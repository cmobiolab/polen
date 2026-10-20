"""Embedding head of the 'ml' mode: permanent multipoles and Slater valence as a function of geometry.

Plugin interface (MLPolMultipoleForce.setMLHead, TorchScript):
    forward(pos (M,3) nm, float64) -> q (M) e, mu (M,3) e·nm, Theta (M,3,3) e·nm² (symmetric, traceless, Tinker/OpenMM
    convention = Buckingham/3), alpha (M) nm³, N (M) valence electrons, s (M) nm
Targets: MBIS in vacuum. alpha: AMOEBA × 1.10 (base) or, with learn_alpha, α_base · exp(δα(R)), trained against the
finite-field QM molecular α through the mutual Thole model (molecular_polarizability).

Model (equivariant by construction; all pairs of the molecule, smooth 6 Å cutoff):
- element -> embedding; T continuous-filter convolution blocks (SchNet) over Gaussian radial bases -> h_i (scalars);
- q_i = q_ref(Z_i) + δq_i, then a weighted shift (learned positive weights) so that Σq = exact total charge;
- mu_i = Σ_j a_ij r̂_ij (a_ij scalar from h_i, h_j and the distance);
- Theta_i = Σ_j c_ij (r̂_ij r̂_ijᵀ − I/3) + Σ_{j,k} d_ijk sym₀(r̂_ij r̂_ikᵀ) (sym₀ = symmetric traceless part);
- N_i = N_ref(Z_i) + δN_i;  s_i = s_ref(Z_i) · exp(δs_i).
Per-element references (training-set means) are stored as buffers; the network learns only the corrections.
"""
from typing import List, Tuple

import torch
from torch import nn

NM_TO_ANG = 10.0


class ShiftedSoftplus(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.softplus(x) - 0.6931471805599453


def mlp(n_in: int, n_hidden: int, n_out: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(n_in, n_hidden), ShiftedSoftplus(), nn.Linear(n_hidden, n_out))


class Interaction(nn.Module):
    def __init__(self, f: int, k: int):
        super().__init__()
        self.filt = mlp(k, f, f)
        self.inp = nn.Linear(f, f, bias=False)
        self.out = mlp(f, f, f)

    def forward(self, h: torch.Tensor, rbf: torch.Tensor, fc: torch.Tensor) -> torch.Tensor:
        w = self.filt(rbf) * fc[..., None]                    # (M, M, F)
        m = (w * self.inp(h)[None, :, :]).sum(1)              # Σ_j W_ij ⊙ x_j
        return h + self.out(m)


class EmbeddingHead(nn.Module):
    """Network shared across ligands; an instance "bound" to a ligand via bind() becomes the plugin module."""

    def __init__(self, n_features: int = 64, n_rbf: int = 16, n_blocks: int = 3, cutoff: float = 6.0, z_max: int = 36,
                 learn_alpha: bool = False):
        super().__init__()
        self.learn_alpha = learn_alpha
        # α(R): multiplicative correction to the base polarizability (AMOEBA × 1.10), α = α_base · exp(δα); the final
        # layer starts at zero (α = α_base at the start of training)
        self.alpha_out = mlp(n_features, n_features, 1)
        torch.nn.init.zeros_(self.alpha_out[2].weight)
        torch.nn.init.zeros_(self.alpha_out[2].bias)
        self.cutoff = cutoff
        self.embed = nn.Embedding(z_max + 1, n_features)
        self.register_buffer('centers', torch.linspace(0.5, cutoff, n_rbf))
        self.gamma: float = float((n_rbf / cutoff) ** 2)
        self.blocks = nn.ModuleList([Interaction(n_features, n_rbf) for _ in range(n_blocks)])
        self.scalar_out = mlp(n_features, n_features, 4)                 # δq, charge weight, δN, δs
        self.pair_out = mlp(2 * n_features + n_rbf, n_features, 2)      # a_ij (dipole), c_ij (quadrupole)
        self.trip_out = mlp(n_features + 2 * n_rbf, n_features, 1)      # d_ijk (cross quadrupole)
        # per-element references (filled by set_references)
        self.register_buffer('q_ref', torch.zeros(z_max + 1))
        self.register_buffer('n_ref', torch.zeros(z_max + 1))
        self.register_buffer('s_ref', torch.ones(z_max + 1))

    def predict(self, x_ang: torch.Tensor, z: torch.Tensor, total_charge: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        q, mu, theta, n_val, s_val, h = self.predict_full(x_ang, z, total_charge)
        return q, mu, theta, n_val, s_val

    def alpha(self, h: torch.Tensor, alpha_base: torch.Tensor) -> torch.Tensor:
        """Polarizabilities: α_base · exp(δα(h)) if learn_alpha, otherwise α_base."""
        if self.learn_alpha:
            return alpha_base * torch.exp(self.alpha_out(h)[:, 0])
        return alpha_base

    def set_references(self, q_ref, n_ref, s_ref):
        self.q_ref.copy_(torch.as_tensor(q_ref, dtype=self.q_ref.dtype))
        self.n_ref.copy_(torch.as_tensor(n_ref, dtype=self.n_ref.dtype))
        self.s_ref.copy_(torch.as_tensor(s_ref, dtype=self.s_ref.dtype))

    def predict_full(self, x_ang: torch.Tensor, z: torch.Tensor, total_charge: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """x in Å; returns q (e), mu (e·Å), Theta (e·Å²), N, s (Å)."""
        m = x_ang.shape[0]
        d = x_ang[None, :, :] - x_ang[:, None, :]                       # d[i, j] = x_j - x_i
        eye = torch.eye(m, dtype=x_ang.dtype, device=x_ang.device)
        r = torch.sqrt((d * d).sum(-1) + eye)                           # 1 on the diagonal (avoids 0/0)
        u = d / r[..., None]
        off = 1.0 - eye
        fc = 0.5 * (torch.cos(torch.pi * torch.clamp(r / self.cutoff, max=1.0)) + 1.0) * off
        rbf = torch.exp(-self.gamma * (r[..., None] - self.centers) ** 2)
        h = self.embed(z).to(x_ang.dtype)
        for blk in self.blocks:
            h = blk(h, rbf, fc)
        s_out = self.scalar_out(h)
        # charges with exact conservation of the total charge
        q_raw = self.q_ref[z] + s_out[:, 0]
        w = nn.functional.softplus(s_out[:, 1]) + 1e-3
        q = q_raw + w * (total_charge - q_raw.sum()) / w.sum()
        n_val = self.n_ref[z] + s_out[:, 2]
        s_val = self.s_ref[z] * torch.exp(s_out[:, 3])
        # dipoles and quadrupoles
        hij = torch.cat([h[:, None, :].expand(m, m, -1), h[None, :, :].expand(m, m, -1), rbf], -1)
        p = self.pair_out(hij) * fc[..., None]                          # (M, M, 2)
        mu = (p[..., 0:1] * u).sum(1)
        uu = u[..., :, None] * u[..., None, :]                          # (M, M, 3, 3)
        i3 = torch.eye(3, dtype=x_ang.dtype, device=x_ang.device)
        theta = (p[..., 1, None, None] * (uu - i3 / 3.0)).sum(1)
        trip_in = torch.cat([h[:, None, None, :].expand(m, m, m, -1), rbf[:, :, None, :].expand(m, m, m, -1),
                             rbf[:, None, :, :].expand(m, m, m, -1)], -1)
        dijk = self.trip_out(trip_in)[..., 0] * fc[:, :, None] * fc[:, None, :]          # (M, M, M)
        cross = u[:, :, None, :, None] * u[:, None, :, None, :]                           # û_ij û_ikᵀ
        cross = 0.5 * (cross + cross.transpose(-1, -2))
        tr = cross.diagonal(dim1=-2, dim2=-1).sum(-1)
        cross = cross - tr[..., None, None] * i3 / 3.0
        theta = theta + (dijk[..., None, None] * cross).sum((1, 2))
        return q, mu, theta, n_val, s_val, h


class EmbeddingHeadV2(EmbeddingHead):
    """v2: same outputs, M² cost (v1 has a triplet term, M³: ~0.5 s per evaluation with 56 atoms on the CPU).
    - factorized pair MLP: W·[h_i; h_j; rbf] = W_i h_i + W_j h_j + W_r rbf (per-atom projections, summed per pair);
    - cross quadrupole from products of C pairs of the atom's own equivariant vectors:
      Theta_i += sym₀(Σ_c v_ic w_icᵀ), v_ic = Σ_j a^c_ij r̂_ij, w_ic = Σ_j b^c_ij r̂_ij.
    """

    def __init__(self, n_features: int = 64, n_rbf: int = 16, n_blocks: int = 3, cutoff: float = 6.0, z_max: int = 36,
                 n_vec: int = 4, learn_alpha: bool = False):
        super().__init__(n_features, n_rbf, n_blocks, cutoff, z_max, learn_alpha)
        del self.pair_out
        del self.trip_out
        self.n_vec = n_vec
        self.pair_i = nn.Linear(n_features, n_features)
        self.pair_j = nn.Linear(n_features, n_features, bias=False)
        self.pair_r = nn.Linear(n_rbf, n_features, bias=False)
        self.pair_act = ShiftedSoftplus()
        self.pair_lin = nn.Linear(n_features, 2 + 2 * n_vec)

    def predict_full(self, x_ang: torch.Tensor, z: torch.Tensor, total_charge: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        m = x_ang.shape[0]
        d = x_ang[None, :, :] - x_ang[:, None, :]
        eye = torch.eye(m, dtype=x_ang.dtype, device=x_ang.device)
        r = torch.sqrt((d * d).sum(-1) + eye)
        u = d / r[..., None]
        off = 1.0 - eye
        fc = 0.5 * (torch.cos(torch.pi * torch.clamp(r / self.cutoff, max=1.0)) + 1.0) * off
        rbf = torch.exp(-self.gamma * (r[..., None] - self.centers) ** 2)
        h = self.embed(z).to(x_ang.dtype)
        for blk in self.blocks:
            h = blk(h, rbf, fc)
        s_out = self.scalar_out(h)
        q_raw = self.q_ref[z] + s_out[:, 0]
        w = nn.functional.softplus(s_out[:, 1]) + 1e-3
        q = q_raw + w * (total_charge - q_raw.sum()) / w.sum()
        n_val = self.n_ref[z] + s_out[:, 2]
        s_val = self.s_ref[z] * torch.exp(s_out[:, 3])
        hid = self.pair_act(self.pair_i(h)[:, None, :] + self.pair_j(h)[None, :, :] + self.pair_r(rbf))
        p = self.pair_lin(hid) * fc[..., None]                          # (M, M, 2 + 2C)
        mu = (p[..., 0:1] * u).sum(1)
        i3 = torch.eye(3, dtype=x_ang.dtype, device=x_ang.device)
        uu = u[..., :, None] * u[..., None, :]
        theta = (p[..., 1, None, None] * (uu - i3 / 3.0)).sum(1)
        nv = self.n_vec
        vv = torch.einsum('ijc,ijx->icx', p[..., 2:2 + nv], u)          # (M, C, 3)
        ww = torch.einsum('ijc,ijx->icx', p[..., 2 + nv:2 + 2 * nv], u)
        vw = (vv[..., :, None] * ww[..., None, :]).sum(1)               # Σ_c v_c w_cᵀ
        vw = 0.5 * (vw + vw.transpose(-1, -2))
        theta = theta + vw - vw.diagonal(dim1=-2, dim2=-1).sum(-1)[:, None, None] * i3 / 3.0
        return q, mu, theta, n_val, s_val, h


def make_head(version: int = 1, **kw) -> EmbeddingHead:
    return EmbeddingHeadV2(**kw) if version == 2 else EmbeddingHead(**{k: v for k, v in kw.items() if k != 'n_vec'})


THOLE_A = 0.39


def thole_tensor(x_ang: torch.Tensor, alpha: torch.Tensor, a: float = THOLE_A) -> torch.Tensor:
    """Thole-damped (AMOEBA) dipole–dipole tensor between all pairs, (3M, 3M), Å⁻³: T_ij = (3λ5 r̂r̂ᵀ − λ3 I)/r³,
    u = r/(α_i α_j)^(1/6), λ3 = 1 − e^(−a u³), λ5 = 1 − (1 + a u³) e^(−a u³). Zero diagonal block."""
    m = x_ang.shape[0]
    d = x_ang[None, :, :] - x_ang[:, None, :]
    eye = torch.eye(m, dtype=x_ang.dtype, device=x_ang.device)
    r = torch.sqrt((d * d).sum(-1) + eye)
    u3 = (r / (alpha[:, None] * alpha[None, :]) ** (1.0 / 6.0)) ** 3
    ex = torch.exp(-a * u3)
    l3, l5 = 1.0 - ex, 1.0 - (1.0 + a * u3) * ex
    rr = d[..., :, None] * d[..., None, :] / (r * r)[..., None, None]
    i3 = torch.eye(3, dtype=x_ang.dtype, device=x_ang.device)
    t = (3.0 * l5[..., None, None] * rr - l3[..., None, None] * i3) / (r ** 3)[..., None, None]
    t = t * (1.0 - eye)[..., None, None]
    return t.permute(0, 2, 1, 3).reshape(3 * m, 3 * m)


def molecular_polarizability(x_ang: torch.Tensor, alpha: torch.Tensor, a: float = THOLE_A) -> torch.Tensor:
    """Molecular α (3×3, Å³) of the induced-dipole model with mutual coupling and Thole between all pairs, as the ML
    region in the kernel (a single polarization group, u = 1): Σ_ij [(α⁻¹ − T)⁻¹]_ij."""
    m = x_ang.shape[0]
    a_inv = torch.diag(torch.repeat_interleave(1.0 / alpha, 3))
    b = torch.linalg.inv(a_inv - thole_tensor(x_ang, alpha, a))
    return b.reshape(m, 3, m, 3).sum((0, 2))


class BoundHead(nn.Module):
    """Plugin module for one ligand: forward(pos nm) -> (q, mu, Theta, alpha, N, s) in OpenMM units."""

    def __init__(self, head: EmbeddingHead, z: List[int], total_charge: float, alpha_nm3: List[float]):
        super().__init__()
        self.head = head
        self.register_buffer('z', torch.tensor(z, dtype=torch.long))
        self.total_charge = float(total_charge)
        self.register_buffer('alpha', torch.tensor(alpha_nm3, dtype=torch.float64))
        self.nm_to_ang = NM_TO_ANG

    def forward(self, pos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        k = self.nm_to_ang
        q, mu, theta, n_val, s_val, h = self.head.predict_full(pos * k, self.z, self.total_charge)
        # α(R) = α_base · exp(δα) with learn_alpha; otherwise α_base (AMOEBA × 1.10), attached to the graph in the same
        # way (the kernel differentiates all outputs with respect to the positions)
        alpha = self.head.alpha(h, self.alpha) + 0.0 * pos.sum()
        return q, mu / k, theta / (k * k), alpha, n_val, s_val / k
