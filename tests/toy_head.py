"""Toy ML head (TorchScript) to test the plugin 'ml' mode: q, mu, Theta, alpha from R_ML.

Differentiable and equivariant; q conserves the total charge; Theta symmetric and traceless.
"""

import torch


class ToyHead(torch.nn.Module):
    def __init__(self, q0, alpha0):
        super().__init__()
        self.register_buffer('q0', torch.as_tensor(q0, dtype=torch.float64))
        self.register_buffer('alpha0', torch.as_tensor(alpha0, dtype=torch.float64))

    def forward(self, r: torch.Tensor):
        n = r.shape[0]
        eye_n = torch.eye(n, dtype=r.dtype, device=r.device)
        d = r[:, None, :] - r[None, :, :]
        dist = torch.sqrt((d * d).sum(-1) + eye_n)
        w = torch.exp(-dist / 0.2) * (1 - eye_n)
        q = self.q0 + 0.05 * torch.tanh(w.sum(1) - 0.5)
        q = q - (q.sum() - self.q0.sum()) / n
        mu = 0.01 * (w[..., None] * (-d)).sum(1)
        dd = d[..., :, None] * d[..., None, :]
        r2 = (d * d).sum(-1)
        eye3 = torch.eye(3, dtype=r.dtype, device=r.device)
        theta = 5e-3 * (w[..., None, None] * (dd - r2[..., None, None] * eye3 / 3.0)).sum(1)
        alpha = self.alpha0 * (1 + 0.2 * torch.tanh(w.sum(1)))
        return q, mu, theta, alpha


class ToyHeadPen(torch.nn.Module):
    """ToyHead plus a geometry-dependent Slater valence (N, s in nm) (composition: TorchScript does not accept super().forward)."""

    def __init__(self, q0, alpha0, n0, s0):
        super().__init__()
        self.base = ToyHead(q0, alpha0)
        self.register_buffer('n0', torch.as_tensor(n0, dtype=torch.float64))
        self.register_buffer('s0', torch.as_tensor(s0, dtype=torch.float64))

    def forward(self, r: torch.Tensor):
        q, mu, theta, alpha = self.base(r)
        n = r.shape[0]
        eye_n = torch.eye(n, dtype=r.dtype, device=r.device)
        d = r[:, None, :] - r[None, :, :]
        dist = torch.sqrt((d * d).sum(-1) + eye_n)
        w = (torch.exp(-dist / 0.2) * (1 - eye_n)).sum(1)
        return q, mu, theta, alpha, self.n0 * (1 + 0.05 * torch.tanh(w)), self.s0 * (1 + 0.1 * torch.tanh(w - 1.0))
