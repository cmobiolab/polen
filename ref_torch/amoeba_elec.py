"""Reference implementation (PyTorch, float64, O(N^2), no PBC) of the AMOEBA electrostatics.

Conventions (same as OpenMM/Tinker):
  - positions in nm, charges in e, dipoles in e*nm, quadrupoles in e*nm^2 (already including the Tinker
    factor 1/3, as stored in AmoebaMultipoleForce), polarizabilities in nm^3;
  - energy in kJ/mol (multiplied by ONE_4PI_EPS0);
  - potential of a site j at the point r_i (d = r_i - r_j):
        phi = q/r + mu.d/r^3 + 3 d.Theta.d/r^5
  - traceless quadrupoles: from_openmm() projects out the residual trace (~1e-8, XML rounding).
    OpenMM does not do this projection, and the residual trace causes differences of ~1e-4
    kJ/mol/nm in the forces; the tests apply the same projection on the OpenMM side.

The polarization energy is  E_pol = -1/2 mu_d . E_p,  with mu_d induced by the field with the "d"
scaling (polarization groups) and E_p the permanent field with the "p" scaling (covalent).
Forces come from autograd through the exact solution of the linear system.
"""

from dataclasses import dataclass

import numpy as np
import torch

ONE_4PI_EPS0 = 138.93545764438198  # kJ mol^-1 nm e^-2 (CODATA 2018, same as OpenMM 8)

# Axis types (openmm.AmoebaMultipoleForce)
Z_THEN_X, BISECTOR, Z_BISECT, THREE_FOLD, Z_ONLY, NO_AXIS_TYPE = range(6)

# AMOEBA scaling factors (amoeba09/amoeba2018)
M_SCALE = {12: 0.0, 13: 0.0, 14: 0.4, 15: 0.8}
P_SCALE = {12: 0.0, 13: 0.0, 14: 1.0, 15: 1.0}
P_SCALE_14_INTRA = 0.5
D_SCALE_11 = 0.0


@dataclass
class AmoebaMultipoleParams:
    charge: torch.Tensor        # (N,)
    dipole: torch.Tensor        # (N,3)   local frame
    quadrupole: torch.Tensor    # (N,3,3) local frame
    axis_type: np.ndarray       # (N,) int
    atom_z: np.ndarray          # (N,) int, -1 if absent
    atom_x: np.ndarray
    atom_y: np.ndarray
    thole: torch.Tensor         # (N,)
    damping: torch.Tensor       # (N,)  = polarity^(1/6)
    polarity: torch.Tensor      # (N,)
    m_scale: torch.Tensor       # (N,N) permanent-permanent scales
    p_scale: torch.Tensor       # (N,N) permanent field -> polarization energy
    d_scale: torch.Tensor       # (N,N) permanent field -> induction
    u_scale: torch.Tensor       # (N,N) induced-induced
    undamped: torch.Tensor = None   # (N,N) optional bool: pairs without Thole damping
    thole_pair: torch.Tensor = None  # (N,N) optional: per-pair parameter a (replaces min(a_i, a_j))

    @property
    def n(self):
        return self.charge.shape[0]


def scale_matrices(n, cov):
    """Build the dense scaling matrices from the covalent maps.

    cov[key] is a (per-atom) list of neighbors, with key in
    {12, 13, 14, 15, 'p11', 'p12', 'p13', 'p14'}.
    """
    m = np.ones((n, n))
    p = np.ones((n, n))
    d = np.ones((n, n))
    u = np.ones((n, n))
    for i in range(n):
        group = set(cov['p11'][i])
        for order in (12, 13, 14, 15):
            for j in cov[order][i]:
                m[i, j] = M_SCALE[order]
                p[i, j] = P_SCALE[order]
                if order == 14 and j in group:
                    p[i, j] = P_SCALE_14_INTRA
        for j in group:
            d[i, j] = D_SCALE_11
    for s in (m, p, d, u):
        np.fill_diagonal(s, 0.0)
    return tuple(torch.tensor(s, dtype=torch.float64) for s in (m, p, d, u))


def from_openmm(force):
    """Extract the parameters of an openmm.AmoebaMultipoleForce."""
    import openmm as mm

    n = force.getNumMultipoles()
    charge, dipole, quad, axis, az, ax, ay, thole, damp, pol = ([] for _ in range(10))
    cov_types = {12: mm.AmoebaMultipoleForce.Covalent12, 13: mm.AmoebaMultipoleForce.Covalent13,
                 14: mm.AmoebaMultipoleForce.Covalent14, 15: mm.AmoebaMultipoleForce.Covalent15,
                 'p11': mm.AmoebaMultipoleForce.PolarizationCovalent11,
                 'p12': mm.AmoebaMultipoleForce.PolarizationCovalent12,
                 'p13': mm.AmoebaMultipoleForce.PolarizationCovalent13,
                 'p14': mm.AmoebaMultipoleForce.PolarizationCovalent14}
    cov = {k: [] for k in cov_types}
    for i in range(n):
        c, dp, qp, at, z, x, y, th, dm, po = force.getMultipoleParameters(i)
        charge.append(float(c._value if hasattr(c, '_value') else c))
        dipole.append([float(v) for v in (dp._value if hasattr(dp, '_value') else dp)])
        quad.append([float(v) for v in (qp._value if hasattr(qp, '_value') else qp)])
        axis.append(at); az.append(z); ax.append(x); ay.append(y)
        thole.append(float(th))
        damp.append(float(dm._value if hasattr(dm, '_value') else dm))
        pol.append(float(po._value if hasattr(po, '_value') else po))
        for k, t in cov_types.items():
            cov[k].append(list(force.getCovalentMap(i, t)))
    f64 = dict(dtype=torch.float64)
    quadrupole = torch.tensor(quad, **f64).reshape(n, 3, 3)
    trace = quadrupole.diagonal(dim1=1, dim2=2).sum(-1)
    quadrupole = quadrupole - trace[:, None, None] / 3 * torch.eye(3, **f64)
    m_scale, p_scale, d_scale, u_scale = scale_matrices(n, cov)
    return AmoebaMultipoleParams(
        charge=torch.tensor(charge, **f64),
        dipole=torch.tensor(dipole, **f64),
        quadrupole=quadrupole,
        axis_type=np.array(axis), atom_z=np.array(az), atom_x=np.array(ax), atom_y=np.array(ay),
        thole=torch.tensor(thole, **f64), damping=torch.tensor(damp, **f64),
        polarity=torch.tensor(pol, **f64),
        m_scale=m_scale, p_scale=p_scale, d_scale=d_scale, u_scale=u_scale)


# ---------------------------------------------------------------------------
# Local frames
# ---------------------------------------------------------------------------

def _unit(v):
    return v / torch.linalg.norm(v, dim=-1, keepdim=True)


def rotation_matrices(pos, prm):
    """(N,3,3) matrices whose rows are the local X, Y, Z axes in the lab frame.

    NoAxisType atoms get the identity.
    """
    n = pos.shape[0]
    rot = torch.eye(3, dtype=pos.dtype).expand(n, 3, 3)
    axis = prm.axis_type
    idx = np.where(axis != NO_AXIS_TYPE)[0]
    if len(idx) == 0:
        return rot
    t = torch.as_tensor(idx)
    ri = pos[t]
    vz = _unit(pos[prm.atom_z[idx]] - ri)

    sub = axis[idx]
    has_x = sub != Z_ONLY
    has_y = (sub == Z_BISECT) | (sub == THREE_FOLD)
    # vectors for X and Y (safe index where not applicable; values discarded through where)
    x_atoms = np.where(has_x, prm.atom_x[idx], prm.atom_z[idx])
    y_atoms = np.where(has_y, prm.atom_y[idx], prm.atom_z[idx])
    vx = _unit(pos[x_atoms] - ri)
    vy = _unit(pos[y_atoms] - ri)

    m = lambda mask: torch.as_tensor(mask)[:, None]
    # Bisector: Z along the bisector of Z and X
    vz = torch.where(m(sub == BISECTOR), _unit(vz + vx), vz)
    # ThreeFold: Z = Z + X + Y
    vz = torch.where(m(sub == THREE_FOLD), _unit(vz + vx + vy), vz)
    # ZBisect: X along the bisector of X and Y
    vx = torch.where(m(sub == Z_BISECT), _unit(vx + vy), vx)
    # ZOnly: X fixed in the lab frame
    ex = torch.tensor([1.0, 0.0, 0.0], dtype=pos.dtype).expand_as(vz)
    ey = torch.tensor([0.0, 1.0, 0.0], dtype=pos.dtype).expand_as(vz)
    zonly_x = torch.where(vz[:, :1].abs() < 0.866, ex, ey)
    vx = torch.where(m(sub == Z_ONLY), zonly_x, vx)

    vx = _unit(vx - (vx * vz).sum(-1, keepdim=True) * vz)
    vy = torch.linalg.cross(vz, vx)
    sub_rot = torch.stack([vx, vy, vz], dim=1)
    return rot.index_put((t,), sub_rot)


def chiral_flip(pos, prm):
    """Sign (N,) that flips mu_y, Q_xy and Q_yz at ZThenX chiral centers (Tinker convention)."""
    n = pos.shape[0]
    sign = torch.ones(n, dtype=pos.dtype)
    idx = np.where((prm.axis_type == Z_THEN_X) & (prm.atom_y >= 0))[0]
    if len(idx) == 0:
        return sign
    with torch.no_grad():
        ry = pos[prm.atom_y[idx]]
        ad = pos[idx] - ry
        bd = pos[prm.atom_z[idx]] - ry
        cd = pos[prm.atom_x[idx]] - ry
        vol = (torch.linalg.cross(bd, cd) * ad).sum(-1)
    return sign.index_put((torch.as_tensor(idx),), torch.where(vol < 0, -1.0, 1.0).to(pos.dtype))


def lab_frame_multipoles(pos, prm, dipole=None, quadrupole=None):
    """Rotate the local multipoles to the lab frame. Returns (mu (N,3), Theta (N,3,3))."""
    dipole = prm.dipole if dipole is None else dipole
    quadrupole = prm.quadrupole if quadrupole is None else quadrupole
    s = chiral_flip(pos, prm)
    flip_d = torch.stack([torch.ones_like(s), s, torch.ones_like(s)], -1)
    one = torch.ones_like(s)
    flip_q = torch.stack([torch.stack([one, s, one], -1),
                          torch.stack([s, one, s], -1),
                          torch.stack([one, s, one], -1)], -2)
    rot = rotation_matrices(pos, prm)                           # rows = axes
    mu = torch.einsum('nk,nka->na', dipole * flip_d, rot)
    theta = torch.einsum('nka,nkl,nlb->nab', rot, quadrupole * flip_q, rot)
    return mu, theta


# ---------------------------------------------------------------------------
# Interaction tensors
# ---------------------------------------------------------------------------

def _pair_geometry(pos):
    d = pos[:, None, :] - pos[None, :, :]          # d_ij = r_i - r_j
    n = pos.shape[0]
    eye = torch.eye(n, dtype=torch.bool)
    # "safe" distance on the diagonal to avoid NaN (diagonal terms are zeroed afterwards)
    d_safe = torch.where(eye[..., None], torch.ones_like(d), d)
    r = torch.linalg.norm(d_safe, dim=-1)
    return d_safe, r, eye


def thole_factors(r, prm, eye):
    """lambda3, lambda5, lambda7 of the Thole damping (N,N)."""
    damp = prm.damping[:, None] * prm.damping[None, :]
    a = torch.minimum(prm.thole[:, None], prm.thole[None, :])
    if prm.thole_pair is not None:
        a = prm.thole_pair
    has = (damp > 0) & ~eye
    if prm.undamped is not None:
        has = has & ~prm.undamped
    damp_safe = torch.where(has, damp, torch.ones_like(damp))
    ratio = (r / damp_safe) ** 3
    e = torch.exp(-a * ratio)
    one = torch.ones_like(r)
    l3 = torch.where(has, 1 - e, one)
    l5 = torch.where(has, 1 - (1 + a * ratio) * e, one)
    l7 = torch.where(has, 1 - (1 + a * ratio + 0.6 * a**2 * ratio**2) * e, one)
    return l3, l5, l7


def permanent_energy(pos, prm, mu, theta):
    """Permanent-permanent energy (undamped), in kJ/mol.

    Uses the Tinker/OpenMM closed form (empole), which assumes traceless quadrupoles: if the quadrupole has a
    residual trace (XML rounding), the result matches OpenMM, not the contraction with full T tensors.
    """
    d, r, eye = _pair_geometry(pos)
    R = -d                                                    # R_ij = r_j - r_i
    ci, ck = prm.charge[:, None], prm.charge[None, :]
    dir_ = torch.einsum('ia,ija->ij', mu, R)                  # mu_i . R
    dkr = torch.einsum('ja,ija->ij', mu, R)                   # mu_k . R
    qi_r = torch.einsum('iab,ijb->ija', theta, R)             # Theta_i R
    qk_r = torch.einsum('jab,ijb->ija', theta, R)             # Theta_k R
    qir = (qi_r * R).sum(-1)
    qkr = (qk_r * R).sum(-1)
    dik = mu @ mu.T
    qik = (qi_r * qk_r).sum(-1)
    diqk = torch.einsum('ia,ija->ij', mu, qk_r)
    dkqi = torch.einsum('ja,ija->ij', mu, qi_r)
    qiqk = torch.einsum('iab,jab->ij', theta, theta)

    term1 = ci * ck
    term2 = ck * dir_ - ci * dkr + dik
    term3 = ci * qkr + ck * qir - dir_ * dkr + 2 * (dkqi - diqk + qiqk)
    term4 = dir_ * qkr - dkr * qir - 4 * qik
    term5 = qir * qkr
    rr1 = 1 / r
    rr3 = rr1 / r**2
    rr5 = 3 * rr3 / r**2
    rr7 = 5 * rr5 / r**2
    rr9 = 7 * rr7 / r**2
    U = term1 * rr1 + term2 * rr3 + term3 * rr5 + term4 * rr7 + term5 * rr9
    U = torch.where(eye, torch.zeros_like(U), U * prm.m_scale)
    return 0.5 * ONE_4PI_EPS0 * U.sum()


def permanent_field(pos, prm, mu, theta, scale, charge=None, damped=True):
    """Field of the permanent multipoles at each site, with per-pair scaling.

    damped=True applies the Thole damping (polarizing field); damped=False gives the "bare" field used in the
    permanent energy. Returns (N,3) in e/nm^2 (without the ONE_4PI_EPS0 factor).
    """
    charge = prm.charge if charge is None else charge
    d, r, eye = _pair_geometry(pos)
    if damped:
        l3, l5, l7 = thole_factors(r, prm, eye)
    else:
        l3 = l5 = l7 = torch.ones_like(r)
    rr3 = l3 / r**3
    rr5 = 3 * l5 / r**5
    rr7 = 15 * l7 / r**7
    mu_d = torch.einsum('jb,ijb->ij', mu, d)                  # mu_j . d_ij
    theta_d = torch.einsum('jab,ijb->ija', theta, d)          # Theta_j d_ij
    dtd = (theta_d * d).sum(-1)                               # d.Theta_j.d
    radial = charge[None, :] * rr3 + mu_d * rr5 + dtd * rr7
    E = d * radial[..., None] - mu[None, :, :] * rr3[..., None] - 2 * theta_d * rr5[..., None]
    E = E * torch.where(eye, torch.zeros_like(scale), scale)[..., None]
    return E.sum(1)


def permanent_potential(pos, prm, mu, theta, scale, charge=None):
    """Undamped potential of the permanent multipoles at each site, with per-pair scaling."""
    charge = prm.charge if charge is None else charge
    d, r, eye = _pair_geometry(pos)
    mu_d = torch.einsum('jb,ijb->ij', mu, d)
    dtd = torch.einsum('ija,jab,ijb->ij', d, theta, d)
    phi = charge[None, :] / r + mu_d / r**3 + 3 * dtd / r**5
    return (phi * torch.where(eye, torch.zeros_like(scale), scale)).sum(1)


def induced_potential_field(pos, prm, pairs):
    """Thole-damped potential and field of induced dipoles at each site.

    pairs: list of (mu (N,3), scale (N,N)); the contributions are summed.
    Returns (phi (N,), E (N,3)).
    """
    d, r, eye = _pair_geometry(pos)
    l3, l5, _ = thole_factors(r, prm, eye)
    phi = torch.zeros(pos.shape[0], dtype=pos.dtype)
    E = torch.zeros_like(pos)
    for mu, scale in pairs:
        w = torch.where(eye, torch.zeros_like(scale), scale)
        mu_d = torch.einsum('jb,ijb->ij', mu, d)
        phi = phi + (w * l3 * mu_d / r**3).sum(1)
        E = E + (w[..., None] * (3 * d * (l5 * mu_d / r**5)[..., None]
                                 - mu[None, :, :] * (l3 / r**3)[..., None])).sum(1)
    return phi, E


def dipole_tensor(pos, prm):
    """(3N,3N) matrix of the damped dipole-dipole tensors, with u scaling (zero diagonal)."""
    d, r, eye = _pair_geometry(pos)
    l3, l5, _ = thole_factors(r, prm, eye)
    I = torch.eye(3, dtype=pos.dtype)
    T = (3 * d[..., :, None] * d[..., None, :] * (l5 / r**5)[..., None, None]
         - I * (l3 / r**3)[..., None, None])
    T = T * torch.where(eye, torch.zeros_like(prm.u_scale), prm.u_scale)[..., None, None]
    n = pos.shape[0]
    return T.permute(0, 2, 1, 3).reshape(3 * n, 3 * n)


def induced_dipoles(pos, prm, E0, polarity=None):
    """Solve (alpha^-1 - T) mu = E0 exactly for the sites with polarizability > 0."""
    polarity = prm.polarity if polarity is None else polarity
    n = pos.shape[0]
    active = torch.nonzero(polarity > 0).squeeze(-1)
    idx3 = (3 * active[:, None] + torch.arange(3)).reshape(-1)
    T = dipole_tensor(pos, prm)[idx3][:, idx3]
    alpha_inv = torch.repeat_interleave(1 / polarity[active], 3)
    A = torch.diag(alpha_inv) - T
    mu_act = torch.linalg.solve(A, E0.reshape(-1)[idx3])
    mu = torch.zeros(3 * n, dtype=pos.dtype).index_put((idx3,), mu_act)
    return mu.reshape(n, 3)


def electrostatic_energy(pos, prm, polarization='mutual', return_terms=False):
    """Total energy of the AmoebaMultipoleForce (no cutoff), in kJ/mol.

    polarization: 'mutual' (exact solution of the linear system) or 'direct' (mu = alpha E_d).
    """
    mu, theta = lab_frame_multipoles(pos, prm)
    e_perm = permanent_energy(pos, prm, mu, theta)
    E_d = permanent_field(pos, prm, mu, theta, prm.d_scale)
    E_p = permanent_field(pos, prm, mu, theta, prm.p_scale)
    if polarization == 'mutual':
        mu_ind = induced_dipoles(pos, prm, E_d)
    elif polarization == 'direct':
        mu_ind = prm.polarity[:, None] * E_d
    else:
        raise ValueError(f'unknown polarization: {polarization}')
    e_pol = -0.5 * ONE_4PI_EPS0 * (mu_ind * E_p).sum()
    if return_terms:
        if polarization == 'mutual':
            mu_ind_p = induced_dipoles(pos, prm, E_p)
        else:
            mu_ind_p = prm.polarity[:, None] * E_p
        return e_perm + e_pol, dict(perm=e_perm, pol=e_pol, mu_lab=mu, theta_lab=theta,
                                    mu_ind=mu_ind, mu_ind_p=mu_ind_p, E_d=E_d, E_p=E_p)
    return e_perm + e_pol
