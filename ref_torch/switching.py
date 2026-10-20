"""Layered polarization with a smooth mutual -> direct switch (for energy-conserving dynamics).

Each molecule m gets a weight s_m in [0, 1] (1 = mutual, 0 = direct) as a function of the distance d_m from its
reference atom to the ML region, using a smooth minimum over the ML atoms (the ordinary minimum has kinks):
    d_m = -tau * log sum_A exp(-|r_m - R_A| / tau)
    s = S(d),  S = 1 for d <= R_in, 0 for d >= R_out, degree-5 (C2) polynomial in between.
The induced–induced coupling of pair (i, j) is multiplied by s_i s_j: with s = 1 it is mutual, with s = 0 it is
direct (dipoles from the permanent field only). The energy stays variational; forces include ds/dR via
autograd.
"""

import torch


def smooth_step(d, r_in, r_out):
    """S(d): 1 -> 0 between r_in and r_out, with continuous 1st and 2nd derivatives."""
    x = ((d - r_in) / (r_out - r_in)).clamp(0.0, 1.0)
    return 1 - x**3 * (10 - 15 * x + 6 * x**2)


def soft_min_distance(points, centers, tau=0.02):
    """Smooth minimum (nm) of the distance from each point (M,3) to the centers (A,3)."""
    d = torch.linalg.norm(points[:, None, :] - centers[None, :, :], dim=-1)
    return -tau * torch.logsumexp(-d / tau, dim=1)


def site_weights(pos, ml, mol_ref, mol_of_site, r_in, r_out, tau=0.02, hard=False):
    """Weight s (N,) per site: 1 on the ML atoms; on MM molecules, the s of the molecule they belong to.

    mol_ref (M,): index of the reference atom of each MM molecule (e.g. the water O);
    mol_of_site (N,): MM molecule index of each site (-1 for the ML atoms).
    hard=True uses an abrupt cut at (r_in + r_out)/2 (discontinuous; for comparison only).
    """
    d = soft_min_distance(pos[mol_ref], pos[ml], tau)
    if hard:
        s_mol = (d <= 0.5 * (r_in + r_out)).to(pos.dtype)
    else:
        s_mol = smooth_step(d, r_in, r_out)
    s = torch.ones(pos.shape[0], dtype=pos.dtype)
    idx = torch.nonzero(mol_of_site >= 0).squeeze(-1)
    return s.index_put((idx,), s_mol[mol_of_site[idx]])


def switched_u_scale(u_scale, s):
    """u scale modulated by the switch: u_ij * s_i * s_j."""
    return u_scale * s[:, None] * s[None, :]
