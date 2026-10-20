"""EMLE-style charge penetration: each ML atom = point core + Slater valence cloud.

Valence density of atom A (population N_A, width s_A; from the MBIS outer shell):
    rho_A(r) = N_A / (8 pi s_A^3) exp(-r / s_A)
Cloud potential: -N_A / r [1 - (1 + r / (2 s_A)) exp(-r / s_A)]. With the "effective" point core (q_A + N_A),
the total potential of the atom is
    phi_A(r) = q_A / r + N_A (1 + r / (2 s_A)) exp(-r / s_A) / r
i.e. that of a point charge q_A plus the penetration correction phi_pen,A (which vanishes for r >> s_A).

The correction energy with the MM point multipoles (Tinker/OpenMM convention, quadrupoles with the 1/3 factor):
  E_pen = sum_j [ q_j phi_pen(r_j) + mu_j . grad phi_pen(r_j) + Theta_j : grad grad phi_pen(r_j) ].

Atomic units (bohr, e, Hartree).
"""

import torch


def slater_overlap_potential(r_points, R_ml, N_val, s_val):
    """phi_pen at the points r_points (M,3), summed over the ML atoms (R_ml (A,3), N_val (A,), s_val (A,))."""
    d = torch.linalg.norm(r_points[:, None, :] - R_ml[None, :, :], dim=-1)      # (M, A)
    return (N_val * (1 + d / (2 * s_val)) * torch.exp(-d / s_val) / d).sum(-1)


def penetration_energy(R_ml, N_val, s_val, R_mm, q_mm, mu_mm=None, theta_mm=None):
    """Penetration energy (Hartree) of the ML Slater valence with the MM point multipoles.

    Differentiable with respect to all inputs (including R_mm): the derivatives of phi use create_graph=True,
    without detaching R_mm from the graph.
    """
    if not R_mm.requires_grad:
        R_mm = R_mm.clone().requires_grad_(True)
    phi = slater_overlap_potential(R_mm, R_ml, N_val, s_val)
    energy = (q_mm * phi).sum()
    if mu_mm is None and theta_mm is None:
        return energy
    grad = torch.autograd.grad(phi.sum(), R_mm, create_graph=True)[0]            # (M,3); independent points
    if mu_mm is not None:
        energy = energy + (mu_mm * grad).sum()
    if theta_mm is not None:
        hess = torch.stack([torch.autograd.grad(grad[:, a].sum(), R_mm, create_graph=True)[0]
                            for a in range(3)], dim=1)                            # (M,3,3)
        energy = energy + (theta_mm * hess).sum()
    return energy
