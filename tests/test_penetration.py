"""Charge penetration (ref_torch/penetration.py): Slater cloud formula and limits."""

import numpy as np
import torch

from ref_torch.penetration import penetration_energy, slater_overlap_potential


def test_slater_potential_matches_numerical_integration():
    """phi_pen + (point charge) = potential of the core + cloud density, integrated numerically."""
    N, s = 5.0, 0.45
    r_grid = np.linspace(1e-6, 40 * s, 400001)
    rho = N / (8 * np.pi * s**3) * np.exp(-r_grid / s)
    for r in [0.2, 0.5, 1.0, 2.0, 4.0]:
        # potential of a spherical distribution: (1/r) int_0^r 4 pi r'^2 rho + int_r^inf 4 pi r' rho
        inner = np.trapezoid(np.where(r_grid < r, 4 * np.pi * r_grid**2 * rho, 0.0), r_grid) / r
        outer = np.trapezoid(np.where(r_grid >= r, 4 * np.pi * r_grid * rho, 0.0), r_grid)
        v_cloud = -(inner + outer)                    # electrons: charge -N
        v_total = N / r + v_cloud                     # effective core +N (q = 0 here)
        phi = slater_overlap_potential(torch.tensor([[r, 0.0, 0.0]]), torch.zeros(1, 3),
                                       torch.tensor([N]), torch.tensor([s])).item()
        assert abs(phi - v_total) < 1e-6 * max(1.0, abs(v_total))


def test_vanishes_far_away():
    R_ml = torch.zeros(1, 3, dtype=torch.float64)
    e = penetration_energy(R_ml, torch.tensor([6.0], dtype=torch.float64), torch.tensor([0.4], dtype=torch.float64),
                           torch.tensor([[30.0, 0, 0]], dtype=torch.float64), torch.tensor([-0.5], dtype=torch.float64))
    assert abs(e.item()) < 1e-25


def test_multipole_terms_match_finite_differences():
    """mu . grad phi and Theta : grad grad phi == finite differences of point charges."""
    torch.manual_seed(0)
    R_ml = torch.randn(3, 3, dtype=torch.float64)
    N = torch.tensor([5.0, 0.8, 6.0], dtype=torch.float64)
    s = torch.tensor([0.5, 0.37, 0.4], dtype=torch.float64)
    x = torch.tensor([[1.5, 0.3, -0.7]], dtype=torch.float64)
    mu = torch.tensor([[0.3, -0.2, 0.5]], dtype=torch.float64)
    e_mu = penetration_energy(R_ml, N, s, x, torch.zeros(1, dtype=torch.float64), mu_mm=mu).item()
    h = 1e-5
    fd = (slater_overlap_potential(x + h * mu / mu.norm(), R_ml, N, s)
          - slater_overlap_potential(x - h * mu / mu.norm(), R_ml, N, s)).item() / (2 * h) * mu.norm().item()
    assert abs(e_mu - fd) < 1e-7 * max(1.0, abs(fd))
    theta = torch.tensor([[[0.2, 0.1, 0.0], [0.1, -0.3, 0.05], [0.0, 0.05, 0.1]]], dtype=torch.float64)
    e_th = penetration_energy(R_ml, N, s, x, torch.zeros(1, dtype=torch.float64), theta_mm=theta).item()
    hess = torch.autograd.functional.hessian(lambda p: slater_overlap_potential(p[None], R_ml, N, s).sum(), x[0])
    assert abs(e_th - (theta[0] * hess).sum().item()) < 1e-10
