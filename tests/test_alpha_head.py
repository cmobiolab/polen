"""Head α(R): (1) old heads (without alpha_out) load and give α = α_base; (2) learn_alpha starts at α_base;
(3) molecular α of the mutual Thole model: AMOEBA water ≈ 1.41 Å³ (Ren & Ponder 2003), and the sum of the atomic α
without coupling when the atoms are far apart; (4) gradient of the molecular α with respect to α and the positions
(finite differences)."""
from pathlib import Path

import numpy as np
import torch

from mlmmpol.embedding_head import make_head, molecular_polarizability
from mlmmpol.export_head import load_head

ROOT = Path(__file__).resolve().parents[1]


def water():
    th = np.radians(108.5 / 2)
    x = torch.tensor([[0, 0, 0], [0.9572 * np.sin(th), 0, 0.9572 * np.cos(th)],
                      [-0.9572 * np.sin(th), 0, 0.9572 * np.cos(th)]], dtype=torch.float64)
    return x, torch.tensor([0.837, 0.496, 0.496], dtype=torch.float64)


def test_amoeba_water_polarizability():
    x, a = water()
    am = molecular_polarizability(x, a)
    iso = torch.trace(am).item() / 3
    assert abs(iso - 1.41) < 0.03, iso
    assert torch.allclose(am, am.T, atol=1e-12)


def test_far_atoms_additive():
    x, a = water()
    am = molecular_polarizability(x * 1e3, a)
    assert torch.allclose(am, torch.eye(3, dtype=torch.float64) * a.sum(), rtol=1e-6)


def test_gradients_fd():
    x, a = water()
    x = x.clone().requires_grad_(True)
    a = a.clone().requires_grad_(True)
    f = lambda xx, aa: torch.trace(molecular_polarizability(xx, aa))
    gx, ga = torch.autograd.grad(f(x, a), [x, a])
    h = 1e-6
    for i in range(3):
        for k in range(3):
            xp, xm = x.detach().clone(), x.detach().clone()
            xp[i, k] += h
            xm[i, k] -= h
            assert abs((f(xp, a.detach()) - f(xm, a.detach())).item() / (2 * h) - gx[i, k].item()) < 1e-6
        ap, am_ = a.detach().clone(), a.detach().clone()
        ap[i] += h
        am_[i] -= h
        assert abs((f(x.detach(), ap) - f(x.detach(), am_)).item() / (2 * h) - ga[i].item()) < 1e-6


def test_old_head_loads_and_alpha_is_base():
    head = load_head(ROOT / 'qm_ref' / 'heads' / 'v2_f128_b3_s0')
    assert head.learn_alpha is False
    x, a = water()
    z = torch.tensor([8, 1, 1])
    *_, h = head.predict_full(x, z, 0.0)
    assert torch.equal(head.alpha(h, a), a)


def test_learn_alpha_starts_at_base():
    head = make_head(2, n_features=32, n_blocks=2, learn_alpha=True).double()
    x, a = water()
    *_, h = head.predict_full(x, torch.tensor([8, 1, 1]), 0.0)
    assert torch.allclose(head.alpha(h, a), a)
