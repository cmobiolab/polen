"""Explicit derivatives (ref_torch/explicit_grads.py) vs autograd: dE/dalpha and dE/ds (smooth switch)."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from openmm_systems import SYSTEMS, build_amoeba
from ref_torch import amoeba_elec as ae
from ref_torch.explicit_grads import alpha_gradient, switch_gradient
from ref_torch.mlmm_pol import mlmm_electrostatic_energy
from ref_torch.switching import switched_u_scale


@pytest.fixture(scope='module')
def nma():
    from openmm import app
    keep = build_amoeba('nma_shell.pdb')
    prm = ae.from_openmm(keep[1])
    top = app.PDBFile(str(SYSTEMS / 'nma_shell.pdb')).topology
    ml = np.array([a.index for a in top.atoms() if a.residue.name in ('ACE', 'NME')])
    t = torch.as_tensor(ml)
    rng = np.random.default_rng(3)
    q = prm.charge[t] + torch.tensor(rng.normal(0, 0.05, len(ml)))
    q = q - (q.sum() - prm.charge[t].sum()) / len(ml)
    mu = torch.tensor(rng.normal(0, 0.004, (len(ml), 3)))
    alpha = prm.polarity[t] * torch.tensor(rng.uniform(0.8, 1.3, len(ml)))
    return keep, prm, keep[3], ml, q, mu, alpha


@pytest.mark.parametrize('coupling', [dict(), dict(a_cross=0.7), dict(damp_ml_mm=False)])
def test_alpha_gradient_matches_autograd(nma, coupling):
    _, prm, pos, ml, q, mu, alpha = nma
    x = torch.tensor(pos)
    a = alpha.clone().requires_grad_()
    e, terms = mlmm_electrostatic_energy(x, prm, ml, q, mu, a, return_terms=True, **coupling)
    g_auto = torch.autograd.grad(e, a)[0]
    g_expl = alpha_gradient(x, {k: (v.detach() if torch.is_tensor(v) else v) for k, v in terms.items()})[torch.as_tensor(ml)]
    assert g_auto.abs().max() > 1.0                               # the term is not negligible
    assert torch.allclose(g_expl, g_auto, rtol=1e-8, atol=1e-6)


def test_switch_gradient_matches_autograd(nma):
    _, prm, pos, ml, *_ = nma
    x = torch.tensor(pos)
    rng = np.random.default_rng(5)
    s0 = torch.tensor(rng.uniform(0, 1, prm.n))
    s0[torch.as_tensor(ml)] = 1.0
    s = s0.clone().requires_grad_()
    p = replace(prm, u_scale=switched_u_scale(prm.u_scale, s))
    e, terms = ae.electrostatic_energy(x, p, return_terms=True)
    g_auto = torch.autograd.grad(e, s)[0]
    terms = {k: (v.detach() if torch.is_tensor(v) else v) for k, v in terms.items()}
    terms['prm'] = replace(prm, u_scale=switched_u_scale(prm.u_scale, s0))
    g_expl = switch_gradient(x, terms, prm.u_scale, s0)
    assert g_auto.abs().max() > 0.1
    assert torch.allclose(g_expl, g_auto, rtol=1e-8, atol=1e-8)


def test_quadrupole_gradient_matches_autograd(nma):
    """Explicit dE/dTheta (symmetric part) vs autograd, with generic (traceless) ML quadrupoles."""
    from ref_torch.explicit_grads import quadrupole_gradient
    _, prm, pos, ml, q, mu, alpha = nma
    rng = np.random.default_rng(9)
    th = torch.tensor(rng.normal(0, 2e-4, (len(ml), 3, 3)))
    th = 0.5 * (th + th.transpose(1, 2))
    th = th - th.diagonal(dim1=1, dim2=2).sum(-1)[:, None, None] / 3 * torch.eye(3, dtype=th.dtype)
    x = torch.tensor(pos)
    for coupling in (dict(), dict(a_cross=0.7)):
        t = th.clone().requires_grad_()
        e, terms = mlmm_electrostatic_energy(x, prm, ml, q, mu, alpha, theta=t, return_terms=True, **coupling)
        g_auto = torch.autograd.grad(e, t)[0]
        g_auto = 0.5 * (g_auto + g_auto.transpose(1, 2))
        g_expl = quadrupole_gradient(x, {k: (v.detach() if torch.is_tensor(v) else v) for k, v in terms.items()}, ml)
        assert g_auto.abs().max() > 1.0
        assert torch.allclose(g_expl, g_auto, rtol=1e-8, atol=1e-6)
