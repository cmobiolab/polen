"""Smooth mutual -> direct switch (ref_torch/switching.py): limits and forces."""

from dataclasses import replace

import numpy as np
import torch

from openmm_systems import build_amoeba
from ref_torch import amoeba_elec as ae
from ref_torch.switching import site_weights, smooth_step, switched_u_scale


def test_smooth_step_limits_and_smoothness():
    d = torch.linspace(0.0, 1.0, 2001, dtype=torch.float64, requires_grad=True)
    s = smooth_step(d, 0.4, 0.6)
    assert s[d < 0.4].eq(1).all() and s[d > 0.6].eq(0).all()
    g = torch.autograd.grad(s.sum(), d, create_graph=True)[0]
    assert g.abs().max() < 2.0 / 0.2                          # bounded derivative
    assert abs(g[(d - 0.4).abs() < 1e-9].item()) < 1e-6 and abs(g[(d - 0.6).abs() < 1e-9].item()) < 1e-6


def _setup():
    system, force, ctx, pos = build_amoeba('nma_shell.pdb')
    prm = ae.from_openmm(force)
    from openmm import app
    from openmm_systems import SYSTEMS
    top = app.PDBFile(str(SYSTEMS / 'nma_shell.pdb')).topology
    ml = torch.as_tensor([a.index for a in top.atoms() if a.residue.name in ('ACE', 'NME')])
    waters = [r for r in top.residues() if r.name == 'HOH']
    mol_ref = torch.as_tensor([next(a.index for a in r.atoms() if a.name == 'O') for r in waters])
    mol_of_site = torch.full((prm.n,), -1, dtype=torch.long)
    for m, r in enumerate(waters):
        for a in r.atoms():
            mol_of_site[a.index] = m
    return (system, ctx), prm, pos, ml, mol_ref, mol_of_site


def test_limits_reproduce_mutual_and_direct():
    keep, prm, pos, ml, mol_ref, mol_of_site = _setup()
    x = torch.tensor(pos)
    e_mut = ae.electrostatic_energy(x, prm).item()
    s_all = site_weights(x, ml, mol_ref, mol_of_site, 10.0, 11.0)      # all inside: mutual
    assert abs(ae.electrostatic_energy(x, replace(prm, u_scale=switched_u_scale(prm.u_scale, s_all))).item() - e_mut) < 1e-10
    s_none = site_weights(x, ml, mol_ref, mol_of_site, 0.0, 1e-3)      # all waters outside: direct
    is_ml = torch.zeros(prm.n, dtype=torch.bool)
    is_ml[ml] = True
    u_dir = torch.where(is_ml[:, None] & is_ml[None, :], prm.u_scale, torch.zeros_like(prm.u_scale))
    e_dir = ae.electrostatic_energy(x, replace(prm, u_scale=u_dir)).item()
    assert abs(ae.electrostatic_energy(x, replace(prm, u_scale=switched_u_scale(prm.u_scale, s_none))).item() - e_dir) < 1e-10


def test_switched_forces_match_finite_differences():
    keep, prm, pos, ml, mol_ref, mol_of_site = _setup()

    def energy(xt):
        s = site_weights(xt, ml, mol_ref, mol_of_site, 0.35, 0.60)
        return ae.electrostatic_energy(xt, replace(prm, u_scale=switched_u_scale(prm.u_scale, s)))

    x = torch.tensor(pos, requires_grad=True)
    s = site_weights(x.detach(), ml, mol_ref, mol_of_site, 0.35, 0.60)
    assert ((s > 1e-3) & (s < 1 - 1e-3)).any()                        # there are waters in the transition
    f = -torch.autograd.grad(energy(x), x)[0].numpy()
    h = 1e-5
    part = [int(i) for i in torch.nonzero((s > 1e-3) & (s < 1 - 1e-3)).squeeze(-1)[:3]] + [0, 5]
    for a in part:
        for c in range(3):
            pp, pm = pos.copy(), pos.copy()
            pp[a, c] += h
            pm[a, c] -= h
            fd = -(energy(torch.tensor(pp)) - energy(torch.tensor(pm))).item() / (2 * h)
            assert abs(fd - f[a, c]) < 1e-5 * max(1.0, abs(fd))
