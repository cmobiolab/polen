"""ML region in the AMOEBA electrostatics (masking, ML parameters, phi/E interface, forces).

Test ML regions: (a) the first two waters of the peptide + waters system;
(b) solvated NMA (ACE–NME), which has intramolecular structure (1-2 to 1-5 pairs, several polarization groups).
"""

import numpy as np
import pytest
import torch

from mlmmpol.amoeba_mixed import mask_multipoles as mask_ml_in_openmm
from openmm_systems import build_amoeba, multipole_energy_forces
from ref_torch import amoeba_elec as ae
from ref_torch.mlmm_pol import AMOEBA_THOLE, mlmm_electrostatic_energy, potential_field_at_ml

SYSTEMS_ML = {
    'waters': ('peptide_shell.pdb', lambda top: [r for r in top.residues() if r.name == 'HOH'][:2]),
    'nma': ('nma_shell.pdb', lambda top: [r for r in top.residues() if r.name in ('ACE', 'NME')]),
}


def ml_atoms(top, case):
    return np.array([a.index for r in SYSTEMS_ML[case][1](top) for a in r.atoms()])


def inject_ml_parameters(force, ml, q, mu, alpha, thole=AMOEBA_THOLE):
    """Put q, mu (lab frame), alpha on the OpenMM ML atoms (NoAxisType, zero quadrupole)."""
    for k, i in enumerate(ml):
        p = force.getMultipoleParameters(int(i))
        p[0] = float(q[k])
        p[1] = [float(v) for v in mu[k]]
        p[2] = [0.0] * 9
        p[3], p[4], p[5], p[6] = ae.NO_AXIS_TYPE, -1, -1, -1
        p[7] = thole
        p[8] = float(alpha[k]) ** (1 / 6)
        p[9] = float(alpha[k])
        force.setMultipoleParameters(int(i), *p)


class ToyEmbedding(torch.nn.Module):
    """Differentiable, equivariant model mimicking the ML head: q, mu, alpha as functions of R_ML.

    q conserves the total charge of the region; mu is a sum of interatomic vectors (l = 1).
    """

    def __init__(self, q0, alpha0):
        super().__init__()
        self.q0 = torch.as_tensor(q0, dtype=torch.float64)
        self.alpha0 = torch.as_tensor(alpha0, dtype=torch.float64)

    def forward(self, r):
        d = r[:, None, :] - r[None, :, :]
        dist = torch.sqrt((d**2).sum(-1) + torch.eye(len(r), dtype=r.dtype))
        w = torch.exp(-dist / 0.2) * (1 - torch.eye(len(r), dtype=r.dtype))
        q = self.q0 + 0.05 * torch.tanh(w.sum(1) - 0.5)
        q = q - (q.sum() - self.q0.sum()) / len(q)
        mu = 0.01 * (w[..., None] * (-d)).sum(1)
        alpha = self.alpha0 * (1 + 0.2 * torch.tanh(w.sum(1)))
        return q, mu, alpha


@pytest.fixture(scope='module', params=list(SYSTEMS_ML))
def base(request):
    from openmm import app
    from openmm_systems import SYSTEMS
    pdb_name = SYSTEMS_ML[request.param][0]
    _, force, ctx, pos = build_amoeba(pdb_name, polarization='mutual', epsilon=1e-10)
    top = app.PDBFile(str(SYSTEMS / pdb_name)).topology
    ml = ml_atoms(top, request.param)
    prm = ae.from_openmm(force)
    model = ToyEmbedding(prm.charge[ml], prm.polarity[ml])
    return pos, prm, ml, model, pdb_name


def test_level1_masking_matches_openmm(base):
    """Level 1 (AMOEBA multipoles on the ML region): torch masks == OpenMM covalent maps."""
    pos, prm, ml, _, pdb_name = base
    _, force, ctx, _ = build_amoeba(pdb_name, epsilon=1e-10, edit=lambda f, t: mask_ml_in_openmm(f, ml))
    e_ref, f_ref = multipole_energy_forces(ctx)
    x = torch.tensor(pos, requires_grad=True)
    e = mlmm_electrostatic_energy(x, prm, ml)
    e.backward()
    assert abs(e.item() - e_ref) < 1e-8
    assert np.abs(-x.grad.numpy() - f_ref).max() < 1e-6


def test_ml_parameters_match_openmm(base):
    """ML parameters injected into OpenMM: energy and forces at fixed parameters agree."""
    pos, prm, ml, model, pdb_name = base
    with torch.no_grad():
        q, mu, alpha = model(torch.tensor(pos[ml]))

    def edit(force, top):
        mask_ml_in_openmm(force, ml)
        inject_ml_parameters(force, ml, q.numpy(), mu.numpy(), alpha.numpy())

    _, force, ctx, _ = build_amoeba(pdb_name, epsilon=1e-10, traceless=True, edit=edit)
    e_ref, f_ref = multipole_energy_forces(ctx)
    x = torch.tensor(pos, requires_grad=True)
    e = mlmm_electrostatic_energy(x, prm, ml, q, mu, alpha)
    e.backward()
    assert abs(e.item() - e_ref) < 1e-8
    assert np.abs(-x.grad.numpy() - f_ref).max() < 1e-6


def test_potential_field_interface(base):
    """Explicit phi and E == dE/dq and -dE/dmu from autograd."""
    pos, prm, ml, model, pdb_name = base
    with torch.no_grad():
        q, mu, alpha = model(torch.tensor(pos[ml]))
    q, mu = q.clone().requires_grad_(), mu.clone().requires_grad_()
    x = torch.tensor(pos)
    e, terms = mlmm_electrostatic_energy(x, prm, ml, q, mu, alpha, return_terms=True)
    dq, dmu = torch.autograd.grad(e, (q, mu))
    phi, E = potential_field_at_ml(x, terms, ml)
    assert torch.allclose(phi, dq, rtol=1e-9, atol=1e-9)
    assert torch.allclose(E, -dmu, rtol=1e-9, atol=1e-9)


def test_chain_rule_forces(base):
    """Forces assembled through the interface (dE/dR|params + VJP through the model) == full autograd == FD."""
    pos, prm, ml, model, pdb_name = base
    ml_t = torch.as_tensor(ml)

    def energy(x):
        q, mu, alpha = model(x[ml_t])
        return mlmm_electrostatic_energy(x, prm, ml, q, mu, alpha)

    # 1) autograd through everything
    x = torch.tensor(pos, requires_grad=True)
    f_auto = -torch.autograd.grad(energy(x), x)[0]

    # 2) plugin-style assembly: parameters as leaves, phi/E gradients from the engine, VJP through the model
    x = torch.tensor(pos, requires_grad=True)
    q, mu, alpha = model(x[ml_t])
    qd, mud, ad = (t.detach().requires_grad_() for t in (q, mu, alpha))
    e, terms = mlmm_electrostatic_energy(x, prm, ml, qd, mud, ad, return_terms=True)
    dE_dR, dE_dalpha = torch.autograd.grad(e, (x, ad), retain_graph=True)
    phi, E = potential_field_at_ml(x.detach(), terms, ml)
    vjp = torch.autograd.grad((q, mu, alpha), x, grad_outputs=(phi.detach(), -E.detach(), dE_dalpha))[0]
    f_chain = -(dE_dR + vjp)
    assert torch.allclose(f_chain, f_auto, rtol=1e-9, atol=1e-8)

    # 3) finite differences on the ML atoms and on some MM atoms
    h = 1e-5
    for a in list(ml[:3]) + [0, 30]:
        for c in range(3):
            pp, pm = pos.copy(), pos.copy()
            pp[a, c] += h
            pm[a, c] -= h
            fd = -(energy(torch.tensor(pp)) - energy(torch.tensor(pm))).item() / (2 * h)
            assert abs(fd - f_auto[a, c].item()) < 1e-5 * max(1.0, abs(fd))


def test_masking_removes_intra_ml_permanent_polarization(base):
    """In the isolated ML region (no MM), only the induced–induced coupling remains: zero energy."""
    pos, prm, ml, model, pdb_name = base
    from dataclasses import replace
    sub = torch.as_tensor(ml)
    iso = replace(prm, charge=prm.charge[sub], dipole=prm.dipole[sub], quadrupole=prm.quadrupole[sub],
                  axis_type=np.full(len(ml), ae.NO_AXIS_TYPE), atom_z=np.full(len(ml), -1),
                  atom_x=np.full(len(ml), -1), atom_y=np.full(len(ml), -1), thole=prm.thole[sub],
                  damping=prm.damping[sub], polarity=prm.polarity[sub],
                  m_scale=prm.m_scale[sub][:, sub], p_scale=prm.p_scale[sub][:, sub],
                  d_scale=prm.d_scale[sub][:, sub], u_scale=prm.u_scale[sub][:, sub])
    with torch.no_grad():
        q, mu, alpha = model(torch.tensor(pos[ml]))
    e = mlmm_electrostatic_energy(torch.tensor(pos[ml]), iso, np.arange(len(ml)), q, mu, alpha)
    assert abs(e.item()) < 1e-12
