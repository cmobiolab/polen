"""The PyTorch AMOEBA electrostatics (ref_torch) reproduces the OpenMM AmoebaMultipoleForce.

System: ACE-LYS-ASP-SER-NME + 11 waters (90 atoms), no cutoff, Reference platform.
"""

import numpy as np
import pytest
import torch

from openmm_systems import build_amoeba, multipole_energy_forces
from ref_torch import amoeba_elec as ae
from ref_torch.amoeba_elec import BISECTOR, NO_AXIS_TYPE, THREE_FOLD, Z_BISECT, Z_ONLY, Z_THEN_X


def torch_energy_forces(pos, prm, polarization='mutual'):
    x = torch.tensor(pos, requires_grad=True)
    e, terms = ae.electrostatic_energy(x, prm, polarization=polarization, return_terms=True)
    e.backward()
    return e.item(), -x.grad.numpy(), terms


@pytest.fixture(scope='module')
def mutual():
    system, force, ctx, pos = build_amoeba(polarization='mutual', epsilon=1e-10)
    return force, ctx, pos, ae.from_openmm(force)


def test_lab_frame_dipoles(mutual):
    force, ctx, pos, prm = mutual
    mu, _ = ae.lab_frame_multipoles(torch.tensor(pos), prm)
    ref = np.array(force.getLabFramePermanentDipoles(ctx))
    assert np.abs(mu.numpy() - ref).max() < 1e-14


def test_chiral_flip_mirror_image(mutual):
    """Mirroring the system (x -> -x), the ZThenX chiral centers flip mu_y, Q_xy and Q_yz."""
    force, ctx, pos, prm = mutual
    chiral = torch.as_tensor((prm.axis_type == Z_THEN_X) & (prm.atom_y >= 0))
    mirror = pos * np.array([-1.0, 1.0, 1.0])
    assert chiral.sum() > 0
    assert (ae.chiral_flip(torch.tensor(pos), prm)[chiral] > 0).all()
    assert (ae.chiral_flip(torch.tensor(mirror), prm)[chiral] < 0).all()

    ctx.setPositions(mirror)
    try:
        mu, _ = ae.lab_frame_multipoles(torch.tensor(mirror), prm)
        assert np.abs(mu.numpy() - np.array(force.getLabFramePermanentDipoles(ctx))).max() < 1e-14
        e_ref, f_ref = multipole_energy_forces(ctx)
        e, f, _ = torch_energy_forces(mirror, prm)
        assert abs(e - e_ref) < 1e-8
        assert np.abs(f - f_ref).max() < 1e-6
    finally:
        ctx.setPositions(pos)


def test_induced_dipoles(mutual):
    force, ctx, pos, prm = mutual
    _, _, terms = torch_energy_forces(pos, prm)
    ref = np.array(force.getInducedDipoles(ctx))
    assert np.abs(terms['mu_ind'].detach().numpy() - ref).max() < 1e-11


@pytest.mark.parametrize('polarization,e_tol,f_tol', [('direct', 1e-9, 1e-9), ('mutual', 1e-8, 1e-6)])
def test_energy_forces(polarization, e_tol, f_tol):
    _, force, ctx, pos = build_amoeba(polarization=polarization, epsilon=1e-10)
    prm = ae.from_openmm(force)
    e_ref, f_ref = multipole_energy_forces(ctx)
    e, f, _ = torch_energy_forces(pos, prm, polarization)
    assert abs(e - e_ref) < e_tol
    assert np.abs(f - f_ref).max() < f_tol


def _atom(top, res_name, atom_name):
    return next(a.index for a in top.atoms() if a.residue.name == res_name and a.name == atom_name)


# Local multipoles with the symmetry each axis type assumes (as in Tinker):
# ZOnly/ThreeFold -> axial symmetry around z; ZBisect -> symmetry in the xz plane.
AXIAL = ([0.0, 0.0, 0.006], [-1.5e-4, 0, 0, 0, -1.5e-4, 0, 0, 0, 3e-4])
PLANAR = ([0.004, 0.0, 0.006], [1e-4, 0, -3e-5, 0, -4e-4, 0, -3e-5, 0, 3e-4])


def _set_axis(force, i, axis, z=-1, x=-1, y=-1, multipoles=AXIAL):
    p = force.getMultipoleParameters(i)
    p[3], p[4], p[5], p[6] = axis, z, x, y
    p[1], p[2] = multipoles
    force.setMultipoleParameters(i, *p)


def edit_axis_types(force, top):
    """Force the axis types the peptide does not use: ZBisect, ZOnly and NoAxisType."""
    _set_axis(force, _atom(top, 'LYS', 'CA'), Z_BISECT, _atom(top, 'LYS', 'N'),
              _atom(top, 'LYS', 'C'), _atom(top, 'LYS', 'CB'), multipoles=PLANAR)
    _set_axis(force, _atom(top, 'ASP', 'OD1'), Z_ONLY, _atom(top, 'ASP', 'CG'))
    _set_axis(force, _atom(top, 'ASP', 'HB2'), NO_AXIS_TYPE)


def edit_three_fold(force, top):
    _set_axis(force, _atom(top, 'LYS', 'NZ'), THREE_FOLD, _atom(top, 'LYS', 'HZ1'),
              _atom(top, 'LYS', 'HZ2'), _atom(top, 'LYS', 'HZ3'))


@pytest.mark.parametrize('polarization', ['direct', 'mutual'])
def test_all_axis_types(polarization):
    _, force, ctx, pos = build_amoeba(polarization=polarization, epsilon=1e-10, edit=edit_axis_types)
    prm = ae.from_openmm(force)
    assert {Z_THEN_X, BISECTOR, Z_BISECT, Z_ONLY, NO_AXIS_TYPE} <= set(prm.axis_type)
    mu, _ = ae.lab_frame_multipoles(torch.tensor(pos), prm)
    assert np.abs(mu.numpy() - np.array(force.getLabFramePermanentDipoles(ctx))).max() < 1e-14
    e_ref, f_ref = multipole_energy_forces(ctx)
    e, f, _ = torch_energy_forces(pos, prm, polarization)
    assert abs(e - e_ref) < 1e-8
    assert np.abs(f - f_ref).max() < 1e-6


def _openmm_fd_forces(ctx, pos, atom, h=1e-6):
    fd = np.zeros(3)
    for c in range(3):
        pp, pm = pos.copy(), pos.copy()
        pp[atom, c] += h
        pm[atom, c] -= h
        ctx.setPositions(pp)
        ep = multipole_energy_forces(ctx)[0]
        ctx.setPositions(pm)
        em = multipole_energy_forces(ctx)[0]
        fd[c] = -(ep - em) / (2 * h)
    ctx.setPositions(pos)
    return fd


def test_three_fold_matches_openmm_energy_gradient():
    """ThreeFold: the energy matches OpenMM and the forces (autograd) are its gradient."""
    _, force, ctx, pos = build_amoeba(polarization='direct', edit=edit_three_fold)
    prm = ae.from_openmm(force)
    e_ref, _ = multipole_energy_forces(ctx)
    e, f, _ = torch_energy_forces(pos, prm, 'direct')
    assert abs(e - e_ref) < 1e-8
    sites = np.where(prm.axis_type == THREE_FOLD)[0]
    atoms = set(sites) | set(prm.atom_z[sites]) | set(prm.atom_x[sites]) | set(prm.atom_y[sites])
    for atom in atoms:
        assert np.abs(_openmm_fd_forces(ctx, pos, atom) - f[atom]).max() < 1e-3


@pytest.mark.xfail(strict=True, reason='OpenMM 8.6.1: the ThreeFold forces (Reference and CUDA) are not '
                                       '-grad of its own energy (fixed in the MLPol plugin)')
def test_three_fold_openmm_forces():
    _, force, ctx, pos = build_amoeba(polarization='direct', edit=edit_three_fold)
    prm = ae.from_openmm(force)
    _, f_ref = multipole_energy_forces(ctx)
    _, f, _ = torch_energy_forces(pos, prm, 'direct')
    assert np.abs(f - f_ref).max() < 1e-6


def test_autograd_vs_finite_difference(mutual):
    """The autograd forces are the gradient of the energy (including the rotation of the local frames)."""
    _, _, pos, prm = mutual
    _, f, _ = torch_energy_forces(pos, prm)
    h = 1e-5
    rng = np.random.default_rng(0)
    for a in rng.choice(len(pos), 8, replace=False):
        for c in range(3):
            pp, pm = pos.copy(), pos.copy()
            pp[a, c] += h
            pm[a, c] -= h
            fd = -(ae.electrostatic_energy(torch.tensor(pp), prm)
                   - ae.electrostatic_energy(torch.tensor(pm), prm)).item() / (2 * h)
            assert abs(fd - f[a, c]) < 1e-5 * max(1.0, abs(f[a, c]))
