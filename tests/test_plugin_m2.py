"""MLPol plugin, milestone M2: local-frame forces from the exact chain rule (correct ThreeFold)."""

from pathlib import Path

import numpy as np
import openmm as mm
import pytest
import torch
from openmm import unit

from openmm_systems import build_amoeba
from ref_torch import amoeba_elec as ae
import test_amoeba_vs_openmm as T

mlpolplugin = pytest.importorskip('mlpolplugin')
PLUGIN = Path(__file__).resolve().parents[1] / 'plugin' / 'install' / 'lib' / 'plugins' / 'libMLPolPluginReference.so'
GENERIC = ([0.004, -0.003, 0.006], [1e-4, 2e-5, -3e-5, 2e-5, -4e-4, 5e-5, -3e-5, 5e-5, 3e-4])


@pytest.fixture(scope='module', autouse=True)
def load_plugin():
    mm.Platform.loadPluginLibrary(str(PLUGIN))


def plugin_context(system):
    s = mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(system))
    i = next(k for k, f in enumerate(s.getForces()) if isinstance(f, mm.AmoebaMultipoleForce))
    f = mlpolplugin.MLPolMultipoleForce(s.getForce(i))
    s.removeForce(i)
    for g in s.getForces():
        g.setForceGroup(0)
    f.setForceGroup(1)
    s.addForce(f)
    return s, mm.Context(s, mm.VerletIntegrator(1e-3), mm.Platform.getPlatformByName('Reference'))


def energy_forces(ctx, pos):
    ctx.setPositions(pos)
    st = ctx.getState(getEnergy=True, getForces=True, groups={1})
    return st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole), \
        st.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)


def three_fold(multipoles):
    def edit(force, top):
        T._set_axis(force, T._atom(top, 'LYS', 'NZ'), ae.THREE_FOLD, T._atom(top, 'LYS', 'HZ1'),
                    T._atom(top, 'LYS', 'HZ2'), T._atom(top, 'LYS', 'HZ3'), multipoles=multipoles)
    return edit


@pytest.mark.parametrize('multipoles', [T.AXIAL, GENERIC], ids=['axial', 'generico'])
@pytest.mark.parametrize('polarization', ['direct', 'mutual'])
def test_three_fold_forces_exact(multipoles, polarization):
    system, force, ctx0, pos = build_amoeba(polarization=polarization, epsilon=1e-10, edit=three_fold(multipoles))
    s, ctx = plugin_context(system)
    e, f = energy_forces(ctx, pos)
    prm = ae.from_openmm(force)
    x = torch.tensor(pos, requires_grad=True)
    e_t = ae.electrostatic_energy(x, prm, polarization=polarization)
    f_t = -torch.autograd.grad(e_t, x)[0].numpy()
    assert abs(e - e_t.item()) < 1e-8
    assert np.abs(f - f_t).max() < 1e-6                  # the original AMOEBA is off by ~9-11 kJ/mol/nm here
    # and the plugin forces are the gradient of the plugin energy (finite differences)
    nz = [a.index for a in __import__('openmm.app', fromlist=['PDBFile']).PDBFile(
        str(Path(__file__).resolve().parents[1] / 'systems' / 'peptide_shell.pdb')).topology.atoms()
          if a.residue.name == 'LYS' and a.name in ('NZ', 'HZ1', 'HZ2', 'HZ3')]
    h = 1e-6
    for a in nz:
        for c in range(3):
            pp, pm = pos.copy(), pos.copy()
            pp[a, c] += h
            pm[a, c] -= h
            fd = -(energy_forces(ctx, pp)[0] - energy_forces(ctx, pm)[0]) / (2 * h)
            assert abs(fd - f[a, c]) < 1e-3


def test_other_axis_types_unchanged():
    """For the other axis types, the plugin stays identical to AMOEBA."""
    system, force, ctx0, pos = build_amoeba(polarization='mutual', epsilon=1e-10, edit=T.edit_axis_types)
    s, ctx = plugin_context(system)
    e, f = energy_forces(ctx, pos)
    st = ctx0.getState(getEnergy=True, getForces=True, groups={1})
    assert abs(e - st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)) < 1e-8
    assert np.abs(f - st.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)).max() < 1e-7
