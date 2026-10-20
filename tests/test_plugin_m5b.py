"""MLPol plugin, milestone M5b: ThreeFold fixed on the GPU (CUDA == plugin Reference, which is -grad of the energy)."""

from pathlib import Path

import numpy as np
import openmm as mm
import pytest
from openmm import unit

from openmm_systems import build_amoeba
import test_amoeba_vs_openmm as T
from test_plugin_m2 import GENERIC, three_fold

mlpolplugin = pytest.importorskip('mlpolplugin')
LIB = Path(__file__).resolve().parents[1] / 'plugin' / 'install' / 'lib' / 'plugins'


def cuda_available():
    try:
        mm.Platform.getPlatformByName('CUDA')
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not cuda_available(), reason='no CUDA')


@pytest.fixture(scope='module', autouse=True)
def load_plugin():
    mm.Platform.loadPluginLibrary(str(LIB / 'libMLPolPluginReference.so'))
    mm.Platform.loadPluginLibrary(str(LIB / 'libMLPolPluginCUDA.so'))


def plugin_energy_forces(system, pos, platform):
    s = mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(system))
    i = next(k for k, f in enumerate(s.getForces()) if isinstance(f, mm.AmoebaMultipoleForce))
    f = mlpolplugin.MLPolMultipoleForce(s.getForce(i))
    s.removeForce(i)
    for g in s.getForces():
        g.setForceGroup(0)
    f.setForceGroup(1)
    s.addForce(f)
    props = {'Precision': 'double'} if platform == 'CUDA' else {}
    ctx = mm.Context(s, mm.VerletIntegrator(1e-3), mm.Platform.getPlatformByName(platform), props)
    ctx.setPositions(pos)
    st = ctx.getState(getEnergy=True, getForces=True, groups={1})
    return st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole), \
        st.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)


@pytest.mark.parametrize('multipoles', [T.AXIAL, GENERIC], ids=['axial', 'generico'])
@pytest.mark.parametrize('polarization', ['direct', 'mutual'])
def test_three_fold_cuda_equals_reference(multipoles, polarization):
    system, force, ctx0, pos = build_amoeba(polarization=polarization, epsilon=1e-10, edit=three_fold(multipoles))
    e0, f0 = plugin_energy_forces(system, pos, 'Reference')
    e1, f1 = plugin_energy_forces(system, pos, 'CUDA')
    assert abs(e1 - e0) < 1e-6 * max(1.0, abs(e0))
    assert np.abs(f1 - f0).max() < 1e-3        # float parameters on the GPU; the original AMOEBA is off by ~9-11 kJ/mol/nm here


def test_other_axis_types_cuda_equals_reference():
    system, force, ctx0, pos = build_amoeba(polarization='mutual', epsilon=1e-10, edit=T.edit_axis_types)
    e0, f0 = plugin_energy_forces(system, pos, 'Reference')
    e1, f1 = plugin_energy_forces(system, pos, 'CUDA')
    assert abs(e1 - e0) < 1e-6 * max(1.0, abs(e0))
    assert np.abs(f1 - f0).max() < 1e-3
