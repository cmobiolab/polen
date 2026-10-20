"""MLPol plugin, milestone M5a: CUDA platform (copy of the common AMOEBA kernel) == OpenMM AmoebaMultipoleForce on CUDA."""

from pathlib import Path

import numpy as np
import openmm as mm
import pytest
from openmm import app, unit

from openmm_systems import SYSTEMS

mlpolplugin = pytest.importorskip('mlpolplugin')
LIB = Path(__file__).resolve().parents[1] / 'plugin' / 'install' / 'lib' / 'plugins'
FF = app.ForceField('amoeba2018.xml')


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


def systems():
    pdb = app.PDBFile(str(SYSTEMS / 'nma_shell.pdb'))
    yield 'nma_nocutoff', pdb.topology, pdb.positions, dict(nonbondedMethod=app.NoCutoff)
    mod = app.Modeller(pdb.topology, pdb.positions)
    mod.delete([r for r in pdb.topology.residues() if r.name == 'HOH'])
    mod.addSolvent(FF, model='tip3p', boxSize=mm.Vec3(2.4, 2.4, 2.4) * unit.nanometer)
    yield 'nma_pme', mod.topology, mod.positions, dict(nonbondedMethod=app.PME, nonbondedCutoff=0.7 * unit.nanometer)


def energy_forces(topology, positions, kw, polarization, plugin, platform):
    system = FF.createSystem(topology, polarization=polarization, mutualInducedTargetEpsilon=1e-8, **kw)
    i = next(k for k, f in enumerate(system.getForces()) if isinstance(f, mm.AmoebaMultipoleForce))
    for f in system.getForces():
        f.setForceGroup(0)
    if plugin:
        f = mlpolplugin.MLPolMultipoleForce(system.getForce(i))
        system.removeForce(i)
        system.addForce(f)
    else:
        f = system.getForce(i)
    f.setForceGroup(1)
    props = {'Precision': 'double'} if platform == 'CUDA' else {}
    ctx = mm.Context(system, mm.VerletIntegrator(1e-3), mm.Platform.getPlatformByName(platform), props)
    ctx.setPositions(positions)
    st = ctx.getState(getEnergy=True, getForces=True, groups={1})
    return st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole), \
        st.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)


@pytest.mark.parametrize('polarization', ['mutual', 'direct', 'extrapolated'])
@pytest.mark.parametrize('case', list(systems()), ids=lambda c: c[0])
def test_cuda_plugin_equals_openmm_cuda(case, polarization):
    _, top, pos, kw = case
    e0, f0 = energy_forces(top, pos, kw, polarization, plugin=False, platform='CUDA')
    e1, f1 = energy_forces(top, pos, kw, polarization, plugin=True, platform='CUDA')
    assert abs(e1 - e0) < 1e-8 * max(1.0, abs(e0))
    assert np.abs(f1 - f0).max() < 1e-6


@pytest.mark.parametrize('case', list(systems()), ids=lambda c: c[0])
def test_cuda_plugin_close_to_reference_plugin(case):
    _, top, pos, kw = case
    e0, f0 = energy_forces(top, pos, kw, 'mutual', plugin=True, platform='Reference')
    e1, f1 = energy_forces(top, pos, kw, 'mutual', plugin=True, platform='CUDA')
    assert abs(e1 - e0) < 1e-5 * max(1.0, abs(e0))
    assert np.abs(f1 - f0).max() < 1e-4 * np.abs(f0).max()      # PME grid (legal FFT dimensions) and SCF
