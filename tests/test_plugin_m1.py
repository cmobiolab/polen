"""MLPol plugin, milestone M1: MLPolMultipoleForce (Reference platform) reproduces AmoebaMultipoleForce."""

from pathlib import Path

import numpy as np
import openmm as mm
import pytest
from openmm import app, unit

mlpolplugin = pytest.importorskip('mlpolplugin')
PLUGIN = Path(__file__).resolve().parents[1] / 'plugin' / 'install' / 'lib' / 'plugins' / 'libMLPolPluginReference.so'


@pytest.fixture(scope='module', autouse=True)
def load_plugin():
    mm.Platform.loadPluginLibrary(str(PLUGIN))


def _swap(system):
    """Copy of the system with the AmoebaMultipoleForce replaced by an MLPolMultipoleForce."""
    s = mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(system))
    i = next(k for k, f in enumerate(s.getForces()) if isinstance(f, mm.AmoebaMultipoleForce))
    f = mlpolplugin.MLPolMultipoleForce(s.getForce(i))
    s.removeForce(i)
    s.addForce(f)
    return s, f


def _state(system, pos):
    for k, f in enumerate(system.getForces()):
        f.setForceGroup(1 if isinstance(f, mm.AmoebaMultipoleForce) else 0)
    ctx = mm.Context(system, mm.VerletIntegrator(1e-3), mm.Platform.getPlatformByName('Reference'))
    ctx.setPositions(pos)
    st = ctx.getState(getEnergy=True, getForces=True, groups={1})
    return ctx, st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole), \
        st.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)


@pytest.mark.parametrize('polarization', ['mutual', 'direct', 'extrapolated'])
def test_nocutoff_matches_amoeba(polarization):
    pdb = app.PDBFile(str(Path(__file__).resolve().parents[1] / 'systems' / 'peptide_shell.pdb'))
    system = app.ForceField('amoeba2018.xml').createSystem(pdb.topology, nonbondedMethod=app.NoCutoff,
                                                           polarization=polarization, mutualInducedTargetEpsilon=1e-8)
    new, f = _swap(system)
    ctx0, e0, f0 = _state(system, pdb.positions)
    ctx1, e1, f1 = _state(new, pdb.positions)
    assert abs(e1 - e0) < 1e-9 * max(1, abs(e0))
    assert np.abs(f1 - f0).max() < 1e-8
    mp0 = next(x for x in system.getForces() if isinstance(x, mm.AmoebaMultipoleForce))
    assert np.abs(np.array(f.getInducedDipoles(ctx1)) - np.array(mp0.getInducedDipoles(ctx0))).max() < 1e-12


def test_pme_matches_amoeba():
    ff = app.ForceField('amoeba2018.xml')
    top = app.Topology()
    mod = app.Modeller(top, [])
    mod.addSolvent(ff, model='tip3p', boxSize=mm.Vec3(2.0, 2.0, 2.0) * unit.nanometer)
    system = ff.createSystem(mod.topology, nonbondedMethod=app.PME, nonbondedCutoff=0.7 * unit.nanometer,
                             polarization='mutual', mutualInducedTargetEpsilon=1e-8)
    new, _ = _swap(system)
    _, e0, f0 = _state(system, mod.positions)
    _, e1, f1 = _state(new, mod.positions)
    assert abs(e1 - e0) < 1e-9 * max(1, abs(e0))
    assert np.abs(f1 - f0).max() < 1e-7
