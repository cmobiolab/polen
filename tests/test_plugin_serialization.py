"""MLPol plugin: XML serialization of MLPolMultipoleForce (round trip), with and without layers."""

from pathlib import Path

import numpy as np
import openmm as mm
import pytest
from openmm import app, unit

from mlmmpol.layered import configure_layered_polarization
from openmm_systems import SYSTEMS

mlpolplugin = pytest.importorskip('mlpolplugin')
PLUGIN = Path(__file__).resolve().parents[1] / 'plugin' / 'install' / 'lib' / 'plugins' / 'libMLPolPluginReference.so'


@pytest.fixture(scope='module', autouse=True)
def load_plugin():
    mm.Platform.loadPluginLibrary(str(PLUGIN))


def make_system(layered):
    pdb = app.PDBFile(str(SYSTEMS / 'nma_shell.pdb'))
    top = pdb.topology
    ml = [a.index for a in top.atoms() if a.residue.name in ('ACE', 'NME')]
    system = app.ForceField('amoeba2018.xml').createSystem(top, nonbondedMethod=app.NoCutoff, polarization='mutual',
                                                           mutualInducedTargetEpsilon=1e-10)
    i = next(k for k, f in enumerate(system.getForces()) if isinstance(f, mm.AmoebaMultipoleForce))
    f = mlpolplugin.MLPolMultipoleForce(system.getForce(i))
    f.setMutualInducedMaxIterations(1000)
    if layered:
        configure_layered_polarization(f, top, ml, 0.30, 0.42)
    system.removeForce(i)
    f.setForceGroup(3)
    system.addForce(f)
    pos = np.load(Path(__file__).resolve().parents[1] / 'qm_ref' / 'nma_frames.npz')['positions_nm'][4]
    return system, pos


def ef(system, pos):
    ctx = mm.Context(system, mm.VerletIntegrator(1e-3), mm.Platform.getPlatformByName('Reference'))
    ctx.setPositions(pos)
    st = ctx.getState(getEnergy=True, getForces=True, groups={3})
    return st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole), \
        st.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)


@pytest.mark.parametrize('layered', [False, True], ids=['no_layers', 'layers'])
def test_roundtrip(layered):
    system, pos = make_system(layered)
    xml = mm.XmlSerializer.serialize(system)
    assert 'MLPolMultipoleForce' in xml and ('LayeredPolarization' in xml)
    assert ('rOut=' in xml) == layered
    system2 = mm.XmlSerializer.deserialize(xml)
    e1, f1 = ef(system, pos)
    e2, f2 = ef(system2, pos)
    assert e1 == pytest.approx(e2, abs=1e-10)
    assert np.abs(f1 - f2).max() < 1e-10
    # and the second serialization is identical to the first
    assert mm.XmlSerializer.serialize(system2) == xml


def test_deserialized_layered_uses_our_kernel():
    """The system read from XML keeps the layers (energy differs from plain mutual polarization)."""
    s_lay, pos = make_system(True)
    s_mut, _ = make_system(False)
    e_lay = ef(mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(s_lay)), pos)[0]
    e_mut = ef(s_mut, pos)[0]
    assert abs(e_lay - e_mut) > 0.5
