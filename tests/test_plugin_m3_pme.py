"""MLPol plugin, milestone M3 with PME: layers (mutual inside the sphere, direct outside), Reference."""

from pathlib import Path

import numpy as np
import openmm as mm
import pytest
from openmm import app, unit

from mlmmpol.layered import configure_layered_polarization
from openmm_systems import SYSTEMS

mlpolplugin = pytest.importorskip('mlpolplugin')
PLUGIN = Path(__file__).resolve().parents[1] / 'plugin' / 'install' / 'lib' / 'plugins' / 'libMLPolPluginReference.so'
FF = app.ForceField('amoeba2018.xml')


@pytest.fixture(scope='module', autouse=True)
def load_plugin():
    mm.Platform.loadPluginLibrary(str(PLUGIN))


def solvated(solute_top, solute_pos, box=3.0, seed=1):
    mod = app.Modeller(solute_top, solute_pos)
    mod.addSolvent(FF, model='tip3p', boxSize=mm.Vec3(box, box, box) * unit.nanometer)
    return mod


def build(mod, polarization, layered=None):
    system = FF.createSystem(mod.topology, nonbondedMethod=app.PME, nonbondedCutoff=0.7 * unit.nanometer,
                             polarization=polarization, mutualInducedTargetEpsilon=1e-9)
    i = next(k for k, f in enumerate(system.getForces()) if isinstance(f, mm.AmoebaMultipoleForce))
    amoeba = system.getForce(i)
    amoeba.setMutualInducedMaxIterations(1000)
    for g in system.getForces():
        g.setForceGroup(0)
    if layered is not None:
        f = mlpolplugin.MLPolMultipoleForce(amoeba)
        configure_layered_polarization(f, mod.topology, *layered)
        system.removeForce(i)
        f.setForceGroup(1)
        system.addForce(f)
    else:
        amoeba.setForceGroup(1)
    ctx = mm.Context(system, mm.VerletIntegrator(1e-3), mm.Platform.getPlatformByName('Reference'))
    return system, ctx


def ef(ctx, pos):
    ctx.setPositions(pos)
    st = ctx.getState(getEnergy=True, getForces=True, groups={1})
    return st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole), \
        st.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)


def test_limit_single_ion_equals_direct_pme():
    """Single ion in the mutual region (no partners): layers == OpenMM PME direct polarization."""
    top = app.Topology()
    chain = top.addChain()
    res = top.addResidue('NA', chain)
    top.addAtom('NA', app.element.sodium, res)
    mod = solvated(top, [mm.Vec3(1.5, 1.5, 1.5)] * unit.nanometer, box=2.5)
    pos = mod.positions
    _, ctx_dir = build(mod, 'direct')
    _, ctx_lay = build(mod, 'mutual', layered=([0], 0.001, 0.002))
    e0, f0 = ef(ctx_dir, pos)
    e1, f1 = ef(ctx_lay, pos)
    assert abs(e1 - e0) < 1e-8 * max(1, abs(e0))
    assert np.abs(f1 - f0).max() < 1e-6


@pytest.fixture(scope='module')
def nma_box():
    pdb = app.PDBFile(str(SYSTEMS / 'nma_shell.pdb'))
    mod = app.Modeller(pdb.topology, pdb.positions)
    mod.delete([r for r in pdb.topology.residues() if r.name == 'HOH'])
    ml = [a.index for a in mod.topology.atoms()]
    box = solvated(mod.topology, mod.positions, box=2.6)
    return box, ml


def test_layered_pme_between_direct_and_mutual(nma_box):
    mod, ml = nma_box
    pos = mod.positions
    e_dir = ef(build(mod, 'direct')[1], pos)[0]
    e_mut = ef(build(mod, 'mutual')[1], pos)[0]
    e_lay = ef(build(mod, 'mutual', layered=(ml, 0.35, 0.55))[1], pos)[0]
    assert min(e_dir, e_mut) < e_lay < max(e_dir, e_mut)


def test_layered_pme_forces_are_energy_gradient(nma_box):
    mod, ml = nma_box
    pos = np.array(mod.positions.value_in_unit(unit.nanometer))
    _, ctx = build(mod, 'mutual', layered=(ml, 0.35, 0.55))
    _, f = ef(ctx, pos)
    # atoms: two from the ML region and the O of the waters closest to the switching window
    O = [a.index for a in mod.topology.atoms() if a.residue.name == 'HOH' and a.name == 'O']
    dmin = {o: np.min(np.linalg.norm(pos[ml] - pos[o], axis=1)) for o in O}
    window = sorted(O, key=lambda o: abs(dmin[o] - 0.45))[:3]
    h = 1e-6
    for a in list(ml[:2]) + window:
        for c in range(3):
            pp, pm = pos.copy(), pos.copy()
            pp[a, c] += h
            pm[a, c] -= h
            fd = -(ef(ctx, pp)[0] - ef(ctx, pm)[0]) / (2 * h)
            assert abs(fd - f[a, c]) < 5e-3 * max(1.0, abs(fd) / 100), (a, c, fd, f[a, c])
