"""Masked pairs beyond the cutoff (ML region larger than the real-space cutoff, e.g. Mg²⁺·ATP + waters).

With PME, a masked pair (scale s < 1) needs the real-space term even beyond the cutoff: erfc - (1-s)*nu, which added to
the reciprocal part gives s*nu. Without it, the distant pair would interact only through the reciprocal part (≈ nu),
double counted with the ML potential.
Test: ML region of two waters ~0.9 nm apart, masked as in ML/MM-AMOEBA (mask_multipoles); energy and forces with a
0.7 nm cutoff (pair beyond the cutoff) and a 1.15 nm cutoff (pair inside) must agree, on Reference and CUDA, and pure
AMOEBA (no mask) still equals the OpenMM AmoebaMultipoleForce.
"""

import os
from pathlib import Path

import numpy as np
import openmm as mm
import pytest
from openmm import app, unit

from mlmmpol.amoeba_mixed import mask_multipoles
from openmm_systems import SYSTEMS

mlpolplugin = pytest.importorskip('mlpolplugin')
LIB = Path(os.environ.get('MLPOL_PLUGIN_DIR', Path(__file__).resolve().parents[1] / 'plugin' / 'install' / 'lib' / 'plugins'))
FF = app.ForceField('amoeba2018.xml')


def platforms():
    out = ['Reference']
    try:
        mm.Platform.getPlatformByName('CUDA')
        out.append('CUDA')
    except Exception:
        pass
    return out


@pytest.fixture(scope='module', autouse=True)
def load_plugin():
    for name in ('libMLPolPluginReference.so', 'libMLPolPluginCUDA.so'):
        if (LIB / name).exists():
            mm.Platform.loadPluginLibrary(str(LIB / name))


@pytest.fixture(scope='module')
def water_box():
    pdb = app.PDBFile(str(SYSTEMS / 'nma_shell.pdb'))
    mod = app.Modeller(pdb.topology, pdb.positions)
    mod.delete(list(pdb.topology.residues()))
    mod.addSolvent(FF, model='tip3p', boxSize=mm.Vec3(2.4, 2.4, 2.4) * unit.nanometer)
    x = np.array(mod.positions.value_in_unit(unit.nanometer))
    box = 2.4
    waters = [[a.index for a in r.atoms()] for r in mod.topology.residues()]
    # water pair with O–O (minimum image) between 0.85 and 0.95 nm
    o = np.array([x[w[0]] for w in waters])
    for i in range(len(waters)):
        d = o - o[i]
        d -= box * np.round(d / box)
        r = np.linalg.norm(d, axis=1)
        j = np.where((r > 0.85) & (r < 0.95))[0]
        if len(j):
            return mod.topology, mod.positions, waters[i] + waters[j[0]]
    raise RuntimeError('no water pair found')


def energy_forces(topology, positions, ml, cutoff, platform, plugin=True, mask=True):
    system = FF.createSystem(topology, nonbondedMethod=app.PME, nonbondedCutoff=cutoff * unit.nanometer,
                             polarization='mutual', mutualInducedTargetEpsilon=1e-9)
    i = next(k for k, f in enumerate(system.getForces()) if isinstance(f, mm.AmoebaMultipoleForce))
    for f in system.getForces():
        f.setForceGroup(0)
    amf = system.getForce(i)
    amf.setEwaldErrorTolerance(1e-7)
    if mask:
        mask_multipoles(amf, ml)
    if plugin:
        f = mlpolplugin.MLPolMultipoleForce(amf)
        system.removeForce(i)
        system.addForce(f)
    else:
        f = amf
    f.setForceGroup(1)
    props = {'Precision': 'double'} if platform == 'CUDA' else {}
    ctx = mm.Context(system, mm.VerletIntegrator(1e-3), mm.Platform.getPlatformByName(platform), props)
    ctx.setPositions(positions)
    st = ctx.getState(getEnergy=True, getForces=True, groups={1})
    return st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole), \
        st.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)


@pytest.mark.parametrize('platform', platforms())
def test_masked_pair_beyond_cutoff(water_box, platform):
    top, pos, ml = water_box
    e_short, f_short = energy_forces(top, pos, ml, 0.7, platform)
    e_long, f_long = energy_forces(top, pos, ml, 1.15, platform)
    print(f'{platform}: E(0.7 nm) = {e_short:.5f}, E(1.15 nm) = {e_long:.5f}, '
          f'max |dF| = {np.abs(f_short - f_long).max():.2e} kJ/mol/nm')
    assert e_short == pytest.approx(e_long, abs=2e-3)
    assert np.abs(f_short - f_long).max() < 5e-2


@pytest.mark.parametrize('platform', platforms())
def test_unmasked_matches_openmm(water_box, platform):
    top, pos, ml = water_box
    e_p, f_p = energy_forces(top, pos, ml, 0.7, platform, plugin=True, mask=False)
    e_o, f_o = energy_forces(top, pos, ml, 0.7, platform, plugin=False, mask=False)
    assert e_p == pytest.approx(e_o, abs=1e-4)
    assert np.abs(f_p - f_o).max() < 1e-3


def ml_context(top, pos, ml, cutoff, platform):
    """'ml' mode (injected parameters) with the masked ML region, as in MuPE(ML) runs."""
    system = FF.createSystem(top, nonbondedMethod=app.PME, nonbondedCutoff=cutoff * unit.nanometer,
                             polarization='mutual', mutualInducedTargetEpsilon=1e-10)
    i = next(k for k, f in enumerate(system.getForces()) if isinstance(f, mm.AmoebaMultipoleForce))
    for f in system.getForces():
        f.setForceGroup(0)
    amf = system.getForce(i)
    amf.setEwaldErrorTolerance(1e-7)
    mask_multipoles(amf, ml)
    prm = ml_parameters(amf, ml)                    # before removing the force from the System (the object is freed)
    f = mlpolplugin.MLPolMultipoleForce(amf)
    f.setMutualInducedMaxIterations(1000)
    f.setMLRegion(ml, 0.39, 0.7)
    system.removeForce(i)
    system.addForce(f)
    f.setForceGroup(1)
    props = {'Precision': 'double'} if platform == 'CUDA' else {}
    ctx = mm.Context(system, mm.VerletIntegrator(1e-3), mm.Platform.getPlatformByName(platform), props)
    ctx.setPositions(pos)
    return f, ctx, prm


def ml_parameters(amf, ml, seed=0):
    rng = np.random.default_rng(seed)
    q = np.array([amf.getMultipoleParameters(a)[0].value_in_unit(unit.elementary_charge) for a in ml])
    al = np.array([amf.getMultipoleParameters(a)[-1].value_in_unit(unit.nanometer ** 3) for a in ml])
    mu = rng.normal(scale=0.003, size=(len(ml), 3))
    th = rng.normal(scale=0.0005, size=(len(ml), 3, 3))
    th = 0.5 * (th + th.transpose(0, 2, 1))
    th -= np.eye(3) * np.trace(th, axis1=1, axis2=2)[:, None, None] / 3
    return [q, mu.reshape(-1), th.reshape(-1), al]


@pytest.mark.parametrize('platform', platforms())
def test_ml_derivatives_masked_pair_beyond_cutoff(water_box, platform):
    """ML derivatives (phi, E, dE/dTheta, dE/dalpha) with an ML–ML pair beyond the cutoff = finite differences of the
    energy, and independent of the cutoff."""
    top, pos, ml = water_box
    out = {}
    for cutoff in (0.7, 1.15):
        f, ctx, prm = ml_context(top, pos, ml, cutoff, platform)

        def energy(p):
            f.setMLParametersInContext(ctx, *[x.tolist() for x in p])
            return ctx.getState(getEnergy=True, groups={1}).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)

        energy(prm)
        der = f.getMLDerivatives(ctx)
        phi, field = np.array(der['phi']), np.array(der['field']).reshape(-1, 3)
        dal = np.array(der['dAlpha'])
        out[cutoff] = (phi, field, np.array(der['dTheta']), dal)
        if cutoff == 0.7:
            worst = 0.0
            for k in range(len(ml)):
                pp, pm = [x.copy() for x in prm], [x.copy() for x in prm]
                pp[0][k] += 1e-4
                pm[0][k] -= 1e-4
                worst = max(worst, abs((energy(pp) - energy(pm)) / 2e-4 - phi[k]) / max(1.0, abs(phi[k])))
                for c in range(3):
                    pp, pm = [x.copy() for x in prm], [x.copy() for x in prm]
                    pp[1][3 * k + c] += 1e-5
                    pm[1][3 * k + c] -= 1e-5
                    worst = max(worst, abs((energy(pp) - energy(pm)) / 2e-5 + field[k, c]) / max(1.0, np.abs(field[k]).max()))
            print(f'{platform}: worst relative error phi/E vs finite differences = {worst:.1e}')
            # on CUDA the energy has ~1e-4 kJ/mol noise (float parameters on the GPU, as in OpenMM): the finite
            # differences fluctuate by ~1%; the derivatives themselves = Reference (~1e-6). The error fixed here was systematic (53%).
            assert worst < (1e-3 if platform == 'Reference' else 2e-2)
    for a, b in zip(out[0.7], out[1.15]):
        assert np.abs(a - b).max() < 2e-2 * max(1.0, np.abs(b).max()), np.abs(a - b).max()
