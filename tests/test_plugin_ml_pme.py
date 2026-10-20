"""MLPol plugin, 'ml' mode with PME: ML derivatives (phi, E, dE/dTheta, dE/dalpha), head and penetration in a periodic box.

There is no periodic torch reference: everything is checked against finite differences of the plugin's own PME energy
(the energy is variational in the induced dipoles, so the total dE/dp = derivative at fixed dipoles).
"""

from pathlib import Path

import numpy as np
import openmm as mm
import pytest
import torch
from openmm import app, unit

from mlmmpol.amoeba_mixed import mask_multipoles
from mlmmpol.layered import configure_layered_polarization
from openmm_systems import SYSTEMS, make_traceless
from toy_head import ToyHead, ToyHeadPen

mlpolplugin = pytest.importorskip('mlpolplugin')
PLUGIN = Path(__file__).resolve().parents[1] / 'plugin' / 'install' / 'lib' / 'plugins' / 'libMLPolPluginReference.so'
FF = app.ForceField('amoeba2018.xml')
CROSS = 0.7
VALENCE = {'H': (0.8, 0.019), 'C': (4.5, 0.026), 'N': (6.0, 0.024), 'O': (7.0, 0.021)}   # N, s (nm) ~ MBIS


@pytest.fixture(scope='module')
def box():
    mm.Platform.loadPluginLibrary(str(PLUGIN))
    pdb = app.PDBFile(str(SYSTEMS / 'nma_shell.pdb'))
    mod = app.Modeller(pdb.topology, pdb.positions)
    mod.delete([r for r in pdb.topology.residues() if r.name == 'HOH'])
    ml = [a.index for a in mod.topology.atoms()]
    mod.addSolvent(FF, model='tip3p', boxSize=mm.Vec3(2.2, 2.2, 2.2) * unit.nanometer)
    pos = np.array(mod.positions.value_in_unit(unit.nanometer))
    system = FF.createSystem(mod.topology, nonbondedMethod=app.PME, nonbondedCutoff=0.7 * unit.nanometer,
                             polarization='mutual', mutualInducedTargetEpsilon=1e-10)
    i = next(k for k, f in enumerate(system.getForces()) if isinstance(f, mm.AmoebaMultipoleForce))
    make_traceless(system.getForce(i))
    amoeba = system.getForce(i)
    q0 = np.array([amoeba.getMultipoleParameters(a)[0].value_in_unit(unit.elementary_charge) for a in ml])
    a0 = np.array([amoeba.getMultipoleParameters(a)[-1].value_in_unit(unit.nanometer**3) for a in ml])
    atoms = list(mod.topology.atoms())
    n0 = torch.tensor([VALENCE[atoms[a].element.symbol][0] for a in ml], dtype=torch.float64)
    s0 = torch.tensor([VALENCE[atoms[a].element.symbol][1] for a in ml], dtype=torch.float64)
    return dict(system=system, index=i, top=mod.topology, ml=ml, pos=pos, q0=q0, a0=a0, n0=n0, s0=s0)


def context(b, head=None, layered=None):
    system = mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(b['system']))
    f = mlpolplugin.MLPolMultipoleForce(system.getForce(b['index']))
    f.setMutualInducedMaxIterations(1000)
    mask_multipoles(f, b['ml'])
    f.setMLRegion(b['ml'], 0.39, CROSS)
    if head is not None:
        f.setMLHead(head)
    if layered is not None:
        configure_layered_polarization(f, b['top'], b['ml'], *layered)
    system.removeForce(b['index'])
    for g in system.getForces():
        g.setForceGroup(0)
    f.setForceGroup(1)
    system.addForce(f)
    ctx = mm.Context(system, mm.VerletIntegrator(1e-3), mm.Platform.getPlatformByName('Reference'))
    ctx.setPositions(b['pos'])
    return f, ctx


def ef(ctx, pos=None):
    if pos is not None:
        ctx.setPositions(pos)
    st = ctx.getState(getEnergy=True, getForces=True, groups={1})
    return st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole), \
        st.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)


def toy_parameters(b):
    t = torch.as_tensor(b['ml'])
    with torch.no_grad():
        q, mu, th, a = ToyHead(torch.tensor(b['q0']), torch.tensor(b['a0']))(torch.tensor(b['pos'][t]))
    return [q.numpy().copy(), mu.numpy().reshape(-1).copy(), th.numpy().reshape(-1).copy(), a.numpy().copy()]


@pytest.mark.parametrize('layered', [None, (0.6, 0.8)], ids=['mutual', 'layered'])
def test_ml_derivatives_pme_match_finite_differences(box, layered):
    b = box
    f, ctx = context(b, layered=layered)
    prm = toy_parameters(b)

    def energy(p):
        f.setMLParametersInContext(ctx, *[x.tolist() for x in p])
        return ef(ctx)[0]

    energy(prm)
    der = f.getMLDerivatives(ctx)
    phi, field = np.array(der['phi']), np.array(der['field']).reshape(-1, 3)
    dth, dal = np.array(der['dTheta']).reshape(-1, 3, 3), np.array(der['dAlpha'])
    assert np.allclose(dth, dth.transpose(0, 2, 1)) and np.abs(np.trace(dth, axis1=1, axis2=2)).max() < 1e-8

    def fd(which, idx, h, direction=None):
        pp = [x.copy() for x in prm]
        pm = [x.copy() for x in prm]
        step = h if direction is None else h * direction
        pp[which][idx] += step
        pm[which][idx] -= step
        return (energy(pp) - energy(pm)) / (2 * h)

    for k in (0, 4, 9):
        assert abs(fd(0, k, 1e-4) - phi[k]) < 1e-3 * max(1.0, abs(phi[k]))
        for c in range(3):
            assert abs(fd(1, 3 * k + c, 1e-5) + field[k, c]) < 1e-3 * max(1.0, np.abs(field[k]).max())
        # Theta: traceless directions (symmetric xz and xx - yy)
        sym = np.zeros(9)
        sym[[2, 6]] = 1.0
        diag = np.zeros(9)
        diag[[0, 4]] = [1.0, -1.0]
        for dvec in (sym, diag):
            ref = (dth[k].reshape(-1) * dvec).sum()
            got = fd(2, slice(9 * k, 9 * k + 9), 1e-5, dvec)
            assert abs(got - ref) < 1e-3 * max(1.0, np.abs(dth[k]).max())
        got = fd(3, k, 1e-7 * prm[3][k] / 1e-3)
        assert abs(got - dal[k]) < 2e-3 * max(1.0, abs(dal[k]))


@pytest.mark.parametrize('pen', [False, True], ids=['no_penetration', 'penetration'])
def test_head_pme_forces_are_energy_gradient(box, pen, tmp_path):
    b = box
    t = torch.as_tensor(b['ml'])
    q0, a0 = torch.tensor(b['q0']), torch.tensor(b['a0'])
    head = ToyHeadPen(q0, a0, b['n0'], b['s0']) if pen else ToyHead(q0, a0)
    path = tmp_path / 'head.pt'
    torch.jit.script(head).save(str(path))
    f, ctx = context(b, head=str(path))
    e, forces = ef(ctx, b['pos'])
    near = np.argsort(np.linalg.norm(b['pos'] - b['pos'][b['ml']].mean(0), axis=1))
    atoms = b['ml'][:3] + [k for k in near if k not in b['ml']][:3]
    h = 1e-4                  # a smaller step suffers from the OpenMM PME/Thole (expf) noise, as with pure AMOEBA
    for k in atoms:
        for c in range(3):
            pp, pm = b['pos'].copy(), b['pos'].copy()
            pp[k, c] += h
            pm[k, c] -= h
            fd = -(ef(ctx, pp)[0] - ef(ctx, pm)[0]) / (2 * h)
            assert abs(fd - forces[k, c]) < 5e-3 * max(1.0, abs(fd) / 100)


def test_penetration_pme_uses_minimum_image(box):
    """Rewrapping the molecules in the box (solute at the edge) does not change the penetration contribution.

    PME itself is not invariant to translations relative to the grid; so the difference with and without N, s is
    compared (penetration does not enter the polarizing field, so this difference is the penetration term alone).
    """
    b = box
    prm = toy_parameters(b)
    f, ctx = context(b)

    def pen(pos):
        f.setMLParametersInContext(ctx, *[x.tolist() for x in prm])
        f.setMLPenetrationInContext(ctx, [], [])
        e0, f0 = ef(ctx, pos)
        f.setMLPenetrationInContext(ctx, b['n0'].tolist(), b['s0'].tolist())
        e1, f1 = ef(ctx, pos)
        return e1 - e0, f1 - f0

    e1, f1 = pen(b['pos'])
    assert abs(e1) > 1.0                                         # the term is not negligible
    L = ctx.getState().getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer)[0, 0]
    shifted = b['pos'] + np.array([L - b['pos'][b['ml']].mean(0)[0], 0.0, 0.0])
    for res in b['top'].residues():
        idx = [a.index for a in res.atoms()]
        shifted[idx] -= np.floor(shifted[idx[0]] / L) * L
    raw = np.linalg.norm(shifted[b['ml']][:, None] - shifted[None], axis=-1)
    assert raw.max() > L                                         # there are pairs whose minimum image is a different one
    e2, f2 = pen(shifted)
    assert abs(e2 - e1) < 1e-8 * max(1.0, abs(e1))
    assert np.abs(f2 - f1).max() < 1e-6
