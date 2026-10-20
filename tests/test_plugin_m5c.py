"""MLPol plugin, milestone M5c: 'ml' mode on the GPU (CUDA) == plugin Reference."""

from pathlib import Path

import numpy as np
import openmm as mm
import pytest
import torch
from openmm import app, unit

from mlmmpol.amoeba_mixed import mask_multipoles
from openmm_systems import SYSTEMS, make_traceless
from toy_head import ToyHead, ToyHeadPen

mlpolplugin = pytest.importorskip('mlpolplugin')
LIB = Path(__file__).resolve().parents[1] / 'plugin' / 'install' / 'lib' / 'plugins'
FF = app.ForceField('amoeba2018.xml')
VALENCE = {'H': (0.8, 0.019), 'C': (4.5, 0.026), 'N': (6.0, 0.024), 'O': (7.0, 0.021)}


def cuda_available():
    try:
        mm.Platform.getPlatformByName('CUDA')
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not cuda_available(), reason='no CUDA')


def build(kind):
    pdb = app.PDBFile(str(SYSTEMS / 'nma_shell.pdb'))
    if kind == 'nocutoff':
        top, pos = pdb.topology, pdb.positions
        kw = dict(nonbondedMethod=app.NoCutoff)
    else:
        mod = app.Modeller(pdb.topology, pdb.positions)
        mod.delete([r for r in pdb.topology.residues() if r.name == 'HOH'])
        mod.addSolvent(FF, model='tip3p', boxSize=mm.Vec3(2.4, 2.4, 2.4) * unit.nanometer)
        top, pos = mod.topology, mod.positions
        kw = dict(nonbondedMethod=app.PME, nonbondedCutoff=0.7 * unit.nanometer)
    ml = [a.index for a in top.atoms() if a.residue.name in ('ACE', 'NME')]
    system = FF.createSystem(top, polarization='mutual', mutualInducedTargetEpsilon=1e-9, **kw)
    i = next(k for k, f in enumerate(system.getForces()) if isinstance(f, mm.AmoebaMultipoleForce))
    make_traceless(system.getForce(i))
    amoeba = system.getForce(i)
    if kind == 'pme':
        amoeba.setPMEParameters(3.4, 36, 36, 36)     # same grid on both platforms
    q0 = np.array([amoeba.getMultipoleParameters(a)[0].value_in_unit(unit.elementary_charge) for a in ml])
    a0 = np.array([amoeba.getMultipoleParameters(a)[-1].value_in_unit(unit.nanometer**3) for a in ml])
    atoms = list(top.atoms())
    n0 = torch.tensor([VALENCE[atoms[a].element.symbol][0] for a in ml], dtype=torch.float64)
    s0 = torch.tensor([VALENCE[atoms[a].element.symbol][1] for a in ml], dtype=torch.float64)
    pos = np.array(pos.value_in_unit(unit.nanometer))
    return dict(system=system, index=i, top=top, ml=ml, pos=pos, q0=q0, a0=a0, n0=n0, s0=s0)


@pytest.fixture(scope='module', params=['nocutoff', 'pme'])
def box(request):
    mm.Platform.loadPluginLibrary(str(LIB / 'libMLPolPluginReference.so'))
    mm.Platform.loadPluginLibrary(str(LIB / 'libMLPolPluginCUDA.so'))
    return build(request.param)


def context(b, platform, cross=0.7, head=None):
    system = mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(b['system']))
    f = mlpolplugin.MLPolMultipoleForce(system.getForce(b['index']))
    f.setMutualInducedMaxIterations(1000)
    mask_multipoles(f, b['ml'])
    f.setMLRegion(b['ml'], 0.39, -1.0 if cross is None else cross)
    if head is not None:
        f.setMLHead(head)
    system.removeForce(b['index'])
    for g in system.getForces():
        g.setForceGroup(0)
    f.setForceGroup(1)
    system.addForce(f)
    props = {'Precision': 'double'} if platform == 'CUDA' else {}
    ctx = mm.Context(system, mm.VerletIntegrator(1e-3), mm.Platform.getPlatformByName(platform), props)
    ctx.setPositions(b['pos'])
    return f, ctx


def ef(ctx):
    st = ctx.getState(getEnergy=True, getForces=True, groups={1})
    return st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole), \
        st.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)


def toy_parameters(b):
    t = torch.as_tensor(b['ml'])
    with torch.no_grad():
        q, mu, th, a = ToyHead(torch.tensor(b['q0']), torch.tensor(b['a0']))(torch.tensor(b['pos'][t]))
    return [q.tolist(), mu.reshape(-1).tolist(), th.reshape(-1).tolist(), a.tolist()]


def close(e0, f0, e1, f1, rel=1e-6):
    assert abs(e1 - e0) < rel * max(1.0, abs(e0)) + 1e-4
    assert np.abs(f1 - f0).max() < 1e-3 * max(1.0, np.abs(f0).max() / 1000)


@pytest.mark.parametrize('cross', [None, 0.7], ids=['min', 'a_cross0.7'])
def test_injected_parameters(box, cross):
    prm = toy_parameters(box)
    out = []
    for platform in ('Reference', 'CUDA'):
        f, ctx = context(box, platform, cross)
        f.setMLParametersInContext(ctx, *prm)
        out.append(ef(ctx))
    close(*out[0], *out[1])


def test_ml_derivatives(box):
    prm = toy_parameters(box)
    der = []
    for platform in ('Reference', 'CUDA'):
        f, ctx = context(box, platform)
        f.setMLParametersInContext(ctx, *prm)
        der.append({k: np.array(v) for k, v in f.getMLDerivatives(ctx).items()})
    for k in ('phi', 'field', 'dTheta', 'dAlpha'):
        ref, got = der[0][k], der[1][k]
        assert np.abs(got - ref).max() < 1e-4 * max(1.0, np.abs(ref).max()), k


def test_head(box, tmp_path):
    head = ToyHead(torch.tensor(box['q0']), torch.tensor(box['a0']))
    path = tmp_path / 'head.pt'
    torch.jit.script(head).save(str(path))
    out = []
    for platform in ('Reference', 'CUDA'):
        f, ctx = context(box, platform, head=str(path))
        out.append(ef(ctx))
    close(*out[0], *out[1])
    # and the head part of the ML-atom forces is not negligible (checks that the VJP was added)
    f, ctx = context(box, 'CUDA')
    prm = toy_parameters(box)
    f.setMLParametersInContext(ctx, *prm)
    e_fixed, f_fixed = ef(ctx)
    assert np.abs(out[1][1][box['ml']] - f_fixed[box['ml']]).max() > 1.0


def test_penetration_injected(box):
    prm = toy_parameters(box)
    out = []
    for platform in ('Reference', 'CUDA'):
        f, ctx = context(box, platform)
        f.setMLParametersInContext(ctx, *prm)
        f.setMLPenetrationInContext(ctx, box['n0'].tolist(), box['s0'].tolist())
        out.append(ef(ctx))
    close(*out[0], *out[1])
    f, ctx = context(box, 'CUDA')
    f.setMLParametersInContext(ctx, *prm)
    e_off = ef(ctx)[0]
    assert abs(out[1][0] - e_off) > 1e-3                          # the term was added (in the PME box the water is farther away)


def test_head_with_penetration(box, tmp_path):
    head = ToyHeadPen(torch.tensor(box['q0']), torch.tensor(box['a0']), box['n0'], box['s0'])
    path = tmp_path / 'head_pen.pt'
    torch.jit.script(head).save(str(path))
    out = []
    for platform in ('Reference', 'CUDA'):
        f, ctx = context(box, platform, head=str(path))
        out.append(ef(ctx))
    close(*out[0], *out[1])
