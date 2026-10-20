"""MLPol plugin, milestone M5d: layered polarization on the GPU (CUDA) == plugin Reference."""

import numpy as np
import pytest
import torch

from mlmmpol.layered import configure_layered_polarization
from test_plugin_m5c import LIB, box, close, context, cuda_available, ef, toy_parameters  # noqa: F401
from toy_head import ToyHeadPen

mlpolplugin = pytest.importorskip('mlpolplugin')
pytestmark = pytest.mark.skipif(not cuda_available(), reason='no CUDA')
LAYERS = (0.45, 0.7)


def layered_context(b, platform, head=None, ml=True):
    f, ctx = context(b, platform, head=head) if ml else plain_context(b, platform)
    configure_layered_polarization(f, b['top'], b['ml'], *LAYERS)
    ctx.reinitialize(preserveState=True)
    return f, ctx


def plain_context(b, platform):
    import openmm as mm
    system = mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(b['system']))
    f = mlpolplugin.MLPolMultipoleForce(system.getForce(b['index']))
    f.setMutualInducedMaxIterations(1000)
    system.removeForce(b['index'])
    for g in system.getForces():
        g.setForceGroup(0)
    f.setForceGroup(1)
    system.addForce(f)
    props = {'Precision': 'double'} if platform == 'CUDA' else {}
    ctx = mm.Context(system, mm.VerletIntegrator(1e-3), mm.Platform.getPlatformByName(platform), props)
    ctx.setPositions(b['pos'])
    return f, ctx


def weights_in_switch(b):
    d = np.linalg.norm(b['pos'][:, None] - b['pos'][b['ml']][None], axis=-1).min(1)
    return int(((d > LAYERS[0]) & (d < LAYERS[1])).sum())


def test_layered_amoeba(box):
    assert weights_in_switch(box) > 0                            # there are atoms in the switching zone
    out = [ef(layered_context(box, p, ml=False)[1]) for p in ('Reference', 'CUDA')]
    close(*out[0], *out[1])


def test_layered_ml_injected(box):
    prm = toy_parameters(box)
    out = []
    for platform in ('Reference', 'CUDA'):
        f, ctx = layered_context(box, platform)
        f.setMLParametersInContext(ctx, *prm)
        out.append(ef(ctx))
    close(*out[0], *out[1])
    der = []
    for platform in ('Reference', 'CUDA'):
        f, ctx = layered_context(box, platform)
        f.setMLParametersInContext(ctx, *prm)
        der.append({k: np.array(v) for k, v in f.getMLDerivatives(ctx).items()})
    for k in ('phi', 'field', 'dTheta', 'dAlpha'):
        assert np.abs(der[1][k] - der[0][k]).max() < 1e-4 * max(1.0, np.abs(der[0][k]).max()), k


def test_layered_ml_head_with_penetration(box, tmp_path):
    head = ToyHeadPen(torch.tensor(box['q0']), torch.tensor(box['a0']), box['n0'], box['s0'])
    path = tmp_path / 'head_pen.pt'
    torch.jit.script(head).save(str(path))
    out = [ef(layered_context(box, p, head=str(path))[1]) for p in ('Reference', 'CUDA')]
    close(*out[0], *out[1])
