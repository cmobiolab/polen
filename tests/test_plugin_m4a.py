"""MLPol plugin, milestone M4a: ML region with injected parameters (lab-frame q, mu, Theta, alpha), cross Thole and
exported derivatives (phi, E, dE/dTheta, dE/dalpha) vs the torch reference. Reference, no cutoff."""

from pathlib import Path

import numpy as np
import openmm as mm
import pytest
import torch
from openmm import app, unit

from mlmmpol.amoeba_mixed import mask_multipoles
from openmm_systems import SYSTEMS, make_traceless
from ref_torch import amoeba_elec as ae
from ref_torch.mlmm_pol import mlmm_electrostatic_energy
from test_mlmm_pol import ToyEmbedding

mlpolplugin = pytest.importorskip('mlpolplugin')
PLUGIN = Path(__file__).resolve().parents[1] / 'plugin' / 'install' / 'lib' / 'plugins' / 'libMLPolPluginReference.so'


@pytest.fixture(scope='module')
def data():
    mm.Platform.loadPluginLibrary(str(PLUGIN))
    pdb = app.PDBFile(str(SYSTEMS / 'nma_shell.pdb'))
    top = pdb.topology
    ml = [a.index for a in top.atoms() if a.residue.name in ('ACE', 'NME')]
    pos = np.load(Path(__file__).resolve().parents[1] / 'qm_ref' / 'nma_frames.npz')['positions_nm'][7]
    system = app.ForceField('amoeba2018.xml').createSystem(top, nonbondedMethod=app.NoCutoff, polarization='mutual',
                                                           mutualInducedTargetEpsilon=1e-11)
    i = next(k for k, f in enumerate(system.getForces()) if isinstance(f, mm.AmoebaMultipoleForce))
    amoeba = system.getForce(i)
    make_traceless(amoeba)
    prm = ae.from_openmm(amoeba)
    t = torch.as_tensor(ml)
    with torch.no_grad():
        q, mu, alpha = ToyEmbedding(prm.charge[t], prm.polarity[t])(torch.tensor(pos[ml]))
    rng = np.random.default_rng(11)
    th = rng.normal(0, 2e-4, (len(ml), 3, 3))
    th = 0.5 * (th + th.transpose(0, 2, 1))
    th -= np.trace(th, axis1=1, axis2=2)[:, None, None] / 3 * np.eye(3)
    return dict(system=system, amoeba_index=i, top=top, ml=ml, pos=pos, prm=prm,
                q=q, mu=mu, alpha=alpha, theta=torch.tensor(th))


def plugin_context(d, cross):
    system = mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(d['system']))
    f = mlpolplugin.MLPolMultipoleForce(system.getForce(d['amoeba_index']))
    f.setMutualInducedMaxIterations(1000)
    mask_multipoles(f, d['ml'])
    f.setMLRegion(d['ml'], 0.39, -1.0 if cross is None else cross)
    system.removeForce(d['amoeba_index'])
    for g in system.getForces():
        g.setForceGroup(0)
    f.setForceGroup(1)
    system.addForce(f)
    ctx = mm.Context(system, mm.VerletIntegrator(1e-3), mm.Platform.getPlatformByName('Reference'))
    ctx.setPositions(d['pos'])
    f.setMLParametersInContext(ctx, d['q'].tolist(), d['mu'].reshape(-1).tolist(), d['theta'].reshape(-1).tolist(),
                               d['alpha'].tolist())
    return f, ctx


@pytest.mark.parametrize('cross', [None, 0.7], ids=['min', 'a_cross0.7'])
def test_energy_forces_and_derivatives(data, cross):
    d = data
    f, ctx = plugin_context(d, cross)
    st = ctx.getState(getEnergy=True, getForces=True, groups={1})
    e = st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    forces = st.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)
    der = f.getMLDerivatives(ctx)

    x = torch.tensor(d['pos'], requires_grad=True)
    q, mu, th, a = (v.clone().requires_grad_() for v in (d['q'], d['mu'], d['theta'], d['alpha']))
    kw = {} if cross is None else dict(a_cross=cross)
    e_t = mlmm_electrostatic_energy(x, d['prm'], d['ml'], q, mu, a, theta=th, **kw)
    gx, gq, gmu, gth, ga = torch.autograd.grad(e_t, (x, q, mu, th, a))
    gth = 0.5 * (gth + gth.transpose(1, 2))
    gth = gth - gth.diagonal(0, 1, 2).sum(-1)[:, None, None] / 3 * torch.eye(3, dtype=gth.dtype)   # the plugin uses only the traceless part

    assert abs(e - e_t.item()) < 1e-7
    assert np.abs(forces + gx.numpy()).max() < 1e-5                # forces at fixed parameters
    assert np.abs(der['phi'] - gq.numpy()).max() < 1e-6
    assert np.abs(der['field'] + gmu.numpy()).max() < 1e-5
    assert np.abs(der['dTheta'] - gth.numpy()).max() < 1e-4 * max(1, np.abs(gth.numpy()).max())
    assert np.abs(der['dAlpha'] - ga.numpy()).max() < 1e-4 * max(1, np.abs(ga.numpy()).max())


def test_ml_region_serialization(data):
    f, ctx = plugin_context(data, 0.7)
    xml = mm.XmlSerializer.serialize(ctx.getSystem())
    import re
    tag = re.search(r'<MLRegion[^>]*>', xml).group(0)
    attr = dict(re.findall(r'(\w+)="([^"]*)"', tag))
    assert float(attr['crossThole']) == 0.7 and float(attr['tholeML']) == 0.39 and attr['head'] == ''
    assert xml.count('<Atom index=') >= len(data['ml'])
    s2 = mm.XmlSerializer.deserialize(xml)
    assert mm.XmlSerializer.serialize(s2) == xml
