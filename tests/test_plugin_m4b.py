"""MLPol plugin, milestone M4b: ML head (TorchScript) inside the kernel + chain rule vs the torch reference."""

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
from toy_head import ToyHead

mlpolplugin = pytest.importorskip('mlpolplugin')
PLUGIN = Path(__file__).resolve().parents[1] / 'plugin' / 'install' / 'lib' / 'plugins' / 'libMLPolPluginReference.so'
CROSS = 0.7


@pytest.fixture(scope='module')
def setup(tmp_path_factory):
    mm.Platform.loadPluginLibrary(str(PLUGIN))
    pdb = app.PDBFile(str(SYSTEMS / 'nma_shell.pdb'))
    top = pdb.topology
    ml = [a.index for a in top.atoms() if a.residue.name in ('ACE', 'NME')]
    pos = np.load(Path(__file__).resolve().parents[1] / 'qm_ref' / 'nma_frames.npz')['positions_nm'][9]
    system = app.ForceField('amoeba2018.xml').createSystem(top, nonbondedMethod=app.NoCutoff, polarization='mutual',
                                                           mutualInducedTargetEpsilon=1e-11)
    i = next(k for k, f in enumerate(system.getForces()) if isinstance(f, mm.AmoebaMultipoleForce))
    amoeba = system.getForce(i)
    make_traceless(amoeba)
    prm = ae.from_openmm(amoeba)
    t = torch.as_tensor(ml)
    head = ToyHead(prm.charge[t], prm.polarity[t])
    path = tmp_path_factory.mktemp('head') / 'toy_head.pt'
    torch.jit.script(head).save(str(path))
    f = mlpolplugin.MLPolMultipoleForce(amoeba)
    f.setMutualInducedMaxIterations(1000)
    mask_multipoles(f, ml)
    f.setMLRegion(ml, 0.39, CROSS)
    f.setMLHead(str(path))
    system.removeForce(i)
    for g in system.getForces():
        g.setForceGroup(0)
    f.setForceGroup(1)
    system.addForce(f)
    ctx = mm.Context(system, mm.VerletIntegrator(1e-3), mm.Platform.getPlatformByName('Reference'))
    return dict(ctx=ctx, pos=pos, prm=prm, ml=ml, head=head, system=system, path=path)


def plugin_ef(d, pos):
    d['ctx'].setPositions(pos)
    st = d['ctx'].getState(getEnergy=True, getForces=True, groups={1})
    return st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole), \
        st.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)


def test_head_forces_match_full_autograd(setup):
    d = setup
    e, f = plugin_ef(d, d['pos'])
    x = torch.tensor(d['pos'], requires_grad=True)
    q, mu, th, a = d['head'](x[torch.as_tensor(d['ml'])])
    e_t = mlmm_electrostatic_energy(x, d['prm'], d['ml'], q, mu, a, theta=th, a_cross=CROSS)
    f_t = -torch.autograd.grad(e_t, x)[0].numpy()
    # forces at fixed parameters (no chain rule), to confirm that the head term matters
    x2 = torch.tensor(d['pos'], requires_grad=True)
    e_fix = mlmm_electrostatic_energy(x2, d['prm'], d['ml'], q.detach(), mu.detach(), a.detach(),
                                      theta=th.detach(), a_cross=CROSS)
    f_fix = -torch.autograd.grad(e_fix, x2)[0].numpy()
    assert np.abs(f_t - f_fix).max() > 1.0
    assert abs(e - e_t.item()) < 1e-7
    assert np.abs(f - f_t).max() < 1e-5


def test_head_forces_are_energy_gradient(setup):
    d = setup
    _, f = plugin_ef(d, d['pos'])
    h = 1e-6
    for a in d['ml'][:4]:
        for c in range(3):
            pp, pm = d['pos'].copy(), d['pos'].copy()
            pp[a, c] += h
            pm[a, c] -= h
            fd = -(plugin_ef(d, pp)[0] - plugin_ef(d, pm)[0]) / (2 * h)
            assert abs(fd - f[a, c]) < 1e-3 * max(1.0, abs(fd) / 100)


def test_head_serialized(setup):
    xml = mm.XmlSerializer.serialize(setup['system'])
    assert f'head="{setup["path"]}"' in xml
