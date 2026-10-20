"""MLPol plugin, milestone M3: layered polarization (mutual -> direct with a smooth switch), Reference, no cutoff."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import openmm as mm
import pytest
import torch
from openmm import app, unit

from mlmmpol.layered import configure_layered_polarization
from openmm_systems import SYSTEMS, make_traceless
from ref_torch import amoeba_elec as ae
from ref_torch.switching import site_weights, switched_u_scale

mlpolplugin = pytest.importorskip('mlpolplugin')
PLUGIN = Path(__file__).resolve().parents[1] / 'plugin' / 'install' / 'lib' / 'plugins' / 'libMLPolPluginReference.so'
R_IN, R_OUT = 0.30, 0.42


@pytest.fixture(scope='module')
def setup():
    mm.Platform.loadPluginLibrary(str(PLUGIN))
    pdb = app.PDBFile(str(SYSTEMS / 'nma_shell.pdb'))
    top = pdb.topology
    pos = np.load(Path(__file__).resolve().parents[1] / 'qm_ref' / 'nma_frames.npz')['positions_nm'][4]
    ml = [a.index for a in top.atoms() if a.residue.name in ('ACE', 'NME')]
    system = app.ForceField('amoeba2018.xml').createSystem(top, nonbondedMethod=app.NoCutoff, polarization='mutual',
                                                           mutualInducedTargetEpsilon=1e-10)
    i = next(k for k, f in enumerate(system.getForces()) if isinstance(f, mm.AmoebaMultipoleForce))
    amoeba = system.getForce(i)
    amoeba.setMutualInducedMaxIterations(1000)
    make_traceless(amoeba)
    prm = ae.from_openmm(amoeba)
    f = mlpolplugin.MLPolMultipoleForce(amoeba)
    group_of, group_ref = configure_layered_polarization(f, top, ml, R_IN, R_OUT)
    system.removeForce(i)
    for g in system.getForces():
        g.setForceGroup(0)
    f.setForceGroup(1)
    system.addForce(f)
    ctx = mm.Context(system, mm.VerletIntegrator(1e-3), mm.Platform.getPlatformByName('Reference'))
    return dict(system=system, f=f, ctx=ctx, pos=pos, prm=prm, ml=ml, group_of=group_of, group_ref=group_ref)


def plugin_ef(d, pos):
    d['ctx'].setPositions(pos)
    st = d['ctx'].getState(getEnergy=True, getForces=True, groups={1})
    return st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole), \
        st.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)


def torch_energy(d, x):
    s = site_weights(x, torch.as_tensor(d['ml']), torch.as_tensor(d['group_ref']),
                     torch.as_tensor(d['group_of']), R_IN, R_OUT)
    return ae.electrostatic_energy(x, replace(d['prm'], u_scale=switched_u_scale(d['prm'].u_scale, s))), s


def test_layered_matches_torch_reference(setup):
    d = setup
    e, f = plugin_ef(d, d['pos'])
    x = torch.tensor(d['pos'], requires_grad=True)
    e_t, s = torch_energy(d, x)
    f_t = -torch.autograd.grad(e_t, x)[0].numpy()
    in_switch = ((s > 1e-6) & (s < 1 - 1e-6)).sum().item()
    assert in_switch >= 3                                        # there are groups in the middle of the switch
    assert abs(e - e_t.item()) < 1e-7
    assert np.abs(f - f_t).max() < 1e-5
    # and it differs from all-mutual (the layer has an effect)
    e_mut = ae.electrostatic_energy(torch.tensor(d['pos']), d['prm']).item()
    assert abs(e - e_mut) > 0.5


def test_layered_forces_are_energy_gradient(setup):
    d = setup
    pos = d['pos']
    _, f = plugin_ef(d, pos)
    _, s = torch_energy(d, torch.tensor(pos))
    grp = [g for g in range(len(d['group_ref'])) if 1e-3 < s[d['group_ref'][g]] < 1 - 1e-3]
    atoms = [d['group_ref'][g] for g in grp[:3]] + list(d['ml'][:2])
    h = 1e-6
    for a in atoms:
        for c in range(3):
            pp, pm = pos.copy(), pos.copy()
            pp[a, c] += h
            pm[a, c] -= h
            fd = -(plugin_ef(d, pp)[0] - plugin_ef(d, pm)[0]) / (2 * h)
            assert abs(fd - f[a, c]) < 2e-3 * max(1.0, abs(fd) / 100)


def test_layered_requires_nocutoff_mutual(setup):
    d = setup
    top = app.PDBFile(str(SYSTEMS / 'nma_shell.pdb')).topology
    system = app.ForceField('amoeba2018.xml').createSystem(top, nonbondedMethod=app.NoCutoff, polarization='direct')
    i = next(k for k, g in enumerate(system.getForces()) if isinstance(g, mm.AmoebaMultipoleForce))
    f2 = mlpolplugin.MLPolMultipoleForce(system.getForce(i))
    configure_layered_polarization(f2, top, d['ml'], R_IN, R_OUT)
    system.removeForce(i)
    system.addForce(f2)
    with pytest.raises(Exception, match='camadas|layered'):
        c = mm.Context(system, mm.VerletIntegrator(1e-3), mm.Platform.getPlatformByName('Reference'))
        c.setPositions(d['pos'])
        c.getState(getEnergy=True)
