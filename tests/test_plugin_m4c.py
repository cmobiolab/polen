"""MLPol plugin, milestone M4c: charge penetration (ML Slater valence × MM multipoles) vs the torch reference."""

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
from ref_torch.penetration import penetration_energy
from toy_head import ToyHead, ToyHeadPen

mlpolplugin = pytest.importorskip('mlpolplugin')
PLUGIN = Path(__file__).resolve().parents[1] / 'plugin' / 'install' / 'lib' / 'plugins' / 'libMLPolPluginReference.so'
BOHR = 0.0529177210903
HARTREE_KJ = 2625.499639479948
CROSS = 0.7
VALENCE = {'H': (0.8, 0.019), 'C': (4.5, 0.026), 'N': (6.0, 0.024), 'O': (7.0, 0.021)}   # N, s (nm) ~ MBIS


@pytest.fixture(scope='module')
def base():
    mm.Platform.loadPluginLibrary(str(PLUGIN))
    pdb = app.PDBFile(str(SYSTEMS / 'nma_shell.pdb'))
    top = pdb.topology
    atoms = list(top.atoms())
    ml = [a.index for a in atoms if a.residue.name in ('ACE', 'NME')]
    pos = np.load(Path(__file__).resolve().parents[1] / 'qm_ref' / 'nma_frames.npz')['positions_nm'][2]
    system = app.ForceField('amoeba2018.xml').createSystem(top, nonbondedMethod=app.NoCutoff, polarization='mutual',
                                                           mutualInducedTargetEpsilon=1e-11)
    i = next(k for k, f in enumerate(system.getForces()) if isinstance(f, mm.AmoebaMultipoleForce))
    make_traceless(system.getForce(i))
    prm = ae.from_openmm(system.getForce(i))
    n0 = torch.tensor([VALENCE[atoms[a].element.symbol][0] for a in ml], dtype=torch.float64)
    s0 = torch.tensor([VALENCE[atoms[a].element.symbol][1] for a in ml], dtype=torch.float64)
    return dict(system=system, index=i, top=top, ml=ml, pos=pos, prm=prm, n0=n0, s0=s0)


def context(b, head=None):
    system = mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(b['system']))
    f = mlpolplugin.MLPolMultipoleForce(system.getForce(b['index']))
    f.setMutualInducedMaxIterations(1000)
    mask_multipoles(f, b['ml'])
    f.setMLRegion(b['ml'], 0.39, CROSS)
    if head is not None:
        f.setMLHead(head)
    system.removeForce(b['index'])
    for g in system.getForces():
        g.setForceGroup(0)
    f.setForceGroup(1)
    system.addForce(f)
    ctx = mm.Context(system, mm.VerletIntegrator(1e-3), mm.Platform.getPlatformByName('Reference'))
    ctx.setPositions(b['pos'])
    return f, ctx


def ef(ctx, pos):
    ctx.setPositions(pos)
    st = ctx.getState(getEnergy=True, getForces=True, groups={1})
    return st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole), \
        st.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)


def torch_penetration(b, x, N, s):
    """E_pen (kJ/mol) from the reference (atomic units), with lab-frame MM multipoles (autograd through the local frames)."""
    t = torch.as_tensor(b['ml'])
    mm_idx = torch.as_tensor(np.setdiff1d(np.arange(b['prm'].n), b['ml']))
    mu_lab, th_lab = ae.lab_frame_multipoles(x, b['prm'])
    return penetration_energy(x[t] / BOHR, N, s / BOHR, x[mm_idx] / BOHR, b['prm'].charge[mm_idx],
                              mu_lab[mm_idx] / BOHR, th_lab[mm_idx] / BOHR**2) * HARTREE_KJ


def test_penetration_energy_and_forces_injected(base):
    b = base
    t = torch.as_tensor(b['ml'])
    with torch.no_grad():
        q, mu, th, a = ToyHead(b['prm'].charge[t], b['prm'].polarity[t])(torch.tensor(b['pos'][b['ml']]))
    f, ctx = context(b)
    f.setMLParametersInContext(ctx, q.tolist(), mu.reshape(-1).tolist(), th.reshape(-1).tolist(), a.tolist())
    e0, f0 = ef(ctx, b['pos'])
    f.setMLPenetrationInContext(ctx, b['n0'].tolist(), b['s0'].tolist())
    e1, f1 = ef(ctx, b['pos'])
    x = torch.tensor(b['pos'], requires_grad=True)
    e_pen = torch_penetration(b, x, b['n0'], b['s0'])
    g = torch.autograd.grad(e_pen, x)[0].numpy()
    assert abs(e_pen.item()) > 1.0                               # the term is not negligible
    assert abs((e1 - e0) - e_pen.item()) < 1e-6
    assert np.abs((f1 - f0) + g).max() < 1e-5                    # includes the torques through the MM local frames


def test_head_with_penetration_matches_full_autograd(base, tmp_path):
    b = base
    t = torch.as_tensor(b['ml'])
    head = ToyHeadPen(b['prm'].charge[t], b['prm'].polarity[t], b['n0'], b['s0'])
    path = tmp_path / 'head_pen.pt'
    torch.jit.script(head).save(str(path))
    f, ctx = context(b, head=str(path))
    e, forces = ef(ctx, b['pos'])
    x = torch.tensor(b['pos'], requires_grad=True)
    q, mu, th, a, N, s = head(x[t])
    e_t = mlmm_electrostatic_energy(x, b['prm'], b['ml'], q, mu, a, theta=th, a_cross=CROSS) + torch_penetration(b, x, N, s)
    f_t = -torch.autograd.grad(e_t, x)[0].numpy()
    assert abs(e - e_t.item()) < 1e-6
    assert np.abs(forces - f_t).max() < 1e-5
    # and the forces are the gradient of the plugin energy (ML atoms and nearby MM atoms)
    near = np.argsort(np.linalg.norm(b['pos'] - b['pos'][b['ml']].mean(0), axis=1))
    atoms = [k for k in near if k not in b['ml']][:3] + b['ml'][:2]
    h = 1e-6
    for k in atoms:
        for c in range(3):
            pp, pm = b['pos'].copy(), b['pos'].copy()
            pp[k, c] += h
            pm[k, c] -= h
            fd = -(ef(ctx, pp)[0] - ef(ctx, pm)[0]) / (2 * h)
            assert abs(fd - forces[k, c]) < 1e-3 * max(1.0, abs(fd) / 100)
