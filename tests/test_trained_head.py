"""Trained head inside the plugin, 'ml' mode: benzamidinium + waters within 4.5 Å (from a trypsin window), AMOEBA
NoCutoff, mutual polarization, Reference.

- forces = central finite differences of the energy (ML and MM atoms);
- injected Σq = +1 and α = AMOEBA × 1.10 (from the derivative ∂E/∂q and from the parameters returned by the head);
- short NVE (all forces of the AMOEBA system; the head only replaces the ML-region electrostatics): small drift.
Skipped if the trypsin window or the trained head are not available.
"""
from pathlib import Path

import numpy as np
import openmm as mm
import pytest
import torch
from openmm import app, unit

from mlmmpol.amber2amoeba import bio18_path
from mlmmpol.amoeba_mixed import mask_multipoles
from mlmmpol.export_head import export_head

mlpolplugin = pytest.importorskip('mlpolplugin')
ROOT = Path(__file__).resolve().parents[1]
PLUGIN = ROOT / 'plugin' / 'install' / 'lib' / 'plugins' / 'libMLPolPluginReference.so'
WIN = ROOT / 'mlmm_windows' / 'trypsin_ben' / 'amoeba' / 'rep1'
HEAD = ROOT / 'qm_ref' / 'heads' / 'v1_f128_b4_s0'
CROSS = 0.7
pytestmark = pytest.mark.skipif(not (WIN / 'amoeba.xyz').exists() or not (HEAD / 'head.pt').exists(),
                                reason='trypsin window or trained head not available')


def read_xyz(path):
    lines = open(path).read().split('\n')
    n = int(lines[0].split()[0])
    rows = [l.split() for l in lines[1:n + 1]]
    return rows


@pytest.fixture(scope='module')
def cluster(tmp_path_factory):
    mm.Platform.loadPluginLibrary(str(PLUGIN))
    rows = read_xyz(WIN / 'amoeba.xyz')
    ml = [int(x) for x in np.loadtxt(WIN / 'ml_atoms.txt')]
    x = np.array([[float(r[2]), float(r[3]), float(r[4])] for r in rows])
    keep = list(ml)
    for k, r in enumerate(rows):
        if r[5] == '349' and np.linalg.norm(x[ml] - x[k], axis=1).min() < 4.5:     # AMOEBA water O
            keep += [k, k + 1, k + 2]
    new = {old: i for i, old in enumerate(keep)}
    tmp = tmp_path_factory.mktemp('ben')
    with open(tmp / 'c.xyz', 'w') as f:
        f.write(f'{len(keep)} benzamidinium + waters\n')
        for old in keep:
            r = rows[old]
            nb = [str(new[int(b) - 1] + 1) for b in r[6:] if int(b) - 1 in new]
            f.write(f'{new[old] + 1:6d}  {r[1]:3s} {r[2]} {r[3]} {r[4]} {r[5]} ' + ' '.join(nb) + '\n')
    tk = app.TinkerFiles(str(tmp / 'c.xyz'), [bio18_path(), str(WIN / 'amoeba.key')])
    system = tk.createSystem(nonbondedMethod=app.NoCutoff, polarization='mutual', mutualInducedTargetEpsilon=1e-10)
    i = next(k for k, f in enumerate(system.getForces()) if isinstance(f, mm.AmoebaMultipoleForce))
    amf = system.getForce(i)
    ml_local = list(range(len(ml)))
    z = [a.element.atomic_number for a in tk.topology.atoms()][:len(ml)]
    alpha = [amf.getMultipoleParameters(a)[-1].value_in_unit(unit.nanometer ** 3) for a in ml_local]
    head_file = tmp / 'ben_head.pt'
    head = export_head(HEAD, z, 1.0, alpha, head_file)
    pos = np.array(tk.getPositions().value_in_unit(unit.nanometer))
    return dict(system=system, index=i, ml=ml_local, pos=pos, head_file=head_file, head=head, alpha=alpha)


def make_context(c, integrator=None, platform='Reference'):
    system = mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(c['system']))
    f = mlpolplugin.MLPolMultipoleForce(system.getForce(c['index']))
    f.setMutualInducedMaxIterations(1000)
    mask_multipoles(f, c['ml'])
    f.setMLRegion(c['ml'], 0.39, CROSS)
    f.setMLHead(str(c['head_file']))
    system.removeForce(c['index'])
    for g in system.getForces():
        g.setForceGroup(0)
    f.setForceGroup(1)
    system.addForce(f)
    props = {'Precision': 'double'} if platform == 'CUDA' else {}
    ctx = mm.Context(system, integrator or mm.VerletIntegrator(1e-3), mm.Platform.getPlatformByName(platform), props)
    ctx.setPositions(c['pos'])
    return f, ctx


def test_head_outputs(cluster):
    q, mu, theta, alpha, n, s = cluster['head'](torch.tensor(cluster['pos'][cluster['ml']]))
    assert abs(q.sum().item() - 1.0) < 1e-10
    assert np.allclose(alpha.numpy(), np.array(cluster['alpha']) * 1.10)
    assert torch.allclose(theta, theta.transpose(-1, -2)) and theta.diagonal(dim1=-2, dim2=-1).sum(-1).abs().max() < 1e-12
    assert (s > 0).all() and (n > 0).all()


def test_forces_finite_differences(cluster):
    _, ctx = make_context(cluster)
    pos = cluster['pos'].copy()
    st = ctx.getState(getEnergy=True, getForces=True, groups={1})
    f = st.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)
    h = 1e-5
    rng = np.random.default_rng(0)
    atoms = list(rng.choice(cluster['ml'], 4, replace=False)) + list(rng.choice(range(len(cluster['ml']), len(pos)), 4, replace=False))
    worst = 0.0
    for a in atoms:
        for k in range(3):
            e = []
            for sgn in (1, -1):
                p = pos.copy()
                p[a, k] += sgn * h
                ctx.setPositions(p)
                e.append(ctx.getState(getEnergy=True, groups={1}).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))
            fd = -(e[0] - e[1]) / (2 * h)
            worst = max(worst, abs(fd - f[a, k]) / max(1.0, abs(f[a, k])))
    print(f'worst relative error force vs finite difference: {worst:.1e}')
    assert worst < 1e-4


def test_nve_short(cluster):
    integ = mm.VerletIntegrator(0.25 * unit.femtosecond)
    _, ctx = make_context(cluster, integ)
    mm.LocalEnergyMinimizer.minimize(ctx, 10.0, 200)
    ctx.setVelocitiesToTemperature(300 * unit.kelvin, 1)
    e = []
    for _ in range(40):
        integ.step(25)
        st = ctx.getState(getEnergy=True)
        e.append((st.getPotentialEnergy() + st.getKineticEnergy()).value_in_unit(unit.kilojoule_per_mole))
    e = np.array(e)
    slope = np.polyfit(np.arange(len(e)) * 25 * 0.25e-3, e, 1)[0]          # kJ/mol/ps
    print(f'NVE 0.25 ps: σ(E) = {e.std():.3f} kJ/mol, drift = {slope:.2f} kJ/mol/ps')
    assert e.std() < 1.0 and abs(slope) < 5.0


def test_cuda_matches_reference(cluster):
    """Trained head in the CUDA kernel (double precision) = Reference: energy and forces."""
    try:
        mm.Platform.loadPluginLibrary(str(PLUGIN.parent / 'libMLPolPluginCUDA.so'))
        _, ctx_c = make_context(cluster, platform='CUDA')
    except Exception as e:
        pytest.skip(f'no CUDA: {e}')
    _, ctx_r = make_context(cluster)
    sr = ctx_r.getState(getEnergy=True, getForces=True, groups={1})
    sc = ctx_c.getState(getEnergy=True, getForces=True, groups={1})
    er, ec = (s.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole) for s in (sr, sc))
    fr, fc = (s.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer) for s in (sr, sc))
    print(f'CUDA vs Reference: dE = {ec - er:.2e} kJ/mol, max |dF| = {np.abs(fc - fr).max():.2e} kJ/mol/nm')
    assert abs(ec - er) < 1e-3 * max(1.0, abs(er))
    assert np.abs(fc - fr).max() < 1e-2
