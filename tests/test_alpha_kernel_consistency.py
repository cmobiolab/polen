"""The polarization model used to train α(R) (mlmmpol.embedding_head.thole_tensor) is the same as in the kernel: in a
masked ML region (NMA, AMOEBA 2018), with the rest of the system switched off (q = μ = Θ = α = 0) and two distant
charges ±Q, the plugin (Reference) induced dipoles = (α⁻¹ − T)⁻¹ E_ext, with E_ext the Coulomb field of the charges."""
from pathlib import Path

import numpy as np
import openmm as mm
import pytest
import torch
from openmm import app, unit

from mlmmpol.amoeba_mixed import mask_multipoles
from mlmmpol.embedding_head import thole_tensor
from openmm_systems import SYSTEMS

mlpolplugin = pytest.importorskip('mlpolplugin')
PLUGIN = Path(__file__).resolve().parents[1] / 'plugin' / 'install' / 'lib' / 'plugins' / 'libMLPolPluginReference.so'


def test_induced_dipoles_match_thole_model():
    mm.Platform.loadPluginLibrary(str(PLUGIN))
    pdb = app.PDBFile(str(SYSTEMS / 'nma_shell.pdb'))
    top = pdb.topology
    ml = [a.index for a in top.atoms() if a.residue.name in ('ACE', 'NME')]
    system = app.ForceField('amoeba2018.xml').createSystem(top, nonbondedMethod=app.NoCutoff, polarization='mutual',
                                                           mutualInducedTargetEpsilon=1e-12)
    for i in reversed(range(system.getNumForces())):
        if not isinstance(system.getForce(i), mm.AmoebaMultipoleForce):
            system.removeForce(i)
    amoeba = system.getForce(0)
    x = np.array(pdb.getPositions().value_in_unit(unit.nanometer))
    center = x[ml].mean(0)
    others = [i for i in range(len(x)) if i not in ml]
    src = others[:2]
    D, Q = 3.0, 2.0
    x[src[0]] = center + [D, 0.3, 0.1]
    x[src[1]] = center - [D, -0.2, 0.4]
    for i in others:
        p = amoeba.getMultipoleParameters(i)
        q = Q if i == src[0] else -Q if i == src[1] else 0.0
        amoeba.setMultipoleParameters(i, q, [0.0] * 3, [0.0] * 9, mm.AmoebaMultipoleForce.NoAxisType, -1, -1, -1,
                                      p[7], p[8], 0.0 * unit.nanometer ** 3)
    mask_multipoles(amoeba, ml)
    alpha = torch.tensor([amoeba.getMultipoleParameters(i)[-1].value_in_unit(unit.nanometer ** 3) for i in ml], dtype=torch.float64)   # nm³ (before removing)
    pf = mlpolplugin.MLPolMultipoleForce(amoeba)
    pf.setMutualInducedMaxIterations(1000)
    system.removeForce(0)
    system.addForce(pf)
    ctx = mm.Context(system, mm.VerletIntegrator(1e-3), mm.Platform.getPlatformByName('Reference'))
    ctx.setPositions(x * unit.nanometer)
    mu = np.array(pf.getInducedDipoles(ctx))[ml]                        # e·nm
    # external field (e/nm²) at the ML positions: only the two charges (the rest is switched off; intra-ML d = 0)
    e = np.zeros((len(ml), 3))
    for s, q in ((src[0], Q), (src[1], -Q)):
        d = x[ml] - x[s]
        e += q * d / np.linalg.norm(d, axis=1)[:, None] ** 3
    t = thole_tensor(torch.tensor(x[ml]), alpha)
    b = torch.linalg.inv(torch.diag(torch.repeat_interleave(1.0 / alpha, 3)) - t).numpy()
    mu_model = (b @ e.ravel()).reshape(-1, 3)
    assert np.abs(mu).max() > 1e-5
    assert np.allclose(mu, mu_model, rtol=1e-5, atol=1e-10), np.abs(mu - mu_model).max() / np.abs(mu).max()
