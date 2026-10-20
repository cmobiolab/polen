"""mlmmpol.zbl: OpenMM energy (ML–MM and ML–ML pairs) = ZBL reference with a smooth switch; zero beyond r_out;
continuous force (finite differences)."""
import numpy as np
import openmm as mm
import pytest
from openmm import app, unit

from mlmmpol.zbl import R_COV, add_zbl, zbl_energy

ELEMENTS = {1: 'H', 8: 'O', 12: 'Mg', 15: 'P'}


def two_atom_system(z1, z2, ml_both, factor_intra=None):
    top = app.Topology()
    ch = top.addChain()
    for k, z in enumerate((z1, z2)):
        res = top.addResidue(f'R{k}', ch)
        top.addAtom(ELEMENTS[z], app.Element.getByAtomicNumber(z), res)
    system = mm.System()
    for z in (z1, z2):
        system.addParticle(1.0)
    system.setDefaultPeriodicBoxVectors(mm.Vec3(3, 0, 0), mm.Vec3(0, 3, 0), mm.Vec3(0, 0, 3))
    add_zbl(system, top, [0, 1] if ml_both else [0], factor_intra=factor_intra)
    return system


@pytest.mark.parametrize('z1,z2', [(1, 8), (15, 15), (12, 8), (1, 1)])
@pytest.mark.parametrize('ml_both', [False, True])
def test_energy_matches_reference(z1, z2, ml_both):
    system = two_atom_system(z1, z2, ml_both)
    ctx = mm.Context(system, mm.VerletIntegrator(1e-3), mm.Platform.getPlatformByName('Reference'))
    rout = 0.8 * (R_COV[z1] + R_COV[z2])
    for r in (0.02, 0.05, 0.4 * rout, 0.7 * rout, 0.95 * rout, 1.1 * rout, 0.25):
        ctx.setPositions([mm.Vec3(1, 1, 1), mm.Vec3(1 + r, 1, 1)] * unit.nanometer)
        e = ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        assert e == pytest.approx(zbl_energy(z1, z2, r), rel=1e-6, abs=1e-8)
        if r >= rout:
            assert e == 0.0
        # force = −dE/dr (finite differences)
        f = ctx.getState(getForces=True).getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)
        h = 1e-6
        fd = -(zbl_energy(z1, z2, r + h) - zbl_energy(z1, z2, r - h)) / (2 * h)
        assert f[1][0] == pytest.approx(fd, rel=1e-4, abs=1e-4)


def test_magnitudes():
    # H–O: strong at 0.5 Å, zero from 0.78 Å on; Mg–O zero at the coordination distance (2.0 Å)
    assert zbl_energy(1, 8, 0.05) > 300
    assert zbl_energy(1, 8, 0.08) == 0.0
    assert zbl_energy(12, 8, 0.20) == 0.0


def test_intra_factor_and_skip():
    # ML–ML pair with intra factor 1.0: r_out = R_cov,i + R_cov,j
    system = two_atom_system(8, 8, True, factor_intra=1.0)
    ctx = mm.Context(system, mm.VerletIntegrator(1e-3), mm.Platform.getPlatformByName('Reference'))
    r = 0.11                                         # between 0.8·1.32 and 1.0·1.32 Å
    ctx.setPositions([mm.Vec3(1, 1, 1), mm.Vec3(1 + r, 1, 1)] * unit.nanometer)
    e = ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    assert e == pytest.approx(zbl_energy(8, 8, r, factor=1.0), rel=1e-6) and e > 0 and zbl_energy(8, 8, r) == 0.0
