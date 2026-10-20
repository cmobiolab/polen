"""mlmmpol.minimize: the scipy L-BFGS reaches the LocalEnergyMinimizer tolerance (RMS force) and stops at an energy of
the same order as the OpenMM minimizer (Lennard-Jones system with bonds, Reference platform)."""
import numpy as np
import openmm as mm
from openmm import unit

from mlmmpol.minimize import minimize, scipy_minimize


def lj_system(n=120, seed=0):
    rng = np.random.default_rng(seed)
    system = mm.System()
    nb = mm.NonbondedForce()
    nb.setNonbondedMethod(mm.NonbondedForce.CutoffPeriodic)
    nb.setCutoffDistance(0.9)
    bonds = mm.HarmonicBondForce()
    for i in range(n):
        system.addParticle(16.0)
        nb.addParticle(0.0, 0.32, 0.65)
    for i in range(0, n, 2):
        bonds.addBond(i, i + 1, 0.15, 2.0e5)
        nb.addException(i, i + 1, 0.0, 1.0, 0.0)
    system.addForce(nb)
    system.addForce(bonds)
    box = 2.2
    system.setDefaultPeriodicBoxVectors(mm.Vec3(box, 0, 0), mm.Vec3(0, box, 0), mm.Vec3(0, 0, box))
    x = rng.uniform(0, box, (n, 3))
    x[1::2] = x[0::2] + 0.12          # compressed bonds and random overlaps
    return system, x


def context(system, x):
    ctx = mm.Context(system, mm.VerletIntegrator(1e-3), mm.Platform.getPlatformByName('Reference'))
    ctx.setPositions(x * unit.nanometer)
    return ctx


def rms_force(ctx):
    f = ctx.getState(getForces=True).getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)
    return np.sqrt(np.mean(np.asarray(f) ** 2))


def energy(ctx):
    return ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)


def test_scipy_reaches_tolerance():
    system, x = lj_system()
    ctx = context(system, x)
    e_start = energy(ctx)
    e0, e1, frms, n, dmax = scipy_minimize(ctx, tolerance=10.0, max_iterations=5000, max_displacement=None)
    assert abs(e0 - e_start) < 1e-6 * abs(e_start)
    assert e1 < e0
    assert frms < 10.0
    assert abs(energy(ctx) - e1) < 1e-6 * max(1.0, abs(e1))
    # same order of energy as the OpenMM minimizer (in an LJ glass the local minima differ)
    ref = context(system, x)
    mm.LocalEnergyMinimizer.minimize(ref, 10.0, 5000)
    assert rms_force(ref) < 10.0
    assert e1 < 0 and abs(e1 - energy(ref)) < 0.5 * abs(energy(ref))


def test_minimize_uses_openmm_when_possible():
    system, x = lj_system(seed=1)
    ctx = context(system, x)
    assert minimize(ctx, 10.0, 5000, max_displacement=None, log=lambda *a, **k: None) == 'openmm'
    assert rms_force(ctx) < 10.0


def test_bounded_displacement():
    system, x = lj_system(seed=2)
    ctx = context(system, x)
    e0, e1, frms, n, dmax = scipy_minimize(ctx, tolerance=10.0, max_iterations=5000, max_displacement=0.02)
    assert e1 < e0
    assert dmax <= 0.02 + 1e-12
    xf = ctx.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    assert np.abs(np.asarray(xf) - x).max() <= 0.02 + 1e-9
