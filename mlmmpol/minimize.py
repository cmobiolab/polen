"""Robust energy minimization for systems with the MLPol plugin.

The OpenMM 8 LocalEnergyMinimizer (CommonMinimizeKernel) recomputes the forces on a CPU platform when, during the
line search, they overflow the GPU fixed point (trial step with overlapping atoms). The plugin has no CPU kernel,
and creating that context fails ("Specified a Platform for a Context which does not support all required
kernels"). Seen in the AMOEBA windows of Mg·ATP and with a random head.

Cause: the AMOEBA buffered 14-7 vdW is finite at r = 0 while the multipolar attraction diverges, so there is a
barrier (~10⁶ kJ/mol) with a "hole" behind it; the same holds for MACE at distances it has never seen (O···O at
0.7 Å with an energy of −4×10¹⁰ kJ/mol). Dynamics never gets there, but the L-BFGS line search takes trial steps
long enough to cross the barrier (the mixed system and pure AMOEBA give the same interaction: not a plugin
defect).

`minimize(..., max_displacement=d)` (default in the window scripts): scipy L-BFGS-B with each coordinate bounded to
±d around the starting point (0.03 nm). An equilibrated frame does not need to move further, and no pair can cross
the barrier. Energy and forces come from the context itself (GPU; float64 on the scipy side); trial points with a
non-finite energy become a barrier.
`minimize(..., max_displacement=None)`: the OpenMM LocalEnergyMinimizer; if it asks for the CPU platform, the
bounded L-BFGS.
"""
import numpy as np
import openmm as mm
from openmm import unit

KJ = unit.kilojoule_per_mole
NM = unit.nanometer
CPU_FALLBACK_MSG = 'does not support all required kernels'


def scipy_minimize(context, tolerance=10.0, max_iterations=5000, max_displacement=0.03, progress=None):
    """L-BFGS (scipy) on the context. Stops when the RMS force per component < tolerance (kJ/mol/nm), like the
    LocalEnergyMinimizer (with bounds: projected force, without the components stuck at a bound), when the relative
    energy stops decreasing (default L-BFGS-B criterion) or at max_iterations. Each coordinate stays within
    ±max_displacement (nm) of the initial one.
    Returns (initial E, final E, final RMS force, number of evaluations, largest displacement in nm)."""
    from scipy.optimize import minimize

    x0 = context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(NM).astype(np.float64)
    shape = x0.shape
    n_eval = [0]
    best = {'e': np.inf, 'x': x0.copy()}
    last = {'x': None, 'g': None}

    def fun(x):
        n_eval[0] += 1
        context.setPositions(x.reshape(shape) * NM)
        context.computeVirtualSites()
        st = context.getState(getEnergy=True, getForces=True)
        e = st.getPotentialEnergy().value_in_unit(KJ)
        f = st.getForces(asNumpy=True).value_in_unit(KJ / NM)
        if not np.isfinite(e) or not np.all(np.isfinite(f)):
            return 1e30, np.zeros_like(x)
        if e < best['e']:
            best['e'], best['x'] = e, x.copy()
        g = -f.ravel().astype(np.float64)
        last['x'], last['g'] = x.copy(), g
        return e, g

    class Converged(Exception):
        pass

    lo = None if max_displacement is None else x0.ravel() - max_displacement
    hi = None if max_displacement is None else x0.ravel() + max_displacement
    it = [0]

    def callback(xk):
        it[0] += 1
        g = last['g'] if last['x'] is not None and np.array_equal(xk, last['x']) else fun(xk)[1]
        if lo is not None:                    # projected gradient: components stuck at a bound do not count
            g = np.where(((xk <= lo + 1e-12) & (g > 0)) | ((xk >= hi - 1e-12) & (g < 0)), 0.0, g)
        if progress and it[0] % 200 == 0:
            progress(f'  L-BFGS it {it[0]}: E {best["e"]:.1f} kJ/mol, RMS force (projected) '
                     f'{np.sqrt(np.mean(g * g)):.2f}', flush=True)
        if np.sqrt(np.mean(g * g)) < tolerance:
            raise Converged

    e0 = fun(x0.ravel())[0]
    try:
        bounds = None if lo is None else np.stack([lo, hi], 1)
        minimize(fun, x0.ravel(), jac=True, method='L-BFGS-B', callback=callback, bounds=bounds,
                 options={'maxiter': max_iterations, 'maxcor': 20, 'gtol': 0.0})
    except Converged:
        pass
    # end at the best configuration found
    context.setPositions(best['x'].reshape(shape) * NM)
    context.computeVirtualSites()
    f = context.getState(getForces=True).getForces(asNumpy=True).value_in_unit(KJ / NM)
    dmax = float(np.abs(best['x'].reshape(shape) - x0).max())
    return e0, best['e'], float(np.sqrt(np.mean(np.asarray(f) ** 2))), n_eval[0], dmax


def minimize(context, tolerance=10.0, max_iterations=5000, max_displacement=0.03, log=print):
    """Bounded L-BFGS (default) or, with max_displacement=None, the OpenMM LocalEnergyMinimizer with the bounded
    L-BFGS as fallback. Returns 'scipy' or 'openmm'."""
    if max_displacement is None:
        x0 = context.getState(getPositions=True).getPositions()
        try:
            mm.LocalEnergyMinimizer.minimize(context, tolerance, max_iterations)
            return 'openmm'
        except mm.OpenMMException as exc:
            if CPU_FALLBACK_MSG not in str(exc):
                raise
        log('the OpenMM minimizer asked for the CPU platform (forces too large on the GPU); using the bounded L-BFGS',
            flush=True)
        context.setPositions(x0)
        context.computeVirtualSites()
        max_displacement = 0.03
    e0, e1, frms, n, dmax = scipy_minimize(context, tolerance, max_iterations, max_displacement, progress=log)
    log(f'bounded L-BFGS (±{max_displacement * 10:.1f} Å): E {e0:.1f} -> {e1:.1f} kJ/mol, total RMS force {frms:.2f} '
        f'kJ/mol/nm, {n} evaluations, largest displacement {dmax * 10:.2f} Å', flush=True)
    return 'scipy'
