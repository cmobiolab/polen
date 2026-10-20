"""ML/MM with mechanical embedding on an Amber system (the standard openmm-ml protocol).

openmm-ml createMixedSystem (interpolate=False): the ML potential replaces the bonded terms and the nonbonded
interactions among ML atoms; the ML–MM interaction stays that of the Amber force field (ligand RESP charges + LJ).
The openmm-ml PythonForce is replaced by the TorchForce (mlmmpol.mace_torchforce), as in ML/MM-AMOEBA.
"""
import openmm as mm
from openmm import app, unit
from openmmml import MLPotential

from .mace_torchforce import make_mace_torchforce, replace_python_force


def create_mechanical_system(prmtop, ml_atoms, model='mace-off23-small', cutoff=1.0, charge=0, multiplicity=1,
                             precision='single'):
    """Return the Amber (topology, system) with the ML region on MACE; rigid water (OPC), no other constraints."""
    prm = app.AmberPrmtopFile(str(prmtop))
    system = prm.createSystem(nonbondedMethod=app.PME, nonbondedCutoff=cutoff * unit.nanometer,
                              constraints=None, rigidWater=True)
    mixed = MLPotential(model).createMixedSystem(prm.topology, system, list(ml_atoms), interpolate=False,
                                                 charge=charge, multiplicity=multiplicity)
    tf = make_mace_torchforce(model, prm.topology, ml_atoms, precision=precision, periodic=True,
                              charge=charge, multiplicity=multiplicity)
    replace_python_force(mixed, tf)
    exclude_ml_virtual_sites(mixed, ml_atoms)
    # exceptions (incl. the ML–ML exclusions of createMixedSystem) by minimum image: in an ML region with several
    # molecules (e.g. Mg²⁺ + ATP + waters) each molecule may sit in a different periodic image of the coordinates
    for f in mixed.getForces():
        if isinstance(f, mm.NonbondedForce):
            f.setExceptionsUsePeriodicBoundaryConditions(True)
    for i in reversed(range(mixed.getNumForces())):
        if isinstance(mixed.getForce(i), mm.CMMotionRemover):
            mixed.removeForce(i)
    return prm.topology, mixed


def ml_virtual_sites(system, ml_atoms):
    """Virtual sites defined only by ML atoms (e.g. the OPC EP of a water in the ML region)."""
    ml = set(ml_atoms)
    sites = []
    for i in range(system.getNumParticles()):
        if system.isVirtualSite(i):
            vs = system.getVirtualSite(i)
            if all(vs.getParticle(k) in ml for k in range(vs.getNumParticles())):
                sites.append(i)
    return sites


def exclude_ml_virtual_sites(system, ml_atoms):
    """Remove the nonbonded interactions between the ML-region virtual sites and the ML region (createMixedSystem only
    excludes pairs of ML atoms; without this, the EP charge of an ML water would interact with the rest of the ML
    region, which the ML potential already describes). Return the handled sites."""
    sites = ml_virtual_sites(system, ml_atoms)
    if not sites:
        return sites
    group = sorted(set(ml_atoms) | set(sites))
    pairs = [(v, j) for v in sites for j in group if j != v and not (j in sites and j < v)]
    for f in system.getForces():
        if isinstance(f, mm.NonbondedForce):
            for v, j in pairs:
                f.addException(v, j, 0, 1, 0, True)
        elif isinstance(f, mm.CustomNonbondedForce):
            existing = {tuple(sorted(f.getExclusionParticles(k))) for k in range(f.getNumExclusions())}
            for v, j in pairs:
                if tuple(sorted((v, j))) not in existing:
                    f.addExclusion(v, j)
    return sites


def set_mts_groups(system, slow=(mm.NonbondedForce, mm.AmoebaMultipoleForce, mm.AmoebaVdwForce)):
    """Group 1 (slow, outer step): electrostatics/vdW; group 0 (fast): bonded terms and the ML potential."""
    for f in system.getForces():
        f.setForceGroup(1 if isinstance(f, slow) else 0)
