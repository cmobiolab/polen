"""Construction of the AMOEBA test systems in OpenMM (no cutoff, Reference platform)."""

from pathlib import Path

import numpy as np
import openmm as mm
from openmm import app, unit

SYSTEMS = Path(__file__).resolve().parents[1] / 'systems'


def make_traceless(force):
    """Project out the residual trace of the quadrupoles (the same as ref_torch.from_openmm does)."""
    for i in range(force.getNumMultipoles()):
        p = force.getMultipoleParameters(i)
        q = np.array(p[2]._value if hasattr(p[2], '_value') else p[2]).reshape(3, 3)
        p[2] = list((q - np.trace(q) / 3 * np.eye(3)).ravel())
        force.setMultipoleParameters(i, *p)


def build_amoeba(pdb_name='peptide_shell.pdb', polarization='mutual', epsilon=1e-10,
                 traceless=True, edit=None):
    """Return (system, multipole_force, context, positions in nm).

    edit(force, topology) allows changing the parameters before the Context is created.
    """
    pdb = app.PDBFile(str(SYSTEMS / pdb_name))
    ff = app.ForceField('amoeba2018.xml')
    system = ff.createSystem(pdb.topology, nonbondedMethod=app.NoCutoff,
                             polarization=polarization, mutualInducedTargetEpsilon=epsilon)
    for f in system.getForces():
        f.setForceGroup(1 if isinstance(f, mm.AmoebaMultipoleForce) else 0)
    multipole = next(f for f in system.getForces() if isinstance(f, mm.AmoebaMultipoleForce))
    multipole.setMutualInducedMaxIterations(1000)
    if traceless:
        make_traceless(multipole)
    if edit is not None:
        edit(multipole, pdb.topology)
    context = mm.Context(system, mm.VerletIntegrator(1e-3), mm.Platform.getPlatformByName('Reference'))
    context.setPositions(pdb.positions)
    pos = np.array(pdb.positions.value_in_unit(unit.nanometer))
    return system, multipole, context, pos


def multipole_energy_forces(context):
    st = context.getState(getEnergy=True, getForces=True, groups={1})
    e = st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    f = np.array(st.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer))
    return e, f
