"""Example 1 — MuPE(FF) from an OpenMM AMOEBA force field (no Amber, no Poltype).

N-methylacetamide (ACE–NME) with 17 waters, AMOEBA 2018 (amoeba2018.xml), no periodic boundaries. The NMA is the ML
region (MACE-OFF23); its force-field multipoles and polarizabilities polarize, and are polarized by, the waters.
Prints the energy by component and runs 2 ps of NVT dynamics.
"""
from pathlib import Path

import openmm as mm
from openmm import app, unit

import mlmmpol as polen

HERE = Path(__file__).parent
pdb = app.PDBFile(str(HERE / 'data' / 'nma' / 'nma_shell.pdb'))
ff = app.ForceField('amoeba2018.xml')
amoeba = ff.createSystem(pdb.topology, nonbondedMethod=app.NoCutoff, polarization='mutual',
                         mutualInducedTargetEpsilon=1e-5)
ml = polen.ml_atoms_from_residues(pdb.topology, ['ACE', 'NME'])
system, force = polen.create_mupe_system(pdb.topology, amoeba, ml, multipoles='amoeba', model='mace-off23-small')
print(f'{len(ml)} ML atoms, {system.getNumParticles()} particles, {system.getNumForces()} forces')

# energy by force group: group 0 = bonded terms + MLIP (fast), group 1 = electrostatics + vdW (slow)
ctx = mm.Context(system, mm.VerletIntegrator(1e-3), mm.Platform.getPlatformByName('CUDA'))
ctx.setPositions(pdb.positions)
for g, name in ((0, 'bonded + MLIP'), (1, 'electrostatics + polarization + vdW')):
    e = ctx.getState(getEnergy=True, groups={g}).getPotentialEnergy()
    print(f'  {name:40s} {e.value_in_unit(unit.kilojoule_per_mole):12.2f} kJ/mol')
del ctx

polen.run_md(system, pdb.topology, pdb.positions, HERE / 'out' / '01_nma_mupe_ff', ns=0.002, equil_ps=1.0,
             pressure=None, report_ps=0.1, checkpoint_ps=10)
