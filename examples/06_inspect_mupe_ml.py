"""Example 6 — Inspecting a MuPE(ML) state: energy components, ML-region parameters and response, ligand dipole.

Builds the MuPE(ML) system of example 5 and, for the starting frame:
  - the energy by force group (bonded + MLIP; electrostatics + polarization + vdW);
  - the derivatives the plugin exports for the ML atoms: phi = dE/dq (potential) and E = -dE/dmu (field);
  - the induced dipoles of the ligand atoms (the response of the ML region to its environment).
"""
from pathlib import Path

import numpy as np
import openmm as mm
from openmm import unit

import mlmmpol as polen

HERE = Path(__file__).parent
D = HERE / 'data' / 'pna'
OUT = HERE / 'out' / '06_inspect'
amo = polen.amber_to_amoeba(D / 'pna.prmtop', D / 'pna.rst7', [('PNA', str(D / 'final.xyz'), str(D / 'final.key'))],
                            OUT / 'amoeba')
ml = amo.atoms_of_residues(['PNA'])
system, force = polen.create_mupe_system(amo.topology, amo.system, ml, multipoles='ml', ligand='pna', workdir=OUT)
ctx = mm.Context(system, mm.VerletIntegrator(1e-3), mm.Platform.getPlatformByName('CUDA'), {'Precision': 'double'})
ctx.setPeriodicBoxVectors(*amo.box)
ctx.setPositions(amo.positions)
for g, name in ((0, 'bonded + MLIP'), (1, 'electrostatics + polarization + vdW')):
    e = ctx.getState(getEnergy=True, groups={g}).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    print(f'{name:40s} {e:14.2f} kJ/mol')

der = force.getMLDerivatives(ctx)                       # also: dTheta, dAlpha
phi, field = np.array(der['phi']), np.array(der['field']).reshape(-1, 3)
names = [a.name for a in amo.topology.atoms()]
print('\nML atom   phi (kJ/mol/e)   |E| (kJ/mol/(e nm))')
for k, i in enumerate(ml):
    print(f'{names[i]:8s} {phi[k]:14.2f} {np.linalg.norm(field[k]):18.1f}')

ind = force.getInducedDipoles(ctx)                      # (N, 3) in e·nm; also getLabFramePermanentDipoles, getTotalDipoles
mu_ind = ind[ml].sum(0) * 48.0321                       # e·nm -> D
print(f'\ninduced dipole of the ligand (sum of its atomic induced dipoles): {np.linalg.norm(mu_ind):.2f} D')
