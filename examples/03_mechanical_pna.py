"""Example 3 — Mechanical embedding (reference): pNA in OPC water with the Amber force field.

The ligand is described by MACE-OFF23; its interaction with the water is the GAFF2/RESP + OPC one (openmm-ml
createMixedSystem, with the MLIP as a TorchForce on the GPU). 5 ps of NPT after 2 ps of equilibration.
"""
from pathlib import Path

from openmm import app

import mlmmpol as polen

HERE = Path(__file__).parent
D = HERE / 'data' / 'pna'
top = app.AmberPrmtopFile(str(D / 'pna.prmtop')).topology
ml = polen.ml_atoms_from_residues(top, ['PNA'])
topology, system = polen.create_mechanical_system(D / 'pna.prmtop', ml, model='mace-off23-small', charge=0)
polen.set_mts_groups(system)
crd = app.AmberInpcrdFile(str(D / 'pna.rst7'))
polen.run_md(system, topology, crd.getPositions(), HERE / 'out' / '03_pna_mechanical', ns=0.005, equil_ps=2.0,
             box=crd.getBoxVectors(), checkpoint_ps=2)
