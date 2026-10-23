"""Example 4 — MuPE(FF): pNA in polarizable AMOEBA water, with the Poltype multipoles on the ML region.

Same system as example 3, converted to AMOEBA (example 2). The ligand keeps its force-field (Poltype) multipoles and
polarizabilities, and the ML region and the water polarize each other mutually (PME). 5 ps of NPT.
"""
from pathlib import Path

import mlmmpol as polen

HERE = Path(__file__).parent
D = HERE / 'data' / 'pna'
OUT = HERE / 'out' / '04_pna_mupe_ff'
amo = polen.amber_to_amoeba(D / 'pna.prmtop', D / 'pna.rst7', [('PNA', str(D / 'final.xyz'), str(D / 'final.key'))],
                            OUT / 'amoeba')
ml = amo.atoms_of_residues(['PNA'])
system, force = polen.create_mupe_system(amo.topology, amo.system, ml, multipoles='amoeba', model='mace-off23-small')
polen.run_md(system, amo.topology, amo.positions, OUT, ns=0.005, equil_ps=2.0, box=amo.box, checkpoint_ps=2)
