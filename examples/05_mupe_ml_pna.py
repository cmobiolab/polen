"""Example 5 — MuPE(ML): pNA with ML multipoles, charge penetration, learned polarizability and refitted ML–MM vdW.

The embedding head of the default preset predicts, at every step, the charges, dipoles, quadrupoles, Slater valence
(N, s) and polarizabilities of the ligand atoms from the MACE geometry; the plugin injects them into the AMOEBA
electrostatics and adds the chain rule through the head to the forces. ligand='pna' selects the ML–MM vdW refit of pNA
(polen.available_ligands() lists the ligands of the preset). 5 ps of NPT.
Equivalent command line:
    python -m mlmmpol run --embedding mupe-ml --prmtop data/pna/pna.prmtop --coords data/pna/pna.rst7 \
        --ml-residues PNA --solute PNA data/pna/final.xyz data/pna/final.key --ligand pna --ns 0.005 --out out/05_cli
"""
from pathlib import Path

import mlmmpol as polen

HERE = Path(__file__).parent
D = HERE / 'data' / 'pna'
OUT = HERE / 'out' / '05_pna_mupe_ml'
print('ligands with ML–MM vdW refit in the default preset:', polen.available_ligands())
amo = polen.amber_to_amoeba(D / 'pna.prmtop', D / 'pna.rst7', [('PNA', str(D / 'final.xyz'), str(D / 'final.key'))],
                            OUT / 'amoeba')
ml = amo.atoms_of_residues(['PNA'])
system, force = polen.create_mupe_system(amo.topology, amo.system, ml, multipoles='ml', model='mace-off23-small',
                                         charge=0, ligand='pna', workdir=OUT)
polen.run_md(system, amo.topology, amo.positions, OUT, ns=0.005, equil_ps=2.0, box=amo.box, checkpoint_ps=2)
