"""Example 2 — Converting an Amber system into AMOEBA (needed for MuPE).

Input (examples/data/pna): pna.prmtop + pna.rst7, built with tleap (pna_tleap.in: GAFF2/RESP ligand, OPC water, ions in
a truncated octahedron), and the Poltype 2 parameters of the ligand (final.xyz, final.key). The ligand is typed from
the Poltype files, water and ions from AMOEBA BIO18; the OPC virtual sites are dropped; the box is kept.
Equivalent command line:
    python -m mlmmpol amber2amoeba data/pna/pna.prmtop data/pna/pna.rst7 out/02_pna/amoeba \
        --solute PNA data/pna/final.xyz data/pna/final.key
(proteins also need Tinker's pdbxyz: set MLMMPOL_PDBXYZ and MLMMPOL_BIO18, see docs/installation.md)
"""
from pathlib import Path

import openmm as mm

import mlmmpol as polen

HERE = Path(__file__).parent
D = HERE / 'data' / 'pna'
amo = polen.amber_to_amoeba(D / 'pna.prmtop', D / 'pna.rst7', [('PNA', str(D / 'final.xyz'), str(D / 'final.key'))],
                            HERE / 'out' / '02_pna' / 'amoeba')
ml = amo.atoms_of_residues(['PNA'])
print(f'Tinker files: {amo.xyz}, {amo.key}')
print(f'{amo.topology.getNumAtoms()} atoms ({amo.topology.getNumResidues()} residues), ligand atoms {ml}')
print('forces:', ', '.join(type(f).__name__ for f in amo.system.getForces()))
print('box vectors (nm):', [[round(float(x), 3) for x in v.value_in_unit(mm.unit.nanometer)] for v in amo.box])
