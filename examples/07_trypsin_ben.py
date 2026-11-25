"""Example 7 — Trypsin–benzamidinium (PDB 3PTB) with the three embeddings of the paper.

The ML region is the benzamidinium cation (charge +1, MACE-OMOL-0 extra large), bound to Asp189 in the S1 pocket;
the environment is the protein (ff19SB), OPC water and 0.15 M NaCl (38 416 particles in the Amber files, including the OPC virtual sites; truncated octahedron). The starting
frame is the end of the first 100 ns Amber replica, as for the ML/MM windows of the paper.

    python 07_trypsin_ben.py --embedding mech        # mechanical embedding (Amber, GAFF2/RESP ligand charges)
    python 07_trypsin_ben.py --embedding mupe-ff     # MuPE(FF): AMOEBA, Poltype multipoles on the ligand
    python 07_trypsin_ben.py --embedding mupe-ml     # MuPE(ML): trained head, default preset (benzamidinium refit)

The AMOEBA embeddings need Tinker's pdbxyz for the protein (MLMMPOL_PDBXYZ, docs/installation.md). By default the
run is short (2 ps equilibration + 5 ps NPT); the paper windows used --equil-ps 200 --ns 5, starting from each of the
five Amber replicas (frames in the Zenodo deposit).
"""
import argparse
from pathlib import Path

from openmm import app

import mlmmpol as polen

HERE = Path(__file__).parent
D = HERE / 'data' / 'trypsin_ben'
MODEL, CHARGE = 'mace-omol-0-extra-large', 1

ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
ap.add_argument('--embedding', choices=['mech', 'mupe-ff', 'mupe-ml'], default='mupe-ml')
ap.add_argument('--ns', type=float, default=0.005, help='production length in ns (paper: 5)')
ap.add_argument('--equil-ps', type=float, default=2.0, help='equilibration in ps, not saved (paper: 200)')
ap.add_argument('--preset', default=None, help='MuPE(ML) preset (default: the validated default, docs/presets.md)')
args = ap.parse_args()
OUT = HERE / 'out' / f'07_trypsin_ben_{args.embedding}'

if args.embedding == 'mech':
    top = app.AmberPrmtopFile(str(D / 'trypsin_ben.prmtop')).topology
    ml = polen.ml_atoms_from_residues(top, ['BEN'])
    topology, system = polen.create_mechanical_system(D / 'trypsin_ben.prmtop', ml, model=MODEL, charge=CHARGE)
    polen.set_mts_groups(system)
    crd = app.AmberInpcrdFile(str(D / 'trypsin_ben.rst7'))
    positions, box = crd.getPositions(), crd.getBoxVectors()
else:
    amo = polen.amber_to_amoeba(D / 'trypsin_ben.prmtop', D / 'trypsin_ben.rst7',
                                [('BEN', str(D / 'final.xyz'), str(D / 'final.key'))], OUT / 'amoeba')
    ml = amo.atoms_of_residues(['BEN'])
    if args.embedding == 'mupe-ff':
        system, force = polen.create_mupe_system(amo.topology, amo.system, ml, multipoles='amoeba', model=MODEL,
                                                 charge=CHARGE)
    else:
        system, force = polen.create_mupe_system(amo.topology, amo.system, ml, multipoles='ml', model=MODEL,
                                                 charge=CHARGE, ligand='ben', preset=args.preset, workdir=OUT)
    topology, positions, box = amo.topology, amo.positions, amo.box

polen.run_md(system, topology, positions, OUT, ns=args.ns, equil_ps=args.equil_ps, box=box, checkpoint_ps=2)
