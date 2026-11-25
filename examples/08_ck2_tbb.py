"""Example 8 — CK2α–TBB⁻ (PDB 6TLL) in the two binding poses, with the three embeddings of the paper.

The ML region is the 4,5,6,7-tetrabromobenzotriazole anion (charge −1, MACE-OMOL-0 extra large). Pose A: salt bridge
with Lys68; pose B: halogen bonds to the hinge (Glu114, Val116). Environment: CK2α (ff19SB), OPC water and 0.15 M NaCl
(81 883 and 81 871 particles in poses A and B, including the OPC virtual sites; truncated octahedron). The starting frame of each pose is the end of the first 100 ns Amber replica.

    python 08_ck2_tbb.py --pose B --embedding mech       # mechanical embedding (RESP charges, no σ-hole)
    python 08_ck2_tbb.py --pose B --embedding mupe-ff    # MuPE(FF): AMOEBA, Poltype multipoles on TBB⁻
    python 08_ck2_tbb.py --pose B --embedding mupe-ml    # MuPE(ML): trained head, default preset (TBB⁻ refit)

The paper windows used the default preset; the pose A window in which a Na⁺ ion reached the triazolate N (replica 3)
was rerun with --preset alpha_s0_vdw2, which adds the ML–Na⁺ pair correction (docs/presets.md). The AMOEBA embeddings
need Tinker's pdbxyz for the protein (MLMMPOL_PDBXYZ, docs/installation.md). By default the run is short (2 ps
equilibration + 5 ps NPT); the paper windows used --equil-ps 200 --ns 5, starting from each of the five Amber replicas
(frames in the Zenodo deposit).
"""
import argparse
from pathlib import Path

from openmm import app

import mlmmpol as polen

HERE = Path(__file__).parent
D = HERE / 'data' / 'ck2_tbb'
MODEL, CHARGE = 'mace-omol-0-extra-large', -1

ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
ap.add_argument('--pose', choices=['A', 'B'], default='B')
ap.add_argument('--embedding', choices=['mech', 'mupe-ff', 'mupe-ml'], default='mupe-ml')
ap.add_argument('--ns', type=float, default=0.005, help='production length in ns (paper: 5)')
ap.add_argument('--equil-ps', type=float, default=2.0, help='equilibration in ps, not saved (paper: 200)')
ap.add_argument('--preset', default=None, help='MuPE(ML) preset (default: the validated default; alpha_s0_vdw2 adds '
                                               'the ML–Na⁺ correction)')
args = ap.parse_args()
name = f'ck2_tbb_anion_pose{args.pose}'
prmtop, rst7 = D / f'{name}.prmtop', D / f'{name}.rst7'
OUT = HERE / 'out' / f'08_ck2_tbb_pose{args.pose}_{args.embedding}'

if args.embedding == 'mech':
    top = app.AmberPrmtopFile(str(prmtop)).topology
    ml = polen.ml_atoms_from_residues(top, ['TBB'])
    topology, system = polen.create_mechanical_system(prmtop, ml, model=MODEL, charge=CHARGE)
    polen.set_mts_groups(system)
    crd = app.AmberInpcrdFile(str(rst7))
    positions, box = crd.getPositions(), crd.getBoxVectors()
else:
    amo = polen.amber_to_amoeba(prmtop, rst7, [('TBB', str(D / 'final.xyz'), str(D / 'final.key'))], OUT / 'amoeba')
    ml = amo.atoms_of_residues(['TBB'])
    if args.embedding == 'mupe-ff':
        system, force = polen.create_mupe_system(amo.topology, amo.system, ml, multipoles='amoeba', model=MODEL,
                                                 charge=CHARGE)
    else:
        system, force = polen.create_mupe_system(amo.topology, amo.system, ml, multipoles='ml', model=MODEL,
                                                 charge=CHARGE, ligand='tbb', preset=args.preset, workdir=OUT)
    topology, positions, box = amo.topology, amo.positions, amo.box

polen.run_md(system, topology, positions, OUT, ns=args.ns, equil_ps=args.equil_ps, box=box, checkpoint_ps=2)
