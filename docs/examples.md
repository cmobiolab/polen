# Examples

The scripts in `examples/` run from that directory (`cd examples; python 01_...py`) with the repository root in the
`PYTHONPATH` and, for examples 2–8, `MLMMPOL_BIO18` set; examples 7 and 8 (proteins) also need Tinker's
`pdbxyz` (`MLMMPOL_PDBXYZ`) for the AMOEBA embeddings ([installation.md](installation.md)). Outputs go to
`examples/out/`. Times are for one RTX 5070 GPU, after the first use (the first run also downloads the MACE model).

| Script | Shows | Time |
|---|---|---|
| `01_mupe_ff_nma_amoeba2018.py` | MuPE(FF) on a system built with OpenMM's `ForceField('amoeba2018.xml')` (NMA + 17 waters, no PBC); energy by component; 2 ps NVT | ~15 s |
| `02_prepare_pna_amber_to_amoeba.py` | Amber (prmtop + rst7) + Poltype → AMOEBA conversion; selecting the ligand by its Amber residue name | a few s |
| `03_mechanical_pna.py` | mechanical embedding of pNA in OPC water (Amber + MACE-OFF23); 5 ps NPT | ~30 s |
| `04_mupe_ff_pna.py` | MuPE(FF): the same pNA system in polarizable AMOEBA water; 5 ps NPT | ~45 s |
| `05_mupe_ml_pna.py` | MuPE(ML) with the default preset and the pNA vdW refit; 5 ps NPT | ~75 s |
| `06_inspect_mupe_ml.py` | energy components, the ML derivatives φ and **E** exported by the plugin, and the induced dipole of the ligand | ~10 s |
| `07_trypsin_ben.py` | trypsin–benzamidinium (PDB 3PTB) with `--embedding mech`, `mupe-ff` or `mupe-ml` (benzamidinium vdW refit); MACE-OMOL-0, charge +1; 5 ps NPT | a few min |
| `08_ck2_tbb.py` | CK2α–TBB⁻ (PDB 6TLL), `--pose A` (Lys68 salt bridge) or `B` (hinge halogen bonds), with the same three embeddings (TBB⁻ vdW refit; `--preset alpha_s0_vdw2` adds the ML–Na⁺ correction); MACE-OMOL-0, charge −1; 5 ps NPT | a few min |

Examples 3–5, 7 and 8 cover the ML/MM windows of the paper. They run 2 ps of equilibration and 5 ps of production by
default; `--equil-ps 200 --ns 5` (examples 7 and 8) gives the protocol of the paper, whose windows started from the
final frame of each of five 100 ns Amber replicas. The data here contain the frame of the first replica; the other
replicas, all window outputs and the analysis scripts are in the Zenodo deposit of the paper.

## Data (`examples/data/`)

- `nma/nma_shell.pdb`: N-methylacetamide (ACE–NME) with 17 waters.
- `pna/`: p-nitroaniline in a truncated octahedron of OPC water with Na⁺/Cl⁻ (6196 atoms):
  - `pna_tleap.in`, `pna.mol2`, `pna.frcmod`, `pna_lig.pdb`: the tleap input and ligand files used to build it
    (GAFF2 with RESP charges);
  - `pna.prmtop`, `pna.rst7`: the Amber topology and an equilibrated frame (end of a 100 ns MD);
  - `final.xyz`, `final.key`: the Poltype 2 AMOEBA parameters of pNA.
- `trypsin_ben/`: trypsin–benzamidinium (PDB 3PTB) in a truncated octahedron of OPC water with 0.15 M NaCl (38 416
  particles, including the OPC virtual sites):
  - `trypsin_ben_tleap.in`, `3PTB_amber.pdb`, `ben_xtal.pdb`, `xwat.pdb`, `ben.mol2`, `ben_final.frcmod`: the tleap
    input and the files used to build it (ff19SB, OPC, GAFF2/RESP ligand with the refined phenyl–amidinium torsion);
  - `trypsin_ben.prmtop`, `trypsin_ben.rst7`: the Amber topology and the final frame of the first 100 ns replica;
  - `final.xyz`, `final.key`, `ben.sdf`: the Poltype 2 AMOEBA parameters of benzamidinium.
- `ck2_tbb/`: CK2α–TBB⁻ (PDB 6TLL) in the two binding poses (81 883 and 81 871 particles):
  - `ck2_tbb_anion_pose{A,B}_tleap.in`, `6TLL_amber.pdb`, `tbb_anion_pose{A,B}.pdb`, `xwat_anion_pose{A,B}.pdb`,
    `tbb_anion.mol2`, `tbb_anion.frcmod`: the tleap inputs and the files used to build them;
  - `ck2_tbb_anion_pose{A,B}.prmtop`, `.rst7`: the Amber topologies and the final frames of the first 100 ns replicas;
  - `final.xyz`, `final.key`, `tbb_anion.sdf`: the Poltype 2 AMOEBA parameters of TBB⁻.

## What to expect

- Example 1 prints the energy split into bonded + MLIP (group 0) and electrostatics + polarization + vdW (group 1).
- Examples 3–5 run the same system with the three embeddings: compare `out/*/energies.csv` (temperature ≈ 300 K,
  density ≈ 1.0 g/cm³) and the speeds in `out/*/done` (about 30, 17 and 9 ns/day).
- Examples 7 and 8 write to `out/07_trypsin_ben_<embedding>/` and `out/08_ck2_tbb_pose<A|B>_<embedding>/`. Typical
  speeds on an RTX 5070 (mechanical / MuPE(FF) / MuPE(ML)): trypsin 4.8 / 3.2 / 2.8 ns/day, CK2 5.4 / 2.4 / 2.2
  ns/day; the first run of an AMOEBA embedding also converts the protein with `pdbxyz` (some tens of seconds).
- Example 6 prints, for each pNA atom, the electrostatic potential and field of the environment at the atom, and the
  induced dipole of the ligand (≈ 2 D): the response of the ML region to the solvent.
- Running an example again with an existing `out/` directory continues from its checkpoint (or does nothing if it
  finished); delete the directory to start over.

## Command line

Example 5 as a single command:

```bash
python -m mlmmpol run --embedding mupe-ml --prmtop data/pna/pna.prmtop --coords data/pna/pna.rst7 \
    --ml-residues PNA --solute PNA data/pna/final.xyz data/pna/final.key --ligand pna --ns 0.005 --out out/05_cli
```
