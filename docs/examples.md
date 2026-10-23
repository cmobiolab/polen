# Examples

The scripts in `examples/` run from that directory (`cd examples; python 01_...py`) with the repository root in the
`PYTHONPATH` and, for examples 2–6, `MLMMPOL_BIO18` set ([installation.md](installation.md)). Outputs go to
`examples/out/`. Times are for one RTX 5070 GPU, after the first use (the first run also downloads the MACE model).

| Script | Shows | Time |
|---|---|---|
| `01_mupe_ff_nma_amoeba2018.py` | MuPE(FF) on a system built with OpenMM's `ForceField('amoeba2018.xml')` (NMA + 17 waters, no PBC); energy by component; 2 ps NVT | ~15 s |
| `02_prepare_pna_amber_to_amoeba.py` | Amber (prmtop + rst7) + Poltype → AMOEBA conversion; selecting the ligand by its Amber residue name | a few s |
| `03_mechanical_pna.py` | mechanical embedding of pNA in OPC water (Amber + MACE-OFF23); 5 ps NPT | ~30 s |
| `04_mupe_ff_pna.py` | MuPE(FF): the same pNA system in polarizable AMOEBA water; 5 ps NPT | ~45 s |
| `05_mupe_ml_pna.py` | MuPE(ML) with the default preset and the pNA vdW refit; 5 ps NPT | ~75 s |
| `06_inspect_mupe_ml.py` | energy components, the ML derivatives φ and **E** exported by the plugin, and the induced dipole of the ligand | ~10 s |

## Data (`examples/data/`)

- `nma/nma_shell.pdb`: N-methylacetamide (ACE–NME) with 17 waters.
- `pna/`: p-nitroaniline in a truncated octahedron of OPC water with Na⁺/Cl⁻ (6196 atoms):
  - `pna_tleap.in`, `pna.mol2`, `pna.frcmod`, `pna_lig.pdb`: the tleap input and ligand files used to build it
    (GAFF2 with RESP charges);
  - `pna.prmtop`, `pna.rst7`: the Amber topology and an equilibrated frame (end of a 100 ns MD);
  - `final.xyz`, `final.key`: the Poltype 2 AMOEBA parameters of pNA.

## What to expect

- Example 1 prints the energy split into bonded + MLIP (group 0) and electrostatics + polarization + vdW (group 1).
- Examples 3–5 run the same system with the three embeddings: compare `out/*/energies.csv` (temperature ≈ 300 K,
  density ≈ 1.0 g/cm³) and the speeds in `out/*/done` (about 30, 17 and 9 ns/day).
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
