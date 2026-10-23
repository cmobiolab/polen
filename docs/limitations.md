# Limitations (version 0.1.0)

**Method**
- **No covalent boundary**: the ML region must consist of whole molecules (no link atoms).
- **The MLIP does not respond to the field**: the polarization of the ML region comes from the induced point dipoles
  (Thole model). Its molecular polarizability is right, but its response to strongly non-uniform fields (e.g. a cation
  in a salt bridge) is overestimated, which overbinds by ~5–10 kJ/mol in such sites. AMOEBA and the other point-dipole
  embeddings share this behaviour.
- **MuPE(ML) needs a trained head**: the shipped heads cover pNA, benzamidinium and TBB⁻
  ([presets.md](presets.md)); training new ligands is not part of 0.1.0.
- **ML–MM van der Waals**: refitted against ligand–water and ligand–Na⁺ dimers only. Other contacts (Cl⁻, protein
  groups) keep the force-field vdW, which can be too attractive with the ML multipoles.
- **MuPE(FF) with highly charged, flexible ML regions** (e.g. Mg²⁺·ATP) can become unstable (runaway ML–MM attraction).

**Code**
- OpenMM platforms: Reference (CPU) and CUDA. No OpenCL/HIP.
- Tested with OpenMM 8.5.2, PyTorch 2.12.1, openmm-torch 1.5.1, openmm-ml 1.7, MACE 0.3.16 (`environment.yml`).
- MLIPs: the MACE models known to `mlmmpol.mace_torchforce` (MACE-OFF23, MACE-OMOL-0).
- In the mechanical embedding the ML atoms have no bonds in the `System`: trajectories written with
  `enforcePeriodicBox` split the ligand across the box; reassemble it for analysis.
- On CUDA the parameters are single precision on the GPU (as in OpenMM): energies have noise of ~10⁻⁴ kJ/mol.
- `amber_to_amoeba` needs Poltype parameters for every non-standard residue and, for proteins, Tinker's `pdbxyz`.
- Not yet packaged for pip/conda: the plugin is built from source and the repository is used in place.
