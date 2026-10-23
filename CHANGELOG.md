# Changelog

## 0.1.0 (unreleased)

First usable version.

- **High-level API** (`mlmmpol`): `amber_to_amoeba` (Amber + Poltype → AMOEBA, with selection by Amber residue
  names), `create_mechanical_system`, `create_mupe_system` (MuPE(FF) and MuPE(ML): head, preset, ML–MM vdW refit,
  layered polarization), `run_md` (restartable MD with checkpoints), `available_ligands`, `load_plugin`.
- **Command line**: `python -m mlmmpol amber2amoeba` and `python -m mlmmpol run`.
- **Examples** (`examples/`) with data, and the **manual** (`docs/`).
- **Presets**: default `alpha_s0_vdw1`; new `alpha_s0_vdw2` with ML–Na⁺ vdW pair factors.
- **Plugin**: the CUDA kernel loads the ML head with PyTorch's CUDA context current and releases it in a destructor
  (creating a second Context, or `reinitialize()`, with an ML head crashed before); ML-class × ion vdW pairs
  (`apply_ml_vdw(..., pair_scale_R, pair_scale_eps)`).
- **Minimizer**: systems with constraints (rigid water) use the OpenMM minimizer; the bounded L-BFGS ignored them.
- **Paths**: no default installation paths; `MLMMPOL_BIO18` and `MLMMPOL_PDBXYZ` (or `pdbxyz` in the PATH).
- Code comments, docstrings, messages and documentation in English.

## 0.1.0.dev2

Development snapshot (private repository).
