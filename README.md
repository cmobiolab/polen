# POLEN

**POLEN** — POLarizable Embedding for Neural-network potentials, a machine-learning/molecular-mechanics (ML/MM)
simulation tool for [OpenMM](https://openmm.org) with a mutually polarizable AMOEBA embedding.

## Status

Under development (version in `VERSION`). Not yet released on PyPI or conda-forge; the `polen` and `openmm-polen`
PyPI names are reserved by placeholder packages (`packaging/`).

## What it does

- An MLIP (e.g. MACE) describes the region of interest; the environment is the polarizable AMOEBA force field.
- **MuPE** (mutual polarizable embedding): the ML region and the AMOEBA environment polarize each other in a single
  linear system, with multipolar PME, on CPU (Reference) and GPU (CUDA).
  - **MuPE(FF)**: the ML region keeps its force-field multipoles and polarizabilities.
  - **MuPE(ML)**: an equivariant head predicts, at every step, the ML-region multipoles, Slater valence (charge
    penetration) and geometry-dependent polarizabilities; forces include the analytic derivatives through the head.
    The ML–MM van der Waals is refitted to QM ligand–water dimers (AMOEBA+ strategy).

## Layout

| path | content |
|---|---|
| `plugin/` | OpenMM plugin (C++/CUDA, SWIG wrapper) |
| `mlmmpol/` | Python tools: mixed ML/AMOEBA systems, head export, presets, ML vdW, minimizer |
| `qm_ref/heads/` | trained heads used by the presets, `presets.json`, `ml_vdw_v1.json` |
| `ref_torch/` | PyTorch reference implementation (used by the tests) |
| `tests/`, `systems/`, `qm_ref/nma_frames.npz` | tests and their small data files |
| `docs/plugin.md` | technical summary of the plugin (Portuguese) |

Installation: see [INSTALL.md](INSTALL.md). Version: [VERSION](VERSION).

## License

GNU Lesser General Public License v3.0 or later (LGPL-3.0-or-later); see `COPYING.LESSER` and `COPYING`.
Third-party code in `plugin/third_party` keeps its own license (see `plugin/third_party/README.md`).
