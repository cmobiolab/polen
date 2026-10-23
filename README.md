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

## Quick start

```python
import mlmmpol as polen

amoeba = polen.amber_to_amoeba('pna.prmtop', 'pna.rst7', [('PNA', 'final.xyz', 'final.key')], 'work/amoeba')
ml = amoeba.atoms_of_residues(['PNA'])
system, force = polen.create_mupe_system(amoeba.topology, amoeba.system, ml, multipoles='ml', ligand='pna',
                                         workdir='work')
polen.run_md(system, amoeba.topology, amoeba.positions, 'work', ns=1.0, box=amoeba.box)
```

Or from the command line: `python -m mlmmpol run --embedding mupe-ml ...` (see the manual).

**Manual:** [`docs/index.md`](docs/index.md) (installation, concepts, user guide, API reference, presets, examples,
limitations). **Examples:** [`examples/`](examples) (runnable scripts with data). **Changes:** [`CHANGELOG.md`](CHANGELOG.md).

## Layout

| path | content |
|---|---|
| `plugin/` | OpenMM plugin (C++/CUDA, SWIG wrapper) |
| `mlmmpol/` | Python package: high-level API (`create_mupe_system`, `amber_to_amoeba`, `run_md`, ...), command line, mixed systems, head export, presets, ML vdW, minimizer |
| `qm_ref/heads/` | trained heads used by the presets, `presets.json`, ML–MM vdW refits |
| `examples/` | example scripts and their data |
| `docs/` | manual |
| `ref_torch/` | PyTorch reference implementation (used by the tests) |
| `tests/`, `systems/`, `qm_ref/nma_frames.npz` | tests and their small data files |

Installation: see [`docs/installation.md`](docs/installation.md). Version: [VERSION](VERSION).

## License

GNU Lesser General Public License v3.0 or later (LGPL-3.0-or-later); see `COPYING.LESSER` and `COPYING`.
Third-party code in `plugin/third_party` keeps its own license (see `plugin/third_party/README.md`).
