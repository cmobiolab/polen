# Installing POLEN (development version)

Tested on Linux x86-64 with NVIDIA GPUs (RTX 5070, driver CUDA 12.8; A30 on CUDA 12.6 with the conda toolkit 12.9).

## 1. Environment

```bash
conda env create -f environment.yml          # OpenMM 8.5.2, PyTorch 2.12.1, openmm-torch 1.5.1, openmm-ml 1.7, MACE
conda activate mlmm-pol                       # (rename in environment.yml if you prefer, e.g. "polen")
```

On machines whose login node has no GPU driver, create the environment with `CONDA_OVERRIDE_CUDA=12.9` so that
conda selects CUDA builds of PyTorch/OpenMM.

## 2. Build the plugin (C++/CUDA + Python wrapper)

```bash
export OPENMM_DIR=$CONDA_PREFIX               # OpenMM installation (the conda environment)
export CUDA_ROOT=/usr/local/cuda-12.8         # CUDA toolkit with nvcc (or $CONDA_PREFIX with the conda toolkit)
bash plugin/build.sh
```

`build.sh` configures with CMake, builds the Reference and CUDA platforms, installs the libraries in
`plugin/install`, generates the SWIG wrapper and installs the `mlpolplugin` Python module in the active environment.
`plugin/build_carbono.sh` is the variant for a cluster with older glibc, using the conda compilers and CUDA toolkit.

Manual alternative (no install into the environment):

```bash
cmake -S plugin -B plugin/build -DOPENMM_DIR=$OPENMM_DIR -DCUDAToolkit_ROOT=$CUDA_ROOT \
      -DCMAKE_CUDA_COMPILER=$CUDA_ROOT/bin/nvcc -DCMAKE_INSTALL_PREFIX=$PWD/plugin/install
cmake --build plugin/build -j8 && cmake --install plugin/build
cd plugin/build && mkdir -p python && cd python
swig -python -c++ -I$OPENMM_DIR/include -o MLPolPluginWrapper.cpp ../../python/mlpolplugin.i
cp ../../python/setup.py . && OPENMM_DIR=$OPENMM_DIR MLPOL_INSTALL=$PWD/../../install python setup.py build_ext --inplace
export PYTHONPATH=$PWD:$PYTHONPATH            # then also add the repository root for `mlmmpol`
```

## 3. Test

```bash
export PYTHONPATH=$PWD:$PYTHONPATH            # repository root (mlmmpol, ref_torch)
python -m pytest -q tests/                    # CUDA tests are skipped when no GPU is visible
```

A few tests read data from the original project (ML/MM window outputs) and are skipped or fail outside it.

## 4. Paths used by the Amber → AMOEBA conversion

`mlmmpol.amber2amoeba` (and `amber_to_amoeba`) needs the AMOEBA BIO18 parameter file and, for systems with protein
residues, Tinker's `pdbxyz`:

```bash
export MLMMPOL_BIO18=/path/to/poltype2/ParameterFiles/amoebabio18.prm   # the Poltype 2 copy (ion types 352/361/357/358)
export MLMMPOL_PDBXYZ=/path/to/tinker/bin/pdbxyz                         # or have pdbxyz in the PATH
```

Use the `amoebabio18.prm` distributed with Poltype 2: other copies number the ion types differently.

## 5. Plugin location

`mlmmpol.load_plugin()` (called by `create_mupe_system`) loads the libraries from `$MLPOL_PLUGIN_DIR` or, by default,
`plugin/install/lib/plugins` in the repository.

## 6. Check the installation

```bash
export PYTHONPATH=$PWD:$PYTHONPATH
cd examples && python 01_mupe_ff_nma_amoeba2018.py      # ~15 s on a GPU (plus the MACE download the first time)
```
