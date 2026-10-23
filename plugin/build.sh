#!/bin/bash
# Build and install the plugin (libraries in plugin/install, Python extension in the OPENMM_DIR environment).
set -e
cd "$(dirname "$0")"
export OPENMM_DIR=${OPENMM_DIR:-/home/rdcunha/miniforge3/envs/mlmm-pol}
export MLPOL_INSTALL=$(pwd)/install
# CUDA toolkit (needed by TorchConfig for the CUDA build of PyTorch and by the CUDA platform)
CUDA_ROOT=${CUDA_ROOT:-/usr/local/cuda-12.8}
PY=$OPENMM_DIR/bin/python
cmake -S . -B build -DOPENMM_DIR=$OPENMM_DIR -DCUDAToolkit_ROOT=$CUDA_ROOT -DCMAKE_CUDA_COMPILER=$CUDA_ROOT/bin/nvcc > build/cmake.log 2>&1 || cmake -S . -B build -DOPENMM_DIR=$OPENMM_DIR -DCUDAToolkit_ROOT=$CUDA_ROOT -DCMAKE_CUDA_COMPILER=$CUDA_ROOT/bin/nvcc
cmake --build build -j${JOBS:-16}
cmake --install build > /dev/null
mkdir -p build/python && cd build/python
$OPENMM_DIR/bin/swig -python -c++ -I$OPENMM_DIR/include -o MLPolPluginWrapper.cpp ../../python/mlpolplugin.i
cp ../../python/setup.py .
$PY setup.py build_ext --inplace > setup.log 2>&1
$PY -m pip install --no-deps --no-build-isolation -q . 2>&1 | tail -2
echo "plugin installed: $MLPOL_INSTALL (libraries) + Python module mlpolplugin"
