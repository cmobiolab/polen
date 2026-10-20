#!/bin/bash
# Build the plugin on an HPC cluster (glibc 2.28, CUDA 12.6 driver) with a conda environment `polen` (OpenMM 8.5.2,
# PyTorch 2.12.1 cuda129, conda-forge compilers and CUDA 12.9 toolkit). Usage: bash plugin/build_carbono.sh
# The system MKL (/usr/local/lib/cmake/mkl-2023, .so.2) was dragged in by TorchConfig (caffe2::mkl) and does not exist on
# the compute nodes; CMakeLists empties the caffe2::mkl link interface. --as-needed is kept as an extra safeguard.
set -e
cd "$(dirname "$0")"
export OPENMM_DIR=${OPENMM_DIR:-$HOME/anaconda3/envs/polen}
export MLPOL_INSTALL=$(pwd)/install
export PATH=$OPENMM_DIR/bin:$PATH
export CC=$OPENMM_DIR/bin/x86_64-conda-linux-gnu-gcc CXX=$OPENMM_DIR/bin/x86_64-conda-linux-gnu-g++
export CUDACXX=$OPENMM_DIR/bin/nvcc
PY=$OPENMM_DIR/bin/python
mkdir -p build
cmake -S . -B build -DOPENMM_DIR=$OPENMM_DIR -DCUDAToolkit_ROOT=$OPENMM_DIR -DCMAKE_CUDA_COMPILER=$OPENMM_DIR/bin/nvcc \
    -DCMAKE_PREFIX_PATH=$OPENMM_DIR -DCMAKE_SHARED_LINKER_FLAGS="-Wl,--as-needed" > build/cmake.log 2>&1
cmake --build build -j8 > build/make.log 2>&1
cmake --install build > /dev/null
mkdir -p build/python && cd build/python
$OPENMM_DIR/bin/swig -python -c++ -I$OPENMM_DIR/include -o MLPolPluginWrapper.cpp ../../python/mlpolplugin.i
cp ../../python/setup.py .
$PY setup.py build_ext --inplace > setup.log 2>&1
$PY -m pip install --no-deps --no-build-isolation -q . 2>&1 | tail -2
echo "plugin installed: $MLPOL_INSTALL + Python module mlpolplugin ($OPENMM_DIR)"
