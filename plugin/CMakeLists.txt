# OpenMM plugin of the POLEN project (ML/MM with polarizable AMOEBA embedding).
# Usage:
#   cmake -S plugin -B plugin/build -DOPENMM_DIR=$CONDA_PREFIX
#   cmake --build plugin/build -j && cmake --install plugin/build
CMAKE_MINIMUM_REQUIRED(VERSION 3.17)
PROJECT(MLPolPlugin CXX)

SET(OPENMM_DIR "" CACHE PATH "Where OpenMM is installed (e.g. the conda environment)")
IF(NOT OPENMM_DIR)
    MESSAGE(FATAL_ERROR "Set OPENMM_DIR")
ENDIF()
IF(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    SET(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/install" CACHE PATH "Local plugin installation" FORCE)
ENDIF()

SET(CMAKE_CXX_STANDARD 17)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)
IF(NOT CMAKE_BUILD_TYPE)
    SET(CMAKE_BUILD_TYPE Release)
ENDIF()
SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib;${OPENMM_DIR}/lib")

INCLUDE_DIRECTORIES(
    "${CMAKE_SOURCE_DIR}/openmmapi/include"
    "${OPENMM_DIR}/include"
    "${OPENMM_DIR}/include/openmm/reference"
    "${CMAKE_SOURCE_DIR}/third_party/jama/include"            # from the OpenMM 8.5.2 source code
    "${CMAKE_SOURCE_DIR}/third_party/pocketfft/include"
    "${CMAKE_SOURCE_DIR}/third_party/openmm_amoeba/include")    # declaration of the (exported) AMOEBA proxy
LINK_DIRECTORIES("${OPENMM_DIR}/lib" "${OPENMM_DIR}/lib/plugins")

# API
FILE(GLOB API_SOURCES openmmapi/src/*.cpp serialization/src/*.cpp)
ADD_LIBRARY(MLPolPlugin SHARED ${API_SOURCES})
TARGET_LINK_LIBRARIES(MLPolPlugin OpenMM OpenMMAmoeba)

# libtorch (TorchScript ML head inside the kernel; the same as in the conda environment / openmm-torch)
LIST(APPEND CMAKE_PREFIX_PATH "${OPENMM_DIR}")
FIND_PACKAGE(Torch REQUIRED)
# The plugin does not use MKL directly. TorchConfig exports caffe2::mkl, and on machines with a system MKLConfig
# (e.g. /usr/local/lib/cmake/mkl-2023, ilp64 interface, .so.2) this linked the plugin against an MKL different from the
# one used by the environment's libtorch_cpu (MKL 3), missing on compute nodes. libtorch_cpu already brings its own MKL.
IF(TARGET caffe2::mkl)
    SET_TARGET_PROPERTIES(caffe2::mkl PROPERTIES INTERFACE_LINK_LIBRARIES "" INTERFACE_LINK_OPTIONS "")
ENDIF()

# Reference platform
FILE(GLOB REF_SOURCES platforms/reference/src/*.cpp)
ADD_LIBRARY(MLPolPluginReference SHARED ${REF_SOURCES})
TARGET_INCLUDE_DIRECTORIES(MLPolPluginReference PRIVATE platforms/reference/src)
TARGET_LINK_LIBRARIES(MLPolPluginReference MLPolPlugin OpenMM OpenMMAmoeba ${TORCH_LIBRARIES})
SET_TARGET_PROPERTIES(MLPolPluginReference PROPERTIES COMPILE_FLAGS "-DOPENMM_BUILDING_SHARED_LIBRARY ${TORCH_CXX_FLAGS}")

# CUDA platform: copy of the OpenMM 8.5.2 "common" AMOEBA kernel (.cc sources embedded as strings)
FIND_PACKAGE(CUDAToolkit REQUIRED)
FILE(GLOB CUDA_KERNEL_FILES platforms/cuda/src/kernels/*.cc)
SET_PROPERTY(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${CUDA_KERNEL_FILES})
SET(KS_DIR "${CMAKE_BINARY_DIR}/kernel_sources")
SET(KS_DECL "")
SET(KS_DEF "")
FOREACH(file ${CUDA_KERNEL_FILES})
    GET_FILENAME_COMPONENT(name ${file} NAME_WE)
    FILE(READ ${file} content)
    STRING(APPEND KS_DECL "    static const std::string ${name};\n")
    STRING(APPEND KS_DEF "const std::string MLPolKernelSources::${name} = R\"MLPOLSRC(${content})MLPOLSRC\";\n")
ENDFOREACH()
FILE(WRITE "${KS_DIR}/MLPolKernelSources.h.tmp"
     "#ifndef MLPOL_KERNEL_SOURCES_H_\n#define MLPOL_KERNEL_SOURCES_H_\n#include <string>\nnamespace MLPol {\nclass MLPolKernelSources {\npublic:\n${KS_DECL}};\n}\n#endif\n")
FILE(WRITE "${KS_DIR}/MLPolKernelSources.cpp.tmp" "#include \"MLPolKernelSources.h\"\nusing namespace MLPol;\n${KS_DEF}")
CONFIGURE_FILE("${KS_DIR}/MLPolKernelSources.h.tmp" "${KS_DIR}/MLPolKernelSources.h" COPYONLY)
CONFIGURE_FILE("${KS_DIR}/MLPolKernelSources.cpp.tmp" "${KS_DIR}/MLPolKernelSources.cpp" COPYONLY)
FILE(GLOB CUDA_SOURCES platforms/cuda/src/*.cpp)
ADD_LIBRARY(MLPolPluginCUDA SHARED ${CUDA_SOURCES} "${KS_DIR}/MLPolKernelSources.cpp")
TARGET_INCLUDE_DIRECTORIES(MLPolPluginCUDA PRIVATE platforms/cuda/src "${KS_DIR}" "${OPENMM_DIR}/include/openmm/cuda"
                           "${OPENMM_DIR}/include/openmm/common" ${CUDAToolkit_INCLUDE_DIRS})
TARGET_LINK_LIBRARIES(MLPolPluginCUDA MLPolPlugin OpenMM OpenMMAmoeba OpenMMCUDA ${TORCH_LIBRARIES})
SET_TARGET_PROPERTIES(MLPolPluginCUDA PROPERTIES COMPILE_FLAGS "-DOPENMM_BUILDING_SHARED_LIBRARY ${TORCH_CXX_FLAGS}")
INSTALL(TARGETS MLPolPluginCUDA LIBRARY DESTINATION lib/plugins)

INSTALL(TARGETS MLPolPlugin LIBRARY DESTINATION lib)
INSTALL(TARGETS MLPolPluginReference LIBRARY DESTINATION lib/plugins)
INSTALL(FILES openmmapi/include/MLPolMultipoleForce.h openmmapi/include/MLPolKernels.h DESTINATION include)
