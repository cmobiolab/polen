Headers copied from the OpenMM 8.5.2 source code (`libraries/`), used by the AMOEBA reference code:
- `jama/`: JAMA/TNT (public domain; NIST).
- `pocketfft/include/pocketfft_hdronly.h`: pocketfft (3-clause BSD license, in the file header).

CUDA platform (`platforms/cuda/src`): `MLPolCommonKernels.{h,cpp}` derive from
`plugins/amoeba/platforms/common/src/AmoebaCommonKernels.{h,cpp}` of OpenMM 8.5.2 (LGPL 3 or later; header kept),
and `kernels/*.cc` are copies of the corresponding kernel sources (same license as the original file).
