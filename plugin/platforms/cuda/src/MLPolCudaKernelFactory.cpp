/* CUDA kernel factory of the MLPol plugin (POLEN). */

#include "MLPolCommonKernels.h"
#include "CudaPlatform.h"
#include "CudaContext.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/KernelFactory.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/windowsExport.h"

using namespace MLPol;
using namespace OpenMM;

namespace {

class CudaCalcMLPolMultipoleForceKernel : public CommonCalcMLPolMultipoleForceKernel {
public:
    CudaCalcMLPolMultipoleForceKernel(const std::string& name, const Platform& platform, CudaContext& cu, const System& system) :
            CommonCalcMLPolMultipoleForceKernel(name, platform, cu, system) {
    }
    bool useFixedPointChargeSpreading() const {
        return cc.getUseDoublePrecision();
    }
};

class MLPolCudaKernelFactory : public KernelFactory {
public:
    KernelImpl* createKernelImpl(std::string name, const Platform& platform, ContextImpl& context) const {
        CudaPlatform::PlatformData& data = *static_cast<CudaPlatform::PlatformData*>(context.getPlatformData());
        CudaContext& cu = *data.contexts[0];
        if (name == CalcMLPolMultipoleForceKernel::Name())
            return new CudaCalcMLPolMultipoleForceKernel(name, platform, cu, context.getSystem());
        throw OpenMMException("MLPol: unknown kernel name '" + name + "'");
    }
};

} // namespace

extern "C" OPENMM_EXPORT void registerPlatforms() {
}

extern "C" OPENMM_EXPORT void registerKernelFactories() {
    try {
        Platform& platform = Platform::getPlatformByName("CUDA");
        platform.registerKernelFactory(CalcMLPolMultipoleForceKernel::Name(), new MLPolCudaKernelFactory());
    }
    catch (...) {
        // CUDA platform not available
    }
}

extern "C" OPENMM_EXPORT void registerMLPolCudaKernelFactories() {
    registerKernelFactories();
}
