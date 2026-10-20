#include "MLPolReferenceKernels.h"
#include "ReferencePlatform.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/KernelFactory.h"
#include "openmm/OpenMMException.h"

using namespace MLPol;
using namespace OpenMM;

namespace {
class MLPolReferenceKernelFactory : public KernelFactory {
public:
    KernelImpl* createKernelImpl(std::string name, const Platform& platform, ContextImpl& context) const {
        if (name == CalcMLPolMultipoleForceKernel::Name())
            return new ReferenceCalcMLPolMultipoleForceKernel(name, platform, context.getSystem());
        throw OpenMMException("MLPol: unknown kernel name '" + name + "'");
    }
};
}

extern "C" OPENMM_EXPORT void registerPlatforms() {
}

extern "C" OPENMM_EXPORT void registerKernelFactories() {
    for (int i = 0; i < Platform::getNumPlatforms(); i++) {
        Platform& platform = Platform::getPlatform(i);
        if (dynamic_cast<ReferencePlatform*>(&platform) != NULL)
            platform.registerKernelFactory(CalcMLPolMultipoleForceKernel::Name(), new MLPolReferenceKernelFactory());
    }
}

extern "C" OPENMM_EXPORT void registerMLPolReferenceKernelFactories() {
    registerKernelFactories();
}
