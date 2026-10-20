#ifndef MLPOL_MULTIPOLE_FORCE_IMPL_H_
#define MLPOL_MULTIPOLE_FORCE_IMPL_H_

#include "MLPolMultipoleForce.h"
#include "openmm/internal/ForceImpl.h"
#include "openmm/Kernel.h"
#include <map>
#include <string>
#include <vector>

namespace MLPol {

class MLPolMultipoleForceImpl : public OpenMM::ForceImpl {
public:
    MLPolMultipoleForceImpl(const MLPolMultipoleForce& owner);
    void initialize(OpenMM::ContextImpl& context);
    const MLPolMultipoleForce& getOwner() const {
        return owner;
    }
    void updateContextState(OpenMM::ContextImpl& context, bool& forcesInvalid) {
    }
    double calcForcesAndEnergy(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy, int groups);
    std::map<std::string, double> getDefaultParameters() {
        return {};
    }
    std::vector<std::string> getKernelNames();
    void getLabFramePermanentDipoles(OpenMM::ContextImpl& context, std::vector<OpenMM::Vec3>& dipoles);
    void getInducedDipoles(OpenMM::ContextImpl& context, std::vector<OpenMM::Vec3>& dipoles);
    void getTotalDipoles(OpenMM::ContextImpl& context, std::vector<OpenMM::Vec3>& dipoles);
    void updateParametersInContext(OpenMM::ContextImpl& context);
    void setMLParameters(OpenMM::ContextImpl& context, const std::vector<double>& charges, const std::vector<double>& dipoles,
                         const std::vector<double>& quadrupoles, const std::vector<double>& polarities);
    void setMLPenetration(OpenMM::ContextImpl& context, const std::vector<double>& N, const std::vector<double>& s);
    void getMLDerivatives(OpenMM::ContextImpl& context, std::vector<double>& phi, std::vector<double>& field,
                          std::vector<double>& dTheta, std::vector<double>& dAlpha);
private:
    const MLPolMultipoleForce& owner;
    OpenMM::Kernel kernel;
};

} // namespace MLPol

#endif
