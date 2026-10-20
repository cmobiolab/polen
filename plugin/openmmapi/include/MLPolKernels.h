#ifndef MLPOL_KERNELS_H_
#define MLPOL_KERNELS_H_

#include "openmm/amoebaKernels.h"
#include <string>
#include <vector>

namespace MLPol {

/** Same interface as the AMOEBA kernel, under another name (so that the platform picks ours). */
class CalcMLPolMultipoleForceKernel : public OpenMM::CalcAmoebaMultipoleForceKernel {
public:
    static std::string Name() {
        return "CalcMLPolMultipoleForce";
    }
    CalcMLPolMultipoleForceKernel(std::string name, const OpenMM::Platform& platform) :
        OpenMM::CalcAmoebaMultipoleForceKernel(name, platform) {
    }
    /** Inject q, mu (3M), Theta (9M, lab frame) and alpha of the ML atoms (setMLRegion order). */
    virtual void setMLParameters(OpenMM::ContextImpl& context, const std::vector<double>& charges, const std::vector<double>& dipoles,
                                 const std::vector<double>& quadrupoles, const std::vector<double>& polarities) = 0;
    /** Charge penetration: population N and width s (nm) of the Slater valence of the ML atoms (empty lists disable it). */
    virtual void setMLPenetration(OpenMM::ContextImpl& context, const std::vector<double>& N, const std::vector<double>& s) = 0;
    /** phi = dE/dq (M), field = -dE/dmu (3M), dE/dTheta (9M), dE/dalpha (M), in kJ/mol and OpenMM units. */
    virtual void getMLDerivatives(OpenMM::ContextImpl& context, std::vector<double>& phi, std::vector<double>& field,
                                  std::vector<double>& dTheta, std::vector<double>& dAlpha) = 0;
};

} // namespace MLPol

#endif
