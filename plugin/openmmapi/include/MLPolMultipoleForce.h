#ifndef MLPOL_MULTIPOLE_FORCE_H_
#define MLPOL_MULTIPOLE_FORCE_H_

/* POLEN: AMOEBA multipole force extended for ML/MM with polarizable embedding.
 *
 * Inherits the whole AmoebaMultipoleForce API (multipoles, covalent maps, methods, polarization); what changes is
 * the kernel it dispatches to (Reference and CUDA platforms). Without extensions it reproduces AmoebaMultipoleForce
 * exactly. Extensions: local-frame forces from the exact chain rule (fixes ThreeFold), layered polarization, injected
 * ML parameters, an ML head evaluated inside the kernel and charge penetration.
 */

#include "openmm/AmoebaMultipoleForce.h"
#include "openmm/Context.h"
#include <string>
#include <vector>

namespace MLPol {

class MLPolMultipoleForce : public OpenMM::AmoebaMultipoleForce {
public:
    MLPolMultipoleForce();
    /** Copy all parameters of an existing AmoebaMultipoleForce. */
    explicit MLPolMultipoleForce(const OpenMM::AmoebaMultipoleForce& source);

    void getInducedDipoles(OpenMM::Context& context, std::vector<OpenMM::Vec3>& dipoles);
    void getLabFramePermanentDipoles(OpenMM::Context& context, std::vector<OpenMM::Vec3>& dipoles);
    void getTotalDipoles(OpenMM::Context& context, std::vector<OpenMM::Vec3>& dipoles);
    void updateParametersInContext(OpenMM::Context& context);

    /**
     * Layered polarization: mutual near the central atoms, direct far away, with a smooth per-group switch.
     *
     * @param centerAtoms   atoms that define the center of the mutual region (e.g. the ML region)
     * @param groupOf       per particle, the switching group (-1 = always mutual)
     * @param groupRefAtoms per group, the reference atom whose distance to the center sets the weight
     * @param rIn, rOut     start and end of the mutual -> direct switch (nm)
     * @param tau           width of the smooth minimum of the distances to the central atoms (nm)
     * Requires mutual polarization. With PME: permanent field with full PME; induced–induced coupling by direct
     * summation (minimum image) among the sites with s > 0, so rOut must be smaller than ~1/4 of the box.
     */
    void setLayeredPolarization(const std::vector<int>& centerAtoms, const std::vector<int>& groupOf,
                                const std::vector<int>& groupRefAtoms, double rIn, double rOut, double tau=0.02);
    void disableLayeredPolarization() {
        layeredEnabled = false;
    }
    bool usesLayeredPolarization() const {
        return layeredEnabled;
    }
    void getLayeredPolarization(std::vector<int>& centerAtoms, std::vector<int>& groupOf,
                                std::vector<int>& groupRefAtoms, double& rIn, double& rOut, double& tau) const;

    /**
     * ML region ('ml' mode): the parameters of these atoms are injected in the lab frame (NoAxisType;
     * damping = alpha^(1/6); thole = tholeML). crossThole: Thole parameter of the ML–MM pairs (< 0: minimum rule).
     * The intra-ML masking (covalent maps) is the responsibility of whoever builds the force.
     */
    void setMLRegion(const std::vector<int>& mlAtoms, double tholeML=0.39, double crossThole=-1.0);
    const std::vector<int>& getMLAtoms() const {
        return mlAtoms;
    }
    double getMLThole() const {
        return mlThole;
    }
    double getCrossThole() const {
        return crossThole;
    }
    /** Inject q (M), mu (3M, e*nm), Theta (9M, e*nm^2; the symmetric traceless part is used) and alpha (M, nm^3) of the ML atoms. */
    void setMLParametersInContext(OpenMM::Context& context, const std::vector<double>& charges, const std::vector<double>& dipoles,
                                  const std::vector<double>& quadrupoles, const std::vector<double>& polarities);
    /** Charge penetration of the ML atoms: population N (M) and width s (M, nm) of the Slater valence (empty lists disable it). */
    void setMLPenetrationInContext(OpenMM::Context& context, const std::vector<double>& N, const std::vector<double>& s);
    /**
     * ML head ('ml' mode): TorchScript file with forward(pos: (M,3) nm, float64) -> (q (M), mu (M,3) e*nm,
     * Theta (M,3,3) e*nm^2 symmetric and traceless, alpha (M) nm^3[, N (M), s (M) nm]), in setMLRegion order;
     * with the last two outputs, charge penetration is enabled. At every evaluation the kernel runs the head,
     * injects the parameters and adds the chain rule (-VJP) through the head to the forces.
     * Empty string: no head (parameters via setMLParametersInContext).
     */
    void setMLHead(const std::string& torchScriptFile) {
        mlHeadFile = torchScriptFile;
    }
    const std::string& getMLHead() const {
        return mlHeadFile;
    }
    /** Derivatives of the energy with respect to the ML parameters (see CalcMLPolMultipoleForceKernel::getMLDerivatives). */
    void getMLDerivatives(OpenMM::Context& context, std::vector<double>& phi, std::vector<double>& field,
                          std::vector<double>& dTheta, std::vector<double>& dAlpha);
protected:
    OpenMM::ForceImpl* createImpl() const;
private:
    bool layeredEnabled = false;
    std::vector<int> layeredCenter, layeredGroupOf, layeredGroupRef;
    double layeredRIn = 0.0, layeredROut = 0.0, layeredTau = 0.02;
    std::vector<int> mlAtoms;
    double mlThole = 0.39, crossThole = -1.0;
    std::string mlHeadFile;
};

} // namespace MLPol

#endif
