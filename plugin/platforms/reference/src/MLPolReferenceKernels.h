#ifndef MLPOL_REFERENCE_KERNELS_H_
#define MLPOL_REFERENCE_KERNELS_H_

#include "MLPolKernels.h"
#include "MLPolMultipoleForce.h"
#include <torch/script.h>
#include "MLPolRefMultipoleForce.h"
#include "openmm/System.h"
#include "openmm/Platform.h"
#include <vector>

namespace MLPol {

using namespace OpenMM;

class ReferenceCalcMLPolMultipoleForceKernel : public CalcMLPolMultipoleForceKernel {
public:
    ReferenceCalcMLPolMultipoleForceKernel(const std::string& name, const Platform& platform, const System& system);
    ~ReferenceCalcMLPolMultipoleForceKernel();
    /**
     * Initialize the kernel.
     * 
     * @param system     the System this kernel will be applied to
     * @param force      the AmoebaMultipoleForce this kernel will be used for
     */
    void initialize(const System& system, const AmoebaMultipoleForce& force);
    /**
     * Setup for AmoebaReferenceMultipoleForce instance. 
     *
     * @param context        the current context
     *
     * @return pointer to initialized instance of AmoebaReferenceMultipoleForce
     */
    OpenMM::MLPolRef::AmoebaReferenceMultipoleForce* setupAmoebaReferenceMultipoleForce(ContextImpl& context);
    /**
     * Execute the kernel to calculate the forces and/or energy.
     *
     * @param context        the context in which to execute this kernel
     * @param includeForces  true if forces should be calculated
     * @param includeEnergy  true if the energy should be calculated
     * @return the potential energy due to the force
     */
    double execute(ContextImpl& context, bool includeForces, bool includeEnergy);
    /**
     * Get the induced dipole moments of all particles.
     * 
     * @param context    the Context for which to get the induced dipoles
     * @param dipoles    the induced dipole moment of particle i is stored into the i'th element
     */
    void getInducedDipoles(ContextImpl& context, std::vector<Vec3>& dipoles);
    /**
     * Get the fixed dipole moments of all particles in the global reference frame.
     * 
     * @param context    the Context for which to get the fixed dipoles
     * @param dipoles    the fixed dipole moment of particle i is stored into the i'th element
     */
    void getLabFramePermanentDipoles(ContextImpl& context, std::vector<Vec3>& dipoles);
    /**
     * Get the total dipole moments of all particles in the global reference frame.
     * 
     * @param context    the Context for which to get the fixed dipoles
     * @param dipoles    the fixed dipole moment of particle i is stored into the i'th element
     */
    void getTotalDipoles(ContextImpl& context, std::vector<Vec3>& dipoles);
    /** 
     * Calculate the electrostatic potential given vector of grid coordinates.
     *
     * @param context                      context
     * @param inputGrid                    input grid coordinates
     * @param outputElectrostaticPotential output potential 
     */
    void getElectrostaticPotential(ContextImpl& context, const std::vector< Vec3 >& inputGrid,
                                   std::vector< double >& outputElectrostaticPotential);

    /**
     * Get the system multipole moments.
     *
     * @param context                context 
     * @param outputMultipoleMoments vector of multipole moments:
                                     (charge,
                                      dipole_x, dipole_y, dipole_z,
                                      quadrupole_xx, quadrupole_xy, quadrupole_xz,
                                      quadrupole_yx, quadrupole_yy, quadrupole_yz,
                                      quadrupole_zx, quadrupole_zy, quadrupole_zz)
     */
    void getSystemMultipoleMoments(ContextImpl& context, std::vector< double >& outputMultipoleMoments);
    /**
     * Copy changed parameters over to a context.
     *
     * @param context    the context to copy parameters to
     * @param force      the AmoebaMultipoleForce to copy the parameters from
     */
    void copyParametersToContext(ContextImpl& context, const AmoebaMultipoleForce& force);
    /**
     * Get the parameters being used for PME.
     * 
     * @param alpha   the separation parameter
     * @param nx      the number of grid points along the X axis
     * @param ny      the number of grid points along the Y axis
     * @param nz      the number of grid points along the Z axis
     */
    void getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const;

private:

    int numMultipoles;
    AmoebaMultipoleForce::NonbondedMethod nonbondedMethod;
    AmoebaMultipoleForce::PolarizationType polarizationType;
    std::vector<double> charges;
    std::vector<double> dipoles;
    std::vector<double> quadrupoles;
    std::vector<double> tholes;
    std::vector<double> dampingFactors;
    std::vector<double> polarity;
    std::vector<int>   axisTypes;
    std::vector<int>   multipoleAtomZs;
    std::vector<int>   multipoleAtomXs;
    std::vector<int>   multipoleAtomYs;
    std::vector< std::vector< std::vector<int> > > multipoleAtomCovalentInfo;

    int mutualInducedMaxIterations;
    double mutualInducedTargetEpsilon;
    std::vector<double> extrapolationCoefficients;

    bool usePme;
    double alphaEwald;
    double cutoffDistance;
    std::vector<int> pmeGridDimension;

    const System& system;
    OpenMM::MLPolRef::AmoebaReferenceMultipoleForce::LayeredPolarization layered;   // POLEN
    void readLayered(const AmoebaMultipoleForce& force);
    // POLEN: ML region
    std::vector<int> mlAtoms;
    double mlThole = 0.39, crossThole = -1.0;
    std::string mlHeadFile;
    std::vector<double> penN, penS;      // penetration (Slater valence of the ML atoms)
    bool hasHead = false;
    torch::jit::script::Module mlHead;
    void loadHead(const std::string& file);
public:
    void setMLParameters(ContextImpl& context, const std::vector<double>& q, const std::vector<double>& mu,
                         const std::vector<double>& theta, const std::vector<double>& alpha);
    void getMLDerivatives(ContextImpl& context, std::vector<double>& phi, std::vector<double>& field,
                          std::vector<double>& dTheta, std::vector<double>& dAlpha);
    void setMLPenetration(ContextImpl& context, const std::vector<double>& N, const std::vector<double>& s);
};

} // namespace MLPol

#endif
