/* -------------------------------------------------------------------------- *
 *                               OpenMMAmoeba                                 *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit.                   *
 * See https://openmm.org/development.                                        *
 *                                                                            *
 * Portions copyright (c) 2008-2025 Stanford University and the Authors.      *
 * Authors: Peter Eastman, Mark Friedrichs                                    *
 * Contributors:                                                              *
 *                                                                            *
 * This program is free software: you can redistribute it and/or modify       *
 * it under the terms of the GNU Lesser General Public License as published   *
 * by the Free Software Foundation, either version 3 of the License, or       *
 * (at your option) any later version.                                        *
 *                                                                            *
 * This program is distributed in the hope that it will be useful,            *
 * but WITHOUT ANY WARRANTY; without even the implied warranty of             *
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the              *
 * GNU Lesser General Public License for more details.                        *
 *                                                                            *
 * You should have received a copy of the GNU Lesser General Public License   *
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.      *
 * -------------------------------------------------------------------------- */
/* POLEN: copy of the OpenMM 8.5.2 CommonCalcAmoebaMultipoleForceKernel class
 * (plugins/amoeba/platforms/common/src/AmoebaCommonKernels.{h,cpp}, LGPL), the basis of the MLPolMultipoleForce
 * CUDA kernel. Without GK (AmoebaGeneralizedKirkwoodForce). Modifications are marked with POLEN. */

#ifndef MLPOL_COMMON_KERNELS_H_
#define MLPOL_COMMON_KERNELS_H_

#include "MLPolKernels.h"
#include <torch/script.h>
#include "openmm/AmoebaMultipoleForce.h"
#include "openmm/System.h"
#include "openmm/common/ComputeContext.h"
#include "openmm/common/ComputeArray.h"
#include "openmm/common/FFT3D.h"
#include "openmm/common/NonbondedUtilities.h"

namespace MLPol {

using namespace OpenMM;

class CommonCalcMLPolMultipoleForceKernel : public CalcMLPolMultipoleForceKernel {
public:
    CommonCalcMLPolMultipoleForceKernel(const std::string& name, const Platform& platform, ComputeContext& cc, const System& system);
    /**
     * Initialize the kernel.
     * 
     * @param system     the System this kernel will be applied to
     * @param force      the AmoebaMultipoleForce this kernel will be used for
     */
    void initialize(const System& system, const AmoebaMultipoleForce& force);
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
     * Get the LabFrame dipole moments of all particles.
     * 
     * @param context    the Context for which to get the induced dipoles
     * @param dipoles    the induced dipole moment of particle i is stored into the i'th element
     */
    void getLabFramePermanentDipoles(ContextImpl& context, std::vector<Vec3>& dipoles);
    /**
     * Get the induced dipole moments of all particles.
     * 
     * @param context    the Context for which to get the induced dipoles
     * @param dipoles    the induced dipole moment of particle i is stored into the i'th element
     */
    void getInducedDipoles(ContextImpl& context, std::vector<Vec3>& dipoles);
    /**
     * Get the total dipole moments of all particles.
     * 
     * @param context    the Context for which to get the induced dipoles
     * @param dipoles    the induced dipole moment of particle i is stored into the i'th element
     */
    void getTotalDipoles(ContextImpl& context, std::vector<Vec3>& dipoles);
    /**
     * Execute the kernel to calculate the electrostatic potential
     *
     * @param context        the context in which to execute this kernel
     * @param inputGrid      input grid coordinates
     * @param outputElectrostaticPotential output potential 
     */
    void getElectrostaticPotential(ContextImpl& context, const std::vector< Vec3 >& inputGrid,
                                   std::vector< double >& outputElectrostaticPotential);

   /** 
     * Get the system multipole moments
     *
     * @param context      context
     * @param outputMultipoleMoments (charge,
     *                                dipole_x, dipole_y, dipole_z,
     *                                quadrupole_xx, quadrupole_xy, quadrupole_xz,
     *                                quadrupole_yx, quadrupole_yy, quadrupole_yz,
     *                                quadrupole_zx, quadrupole_zy, quadrupole_zz)
     */
    void getSystemMultipoleMoments(ContextImpl& context, std::vector<double>& outputMultipoleMoments);
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
    /**
     * Get whether charge spreading should be done in fixed point.
     */
    virtual bool useFixedPointChargeSpreading() const = 0;
    // POLEN: ML region
    void setMLParameters(ContextImpl& context, const std::vector<double>& charges, const std::vector<double>& dipoles,
                         const std::vector<double>& quadrupoles, const std::vector<double>& polarities);
    void setMLPenetration(ContextImpl& context, const std::vector<double>& N, const std::vector<double>& s);
    void getMLDerivatives(ContextImpl& context, std::vector<double>& phi, std::vector<double>& field,
                          std::vector<double>& dTheta, std::vector<double>& dAlpha);
protected:
    class ForceInfo;
    void initializeScaleFactors();
    void computeInducedField();
    bool iterateDipolesByDIIS(int iteration);
    void computeExtrapolatedDipoles();
    void ensureMultipolesValid(ContextImpl& context);
    template <class T, class T4, class M4> void computeSystemMultipoleMoments(ContextImpl& context, std::vector<double>& outputMultipoleMoments);
    int numMultipoles, maxInducedIterations, maxExtrapolationOrder;
    int fixedFieldThreads, inducedFieldThreads, electrostaticsThreads;
    int gridSizeX, gridSizeY, gridSizeZ;
    double pmeAlpha, inducedEpsilon, totalCharge;
    bool usePME, hasQuadrupoles, hasInitializedScaleFactors, multipolesAreValid, hasCreatedEvent;
    AmoebaMultipoleForce::PolarizationType polarizationType;
    ComputeContext& cc;
    const System& system;
    std::vector<mm_int4> covalentFlagValues;
    std::vector<mm_int2> polarizationFlagValues;
    ComputeArray multipoleParticles;
    ComputeArray localDipoles;
    ComputeArray localQuadrupoles;
    ComputeArray labDipoles;
    ComputeArray labQuadrupoles;
    ComputeArray sphericalDipoles;
    ComputeArray sphericalQuadrupoles;
    ComputeArray fracDipoles;
    ComputeArray fracQuadrupoles;
    ComputeArray field;
    ComputeArray fieldPolar;
    ComputeArray inducedField;
    ComputeArray inducedFieldPolar;
    ComputeArray torque;
    ComputeArray dampingAndThole;
    ComputeArray inducedDipole;
    ComputeArray inducedDipolePolar;
    ComputeArray inducedDipoleErrors;
    ComputeArray prevDipoles;
    ComputeArray prevDipolesPolar;
    ComputeArray prevDipolesGk;
    ComputeArray prevDipolesGkPolar;
    ComputeArray prevErrors;
    ComputeArray diisMatrix;
    ComputeArray diisCoefficients;
    ComputeArray extrapolatedDipole;
    ComputeArray extrapolatedDipolePolar;
    ComputeArray extrapolatedDipoleGk;
    ComputeArray extrapolatedDipoleGkPolar;
    ComputeArray inducedDipoleFieldGradient;
    ComputeArray inducedDipoleFieldGradientPolar;
    ComputeArray inducedDipoleFieldGradientGk;
    ComputeArray inducedDipoleFieldGradientGkPolar;
    ComputeArray extrapolatedDipoleFieldGradient;
    ComputeArray extrapolatedDipoleFieldGradientPolar;
    ComputeArray extrapolatedDipoleFieldGradientGk;
    ComputeArray extrapolatedDipoleFieldGradientGkPolar;
    ComputeArray polarizability;
    ComputeArray covalentFlags;
    ComputeArray polarizationGroupFlags;
    ComputeArray pmeGrid1;
    ComputeArray pmeGrid2;
    ComputeArray pmeGridLong;
    ComputeArray pmeBsplineModuliX;
    ComputeArray pmeBsplineModuliY;
    ComputeArray pmeBsplineModuliZ;
    ComputeArray pmePhi;
    ComputeArray pmePhid;
    ComputeArray pmePhip;
    ComputeArray pmePhidp;
    ComputeArray pmeCphi;
    ComputeArray lastPositions;
    ComputeKernel computeMomentsKernel, recordInducedDipolesKernel, mapTorqueKernel, computePotentialKernel, electrostaticsKernel;
    ComputeKernel computeFixedFieldKernel, computeInducedFieldKernel, updateInducedFieldKernel;
    ComputeKernel recordDIISDipolesKernel, buildMatrixKernel, solveMatrixKernel;
    ComputeKernel initExtrapolatedKernel, iterateExtrapolatedKernel, computeExtrapolatedKernel, addExtrapolatedGradientKernel;
    ComputeKernel pmeSpreadFixedMultipolesKernel, pmeSpreadInducedDipolesKernel, pmeFinishSpreadChargeKernel, pmeConvolutionKernel;
    ComputeKernel pmeFixedPotentialKernel, pmeInducedPotentialKernel, pmeFixedForceKernel, pmeInducedForceKernel, pmeRecordInducedFieldDipolesKernel;
    ComputeKernel pmeTransformMultipolesKernel, pmeTransformPotentialKernel;
    ComputeEvent syncEvent;
    FFT3D fft;
    // POLEN: ML region
    std::vector<int> mlAtoms;
    std::vector<char> isML;
    double mlThole = 0.39, crossThole = -1.0;
    std::vector<double> mlCharges;         // current ML charges (for the PME total charge)
    ComputeArray mlIndex, mlParams;
    ComputeKernel injectMLKernel, mlDerivativesKernel;
    ComputeArray mlSpecialStart, mlSpecialAtom, mlSpecialScale, mlDerivatives, mlPos, mlForce;
    ComputeKernel gatherMLKernel, addMLForcesKernel, penetrationKernel;
    ComputeArray penParams, penDerivs;
    std::string mlHeadFile;
    bool hasHead = false;
    torch::jit::script::Module mlHead;
    torch::Device headDevice = torch::kCPU;   // POLEN: the context CUDA device (default) or CPU (MLPOL_HEAD_DEVICE=cpu)
    void* torchContext = NULL;                // PyTorch primary CUcontext (OpenMM uses its own context)
    std::vector<double> penN, penS, penDN, penDS;
    double executeAmoeba(ContextImpl& context, bool includeForces, bool includeEnergy);
    // layers
    bool layered = false;
    ComputeArray layerGroupRef, layerCenters, layerGroupOf, layerGroupS, layerGroupDS, layerSoftminW, layerAtomWeight;
    ComputeArray layerActive, layerTrans, layerCounters, layerDEds, layerGroupAtomStart, layerGroupAtoms;
    ComputeKernel groupSwitchKernel, atomWeightsKernel, layeredFieldKernel, layeredCouplingKernel;
    ComputeKernel layeredSwitchDerivKernel, layeredSwitchForceKernel;
    void layeredComputeInducedField();
    static const int MLThreads = 128;
    void computeMLDerivatives(std::vector<double>& out);
    void readMLRegion(const AmoebaMultipoleForce& force);
    static const int PmeOrder = 5;
    static const int MaxPrevDIISDipoles = 20;
};

} // namespace MLPol

#endif
