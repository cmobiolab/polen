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

/* POLEN: see MLPolCommonKernels.h. */

#ifdef WIN32
  #define _USE_MATH_DEFINES
#endif
#include "MLPolCommonKernels.h"
#include "MLPolMultipoleForce.h"
#include <torch/csrc/autograd/autograd.h>
#include "MLPolKernelSources.h"
#include "CudaContext.h"
#include <torch/cuda.h>
#include <cstdlib>
#include "openmm/common/ContextSelector.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/internal/AmoebaMultipoleForceImpl.h"
#include "openmm/internal/NonbondedForceImpl.h"
#include "openmm/AmoebaGeneralizedKirkwoodForce.h"
#include "openmm/NonbondedForce.h"
#include "openmm/OpenMMException.h"
#include "SimTKOpenMMRealType.h"
#include "jama_lu.h"
#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <sstream>

using namespace MLPol;
using namespace OpenMM;
using namespace std;

static vector<double> mlChargesOf(const AmoebaMultipoleForce& force, const vector<int>& atoms) {
    vector<double> q;
    for (int a : atoms) {
        double charge, thole, damping, polarity;
        int axisType, atomX, atomY, atomZ;
        vector<double> dipole, quadrupole;
        force.getMultipoleParameters(a, charge, dipole, quadrupole, axisType, atomZ, atomX, atomY, thole, damping, polarity);
        q.push_back(charge);
    }
    return q;
}

// POLEN: scaling factors (m, d, p, u) of a pair, with the AmoebaReferenceMultipoleForce rules (maps of the lower index)
static mm_float4 pairScaleFactors(const AmoebaMultipoleForce& force, int a, int b) {
    const float mScale[] = {0.0f, 0.0f, 0.4f, 0.8f}, pScale[] = {0.0f, 0.0f, 1.0f, 1.0f}, dScale[] = {0.0f, 1.0f, 1.0f, 1.0f};
    int i = min(a, b), j = max(a, b);
    vector<vector<int> > cov;
    force.getCovalentMaps(i, cov);
    float m = 1, d = 1, p = 1, u = 1;
    for (int t = 0; t < 4; t++)
        if (find(cov[t].begin(), cov[t].end(), j) != cov[t].end()) {
            bool hit = (t == AmoebaMultipoleForce::Covalent14 &&
                        find(cov[AmoebaMultipoleForce::PolarizationCovalent11].begin(), cov[AmoebaMultipoleForce::PolarizationCovalent11].end(), j) != cov[AmoebaMultipoleForce::PolarizationCovalent11].end());
            p = hit ? 0.5f*pScale[t] : pScale[t];
            m = mScale[t];
        }
    for (int t = 4; t < 8; t++)
        if (find(cov[t].begin(), cov[t].end(), j) != cov[t].end()) {
            d = dScale[t-4];
            u = 1;
        }
    return mm_float4(m, d, p, u);
}

static void setPeriodicBoxArgs(ComputeContext& cc, ComputeKernel kernel, int index) {
    Vec3 a, b, c;
    cc.getPeriodicBoxVectors(a, b, c);
    if (cc.getUseDoublePrecision()) {
        kernel->setArg(index++, mm_double4(a[0], b[1], c[2], 0.0));
        kernel->setArg(index++, mm_double4(1.0/a[0], 1.0/b[1], 1.0/c[2], 0.0));
        kernel->setArg(index++, mm_double4(a[0], a[1], a[2], 0.0));
        kernel->setArg(index++, mm_double4(b[0], b[1], b[2], 0.0));
        kernel->setArg(index, mm_double4(c[0], c[1], c[2], 0.0));
    }
    else {
        kernel->setArg(index++, mm_float4((float) a[0], (float) b[1], (float) c[2], 0.0f));
        kernel->setArg(index++, mm_float4(1.0f/(float) a[0], 1.0f/(float) b[1], 1.0f/(float) c[2], 0.0f));
        kernel->setArg(index++, mm_float4((float) a[0], (float) a[1], (float) a[2], 0.0f));
        kernel->setArg(index++, mm_float4((float) b[0], (float) b[1], (float) b[2], 0.0f));
        kernel->setArg(index, mm_float4((float) c[0], (float) c[1], (float) c[2], 0.0f));
    }
}

class CommonCalcMLPolMultipoleForceKernel::ForceInfo : public ComputeForceInfo {
public:
    ForceInfo(const AmoebaMultipoleForce& force, const vector<char>* isML) : force(force), isML(isML) {
    }
    bool areParticlesIdentical(int particle1, int particle2) {
        // POLEN: the ML atoms (parameters injected at every step) never get reordered on the GPU
        if (isML != NULL && ((*isML)[particle1] || (*isML)[particle2]))
            return false;
        double charge1, charge2, thole1, thole2, damping1, damping2, polarity1, polarity2;
        int axis1, axis2, multipole11, multipole12, multipole21, multipole22, multipole31, multipole32;
        vector<double> dipole1, dipole2, quadrupole1, quadrupole2;
        force.getMultipoleParameters(particle1, charge1, dipole1, quadrupole1, axis1, multipole11, multipole21, multipole31, thole1, damping1, polarity1);
        force.getMultipoleParameters(particle2, charge2, dipole2, quadrupole2, axis2, multipole12, multipole22, multipole32, thole2, damping2, polarity2);
        if (charge1 != charge2 || thole1 != thole2 || damping1 != damping2 || polarity1 != polarity2 || axis1 != axis2) {
            return false;
        }
        for (int i = 0; i < (int) dipole1.size(); ++i) {
            if (dipole1[i] != dipole2[i]) {
                return false;
            }
        }
        for (int i = 0; i < (int) quadrupole1.size(); ++i) {
            if (quadrupole1[i] != quadrupole2[i]) {
                return false;
            }
        }
        return true;
    }
    int getNumParticleGroups() {
        return 7*force.getNumMultipoles();
    }
    void getParticlesInGroup(int index, vector<int>& particles) {
        int particle = index/7;
        int type = index-7*particle;
        force.getCovalentMap(particle, AmoebaMultipoleForce::CovalentType(type), particles);
    }
    bool areGroupsIdentical(int group1, int group2) {
        return ((group1%7) == (group2%7));
    }
private:
    const AmoebaMultipoleForce& force;
    const vector<char>* isML;
};

CommonCalcMLPolMultipoleForceKernel::CommonCalcMLPolMultipoleForceKernel(const std::string& name, const Platform& platform, ComputeContext& cc, const System& system) :
        CalcMLPolMultipoleForceKernel(name, platform), cc(cc), system(system), hasInitializedScaleFactors(false), multipolesAreValid(false), hasCreatedEvent(false) {
}

void CommonCalcMLPolMultipoleForceKernel::initialize(const System& system, const AmoebaMultipoleForce& force) {
    ContextSelector selector(cc);

    // Initialize multipole parameters.

    numMultipoles = force.getNumMultipoles();
    readMLRegion(force);   // POLEN
    ArrayInterface& posq = cc.getPosq();
    vector<mm_double4> temp(posq.getSize());
    mm_float4* posqf = (mm_float4*) &temp[0];
    mm_double4* posqd = (mm_double4*) &temp[0];
    vector<mm_float2> dampingAndTholeVec;
    vector<float> polarizabilityVec;
    vector<float> localDipolesVec;
    vector<float> localQuadrupolesVec;
    vector<mm_int4> multipoleParticlesVec;
    totalCharge = 0.0;
    for (int i = 0; i < numMultipoles; i++) {
        double charge, thole, damping, polarity;
        int axisType, atomX, atomY, atomZ;
        vector<double> dipole, quadrupole;
        force.getMultipoleParameters(i, charge, dipole, quadrupole, axisType, atomZ, atomX, atomY, thole, damping, polarity);
        totalCharge += charge;
        if (cc.getUseDoublePrecision())
            posqd[i] = mm_double4(0, 0, 0, charge);
        else
            posqf[i] = mm_float4(0, 0, 0, (float) charge);
        dampingAndTholeVec.push_back(mm_float2((float) damping, (float) (isML[i] ? -thole : thole)));   // POLEN: sign = ML
        polarizabilityVec.push_back((float) polarity);
        multipoleParticlesVec.push_back(mm_int4(atomX, atomY, atomZ, axisType));
        for (int j = 0; j < 3; j++)
            localDipolesVec.push_back((float) dipole[j]);
        localQuadrupolesVec.push_back((float) quadrupole[0]);
        localQuadrupolesVec.push_back((float) quadrupole[1]);
        localQuadrupolesVec.push_back((float) quadrupole[2]);
        localQuadrupolesVec.push_back((float) quadrupole[4]);
        localQuadrupolesVec.push_back((float) quadrupole[5]);
    }
    hasQuadrupoles = !mlAtoms.empty();   // POLEN: injected Theta
    for (auto q : localQuadrupolesVec)
        if (q != 0.0)
            hasQuadrupoles = true;
    int paddedNumAtoms = cc.getPaddedNumAtoms();
    for (int i = numMultipoles; i < paddedNumAtoms; i++) {
        dampingAndTholeVec.push_back(mm_float2(0, 0));
        polarizabilityVec.push_back(0);
        multipoleParticlesVec.push_back(mm_int4(0, 0, 0, 0));
        for (int j = 0; j < 3; j++)
            localDipolesVec.push_back(0);
        for (int j = 0; j < 5; j++)
            localQuadrupolesVec.push_back(0);
        if (cc.getUseDoublePrecision())
            posqd[i] = mm_double4(0, 0, 0, 0);
        else
            posqf[i] = mm_float4(0, 0, 0, 0);
    }
    dampingAndThole.initialize<mm_float2>(cc, paddedNumAtoms, "dampingAndThole");
    polarizability.initialize<float>(cc, paddedNumAtoms, "polarizability");
    multipoleParticles.initialize<mm_int4>(cc, paddedNumAtoms, "multipoleParticles");
    localDipoles.initialize<float>(cc, 3*paddedNumAtoms, "localDipoles");
    localQuadrupoles.initialize<float>(cc, 5*paddedNumAtoms, "localQuadrupoles");
    lastPositions.initialize(cc, cc.getPosq().getSize(), cc.getPosq().getElementSize(), "lastPositions");
    dampingAndThole.upload(dampingAndTholeVec);
    polarizability.upload(polarizabilityVec);
    multipoleParticles.upload(multipoleParticlesVec);
    localDipoles.upload(localDipolesVec);
    localQuadrupoles.upload(localQuadrupolesVec);
    posq.upload(&temp[0]);
    
    // Create workspace arrays.
    
    polarizationType = force.getPolarizationType();
    int elementSize = (cc.getUseDoublePrecision() ? sizeof(double) : sizeof(float));
    labDipoles.initialize(cc, 3*paddedNumAtoms, elementSize, "labDipoles");
    labQuadrupoles.initialize(cc, 5*paddedNumAtoms, elementSize, "labQuadrupoles");
    sphericalDipoles.initialize(cc, 3*paddedNumAtoms, elementSize, "sphericalDipoles");
    sphericalQuadrupoles.initialize(cc, 5*paddedNumAtoms, elementSize, "sphericalQuadrupoles");
    fracDipoles.initialize(cc, 3*paddedNumAtoms, elementSize, "fracDipoles");
    fracQuadrupoles.initialize(cc, 6*paddedNumAtoms, elementSize, "fracQuadrupoles");
    field.initialize(cc, 3*paddedNumAtoms, sizeof(long long), "field");
    fieldPolar.initialize(cc, 3*paddedNumAtoms, sizeof(long long), "fieldPolar");
    torque.initialize(cc, 3*paddedNumAtoms, sizeof(long long), "torque");
    inducedDipole.initialize(cc, 3*paddedNumAtoms, elementSize, "inducedDipole");
    inducedDipolePolar.initialize(cc, 3*paddedNumAtoms, elementSize, "inducedDipolePolar");
    if (polarizationType == AmoebaMultipoleForce::Mutual) {
        inducedDipoleErrors.initialize(cc, cc.getNumThreadBlocks(), sizeof(mm_float2), "inducedDipoleErrors");
        prevDipoles.initialize(cc, 3*numMultipoles*MaxPrevDIISDipoles, elementSize, "prevDipoles");
        prevDipolesPolar.initialize(cc, 3*numMultipoles*MaxPrevDIISDipoles, elementSize, "prevDipolesPolar");
        prevErrors.initialize(cc, 3*numMultipoles*MaxPrevDIISDipoles, elementSize, "prevErrors");
        diisMatrix.initialize(cc, MaxPrevDIISDipoles*MaxPrevDIISDipoles, elementSize, "diisMatrix");
        diisCoefficients.initialize(cc, MaxPrevDIISDipoles+1, sizeof(float), "diisCoefficients");
        syncEvent = cc.createEvent();
        hasCreatedEvent = true;
    }
    else if (polarizationType == AmoebaMultipoleForce::Extrapolated) {
        int numOrders = force.getExtrapolationCoefficients().size();
        extrapolatedDipole.initialize(cc, 3*numMultipoles*numOrders, elementSize, "extrapolatedDipole");
        extrapolatedDipolePolar.initialize(cc, 3*numMultipoles*numOrders, elementSize, "extrapolatedDipolePolar");
        inducedDipoleFieldGradient.initialize(cc, 6*paddedNumAtoms, sizeof(long long), "inducedDipoleFieldGradient");
        inducedDipoleFieldGradientPolar.initialize(cc, 6*paddedNumAtoms, sizeof(long long), "inducedDipoleFieldGradientPolar");
        extrapolatedDipoleFieldGradient.initialize(cc, 6*paddedNumAtoms*(numOrders-1), elementSize, "extrapolatedDipoleFieldGradient");
        extrapolatedDipoleFieldGradientPolar.initialize(cc, 6*paddedNumAtoms*(numOrders-1), elementSize, "extrapolatedDipoleFieldGradientPolar");
    }
    // The next two arrays will get resized once we know the number of exclusions.
    covalentFlags.initialize<mm_int2>(cc, 1, "covalentFlags");
    polarizationGroupFlags.initialize<unsigned int>(cc, 1, "polarizationGroupFlags");
    cc.addAutoclearBuffer(field);
    cc.addAutoclearBuffer(fieldPolar);
    cc.addAutoclearBuffer(torque);
    
    // Record which atoms should be flagged as exclusions based on covalent groups, and determine
    // the values for the covalent group flags.
    
    vector<vector<int> > exclusions(numMultipoles);
    for (int i = 0; i < numMultipoles; i++) {
        vector<int> atoms;
        set<int> allAtoms;
        allAtoms.insert(i);
        force.getCovalentMap(i, AmoebaMultipoleForce::Covalent12, atoms);
        allAtoms.insert(atoms.begin(), atoms.end());
        force.getCovalentMap(i, AmoebaMultipoleForce::Covalent13, atoms);
        allAtoms.insert(atoms.begin(), atoms.end());
        for (int atom : allAtoms)
            covalentFlagValues.push_back(mm_int4(i, atom, 0, 0));
        force.getCovalentMap(i, AmoebaMultipoleForce::Covalent14, atoms);
        allAtoms.insert(atoms.begin(), atoms.end());
        for (int atom : atoms)
            covalentFlagValues.push_back(mm_int4(i, atom, 1, 0));
        force.getCovalentMap(i, AmoebaMultipoleForce::Covalent15, atoms);
        for (int atom : atoms)
            covalentFlagValues.push_back(mm_int4(i, atom, 2, 0));
        allAtoms.insert(atoms.begin(), atoms.end());
        force.getCovalentMap(i, AmoebaMultipoleForce::PolarizationCovalent11, atoms);
        allAtoms.insert(atoms.begin(), atoms.end());
        exclusions[i].insert(exclusions[i].end(), allAtoms.begin(), allAtoms.end());

        // Workaround for bug in TINKER: if an atom is listed in both the PolarizationCovalent11
        // and PolarizationCovalent12 maps, the latter takes precedence.

        vector<int> atoms12;
        force.getCovalentMap(i, AmoebaMultipoleForce::PolarizationCovalent12, atoms12);
        for (int atom : atoms)
            if (find(atoms12.begin(), atoms12.end(), atom) == atoms12.end())
                polarizationFlagValues.push_back(mm_int2(i, atom));
    }
    set<pair<int, int> > tilesWithExclusions;
    for (int atom1 = 0; atom1 < (int) exclusions.size(); ++atom1) {
        int x = atom1/ComputeContext::TileSize;
        for (int atom2 : exclusions[atom1]) {
            int y = atom2/ComputeContext::TileSize;
            tilesWithExclusions.insert(make_pair(max(x, y), min(x, y)));
        }
    }
    
    // Record other options.
    
    if (polarizationType == AmoebaMultipoleForce::Mutual) {
        maxInducedIterations = force.getMutualInducedMaxIterations();
        inducedEpsilon = force.getMutualInducedTargetEpsilon();
    }
    else
        maxInducedIterations = 0;
    if (polarizationType != AmoebaMultipoleForce::Direct) {
        inducedField.initialize(cc, 3*paddedNumAtoms, sizeof(long long), "inducedField");
        inducedFieldPolar.initialize(cc, 3*paddedNumAtoms, sizeof(long long), "inducedFieldPolar");
    }
    usePME = (force.getNonbondedMethod() == AmoebaMultipoleForce::PME);
    
    // See whether there's an AmoebaGeneralizedKirkwoodForce in the System.

    const AmoebaGeneralizedKirkwoodForce* gk = NULL;
    for (int i = 0; i < system.getNumForces() && gk == NULL; i++)
        gk = dynamic_cast<const AmoebaGeneralizedKirkwoodForce*>(&system.getForce(i));
    if (gk != NULL)
        throw OpenMMException("MLPolMultipoleForce: AmoebaGeneralizedKirkwoodForce is not supported.");   // POLEN
    double innerDielectric = 1.0;
    
    // Create the kernels.

    bool useShuffle = false;//(cc.getComputeCapability() >= 3.0 && !cc.getUseDoublePrecision());
    double fixedThreadMemory = 19*elementSize+2*sizeof(float)+3*sizeof(int)/(double) cc.TileSize;
    double inducedThreadMemory = 15*elementSize+2*sizeof(float);
    if (polarizationType == AmoebaMultipoleForce::Extrapolated)
        inducedThreadMemory += 12*elementSize;
    double electrostaticsThreadMemory = 0;
    if (!useShuffle)
        fixedThreadMemory += 3*elementSize;
    map<string, string> defines;
    defines["NUM_ATOMS"] = cc.intToString(numMultipoles);
    defines["PADDED_NUM_ATOMS"] = cc.intToString(cc.getPaddedNumAtoms());
    defines["NUM_BLOCKS"] = cc.intToString(cc.getNumAtomBlocks());
    defines["ENERGY_SCALE_FACTOR"] = cc.doubleToString(ONE_4PI_EPS0/innerDielectric);
    if (polarizationType == AmoebaMultipoleForce::Direct)
        defines["DIRECT_POLARIZATION"] = "";
    else if (polarizationType == AmoebaMultipoleForce::Mutual)
        defines["MUTUAL_POLARIZATION"] = "";
    else if (polarizationType == AmoebaMultipoleForce::Extrapolated)
        defines["EXTRAPOLATED_POLARIZATION"] = "";
    if (useShuffle)
        defines["USE_SHUFFLE"] = "";
    if (hasQuadrupoles)
        defines["INCLUDE_QUADRUPOLES"] = "";
    defines["TILE_SIZE"] = cc.intToString(ComputeContext::TileSize);
    if (!mlAtoms.empty() && crossThole >= 0.0)
        defines["MLPOL_CROSS_THOLE"] = cc.doubleToString(crossThole);   // POLEN
    int numExclusionTiles = tilesWithExclusions.size();
    defines["NUM_TILES_WITH_EXCLUSIONS"] = cc.intToString(numExclusionTiles);
    int numContexts = cc.getNumContexts();
    int startExclusionIndex = cc.getContextIndex()*numExclusionTiles/numContexts;
    int endExclusionIndex = (cc.getContextIndex()+1)*numExclusionTiles/numContexts;
    defines["FIRST_EXCLUSION_TILE"] = cc.intToString(startExclusionIndex);
    defines["LAST_EXCLUSION_TILE"] = cc.intToString(endExclusionIndex);
    maxExtrapolationOrder = force.getExtrapolationCoefficients().size();
    defines["MAX_EXTRAPOLATION_ORDER"] = cc.intToString(maxExtrapolationOrder);
    stringstream coefficients;
    for (int i = 0; i < maxExtrapolationOrder; i++) {
        if (i > 0)
            coefficients << ",";
        double sum = 0;
        for (int j = i; j < maxExtrapolationOrder; j++)
            sum += force.getExtrapolationCoefficients()[j];
        coefficients << cc.doubleToString(sum);
    }
    defines["EXTRAPOLATION_COEFFICIENTS_SUM"] = coefficients.str();
    if (usePME) {
        int nx, ny, nz;
        force.getPMEParameters(pmeAlpha, nx, ny, nz);
        if (nx == 0 || pmeAlpha == 0) {
            NonbondedForce nb;
            nb.setEwaldErrorTolerance(force.getEwaldErrorTolerance());
            nb.setCutoffDistance(force.getCutoffDistance());
            NonbondedForceImpl::calcPMEParameters(system, nb, pmeAlpha, gridSizeX, gridSizeY, gridSizeZ, false);
            gridSizeX = cc.findLegalFFTDimension(gridSizeX);
            gridSizeY = cc.findLegalFFTDimension(gridSizeY);
            gridSizeZ = cc.findLegalFFTDimension(gridSizeZ);
        }
        else {
            gridSizeX = cc.findLegalFFTDimension(nx);
            gridSizeY = cc.findLegalFFTDimension(ny);
            gridSizeZ = cc.findLegalFFTDimension(nz);
        }
        defines["EWALD_ALPHA"] = cc.doubleToString(pmeAlpha);
        defines["SQRT_PI"] = cc.doubleToString(sqrt(M_PI));
        defines["USE_EWALD"] = "";
        defines["USE_CUTOFF"] = "";
        defines["USE_PERIODIC"] = "";
        defines["CUTOFF_SQUARED"] = cc.doubleToString(force.getCutoffDistance()*force.getCutoffDistance());
    }
    if (gk != NULL) {
        defines["USE_GK"] = "";
        defines["GK_C"] = cc.doubleToString(2.455);
        double solventDielectric = gk->getSolventDielectric();
        defines["GK_FC"] = cc.doubleToString(1*(1-solventDielectric)/(0+1*solventDielectric));
        defines["GK_FD"] = cc.doubleToString(2*(1-solventDielectric)/(1+2*solventDielectric));
        defines["GK_FQ"] = cc.doubleToString(3*(1-solventDielectric)/(2+3*solventDielectric));
        fixedThreadMemory += 4*elementSize;
        inducedThreadMemory += 13*elementSize;
        if (polarizationType == AmoebaMultipoleForce::Mutual) {
            prevDipolesGk.initialize(cc, 3*numMultipoles*MaxPrevDIISDipoles, elementSize, "prevDipolesGk");
            prevDipolesGkPolar.initialize(cc, 3*numMultipoles*MaxPrevDIISDipoles, elementSize, "prevDipolesGkPolar");
        }
        else if (polarizationType == AmoebaMultipoleForce::Extrapolated) {
            inducedThreadMemory += 12*elementSize;
            int numOrders = force.getExtrapolationCoefficients().size();
            extrapolatedDipoleGk.initialize(cc, 3*numMultipoles*numOrders, elementSize, "extrapolatedDipoleGk");
            extrapolatedDipoleGkPolar.initialize(cc, 3*numMultipoles*numOrders, elementSize, "extrapolatedDipoleGkPolar");
            inducedDipoleFieldGradientGk.initialize(cc, 6*paddedNumAtoms, sizeof(long long), "inducedDipoleFieldGradientGk");
            inducedDipoleFieldGradientGkPolar.initialize(cc, 6*paddedNumAtoms, sizeof(long long), "inducedDipoleFieldGradientGkPolar");
            extrapolatedDipoleFieldGradientGk.initialize(cc, 6*paddedNumAtoms*(numOrders-1), elementSize, "extrapolatedDipoleFieldGradientGk");
            extrapolatedDipoleFieldGradientGkPolar.initialize(cc, 6*paddedNumAtoms*(numOrders-1), elementSize, "extrapolatedDipoleFieldGradientGkPolar");
        }
    }
    NonbondedUtilities& nb = cc.getNonbondedUtilities();
    int maxThreads = max(32, nb.getForceThreadBlockSize());
    fixedFieldThreads = min(maxThreads, cc.computeThreadBlockSize(fixedThreadMemory));
    inducedFieldThreads = min(maxThreads, cc.computeThreadBlockSize(inducedThreadMemory));
    ComputeProgram program = cc.compileProgram(MLPolKernelSources::multipoles, defines);
    computeMomentsKernel = program->createKernel("computeLabFrameMoments");
    computeMomentsKernel->addArg(cc.getPosq());
    computeMomentsKernel->addArg(multipoleParticles);
    computeMomentsKernel->addArg(localDipoles);
    computeMomentsKernel->addArg(localQuadrupoles);
    computeMomentsKernel->addArg(labDipoles);
    computeMomentsKernel->addArg(labQuadrupoles);
    computeMomentsKernel->addArg(sphericalDipoles);
    computeMomentsKernel->addArg(sphericalQuadrupoles);
    recordInducedDipolesKernel = program->createKernel("recordInducedDipoles");
    recordInducedDipolesKernel->addArg(field);
    recordInducedDipolesKernel->addArg(fieldPolar);
    if (gk != NULL)
        for (int i = 0; i < 3; i++)
            recordInducedDipolesKernel->addArg();
    recordInducedDipolesKernel->addArg(inducedDipole);
    recordInducedDipolesKernel->addArg(inducedDipolePolar);
    recordInducedDipolesKernel->addArg(polarizability);
    mapTorqueKernel = program->createKernel("mapTorqueToForce");
    mapTorqueKernel->addArg(cc.getLongForceBuffer());
    mapTorqueKernel->addArg(torque);
    mapTorqueKernel->addArg(cc.getPosq());
    mapTorqueKernel->addArg(multipoleParticles);
    computePotentialKernel = program->createKernel("computePotentialAtPoints");
    computePotentialKernel->addArg(cc.getPosq());
    computePotentialKernel->addArg(labDipoles);
    computePotentialKernel->addArg(labQuadrupoles);
    computePotentialKernel->addArg(inducedDipole);
    for (int i = 0; i < 8; i++)
        computePotentialKernel->addArg();
    defines["THREAD_BLOCK_SIZE"] = cc.intToString(fixedFieldThreads);
    program = cc.compileProgram(MLPolKernelSources::multipoleFixedField, defines);
    computeFixedFieldKernel = program->createKernel("computeFixedField");
    computeFixedFieldKernel->addArg(field);
    computeFixedFieldKernel->addArg(fieldPolar);
    computeFixedFieldKernel->addArg(cc.getPosq());
    computeFixedFieldKernel->addArg(covalentFlags);
    computeFixedFieldKernel->addArg(polarizationGroupFlags);
    computeFixedFieldKernel->addArg(nb.getExclusionTiles());
    computeFixedFieldKernel->addArg();
    computeFixedFieldKernel->addArg();
    if (gk != NULL) {
        computeFixedFieldKernel->addArg();
        computeFixedFieldKernel->addArg();
    }
    else if (usePME) {
        computeFixedFieldKernel->addArg(nb.getInteractingTiles());
        computeFixedFieldKernel->addArg(nb.getInteractionCount());
        for (int i = 0; i < 6; i++)
            computeFixedFieldKernel->addArg();
        computeFixedFieldKernel->addArg(nb.getBlockCenters());
        computeFixedFieldKernel->addArg(nb.getInteractingAtoms());
    }
    computeFixedFieldKernel->addArg(labDipoles);
    computeFixedFieldKernel->addArg(labQuadrupoles);
    computeFixedFieldKernel->addArg(dampingAndThole);
    if (polarizationType != AmoebaMultipoleForce::Direct) {
        defines["THREAD_BLOCK_SIZE"] = cc.intToString(inducedFieldThreads);
        defines["MAX_PREV_DIIS_DIPOLES"] = cc.intToString(MaxPrevDIISDipoles);
        program = cc.compileProgram(MLPolKernelSources::multipoleInducedField, defines);
        computeInducedFieldKernel = program->createKernel("computeInducedField");
        computeInducedFieldKernel->addArg(inducedField);
        computeInducedFieldKernel->addArg(inducedFieldPolar);
        computeInducedFieldKernel->addArg(cc.getPosq());
        computeInducedFieldKernel->addArg(nb.getExclusionTiles());
        computeInducedFieldKernel->addArg(inducedDipole);
        computeInducedFieldKernel->addArg(inducedDipolePolar);
        computeInducedFieldKernel->addArg();
        computeInducedFieldKernel->addArg();
        if (usePME) {
            computeInducedFieldKernel->addArg(nb.getInteractingTiles());
            computeInducedFieldKernel->addArg(nb.getInteractionCount());
            for (int i = 0; i < 6; i++)
                computeInducedFieldKernel->addArg();
            computeInducedFieldKernel->addArg(nb.getBlockCenters());
            computeInducedFieldKernel->addArg(nb.getInteractingAtoms());
        }
        if (gk != NULL) {
            for (int i = 0; i < 5; i++)
                computeInducedFieldKernel->addArg();
        }
        if (polarizationType == AmoebaMultipoleForce::Extrapolated) {
            computeInducedFieldKernel->addArg(inducedDipoleFieldGradient);
            computeInducedFieldKernel->addArg(inducedDipoleFieldGradientPolar);
            if (gk != NULL) {
                computeInducedFieldKernel->addArg(inducedDipoleFieldGradientGk);
                computeInducedFieldKernel->addArg(inducedDipoleFieldGradientGkPolar);
            }
        }
        computeInducedFieldKernel->addArg(dampingAndThole);
        if (polarizationType == AmoebaMultipoleForce::Mutual) {
            updateInducedFieldKernel = program->createKernel("updateInducedFieldByDIIS");
            for (int i = 0; i < 4; i++)
                updateInducedFieldKernel->addArg();
            updateInducedFieldKernel->addArg(diisCoefficients);
            updateInducedFieldKernel->addArg();
            recordDIISDipolesKernel = program->createKernel("recordInducedDipolesForDIIS");
            recordDIISDipolesKernel->addArg(field);
            recordDIISDipolesKernel->addArg(fieldPolar);
            recordDIISDipolesKernel->addArg(polarizability);
            recordDIISDipolesKernel->addArg(inducedDipoleErrors);
            recordDIISDipolesKernel->addArg(prevErrors);
            recordDIISDipolesKernel->addArg(diisMatrix);
            for (int i = 0; i < 9; i++)
                recordDIISDipolesKernel->addArg();
            buildMatrixKernel = program->createKernel("computeDIISMatrix");
            buildMatrixKernel->addArg(prevErrors);
            buildMatrixKernel->addArg();
            buildMatrixKernel->addArg(diisMatrix);
            solveMatrixKernel = program->createKernel("solveDIISMatrix");
            solveMatrixKernel->addArg();
            solveMatrixKernel->addArg(diisMatrix);
            solveMatrixKernel->addArg(diisCoefficients);
        }
        if (polarizationType == AmoebaMultipoleForce::Extrapolated) {
            initExtrapolatedKernel = program->createKernel("initExtrapolatedDipoles");
            initExtrapolatedKernel->addArg(inducedDipole);
            initExtrapolatedKernel->addArg(extrapolatedDipole);
            initExtrapolatedKernel->addArg(inducedDipolePolar);
            initExtrapolatedKernel->addArg(extrapolatedDipolePolar);
            initExtrapolatedKernel->addArg(inducedDipoleFieldGradient);
            initExtrapolatedKernel->addArg(inducedDipoleFieldGradientPolar);
            if (gk != NULL) {
                initExtrapolatedKernel->addArg();
                initExtrapolatedKernel->addArg();
                initExtrapolatedKernel->addArg(extrapolatedDipoleGk);
                initExtrapolatedKernel->addArg(extrapolatedDipoleGkPolar);
                initExtrapolatedKernel->addArg(inducedDipoleFieldGradientGk);
                initExtrapolatedKernel->addArg(inducedDipoleFieldGradientGkPolar);
            }
            iterateExtrapolatedKernel = program->createKernel("iterateExtrapolatedDipoles");
            iterateExtrapolatedKernel->addArg();
            iterateExtrapolatedKernel->addArg(inducedDipole);
            iterateExtrapolatedKernel->addArg(extrapolatedDipole);
            iterateExtrapolatedKernel->addArg(inducedField);
            iterateExtrapolatedKernel->addArg(inducedDipolePolar);
            iterateExtrapolatedKernel->addArg(extrapolatedDipolePolar);
            iterateExtrapolatedKernel->addArg(inducedFieldPolar);
            iterateExtrapolatedKernel->addArg(inducedDipoleFieldGradient);
            iterateExtrapolatedKernel->addArg(inducedDipoleFieldGradientPolar);
            iterateExtrapolatedKernel->addArg(extrapolatedDipoleFieldGradient);
            iterateExtrapolatedKernel->addArg(extrapolatedDipoleFieldGradientPolar);
            if (gk != NULL) {
                iterateExtrapolatedKernel->addArg();
                iterateExtrapolatedKernel->addArg();
                iterateExtrapolatedKernel->addArg(extrapolatedDipoleGk);
                iterateExtrapolatedKernel->addArg(extrapolatedDipoleGkPolar);
                iterateExtrapolatedKernel->addArg(inducedDipoleFieldGradientGk);
                iterateExtrapolatedKernel->addArg(inducedDipoleFieldGradientGkPolar);
                iterateExtrapolatedKernel->addArg();
                iterateExtrapolatedKernel->addArg();
                iterateExtrapolatedKernel->addArg(extrapolatedDipoleFieldGradientGk);
                iterateExtrapolatedKernel->addArg(extrapolatedDipoleFieldGradientGkPolar);
            }
            iterateExtrapolatedKernel->addArg(polarizability);
            computeExtrapolatedKernel = program->createKernel("computeExtrapolatedDipoles");
            computeExtrapolatedKernel->addArg(inducedDipole);
            computeExtrapolatedKernel->addArg(extrapolatedDipole);
            computeExtrapolatedKernel->addArg(inducedDipolePolar);
            computeExtrapolatedKernel->addArg(extrapolatedDipolePolar);
            if (gk != NULL) {
                computeExtrapolatedKernel->addArg();
                computeExtrapolatedKernel->addArg();
                computeExtrapolatedKernel->addArg(extrapolatedDipoleGk);
                computeExtrapolatedKernel->addArg(extrapolatedDipoleGkPolar);
            }
            addExtrapolatedGradientKernel = program->createKernel("addExtrapolatedFieldGradientToForce");
            addExtrapolatedGradientKernel->addArg(cc.getLongForceBuffer());
            addExtrapolatedGradientKernel->addArg(extrapolatedDipole);
            addExtrapolatedGradientKernel->addArg(extrapolatedDipolePolar);
            addExtrapolatedGradientKernel->addArg(extrapolatedDipoleFieldGradient);
            addExtrapolatedGradientKernel->addArg(extrapolatedDipoleFieldGradientPolar);
            if (gk != NULL) {
                addExtrapolatedGradientKernel->addArg(extrapolatedDipoleGk);
                addExtrapolatedGradientKernel->addArg(extrapolatedDipoleGkPolar);
                addExtrapolatedGradientKernel->addArg(extrapolatedDipoleFieldGradientGk);
                addExtrapolatedGradientKernel->addArg(extrapolatedDipoleFieldGradientGkPolar);
            }
        }
    }
    stringstream electrostaticsSource;
    electrostaticsSource << MLPolKernelSources::sphericalMultipoles;
    if (usePME)
        electrostaticsSource << MLPolKernelSources::pmeMultipoleElectrostatics;
    else
        electrostaticsSource << MLPolKernelSources::multipoleElectrostatics;
    electrostaticsThreadMemory = 24*elementSize+3*sizeof(float)+3*sizeof(int)/(double) cc.TileSize;
    electrostaticsThreads = min(maxThreads, cc.computeThreadBlockSize(electrostaticsThreadMemory));
    defines["THREAD_BLOCK_SIZE"] = cc.intToString(electrostaticsThreads);
    program = cc.compileProgram(electrostaticsSource.str(), defines);
    electrostaticsKernel = program->createKernel("computeElectrostatics");
    electrostaticsKernel->addArg(cc.getLongForceBuffer());
    electrostaticsKernel->addArg(torque);
    electrostaticsKernel->addArg(cc.getEnergyBuffer());
    electrostaticsKernel->addArg(cc.getPosq());
    electrostaticsKernel->addArg(covalentFlags);
    electrostaticsKernel->addArg(polarizationGroupFlags);
    electrostaticsKernel->addArg(nb.getExclusionTiles());
    electrostaticsKernel->addArg();
    electrostaticsKernel->addArg();
    if (usePME) {
        electrostaticsKernel->addArg(nb.getInteractingTiles());
        electrostaticsKernel->addArg(nb.getInteractionCount());
        for (int i = 0; i < 6; i++)
            electrostaticsKernel->addArg();
        electrostaticsKernel->addArg(nb.getBlockCenters());
        electrostaticsKernel->addArg(nb.getInteractingAtoms());
    }
    electrostaticsKernel->addArg(sphericalDipoles);
    electrostaticsKernel->addArg(sphericalQuadrupoles);
    electrostaticsKernel->addArg(inducedDipole);
    electrostaticsKernel->addArg(inducedDipolePolar);
    electrostaticsKernel->addArg(dampingAndThole);
    if (layered) {
        // POLEN: with layers, energy and forces come from the direct-polarization kernels with the converged dipoles;
        // the induced–induced coupling is added separately (layeredCouplingForces)
        map<string, string> directDefines = defines;
        directDefines.erase("MUTUAL_POLARIZATION");
        directDefines["DIRECT_POLARIZATION"] = "";
        ComputeProgram directProgram = cc.compileProgram(electrostaticsSource.str(), directDefines);
        ComputeKernel k = directProgram->createKernel("computeElectrostatics");
        k->addArg(cc.getLongForceBuffer());
        k->addArg(torque);
        k->addArg(cc.getEnergyBuffer());
        k->addArg(cc.getPosq());
        k->addArg(covalentFlags);
        k->addArg(polarizationGroupFlags);
        k->addArg(nb.getExclusionTiles());
        k->addArg();
        k->addArg();
        if (usePME) {
            k->addArg(nb.getInteractingTiles());
            k->addArg(nb.getInteractionCount());
            for (int i = 0; i < 6; i++)
                k->addArg();
            k->addArg(nb.getBlockCenters());
            k->addArg(nb.getInteractingAtoms());
        }
        k->addArg(sphericalDipoles);
        k->addArg(sphericalQuadrupoles);
        k->addArg(inducedDipole);
        k->addArg(inducedDipolePolar);
        k->addArg(dampingAndThole);
        electrostaticsKernel = k;
    }

    // POLEN: layered polarization
    if (layered) {
        const MLPolMultipoleForce& f = dynamic_cast<const MLPolMultipoleForce&>(force);
        vector<int> center, groupOf, groupRef;
        double rIn, rOut, tau;
        f.getLayeredPolarization(center, groupOf, groupRef, rIn, rOut, tau);
        int numGroups = groupRef.size(), numCenters = center.size();
        vector<int> start(numGroups+1, 0), atoms;
        vector<vector<int> > members(numGroups);
        for (int i = 0; i < (int) groupOf.size(); i++)
            if (groupOf[i] >= 0)
                members[groupOf[i]].push_back(i);
        for (int m = 0; m < numGroups; m++) {
            atoms.insert(atoms.end(), members[m].begin(), members[m].end());
            start[m+1] = atoms.size();
        }
        if (atoms.empty())
            atoms.push_back(0);
        layerGroupRef.initialize<int>(cc, max(numGroups, 1), "layerGroupRef");
        layerCenters.initialize<int>(cc, numCenters, "layerCenters");
        layerGroupOf.initialize<int>(cc, groupOf.size(), "layerGroupOf");
        layerGroupAtomStart.initialize<int>(cc, start.size(), "layerGroupAtomStart");
        layerGroupAtoms.initialize<int>(cc, atoms.size(), "layerGroupAtoms");
        if (numGroups > 0)
            layerGroupRef.upload(groupRef);
        layerCenters.upload(center);
        layerGroupOf.upload(groupOf);
        layerGroupAtomStart.upload(start);
        layerGroupAtoms.upload(atoms);
        layerGroupS.initialize(cc, max(numGroups, 1), elementSize, "layerGroupS");
        layerGroupDS.initialize(cc, max(numGroups, 1), elementSize, "layerGroupDS");
        layerSoftminW.initialize(cc, max(numGroups*numCenters, 1), elementSize, "layerSoftminW");
        layerAtomWeight.initialize(cc, paddedNumAtoms, elementSize, "layerAtomWeight");
        layerDEds.initialize(cc, paddedNumAtoms, elementSize, "layerDEds");
        layerActive.initialize<int>(cc, paddedNumAtoms, "layerActive");
        layerTrans.initialize<int>(cc, paddedNumAtoms, "layerTrans");
        layerCounters.initialize<int>(cc, 2, "layerCounters");
        map<string, string> layerDefines = defines;
        layerDefines["NUM_LAYER_GROUPS"] = cc.intToString(numGroups);
        layerDefines["NUM_CENTERS"] = cc.intToString(numCenters);
        layerDefines["LAYER_TAU"] = cc.doubleToString(tau);
        layerDefines["LAYER_R_IN"] = cc.doubleToString(rIn);
        layerDefines["LAYER_R_OUT"] = cc.doubleToString(rOut);
        ComputeProgram layerProgram = cc.compileProgram(MLPolKernelSources::layered, layerDefines);
        auto periodic = [&](ComputeKernel& k) {
            if (usePME)
                for (int i = 0; i < 5; i++)
                    k->addArg();
        };
        groupSwitchKernel = layerProgram->createKernel("computeGroupSwitch");
        groupSwitchKernel->addArg(cc.getPosq());
        groupSwitchKernel->addArg(layerGroupRef);
        groupSwitchKernel->addArg(layerCenters);
        groupSwitchKernel->addArg(layerGroupS);
        groupSwitchKernel->addArg(layerGroupDS);
        groupSwitchKernel->addArg(layerSoftminW);
        periodic(groupSwitchKernel);
        atomWeightsKernel = layerProgram->createKernel("computeAtomWeights");
        atomWeightsKernel->addArg(layerGroupOf);
        atomWeightsKernel->addArg(layerGroupS);
        atomWeightsKernel->addArg(layerGroupDS);
        atomWeightsKernel->addArg(layerAtomWeight);
        atomWeightsKernel->addArg(layerActive);
        atomWeightsKernel->addArg(layerTrans);
        atomWeightsKernel->addArg(layerCounters);
        layeredFieldKernel = layerProgram->createKernel("layeredInducedField");
        layeredFieldKernel->addArg(cc.getPosq());
        layeredFieldKernel->addArg(dampingAndThole);
        layeredFieldKernel->addArg(layerActive);
        layeredFieldKernel->addArg(layerCounters);
        layeredFieldKernel->addArg(layerAtomWeight);
        layeredFieldKernel->addArg(inducedDipole);
        layeredFieldKernel->addArg(inducedDipolePolar);
        layeredFieldKernel->addArg(inducedField);
        layeredFieldKernel->addArg(inducedFieldPolar);
        periodic(layeredFieldKernel);
        layeredCouplingKernel = layerProgram->createKernel("layeredCouplingForces");
        layeredCouplingKernel->addArg(cc.getPosq());
        layeredCouplingKernel->addArg(dampingAndThole);
        layeredCouplingKernel->addArg(layerActive);
        layeredCouplingKernel->addArg(layerCounters);
        layeredCouplingKernel->addArg(layerAtomWeight);
        layeredCouplingKernel->addArg(inducedDipole);
        layeredCouplingKernel->addArg(inducedDipolePolar);
        layeredCouplingKernel->addArg(cc.getLongForceBuffer());
        periodic(layeredCouplingKernel);
        layeredSwitchDerivKernel = layerProgram->createKernel("layeredSwitchDerivatives");
        layeredSwitchDerivKernel->addArg(cc.getPosq());
        layeredSwitchDerivKernel->addArg(dampingAndThole);
        layeredSwitchDerivKernel->addArg(layerActive);
        layeredSwitchDerivKernel->addArg(layerTrans);
        layeredSwitchDerivKernel->addArg(layerCounters);
        layeredSwitchDerivKernel->addArg(layerAtomWeight);
        layeredSwitchDerivKernel->addArg(inducedDipole);
        layeredSwitchDerivKernel->addArg(inducedDipolePolar);
        layeredSwitchDerivKernel->addArg(layerDEds);
        periodic(layeredSwitchDerivKernel);
        layeredSwitchForceKernel = layerProgram->createKernel("layeredSwitchForces");
        layeredSwitchForceKernel->addArg(cc.getPosq());
        layeredSwitchForceKernel->addArg(layerGroupRef);
        layeredSwitchForceKernel->addArg(layerCenters);
        layeredSwitchForceKernel->addArg(layerGroupDS);
        layeredSwitchForceKernel->addArg(layerSoftminW);
        layeredSwitchForceKernel->addArg(layerGroupAtomStart);
        layeredSwitchForceKernel->addArg(layerGroupAtoms);
        layeredSwitchForceKernel->addArg(layerDEds);
        layeredSwitchForceKernel->addArg(cc.getLongForceBuffer());
        periodic(layeredSwitchForceKernel);
    }

    // POLEN: ML-region kernels
    if (!mlAtoms.empty()) {
        map<string, string> mlDefines = defines;
        mlDefines["NUM_ML"] = cc.intToString(mlAtoms.size());
        mlDefines["ML_THREADS"] = cc.intToString(MLThreads);
        if (layered)
            mlDefines["MLPOL_LAYERED"] = "";
        if (usePME) {
            mlDefines["GRID_SIZE_X"] = cc.intToString(gridSizeX);
            mlDefines["GRID_SIZE_Y"] = cc.intToString(gridSizeY);
            mlDefines["GRID_SIZE_Z"] = cc.intToString(gridSizeZ);
        }
        ComputeProgram mlProgram = cc.compileProgram(MLPolKernelSources::mlpol, mlDefines);
        mlIndex.initialize<int>(cc, mlAtoms.size(), "mlIndex");
        mlIndex.upload(mlAtoms);
        mlParams.initialize(cc, 11*mlAtoms.size(), elementSize, "mlParams");
        injectMLKernel = mlProgram->createKernel("injectMLParameters");
        injectMLKernel->addArg(cc.getPosq());
        injectMLKernel->addArg(localDipoles);
        injectMLKernel->addArg(localQuadrupoles);
        injectMLKernel->addArg(polarizability);
        injectMLKernel->addArg(dampingAndThole);
        injectMLKernel->addArg(multipoleParticles);
        injectMLKernel->addArg(mlIndex);
        injectMLKernel->addArg(mlParams);
        injectMLKernel->addArg((float) mlThole);

        // derivatives: pairs with scale != 1 of each ML atom (sorted lists)
        vector<int> start(1, 0), partners;
        vector<mm_float4> scales;
        for (int a : mlAtoms) {
            set<int> cand;
            vector<vector<int> > cov;
            force.getCovalentMaps(a, cov);
            for (auto& list : cov)
                cand.insert(list.begin(), list.end());
            cand.erase(a);
            for (int b : cand) {
                mm_float4 sc = pairScaleFactors(force, a, b);
                if (sc.x != 1 || sc.y != 1 || sc.z != 1 || sc.w != 1) {
                    partners.push_back(b);
                    scales.push_back(sc);
                }
            }
            start.push_back(partners.size());
        }
        if (partners.empty()) {
            partners.push_back(-1);
            scales.push_back(mm_float4(1, 1, 1, 1));
        }
        mlSpecialStart.initialize<int>(cc, start.size(), "mlSpecialStart");
        mlSpecialAtom.initialize<int>(cc, partners.size(), "mlSpecialAtom");
        mlSpecialScale.initialize<mm_float4>(cc, scales.size(), "mlSpecialScale");
        mlSpecialStart.upload(start);
        mlSpecialAtom.upload(partners);
        mlSpecialScale.upload(scales);
        int mixedSize = (cc.getUseDoublePrecision() || cc.getUseMixedPrecision() ? sizeof(double) : sizeof(float));
        mlDerivatives.initialize(cc, 14*mlAtoms.size(), mixedSize, "mlDerivatives");
        mlPos.initialize(cc, mlAtoms.size(), 4*elementSize, "mlPos");
        mlForce.initialize(cc, 3*mlAtoms.size(), elementSize, "mlForce");
        gatherMLKernel = mlProgram->createKernel("gatherMLPositions");
        gatherMLKernel->addArg(cc.getPosq());
        gatherMLKernel->addArg(mlIndex);
        gatherMLKernel->addArg(mlPos);
        addMLForcesKernel = mlProgram->createKernel("addMLForces");
        addMLForcesKernel->addArg(cc.getLongForceBuffer());
        addMLForcesKernel->addArg(mlIndex);
        addMLForcesKernel->addArg(mlForce);
        penParams.initialize(cc, 2*mlAtoms.size(), elementSize, "penParams");
        penDerivs.initialize(cc, 2*mlAtoms.size(), mixedSize, "penDerivs");
        penetrationKernel = mlProgram->createKernel("computePenetration");
        penetrationKernel->addArg(cc.getPosq());
        penetrationKernel->addArg(labDipoles);
        penetrationKernel->addArg(labQuadrupoles);
        penetrationKernel->addArg(dampingAndThole);
        penetrationKernel->addArg(mlIndex);
        penetrationKernel->addArg(penParams);
        penetrationKernel->addArg(cc.getLongForceBuffer());
        penetrationKernel->addArg(torque);
        penetrationKernel->addArg(cc.getEnergyBuffer());
        penetrationKernel->addArg(penDerivs);
        if (usePME)
            for (int i = 0; i < 5; i++)
                penetrationKernel->addArg();
        mlDerivativesKernel = mlProgram->createKernel("computeMLDerivatives");
        mlDerivativesKernel->addArg(cc.getPosq());
        mlDerivativesKernel->addArg(labDipoles);
        mlDerivativesKernel->addArg(labQuadrupoles);
        mlDerivativesKernel->addArg(inducedDipole);
        mlDerivativesKernel->addArg(inducedDipolePolar);
        mlDerivativesKernel->addArg(dampingAndThole);
        mlDerivativesKernel->addArg(polarizability);
        mlDerivativesKernel->addArg(mlIndex);
        mlDerivativesKernel->addArg(mlSpecialStart);
        mlDerivativesKernel->addArg(mlSpecialAtom);
        mlDerivativesKernel->addArg(mlSpecialScale);
        mlDerivativesKernel->addArg(mlDerivatives);
        if (usePME)
            for (int i = 0; i < 11; i++)
                mlDerivativesKernel->addArg();   // box (5), phi, phidp, reciprocal (3) and the plasma term: set per call
        if (layered)
            mlDerivativesKernel->addArg(layerAtomWeight);
    }

    // Set up PME.
    
    if (usePME) {
        // Create required data structures.

        int elementSize = (cc.getUseDoublePrecision() ? sizeof(double) : sizeof(float));
        pmeGrid1.initialize(cc, gridSizeX*gridSizeY*gridSizeZ, 2*elementSize, "pmeGrid1");
        pmeGrid2.initialize(cc, gridSizeX*gridSizeY*gridSizeZ, 2*elementSize, "pmeGrid2");
        if (useFixedPointChargeSpreading()) {
            pmeGridLong.initialize(cc, 2*gridSizeX*gridSizeY*gridSizeZ, sizeof(long long), "pmeGridLong");
            cc.addAutoclearBuffer(pmeGridLong);
        }
        else
            cc.addAutoclearBuffer(pmeGrid1);
        pmeBsplineModuliX.initialize(cc, gridSizeX, elementSize, "pmeBsplineModuliX");
        pmeBsplineModuliY.initialize(cc, gridSizeY, elementSize, "pmeBsplineModuliY");
        pmeBsplineModuliZ.initialize(cc, gridSizeZ, elementSize, "pmeBsplineModuliZ");
        pmePhi.initialize(cc, 20*numMultipoles, elementSize, "pmePhi");
        pmePhid.initialize(cc, 10*numMultipoles, elementSize, "pmePhid");
        pmePhip.initialize(cc, 10*numMultipoles, elementSize, "pmePhip");
        pmePhidp.initialize(cc, 20*numMultipoles, elementSize, "pmePhidp");
        pmeCphi.initialize(cc, 10*numMultipoles, elementSize, "pmeCphi");
        fft = cc.createFFT(gridSizeX, gridSizeY, gridSizeZ, false);

        // Create the PME kernels.

        map<string, string> pmeDefines;
        pmeDefines["EWALD_ALPHA"] = cc.doubleToString(pmeAlpha);
        pmeDefines["PME_ORDER"] = cc.intToString(PmeOrder);
        pmeDefines["NUM_ATOMS"] = cc.intToString(numMultipoles);
        pmeDefines["PADDED_NUM_ATOMS"] = cc.intToString(cc.getPaddedNumAtoms());
        pmeDefines["NUM_INDICES"] = "0";
        pmeDefines["EPSILON_FACTOR"] = cc.doubleToString(ONE_4PI_EPS0);
        pmeDefines["GRID_SIZE_X"] = cc.intToString(gridSizeX);
        pmeDefines["GRID_SIZE_Y"] = cc.intToString(gridSizeY);
        pmeDefines["GRID_SIZE_Z"] = cc.intToString(gridSizeZ);
        pmeDefines["M_PI"] = cc.doubleToString(M_PI);
        pmeDefines["SQRT_PI"] = cc.doubleToString(sqrt(M_PI));
        if (polarizationType == AmoebaMultipoleForce::Direct)
            pmeDefines["DIRECT_POLARIZATION"] = "";
        else if (polarizationType == AmoebaMultipoleForce::Mutual)
            pmeDefines["MUTUAL_POLARIZATION"] = "";
        else if (polarizationType == AmoebaMultipoleForce::Extrapolated)
            pmeDefines["EXTRAPOLATED_POLARIZATION"] = "";
        if (useFixedPointChargeSpreading())
            pmeDefines["USE_FIXED_POINT_CHARGE_SPREADING"] = "";
        program = cc.compileProgram(MLPolKernelSources::multipolePme, pmeDefines);
        pmeTransformMultipolesKernel = program->createKernel("transformMultipolesToFractionalCoordinates");
        pmeTransformMultipolesKernel->addArg(labDipoles);
        pmeTransformMultipolesKernel->addArg(labQuadrupoles);
        pmeTransformMultipolesKernel->addArg(fracDipoles);
        pmeTransformMultipolesKernel->addArg(fracQuadrupoles);
        for (int i = 0; i < 3; i++)
            pmeTransformMultipolesKernel->addArg();
        pmeTransformPotentialKernel = program->createKernel("transformPotentialToCartesianCoordinates");
        pmeTransformPotentialKernel->addArg();
        pmeTransformPotentialKernel->addArg(pmeCphi);
        for (int i = 0; i < 3; i++)
            pmeTransformPotentialKernel->addArg();
        pmeSpreadFixedMultipolesKernel = program->createKernel("gridSpreadFixedMultipoles");
        pmeSpreadFixedMultipolesKernel->addArg(cc.getPosq());
        pmeSpreadFixedMultipolesKernel->addArg(fracDipoles);
        pmeSpreadFixedMultipolesKernel->addArg(fracQuadrupoles);
        if (useFixedPointChargeSpreading())
            pmeSpreadFixedMultipolesKernel->addArg(pmeGridLong);
        else
            pmeSpreadFixedMultipolesKernel->addArg(pmeGrid1);
        for (int i = 0; i < 6; i++)
            pmeSpreadFixedMultipolesKernel->addArg();
        pmeSpreadInducedDipolesKernel = program->createKernel("gridSpreadInducedDipoles");
        pmeSpreadInducedDipolesKernel->addArg(cc.getPosq());
        pmeSpreadInducedDipolesKernel->addArg(inducedDipole);
        pmeSpreadInducedDipolesKernel->addArg(inducedDipolePolar);
        if (useFixedPointChargeSpreading())
            pmeSpreadInducedDipolesKernel->addArg(pmeGridLong);
        else
            pmeSpreadInducedDipolesKernel->addArg(pmeGrid1);
        for (int i = 0; i < 6; i++)
            pmeSpreadInducedDipolesKernel->addArg();
        if (useFixedPointChargeSpreading()) {
            pmeFinishSpreadChargeKernel = program->createKernel("finishSpreadCharge");
            pmeFinishSpreadChargeKernel->addArg(pmeGridLong);
            pmeFinishSpreadChargeKernel->addArg(pmeGrid1);
        }
        pmeConvolutionKernel = program->createKernel("reciprocalConvolution");
        pmeConvolutionKernel->addArg(pmeGrid2);
        pmeConvolutionKernel->addArg(pmeBsplineModuliX);
        pmeConvolutionKernel->addArg(pmeBsplineModuliY);
        pmeConvolutionKernel->addArg(pmeBsplineModuliZ);
        for (int i = 0; i < 4; i++)
            pmeConvolutionKernel->addArg();
        pmeFixedPotentialKernel = program->createKernel("computeFixedPotentialFromGrid");
        pmeFixedPotentialKernel->addArg(pmeGrid1);
        pmeFixedPotentialKernel->addArg(pmePhi);
        pmeFixedPotentialKernel->addArg(field);
        pmeFixedPotentialKernel->addArg(fieldPolar);
        pmeFixedPotentialKernel->addArg(cc.getPosq());
        pmeFixedPotentialKernel->addArg(labDipoles);
        for (int i = 0; i < 6; i++)
            pmeFixedPotentialKernel->addArg();
        pmeInducedPotentialKernel = program->createKernel("computeInducedPotentialFromGrid");
        pmeInducedPotentialKernel->addArg(pmeGrid1);
        pmeInducedPotentialKernel->addArg(pmePhid);
        pmeInducedPotentialKernel->addArg(pmePhip);
        pmeInducedPotentialKernel->addArg(pmePhidp);
        pmeInducedPotentialKernel->addArg(cc.getPosq());
        for (int i = 0; i < 6; i++)
            pmeInducedPotentialKernel->addArg();
        pmeFixedForceKernel = program->createKernel("computeFixedMultipoleForceAndEnergy");
        pmeFixedForceKernel->addArg(cc.getPosq());
        pmeFixedForceKernel->addArg(cc.getLongForceBuffer());
        pmeFixedForceKernel->addArg(torque);
        pmeFixedForceKernel->addArg(cc.getEnergyBuffer());
        pmeFixedForceKernel->addArg(labDipoles);
        pmeFixedForceKernel->addArg(labQuadrupoles);
        pmeFixedForceKernel->addArg(fracDipoles);
        pmeFixedForceKernel->addArg(fracQuadrupoles);
        pmeFixedForceKernel->addArg(pmePhi);
        pmeFixedForceKernel->addArg(pmeCphi);
        for (int i = 0; i < 3; i++)
            pmeFixedForceKernel->addArg();
        pmeInducedForceKernel = program->createKernel("computeInducedDipoleForceAndEnergy");
        pmeInducedForceKernel->addArg(cc.getPosq());
        pmeInducedForceKernel->addArg(cc.getLongForceBuffer());
        pmeInducedForceKernel->addArg(torque);
        pmeInducedForceKernel->addArg(cc.getEnergyBuffer());
        pmeInducedForceKernel->addArg(labDipoles);
        pmeInducedForceKernel->addArg(labQuadrupoles);
        pmeInducedForceKernel->addArg(fracDipoles);
        pmeInducedForceKernel->addArg(fracQuadrupoles);
        pmeInducedForceKernel->addArg(inducedDipole);
        pmeInducedForceKernel->addArg(inducedDipolePolar);
        pmeInducedForceKernel->addArg(pmePhi);
        pmeInducedForceKernel->addArg(pmePhid);
        pmeInducedForceKernel->addArg(pmePhip);
        pmeInducedForceKernel->addArg(pmePhidp);
        pmeInducedForceKernel->addArg(pmeCphi);
        for (int i = 0; i < 3; i++)
            pmeInducedForceKernel->addArg();
        if (layered) {
            map<string, string> directDefines = pmeDefines;
            directDefines.erase("MUTUAL_POLARIZATION");
            directDefines["DIRECT_POLARIZATION"] = "";
            ComputeProgram directProgram = cc.compileProgram(MLPolKernelSources::multipolePme, directDefines);
            ComputeKernel k = directProgram->createKernel("computeInducedDipoleForceAndEnergy");
            k->addArg(cc.getPosq());
            k->addArg(cc.getLongForceBuffer());
            k->addArg(torque);
            k->addArg(cc.getEnergyBuffer());
            k->addArg(labDipoles);
            k->addArg(labQuadrupoles);
            k->addArg(fracDipoles);
            k->addArg(fracQuadrupoles);
            k->addArg(inducedDipole);
            k->addArg(inducedDipolePolar);
            k->addArg(pmePhi);
            k->addArg(pmePhid);
            k->addArg(pmePhip);
            k->addArg(pmePhidp);
            k->addArg(pmeCphi);
            for (int i = 0; i < 3; i++)
                k->addArg();
            pmeInducedForceKernel = k;
        }
        if (polarizationType != AmoebaMultipoleForce::Direct) {
            pmeRecordInducedFieldDipolesKernel = program->createKernel("recordInducedFieldDipoles");
            pmeRecordInducedFieldDipolesKernel->addArg(pmePhid);
            pmeRecordInducedFieldDipolesKernel->addArg(pmePhip);
            pmeRecordInducedFieldDipolesKernel->addArg(inducedField);
            pmeRecordInducedFieldDipolesKernel->addArg(inducedFieldPolar);
            pmeRecordInducedFieldDipolesKernel->addArg(inducedDipole);
            pmeRecordInducedFieldDipolesKernel->addArg(inducedDipolePolar);
            for (int i = 0; i < 3; i++)
                pmeRecordInducedFieldDipolesKernel->addArg();
            if (polarizationType == AmoebaMultipoleForce::Extrapolated) {
                pmeRecordInducedFieldDipolesKernel->addArg(inducedDipoleFieldGradient);
                pmeRecordInducedFieldDipolesKernel->addArg(inducedDipoleFieldGradientPolar);
            }
        }

        // Initialize the B-spline moduli.

        double data[PmeOrder];
        double x = 0.0;
        data[0] = 1.0 - x;
        data[1] = x;
        for (int i = 2; i < PmeOrder; i++) {
            double denom = 1.0/i;
            data[i] = x*data[i-1]*denom;
            for (int j = 1; j < i; j++)
                data[i-j] = ((x+j)*data[i-j-1] + ((i-j+1)-x)*data[i-j])*denom;
            data[0] = (1.0-x)*data[0]*denom;
        }
        int maxSize = max(max(gridSizeX, gridSizeY), gridSizeZ);
        vector<double> bsplines_data(maxSize+1, 0.0);
        for (int i = 2; i <= PmeOrder+1; i++)
            bsplines_data[i] = data[i-2];
        for (int dim = 0; dim < 3; dim++) {
            int ndata = (dim == 0 ? gridSizeX : dim == 1 ? gridSizeY : gridSizeZ);
            vector<double> moduli(ndata);

            // get the modulus of the discrete Fourier transform

            double factor = 2.0*M_PI/ndata;
            for (int i = 0; i < ndata; i++) {
                double sc = 0.0;
                double ss = 0.0;
                for (int j = 1; j <= ndata; j++) {
                    double arg = factor*i*(j-1);
                    sc += bsplines_data[j]*cos(arg);
                    ss += bsplines_data[j]*sin(arg);
                }
                moduli[i] = sc*sc+ss*ss;
            }

            // Fix for exponential Euler spline interpolation failure.

            double eps = 1.0e-7;
            if (moduli[0] < eps)
                moduli[0] = 0.9*moduli[1];
            for (int i = 1; i < ndata-1; i++)
                if (moduli[i] < eps)
                    moduli[i] = 0.9*(moduli[i-1]+moduli[i+1]);
            if (moduli[ndata-1] < eps)
                moduli[ndata-1] = 0.9*moduli[ndata-2];

            // Compute and apply the optimal zeta coefficient.

            int jcut = 50;
            for (int i = 1; i <= ndata; i++) {
                int k = i - 1;
                if (i > ndata/2)
                    k = k - ndata;
                double zeta;
                if (k == 0)
                    zeta = 1.0;
                else {
                    double sum1 = 1.0;
                    double sum2 = 1.0;
                    factor = M_PI*k/ndata;
                    for (int j = 1; j <= jcut; j++) {
                        double arg = factor/(factor+M_PI*j);
                        sum1 += pow(arg, PmeOrder);
                        sum2 += pow(arg, 2*PmeOrder);
                    }
                    for (int j = 1; j <= jcut; j++) {
                        double arg = factor/(factor-M_PI*j);
                        sum1 += pow(arg, PmeOrder);
                        sum2 += pow(arg, 2*PmeOrder);
                    }
                    zeta = sum2/sum1;
                }
                moduli[i-1] = moduli[i-1]*zeta*zeta;
            }
            if (cc.getUseDoublePrecision()) {
                if (dim == 0)
                    pmeBsplineModuliX.upload(moduli);
                else if (dim == 1)
                    pmeBsplineModuliY.upload(moduli);
                else
                    pmeBsplineModuliZ.upload(moduli);
            }
            else {
                vector<float> modulif(ndata);
                for (int i = 0; i < ndata; i++)
                    modulif[i] = (float) moduli[i];
                if (dim == 0)
                    pmeBsplineModuliX.upload(modulif);
                else if (dim == 1)
                    pmeBsplineModuliY.upload(modulif);
                else
                    pmeBsplineModuliZ.upload(modulif);
            }
        }
    }

    // Add an interaction to the default nonbonded kernel.  This doesn't actually do any calculations.  It's
    // just so that NonbondedUtilities will build the exclusion flags and maintain the neighbor list.
    
    cc.getNonbondedUtilities().addInteraction(usePME, usePME, true, force.getCutoffDistance(), exclusions, "", force.getForceGroup());
    cc.getNonbondedUtilities().setUsePadding(false);
    cc.addForce(new ForceInfo(force, &isML));
}

void CommonCalcMLPolMultipoleForceKernel::initializeScaleFactors() {
    hasInitializedScaleFactors = true;
    NonbondedUtilities& nb = cc.getNonbondedUtilities();
    
    // Figure out the covalent flag values to use for each atom pair.

    vector<mm_int2> exclusionTiles;
    nb.getExclusionTiles().download(exclusionTiles);
    map<pair<int, int>, int> exclusionTileMap;
    for (int i = 0; i < (int) exclusionTiles.size(); i++) {
        mm_int2 tile = exclusionTiles[i];
        exclusionTileMap[make_pair(tile.x, tile.y)] = i;
    }
    covalentFlags.resize(nb.getExclusions().getSize());
    vector<mm_int2> covalentFlagsVec(nb.getExclusions().getSize(), mm_int2(0, 0));
    for (mm_int4 values : covalentFlagValues) {
        int atom1 = values.x;
        int atom2 = values.y;
        int value = values.z;
        int x = atom1/ComputeContext::TileSize;
        int offset1 = atom1-x*ComputeContext::TileSize;
        int y = atom2/ComputeContext::TileSize;
        int offset2 = atom2-y*ComputeContext::TileSize;
        int f1 = (value == 0 || value == 1 ? 1 : 0);
        int f2 = (value == 0 || value == 2 ? 1 : 0);
        if (x == y) {
            int index = exclusionTileMap[make_pair(x, y)]*ComputeContext::TileSize;
            covalentFlagsVec[index+offset1].x |= f1<<offset2;
            covalentFlagsVec[index+offset1].y |= f2<<offset2;
            covalentFlagsVec[index+offset2].x |= f1<<offset1;
            covalentFlagsVec[index+offset2].y |= f2<<offset1;
        }
        else if (x > y) {
            int index = exclusionTileMap[make_pair(x, y)]*ComputeContext::TileSize;
            covalentFlagsVec[index+offset1].x |= f1<<offset2;
            covalentFlagsVec[index+offset1].y |= f2<<offset2;
        }
        else {
            int index = exclusionTileMap[make_pair(y, x)]*ComputeContext::TileSize;
            covalentFlagsVec[index+offset2].x |= f1<<offset1;
            covalentFlagsVec[index+offset2].y |= f2<<offset1;
        }
    }
    covalentFlags.upload(covalentFlagsVec);
    
    // Do the same for the polarization flags.
    
    polarizationGroupFlags.resize(nb.getExclusions().getSize());
    vector<unsigned int> polarizationGroupFlagsVec(nb.getExclusions().getSize(), 0);
    for (mm_int2 values : polarizationFlagValues) {
        int atom1 = values.x;
        int atom2 = values.y;
        int x = atom1/ComputeContext::TileSize;
        int offset1 = atom1-x*ComputeContext::TileSize;
        int y = atom2/ComputeContext::TileSize;
        int offset2 = atom2-y*ComputeContext::TileSize;
        if (x == y) {
            int index = exclusionTileMap[make_pair(x, y)]*ComputeContext::TileSize;
            polarizationGroupFlagsVec[index+offset1] |= 1<<offset2;
            polarizationGroupFlagsVec[index+offset2] |= 1<<offset1;
        }
        else if (x > y) {
            int index = exclusionTileMap[make_pair(x, y)]*ComputeContext::TileSize;
            polarizationGroupFlagsVec[index+offset1] |= 1<<offset2;
        }
        else {
            int index = exclusionTileMap[make_pair(y, x)]*ComputeContext::TileSize;
            polarizationGroupFlagsVec[index+offset2] |= 1<<offset1;
        }
    }
    polarizationGroupFlags.upload(polarizationGroupFlagsVec);
}

double CommonCalcMLPolMultipoleForceKernel::executeAmoeba(ContextImpl& context, bool includeForces, bool includeEnergy) {
    ContextSelector selector(cc);
    if (!hasInitializedScaleFactors) {
        initializeScaleFactors();
    }
    NonbondedUtilities& nb = cc.getNonbondedUtilities();
    
    // Compute the lab frame moments.

    computeMomentsKernel->execute(cc.getNumAtoms());
    if (layered) {   // POLEN: layer weights and site lists (before the solver)
        cc.clearBuffer(layerCounters);
        if (usePME)
            setPeriodicBoxArgs(cc, groupSwitchKernel, 6);
        groupSwitchKernel->execute(max((int) layerGroupS.getSize(), 1));
        atomWeightsKernel->execute(cc.getNumAtoms());
    }
    int startTileIndex = nb.getStartTileIndex();
    int numTileIndices = nb.getNumTiles();
    int numForceThreadBlocks = nb.getNumForceThreadBlocks();
    electrostaticsKernel->setArg(7, startTileIndex);
    electrostaticsKernel->setArg(8, numTileIndices);
    computeFixedFieldKernel->setArg(6, startTileIndex);
    computeFixedFieldKernel->setArg(7, numTileIndices);
    if (!pmeGrid1.isInitialized()) {
        // Compute induced dipoles.
        
        computeFixedFieldKernel->execute(numForceThreadBlocks*fixedFieldThreads, fixedFieldThreads);
        recordInducedDipolesKernel->execute(cc.getNumAtoms());
        
        // Iterate until the dipoles converge.
        
        if (polarizationType == AmoebaMultipoleForce::Extrapolated)
            computeExtrapolatedDipoles();
        for (int i = 0; i < maxInducedIterations; i++) {
            computeInducedField();
            bool converged = iterateDipolesByDIIS(i);
            if (converged)
                break;
        }
        
        // Compute electrostatic force.
        
        electrostaticsKernel->execute(numForceThreadBlocks*electrostaticsThreads, electrostaticsThreads);
    }
    else {
        // Compute reciprocal box vectors.
        
        Vec3 a, b, c;
        cc.getPeriodicBoxVectors(a, b, c);
        mm_double4 recipBoxVectors[3];
        cc.computeReciprocalBoxVectors(recipBoxVectors);
        if (cc.getUseDoublePrecision()) {
            mm_double4 boxVectors[] = {mm_double4(a[0], a[1], a[2], 0), mm_double4(b[0], b[1], b[2], 0), mm_double4(c[0], c[1], c[2], 0)};
            pmeConvolutionKernel->setArg(4, mm_double4(a[0], b[1], c[2], 0));
            for (int i = 0; i < 3; i++) {
                pmeTransformMultipolesKernel->setArg(4+i, recipBoxVectors[i]);
                pmeTransformPotentialKernel->setArg(2+i, recipBoxVectors[i]);
                pmeSpreadFixedMultipolesKernel->setArg(4+i, boxVectors[i]);
                pmeSpreadFixedMultipolesKernel->setArg(7+i, recipBoxVectors[i]);
                pmeSpreadInducedDipolesKernel->setArg(4+i, boxVectors[i]);
                pmeSpreadInducedDipolesKernel->setArg(7+i, recipBoxVectors[i]);
                pmeConvolutionKernel->setArg(5+i, recipBoxVectors[i]);
                pmeFixedPotentialKernel->setArg(6+i, boxVectors[i]);
                pmeFixedPotentialKernel->setArg(9+i, recipBoxVectors[i]);
                pmeInducedPotentialKernel->setArg(5+i, boxVectors[i]);
                pmeInducedPotentialKernel->setArg(8+i, recipBoxVectors[i]);
                pmeFixedForceKernel->setArg(10+i, recipBoxVectors[i]);
                pmeInducedForceKernel->setArg(15+i, recipBoxVectors[i]);
                if (polarizationType != AmoebaMultipoleForce::Direct)
                    pmeRecordInducedFieldDipolesKernel->setArg(6+i, recipBoxVectors[i]);
            }
        }
        else {
            mm_float4 recipBoxVectorsFloat[3];
            recipBoxVectorsFloat[0] = mm_float4((float) recipBoxVectors[0].x, 0, 0, 0);
            recipBoxVectorsFloat[1] = mm_float4((float) recipBoxVectors[1].x, (float) recipBoxVectors[1].y, 0, 0);
            recipBoxVectorsFloat[2] = mm_float4((float) recipBoxVectors[2].x, (float) recipBoxVectors[2].y, (float) recipBoxVectors[2].z, 0);
            mm_float4 boxVectors[] = {mm_float4(a[0], a[1], a[2], 0), mm_float4(b[0], b[1], b[2], 0), mm_float4(c[0], c[1], c[2], 0)};
            pmeConvolutionKernel->setArg(4, mm_float4(a[0], b[1], c[2], 0));
            for (int i = 0; i < 3; i++) {
                pmeTransformMultipolesKernel->setArg(4+i, recipBoxVectorsFloat[i]);
                pmeTransformPotentialKernel->setArg(2+i, recipBoxVectorsFloat[i]);
                pmeSpreadFixedMultipolesKernel->setArg(4+i, boxVectors[i]);
                pmeSpreadFixedMultipolesKernel->setArg(7+i, recipBoxVectorsFloat[i]);
                pmeSpreadInducedDipolesKernel->setArg(4+i, boxVectors[i]);
                pmeSpreadInducedDipolesKernel->setArg(7+i, recipBoxVectorsFloat[i]);
                pmeConvolutionKernel->setArg(5+i, recipBoxVectorsFloat[i]);
                pmeFixedPotentialKernel->setArg(6+i, boxVectors[i]);
                pmeFixedPotentialKernel->setArg(9+i, recipBoxVectorsFloat[i]);
                pmeInducedPotentialKernel->setArg(5+i, boxVectors[i]);
                pmeInducedPotentialKernel->setArg(8+i, recipBoxVectorsFloat[i]);
                pmeFixedForceKernel->setArg(10+i, recipBoxVectorsFloat[i]);
                pmeInducedForceKernel->setArg(15+i, recipBoxVectorsFloat[i]);
                if (polarizationType != AmoebaMultipoleForce::Direct)
                    pmeRecordInducedFieldDipolesKernel->setArg(6+i, recipBoxVectorsFloat[i]);
            }
        }

        // Reciprocal space calculation.
        
        unsigned int maxTiles = nb.getInteractingTiles().getSize();
        pmeTransformMultipolesKernel->execute(cc.getNumAtoms());
        pmeSpreadFixedMultipolesKernel->execute(cc.getNumAtoms());
        if (useFixedPointChargeSpreading())
            pmeFinishSpreadChargeKernel->execute(pmeGrid1.getSize());
        fft->execFFT(pmeGrid1, pmeGrid2, true);
        pmeConvolutionKernel->execute(gridSizeX*gridSizeY*gridSizeZ, 256);
        fft->execFFT(pmeGrid2, pmeGrid1, false);
        pmeFixedPotentialKernel->execute(cc.getNumAtoms());
        pmeTransformPotentialKernel->setArg(0, pmePhi);
        pmeTransformPotentialKernel->execute(cc.getNumAtoms());
        pmeFixedForceKernel->execute(cc.getNumAtoms());

        // Direct space calculation.

        setPeriodicBoxArgs(cc, computeFixedFieldKernel, 10);
        computeFixedFieldKernel->setArg(15, maxTiles);
        computeFixedFieldKernel->execute(numForceThreadBlocks*fixedFieldThreads, fixedFieldThreads);
        recordInducedDipolesKernel->execute(cc.getNumAtoms());

        // Reciprocal space calculation for the induced dipoles.

        if (useFixedPointChargeSpreading())
            cc.clearBuffer(pmeGridLong);
        else
            cc.clearBuffer(pmeGrid1);
        pmeSpreadInducedDipolesKernel->execute(cc.getNumAtoms());
        if (useFixedPointChargeSpreading())
            pmeFinishSpreadChargeKernel->execute(pmeGrid1.getSize());
        fft->execFFT(pmeGrid1, pmeGrid2, true);
        pmeConvolutionKernel->execute(gridSizeX*gridSizeY*gridSizeZ, 256);
        fft->execFFT(pmeGrid2, pmeGrid1, false);
        pmeInducedPotentialKernel->execute(cc.getNumAtoms());
        
        // Iterate until the dipoles converge.
        
        if (polarizationType == AmoebaMultipoleForce::Extrapolated)
            computeExtrapolatedDipoles();
        for (int i = 0; i < maxInducedIterations; i++) {
            computeInducedField();
            bool converged = iterateDipolesByDIIS(i);
            if (converged)
                break;
        }
        
        if (layered) {
            // POLEN: reciprocal potentials of the converged dipoles (the layered SCF does not use PME)
            if (useFixedPointChargeSpreading())
                cc.clearBuffer(pmeGridLong);
            else
                cc.clearBuffer(pmeGrid1);
            pmeSpreadInducedDipolesKernel->execute(cc.getNumAtoms());
            if (useFixedPointChargeSpreading())
                pmeFinishSpreadChargeKernel->execute(pmeGrid1.getSize());
            fft->execFFT(pmeGrid1, pmeGrid2, true);
            pmeConvolutionKernel->execute(gridSizeX*gridSizeY*gridSizeZ, 256);
            fft->execFFT(pmeGrid2, pmeGrid1, false);
            pmeInducedPotentialKernel->execute(cc.getNumAtoms());
        }

        // Compute electrostatic force.
        
        setPeriodicBoxArgs(cc, electrostaticsKernel, 11);
        electrostaticsKernel->setArg(16, maxTiles);
        electrostaticsKernel->execute(numForceThreadBlocks*electrostaticsThreads, electrostaticsThreads);
        pmeTransformPotentialKernel->setArg(0, pmePhidp);
        pmeTransformPotentialKernel->execute(cc.getNumAtoms());
        pmeInducedForceKernel->execute(cc.getNumAtoms());
    }
    
    // If using extrapolated polarization, add in force contributions from µ(m) T µ(n).
    
    if (polarizationType == AmoebaMultipoleForce::Extrapolated)
        addExtrapolatedGradientKernel->execute(numMultipoles);

    // POLEN: layers: forces of the induced–induced coupling and of the smooth switch

    if (layered) {
        if (usePME) {
            setPeriodicBoxArgs(cc, layeredCouplingKernel, 8);
            setPeriodicBoxArgs(cc, layeredSwitchDerivKernel, 9);
            setPeriodicBoxArgs(cc, layeredSwitchForceKernel, 9);
        }
        layeredCouplingKernel->execute(cc.getNumAtoms());
        layeredSwitchDerivKernel->execute(cc.getNumAtoms());
        layeredSwitchForceKernel->execute(max((int) layerGroupS.getSize(), 1));
    }

    // POLEN: charge penetration (forces and torques; the torques enter the mapping below)

    penDN.clear();
    penDS.clear();
    if (!penN.empty()) {
        if (usePME)
            setPeriodicBoxArgs(cc, penetrationKernel, 10);
        penetrationKernel->execute(mlAtoms.size()*MLThreads, MLThreads);
        vector<double> d;
        if (penDerivs.getElementSize() == sizeof(double))
            penDerivs.download(d);
        else {
            vector<float> f;
            penDerivs.download(f);
            d.assign(f.begin(), f.end());
        }
        for (int k = 0; k < (int) mlAtoms.size(); k++) {
            penDN.push_back(d[2*k]);
            penDS.push_back(d[2*k+1]);
        }
    }

    // Map torques to force.

    mapTorqueKernel->execute(cc.getNumAtoms());
    
    // Record the current atom positions so we can tell later if they have changed.
    
    cc.getPosq().copyTo(lastPositions);
    multipolesAreValid = true;

    // Correction for the neutralizing plasma.

    if (usePME) {
        Vec3 a, b, c;
        cc.getPeriodicBoxVectors(a, b, c);
        double volume = a[0] * b[1] * c[2];
        return -totalCharge*totalCharge/(8*EPSILON0*volume*pmeAlpha*pmeAlpha);
    }
    else
        return 0.0;
}

void CommonCalcMLPolMultipoleForceKernel::layeredComputeInducedField() {
    // POLEN: coupling by direct summation among the active sites (without the Ewald real/reciprocal field)
    cc.clearBuffer(inducedField);
    cc.clearBuffer(inducedFieldPolar);
    if (usePME)
        setPeriodicBoxArgs(cc, layeredFieldKernel, 9);
    layeredFieldKernel->execute(cc.getNumAtoms());
}

void CommonCalcMLPolMultipoleForceKernel::computeInducedField() {
    if (layered) {
        layeredComputeInducedField();
        return;
    }
    NonbondedUtilities& nb = cc.getNonbondedUtilities();
    int startTileIndex = nb.getStartTileIndex();
    int numTileIndices = nb.getNumTiles();
    int numForceThreadBlocks = nb.getNumForceThreadBlocks();
    computeInducedFieldKernel->setArg(6, startTileIndex);
    computeInducedFieldKernel->setArg(7, numTileIndices);
    if (usePME) {
        setPeriodicBoxArgs(cc, computeInducedFieldKernel, 10);
        computeInducedFieldKernel->setArg(15, (int) nb.getInteractingTiles().getSize());
    }
    cc.clearBuffer(inducedField);
    cc.clearBuffer(inducedFieldPolar);
    if (polarizationType == AmoebaMultipoleForce::Extrapolated) {
        cc.clearBuffer(inducedDipoleFieldGradient);
        cc.clearBuffer(inducedDipoleFieldGradientPolar);
    }
    computeInducedFieldKernel->execute(numForceThreadBlocks*inducedFieldThreads, inducedFieldThreads);
    if (pmeGrid1.isInitialized()) {
        if (useFixedPointChargeSpreading())
            cc.clearBuffer(pmeGridLong);
        else
            cc.clearBuffer(pmeGrid1);
        pmeSpreadInducedDipolesKernel->execute(cc.getNumAtoms());
        if (useFixedPointChargeSpreading())
            pmeFinishSpreadChargeKernel->execute(pmeGrid1.getSize());
        fft->execFFT(pmeGrid1, pmeGrid2, true);
        pmeConvolutionKernel->execute(gridSizeX*gridSizeY*gridSizeZ, 256);
        fft->execFFT(pmeGrid2, pmeGrid1, false);
        pmeInducedPotentialKernel->execute(cc.getNumAtoms());
        if (polarizationType == AmoebaMultipoleForce::Extrapolated) {
            pmeRecordInducedFieldDipolesKernel->execute(cc.getNumAtoms());
        }
        else {
            pmeRecordInducedFieldDipolesKernel->execute(cc.getNumAtoms());
        }
    }
}

bool CommonCalcMLPolMultipoleForceKernel::iterateDipolesByDIIS(int iteration) {
    void* npt = NULL;

    // Record the dipoles and errors into the lists of previous dipoles.

    recordDIISDipolesKernel->setArg(13, iteration);
    recordDIISDipolesKernel->setArg(6, npt);
    recordDIISDipolesKernel->setArg(7, inducedField);
    recordDIISDipolesKernel->setArg(8, inducedFieldPolar);
    recordDIISDipolesKernel->setArg(9, inducedDipole);
    recordDIISDipolesKernel->setArg(10, inducedDipolePolar);
    recordDIISDipolesKernel->setArg(11, prevDipoles);
    recordDIISDipolesKernel->setArg(12, prevDipolesPolar);
    recordDIISDipolesKernel->setArg(14, 0);
    recordDIISDipolesKernel->execute(cc.getNumThreadBlocks()*64, 64);
    mm_float2* errors = (mm_float2*) cc.getPinnedBuffer();
    inducedDipoleErrors.download(errors, false);
    syncEvent->enqueue();
    
    // Build the DIIS matrix.
    
    int numPrev = (iteration+1 < MaxPrevDIISDipoles ? iteration+1 : MaxPrevDIISDipoles);
    int threadBlocks = min(numPrev, cc.getNumThreadBlocks());
    int blockSize = min(512, buildMatrixKernel->getMaxBlockSize());
    buildMatrixKernel->setArg(1, iteration);
    buildMatrixKernel->execute(threadBlocks*blockSize, blockSize);
    
    // Solve the matrix.

    solveMatrixKernel->setArg(0, iteration);
    solveMatrixKernel->execute(32, 32);
    
    // Determine whether the iteration has converged.
    
    syncEvent->wait();
    double total1 = 0.0, total2 = 0.0;
    for (int j = 0; j < inducedDipoleErrors.getSize(); j++) {
        total1 += errors[j].x;
        total2 += errors[j].y;
    }
    if (48.033324*sqrt(max(total1, total2)/cc.getNumAtoms()) < inducedEpsilon)
        return true;
    
    // Compute the dipoles.
    
    updateInducedFieldKernel->setArg(0, inducedDipole);
    updateInducedFieldKernel->setArg(1, inducedDipolePolar);
    updateInducedFieldKernel->setArg(2, prevDipoles);
    updateInducedFieldKernel->setArg(3, prevDipolesPolar);
    updateInducedFieldKernel->setArg(5, numPrev);
    updateInducedFieldKernel->execute(3*cc.getNumAtoms(), 256);
    return false;
}

void CommonCalcMLPolMultipoleForceKernel::computeExtrapolatedDipoles() {
    // Start by storing the direct dipoles as PT0

    initExtrapolatedKernel->execute(extrapolatedDipole.getSize());

    // Recursively apply alpha.Tau to the µ_(n) components to generate µ_(n+1), and store the result

    for (int order = 1; order < maxExtrapolationOrder; ++order) {
        computeInducedField();
        iterateExtrapolatedKernel->setArg(0, order);
        iterateExtrapolatedKernel->execute(extrapolatedDipole.getSize());
    }
    
    // Take a linear combination of the µ_(n) components to form the total dipole

    computeExtrapolatedKernel->execute(extrapolatedDipole.getSize());
    computeInducedField();
}

void CommonCalcMLPolMultipoleForceKernel::ensureMultipolesValid(ContextImpl& context) {
    if (multipolesAreValid) {
        int numParticles = cc.getNumAtoms();
        if (cc.getUseDoublePrecision()) {
            vector<mm_double4> pos1, pos2;
            cc.getPosq().download(pos1);
            lastPositions.download(pos2);
            for (int i = 0; i < numParticles; i++)
                if (pos1[i].x != pos2[i].x || pos1[i].y != pos2[i].y || pos1[i].z != pos2[i].z) {
                    multipolesAreValid = false;
                    break;
                }
        }
        else {
            vector<mm_float4> pos1, pos2;
            cc.getPosq().download(pos1);
            lastPositions.download(pos2);
            for (int i = 0; i < numParticles; i++)
                if (pos1[i].x != pos2[i].x || pos1[i].y != pos2[i].y || pos1[i].z != pos2[i].z) {
                    multipolesAreValid = false;
                    break;
                }
        }
    }
    if (!multipolesAreValid) {
        ContextDeselector deselector(cc);
        context.calcForcesAndEnergy(false, false, context.getIntegrator().getIntegrationForceGroups());
    }
}

void CommonCalcMLPolMultipoleForceKernel::getLabFramePermanentDipoles(ContextImpl& context, vector<Vec3>& dipoles) {
    ContextSelector selector(cc);
    ensureMultipolesValid(context);
    int numParticles = cc.getNumAtoms();
    dipoles.resize(numParticles);
    const vector<int>& order = cc.getAtomIndex();
    if (cc.getUseDoublePrecision()) {
        vector<double> labDipoleVec;
        labDipoles.download(labDipoleVec);
        for (int i = 0; i < numParticles; i++)
            dipoles[order[i]] = Vec3(labDipoleVec[3*i], labDipoleVec[3*i+1], labDipoleVec[3*i+2]);
    }
    else {
        vector<float> labDipoleVec;
        labDipoles.download(labDipoleVec);
        for (int i = 0; i < numParticles; i++)
            dipoles[order[i]] = Vec3(labDipoleVec[3*i], labDipoleVec[3*i+1], labDipoleVec[3*i+2]);
    }
}


void CommonCalcMLPolMultipoleForceKernel::getInducedDipoles(ContextImpl& context, vector<Vec3>& dipoles) {
    ContextSelector selector(cc);
    ensureMultipolesValid(context);
    int numParticles = cc.getNumAtoms();
    dipoles.resize(numParticles);
    const vector<int>& order = cc.getAtomIndex();
    if (cc.getUseDoublePrecision()) {
        vector<double> d;
        inducedDipole.download(d);
        for (int i = 0; i < numParticles; i++)
            dipoles[order[i]] = Vec3(d[3*i], d[3*i+1], d[3*i+2]);
    }
    else {
        vector<float> d;
        inducedDipole.download(d);
        for (int i = 0; i < numParticles; i++)
            dipoles[order[i]] = Vec3(d[3*i], d[3*i+1], d[3*i+2]);
    }
}


void CommonCalcMLPolMultipoleForceKernel::getTotalDipoles(ContextImpl& context, vector<Vec3>& dipoles) {
    ContextSelector selector(cc);
    ensureMultipolesValid(context);
    int numParticles = cc.getNumAtoms();
    dipoles.resize(numParticles);
    const vector<int>& order = cc.getAtomIndex();
    if (cc.getUseDoublePrecision()) {
        vector<mm_double4> posqVec;
        vector<double> labDipoleVec;
        vector<double> inducedDipoleVec;
        double totalDipoleVecX;
        double totalDipoleVecY;
        double totalDipoleVecZ;
        inducedDipole.download(inducedDipoleVec);
        labDipoles.download(labDipoleVec);
        cc.getPosq().download(posqVec);
        for (int i = 0; i < numParticles; i++) {
            totalDipoleVecX = labDipoleVec[3*i] + inducedDipoleVec[3*i];
            totalDipoleVecY = labDipoleVec[3*i+1] + inducedDipoleVec[3*i+1];
            totalDipoleVecZ = labDipoleVec[3*i+2] + inducedDipoleVec[3*i+2];
            dipoles[order[i]] = Vec3(totalDipoleVecX, totalDipoleVecY, totalDipoleVecZ);
        }
    }
    else {
        vector<mm_float4> posqVec;
        vector<float> labDipoleVec;
        vector<float> inducedDipoleVec;
        float totalDipoleVecX;
        float totalDipoleVecY;
        float totalDipoleVecZ;
        inducedDipole.download(inducedDipoleVec);
        labDipoles.download(labDipoleVec);
        cc.getPosq().download(posqVec);
        for (int i = 0; i < numParticles; i++) {
            totalDipoleVecX = labDipoleVec[3*i] + inducedDipoleVec[3*i];
            totalDipoleVecY = labDipoleVec[3*i+1] + inducedDipoleVec[3*i+1];
            totalDipoleVecZ = labDipoleVec[3*i+2] + inducedDipoleVec[3*i+2];
            dipoles[order[i]] = Vec3(totalDipoleVecX, totalDipoleVecY, totalDipoleVecZ);
        }
    }
}

void CommonCalcMLPolMultipoleForceKernel::getElectrostaticPotential(ContextImpl& context, const vector<Vec3>& inputGrid, vector<double>& outputElectrostaticPotential) {
    ContextSelector selector(cc);
    ensureMultipolesValid(context);
    int numPoints = inputGrid.size();
    int elementSize = (cc.getUseDoublePrecision() ? sizeof(double) : sizeof(float));
    ComputeArray points, potential;
    points.initialize(cc, numPoints, 4*elementSize, "points");
    potential.initialize(cc, numPoints, elementSize, "potential");
    
    // Copy the grid points to the GPU.
    
    if (cc.getUseDoublePrecision()) {
        vector<mm_double4> p(numPoints);
        for (int i = 0; i < numPoints; i++)
            p[i] = mm_double4(inputGrid[i][0], inputGrid[i][1], inputGrid[i][2], 0);
        points.upload(p);
    }
    else {
        vector<mm_float4> p(numPoints);
        for (int i = 0; i < numPoints; i++)
            p[i] = mm_float4((float) inputGrid[i][0], (float) inputGrid[i][1], (float) inputGrid[i][2], 0);
        points.upload(p);
    }
    
    // Compute the potential.
    
    computePotentialKernel->setArg(4, points);
    computePotentialKernel->setArg(5, potential);
    computePotentialKernel->setArg(6, numPoints);
    setPeriodicBoxArgs(cc, computePotentialKernel, 7);
    computePotentialKernel->execute(numPoints, 128);
    outputElectrostaticPotential.resize(numPoints);
    if (cc.getUseDoublePrecision())
        potential.download(outputElectrostaticPotential);
    else {
        vector<float> p(numPoints);
        potential.download(p);
        for (int i = 0; i < numPoints; i++)
            outputElectrostaticPotential[i] = p[i];
    }
}

template <class T, class T4, class M4>
void CommonCalcMLPolMultipoleForceKernel::computeSystemMultipoleMoments(ContextImpl& context, vector<double>& outputMultipoleMoments) {
    // Compute the local coordinates relative to the center of mass.
    int numAtoms = cc.getNumAtoms();
    vector<T4> posq;
    vector<M4> velm;
    cc.getPosq().download(posq);
    cc.getVelm().download(velm);
    double totalMass = 0.0;
    Vec3 centerOfMass(0, 0, 0);
    for (int i = 0; i < numAtoms; i++) {
        double mass = (velm[i].w > 0 ? 1.0/velm[i].w : 0.0);
        totalMass += mass;
        centerOfMass[0] += mass*posq[i].x;
        centerOfMass[1] += mass*posq[i].y;
        centerOfMass[2] += mass*posq[i].z;
    }
    if (totalMass > 0.0) {
        centerOfMass[0] /= totalMass;
        centerOfMass[1] /= totalMass;
        centerOfMass[2] /= totalMass;
    }
    vector<mm_double4> posqLocal(numAtoms);
    for (int i = 0; i < numAtoms; i++) {
        posqLocal[i].x = posq[i].x - centerOfMass[0];
        posqLocal[i].y = posq[i].y - centerOfMass[1];
        posqLocal[i].z = posq[i].z - centerOfMass[2];
        posqLocal[i].w = posq[i].w;
    }

    // Compute the multipole moments.
    
    double totalCharge = 0.0;
    double xdpl = 0.0;
    double ydpl = 0.0;
    double zdpl = 0.0;
    double xxqdp = 0.0;
    double xyqdp = 0.0;
    double xzqdp = 0.0;
    double yxqdp = 0.0;
    double yyqdp = 0.0;
    double yzqdp = 0.0;
    double zxqdp = 0.0;
    double zyqdp = 0.0;
    double zzqdp = 0.0;
    vector<T> labDipoleVec, inducedDipoleVec, quadrupoleVec;
    labDipoles.download(labDipoleVec);
    inducedDipole.download(inducedDipoleVec);
    labQuadrupoles.download(quadrupoleVec);
    for (int i = 0; i < numAtoms; i++) {
        totalCharge += posqLocal[i].w;
        double netDipoleX = (labDipoleVec[3*i] + inducedDipoleVec[3*i]);
        double netDipoleY = (labDipoleVec[3*i+1] + inducedDipoleVec[3*i+1]);
        double netDipoleZ = (labDipoleVec[3*i+2] + inducedDipoleVec[3*i+2]);
        xdpl += posqLocal[i].x*posqLocal[i].w + netDipoleX;
        ydpl += posqLocal[i].y*posqLocal[i].w + netDipoleY;
        zdpl += posqLocal[i].z*posqLocal[i].w + netDipoleZ;
        xxqdp += posqLocal[i].x*posqLocal[i].x*posqLocal[i].w + 2*posqLocal[i].x*netDipoleX;
        xyqdp += posqLocal[i].x*posqLocal[i].y*posqLocal[i].w + posqLocal[i].x*netDipoleY + posqLocal[i].y*netDipoleX;
        xzqdp += posqLocal[i].x*posqLocal[i].z*posqLocal[i].w + posqLocal[i].x*netDipoleZ + posqLocal[i].z*netDipoleX;
        yxqdp += posqLocal[i].y*posqLocal[i].x*posqLocal[i].w + posqLocal[i].y*netDipoleX + posqLocal[i].x*netDipoleY;
        yyqdp += posqLocal[i].y*posqLocal[i].y*posqLocal[i].w + 2*posqLocal[i].y*netDipoleY;
        yzqdp += posqLocal[i].y*posqLocal[i].z*posqLocal[i].w + posqLocal[i].y*netDipoleZ + posqLocal[i].z*netDipoleY;
        zxqdp += posqLocal[i].z*posqLocal[i].x*posqLocal[i].w + posqLocal[i].z*netDipoleX + posqLocal[i].x*netDipoleZ;
        zyqdp += posqLocal[i].z*posqLocal[i].y*posqLocal[i].w + posqLocal[i].z*netDipoleY + posqLocal[i].y*netDipoleZ;
        zzqdp += posqLocal[i].z*posqLocal[i].z*posqLocal[i].w + 2*posqLocal[i].z*netDipoleZ;
    }

    // Convert the quadrupole from traced to traceless form.
 
    double qave = (xxqdp + yyqdp + zzqdp)/3;
    xxqdp = 1.5*(xxqdp-qave);
    xyqdp = 1.5*xyqdp;
    xzqdp = 1.5*xzqdp;
    yxqdp = 1.5*yxqdp;
    yyqdp = 1.5*(yyqdp-qave);
    yzqdp = 1.5*yzqdp;
    zxqdp = 1.5*zxqdp;
    zyqdp = 1.5*zyqdp;
    zzqdp = 1.5*(zzqdp-qave);

    // Add the traceless atomic quadrupoles to the total quadrupole moment.

    for (int i = 0; i < numAtoms; i++) {
        xxqdp = xxqdp + 3*quadrupoleVec[5*i];
        xyqdp = xyqdp + 3*quadrupoleVec[5*i+1];
        xzqdp = xzqdp + 3*quadrupoleVec[5*i+2];
        yxqdp = yxqdp + 3*quadrupoleVec[5*i+1];
        yyqdp = yyqdp + 3*quadrupoleVec[5*i+3];
        yzqdp = yzqdp + 3*quadrupoleVec[5*i+4];
        zxqdp = zxqdp + 3*quadrupoleVec[5*i+2];
        zyqdp = zyqdp + 3*quadrupoleVec[5*i+4];
        zzqdp = zzqdp + -3*(quadrupoleVec[5*i]+quadrupoleVec[5*i+3]);
    }
 
    double debye = 4.80321;
    outputMultipoleMoments.resize(13);
    outputMultipoleMoments[0] = totalCharge;
    outputMultipoleMoments[1] = 10.0*xdpl*debye;
    outputMultipoleMoments[2] = 10.0*ydpl*debye;
    outputMultipoleMoments[3] = 10.0*zdpl*debye;
    outputMultipoleMoments[4] = 100.0*xxqdp*debye;
    outputMultipoleMoments[5] = 100.0*xyqdp*debye;
    outputMultipoleMoments[6] = 100.0*xzqdp*debye;
    outputMultipoleMoments[7] = 100.0*yxqdp*debye;
    outputMultipoleMoments[8] = 100.0*yyqdp*debye;
    outputMultipoleMoments[9] = 100.0*yzqdp*debye;
    outputMultipoleMoments[10] = 100.0*zxqdp*debye;
    outputMultipoleMoments[11] = 100.0*zyqdp*debye;
    outputMultipoleMoments[12] = 100.0*zzqdp*debye;
}


void CommonCalcMLPolMultipoleForceKernel::getSystemMultipoleMoments(ContextImpl& context, vector<double>& outputMultipoleMoments) {
    ContextSelector selector(cc);
    ensureMultipolesValid(context);
    if (cc.getUseDoublePrecision())
        computeSystemMultipoleMoments<double, mm_double4, mm_double4>(context, outputMultipoleMoments);
    else if (cc.getUseMixedPrecision())
        computeSystemMultipoleMoments<float, mm_float4, mm_double4>(context, outputMultipoleMoments);
    else
        computeSystemMultipoleMoments<float, mm_float4, mm_float4>(context, outputMultipoleMoments);
}

void CommonCalcMLPolMultipoleForceKernel::copyParametersToContext(ContextImpl& context, const AmoebaMultipoleForce& force) {
    // Make sure the new parameters are acceptable.
    
    ContextSelector selector(cc);
    if (force.getNumMultipoles() != cc.getNumAtoms())
        throw OpenMMException("updateParametersInContext: The number of multipoles has changed");
    
    // Record the per-multipole parameters.
    
    cc.getPosq().download(cc.getPinnedBuffer());
    mm_float4* posqf = (mm_float4*) cc.getPinnedBuffer();
    mm_double4* posqd = (mm_double4*) cc.getPinnedBuffer();
    vector<mm_float2> dampingAndTholeVec;
    vector<float> polarizabilityVec;
    vector<float> localDipolesVec;
    vector<float> localQuadrupolesVec;
    vector<mm_int4> multipoleParticlesVec;
    totalCharge = 0.0;
    for (int i = 0; i < force.getNumMultipoles(); i++) {
        double charge, thole, damping, polarity;
        int axisType, atomX, atomY, atomZ;
        vector<double> dipole, quadrupole;
        force.getMultipoleParameters(i, charge, dipole, quadrupole, axisType, atomZ, atomX, atomY, thole, damping, polarity);
        totalCharge += charge;
        if (cc.getUseDoublePrecision())
            posqd[i].w = charge;
        else
            posqf[i].w = (float) charge;
        dampingAndTholeVec.push_back(mm_float2((float) damping, (float) (isML[i] ? -thole : thole)));   // POLEN
        polarizabilityVec.push_back((float) polarity);
        multipoleParticlesVec.push_back(mm_int4(atomX, atomY, atomZ, axisType));
        for (int j = 0; j < 3; j++)
            localDipolesVec.push_back((float) dipole[j]);
        localQuadrupolesVec.push_back((float) quadrupole[0]);
        localQuadrupolesVec.push_back((float) quadrupole[1]);
        localQuadrupolesVec.push_back((float) quadrupole[2]);
        localQuadrupolesVec.push_back((float) quadrupole[4]);
        localQuadrupolesVec.push_back((float) quadrupole[5]);
    }
    mlCharges = mlChargesOf(force, mlAtoms);   // POLEN: back to the force parameters
    if (!hasQuadrupoles) {
        for (auto q : localQuadrupolesVec)
            if (q != 0.0)
                throw OpenMMException("updateParametersInContext: Cannot set a non-zero quadrupole moment, because quadrupoles were excluded from the kernel");
    }
    for (int i = force.getNumMultipoles(); i < cc.getPaddedNumAtoms(); i++) {
        dampingAndTholeVec.push_back(mm_float2(0, 0));
        polarizabilityVec.push_back(0);
        multipoleParticlesVec.push_back(mm_int4(0, 0, 0, 0));
        for (int j = 0; j < 3; j++)
            localDipolesVec.push_back(0);
        for (int j = 0; j < 5; j++)
            localQuadrupolesVec.push_back(0);
    }
    dampingAndThole.upload(dampingAndTholeVec);
    polarizability.upload(polarizabilityVec);
    multipoleParticles.upload(multipoleParticlesVec);
    localDipoles.upload(localDipolesVec);
    localQuadrupoles.upload(localQuadrupolesVec);
    cc.getPosq().upload(cc.getPinnedBuffer());
    cc.invalidateMolecules();
    multipolesAreValid = false;
}

void CommonCalcMLPolMultipoleForceKernel::getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const {
    if (!usePME)
        throw OpenMMException("getPMEParametersInContext: This Context is not using PME");
    alpha = pmeAlpha;
    nx = gridSizeX;
    ny = gridSizeY;
    nz = gridSizeZ;
}

// ---------------------------------------------------------------------------------------------------------
// POLEN
// ---------------------------------------------------------------------------------------------------------

void CommonCalcMLPolMultipoleForceKernel::readMLRegion(const AmoebaMultipoleForce& force) {
    mlAtoms.clear();
    layered = false;
    isML.assign(force.getNumMultipoles(), 0);
    const MLPolMultipoleForce* f = dynamic_cast<const MLPolMultipoleForce*>(&force);
    if (f == NULL)
        return;
    layered = f->usesLayeredPolarization();
    if (layered && force.getPolarizationType() != AmoebaMultipoleForce::Mutual)
        throw OpenMMException("MLPolMultipoleForce: layered polarization requires mutual polarization");
    mlAtoms = f->getMLAtoms();
    mlHeadFile = f->getMLHead();
    hasHead = false;
    if (!mlHeadFile.empty()) {
        if (mlAtoms.empty())
            throw OpenMMException("MLPolMultipoleForce: ML head set without an ML region (setMLRegion)");
        // the head runs on the same CUDA device as the context (latency ~3-5 ms, nearly independent of the ML-region
        // size; on the CPU the cost grows quickly with the number of atoms), unless MLPOL_HEAD_DEVICE=cpu
        const char* env = std::getenv("MLPOL_HEAD_DEVICE");
        bool wantCPU = (env != NULL && std::string(env) == "cpu");
        CudaContext* cu = dynamic_cast<CudaContext*>(&cc);
        headDevice = torch::kCPU;
        if (!wantCPU && cu != NULL && torch::cuda::is_available())
            headDevice = torch::Device(torch::kCUDA, cu->getDeviceIndex());
        try {
            // PyTorch uses the device primary context and makes it current when initializing; OpenMM uses its own
            // context: save the current one, load, grab the primary one and restore (as in openmm-torch)
            CUcontext before = NULL;
            if (headDevice.is_cuda())
                cuCtxGetCurrent(&before);
            mlHead = torch::jit::load(mlHeadFile, headDevice);
            if (headDevice.is_cuda()) {
                CUcontext primary;
                cuDevicePrimaryCtxRetain(&primary, cu->getDevice());
                torchContext = primary;
                cuCtxSetCurrent(before);
            }
        }
        catch (const std::exception& e) {
            throw OpenMMException("MLPolMultipoleForce: could not load the ML head '" + mlHeadFile + "': " + e.what());
        }
        mlHead.eval();
        hasHead = true;
    }
    mlThole = f->getMLThole();
    crossThole = f->getCrossThole();
    for (int a : mlAtoms)
        isML[a] = 1;
    mlCharges = mlChargesOf(force, mlAtoms);
}

void CommonCalcMLPolMultipoleForceKernel::setMLParameters(ContextImpl& context, const vector<double>& q, const vector<double>& mu,
                                                          const vector<double>& theta, const vector<double>& alpha) {
    ContextSelector selector(cc);
    int M = mlAtoms.size();
    if (M == 0)
        throw OpenMMException("MLPolMultipoleForce: setMLParametersInContext without an ML region (setMLRegion)");
    if ((int) q.size() != M || (int) mu.size() != 3*M || (int) theta.size() != 9*M || (int) alpha.size() != M)
        throw OpenMMException("MLPolMultipoleForce: wrong sizes in setMLParametersInContext");
    const vector<int>& order = cc.getAtomIndex();
    for (int a : mlAtoms)
        if (order[a] != a)
            throw OpenMMException("MLPolMultipoleForce (CUDA): ML atom reordered on the GPU");
    vector<double> p(11*M);
    for (int k = 0; k < M; k++) {
        totalCharge += q[k] - mlCharges[k];
        mlCharges[k] = q[k];
        p[11*k] = q[k];
        for (int a = 0; a < 3; a++)
            p[11*k+1+a] = mu[3*k+a];
        // symmetric traceless part (as in Reference)
        const double* t = &theta[9*k];
        double tr = (t[0] + t[4] + t[8])/3.0;
        p[11*k+4] = t[0] - tr;
        p[11*k+5] = 0.5*(t[1] + t[3]);
        p[11*k+6] = 0.5*(t[2] + t[6]);
        p[11*k+7] = t[4] - tr;
        p[11*k+8] = 0.5*(t[5] + t[7]);
        p[11*k+9] = alpha[k];
        p[11*k+10] = pow(alpha[k], 1.0/6.0);
    }
    if (cc.getUseDoublePrecision())
        mlParams.upload(p);
    else {
        vector<float> pf(p.begin(), p.end());
        mlParams.upload(pf);
    }
    injectMLKernel->execute(M);
    multipolesAreValid = false;
}

void CommonCalcMLPolMultipoleForceKernel::setMLPenetration(ContextImpl& context, const vector<double>& N, const vector<double>& sv) {
    ContextSelector selector(cc);
    int M = mlAtoms.size();
    if (N.empty() && sv.empty()) {                  // empty lists disable penetration
        penN.clear();
        penS.clear();
        multipolesAreValid = false;
        return;
    }
    if (M == 0 || (int) N.size() != M || (int) sv.size() != M)
        throw OpenMMException("MLPolMultipoleForce: penetration requires N and s for every ML atom");
    penN = N;
    penS = sv;
    vector<double> p(2*M);
    for (int k = 0; k < M; k++) {
        p[2*k] = N[k];
        p[2*k+1] = sv[k];
    }
    if (cc.getUseDoublePrecision())
        penParams.upload(p);
    else {
        vector<float> pf(p.begin(), p.end());
        penParams.upload(pf);
    }
    multipolesAreValid = false;
}

void CommonCalcMLPolMultipoleForceKernel::computeMLDerivatives(vector<double>& out) {
    // uses the converged state of the last evaluation (induced dipoles, lab-frame multipoles, PME potentials)
    int M = mlAtoms.size();
    if (usePME) {
        setPeriodicBoxArgs(cc, mlDerivativesKernel, 12);
        mlDerivativesKernel->setArg(17, pmePhi);
        mlDerivativesKernel->setArg(18, pmePhidp);
        Vec3 a, b, c;
        cc.getPeriodicBoxVectors(a, b, c);
        mm_double4 recip[3];
        cc.computeReciprocalBoxVectors(recip);
        double plasma = -M_PI*totalCharge/(a[0]*b[1]*c[2]*pmeAlpha*pmeAlpha);
        if (cc.getUseDoublePrecision()) {
            for (int i = 0; i < 3; i++)
                mlDerivativesKernel->setArg(19+i, recip[i]);
            mlDerivativesKernel->setArg(22, plasma);
        }
        else {
            for (int i = 0; i < 3; i++)
                mlDerivativesKernel->setArg(19+i, mm_float4((float) recip[i].x, (float) recip[i].y, (float) recip[i].z, 0));
            mlDerivativesKernel->setArg(22, (float) plasma);
        }
    }
    mlDerivativesKernel->execute(M*MLThreads, MLThreads);
    if (mlDerivatives.getElementSize() == sizeof(double))
        mlDerivatives.download(out);
    else {
        vector<float> f;
        mlDerivatives.download(f);
        out.assign(f.begin(), f.end());
    }
}

void CommonCalcMLPolMultipoleForceKernel::getMLDerivatives(ContextImpl& context, vector<double>& phi, vector<double>& field,
                                                           vector<double>& dTheta, vector<double>& dAlpha) {
    ContextSelector selector(cc);
    if (mlAtoms.empty())
        throw OpenMMException("MLPolMultipoleForce: getMLDerivatives without an ML region (setMLRegion)");
    ensureMultipolesValid(context);
    vector<double> out;
    computeMLDerivatives(out);
    int M = mlAtoms.size();
    phi.resize(M);
    field.resize(3*M);
    dTheta.resize(9*M);
    dAlpha.resize(M);
    for (int k = 0; k < M; k++) {
        phi[k] = out[14*k];
        for (int a = 0; a < 3; a++)
            field[3*k+a] = out[14*k+1+a];
        for (int a = 0; a < 9; a++)
            dTheta[9*k+a] = out[14*k+4+a];
        dAlpha[k] = out[14*k+13];
    }
}

double CommonCalcMLPolMultipoleForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    if (!hasHead)
        return executeAmoeba(context, includeForces, includeEnergy);
    ContextSelector selector(cc);

    // ML head -> parameters of the ML atoms (autograd graph back to the ML positions)
    int M = mlAtoms.size();
    torch::Tensor pos = torch::empty({M, 3}, torch::kFloat64);
    {
        gatherMLKernel->execute(M);
        auto acc = pos.accessor<double, 2>();
        if (cc.getUseDoublePrecision()) {
            vector<mm_double4> p;
            mlPos.download(p);
            for (int k = 0; k < M; k++) {
                acc[k][0] = p[k].x;
                acc[k][1] = p[k].y;
                acc[k][2] = p[k].z;
            }
        }
        else {
            vector<mm_float4> p;
            mlPos.download(p);
            for (int k = 0; k < M; k++) {
                acc[k][0] = p[k].x;
                acc[k][1] = p[k].y;
                acc[k][2] = p[k].z;
            }
        }
    }
    // PyTorch operations in its primary context (pushed only while the head runs)
    struct TorchCtx {
        bool on;
        TorchCtx(void* c) : on(c != NULL) { if (on) cuCtxPushCurrent((CUcontext) c); }
        ~TorchCtx() { if (on) { CUcontext dummy; cuCtxPopCurrent(&dummy); } }
    };
    torch::Tensor posDev;
    c10::intrusive_ptr<c10::ivalue::Tuple> out;
    {
        TorchCtx tc(torchContext);
        posDev = pos.to(headDevice).detach();
        posDev.set_requires_grad(true);
        out = mlHead.forward({posDev}).toTuple();
    }
    vector<torch::Tensor> outs;
    vector<vector<double> > outv;
    {
        TorchCtx tc(torchContext);
        for (int n = 0; n < (int) out->elements().size() && n < 6; n++)
            outs.push_back(out->elements()[n].toTensor().to(torch::kFloat64));
        for (auto& t : outs) {
            torch::Tensor c = t.detach().to(torch::kCPU).contiguous().reshape({-1});
            outv.push_back(vector<double>(c.data_ptr<double>(), c.data_ptr<double>() + c.numel()));
        }
    }
    setMLParameters(context, outv[0], outv[1], outv[2], outv[3]);
    bool pen = (outs.size() >= 6);
    if (pen)
        setMLPenetration(context, outv[4], outv[5]);

    double energy = executeAmoeba(context, includeForces, includeEnergy);

    // chain rule through the head: F_k -= sum_p (dE/dp)(dp/dR_k)
    if (includeForces) {
        vector<double> d;
        computeMLDerivatives(d);
        vector<double> fv;
        {
        TorchCtx tc(torchContext);
        torch::Tensor D = torch::tensor(d, torch::kFloat64).reshape({M, 14});
        vector<torch::Tensor> gouts = {D.index({torch::indexing::Slice(), 0}),
                                       -D.index({torch::indexing::Slice(), torch::indexing::Slice(1, 4)}),
                                       D.index({torch::indexing::Slice(), torch::indexing::Slice(4, 13)}).reshape({M, 3, 3}),
                                       D.index({torch::indexing::Slice(), 13})};
        vector<torch::Tensor> heads(outs.begin(), outs.begin()+4);
        if (pen && !penDN.empty()) {
            heads.push_back(outs[4]);
            heads.push_back(outs[5]);
            gouts.push_back(torch::tensor(penDN, torch::kFloat64));
            gouts.push_back(torch::tensor(penDS, torch::kFloat64));
        }
        for (int n = 0; n < (int) heads.size(); n++)
            gouts[n] = gouts[n].reshape(heads[n].sizes()).to(headDevice);
        vector<torch::Tensor> grads = torch::autograd::grad(heads, {posDev}, gouts, false, false, true);
        if (grads[0].defined()) {
            torch::Tensor g = -grads[0].to(torch::kCPU).contiguous();
            fv.assign(g.data_ptr<double>(), g.data_ptr<double>() + 3*M);
        }
        }                                           // end of the PyTorch context: the rest is OpenMM
        if (!fv.empty()) {
            if (cc.getUseDoublePrecision())
                mlForce.upload(fv);
            else {
                vector<float> ff(fv.begin(), fv.end());
                mlForce.upload(ff);
            }
            addMLForcesKernel->execute(M);
        }
    }
    return energy;
}
