/* Reference kernel of MLPolMultipoleForce. Based on the OpenMM 8.5.2 AMOEBA kernel (MIT license), using the
 * classes isolated in OpenMM::MLPolRef. No GK support. */

#include "MLPolReferenceKernels.h"
#include <torch/csrc/autograd/autograd.h>
#include "ReferencePlatform.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/internal/AmoebaMultipoleForceImpl.h"
#include "openmm/internal/AmoebaGeneralizedKirkwoodForceImpl.h"
#include "openmm/NonbondedForce.h"
#include "openmm/internal/NonbondedForceImpl.h"
#include "openmm/OpenMMException.h"
#include <cmath>

using namespace MLPol;
using namespace OpenMM;
using namespace OpenMM::MLPolRef;
using namespace std;

static vector<Vec3>& extractPositions(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return *data->positions;
}

static vector<Vec3>& extractForces(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return *data->forces;
}

static Vec3* extractBoxVectors(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return data->periodicBoxVectors;
}

ReferenceCalcMLPolMultipoleForceKernel::ReferenceCalcMLPolMultipoleForceKernel(const std::string& name, const Platform& platform, const System& system) :
         CalcMLPolMultipoleForceKernel(name, platform), system(system), numMultipoles(0), mutualInducedMaxIterations(60), mutualInducedTargetEpsilon(1.0e-03),
                                                         usePme(false),alphaEwald(0.0), cutoffDistance(1.0) {  

}

ReferenceCalcMLPolMultipoleForceKernel::~ReferenceCalcMLPolMultipoleForceKernel() {
}

void ReferenceCalcMLPolMultipoleForceKernel::readLayered(const AmoebaMultipoleForce& force) {
    layered = OpenMM::MLPolRef::AmoebaReferenceMultipoleForce::LayeredPolarization();
    const MLPolMultipoleForce* f = dynamic_cast<const MLPolMultipoleForce*>(&force);
    mlAtoms.clear();
    if (f != NULL) {
        mlAtoms = f->getMLAtoms();
        mlThole = f->getMLThole();
        crossThole = f->getCrossThole();
        loadHead(f->getMLHead());
    }
    if (f == NULL || !f->usesLayeredPolarization())
        return;
    layered.enabled = true;
    f->getLayeredPolarization(layered.centerAtoms, layered.groupOf, layered.groupRefAtom, layered.rIn, layered.rOut, layered.tau);
}

void ReferenceCalcMLPolMultipoleForceKernel::initialize(const System& system, const AmoebaMultipoleForce& force) {
    readLayered(force);

    numMultipoles   = force.getNumMultipoles();

    charges.resize(numMultipoles);
    dipoles.resize(3*numMultipoles);
    quadrupoles.resize(9*numMultipoles);
    tholes.resize(numMultipoles);
    dampingFactors.resize(numMultipoles);
    polarity.resize(numMultipoles);
    axisTypes.resize(numMultipoles);
    multipoleAtomZs.resize(numMultipoles);
    multipoleAtomXs.resize(numMultipoles);
    multipoleAtomYs.resize(numMultipoles);
    multipoleAtomCovalentInfo.resize(numMultipoles);

    int dipoleIndex      = 0;
    int quadrupoleIndex  = 0;
    double totalCharge   = 0.0;
    for (int ii = 0; ii < numMultipoles; ii++) {

        // multipoles

        int axisType, multipoleAtomZ, multipoleAtomX, multipoleAtomY;
        double charge, tholeD, dampingFactorD, polarityD;
        std::vector<double> dipolesD;
        std::vector<double> quadrupolesD;
        force.getMultipoleParameters(ii, charge, dipolesD, quadrupolesD, axisType, multipoleAtomZ, multipoleAtomX, multipoleAtomY,
                                     tholeD, dampingFactorD, polarityD);

        totalCharge                       += charge;
        axisTypes[ii]                      = axisType;
        multipoleAtomZs[ii]                = multipoleAtomZ;
        multipoleAtomXs[ii]                = multipoleAtomX;
        multipoleAtomYs[ii]                = multipoleAtomY;

        charges[ii]                        = charge;
        tholes[ii]                         = tholeD;
        dampingFactors[ii]                 = dampingFactorD;
        polarity[ii]                       = polarityD;

        dipoles[dipoleIndex++]             = dipolesD[0];
        dipoles[dipoleIndex++]             = dipolesD[1];
        dipoles[dipoleIndex++]             = dipolesD[2];
        
        quadrupoles[quadrupoleIndex++]     = quadrupolesD[0];
        quadrupoles[quadrupoleIndex++]     = quadrupolesD[1];
        quadrupoles[quadrupoleIndex++]     = quadrupolesD[2];
        quadrupoles[quadrupoleIndex++]     = quadrupolesD[3];
        quadrupoles[quadrupoleIndex++]     = quadrupolesD[4];
        quadrupoles[quadrupoleIndex++]     = quadrupolesD[5];
        quadrupoles[quadrupoleIndex++]     = quadrupolesD[6];
        quadrupoles[quadrupoleIndex++]     = quadrupolesD[7];
        quadrupoles[quadrupoleIndex++]     = quadrupolesD[8];

        // covalent info

        std::vector< std::vector<int> > covalentLists;
        force.getCovalentMaps(ii, covalentLists);
        multipoleAtomCovalentInfo[ii] = covalentLists;

    }

    polarizationType = force.getPolarizationType();
    if (polarizationType == AmoebaMultipoleForce::Mutual) {
        mutualInducedMaxIterations = force.getMutualInducedMaxIterations();
        mutualInducedTargetEpsilon = force.getMutualInducedTargetEpsilon();
    }
    else if (polarizationType == AmoebaMultipoleForce::Extrapolated) {
        extrapolationCoefficients = force.getExtrapolationCoefficients();
    }

    // PME

    nonbondedMethod  = force.getNonbondedMethod();
    if (nonbondedMethod == AmoebaMultipoleForce::PME) {
        usePme     = true;
        pmeGridDimension.resize(3);
        force.getPMEParameters(alphaEwald, pmeGridDimension[0], pmeGridDimension[1], pmeGridDimension[2]);
        cutoffDistance = force.getCutoffDistance();
        if (pmeGridDimension[0] == 0 || alphaEwald == 0.0) {
            NonbondedForce nb;
            nb.setEwaldErrorTolerance(force.getEwaldErrorTolerance());
            nb.setCutoffDistance(force.getCutoffDistance());
            int gridSizeX, gridSizeY, gridSizeZ;
            NonbondedForceImpl::calcPMEParameters(system, nb, alphaEwald, gridSizeX, gridSizeY, gridSizeZ, false);
            pmeGridDimension[0] = gridSizeX;
            pmeGridDimension[1] = gridSizeY;
            pmeGridDimension[2] = gridSizeZ;
        }    
    }
    else {
        usePme = false;
    }
    return;
}

AmoebaReferenceMultipoleForce* ReferenceCalcMLPolMultipoleForceKernel::setupAmoebaReferenceMultipoleForce(ContextImpl& context)
{

    // amoebaReferenceMultipoleForce is set to AmoebaReferenceGeneralizedKirkwoodForce if AmoebaGeneralizedKirkwoodForce is present
    // amoebaReferenceMultipoleForce is set to AmoebaReferencePmeMultipoleForce if 'usePme' is set
    // amoebaReferenceMultipoleForce is set to AmoebaReferenceMultipoleForce otherwise

    // check if AmoebaGeneralizedKirkwoodForce is present 

    for (unsigned int ii = 0; ii < context.getForceImpls().size(); ii++)
        if (dynamic_cast<AmoebaGeneralizedKirkwoodForceImpl*>(context.getForceImpls()[ii]) != NULL)
            throw OpenMMException("MLPolMultipoleForce: AmoebaGeneralizedKirkwoodForce is not supported.");

    AmoebaReferenceMultipoleForce* amoebaReferenceMultipoleForce = NULL;
    if (usePme) {

        AmoebaReferencePmeMultipoleForce* amoebaReferencePmeMultipoleForce = new AmoebaReferencePmeMultipoleForce();
        amoebaReferencePmeMultipoleForce->setAlphaEwald(alphaEwald);
        amoebaReferencePmeMultipoleForce->setCutoffDistance(cutoffDistance);
        amoebaReferencePmeMultipoleForce->setPmeGridDimensions(pmeGridDimension);
        Vec3* boxVectors = extractBoxVectors(context);
        double minAllowedSize = 1.999999*cutoffDistance;
        if (boxVectors[0][0] < minAllowedSize || boxVectors[1][1] < minAllowedSize || boxVectors[2][2] < minAllowedSize) {
            throw OpenMMException("The periodic box size has decreased to less than twice the nonbonded cutoff.");
        }
        amoebaReferencePmeMultipoleForce->setPeriodicBoxSize(boxVectors);
        amoebaReferenceMultipoleForce = static_cast<AmoebaReferenceMultipoleForce*>(amoebaReferencePmeMultipoleForce);

    }
    else {
         amoebaReferenceMultipoleForce = new AmoebaReferenceMultipoleForce(AmoebaReferenceMultipoleForce::NoCutoff);
    }

    // set polarization type

    if (polarizationType == AmoebaMultipoleForce::Mutual) {
        amoebaReferenceMultipoleForce->setPolarizationType(AmoebaReferenceMultipoleForce::Mutual);
        amoebaReferenceMultipoleForce->setMutualInducedDipoleTargetEpsilon(mutualInducedTargetEpsilon);
        amoebaReferenceMultipoleForce->setMaximumMutualInducedDipoleIterations(mutualInducedMaxIterations);
    }
    else if (polarizationType == AmoebaMultipoleForce::Direct) {
        amoebaReferenceMultipoleForce->setPolarizationType(AmoebaReferenceMultipoleForce::Direct);
    }
    else if (polarizationType == AmoebaMultipoleForce::Extrapolated) {
        amoebaReferenceMultipoleForce->setPolarizationType(AmoebaReferenceMultipoleForce::Extrapolated);
        amoebaReferenceMultipoleForce->setExtrapolationCoefficients(extrapolationCoefficients);
    }
    else {
        throw OpenMMException("Polarization type not recognzied.");
    }

    if (!mlAtoms.empty()) {
        amoebaReferenceMultipoleForce->setMLRegion(mlAtoms, crossThole);
        if (!penN.empty())
            amoebaReferenceMultipoleForce->setPenetration(penN, penS);
    }
    if (layered.enabled) {
        if (polarizationType != AmoebaMultipoleForce::Mutual) {
            delete amoebaReferenceMultipoleForce;
            throw OpenMMException("MLPolMultipoleForce: layered polarization requires mutual polarization");
        }
        amoebaReferenceMultipoleForce->setLayeredPolarization(layered);
    }
    return amoebaReferenceMultipoleForce;

}

double ReferenceCalcMLPolMultipoleForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {

    vector<Vec3>& posData = extractPositions(context);
    vector<Vec3>& forceData = extractForces(context);

    // POLEN: ML head -> parameters of the ML atoms (with the autograd graph back to the ML positions)
    int M = mlAtoms.size();
    torch::Tensor pos, headQ, headMu, headTheta, headAlpha, headN, headS;
    if (hasHead) {
        pos = torch::empty({M, 3}, torch::kFloat64);
        auto acc = pos.accessor<double, 2>();
        for (int k = 0; k < M; k++)
            for (int a = 0; a < 3; a++)
                acc[k][a] = posData[mlAtoms[k]][a];
        pos.set_requires_grad(true);
        auto out = mlHead.forward({pos}).toTuple();
        headQ = out->elements()[0].toTensor().to(torch::kFloat64);
        headMu = out->elements()[1].toTensor().to(torch::kFloat64);
        headTheta = out->elements()[2].toTensor().to(torch::kFloat64);
        headAlpha = out->elements()[3].toTensor().to(torch::kFloat64);
        if (out->elements().size() >= 6) {       // head with penetration: (..., N, s)
            headN = out->elements()[4].toTensor().to(torch::kFloat64);
            headS = out->elements()[5].toTensor().to(torch::kFloat64);
        }
        auto toVec = [](const torch::Tensor& t) {
            torch::Tensor c = t.detach().contiguous().reshape({-1});
            return vector<double>(c.data_ptr<double>(), c.data_ptr<double>() + c.numel());
        };
        setMLParameters(context, toVec(headQ), toVec(headMu), toVec(headTheta), toVec(headAlpha));
        if (headN.defined())
            setMLPenetration(context, toVec(headN), toVec(headS));
    }

    AmoebaReferenceMultipoleForce* amoebaReferenceMultipoleForce = setupAmoebaReferenceMultipoleForce(context);
    bool chain = hasHead && includeForces;
    amoebaReferenceMultipoleForce->setComputeMLDerivatives(chain);   // derivatives from the same SCF as the energy
    double energy = amoebaReferenceMultipoleForce->calculateForceAndEnergy(posData, charges, dipoles, quadrupoles, tholes,
                                                                           dampingFactors, polarity, axisTypes, 
                                                                           multipoleAtomZs, multipoleAtomXs, multipoleAtomYs,
                                                                           multipoleAtomCovalentInfo, forceData);
    vector<double> penDN, penDS;
    amoebaReferenceMultipoleForce->getPenetrationDerivatives(penDN, penDS);
    vector<double> phi, field, dTheta, dAlpha;
    if (chain) {
        vector<Vec3> fieldVec;
        amoebaReferenceMultipoleForce->getStoredMLDerivatives(phi, fieldVec, dTheta, dAlpha);
        field.resize(3*fieldVec.size());
        for (unsigned int k = 0; k < fieldVec.size(); k++)
            for (int a = 0; a < 3; a++)
                field[3*k+a] = fieldVec[k][a];
    }

    delete amoebaReferenceMultipoleForce;

    // chain rule through the head: F_k -= sum_p (dE/dp)(dp/dR_k), with dE/dp from the converged quantities
    if (chain) {
        auto fromVec = [](const vector<double>& v, torch::IntArrayRef shape, double sign) {
            return torch::tensor(v, torch::kFloat64).reshape(shape)*sign;
        };
        vector<torch::Tensor> outs = {headQ, headMu, headTheta, headAlpha};
        vector<torch::Tensor> gouts = {fromVec(phi, {M}, 1.0), fromVec(field, {M, 3}, -1.0), fromVec(dTheta, {M, 3, 3}, 1.0),
                                       fromVec(dAlpha, {M}, 1.0)};
        if (headN.defined() && !penDN.empty()) {
            outs.push_back(headN);
            outs.push_back(headS);
            gouts.push_back(fromVec(penDN, {M}, 1.0));
            gouts.push_back(fromVec(penDS, {M}, 1.0));
        }
        vector<torch::Tensor> grads = torch::autograd::grad(outs, {pos}, gouts, false, false, true);
        if (grads[0].defined()) {
            auto g = grads[0].accessor<double, 2>();
            for (int k = 0; k < M; k++)
                for (int a = 0; a < 3; a++)
                    forceData[mlAtoms[k]][a] -= g[k][a];
        }
    }

    return static_cast<double>(energy);
}

void ReferenceCalcMLPolMultipoleForceKernel::loadHead(const std::string& file) {
    if (file == mlHeadFile && (hasHead || file.empty()))
        return;
    mlHeadFile = file;
    hasHead = false;
    if (file.empty())
        return;
    if (mlAtoms.empty())
        throw OpenMMException("MLPolMultipoleForce: ML head set without an ML region (setMLRegion)");
    try {
        mlHead = torch::jit::load(file, torch::kCPU);
    }
    catch (const std::exception& e) {
        throw OpenMMException("MLPolMultipoleForce: could not load the ML head '" + file + "': " + e.what());
    }
    mlHead.eval();
    hasHead = true;
}

void ReferenceCalcMLPolMultipoleForceKernel::getInducedDipoles(ContextImpl& context, vector<Vec3>& outputDipoles) {
    int numParticles = context.getSystem().getNumParticles();
    outputDipoles.resize(numParticles);

    // Create an AmoebaReferenceMultipoleForce to do the calculation.
    
    AmoebaReferenceMultipoleForce* amoebaReferenceMultipoleForce = setupAmoebaReferenceMultipoleForce(context);
    vector<Vec3>& posData = extractPositions(context);
    
    // Retrieve the induced dipoles.
    
    vector<Vec3> inducedDipoles;
    amoebaReferenceMultipoleForce->calculateInducedDipoles(posData, charges, dipoles, quadrupoles, tholes,
            dampingFactors, polarity, axisTypes, multipoleAtomZs, multipoleAtomXs, multipoleAtomYs, multipoleAtomCovalentInfo, inducedDipoles);
    for (int i = 0; i < numParticles; i++)
        outputDipoles[i] = inducedDipoles[i];
    delete amoebaReferenceMultipoleForce;
}

void ReferenceCalcMLPolMultipoleForceKernel::getLabFramePermanentDipoles(ContextImpl& context, vector<Vec3>& outputDipoles) {
    int numParticles = context.getSystem().getNumParticles();
    outputDipoles.resize(numParticles);

    // Create an AmoebaReferenceMultipoleForce to do the calculation.
    
    AmoebaReferenceMultipoleForce* amoebaReferenceMultipoleForce = setupAmoebaReferenceMultipoleForce(context);
    vector<Vec3>& posData = extractPositions(context);
    
    // Retrieve the permanent dipoles in the lab frame.
    
    vector<Vec3> labFramePermanentDipoles;
    amoebaReferenceMultipoleForce->calculateLabFramePermanentDipoles(posData, charges, dipoles, quadrupoles, tholes, 
            dampingFactors, polarity, axisTypes, multipoleAtomZs, multipoleAtomXs, multipoleAtomYs, multipoleAtomCovalentInfo, labFramePermanentDipoles);
    for (int i = 0; i < numParticles; i++)
        outputDipoles[i] = labFramePermanentDipoles[i];
    delete amoebaReferenceMultipoleForce;
}


void ReferenceCalcMLPolMultipoleForceKernel::getTotalDipoles(ContextImpl& context, vector<Vec3>& outputDipoles) {
    int numParticles = context.getSystem().getNumParticles();
    outputDipoles.resize(numParticles);

    // Create an AmoebaReferenceMultipoleForce to do the calculation.
    
    AmoebaReferenceMultipoleForce* amoebaReferenceMultipoleForce = setupAmoebaReferenceMultipoleForce(context);
    vector<Vec3>& posData = extractPositions(context);
    
    // Retrieve the permanent dipoles in the lab frame.
    
    vector<Vec3> totalDipoles;
    amoebaReferenceMultipoleForce->calculateTotalDipoles(posData, charges, dipoles, quadrupoles, tholes,
            dampingFactors, polarity, axisTypes, multipoleAtomZs, multipoleAtomXs, multipoleAtomYs, multipoleAtomCovalentInfo, totalDipoles);

    for (int i = 0; i < numParticles; i++)
        outputDipoles[i] = totalDipoles[i];
    delete amoebaReferenceMultipoleForce;
}



void ReferenceCalcMLPolMultipoleForceKernel::getElectrostaticPotential(ContextImpl& context, const std::vector< Vec3 >& inputGrid,
                                                                        std::vector< double >& outputElectrostaticPotential) {

    AmoebaReferenceMultipoleForce* amoebaReferenceMultipoleForce = setupAmoebaReferenceMultipoleForce(context);
    vector<Vec3>& posData                                     = extractPositions(context);
    vector<Vec3> grid(inputGrid.size());
    vector<double> potential(inputGrid.size());
    for (unsigned int ii = 0; ii < inputGrid.size(); ii++) {
        grid[ii] = inputGrid[ii];
    }
    amoebaReferenceMultipoleForce->calculateElectrostaticPotential(posData, charges, dipoles, quadrupoles, tholes,
                                                                   dampingFactors, polarity, axisTypes, 
                                                                   multipoleAtomZs, multipoleAtomXs, multipoleAtomYs,
                                                                   multipoleAtomCovalentInfo, grid, potential);

    outputElectrostaticPotential.resize(inputGrid.size());
    for (unsigned int ii = 0; ii < inputGrid.size(); ii++) {
        outputElectrostaticPotential[ii] = potential[ii];
    }

    delete amoebaReferenceMultipoleForce;
}

void ReferenceCalcMLPolMultipoleForceKernel::getSystemMultipoleMoments(ContextImpl& context, std::vector< double >& outputMultipoleMoments) {

    // retrieve masses

    const System& system             = context.getSystem();
    vector<double> masses;
    for (int i = 0; i <  system.getNumParticles(); ++i) {
        masses.push_back(system.getParticleMass(i));
    }    

    AmoebaReferenceMultipoleForce* amoebaReferenceMultipoleForce = setupAmoebaReferenceMultipoleForce(context);
    vector<Vec3>& posData                                     = extractPositions(context);
    amoebaReferenceMultipoleForce->calculateAmoebaSystemMultipoleMoments(masses, posData, charges, dipoles, quadrupoles, tholes,
                                                                         dampingFactors, polarity, axisTypes, 
                                                                         multipoleAtomZs, multipoleAtomXs, multipoleAtomYs,
                                                                         multipoleAtomCovalentInfo, outputMultipoleMoments);

    delete amoebaReferenceMultipoleForce;
}

void ReferenceCalcMLPolMultipoleForceKernel::copyParametersToContext(ContextImpl& context, const AmoebaMultipoleForce& force) {
    readLayered(force);
    if (numMultipoles != force.getNumMultipoles())
        throw OpenMMException("updateParametersInContext: The number of multipoles has changed");

    // Record the values.

    int dipoleIndex = 0;
    int quadrupoleIndex = 0;
    for (int i = 0; i < numMultipoles; ++i) {
        int axisType, multipoleAtomZ, multipoleAtomX, multipoleAtomY;
        double charge, tholeD, dampingFactorD, polarityD;
        std::vector<double> dipolesD;
        std::vector<double> quadrupolesD;
        force.getMultipoleParameters(i, charge, dipolesD, quadrupolesD, axisType, multipoleAtomZ, multipoleAtomX, multipoleAtomY, tholeD, dampingFactorD, polarityD);
        axisTypes[i] = axisType;
        multipoleAtomZs[i] = multipoleAtomZ;
        multipoleAtomXs[i] = multipoleAtomX;
        multipoleAtomYs[i] = multipoleAtomY;
        charges[i] = charge;
        tholes[i] = tholeD;
        dampingFactors[i] = dampingFactorD;
        polarity[i] = polarityD;
        dipoles[dipoleIndex++] = dipolesD[0];
        dipoles[dipoleIndex++] = dipolesD[1];
        dipoles[dipoleIndex++] = dipolesD[2];
        quadrupoles[quadrupoleIndex++] = quadrupolesD[0];
        quadrupoles[quadrupoleIndex++] = quadrupolesD[1];
        quadrupoles[quadrupoleIndex++] = quadrupolesD[2];
        quadrupoles[quadrupoleIndex++] = quadrupolesD[3];
        quadrupoles[quadrupoleIndex++] = quadrupolesD[4];
        quadrupoles[quadrupoleIndex++] = quadrupolesD[5];
        quadrupoles[quadrupoleIndex++] = quadrupolesD[6];
        quadrupoles[quadrupoleIndex++] = quadrupolesD[7];
        quadrupoles[quadrupoleIndex++] = quadrupolesD[8];
    }
}

void ReferenceCalcMLPolMultipoleForceKernel::getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const {
    if (!usePme)
        throw OpenMMException("getPMEParametersInContext: This Context is not using PME");
    alpha = alphaEwald;
    nx = pmeGridDimension[0];
    ny = pmeGridDimension[1];
    nz = pmeGridDimension[2];
}

// POLEN: injected ML parameters (lab frame) and derivatives
void ReferenceCalcMLPolMultipoleForceKernel::setMLParameters(ContextImpl& context, const vector<double>& q, const vector<double>& mu,
                                                             const vector<double>& theta, const vector<double>& alpha) {
    for (unsigned int k = 0; k < mlAtoms.size(); k++) {
        int i = mlAtoms[k];
        charges[i] = q[k];
        for (int a = 0; a < 3; a++)
            dipoles[3*i+a] = mu[3*k+a];
        // symmetric traceless part (OpenMM PME ignores the trace in direct space; getMLDerivatives returns the
        // derivative projected in the same way)
        const double* t = &theta[9*k];
        double tr = (t[0] + t[4] + t[8])/3.0;
        for (int a = 0; a < 3; a++)
            for (int b = 0; b < 3; b++)
                quadrupoles[9*i+3*a+b] = 0.5*(t[3*a+b] + t[3*b+a]) - (a == b ? tr : 0.0);
        axisTypes[i] = AmoebaMultipoleForce::NoAxisType;
        multipoleAtomZs[i] = multipoleAtomXs[i] = multipoleAtomYs[i] = -1;
        polarity[i] = alpha[k];
        dampingFactors[i] = pow(alpha[k], 1.0/6.0);
        tholes[i] = mlThole;
    }
}

void ReferenceCalcMLPolMultipoleForceKernel::getMLDerivatives(ContextImpl& context, vector<double>& phi, vector<double>& field,
                                                              vector<double>& dTheta, vector<double>& dAlpha) {
    OpenMM::MLPolRef::AmoebaReferenceMultipoleForce* ref = setupAmoebaReferenceMultipoleForce(context);
    vector<Vec3> fieldVec;
    try {
        ref->calculateMLDerivatives(extractPositions(context), charges, dipoles, quadrupoles, tholes, dampingFactors, polarity,
                                    axisTypes, multipoleAtomZs, multipoleAtomXs, multipoleAtomYs, multipoleAtomCovalentInfo,
                                    phi, fieldVec, dTheta, dAlpha);
    }
    catch (...) {
        delete ref;
        throw;
    }
    delete ref;
    field.resize(3*fieldVec.size());
    for (unsigned int k = 0; k < fieldVec.size(); k++)
        for (int a = 0; a < 3; a++)
            field[3*k+a] = fieldVec[k][a];
}

void ReferenceCalcMLPolMultipoleForceKernel::setMLPenetration(ContextImpl& context, const vector<double>& N, const vector<double>& s) {
    bool off = N.empty() && s.empty();              // empty lists disable penetration
    if (!off && (N.size() != mlAtoms.size() || s.size() != mlAtoms.size()))
        throw OpenMMException("MLPolMultipoleForce: penetration requires N and s for every ML atom");
    penN = N;
    penS = s;
}
