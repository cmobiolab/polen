#include "internal/MLPolMultipoleForceImpl.h"
#include "MLPolKernels.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/OpenMMException.h"

using namespace MLPol;
using namespace OpenMM;
using namespace std;

MLPolMultipoleForceImpl::MLPolMultipoleForceImpl(const MLPolMultipoleForce& owner) : owner(owner) {
}

void MLPolMultipoleForceImpl::initialize(ContextImpl& context) {
    // same checks as AmoebaMultipoleForceImpl
    const System& system = context.getSystem();
    if (owner.getNumMultipoles() != system.getNumParticles())
        throw OpenMMException("MLPolMultipoleForce must have exactly as many particles as the System it belongs to.");
    if (owner.getNonbondedMethod() == AmoebaMultipoleForce::PME) {
        Vec3 boxVectors[3];
        system.getDefaultPeriodicBoxVectors(boxVectors[0], boxVectors[1], boxVectors[2]);
        double cutoff = owner.getCutoffDistance();
        if (cutoff > 0.5*boxVectors[0][0] || cutoff > 0.5*boxVectors[1][1] || cutoff > 0.5*boxVectors[2][2])
            throw OpenMMException("MLPolMultipoleForce: The cutoff distance cannot be greater than half the periodic box size.");
    }
    for (int i = 0; i < owner.getNumMultipoles(); i++) {
        double charge, thole, damping, polarity;
        int axisType, atomX, atomY, atomZ;
        vector<double> dipole, quadrupole;
        owner.getMultipoleParameters(i, charge, dipole, quadrupole, axisType, atomZ, atomX, atomY, thole, damping, polarity);
        if (axisType != AmoebaMultipoleForce::NoAxisType && axisType != AmoebaMultipoleForce::ZOnly &&
            axisType != AmoebaMultipoleForce::Bisector && axisType != AmoebaMultipoleForce::ZThenX &&
            axisType != AmoebaMultipoleForce::ZBisect && axisType != AmoebaMultipoleForce::ThreeFold)
            throw OpenMMException("MLPolMultipoleForce: axis type not supported for particle " + to_string(i));
    }
    kernel = context.getPlatform().createKernel(CalcMLPolMultipoleForceKernel::Name(), context);
    kernel.getAs<CalcMLPolMultipoleForceKernel>().initialize(context.getSystem(), owner);
}

double MLPolMultipoleForceImpl::calcForcesAndEnergy(ContextImpl& context, bool includeForces, bool includeEnergy, int groups) {
    if ((groups&(1<<owner.getForceGroup())) != 0)
        return kernel.getAs<CalcMLPolMultipoleForceKernel>().execute(context, includeForces, includeEnergy);
    return 0.0;
}

vector<string> MLPolMultipoleForceImpl::getKernelNames() {
    return {CalcMLPolMultipoleForceKernel::Name()};
}

void MLPolMultipoleForceImpl::getLabFramePermanentDipoles(ContextImpl& context, vector<Vec3>& dipoles) {
    kernel.getAs<CalcMLPolMultipoleForceKernel>().getLabFramePermanentDipoles(context, dipoles);
}

void MLPolMultipoleForceImpl::getInducedDipoles(ContextImpl& context, vector<Vec3>& dipoles) {
    kernel.getAs<CalcMLPolMultipoleForceKernel>().getInducedDipoles(context, dipoles);
}

void MLPolMultipoleForceImpl::getTotalDipoles(ContextImpl& context, vector<Vec3>& dipoles) {
    kernel.getAs<CalcMLPolMultipoleForceKernel>().getTotalDipoles(context, dipoles);
}

void MLPolMultipoleForceImpl::updateParametersInContext(ContextImpl& context) {
    kernel.getAs<CalcMLPolMultipoleForceKernel>().copyParametersToContext(context, owner);
    context.systemChanged();
}

void MLPolMultipoleForceImpl::setMLParameters(ContextImpl& context, const vector<double>& charges, const vector<double>& dipoles,
                                              const vector<double>& quadrupoles, const vector<double>& polarities) {
    kernel.getAs<CalcMLPolMultipoleForceKernel>().setMLParameters(context, charges, dipoles, quadrupoles, polarities);
    context.systemChanged();
}

void MLPolMultipoleForceImpl::getMLDerivatives(ContextImpl& context, vector<double>& phi, vector<double>& field,
                                               vector<double>& dTheta, vector<double>& dAlpha) {
    kernel.getAs<CalcMLPolMultipoleForceKernel>().getMLDerivatives(context, phi, field, dTheta, dAlpha);
}

void MLPolMultipoleForceImpl::setMLPenetration(ContextImpl& context, const vector<double>& N, const vector<double>& s) {
    kernel.getAs<CalcMLPolMultipoleForceKernel>().setMLPenetration(context, N, s);
    context.systemChanged();
}
