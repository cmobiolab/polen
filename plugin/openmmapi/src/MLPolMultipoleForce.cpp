#include "MLPolMultipoleForce.h"
#include "internal/MLPolMultipoleForceImpl.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/OpenMMException.h"

using namespace MLPol;
using namespace OpenMM;
using namespace std;

MLPolMultipoleForce::MLPolMultipoleForce() {
}

MLPolMultipoleForce::MLPolMultipoleForce(const AmoebaMultipoleForce& source) : AmoebaMultipoleForce(source) {
}

ForceImpl* MLPolMultipoleForce::createImpl() const {
    return new MLPolMultipoleForceImpl(*this);
}

void MLPolMultipoleForce::getInducedDipoles(Context& context, vector<Vec3>& dipoles) {
    dynamic_cast<MLPolMultipoleForceImpl&>(getImplInContext(context)).getInducedDipoles(getContextImpl(context), dipoles);
}

void MLPolMultipoleForce::getLabFramePermanentDipoles(Context& context, vector<Vec3>& dipoles) {
    dynamic_cast<MLPolMultipoleForceImpl&>(getImplInContext(context)).getLabFramePermanentDipoles(getContextImpl(context), dipoles);
}

void MLPolMultipoleForce::getTotalDipoles(Context& context, vector<Vec3>& dipoles) {
    dynamic_cast<MLPolMultipoleForceImpl&>(getImplInContext(context)).getTotalDipoles(getContextImpl(context), dipoles);
}

void MLPolMultipoleForce::updateParametersInContext(Context& context) {
    dynamic_cast<MLPolMultipoleForceImpl&>(getImplInContext(context)).updateParametersInContext(getContextImpl(context));
}

void MLPolMultipoleForce::setLayeredPolarization(const vector<int>& centerAtoms, const vector<int>& groupOf,
                                                 const vector<int>& groupRefAtoms, double rIn, double rOut, double tau) {
    if (centerAtoms.empty())
        throw OpenMMException("setLayeredPolarization: empty centerAtoms");
    if ((int) groupOf.size() != getNumMultipoles())
        throw OpenMMException("setLayeredPolarization: groupOf must have one entry per particle");
    for (int g : groupOf)
        if (g >= (int) groupRefAtoms.size())
            throw OpenMMException("setLayeredPolarization: group index without a reference atom");
    if (!(rOut > rIn) || rIn < 0.0 || tau <= 0.0)
        throw OpenMMException("setLayeredPolarization: requires 0 <= rIn < rOut and tau > 0");
    layeredEnabled = true;
    layeredCenter = centerAtoms;
    layeredGroupOf = groupOf;
    layeredGroupRef = groupRefAtoms;
    layeredRIn = rIn;
    layeredROut = rOut;
    layeredTau = tau;
}

void MLPolMultipoleForce::getLayeredPolarization(vector<int>& centerAtoms, vector<int>& groupOf, vector<int>& groupRefAtoms,
                                                 double& rIn, double& rOut, double& tau) const {
    centerAtoms = layeredCenter;
    groupOf = layeredGroupOf;
    groupRefAtoms = layeredGroupRef;
    rIn = layeredRIn;
    rOut = layeredROut;
    tau = layeredTau;
}

void MLPolMultipoleForce::setMLRegion(const vector<int>& atoms, double tholeML, double cross) {
    for (int a : atoms)
        if (a < 0 || a >= getNumMultipoles())
            throw OpenMMException("setMLRegion: invalid atom index");
    mlAtoms = atoms;
    mlThole = tholeML;
    crossThole = cross;
}

void MLPolMultipoleForce::setMLParametersInContext(Context& context, const vector<double>& charges, const vector<double>& dipoles,
                                                   const vector<double>& quadrupoles, const vector<double>& polarities) {
    int M = mlAtoms.size();
    if ((int) charges.size() != M || (int) dipoles.size() != 3*M || (int) quadrupoles.size() != 9*M || (int) polarities.size() != M)
        throw OpenMMException("setMLParametersInContext: expected sizes M, 3M, 9M, M");
    dynamic_cast<MLPolMultipoleForceImpl&>(getImplInContext(context)).setMLParameters(getContextImpl(context), charges, dipoles, quadrupoles, polarities);
}

void MLPolMultipoleForce::getMLDerivatives(Context& context, vector<double>& phi, vector<double>& field,
                                           vector<double>& dTheta, vector<double>& dAlpha) {
    dynamic_cast<MLPolMultipoleForceImpl&>(getImplInContext(context)).getMLDerivatives(getContextImpl(context), phi, field, dTheta, dAlpha);
}

void MLPolMultipoleForce::setMLPenetrationInContext(Context& context, const vector<double>& N, const vector<double>& s) {
    dynamic_cast<MLPolMultipoleForceImpl&>(getImplInContext(context)).setMLPenetration(getContextImpl(context), N, s);
}
