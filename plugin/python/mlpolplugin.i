%module mlpolplugin

%import(module="openmm.openmm") "swig/OpenMMSwigHeaders.i"
using namespace OpenMM;
%include "swig/typemaps.i"
%include "std_vector.i"
%include "std_string.i"
namespace std {
  %template(vectord) vector<double>;
  %template(vectori) vector<int>;
};

%{
#include "MLPolMultipoleForce.h"
#include "OpenMM.h"
#include "OpenMMAmoeba.h"
#include "OpenMMDrude.h"
#include "openmm/RPMDIntegrator.h"
#include "openmm/RPMDMonteCarloBarostat.h"
%}

%pythoncode %{
import openmm as mm
%}

%exception {
    try {
        $action
    } catch (std::exception &e) {
        PyErr_SetString(PyExc_Exception, const_cast<char*>(e.what()));
        return NULL;
    }
}

namespace MLPol {

class MLPolMultipoleForce : public OpenMM::AmoebaMultipoleForce {
public:
    MLPolMultipoleForce();
    MLPolMultipoleForce(const OpenMM::AmoebaMultipoleForce& source);
    void updateParametersInContext(OpenMM::Context& context);
    void setLayeredPolarization(const std::vector<int>& centerAtoms, const std::vector<int>& groupOf,
                                const std::vector<int>& groupRefAtoms, double rIn, double rOut, double tau=0.02);
    void disableLayeredPolarization();
    bool usesLayeredPolarization() const;
    void setMLRegion(const std::vector<int>& mlAtoms, double tholeML=0.39, double crossThole=-1.0);
    void setMLHead(const std::string& torchScriptFile);
    void setMLPenetrationInContext(OpenMM::Context& context, const std::vector<double>& N, const std::vector<double>& s);
    const std::string& getMLHead() const;
    double getMLThole() const;
    double getCrossThole() const;
    void setMLParametersInContext(OpenMM::Context& context, const std::vector<double>& charges, const std::vector<double>& dipoles,
                                  const std::vector<double>& quadrupoles, const std::vector<double>& polarities);
};

}

/* Dipole getters as flat vectors (N*3, e*nm); Python reshapes them to (N, 3). */
%extend MLPol::MLPolMultipoleForce {
    std::vector<double> _dipoles(OpenMM::Context& context, int kind) {
        std::vector<OpenMM::Vec3> d;
        if (kind == 0) $self->getInducedDipoles(context, d);
        else if (kind == 1) $self->getLabFramePermanentDipoles(context, d);
        else $self->getTotalDipoles(context, d);
        std::vector<double> out;
        for (auto& v : d) { out.push_back(v[0]); out.push_back(v[1]); out.push_back(v[2]); }
        return out;
    }
    std::vector<int> _mlAtoms() {
        return $self->getMLAtoms();
    }
    std::vector<double> _mlDerivatives(OpenMM::Context& context) {
        std::vector<double> phi, field, dTheta, dAlpha;
        $self->getMLDerivatives(context, phi, field, dTheta, dAlpha);
        std::vector<double> out(phi);
        out.insert(out.end(), field.begin(), field.end());
        out.insert(out.end(), dTheta.begin(), dTheta.end());
        out.insert(out.end(), dAlpha.begin(), dAlpha.end());
        return out;
    }
    %pythoncode %{
    def getMLAtoms(self):
        return list(self._mlAtoms())
    def getMLDerivatives(self, context):
        """Derivatives of the energy with respect to the ML parameters: dict with phi (M), field (M,3),
        dTheta (M,3,3), dAlpha (M), in kJ/mol and OpenMM units (e, nm)."""
        import numpy as np
        v = np.array(self._mlDerivatives(context))
        M = len(self._mlAtoms())
        return dict(phi=v[:M], field=v[M:4*M].reshape(M, 3), dTheta=v[4*M:13*M].reshape(M, 3, 3), dAlpha=v[13*M:14*M])
    def getInducedDipoles(self, context):
        import numpy as np
        return np.array(self._dipoles(context, 0)).reshape(-1, 3)
    def getLabFramePermanentDipoles(self, context):
        import numpy as np
        return np.array(self._dipoles(context, 1)).reshape(-1, 3)
    def getTotalDipoles(self, context):
        import numpy as np
        return np.array(self._dipoles(context, 2)).reshape(-1, 3)
    %}
}
