/* XML serialization of MLPolMultipoleForce: the AMOEBA part is delegated to the OpenMM proxy (exported by
 * libOpenMMAmoeba) + the layered-polarization and ML settings. Registered automatically when the library loads. */

#include "MLPolMultipoleForce.h"
#include "openmm/serialization/AmoebaMultipoleForceProxy.h"
#include "openmm/serialization/SerializationNode.h"
#include "openmm/serialization/SerializationProxy.h"
#include "openmm/OpenMMException.h"
#include <typeinfo>

using namespace MLPol;
using namespace OpenMM;
using namespace std;

namespace {

class MLPolMultipoleForceProxy : public SerializationProxy {
public:
    MLPolMultipoleForceProxy() : SerializationProxy("MLPolMultipoleForce") {
    }

    void serialize(const void* object, SerializationNode& node) const {
        node.setIntProperty("version", 1);
        const MLPolMultipoleForce& force = *reinterpret_cast<const MLPolMultipoleForce*>(object);
        AmoebaMultipoleForceProxy amoebaProxy;
        amoebaProxy.serialize(static_cast<const AmoebaMultipoleForce*>(&force), node.createChildNode("AmoebaMultipoleForce"));
        SerializationNode& layered = node.createChildNode("LayeredPolarization");
        layered.setBoolProperty("enabled", force.usesLayeredPolarization());
        if (force.usesLayeredPolarization()) {
            vector<int> center, groupOf, groupRef;
            double rIn, rOut, tau;
            force.getLayeredPolarization(center, groupOf, groupRef, rIn, rOut, tau);
            layered.setDoubleProperty("rIn", rIn).setDoubleProperty("rOut", rOut).setDoubleProperty("tau", tau);
            SerializationNode& c = layered.createChildNode("CenterAtoms");
            for (int a : center)
                c.createChildNode("Atom").setIntProperty("index", a);
            SerializationNode& g = layered.createChildNode("GroupOf");
            for (int a : groupOf)
                g.createChildNode("Particle").setIntProperty("group", a);
            SerializationNode& r = layered.createChildNode("GroupRefAtoms");
            for (int a : groupRef)
                r.createChildNode("Group").setIntProperty("ref", a);
        }
        SerializationNode& ml = node.createChildNode("MLRegion");
        ml.setDoubleProperty("tholeML", force.getMLThole()).setDoubleProperty("crossThole", force.getCrossThole());
        ml.setStringProperty("head", force.getMLHead());
        for (int a : force.getMLAtoms())
            ml.createChildNode("Atom").setIntProperty("index", a);
    }

    void* deserialize(const SerializationNode& node) const {
        if (node.getIntProperty("version") != 1)
            throw OpenMMException("MLPolMultipoleForce: unsupported serialization version");
        AmoebaMultipoleForceProxy amoebaProxy;
        AmoebaMultipoleForce* amoeba = reinterpret_cast<AmoebaMultipoleForce*>(amoebaProxy.deserialize(node.getChildNode("AmoebaMultipoleForce")));
        MLPolMultipoleForce* force = new MLPolMultipoleForce(*amoeba);
        delete amoeba;
        try {
            const SerializationNode& layered = node.getChildNode("LayeredPolarization");
            if (layered.getBoolProperty("enabled")) {
                vector<int> center, groupOf, groupRef;
                for (auto& child : layered.getChildNode("CenterAtoms").getChildren())
                    center.push_back(child.getIntProperty("index"));
                for (auto& child : layered.getChildNode("GroupOf").getChildren())
                    groupOf.push_back(child.getIntProperty("group"));
                for (auto& child : layered.getChildNode("GroupRefAtoms").getChildren())
                    groupRef.push_back(child.getIntProperty("ref"));
                force->setLayeredPolarization(center, groupOf, groupRef, layered.getDoubleProperty("rIn"),
                                              layered.getDoubleProperty("rOut"), layered.getDoubleProperty("tau"));
            }
            for (auto& ml : node.getChildren()) {       // optional node (older files do not have it)
                if (ml.getName() != "MLRegion")
                    continue;
                vector<int> atoms;
                for (auto& child : ml.getChildren())
                    atoms.push_back(child.getIntProperty("index"));
                if (!atoms.empty())
                    force->setMLRegion(atoms, ml.getDoubleProperty("tholeML"), ml.getDoubleProperty("crossThole"));
                force->setMLHead(ml.getStringProperty("head", ""));
            }
        }
        catch (...) {
            delete force;
            throw;
        }
        return force;
    }
};

} // namespace

extern "C" void __attribute__((constructor)) registerMLPolSerializationProxies() {
    SerializationProxy::registerProxy(typeid(MLPolMultipoleForce), new MLPolMultipoleForceProxy());
}
