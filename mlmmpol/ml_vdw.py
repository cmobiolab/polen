"""Buffered 14-7 vdW of the ML atoms refitted to the MuPE(ML) electrostatics, following AMOEBA+ (Liu, Piquemal & Ren,
JCTC 15, 4122, 2019): R_min and ε factors per ML atom class (H bonded to N/O/C, N, O, C, Br...), fitted to QM
dimer curves.

Each ML atom gets a new vdW type (copy of the AMOEBA type with scaled R and ε); the specific pairs (vdwpr) of the
original type are copied with the same factors. ML–ML pairs are already exclusions (amoeba_mixed.exclude_vdw), so
only ML–MM pairs change; MM–MM is untouched.
"""
import json

import openmm as mm


def atom_classes(topology, ml):
    """Class of each ML atom: the element, or 'H' + element of the neighbor for hydrogens."""
    atoms = list(topology.atoms())
    nb = {a.index: [] for a in atoms}
    for b in topology.bonds():
        nb[b[0].index].append(b[1])
        nb[b[1].index].append(b[0])
    out = {}
    for i in ml:
        el = atoms[i].element.symbol
        out[i] = 'H' + nb[i][0].element.symbol if el == 'H' and nb[i] else el
    return out


def apply_ml_vdw(system, topology, ml, scale_R, scale_eps):
    """Apply the factors (dicts class -> factor; missing classes stay 1) to the ML atoms of the AmoebaVdwForce."""
    f = next(x for x in system.getForces() if isinstance(x, mm.AmoebaVdwForce))
    if not f.getUseParticleTypes():
        raise ValueError('AmoebaVdwForce without particle types')
    cls = atom_classes(topology, ml)
    pairs = [f.getTypePairParameters(k) for k in range(f.getNumTypePairs())]
    new_type = {}
    for i in ml:
        p = list(f.getParticleParameters(i))
        typ = p[5]
        sR, sE = scale_R.get(cls[i], 1.0), scale_eps.get(cls[i], 1.0)
        key = (typ, sR, sE)
        if key not in new_type:
            sig, eps = f.getParticleTypeParameters(typ)
            t = f.addParticleType(sig * sR, eps * sE)
            for ta, tb, sg, ep in pairs:          # vdwpr of the original type, with the same factors
                if ta == typ:
                    f.addTypePair(t, tb, sg * sR, ep * sE)
                elif tb == typ:
                    f.addTypePair(ta, t, sg * sR, ep * sE)
            new_type[key] = t
        p[5] = new_type[key]
        f.setParticleParameters(i, *p)
    return cls


def load_fit(path):
    d = json.load(open(path))
    return d['scale_R'], d['scale_eps']
