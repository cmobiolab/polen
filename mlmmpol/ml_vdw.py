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


def apply_ml_vdw(system, topology, ml, scale_R, scale_eps, pair_scale_R=None, pair_scale_eps=None):
    """Apply the factors (dicts class -> factor; missing classes stay 1) to the ML atoms of the AmoebaVdwForce.

    pair_scale_R / pair_scale_eps (optional): extra factors for specific ML-class × ion pairs, keyed 'class|element'
    (e.g. 'N|Na'), applied on top of the per-class factors to the combined pair parameters (CUBIC-MEAN R_min, HHG ε)
    and stored as type pairs; the ML–water and other ML–MM pairs are unaffected."""
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
    if pair_scale_R or pair_scale_eps:
        add_ion_pairs(f, topology, ml, cls, new_type, pair_scale_R or {}, pair_scale_eps or {})
    return cls


def add_ion_pairs(f, topology, ml, cls, new_type, pair_scale_R, pair_scale_eps):
    """Type pairs (new ML type, ion type) with the combined parameters scaled by the pair factors."""
    atoms = list(topology.atoms())
    mlset = set(ml)
    ion_types = {}                                   # element -> set of particle types of the (non-ML) ions
    for a in atoms:
        if a.index not in mlset and a.element is not None and a.element.symbol in ('Na', 'Cl', 'K', 'Mg', 'Ca'):
            if not any(True for _ in a.residue.bonds()):
                ion_types.setdefault(a.element.symbol, set()).add(f.getParticleParameters(a.index)[5])
    existing = {(f.getTypePairParameters(k)[0], f.getTypePairParameters(k)[1]): k for k in range(f.getNumTypePairs())}
    done = set()
    for i in ml:
        t = f.getParticleParameters(i)[5]
        for key in set(pair_scale_R) | set(pair_scale_eps):
            c, el = key.split('|')
            if c != cls[i]:
                continue
            fR, fE = pair_scale_R.get(key, 1.0), pair_scale_eps.get(key, 1.0)
            for ti in ion_types.get(el, ()):
                if (t, ti) in done:
                    continue
                done.add((t, ti))
                k = existing.get((t, ti), existing.get((ti, t)))
                if k is not None:                    # vdwpr already present (copied from the original type): scale it
                    ta, tb, sg, ep = f.getTypePairParameters(k)
                    f.setTypePairParameters(k, ta, tb, sg * fR, ep * fE)
                    continue
                si, ei = f.getParticleTypeParameters(t)
                sj, ej = f.getParticleTypeParameters(ti)
                si, sj = 2 * _val(si, mm.unit.nanometer), 2 * _val(sj, mm.unit.nanometer)   # types store R_min/2
                ei, ej = _val(ei, mm.unit.kilojoule_per_mole), _val(ej, mm.unit.kilojoule_per_mole)
                rmin = (si ** 3 + sj ** 3) / (si ** 2 + sj ** 2)                                # CUBIC-MEAN
                eps = 4 * ei * ej / (ei ** 0.5 + ej ** 0.5) ** 2                               # HHG
                f.addTypePair(t, ti, rmin * fR, eps * fE)


def _val(q, u):
    return q.value_in_unit(u) if hasattr(q, 'value_in_unit') else q


def load_fit(path):
    d = json.load(open(path))
    return d['scale_R'], d['scale_eps']
