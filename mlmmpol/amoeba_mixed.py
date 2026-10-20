"""Mixed ML/MM system for AMOEBA in OpenMM (level 1: multipolar mechanical embedding).

openmm-ml (1.7) `MLPotential.createMixedSystem` correctly removes the bonded terms inside the ML region and adds
the ML potential, but it does **not** handle the AMOEBA nonbonded forces: intra-ML vdW and electrostatics stay in
the system and are double counted. Here:

- AmoebaVdwForce: all ML–ML pairs become exclusions;
- AmoebaMultipoleForce: ML–ML pairs go to Covalent12 and the ML region becomes a single polarization group
  (PolarizationCovalent11). This zeroes m, p and d for these pairs and keeps u = 1
  (validated against ref_torch in tests/test_mlmm_pol.py);
- any remaining bonded term entirely inside the ML region, or any unknown force type, raises an error instead of
  passing silently.
"""

import openmm as mm

AMF = mm.AmoebaMultipoleForce

# Forces handled by createMixedSystem (bonded terms) or independent of the ML region.
_BONDED = (mm.HarmonicBondForce, mm.HarmonicAngleForce, mm.PeriodicTorsionForce, mm.RBTorsionForce,
           mm.CustomBondForce, mm.CustomAngleForce, mm.CustomTorsionForce, mm.CustomCompoundBondForce)
_NEUTRAL = (mm.CMMotionRemover,)


def mask_multipoles(force, ml):
    """Intra-ML masking in the AmoebaMultipoleForce through the covalent maps.

    All ML–ML pairs go to Covalent12 and are removed from Covalent13/14/15: 1-4 and 1-5 pairs inside the region
    would otherwise get m = 0.4/0.8 and p = 1 (or intra-group p14 = 0.5) instead of zero.
    """
    ml = sorted(int(i) for i in ml)
    ml_set = set(ml)
    for i in ml:
        c12 = set(force.getCovalentMap(i, AMF.Covalent12)) | (ml_set - {i})
        force.setCovalentMap(i, AMF.Covalent12, sorted(c12))
        for t in (AMF.Covalent13, AMF.Covalent14, AMF.Covalent15, AMF.PolarizationCovalent12,
                  AMF.PolarizationCovalent13, AMF.PolarizationCovalent14):
            force.setCovalentMap(i, t, [j for j in force.getCovalentMap(i, t) if j not in ml_set])
        force.setCovalentMap(i, AMF.PolarizationCovalent11, ml)


def exclude_vdw(force, ml):
    """Exclude all ML–ML pairs from the AmoebaVdwForce."""
    ml = sorted(int(i) for i in ml)
    for i in ml:
        excl = set(force.getParticleExclusions(i)) | set(ml)
        force.setParticleExclusions(i, sorted(excl))


def _term_atoms(force, k):
    """Atoms of the k-th term of a bonded force."""
    if isinstance(force, (mm.HarmonicBondForce, mm.CustomBondForce)):
        return force.getBondParameters(k)[:2]
    if isinstance(force, (mm.HarmonicAngleForce, mm.CustomAngleForce)):
        return force.getAngleParameters(k)[:3]
    if isinstance(force, (mm.PeriodicTorsionForce, mm.RBTorsionForce, mm.CustomTorsionForce)):
        return force.getTorsionParameters(k)[:4]
    if isinstance(force, mm.CustomCompoundBondForce):
        return force.getBondParameters(k)[0]
    if isinstance(force, mm.AmoebaTorsionTorsionForce):
        return force.getTorsionTorsionParameters(k)[:5]
    raise TypeError(type(force).__name__)


def _num_terms(force):
    for m in ('getNumBonds', 'getNumAngles', 'getNumTorsions', 'getNumTorsionTorsions'):
        if hasattr(force, m):
            return getattr(force, m)()
    return 0


def check_no_intra_ml_terms(system, ml):
    """Check that no remaining bonded term lies entirely inside the ML region."""
    ml = set(int(i) for i in ml)
    for force in system.getForces():
        if isinstance(force, _BONDED + (mm.AmoebaTorsionTorsionForce,)):
            for k in range(_num_terms(force)):
                if set(_term_atoms(force, k)) <= ml:
                    raise RuntimeError(f'{type(force).__name__}: term {k} entirely inside the ML region')


def create_mixed_amoeba_system(potential, topology, system, ml_atoms, force_group=0, ml_backend='torchforce',
                               **args):
    """Create the level-1 ML/MM system from an AMOEBA system.

    potential: openmmml.MLPotential; args are passed on to createMixedSystem (e.g. precision).
    ml_backend: 'torchforce' (default; MACE as TorchScript on the GPU) or 'pythonforce' (the openmm-ml 1.7 one, which
    on the CUDA platform with an AmoebaMultipoleForce receives corrupted positions).
    """
    ml = sorted(int(i) for i in ml_atoms)
    known = _BONDED + _NEUTRAL + (mm.AmoebaMultipoleForce, mm.AmoebaVdwForce, mm.AmoebaTorsionTorsionForce)
    for f in system.getForces():
        if not isinstance(f, known):
            raise NotImplementedError(f'force not supported in the AMOEBA mixed system: {type(f).__name__}')
        if isinstance(f, mm.AmoebaTorsionTorsionForce):
            if any(set(_term_atoms(f, k)) <= set(ml) for k in range(_num_terms(f))):
                raise NotImplementedError('AmoebaTorsionTorsionForce with terms in the ML region')

    mixed = potential.createMixedSystem(topology, system, ml, forceGroup=force_group, **args)
    for f in mixed.getForces():
        if isinstance(f, mm.AmoebaMultipoleForce):
            mask_multipoles(f, ml)
        elif isinstance(f, mm.AmoebaVdwForce):
            exclude_vdw(f, ml)
    check_no_intra_ml_terms(mixed, ml)
    if ml_backend == 'torchforce':
        from .mace_torchforce import MACE_MODELS, make_mace_torchforce, replace_python_force
        name = potential._impl.name
        if name not in MACE_MODELS:
            raise NotImplementedError(f'TorchForce only implemented for {sorted(MACE_MODELS)}')
        tf = make_mace_torchforce(name, topology, ml, precision=args.get('precision') or 'single',
                                  periodic=system.usesPeriodicBoundaryConditions(), force_group=force_group,
                                  charge=float(args.get('charge', 0)), multiplicity=float(args.get('multiplicity', 1)))
        replace_python_force(mixed, tf)
    elif ml_backend != 'pythonforce':
        raise ValueError(f'unknown ml_backend: {ml_backend}')
    return mixed
