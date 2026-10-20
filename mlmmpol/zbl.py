"""Short-range screened nuclear repulsion (ZBL) as an ML/MM safety net.

V(r) = (Z_i Z_j e²/4πε₀r) · φ(r/a) · S(r),  a = 0.46850 Å / (Z_i^0.23 + Z_j^0.23)
φ(x) = 0.18175 e^(−3.19980x) + 0.50986 e^(−0.94229x) + 0.28022 e^(−0.40290x) + 0.02817 e^(−0.20162x)
(Ziegler, Biersack & Littmark, 1985; the Bohr potential is the single-exponential case.)

Pure ZBL is still worth tens of kJ/mol at normal contact distances (H···O at 1.8 Å: ~50 kJ/mol), so, as in MLIPs
(MACE, LAMMPS pair zbl), it is switched off by a C² switching function: S = 1 up to r_in = r_out/2 and S = 0 from
r_out = f·(R_cov,i + R_cov,j) (covalent radii of Cordero et al. 2008; f = 0.8 by default: H–O 0.78 Å, Mg–O 1.66 Å).
It only acts at distances normal dynamics never reaches and prevents the short-range "holes" (buffered 14-7 vdW
finite at r = 0 against the diverging multipolar attraction; regions outside the MLIP training).

add_zbl(system, topology, ml): nonbonded ML–MM and ML–ML pairs, in a CustomNonbondedForce with two interaction
groups. Energy in kJ/mol, distances in nm.
"""
import openmm as mm

COULOMB = 138.935458                  # kJ/mol·nm/e²
R_COV = {1: 0.031, 6: 0.076, 7: 0.071, 8: 0.066, 9: 0.057, 11: 0.166, 12: 0.141, 15: 0.107, 16: 0.105, 17: 0.102,
         19: 0.203, 20: 0.176, 35: 0.120, 53: 0.139}     # nm

PHI = ('0.18175*exp(-3.19980*x)+0.50986*exp(-0.94229*x)+0.28022*exp(-0.40290*x)+0.02817*exp(-0.20162*x)')


def _expression():
    """OpenMM expression: r_out factor = f_inter for ML–MM pairs and f_intra for ML–ML pairs (ml1·ml2 = 1); ML–ML pairs
    with an element in skip (e.g. the coordinated Mg²⁺) get no ZBL."""
    return (f'{COULOMB}*z1*z2/r*phi*sw*(1-ml1*ml2*max(skip1,skip2)); '
            f'phi={PHI}; x=r/a; a=0.046850/(z1^0.23+z2^0.23); '
            'sw=1-y^3*(10-15*y+6*y^2); y=min(max((r-rin)/(rout-rin),0),1); rin=0.5*rout; '
            'rout=(zblf+ml1*ml2*(zblf_intra-zblf))*(rc1+rc2)')


def add_zbl(system, topology, ml, factor=0.8, force_group=0, mm_atoms=None, factor_intra=None, skip_intra_z=()):
    """Add the ZBL repulsion with a smooth switch on nonbonded ML–MM and ML–ML pairs, in a single CustomNonbondedForce
    with two interaction groups (ML×MM and ML×ML) and exclusions for the bonded (1-2) pairs of the ML region. Adds no
    bonds to the system (a CustomBondForce between ML atoms would change the OpenMM "molecules" and, with the plugin,
    CUDA dynamics blew up in ~50 fs). factor_intra: r_out factor for ML–ML pairs (default: the same);
    skip_intra_z: elements without intra-ML ZBL."""
    factor_intra = factor if factor_intra is None else factor_intra
    atoms = list(topology.atoms())
    z = [a.element.atomic_number if a.element is not None else 0 for a in atoms]
    ml = sorted(int(i) for i in ml)
    mlset = set(ml)
    real = [i for i in range(len(atoms)) if z[i] > 0]
    mm_atoms = sorted(set(real) - mlset) if mm_atoms is None else sorted(mm_atoms)
    for i in ml + mm_atoms:
        assert z[i] in R_COV, f'no covalent radius for Z = {z[i]}'
    rmax = max(factor, factor_intra) * 2 * max(R_COV[z[i]] for i in ml + mm_atoms)

    nb = mm.CustomNonbondedForce(_expression())
    nb.addGlobalParameter('zblf', factor)
    nb.addGlobalParameter('zblf_intra', factor_intra)
    for name in ('z', 'rc', 'ml', 'skip'):
        nb.addPerParticleParameter(name)
    for i in range(system.getNumParticles()):
        zi = z[i] if i < len(z) else 0
        nb.addParticle([float(zi), R_COV.get(zi, 0.1), 1.0 if i in mlset else 0.0, 1.0 if zi in skip_intra_z else 0.0])
    periodic = system.usesPeriodicBoundaryConditions()
    nb.setNonbondedMethod(mm.CustomNonbondedForce.CutoffPeriodic if periodic else mm.CustomNonbondedForce.CutoffNonPeriodic)
    nb.setCutoffDistance(max(rmax, 0.2))
    nb.setUseSwitchingFunction(False)
    nb.setUseLongRangeCorrection(False)
    nb.addInteractionGroup(ml, mm_atoms)
    nb.addInteractionGroup(ml, ml)
    for b in topology.bonds():
        i, j = b[0].index, b[1].index
        if i in mlset and j in mlset:
            nb.addExclusion(i, j)
    nb.setForceGroup(force_group)
    nb.setName('ZBL')
    system.addForce(nb)
    return nb


def zbl_energy(zi, zj, r_nm, factor=0.8):
    """Python reference (kJ/mol) for the tests."""
    import math
    a = 0.046850 / (zi ** 0.23 + zj ** 0.23)
    x = r_nm / a
    phi = (0.18175 * math.exp(-3.19980 * x) + 0.50986 * math.exp(-0.94229 * x) + 0.28022 * math.exp(-0.40290 * x)
           + 0.02817 * math.exp(-0.20162 * x))
    rout = factor * (R_COV[zi] + R_COV[zj])
    rin = 0.5 * rout
    y = min(max((r_nm - rin) / (rout - rin), 0.0), 1.0)
    return COULOMB * zi * zj / r_nm * phi * (1 - y ** 3 * (10 - 15 * y + 6 * y ** 2))
