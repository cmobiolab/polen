"""Level-1 mixed system (mlmmpol.amoeba_mixed) with NMA as the ML region.

NMA (= ACE–NME in amoeba2018.xml) + 17 waters, no cutoff. Each term is validated separately:
- bonded and vdW: mixed == full − isolated NMA (terms additive per pair/term);
- multipoles: mixed == level-1 ref_torch (intra-ML masking);
- MACE: mixed == MACE on the isolated NMA; zero force on the waters.
"""

import numpy as np
import openmm as mm
import pytest
import torch
from openmm import app, unit

from mlmmpol.amoeba_mixed import create_mixed_amoeba_system
from openmm_systems import SYSTEMS, make_traceless
from ref_torch import amoeba_elec as ae
from ref_torch.mlmm_pol import mlmm_electrostatic_energy

openmmml = pytest.importorskip('openmmml')

BONDED, MULTIPOLE, VDW, ML = 0, 1, 2, 3


def _groups(system):
    for f in system.getForces():
        if isinstance(f, mm.AmoebaMultipoleForce):
            f.setForceGroup(MULTIPOLE)
            f.setMutualInducedMaxIterations(1000)
        elif isinstance(f, mm.AmoebaVdwForce):
            f.setForceGroup(VDW)
        elif f.getForceGroup() != ML:
            f.setForceGroup(BONDED)


def _amoeba(topology):
    system = app.ForceField('amoeba2018.xml').createSystem(
        topology, nonbondedMethod=app.NoCutoff, polarization='mutual', mutualInducedTargetEpsilon=1e-10)
    make_traceless(next(f for f in system.getForces() if isinstance(f, mm.AmoebaMultipoleForce)))
    return system


def _eval(system, positions, group):
    ctx = mm.Context(system, mm.VerletIntegrator(1e-3), mm.Platform.getPlatformByName('Reference'))
    ctx.setPositions(positions)
    st = ctx.getState(getEnergy=True, getForces=True, groups={group})
    return (st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole),
            np.array(st.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)))


@pytest.fixture(scope='module')
def nma():
    pdb = app.PDBFile(str(SYSTEMS / 'nma_shell.pdb'))
    top, pos = pdb.topology, pdb.positions
    ml = [a.index for a in top.atoms() if a.residue.name in ('ACE', 'NME')]
    assert ml == list(range(len(ml)))          # NMA comes first: makes comparing forces easier

    full = _amoeba(top)
    potential = openmmml.MLPotential('mace-off23-small')
    mixed = create_mixed_amoeba_system(potential, top, full, ml, force_group=ML, precision='double')

    vac = app.Modeller(top, pos)
    vac.delete([r for r in top.residues() if r.name == 'HOH'])
    iso = _amoeba(vac.topology)
    iso_ml = potential.createSystem(vac.topology, precision='double')
    for s in (full, mixed, iso):
        _groups(s)
    return dict(top=top, pos=pos, ml=ml, full=full, mixed=mixed, iso=iso, iso_ml=iso_ml,
                vac_pos=vac.positions)


@pytest.mark.parametrize('group', [BONDED, VDW])
def test_additive_terms_remove_ml_region(nma, group):
    e_mix, f_mix = _eval(nma['mixed'], nma['pos'], group)
    e_full, f_full = _eval(nma['full'], nma['pos'], group)
    e_iso, f_iso = _eval(nma['iso'], nma['vac_pos'], group)
    f_ref = f_full.copy()
    f_ref[:len(nma['ml'])] -= f_iso
    assert abs(e_mix - (e_full - e_iso)) < 1e-9
    assert np.abs(f_mix - f_ref).max() < 1e-8


def test_multipoles_match_reference(nma):
    e_mix, f_mix = _eval(nma['mixed'], nma['pos'], MULTIPOLE)
    force = next(f for f in nma['full'].getForces() if isinstance(f, mm.AmoebaMultipoleForce))
    prm = ae.from_openmm(force)
    x = torch.tensor(np.array(nma['pos'].value_in_unit(unit.nanometer)), requires_grad=True)
    e = mlmm_electrostatic_energy(x, prm, nma['ml'])
    e.backward()
    assert abs(e.item() - e_mix) < 1e-8
    assert np.abs(-x.grad.numpy() - f_mix).max() < 1e-6


def test_ml_potential_only_on_ml_region(nma):
    e_mix, f_mix = _eval(nma['mixed'], nma['pos'], ML)
    ctx = mm.Context(nma['iso_ml'], mm.VerletIntegrator(1e-3), mm.Platform.getPlatformByName('Reference'))
    ctx.setPositions(nma['vac_pos'])
    st = ctx.getState(getEnergy=True, getForces=True)
    e_iso = st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    f_iso = st.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)
    n = len(nma['ml'])
    assert abs(e_mix - e_iso) < 1e-6
    assert np.abs(f_mix[:n] - f_iso).max() < 1e-5
    assert np.abs(f_mix[n:]).max() == 0.0


def test_openmm_ml_alone_double_counts(nma):
    """Documents the gap: plain createMixedSystem masks neither the AMOEBA vdW nor the multipoles."""
    plain = openmmml.MLPotential('mace-off23-small').createMixedSystem(
        nma['top'], nma['full'], nma['ml'], forceGroup=ML, precision='double')
    _groups(plain)
    e_plain, _ = _eval(plain, nma['pos'], VDW)
    e_full, _ = _eval(nma['full'], nma['pos'], VDW)
    assert abs(e_plain - e_full) < 1e-12          # intra-NMA vdW was not removed
    e_plain, _ = _eval(plain, nma['pos'], MULTIPOLE)
    e_full, _ = _eval(nma['full'], nma['pos'], MULTIPOLE)
    assert abs(e_plain - e_full) < 1e-8           # intra-NMA electrostatics was not removed
