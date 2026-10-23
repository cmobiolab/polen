"""High-level interface of POLEN: build ML/MM systems with mechanical or mutually polarizable (MuPE) embedding.

Typical use (see docs/user_guide.md and examples/):

    import mlmmpol as polen
    polen.load_plugin()
    amoeba = polen.amber_to_amoeba('lig.prmtop', 'lig.rst7', [('LIG', 'final.xyz', 'final.key')], 'work/amoeba')
    ml = amoeba.atoms_of_residues(['LIG'])
    system, force = polen.create_mupe_system(amoeba.topology, amoeba.system, ml, multipoles='ml', ligand='pna')

The three embeddings:
  - 'mechanical': create_mechanical_system (Amber/GAFF2 environment, ML–MM interaction from the force field);
  - MuPE(FF):     create_mupe_system(..., multipoles='amoeba') (AMOEBA environment, force-field multipoles and
                  polarizabilities on the ML region, mutual polarization);
  - MuPE(ML):     create_mupe_system(..., multipoles='ml') (the same, with ML-predicted multipoles, charge penetration
                  and polarizability from an embedding head, and the refitted ML–MM vdW).
"""
import json
import os
from collections import namedtuple
from pathlib import Path

import openmm as mm
from openmm import unit

from .amber2amoeba import load_amoeba, write_system
from .amoeba_mixed import create_mixed_amoeba_system
from .mechanical import create_mechanical_system, set_mts_groups  # noqa: F401  (re-exported)
from .presets import ml_preset

ROOT = Path(__file__).resolve().parents[1]
class AmoebaSystem(namedtuple('AmoebaSystem', 'topology system positions box mapping xyz key amber_topology')):
    """Result of amber_to_amoeba. The AMOEBA topology (from TinkerFiles) does not keep the Amber residue names; use
    atoms_of_residues to select atoms by their Amber residue names (e.g. the ML region)."""

    def atoms_of_residues(self, residue_names):
        """AMOEBA indices of the atoms of the Amber residues with these names (non-protein residues, e.g. ligands)."""
        idx = [self.mapping[i] for i in ml_atoms_from_residues(self.amber_topology, residue_names)]
        if any(i is None for i in idx):
            raise ValueError('some of these residues are protein residues or virtual sites (not mapped)')
        return idx
_PLUGIN_LOADED = False


def plugin_dir():
    """Directory with the MLPol plugin libraries: $MLPOL_PLUGIN_DIR, otherwise plugin/install/lib/plugins in the repository."""
    env = os.environ.get('MLPOL_PLUGIN_DIR')
    return Path(env) if env else ROOT / 'plugin' / 'install' / 'lib' / 'plugins'


def load_plugin(directory=None):
    """Load the MLPol plugin libraries (Reference and, if present, CUDA) into OpenMM. Safe to call more than once."""
    global _PLUGIN_LOADED
    if _PLUGIN_LOADED:
        return
    d = Path(directory) if directory else plugin_dir()
    libs = [d / 'libMLPolPluginReference.so', d / 'libMLPolPluginCUDA.so']
    if not libs[0].exists():
        raise FileNotFoundError(f'MLPol plugin not found in {d} (build it with plugin/build.sh or set MLPOL_PLUGIN_DIR)')
    for lib in libs:
        if lib.exists():
            mm.Platform.loadPluginLibrary(str(lib))
    _PLUGIN_LOADED = True


def ml_atoms_from_residues(topology, residue_names):
    """Indices of the atoms of the residues with the given names (e.g. the ligand)."""
    names = set(residue_names)
    return [a.index for a in topology.atoms() if a.residue.name in names]


def amber_to_amoeba(prmtop, coordinates, solutes, out_prefix, ion_types=None, cutoff=0.9, polarization='mutual'):
    """Convert an Amber system (prmtop + inpcrd/rst7) into an OpenMM AMOEBA system.

    solutes: list of (Amber residue name, Poltype final.xyz, Poltype final.key) for each non-standard molecule (ligands);
    protein residues are typed with Tinker's pdbxyz (AMOEBA BIO18; set MLMMPOL_BIO18 and MLMMPOL_PDBXYZ), water and
    the common ions with amoebabio18. The Tinker .xyz/.key are written as out_prefix.xyz/.key.
    Returns AmoebaSystem(topology, system, positions, box, mapping, xyz, key, amber_topology); mapping[i_amber] is the
    AMOEBA index of each non-protein Amber atom (None for protein atoms and the dropped OPC virtual sites); select the
    ML region with result.atoms_of_residues(['LIG']).
    """
    Path(out_prefix).parent.mkdir(parents=True, exist_ok=True)
    from openmm import app
    xyz, key, box, mapping = write_system(prmtop, coordinates, solutes, str(out_prefix), ion_types=ion_types)
    tk, system = load_amoeba(xyz, key, box, cutoff=cutoff, polarization=polarization)
    return AmoebaSystem(tk.topology, system, tk.getPositions(), box, mapping, xyz, key,
                        app.AmberPrmtopFile(str(prmtop)).topology)


def available_ligands(preset=None, root=None):
    """Ligands with refitted ML–MM vdW in the given preset (names accepted by create_mupe_system(..., ligand=))."""
    p = ml_preset(preset, root)
    if not p.get('ml_vdw'):
        return []
    fit = json.loads((Path(root or ROOT) / 'qm_ref' / 'heads' / p['ml_vdw']).read_text())
    return sorted(fit['scale_R'])


def create_mupe_system(topology, system, ml_atoms, multipoles='ml', model='mace-off23-small', charge=0, multiplicity=1,
                       preset=None, head=None, a_cross=None, ligand=None, ml_vdw=True, layered=None,
                       max_iterations=500, workdir='.', mts=True, root=None):
    """ML/MM system with mutual polarizable embedding (MuPE) on an OpenMM AMOEBA system.

    topology, system: the full AMOEBA system (e.g. from amber_to_amoeba, TinkerFiles or ForceField('amoeba2018.xml')).
    ml_atoms:   indices of the ML region (whole molecules; no covalent boundary).
    multipoles: 'amoeba' for MuPE(FF) (force-field multipoles and polarizabilities on the ML region) or 'ml' for
                MuPE(ML) (multipoles, Slater valence and polarizability predicted by the embedding head at every step).
    model:      MLIP of the ML region (openmm-ml name, e.g. 'mace-off23-small' or 'mace-omol-0-extra-large').
    charge, multiplicity: of the ML region (used by MACE-OMOL and by the head).
    MuPE(ML) only:
      preset:   name in qm_ref/heads/presets.json (default: the validated default), giving the head, a_cross, the
                ML Thole parameter and the ML–MM vdW refit;
      head:     path of a trained head (overrides the preset's); a_cross: ML–MM Thole parameter (overrides the preset's);
      ligand:   key of the ligand in the preset's vdW refit (see available_ligands); ml_vdw=False skips the refit.
    layered:    (r_in, r_out) in nm for layered polarization (mutual inside r_in, direct beyond r_out), or None.
    workdir:    where the TorchScript head for this ML region is written (head.pt).
    mts:        put the slow forces (electrostatics, vdW) in force group 1 for a multiple-time-step integrator.
    Returns (system, MLPolMultipoleForce).
    """
    from openmmml import MLPotential
    import mlpolplugin
    load_plugin()
    if multipoles not in ('amoeba', 'ml'):
        raise ValueError("multipoles must be 'amoeba' (MuPE(FF)) or 'ml' (MuPE(ML))")
    ml = [int(i) for i in ml_atoms]
    mixed = create_mixed_amoeba_system(MLPotential(model), topology, system, ml, force_group=0, charge=charge,
                                       multiplicity=multiplicity)
    for i in reversed(range(mixed.getNumForces())):
        if isinstance(mixed.getForce(i), mm.CMMotionRemover):
            mixed.removeForce(i)
    i = next(k for k, f in enumerate(mixed.getForces()) if isinstance(f, mm.AmoebaMultipoleForce))
    amoeba = mixed.getForce(i)
    alpha_ml = [amoeba.getMultipoleParameters(a)[-1].value_in_unit(unit.nanometer ** 3) for a in ml]
    force = mlpolplugin.MLPolMultipoleForce(amoeba)
    force.setMutualInducedMaxIterations(max_iterations)
    if multipoles == 'ml':
        from .export_head import export_head
        p = ml_preset(preset, root)
        z = [a.element.atomic_number for a in topology.atoms()]
        out = Path(workdir) / 'head.pt'
        out.parent.mkdir(parents=True, exist_ok=True)
        export_head(head or p['head'], [z[k] for k in ml], charge, alpha_ml, out)
        force.setMLRegion(ml, p['thole_ml'], p['a_cross'] if a_cross is None else a_cross)
        force.setMLHead(str(out))
        if ml_vdw and p.get('ml_vdw'):
            from .ml_vdw import apply_ml_vdw
            fit = json.loads((Path(root or ROOT) / 'qm_ref' / 'heads' / p['ml_vdw']).read_text())
            if ligand not in fit['scale_R']:
                raise ValueError(f"preset '{p['name']}' has an ML–MM vdW refit for {sorted(fit['scale_R'])}; pass "
                                 f"ligand=<one of them>, or ml_vdw=False to keep the force-field vdW")
            apply_ml_vdw(mixed, topology, ml, fit['scale_R'][ligand], fit['scale_eps'][ligand],
                         fit.get('pair_scale_R', {}).get(ligand), fit.get('pair_scale_eps', {}).get(ligand))
    if layered:
        from .layered import configure_layered_polarization
        configure_layered_polarization(force, topology, ml, *layered)
    mixed.removeForce(i)
    mixed.addForce(force)
    if mts:
        set_mts_groups(mixed)
    return mixed, force
