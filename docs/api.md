# API reference

```python
import mlmmpol as polen
```

The heavy dependencies (OpenMM, PyTorch, MACE) are imported only when a function is used.

## Building systems

### `polen.load_plugin(directory=None)`
Loads the MLPol plugin libraries (Reference and, if built, CUDA) into OpenMM. Called automatically by
`create_mupe_system`. The directory is `$MLPOL_PLUGIN_DIR` or `plugin/install/lib/plugins` in the repository.

### `polen.amber_to_amoeba(prmtop, coordinates, solutes, out_prefix, ion_types=None, cutoff=0.9, polarization='mutual')`
Converts an Amber system into an OpenMM AMOEBA system (PME, cutoff in nm).
- `solutes`: list of `(amber_residue_name, final.xyz, final.key)` with the Poltype parameters of each non-standard
  molecule.
- `ion_types`: `{amber_residue_name: amoeba_type}` for ions other than Na⁺, Cl⁻, Mg²⁺, Ca²⁺.
- Returns an `AmoebaSystem` with `topology`, `system`, `positions`, `box`, `mapping` (Amber → AMOEBA index, `None` for
  protein atoms and OPC virtual sites), `xyz`, `key` (the Tinker files written), `amber_topology`, and the method
  `atoms_of_residues(names)`.

### `polen.ml_atoms_from_residues(topology, residue_names)`
Indices of the atoms of the residues with these names (for Amber topologies or topologies that keep the names).

### `polen.create_mechanical_system(prmtop, ml_atoms, model='mace-off23-small', cutoff=1.0, charge=0, multiplicity=1, precision='single')`
Mechanical embedding on an Amber system (PME, rigid water). The MLIP replaces the bonded terms and the nonbonded
interactions inside the ML region; the ML–MM interaction stays that of the force field. Returns `(topology, system)`.

### `polen.create_mupe_system(topology, system, ml_atoms, multipoles='ml', model='mace-off23-small', charge=0, multiplicity=1, preset=None, head=None, a_cross=None, ligand=None, ml_vdw=True, layered=None, max_iterations=500, workdir='.', mts=True, root=None)`
Mutual polarizable embedding on an AMOEBA system. Returns `(system, force)` with `force` the `MLPolMultipoleForce`.

| Argument | Meaning |
|---|---|
| `multipoles` | `'amoeba'` = MuPE(FF); `'ml'` = MuPE(ML) |
| `model` | MLIP (`'mace-off23-small'`, `'mace-off23-medium'`, `'mace-omol-0-extra-large'`, ...) |
| `charge`, `multiplicity` | of the ML region |
| `preset` | MuPE(ML): name in `qm_ref/heads/presets.json` (default: the validated default) |
| `head`, `a_cross` | MuPE(ML): override the preset's head (directory with `head.pt` + `metrics.json`) and ML–MM Thole parameter |
| `ligand`, `ml_vdw` | MuPE(ML): ligand key of the vdW refit (`available_ligands()`); `ml_vdw=False` keeps the force-field vdW |
| `layered` | `(r_in, r_out)` in nm: mutual induction inside `r_in`, direct beyond `r_out`, smooth switch between |
| `max_iterations` | maximum SCF iterations of the induced dipoles |
| `workdir` | where the region-specific TorchScript head (`head.pt`) is written |
| `mts` | electrostatics and vdW in force group 1, bonded terms and MLIP in group 0 |
| `root` | repository root holding `qm_ref/heads` (default: the one of the installed package) |

### `polen.available_ligands(preset=None)`
Ligand keys with an ML–MM vdW refit in the preset.

### `polen.set_mts_groups(system)`
Puts `NonbondedForce`, `AmoebaMultipoleForce`/`MLPolMultipoleForce` and `AmoebaVdwForce` in group 1 and the rest in group 0.

## Running

### `polen.run_md(system, topology, positions, out_dir, ns=1.0, box=None, equil_ps=200.0, temperature=300.0, pressure=1.0, timestep_fs=1.0, mts=True, report_ps=1.0, checkpoint_ps=50, platform='CUDA', properties=None, minimize=True, seed=None, log=print)`
Minimization, equilibration and production with checkpoints; continues from `out_dir/restart.xml` if present. Writes
`topo.pdb`, `prod.dcd`, `energies.csv`, `final.xml`, `done`. Returns the speed in ns/day. `checkpoint_ps` is counted
in report intervals. Default platform properties for CUDA: mixed precision.

### `polen.minimize(context, tolerance=10.0, max_iterations=5000, max_displacement=0.03, log=print)`
Bounded L-BFGS (each coordinate within ±`max_displacement` nm of the start); avoids the short-range "holes" that the
OpenMM minimizer can fall into with buffered vdW + multipoles or with an MLIP outside its training range.

## `mlpolplugin.MLPolMultipoleForce`

Subclass of `openmm.AmoebaMultipoleForce`: everything of `AmoebaMultipoleForce` is available, plus:

| Method | Description |
|---|---|
| `MLPolMultipoleForce(amoeba_force)` | copy of an existing `AmoebaMultipoleForce` |
| `setMLRegion(atoms, tholeML=0.39, crossThole=-1)` | ML region (parameters injected in the lab frame); `crossThole` = ML–MM Thole parameter (< 0: AMOEBA minimum rule) |
| `setMLHead(path)` | TorchScript head: `forward(pos (M,3) nm) -> (q, mu, Theta, alpha[, N, s])`; enables MuPE(ML) |
| `setMLParametersInContext(context, q, mu, theta, alpha)` | inject ML parameters by hand (no head) |
| `setMLPenetrationInContext(context, N, s)` | Slater valence of the ML atoms by hand (empty lists disable penetration) |
| `getMLDerivatives(context)` | dict `phi` (M), `field` (M,3), `dTheta` (M,3,3), `dAlpha` (M) |
| `getInducedDipoles(context)`, `getLabFramePermanentDipoles(context)`, `getTotalDipoles(context)` | (N,3) arrays in e·nm |
| `setLayeredPolarization(centerAtoms, groupOf, groupRefAtoms, rIn, rOut, tau)`, `disableLayeredPolarization()` | layered polarization (normally via `create_mupe_system(layered=...)`) |
| `getMLAtoms()` | indices of the ML region |

## Lower-level modules

| Module | Contents |
|---|---|
| `mlmmpol.amoeba_mixed` | `create_mixed_amoeba_system` (openmm-ml mixed system with the intra-ML AMOEBA terms masked), `mask_multipoles`, `exclude_vdw` |
| `mlmmpol.mace_torchforce` | MACE as an openmm-torch `TorchForce` |
| `mlmmpol.embedding_head`, `mlmmpol.export_head` | the head architecture and its export to TorchScript for one ML region |
| `mlmmpol.ml_vdw` | `apply_ml_vdw` (per-class and ML-class × ion vdW factors) |
| `mlmmpol.layered` | `configure_layered_polarization` |
| `mlmmpol.zbl` | short-range ZBL repulsion (optional safety net) |
| `mlmmpol.amber2amoeba` | `write_system`, `load_amoeba` (used by `amber_to_amoeba`) |
| `mlmmpol.presets` | `ml_preset` |
| `ref_torch` | PyTorch reference implementation of the AMOEBA/ML electrostatics (used by the tests) |
