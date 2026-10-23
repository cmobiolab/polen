# User guide

This guide follows the path used for the systems of the paper: an Amber system (built with tleap), Poltype parameters
for the ligand, conversion to AMOEBA, the ML/MM system and the simulation. Every step has a runnable counterpart in
[`examples/`](examples.md).

```python
import mlmmpol as polen      # the package is called mlmmpol; polen is just a short alias
```

## 1. Inputs

| What | How it was made | Used by |
|---|---|---|
| `system.prmtop`, `system.rst7` | tleap (e.g. ff19SB + OPC + GAFF2/RESP ligand; see `examples/data/pna/pna_tleap.in`), then equilibration | all embeddings |
| `final.xyz`, `final.key` of each ligand | Poltype 2 (AMOEBA parameters of the ligand) | MuPE(FF), MuPE(ML) |
| a trained head + preset covering the ligand | shipped presets ([presets.md](presets.md)) | MuPE(ML) |

The mechanical embedding only needs the Amber files.

## 2. Mechanical embedding

```python
from openmm import app
top = app.AmberPrmtopFile('system.prmtop').topology
ml = polen.ml_atoms_from_residues(top, ['LIG'])
topology, system = polen.create_mechanical_system('system.prmtop', ml, model='mace-off23-small', charge=0)
polen.set_mts_groups(system)                    # fast forces (bonded + MLIP) in group 0, the rest in group 1
crd = app.AmberInpcrdFile('system.rst7')
polen.run_md(system, topology, crd.getPositions(), 'out/mech', ns=5, box=crd.getBoxVectors())
```

For charged ML regions or ions use `model='mace-omol-0-extra-large'` and set `charge` (and `multiplicity`).

## 3. Converting to AMOEBA

```python
amo = polen.amber_to_amoeba('system.prmtop', 'system.rst7', [('LIG', 'final.xyz', 'final.key')], 'out/amoeba')
ml = amo.atoms_of_residues(['LIG'])             # select by the Amber residue names
```

- The ligand is typed with its Poltype parameters (same atom order as the mol2/sdf given to Poltype); water and the
  common ions (Na⁺, Cl⁻, Mg²⁺, Ca²⁺) with AMOEBA BIO18; protein residues with Tinker's `pdbxyz` (set `MLMMPOL_PDBXYZ` and
  `MLMMPOL_BIO18`, [installation.md](installation.md)). The OPC virtual sites are dropped, and the Amber box is kept.
- The result keeps the Tinker files (`amo.xyz`, `amo.key`), the AMOEBA system, positions and box, and the mapping from
  Amber to AMOEBA indices (`amo.mapping`). The protein atom order is Tinker's, so use `atoms_of_residues` (or the
  mapping) rather than Amber indices.
- Systems already in AMOEBA (Tinker files, or OpenMM's `ForceField('amoeba2018.xml')`) can go directly to step 4.

## 4. MuPE(FF) and MuPE(ML)

```python
# MuPE(FF): force-field multipoles on the ML region
system, force = polen.create_mupe_system(amo.topology, amo.system, ml, multipoles='amoeba')

# MuPE(ML): learned multipoles, penetration and polarizability, refitted ML–MM vdW
system, force = polen.create_mupe_system(amo.topology, amo.system, ml, multipoles='ml', ligand='pna',
                                         charge=0, workdir='out/mupe_ml')
```

Main options of `create_mupe_system` (full list in [api.md](api.md)):

- `model`, `charge`, `multiplicity`: the MLIP of the ML region;
- `preset`: head + coupling + vdW refit (default: the validated default, [presets.md](presets.md)); `head` and `a_cross`
  override the preset's;
- `ligand`: which ligand of the preset's vdW refit to use (`polen.available_ligands()`); `ml_vdw=False` keeps the
  force-field vdW;
- `layered=(r_in, r_out)`: layered polarization (nm), for large systems;
- `mts=True`: fast forces in group 0 (for the multiple-time-step integrator of `run_md`).

The function returns the new `System` and its `MLPolMultipoleForce`, which you can use to inspect the state (step 6).
`workdir` receives `head.pt`, the TorchScript head specialised to this ML region (atoms, charge, base polarizabilities).

## 5. Running

```python
polen.run_md(system, amo.topology, amo.positions, 'out/mupe_ml', ns=5, box=amo.box, equil_ps=200,
             temperature=300, pressure=1.0, timestep_fs=1.0, mts=True, platform='CUDA')
```

- Bounded minimization (`mlmmpol.minimize`, robust to the short-range "holes" of buffered vdW + multipoles), then
  equilibration (`equil_ps`, not saved), then production with one frame and one line of `energies.csv` per
  `report_ps`.
- Multiple time step: outer step `timestep_fs` for the electrostatics and vdW, half of it for the bonded terms and the
  MLIP (MTS Langevin, 1 ps⁻¹). `pressure=None` gives NVT.
- **Restarts**: a checkpoint (`restart.xml`) is written every `checkpoint_ps`; calling `run_md` again with the same
  `out_dir` continues from it (trajectory and CSV are cut at the checkpoint). This is how runs longer than a queue
  limit are chained.
- Outputs: `topo.pdb`, `prod.dcd`, `energies.csv`, `final.xml`, `done` (speed). Typical speeds on an RTX 5070:
  pNA in water (8k atoms) 34 / 15 / 9 ns/day (mechanical / MuPE(FF) / MuPE(ML)); trypsin (38k) 4.8 / 3.2 / 2.8;
  CK2 (82k) 5.4 / 2.4 / 2.2.

## 6. Inspecting a state

```python
import openmm as mm
ctx = mm.Context(system, mm.VerletIntegrator(1e-3), mm.Platform.getPlatformByName('CUDA'))
ctx.setPeriodicBoxVectors(*amo.box); ctx.setPositions(amo.positions)
der = force.getMLDerivatives(ctx)          # dict: phi (M), field (M,3), dTheta (M,3,3), dAlpha (M), kJ/mol units
ind = force.getInducedDipoles(ctx)         # (N,3) e·nm; also getLabFramePermanentDipoles, getTotalDipoles
```

Energy by component: group 0 = bonded + MLIP, group 1 = electrostatics + polarization + vdW
(`ctx.getState(getEnergy=True, groups={1})`).

**Trajectories:** in the mechanical embedding the ML atoms have no bonds in the `System`, so OpenMM wraps them into the
box one by one when writing the DCD; reassemble the ligand (minimum image) before analysis.

## 7. Command line

The same workflow without writing Python:

```bash
python -m mlmmpol amber2amoeba system.prmtop system.rst7 out/amoeba --solute LIG final.xyz final.key
python -m mlmmpol run --embedding mupe-ml --prmtop system.prmtop --coords system.rst7 --ml-residues LIG \
    --solute LIG final.xyz final.key --ligand pna --ns 5 --out out/mupe_ml
```

`python -m mlmmpol run -h` lists all options (embedding, model, charge, preset, layers, length, temperature,
pressure, platform and device). Running the same command again continues an interrupted run.
