# POLEN manual

**POLEN** (*POLarizable Embedding for Neural-network potentials*) runs ML/MM simulations in OpenMM in which a
machine-learning interatomic potential (MLIP; MACE) describes a region of interest embedded in the polarizable AMOEBA
force field, with **mutual** polarization between the two (MuPE).

| Page | Contents |
|---|---|
| [Installation](installation.md) | environment, building the plugin, paths, testing |
| [Concepts](concepts.md) | the three embeddings, what MuPE computes, the embedding head, presets |
| [User guide](user_guide.md) | from Amber/Poltype files to a running simulation, restarts, analysis, command line |
| [API reference](api.md) | `mlmmpol` functions and the `MLPolMultipoleForce` class |
| [Presets and supported ligands](presets.md) | the trained heads, couplings and vdW refits that ship with the code |
| [Examples](examples.md) | the scripts in `examples/`, what they show and how long they take |
| [Limitations](limitations.md) | what the method and the code do not do (yet) |
| [Technical summary](plugin.md) | physics and implementation of the plugin, validation, future directions |

## The three embeddings in one table

| | Mechanical | MuPE(FF) | MuPE(ML) |
|---|---|---|---|
| environment | fixed-charge (Amber) | polarizable AMOEBA | polarizable AMOEBA |
| ML-region electrostatics | force-field charges (RESP) | force-field multipoles + polarizabilities (Poltype) | multipoles, Slater valence and polarizability predicted by a neural head at every step |
| polarization ML ↔ MM | no | mutual | mutual, with charge penetration |
| function | `create_mechanical_system` | `create_mupe_system(..., multipoles='amoeba')` | `create_mupe_system(..., multipoles='ml')` |

## Minimal example

```python
import mlmmpol as polen

amoeba = polen.amber_to_amoeba('pna.prmtop', 'pna.rst7', [('PNA', 'final.xyz', 'final.key')], 'work/amoeba')
ml = amoeba.atoms_of_residues(['PNA'])
system, force = polen.create_mupe_system(amoeba.topology, amoeba.system, ml, multipoles='ml', ligand='pna',
                                         workdir='work')
polen.run_md(system, amoeba.topology, amoeba.positions, 'work', ns=1.0, box=amoeba.box)
```

Version: see the `VERSION` file; changes in [`CHANGELOG.md`](../CHANGELOG.md).
