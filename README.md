# POLEN

**POLEN** — *POLarizable Embedding of Neural-network potentials* — is an ML/MM framework for
[OpenMM](https://openmm.org) in which a machine-learning interatomic potential (e.g. MACE) describes a region of
interest embedded in the AMOEBA polarizable force field.

Its core is **MuPE** (*Mutual Polarizable Embedding*):

- the ML region and the polarizable AMOEBA environment polarize each other self-consistently (mutual induction),
  with particle-mesh Ewald and GPU (CUDA) support;
- **MuPE(FF)**: the ML region carries the force-field multipoles and polarizabilities;
- **MuPE(ML)**: atomic charges, dipoles, quadrupoles, valence (charge-penetration) parameters and polarizabilities of
  the ML region are predicted by a neural network at every step, with analytic forces.

## Status

Under active development. The code and the accompanying paper are being finalized; this repository currently holds
placeholder packages that reserve the `polen` and `openmm-polen` names on PyPI.

## License

GNU Lesser General Public License v3.0 or later (LGPL-3.0-or-later); see `COPYING.LESSER` and `COPYING`.
