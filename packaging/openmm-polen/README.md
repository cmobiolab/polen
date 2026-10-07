# openmm-polen

**POLEN** — *POLarizable Embedding of Neural-network potentials*.

POLEN is an ML/MM framework for [OpenMM](https://openmm.org) in which a machine-learning interatomic potential (e.g. MACE)
describes a region of interest embedded in the AMOEBA polarizable force field, with **mutual polarization** between
the two regions (MuPE, *Mutual Polarizable Embedding*), periodic boundary conditions (PME) and GPU support. In the MuPE(ML)
variant, atomic multipoles, valence (charge-penetration) parameters and polarizabilities of the ML region are predicted by
a neural network at every step.

This is a **placeholder release** that reserves the package name while the code and the accompanying paper are being
finalized. The first functional release will be announced at https://github.com/cmobiolab/polen.

License: GNU Lesser General Public License v3.0 or later (LGPL-3.0-or-later).
