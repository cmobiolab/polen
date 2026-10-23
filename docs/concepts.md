# Concepts

## ML/MM and the embedding problem

In ML/MM the energy of a system is split into a region of interest described by an MLIP (here MACE: MACE-OFF23 for
neutral organic molecules, MACE-OMOL-0 for ions and metal complexes) and an environment described by a force field. The
*embedding* is how the two interact. The MLIP gives the internal energy of the ML region in vacuum; it does not see the
electric field of the environment. Everything that depends on the environment—electrostatics, polarization, van der
Waals—has to come from the embedding.

## The three embeddings

**Mechanical** (`create_mechanical_system`). The ML–MM interaction is the force field's own: fixed point charges
(RESP/GAFF2) on the ML atoms interacting with a fixed-charge environment (Amber), plus Lennard-Jones. This is the
standard openmm-ml protocol. Neither side polarizes.

**MuPE(FF)** (`create_mupe_system(..., multipoles='amoeba')`). The environment is AMOEBA: permanent multipoles up to
quadrupoles and atomic polarizabilities, with mutual induction solved self-consistently with PME. The ML region keeps
its force-field (Poltype) multipoles and polarizabilities and takes part in the same mutual induction. Inside the ML
region the permanent electrostatics and vdW are removed (the MLIP already contains them), but the induced–induced
coupling is kept, so the ML region has the right molecular polarizability.

**MuPE(ML)** (`create_mupe_system(..., multipoles='ml')`). As MuPE(FF), but the permanent electrostatics of the ML region
are predicted at every step by an *embedding head* from the current geometry: charge q, dipole μ and quadrupole Θ of
each atom (in the lab frame), the Slater valence population N and width s (for charge penetration: the ML electron
cloud overlapping with the environment's multipoles), and the atomic polarizabilities α(R). The ML–MM Thole damping is
`a_cross`, and the ML–MM vdW is refitted to QM dimer curves (AMOEBA+ strategy). The forces include the chain rule
through the head.

## What the plugin does

`MLPolMultipoleForce` (the OpenMM plugin, Reference and CUDA platforms) is a subclass of `AmoebaMultipoleForce`. For
MuPE(FF) it behaves exactly like it (it also fixes the forces of the ThreeFold local-frame type, which are not the
gradient of the energy in OpenMM 8.5.2). For MuPE(ML) it additionally:

1. runs the TorchScript head (on the GPU) with the ML positions;
2. injects q, μ, Θ, α of the ML atoms and the Slater valence (N, s);
3. solves the mutual induction of the whole system and computes energy and forces, including charge penetration;
4. computes the derivatives of the energy with respect to the injected parameters (φ = ∂E/∂q, **E** = −∂E/∂μ,
   ∂E/∂Θ, ∂E/∂α, ∂E/∂N, ∂E/∂s) and adds the chain rule through the head (one vector–Jacobian product) to the forces.

The energy is variational in the induced dipoles, so these derivatives are exact. Optional *layered polarization*
(`layered=(r_in, r_out)`) makes the induction mutual only near the ML region and direct beyond, with a smooth switch.

## The embedding head and presets

The head is a small equivariant network trained on MBIS multipoles and Slater valence (ωB97M/def2-TZVPPD, vacuum) of
the ligand conformations, and on QM molecular polarizabilities (finite field). A head is trained for a **set of
ligands**: it can only be used for molecules of that set. A *preset* bundles a head with its coupling (`a_cross`, the
ML Thole parameter) and, optionally, an ML–MM vdW refit per ligand. See [presets.md](presets.md).

## Units and conventions

OpenMM units throughout (nm, kJ/mol, e). Multipoles: dipoles in e·nm, quadrupoles in e·nm² in the Tinker/OpenMM
convention (traceless, Buckingham/3), polarizabilities in nm³. The ML region must be made of whole molecules (no
covalent ML/MM boundary).
