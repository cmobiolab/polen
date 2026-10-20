# POLEN / MuPE — the OpenMM plugin (self-contained technical summary)

> Document for discussing future directions of the implementation without the project history.
> Code in `plugin/` (C++/CUDA) and `mlmmpol/` (Python).

## 1. What it is

- **POLEN** (*POLarizable Embedding of Neural-network potentials*): an ML/MM framework for OpenMM in which a
  machine-learning interatomic potential (MLIP; currently MACE-OFF23 and MACE-OMOL-0) describes a region of interest
  embedded in the polarizable **AMOEBA** force field.
- **MuPE** (*Mutual Polarizable Embedding*): the embedding scheme. The ML region and the AMOEBA environment polarize
  each other **mutually** in a single linear system, with PME and on the GPU.
  - **MuPE(FF)** (`multipoles='amoeba'`, "level 1"): the ML region uses the force-field multipoles and
    polarizabilities (Poltype 2). Equivalent to Deep-HP/Tinker-HP (ANI-2X/AMOEBA), but in OpenMM.
  - **MuPE(ML)** (`multipoles='ml'`): at every step a network ("head") predicts the charge, dipole, quadrupole and
    Slater valence (N, s) of each ML atom, and the polarizability α(R) = α_AMOEBA × 1.10 × exp(δα(R)). It includes
    charge penetration and an ML–MM cross Thole damping. Forces come from analytic derivatives propagated through the
    head.
- Novelty with respect to the literature:
  - EMLE, ANI-MBIS and AMP (Riniker) have ML charges, multipoles and α, but in a **fixed-charge** environment;
  - Deep-HP has an AMOEBA environment, but **fixed** ML-region multipoles;
  - the EMLE group mentions "embedding EMLE in AMOEBA" as a perspective (Gradisteanu et al., *Chem. Sci.* 2026).
- License LGPL-3.0-or-later. Package names `polen` and `openmm-polen` reserved on PyPI (0.0.1 placeholders,
  repository `github.com/cmobiolab/polen`).

## 2. Implemented physics

- **Energy:** E = E_ML(R_ML) + AMOEBA bonded and vdW terms (MM–MM and ML–MM) + permanent electrostatics
  + polarization + [penetration].
- **Intra-ML masking** (the MLIP already describes the inside of the ML region):
  - zeroed: the ML–ML permanent–permanent interaction and the intra-ML permanent field that polarizes the ML region
    itself;
  - kept: the ML–ML induced–induced coupling, which gives the molecular polarizability;
  - in OpenMM: all ML–ML pairs in `Covalent12` and the ML region in a single polarization group
    (`PolarizationCovalent11`); see `mlmmpol/amoeba_mixed.py: mask_multipoles`.
- **Masked pairs beyond the real-space cutoff** (large ML regions, e.g. Mg²⁺·ATP + waters, up to 1.7 nm): they get
  the `erfc − (1−s)·nu` term at any distance, so that, added to the reciprocal part, they interact with exactly the
  scale s. This holds for the energy, the forces and the ML derivatives, on CUDA and Reference.
- **`'ml'` mode:**
  - per-ML-atom parameters: lab-frame q, μ and Θ (no local axes), α, N and s;
  - penetration: Slater valence ρ = N e^{−r/s}/(8πs³) × MM multipoles (q, μ, Θ), with closed-form radial
    coefficients in the kernel;
  - ML–MM cross Thole a_cross (0.55 in the default preset);
  - ML–MM vdW: the buffered 14-7 parameters of the ML atoms can be refitted to QM dimer curves following AMOEBA+
    (`mlmmpol/ml_vdw.py`); ML–ML vdW pairs are exclusions, so only ML–MM pairs change.
- **Forces:**
  - the energy is variational in the induced dipoles, so F = explicit part − Σ [φ ∂q/∂R − **E**·∂μ/∂R
    + ∂E/∂Θ:∂Θ/∂R + ∂E/∂α ∂α/∂R + ∂E/∂N ∂N/∂R + ∂E/∂s ∂s/∂R];
  - the kernel computes φ, **E**, ∂E/∂Θ, ∂E/∂α (including through the Thole damping), ∂E/∂N and ∂E/∂s in real
    space, reciprocal space and the PME self terms;
  - a single vector–Jacobian product (`torch::autograd::grad`) through the head closes the chain rule.
- **Layered polarization (optional):** mutual up to R, direct beyond. Per-molecule weight s = S(d), with d a smooth
  minimum of the distance to the ML region and S of class C²; induced–induced coupling × s_i s_j. The energy stays
  variational, and the forces include ∂s/∂R.
- **Exact ThreeFold:** local-frame forces come from the exact chain rule. OpenMM 8.5.2 has an error of
  8.6–11 kJ/mol/nm for this axis type.

## 3. Code architecture

- `plugin/openmmapi/`: `MLPolMultipoleForce` (subclass of `AmoebaMultipoleForce`), with XML serialization and a
  Python interface via SWIG (`mlpolplugin`). Main API:
  - `setMLRegion(atoms, tholeML=0.39, crossThole=-1)`;
  - `setMLHead(TorchScript file)`;
  - `setMLParametersInContext(ctx, q, mu, theta, alpha)`;
  - `setMLPenetrationInContext(ctx, N, s)`;
  - `getMLDerivatives(ctx)`, which returns phi, field, dTheta and dAlpha;
  - `setLayeredPolarization(...)` and `disableLayeredPolarization()`;
  - plus everything `AmoebaMultipoleForce` already has (induced dipoles, electrostatic potential, etc.).
- `plugin/platforms/reference/`: CPU kernel (copy of the OpenMM AMOEBA Reference kernel, extended).
- `plugin/platforms/cuda/`: copy of the OpenMM 8.5.2 "common" AMOEBA kernel, with:
  - the kernel sources embedded as strings (`MLPolKernelSources`) and compiled at run time by OpenMM;
  - the ML atoms marked by the **sign of the Thole parameter** (`PAIR_THOLE`);
  - own kernels in `kernels/mlpol.cc` (injection, derivatives, penetration, ML forces) and `kernels/layered.cc`.
- **Head on the GPU:**
  - runs on the same CUDA device as the context, with the PyTorch primary context pushed only while the head runs,
    as in openmm-torch;
  - `MLPOL_HEAD_DEVICE=cpu` switches back to the CPU;
  - the ML positions (M×3) and derivatives (M×14) go through the CPU, which is cheap and guarantees ordering between
    streams.
- **MLIP:** `TorchForce` (openmm-torch), through `mlmmpol/mace_torchforce.py`:
  - fixed list of all pairs and CUDA graphs (2.4× faster);
  - precomputed MACE-OMOL charge/spin embedding;
  - minimum image of the ML region through its centroid.
- **Python (`mlmmpol/`):**
  - `amoeba_mixed.py`: builds the mixed system;
  - `mechanical.py`: mechanical embedding for comparison (openmm-ml on Amber, with OPC EP exclusions and PBC-aware
    exceptions);
  - `amber2amoeba.py`: Amber → AMOEBA conversion (Tinker pdbxyz + Poltype);
  - `embedding_head.py`, `export_head.py`: head and TorchScript export;
  - `ml_vdw.py`: ML–MM vdW refit (AMOEBA+ strategy);
  - `layered.py`: layer setup;
  - `minimize.py`, `zbl.py`: robust bounded minimization and a short-range ZBL safety net.
- **Head (`embedding_head.py`):**
  - v1: SchNet-like scalar network with a triplet term (M³ cost);
  - **v2** (default): factorized pair MLP and quadrupole from products of equivariant vectors (M² cost);
  - exact total charge, outputs equivariant by construction;
  - trained on MBIS targets (ωB97M/def2-TZVPPD in vacuum), with α(R) trained against finite-field QM molecular
    polarizabilities through the intra-ML mutual Thole model.
- **Tests:** ~100 in `tests/`, comparing plugin vs OpenMM, Reference vs CUDA, the PyTorch reference (`ref_torch/`)
  and finite differences.

## 4. Validation and main numbers

- **Plugin vs OpenMM:** energy relative difference < 10⁻⁹ and forces < 10⁻⁸ (Reference); CUDA equal to Reference.
- **`'ml'` mode:**
  - forces equal to finite differences of the PME energy;
  - trained head on CUDA equal to Reference (ΔE ~10⁻⁶ kJ/mol);
  - NVE conserves energy (σ = 0.04 kJ/mol on Reference; drift ≤ 0.12 kJ/mol/ps on CUDA with MACE).
- **NMA vs QM/AMOEBA** (20 frames, ωB97M/def2-TZVPPD), interaction-energy RMSE: 8.9 kJ/mol for MuPE(FF) and
  **2.4 kJ/mol** with the MuPE(ML) physics using exact MBIS multipoles (the "ceiling" of the method).
- **Head v2:** ESP error of 2.1–2.6 kJ/mol/e, ~1% of the typical value, on unseen replicas.
- **Performance** (RTX 5070, complete MLIP + environment runs, 1/0.5 fs multiple time step), in ns/day, mechanical
  vs MuPE(FF):

  | system | atoms | mechanical | MuPE(FF) |
  |---|---|---|---|
  | pNA | 8k | 34.5 | 17.5 |
  | trypsin | 38k | 4.8 | 3.2 |
  | CK2 | 82k | 5.4 | 2.4 |

  - Electrostatics alone: the `'ml'` mode adds ~1–2 ms per step on top of the head cost.

## 5. Known limitations and pitfalls

- **No covalent boundary:** the ML region must consist of whole molecules (there are no link atoms).
- **The MLIP does not respond electronically to the environment field:** the polarization of the ML region comes
  from the Thole model.
- **Head:** for now, one network for a fixed set of ligands; trained mostly on classical geometries.
- **Ligand induction in non-uniform fields:** like AMOEBA itself, the learned α(R) overestimates the response to
  strongly non-uniform fields (e.g. a salt bridge), which biases interaction energies in such sites.
- **OpenMM 8 minimizer:** with huge energies (very strained structures) it switches to a fallback context on the CPU
  platform, where the plugin has no kernel, and fails; use `mlmmpol.minimize`.
- **ML atoms without bonds in the `System`:** OpenMM treats them as separate "molecules". When writing trajectories
  with `enforcePeriodicBox`, each one is wrapped into the box on its own (the ligand comes out "broken"; analysis has
  to reassemble it). The barostat also scales each one as a separate molecule (negligible effect).
- **CUDA precision:** `float` parameters on the GPU, as in OpenMM; the energy has ~10⁻⁴ kJ/mol noise.
- **`PythonForce` + `AmoebaMultipoleForce` on CUDA:** gives NaN (an OpenMM issue); this is why the MLIP enters as a
  `TorchForce`.
- **cuEquivariance:** does not compile in TorchScript with mace-torch 0.3.16.
- **Deployment:** binaries depend on the system glibc and CUDA. There is no conda package yet.

## 6. Future directions (already considered, open)

- **EMLE models as heads** ("EMLE inside AMOEBA"), to compare mechanical vs EMLE vs EMLE-in-AMOEBA vs MuPE(ML).
- **Head as extra outputs of the MACE model itself** (multi-head), reusing the features already computed.
- **CUDA graphs for the head;** positions and derivatives without going through the CPU.
- **Link atoms / covalent boundary** (e.g. protein residues in the ML region).
- **Free energy (FEP/alchemical)** with MuPE:
  - common basis: a **global λ in the plugin**, scaling the intra-ML masking (1 = full AMOEBA, 0 = MuPE), with
    E_intra = (1−λ)·E_AMOEBA,intra + λ·E_ML and interpolated multipoles p(λ) = (1−λ)·p_FF + λ·p_ML. The energy stays
    variational, so dE/dλ = Σ [φ·Δq − **E**·Δμ + ∂E/∂Θ:ΔΘ + ∂E/∂α·Δα] comes from the derivatives the kernel already
    computes (TI almost for free; MBAR requires solving the polarization at each λ). Test: finite differences in λ;
  - first use: an **AMOEBA → MuPE end-point correction** on top of a pure AMOEBA (Poltype) alchemical cycle. With
    MuPE(FF) the difference stays inside the ligand, so the overlap should be good. Validation: hydration free
    energies (a FreeSolv subset), AMOEBA vs MuPE(FF) vs MuPE(ML);
  - later: **absolute decoupling** with MuPE in every window. The ligand stays ML throughout and only the embedding
    is switched off (electrostatics/polarization, then vdW with soft-core and Boresch restraints). Care is needed
    with α → 0 in the Thole damping. Expensive (~3 ns/day per window);
  - `openmmtools.alchemy` does not support AMOEBA multipoles, so this would be a dedicated module (`polen.fep`, with
    pymbar).
- **τRAMD** and other unbinding-kinetics methods:
  - random force on a `CustomCentroidBondForce` of one group (the ligand), energy `-f*(ex*x1+ey*y1+ez*z1)`, with the
    direction redrawn through `context.setParameter` when the center of mass moves less than the threshold;
  - ML region = the ligand only (ML waters would diffuse away); direction changes at the outer-step boundary of the
    multiple time step; check that the layered-polarization set follows the leaving ligand (`layered.py`);
  - scientific question: does mutual polarization change relative residence times (mechanical vs AMOEBA vs MuPE(FF)
    vs MuPE(ML))? First case: trypsin–benzamidine (experimental k_off) with MuPE(FF). Benchmarks with many ligands
    (HSP90) need a transferable head.
- **Protonation state and pKa:**
  - recommended route: relative ΔpKa (site vs water) through the AMOEBA alchemical cycle + an AMOEBA → MuPE
    end-point correction in both states, each with a fixed composition (the MLIP does not accept "half an atom"). The
    MLIP error in the vacuum deprotonation energy largely cancels;
  - explicit proton transfer only **inside** the ML region (donor, acceptor and proton ML) and only in `'ml'` mode.
    It needs: MACE-OMOL validated on the reaction (scan vs ωB97M), a head trained along the path, α(R), a vdW
    reduction factor = 1 on ML hydrogens (otherwise the vdW site follows the old parent atom), ML waters held by
    flat-bottom restraints, umbrella sampling/metadynamics and a check against QM/MMPol. It gives the local
    equilibrium (neutral ⇌ ion pair), not the pKa: the proton does not leave into the solvent. The response of the ML
    region to the field remains dipolar only (no charge flux).
- **Constant-pH MD** (further away; depends on the global λ and the end-point correction):
  - sites in the MM region (AMOEBA residues): this is CpHMD with AMOEBA. Independent of ML, but requires switching
    multipoles, α and vdW between states (with "ghost" H). Review the literature on CpHMD with polarizable force
    fields (Drude, AMOEBA);
  - site in the ML region: discrete MC between the states, with the deprotonated H as a ghost particle. Changing the
    charge in explicit solvent has low acceptance, so this would be NCMC, interpolating E_ML = (1−λ)·E_ML(prot) +
    λ·E_ML(deprot) (two MLIP evaluations only during switching) and the head multipoles in both states. It needs a
    reference potential calibrated on the compound in water and a co-titrating ion or water to keep the PME system
    neutral;
  - only worthwhile when protonation couples to conformation. When it does not, the free-energy pKa is enough.
- **Layered polarization** in production for very large systems; dynamic selection of the mutual region.
- **High-level openmm-ml-style interface:**
  `MLPolPotential(model).createMixedSystem(top, system, ml_atoms, multipoles='amoeba'|'ml', head=..., polarization='mutual'|'layered', ...)`.
- **Other platforms** (OpenCL/HIP); proposing *hooks* in OpenMM itself so that a copy of the AMOEBA kernel does not
  have to be maintained.
- **Other polarizable force fields** with the same format (AMOEBA+, HIPPO?), environment-dependent ML–MM vdW.
