# Presets and supported ligands

MuPE(ML) needs a trained embedding head for the ligand. The heads that ship with POLEN (`qm_ref/heads/`) were trained
for the ligands of the paper; a *preset* (`qm_ref/heads/presets.json`) combines a head with its coupling and vdW refit.

## Supported ligands

| Ligand | Charge | Vdw refit key (`ligand=`) | MLIP used in the paper |
|---|---|---|---|
| p-nitroaniline (pNA) | 0 | `pna` | `mace-off23-small` |
| benzamidinium | +1 | `ben` | `mace-omol-0-extra-large` |
| 4,5,6,7-tetrabromobenzotriazole anion (TBB⁻) | −1 | `tbb` | `mace-omol-0-extra-large` |

The heads were trained on 1500 conformations of these ligands (classical MD and ML/MM windows; see the paper's SI).
Other molecules require training a new head (MBIS targets in vacuum + finite-field polarizabilities) and, for the
best ML–MM contacts, a vdW refit against QM dimer curves; the training and fitting scripts are not part of 0.1.0.

## Presets

| Name | Head | `a_cross` | ML–MM vdW | Note |
|---|---|---|---|---|
| **`alpha_s0_vdw1`** (default) | `v2_final2_alpha_s0` (learned α(R)) | 0.55 | `ml_vdw_v1.json` (ligand–water) | validated default |
| `alpha_s0_vdw2` | same | 0.55 | `ml_vdw_v2.json` (ligand–water + ML–Na⁺ pairs) | adds the Na⁺ contact correction |
| `alpha_s0` | same | 0.55 | none (force-field vdW) | |
| `noalpha_s1` | `v2_final2_noalpha_s1` (α = 1.10 α_AMOEBA) | 0.57 | none | fixed polarizability |
| `alpha_s0_a070` | `v2_final2_alpha_s0` | 0.70 | none | NMA coupling (overbinds CK2) |
| `v2_old` | `v2_f128_b3_s0` | 0.70 | none | earlier head |

- **`a_cross`** is the Thole damping parameter of the ML–MM pairs. 0.55 minimizes the error of the interaction energy
  against QM/AMOEBA over pNA, trypsin and CK2 (MAE ≈ 5 kJ/mol).
- **ML–MM vdW refit** (AMOEBA+ strategy): the buffered 14-7 parameters of the ML atoms are scaled per atom class to
  reproduce QM ligand–water dimer curves; `v2` adds pair-specific factors for ML atoms in contact with Na⁺ (the
  triazolate N of TBB⁻ and the nitro O of pNA). Ligand–Cl⁻ contacts are not refitted.
- Use `preset='alpha_s0_vdw2'` for systems where a Na⁺ ion can reach the ligand's N/O acceptors.

## Using your own head

`create_mupe_system(..., head='path/to/head_dir', a_cross=...)` uses any head directory with `head.pt` and
`metrics.json` in the format of `mlmmpol.embedding_head` (see `mlmmpol/export_head.py`). The plugin also accepts any
TorchScript module with `forward(pos: (M,3) nm, float64) -> (q (M), mu (M,3), Theta (M,3,3), alpha (M)[, N (M), s (M)])`
through `MLPolMultipoleForce.setMLHead`.
