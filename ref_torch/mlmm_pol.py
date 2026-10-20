"""ML/MM electrostatics with mutual polarizable embedding (reference implementation).

The ML region (indices `ml`) enters the same AMOEBA electrostatics, with two changes:

1. Intra-ML masking: for ML–ML pairs the m (permanent–permanent), p and d (permanent -> induced) scales are
   zeroed; u = 1 is kept, because the intra-ML induced–induced coupling is what gives the molecular
   polarizability. In OpenMM this amounts to marking the ML–ML pairs as Covalent12 + PolarizationCovalent11.

2. ML-predicted parameters (optional): on the ML atoms, q, mu (lab frame, NoAxisType), a zero quadrupole and
   alpha (with damping = alpha^(1/6)) replace the AMOEBA ones. Without ML parameters, the ML atoms keep the
   AMOEBA multipoles: this is the multipolar mechanical embedding (level 1).

Forces:
    F = -dE/dR|_params - [phi . dq/dR - E . dmu/dR + dE/dalpha . dalpha/dR]
with phi = dE/dq and E = -dE/dmu provided by `potential_field_at_ml` (what the plugin exports).
"""

from dataclasses import replace

import numpy as np
import torch

from .amoeba_elec import (NO_AXIS_TYPE, ONE_4PI_EPS0, electrostatic_energy, induced_potential_field,
                          lab_frame_multipoles, permanent_field, permanent_potential)

AMOEBA_THOLE = 0.39


def ml_pair_mask(n, ml):
    """Boolean (N,N) matrix, true for ML–ML pairs (i != j)."""
    is_ml = np.zeros(n, dtype=bool)
    is_ml[ml] = True
    mask = is_ml[:, None] & is_ml[None, :]
    np.fill_diagonal(mask, False)
    return torch.as_tensor(mask)


def mask_ml_region(prm, ml):
    """Copy of prm with m, p, d zeroed for ML–ML pairs (u kept)."""
    mask = ml_pair_mask(prm.n, ml)
    zero = lambda s: torch.where(mask, torch.zeros_like(s), s)
    return replace(prm, m_scale=zero(prm.m_scale), p_scale=zero(prm.p_scale),
                   d_scale=zero(prm.d_scale))


def with_ml_parameters(prm, ml, q, mu, alpha, thole=AMOEBA_THOLE, theta=None):
    """Copy of prm with the ML parameters on the `ml` atoms (differentiable in q, mu, alpha, theta).

    q (M,), mu (M,3) in the lab frame in e*nm, alpha (M,) in nm^3; optional theta (M,3,3), lab frame, in e*nm^2
    (Tinker/OpenMM convention); without theta, zero quadrupole.
    """
    t = torch.as_tensor(np.asarray(ml))
    axis = prm.axis_type.copy()
    axis[ml] = NO_AXIS_TYPE
    none = lambda a: np.where(np.isin(np.arange(prm.n), ml), -1, a)
    return replace(
        prm,
        charge=prm.charge.index_put((t,), q),
        dipole=prm.dipole.index_put((t,), mu),
        quadrupole=prm.quadrupole.index_put((t,), torch.zeros(len(ml), 3, 3, dtype=q.dtype) if theta is None else theta),
        axis_type=axis, atom_z=none(prm.atom_z), atom_x=none(prm.atom_x), atom_y=none(prm.atom_y),
        polarity=prm.polarity.index_put((t,), alpha),
        damping=prm.damping.index_put((t,), alpha ** (1 / 6)),
        thole=prm.thole.index_put((t,), torch.full_like(alpha, thole)),
    )


def mlmm_electrostatic_energy(pos, prm, ml, q=None, mu=None, alpha=None, thole=AMOEBA_THOLE,
                              damp_ml_mm=True, a_cross=None, theta=None, return_terms=False):
    """ML/MM electrostatic energy (kJ/mol).

    Without q/mu/alpha: multipolar mechanical embedding (level 1, AMOEBA multipoles on the ML region).
    With q/mu/alpha: mutual polarizable embedding with ML parameters.
    Does not include E_MACE or vdW. Thole damping for ML–MM pairs: damp_ml_mm=False turns it off; a_cross sets
    the parameter a of these pairs (a -> infinity = undamped; AMOEBA uses 0.39).
    """
    p = mask_ml_region(prm, ml)
    if q is not None:
        p = with_ml_parameters(p, ml, q, mu, alpha, thole, theta)
    if not damp_ml_mm:
        # no Thole for ML–MM pairs ("bare" cross fields, as in QM/MM); kept within each region
        is_ml = torch.zeros(prm.n, dtype=torch.bool)
        is_ml[torch.as_tensor(np.asarray(ml))] = True
        p = replace(p, undamped=is_ml[:, None] ^ is_ml[None, :])
    elif a_cross is not None:
        is_ml = torch.zeros(prm.n, dtype=torch.bool)
        is_ml[torch.as_tensor(np.asarray(ml))] = True
        a = torch.minimum(p.thole[:, None], p.thole[None, :])
        p = replace(p, thole_pair=torch.where(is_ml[:, None] ^ is_ml[None, :], torch.as_tensor(a_cross, dtype=a.dtype), a))
    e, terms = electrostatic_energy(pos, p, return_terms=True)
    terms['prm'] = p
    return (e, terms) if return_terms else e


def potential_field_at_ml(pos, terms, ml):
    """phi = dE/dq and E = -dE/dmu at the ML sites (kJ/mol/e and kJ/mol/(e*nm)).

    Explicit, from the converged quantities: this is the electrostatics engine -> PyTorch interface.
      phi_i = sum_j m_ij phi_perm,j(r_i) + 1/2 sum_j lambda3 [d_ij mu_p,j + p_ij mu_d,j] . d_ij / r^3
      E_i   = sum_j m_ij E_perm,j(r_i)   + 1/2 sum_j T_ij(lambda) [d_ij mu_p,j + p_ij mu_d,j]
    """
    p = terms['prm']
    mu, theta = terms['mu_lab'], terms['theta_lab']
    phi = permanent_potential(pos, p, mu, theta, p.m_scale)
    E = permanent_field(pos, p, mu, theta, p.m_scale, damped=False)
    phi_i, E_i = induced_potential_field(pos, p, [(terms['mu_ind_p'], p.d_scale),
                                                  (terms['mu_ind'], p.p_scale)])
    phi = ONE_4PI_EPS0 * (phi + 0.5 * phi_i)
    E = ONE_4PI_EPS0 * (E + 0.5 * E_i)
    return phi[ml], E[ml]
