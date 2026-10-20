"""Explicit derivatives of the AMOEBA energy with respect to alpha and to the mutual -> direct switching weights.

For any parameter theta, with symmetric A = alpha^-1 - T~, mu_d = A^-1 E_d, mu_p = A^-1 E_p and
E = E_perm - 1/2 mu_d . E_p (energy variational in the dipoles):

    dE/dtheta = dE_perm/dtheta - 1/2 [dE_d/dtheta . mu_p + mu_d . dE_p/dtheta] + 1/2 mu_d . (dA/dtheta) mu_p

(a) alpha_k (sites with damping = alpha^(1/6)): alpha enters the diagonal of A and the Thole damping
    (dipole-dipole tensor and permanent fields). With rho = a r^3 / sqrt(alpha_i alpha_j),
    d rho / d alpha_k = -rho / (2 alpha_k), and each factor lambda_n becomes g_n = rho * d lambda_n / d rho:
        g3 = rho e^-rho,  g5 = rho^2 e^-rho,  g7 = (0.6 rho^3 - 0.2 rho^2) e^-rho.
(b) weight s_m (u_ij -> u_ij s_i s_j): dE/ds_m = -1/2 sum_{i in m} (mu_d,i . F^p_i + mu_p,i . F^d_i),
    F^x_i = sum_j u_ij s_j T~_ij mu_x,j.

These are the quantities the plugin kernel has to export/compute (one extra pass over the pairs).
Everything in kJ/mol (ONE_4PI_EPS0 factor included).
"""

import torch

from .amoeba_elec import ONE_4PI_EPS0, _pair_geometry, thole_factors


def thole_rho_derivs(r, prm, eye):
    """g3, g5, g7 = rho * d(lambda_n)/d(rho) (zero for undamped pairs)."""
    damp = prm.damping[:, None] * prm.damping[None, :]
    a = torch.minimum(prm.thole[:, None], prm.thole[None, :])
    if prm.thole_pair is not None:
        a = prm.thole_pair
    has = (damp > 0) & ~eye
    if prm.undamped is not None:
        has = has & ~prm.undamped
    damp_safe = torch.where(has, damp, torch.ones_like(damp))
    rho = a * (r / damp_safe) ** 3
    e = torch.exp(-rho)
    zero = torch.zeros_like(r)
    g3 = torch.where(has, rho * e, zero)
    g5 = torch.where(has, rho**2 * e, zero)
    g7 = torch.where(has, (0.6 * rho**3 - 0.2 * rho**2) * e, zero)
    return g3, g5, g7


def _pair_fields(d, r, mu, theta, charge, f3, f5, f7):
    """Field (N,N,3) at site i due to the multipoles of j, with radial factors f3, f5, f7 in place of lambda."""
    rr3, rr5, rr7 = f3 / r**3, 3 * f5 / r**5, 15 * f7 / r**7
    mu_d = torch.einsum('jb,ijb->ij', mu, d)
    theta_d = torch.einsum('jab,ijb->ija', theta, d)
    dtd = (theta_d * d).sum(-1)
    radial = charge[None, :] * rr3 + mu_d * rr5 + dtd * rr7
    return d * radial[..., None] - mu[None, :, :] * rr3[..., None] - 2 * theta_d * rr5[..., None]


def _pair_dipole_tensor(d, r, f3, f5):
    """Tensor (N,N,3,3) 3 d d f5 / r^5 - I f3 / r^3."""
    I = torch.eye(3, dtype=d.dtype)
    return (3 * d[..., :, None] * d[..., None, :] * (f5 / r**5)[..., None, None]
            - I * (f3 / r**3)[..., None, None])


def alpha_gradient(pos, terms):
    """dE/dalpha_k (N,) in kJ/(mol nm^3), from the converged quantities (terms of electrostatic_energy).

    Valid for sites with damping = alpha^(1/6) (e.g. the ML sites of with_ml_parameters).
    """
    prm = terms['prm'] if 'prm' in terms else None
    if prm is None:
        raise ValueError('terms must contain "prm" (use mlmm_electrostatic_energy(..., return_terms=True))')
    mu, theta = terms['mu_lab'], terms['theta_lab']
    mu_d, mu_p = terms['mu_ind'], terms['mu_ind_p']
    d, r, eye = _pair_geometry(pos)
    g3, g5, g7 = thole_rho_derivs(r, prm, eye)
    alpha = prm.polarity
    active = alpha > 0
    alpha_safe = torch.where(active, alpha, torch.ones_like(alpha))
    zero_diag = lambda s: torch.where(eye, torch.zeros_like(s), s)

    # "derivative" permanent fields: G[i, j] = field at i of the multipoles of j with g factors
    G = _pair_fields(d, r, mu, theta, prm.charge, g3, g5, g7)
    Gd = G * zero_diag(prm.d_scale)[..., None]
    Gp = G * zero_diag(prm.p_scale)[..., None]
    # sum_i (G^d_ik . mu_p,i + mu_d,i . G^p_ik)  (k = source)  +  sum_j (G^d_kj . mu_p,k + mu_d,k . G^p_kj)  (k = target)
    as_source = torch.einsum('ikx,ix->k', Gd, mu_p) + torch.einsum('ix,ikx->k', mu_d, Gp)
    as_target = torch.einsum('kjx,kx->k', Gd, mu_p) + torch.einsum('kx,kjx->k', mu_d, Gp)
    field_term = (as_source + as_target) / (4 * alpha_safe)

    # "derivative" dipole-dipole tensor
    H = _pair_dipole_tensor(d, r, g3, g5) * zero_diag(prm.u_scale)[..., None, None]
    tensor_term = (torch.einsum('kx,kjxy,jy->k', mu_d, H, mu_p)
                   + torch.einsum('ix,ikxy,ky->k', mu_d, H, mu_p)) / (4 * alpha_safe)

    diag_term = -(mu_d * mu_p).sum(-1) / (2 * alpha_safe**2)
    grad = ONE_4PI_EPS0 * (diag_term + field_term + tensor_term)
    return torch.where(active, grad, torch.zeros_like(grad))


def switch_gradient(pos, terms, u_base, s):
    """dE/ds_i (N,) per site, for u_ij = u_base_ij s_i s_j (sum per molecule to get dE/ds_m)."""
    prm = terms['prm']
    mu_d, mu_p = terms['mu_ind'], terms['mu_ind_p']
    d, r, eye = _pair_geometry(pos)
    l3, l5, _ = thole_factors(r, prm, eye)
    T = _pair_dipole_tensor(d, r, l3, l5)
    w = torch.where(eye, torch.zeros_like(u_base), u_base) * s[None, :]              # u_ij s_j
    F_p = torch.einsum('ij,ijxy,jy->ix', w, T, mu_p)
    F_d = torch.einsum('ij,ijxy,jy->ix', w, T, mu_d)
    return -0.5 * ONE_4PI_EPS0 * ((mu_d * F_p).sum(-1) + (mu_p * F_d).sum(-1))


def quadrupole_gradient(pos, terms, sites):
    """dE/dTheta_i (M,3,3), symmetric part, at the given sites (lab-frame quadrupoles; Tinker convention).

    Permanent part (Tinker closed form, undamped, m scaling) + polarization part (damped field of the quadrupole
    at j, contracted with w_j = 1/2 (d_ij mu_p,j + p_ij mu_d,j)).
    In the kernel it is the same field gradient used for the quadrupole torques.
    """
    prm = terms['prm']
    mu, theta = terms['mu_lab'], terms['theta_lab']
    mu_d, mu_p = terms['mu_ind'], terms['mu_ind_p']
    d, r, eye = _pair_geometry(pos)
    zero_diag = lambda s: torch.where(eye, torch.zeros_like(s), s)
    R = -d                                                          # R_ij = r_j - r_i
    rr5 = 3 / r**5
    rr7 = 15 / r**7
    rr9 = 105 / r**9
    RR = R[..., :, None] * R[..., None, :]
    ck = prm.charge[None, :]
    dkr = torch.einsum('ja,ija->ij', mu, R)
    qkR = torch.einsum('jab,ijb->ija', theta, R)
    qkr = (qkR * R).sum(-1)
    # derivatives of the closed-form terms with respect to Theta_i (before symmetrizing)
    g = (RR * (ck * rr5 - dkr * rr7 + qkr * rr9)[..., None, None]
         + 2 * mu[None, :, :, None] * R[..., None, :] * rr5[..., None, None]
         + 2 * theta[None, :, :, :] * rr5[..., None, None]
         - 4 * R[..., :, None] * qkR[..., None, :] * rr7[..., None, None])
    perm = (g * zero_diag(prm.m_scale)[..., None, None]).sum(1)

    # polarization: the field at j of the quadrupole at i is E_j = 15 (dTd) d l7/r^7 - 6 T d l5/r^5, d = r_j - r_i
    l3, l5, l7 = thole_factors(r, prm, eye)
    w = 0.5 * (zero_diag(prm.d_scale)[..., None] * mu_p[None, :, :] + zero_diag(prm.p_scale)[..., None] * mu_d[None, :, :])
    wd = (w * R).sum(-1)                                            # w_j . d  with d = r_j - r_i = R_ij
    dfield = (15 * wd * l7 / r**7)[..., None, None] * RR - 6 * (l5 / r**5)[..., None, None] * w[..., :, None] * R[..., None, :]
    pol = -dfield.sum(1)

    grad = ONE_4PI_EPS0 * (perm + pol)
    grad = 0.5 * (grad + grad.transpose(-1, -2))
    return grad[torch.as_tensor(sites)]
