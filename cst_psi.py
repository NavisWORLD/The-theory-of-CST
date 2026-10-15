# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Cory Shane Davis.
"""Unit-consistent CST psi_i (2026-10-15).

Every term is an energy divided by the reference energy E_ref,i = m_i c^2 (1 + lambda_i * tau),
so each term and their sum are DIMENSIONLESS. This replaces the mixed-unit sum in
CST_Formula_Explanation.markdown (see its Erratum section). The legacy compute_psi_i in
cst_functions.py is kept unchanged for history.

  K_i = (1 + phi + lambda_i tau) * (1/2 m_i |v_i|^2) / E_ref,i                       kinetic/chaotic
  S_i = sum_j exp(-r_ij/r0) * G m_i m_j / r_ij / E_ref,i                            screened ("synaptic")
  U_i = - sum_j G m_i m_j / r_ij / E_ref,i                                          Newtonian
  I_i = - sum_j k_B T_CMB * N_i N_j * (ell / r_ij) / E_ref,i                        informational
        N = A / (4 l_P^2 ln 2) = c^3 A / (4 hbar G ln 2)  [bits, Bekenstein-Hawking]
  psi_i = K_i + S_i + U_i + I_i

tau [s], r0 [m] and ell [m] are explicit model parameters, not physical constants.
"""
from __future__ import annotations
import numpy as np

G = 6.67430e-11
C = 2.99792458e8
K_B = 1.380649e-23
HBAR = 1.054571817e-34
T_CMB = 2.725
LN_2 = float(np.log(2.0))
MPC = 3.0856775814913673e22
L_PLANCK = float(np.sqrt(HBAR * G / C**3))


def bekenstein_bits(radius_m):
    """Bekenstein-Hawking entropy in bits for a sphere of radius R (dimensionless)."""
    area = 4 * np.pi * np.asarray(radius_m, dtype=float) ** 2
    return C**3 * area / (4 * HBAR * G * LN_2)


def pairwise_distances(positions):
    """Symmetric Euclidean distances from the SAME position vectors (any dimension)."""
    p = np.asarray(positions, dtype=float)
    d = np.sqrt(((p[:, None, :] - p[None, :, :]) ** 2).sum(-1))
    return d


def psi_terms(masses, positions, velocities, radii, lyapunovs, *, phi=(1 + 5**0.5) / 2,
              tau=1.0, r0=MPC, ell=L_PLANCK):
    m = np.asarray(masses, dtype=float)
    v = np.asarray(velocities, dtype=float)
    lam = np.asarray(lyapunovs, dtype=float)
    r = pairwise_distances(positions)
    n = len(m)
    off = ~np.eye(n, dtype=bool)
    if np.any(r[off] <= 0):
        raise ValueError("coincident entities: r_ij must be > 0")
    rs = np.where(off, r, np.inf)
    e_ref = m * C**2 * (1 + lam * tau)
    ke = 0.5 * m * (v**2).sum(-1)
    K = (1 + phi + lam * tau) * ke / e_ref
    gpair = G * m[:, None] * m[None, :] / rs
    screen = np.where(off, np.exp(-r / r0), 0.0)
    S = (screen * gpair).sum(1) / e_ref
    U = -gpair.sum(1) / e_ref
    N = bekenstein_bits(radii)
    I = -(K_B * T_CMB * N[:, None] * N[None, :] * ell / rs).sum(1) / e_ref
    return {"K": K, "S": S, "U": U, "I": I, "psi": K + S + U + I}


def psi(*args, **kw):
    return psi_terms(*args, **kw)["psi"]
