"""CST v2: a dimensionally consistent, nonvalidated diagnostic energy model.

Independent of, and never a replacement for, historical cst_functions.compute_psi_i.
Only 3D physical coordinates are accepted; 12-channel game state is separate.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import exp, fsum, isfinite, sqrt
from typing import Tuple

G = 6.67430e-11                    # m^3 kg^-1 s^-2
C = 299792458.0                   # m/s
K_B = 1.380649e-23                # J/K


def _real(name: str, value: float, *, positive: bool = False) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite real number") from exc
    if not isfinite(v) or (positive and v <= 0) or (not positive and v < 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return v


@dataclass(frozen=True)
class Parameters:
    phi: float = 1.0
    tau_s: float = 1.0
    alpha_connect: float = 1.0
    alpha_information: float = 1.0
    decay_length_m: float = 1.0e6
    information_length_m: float = 1.0e6
    reference_bits: float = 1.0
    temperature_k: float = 2.725
    reference_energy_j: float = 1.0

    def __post_init__(self) -> None:
        for name in ('phi', 'alpha_connect', 'alpha_information', 'temperature_k'):
            _real(name, getattr(self, name))
        for name in ('tau_s', 'decay_length_m', 'information_length_m',
                     'reference_bits', 'reference_energy_j'):
            _real(name, getattr(self, name), positive=True)


@dataclass(frozen=True)
class Node:
    mass_kg: float
    position_m: Tuple[float, float, float]
    velocity_m_s: Tuple[float, float, float]
    information_bits: float = 0.0
    lyapunov_per_s: float = 0.0
    chaos_energy_j: float = 0.0

    def __post_init__(self) -> None:
        for name in ('mass_kg',):
            _real(name, getattr(self, name), positive=True)
        for name in ('information_bits', 'lyapunov_per_s', 'chaos_energy_j'):
            _real(name, getattr(self, name))
        for name in ('position_m', 'velocity_m_s'):
            vector = getattr(self, name)
            if len(vector) != 3:
                raise ValueError(f'{name} must contain exactly three physical components')
            for x in vector:
                try:
                    valid = isfinite(float(x))
                except (TypeError, ValueError, OverflowError) as exc:
                    raise ValueError(f'{name} must be finite') from exc
                if not valid:
                    raise ValueError(f'{name} must be finite')


@dataclass(frozen=True)
class Terms:
    kinetic_j: float
    chaos_modulation_j: float
    gravitational_j: float
    connectivity_j: float
    information_j: float
    total_j: float
    psi_proxy: float


def evaluate(nodes: Tuple[Node, ...], params: Parameters = Parameters(),
             mode: str = 'corrected') -> Tuple[Terms, ...]:
    """Calculate per-node diagnostics in joules; no additional force law is implied.

    Modes: corrected, classical, no_information, no_chaos, no_connectivity.
    Shuffling information requires the caller to supply a copied/permuted node tuple.
    Self interactions are absent; coincident distinct nodes raise ValueError.
    """
    modes = ('corrected', 'classical', 'no_information', 'no_chaos', 'no_connectivity')
    if mode not in modes:
        raise ValueError(f'mode must be one of {modes}')
    if not isinstance(params, Parameters):
        raise TypeError('params must be Parameters')
    nodes = tuple(nodes)
    if not all(isinstance(n, Node) for n in nodes):
        raise TypeError('nodes must contain Node instances')
    gravity = [[] for _ in nodes]
    connectivity = [[] for _ in nodes]
    information = [[] for _ in nodes]
    for i, a in enumerate(nodes):
        for j in range(i + 1, len(nodes)):
            b = nodes[j]
            delta = tuple(float(x) - float(y) for x, y in zip(a.position_m, b.position_m))
            r = sqrt(fsum((d*d for d in delta)))
            if not isfinite(r) or r <= 0:
                raise ValueError('distinct nodes require finite, strictly positive separation')
            base = G * a.mass_kg * b.mass_kg / r
            grav = -0.5 * base
            conn = (0.0 if mode in ('classical', 'no_connectivity') else
                    0.5 * params.alpha_connect * base * exp(-r / params.decay_length_m))
            info = (0.0 if mode in ('classical', 'no_information') else
                    -0.5 * params.alpha_information * K_B * params.temperature_k *
                    (params.information_length_m / r) *
                    (a.information_bits / params.reference_bits) *
                    (b.information_bits / params.reference_bits))
            for index in (i, j):
                gravity[index].append(grav)
                connectivity[index].append(conn)
                information[index].append(info)
    results = []
    for index, node in enumerate(nodes):
        v2 = fsum((float(v)*float(v) for v in node.velocity_m_s))
        kinetic = 0.5 * node.mass_kg * v2
        mod = (0.0 if mode in ('classical', 'no_chaos') else
               kinetic * (params.phi - 1.0 + params.tau_s * node.lyapunov_per_s +
                          (node.chaos_energy_j / (node.mass_kg * C*C))))
        if mode == 'no_chaos':
            mod = 0.0
        g, c, info = (fsum(v) for v in (gravity[index], connectivity[index], information[index]))
        total = fsum((kinetic, mod, g, c, info))
        psi = total / params.reference_energy_j
        if not all(isfinite(v) for v in (kinetic, mod, g, c, info, total, psi)):
            raise ValueError('numeric overflow; reduce inputs or use a higher-precision analysis')
        results.append(Terms(kinetic, mod, g, c, info, total, psi))
    return tuple(results)
