# CST mathematical audit — 2026-09-19

Baseline: [`306362e45af5ed3ac69a2b69cee4e3cddd066b77`](https://github.com/NavisWORLD/The-theory-of-CST/commit/306362e45af5ed3ac69a2b69cee4e3cddd066b77). Historical formula and functions are intentionally preserved. This document corrects claims about their units; it does not retroactively change their publication.

## Actual source and lineage

1. [Early recoverable `test1maybe.py` commit](https://github.com/PHERACLEASE/test/commit/10e86764c6d743b5ceaaaf1baab7279a0d6f0ba5) authored **2025-01-30 07:13:05 UTC** (January 29 in UTC−8), contains `compute_Psi: return self.alpha * Omega * Ec`. `Omega` is a gravitational acceleration sum divided by `a0`; `alpha` is a free coefficient. The source header or recollection may suggest earlier January work, but that date is not established by this commit.
2. [CosmicSynapse](https://github.com/NavisWORLD/CosmicSynapse) has publicly accessible commits dated 2025-02-26 UTC. It is related lineage, not independently validated physics.
3. The legacy `cst_engine.py` uses 11-component position and velocity vectors, 12-value memory, `VOLUME_11D=1e132`, and `psi` clipping to ±1e-10.
4. `cst_functions.py` implements a later 12-component velocity expression with `v_12d=1e-36`, `R_0=1e6`, and nominal information input called `entropies`.
5. `CST_Formula_Explanation.markdown` publishes a related, **nonidentical** 12D formula, including area-derived bits and a stated `V_12D≈1e144`. These values differ from the Python helper and must not be conflated.
6. `cst_math_v2.py` defines a separately named candidate, with standalone numerical test evidence; the old implementations remain unchanged.

The earlier `cst_math_sandbox_001.zip` was not available here. `experiments/cst_math_001/` independently reconstructs the synthetic comparison from the published source.

## Term and unit audit

| Historical symbol or term | Declared meaning / actual domain | Dimensional result | Status / correction |
|---|---|---|---|
| `m_i c²` | mass kg, speed m/s | J | Defined rest-energy scale; not automatically free energy |
| `E_chaos` | claimed energy J | J in publication, but `m_i c² λ_i` in Python gives **J/s** for `λ_i` in s⁻¹ | INVALID; v2 uses independent J input `H_i` and `λ_i τ` |
| `½m_i Σ_1^12 v_ik²` | velocities claimed m/s | J, conditional on all coordinates being physical lengths | 12 software channels are not 12 measured spatial dimensions; v2 physical candidate accepts only 3D velocity |
| `φ(m_i c²+E_chaos)+c λ_i+1` | `φ` dimensionless, `λ` s⁻¹ | J + m/s² + dimensionless; incompatible | INVALID; `cλ` is acceleration, **not** energy; use dimensionless `φ+λτ+H_i/(m_i c²)` multiplying kinetic J |
| `exp(-r/r0)` | r and r0 in m | dimensionless | Defined for positive r0 |
| `G m_i m_j/(r c²)` | m in kg, r in m | **kg**, not dimensionless | INVALID; v2 uses pair interaction energy `G m_i m_j exp(-r/r0)/r` multiplied by declared dimensionless coefficient |
| `-G m_i m_j/r` | Newton potential energy | J | Classical pair energy; allocate half to each node to avoid total-energy double count |
| `k_B T_CMB/c` | J/K × K divided by m/s | **J·s/m**, not J/m | INVALID as published prefactor; v2 uses `k_B T × (ℓ_I/r)` |
| `S_i/(k_B ln2)` | information bits from horizon entropy | dimensionless, **only for an actual applicable black-hole horizon** | Arbitrary stellar/planetary radii do not satisfy this premise; v2 accepts explicit software score `b_i/b0` |
| `(k_B T/c) Σ B_i B_j/r` | B dimensionless | **J·s/m²**, not J | INVALID; v2 information energy `-α_I k_B T (ℓ_I/r) (b_i/b0)(b_j/b0)` (J) is a *hypothesis*, not a derived force |
| `V_12D` | claimed m¹² | m¹² *if 12 genuine spatial axes and a metric exist* | Not established; v2 returns J and dimensionless `E/E_ref`; no 12D physical density |
| `positions,distances` helper arguments | 12D positions supplied; supplied pair-distance matrix | positions unused in helper; distance symmetry and positivity unvalidated | v2 computes 3D separation internally and rejects coincident nodes |
| `R_0=1e6` comment | comment says approximately 1 Mpc | 1e6 m is **not** 1 Mpc (~3.086e22 m) | Inconsistent physical description; v2 parameter explicitly meters |
| `psi` clipping | legacy simulator `[-1e-10,1e-10]` | J/m¹¹ if premise held | Can hide divergences; v2 fails closed rather than clipping |

All new node/pair additions have units of joules and every denominator has an explicit positive scale. Individual terms remain separately inspectable. No numerical unit proof establishes a new physical mechanism.

## Symmetry, limits and numeric behavior

- The pair energy is symmetric in `i,j`; by allocating half of each pair energy to each node, summing node diagnostics counts each pair once. A canonical mass multiplication order makes float64 results permutation invariant under tested reorderings.
- At `α_C=α_I=0, φ=1, λ=H=0`, the diagnostic reduces to Newtonian mechanical energy. It is not a trajectory equation or force derivation.
- `r→0` creates inverse-distance singularities; exact coincident distinct nodes raise `ValueError`. For `r≫r0`, connectivity decays exponentially; inverse-distance terms tend to zero.
- Float64 finite checks reject overflows. Large and tiny terms must be inspected separately: the information contribution can vanish in the rounded total.
- Parameters `φ, α_C, α_I, τ, r0, ℓ_I, b0, T, E_ref` are explicit. The numerical choices in the experiment are conventions; no physical value was fitted.
- Missing work: independent measured target, additional-force derivation, long-run *coupled* dynamics, an empirically justified 12D geometry and calibrated information observables.

## Provenance and evidence boundary

See `docs/CST_CORRECTED_FORMULATION.md`, `docs/CST_MATH_EXPERIMENT_001.md`, and `experiments/cst_math_001/results.json`. A passing test establishes code-level behavior only. IBM quantum workloads, entanglement-tool timestamps, CERN measurements and gameplay output are not validation of this equation.
