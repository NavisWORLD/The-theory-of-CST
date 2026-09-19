# CST candidate v2 — dimensional bookkeeping

This proposed model separates measurable classical terms from dimensionless software proxies. It is **not** a new physical law. Historical code and the original formula remain unchanged.

For each node `i` with mass `m_i>0`, three-dimensional position `x_i`, velocity `v_i`, chaos energy `H_i≥0` (J), and nonnegative Lyapunov rate `λ_i` (s⁻¹), define `E0_i=m_i c²` and `K_i=½m_i|v_i|²`. A declared time scale `τ>0` makes `λ_iτ` dimensionless. The kinetic contribution is

`K'_i = K_i [φ + λ_i τ + H_i/E0_i]`, with dimensionless `φ≥0`.

For unordered pairs `i<j`, at a strictly positive physical three-dimensional separation `r_ij=|x_i-x_j|`:

- Newton: `U_ij = -G m_i m_j/r_ij` (J).
- Model connectivity: `C_ij = α_C G m_i m_j exp(-r_ij/r0)/r_ij` (J).
- Hypothetical information coupling: `I_ij = -α_I k_B T (ℓ_I/r_ij) (b_i/b0)(b_j/b0)` (J).

Here `α_C,α_I≥0` are dimensionless **free model parameters**, `r0,ℓ_I>0` are independently declared length scales (m), `b_i≥0` is a software information score in bits, `b0>0` is a declared reference in bits, and `T≥0` is temperature (K). `φ=1, α_C=1, α_I=1, τ=1 s` in the synthetic pre-registration are transparent conventions, **not fitted physical constants**. Do not claim that `b_i` measures ordinary objects' black-hole horizon entropy. The information coupling is a defined hypothesis, not a derived force.

The node diagnostic energy is `E_i=K'_i+½Σ_{j≠i}(U_ij+C_ij+I_ij)` (J), with half-allocation preventing pair double-counting in `Σ_i E_i`. `ψ_proxy,i=E_i/E_ref` is dimensionless for explicitly supplied `E_ref>0` (J). There is **no physical 12D density**: no geometric 12D volume measure is established. Three spatial coordinates here are physical model inputs; any twelve-channel numerical state belongs to the separate computational implementation, not measured spacetime.

**Limits:** `α_C=α_I=0,φ=1,λ_i=H_i=0` recovers Newtonian mechanical energy (not new dynamics). Exponential connectivity tends to zero at large separation; every pair term is singular as `r→0` and exact collisions fail closed. All pair terms are invariant under node permutations; the input positions/velocities are 3D. Double precision may discard tiny information terms when added to much larger classical energies; return term-separated diagnostics and never infer an effect from a rounded total.

**No prediction of trajectories follows from this diagnostic alone.** Deriving forces for the additional terms would require an explicit variational/dynamical hypothesis and independent testing. The existing 11D/12D historical code and the Mars game remain unchanged.