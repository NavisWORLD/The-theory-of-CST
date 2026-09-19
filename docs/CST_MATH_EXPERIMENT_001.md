# CST math experiment 001 — synthetic, frozen

**Status: MEASURED arithmetic; NULL external physical evidence.** Baseline `306362e45af5ed3ac69a2b69cee4e3cddd066b77`. Protocol: `experiments/cst_math_001/protocol.json`; implementation: `cst_math_v2.py`; raw inputs, per-node terms, all control outputs: `experiments/cst_math_001/results.json`; executable: `experiments/cst_math_001/run.py`. Seed `20260919`.

## Procedure

Run from repository root:

```bash
PYTHONPATH=. python experiments/cst_math_001/run.py
MOCK_AUDIO=1 PYTHONPATH=. pytest -q
(cd game && npm test && npm run check)
```

The first command overwrites only the local experiment result file. Existing production runtime, historical helper, and gameplay are never redirected. The `original_reference` arm is an independent scalar transcription of the historical `cst_functions.compute_psi_i`; it deliberately retains its invalid dimensions, legacy constants and direct entropy-as-area substitution. The source’s unrelated `web3` dependency is not required for this isolated numeric comparison.

## Executed local findings

| Measure | Observed result |
|---|---|
| Historical information-dominance on synthetic three nodes | All 3 true; invalid historical information contribution approximately -1.60e56, -2.14e56, -1.91e56 in the old arithmetic |
| Historical `psi` (old arbitrary volume normalization) | -1.60e92, -2.14e92, -1.91e92; **not physically meaningful energy densities** |
| Corrected information energy `I_i` (J) | -6.584e-23, -8.810e-23, -7.869e-23 |
| Corrected minus no-information float64 total energy (J) | [0, 0, 0]: **NULL at this precision and scale** |
| Information coefficient 0 / 1 / 2 | Isolated term 0 / baseline / exactly twice baseline; construction, not evidence of utility |
| Shuffled information inputs | Changes isolated information terms; float64 node totals unchanged |
| No-chaos control | Removes the explicit kinetic modulation; a constructed change, not prediction |
| Classical control | Returns 3D Newtonian mechanical diagnostic |
| Permutation and repeated evaluation | Both true after fixing canonical multiplication order |
| Distance scaling 0.1× / 1× / 10× | Newtonian pair contribution follows inverse-distance scaling |
| High-precision reporting | Decimal(60) sums already-computed float64 terms retain tiny contribution; **not a higher-precision recomputation of the original expressions** |

Local `pytest -q tests/test_cst_math_v2.py tests/test_cst_math_experiment.py` on the isolated v2 files: 10 passed across v2 and experiment-replay checks after correction. Historical full-suite results are separately reported in migration documentation and CI; do not conflate those with a local full-repository run.

The standalone run originally found a failure of strict float64 permutation invariance from operand ordering. It was fixed in `cst_math_v2.py`; the earlier failure is retained here rather than misrepresented as a first-pass success.

## Reproducibility / scope

Local full-format generated `results.json` SHA-256: `805896cdf5ca6adf94aa59b90c86a1941a068131fbfdc13ab295806d2594b845`. The repository includes the **same raw numeric fields in a compact JSON serialization**, so byte-level checksums differ; regenerate the full-format file with the above command to compare with the local hash. Run results are specific to the recorded Python/float64 implementation and synthetic seed.

**Unexecuted:** independent measured-physical-system comparison; live IBM QPU use; actual new-force-driven dynamics; long-run CST-coupled trajectory stability; full game/browser test in this isolated local container. Do not mark these as passes.

**Interpretation:** unit consistency and executable control outputs support a research *candidate*, not a new law. The corrected thermal term has no measurable effect on the total at this synthetic float64 scale. No arbitrary magnification was introduced.