# CST v2 compatibility and migration

Baseline main SHA: `306362e45af5ed3ac69a2b69cee4e3cddd066b77`. Dedicated update branch: `research/cst-math-closure-001`.

**No automatic migration.** The new `cst_math_v2.evaluate(nodes, params, mode)` entry point is opt-in and never monkeypatches the original `cst_functions.compute_psi_i`, `cst_engine.CSTEntity.compute_psi`, network protocol, or game `psiProxy`. Current cst_functions accepts 12-component velocities and externally supplied distances with an arbitrary historic volume; v2 instead accepts 3 physical coordinates, immutable `Node`/`Parameters`, derives pair separation, returns separate J terms and a dimensionless proxy. These are intentionally incompatible semantic contracts. Callers must explicitly map dimensions, information scores and units; do not silently adapt a saved game or legacy simulator.

Mars Synapse: Red Genesis remains unchanged. `game/src/cst.js`, `game/src/save.js`, its schema and browser state are historical computational gameplay semantics. No replacement equations are injected, no save migration is needed, and no physical claims are transferred to the game.

**Verification ledger:**
- Baseline GitHub Actions [CST verification run 32338611513](https://github.com/NavisWORLD/The-theory-of-CST/actions/runs/32338611513): success at main baseline (2026-08-20), not a locally rerun baseline.
- Isolated local v2 and experiment-replay tests after the float64 permutation and JSON tuple-normalization fixes: 10 passed via `PYTHONPATH=. python -m pytest tests/test_cst_math_v2.py tests/test_cst_math_experiment.py -q` in a reconstructed subset of files.
- Local full legacy Python suite: UNEXECUTED (container has no network route to clone GitHub; connected GitHub is available separately).
- Local Node game suite and visual/browser checks: UNEXECUTED in this container; unchanged source and save contract.
- CI on update branch: inspect actual run result before claiming pass; GitHub run status is not inferred from individual file commits.

**No Beast Box files, historical records, weights or game saves were modified.**