# CST validation protocol — separate from mathematical consistency

**Target:** test a prospective physical prediction that differs from conventional models, using independently measured observations and blinded or held-out conditions. A diagnostic energy function alone is not a dynamical prediction.

1. Define a specific system and measurable observable *before* choosing coefficients. Specify whether the observable is orbital trajectory, measured energy, or another instrumented quantity. Verify the applicability of any black-hole horizon entropy only if working with black holes.
2. Derive additional forces or evolution equations from an explicit physical action or a declared alternative law. Check conservation laws, energy bookkeeping, symmetry, and parameter identifiability. Do not infer a force from a diagnostic number without that derivation.
3. Select an independent, versioned dataset and record units, uncertainties, selection criteria, provenance and train/test partition. Keep all tuning strictly on training data. No measured dataset is bundled by this patch.
4. Register primary error metric (with physical units), uncertainty interval, baseline (e.g. Newtonian dynamics where applicable), evaluation horizon, calibration budget, statistical test and rejection criteria. Compare noninformation, shuffled information and no-chaos arms on identical held-out conditions.
5. Report all results, including nulls and boundary failures, alongside computational budget and hashes. Repeat with floating and higher precision only where numerical conditioning warrants it.
6. For purely computational tasks, select task-level datasets, matched algorithms and statistical controls; report these as software benchmarks, not physical validation.

Current status: **NOT ESTABLISHED**. No empirically grounded additional-force evolution law, fitted measurement mapping, or independent held-out physical dataset is available in this update. The synthetic mathematics experiment must not be relabeled as validation. IBM quantum data are separate and would require matched measurement-basis controls and independent causal hypotheses.