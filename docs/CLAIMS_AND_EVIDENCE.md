# CST Claims and Evidence Ledger

## Why this file exists

Large research programs become difficult to evaluate when implementation, observation, interpretation, and ambition are mixed together. This ledger keeps them separate.

Use these labels:

- **IMPLEMENTED**: code exists;
- **OBSERVED**: a captured runtime demonstrates execution;
- **MEASURED**: a defined experiment produced a metric;
- **NULL**: a test failed its success criterion;
- **HYPOTHESIS**: a falsifiable proposition awaiting stronger evidence;
- **MODEL / METAPHOR**: conceptual language, not a literal scientific claim.

## Repository-level ledger

| Claim | Status | Where to inspect | What it does not prove |
|---|---|---|---|
| The simulator stores multi-component entity state | IMPLEMENTED | `cst_engine.py` | That physical reality uses the same state space |
| The legacy runtime uses 11D position and velocity vectors | IMPLEMENTED | `CSTEntity`, `CSTUniverse` | Eleven physical dimensions |
| The later formula helper accepts 12D position and velocity arrays | IMPLEMENTED | `cst_functions.py` | Twelve physical dimensions |
| Each legacy entity contains a 12-value internal memory vector | IMPLEMENTED | `CSTEntity.memory_vector` | Biological memory or consciousness |
| Audio can be summarized and fed into the simulator | IMPLEMENTED | `AudioProcessor`, `process_audio()` | That sound controls real cosmic structure |
| Frequency can be mapped to RGB/HSV values | IMPLEMENTED | `freq_to_light()` | A newly discovered law connecting audible frequency to photon wavelength |
| The engine computes gravitational-style pairwise forces | IMPLEMENTED | `compute_net_force()` | Astrophysical accuracy of the full CST model |
| The engine computes a CST `psi` value | IMPLEMENTED | `compute_psi()` | That `psi` is a measured physical field |
| State events are written to CSV/JSON | IMPLEMENTED | `MemoryNodeLog` | That the log is an immutable blockchain |
| Logged tokens are SHA-256 hashed and linked to the previous token ID | IMPLEMENTED | `MemoryNodeLog.generate_token_id()` and `log()` | Consensus, decentralization, or tamper-proof storage |
| A local TCP server exposes `ping`, `update`, and `audio` commands | IMPLEMENTED | `socket_server.py` | Internet-scale service readiness |
| Unity-side scripts consume and visualize exported state | IMPLEMENTED source components | `*.cs` | That a complete Unity project is packaged in this repository |
| The missing ecosystem dependency has a bounded compatible implementation | IMPLEMENTED | `ecosystem_engine.py`, tests | Biological realism |
| The universe is literally a neural network | HYPOTHESIS / MODEL | CST theory | Established cosmology |
| The informational term describes a new physical force | HYPOTHESIS | `CST_Formula_Explanation.markdown` | Experimental detection of that force |
| The golden ratio is a privileged fundamental constant in CST physics | HYPOTHESIS | formula documents | A demonstrated new law of nature |
| CST proves simulation theory | NOT ESTABLISHED | n/a | n/a |
| CST provides unhackable biometric identity | NOT ESTABLISHED | n/a | n/a |
| CST provides medical diagnosis | NOT ESTABLISHED | n/a | n/a |
| Persistent CST software state is consciousness | NOT ESTABLISHED | n/a | n/a |

## Reproducible software checks

Run:

```bash
pip install -r requirements-dev.txt
MOCK_AUDIO=1 PYTHONPATH=. pytest -q
```

The tests are intentionally narrower than the theory. A passing unit test verifies declared software behavior, not the truth of the physical interpretation.

## Minimum standard for a new CST claim

A new claim should include:

1. **Mechanism**: exact code path or mathematical term.
2. **Control**: what is disabled or replaced in the baseline.
3. **Metric**: what number changes if the mechanism matters.
4. **Protocol**: exact command, seed, dataset, and environment.
5. **Result**: raw output plus summary statistics.
6. **Failure condition**: a criterion that would count against the claim.
7. **Scope**: what the result does and does not generalize to.

## Same-architecture ablation template

A strong CST software experiment holds the surrounding system constant and changes only the mechanism being tested.

```text
A: same model + same prompt + same memory + same tools + CST mechanism OFF
B: same model + same prompt + same memory + same tools + CST mechanism ON
```

Repeat across enough trials to estimate variance. Report wins, losses, nulls, and failures.

## Physical-hypothesis boundary

A computational analogy becomes a physical theory only when it makes precise empirical predictions and survives comparison with observation. Code execution, visual beauty, internal consistency, and surprising emergent behavior can motivate a hypothesis, but none of them substitute for external measurement.
