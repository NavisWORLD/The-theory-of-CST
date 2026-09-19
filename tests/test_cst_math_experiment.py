"""Verify the frozen experiment can be replayed without external services."""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / 'experiments' / 'cst_math_001'


def _run_module():
    spec = importlib.util.spec_from_file_location('cst_math_001_run', EXP / 'run.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_frozen_results_match_reexecution():
    expected = json.loads((EXP / 'results.json').read_text(encoding='utf-8'))
    # JSON round-trip normalizes dataclass tuple coordinates to JSON arrays.
    actual = json.loads(json.dumps(_run_module().execute()))
    assert actual == expected


def test_synthetic_null_and_legacy_dominance_are_preserved():
    result = _run_module().execute()
    metrics = result['metrics']
    assert metrics['finite_corrected']
    assert metrics['legacy_information_dominates']
    assert metrics['permutation_invariant']
    assert metrics['repeated_evaluation_equal']
    assert metrics['float64_info_ablation_delta_j'] == [0.0, 0.0, 0.0]
    for full, without in zip(metrics['precise_sum_corrected_j'],
                             metrics['precise_sum_no_information_j']):
        assert full != without
