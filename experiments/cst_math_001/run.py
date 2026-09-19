"""Reproducible CST 001 frozen synthetic comparison; run with PYTHONPATH=."""
import argparse
from dataclasses import asdict, replace
from decimal import Decimal, localcontext
import hashlib
import json
import math
from pathlib import Path
from random import Random

from cst_math_v2 import C, G, K_B, Node, Parameters, evaluate

BASE_SHA = '306362e45af5ed3ac69a2b69cee4e3cddd066b77'


def original_reference(nodes):
    """Independent scalar transcription of historical cst_functions.compute_psi_i.

    Precisely retains original R_0=1e6, 1e-36 volume, phi=0.1,
    lambda used as raw rest-energy multiplier, and entropies as area argument.
    It is INVALID dimensionally. Reuse only for a frozen numerical comparison.
    """
    hbar, ln2, tcmb, r0, volume = 1.054571817e-34, .693147, 2.725, 1e6, 1e-36
    result = []
    for i, a in enumerate(nodes):
        rest = a.mass_kg * C*C
        chaos = rest * a.lyapunov_per_s
        kinetic = .5*a.mass_kg * sum(x*x for x in a.velocity_m_s)
        weight = .1*(rest + chaos) + C * a.lyapunov_per_s + 1
        term_k = weight * kinetic / (rest + chaos)
        term_s, term_g, term_info = 0., 0., 0.
        s_i = K_B * C**3 * a.information_bits / (4*hbar*G*ln2)
        for j, b in enumerate(nodes):
            if i == j: continue
            r = math.dist(a.position_m, b.position_m)
            s_j = K_B * C**3 * b.information_bits / (4*hbar*G*ln2)
            term_s += math.exp(-r/r0) * G * a.mass_kg * b.mass_kg / (r*C*C)
            term_g += G*a.mass_kg*b.mass_kg/r
            term_info += (K_B*tcmb/C)*(s_i*s_j)/r
        term_s *= (rest + chaos)
        result.append(dict(kinetic_j=term_k, connectivity_historical=term_s,
                           gravity_j=-term_g, information_invalid=-term_info,
                           psi_historical=(term_k+term_s-term_g-term_info)/volume))
    return result


def generate(seed):
    rng = Random(seed)
    positions = ((0.,0.,0.), (1e6,0.,0.), (0.,2e6,0.))
    return tuple(Node(1e5 + rng.random()*1e5, pos,
                      tuple(rng.uniform(-100, 100) for _ in range(3)),
                      float(i+1), rng.uniform(0, .1), 0.)
                 for i, pos in enumerate(positions))


def dsum(terms):
    with localcontext() as ctx:
        ctx.prec = 60
        return str(sum((Decimal.from_float(float(x)) for x in terms), Decimal(0)))


def execute(seed=20260919):
    nodes = generate(seed)
    p = Parameters()
    arms = {name: evaluate(nodes, p, mode=name) for name in
            ('corrected', 'no_information', 'no_chaos', 'classical')}
    shifted = tuple(replace(n, information_bits=nodes[(i+1)%len(nodes)].information_bits)
                    for i,n in enumerate(nodes))
    arms['shuffled_information'] = evaluate(shifted, p)
    old = original_reference(nodes)
    corrected = arms['corrected']
    without = arms['no_information']
    results = {
        'experiment': 'cst_math_001', 'base_sha': BASE_SHA, 'seed': seed,
        'inputs': [asdict(n) for n in nodes], 'params': asdict(p),
        'arms': {name: [asdict(t) for t in vals] for name, vals in arms.items()},
        'original': old,
        'metrics': {
            'finite_corrected': all(math.isfinite(t.total_j) for t in corrected),
            'legacy_information_dominates': all(abs(t['information_invalid']) >
                abs(t['kinetic_j'])+abs(t['gravity_j'])+abs(t['connectivity_historical'])
                for t in old),
            'float64_info_ablation_delta_j': [a.total_j-b.total_j for a,b in zip(corrected, without)],
            'precise_sum_corrected_j': [dsum((t.kinetic_j, t.chaos_modulation_j,
                t.gravitational_j, t.connectivity_j, t.information_j)) for t in corrected],
            'precise_sum_no_information_j': [dsum((t.kinetic_j, t.chaos_modulation_j,
                t.gravitational_j, t.connectivity_j)) for t in without],
            'permutation_invariant': tuple(reversed(evaluate(tuple(reversed(nodes)), p))) == corrected,
            'repeated_evaluation_equal': evaluate(nodes, p) == corrected,
            'information_scaling': {str(a): [t.information_j for t in evaluate(nodes,
                  replace(p, alpha_information=float(a)))] for a in (0,1,2)},
            'distance_scaling': {str(scale): [t.gravitational_j for t in evaluate(tuple(
                  replace(n, position_m=tuple(x*scale for x in n.position_m)) for n in nodes), p)]
                  for scale in (.1, 1., 10.)},
        },
        'scope': 'synthetic arithmetic; no independent external dataset or new force law',
    }
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, default=Path(__file__).with_name('results.json'))
    args = ap.parse_args()
    payload = (json.dumps(execute(), indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
    args.output.write_bytes(payload)
    print(f'output={args.output} sha256={hashlib.sha256(payload).hexdigest()} bytes={len(payload)}')
    s = json.loads(payload)
    print(json.dumps(s['metrics'], indent=2))


if __name__ == '__main__':
    main()
