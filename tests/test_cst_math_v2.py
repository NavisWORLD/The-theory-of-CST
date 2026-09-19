from dataclasses import replace
from math import isclose, isfinite
import pytest
from cst_math_v2 import C, G, K_B, Node, Parameters, evaluate

A = Node(2.0, (0., 0., 0.), (1., 2., 0.), 3., .2, 4.)
B = Node(3.0, (10., 0., 0.), (0., 0., 0.), 5.)
P = Parameters(decay_length_m=10, information_length_m=10, temperature_k=2,
               reference_bits=1, reference_energy_j=1)


def test_classical_exact_pair_half_allocation():
    x, y = evaluate((A, B), P, 'classical')
    u = -G * A.mass_kg * B.mass_kg / 10
    assert isclose(x.kinetic_j, 5.)
    assert isclose(x.gravitational_j, u / 2)
    assert isclose(y.gravitational_j, u / 2)
    assert x.information_j == y.information_j == 0
    assert x.connectivity_j == y.connectivity_j == 0


def test_pair_symmetry_and_information_units():
    x, y = evaluate((A, B), P)
    expected = -K_B * 2. * (10. / 10.) * (3. * 5.) / 2
    assert isclose(x.information_j, expected)
    assert x.information_j == y.information_j
    assert x.connectivity_j == y.connectivity_j
    assert isclose(x.chaos_modulation_j, 5 * (.2 + 4 / (2 * C*C)))


def test_modes_are_explicit_and_nonmutating():
    original = (A, B)
    x = evaluate(original, P)
    noinfo = evaluate(original, P, 'no_information')
    assert noinfo[0].information_j == 0.
    assert noinfo[0].kinetic_j == x[0].kinetic_j
    assert evaluate(original, P, 'no_chaos')[0].chaos_modulation_j == 0
    assert evaluate(original, P, 'no_connectivity')[0].connectivity_j == 0
    assert evaluate(original, P) == x
    assert original == (A, B)


def test_permutation_invariance():
    original = evaluate((A, B), P)
    reversed_nodes = evaluate((B, A), P)
    assert original == tuple(reversed(reversed_nodes))


def test_zero_info_and_far_distance_limits():
    a = replace(A, information_bits=0.)
    assert evaluate((a, B), P)[0].information_j == 0.
    far = replace(B, position_m=(1e9, 0., 0.))
    result = evaluate((A, far), P)[0]
    assert result.connectivity_j == 0.
    assert abs(result.gravitational_j) < abs(evaluate((A, B), P)[0].gravitational_j)


def test_invalid_inputs_fail_closed():
    with pytest.raises(ValueError): Node(-1, (0,0,0), (0,0,0))
    with pytest.raises(ValueError): Node(1, (0,0), (0,0,0))
    with pytest.raises(ValueError): Node(1, (0,0,0), (0,0,float('nan')))
    with pytest.raises(ValueError): Parameters(reference_bits=0)
    with pytest.raises(ValueError): evaluate((A, replace(B, position_m=A.position_m)))
    with pytest.raises(ValueError): evaluate((A, B), P, 'unsupported')
    with pytest.raises(ValueError): evaluate((replace(A, mass_kg=1e300), replace(B, mass_kg=1e300)), P)


def test_isolated_node_and_empty_network():
    assert evaluate((), P) == ()
    assert evaluate((A,), P)[0].gravitational_j == 0
    assert all(isfinite(v) for v in vars(evaluate((A,), P)[0]).values())


def test_magnitude_is_not_an_effect():
    x = evaluate((A, B), P)[0]
    assert x.total_j == x.psi_proxy * P.reference_energy_j
    assert x.information_j != 0
    assert x.total_j == evaluate((A, B), P, 'no_information')[0].total_j
    # Tiny informational terms are lost in the much larger total: record separately.
