import math, numpy as np, pytest
import cst_psi as P

# Worked example from CST_Formula_Explanation.markdown (m=1e30 kg, r=1e10 m, R=3.5e8 m, c=3e8 as in the doc)
def test_doc_arithmetic_corrected():
    c = 3e8; G = 6.67430e-11
    assert G * 1e60 / (1e10 * c**2) == pytest.approx(7.4159e22, rel=1e-4)          # doc said 7.42e-15
    assert 6.67430e-11 * 1e60 / 1e10 == pytest.approx(6.6743e39)
    hbar = 1.0545718e-34; A = 4 * math.pi * (3.5e8) ** 2
    N = c**3 * A / (4 * hbar * G * math.log(2))
    assert N == pytest.approx(2.1298e87, rel=1e-3)                                    # doc said 2.13e88
    assert (1.380649e-23 * 2.725 / c) * N**2 / 1e10 == pytest.approx(5.689e133, rel=1e-3)  # doc said 5.67e145

def test_bits_dimensionless_planck_form():
    R = 3.5e8
    assert P.bekenstein_bits(R) == pytest.approx(4 * math.pi * R**2 / (4 * P.L_PLANCK**2 * P.LN_2), rel=1e-12)

def _system(n=4, seed=0):
    rng = np.random.default_rng(seed)
    return dict(masses=rng.uniform(1e29, 1e31, n), positions=rng.uniform(-1e12, 1e12, (n, 12)),
                velocities=rng.uniform(-1e5, 1e5, (n, 12)), radii=rng.uniform(1e7, 1e9, n),
                lyapunovs=rng.uniform(0, 0.1, n))

def test_terms_dimensionless_scale_invariance():
    # Dimensionless ratios are invariant under a change of mass unit (kg -> g) when G is rescaled consistently.
    s = _system(); a = P.psi_terms(**s)
    assert np.all(np.isfinite(a["psi"]))
    # K is pure kinematics: (1+phi+lam tau) v^2/(2c^2(1+lam tau)), independent of mass
    s2 = dict(s, masses=s["masses"] * 7.0)
    b = P.psi_terms(**s2)
    assert np.allclose(a["K"], b["K"])
    # Gravity ratios scale linearly with the OTHER masses: G m_j / (r c^2)
    assert np.allclose(b["U"], 7.0 * a["U"]) and np.allclose(b["S"], 7.0 * a["S"])

def test_distances_symmetric_and_from_positions():
    s = _system(); d = P.pairwise_distances(s["positions"])
    assert np.allclose(d, d.T) and np.allclose(np.diag(d), 0)

def test_newtonian_limit_two_body():
    # With r0 -> inf screening vanishes: S = -U exactly; U_i = -G m_j / (r c^2) for lam=0
    s = dict(masses=[2e30, 1e30], positions=[[0, 0, 0], [1e11, 0, 0]], velocities=[[0, 0, 0], [0, 0, 0]],
             radii=[1.0, 1.0], lyapunovs=[0, 0])
    t = P.psi_terms(**s, r0=float("inf"))
    assert t["S"] == pytest.approx(-t["U"])
    assert t["U"][0] == pytest.approx(-P.G * 1e30 / (1e11 * P.C**2))

def test_coincident_rejected():
    with pytest.raises(ValueError):
        P.psi_terms([1, 1], [[0, 0], [0, 0]], [[0, 0], [0, 0]], [1, 1], [0, 0])

def test_informational_term_still_dominates_with_planck_ell():
    # Honest regression: even unit-consistent, I dwarfs gravity for star-sized radii with ell = l_P.
    s = dict(masses=[1e30, 1e30], positions=[[0, 0, 0], [1e10, 0, 0]], velocities=[[0, 0, 0], [0, 0, 0]],
             radii=[3.5e8, 3.5e8], lyapunovs=[0, 0])
    t = P.psi_terms(**s)
    assert abs(t["I"][0]) > 1e50 * abs(t["U"][0])
