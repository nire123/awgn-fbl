"""
Tests for :class:`awgn_fbl.ErsegheConverse` — the robust (Temme) evaluation of
the relaxed (Q_Y = N(0,(1+P)I)) PPV converse.

The reference oracle is the library's scipy-``ncx2`` :class:`ChiSquaredConverse`
(exact where scipy is well-behaved); Erseghe must match it there and stay
finite past scipy's NaN wall.
"""

import numpy as np
import pytest

from awgn_fbl import ChiSquaredConverse, ErsegheConverse, NoncentralTConverse


pytestmark = pytest.mark.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Exact agreement with the scipy ncx2 reference where scipy works
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", [50, 100, 200, 500])
@pytest.mark.parametrize("snr_db", [0.0, 3.0, 6.0])
@pytest.mark.parametrize("eps", [1e-3, 1e-6])
def test_integral_matches_scipy_ncx2(n, snr_db, eps):
    ref = ChiSquaredConverse(n=n, snr_db=snr_db).converse_rate(eps)
    assert np.isfinite(ref)
    got = ErsegheConverse(n=n, snr_db=snr_db).converse_rate(eps)
    assert got == pytest.approx(ref, rel=2e-4)


# ---------------------------------------------------------------------------
# Robustness past scipy's NaN wall
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n,snr_db", [(1000, 6.0), (2000, 9.0)])
def test_finite_where_scipy_nans(n, snr_db):
    eps = 1e-6
    scipy_ref = ChiSquaredConverse(n=n, snr_db=snr_db).converse_rate(eps)
    erseghe = ErsegheConverse(n=n, snr_db=snr_db).converse_rate(eps)
    assert np.isnan(scipy_ref)          # scipy ncx2.ppf gives up here
    assert np.isfinite(erseghe)         # Erseghe does not
    assert 0 < erseghe < 0.5 * np.log2(1 + 10 ** (snr_db / 10))  # below capacity


# ---------------------------------------------------------------------------
# The relaxed bound is strictly looser (higher rate) than the optimal NCT
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", [100, 200, 500])
def test_looser_than_nct(n):
    eps, snr_db = 1e-3, 0.0
    nct = NoncentralTConverse(n=n, snr_db=snr_db).converse_rate_log(eps)
    erseghe = ErsegheConverse(n=n, snr_db=snr_db).converse_rate(eps)
    assert erseghe > nct                 # relaxation => looser converse
    assert erseghe - nct < 0.05          # but only by an O(1/n) margin


# ---------------------------------------------------------------------------
# Round-trip eps -> R -> eps
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n,snr_db,eps", [(200, 0.0, 1e-3), (500, 3.0, 1e-6),
                                          (1000, 6.0, 1e-4)])
def test_round_trip(n, snr_db, eps):
    conv = ErsegheConverse(n=n, snr_db=snr_db)
    R = conv.converse_rate(eps)
    eps_back = conv.converse_error(R)
    assert eps_back == pytest.approx(eps, rel=1e-3)


# ---------------------------------------------------------------------------
# The two-term asymptotic agrees with the exact integral
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", [100, 500, 1000])
def test_asymptotic_matches_integral(n):
    eps, snr_db = 1e-4, 3.0
    exact = ErsegheConverse(n=n, snr_db=snr_db, method="integral").converse_rate(eps)
    asy = ErsegheConverse(n=n, snr_db=snr_db, method="asymptotic").converse_rate(eps)
    assert asy == pytest.approx(exact, rel=3e-3)


# ---------------------------------------------------------------------------
# Monotonicity
# ---------------------------------------------------------------------------

def test_rate_monotone_in_eps():
    conv = ErsegheConverse(n=200, snr_db=0.0)
    rates = [conv.converse_rate(e) for e in [1e-8, 1e-6, 1e-4, 1e-2, 1e-1]]
    assert all(b > a for a, b in zip(rates, rates[1:]))


def test_rate_monotone_in_snr():
    rates = [ErsegheConverse(n=200, snr_db=s).converse_rate(1e-3)
             for s in [-2, 0, 2, 4, 6, 8]]
    assert all(b > a for a, b in zip(rates, rates[1:]))


def test_invalid_epsilon_raises():
    conv = ErsegheConverse(n=200, snr_db=0.0)
    with pytest.raises(ValueError):
        conv.converse_rate(0.0)
    with pytest.raises(ValueError):
        conv.converse_rate(1.0)
