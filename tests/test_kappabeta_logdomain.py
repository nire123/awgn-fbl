"""
Tests for the log-domain robustness improvements to the κβ bound:

* complementary upper-tail probabilities (``isf``/``sf`` and ``erfinv``/``erf``)
  remove the ``1 - ε`` / ``(1+τ)/2`` rounding that made the bound NaN at large
  n / high SNR / small ε;
* :func:`awgn_fbl.achievable._log_ncx2_cdf_series` evaluates the β term's
  non-central χ² lower tail in log domain, finite where scipy underflows.
"""

import numpy as np
import pytest

from awgn_fbl import KappaBetaAchievablePPV, KappaBetaAchievable
from awgn_fbl.achievable import _log_ncx2_cdf_series
from scipy import stats


pytestmark = pytest.mark.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# log_ncx2 lower-tail CDF vs scipy where scipy is well-behaved
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("df,nc,x", [
    (50, 30, 40), (200, 127, 150), (2000, 20, 1500), (500, 500, 600),
    (1000, 250, 400),
])
def test_log_ncx2_series_matches_scipy(df, nc, x):
    ref = stats.ncx2.cdf(x, df=df, nc=nc)
    assert ref > 0                       # scipy still resolves it
    got = _log_ncx2_cdf_series(x, df, nc)
    assert got == pytest.approx(np.log(ref), rel=1e-6, abs=1e-6)


def test_log_ncx2_series_finite_in_deep_tail():
    # scipy underflows to 0 / -inf here; the series stays finite
    assert stats.ncx2.cdf(146, df=8000, nc=8127) == 0.0
    val = _log_ncx2_cdf_series(146, 8000, 8127)
    assert np.isfinite(val) and val < -1000


def test_log_ncx2_series_central_chi2():
    # nc = 0 reduces to the central chi-squared lower tail
    got = _log_ncx2_cdf_series(40.0, 50, 0.0)
    assert got == pytest.approx(stats.chi2.logcdf(40.0, df=50), rel=1e-9)


# ---------------------------------------------------------------------------
# κβ stays finite and valid where it used to NaN
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n,snr_db,eps", [
    (5000, 15, 1e-15), (8000, 18, 1e-20), (2000, 20, 1e-30), (20000, 20, 1e-50),
])
def test_kappabeta_finite_and_below_capacity_extreme(n, snr_db, eps):
    C = 0.5 * np.log2(1 + 10 ** (snr_db / 10))
    R = KappaBetaAchievablePPV(n=n, snr_db=snr_db).achievable_rate(eps)
    assert np.isfinite(R)
    assert 0 < R < C                     # a valid achievability rate


def test_kappabeta_below_capacity_large_n_moderate_snr():
    # n=3000, 0 dB used to (incorrectly) exceed C with the broken windowing
    R = KappaBetaAchievablePPV(n=3000, snr_db=0.0).achievable_rate(1e-6)
    assert 0 < R < 0.5


# ---------------------------------------------------------------------------
# Unchanged where scipy already worked (no regression on reference points)
# ---------------------------------------------------------------------------

def test_kappabeta_reference_point_unchanged():
    R = KappaBetaAchievablePPV(n=200, snr_db=0.0).achievable_rate(1e-3)
    assert R == pytest.approx(0.2837, abs=1e-3)


def test_ppv_and_simple_agree():
    for n, s, e in [(200, 0.0, 1e-3), (500, 3.0, 1e-6), (1000, 6.0, 1e-9)]:
        rp = KappaBetaAchievablePPV(n=n, snr_db=s).achievable_rate(e)
        rs = KappaBetaAchievable(n=n, snr_db=s).achievable_rate(e)
        assert rp == pytest.approx(rs, rel=5e-3)


# ---------------------------------------------------------------------------
# Monotonicity preserved
# ---------------------------------------------------------------------------

def test_monotone_in_eps_deep():
    kb = KappaBetaAchievablePPV(n=1000, snr_db=6.0)
    rates = [kb.achievable_rate(e) for e in [1e-12, 1e-9, 1e-6, 1e-3]]
    assert all(b > a for a, b in zip(rates, rates[1:]))
