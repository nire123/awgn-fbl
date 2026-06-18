"""
Tests for the log-domain NCT CDF and log-domain Lemma 1 implementations.

Coverage:
    A. Lemma 1:    log_pairwise_error_prob  vs  log(_pairwise_error_prob)
    B. NCT CDF:    log_nct_cdf  vs  scipy.stats.nct at moderate args
    C. Central-t:  log_nct_cdf(nc=0)  vs  scipy.stats.t.logcdf  (exact reference)
    D. Tail extension: log versions produce finite values where
       linear scipy.stats.nct / linear Lemma 1 underflow to 0 / NaN.
"""

import numpy as np
import pytest
from scipy import stats

from awgn_fbl import NoncentralTConverse, log_nct_cdf, log_nct_sf


pytestmark = pytest.mark.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# A. Lemma 1 log vs linear
# ---------------------------------------------------------------------------

class TestLemma1LogVsLinear:
    @pytest.mark.parametrize("n,snr_db", [(50, 0.0), (200, 0.0),
                                          (500, 3.0), (1000, 3.0)])
    def test_agrees_with_linear(self, n, snr_db):
        """Where linear Lemma 1 is well-defined, log version matches."""
        conv = NoncentralTConverse(n=n, snr_db=snr_db)
        for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
            P = conv._pairwise_error_prob(t)
            if P <= 0:
                continue
            log_lin = np.log(P)
            log_new = conv.log_pairwise_error_prob(t)
            # Absolute agreement in log space (drifts with n due to grid, so
            # we allow ~1e-3 which is well below any practical use).
            assert abs(log_new - log_lin) < 1e-3

    def test_tail_extension(self):
        """Log version remains finite where linear underflows to 0."""
        conv = NoncentralTConverse(n=2000, snr_db=6.0)
        # At t=0.9 or t=0.99, P is around 10^-1500 — linear can't represent it
        for t in [0.9, 0.99]:
            P_lin = conv._pairwise_error_prob(t)
            log_log = conv.log_pairwise_error_prob(t)
            assert P_lin == 0.0, "Linear should underflow here"
            assert np.isfinite(log_log), "Log version should be finite"
            assert log_log < -500, "Should be a very large negative number"


# ---------------------------------------------------------------------------
# B. log_nct_cdf vs scipy at MODERATE args
# ---------------------------------------------------------------------------

class TestLogNctCdfVsScipy:
    @pytest.mark.parametrize("df", [5, 20, 50, 200, 500])
    def test_near_mode(self, df):
        """At args near the distribution's center, scipy is reliable;
        log_nct_cdf should match to near machine precision."""
        nc = np.sqrt(df)  # realistic magnitude for our AWGN use case
        for dx in [-2, 0, 2, 5, 10]:
            x = nc + dx
            scipy_lc = stats.nct.logcdf(x, df=df, nc=nc)
            our_lc = log_nct_cdf(x, df=df, nc=nc)
            if np.isfinite(scipy_lc):
                assert abs(our_lc - scipy_lc) < 1e-6, (
                    f"df={df} nc={nc} x={x}: scipy={scipy_lc} ours={our_lc}"
                )


# ---------------------------------------------------------------------------
# C. Central-t cross-check (nc=0): scipy.stats.t is exact reference
# ---------------------------------------------------------------------------

class TestCentralTCrossCheck:
    """At nc=0, the NCT is the central t, which scipy computes accurately.
    This is our ground-truth independent reference.
    """

    @pytest.mark.parametrize("df", [5, 20, 50, 100, 500])
    def test_central_t_agreement(self, df):
        for x in [-10, -5, -2, 0, 2, 5, 10]:
            t_ref = stats.t.logcdf(x, df=df)
            our = log_nct_cdf(x, df=df, nc=0.0)
            if np.isfinite(t_ref):
                assert abs(our - t_ref) < 1e-6, (
                    f"df={df} x={x}: t.logcdf={t_ref} ours={our}"
                )


# ---------------------------------------------------------------------------
# D. Tail extension: we work where scipy NCT breaks
# ---------------------------------------------------------------------------

class TestNctTailExtension:
    def test_finite_where_scipy_returns_nan(self):
        """scipy.stats.nct returns NaN at certain x<nc, nc>0 combos.
        Ours should give finite values."""
        df, nc = 50, 5
        for x in [-30, -50, -100]:
            scipy_lc = stats.nct.logcdf(x, df=df, nc=nc)
            our_lc = log_nct_cdf(x, df=df, nc=nc)
            assert np.isfinite(our_lc)
            assert our_lc < -100  # deep in the tail

    def test_sf_and_cdf_complement(self):
        """log_nct_cdf(x) + log_nct_sf(x) should be close to log(1) = 0 in the
        linear-combined sense: exp(logcdf) + exp(logsf) ≈ 1."""
        df, nc = 20, 4
        for x in [nc - 3, nc, nc + 3]:
            lc = log_nct_cdf(x, df=df, nc=nc)
            ls = log_nct_sf(x, df=df, nc=nc)
            # exp(lc) + exp(ls) should be approximately 1
            tot = np.exp(lc) + np.exp(ls)
            assert abs(tot - 1.0) < 1e-3


# ---------------------------------------------------------------------------
# E. Internal monotonicity sanity
# ---------------------------------------------------------------------------

class TestMonotonicity:
    def test_log_cdf_monotone_in_x(self):
        df, nc = 50, 5
        xs = np.linspace(-20, 20, 50)
        vals = [log_nct_cdf(x, df=df, nc=nc) for x in xs]
        # log CDF should be monotone increasing
        diffs = np.diff(vals)
        assert np.all(diffs >= -1e-6)
