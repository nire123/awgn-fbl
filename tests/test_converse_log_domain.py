"""
Regression tests for the log-domain converse API.

    NoncentralTConverse.converse_rate_log      (ε → R, log-safe)
    NoncentralTConverse.converse_error_log     (R → ε, log-safe, returns ε)
    NoncentralTConverse.log_converse_error     (R → log ε, most stable)

These go through `log_pairwise_error_prob` and `log_nct_cdf` — never calling
`scipy.stats.nct.ppf` or `nct.cdf` — and therefore extend the working range
to large (n, SNR) where scipy's linear NCT returns NaN.
"""

import numpy as np
import pytest

from awgn_fbl import NoncentralTConverse


pytestmark = pytest.mark.filterwarnings("ignore")


class TestAgreementInModerateRegime:
    """Where scipy's linear path works, log version must match to ~1e-5."""

    @pytest.mark.parametrize("n,snr_db,eps", [
        (100, 0.0, 1e-3),
        (200, 0.0, 1e-3),
        (500, 0.0, 1e-3),
        (200, 3.0, 1e-4),
        (200, 0.0, 1e-8),
        (500, -2.0, 1e-3),
    ])
    def test_rate_direction(self, n, snr_db, eps):
        conv = NoncentralTConverse(n=n, snr_db=snr_db)
        R_lin = conv.converse_rate(eps)
        R_log = conv.converse_rate_log(eps)
        assert np.isfinite(R_lin) and np.isfinite(R_log)
        assert abs(R_lin - R_log) < 1e-5

    @pytest.mark.parametrize("n,snr_db,R", [
        (200, 0.0, 0.30),
        (200, 0.0, 0.40),
        (500, 0.0, 0.38),
        (1000, 3.0, 0.65),
        (100, 0.0, 0.20),
    ])
    def test_error_direction(self, n, snr_db, R):
        conv = NoncentralTConverse(n=n, snr_db=snr_db)
        e_lin = conv.converse_error(R)
        e_log = conv.converse_error_log(R)
        assert np.isfinite(e_lin) and np.isfinite(e_log)
        # Agreement in log space (the right metric for probabilities)
        assert abs(np.log(e_lin) - np.log(e_log)) < 1e-4


class TestExtremeRegimeExtension:
    """Regimes where scipy's nct.ppf returns NaN — log version should
    still produce a meaningful rate."""

    @pytest.mark.parametrize("n,snr_db,eps", [
        (1500, 5.0, 1e-3),
        (2000, 3.0, 1e-3),
        (3000, 0.0, 1e-3),
    ])
    def test_log_works_where_linear_fails(self, n, snr_db, eps):
        conv = NoncentralTConverse(n=n, snr_db=snr_db)
        R_lin = conv.converse_rate(eps)          # expected NaN
        R_log = conv.converse_rate_log(eps)      # should be finite
        assert np.isnan(R_lin), "scipy NCT was expected to fail here"
        assert np.isfinite(R_log), f"log-domain produced {R_log}"
        # Sanity: must be below capacity
        C = 0.5 * np.log2(1 + 10 ** (snr_db / 10))
        assert R_log < C


class TestRoundTrip:
    """converse_rate(ε) should invert converse_error(R) in log domain."""

    @pytest.mark.parametrize("n,snr_db,eps", [
        (200, 0.0, 1e-3),
        (500, 0.0, 1e-3),
        (200, 3.0, 1e-4),
    ])
    def test_eps_to_R_and_back(self, n, snr_db, eps):
        conv = NoncentralTConverse(n=n, snr_db=snr_db)
        R = conv.converse_rate_log(eps)
        eps_back = conv.converse_error_log(R)
        # Agree in log space
        assert abs(np.log(eps) - np.log(eps_back)) < 1e-3


class TestSanityProperties:
    """Sanity checks on the log-domain direction."""

    def test_rate_below_capacity(self):
        for n in [100, 500, 2000]:
            for snr_db in [-3, 0, 3, 6]:
                conv = NoncentralTConverse(n=n, snr_db=snr_db)
                C = 0.5 * np.log2(1 + 10 ** (snr_db / 10))
                R = conv.converse_rate_log(1e-3)
                if np.isfinite(R):
                    assert R < C

    def test_rate_monotone_in_eps(self):
        """Converse rate should increase monotonically with ε."""
        conv = NoncentralTConverse(n=200, snr_db=0.0)
        eps_values = [1e-10, 1e-7, 1e-4, 1e-2, 1e-1]
        rates = [conv.converse_rate_log(e) for e in eps_values]
        assert all(rates[i] < rates[i + 1] for i in range(len(rates) - 1))

    def test_rate_monotone_in_n(self):
        """Converse rate should increase with n (fixed SNR, ε)."""
        rates = []
        for n in [50, 200, 1000, 5000]:
            conv = NoncentralTConverse(n=n, snr_db=0.0)
            R = conv.converse_rate_log(1e-3)
            rates.append(R)
        # Monotone non-decreasing, no NaN
        assert all(np.isfinite(r) for r in rates)
        assert all(rates[i] < rates[i + 1] for i in range(len(rates) - 1))

    def test_log_vs_linear_error_in_deep_tail(self):
        """log_converse_error should give a finite log where linear underflows.

        For the converse map R → ε, *smaller* R means *smaller* ε
        (rarer achievable error). Pick R well below capacity.
        """
        conv = NoncentralTConverse(n=1000, snr_db=3.0)
        # Capacity ≈ 0.7913 at SNR=3 dB.  R=0.4 is well below.
        R = 0.4
        log_eps = conv.log_converse_error(R)
        assert np.isfinite(log_eps)
        assert log_eps < -20  # very small error probability


class TestRCUExtendedRange:
    """RCU+ uses the F-evaluator which now defaults to log-domain. This
    propagates the extended working range to the achievability side.
    """

    @pytest.mark.parametrize("n,snr_db,eps", [
        (1500, 5.0, 1e-3),
        (2000, 3.0, 1e-3),
        (3000, 0.0, 1e-3),
        (5000, 0.0, 1e-3),
    ])
    def test_rcu_works_where_linear_nct_fails(self, n, snr_db, eps):
        from awgn_fbl import RCUAchievable
        rcu = RCUAchievable(n=n, snr_db=snr_db)
        R_rcu = rcu.achievable_rate(eps)

        # Must be a finite, non-negative, below-capacity rate
        assert np.isfinite(R_rcu)
        assert R_rcu > 0
        C = 0.5 * np.log2(1 + 10 ** (snr_db / 10))
        assert R_rcu < C

        # And below the converse (the fundamental sanity property)
        R_nct = NoncentralTConverse(n=n, snr_db=snr_db).converse_rate_log(eps)
        assert R_rcu <= R_nct + 1e-4

    def test_converse_achievable_gap_shrinks_with_n(self):
        """Sanity: gap should decrease as n grows, well into the extended
        regime where the linear pipeline would have stopped at NaN."""
        from awgn_fbl import RCUAchievable
        gaps = []
        for n in [500, 1000, 2000, 3000]:
            conv = NoncentralTConverse(n=n, snr_db=3.0)
            rcu = RCUAchievable(n=n, snr_db=3.0)
            R_nct = conv.converse_rate_log(1e-3)
            R_rcu = rcu.achievable_rate(1e-3)
            gaps.append(R_nct - R_rcu)
        # monotone non-increasing
        for i in range(len(gaps) - 1):
            assert gaps[i + 1] <= gaps[i] + 1e-4, f"gaps = {gaps}"
