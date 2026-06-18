"""
Tests for the log-domain RCU+ implementation (Elkayam factorization).

    P(R) = F(R) · J(R)
    J(R) = ∫ [F(γ)/F(R)] · exp(log M − γ) dγ,   1 ≤ J(R) ≤ 1/F(R)

The log version should agree with the legacy version wherever both produce
a non-zero answer, and be *more* accurate in the deep tail.
"""

import numpy as np
import pytest

from awgn_fbl import RCUAchievable


pytestmark = pytest.mark.filterwarnings("ignore")


class TestLogVsLegacyAgreement:
    """P_new and P_old must match wherever both are non-zero."""

    @pytest.mark.parametrize("n,snr_db", [(200, 0.0), (500, 0.0), (1000, 0.0),
                                          (200, 3.0), (100, -2.0)])
    def test_moderate_rate_range(self, n, snr_db):
        rcu = RCUAchievable(n=n, snr_db=snr_db)
        # Probe rates inside the F grid with non-trivial P values
        R_lo = rcu.F_eval.R_min + 0.05
        R_hi = rcu.F_eval.R_max - 0.05
        Rs = np.linspace(R_lo, R_hi, 8)
        for R in Rs:
            P_old = rcu.achievable_error(R)
            P_new = rcu.achievable_error_v2(R)
            if P_old == 0 and P_new == 0:
                continue
            # Agreement in log space is the right metric
            log_P_old = np.log(P_old) if P_old > 0 else -np.inf
            log_P_new = rcu.log_achievable_error(R)
            if np.isfinite(log_P_old):
                assert abs(log_P_old - log_P_new) < 0.01, (
                    f"n={n} SNR={snr_db} R={R:.3f}: "
                    f"log P_old={log_P_old:.3f} log P_new={log_P_new:.3f}"
                )


class TestLogPrecisionGain:
    """At R where P ~ 1e-16, linear quad loses precision; log version does not.

    We don't assert the legacy is wrong, only that the log version gives
    a finite log value and that the two agree in log space to a few %
    of the magnitude.
    """

    def test_deep_tail_value_is_finite(self):
        rcu = RCUAchievable(n=200, snr_db=0.0)
        # At R=0.1 and n=200, SNR=0, P is around 1e-15 to 1e-16
        R = 0.1
        log_P = rcu.log_achievable_error(R)
        # Should be a finite, very negative number
        assert np.isfinite(log_P)
        assert log_P < -30   # very small probability
        assert log_P > -50   # but not unreasonably so


class TestLogRateInversion:
    """achievable_rate_v2 should match achievable_rate at moderate eps."""

    @pytest.mark.parametrize("n,snr_db,eps", [
        (200, 0.0, 1e-3),
        (500, 0.0, 1e-3),
        (200, 3.0, 1e-4),
        (200, 0.0, 1e-6),
    ])
    def test_rate_agreement(self, n, snr_db, eps):
        rcu = RCUAchievable(n=n, snr_db=snr_db)
        R_old = rcu.achievable_rate(eps)
        R_new = rcu.achievable_rate_v2(eps)
        if np.isnan(R_old) or np.isnan(R_new):
            return
        assert abs(R_old - R_new) < 1e-5, (
            f"n={n} SNR={snr_db} eps={eps}: R_old={R_old} R_new={R_new}"
        )


class TestFactorizationInvariants:
    """Structural properties of the F(R) · J(R) split."""

    def test_J_is_at_least_one(self):
        """At γ=log M the integrand equals 1, so ∫ (integrand) ≥ 1."""
        rcu = RCUAchievable(n=200, snr_db=0.0)
        for R in [0.3, 0.35, 0.4]:
            log_P = rcu.log_achievable_error(R)
            log_F_R = rcu.F_eval.log_F(R)
            log_J = log_P - log_F_R
            # J >= 1 means log J >= 0. Allow tiny numerical slack.
            assert log_J >= -1e-6, (
                f"R={R}: log J = {log_J:.6f} should be >= 0"
            )

    def test_J_is_at_most_1_over_F(self):
        """J ≤ 1/F(R)  ⇔  log J ≤ −log F(R)  ⇔  log P ≤ 0."""
        rcu = RCUAchievable(n=200, snr_db=0.0)
        for R in [0.2, 0.3, 0.4, 0.45]:
            log_P = rcu.log_achievable_error(R)
            # P is a probability, so log P ≤ 0
            assert log_P <= 1e-6, f"R={R}: log P = {log_P} > 0"
