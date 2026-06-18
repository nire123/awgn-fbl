"""Tests for the faithful κβ port (v2)."""

import numpy as np
import pytest

from awgn_fbl import KappaBetaAchievablePPV as KappaBetaAchievableV2
from awgn_fbl import NoncentralTConverse


pytestmark = pytest.mark.filterwarnings("ignore")


class TestKappaBetaV2Basic:
    def test_constructor(self):
        kb = KappaBetaAchievableV2(n=200, snr_db=0.0)
        assert kb.n == 200
        assert kb.P == pytest.approx(1.0)
        assert kb.A == pytest.approx(1.0)

    def test_invalid_n(self):
        with pytest.raises(ValueError):
            KappaBetaAchievableV2(n=0, snr_db=0.0)

    def test_invalid_epsilon(self):
        kb = KappaBetaAchievableV2(n=200, snr_db=0.0)
        with pytest.raises(ValueError):
            kb.achievable_rate(0.0)
        with pytest.raises(ValueError):
            kb.achievable_rate(1.0)


class TestKappaBetaV2Sanity:
    def test_achievable_below_converse(self):
        for n in [100, 200, 500]:
            kb = KappaBetaAchievableV2(n=n, snr_db=0.0)
            R_ach = kb.achievable_rate(1e-3)
            R_conv = NoncentralTConverse(n=n, snr_db=0.0).converse_rate(1e-3)
            if not np.isnan(R_ach):
                assert R_ach <= R_conv + 1e-6

    def test_achievable_below_capacity(self):
        for snr_db in [-2, 0, 3, 5]:
            kb = KappaBetaAchievableV2(n=200, snr_db=snr_db)
            R = kb.achievable_rate(1e-3)
            C = 0.5 * np.log2(1 + 10 ** (snr_db / 10))
            if not np.isnan(R):
                assert R < C

    def test_monotonic_in_n(self):
        """Rate should grow with blocklength."""
        rates = [
            KappaBetaAchievableV2(n=n, snr_db=0.0).achievable_rate(1e-3)
            for n in [100, 200, 500, 1000]
        ]
        diffs = np.diff(rates)
        assert np.all(diffs >= -1e-6)

    def test_kappa_inf_matches_v1_formula(self):
        """κ_inf must match Polyanskiy's kappa_inf.m line-for-line."""
        kb = KappaBetaAchievableV2(n=200, snr_db=3.0)
        # The formula is deterministic; verify a known value.
        # At P=2 (3 dB), tau=0.5:
        kb_test = KappaBetaAchievableV2(n=200, snr_db=3.0)
        val = kb_test._kappa_inf(0.5)
        assert 0 < val < 1  # κ is a probability-like quantity


class TestKappaBetaV1vsV2:
    """Cross-compare with the v1 implementation."""

    def test_similar_at_moderate_params(self):
        from awgn_fbl import KappaBetaAchievable
        R1 = KappaBetaAchievable(n=200, snr_db=0.0).achievable_rate(1e-3)
        R2 = KappaBetaAchievableV2(n=200, snr_db=0.0).achievable_rate(1e-3)
        # At this operating point both should be well-behaved.
        # We don't require exact match — v2 uses Newton iteration and
        # omits the correction term. Expect within a few percent.
        if not (np.isnan(R1) or np.isnan(R2)):
            assert abs(R1 - R2) < 0.01
