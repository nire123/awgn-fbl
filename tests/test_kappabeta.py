import numpy as np
import pytest

from awgn_fbl import KappaBetaAchievable
from awgn_fbl import NoncentralTConverse


@pytest.fixture
def kb():
    return KappaBetaAchievable(n=200, snr_db=0.0)


class TestKnownValues:
    def test_rate_n200_snr0(self, kb):
        """Doc says κβ gives R ≈ 0.2811 at n=200, SNR=0dB, ε=1e-3."""
        R = kb.achievable_rate(1e-3)
        assert 0.20 < R < 0.35

    def test_below_converse(self, kb):
        R_kb = kb.achievable_rate(1e-3)
        R_conv = NoncentralTConverse(n=200, snr_db=0.0).converse_rate(1e-3)
        assert R_kb < R_conv


class TestMonotonicity:
    def test_higher_eps_gives_higher_rate(self, kb):
        R_strict = kb.achievable_rate(1e-4)
        R_relaxed = kb.achievable_rate(1e-2)
        assert R_relaxed > R_strict

    def test_higher_snr_gives_higher_rate(self):
        R_lo = KappaBetaAchievable(n=200, snr_db=0.0).achievable_rate(1e-3)
        R_hi = KappaBetaAchievable(n=200, snr_db=5.0).achievable_rate(1e-3)
        assert R_hi > R_lo


class TestValidation:
    def test_invalid_n(self):
        with pytest.raises(ValueError):
            KappaBetaAchievable(n=0, snr_db=0.0)


class TestAcrossSNR:
    @pytest.mark.parametrize("snr_db", [0.0, 3.0, 5.0])
    def test_below_capacity(self, snr_db):
        kb = KappaBetaAchievable(n=200, snr_db=snr_db)
        C = 0.5 * np.log2(1 + 10 ** (snr_db / 10))
        R = kb.achievable_rate(1e-3)
        assert 0 < R < C
