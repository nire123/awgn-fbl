import numpy as np
import pytest

from awgn_fbl import (
    AWGNConverseBase,
    NoncentralTConverse,
    ChiSquaredConverse,
    SolidAngleConverse,
    awgn_converse_rate,
    awgn_converse_error,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def nct():
    return NoncentralTConverse(n=200, snr_db=0.0)


@pytest.fixture
def chi2():
    return ChiSquaredConverse(n=200, snr_db=0.0)


# ---------------------------------------------------------------------------
# Shannon capacity
# ---------------------------------------------------------------------------

class TestShannonCapacity:
    def test_known_value_0dB(self, nct):
        # C = 0.5 * log2(1 + 1) = 0.5
        assert nct.shannon_capacity() == pytest.approx(0.5, abs=1e-10)

    def test_known_value_3dB(self):
        # SNR=2 → C = 0.5 * log2(3) ≈ 0.7925
        c = NoncentralTConverse(n=100, snr_db=10 * np.log10(2))
        assert c.shannon_capacity() == pytest.approx(0.5 * np.log2(3), abs=1e-10)

    def test_both_methods_agree(self, nct, chi2):
        assert nct.shannon_capacity() == pytest.approx(chi2.shannon_capacity(), abs=1e-12)


# ---------------------------------------------------------------------------
# Known-value tests (from documentation)
# ---------------------------------------------------------------------------

class TestKnownValues:
    """At n=200, SNR=0dB, ε=10⁻³ the converse should give R ≈ 0.34 bits/use."""

    def test_nct_rate(self, nct):
        R = nct.converse_rate(1e-3)
        assert 0.30 < R < 0.40

    def test_chi2_rate(self, chi2):
        R = chi2.converse_rate(1e-3)
        assert 0.30 < R < 0.40


# ---------------------------------------------------------------------------
# Cross-validation: both methods should be close
# ---------------------------------------------------------------------------

class TestCrossValidation:
    @pytest.mark.parametrize("eps", [1e-2, 1e-3, 1e-4])
    def test_methods_agree_n200(self, eps):
        nct = NoncentralTConverse(n=200, snr_db=0.0)
        chi2 = ChiSquaredConverse(n=200, snr_db=0.0)
        R_nct = nct.converse_rate(eps)
        R_chi2 = chi2.converse_rate(eps)
        assert R_nct == pytest.approx(R_chi2, abs=0.02)

    @pytest.mark.parametrize("snr_db", [-2.0, 0.0, 3.0, 5.0, 10.0])
    def test_methods_agree_across_snr(self, snr_db):
        """Both methods should agree closely across SNR range."""
        nct = NoncentralTConverse(n=200, snr_db=snr_db)
        chi2 = ChiSquaredConverse(n=200, snr_db=snr_db)
        R_nct = nct.converse_rate(1e-3)
        R_chi2 = chi2.converse_rate(1e-3)
        assert R_nct == pytest.approx(R_chi2, abs=0.01)


# ---------------------------------------------------------------------------
# Bidirectional round-trip (NCT only)
# ---------------------------------------------------------------------------

class TestBidirectional:
    @pytest.mark.parametrize("eps", [1e-2, 1e-3, 1e-4])
    def test_round_trip(self, nct, eps):
        R = nct.converse_rate(eps)
        eps_back = nct.converse_error(R)
        assert eps_back == pytest.approx(eps, rel=0.05)


# ---------------------------------------------------------------------------
# Monotonicity
# ---------------------------------------------------------------------------

class TestMonotonicity:
    def test_higher_epsilon_gives_higher_rate(self, nct):
        R_low = nct.converse_rate(1e-4)
        R_high = nct.converse_rate(1e-1)
        assert R_high > R_low

    def test_higher_snr_gives_higher_rate(self):
        R_low = NoncentralTConverse(n=200, snr_db=0.0).converse_rate(1e-3)
        R_high = NoncentralTConverse(n=200, snr_db=5.0).converse_rate(1e-3)
        assert R_high > R_low

    def test_rate_below_capacity(self, nct):
        R = nct.converse_rate(1e-3)
        assert R < nct.shannon_capacity()


# ---------------------------------------------------------------------------
# Edge cases / input validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_invalid_n(self):
        with pytest.raises(ValueError):
            NoncentralTConverse(n=0, snr_db=0.0)

    def test_epsilon_zero(self, nct):
        with pytest.raises(ValueError):
            nct.converse_rate(0.0)

    def test_epsilon_one(self, nct):
        with pytest.raises(ValueError):
            nct.converse_rate(1.0)

    def test_negative_rate(self, nct):
        with pytest.raises(ValueError):
            nct.converse_error(-0.1)


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

class TestConvenience:
    def test_awgn_converse_rate_nct(self):
        R = awgn_converse_rate(n=200, epsilon=1e-3, snr_db=0.0, method="nct")
        assert 0.30 < R < 0.40

    def test_awgn_converse_rate_chi2(self):
        R = awgn_converse_rate(n=200, epsilon=1e-3, snr_db=0.0, method="chi2")
        assert 0.30 < R < 0.40

    def test_awgn_converse_rate_solid_angle(self):
        R = awgn_converse_rate(n=30, epsilon=1e-3, snr_db=0.0, method="solid_angle")
        assert 0 < R < 0.5

    def test_awgn_converse_error(self):
        eps = awgn_converse_error(n=200, rate_bits=0.34, snr_db=0.0)
        assert 0 < eps < 1


# ---------------------------------------------------------------------------
# Shannon solid angle (1959)
# ---------------------------------------------------------------------------

class TestSolidAngle:
    """Shannon's solid angle method — only for small n."""

    @pytest.mark.parametrize("snr_db", [0.0, 3.0, 5.0])
    def test_matches_nct_small_n(self, snr_db):
        """Solid angle and NCT should give identical results for small n."""
        n = 30
        nct = NoncentralTConverse(n=n, snr_db=snr_db)
        sa = SolidAngleConverse(n=n, snr_db=snr_db)
        R_nct = nct.converse_rate(1e-3)
        R_sa = sa.converse_rate(1e-3)
        assert R_nct == pytest.approx(R_sa, abs=1e-6)

    @pytest.mark.parametrize("eps", [1e-2, 1e-3, 1e-4])
    def test_matches_nct_across_epsilon(self, eps):
        n = 20
        nct = NoncentralTConverse(n=n, snr_db=0.0)
        sa = SolidAngleConverse(n=n, snr_db=0.0)
        R_nct = nct.converse_rate(eps)
        R_sa = sa.converse_rate(eps)
        assert R_nct == pytest.approx(R_sa, abs=1e-6)

    def test_warns_large_n(self):
        with pytest.warns(UserWarning, match="unreliable"):
            SolidAngleConverse(n=200, snr_db=0.0)

    def test_rate_below_capacity(self):
        sa = SolidAngleConverse(n=30, snr_db=5.0)
        R = sa.converse_rate(1e-3)
        assert 0 < R < sa.shannon_capacity()
