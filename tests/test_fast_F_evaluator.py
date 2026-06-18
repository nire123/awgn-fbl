import numpy as np
import pytest

from awgn_fbl import FastFREvaluator
from awgn_fbl import NoncentralTConverse


@pytest.fixture(scope="module")
def feval():
    """Shared evaluator — expensive to build, so module-scoped."""
    return FastFREvaluator(n=200, snr_db=0.0, method="nct")


# ---------------------------------------------------------------------------
# Basic properties
# ---------------------------------------------------------------------------

class TestGridProperties:
    def test_grid_sorted(self, feval):
        assert np.all(np.diff(feval.R_grid) > 0)

    def test_grid_nonempty(self, feval):
        assert len(feval.R_grid) > 50

    def test_eps_increasing_with_R(self, feval):
        """Higher rate → higher ε needed (converse is more relaxed)."""
        Rs = np.linspace(feval.R_min + 0.01, feval.R_max - 0.01, 50)
        eps_vals = feval(Rs)
        assert np.all(np.diff(eps_vals) > 0)

    def test_bounds_consistent(self, feval):
        assert feval.R_min < feval.R_max
        assert feval.eps_min < feval.eps_max


# ---------------------------------------------------------------------------
# Accuracy vs direct converse computation
# ---------------------------------------------------------------------------

class TestAccuracy:
    @pytest.mark.parametrize("eps", [1e-2, 1e-3, 1e-5])
    def test_matches_direct_converse(self, feval, eps):
        """F(R) at the converse rate should recover ε."""
        nct = NoncentralTConverse(n=200, snr_db=0.0)
        R = nct.converse_rate(eps)
        eps_interp = feval(R)
        assert eps_interp == pytest.approx(eps, rel=0.05)

    def test_vectorised_call(self, feval):
        Rs = np.linspace(feval.R_min + 0.01, feval.R_max - 0.01, 10)
        eps_arr = feval(Rs)
        assert eps_arr.shape == (10,)
        assert np.all(eps_arr > 0)
        assert np.all(eps_arr < 1)

    def test_scalar_returns_float(self, feval):
        val = feval(0.3)
        assert isinstance(val, float)


# ---------------------------------------------------------------------------
# Monotonicity
# ---------------------------------------------------------------------------

class TestMonotonicity:
    def test_F_increasing(self, feval):
        """F(R) should be monotonically increasing (higher R needs higher ε)."""
        Rs = np.linspace(feval.R_min + 0.01, feval.R_max - 0.01, 50)
        eps_vals = feval(Rs)
        assert np.all(np.diff(eps_vals) > 0)


# ---------------------------------------------------------------------------
# Convenience method
# ---------------------------------------------------------------------------

class TestConvenienceMethod:
    def test_converse_rate(self, feval):
        R = feval.converse_rate(1e-3)
        assert 0.30 < R < 0.40


# ---------------------------------------------------------------------------
# Chi-squared method variant
# ---------------------------------------------------------------------------

class TestChi2Method:
    def test_chi2_evaluator_builds(self):
        feval = FastFREvaluator(n=200, snr_db=0.0, method="chi2")
        assert feval.R_min < feval.R_max

    def test_chi2_agrees_with_nct_at_known_eps(self):
        """Both methods should give similar R for the same ε."""
        feval_nct = FastFREvaluator(n=200, snr_db=0.0, method="nct")
        feval_chi2 = FastFREvaluator(n=200, snr_db=0.0, method="chi2")
        R_nct = feval_nct.converse_rate(1e-3)
        R_chi2 = feval_chi2.converse_rate(1e-3)
        assert R_nct == pytest.approx(R_chi2, abs=0.01)
