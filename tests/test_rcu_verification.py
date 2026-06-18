"""
Verification tests for the RCU+ pipeline.

The RCU+ bound depends on precomputing F(R) (the converse inverse).
The converse itself is a parametric curve over t in (0, 1), with two
independent maps:

    t -> R   via Lemma 1 pairwise error:  P(t) = (1/pi) (1-t^2)^((n-1)/2) * int ...
                                          R = -log2 P(t) / n

    t -> eps via NCT CDF (forward):       beta(t) = t^2 / (1 - t^2)
                                          threshold = sqrt(beta * (n-1))
                                          eps = NCT_CDF(threshold; df=n-1, nc=sqrt(nP))

These tests exercise the pipeline independently of the production code path
(which samples eps and inverts via NCT quantile).
"""

import numpy as np
import pytest
from scipy import stats

from awgn_fbl import NoncentralTConverse
from awgn_fbl import FastFREvaluator
from awgn_fbl import RCUAchievable


pytestmark = pytest.mark.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Helper: the "forward" t-parameterization (independent reimplementation)
# ---------------------------------------------------------------------------

def t_to_R_eps(t: float, n: int, snr: float) -> tuple[float, float]:
    """Given t in (0,1), return (R_bits, eps) independently of converse_rate."""
    mu = np.sqrt(n * snr)
    nu = n - 1

    # t -> R via Lemma 1 integral
    conv = NoncentralTConverse(n=n, snr_db=10 * np.log10(snr) if snr > 0 else -np.inf)
    P_t = conv._pairwise_error_prob(t)
    R_bits = -np.log(P_t) / n / np.log(2)

    # t -> eps via NCT CDF
    beta = t ** 2 / (1 - t ** 2)
    threshold = np.sqrt(beta * nu)
    eps = stats.nct.cdf(threshold, df=nu, nc=mu)

    return R_bits, eps


# ---------------------------------------------------------------------------
# 1. Monotonicity of t -> R and t -> eps
# ---------------------------------------------------------------------------

class TestParametricMonotonicity:
    def test_R_monotone_in_t(self):
        """R(t) must be strictly increasing on (0, 1)."""
        n, snr_db = 200, 0.0
        snr = 10 ** (snr_db / 10)
        ts = np.linspace(0.1, 0.9, 30)
        Rs = [t_to_R_eps(t, n, snr)[0] for t in ts]
        diffs = np.diff(Rs)
        assert np.all(diffs > 0), "R(t) must be strictly increasing"

    def test_eps_monotone_in_t(self):
        """eps(t) must be strictly increasing on (0, 1)."""
        n, snr_db = 200, 0.0
        snr = 10 ** (snr_db / 10)
        ts = np.linspace(0.1, 0.9, 30)
        eps_vals = [t_to_R_eps(t, n, snr)[1] for t in ts]
        diffs = np.diff(eps_vals)
        assert np.all(diffs > 0), "eps(t) must be strictly increasing"


# ---------------------------------------------------------------------------
# 2. Round-trip: stored (eps, R) pairs must be internally consistent
# ---------------------------------------------------------------------------

class TestFGridRoundTrip:
    """Each stored (eps, R) pair should recover itself under the converse."""

    @pytest.mark.parametrize("n,snr_db", [(100, 0), (200, 0), (500, 0), (200, 3)])
    def test_grid_eps_to_R_matches_stored(self, n, snr_db):
        F = FastFREvaluator(n=n, snr_db=snr_db)
        conv = NoncentralTConverse(n=n, snr_db=snr_db)
        # Re-run whichever converse flavor the F-evaluator used.
        rate_fn = conv.converse_rate_log if F.log_domain else conv.converse_rate
        idx = np.linspace(5, len(F.eps_grid) - 6, 8).astype(int)
        for i in idx:
            eps_i = F.eps_grid[i]
            R_i = F.R_grid[i]
            R_recomputed = rate_fn(eps_i)
            assert R_recomputed == pytest.approx(R_i, rel=1e-6, abs=1e-8)


# ---------------------------------------------------------------------------
# 3. t-parameterization and eps-parameterization must agree
# ---------------------------------------------------------------------------

class TestParameterizationEquivalence:
    """For each grid point, recover the implied t and verify both maps agree."""

    @pytest.mark.parametrize("n,snr_db", [(100, 0), (200, 3), (500, -2)])
    def test_t_implied_by_eps_gives_same_R(self, n, snr_db):
        F = FastFREvaluator(n=n, snr_db=snr_db)
        snr = 10 ** (snr_db / 10)
        mu = np.sqrt(n * snr)
        nu = n - 1

        idx = np.linspace(3, len(F.eps_grid) - 4, 6).astype(int)
        for i in idx:
            eps_i = F.eps_grid[i]
            R_stored = F.R_grid[i]

            # Recover t from eps via NCT quantile
            threshold = stats.nct.ppf(eps_i, df=nu, nc=mu)
            beta = threshold ** 2 / nu
            t = np.sqrt(beta / (1 + beta))

            # Forward: t -> R via Lemma 1
            R_forward, eps_forward = t_to_R_eps(t, n, snr)

            assert R_forward == pytest.approx(R_stored, rel=1e-5, abs=1e-7)
            assert eps_forward == pytest.approx(eps_i, rel=1e-5, abs=1e-10)


# ---------------------------------------------------------------------------
# 4. F(R) interpolator fidelity
# ---------------------------------------------------------------------------

class TestFInterpolatorFidelity:
    def test_F_at_grid_points_matches_stored(self):
        """F(R_grid[i]) should return eps_grid[i] to interpolation precision."""
        F = FastFREvaluator(n=200, snr_db=0.0)
        # Evaluate at each grid point; PCHIP is exact at nodes
        for i in range(len(F.R_grid)):
            eps_i = F(F.R_grid[i])
            assert eps_i == pytest.approx(F.eps_grid[i], rel=1e-10)

    def test_F_matches_direct_converse_off_grid(self):
        """Off-grid F(R) should match direct converse (bisection ground truth)
        to very high precision. PCHIP on the geometric eps_factor=0.1 grid
        achieves ~3 ppm absolute error in log(eps) in the RCU+-relevant range.
        """
        F = FastFREvaluator(n=200, snr_db=0.0)
        conv = NoncentralTConverse(n=200, snr_db=0.0)

        def exact_F(R_target):
            eps_lo, eps_hi = 1e-14, 1 - 1e-10
            for _ in range(80):
                eps_mid = np.sqrt(eps_lo * eps_hi)
                R_mid = conv.converse_rate(eps_mid)
                if R_mid < R_target:
                    eps_lo = eps_mid
                else:
                    eps_hi = eps_mid
            return np.sqrt(eps_lo * eps_hi)

        # Test in the range relevant for RCU+ integration (rates near C)
        R_test = np.linspace(0.25, 0.45, 30)
        max_err = 0.0
        for R in R_test:
            eps_interp = F(R)
            eps_direct = exact_F(R)
            err = abs(np.log(eps_interp) - np.log(eps_direct))
            max_err = max(max_err, err)

        # Empirically observed max ~3.6e-6; leave 10x headroom
        assert max_err < 5e-5, f"max abs error in log(eps) = {max_err:.3e}"

    def test_F_grid_density_does_not_affect_rcu_rate(self):
        """Refining the grid should not meaningfully change the RCU+ rate,
        confirming PCHIP on the default grid is effectively exact."""
        from awgn_fbl import RCUAchievable
        rcu_default = RCUAchievable(n=200, snr_db=0.0)
        R_default = rcu_default.achievable_rate(1e-3)

        rcu_fine = RCUAchievable(n=200, snr_db=0.0)
        rcu_fine.F_eval = FastFREvaluator(n=200, snr_db=0.0, eps_factor=0.02)
        R_fine = rcu_fine.achievable_rate(1e-3)

        # Difference should be below 1e-6 bits/use (empirically 1e-10)
        assert abs(R_default - R_fine) < 1e-6

    def test_F_monotone_increasing(self):
        """F(R) must be monotonically increasing in R (more rate -> more error)."""
        F = FastFREvaluator(n=200, snr_db=0.0)
        R_test = np.linspace(F.R_min + 1e-3, F.R_max - 1e-3, 100)
        eps_test = F(R_test)
        assert np.all(np.diff(eps_test) > 0)


# ---------------------------------------------------------------------------
# 5. RCU+ integral itself: sanity on the structure
# ---------------------------------------------------------------------------

class TestRCUIntegralStructure:
    """The RCU+ integral P(log M) = int_{log M}^inf F(gamma) exp(log M - gamma) dgamma
    has known structural properties we can verify."""

    def test_achievable_error_monotone_in_rate(self):
        """P(R) should be monotonically increasing in R."""
        rcu = RCUAchievable(n=200, snr_db=0.0)
        Rs = np.linspace(0.15, 0.35, 20)
        pes = [rcu.achievable_error(R) for R in Rs]
        # Allow tiny numerical noise
        assert np.all(np.diff(pes) > -1e-10)

    def test_achievable_error_at_low_rate_is_small(self):
        """At rates well below capacity, P(R) should be very small."""
        rcu = RCUAchievable(n=200, snr_db=0.0)
        pe = rcu.achievable_error(0.1)  # well below C=0.5
        assert pe < 1e-4

    def test_achievable_error_approaches_one_near_capacity(self):
        """As R approaches capacity, P(R) grows."""
        rcu = RCUAchievable(n=200, snr_db=0.0)
        pe_low = rcu.achievable_error(0.25)
        pe_hi = rcu.achievable_error(0.40)
        assert pe_hi > pe_low

    def test_integrand_decay(self):
        """Integrand F(gamma) * exp(log M - gamma) decays for gamma > log M."""
        rcu = RCUAchievable(n=200, snr_db=0.0)
        R_bits = 0.3
        log_M = rcu.n * R_bits * np.log(2)
        F = rcu.F_eval

        def integrand(gamma_nats):
            R_per_use = gamma_nats / (rcu.n * np.log(2))
            if R_per_use < F.R_min or R_per_use > F.R_max:
                return 0.0
            return F(R_per_use) * np.exp(log_M - gamma_nats)

        # Values at log_M, log_M+1, log_M+5 (all in nats)
        v0 = integrand(log_M)
        v1 = integrand(log_M + 1)
        v5 = integrand(log_M + 5)
        # Expected decay: exp(-1) ~0.37, exp(-5) ~0.007 multiplicative
        assert v1 < v0
        assert v5 < v1


# ---------------------------------------------------------------------------
# 6. RCU+ round-trip
# ---------------------------------------------------------------------------

class TestRCURoundTrip:
    @pytest.mark.parametrize("n,snr_db,eps", [
        (200, 0.0, 1e-3),
        (500, 0.0, 1e-3),
        (200, 3.0, 1e-4),
    ])
    def test_rate_and_error_are_inverses(self, n, snr_db, eps):
        rcu = RCUAchievable(n=n, snr_db=snr_db)
        R = rcu.achievable_rate(eps)
        eps_back = rcu.achievable_error(R)
        # Brent's method returns R such that P(R) = eps within its tolerance
        assert eps_back == pytest.approx(eps, rel=1e-3)
