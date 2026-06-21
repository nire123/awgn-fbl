"""
Automated cross-checks against the Ahmed–Ambroze–Tomlinson (2007) algorithms.

The reference reimplementations live in ``analysis/verify_ahmed.py``:

* ``ahmed_solid_angle_ratio`` — the cone/solid-angle (≡ pairwise-error /
  ``t → R``) quantity via Ahmed's closed-form trigonometric reduction;
* ``ahmed_nct_cdf_log`` — the non-central t CDF (``t → ε``) via Ahmed's
  log-domain incomplete-beta recursion (their Table 1).

These tests lock in the agreement with the library where Ahmed's forms are
valid, and document the large-n breakdown of the closed-form geometric sum
(alternating binomials → catastrophic cancellation), which is exactly why the
library evaluates the same geometric quantity through the log-domain Lemma 1
instead.
"""

import math

import numpy as np
import pytest
from scipy import stats

from analysis.verify_ahmed import ahmed_solid_angle_ratio, ahmed_nct_cdf_log
from awgn_fbl import log_nct_cdf
from awgn_fbl._pairwise import pairwise_error_prob


pytestmark = pytest.mark.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# 1. Ahmed's geometric trig reduction == our Lemma 1, where it is well-behaved
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", [6, 10, 20, 40, 60, 80])
@pytest.mark.parametrize("t", [0.3, 0.5, 0.7])
def test_ahmed_trig_matches_lemma1_small_n(n, t):
    theta = math.acos(t)
    got = ahmed_solid_angle_ratio(theta, n)
    ref = pairwise_error_prob(t, n)        # the library's Lemma 1
    assert got == pytest.approx(ref, rel=1e-6)


# ---------------------------------------------------------------------------
# 2. The closed-form geometric sum is *not* stable at large n
#    (documents why the library uses the log-domain Lemma 1 instead)
# ---------------------------------------------------------------------------

def test_ahmed_trig_loses_precision_at_large_n():
    t = 0.5
    theta = math.acos(t)
    ref = pairwise_error_prob(t, 400)              # ~4.7e-27, correct
    ahmed = ahmed_solid_angle_ratio(theta, 400)
    # Ahmed's float reduction has lost all precision here (wrong by many orders,
    # often even the sign); the library's Lemma 1 stays exact.
    assert abs(ahmed - ref) > 1e3 * abs(ref)


# ---------------------------------------------------------------------------
# 3. Ahmed's log-domain incomplete-beta NCT == scipy and our log_nct_cdf
#    in the body of the distribution
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n,snr_db", [(50, 0.0), (200, 0.0), (200, 3.0)])
def test_ahmed_nct_matches_scipy_and_ours_in_body(n, snr_db):
    f = n - 1
    delta = math.sqrt(n * 10 ** (snr_db / 10))
    t = delta * 0.6                                 # in the body, CDF ~ 1e-3..1e-7
    P, _ = ahmed_nct_cdf_log(t, f, delta)
    assert P == pytest.approx(stats.nct.cdf(t, df=f, nc=delta), rel=1e-5)
    assert P == pytest.approx(math.exp(log_nct_cdf(t, df=f, nc=delta)), rel=1e-5)


# ---------------------------------------------------------------------------
# 4. Deep tail: our log_nct_cdf stays accurate; Ahmed's linear 1-(.) recovery
#    floors (~1e-12), which is the one methodological difference between the
#    two log-domain CDF evaluations.
# ---------------------------------------------------------------------------

def test_ours_log_nct_reaches_deeper_than_ahmed():
    n = 200
    f = n - 1
    delta = math.sqrt(n * 1.0)                       # 0 dB
    t = delta * 0.3                                  # deep tail: true CDF ~ 1e-22

    true_log = log_nct_cdf(t, df=f, nc=delta)        # ours, natural log
    scipy_cdf = stats.nct.cdf(t, df=f, nc=delta)     # scipy still resolves it here

    # ours tracks the true (scipy) value far into the tail ...
    assert math.exp(true_log) == pytest.approx(scipy_cdf, rel=1e-3)
    assert true_log < -40                            # ~1e-18 or smaller, well past 1e-12

    # ... whereas Ahmed's recovered probability cannot reach this depth: the
    # final linear subtraction floors it many orders of magnitude above the
    # true value.
    P_ahmed, _ = ahmed_nct_cdf_log(t, f, delta)
    assert P_ahmed > 1e6 * math.exp(true_log)
