"""
Cross-checks of this library's converse primitives against the closed-form /
log-domain algorithms of Ahmed, Ambroze & Tomlinson, "On Computing Shannon's
Sphere Packing Bound and Applications" (ISCTA 2007).

Two independent reimplementations of Ahmed's paper are provided and compared
against the library:

1. ``ahmed_solid_angle_ratio`` — the cone/solid-angle integral
   ``∫_0^θ sin^{n-2}φ dφ / ∫_0^π sin^{n-2}φ dφ`` via Ahmed's closed-form
   trigonometric reduction (their eqs 6-7).  This is the quantity the library
   computes via Lemma 1 (``pairwise_error_prob``); the two should agree.

2. ``ahmed_nct_cdf_log`` — the non-central t CDF via Ahmed's log-domain
   incomplete-beta recursion (their Table 1).  Compared against the library's
   ``log_nct_cdf`` and ``scipy.stats.nct``.

Run:  python analysis/verify_ahmed.py
"""

from __future__ import annotations

import math

import numpy as np
from scipy import special, stats

from awgn_fbl import log_nct_cdf
from awgn_fbl._pairwise import pairwise_error_prob


# ===========================================================================
# 1. Ahmed's closed-form solid-angle integral (eqs 6-7) vs Lemma 1
# ===========================================================================

def _int_sin_pow(theta: float, m: int) -> float:
    """Definite integral ∫_0^theta sin^m(x) dx via Ahmed's reduction (eqs 6-7).

    m = n - 2.  Uses the even branch (eq 6) for even m and the odd branch
    (eq 7) for odd m.  Exact in real arithmetic; the binomial coefficients
    grow like 2^m and alternate in sign, so the float evaluation loses
    precision to cancellation as m grows — exactly the instability that
    motivates the library's log-domain Lemma 1.
    """
    if m == 0:
        return theta
    if m == 1:
        return 1.0 - math.cos(theta)

    if m % 2 == 0:  # eq (6): m = 2p
        p = m // 2
        out = special.comb(2 * p, p) / (2.0 ** (2 * p)) * theta
        acc = 0.0
        for k in range(p):
            acc += (-1) ** k * special.comb(2 * p, k) * \
                math.sin((2 * p - 2 * k) * theta) / (2 * p - 2 * k)
        out += (-1) ** p / (2.0 ** (2 * p - 1)) * acc
        return out
    else:           # eq (7): m = 2q + 1
        q = (m - 1) // 2
        pref = (-1) ** (q + 1) / (2.0 ** (2 * q))

        def F(x):
            acc = 0.0
            for k in range(q + 1):
                acc += (-1) ** k * special.comb(2 * q + 1, k) * \
                    math.cos((2 * q + 1 - 2 * k) * x) / (2 * q + 1 - 2 * k)
            return pref * acc

        return F(theta) - F(0.0)


def ahmed_solid_angle_ratio(theta: float, n: int) -> float:
    """Ω(θ)/Ω(π) for dimension n via Ahmed's closed form (eqs 6-7)."""
    num = _int_sin_pow(theta, n - 2)
    den = _int_sin_pow(math.pi, n - 2)
    return num / den


def task1():
    print("=" * 72)
    print("TASK 1  Ahmed closed-form solid angle (eqs 6-7)  vs  Lemma 1")
    print("=" * 72)
    print("  P(rho_hat >= t) with t = cos(theta) must equal Omega(theta)/Omega(pi).")
    print()
    print(f"{'n':>5} {'t':>6} {'Lemma1 P(t)':>16} {'Ahmed ratio':>16} {'rel.err':>12}")
    for n in [10, 20, 40, 60, 80, 120, 200, 400]:
        t = 0.5
        theta = math.acos(t)
        lemma1 = pairwise_error_prob(t, n)
        ahmed = ahmed_solid_angle_ratio(theta, n)
        rel = abs(ahmed - lemma1) / abs(lemma1) if lemma1 else float("nan")
        print(f"{n:>5} {t:>6.2f} {lemma1:>16.6e} {ahmed:>16.6e} {rel:>12.2e}")
    print()
    print("  (Ahmed's float reduction loses precision to binomial cancellation")
    print("   as n grows; agreement at small n confirms the two compute the")
    print("   same quantity.)")
    print()


# ===========================================================================
# 2. Ahmed's log-domain incomplete-beta non-central t (Table 1) vs ours
# ===========================================================================

def _logsumexp2(a: float, b: float) -> float:
    """ln(e^a + e^b), numerically safe."""
    hi, lo = (a, b) if a >= b else (b, a)
    if hi == -math.inf:
        return -math.inf
    return hi + math.log1p(math.exp(lo - hi))


def ahmed_nct_cdf_log(t: float, f: float, delta: float, N: int = 4000):
    """log of the non-central t CDF P(f, delta, t) via Ahmed's Table 1.

    P(f, delta, t) = Pr{ (z + delta) / sqrt((1/f) sum_{i=1}^f x_i^2) <= t }
                   = nct.cdf(t; df=f, nc=delta).

    Returns (P, logP) where the log form keeps the *complement-free* series
    sum; P is recovered as 1 - 0.5 e^{-lambda + SUM}.  The series itself is
    accumulated in log domain (Ahmed's contribution); the final subtraction
    is linear, so P saturates near 1e-15 in the deep lower tail — see notes.
    """
    if t <= 0:
        # not needed for our operating points; fall back to scipy
        return stats.nct.cdf(t, df=f, nc=delta), None

    lam = delta * delta / 2.0
    x = f / (f + t * t)                       # = sin^2(theta); eq (12)
    lx = math.log(x)
    l1mx = math.log1p(-x)

    lnG = special.gammaln
    # B(i)  = I_x(f/2, i + 1/2)   (regularised incomplete beta)
    # BB(i) = I_x(f/2, i + 1)
    B = math.log(special.betainc(f / 2.0, 0.5, x))
    BB = math.log(special.betainc(f / 2.0, 1.0, x))
    D = 0.0                                    # ln T_0 = ln 1
    E = math.log(delta * math.sqrt(2.0 / math.pi))
    S = (math.log(2.0) + lnG((f + 1) / 2.0) - lnG(f / 2.0)
         - 0.5 * math.log(math.pi) + (f / 2.0) * lx + 0.5 * l1mx)
    SS = lnG(1 + f / 2.0) - lnG(f / 2.0) + (f / 2.0) * lx + l1mx
    SUM = _logsumexp2(D + B, E + BB)

    llam = math.log(lam) if lam > 0 else -math.inf
    for i in range(1, N + 1):
        B = _logsumexp2(B, S)
        BB = _logsumexp2(BB, SS)
        D = llam + D - math.log(i)
        E = llam + E - math.log(i + 0.5)
        SUM = _logsumexp2(_logsumexp2(SUM, D + B), E + BB)
        S = l1mx + math.log(f + 2 * i - 1) - math.log(1 + 2 * i) + S
        SS = l1mx + math.log(f + 2 * i) - math.log(2 + 2 * i) + SS

    # P = 1 - 0.5 e^{-lambda} e^{SUM}  (eq 16)
    log_complement = -lam + SUM + math.log(0.5)   # ln(1 - P)
    P = -math.expm1(log_complement)               # log-safe 1 - e^{...}
    return P, log_complement


def task2():
    print("=" * 72)
    print("TASK 2  Ahmed log incomplete-beta NCT (Table 1)  vs  our log_nct_cdf")
    print("=" * 72)
    print("  All three evaluate nct.cdf(t; df=f, nc=delta).")
    print()
    hdr = f"{'n':>5} {'snr_dB':>7} {'t':>9} {'scipy cdf':>14} {'Ahmed cdf':>14} {'our exp(logcdf)':>16}"
    print(hdr)
    for n in [50, 200, 500]:
        for snr_db in [0.0, 3.0]:
            f = n - 1
            delta = math.sqrt(n * 10 ** (snr_db / 10))
            # pick a threshold in the body of the distribution
            t = delta * 0.6
            sp = stats.nct.cdf(t, df=f, nc=delta)
            ah, _ = ahmed_nct_cdf_log(t, f, delta)
            ours = math.exp(log_nct_cdf(t, df=f, nc=delta))
            print(f"{n:>5} {snr_db:>7.1f} {t:>9.3f} {sp:>14.6e} {ah:>14.6e} {ours:>16.6e}")
    print()
    print("  Deep lower tail — where scipy starts to struggle:")
    print(f"{'n':>5} {'t':>9} {'scipy cdf':>14} {'Ahmed cdf':>14} {'our log cdf':>16}")
    n = 200
    f = n - 1
    delta = math.sqrt(n * 1.0)  # snr 0 dB
    for frac in [0.5, 0.4, 0.3, 0.2]:
        t = delta * frac
        sp = stats.nct.cdf(t, df=f, nc=delta)
        ah, ahlog = ahmed_nct_cdf_log(t, f, delta)
        ourslog = log_nct_cdf(t, df=f, nc=delta)
        print(f"{n:>5} {t:>9.3f} {sp:>14.6e} {ah:>14.6e} {ourslog:>16.4f} (nat log)")
    print()
    print("  Note: Ahmed accumulates the *series* in log domain (enabling large")
    print("  n/delta), but recovers P = 1 - (.) linearly, so its lower-tail floor")
    print("  is ~1e-12.  Our log_nct_cdf returns log P directly and goes far")
    print("  deeper (that is the one methodological difference).")
    print()


if __name__ == "__main__":
    task1()
    task2()
