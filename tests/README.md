# Test suite

`pytest tests/ -q` — **261 cases** (133 test functions, expanded by
parametrisation), ~6 minutes.

The suite is built around one idea: most quantities have **two or more
independent implementations** (a simple/scipy oracle and a robust log-domain
default — see the *Bounds inventory* table in the top-level
[README](../README.md)), and the tests check them against each other, against
published reference points, and for the structural properties a bound must
have (monotonicity, ordering, round-trip inverses, tail finiteness).

Five recurring kinds of check:

1. **Implementation cross-validation** — log-domain vs linear/scipy, exact vs
   Monte-Carlo, simple vs faithful.
2. **Published reference points** — Gallager `n=3000, ε=10⁻⁶ → log M = 1225`;
   κβ matches Polyanskiy's `betaq_up_v2.m`.
3. **Round-trip identities** — `error(rate(ε)) ≈ ε`, grid round-trips.
4. **Monotonicity & ordering** — every bound monotone in `n, ε, SNR, R`, and
   `achievable ≤ true ≤ converse ≤ capacity` (with the finite-`n` caveat near
   `ε→½`).
5. **Tail / range robustness** — the log-domain forms stay finite where scipy
   NaNs or underflows.

---

## Converse

| File | cases | Covers |
|---|---:|---|
| `test_converse.py` | 23 | Shannon capacity values; NCT and χ² rate at known points; **NCT vs χ² agree** across `n`/SNR; bidirectional round-trip; monotonicity; input validation; convenience wrappers; **solid-angle vs NCT** at small `n` and the large-`n` warning. |
| `test_converse_log_domain.py` | 10 | `converse_rate_log` / `converse_error_log` **vs the linear path** in the shared regime; log path finite where linear scipy NCT NaNs; round-trip; monotonicity in `ε`/`n`; deep-tail error; RCU⁺ extended range; converse–achievability gap shrinks with `n`. |
| `test_erseghe.py` | 8 | `ErsegheConverse` **integral vs scipy `ncx2`** (~10⁻¹²); finite past scipy's NaN wall; strictly **looser than NCT** (the relaxation); round-trip; two-term asymptotic vs integral; monotonicity; validation. |
| `test_log_domain.py` | 7 | log-domain **Lemma 1 vs linear** + tail extension; `log_nct_cdf` vs scipy near the mode; **central-`t` cross-check** (`nc=0`); NCT tail finite where scipy returns NaN; SF/CDF complement; monotonicity. |
| `test_ahmed.py` | 22 | **Ahmed (2007) vs the library**: the closed-form trigonometric reduction == Lemma 1 at small `n` (and its large-`n` cancellation breakdown); Ahmed's incomplete-beta log-NCT == scipy and `log_nct_cdf` in the body. |

## Achievability

| File | cases | Covers |
|---|---:|---|
| `test_rcu_achievable.py` | 7 | RCU⁺ known values; **below the converse**; small gap; error/​rate monotonicity; round-trip. |
| `test_rcu_log_domain.py` | 5 | log-safe path **vs the linear path**; deep-tail value finite; rate inversion agreement; **`F·J` factorisation invariants** `J ∈ [1, 1/F]`. |
| `test_rcu_verification.py` | 13 | independent reimplementation of the `t`-parameterisation: monotonicity of `R(t)`, `ε(t)`; F-grid round-trip; parameterisation equivalence; **F interpolator fidelity** (exact at grid nodes, off-grid vs direct converse, grid-density invariance); integral structure; rate↔error inverses. |
| `test_exact_random_coding.py` | 6 | the ordering **exact ≤ union ≤ RCU⁺ envelope**, and exact ≥ converse; Monte-Carlo convergence with sample count; trivial cases; vectorised vs scalar log-`G`. |
| `test_kappabeta.py` | 6 | κβ (simple) known value; below converse; monotonicity; validation; below capacity. |
| `test_kappabeta_v2.py` | 8 | κβ (PPV-faithful) basics and sanity; `kappa_inf` matches the v1 formula; **v1 vs v2 agree** at moderate params. |
| `test_kappabeta_logdomain.py` | 8 | `_log_ncx2_cdf_series` **vs scipy `ncx2`** + deep-tail finiteness + central-χ² reduction; κβ **finite and `< C`** at the extreme points that used to NaN; large-`n`/moderate-SNR `< C`; reference point unchanged; **PPV vs simple agree**; deep-`ε` monotonicity. |
| `test_gallager.py` | 12 | constructor/validation; `R_cr` value; **Polyanskiy reference** `log M = 1225`; sanity (below converse/capacity, monotone); error-direction round-trip and monotonicity. |
| `test_gallager_logdomain.py` | 6 | `log_achievable_error` **vs the linear form**; finite arbitrarily deep; clamped at `log 1`; deep-`ε` monotonicity; `< C` deep; reference `1225`. |

## Approximation & evaluator

| File | cases | Covers |
|---|---:|---|
| `test_normal_approximation.py` | 10 | channel dispersion `V`; rate (below capacity, → capacity, monotone, value, **between achievable and converse**); error round-trip and monotonicity. |
| `test_fast_F_evaluator.py` | 11 | `FastFREvaluator` grid properties; accuracy **vs the direct converse**; vectorised/scalar calls; monotonicity; convenience method; χ² (Erseghe-fed) variant. |

---

## Notes

* The Ahmed–Ambroze–Tomlinson (2007) reference reimplementations live in
  [`analysis/verify_ahmed.py`](../analysis/verify_ahmed.py); `test_ahmed.py`
  imports them and asserts agreement with the library, and the script itself
  (`python analysis/verify_ahmed.py`) prints the full table that *shows* where
  the closed form loses precision at large `n`.
* The theory behind those alternative evaluations is written up in
  [`docs/notes/alternative-evaluations.md`](../docs/notes/alternative-evaluations.md)
  (Ahmed) and [`docs/notes/kappabeta-log-domain.md`](../docs/notes/kappabeta-log-domain.md)
  (the κβ log-domain machinery).
* `tests/__init__.py` is empty; tests import the installed package
  (`pip install -e .`).
