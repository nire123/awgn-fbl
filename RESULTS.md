# Results

Numerical results for the AWGN finite-blocklength bounds in `awgn_fbl`.
Every number and figure below is **reproducible** from the library — there is
no hand-tuned data.

```bash
pip install -e .
python generate_chapter_figures.py     # regenerates every figure in plots/chapter/
python analysis/stress_plots.py         # regenerates the stress/ sweeps
pytest tests/ -q                        # 262 tests, ~6 min
```

All rates are in **bits/channel use**; `C = ½·log₂(1 + SNR)` is the Shannon
capacity. Converse is an **upper** bound on the achievable rate, the
achievability bounds (RCU⁺, κβ, Gallager) are **lower** bounds, and the normal
approximation is a (non-rigorous) benchmark that sits between them.

---

## 1. Headline result

At the reference operating point `n = 200`, `SNR = 0 dB`, `ε = 10⁻³`:

| Bound | Rate | Gap to capacity |
|---|---|---|
| Shannon capacity | 0.5000 | — |
| **Shannon cone-packing converse** | **0.3456** | 0.154 |
| **RCU⁺ achievable** | **0.3365** | 0.164 |
| Normal approximation | 0.3261 | 0.174 |
| κβ (Polyanskiy, PPV-faithful) | 0.2837 | 0.216 |
| Gallager | 0.2540 | 0.246 |

The **converse → achievability gap is 0.0092 bits/use** — the two bounds
bracket the maximum coding rate to within ~1% of capacity, about an order of
magnitude closer than the next achievability bound (κβ).

> **The converse row is Shannon's cone-packing bound.**  For an equal-power
> AWGN codebook the maximum-likelihood decision regions are circular cones
> around each codeword; Shannon's 1959 sphere/cone-packing argument gives the
> β-optimal converse, whose error probability is exactly a non-central t tail.
> The formal identity between this bound and the Polyanskiy–Poor–Verdú
> meta-converse (in the minimax/saddle-point sense) is due to Polyanskiy
> (2013).  So the bracket above is the RCU⁺ achievability against the β-optimal
> converse, not a loose one.

### Across operating points

Computed directly from the library (`NoncentralTConverse.converse_rate_log`,
`RCUAchievable.achievable_rate`, `KappaBetaAchievablePPV`,
`GallagerAchievable`, `normal_approx_rate`):

| n | SNR (dB) | ε | C | Converse | RCU⁺ | Normal | κβ (PPV) | Gallager |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 200 | 0 | 10⁻³ | 0.5000 | 0.3456 | 0.3365 | 0.3261 | 0.2837 | 0.2540 |
| 500 | 0 | 10⁻³ | 0.5000 | 0.3949 | 0.3916 | 0.3869 | 0.3688 | 0.3351 |
| 1000 | 0 | 10⁻³ | 0.5000 | 0.4227 | 0.4211 | 0.4186 | 0.4092 | 0.3801 |
| 200 | 3 | 10⁻³ | 0.7913 | 0.6179 | 0.6090 | 0.6003 | 0.5535 | 0.5098 |
| 500 | 3 | 10⁻⁶ | 0.7913 | 0.6100 | 0.6065 | 0.5959 | 0.5653 | 0.5554 |

* **The converse/RCU⁺ gap shrinks with n** — 0.0092 at `n=200`, 0.0033 at
  `n=500`, 0.0016 at `n=1000` (SNR 0 dB, ε = 10⁻³). RCU⁺ tracks the converse
  to `O(1/n)`.
* **RCU⁺ beats the normal approximation everywhere**, as a true bound should,
  while κβ and Gallager are progressively looser — the cost of their
  respective relaxations.

---

## 2. Converse vs RCU⁺ across SNR

![showcase waterfall](plots/chapter/showcase_waterfall_n500.png)

**`showcase_waterfall_n500.png`** — error probability `P_e` vs rate `R` at
`n = 500`, for six SNRs from 0 to 20 dB (one colour each). Dashed = NCT
converse, solid ○ = RCU⁺, dotted = normal approximation; thin vertical lines
mark capacity at each SNR. The zoom inset (SNR = 8 dB) shows the converse and
RCU⁺ curves are visually indistinguishable wherever both are drawn — the
achievability and converse essentially coincide across the whole operating
range. (This figure uses the default ε-grid, so the RCU⁺ curves stop at their
`≈10⁻¹⁰` floor while the converse continues lower; §9 shows a deep-grid RCU⁺
tracking the converse far past that.)

---

## 3. Bound comparisons — rate vs SNR / n / ε

All six standard curves (`capacity`, `converse_nct`, `rcu`, `gallager`,
`kappabeta_ppv`, `normal`) on common axes.

### Rate vs SNR (ε = 10⁻³)

At three blocklengths, sweeping SNR over `[-2, 10] dB`:

| n = 50 | n = 200 | n = 1000 |
|---|---|---|
| ![rate vs snr n50](plots/chapter/rate_vs_snr_n50.png) | ![rate vs snr n200](plots/chapter/rate_vs_snr_n200.png) | ![rate vs snr n1000](plots/chapter/rate_vs_snr_n1000.png) |

As `n` grows the converse and RCU⁺ curves close on each other and on capacity;
at `n = 50` the finite-blocklength penalty is large, at `n = 1000` all bounds
are tightly bunched.

### Rate vs blocklength (ε = 10⁻³)

Convergence to capacity as `n → ∞`, at two SNRs:

| SNR = 0 dB | SNR = 3 dB |
|---|---|
| ![rate vs n snr0](plots/chapter/rate_vs_n_snr0.png) | ![rate vs n snr3](plots/chapter/rate_vs_n_snr3.png) |

### Rate vs target error probability (n = 200)

Sweeping `ε` over `[10⁻⁵, 0.5]`, at two SNRs:

| SNR = 0 dB | SNR = 3 dB |
|---|---|
| ![rate vs eps snr0](plots/chapter/rate_vs_eps_snr0.png) | ![rate vs eps snr3](plots/chapter/rate_vs_eps_snr3.png) |

> **Why the curves rise above the capacity line near ε → 0.5.** This is correct,
> not a plotting error.  Shannon capacity `C` is the `n → ∞`, vanishing-error
> limit; at *finite* `n` with a *large* allowed error probability the maximum
> coding rate genuinely exceeds `C`.  The normal approximation makes it exact:
> `R*(n,ε) ≈ C − √(V/n)·Q⁻¹(ε) + log₂(n)/(2n)`, and since `Q⁻¹(ε) → 0` as
> `ε → ½` the `+log₂(n)/(2n)` term alone lifts the rate to `C + 0.019` at
> `ε = 0.5`, `n = 200`.  The converse, RCU⁺ and normal-approximation curves all
> cross above `C` together, in the correct order (converse ≥ RCU⁺), exactly as
> expected.  The `C` line is the asymptotic reference, not a finite-`n` ceiling.

---

## 4. Waterfall — error probability vs SNR (fixed rate R = 0.3)

The `R → ε` direction: how error probability falls as SNR increases, at fixed
rate, for converse / RCU⁺ / normal approximation.

| n = 200 | n = 500 |
|---|---|
| ![error vs snr n200](plots/chapter/error_vs_snr_n200.png) | ![error vs snr n500](plots/chapter/error_vs_snr_n500.png) |

---

## 5. Converse optimality — cone-packing vs Polyanskiy's relaxed χ²

The NCT converse here **is** Shannon's 1959 cone-packing bound — the optimal
AWGN converse (the PPV meta-converse at its minimax saddle point; Polyanskiy
2013). A commonly-used *relaxed* form instead takes the output measure
`Q_Y = N(0,(1+P)·I)`, which is **not** β-optimal. These figures quantify how
much rate that relaxation gives up relative to the optimal cone-packing
(Lemma 1 / NCT) converse, `R_χ² − R_NCT`.

> The relaxed χ² curve is evaluated with **Erseghe's (2015) Temme method**
> (`ErsegheConverse`), which agrees with scipy `ncx2` to ~10⁻¹² where scipy
> works and stays finite past its NaN wall — so the mismatch curves remain
> complete at large n / high SNR.  Erseghe's paper evaluates exactly this
> *relaxed* bound (it is not the optimal cone-packing one); its claimed
> equivalence to Shannon's bound is the asymptotic/minimax statement above.

### vs SNR (ε = 10⁻³, several blocklengths)

![mismatch gap vs snr](plots/chapter/mismatch_gap_vs_snr.png)

### vs blocklength (log–log, showing the O(1/n) decay)

![mismatch gap vs n](plots/chapter/mismatch_gap_vs_n.png)

**`mismatch_gap_vs_n.png`** — the gap decays as `O(1/n)` (dashed `∝ 1/n`
reference line): both converses converge to capacity, but the NCT converse is
strictly tighter at every finite `n`.

---

## 6. Exact random coding vs the RCU⁺ envelope (small n)

![exact rc vs bounds](plots/chapter/exact_rc_vs_bounds.png)

**`exact_rc_vs_bounds.png`** — at `n = 30, 50, 100` (SNR = 0 dB), the *exact*
Monte-Carlo random-coding error `E[1 − (1 − G(T))^(M−1)]` (black, 500k
samples) against the RCU⁺ min-with-1 envelope `E[min(1, (M−1)·G(T))]` (green)
and the NCT converse (red, a lower bound on `P_e`). The envelope sits just
above the exact curve — quantifying the (small) cost of the union-bound
relaxation that RCU⁺ applies, and serving as an independent Monte-Carlo
cross-check of the RCU⁺ integral form.

---

## 7. Robustness — stress and edge-case sweeps

`analysis/stress_plots.py` regenerates a broader battery under
[`plots/stress/`](plots/stress/) — ~100 figures with companion `.csv` data —
exercising the bounds far outside the comfortable regime:

* **`standard/`** — dense `rate_vs_snr`, `rate_vs_n`, `rate_vs_eps`,
  `error_vs_snr` sweeps across many `(n, SNR, ε)` combinations.
* **`inversions/`** — the inverse directions (`snr_vs_n`, `snr_vs_eps`,
  `error_vs_n`), which exercise the Brent root-finders.
* **`edge/`** — deliberately hostile operating points (below).

Two representative edge cases:

| Extreme SNR, n = 1000 | Large n (n→∞ regime), SNR = +9 dB |
|---|---|
| ![extreme snr](plots/stress/edge/extreme_snr/n1000.png) | ![large n](plots/stress/edge/large_n/snr+9dB.png) |

The `edge/` set also covers `tiny_eps` (ε down to deep tails), `small_n`,
`high_rate_waterfall`, `low_rate_waterfall`, and the `gallager_regime` — these
are where scipy's linear NCT/ncx² break down and the library's log-domain
paths take over. See the directory for the full set plus the raw `.csv`.

---

## 8. Validation

The numbers above are backed by **262 passing tests** (`pytest tests/`,
~6 min). Coverage relevant to result correctness:

* **Implementation cross-validation** — every bound with two implementations is
  checked against its oracle: NCT log-domain vs linear (~10⁻⁶ bits/use),
  `ErsegheConverse` vs scipy ncx², RCU⁺ log vs linear and vs Monte-Carlo union,
  κβ simple vs PPV-faithful and `_log_ncx2_cdf_series` vs scipy ncx², Gallager
  log vs linear.  See the inventory table in the README.
* **Published reference points** — Gallager `n=3000, ε=10⁻⁶ → log M = 1225`;
  κβ_PPV β-formula matches Polyanskiy's `betaq_up_v2.m`.
* **Round-trip identities** — `achievable_error(achievable_rate(ε)) ≈ ε`;
  the `log F` interpolator is exact at its grid nodes.
* **Monotonicity** of every bound in `n`, `ε`, `SNR`, `R`.
* **Tail / range robustness** — the log-domain forms stay finite where the
  linear/scipy ones NaN or underflow: κβ and the χ² converse valid to n≳10⁴ and
  ε≲10⁻³⁰, Gallager finite arbitrarily deep, `log_nct_cdf` cross-checked against
  `scipy.stats.t` at `nc = 0`, solid-angle vs NCT below 10⁻¹⁰ at small n.

---

## 9. Numerical reach — the fully log-domain pipeline

**Every bound in the library is now evaluated in the log domain.**  The
cone-packing converse and RCU⁺ are shown in the reach figure below; the
reference bounds (Erseghe χ², κβ, Gallager) were made log-domain in the same
way (see the closing note), so nothing in the library silently NaNs or
underflows in the regimes that matter.  The figure makes the converse / RCU⁺
reach explicit.

![extended reach](plots/chapter/extended_reach.png)

**`extended_reach.png`** —

* **Left (reach in n).** The cone-packing converse, evaluated in log-domain
  (`converse_rate_log` → `log_nct_cdf`, the integral form
  `E_X[Φ(x·√(X/df) − nc)]`), holds smoothly out to **n = 5000**.  The *same
  bound* via scipy's linear `nct.ppf`, and the χ² relaxation via `ncx2.ppf`,
  both hit a **NaN wall** at far smaller n (vertical markers) — scipy's
  routines underflow internally.
* **Right (depth in ε).** A deep waterfall at n = 500, SNR = 6 dB.  The
  log-domain converse and a **deep-grid RCU⁺** (`eps_min = 10⁻¹⁰⁰`) track each
  other down to `P_e ≈ 10⁻⁴⁵`, while the **default-grid RCU⁺** flattens at its
  ε-floor (grey band).

The two bounds differ slightly in *how* the log domain buys reach:

| Bound | Log-domain mechanism | Reach |
|---|---|---|
| **Converse** (cone-packing) | `log_nct_cdf` integral representation on a log-uniform χ²-grid + log-domain Lemma 1 | Essentially unlimited — verified to `n = 5000`, `ε` far below `10⁻³⁰⁰`, out of the box. |
| **RCU⁺** (achievable) | the `F·J` factorisation `log P = log F(R) + log J(R)`, two well-conditioned terms | Set by the depth of the converse curve `F(R)` it integrates: the default grid floors at `ε = 10⁻¹⁰`; pass `RCUAchievable(..., eps_min=1e-100)` to reach the deep tail (the converse feeding it is accurate that far). |

* **scipy's `nct.logcdf` / `ncx2.logcdf` are not log-stable** — they are
  computed as `log(cdf(x))` and inherit the linear underflow.  The library's
  `log_nct_cdf` stays in the log domain throughout, which is the whole reason
  the converse reaches where it does.
* The **F(R) interpolator** carries ~3.6 ppm error in `log ε`; a 5× grid
  refinement moves the final RCU⁺ rate by ~10⁻¹⁰ bits/use.

**The reference bounds, too.**  The same complementary / log-domain idea was
applied to the rest of the library: the **χ² converse** via Erseghe's Temme
method (`ErsegheConverse`), **κβ** via upper-tail quantiles (`isf`/`sf`,
`erfinv`) and a saddle-point Poisson-mixture `_log_ncx2_cdf_series` for its
non-central χ² tail, and **Gallager** via `log_achievable_error = log μ − n·E_r`.
κβ and the χ² converse are now finite and valid to n ≳ 10⁴, SNR 20, ε ≲ 10⁻³⁰
(previously NaN past n ≈ 5000); Gallager is finite arbitrarily deep.
