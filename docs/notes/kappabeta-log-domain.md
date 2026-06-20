# Making the κβ bound log-domain

How `KappaBetaAchievable` / `KappaBetaAchievablePPV` were made numerically
robust, and how the same idea unifies every bound in the library.

GitHub renders the `$…$` math below.

---

## 1. The bound, and where it breaks

Polyanskiy's κβ achievability bound for the AWGN power-shell ensemble is

$$\log M^\*(n,\varepsilon) \;\ge\; \max_{0<\tau<\varepsilon}\;\Big[\,\log_2 \kappa(\tau)\;-\;\log_2 \beta_q(\tau)\,\Big],
\qquad q = 1-\varepsilon+\tau .$$

`achievable_rate(ε)` sweeps $\tau$ and returns $\max(\cdot)/n$.  Two ingredients:

* $\kappa(\tau)$ — an auxiliary quantity; we use Polyanskiy's asymptotic
  `kappa_inf`,
  $$\kappa(\tau)=2\,\Phi\!\Big(\sqrt{V_P/V_Q}\,x_0\Big)-1,\qquad
    x_0=\Phi^{-1}\!\big(\tfrac{1+\tau}{2}\big),\quad
    V_P=2(1{+}2P),\ V_Q=2(1{+}P)^2 .$$
* $\beta_q$ — the Neyman–Pearson type-II error of the binary test between the
  shell output and the auxiliary output $Q_Y$, obtained through a change of
  measure built on **non-central $\chi^2$** quantiles and CDFs.

Each of those three sub-computations carried a latent numerical failure that
made the bound return `NaN` (or, worse, a rate **above capacity**) at large
$n$, high SNR, or small $\varepsilon$:

| # | sub-computation | failure | trigger |
|---|---|---|---|
| 1 | quantile `pp0 = ncx2.ppf(q)` | $q=1-\varepsilon+\tau \to 1.0$ in float64, `ppf(1.0)=∞` | $\varepsilon \lesssim 10^{-16}$ |
| 2 | $\kappa$ via `norm.ppf((τ+1)/2)`, `2·cdf−1` | $(1{+}\tau)/2\to 0.5$, so $x_0=0$, $\kappa=0$, every $\tau$ skipped | $\tau \lesssim 10^{-16}$ |
| 3 | `term1 = ncx2.cdf(qq0)` | deep lower tail: `cdf=0`, and even `logcdf=−∞` | large $n$ / high SNR |

The fix for all three is the **same principle**.

---

## 2. The principle: never form $1-(\text{tiny})$; stay in the tail / in logs

Failures 1 and 2 are *catastrophic cancellation by rounding*: a probability of
interest is a tiny number $p$, but the code forms $1-p$ (or $\tfrac{1+\tau}{2}$),
which rounds to exactly $1$ (or $\tfrac12$) once $p<\varepsilon_{\text{mach}}\approx
2\times10^{-16}$, destroying $p$ before it is ever used.

The cure is to carry the **complementary / small** quantity directly and use
the library functions that accept it without forming the difference.

### 2.1 The β quantile — `isf`/`sf` instead of `ppf(1−·)`

We need `pp0` with $F(\texttt{pp0})=q$, i.e. the *upper-tail* probability
$\operatorname{SF}(\texttt{pp0}) = 1-q = \varepsilon-\tau =: p_{\uparrow}$.
Instead of `ncx2.ppf(q)` we invert the survival function directly,

```python
pp0 = stats.ncx2.isf(p_up, df=n, nc=n/A**2)          # SF(pp0) = p_up, no 1−ε
```

and the PPV "overshoot" Newton refinement is rewritten in the tail too,

```python
while stats.ncx2.sf(pp0, df=n, nc=nc_p) > p_up:       # push pp0 up
    pp0 += (stats.ncx2.sf(pp0, df=n, nc=nc_p) - p_up) / stats.ncx2.pdf(pp0, df=n, nc=nc_p)
```

`p_up = ε − τ` is passed down from the driver, so $1-\varepsilon+\tau$ is never
formed.  `ncx2.isf(10⁻²⁰)` returns a finite quantile where `ncx2.ppf(1−10⁻²⁰)=
ncx2.ppf(1.0)=∞`.

### 2.2 κ — `erfinv`/`erf` instead of `ppf((τ+1)/2)` and `2·cdf−1`

Using $\Phi(x)=\tfrac12\big(1+\operatorname{erf}(x/\sqrt2)\big)$ twice,

$$x_0=\Phi^{-1}\!\big(\tfrac{1+\tau}{2}\big)=\sqrt2\,\operatorname{erf}^{-1}(\tau),
\qquad
\kappa(\tau)=\operatorname{erf}\!\Big(\sqrt{V_P/V_Q}\;x_0/\sqrt2\Big).$$

```python
x0 = np.sqrt(2.0) * special.erfinv(tau)                # = norm.ppf((τ+1)/2)
kappa = special.erf(np.sqrt(VP/VQ) * x0 / np.sqrt(2))  # = 2Φ(·) − 1
```

`erfinv(τ)` and `erf(·)` take the *small* argument $\tau$ directly — no
$\tfrac{1+\tau}{2}\to\tfrac12$ rounding — so $\kappa>0$ for arbitrarily small
$\tau$.

---

## 3. The deep-tail term: a log-domain non-central $\chi^2$ CDF

Failure 3 is genuine *underflow*, not rounding: `term1` is a non-central
$\chi^2$ **lower-tail** CDF that is legitimately $\ll 10^{-308}$, so `ncx2.cdf`
returns $0$ and scipy's `ncx2.logcdf` (computed as $\log(\texttt{cdf})$) returns
$-\infty$.  We need it in log domain.  `_log_ncx2_cdf_series` does this.

### 3.1 Poisson mixture over central $\chi^2$ lower tails

A non-central $\chi^2_k(\lambda)$ is a Poisson$(\lambda/2)$ mixture of central
$\chi^2_{k+2j}$, so

$$F_{\chi^2}(x;k,\lambda)=\sum_{j\ge0} e^{-\lambda/2}\frac{(\lambda/2)^j}{j!}\;
   P\!\big(\tfrac{k}{2}+j,\ \tfrac{x}{2}\big),$$

where $P(a,z)=\gamma(a,z)/\Gamma(a)$ is the regularised lower incomplete gamma.
In log domain,

$$\log F = \operatorname*{logsumexp}_{j}\Big[\underbrace{-\tfrac\lambda2 + j\log\tfrac\lambda2 - \log j!}_{\log\text{Poisson}(j)} \;+\; \log P\!\big(\tfrac k2+j,\tfrac x2\big)\Big].$$

### 3.2 A log-stable lower incomplete gamma

For each component we need $\log P(a,z)$ accurate when $P$ underflows.  The
always-convergent series

$$P(a,z)=e^{-z+a\log z-\log\Gamma(a)}\;\sum_{m\ge0}\frac{z^{m}}{a(a+1)\cdots(a+m)}$$

keeps the magnitude in the exponent; the sum is $O(1)$ and converges
geometrically for $z<a$ (the lower-tail regime).  So `_log_reg_lower_gamma`
returns a finite value where `gammainc` is $0$.

### 3.3 The subtlety that bit: the mixture peaks at a **saddle**, not the mean

The summand $f(j)=\log\text{Poisson}(j)+\log P(\tfrac k2+j,\tfrac x2)$ is
unimodal, but **its peak is not at the Poisson mean $\lambda/2$.**  Setting
$\partial_j f=0$ with $\partial_j\log\text{Poisson}\approx\log(\tfrac{\lambda/2}{j})$
and $\partial_j\log P\approx\log(\tfrac{z}{k/2+j})$ gives the saddle

$$j_\*\big(\tfrac k2+j_\*\big)\approx \tfrac\lambda2\cdot\tfrac x2 .$$

Deep in the lower tail ($x\ll k+\lambda$) this is **far below** $\lambda/2$: the
mixture is dominated by the *smallest*-order components, where $x$ is least
extreme.  A first implementation that summed a window around $\lambda/2$ missed
the peak entirely, undershot `term1` by tens of thousands of nats, and so
produced $\log_2\beta_q$ too negative and a **rate above capacity**.  The fix is
to seed at $j_\*$, hill-climb to the true peak, and sum outward until the terms
fall a fixed number of nats below it:

```python
j_star = int(round(0.5*(-a0 + np.sqrt(a0*a0 + 4*mu*z))))   # a0=k/2, mu=λ/2, z=x/2
# hill-climb to the unimodal peak, then logsumexp a window around it
```

Validated to $\sim10^{-13}$ against `scipy.stats.ncx2` wherever scipy resolves
the value, and finite/sane far beyond (e.g. $F_{\chi^2}(146;8000,8127)\approx
e^{-16082}$, where scipy returns $0$).

---

## 4. Connection to the other bounds

The κβ fix is not a one-off — it is the same move used everywhere in the
library.  **Work in the complementary or logarithmic representation so the
quantity of interest is never the small remainder of an $O(1)$ subtraction.**

| Bound | Same idea, applied as |
|---|---|
| **Cone-packing converse** (`converse_rate_log`) | never calls `nct.ppf`; instead Brent-inverts `log_nct_cdf` (an integral representation $\mathbb E_X[\Phi(x\sqrt{X/\nu}-\mu)]$ kept in logs), and maps $t\to R$ through the log-domain Lemma 1 |
| **Relaxed χ² converse** (`ErsegheConverse`) | non-central $\chi^2$ tails via Temme's expansion, $\log P=\log g-\tfrac n2 v$ — see [alternative-evaluations.md](alternative-evaluations.md) |
| **RCU⁺** (`log_achievable_error`) | Elkayam factorisation $\log P(R)=\log F(R)+\log J(R)$, a sum of two well-conditioned terms instead of one underflowing integral |
| **Gallager** (`log_achievable_error`) | the bound is an exponential, so $\log P_e=\log\mu-nE_r$ directly; the prefactor $\mu$ is a central-$\chi^2$ difference near the mean |
| **κβ** (this note) | upper-tail quantiles (`isf`/`sf`, `erfinv`/`erf`) + the saddle-point Poisson-mixture `_log_ncx2_cdf_series` |

The shared non-central-$\chi^2$ primitive `_log_ncx2_cdf_series` is also the
natural backend for a future fully-log `ChiSquaredConverse`.

---

## 5. Outcome

* κβ is **finite and below capacity** at $n$ up to $2\times10^4$, SNR $20$ dB,
  $\varepsilon$ down to $10^{-50}$ — previously `NaN` beyond $n\approx5000$.
* Where scipy's `ncx2` is well-conditioned the new code reproduces it exactly,
  and the Polyanskiy reference points are unchanged.
* Covered by `tests/test_kappabeta_logdomain.py` (series vs scipy; PPV vs
  simple; finite-and-$<C$ at the extreme points; monotonicity).
