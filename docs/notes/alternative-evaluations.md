# Alternative evaluations of the same bounds

The cone-packing converse and the relaxed-$Q_Y$ converse each admit several
independent numerical evaluations.  Beyond the library's own log-domain
methods, two come from the literature and are implemented here as
cross-checks: **Ahmed–Ambroze–Tomlinson (2007)** for the geometric
(solid-angle) quantity, and **Erseghe (2015)** for the relaxed $\chi^2$
converse.  This note explains what each computes, how it relates to our
default method, and where it breaks.

GitHub renders the `$…$` math below.

---

## 1. Ahmed's trigonometric reduction vs our geometric Lemma 1

### 1.1 The shared quantity

The fundamental geometric object is the probability that a spherically
symmetric vector has empirical correlation $\hat\rho$ with a fixed direction
exceeding a threshold $t$.  It appears two ways:

* **Our Lemma 1** — a half-period trigonometric integral,
  $$P(\hat\rho\ge t)=\frac1\pi\,(1-t^2)^{\frac{n-1}{2}}
    \int_0^{\pi/2}\big(1+t^2\tan^2\theta\big)^{\frac{1-n}{2}}\,d\theta,$$
  evaluated in log domain (`log_pairwise_error_prob`: a `logsumexp` over a
  concentrated $\theta$-grid, with the $(1-t^2)^{(n-1)/2}$ prefactor kept as a
  log-linear term).

* **Shannon's solid angle** — the same probability is the fraction of the
  sphere inside the cap of half-angle $\theta=\arccos t$,
  $$\frac{\Omega_n(\theta)}{\Omega_n(\pi)}
    =\frac{\int_0^{\theta}\sin^{\,n-2}\!\varphi\,d\varphi}
           {\int_0^{\pi}\sin^{\,n-2}\!\varphi\,d\varphi}.$$

They are equal with $t=\cos\theta$.  Our `SolidAngleConverse` evaluates the
ratio by quadrature; Ahmed evaluate the integral in **closed form**.

### 1.2 Ahmed's closed form

Ahmed use the standard trigonometric reduction formulas (Gradshteyn–Ryzhik) for
$\int\sin^m\!\varphi\,d\varphi$ as a finite sum.  For $m=2p$ (even),

$$\int_0^{\theta}\!\sin^{2p}\!\varphi\,d\varphi
  =\frac{1}{2^{2p}}\binom{2p}{p}\theta
   +\frac{(-1)^p}{2^{2p-1}}\sum_{k=0}^{p-1}(-1)^k\binom{2p}{k}
     \frac{\sin\big((2p-2k)\theta\big)}{2p-2k},$$

with an analogous cosine sum for odd $m$ (`analysis/verify_ahmed.py`,
`_int_sin_pow`).  So Ahmed **replace the integral entirely with a finite
trigonometric sum** — no quadrature, no log-domain machinery for this piece.

### 1.3 Equivalence, and why we keep the integral

The cross-check `analysis/verify_ahmed.py` (task 1) confirms the two agree to
machine precision at small $n$:

| $n$ | Lemma 1 $P(t{=}0.5)$ | Ahmed closed form | rel. err |
|---:|---|---|---:|
| 10  | $5.865\times10^{-2}$ | $5.865\times10^{-2}$ | $1\times10^{-15}$ |
| 80  | $1.004\times10^{-6}$ | $1.004\times10^{-6}$ | $2\times10^{-10}$ |
| 200 | $2.062\times10^{-14}$ | $1.872\times10^{-14}$ | $9\times10^{-2}$ |
| 400 | $4.707\times10^{-27}$ | $-7\times10^{-14}$ | — (garbage) |

The closed form carries binomials $\binom{2p}{k}\sim 2^{m}$ with **alternating
signs**; in float64 their partial sums cancel catastrophically, so the result
loses all precision by $n\approx200$ and goes negative by $n\approx400$.  Our
log-domain Lemma 1 has no such cancellation and stays exact to arbitrary $n$.

This matters for Ahmed's method, and is worth stating precisely because it is
easy to get wrong.  Ahmed's pipeline has **two halves**:

* **the geometry** — finding the cone angle $\theta_1$ from the rate, via the
  trig reduction (eqs 6–7).  The paper is explicit: *"These expressions can be
  used to evaluate the angle $\theta_1$ given an $(n,k)$ code"* (§II.B).  There
  is **no log-domain treatment of this step** in the paper.
* **the error probability** $Q(\theta_1)$ — the non-central $t$ CDF, evaluated
  by the log-domain incomplete-beta recursion of §1.4 below (their Table 1,
  titled *"Logarithmic Version of Algorithm to Compute (9)"*, and eq 9 is
  $Q(\theta)$).

So the robustness Ahmed advertise — "exact for $k$ of thousands", demonstrated
on DVB-S2 with $n\approx65{,}000$ — is carried entirely by the **incomplete-beta
CDF** (the second half).  The **geometric step is the trig sum**, a *signed*
closed form with no log treatment, and it is exactly the part that cancels at
large $n$: to make the integral equal a tiny $2^{-k}$ target, its leading
$O(\theta_1)$ term must cancel down to $O(\theta_1^{\,n-1})$, losing
$\sim (n-1)\log(1/\theta_1)$ digits.  The paper does not analyse this, and as
written eqs 6–7 would not survive $n\approx65{,}000$ — so their implementation
must do something the paper does not state (extended precision, or computing
$\theta_1$ through the $\delta=0$ incomplete beta instead, since the
solid-angle fraction is the central-$t$ CDF).

**Conclusion:** the log-domain robustness Ahmed claim applies to the *CDF* half;
the *geometric* half, as described, is the trig sum, and our cross-check
confirms it is the unstable piece.  The integral form (our log-domain Lemma 1,
or equivalently the central incomplete beta) is the stable way to evaluate the
same geometric quantity.

### 1.4 Ahmed's log-domain non-central $t$ CDF

Ahmed also give a log-domain evaluation of the non-central $t$ CDF (their
Table 1) — the *other* half of the converse pipeline.  It expands the CDF as a
series in the **regularised incomplete beta** function and accumulates all four
recursions (`logsumexp`, $\log\Gamma$, $\log I_x$) in log domain, with a Poisson
tail bounding the truncation.  This is a genuine alternative to our
`log_nct_cdf` (which instead uses the integral representation
$\mathbb E_{X\sim\chi^2_\nu}[\Phi(x\sqrt{X/\nu}-\mu)]$ on a log-uniform grid).

The cross-check (task 2) shows they agree to $\sim6$ digits in the body of the
distribution.  One difference: Ahmed recover the probability as
$P=1-\tfrac12 e^{-\lambda}\,\Sigma$, a linear subtraction that floors the
lower-tail accuracy at $\sim10^{-12}$, whereas `log_nct_cdf` returns $\log P$
directly and reaches $e^{-700}$ and beyond.  This is below Ahmed's own operating
range — their applications are real codes at codeword-error rates $\sim10^{-6}$,
so the floor never arises there; it matters only for evaluating the *bound
itself* deep in the tail (e.g. $\varepsilon\sim10^{-50}$), which is this
library's use case rather than theirs.

**References:** M. Z. Ahmed, M. A. Ambroze, M. Tomlinson, *On computing
Shannon's sphere packing bound and applications*, ISCTA 2007.

---

## 2. Erseghe's Temme evaluation of the relaxed converse

### 2.1 What it computes

The Polyanskiy–Poor–Verdú meta-converse with the **capacity-achieving output**
$Q_Y=\mathcal N(0,(1+P)I)$ — the library's `ChiSquaredConverse`, *not* the
optimal cone-packing bound — reduces (Erseghe, Thm 1) to false-alarm / missed-
detection probabilities that are **non-central $\chi^2$ tails**:

$$P_{\mathrm{MD}}=\overline F_{\chi}\!\big(n\lambda';\,n,\tfrac n\Omega\big),
\qquad
P_{\mathrm{FA}}= F_{\chi}\!\Big(\tfrac{n\lambda'}{1+\Omega};\,n,\,n\tfrac{1+\Omega}\Omega\Big),$$

and the converse rate is $R=-\tfrac1n\log_2 P_{\mathrm{FA}}$ with $\lambda$ set
by $P_{\mathrm{MD}}=\varepsilon$.  Evaluated with `scipy.stats.ncx2` these hit a
`NaN` wall at large $n$ / high SNR (the same wall that motivates the whole
log-domain effort).

### 2.2 Temme's uniform expansion

Erseghe applies Temme's method for the non-central $\chi^2$ / Marcum-$Q$
function.  Writing the tail as

$$P = \mathbf 1(\cdot) \;\pm\; g(\gamma)\,e^{-\frac n2 v(\gamma)},$$

the exponential rate $v(\gamma)=-\alpha(\theta)/s_\gamma$ and the threshold map
are elementary functions of a single variable $\gamma$ (with $s_\gamma=\sinh\gamma$,
$\theta=\tfrac12\log(y/x)$), and the prefactor $g(\gamma)$ is an $O(1)$ **single
integral** (his Theorem 4),

$$g(\gamma)=\frac1\pi\int_0^{\pi}\tilde g(\varphi)\,
            e^{-\,n\,u^2(\varphi)/(4 s_\gamma)}\,d\varphi,$$

in which **$n$ appears only in the exponent** — so the integrand is
well-conditioned and the whole thing is evaluated in log domain as
$\log P=\log|g|-\tfrac n2 v$.  `ErsegheConverse` implements this exactly
(`method="integral"`), plus the cheaper two-term asymptotic expansion
(`method="asymptotic"`, his eqs 34–36).

### 2.3 Validation and a root-finding subtlety

The integral form matches `scipy.stats.ncx2` to $\sim10^{-12}$ wherever scipy
resolves the value, and stays finite past its `NaN` wall (e.g. it returns a
finite converse rate at $n=1000$, SNR $6$ dB where scipy gives `NaN`).
`tests/test_erseghe.py` covers this.

One numerical care point: the leading Temme expansion is accurate in the tail
but **turns over** as the probability approaches $O(1)$ near the $P_e=\tfrac12$
boundary, so $\log P_{\mathrm{MD}}(\gamma)$ is *not monotone* on the whole
$\gamma$ interval.  The solver locates the peak (`minimize_scalar`) and
root-finds only on the monotone rising branch below it — otherwise a naïve
bracket spanning the turnover fails the sign test and returns `NaN`.

### 2.4 Theoretical placement — what Erseghe is and isn't

Erseghe evaluates the **relaxed** bound (the $\mathcal N(0,(1+P)I)$ choice),
which is strictly looser than Shannon's cone-packing converse at finite $n$ —
this is exactly the $O(1/n)$ *relaxation cost* the chapter's mismatch figures
quantify.  Erseghe's stated "as accurate as Shannon's 1959 cone-packing bound"
is an asymptotic / minimax statement; the **formal identity** between the PPV
meta-converse and the cone-packing bound (in the saddle-point sense) is due to
Polyanskiy, *Saddle point in the minimax converse for channel coding*, IEEE
TIT 2013 — not to the relaxed evaluation itself.  In the library:

* `NoncentralTConverse` — the **optimal** cone-packing converse (tighter);
* `ErsegheConverse`, `ChiSquaredConverse` — the **relaxed** $Q_Y$ converse
  (looser), the latter via scipy and the former robustly via Temme.

**References:** T. Erseghe, *On the evaluation of the Polyanskiy–Poor–Verdú
converse bound for finite block-length coding in AWGN*, IEEE TIT 61(12), 2015
(arXiv:1401.7169); N. Temme, *Asymptotic and numerical aspects of the
non-central chi-square distribution*, 1993; Y. Polyanskiy, *Saddle point in the
minimax converse for channel coding*, IEEE TIT 2013.

---

## 3. Summary — one quantity, several methods

| Quantity | Default (library) | Cross-check / alternative | Verdict |
|---|---|---|---|
| Solid-angle / pairwise error | log-domain Lemma 1 | Ahmed closed-form trig sum | agree to machine ε at small $n$; the log-domain form stays stable at large $n$ |
| Non-central $t$ CDF | `log_nct_cdf` (integral rep) | Ahmed incomplete-beta recursion | agree in body; the integral form reaches the deep tail |
| Relaxed $\chi^2$ converse | `ErsegheConverse` (Temme) | `ChiSquaredConverse` (scipy ncx²) | agree to $10^{-12}$; Temme survives the NaN wall |

See also [kappabeta-log-domain.md](kappabeta-log-domain.md) for the
complementary-probability / saddle-point machinery shared by these evaluations.
