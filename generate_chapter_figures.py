#!/usr/bin/env python
"""
Top-level reproducibility script for the thesis AWGN chapter.

Generates every figure referenced by ``docs/chapter_awgn.tex`` into the
``plots/chapter/`` directory.  Run from the repository root::

    python generate_chapter_figures.py

Group structure (in the order the figures appear in the chapter):

    1. Mismatch plots    — Polyanskiy χ² vs NCT converse relaxation cost
    2. Showcase waterfall — exact/RCU+/converse/normal on one figure
    3. Rate vs SNR / n / ε — standard bound comparisons
    4. Error vs SNR      — waterfall representation
    5. Exact RC vs RCU+  — small-n envelope-cost figure

Expensive steps (Monte Carlo, F-grid precompute) are cached per operating
point within a single figure's context, but the script as a whole runs
end-to-end in a few minutes.
"""

from __future__ import annotations

import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from awgn_fbl import (
    ChiSquaredConverse,
    ErsegheConverse,
    ExactRandomCoding,
    GallagerAchievable,
    KappaBetaAchievablePPV,
    NoncentralTConverse,
    RCUAchievable,
    normal_approx_error,
    normal_approx_rate,
    plot,
)


warnings.filterwarnings("ignore")
OUT_DIR = "plots/chapter"


# ===========================================================================
# Helpers
# ===========================================================================

def _ensure_out_dir() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)


def _save(fig, name: str) -> None:
    path = f"{OUT_DIR}/{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path}")


def _safe_positive(arr: np.ndarray) -> np.ndarray:
    out = np.asarray(arr, dtype=float).copy()
    out[out <= 0] = np.nan
    return out


def _polyanskiy_chi2_rate(n: int, snr_db: float, eps: float) -> float:
    """Polyanskiy's χ² convention: `log M` at dim `n+1`, rate divided by `n`.

    Evaluated via Erseghe's robust Temme method (exact match to scipy ncx2
    where it works, finite past its NaN wall) so the mismatch curves stay
    complete at large n / high SNR.
    """
    r = ErsegheConverse(n=n + 1, snr_db=snr_db).converse_rate(eps)
    return r * (n + 1) / n


# ===========================================================================
# 1. Mismatch between the NCT converse and Polyanskiy's χ² relaxation
# ===========================================================================

def fig_mismatch_gap_vs_snr():
    """Gap `R_χ² − R_NCT` vs SNR, at ε = 10⁻³, several blocklengths."""
    fig, ax = plt.subplots(figsize=(10, 6))

    snrs = np.linspace(-3, 10, 20)
    ns = [50, 100, 200, 500, 1000]
    cmap = plt.cm.viridis(np.linspace(0.1, 0.9, len(ns)))

    for n, color in zip(ns, cmap):
        gaps = []
        for s in snrs:
            nct = NoncentralTConverse(n=n, snr_db=s).converse_rate_log(1e-3)
            chi2 = _polyanskiy_chi2_rate(n, s, 1e-3)
            gaps.append(chi2 - nct if np.isfinite(chi2) and np.isfinite(nct) else np.nan)
        ax.plot(snrs, gaps, color=color, linewidth=2, marker="o",
                markersize=5, label=f"n = {n}")

    ax.set_xlabel("SNR (dB)", fontsize=13)
    ax.set_ylabel(
        r"$R_{\chi^2}^{\mathrm{Polyanskiy}} - R_{\mathrm{NCT}}^{\mathrm{ours}}$"
        "  (bits/channel use)",
        fontsize=12,
    )
    ax.set_title(
        r"Relaxation cost of Polyanskiy's choice $Q_Y = \mathcal{N}(0,(1{+}P)I)$"
        "\n"
        r"vs.\ optimal sphere-packing (Lemma 1 / NCT) at $\varepsilon = 10^{-3}$",
        fontsize=12,
    )
    ax.axhline(0, color="k", linewidth=0.5, alpha=0.5)
    ax.legend(fontsize=11, title="Blocklength", loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, "mismatch_gap_vs_snr")


def fig_mismatch_gap_vs_n():
    """Same gap on log–log axes, showing the `O(1/n)` decay."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ns = np.array([30, 50, 75, 100, 150, 200, 300, 500, 750, 1000, 1500, 2000])
    snrs_db = [-2, 0, 3, 6]
    cmap = plt.cm.plasma(np.linspace(0.1, 0.8, len(snrs_db)))

    for snr_db, color in zip(snrs_db, cmap):
        gaps = []
        for n in ns:
            nct = NoncentralTConverse(n=n, snr_db=snr_db).converse_rate_log(1e-3)
            chi2 = _polyanskiy_chi2_rate(n, snr_db, 1e-3)
            gaps.append(chi2 - nct if np.isfinite(chi2) and np.isfinite(nct) else np.nan)
        ax.loglog(ns, gaps, color=color, linewidth=2, marker="o",
                  markersize=5, label=f"SNR = {snr_db} dB")

    ns_ref = np.array([30, 2000])
    ax.loglog(ns_ref, 0.8 / ns_ref, "k--", linewidth=1, alpha=0.5,
              label=r"$\propto 1/n$ (reference)")

    ax.set_xlabel("Block length n", fontsize=13)
    ax.set_ylabel(
        r"$R_{\chi^2}^{\mathrm{Polyanskiy}} - R_{\mathrm{NCT}}^{\mathrm{ours}}$"
        "  (bits/channel use)",
        fontsize=12,
    )
    ax.set_title(
        r"Gap shrinks as $O(1/n)$ — both bounds converge to Shannon capacity;"
        r" ours is tighter at every finite $n$",
        fontsize=11,
    )
    ax.legend(fontsize=11, loc="upper right")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    _save(fig, "mismatch_gap_vs_n")


# ===========================================================================
# 2. Showcase waterfall: log P_e vs R at multiple SNRs
# ===========================================================================

def _waterfall_curves(n: int, snr_db: float, R_points: int = 35):
    snr = 10 ** (snr_db / 10)
    C = 0.5 * np.log2(1 + snr)
    R_min = max(0.03, 0.3 * C)
    R_max = 0.985 * C
    Rs = np.linspace(R_min, R_max, R_points)

    conv = NoncentralTConverse(n=n, snr_db=snr_db)
    rcu = RCUAchievable(n=n, snr_db=snr_db)

    eps_conv = np.array([conv.converse_error_log(R) for R in Rs])
    eps_rcu = np.array([rcu.achievable_error(R) for R in Rs])
    eps_norm = np.array([normal_approx_error(n, R, snr_db) for R in Rs])
    return Rs, C, eps_conv, eps_rcu, eps_norm


def _waterfall_panel(ax, n: int, snr_list, cmap, eps_floor: float,
                     show_capacity_labels: bool) -> None:
    for snr_db, color in zip(snr_list, cmap):
        Rs, C, eps_conv, eps_rcu, eps_norm = _waterfall_curves(n, snr_db)
        ax.semilogy(Rs, _safe_positive(eps_conv),
                    color=color, linestyle="--", linewidth=1.8)
        ax.semilogy(Rs, _safe_positive(eps_rcu),
                    color=color, linestyle="-", linewidth=2.0,
                    marker="o", markersize=4, markevery=3)
        ax.semilogy(Rs, _safe_positive(eps_norm),
                    color=color, linestyle=":", linewidth=1.8)

        ax.axvline(C, color=color, linestyle="-", alpha=0.2, linewidth=0.7)
        if show_capacity_labels:
            ax.text(C, eps_floor * 1.8, f"$C$ ({snr_db} dB)",
                    ha="center", va="bottom", fontsize=8, color=color,
                    rotation=90, alpha=0.85)


def fig_showcase_waterfall(n: int = 500,
                           snr_list=(0, 4, 8, 12, 16, 20),
                           inset_snr: int = 8,
                           eps_floor: float = 1e-14):
    """Flagship figure: converse / RCU+ / normal approximation at several SNRs,
    with a zoom inset on one operating point."""
    cmap = plt.cm.viridis(np.linspace(0.08, 0.92, len(snr_list)))
    fig, ax = plt.subplots(figsize=(13, 8))

    _waterfall_panel(ax, n, snr_list, cmap, eps_floor, show_capacity_labels=True)

    ax.set_xlabel("Rate R (bits/channel use)", fontsize=13)
    ax.set_ylabel(r"Error probability  $P_e$", fontsize=13)
    ax.set_title(
        f"AWGN achievable vs converse bounds, n = {n}\n"
        "each colour = one SNR  ·  dashed = converse, solid ○ = RCU$^+$, "
        "dotted = normal approx",
        fontsize=12,
    )
    ax.set_ylim(eps_floor, 1)
    ax.grid(True, which="both", alpha=0.3)

    snr_handles = [
        mlines.Line2D([], [], color=c, linestyle="-", linewidth=2.5,
                      marker="o", markersize=5, label=f"SNR = {s} dB")
        for s, c in zip(snr_list, cmap)
    ]
    ax.legend(handles=snr_handles, loc="upper left", fontsize=10,
              title="Operating point", framealpha=0.92)

    # Zoom inset on one SNR
    Rs, C, eps_conv, eps_rcu, eps_norm = _waterfall_curves(n, inset_snr, R_points=60)
    color = cmap[snr_list.index(inset_snr)]

    ax_in = inset_axes(ax, width="38%", height="38%", loc="lower right",
                       borderpad=1.8)
    ax_in.semilogy(Rs, _safe_positive(eps_conv),
                   color=color, linestyle="--", linewidth=1.8, label="Converse")
    ax_in.semilogy(Rs, _safe_positive(eps_rcu),
                   color=color, linestyle="-", linewidth=2.0,
                   marker="o", markersize=4, markevery=3, label="RCU$^+$")
    ax_in.semilogy(Rs, _safe_positive(eps_norm),
                   color=color, linestyle=":", linewidth=1.8, label="Normal")
    mask = (eps_rcu > eps_floor) & (eps_rcu < 1)
    R_in = Rs[mask]
    if len(R_in) > 3:
        R_lo = R_in[max(0, len(R_in) // 3)]
        R_hi = R_in[min(len(R_in) - 1, 5 * len(R_in) // 6)]
        ax_in.set_xlim(R_lo, R_hi)
    vals = np.concatenate([eps_conv[mask], eps_rcu[mask]])
    vals = vals[vals > 0]
    if len(vals):
        ax_in.set_ylim(max(eps_floor, vals.min() * 0.5),
                       min(1, vals.max() * 3))
    ax_in.grid(True, which="both", alpha=0.3)
    ax_in.set_title(f"Zoom at SNR = {inset_snr} dB", fontsize=10)
    ax_in.tick_params(labelsize=8)
    ax_in.legend(fontsize=8, loc="lower right")

    fig.tight_layout()
    _save(fig, f"showcase_waterfall_n{n}")


# ===========================================================================
# 3. Standard bound comparisons
# ===========================================================================

_STANDARD_CURVES = ["capacity", "converse_nct", "rcu", "gallager",
                    "kappabeta_ppv", "normal"]


def fig_rate_vs_snr():
    for n in [50, 200, 1000]:
        fig, _ = plot(_STANDARD_CURVES,
                      x="snr_db", y="rate_bits", x_range=(-2, 10),
                      n=n, epsilon=1e-3, n_points=22)
        _save(fig, f"rate_vs_snr_n{n}")


def fig_rate_vs_n():
    for snr_db in [0, 3]:
        fig, _ = plot(_STANDARD_CURVES,
                      x="n", y="rate_bits", x_range=(50, 1000),
                      snr_db=snr_db, epsilon=1e-3, n_points=20)
        _save(fig, f"rate_vs_n_snr{snr_db}")


def fig_rate_vs_eps():
    for snr_db in [0, 3]:
        fig, _ = plot(_STANDARD_CURVES,
                      x="epsilon", y="rate_bits", x_range=(-5, -0.3),
                      n=200, snr_db=snr_db, n_points=22)
        _save(fig, f"rate_vs_eps_snr{snr_db}")


# ===========================================================================
# 4. Waterfall (R → ε)
# ===========================================================================

def fig_error_vs_snr():
    for n in [200, 500]:
        fig, _ = plot(
            ["converse_nct", "rcu", "normal"],
            x="snr_db", y="epsilon", x_range=(-4, 10),
            n=n, rate_bits=0.3, n_points=25,
        )
        _save(fig, f"error_vs_snr_n{n}")


# ===========================================================================
# 5. Exact random coding vs RCU+ envelope (small n)
# ===========================================================================

def _exact_rc_panel(ax, n: int, snr_db: float, R_range, n_samples: int,
                    n_points: int, seed: int = 42) -> None:
    Rs = np.linspace(*R_range, n_points)
    ex = ExactRandomCoding(n=n, snr_db=snr_db, seed=seed)
    rcu = RCUAchievable(n=n, snr_db=snr_db)
    conv = NoncentralTConverse(n=n, snr_db=snr_db)

    P_exact, P_envelope, P_conv = [], [], []
    for R in Rs:
        P_exact.append(ex.exact_error(R, n_samples=n_samples))
        P_envelope.append(rcu.achievable_error(R))
        P_conv.append(conv.converse_error_log(R))

    ax.semilogy(Rs, _safe_positive(P_conv), "r-", linewidth=1.8,
                label="NCT converse  (lower bound on $P_e$)")
    ax.semilogy(Rs, _safe_positive(P_exact), "k-", linewidth=2.4,
                marker="o", markersize=6,
                label=r"exact RC:  $\mathbb{E}[1-(1-G(T))^{M-1}]$")
    ax.semilogy(Rs, _safe_positive(P_envelope), "g--", linewidth=2,
                marker="^", markersize=5,
                label=r"RCU$^+$ envelope:  $\mathbb{E}[\min(1,(M-1)G(T))]$")

    C = 0.5 * np.log2(1 + 10 ** (snr_db / 10))
    ax.axvline(C, color="k", linestyle=":", alpha=0.4)
    ax.text(C, 0.6, f"  $C={C:.3f}$", fontsize=9, alpha=0.6)
    ax.set_xlabel("Rate R (bits/channel use)", fontsize=12)
    ax.set_ylabel(r"Error probability  $P_e$", fontsize=12)
    ax.set_title(f"n = {n}", fontsize=12)
    ax.grid(True, which="both", alpha=0.3)
    ax.set_ylim(1e-10, 1)
    ax.legend(fontsize=10, loc="lower right")


def fig_exact_rc_vs_bounds():
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    _exact_rc_panel(axes[0], n=30, snr_db=0, R_range=(0.03, 0.35),
                    n_samples=500_000, n_points=18)
    _exact_rc_panel(axes[1], n=50, snr_db=0, R_range=(0.05, 0.42),
                    n_samples=500_000, n_points=18)
    _exact_rc_panel(axes[2], n=100, snr_db=0, R_range=(0.10, 0.43),
                    n_samples=500_000, n_points=18)
    fig.suptitle(
        "Exact random coding vs the RCU$^+$ min-envelope.  AWGN, SNR = 0 dB.",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout()
    _save(fig, "exact_rc_vs_bounds")


# ===========================================================================
# 6. Extended reach: where the log-domain pipeline works and scipy fails
# ===========================================================================

def fig_extended_reach(n_panelB: int = 500, snr_panelB: int = 6):
    """Two panels showing the numerical reach of the log-domain bounds.

    Left: converse rate vs blocklength out to n = 5000 — the log-domain
    cone-packing converse holds throughout, while the linear scipy NCT and
    the χ² (ncx2) evaluation hit a NaN wall.  Right: a deep waterfall — the
    log-domain converse and a deep-grid RCU⁺ track each other down to
    P_e ≈ 1e-45, while the default-grid RCU⁺ flattens at its ε-floor.
    """
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(17, 6.8))

    # ---- Panel A: converse reach in n -----------------------------------
    ns = np.unique(np.round(np.linspace(50, 5000, 36)).astype(int))
    snr_db, eps = 6, 1e-6
    log_c, lin_c, chi_c = [], [], []
    for n in ns:
        c = NoncentralTConverse(n=n, snr_db=snr_db)
        log_c.append(c.converse_rate_log(eps))
        lin_c.append(c.converse_rate(eps))
        chi_c.append(ChiSquaredConverse(n=n, snr_db=snr_db).converse_rate(eps))
    log_c, lin_c, chi_c = map(np.array, (log_c, lin_c, chi_c))

    axA.plot(ns, log_c, "r-", lw=2.6,
             label="Shannon cone-packing, log-domain (ours)")
    axA.plot(ns, _safe_positive(lin_c), "b--", lw=1.8, marker="s",
             markevery=3, ms=5, label="same bound — linear scipy NCT")
    axA.plot(ns, _safe_positive(chi_c), "g-.", lw=1.8, marker="o",
             markevery=3, ms=5, label=r"$\chi^2$ relaxation — scipy ncx2")

    def _last_finite_n(vals):
        finite = ns[np.isfinite(vals) & (vals > 0)]
        return finite.max() if len(finite) else None

    for vals, color, name in [(lin_c, "b", "linear NCT"),
                              (chi_c, "g", r"$\chi^2$")]:
        n_stop = _last_finite_n(vals)
        if n_stop is not None and n_stop < ns.max():
            axA.axvline(n_stop, color=color, ls=":", alpha=0.5, lw=1.2)
            axA.text(n_stop, axA.get_ylim()[0], f"  {name} NaN\n  beyond n≈{n_stop}",
                     color=color, fontsize=8, va="bottom", rotation=90, alpha=0.8)

    axA.set_xlabel("Block length n", fontsize=12)
    axA.set_ylabel("Converse rate (bits/channel use)", fontsize=12)
    axA.set_title(
        f"Converse reach in n  (SNR = {snr_db} dB, ε = $10^{{-6}}$)\n"
        "log-domain holds to n = 5000; scipy linear paths hit a NaN wall",
        fontsize=11)
    axA.legend(fontsize=10, loc="lower right")
    axA.grid(True, alpha=0.3)

    # ---- Panel B: deep waterfall (depth in ε) ---------------------------
    n, snr_db = n_panelB, snr_panelB
    C = 0.5 * np.log2(1 + 10 ** (snr_db / 10))
    conv = NoncentralTConverse(n=n, snr_db=snr_db)
    rcu_deep = RCUAchievable(n=n, snr_db=snr_db, eps_min=1e-100)  # slow build
    rcu_def = RCUAchievable(n=n, snr_db=snr_db)                   # default 1e-10

    Rs = np.linspace(0.55, 0.985 * C, 42)
    log10e = np.log10(np.e)
    eps_conv = 10.0 ** np.array([conv.log_converse_error(R) * log10e for R in Rs])
    eps_deep = 10.0 ** np.array([rcu_deep.log_achievable_error(R) * log10e for R in Rs])
    eps_def = 10.0 ** np.array([rcu_def.log_achievable_error(R) * log10e for R in Rs])

    floor = 1e-45
    axB.semilogy(Rs, _safe_positive(eps_conv), "r-", lw=2.2,
                 label="Shannon cone-packing converse (log)")
    axB.semilogy(Rs, _safe_positive(eps_deep), "g-", lw=2.4, marker="^",
                 markevery=3, ms=5,
                 label=r"RCU$^+$ achievable — log, $\varepsilon_{\min}=10^{-100}$")
    axB.semilogy(Rs, _safe_positive(eps_def), "m:", lw=2.0,
                 label=r"RCU$^+$ — default grid $\varepsilon_{\min}=10^{-10}$")
    axB.axhspan(floor, 1e-10, color="grey", alpha=0.08)
    axB.text(Rs[0], 3e-10, "  default-grid floor", fontsize=8, color="grey",
             va="bottom")

    axB.set_xlabel("Rate R (bits/channel use)", fontsize=12)
    axB.set_ylabel(r"Error probability  $P_e$", fontsize=12)
    axB.set_title(
        f"Deep waterfall  (n = {n}, SNR = {snr_db} dB)\n"
        "log-domain + deep grid tracks the converse to $P_e\\approx10^{-45}$",
        fontsize=11)
    axB.set_ylim(floor, 1)
    axB.grid(True, which="both", alpha=0.3)
    axB.legend(fontsize=10, loc="upper right")

    fig.tight_layout()
    _save(fig, "extended_reach")


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    _ensure_out_dir()
    print(f"Generating chapter figures into {OUT_DIR}/")

    print("\n[1] mismatch plots ...")
    fig_mismatch_gap_vs_snr()
    fig_mismatch_gap_vs_n()

    print("\n[2] showcase waterfall ...")
    fig_showcase_waterfall(n=500)

    print("\n[3] rate vs SNR / n / eps ...")
    fig_rate_vs_snr()
    fig_rate_vs_n()
    fig_rate_vs_eps()

    print("\n[4] error vs SNR ...")
    fig_error_vs_snr()

    print("\n[5] exact RC vs RCU+ envelope ...")
    fig_exact_rc_vs_bounds()

    print("\n[6] extended reach (log-domain vs scipy NaN wall) ...")
    fig_extended_reach()

    print(f"\nAll figures written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
