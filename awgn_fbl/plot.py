"""
Plotting utilities for AWGN finite-blocklength bounds.

Every bound curve is parametrised by four variables
`(n, snr_db, epsilon, rate_bits)`; a plot fixes two, puts a third on the
x-axis, and computes the fourth for the y-axis.  The main entry point
:func:`plot` handles all 12 such combinations uniformly.  Thin wrappers
:func:`plot_rate_vs_snr`, :func:`plot_rate_vs_epsilon`, etc. cover the
common cases with friendly parameter names.

Example::

    from awgn_fbl import plot

    # Rate vs SNR at fixed (n, ε)
    fig, ax = plot(["rcu", "converse_nct"],
                   x="snr_db", y="rate_bits", x_range=(-2, 10),
                   n=200, epsilon=1e-3)

    # Waterfall: ε vs SNR at fixed (n, R)
    fig, ax = plot(["rcu", "converse_nct", "normal"],
                   x="snr_db", y="epsilon", x_range=(-2, 8),
                   n=200, rate_bits=0.3)

The ``CURVE_STYLES`` dict controls colour / linestyle / marker / label for
each curve name; override entries to customise.  ``ALL_CURVES`` lists the
available names and ``ERROR_CURVES`` the subset that supports the
`R → ε` direction.
"""

from __future__ import annotations

import warnings
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize

from .achievable import (
    ExactRandomCoding,
    GallagerAchievable,
    KappaBetaAchievable,
    KappaBetaAchievablePPV,
    RCUAchievable,
)
from .converse import ChiSquaredConverse, NoncentralTConverse
from .normal_approx import normal_approx_error, normal_approx_rate


# ===========================================================================
# Curve styles
# ===========================================================================

CURVE_STYLES = {
    "capacity":      {"color": "k",      "ls": "-",  "lw": 2.5, "marker": None,
                      "label": "Shannon capacity"},
    "converse_nct":  {"color": "r",      "ls": "-",  "lw": 2,   "marker": "o",
                      "label": "NCT converse (ours)"},
    "converse_chi2": {"color": "b",      "ls": "--", "lw": 2,   "marker": "s",
                      "label": r"$\chi^2$ converse (Polyanskiy)"},
    "rcu":           {"color": "g",      "ls": "-",  "lw": 2.5, "marker": "^",
                      "label": "RCU$^+$ achievable (ours)"},
    "kappabeta":     {"color": "m",      "ls": "--", "lw": 1.5, "marker": "D",
                      "label": r"$\kappa\beta$ achievable (simple)"},
    "kappabeta_ppv": {"color": "purple", "ls": "-.", "lw": 1.5, "marker": "v",
                      "label": r"$\kappa\beta$ achievable (PPV-faithful)"},
    "gallager":      {"color": "orange", "ls": "--", "lw": 2,   "marker": "x",
                      "label": "Gallager achievable"},
    "normal":        {"color": "c",      "ls": ":",  "lw": 2,   "marker": None,
                      "label": "Normal approximation"},
}

ALL_CURVES = list(CURVE_STYLES.keys())

# Curves that also support the R → ε direction
ERROR_CURVES = {"converse_nct", "rcu", "normal", "gallager", "capacity"}


# ===========================================================================
# Variable metadata
# ===========================================================================

_VAR_LABEL = {
    "n":         "Block length n",
    "snr_db":    "SNR (dB)",
    "epsilon":   r"Error probability $\varepsilon$",
    "rate_bits": "Rate R (bits/channel use)",
}

_VAR_SCALE = {
    "n":         "linear",
    "snr_db":    "linear",
    "epsilon":   "log",
    "rate_bits": "linear",
}


# ===========================================================================
# Core dispatch
# ===========================================================================

def _rate_for(curve: str, *, n: int, snr_db: float, epsilon: float,
              cache: dict) -> float:
    """`curve, (n, SNR, ε) → R`."""
    snr = 10 ** (snr_db / 10)
    if curve == "capacity":
        return 0.5 * np.log2(1 + snr)
    if curve == "converse_nct":
        return NoncentralTConverse(n=n, snr_db=snr_db).converse_rate_log(epsilon)
    if curve == "converse_chi2":
        return ChiSquaredConverse(n=n, snr_db=snr_db).converse_rate(epsilon)
    if curve == "rcu":
        key = ("rcu", n, snr_db)
        if key not in cache:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cache[key] = RCUAchievable(n=n, snr_db=snr_db)
        return cache[key].achievable_rate(epsilon)
    if curve == "kappabeta":
        return KappaBetaAchievable(n=n, snr_db=snr_db).achievable_rate(epsilon)
    if curve == "kappabeta_ppv":
        return KappaBetaAchievablePPV(n=n, snr_db=snr_db).achievable_rate(epsilon)
    if curve == "gallager":
        return GallagerAchievable(n=n, snr_db=snr_db).achievable_rate(epsilon)
    if curve == "normal":
        return normal_approx_rate(n, epsilon, snr_db)
    raise ValueError(f"Unknown curve: {curve}")


def _error_for(curve: str, *, n: int, snr_db: float, rate_bits: float,
               cache: dict) -> float:
    """`curve, (n, SNR, R) → ε`."""
    snr = 10 ** (snr_db / 10)
    C = 0.5 * np.log2(1 + snr)
    if curve == "capacity":
        return 0.0 if rate_bits < C else 1.0
    if curve == "converse_nct":
        return NoncentralTConverse(n=n, snr_db=snr_db).converse_error_log(rate_bits)
    if curve == "rcu":
        key = ("rcu", n, snr_db)
        if key not in cache:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cache[key] = RCUAchievable(n=n, snr_db=snr_db)
        return cache[key].achievable_error(rate_bits)
    if curve == "gallager":
        return GallagerAchievable(n=n, snr_db=snr_db).achievable_error(rate_bits)
    if curve == "normal":
        return normal_approx_error(n, rate_bits, snr_db)
    raise ValueError(f"Curve '{curve}' does not support R → ε")


def _invert_for_snr(curve: str, *, n: int, epsilon: float, rate_bits: float,
                    cache: dict,
                    snr_range: tuple[float, float] = (-10, 30)) -> float:
    """Find `SNR` s.t. `curve(n, SNR, ε) = R` via Brent bisection."""
    def f(snr_db):
        return _rate_for(curve, n=n, snr_db=snr_db, epsilon=epsilon,
                         cache=cache) - rate_bits
    try:
        return optimize.brentq(f, *snr_range, xtol=1e-3)
    except (ValueError, RuntimeError):
        return np.nan


def _invert_for_n(curve: str, *, snr_db: float, epsilon: float, rate_bits: float,
                  cache: dict,
                  n_range: tuple[int, int] = (20, 5000)) -> float:
    """Find `n` s.t. `curve(n, SNR, ε) = R` via Brent bisection."""
    def f(n_float):
        return _rate_for(curve, n=int(round(n_float)), snr_db=snr_db,
                         epsilon=epsilon, cache=cache) - rate_bits
    try:
        return optimize.brentq(f, *n_range, xtol=0.5)
    except (ValueError, RuntimeError):
        return np.nan


def _compute_y(curve: str, *, y: str, cache: dict, **fixed) -> float:
    """Dispatch on the requested y-axis variable."""
    if y == "rate_bits":
        return _rate_for(curve, cache=cache, **fixed)
    if y == "epsilon":
        return _error_for(curve, cache=cache, **fixed)
    if y == "snr_db":
        return _invert_for_snr(curve, cache=cache, **fixed)
    if y == "n":
        return _invert_for_n(curve, cache=cache, **fixed)
    raise ValueError(f"Unknown y axis: {y}")


def _sample_x(x: str, x_range: tuple, n_points: int) -> np.ndarray:
    """Generate the independent-variable samples along the x-axis."""
    lo, hi = x_range
    if x == "epsilon":
        return np.logspace(lo, hi, n_points)
    if x == "n":
        return np.unique(np.round(np.linspace(lo, hi, n_points)).astype(int))
    return np.linspace(lo, hi, n_points)


# ===========================================================================
# Main plot function
# ===========================================================================

_ALL_VARS = {"n", "snr_db", "epsilon", "rate_bits"}


def plot(curves: Sequence[str],
         *,
         x: str,
         y: str,
         x_range: tuple,
         n_points: int = 25,
         ax: plt.Axes | None = None,
         title: str | None = None,
         **fixed):
    """Plot finite-blocklength bound curves.

    Fixes two of the four variables `(n, snr_db, epsilon, rate_bits)` via
    keyword arguments, sweeps a third along the x-axis, and computes the
    fourth for the y-axis.

    Parameters
    ----------
    curves : sequence of str
        Curve names from :data:`ALL_CURVES`.
    x, y : str
        One of `"n", "snr_db", "epsilon", "rate_bits"`, must be distinct.
    x_range : tuple
        Sampling range.  For ``x="epsilon"`` interpreted as
        ``(log10 lo, log10 hi)``.
    n_points : int
        Number of x-axis samples.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.  A new figure is created if omitted.
    title : str, optional
        Title; auto-generated from the fixed variables when omitted.
    **fixed
        Values of the two variables not on either axis; must be exactly
        the remaining pair.
    """
    if x == y:
        raise ValueError("x and y must be different variables")
    if x not in _ALL_VARS or y not in _ALL_VARS:
        raise ValueError(f"x, y must be in {_ALL_VARS}")

    remaining = _ALL_VARS - {x, y}
    if set(fixed) != remaining:
        raise ValueError(f"Must fix exactly: {remaining}, got: {set(fixed)}")

    xs = _sample_x(x, x_range, n_points)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 7))
    else:
        fig = ax.get_figure()

    cache: dict = {}
    for curve in curves:
        ys = []
        for xi in xs:
            try:
                y_val = _compute_y(
                    curve, y=y, cache=cache,
                    **{x: xi if not isinstance(xi, np.integer) else int(xi)},
                    **fixed,
                )
            except (ValueError, RuntimeError):
                y_val = np.nan
            ys.append(y_val)
        ys = np.asarray(ys, dtype=float)

        mask = np.isfinite(ys)
        if y == "epsilon":
            mask &= ys > 0
        if x == "epsilon":
            mask &= xs > 0

        _plot_curve(ax, np.asarray(xs)[mask], ys[mask], curve)

    ax.set_xlabel(_VAR_LABEL[x], fontsize=13)
    ax.set_ylabel(_VAR_LABEL[y], fontsize=13)
    if _VAR_SCALE[x] == "log":
        ax.set_xscale("log")
    if _VAR_SCALE[y] == "log":
        ax.set_yscale("log")
        ax.set_ylim(bottom=1e-15)

    if title is None:
        fixed_str = ", ".join(_fmt_fixed(k, v) for k, v in sorted(fixed.items()))
        title = f"{_VAR_LABEL[y]} vs {_VAR_LABEL[x]}   ({fixed_str})"
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax


def _plot_curve(ax, xs, ys, curve: str) -> None:
    style = CURVE_STYLES[curve]
    if len(xs) == 0:
        return
    ax.plot(
        xs, ys,
        color=style["color"], linestyle=style["ls"], linewidth=style["lw"],
        marker=style["marker"], markersize=4,
        markevery=max(1, len(xs) // 8),
        label=style["label"],
    )


def _fmt_fixed(key: str, val) -> str:
    if key == "n":
        return f"n={val}"
    if key == "snr_db":
        return f"SNR={val} dB"
    if key == "epsilon":
        return rf"$\varepsilon$={val:.0e}"
    if key == "rate_bits":
        return f"R={val:.2f}"
    return f"{key}={val}"


# ===========================================================================
# Convenience wrappers
# ===========================================================================

def plot_rate_vs_snr(n, epsilon, curves=None, snr_range=(-2, 10),
                     n_points=25, ax=None):
    """Rate vs SNR at fixed (n, ε)."""
    return plot(
        curves or ALL_CURVES,
        x="snr_db", y="rate_bits", x_range=snr_range,
        n_points=n_points, ax=ax,
        n=n, epsilon=epsilon,
    )


def plot_rate_vs_epsilon(n, snr_db, curves=None, eps_range=(-5, -0.3),
                         n_points=25, ax=None):
    """Rate vs ε at fixed (n, SNR).  ``eps_range`` is `(log10 lo, log10 hi)`."""
    return plot(
        curves or ALL_CURVES,
        x="epsilon", y="rate_bits", x_range=eps_range,
        n_points=n_points, ax=ax,
        n=n, snr_db=snr_db,
    )


def plot_rate_vs_n(snr_db, epsilon, curves=None, n_range=(50, 1000),
                   n_step=20, ax=None):
    """Rate vs blocklength at fixed (SNR, ε)."""
    n_points = max(5, int((n_range[1] - n_range[0]) / n_step) + 1)
    return plot(
        curves or ALL_CURVES,
        x="n", y="rate_bits", x_range=n_range,
        n_points=n_points, ax=ax,
        snr_db=snr_db, epsilon=epsilon,
    )


def plot_error_vs_snr(n, rate_bits, curves=None, snr_range=(-2, 10),
                      n_points=25, ax=None):
    """P_e vs SNR at fixed (n, R).  Only avg-error curves supported."""
    if curves is None:
        curves = [c for c in ALL_CURVES if c in ERROR_CURVES]
    return plot(
        curves,
        x="snr_db", y="epsilon", x_range=snr_range,
        n_points=n_points, ax=ax,
        n=n, rate_bits=rate_bits,
    )


def plot_error_vs_n(snr_db, rate_bits, curves=None, n_range=(50, 1000),
                    n_step=20, ax=None):
    """P_e vs blocklength at fixed (SNR, R)."""
    if curves is None:
        curves = [c for c in ALL_CURVES if c in ERROR_CURVES]
    n_points = max(5, int((n_range[1] - n_range[0]) / n_step) + 1)
    return plot(
        curves,
        x="n", y="epsilon", x_range=n_range,
        n_points=n_points, ax=ax,
        snr_db=snr_db, rate_bits=rate_bits,
    )


def plot_snr_vs_n(epsilon, rate_bits, curves=None, n_range=(50, 1000),
                  n_step=20, ax=None):
    """SNR required (y) vs blocklength (x) at fixed (ε, R) — inversion plot."""
    if curves is None:
        curves = ["converse_nct", "rcu", "gallager", "normal"]
    n_points = max(5, int((n_range[1] - n_range[0]) / n_step) + 1)
    return plot(
        curves,
        x="n", y="snr_db", x_range=n_range,
        n_points=n_points, ax=ax,
        epsilon=epsilon, rate_bits=rate_bits,
    )
