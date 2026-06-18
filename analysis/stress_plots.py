#!/usr/bin/env python
"""
Stress-test the plotting pipeline by exercising all axis combinations
across a wide range of parameters.

Each plot is accompanied by a CSV containing the raw (x, y) data per curve,
which ``sanity_check.py`` consumes to validate the results programmatically.

Run::

    python stress_plots.py
"""

from __future__ import annotations

import csv
import os
import warnings
from typing import Iterable, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")

from awgn_fbl.plot import plot, ALL_CURVES, ERROR_CURVES


OUT = "plots/stress"

RATE_CURVES = ["capacity", "converse_nct", "rcu", "gallager",
               "kappabeta_v2", "normal"]
ERR_CURVES = [c for c in ALL_CURVES if c in ERROR_CURVES]
INV_CURVES = ["converse_nct", "rcu", "gallager", "normal"]


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------

def _extract_data(ax) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Pull the (x, y) arrays back out of a matplotlib Axes, keyed by label."""
    out = {}
    for line in ax.get_lines():
        label = line.get_label()
        if label.startswith("_"):
            continue
        x = np.asarray(line.get_xdata(), dtype=float)
        y = np.asarray(line.get_ydata(), dtype=float)
        out[label] = (x, y)
    return out


def _save_csv(path: str, data: dict, x_name: str, y_name: str,
              fixed: dict):
    """Write one CSV per (plot, curve) entry.

    Each row: curve, <fixed cols>, x, y.
    """
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        fixed_cols = sorted(fixed)
        w.writerow(["curve", *fixed_cols, x_name, y_name])
        for curve, (xs, ys) in data.items():
            for x, y in zip(xs, ys):
                w.writerow([
                    curve,
                    *[fixed[k] for k in fixed_cols],
                    f"{x:g}", f"{y:g}",
                ])


# ---------------------------------------------------------------------------
# Wrap plot() to save PNG + CSV
# ---------------------------------------------------------------------------

def _save(fig_ax, subdir: str, name: str,
          x: str, y: str, fixed: dict):
    fig, ax = fig_ax
    path = f"{OUT}/{subdir}"
    os.makedirs(path, exist_ok=True)
    png = f"{path}/{name}.png"
    csv_path = f"{path}/{name}.csv"

    data = _extract_data(ax)
    _save_csv(csv_path, data, x, y, fixed)

    fig.savefig(png, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  {png}")


# ---------------------------------------------------------------------------
# 1. STANDARD
# ---------------------------------------------------------------------------

def standard_suite():
    print("\n[1] STANDARD — parameter sweeps for each plot type")

    # Rate vs SNR
    for n in [50, 100, 200, 500, 1000]:
        for eps in [1e-2, 1e-3, 1e-6]:
            fa = plot(
                RATE_CURVES, x="snr_db", y="rate_bits", x_range=(-4, 8),
                n=n, epsilon=eps, n_points=20,
            )
            _save(fa, "standard/rate_vs_snr", f"n{n}_eps{eps:.0e}",
                  x="snr_db", y="rate_bits",
                  fixed={"n": n, "epsilon": eps})

    # Rate vs epsilon
    for n in [50, 200, 1000]:
        for snr_db in [-3, 0, 3, 6]:
            fa = plot(
                RATE_CURVES, x="epsilon", y="rate_bits", x_range=(-8, -0.5),
                n=n, snr_db=snr_db, n_points=20,
            )
            _save(fa, "standard/rate_vs_eps", f"n{n}_snr{snr_db:+d}",
                  x="epsilon", y="rate_bits",
                  fixed={"n": n, "snr_db": snr_db})

    # Rate vs n
    for snr_db in [-3, 0, 3, 6]:
        for eps in [1e-2, 1e-3, 1e-6]:
            fa = plot(
                RATE_CURVES, x="n", y="rate_bits", x_range=(30, 2000),
                n_points=25, snr_db=snr_db, epsilon=eps,
            )
            _save(fa, "standard/rate_vs_n", f"snr{snr_db:+d}_eps{eps:.0e}",
                  x="n", y="rate_bits",
                  fixed={"snr_db": snr_db, "epsilon": eps})

    # Error vs SNR
    for n in [100, 200, 500, 1000]:
        for R in [0.1, 0.3, 0.5]:
            fa = plot(
                ERR_CURVES, x="snr_db", y="epsilon", x_range=(-4, 10),
                n_points=20, n=n, rate_bits=R,
            )
            _save(fa, "standard/error_vs_snr", f"n{n}_R{R:.2f}",
                  x="snr_db", y="epsilon",
                  fixed={"n": n, "rate_bits": R})


# ---------------------------------------------------------------------------
# 2. INVERSIONS
# ---------------------------------------------------------------------------

def inversions_suite():
    print("\n[2] INVERSIONS — axis combinations only possible with unified API")

    # Error vs n
    for snr_db in [-2, 0, 3]:
        for R in [0.1, 0.3]:
            fa = plot(
                ERR_CURVES, x="n", y="epsilon", x_range=(50, 2000),
                n_points=25, snr_db=snr_db, rate_bits=R,
            )
            _save(fa, "inversions/error_vs_n", f"snr{snr_db:+d}_R{R:.2f}",
                  x="n", y="epsilon",
                  fixed={"snr_db": snr_db, "rate_bits": R})

    # SNR vs n
    for eps in [1e-3, 1e-6]:
        for R in [0.1, 0.3, 0.5]:
            fa = plot(
                INV_CURVES, x="n", y="snr_db", x_range=(50, 2000),
                n_points=20, epsilon=eps, rate_bits=R,
            )
            _save(fa, "inversions/snr_vs_n", f"R{R:.2f}_eps{eps:.0e}",
                  x="n", y="snr_db",
                  fixed={"epsilon": eps, "rate_bits": R})

    # SNR vs eps
    for n in [100, 500]:
        for R in [0.2, 0.4]:
            fa = plot(
                INV_CURVES, x="epsilon", y="snr_db", x_range=(-8, -0.5),
                n_points=20, n=n, rate_bits=R,
            )
            _save(fa, "inversions/snr_vs_eps", f"n{n}_R{R:.2f}",
                  x="epsilon", y="snr_db",
                  fixed={"n": n, "rate_bits": R})


# ---------------------------------------------------------------------------
# 3. EDGE CASES
# ---------------------------------------------------------------------------

def edge_suite():
    print("\n[3] EDGE CASES — probing numerical limits")

    for eps in [1e-3, 1e-6]:
        fa = plot(
            RATE_CURVES, x="n", y="rate_bits", x_range=(20, 100),
            n_points=20, snr_db=0.0, epsilon=eps,
        )
        _save(fa, "edge/small_n", f"eps{eps:.0e}",
              x="n", y="rate_bits",
              fixed={"snr_db": 0.0, "epsilon": eps})

    for snr_db in [3, 6, 9]:
        fa = plot(
            ["capacity", "converse_nct", "converse_chi2", "rcu", "normal"],
            x="n", y="rate_bits", x_range=(500, 5000),
            n_points=25, snr_db=snr_db, epsilon=1e-3,
        )
        _save(fa, "edge/large_n", f"snr{snr_db:+d}dB",
              x="n", y="rate_bits",
              fixed={"snr_db": snr_db, "epsilon": 1e-3})

    for n in [200, 1000]:
        fa = plot(
            RATE_CURVES, x="snr_db", y="rate_bits", x_range=(-8, 15),
            n_points=30, n=n, epsilon=1e-3,
        )
        _save(fa, "edge/extreme_snr", f"n{n}",
              x="snr_db", y="rate_bits",
              fixed={"n": n, "epsilon": 1e-3})

    for n in [200, 500]:
        fa = plot(
            RATE_CURVES, x="epsilon", y="rate_bits", x_range=(-12, -1),
            n_points=25, n=n, snr_db=0.0,
        )
        _save(fa, "edge/tiny_eps", f"n{n}",
              x="epsilon", y="rate_bits",
              fixed={"n": n, "snr_db": 0.0})

    for n in [200, 1000]:
        fa = plot(
            ERR_CURVES, x="snr_db", y="epsilon", x_range=(-2, 15),
            n_points=25, n=n, rate_bits=0.7,
        )
        _save(fa, "edge/high_rate_waterfall", f"n{n}",
              x="snr_db", y="epsilon",
              fixed={"n": n, "rate_bits": 0.7})

    fa = plot(
        ERR_CURVES, x="snr_db", y="epsilon", x_range=(-5, 5),
        n_points=25, n=200, rate_bits=0.05,
    )
    _save(fa, "edge/low_rate_waterfall", "n200_R0.05",
          x="snr_db", y="epsilon",
          fixed={"n": 200, "rate_bits": 0.05})

    fa = plot(
        ["capacity", "converse_nct", "rcu", "gallager", "normal"],
        x="n", y="rate_bits", x_range=(30, 3000),
        n_points=30, snr_db=0.0, epsilon=1e-3,
    )
    _save(fa, "edge/gallager_regime", "snr0_eps1e-3",
          x="n", y="rate_bits",
          fixed={"snr_db": 0.0, "epsilon": 1e-3})


def main():
    print("Stress-testing the plotting pipeline")
    standard_suite()
    inversions_suite()
    edge_suite()
    print(f"\nDone. PNG + CSV under {OUT}")


if __name__ == "__main__":
    main()
