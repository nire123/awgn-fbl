#!/usr/bin/env python
"""
Sanity check for the bounds library.

Reads every CSV under ``plots/stress/`` (produced by ``stress_plots.py``)
and validates structural properties that should hold regardless of the
operating point:

    S1  Every achievable bound must be below the corresponding converse at
        the same (n, SNR, ε) triple.
    S2  Every rate bound must be below Shannon capacity.
    S3  κβ v1 and v2 should agree to within a few percent.
    S4  NCT and χ² converse should agree to within ~0.003 bits/use.
    S5  Monotonicity along the x-axis where expected:
            R(n) ↗, R(SNR) ↗, R(ε) ↗,
            ε(n) ↘, ε(SNR) ↘.
    S6  No unexplained NaN (a NaN is OK where the bound is genuinely
        undefined; flagged for review only when a majority of points in a
        curve are NaN).

Output: a summary table printed to stdout and saved as
``plots/stress/_sanity_report.csv``.
"""

from __future__ import annotations

import csv
import pathlib
from collections import defaultdict
from typing import Any

import numpy as np


ROOT = pathlib.Path("plots/stress")
OUT_CSV = ROOT / "_sanity_report.csv"


# ---------------------------------------------------------------------------
# Load all CSVs into a uniform structure
# ---------------------------------------------------------------------------

def load_all() -> list[dict[str, Any]]:
    """Return list of {plot, x_var, y_var, fixed, curves: {name: {xs, ys}}}."""
    plots = []
    for csv_file in sorted(ROOT.rglob("*.csv")):
        if csv_file.name.startswith("_"):
            continue
        with open(csv_file, newline="", encoding="utf-8") as f:
            rdr = csv.reader(f)
            header = next(rdr)
            rows = list(rdr)

        # Identify x and y columns: everything after "curve" and fixed cols
        curve_col = header.index("curve")
        x_var = header[-2]
        y_var = header[-1]
        fixed_cols = header[curve_col + 1 : -2]

        fixed = {}
        curves: dict[str, dict[str, list]] = defaultdict(
            lambda: {"xs": [], "ys": []})
        for row in rows:
            cname = row[curve_col]
            for i, col in enumerate(fixed_cols):
                fixed[col] = float(row[curve_col + 1 + i])
            try:
                x = float(row[-2])
                y = float(row[-1])
            except ValueError:
                x, y = float("nan"), float("nan")
            curves[cname]["xs"].append(x)
            curves[cname]["ys"].append(y)

        # Convert to numpy arrays
        for c in curves:
            curves[c]["xs"] = np.asarray(curves[c]["xs"])
            curves[c]["ys"] = np.asarray(curves[c]["ys"])

        plots.append({
            "file": str(csv_file.relative_to(ROOT)),
            "x_var": x_var,
            "y_var": y_var,
            "fixed": dict(fixed),
            "curves": dict(curves),
        })
    return plots


# ---------------------------------------------------------------------------
# Classifiers
# ---------------------------------------------------------------------------

CONVERSE = {"converse_nct", "converse_chi2", "capacity"}
ACHIEVABLE = {"rcu", "kappabeta", "kappabeta_ppv", "gallager"}
APPROX = {"normal"}


# Curve label → internal id (labels are what matplotlib shows in the legend);
# must stay in sync with ``awgn_fbl.plot.CURVE_STYLES``.
LABEL_TO_ID = {
    "Shannon capacity": "capacity",
    "Shannon cone-packing converse": "converse_nct",
    r"$\chi^2$ converse (Polyanskiy, relaxed $Q_Y$)": "converse_chi2",
    "RCU$^+$ achievable (ours)": "rcu",
    r"$\kappa\beta$ achievable (simple)": "kappabeta",
    r"$\kappa\beta$ achievable (PPV-faithful)": "kappabeta_ppv",
    "Gallager achievable": "gallager",
    "Normal approximation": "normal",
}


def _curve_id(label: str) -> str:
    return LABEL_TO_ID.get(label, label)


# ---------------------------------------------------------------------------
# Individual sanity tests
# ---------------------------------------------------------------------------

def check_s1_achievable_vs_converse(plot) -> list[str]:
    """S1: every achievable ≤ converse (NCT, preferred) at each x."""
    y_var = plot["y_var"]
    if y_var != "rate_bits":
        return []  # S1 only meaningful on rate-axis plots

    curves = {_curve_id(k): v for k, v in plot["curves"].items()}
    if "converse_nct" not in curves:
        return []

    conv_xs = curves["converse_nct"]["xs"]
    conv_ys = curves["converse_nct"]["ys"]

    violations = []
    for name, data in curves.items():
        cid = name
        if cid not in ACHIEVABLE:
            continue
        ach_xs = data["xs"]
        ach_ys = data["ys"]
        # Interpolate converse at the achievable's x values
        mask = np.isfinite(conv_ys)
        if mask.sum() < 2:
            continue
        try:
            conv_at = np.interp(
                ach_xs, conv_xs[mask], conv_ys[mask],
                left=np.nan, right=np.nan,
            )
        except Exception:
            continue
        for x, ay, cy in zip(ach_xs, ach_ys, conv_at):
            if np.isfinite(ay) and np.isfinite(cy) and ay > cy + 1e-6:
                violations.append(
                    f"{cid} > NCT converse at {plot['x_var']}={x:g}: "
                    f"{ay:.4f} > {cy:.4f}"
                )
    return violations


def check_s2_below_capacity(plot) -> list[str]:
    """S2: *achievable* bounds must be ≤ Shannon capacity at all ε.

    The converse and normal approximation can legitimately exceed C at large
    ε (Shannon's capacity is the ε→0 limit), so they are exempted.
    """
    if plot["y_var"] != "rate_bits":
        return []
    curves = {_curve_id(k): v for k, v in plot["curves"].items()}
    if "capacity" not in curves:
        return []

    cap_ys = curves["capacity"]["ys"]
    if not np.all(np.isfinite(cap_ys)):
        return []
    cap_max = np.nanmax(cap_ys) + 1e-6

    violations = []
    for name, data in curves.items():
        # Skip capacity itself, the converse, and the (non-rigorous)
        # normal approximation — those may exceed C at large ε.
        if name in {"capacity", "converse_nct", "converse_chi2", "normal"}:
            continue
        for x, y in zip(data["xs"], data["ys"]):
            if np.isfinite(y) and y > cap_max:
                violations.append(
                    f"{name} > capacity at {plot['x_var']}={x:g}: "
                    f"{y:.4f} > {cap_max:.4f}"
                )
    return violations


def check_s3_kb_v1_v2_agree(plot, tol: float = 0.02) -> list[str]:
    """S3: κβ v1 and v2 should agree (same formulas, different numerics)."""
    curves = {_curve_id(k): v for k, v in plot["curves"].items()}
    if "kappabeta" not in curves or "kappabeta_ppv" not in curves:
        return []
    x1, y1 = curves["kappabeta"]["xs"], curves["kappabeta"]["ys"]
    x2, y2 = curves["kappabeta_ppv"]["xs"], curves["kappabeta_ppv"]["ys"]
    if not np.array_equal(x1, x2):
        return []
    violations = []
    for x, a, b in zip(x1, y1, y2):
        if np.isfinite(a) and np.isfinite(b):
            if abs(a - b) > tol:
                violations.append(
                    f"κβ v1 vs v2 disagree at {plot['x_var']}={x:g}: "
                    f"v1={a:.4f} v2={b:.4f} (|Δ|={abs(a-b):.4f})"
                )
    return violations


def check_s4_nct_vs_chi2(plot, tol: float = 0.01) -> list[str]:
    """S4: NCT and χ² converse should agree to within ~0.003–0.01 bits."""
    curves = {_curve_id(k): v for k, v in plot["curves"].items()}
    if "converse_nct" not in curves or "converse_chi2" not in curves:
        return []
    x1, y1 = curves["converse_nct"]["xs"], curves["converse_nct"]["ys"]
    x2, y2 = curves["converse_chi2"]["xs"], curves["converse_chi2"]["ys"]
    if not np.array_equal(x1, x2):
        return []
    violations = []
    for x, a, b in zip(x1, y1, y2):
        if np.isfinite(a) and np.isfinite(b):
            if abs(a - b) > tol:
                violations.append(
                    f"NCT vs χ² disagree at {plot['x_var']}={x:g}: "
                    f"NCT={a:.4f} χ²={b:.4f} (|Δ|={abs(a-b):.4f})"
                )
    return violations


def check_s5_monotonicity(plot) -> list[str]:
    """S5: rate increases in n, SNR, ε; error decreases in n, SNR."""
    x_var = plot["x_var"]
    y_var = plot["y_var"]
    if y_var == "rate_bits" and x_var in ("n", "snr_db", "epsilon"):
        expected = "increasing"
    elif y_var == "epsilon" and x_var in ("n", "snr_db"):
        expected = "decreasing"
    elif y_var == "snr_db" and x_var in ("n",):
        # SNR required for a given rate should DECREASE as n grows (easier)
        expected = "decreasing"
    else:
        return []  # no strong expectation

    violations = []
    for name, data in plot["curves"].items():
        cid = _curve_id(name)
        xs = np.asarray(data["xs"])
        ys = np.asarray(data["ys"])
        mask = np.isfinite(ys) & np.isfinite(xs)
        if mask.sum() < 3:
            continue
        # Sort by x
        order = np.argsort(xs[mask])
        xs_s = xs[mask][order]
        ys_s = ys[mask][order]
        dy = np.diff(ys_s)
        # Allow small downward noise (tol of few ppm relative)
        tol = max(1e-6, 1e-4 * np.nanmean(np.abs(ys_s)))
        if expected == "increasing":
            if np.any(dy < -tol):
                bad_idx = int(np.argmin(dy))
                violations.append(
                    f"{cid} not monotone ↑ in {x_var}: "
                    f"drop {dy[bad_idx]:.4g} at {x_var}={xs_s[bad_idx+1]:g}"
                )
        else:
            if np.any(dy > tol):
                bad_idx = int(np.argmax(dy))
                violations.append(
                    f"{cid} not monotone ↓ in {x_var}: "
                    f"rise {dy[bad_idx]:.4g} at {x_var}={xs_s[bad_idx+1]:g}"
                )
    return violations


def check_s6_nan_density(plot, threshold: float = 0.8) -> list[str]:
    """S6: flag curves where > `threshold` fraction of points are NaN."""
    violations = []
    for name, data in plot["curves"].items():
        cid = _curve_id(name)
        ys = np.asarray(data["ys"])
        if len(ys) == 0:
            continue
        frac_nan = np.mean(~np.isfinite(ys))
        # Note: for inversion plots where the curve is fundamentally undefined
        # for parts of the domain, we may see legitimate NaN, so threshold is
        # set high (80%) and intended as a smoke check, not a hard fail.
        if frac_nan > threshold:
            violations.append(
                f"{cid} is {frac_nan:.0%} NaN on this plot"
            )
    return violations


ALL_CHECKS = [
    ("S1_achievable<=converse",    check_s1_achievable_vs_converse),
    ("S2_below_capacity",          check_s2_below_capacity),
    ("S3_kb_v1_v2_agree",          check_s3_kb_v1_v2_agree),
    ("S4_nct_chi2_agree",          check_s4_nct_vs_chi2),
    ("S5_monotonicity",            check_s5_monotonicity),
    ("S6_nan_density",             check_s6_nan_density),
]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def build_summary_table() -> list[dict]:
    """Compute every bound at a standard grid of (n, SNR, ε) operating points
    and return rows for a CSV summary.

    Rows include per-bound rate and gap-to-NCT-converse, plus the min/max
    achievable bound so the ordering is obvious.
    """
    import warnings
    warnings.filterwarnings("ignore")

    from awgn_fbl import NoncentralTConverse, ChiSquaredConverse
    from awgn_fbl import RCUAchievable
    from awgn_fbl import KappaBetaAchievablePPV as KappaBetaAchievableV2
    from awgn_fbl import GallagerAchievable
    from awgn_fbl import normal_approx_rate

    # Canonical grid — 3D sweep.
    Ns = [50, 100, 200, 500, 1000]
    SNRS = [-3, 0, 3, 6]
    EPS = [1e-2, 1e-3, 1e-6]

    rows = []
    for n in Ns:
        for snr_db in SNRS:
            for eps in EPS:
                snr = 10 ** (snr_db / 10)
                C = 0.5 * np.log2(1 + snr)
                try:
                    nct = NoncentralTConverse(n, snr_db).converse_rate(eps)
                except Exception:
                    nct = np.nan
                try:
                    chi2 = ChiSquaredConverse(n, snr_db).converse_rate(eps)
                except Exception:
                    chi2 = np.nan
                try:
                    rcu = RCUAchievable(n, snr_db).achievable_rate(eps)
                except Exception:
                    rcu = np.nan
                try:
                    kb = KappaBetaAchievableV2(n, snr_db).achievable_rate(eps)
                except Exception:
                    kb = np.nan
                try:
                    gal = GallagerAchievable(n, snr_db).achievable_rate(eps)
                except Exception:
                    gal = np.nan
                try:
                    nap = normal_approx_rate(n, eps, snr_db)
                except Exception:
                    nap = np.nan

                # best achievable bound at this point
                achs = {k: v for k, v in
                        [("rcu", rcu), ("gallager", gal), ("kappabeta_ppv", kb)]
                        if np.isfinite(v)}
                best_ach = max(achs.values()) if achs else np.nan
                best_ach_name = (
                    max(achs, key=achs.get) if achs else "—"
                )
                gap = (nct - rcu) if np.isfinite(nct) and np.isfinite(rcu) else np.nan

                rows.append({
                    "n": n, "snr_db": snr_db, "epsilon": eps,
                    "C": round(C, 4),
                    "nct": round(nct, 4) if np.isfinite(nct) else "nan",
                    "chi2": round(chi2, 4) if np.isfinite(chi2) else "nan",
                    "rcu": round(rcu, 4) if np.isfinite(rcu) else "nan",
                    "kappabeta_ppv": round(kb, 4) if np.isfinite(kb) else "nan",
                    "gallager": round(gal, 4) if np.isfinite(gal) else "nan",
                    "normal": round(nap, 4) if np.isfinite(nap) else "nan",
                    "best_ach": best_ach_name,
                    "gap_nct_rcu": round(gap, 4) if np.isfinite(gap) else "nan",
                })
    return rows


def write_summary_csv(rows: list[dict], path: pathlib.Path):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    print("Loading stress-plot data ...")
    plots = load_all()
    print(f"  {len(plots)} plots")

    results = []
    for p in plots:
        for check_name, fn in ALL_CHECKS:
            try:
                violations = fn(p)
            except Exception as e:
                violations = [f"check raised: {type(e).__name__}: {e}"]
            for v in violations:
                results.append({
                    "plot": p["file"],
                    "check": check_name,
                    "detail": v,
                })

    # Print summary
    print(f"\n=== RESULTS: {len(results)} violations ===\n")
    by_check: dict[str, int] = defaultdict(int)
    by_plot: dict[str, int] = defaultdict(int)
    for r in results:
        by_check[r["check"]] += 1
        by_plot[r["plot"]] += 1

    print("By check:")
    for name, _ in ALL_CHECKS:
        print(f"  {name:28s}  {by_check[name]:4d}")

    print(f"\nTop 10 plots with most violations:")
    for plot_name, count in sorted(by_plot.items(),
                                   key=lambda kv: -kv[1])[:10]:
        print(f"  {count:4d}  {plot_name}")

    # Save full report
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["plot", "check", "detail"])
        for r in results:
            w.writerow([r["plot"], r["check"], r["detail"]])
    print(f"\nFull report: {OUT_CSV}")

    # Build compact summary table
    print("\nBuilding summary table at canonical (n, SNR, eps) grid ...")
    summary_rows = build_summary_table()
    summary_path = ROOT / "_summary_table.csv"
    write_summary_csv(summary_rows, summary_path)
    print(f"Summary: {summary_path}  ({len(summary_rows)} rows)")


if __name__ == "__main__":
    main()
