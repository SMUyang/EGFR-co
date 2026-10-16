#!/usr/bin/env python3
"""Task3 GC/AT-biased 4-mer frequency analysis from end-mer CSVs.

Reads the 15 Task3 5' fragment end-mer CSVs (cf1-5 Single, ec1-5 Co-extraction,
ev1-5 Exosome), aggregates normalized 5' 4-mer frequencies into GC-biased
(GC count >= 3) and AT-biased (GC count <= 1) sums per sample, then compares
the three groups with Friedman + pairwise Wilcoxon on the matched sample index.

Usage:
    python3 scripts/analyze_task3_gc_bias_4mer.py \
      --endmer-dir data/endmer/task3 \
      --out outputs/gc_bias_4mer
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import friedmanchisquare, rankdata, wilcoxon  # noqa: E402

GROUPS = ["cf", "ec", "ev"]
GROUP_LABELS = {"cf": "Single (cf)", "ec": "Co-extract (ec)", "ev": "Exosome (ev)"}
GROUP_COLORS = {"cf": "#2F6B9A", "ec": "#D47A34", "ev": "#5B9E5B"}
METRIC = "normalized_5prime"
EXPECTED_SAMPLES = [f"{g}{i}" for g in GROUPS for i in range(1, 6)]  # cf1..ev5


def gc_count(mer: str) -> int:
    return sum(1 for c in str(mer) if c in "GCgc")


def derive_group(sample: str) -> str:
    s = str(sample).lower()
    for g in GROUPS:
        if s.startswith(g):
            return g
    return "unknown"


def derive_index(sample: str) -> str:
    m = re.search(r"(?:cf|ec|ev)(\d+)", str(sample).lower())
    return m.group(1) if m else "0"


def rank_biserial(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x) & (x != 0)]
    if len(x) == 0:
        return 0.0
    ranks = rankdata(np.abs(x), method="average")
    pos = ranks[x > 0].sum()
    neg = ranks[x < 0].sum()
    return float((pos - neg) / (pos + neg))


def bootstrap_median_delta(a: np.ndarray, b: np.ndarray, n_boot: int = 5000, seed: int = 42) -> tuple[float, float]:
    if len(a) != len(b):
        raise ValueError("paired arrays must have the same length")
    if len(a) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(a), size=(n_boot, len(a)), replace=True)
    boots = np.median(b[indices] - a[indices], axis=1)
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
    adjusted = np.full_like(p_values, np.nan)
    valid = np.isfinite(p_values)
    if not valid.any():
        return adjusted
    valid_values = p_values[valid]
    order = np.argsort(valid_values)
    ranked = valid_values[order]
    m = len(ranked)
    ranked_adjusted = ranked * m / np.arange(1, m + 1)
    ranked_adjusted = np.minimum.accumulate(ranked_adjusted[::-1])[::-1]
    restored = np.empty_like(ranked_adjusted)
    restored[order] = np.clip(ranked_adjusted, 0.0, 1.0)
    adjusted[valid] = restored
    return adjusted


def run_friedman_pairwise(cf: np.ndarray, ec: np.ndarray, ev: np.ndarray) -> dict:
    """Friedman test + pairwise Wilcoxon on matched arrays."""
    n = len(cf)
    result = {"n": n}
    if n >= 3:
        try:
            stat, p = friedmanchisquare(cf, ec, ev)
            result["friedman_stat"] = float(stat)
            result["friedman_p"] = float(p)
        except Exception:
            result["friedman_stat"] = float("nan")
            result["friedman_p"] = float("nan")
    else:
        result["friedman_stat"] = float("nan")
        result["friedman_p"] = float("nan")

    for name_a, a, name_b, b in [
        ("cf", cf, "ec", ec), ("cf", cf, "ev", ev), ("ec", ec, "ev", ev)
    ]:
        delta = b - a
        if n >= 3 and np.any(delta != 0):
            try:
                w_stat, w_p = wilcoxon(delta, zero_method="wilcox", alternative="two-sided")
            except Exception:
                w_stat, w_p = float("nan"), float("nan")
        else:
            w_stat, w_p = float("nan"), float("nan")
        rb = rank_biserial(delta)
        ci_lo, ci_hi = bootstrap_median_delta(a, b)
        result[f"{name_a}_vs_{name_b}_wilcoxon"] = float(w_stat)
        result[f"{name_a}_vs_{name_b}_p"] = float(w_p)
        result[f"{name_a}_vs_{name_b}_rank_biserial"] = float(rb)
        result[f"{name_a}_vs_{name_b}_delta_median"] = float(np.median(delta))
        result[f"{name_a}_vs_{name_b}_ci_low"] = ci_lo
        result[f"{name_a}_vs_{name_b}_ci_high"] = ci_hi

    return result


def load_matrix(endmer_dir: Path) -> pd.DataFrame:
    """Load the 15 requested end-mer CSVs into a sample×mer matrix of METRIC values."""
    rows: dict[str, pd.Series] = {}
    missing: list[str] = []
    for sample in EXPECTED_SAMPLES:
        csv = endmer_dir / f"{sample}.recal.end_mer.csv"
        if not csv.exists():
            missing.append(sample)
            continue
        df = pd.read_csv(csv)
        if {"mer", METRIC} - set(df.columns):
            raise ValueError(f"{csv} missing columns mer/{METRIC}")
        rows[sample] = df.set_index("mer")[METRIC].astype(float)

    if missing:
        raise FileNotFoundError(f"Missing end-mer CSVs: {', '.join(missing)}")
    matrix = pd.DataFrame(rows).T.loc[EXPECTED_SAMPLES].fillna(0.0)
    matrix.columns.name = "mer"
    return matrix


def aggregate_biased_frequencies(matrix: pd.DataFrame) -> pd.DataFrame:
    """Per-sample GC-biased (GC>=3) and AT-biased (GC<=1) normalized 4-mer sums."""
    mer_gc = pd.Series({mer: gc_count(str(mer)) for mer in matrix.columns})
    gc_cols = mer_gc[mer_gc >= 3].index
    at_cols = mer_gc[mer_gc <= 1].index

    out = pd.DataFrame({
        "sample": matrix.index,
        "group": matrix.index.map(derive_group),
        "bio_idx": matrix.index.map(derive_index),
        "gc_biased_freq": matrix.loc[:, gc_cols].sum(axis=1),
        "at_biased_freq": matrix.loc[:, at_cols].sum(axis=1),
    })
    unexpected = sorted(set(out["group"]) - set(GROUPS))
    if unexpected:
        raise RuntimeError(f"Unexpected groups: {unexpected}")
    return out.reset_index(drop=True)


def run_comparison(freq_df: pd.DataFrame, metric: str, out_dir: Path) -> pd.DataFrame:
    wide = freq_df.pivot_table(index="bio_idx", columns="group", values=metric, aggfunc="first")
    common = wide.dropna()
    if len(common) < 3:
        return pd.DataFrame()
    cf = common["cf"].to_numpy()
    ec = common["ec"].to_numpy()
    ev = common["ev"].to_numpy()
    res = run_friedman_pairwise(cf, ec, ev)
    res["metric"] = metric
    res["cf_median"] = float(np.median(cf))
    res["ec_median"] = float(np.median(ec))
    res["ev_median"] = float(np.median(ev))
    results = pd.DataFrame([res])
    results["friedman_p_BH"] = benjamini_hochberg(results["friedman_p"].to_numpy())
    return results


def plot_bias_boxplot(freq_df: pd.DataFrame, out_dir: Path) -> None:
    """Two-panel boxplot (GC-biased / AT-biased) with paired points + brackets."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharey=False)
    panels = [
        ("gc_biased_freq", "5' End: GC-biased 4-mer Frequency (GC\u22653)"),
        ("at_biased_freq", "5' End: AT-biased 4-mer Frequency (GC\u22641)"),
    ]
    positions = np.arange(1, len(GROUPS) + 1)

    for ax, (value_col, title) in zip(axes, panels, strict=True):
        data = [freq_df[freq_df["group"] == g][value_col].dropna().to_numpy() for g in GROUPS]
        labels = [f"{GROUP_LABELS[g]}\n(n={len(v)})" for g, v in zip(GROUPS, data, strict=True)]
        colors = [GROUP_COLORS[g] for g in GROUPS]

        bp = ax.boxplot(data, positions=positions, tick_labels=labels, patch_artist=True,
                        medianprops={"color": "black", "linewidth": 1.2}, showfliers=False)
        for patch, color in zip(bp["boxes"], colors, strict=True):
            patch.set_facecolor(color)
            patch.set_alpha(0.55)

        # paired points: same bio_idx connected across groups
        rng = np.random.default_rng(42)
        wide = freq_df.pivot_table(index="bio_idx", columns="group", values=value_col, aggfunc="first").dropna()
        for _, row in wide.iterrows():
            x = positions + rng.uniform(-0.06, 0.06, len(positions))
            ax.plot(x, [row[g] for g in GROUPS], color="#888888", alpha=0.18, linewidth=0.6, zorder=2)
        for i, (g, vals) in enumerate(zip(GROUPS, data, strict=True)):
            if len(vals) == 0:
                continue
            jitter = rng.uniform(-0.05, 0.05, len(vals))
            ax.scatter(np.full(len(vals), i + 1) + jitter, vals, s=26, alpha=0.7,
                       color=colors[i], edgecolors="white", linewidths=0.3, zorder=3)

        # Friedman p annotation
        wide_vals = [freq_df[freq_df["group"] == g][value_col].dropna().to_numpy() for g in GROUPS]
        common_len = min(len(v) for v in wide_vals)
        if common_len >= 3:
            stat, p = friedmanchisquare(*[v[:common_len] for v in wide_vals])
            all_vals = np.concatenate(wide_vals)
            span = max(all_vals.max() - all_vals.min(), 1e-6)
            ax.text(0.5, 0.96, f"Friedman p={p:.4f}", transform=ax.transAxes,
                    ha="center", va="top", fontsize=9, fontweight="bold")

        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylabel("Normalized frequency sum")
        ax.grid(axis="y", alpha=0.22, linewidth=0.6)
        ax.tick_params(axis="x", labelrotation=15)

    fig.tight_layout()
    fig.savefig(out_dir / "gc_bias_4mer_box.png", dpi=220, bbox_inches="tight")
    fig.savefig(out_dir / "gc_bias_4mer_box.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_paired_4mer_bias(freq_df: pd.DataFrame, out_dir: Path) -> None:
    """Paired comparisons: cf vs ec, cf vs ev, ec vs ev on matched sample index."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharey=False)
    panels = [
        ("gc_biased_freq", "Paired GC-biased 4-mer Frequency"),
        ("at_biased_freq", "Paired AT-biased 4-mer Frequency"),
    ]
    pairs = [("cf", "ec", "Single", "Co-extract"), ("cf", "ev", "Single", "Exosome"),
             ("ec", "ev", "Co-extract", "Exosome")]
    rng = np.random.default_rng(7)

    for ax, (value_col, title) in zip(axes, panels, strict=True):
        wide = freq_df.pivot_table(index="bio_idx", columns="group", values=value_col, aggfunc="first").dropna()
        for i, (a, b, a_lab, b_lab) in enumerate(pairs):
            if a not in wide.columns or b not in wide.columns:
                continue
            a_vals = wide[a].to_numpy()
            b_vals = wide[b].to_numpy()
            pos_a, pos_b = 1 + i * 3, 2 + i * 3
            ax.boxplot([a_vals, b_vals], positions=[pos_a, pos_b], widths=0.5,
                       patch_artist=True, medianprops={"color": "black", "linewidth": 1.0},
                       showfliers=False)
            for patch, g in zip(ax.patches[-2:], [a, b], strict=True):
                patch.set_facecolor(GROUP_COLORS[g])
                patch.set_alpha(0.55)
            jitter = rng.uniform(-0.06, 0.06, len(wide))
            for idx, row in wide.iterrows():
                ax.plot([pos_a, pos_b], [row[a], row[b]], color="#888888", alpha=0.2,
                        linewidth=0.6, zorder=2)
            ax.scatter(np.full(len(wide), pos_a) + jitter, a_vals, s=20, alpha=0.7, color=GROUP_COLORS[a], zorder=3)
            ax.scatter(np.full(len(wide), pos_b) + jitter, b_vals, s=20, alpha=0.7, color=GROUP_COLORS[b], zorder=3)
            # wilcoxon
            delta = b_vals - a_vals
            if np.any(delta != 0):
                _, p = wilcoxon(delta, zero_method="wilcox", alternative="two-sided")
                all_vals = np.concatenate([a_vals, b_vals])
                span = max(all_vals.max() - all_vals.min(), 1e-6)
                y = all_vals.max() + span * 0.06
                cap = span * 0.015
                ax.plot([pos_a, pos_a, pos_b, pos_b], [y - cap, y, y, y - cap],
                        color="#333333", linewidth=0.9, zorder=4)
                ax.text((pos_a + pos_b) / 2, y + cap, f"p={p:.3f}", ha="center", va="bottom",
                        fontsize=8, fontweight="bold")
        ax.set_xticks([1.5, 4.5, 7.5])
        ax.set_xticklabels([f"{a_lab} vs {b_lab}" for a, b, a_lab, b_lab in pairs], fontsize=9)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylabel("Normalized frequency sum")
        ax.grid(axis="y", alpha=0.22, linewidth=0.6)

    fig.tight_layout()
    fig.savefig(out_dir / "gc_bias_4mer_paired.png", dpi=220, bbox_inches="tight")
    fig.savefig(out_dir / "gc_bias_4mer_paired.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--endmer-dir", type=Path, required=True,
                    help="Task3 end_mer2 directory with *.recal.end_mer.csv files")
    ap.add_argument("--out", type=Path, required=True, help="Output directory")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    matrix = load_matrix(args.endmer_dir)
    freq_df = aggregate_biased_frequencies(matrix)
    freq_df[["sample", "group", "bio_idx", "gc_biased_freq", "at_biased_freq"]].to_csv(
        args.out / "gc_bias_4mer_by_sample.tsv", sep="\t", index=False)

    comparisons = []
    for metric in ["gc_biased_freq", "at_biased_freq"]:
        res = run_comparison(freq_df, metric, args.out)
        if not res.empty:
            comparisons.append(res)
    if comparisons:
        results = pd.concat(comparisons, ignore_index=True)
        results.to_csv(args.out / "gc_bias_4mer_comparison.tsv", sep="\t", index=False)
        print(results.to_string(index=False), flush=True)

    plot_bias_boxplot(freq_df, args.out)
    plot_paired_4mer_bias(freq_df, args.out)

    print(f"\nSaved outputs to: {args.out}", flush=True)


if __name__ == "__main__":
    main()
