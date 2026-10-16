#!/usr/bin/env python3
"""Three-group paired comparison for Task3: cf (Single) vs ec (Co-extraction) vs ev (Exosome).

Same biological sample index across groups → Friedman test + pairwise Wilcoxon signed-rank.
Analyzes cCRE matched log(OR), TSS center/flank ratio, and TSS coverage NDR depth.

Usage:
    python3 scripts/analyze_task3_three_group.py \
      --root /your/task3_root \
      --out  outputs/three_group_comparison
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import friedmanchisquare, rankdata, wilcoxon

GROUPS = ["cf", "ec", "ev"]
GROUP_LABELS = {"cf": "Single (cf)", "ec": "Co-extract (ec)", "ev": "Exosome (ev)"}
GROUP_COLORS = {"cf": "#2F6B9A", "ec": "#D47A34", "ev": "#5B9E5B"}


def derive_group(analysis_id: str) -> str:
    """cf/ec/ev from analysis_id like 'task3__cf1'."""
    aid = analysis_id.lower()
    for g in GROUPS:
        if f"__{g}" in aid:
            return g
    return "unknown"


def derive_index(analysis_id: str) -> str:
    """Extract numeric index: task3__cf3 → '3'."""
    import re
    m = re.search(r"__(?:cf|ec|ev)(\d+)", analysis_id.lower())
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


def paired_analysis_pivot(df: pd.DataFrame, group_col: str, index_col: str, value_col: str) -> dict:
    """Pivot to wide format: rows=index, cols=groups, values=value. Return dict of arrays."""
    wide = df.pivot_table(index=index_col, columns=group_col, values=value_col, aggfunc="first")
    result = {}
    for g in GROUPS:
        if g in wide.columns:
            result[g] = wide[g].dropna().to_numpy()
    # Only keep indices present in all groups
    common_idx = wide.dropna().index
    if len(common_idx) >= 2:
        result["_common_idx"] = common_idx.to_numpy()
        for g in GROUPS:
            if g in wide.columns:
                result[f"_{g}_paired"] = wide.loc[common_idx, g].to_numpy()
    return result

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

    # Pairwise Wilcoxon
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


# ============================================================
# cCRE analysis
# ============================================================

def analyze_ccre(ccre_dir: Path, out_dir: Path) -> pd.DataFrame:
    """Per-sample median log(OR) per feature×gc_definition×length_bin, compare 3 groups."""
    f = ccre_dir / "final" / "matched_or_summary.tsv"
    if not f.exists():
        print("ERROR: matched_or_summary.tsv not found", file=sys.stderr)
        return pd.DataFrame()

    df = pd.read_csv(f, sep="\t")
    df["log_or"] = np.log(df["or"].astype(float))
    df["group"] = df["analysis_id"].apply(derive_group)
    df["bio_idx"] = df["analysis_id"].apply(derive_index)

    # Per-sample median log(OR) within each stratum
    strata = ["feature", "gc_definition", "local_gc_control", "length_bin"]
    agg = df.groupby(["group", "bio_idx", "analysis_id"] + strata)["log_or"].median().reset_index()

    records = []
    for stratum_vals, grp in agg.groupby(strata):
        entry = dict(zip(strata, stratum_vals))
        # Pivot: bio_idx × group
        wide = grp.pivot_table(index="bio_idx", columns="group", values="log_or", aggfunc="first")
        common = wide.dropna()
        if len(common) < 3:
            continue
        cf = common["cf"].to_numpy() if "cf" in common else None
        ec = common["ec"].to_numpy() if "ec" in common else None
        ev = common["ev"].to_numpy() if "ev" in common else None
        if cf is None or ec is None or ev is None:
            continue
        res = run_friedman_pairwise(cf, ec, ev)
        res.update(entry)
        res["cf_median"] = float(np.median(cf))
        res["ec_median"] = float(np.median(ec))
        res["ev_median"] = float(np.median(ev))
        records.append(res)

    results = pd.DataFrame(records)
    if "friedman_p" in results.columns:
        results["friedman_p_BH"] = benjamini_hochberg(results["friedman_p"].to_numpy())

    results.to_csv(out_dir / "ccre_three_group_comparison.tsv", sep="\t", index=False)
    print(f"cCRE: {len(results)} strata tested", flush=True)
    return results


# ============================================================
# TSS center/flank analysis
# ============================================================

def analyze_tss(tss_dir: Path, out_dir: Path) -> pd.DataFrame:
    f = tss_dir / "tss_center_flank_by_sample.tsv"
    if not f.exists():
        print("ERROR: tss_center_flank_by_sample.tsv not found", file=sys.stderr)
        return pd.DataFrame()

    df = pd.read_csv(f, sep="\t")
    df["group"] = df["analysis_id"].apply(derive_group)
    df["bio_idx"] = df["analysis_id"].apply(derive_index)

    strata = ["tss_type", "gc_class", "length_bin"]
    records = []
    for stratum_vals, grp in df.groupby(strata):
        entry = dict(zip(strata, stratum_vals))
        wide = grp.pivot_table(index="bio_idx", columns="group", values="center_flank_ratio", aggfunc="first")
        common = wide.dropna()
        if len(common) < 3:
            continue
        cf = common["cf"].to_numpy() if "cf" in common else None
        ec = common["ec"].to_numpy() if "ec" in common else None
        ev = common["ev"].to_numpy() if "ev" in common else None
        if cf is None or ec is None or ev is None:
            continue
        res = run_friedman_pairwise(cf, ec, ev)
        res.update(entry)
        res["cf_median"] = float(np.median(cf))
        res["ec_median"] = float(np.median(ec))
        res["ev_median"] = float(np.median(ev))
        records.append(res)

    results = pd.DataFrame(records)
    if "friedman_p" in results.columns:
        results["friedman_p_BH"] = benjamini_hochberg(results["friedman_p"].to_numpy())

    results.to_csv(out_dir / "tss_three_group_comparison.tsv", sep="\t", index=False)
    print(f"TSS: {len(results)} strata tested", flush=True)
    return results


# ============================================================
# Coverage NDR depth
# ============================================================

def analyze_coverage(cov_dir: Path, out_dir: Path) -> pd.DataFrame | None:
    per_sample_dir = cov_dir / "per_sample"
    if not per_sample_dir.exists():
        print("WARN: coverage per_sample dir not found")
        return None

    records = []
    for sample_dir in sorted(per_sample_dir.iterdir()):
        aid = sample_dir.name
        cov_file = sample_dir / "coverage_summary.tsv"
        if not cov_file.exists():
            continue
        group = derive_group(aid)
        if group == "unknown":
            continue
        idx = derive_index(aid)
        df = pd.read_csv(cov_file, sep="\t")
        center_mask = (df["position"] >= -500) & (df["position"] <= 500)
        flank_mask = ((df["position"] >= -2000) & (df["position"] <= -1500)) | \
                     ((df["position"] >= 1500) & (df["position"] <= 2000))
        for col in ["coverage_all", "coverage_160_180"]:
            if col not in df.columns:
                continue
            center = df.loc[center_mask, col].mean()
            flank = df.loc[flank_mask, col].mean()
            ratio = center / flank if flank > 0 else float("nan")
            metric = f"ndr_ratio_{col.replace('coverage_', '')}"
            records.append({"analysis_id": aid, "group": group, "bio_idx": idx, "metric": metric, "value": ratio})

    if not records:
        return None
    cov_df = pd.DataFrame(records)
    # Aggregate coverage summary for reuse
    summary = cov_df.pivot_table(index=["analysis_id", "group", "bio_idx"],
                                  columns="metric", values="value", aggfunc="first").reset_index()
    summary.to_csv(cov_dir / "coverage_summary.tsv", sep="\t", index=False)

    results_list = []
    for metric, grp in cov_df.groupby("metric"):
        wide = grp.pivot_table(index="bio_idx", columns="group", values="value", aggfunc="first")
        common = wide.dropna()
        if len(common) < 3:
            continue
        cf = common["cf"].to_numpy()
        ec = common["ec"].to_numpy()
        ev = common["ev"].to_numpy()
        res = run_friedman_pairwise(cf, ec, ev)
        res["metric"] = metric
        res["cf_median"] = float(np.median(cf))
        res["ec_median"] = float(np.median(ec))
        res["ev_median"] = float(np.median(ev))
        results_list.append(res)

    if not results_list:
        return None
    results = pd.DataFrame(results_list)
    results["friedman_p_BH"] = benjamini_hochberg(results["friedman_p"].to_numpy())
    results.to_csv(out_dir / "coverage_three_group_comparison.tsv", sep="\t", index=False)
    print(f"Coverage: {len(results)} metrics tested", flush=True)
    return results


# ============================================================
# Plots
# ============================================================

def plot_ccre_box(ccre_dir: Path, results: pd.DataFrame, out_dir: Path) -> None:
    """Box plot of per-sample log(OR) for top strata."""
    if results.empty:
        return
    f = ccre_dir / "final" / "matched_or_summary.tsv"
    df = pd.read_csv(f, sep="\t")
    df["log_or"] = np.log(df["or"].astype(float))
    df["group"] = df["analysis_id"].apply(derive_group)
    df["bio_idx"] = df["analysis_id"].apply(derive_index)

    # Pick top 6 most significant strata for ccre_all
    primary = results[(results["feature"] == "ccre_all") & (results["gc_definition"] == "q90")].copy()
    primary = primary.sort_values("friedman_p").head(6)

    n_plots = len(primary)
    if n_plots == 0:
        return
    fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 5), squeeze=False)
    for ax_idx, (_, row) in enumerate(primary.iterrows()):
        ax = axes[0, ax_idx]
        subset = df[(df["feature"] == row["feature"]) &
                    (df["gc_definition"] == row["gc_definition"]) &
                    (df["local_gc_control"] == row["local_gc_control"]) &
                    (df["length_bin"] == row["length_bin"])]
        per_sample = subset.groupby(["group", "bio_idx", "analysis_id"])["log_or"].median().reset_index()
        data = [per_sample[per_sample["group"] == g]["log_or"].dropna().to_numpy() for g in GROUPS]
        bp = ax.boxplot(data, tick_labels=[GROUP_LABELS[g] for g in GROUPS], patch_artist=True, widths=0.5)
        for patch, g in zip(bp["boxes"], GROUPS):
            patch.set_facecolor(GROUP_COLORS[g])
            patch.set_alpha(0.6)
        title = f"{row['length_bin']}\n{row['local_gc_control']}"
        ax.set_title(title, fontsize=9)
        ax.set_ylabel("log(OR)")
        p = row.get("friedman_p", 1.0)
        ax.text(0.5, 0.95, f"Friedman p={p:.4f}", transform=ax.transAxes, ha="center", va="top", fontsize=8)
    fig.suptitle("cCRE enrichment (q90, ccre_all): cf vs ec vs ev", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_dir / "ccre_three_group_box.png", dpi=150, bbox_inches="tight")
    fig.savefig(out_dir / "ccre_three_group_box.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_tss_box(tss_dir: Path, results: pd.DataFrame, out_dir: Path) -> None:
    """Box plot of center/flank ratio for q90_high strata."""
    if results.empty:
        return
    df = pd.read_csv(tss_dir / "tss_center_flank_by_sample.tsv", sep="\t")
    df["group"] = df["analysis_id"].apply(derive_group)

    primary = results[results["gc_class"] == "q90_high"].copy()
    primary = primary.sort_values("friedman_p")

    n_plots = min(len(primary), 6)
    if n_plots == 0:
        return
    fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 5), squeeze=False)
    for ax_idx in range(n_plots):
        row = primary.iloc[ax_idx]
        ax = axes[0, ax_idx]
        subset = df[(df["tss_type"] == row["tss_type"]) &
                    (df["gc_class"] == row["gc_class"]) &
                    (df["length_bin"] == row["length_bin"])]
        data = [subset[subset["group"] == g]["center_flank_ratio"].dropna().to_numpy() for g in GROUPS]
        bp = ax.boxplot(data, tick_labels=[GROUP_LABELS[g] for g in GROUPS], patch_artist=True, widths=0.5)
        for patch, g in zip(bp["boxes"], GROUPS):
            patch.set_facecolor(GROUP_COLORS[g])
            patch.set_alpha(0.6)
        title = f"{row['tss_type']} | {row['length_bin']}"
        ax.set_title(title, fontsize=9)
        ax.set_ylabel("Center/Flank Ratio")
        p = row.get("friedman_p", 1.0)
        ax.text(0.5, 0.95, f"Friedman p={p:.4f}", transform=ax.transAxes, ha="center", va="top", fontsize=8)
    fig.suptitle("TSS center/flank ratio (q90_high): cf vs ec vs ev", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_dir / "tss_three_group_box.png", dpi=150, bbox_inches="tight")
    fig.savefig(out_dir / "tss_three_group_box.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_tss_metaplot(tss_dir: Path, out_dir: Path) -> None:
    """Three-group overlay metaplot from per-sample RPM profiles."""
    f = tss_dir / "tss_rpm_by_sample.tsv"
    if not f.exists():
        return
    df = pd.read_csv(f, sep="\t")
    df["group"] = df["analysis_id"].apply(derive_group)
    df = df[df["group"].isin(GROUPS)]
    primary = df[(df["tss_type"] == "all") & (df["gc_class"] == "q90_high")]
    if primary.empty:
        return
    summary = primary.groupby(["group", "length_bin", "position"])["rpm"].agg(["mean", "sem"]).reset_index()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax_idx, length_bin in enumerate(["short_90_150", "mono_151_220", "long_221_500"]):
        ax = axes[ax_idx]
        for group in GROUPS:
            subset = summary[(summary["group"] == group) & (summary["length_bin"] == length_bin)].sort_values("position")
            if subset.empty:
                continue
            ax.plot(subset["position"], subset["mean"], label=GROUP_LABELS[group],
                    color=GROUP_COLORS[group], linewidth=1.5)
            ax.fill_between(subset["position"], subset["mean"] - subset["sem"], subset["mean"] + subset["sem"],
                            alpha=0.15, color=GROUP_COLORS[group])
        ax.set_title(length_bin, fontsize=11)
        ax.set_xlabel("Position from TSS (bp)")
        ax.set_ylabel("Mean RPM")
        ax.axvline(0, color="grey", linestyle="--", alpha=0.5)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=8)
    fig.suptitle("TSS metaplot (q90_high): cf vs ec vs ev", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_dir / "tss_metaplot_three_group.png", dpi=150, bbox_inches="tight")
    fig.savefig(out_dir / "tss_metaplot_three_group.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_coverage_overlay(cov_dir: Path, out_dir: Path) -> None:
    """Three-group overlay of per-sample flank-normalized TSS coverage."""
    per_sample_dir = cov_dir / "per_sample"
    if not per_sample_dir.exists():
        return
    rows = []
    for sample_dir in sorted(per_sample_dir.iterdir()):
        group = derive_group(sample_dir.name)
        f = sample_dir / "coverage_summary.tsv"
        if group not in GROUPS or not f.exists():
            continue
        df = pd.read_csv(f, sep="\t")
        flank_mask = ((df["position"] >= -2000) & (df["position"] <= -1500)) | \
                     ((df["position"] >= 1500) & (df["position"] <= 2000))
        for column, label in [("coverage_160_180", "160-180bp"), ("coverage_all", "All fragments")]:
            flank_mean = df.loc[flank_mask, column].mean()
            if flank_mean <= 0:
                continue
            rows.extend({"group": group, "position": int(position), "track": label, "normalized": float(value)}
                        for position, value in zip(df["position"], df[column] / flank_mean, strict=True))
    if not rows:
        return
    tracks = pd.DataFrame(rows).groupby(["group", "track", "position"])["normalized"].agg(["mean", "sem"]).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    for ax, label in zip(axes, ["160-180bp", "All fragments"], strict=True):
        for group in GROUPS:
            subset = tracks[(tracks["group"] == group) & (tracks["track"] == label)].sort_values("position")
            if subset.empty:
                continue
            ax.plot(subset["position"], subset["mean"], label=GROUP_LABELS[group], color=GROUP_COLORS[group])
            ax.fill_between(subset["position"], subset["mean"] - subset["sem"], subset["mean"] + subset["sem"],
                            alpha=0.15, color=GROUP_COLORS[group])
        ax.set_xlabel("Position from TSS (bp)")
        ax.set_ylabel("Flank-normalized coverage")
        ax.axvline(0, color="grey", linestyle="--", alpha=0.5)
        ax.set_title(label)
        ax.set_xlim(-2000, 2000)
        ax.legend(fontsize=8)
    fig.suptitle("TSS coverage: cf vs ec vs ev", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_dir / "coverage_three_group_overlay.png", dpi=150, bbox_inches="tight")
    fig.savefig(out_dir / "coverage_three_group_overlay.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True,
                    help="Task3 output root (containing ccre/, tss/, tss_coverage/)")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output directory for comparison results")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print("=== cCRE three-group comparison ===", flush=True)
    try:
        ccre_res = analyze_ccre(args.root / "ccre", args.out)
        plot_ccre_box(args.root / "ccre", ccre_res, args.out)
    except Exception as e:
        print(f"cCRE analysis failed: {e}", file=sys.stderr)
        import traceback; traceback.print_exc()

    print("\n=== TSS center/flank three-group comparison ===", flush=True)
    try:
        tss_res = analyze_tss(args.root / "tss", args.out)
        plot_tss_box(args.root / "tss", tss_res, args.out)
        plot_tss_metaplot(args.root / "tss", args.out)
    except Exception as e:
        print(f"TSS analysis failed: {e}", file=sys.stderr)
        import traceback; traceback.print_exc()

    print("\n=== TSS coverage three-group comparison ===", flush=True)
    try:
        cov_res = analyze_coverage(args.root / "tss_coverage", args.out)
        plot_coverage_overlay(args.root / "tss_coverage", args.out)
    except Exception as e:
        print(f"Coverage analysis failed: {e}", file=sys.stderr)
        import traceback; traceback.print_exc()

    print(f"\n=== Done. Outputs in {args.out} ===", flush=True)


if __name__ == "__main__":
    main()
