#!/usr/bin/env python3
"""Generate 5 publication-quality figures for the fragmentomics analysis.

Run on server:
  cd EGFR-co
  python3 scripts/generate_final_figures.py
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"]   = 42
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy.stats import wilcoxon
from fragmentomics_figure_counts import cohort_counts
from fragmentomics_disease_stats import (
    disease_heatmap_stats,
    gc_control_stats,
    source_consistency_stats,
)

import sys as _sys
from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import pathconfig as _pc

V2 = _pc.get("frag_v2_final_dir", "data/fragmentomics_genome_wide_v2/final")
OUT = _P("outputs/final_figures"); OUT.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

# ── Exclusion criteria ──
EXCLUDE = {f"H{i}.recal" for i in range(1, 11)} | {"SH7.recal", "SB1.recal", "B1.recal"}

def load(name):
    df = pd.read_csv(V2 / name, sep="\t")
    if "sample" in df.columns:
        df = df[~df["sample"].isin(EXCLUDE)].copy()
    return df

def complete_pair_ids(sample_table):
    arms_by_pair = sample_table.groupby("pair_id")["extraction"].agg(lambda values: set(values))
    return {str(pair_id) for pair_id, arms in arms_by_pair.items()
            if {"Co-extraction", "Single"}.issubset(arms)}

def keep_complete_pairs(df, pair_ids):
    if "pair_id" not in df.columns:
        return df
    return df[df["pair_id"].astype(str).isin(pair_ids)].copy()

# Load and enforce the confirmed sample policy before any paired summary.
# Strict 58-pair cohort: 45 Cancer + 13 Healthy (Post-op excluded).
thr = load("gc_thresholds.tsv")
COHORT = cohort_counts(thr, exclude=EXCLUDE)
COHORT["pair_ids"] = {str(pid) for pid in COHORT["pair_ids"]
                      if thr[thr["pair_id"].astype(str) == str(pid)]["disease"].iloc[0] != "Post-op"}
COHORT["n_pairs"] = len(COHORT["pair_ids"])
PAIR_IDS = COHORT["pair_ids"]
mh = keep_complete_pairs(load("matched_or_summary.tsv"), PAIR_IDS)
mh_d = keep_complete_pairs(load("matched_or_paired_delta.tsv"), PAIR_IDS)
ov = keep_complete_pairs(load("raw_overlap_summary.tsv"), PAIR_IDS)

# Feature order (biological → sequence)
FEAT_ORDER = ["ccre_all", "promoter_500bp_cpg", "promoter_500bp_non_cpg", "dnase_footprint",
              "cpg_all", "cpg_high_gc70"]
FEAT_LABELS = {
    "ccre_all": "cCRE (regulatory)",
    "promoter_500bp_cpg": "CpG promoter",
    "promoter_500bp_non_cpg": "Non-CpG promoter",
    "dnase_footprint": "DNase footprint",
    "cpg_all": "CpG island",
    "cpg_high_gc70": "CpG high-GC70",
}
LENGTH_BINS = ["short_90_150", "mono_151_220", "long_221_500"]
LB_LABELS = {"short_90_150": "Short\n(90-150bp)", "mono_151_220": "Mono\n(151-220bp)", "long_221_500": "Long\n(221-500bp)"}
METHOD_COLORS = {"Single": "#5B4B8A", "Co-extraction": "#C8537A"}
DISEASES = ["Cancer", "Healthy"]
METHODS = ["Co-extraction", "Single"]
DISEASE_PAIR_COUNTS = COHORT["pairs_by_disease"]


# ── Compute data tables ──
# Table 1: Δlog(OR) for flank_500bp, q90, all length bins
flank_d = mh_d[(mh_d["local_gc_control"] == "flank_500bp")]
table1 = {}
for feat in FEAT_ORDER:
    for lb in LENGTH_BINS:
        s = flank_d[(flank_d["gc_definition"] == "q90") & (flank_d["length_bin"] == lb) &
                    (flank_d["feature"] == feat) & flank_d["log_or_delta"].notna()]
        if len(s) < 5:
            continue
        _, p = wilcoxon(s["log_or_delta"])
        table1[(feat, lb)] = {"median": s["log_or_delta"].median(), "p": p, "n": len(s)}

# Table 2: Raw overlap delta (q90, all bins)
raw_delta = {}
for feat in FEAT_ORDER:
    for lb in LENGTH_BINS:
        d = []
        for pid in ov["pair_id"].unique():
            sub = ov[(ov["pair_id"] == pid) & (ov["length_bin"] == lb) &
                     (ov["gc_definition"] == "q90") & (ov["feature"] == feat)]
            co = sub[sub["extraction"] == "Co-extraction"]
            si = sub[sub["extraction"] == "Single"]
            if len(co) == 1 and len(si) == 1:
                d.append(co["high_overlap_fraction"].iloc[0] - si["high_overlap_fraction"].iloc[0])
        if len(d) < 5:
            continue
        _, p = wilcoxon(d)
        raw_delta[(feat, lb)] = {"median": np.median(d), "p": p, "n": len(d)}

# Per-pair OR data for paired dot plot
pair_or = {}
mh_pair = mh_d[(mh_d["local_gc_control"] == "flank_500bp") &
               (mh_d["gc_definition"] == "q90") &
               (mh_d["length_bin"] == "mono_151_220")]
for feat in FEAT_ORDER:
    s = mh_pair[(mh_pair["feature"] == feat) &
                mh_pair["co_or"].notna() & mh_pair["single_or"].notna()].copy()
    s = s[np.isfinite(s["co_or"]) & np.isfinite(s["single_or"]) &
          (s["co_or"] > 0) & (s["single_or"] > 0)]
    pair_or[feat] = pd.DataFrame({
        "pair_id": s["pair_id"].values,
        "co": s["co_or"].values,
        "si": s["single_or"].values,
        "log_co": np.log(s["co_or"].values),
        "log_si": np.log(s["single_or"].values),
        "disease": s["disease"].values,
    })

# Disease-stratified flank OR
mh_f = mh[mh["local_gc_control"] == "flank_500bp"]
mh_f2 = mh_f[(mh_f["gc_definition"] == "q90") & (mh_f["length_bin"] == "mono_151_220")]
disease_or = {}
for feat in ["ccre_all"]:
    for dis in ["Cancer", "Healthy", "Post-op"]:
        sub = mh_f2[(mh_f2["feature"] == feat) & (mh_f2["disease"] == dis)]
        sub = sub[sub["or"].notna() & (sub["or"] < float("inf"))]
        if len(sub) < 3:
            continue
        disease_or[(feat, dis)] = {"or_median": sub["or"].median(),
                                   "ci_low": sub["ci_low"].median(),
                                   "ci_high": sub["ci_high"].median(),
                                   "n_pairs": COHORT["pairs_by_disease"].get(dis, 0),
                                   "n_ors": len(sub),
                                   "or_values": sub["or"].values}

# GC shift by disease
gc_shift = {}
for lb in ["mono_151_220"]:
    for dis in ["Cancer", "Healthy", "Post-op"]:
        co_s = thr[(thr["extraction"]=="Co-extraction")&(thr["disease"]==dis)&(thr["length_bin"]==lb)].set_index("pair_id")["mean_gc"]
        si_s = thr[(thr["extraction"]=="Single")&(thr["disease"]==dis)&(thr["length_bin"]==lb)].set_index("pair_id")["mean_gc"]
        common = co_s.index.intersection(si_s.index)
        if len(common) > 0:
            deltas = (co_s.loc[common] - si_s.loc[common]).values
            gc_shift[(dis,)] = {"mean": np.mean(deltas), "std": np.std(deltas), "values": deltas,
                                "co_mean": co_s.loc[common].mean(), "si_mean": si_s.loc[common].mean()}

# ── Helper ──
def sig_stars(p):
    if p < 0.001: return "***"
    elif p < 0.01: return "**"
    elif p < 0.05: return "*"
    return "ns"

def p_label(p):
    if not np.isfinite(p):
        return "p=NA"
    if p < 1e-99:
        return "p<1e-99"
    return f"p={p:.1e}"


def p_label_compact(p):
    label = p_label(p)
    if label.startswith("p="):
        return label[2:]
    if label.startswith("p"):
        return label[1:]
    return label
def feat_color(feat):
    if feat in ["ccre_all", "promoter_500bp_cpg", "promoter_500bp_non_cpg", "dnase_footprint"]:
        return "#E45756"  # warm red for regulatory features
    else:
        return "#4C78A8"  # cool blue for sequence/GC features


# ═══════════════════════════════════════════════════════════════
# Figure 1: Diverging Bar Plot — cCRE↑ / CpG↓ Separation
# ═══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 6))
feats_mono = [f for f in FEAT_ORDER if (f, "mono_151_220") in table1]
y_pos = np.arange(len(feats_mono))
deltas = [table1[(f, "mono_151_220")]["median"] for f in feats_mono]
pvals = [table1[(f, "mono_151_220")]["p"] for f in feats_mono]
colors = [feat_color(f) for f in feats_mono]
labels = [FEAT_LABELS[f] for f in feats_mono]

bars = ax.barh(y_pos, deltas, color=colors, edgecolor="white", height=0.6)
ax.axvline(0, color="black", linewidth=0.8, linestyle="-")
ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=11)
ax.set_xlabel("Δ log(OR)  Co − Single  (flank_500bp, q90, mono)", fontsize=11)
ax.set_title("Figure 1. Regulatory-specific enrichment: cCRE↑ / CpG↓ separation\n"
             "Co-extraction shows higher specificity for regulatory regions, not just GC-rich regions",
             fontsize=12, fontweight="bold")
ax.invert_yaxis()

# Add exact p values
for i, (d, p) in enumerate(zip(deltas, pvals)):
    star = sig_stars(p)
    x_pos = d + 0.006 if d > 0 else d - 0.006
    ha = "left" if d > 0 else "right"
    ax.text(x_pos, i, f" {p_label(p)} {star}", ha=ha, va="center", fontsize=9, fontweight="bold")
reg_count = sum(1 for f in feats_mono if f in ["ccre_all", "promoter_500bp_cpg", "promoter_500bp_non_cpg", "dnase_footprint"])
ax.axhline(reg_count - 0.5, color="grey", linewidth=0.5, linestyle="--")

# Legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor="#E45756", label="Regulatory features"),
                   Patch(facecolor="#4C78A8", label="Sequence/GC features")]
ax.legend(handles=legend_elements, loc="lower right", fontsize=9)
ax.set_xlim(min(deltas) - 0.05, max(deltas) + 0.05)

plt.tight_layout()
plt.savefig(OUT / "fig1_diverging_bar_cCRE_up_CpG_down.png", dpi=200, bbox_inches="tight")
plt.savefig(OUT / "fig1_diverging_bar_cCRE_up_CpG_down.pdf", bbox_inches="tight")
print("Figure 1 saved", flush=True)


# ═══════════════════════════════════════════════════════════════
# Figure 2: Heatmap — Length Consistency
# ═══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9.5, 6.5))
feats_hm = [f for f in FEAT_ORDER if all((f, lb) in table1 for lb in LENGTH_BINS)]
data_hm = np.array([[table1[(f, lb)]["median"] for lb in LENGTH_BINS] for f in feats_hm])
pvals_hm = np.array([[table1[(f, lb)]["p"] for lb in LENGTH_BINS] for f in feats_hm])

from matplotlib.colors import TwoSlopeNorm
vmax = max(abs(data_hm.min()), data_hm.max())
norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
im = ax.imshow(data_hm, cmap="RdBu_r", norm=norm, aspect="auto")

ax.set_xticks(range(len(LENGTH_BINS)))
ax.set_xticklabels([LB_LABELS[lb] for lb in LENGTH_BINS], fontsize=10)
ax.set_yticks(range(len(feats_hm)))
ax.set_yticklabels([FEAT_LABELS[f] for f in feats_hm], fontsize=10)
ax.set_title("Figure 2. Length-stratified consistency of regulatory enrichment\n"
             "Δ log(OR) Co − Single (flank_500bp, q90)", fontsize=12, fontweight="bold")

# Add values and exact p values
for i in range(len(feats_hm)):
    for j in range(len(LENGTH_BINS)):
        val = data_hm[i, j]
        p = pvals_hm[i, j]
        text_color = "white" if abs(val) > vmax * 0.5 else "black"
        ax.text(j, i, f"{val:+.3f}\n{p_label_compact(p)}", ha="center", va="center",
                fontsize=8, color=text_color, fontweight="bold")

# Separator line
reg_count_hm = sum(1 for f in feats_hm if f in ["ccre_all", "promoter_500bp_cpg", "promoter_500bp_non_cpg", "dnase_footprint"])
ax.axhline(reg_count_hm - 0.5, color="white", linewidth=2, linestyle="-")

cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label("Δ log(OR)", fontsize=10)

plt.tight_layout()
plt.savefig(OUT / "fig2_heatmap_length_consistency.png", dpi=200, bbox_inches="tight")
plt.savefig(OUT / "fig2_heatmap_length_consistency.pdf", bbox_inches="tight")
print("Figure 2 saved", flush=True)


# ═══════════════════════════════════════════════════════════════
# Figure 3: Raw overlap vs GC-adjusted (side by side)
# ═══════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
feats_cmp = ["ccre_all", "promoter_500bp_cpg", "cpg_all"]
labels_cmp = [FEAT_LABELS[f] for f in feats_cmp]
y_pos = np.arange(len(feats_cmp))

# Left: raw overlap delta (mono)
raw_vals = [raw_delta.get((f, "mono_151_220"), {"median": 0})["median"] for f in feats_cmp]
raw_p = [raw_delta.get((f, "mono_151_220"), {"p": 1})["p"] for f in feats_cmp]
colors_raw = ["#E45756" if v > 0 else "#4C78A8" for v in raw_vals]
ax1.barh(y_pos, raw_vals, color=colors_raw, edgecolor="white", height=0.5)
ax1.axvline(0, color="black", linewidth=0.8)
ax1.set_yticks(y_pos)
ax1.set_yticklabels(labels_cmp, fontsize=10)
ax1.set_xlabel("Δ overlap fraction (pp)", fontsize=10)
ax1.set_title("A. Raw overlap difference\n(q90, mono, Co − Single)", fontsize=11)
ax1.invert_yaxis()
for i, (v, p) in enumerate(zip(raw_vals, raw_p)):
    star = sig_stars(p)
    ax1.text(v + 0.002, i, f" {star}", ha="left", va="center", fontsize=10, fontweight="bold")

# Right: GC-adjusted flank OR delta (mono)
adj_vals = [table1.get((f, "mono_151_220"), {"median": 0})["median"] for f in feats_cmp]
adj_p = [table1.get((f, "mono_151_220"), {"p": 1})["p"] for f in feats_cmp]
colors_adj = ["#E45756" if v > 0 else "#4C78A8" for v in adj_vals]
ax2.barh(y_pos, adj_vals, color=colors_adj, edgecolor="white", height=0.5)
ax2.axvline(0, color="black", linewidth=0.8)
ax2.set_yticks(y_pos)
ax2.set_yticklabels(labels_cmp, fontsize=10)
ax2.set_xlabel("Δ log(OR) (flank-adjusted)", fontsize=10)
ax2.set_title("B. GC-adjusted enrichment\n(flank_500bp, q90, mono, Co − Single)", fontsize=11)
ax2.invert_yaxis()
for i, (v, p) in enumerate(zip(adj_vals, adj_p)):
    star = sig_stars(p)
    xp = v + 0.005 if v > 0 else v - 0.005
    ha = "left" if v > 0 else "right"
    ax2.text(xp, i, f" {star}", ha=ha, va="center", fontsize=10, fontweight="bold")

fig.suptitle("Figure 3. Raw overlap vs GC-adjusted: Co-extraction specificity\n"
             "CpG island: raw overlap ↑ but GC-adjusted ↓ = GC composition, not regulatory preference",
             fontsize=12, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(OUT / "fig3_raw_vs_adjusted.png", dpi=200, bbox_inches="tight")
plt.savefig(OUT / "fig3_raw_vs_adjusted.pdf", bbox_inches="tight")
print("Figure 3 saved", flush=True)


# ═══════════════════════════════════════════════════════════════
# Figure 4: Paired dot plot
# ═══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(15, 6), sharey=False)
for ax, feat in zip(axes, ["ccre_all", "promoter_500bp_cpg", "cpg_high_gc70"]):
    df_p = pair_or.get(feat)
    if df_p is None or len(df_p) < 5:
        ax.set_title(FEAT_LABELS[feat])
        continue

    rng = np.random.default_rng(42)
    bp = ax.boxplot([df_p["log_si"], df_p["log_co"]], positions=[0, 1], widths=0.22,
                    patch_artist=True, showfliers=False, zorder=1,
                    medianprops={"color": "black", "linewidth": 1.0},
                    whiskerprops={"color": "#555555", "linewidth": 0.8},
                    capprops={"color": "#555555", "linewidth": 0.8})
    for patch, color in zip(bp["boxes"], [METHOD_COLORS["Single"], METHOD_COLORS["Co-extraction"]]):
        patch.set_facecolor(color)
        patch.set_alpha(0.30)
        patch.set_edgecolor("#555555")
        patch.set_linewidth(0.8)

    si_x = rng.uniform(-0.055, 0.055, len(df_p))
    co_x = 1 + rng.uniform(-0.055, 0.055, len(df_p))
    ax.scatter(si_x, df_p["log_si"], s=28, color=METHOD_COLORS["Single"], alpha=0.65,
               edgecolors="white", linewidth=0.3, zorder=3)
    ax.scatter(co_x, df_p["log_co"], s=28, color=METHOD_COLORS["Co-extraction"], alpha=0.65,
               edgecolors="white", linewidth=0.3, zorder=3)

    for _, row in df_p.iterrows():
        ax.plot([0, 1], [row["log_si"], row["log_co"]], color="grey", alpha=0.12, linewidth=0.45, zorder=2)

    # Median overlay
    si_med = df_p["log_si"].median()
    co_med = df_p["log_co"].median()
    ax.plot([0, 1], [si_med, co_med], color="#333333", linewidth=2, marker="D", markersize=8, zorder=4)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Single", "Co-extraction"], fontsize=11)
    ax.set_title(FEAT_LABELS[feat], fontsize=11)
    if feat == "ccre_all":
        ax.set_ylabel("log(OR) per pair (flank_500bp, q90, mono)", fontsize=10)

    # Wilcoxon p
    _, p = wilcoxon(df_p["log_co"] - df_p["log_si"])
    star = sig_stars(p)
    ax.text(0.5, ax.get_ylim()[1] - 0.05 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
            f"p = {p:.1e} {star}", ha="center", fontsize=10, fontweight="bold")

fig.suptitle("Figure 4. Paired comparison: Co vs Single log(OR) per pair\n"
             "Each grey line = one patient pair; points = pairs; boxes = IQR; dark diamond = median",
             fontsize=12, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(OUT / "fig4_paired_dot_plot.png", dpi=200, bbox_inches="tight")
plt.savefig(OUT / "fig4_paired_dot_plot.pdf", bbox_inches="tight")
print("Figure 4 saved", flush=True)


# ═══════════════════════════════════════════════════════════════
# Figure 5: Disease stratification
# ═══════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Panel A: cCRE flank OR by disease
diseases = ["Healthy", "Cancer", "Post-op"]
x_pos = np.arange(len(diseases))
or_medians = [disease_or.get(("ccre_all", d), {}).get("or_median", np.nan) for d in diseases]
ci_lows = [disease_or.get(("ccre_all", d), {}).get("ci_low", np.nan) for d in diseases]
ci_highs = [disease_or.get(("ccre_all", d), {}).get("ci_high", np.nan) for d in diseases]
count_labels = [f"{d}\n{disease_or.get(('ccre_all', d), {}).get('n_pairs', 0)} pairs\n{disease_or.get(('ccre_all', d), {}).get('n_ors', 0)} ORs" for d in diseases]

# Also get individual OR values for boxplot overlay
or_values_by_dis = [disease_or.get(("ccre_all", d), {}).get("or_values", []) for d in diseases]

# Boxplot
bp = ax1.boxplot(or_values_by_dis, positions=x_pos, widths=0.4, patch_artist=True,
                 showmeans=False, showfliers=False)
colors_dis = ["#82b366", "#b85450", "#9673a6"]
for patch, color in zip(bp["boxes"], colors_dis):
    patch.set_facecolor(color)
    patch.set_alpha(0.3)


# Scatter overlay
rng5 = np.random.default_rng(42)
for i, vals in enumerate(or_values_by_dis):
    jx = i + rng5.uniform(-0.1, 0.1, len(vals))
    ax1.scatter(jx, vals, s=18, color=colors_dis[i], alpha=0.55,
                edgecolors="white", linewidth=0.3, zorder=3)

# Point-range for median
ax1.scatter(x_pos, or_medians, color=colors_dis, s=80, zorder=5, edgecolors="black")
for i in range(len(diseases)):
    if not np.isnan(or_medians[i]):
        ax1.plot([x_pos[i], x_pos[i]], [ci_lows[i], ci_highs[i]], color=colors_dis[i], linewidth=2)

ax1.set_xticks(x_pos)
ax1.set_xticklabels(count_labels, fontsize=10)
ax1.set_ylabel("cCRE MH-OR (flank_500bp, q90, mono)", fontsize=10)
ax1.set_title("A. cCRE enrichment by disease", fontsize=11)
ax1.axhline(1, color="grey", linestyle="--", linewidth=0.5)

# Panel B: GC shift by disease
gc_diseases = ["Cancer", "Healthy", "Post-op"]
gc_deltas = [gc_shift.get((d,), {}).get("values", []) for d in gc_diseases]
gc_colors = ["#b85450", "#82b366", "#9673a6"]

bp2 = ax2.boxplot(gc_deltas, positions=np.arange(len(gc_diseases)), widths=0.4, patch_artist=True)
for patch, color in zip(bp2["boxes"], gc_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.3)


# Scatter overlay
rng5b = np.random.default_rng(43)
for i, vals in enumerate(gc_deltas):
    if len(vals):
        jx = i + rng5b.uniform(-0.1, 0.1, len(vals))
        ax2.scatter(jx, vals, s=18, color=gc_colors[i], alpha=0.55,
                    edgecolors="white", linewidth=0.3, zorder=3)

# Add mean
for i, vals in enumerate(gc_deltas):
    if len(vals):
        ax2.scatter(i, np.mean(vals), color=gc_colors[i], marker="D", s=50, zorder=5, edgecolors="black")
        ax2.text(i, np.mean(vals) + 0.002, f"+{np.mean(vals):.3f}", ha="center", fontsize=9, fontweight="bold")

ax2.set_xticks(np.arange(len(gc_diseases)))
ax2.set_xticklabels(gc_diseases, fontsize=11)
ax2.set_ylabel("Δ mean GC (Co − Single, mono)", fontsize=10)
ax2.set_title("B. GC shift by disease", fontsize=11)
ax2.axhline(0, color="grey", linestyle="--", linewidth=0.5)

fig.suptitle("Figure 5. Disease stratification: cCRE enrichment and GC shift\n"
             "Healthy shows highest cCRE specificity; Cancer shows largest GC shift",
             fontsize=12, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(OUT / "fig5_disease_stratification.png", dpi=200, bbox_inches="tight")
plt.savefig(OUT / "fig5_disease_stratification.pdf", bbox_inches="tight")

# ═══════════════════════════════════════════════════════════════════
# Figure 6: Paired boxplot across all 6 features (Co vs Single log OR)
# ═══════════════════════════════════════════════════════════════════
rng = np.random.default_rng(42)
n_feats = len(FEAT_ORDER)
n_cols = 3
n_rows = (n_feats + n_cols - 1) // n_cols
fig6, axes6 = plt.subplots(n_rows, n_cols, figsize=(5.0 * n_cols, 5.5 * n_rows),
                            sharey=False)
axes6 = np.asarray(axes6).flatten()
method_colors = [METHOD_COLORS["Single"], METHOD_COLORS["Co-extraction"]]
for idx, feat in enumerate(FEAT_ORDER):
    ax = axes6[idx]
    df_p = pair_or.get(feat)
    if df_p is None or len(df_p) < 3:
        ax.set_title(FEAT_LABELS[feat] + " (no data)")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Single", "Co-extraction"])
        continue
    si_v = df_p["log_si"].dropna().values
    co_v = df_p["log_co"].dropna().values
    bp = ax.boxplot([si_v, co_v], positions=[0, 1], widths=0.30,
                    patch_artist=True, showfliers=False, zorder=1,
                    medianprops={"color": "black", "linewidth": 1.2},
                    whiskerprops={"color": "#555555", "linewidth": 0.8},
                    capprops={"color": "#555555", "linewidth": 0.8})
    for patch, color in zip(bp["boxes"], method_colors):
        patch.set_facecolor(color); patch.set_alpha(0.32)
        patch.set_edgecolor("#555555"); patch.set_linewidth(0.8)
    j_si = rng.uniform(-0.07, 0.07, len(df_p))
    j_co = 1 + rng.uniform(-0.07, 0.07, len(df_p))
    ax.scatter(j_si, si_v, s=22, color=method_colors[0], alpha=0.65,
               edgecolors="white", linewidth=0.3, zorder=3)
    ax.scatter(j_co, co_v, s=22, color=method_colors[1], alpha=0.65,
               edgecolors="white", linewidth=0.3, zorder=3)
    for _, row in df_p.iterrows():
        ax.plot([0, 1], [row["log_si"], row["log_co"]],
                color="grey", alpha=0.10, linewidth=0.4, zorder=2)
    si_med, co_med = np.median(si_v), np.median(co_v)
    ax.plot([0, 1], [si_med, co_med], color="#333333", linewidth=2,
            marker="D", markersize=7, zorder=4)
    try:
        _, p = wilcoxon(co_v - si_v)
        star = sig_stars(p)
        p_text = f"p={p:.1e} {star}"
    except ValueError:
        p_text = "p=n/a"
    ax.text(0.5, 0.96, f"n={len(df_p)}  {p_text}",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=9, fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "#999", "alpha": 0.85, "pad": 2})
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Single", "Co-extraction"], fontsize=10)
    ax.set_title(FEAT_LABELS[feat], fontsize=11)
    ax.set_ylabel("log(OR) per pair" if idx % n_cols == 0 else "")
    ax.grid(axis="y", alpha=0.20, linewidth=0.6)
for j in range(n_feats, n_rows * n_cols):
    axes6[j].set_visible(False)
fig6.suptitle("Figure 6. Paired boxplot: log(OR) per pair, Co vs Single, across 6 cCRE features\n"
              "flank_500bp local GC control, q90 high-GC, mono_151_220 fragment size\n"
              "n=pair count; Wilcoxon signed-rank on log(co) − log(single)",
              fontsize=12, fontweight="bold", y=1.01)
plt.tight_layout()
fig6.savefig(OUT / "fig6_paired_boxplot_all_features.png", dpi=200, bbox_inches="tight")
fig6.savefig(OUT / "fig6_paired_boxplot_all_features.pdf", bbox_inches="tight")
plt.close(fig6)
print("Figure 6 saved (6-feature paired boxplot)", flush=True)

# ═══════════════════════════════════════════════════════════════
# Figure A: GC-control attenuation paired forest
# ═══════════════════════════════════════════════════════════════
def _bootstrap_median_ci(values, n_iter=1000, seed=42, ci=95):
    """Return the observed median and a percentile bootstrap confidence interval."""
    values = np.asarray(values, dtype=float)
    median = np.median(values)
    n = len(values)
    if n < 3:
        return median, (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    boot_meds = np.array([np.median(rng.choice(values, size=n, replace=True)) for _ in range(n_iter)])
    lo = (100 - ci) / 2
    hi = 100 - lo
    return median, (np.percentile(boot_meds, lo), np.percentile(boot_meds, hi))


def figure_gc_control_attenuation():
    """Figure A: GC-control shift split by disease and extraction method."""
    records, stats = gc_control_stats(mh_d, FEAT_ORDER)

    fig, axes = plt.subplots(2, 2, figsize=(18, 14), sharex=True, sharey=True)
    panels = [
        ("Cancer", "Co-extraction", "Panel A"),
        ("Cancer", "Single", "Panel B"),
        ("Healthy", "Co-extraction", "Panel C"),
        ("Healthy", "Single", "Panel D"),
    ]

    for ax, (disease, method, panel_name) in zip(axes.flat, panels):
        y_ticks = []
        y_labels = []
        for yi, feature in enumerate(FEAT_ORDER):
            y = len(FEAT_ORDER) - 1 - yi
            summary = stats[(disease, feature, method)]
            if summary["n"] == 0:
                continue

            color = feat_color(feature)
            low_count = summary["n"] < 3
            xerr = None
            if (
                not low_count
                and np.isfinite(summary["ci_lo"])
                and np.isfinite(summary["ci_hi"])
            ):
                xerr = [
                    [summary["median"] - summary["ci_lo"]],
                    [summary["ci_hi"] - summary["median"]],
                ]
            ax.errorbar(
                summary["median"], y, xerr=xerr, fmt="o", color=color,
                markerfacecolor="none" if low_count else color,
                markeredgecolor=color, markersize=8, capsize=3, linewidth=1.4,
                alpha=0.5 if low_count else 0.9,
            )
            right_edge = (
                summary["ci_hi"]
                if np.isfinite(summary["ci_hi"])
                else summary["median"]
            )
            ax.text(
                right_edge + 0.02, y,
                f"median shift={summary['median']:+.2f}; n={summary['n']}",
                fontsize=7.8, va="center", color="#444444",
            )
            y_ticks.append(y)
            y_labels.append(FEAT_LABELS.get(feature, feature))

        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels, fontsize=9)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Paired GC-control shift: log(OR flank_500bp) − log(OR local_1kb)", fontsize=9)
        ax.set_title(
            f"{panel_name}: {disease} / {method}\n"
            f"strict cohort n={DISEASE_PAIR_COUNTS.get(disease, 0)} pairs",
            fontsize=11, fontweight="bold",
        )
        ax.grid(axis="x", alpha=0.15, linewidth=0.5)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="grey",
               markeredgecolor="grey", markersize=8,
               label="Median paired GC-control shift"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=1, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, 0.005))
    fig.suptitle(
        "Figure A. Paired GC-control shift by disease and extraction method | q90, mono\n"
        "Positive values indicate higher OR under flank_500bp control; percentile bootstrap CI (1,000, seed 42)",
        fontsize=12, fontweight="bold", y=0.995,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    fig.savefig(OUT / "figA_gc_control_forest.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / "figA_gc_control_forest.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Figure A saved (disease/method GC-control attenuation)", flush=True)
    return records, stats


# ═══════════════════════════════════════════════════════════════
# Figure B: Length-stratified raw-vs-adjusted heatmap, all 6 features
# ═══════════════════════════════════════════════════════════════
def figure_length_stratified_raw_vs_adjusted():
    """Figure B: disease-stratified raw and flank-adjusted paired heatmaps."""
    stats = disease_heatmap_stats(ov, mh_d, FEAT_ORDER, LENGTH_BINS)

    raw_values = [
        cell["median"]
        for disease in DISEASES
        for cell in stats["raw"][disease].values()
        if np.isfinite(cell["median"])
    ]
    adjusted_values = [
        cell["median"]
        for disease in DISEASES
        for cell in stats["adjusted"][disease].values()
        if np.isfinite(cell["median"])
    ]
    raw_vmax = max([abs(value) for value in raw_values] + [1e-6])
    adjusted_vmax = max([abs(value) for value in adjusted_values] + [1e-6])

    fig, axes = plt.subplots(2, 2, figsize=(18, 15), sharex=True, sharey=True)
    plot_specs = [
        ("Cancer", "raw", "Panel A", raw_vmax),
        ("Cancer", "adjusted", "Panel B", adjusted_vmax),
        ("Healthy", "raw", "Panel C", raw_vmax),
        ("Healthy", "adjusted", "Panel D", adjusted_vmax),
    ]
    images = {}

    for ax, (disease, metric, panel_name, vmax) in zip(axes.flat, plot_specs):
        matrix = np.full((len(FEAT_ORDER), len(LENGTH_BINS)), np.nan)
        annotations = [["" for _ in LENGTH_BINS] for _ in FEAT_ORDER]
        for i, feature in enumerate(FEAT_ORDER):
            for j, length_bin in enumerate(LENGTH_BINS):
                cell = stats[metric][disease][(feature, length_bin)]
                matrix[i, j] = cell["median"]
                if cell["n"]:
                    value_text = (
                        f"{cell['median']:+.3f}"
                        if metric == "raw"
                        else f"{cell['median']:+.2f}"
                    )
                    annotations[i][j] = (
                        f"{value_text}\n{p_label_compact(cell['p'])}"
                        f"{sig_stars(cell['p'])}\nn={cell['n']}"
                    )
                else:
                    annotations[i][j] = "NA\nn=0"

        image = ax.imshow(
            matrix, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax
        )
        images[(disease, metric)] = image
        ax.set_xticks(range(len(LENGTH_BINS)))
        ax.set_xticklabels(
            [LB_LABELS.get(length_bin, length_bin) for length_bin in LENGTH_BINS],
            fontsize=9,
        )
        ax.set_yticks(range(len(FEAT_ORDER)))
        ax.set_yticklabels(
            [FEAT_LABELS.get(feature, feature) for feature in FEAT_ORDER],
            fontsize=9,
        )
        for i in range(len(FEAT_ORDER)):
            for j in range(len(LENGTH_BINS)):
                value = matrix[i, j]
                color = (
                    "white"
                    if np.isfinite(value) and abs(value) > 0.55 * vmax
                    else "black"
                )
                ax.text(
                    j, i, annotations[i][j], ha="center", va="center",
                    fontsize=7.4, fontweight="bold", color=color,
                )

        metric_title = (
            "Raw overlap Δ(fraction)\nCo − Single, q90"
            if metric == "raw"
            else "Flank-adjusted Δlog(OR)\nCo − Single, flank_500bp, q90"
        )
        ax.set_title(
            f"{panel_name}: {disease}\n{metric_title}\n"
            f"strict cohort n={DISEASE_PAIR_COUNTS.get(disease, 0)} pairs",
            fontsize=10.5, fontweight="bold",
        )

    raw_cbar = fig.colorbar(
        images[("Cancer", "raw")], ax=axes[:, 0], orientation="horizontal",
        fraction=0.04, pad=0.09, aspect=45,
    )
    raw_cbar.set_label("Δ overlap fraction", fontsize=9)
    adjusted_cbar = fig.colorbar(
        images[("Cancer", "adjusted")], ax=axes[:, 1], orientation="horizontal",
        fraction=0.04, pad=0.09, aspect=45,
    )
    adjusted_cbar.set_label("GC-adjusted Δ log(OR)", fontsize=9)
    fig.suptitle(
        "Figure B. Fragment-length structure by disease | pair-level statistics\n"
        "Raw overlap composition versus flank-adjusted enrichment",
        fontsize=12, fontweight="bold", y=0.995,
    )
    fig.subplots_adjust(
        top=0.89, bottom=0.15, left=0.17, right=0.97,
        hspace=0.34, wspace=0.12,
    )
    fig.savefig(
        OUT / "figB_length_raw_vs_adjusted_heatmap.png",
        dpi=200, bbox_inches="tight",
    )
    fig.savefig(
        OUT / "figB_length_raw_vs_adjusted_heatmap.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)
    print("Figure B saved (disease-stratified raw vs adjusted)", flush=True)
    return stats



def _collect_pooled_ors(frame, disease, feature, method):
    """Return raw OR values for a (disease, feature, method) pooled across all sources."""
    sub = frame[
        (frame["local_gc_control"] == "flank_500bp")
        & (frame["gc_definition"] == "q90")
        & (frame["length_bin"] == "mono_151_220")
        & (frame["disease"] == disease)
        & (frame["feature"] == feature)
        & (frame["extraction"] == method)
        & frame["or"].notna()
    ].copy()
    sub = sub[np.isfinite(sub["or"].to_numpy(dtype=float)) & (sub["or"] > 0)]
    # One value per pair (dedup if multiple sources for same pair)
    pairs = sub.groupby("pair_id")["or"].first()
    return pairs.to_numpy(dtype=float)

def figure_source_consistency():
    """Figure C: Disease effect by method — disease Δ within each extraction method."""
    stats = source_consistency_stats(mh, FEAT_ORDER)

    method_colors = {"Co-extraction": "#C8537A", "Single": "#5B4B8A"}
    disease_markers = {"Cancer": "o", "Healthy": "s"}

    # Pool OR values across sources for each (disease, feature, method)
    pooled = {}
    for disease in DISEASES:
        for feature in FEAT_ORDER:
            for method in METHODS:
                raw_vals = _collect_pooled_ors(mh, disease, feature, method)
                if len(raw_vals) < 3:
                    continue
                median, (ci_lo, ci_hi) = _bootstrap_median_ci(raw_vals)
                pooled[(disease, feature, method)] = {
                    "median": median, "ci_lo": ci_lo, "ci_hi": ci_hi, "n": len(raw_vals),
                }
    n_features = len(FEAT_ORDER)
    fig, axes = plt.subplots(n_features, 1, figsize=(10, 1.5 * n_features),
                             sharex=False, sharey=False)
    if n_features == 1:
        axes = [axes]

    for row_idx, feature in enumerate(FEAT_ORDER):
        ax = axes[row_idx]
        cell_ors = []
        method_y = {"Co-extraction": 0.06, "Single": -0.06}

        for method in METHODS:
            ca_s = pooled.get(("Cancer", feature, method))
            he_s = pooled.get(("Healthy", feature, method))
            if not ca_s or not he_s:
                continue
            y = method_y[method]
            method_color = method_colors[method]
            mid_x = (ca_s["median"] + he_s["median"]) / 2

            for s in [ca_s, he_s]:
                cell_ors.append(s["median"])
                if np.isfinite(s.get("ci_lo", np.nan)):
                    cell_ors.append(s["ci_lo"])
                if np.isfinite(s.get("ci_hi", np.nan)):
                    cell_ors.append(s["ci_hi"])

            # Grey connecting line: Cancer → Healthy
            ax.plot(
                [ca_s["median"], he_s["median"]],
                [y, y],
                color="#777777", linewidth=1.8, alpha=0.55, zorder=1,
            )

            # Cancer marker (circle)
            low_ca = ca_s["n"] < 3
            xerr_ca = None
            if not low_ca and np.isfinite(ca_s["ci_lo"]) and np.isfinite(ca_s["ci_hi"]):
                xerr_ca = [[ca_s["median"] - ca_s["ci_lo"]], [ca_s["ci_hi"] - ca_s["median"]]]
            ax.errorbar(
                ca_s["median"], y, xerr=xerr_ca, fmt="o",
                color=method_color,
                markerfacecolor="none" if low_ca else method_color,
                markeredgecolor=method_color, markersize=9, capsize=3,
                linewidth=1.4, alpha=0.45 if low_ca else 0.9, zorder=3,
            )
            ax.annotate(
                f"n={ca_s['n']}",
                (ca_s["median"], y),
                xytext=(-2, -10), textcoords="offset points",
                fontsize=7.5, ha="right", color="#444444",
            )

            # Healthy marker (square)
            low_he = he_s["n"] < 3
            xerr_he = None
            if not low_he and np.isfinite(he_s["ci_lo"]) and np.isfinite(he_s["ci_hi"]):
                xerr_he = [[he_s["median"] - he_s["ci_lo"]], [he_s["ci_hi"] - he_s["median"]]]
            ax.errorbar(
                he_s["median"], y, xerr=xerr_he, fmt="s",
                color=method_color,
                markerfacecolor="none" if low_he else method_color,
                markeredgecolor=method_color, markersize=9, capsize=3,
                linewidth=1.4, alpha=0.45 if low_he else 0.9, zorder=3,
            )
            ax.annotate(
                f"n={he_s['n']}",
                (he_s["median"], y),
                xytext=(2, -10), textcoords="offset points",
                fontsize=7.5, ha="left", color="#444444",
            )

            # Δ = Cancer − Healthy (within same method)
            delta = ca_s["median"] - he_s["median"]
            ax.annotate(
                f"Δ={delta:+.3f}",
                (mid_x, y),
                xytext=(0, 8), textcoords="offset points",
                fontsize=7.5, ha="center", color="#555555", fontweight="bold",
            )
        ax.set_ylabel(
            FEAT_LABELS.get(feature, feature), fontsize=10,
            fontweight="bold", rotation=0, ha="right", va="center",
        )
        ax.tick_params(axis="y", left=False, labelleft=False)
        if row_idx == n_features - 1:
            ax.set_xlabel("OR (flank_500bp, q90, mono) — pooled across sources", fontsize=9)
        ax.set_ylim(-0.14, 0.16)

        if cell_ors:
            cmin, cmax = min(cell_ors), max(cell_ors)
            span = cmax - cmin
            if span < 0.10:
                mid = (cmin + cmax) / 2
                span = max(span, 0.25)
                cmin, cmax = mid - span / 2, mid + span / 2
            pad = span * 0.35
            ax.set_xlim(max(0.0, cmin - pad), cmax + pad)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = []
    for method, color in method_colors.items():
        legend_elements.append(
            Line2D([0], [0], marker="s", color=color, markerfacecolor=color,
                   markersize=9, linewidth=0, label=method)
        )
    for disease, marker in disease_markers.items():
        shape_label = "●" if marker == "o" else "■"
        legend_elements.append(
            Line2D([0], [0], marker=marker, color="grey", markerfacecolor="white",
                   markeredgecolor="grey", markersize=9, linewidth=0,
                   label=f"{disease} ({shape_label})")
        )
    fig.legend(
        handles=legend_elements, loc="lower center", ncol=4,
        fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.005),
    )
    fig.suptitle(
        "Figure C. Disease effect within each extraction method: Cancer − Healthy OR shift\n"
        "Color = extraction method, shape = disease; "
        "Δ = median(Cancer) − median(Healthy)",
        fontsize=12, fontweight="bold", y=0.998,
    )
    fig.subplots_adjust(
        top=0.93, bottom=0.10, left=0.22, right=0.95, hspace=0.45,
    )
    fig.savefig(OUT / "figC_source_consistency.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / "figC_source_consistency.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Figure C saved (disease × method pooled)", flush=True)
    return pooled



# ── Generate new figures A-C ──
figure_gc_control_attenuation()
figure_length_stratified_raw_vs_adjusted()
figure_source_consistency()

print(f"\n=== All figures saved to {OUT} ===", flush=True)
for f in sorted(OUT.glob("fig*.png")):
    print(f"  {f.name}  ({f.stat().st_size // 1024}KB)", flush=True)
print("=== DONE ===", flush=True)
