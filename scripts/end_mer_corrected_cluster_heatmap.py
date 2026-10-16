#!/usr/bin/env python3
"""Cluster corrected end-mer values using differential 4-mers.

Primary input is the paired non-differential value matrix produced by
`scripts/end_mer_latest_paired_analysis.py`. Batch effect is removed by
mean-centering each pair-set per motif while preserving within-method and
within-disease differences.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import re


DISEASE_ORDER = ["Healthy", "Post-op", "Cancer"]
EXTRACTION_ORDER = ["Co-extraction", "Single"]
DISEASE_COLORS = {"Healthy": "#2A9D8F", "Post-op": "#E9C46A", "Cancer": "#E76F51"}
EXCLUDED_SAMPLE_IDS = {"H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8", "H9", "H10"}
EXTRACTION_COLORS = {"cfDNA": "#524682", "CoDNA": "#BD507D"}
PAIR_SET_COLORS = {"Retrospective cohort": "#8ECAE6", "Prospective cohort": "#264653"}
PAIR_SET_MAP = {
    "first_FQ_within_dir": "Retrospective cohort",
    "raw03_vs_stage1_b2": "Prospective cohort",
    "first_FQ": "Retrospective cohort",
    "raw03": "Prospective cohort",
}
EXTRACTION_MAP = {
    "Single": "cfDNA",
    "single": "cfDNA",
    "cf": "cfDNA",
    "cfDNA": "cfDNA",
    "Co-extraction": "CoDNA",
    "co": "CoDNA",
    "CoDNA": "CoDNA",
}

import sys as _sys
from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import pathconfig as _pc

DEFAULT_VALUE_MATRIX = _pc.get("value_matrix", "outputs/end_mer_latest_paired_analysis/latest_paired_ordered_value_matrix_normalized_5prime.tsv")
DEFAULT_PAIRS = _pc.get("endmer_pairs", "outputs/end_mer_latest_paired_analysis/latest_end_mer_correct_pairs.tsv")
DEFAULT_DIFF = _pc.get("endmer_diff", "outputs/end_mer_group_analysis/end_mer_paired_differential_normalized_5prime.tsv")
DEFAULT_OUT_DIR = _pc.get("cluster_out_dir", "outputs/end_mer_latest_paired_analysis")


def batch_correct_by_pair_set(matrix: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    corrected = matrix.copy().astype(float)
    global_mean = matrix.mean(axis=0)
    for pair_set in metadata["pair_set"].dropna().unique():
        rows = metadata.index[metadata["pair_set"] == pair_set]
        batch_mean = matrix.loc[rows].mean(axis=0)
        corrected.loc[rows] = matrix.loc[rows] + (global_mean - batch_mean)
    return corrected


def select_differential_mers(matrix: pd.DataFrame, diff_table: pd.DataFrame | None, top_n: int) -> list[str]:
    if diff_table is not None and "mer" in diff_table.columns:
        score_col = None
        for candidate in ["pvalue_ttest", "pvalue", "pvalue_wilcoxon"]:
            if candidate in diff_table.columns:
                score_col = candidate
                break
        available = diff_table[diff_table["mer"].isin(matrix.columns)].copy()
        if not available.empty:
            if score_col is not None:
                available = available.sort_values(score_col, ascending=True)
            elif "neglog10p" in available.columns:
                available = available.sort_values("neglog10p", ascending=False)
            return available["mer"].drop_duplicates().head(top_n).tolist()
    cv = matrix.std(axis=0) / (matrix.mean(axis=0).abs() + 1e-12)
    return cv.sort_values(ascending=False).head(top_n).index.tolist()


def disease_from_pair_id(pair_id: str) -> str | None:
    text = str(pair_id)
    if text.startswith("MQ"):
        return "Post-op"
    if text.startswith(("H", "C")) or text.replace(".0", "").isdigit():
        return "Healthy"
    if text.startswith(("B", "E")):
        return "Cancer"
    return None


def sample_metadata(value_matrix: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    pair_meta = pairs[["pair_set", "pair_id", "disease"]].drop_duplicates("pair_id").set_index("pair_id")
    rows = []
    for sample in value_matrix.index:
        pair_id, extraction = str(sample).split("|", 1)
        disease = pair_meta.loc[pair_id, "disease"] if pair_id in pair_meta.index else disease_from_pair_id(pair_id)
        pair_set = pair_meta.loc[pair_id, "pair_set"] if pair_id in pair_meta.index else "unknown"
        rows.append({"sample": sample, "pair_id": pair_id, "extraction": extraction, "disease": disease, "pair_set": pair_set})
    return pd.DataFrame(rows).set_index("sample")


def load_inputs(value_matrix_path: Path, pairs_path: Path, diff_path: Path | None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    matrix = pd.read_csv(value_matrix_path, sep="\t", index_col=0)
    pairs = pd.read_csv(pairs_path, sep="\t")
    diff = pd.read_csv(diff_path, sep="\t") if diff_path is not None and diff_path.exists() else None
    return matrix, pairs, diff



def _pair_sort_key(pair_id: str) -> tuple[str, int, str]:
    text = str(pair_id)
    match = re.fullmatch(r"([A-Za-z]+)?(\d+)(?:\.0)?", text)
    if match:
        return (match.group(1) or "", int(match.group(2)), text)
    return (text, 10**9, text)


def order_samples_by_method_disease(matrix: pd.DataFrame, metadata: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered_samples: list[str] = []
    for disease in DISEASE_ORDER:
        for extraction in EXTRACTION_ORDER:
            sub = metadata[
                (metadata["extraction"] == extraction)
                & (metadata["disease"] == disease)
            ].copy()
            mask = ~sub.index.astype(str).str.split("|").str[0].isin(EXCLUDED_SAMPLE_IDS)
            sub = sub[mask]
            ordered_samples.extend(sorted(sub.index.astype(str), key=lambda sample: _pair_sort_key(str(metadata.loc[sample, "pair_id"]))))
    return matrix.loc[ordered_samples], metadata.loc[ordered_samples]


def plot_clustered_heatmap(matrix: pd.DataFrame, metadata: pd.DataFrame, out_path: Path) -> None:
    import seaborn as sns

    if matrix.index.has_duplicates:
        duplicates = matrix.index[matrix.index.duplicated()].unique().tolist()
        raise ValueError(f"Duplicate sample rows before plotting: {duplicates[:10]}")
    if metadata.index.has_duplicates:
        duplicates = metadata.index[metadata.index.duplicated()].unique().tolist()
        raise ValueError(f"Duplicate metadata rows before plotting: {duplicates[:10]}")
    if not matrix.index.equals(metadata.index):
        raise ValueError("Matrix and metadata sample order are not aligned before plotting")

    row_data = matrix.T
    z = row_data.sub(row_data.mean(axis=1), axis=0).div(row_data.std(axis=1).replace(0, 1), axis=0)

    disease_list = metadata["disease"].astype(str).tolist()
    boundaries = [
        d for d in range(1, len(disease_list))
        if disease_list[d] != disease_list[d - 1]
    ]

    # Insert one real blank column after each disease group. The inserted columns
    # are constant zeros for clustering (no row-distance contribution) and masked
    # to render as white gaps in the heatmap.
    z_parts: list[pd.DataFrame] = []
    mask_parts: list[pd.DataFrame] = []
    color_disease: list[str] = []
    color_type: list[str] = []
    color_pair: list[str] = []
    start = 0
    for boundary_idx, end in enumerate(boundaries + [z.shape[1]]):
        block = z.iloc[:, start:end]
        z_parts.append(block)
        mask_parts.append(pd.DataFrame(False, index=z.index, columns=block.columns))

        block_meta = metadata.iloc[start:end]
        mapped_extraction = block_meta["extraction"].map(EXTRACTION_MAP).fillna(block_meta["extraction"])
        mapped_pair_set = block_meta["pair_set"].map(PAIR_SET_MAP).fillna(block_meta["pair_set"])
        color_disease.extend(block_meta["disease"].map(DISEASE_COLORS).tolist())
        color_type.extend(mapped_extraction.map(EXTRACTION_COLORS).tolist())
        color_pair.extend(mapped_pair_set.map(PAIR_SET_COLORS).fillna("#999999").tolist())

        if boundary_idx < len(boundaries):
            sep_name = f"__separator_{boundary_idx + 1}__"
            sep = pd.DataFrame({sep_name: np.zeros(z.shape[0], dtype=float)}, index=z.index)
            z_parts.append(sep)
            mask_parts.append(pd.DataFrame(True, index=z.index, columns=[sep_name]))
            color_disease.append("#FFFFFF")
            color_type.append("#FFFFFF")
            color_pair.append("#FFFFFF")
        start = end

    z_with_gaps = pd.concat(z_parts, axis=1)
    mask_with_gaps = pd.concat(mask_parts, axis=1)
    col_colors = pd.DataFrame(
        {
            "Disease": color_disease,
            "Type": color_type,
            "Pair set": color_pair,
        },
        index=z_with_gaps.columns,
    )

    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#FFFFFF")
    grid = sns.clustermap(
        z_with_gaps,
        mask=mask_with_gaps,
        method="average",
        metric="euclidean",
        cmap=cmap,
        center=0,
        vmin=-2.5,
        vmax=2.5,
        figsize=(max(18, z_with_gaps.shape[1] * 0.16), max(10, z_with_gaps.shape[0] * 0.07)),
        row_cluster=True,
        col_cluster=False,
        col_colors=col_colors,
        xticklabels=False,
        yticklabels=True,
        cbar_kws={"label": "Z-score after pair-set correction"},
        dendrogram_ratio=0.08,
        colors_ratio=0.035,
    )
    grid.ax_heatmap.set_xlabel("Sample")
    grid.ax_heatmap.set_ylabel("Differential 5' end 4-mer")
    grid.ax_heatmap.set_yticklabels(grid.ax_heatmap.get_yticklabels(), rotation=0, fontsize=5)
    handles = []
    for title, colors in [("Disease", DISEASE_COLORS), ("Type", EXTRACTION_COLORS), ("Pair set", PAIR_SET_COLORS)]:
        handles.append(mpatches.Patch(color="white", label=title))
        handles.extend(mpatches.Patch(color=color, label=f"  {label}") for label, color in colors.items())
    grid.ax_heatmap.legend(handles=handles, bbox_to_anchor=(1.02, 1.05), loc="upper left", frameon=False, fontsize=8)
    grid.fig.suptitle(f"Batch-corrected end-mer clustered heatmap — {matrix.shape[0]} samples, {matrix.shape[1]} differential motifs", y=1.02)
    grid.savefig(out_path, dpi=220, bbox_inches="tight")
    grid.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close("all")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot batch-corrected differential end-mer clustered heatmap")
    parser.add_argument("--value-matrix", type=Path, default=DEFAULT_VALUE_MATRIX)
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--diff-table", type=Path, default=DEFAULT_DIFF)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--top-n", type=int, default=128)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    matrix, pairs, diff = load_inputs(args.value_matrix, args.pairs, args.diff_table)
    metadata = sample_metadata(matrix, pairs)
    corrected = batch_correct_by_pair_set(matrix, metadata)
    mers = select_differential_mers(corrected, diff, args.top_n)
    selected = corrected[mers]
    selected_ordered, metadata_ordered = order_samples_by_method_disease(selected, metadata)
    selected_ordered.to_csv(args.out_dir / f"latest_paired_batch_corrected_differential_mers_top{len(mers)}.tsv", sep="\t")
    metadata_ordered.to_csv(args.out_dir / "latest_paired_cluster_sample_metadata.tsv", sep="\t")
    plot_clustered_heatmap(selected_ordered, metadata_ordered, args.out_dir / f"latest_paired_batch_corrected_clustered_heatmap_top{len(mers)}.png")
    print(f"Samples: {selected.shape[0]}")
    print(f"Mers: {selected.shape[1]}")
    print(f"By disease: {metadata['disease'].value_counts().to_dict()}")
    print(f"By type: {metadata['extraction'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
