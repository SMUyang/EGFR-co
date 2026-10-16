#!/usr/bin/env python3
"""Latest end-mer paired analysis using corrected sample mappings.

Uses latest fragment end-mer CSVs from:
- Validation cohort (raw03) co-extraction: data/endmer/raw03 (or paths.yaml raw03_dir)
- Baseline single extraction (stage1_b2): data/endmer/stage1_b2
- Discovery cohort (first_fq) paired set: data/endmer/first_fq

Corrected aliases:
- raw03 NA-P1 -> B31 co-extraction
- raw03 NA-2-SP4 -> B13 co-extraction
- stage1_b2 NA-1-SP4 -> MQ10 single
- stage1_b2 NA-2-SP4 -> B13 single
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


import sys as _sys
from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import pathconfig as _pc

DEFAULT_RAW03_DIR = _pc.get("raw03_dir", "data/endmer/raw03")
DEFAULT_FIRST_FQ_DIR = _pc.get("first_fq_dir", "data/endmer/first_fq")
DEFAULT_STAGE1_B2_DIR = _pc.get("stage1_b2_dir", DEFAULT_RAW03_DIR)
DEFAULT_RAW03_MANIFEST = _pc.get("raw03_manifest", "data/endmer/raw03_manifest.json")
DEFAULT_OUT_DIR = Path("./end_mer_latest_paired_analysis")
VALID_METRICS = ["normalized_5prime", "normalized_3prime", "oe_ratio_5prime", "oe_ratio_3prime"]
DISEASE_ORDER = ["Healthy", "Cancer", "Post-op"]
EXTRACTION_ORDER = ["Single", "Co-extraction"]
EXCLUDED_SAMPLE_IDS = {"H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8", "H9", "H10"}


SOURCE_SAMPLE_ALIASES = {
    ("raw03_endmer", "NA-P1.recal"): ("B31", "co", "Cancer"),
    ("raw03_endmer", "NA-2-SP4.recal"): ("B13", "co", "Cancer"),
    ("stage1_b2_endmer", "NA-1-SP4.recal"): ("MQ10", "single", "Post-op"),
    ("stage1_b2_endmer", "NA-2-SP4.recal"): ("B13", "single", "Cancer"),
}


def sample_from_csv(path: Path) -> str:
    name = path.name
    suffix = ".recal.end_mer.csv"
    if name.endswith(suffix):
        return name[: -len(suffix)] + ".recal"
    return path.stem + ".recal"


def disease_from_id(sample_id: str) -> str | None:
    if sample_id.startswith("MQ"):
        return "Post-op"
    if sample_id.startswith(("H", "C")):
        return "Healthy"
    if sample_id.startswith(("B", "E")):
        return "Cancer"
    return None


def classify_standard_sample(sample: str) -> tuple[str, str, str | None]:
    name = sample.replace(".recal", "")
    extraction = "single" if name.startswith("S") else "co"
    sample_id = name[1:] if extraction == "single" else name
    return sample_id, extraction, disease_from_id(sample_id)


def classify_first_fq_sample(sample: str) -> tuple[str, str, str | None]:
    name = sample.replace(".recal", "")
    if re.fullmatch(r"\d+(?:\.0)?", name):
        return name.split(".")[0], "single", None
    if re.fullmatch(r"C\d+", name):
        return name[1:], "co", "Healthy"
    if re.fullmatch(r"E\d+", name):
        return name[1:], "co", "Cancer"
    return classify_standard_sample(sample)


def classify_sample_for_source(sample: str, source: str) -> tuple[str, str, str | None]:
    alias = SOURCE_SAMPLE_ALIASES.get((source, sample))
    if alias is not None:
        return alias
    if source == "first_FQ_endmer":
        return classify_first_fq_sample(sample)
    return classify_standard_sample(sample)


def is_alias_sample(sample: str, source: str) -> bool:
    return (source, sample) in SOURCE_SAMPLE_ALIASES


def read_manifest_sample_names(path: Path) -> set[str]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    samples = manifest.get("samples")
    if not isinstance(samples, dict):
        raise ValueError(f"Manifest missing object field 'samples': {path}")
    return set(samples.keys())


def load_inventory(source_dirs: dict[str, Path]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for source, directory in source_dirs.items():
        for csv_path in sorted(directory.glob("*.csv")):
            sample = sample_from_csv(csv_path)
            pair_id, extraction, disease = classify_sample_for_source(sample, source)
            rows.append({
                "source": source,
                "sample": sample,
                "pair_id": pair_id,
                "extraction": extraction,
                "disease": disease,
                "path": str(csv_path),
            })
    if not rows:
        raise RuntimeError("No end-mer CSV files found")
    return pd.DataFrame(rows)


def _prefer_alias_rows(table: pd.DataFrame, source: str, keys: list[str]) -> pd.DataFrame:
    if table.empty:
        return table
    tmp = table.copy()
    tmp["_alias_priority"] = tmp["sample"].map(lambda sample: 1 if is_alias_sample(str(sample), source) else 0)
    tmp = tmp.sort_values([*keys, "_alias_priority", "sample"])
    return tmp.drop_duplicates(keys, keep="last").drop(columns=["_alias_priority"])


def build_correct_pairs(inventory: pd.DataFrame, raw_manifest_samples: set[str]) -> pd.DataFrame:
    raw = inventory[inventory["source"] == "raw03_endmer"]
    stage = inventory[inventory["source"] == "stage1_b2_endmer"]
    first = inventory[inventory["source"] == "first_FQ_endmer"]

    raw_co = raw[
        ((raw["sample"].isin(raw_manifest_samples)) | raw["sample"].map(lambda s: is_alias_sample(str(s), "raw03_endmer")))
        & (raw["extraction"] == "co")
    ]
    raw_co = _prefer_alias_rows(raw_co, "raw03_endmer", ["pair_id"])
    stage_single = stage[stage["extraction"] == "single"]
    # Complement stage1_b2 Single with raw03 Single samples for pair_ids missing in stage1
    raw_single = raw[raw["extraction"] == "single"]
    missing_ids = set(raw_co["pair_id"]) - set(stage_single["pair_id"])
    if missing_ids:
        raw_single_complement = raw_single[raw_single["pair_id"].isin(missing_ids)]
        if not raw_single_complement.empty:
            stage_single = pd.concat([stage_single, raw_single_complement], ignore_index=True)
    stage_single = _prefer_alias_rows(stage_single, "stage1_b2_endmer", ["pair_id"])
    first_single = first[first["extraction"] == "single"]
    first_co = first[first["extraction"] == "co"]

    raw_pairs = raw_co.merge(stage_single, on="pair_id", suffixes=("_co", "_single"))
    first_pairs = first_co.merge(first_single, on="pair_id", suffixes=("_co", "_single"))

    raw_out = pd.DataFrame({
        "pair_set": "raw03_vs_stage1_b2",
        "pair_id": raw_pairs["pair_id"],
        "disease": raw_pairs["disease_co"].combine_first(raw_pairs["disease_single"]),
        "single_sample": raw_pairs["sample_single"],
        "single_source": raw_pairs["source_single"],
        "co_sample": raw_pairs["sample_co"],
        "co_source": raw_pairs["source_co"],
        "single_path": raw_pairs["path_single"],
        "co_path": raw_pairs["path_co"],
    })
    first_out = pd.DataFrame({
        "pair_set": "first_FQ_within_dir",
        "pair_id": first_pairs["pair_id"],
        "disease": first_pairs["disease_co"].combine_first(first_pairs["disease_single"]),
        "single_sample": first_pairs["sample_single"],
        "single_source": first_pairs["source_single"],
        "co_sample": first_pairs["sample_co"],
        "co_source": first_pairs["source_co"],
        "single_path": first_pairs["path_single"],
        "co_path": first_pairs["path_co"],
    })
    return pd.concat([raw_out, first_out], ignore_index=True).sort_values(["pair_set", "disease", "pair_id"]).reset_index(drop=True)


def _load_metric_series(path: str, metric: str) -> pd.Series:
    df = pd.read_csv(path)
    if "mer" not in df.columns or metric not in df.columns:
        raise ValueError(f"Missing mer or {metric} in {path}")
    return df.set_index("mer")[metric].astype(float)


def load_difference_matrix(pairs: pd.DataFrame, metric: str) -> pd.DataFrame:
    rows = {}
    for row in pairs.itertuples(index=False):
        single = _load_metric_series(row.single_path, metric)
        co = _load_metric_series(row.co_path, metric)
        rows[str(row.pair_id)] = co.subtract(single, fill_value=0.0)
    matrix = pd.DataFrame(rows).T.fillna(0.0)
    matrix.index.name = "pair_id"
    matrix.columns.name = "mer"
    return matrix


def load_pair_value_matrix(pairs: pd.DataFrame, metric: str) -> pd.DataFrame:
    rows = {}
    labels = []
    for row in pairs.itertuples(index=False):
        single_key = f"{row.pair_id}|Single"
        co_key = f"{row.pair_id}|Co-extraction"
        rows[single_key] = _load_metric_series(row.single_path, metric)
        rows[co_key] = _load_metric_series(row.co_path, metric)
        labels.extend([single_key, co_key])
    matrix = pd.DataFrame(rows).T.fillna(0.0)
    matrix = matrix.loc[labels]
    matrix.index.name = "sample"
    matrix.columns.name = "mer"
    return matrix


def _pair_sort_key(pair_id: str) -> tuple[str, int, str]:
    text = str(pair_id)
    match = re.fullmatch(r"([A-Za-z]+)?(\d+)(?:\.0)?", text)
    if match:
        return (match.group(1) or "", int(match.group(2)), text)
    return (text, 10**9, text)


def build_ordered_value_matrix(value_matrix: pd.DataFrame, pairs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pair_meta = pairs[["pair_id", "disease"]].drop_duplicates().copy()
    ordered_rows: list[str] = []
    meta_rows: list[dict[str, str]] = []
    for disease in DISEASE_ORDER:
        disease_pairs = pair_meta[pair_meta["disease"] == disease]["pair_id"].astype(str).tolist()
        for pair_id in sorted(disease_pairs, key=_pair_sort_key):
            for extraction in EXTRACTION_ORDER:
                row_id = f"{pair_id}|{extraction}"
                if row_id in value_matrix.index:
                    ordered_rows.append(row_id)
                    meta_rows.append({"sample": row_id, "pair_id": pair_id, "disease": disease, "extraction": extraction})
    ordered = value_matrix.loc[ordered_rows]
    metadata = pd.DataFrame(meta_rows).set_index("sample")
    return ordered, metadata



def select_variable_motifs(matrix: pd.DataFrame, top_n: int) -> list[str]:
    variability = matrix.std(axis=0) / (matrix.mean(axis=0).abs() + 1e-12)
    return variability.sort_values(ascending=False).head(top_n).index.tolist()


def plot_difference_heatmap(diff_matrix: pd.DataFrame, out_dir: Path, metric: str, top_n: int) -> None:
    motifs = select_variable_motifs(diff_matrix, min(top_n, diff_matrix.shape[1]))
    data = diff_matrix[motifs]
    values = data.to_numpy(dtype=float)
    vmax = np.nanpercentile(np.abs(values), 98) if values.size else 1.0
    if vmax <= 0:
        vmax = 1.0
    fig, ax = plt.subplots(figsize=(14, max(8, len(data) * 0.12)))
    im = ax.imshow(values, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(data.columns)))
    ax.set_xticklabels(data.columns, rotation=90, fontsize=6)
    ax.set_yticks(range(len(data.index)))
    ax.set_yticklabels(data.index, fontsize=7)
    ax.set_xlabel("4-mer motif")
    ax.set_ylabel("Pair ID")
    ax.set_title(f"Latest paired end-mer differences ({metric})", fontweight="bold")
    fig.colorbar(im, ax=ax, label="Co-extraction - Single")
    fig.tight_layout()
    fig.savefig(out_dir / f"latest_paired_difference_heatmap_{metric}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)




def plot_ordered_value_heatmap(
    value_matrix: pd.DataFrame,
    pairs: pd.DataFrame,
    out_dir: Path,
    metric: str,
    top_n: int,
) -> None:
    ordered, metadata = build_ordered_value_matrix(value_matrix, pairs)
    motifs = select_variable_motifs(ordered, min(top_n, ordered.shape[1]))
    data = ordered[motifs]
    values = data.to_numpy(dtype=float)
    vmin = np.nanpercentile(values, 2) if values.size else 0.0
    vmax = np.nanpercentile(values, 98) if values.size else 1.0
    if vmax <= vmin:
        vmax = vmin + 1.0

    disease_colors = {"Healthy": "#2A9D8F", "Post-op": "#E9C46A", "Cancer": "#E76F51"}
    extraction_colors = {"Single": "#457B9D", "Co-extraction": "#E63946"}

    fig = plt.figure(figsize=(17, max(10, len(data) * 0.16)))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.18, 0.18, 12], wspace=0.02)
    ax_disease = fig.add_subplot(gs[0, 0])
    ax_extraction = fig.add_subplot(gs[0, 1])
    ax = fig.add_subplot(gs[0, 2])

    disease_rgb = np.array([
        tuple(int(disease_colors[disease].lstrip("#")[i:i+2], 16) / 255 for i in (0, 2, 4))
        for disease in metadata["disease"]
    ])[:, None, :]
    extraction_rgb = np.array([
        tuple(int(extraction_colors[extraction].lstrip("#")[i:i+2], 16) / 255 for i in (0, 2, 4))
        for extraction in metadata["extraction"]
    ])[:, None, :]

    ax_disease.imshow(disease_rgb, aspect="auto")
    ax_extraction.imshow(extraction_rgb, aspect="auto")
    ax_disease.set_xticks([])
    ax_extraction.set_xticks([])
    ax_disease.set_yticks([])
    ax_extraction.set_yticks([])
    ax_disease.set_title("Disease", fontsize=8, rotation=90, va="bottom")
    ax_extraction.set_title("Type", fontsize=8, rotation=90, va="bottom")

    im = ax.imshow(values, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(data.columns)))
    ax.set_yticks(range(len(data.index)))
    # Defensive: align metadata with data rows
    ylabels = []
    for idx in data.index:
        meta_row = metadata.loc[idx] if idx in metadata.index else None
        if meta_row is not None:
            ylabels.append(f"{meta_row['pair_id']} {'S' if meta_row['extraction'] == 'Single' else 'C'}")
        else:
            ylabels.append(str(idx))
    ax.set_yticklabels(ylabels, fontsize=5)
    ax.set_xlabel("4-mer motif")
    ax.set_ylabel("Sample (ordered by disease, pair, type)")
    ax.set_title(f"Latest paired end-mer values ({metric})", fontweight="bold")

    start = 0
    for disease in DISEASE_ORDER:
        count = int((metadata["disease"] == disease).sum())
        if count:
            ax.axhline(start - 0.5, color="white", linewidth=1.0)
            ax.text(-len(data.columns) * 0.015, start + count / 2 - 0.5, disease, ha="right", va="center", fontsize=8)
            start += count
    ax.axhline(len(data) - 0.5, color="white", linewidth=1.0)

    legend_handles = [
        plt.Line2D([0], [0], marker="s", color="w", label=label, markerfacecolor=color, markersize=8)
        for label, color in {**disease_colors, **extraction_colors}.items()
    ]
    ax.legend(handles=legend_handles, bbox_to_anchor=(1.02, 1.0), loc="upper left", frameon=False, fontsize=8)
    fig.colorbar(im, ax=ax, label=metric)
    fig.tight_layout()
    fig.savefig(out_dir / f"latest_paired_value_heatmap_{metric}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    ordered.to_csv(out_dir / f"latest_paired_ordered_value_matrix_{metric}.tsv", sep="\t")
    metadata.to_csv(out_dir / "latest_paired_ordered_sample_metadata.tsv", sep="\t")

def plot_mean_difference_bar(diff_matrix: pd.DataFrame, out_dir: Path, metric: str, top_n: int) -> None:
    means = diff_matrix.mean(axis=0).sort_values()
    top = pd.concat([means.head(top_n), means.tail(top_n)]).drop_duplicates()
    colors = ["#457B9D" if value < 0 else "#E63946" for value in top]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(range(len(top)), top.values, color=colors)
    ax.axhline(0.0, color="#555555", linestyle="--", linewidth=0.9)
    ax.set_xticks(range(len(top)))
    ax.set_xticklabels(top.index, rotation=60, ha="right")
    ax.set_ylabel(f"Mean difference ({metric})")
    ax.set_title("Latest paired end-mer mean difference: Co-extraction minus Single", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / f"latest_paired_mean_difference_{metric}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

# User-specified colors: Single=#330287, Co-extraction=#DB3182
EXTRACTION_COLORS_ENDMER = {"Single": "#330287", "Co-extraction": "#DB3182"}

def _shannon_entropy(series: pd.Series) -> float:
    """Compute Shannon entropy (base 2) from a frequency distribution."""
    p = series.astype(float)
    p = p[p > 0] / p.sum()
    return - (p * np.log2(p)).sum()

def plot_endmer_diversity_boxplot(value_matrix: pd.DataFrame, pairs: pd.DataFrame, out_dir: Path, metric: str) -> None:
    """Plot end-mer Shannon diversity (entropy) by disease and extraction as boxplots."""
    pair_meta = pairs[["pair_id", "disease"]].drop_duplicates().set_index("pair_id")["disease"]

    entropy_rows = []
    for idx in value_matrix.index:
        pair_id, extraction = idx.split("|")
        disease = pair_meta.get(pair_id)
        if disease not in DISEASE_ORDER:
            continue
        entropy = _shannon_entropy(value_matrix.loc[idx])
        entropy_rows.append({
            "pair_id": pair_id,
            "disease": disease,
            "extraction": extraction,
            "entropy": entropy,
        })
    df = pd.DataFrame(entropy_rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "Hiragino Sans GB", "Heiti TC", "PingFang SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), sharey=True)
    for ax, disease in zip(axes, DISEASE_ORDER, strict=True):
        sub = df[df["disease"] == disease]
        if sub.empty:
            ax.set_title(f"{disease} (n=0)", fontweight="bold")
            ax.axis("off")
            continue
        data_to_plot = []
        labels = []
        colors = []
        for extraction in EXTRACTION_ORDER:
            vals = sub[sub["extraction"] == extraction]["entropy"].tolist()
            if vals:
                data_to_plot.append(np.array(vals, dtype=float))
                labels.append(f"{extraction}\n(n={len(vals)})")
                colors.append(EXTRACTION_COLORS_ENDMER[extraction])

        bp = ax.boxplot(data_to_plot, tick_labels=labels, patch_artist=True,
                        medianprops={"color": "black", "linewidth": 1.5})
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_title(disease, fontweight="bold", fontsize=12)
        ax.set_ylabel("Shannon entropy (bits)")
        ax.grid(axis="y", alpha=0.2)

    fig.suptitle(f"End-mer diversity ({metric}) by disease group", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / f"latest_endmer_diversity_boxplot_{metric}.png", dpi=220, bbox_inches="tight")
    fig.savefig(out_dir / f"latest_endmer_diversity_boxplot_{metric}.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Latest corrected paired end-mer analysis")
    parser.add_argument("--raw03-dir", type=Path, default=DEFAULT_RAW03_DIR)
    parser.add_argument("--stage1-b2-dir", type=Path, default=DEFAULT_STAGE1_B2_DIR)
    parser.add_argument("--first-fq-dir", type=Path, default=DEFAULT_FIRST_FQ_DIR)
    parser.add_argument("--raw03-manifest", type=Path, default=DEFAULT_RAW03_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--metric", choices=VALID_METRICS, default="normalized_5prime")
    parser.add_argument("--top-heatmap", type=int, default=80)
    parser.add_argument("--top-bar", type=int, default=20)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    inventory = load_inventory({
        "raw03_endmer": args.raw03_dir,
        "stage1_b2_endmer": args.stage1_b2_dir,
        "first_FQ_endmer": args.first_fq_dir,
    })
    # Propagate disease from Co to Single for first_FQ samples
    first_fq = inventory[inventory["source"] == "first_FQ_endmer"]
    single_mask = (first_fq["extraction"] == "single") & (first_fq["disease"].isna())
    if single_mask.any():
        co_map = (
            first_fq[first_fq["extraction"] == "co"]
            .drop_duplicates("pair_id")
            .set_index("pair_id")["disease"]
        )
        inventory.loc[first_fq[single_mask].index, "disease"] = first_fq.loc[single_mask, "pair_id"].map(co_map).values
    inventory = inventory[~inventory["pair_id"].astype(str).isin(EXCLUDED_SAMPLE_IDS)].copy()
    raw_manifest_samples = read_manifest_sample_names(args.raw03_manifest)
    pairs = build_correct_pairs(inventory, raw_manifest_samples)
    value_matrix = load_pair_value_matrix(pairs, args.metric)
    diff_matrix = load_difference_matrix(pairs, args.metric)

    inventory.to_csv(args.out_dir / "latest_end_mer_sample_inventory.tsv", sep="\t", index=False)
    pairs.to_csv(args.out_dir / "latest_end_mer_correct_pairs.tsv", sep="\t", index=False)
    plot_ordered_value_heatmap(value_matrix, pairs, args.out_dir, args.metric, args.top_heatmap)
    plot_difference_heatmap(diff_matrix, args.out_dir, args.metric, args.top_heatmap)
    plot_mean_difference_bar(diff_matrix, args.out_dir, args.metric, args.top_bar)
    plot_endmer_diversity_boxplot(value_matrix, pairs, args.out_dir, args.metric)

    print(f"Inventory samples: {len(inventory)}")
    print(f"Pairs: {pairs['pair_id'].nunique()}")
    print(f"Pairs by set: {pairs.groupby('pair_set')['pair_id'].nunique().to_dict()}")
    print(f"Pairs by disease: {pairs.groupby('disease')['pair_id'].nunique().to_dict()}")
    print(f"Saved outputs to: {args.out_dir}")


if __name__ == "__main__":
    main()
