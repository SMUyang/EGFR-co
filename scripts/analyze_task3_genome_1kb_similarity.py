"""Statistics for Task3 whole-genome 1 kb depth similarity.

The input is the extraction table ``genome_1kb_depth_by_bin.tsv.gz``.  Raw
coverage is retained, while similarity is computed on finite, non-negative
tracks normalized to unit sum over bins having signal in at least one arm.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {
    "chrom",
    "bin_start",
    "bin_end",
    "bin_width",
    "cf_raw_depth",
    "ev_raw_depth",
    "ec_raw_depth",
}
OUTPUT_NAMES = (
    "genome_1kb_similarity_by_bin.tsv.gz",
    "genome_1kb_similarity_by_chromosome.tsv",
    "genome_1kb_similarity_by_gc.tsv",
    "genome_1kb_similarity_summary.tsv",
)


@dataclass(frozen=True)
class AnalysisResult:
    per_bin: pd.DataFrame
    chromosome: pd.DataFrame
    gc: pd.DataFrame
    summary: pd.DataFrame


def normalize_track(values: Sequence[float] | np.ndarray) -> np.ndarray:
    """Return a finite, non-negative vector normalized to unit sum.

    An all-zero vector is represented by zeros.  Empty, non-finite, negative,
    or non-vector input is rejected so invalid coverage cannot become NaN.
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("track must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(arr)):
        raise ValueError("track values must be finite")
    if np.any(arr < 0):
        raise ValueError("track values must be non-negative")
    total = float(arr.sum())
    return np.zeros_like(arr) if total == 0.0 else arr / total


def position_similarity(
    a: Sequence[float] | np.ndarray,
    b: Sequence[float] | np.ndarray,
    epsilon: float = 1e-12,
) -> dict[str, float]:
    """Compare two tracks with normalized L1 similarity in ``[0, 1]``."""
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    if aa.ndim != 1 or bb.ndim != 1 or aa.size != bb.size:
        raise ValueError("tracks must be one-dimensional arrays of equal length")
    if not np.isfinite(epsilon) or epsilon < 0:
        raise ValueError("epsilon must be a finite non-negative number")
    delta = normalize_track(aa) - normalize_track(bb)
    absolute = float(np.abs(delta).sum())
    if absolute <= epsilon:
        absolute = 0.0
    return {
        "similarity": float(np.clip(1.0 - absolute / 2.0, 0.0, 1.0)),
        "signed_difference": float(np.dot(delta, np.arange(delta.size) - (delta.size - 1) / 2.0)),
        "absolute_difference": absolute,
    }


def virtual_merged_track(cf: Sequence[float] | np.ndarray, ev: Sequence[float] | np.ndarray) -> np.ndarray:
    """Build the normalized virtual merged track from raw ``CF + EV`` depth."""
    cf_arr = np.asarray(cf, dtype=np.float64)
    ev_arr = np.asarray(ev, dtype=np.float64)
    if cf_arr.ndim != 1 or ev_arr.ndim != 1 or cf_arr.size != ev_arr.size:
        raise ValueError("CF and EV tracks must be one-dimensional arrays of equal length")
    return normalize_track(cf_arr + ev_arr)


def _validate_input(table: pd.DataFrame, bin_size: int | None) -> None:
    missing = sorted(REQUIRED_COLUMNS - set(table.columns))
    if missing:
        raise ValueError(f"missing required columns: {', '.join(missing)}")
    if table.empty:
        raise ValueError("input table must not be empty")
    if bin_size is not None and bin_size <= 0:
        raise ValueError("bin_size must be positive")
    for column in ("cf_raw_depth", "ev_raw_depth", "ec_raw_depth"):
        values = pd.to_numeric(table[column], errors="coerce").to_numpy(dtype=float)
        if not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError(f"{column} must contain finite non-negative values")


def _metric_columns(ec: np.ndarray, cf: np.ndarray, ev: np.ndarray, merged: np.ndarray) -> dict[str, float]:
    ec_cf = position_similarity(ec, cf)
    ec_ev = position_similarity(ec, ev)
    ec_merged = position_similarity(ec, merged)
    best_single = max(ec_cf["similarity"], ec_ev["similarity"])
    return {
        "ec_vs_cf_similarity": ec_cf["similarity"],
        "ec_vs_ev_similarity": ec_ev["similarity"],
        "ec_vs_merged_similarity": ec_merged["similarity"],
        "ec_vs_cf_l1": ec_cf["absolute_difference"],
        "ec_vs_ev_l1": ec_ev["absolute_difference"],
        "ec_vs_merged_l1": ec_merged["absolute_difference"],
        "merged_gain_vs_best_single": ec_merged["similarity"] - best_single,
    }


def _summarize(group: pd.DataFrame, group_name: str, group_value: str) -> dict[str, object]:
    row: dict[str, object] = {group_name: group_value, "n_bins": int(len(group))}
    metrics = (
        "ec_vs_cf_similarity",
        "ec_vs_ev_similarity",
        "ec_vs_merged_similarity",
        "merged_gain_vs_best_single",
        "ec_cf_bin_similarity",
        "ec_ev_bin_similarity",
        "ec_merged_bin_similarity",
    )
    for metric in metrics:
        row[f"mean_{metric}"] = float(group[metric].mean())
        row[f"median_{metric}"] = float(group[metric].median())
    return row


def _analyze_group(table: pd.DataFrame, *, bin_size: int | None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    frame = table.copy()
    for column in ("cf_raw_depth", "ev_raw_depth", "ec_raw_depth"):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(float)
    valid = frame[["cf_raw_depth", "ev_raw_depth", "ec_raw_depth"]].sum(axis=1) > 0
    frame = frame.loc[valid].copy().reset_index(drop=True)
    if frame.empty:
        raise ValueError("input contains no non-zero bins")
    raw_cf = frame["cf_raw_depth"].to_numpy(dtype=np.float64)
    raw_ev = frame["ev_raw_depth"].to_numpy(dtype=np.float64)
    raw_ec = frame["ec_raw_depth"].to_numpy(dtype=np.float64)
    epsilon = 1e-12
    frame["ec_cf_bin_similarity"] = np.clip(
        1.0 - np.abs(raw_ec - raw_cf) / (raw_ec + raw_cf + epsilon), 0.0, 1.0
    )
    frame["ec_ev_bin_similarity"] = np.clip(
        1.0 - np.abs(raw_ec - raw_ev) / (raw_ec + raw_ev + epsilon), 0.0, 1.0
    )
    frame["ec_merged_bin_similarity"] = np.clip(
        1.0 - np.abs(raw_ec - (raw_cf + raw_ev)) / (raw_ec + raw_cf + raw_ev + epsilon),
        0.0,
        1.0,
    )

    cf = normalize_track(frame["cf_raw_depth"].to_numpy())
    ev = normalize_track(frame["ev_raw_depth"].to_numpy())
    ec = normalize_track(frame["ec_raw_depth"].to_numpy())
    merged = virtual_merged_track(frame["cf_raw_depth"].to_numpy(), frame["ev_raw_depth"].to_numpy())
    frame["cf_normalized_depth"] = cf
    frame["ev_normalized_depth"] = ev
    frame["ec_normalized_depth"] = ec
    frame["merged_normalized_depth"] = merged
    metrics = _metric_columns(ec, cf, ev, merged)
    for name, value in metrics.items():
        frame[name] = value

    chromosome_rows = [_summarize(group, "chrom", str(chrom)) for chrom, group in frame.groupby("chrom", sort=True)]
    chromosome = pd.DataFrame(chromosome_rows)
    gc_column = next((name for name in ("gc_fraction", "gc", "gc_content") if name in frame.columns), None)
    if gc_column is None:
        frame["gc_group"] = "unavailable"
    else:
        gc_values = pd.to_numeric(frame[gc_column], errors="coerce")
        frame["gc_group"] = gc_values.map(lambda value: f"{value:.2f}" if np.isfinite(value) else "unavailable")
    gc_rows = [_summarize(group, "gc_group", str(gc_group)) for gc_group, group in frame.groupby("gc_group", sort=True)]
    gc = pd.DataFrame(gc_rows)
    summary_row: dict[str, object] = {"n_bins": int(len(frame)), "n_chromosomes": int(frame["chrom"].nunique())}
    summary_row.update(metrics)
    for metric in (
        "ec_cf_bin_similarity",
        "ec_ev_bin_similarity",
        "ec_merged_bin_similarity",
    ):
        summary_row[f"mean_{metric}"] = float(frame[metric].mean())
        summary_row[f"median_{metric}"] = float(frame[metric].median())
    return frame, chromosome, gc, summary_row


def analyze_table(table: pd.DataFrame, *, bin_size: int | None = 1000, threads: int = 1) -> AnalysisResult:
    """Analyze an extraction table, independently for each optional pair."""
    _validate_input(table, bin_size)
    if threads <= 0:
        raise ValueError("threads must be positive")

    pair_aware = "pair" in table.columns
    if not pair_aware:
        frame, chromosome, gc, summary_row = _analyze_group(table, bin_size=bin_size)
        return AnalysisResult(frame, chromosome, gc, pd.DataFrame([summary_row]))

    source = table.copy()
    source["pair"] = source["pair"].astype(str)
    if source["pair"].eq("").any() or source["pair"].isna().any():
        raise ValueError("pair must contain non-empty values")
    per_bin_parts: list[pd.DataFrame] = []
    chromosome_parts: list[pd.DataFrame] = []
    gc_parts: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    for pair, group in source.groupby("pair", sort=True):
        per_bin, chromosome, gc, summary_row = _analyze_group(group, bin_size=bin_size)
        per_bin["pair"] = pair
        chromosome["pair"] = pair
        gc["pair"] = pair
        summary_row["pair"] = pair
        per_bin_parts.append(per_bin)
        chromosome_parts.append(chromosome)
        gc_parts.append(gc)
        summary_rows.append(summary_row)

    per_bin = pd.concat(per_bin_parts, ignore_index=True)
    chromosome = pd.concat(chromosome_parts, ignore_index=True)
    gc = pd.concat(gc_parts, ignore_index=True)
    overall: dict[str, object] = {
        "pair": "overall",
        "n_bins": int(len(per_bin)),
        "n_chromosomes": int(per_bin["chrom"].nunique()),
    }
    for metric in (
        "ec_vs_cf_similarity",
        "ec_vs_ev_similarity",
        "ec_vs_merged_similarity",
        "merged_gain_vs_best_single",
        "ec_cf_bin_similarity",
        "ec_ev_bin_similarity",
        "ec_merged_bin_similarity",
    ):
        overall[metric] = float(per_bin[metric].mean())
        overall[f"mean_{metric}"] = float(per_bin[metric].mean())
        overall[f"median_{metric}"] = float(per_bin[metric].median())
    summary_rows.append(overall)
    summary = pd.DataFrame(summary_rows)
    return AnalysisResult(per_bin, chromosome, gc, summary)


def write_outputs(result: AnalysisResult, output_dir: str | Path) -> list[Path]:
    """Write the exact four statistics artifacts and return their paths."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tables = (result.per_bin, result.chromosome, result.gc, result.summary)
    paths = [out / name for name in OUTPUT_NAMES]
    for table, path in zip(tables, paths):
        table.to_csv(path, sep="\t", index=False, compression="gzip" if path.suffix == ".gz" else None)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="genome_1kb_depth_by_bin.tsv.gz")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bin-size", type=int, default=1000)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    table = pd.read_csv(args.input, sep="\t", compression="infer")
    write_outputs(analyze_table(table, bin_size=args.bin_size, threads=args.threads), args.output_dir)


if __name__ == "__main__":
    main()
