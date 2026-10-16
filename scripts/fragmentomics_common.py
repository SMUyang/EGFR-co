#!/usr/bin/env python3
"""Shared utilities for the genome-wide fragmentomics pipeline.

Handles pair-table parsing, BAM path resolution, feature loading,
and streaming fragment processing common to all per-sample scripts.
"""

from __future__ import annotations

import bisect
import gzip
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

STANDARD_CHROMS = {f"chr{i}" for i in range(1, 23)} | {"chrX", "chrY"}
CHROMS = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]
LENGTH_BINS = ["short_90_150", "mono_151_220", "long_221_500"]
GC_BINS = 1001

import sys as _sys
from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import pathconfig as _pc

def _feat(name: str) -> str:
    """Feature BED path: paths.yaml features_root takes precedence, falling back to original relative path."""
    try:
        root = _pc.get("features_root")
        sub = {"ccre_all": "encode_ccre_v2_GRCh38_preferred_ENCFF420VPZ.bed.gz",
               "cpg_high_gc70": "generated/cpg_high_gc70.hg38.bed",
               "promoter_500bp": "generated/promoter_500bp.hg38.bed",
               "promoter_500bp_cpg": "generated/promoter_500bp_cpg.hg38.bed",
               "promoter_500bp_non_cpg": "generated/promoter_500bp_non_cpg.hg38.bed"}.get(name)
        if name == "cpg_all":
            return str(root / "generated" / "cpg_all.hg38.bed")
        if name == "dnase_footprint":
            return str(root / "vierstra_consensus_footprints_hg38.bed.gz")
        if sub:
            return str(root / sub)
    except KeyError:
        pass
    return None  # Fallback to original default table

SOURCE_TO_BAM_DIR = {
    "raw03_endmer": str(_pc.get("raw03_bam_dir", "data/bam/raw03")),
    "stage1_b2_endmer": str(_pc.get("stage1_b2_bam_dir", "data/bam/stage1_b2")),
    "first_FQ_endmer": str(_pc.get("first_fq_bam_dir", "data/bam/first_fq")),
}

_FEATURE_DEFAULTS = {
    "ccre_all": "data/features/encode_ccre_v2_GRCh38_preferred_ENCFF420VPZ.bed.gz",
    "cpg_all": "data/features/generated/cpg_all.hg38.bed",
    "cpg_high_gc70": "data/features/generated/cpg_high_gc70.hg38.bed",
    "promoter_500bp": "data/features/generated/promoter_500bp.hg38.bed",
    "promoter_500bp_cpg": "data/features/generated/promoter_500bp_cpg.hg38.bed",
    "promoter_500bp_non_cpg": "data/features/generated/promoter_500bp_non_cpg.hg38.bed",
    "dnase_footprint": "data/features/vierstra_consensus_footprints_hg38.bed.gz",
}

FEATURE_PATHS = {k: (_feat(k) or v) for k, v in _FEATURE_DEFAULTS.items()}


def assign_length_bin(length: int) -> str | None:
    if 90 <= length <= 150:
        return "short_90_150"
    if 151 <= length <= 220:
        return "mono_151_220"
    if 221 <= length <= 500:
        return "long_221_500"
    return None


def gc_prefix_for(seq: str) -> np.ndarray:
    arr = np.frombuffer(seq.upper().encode("ascii"), dtype=np.uint8)
    gc = (arr == ord("G")) | (arr == ord("C"))
    prefix = np.empty(len(gc) + 1, dtype=np.uint32)
    prefix[0] = 0
    np.cumsum(gc, dtype=np.uint32, out=prefix[1:])
    return prefix


def point_overlap(starts: list[int], ends: list[int], pos: int) -> bool:
    """Return whether pos overlaps merged, sorted half-open intervals."""
    idx = bisect.bisect_right(starts, pos) - 1
    return idx >= 0 and pos < ends[idx]


def open_text(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt")
    return path.open()


def load_pair_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t")


def resolve_bam_path(sample_name: str, source: str) -> Path:
    bam_dir = SOURCE_TO_BAM_DIR.get(source)
    if bam_dir is None:
        raise ValueError(f"unknown source '{source}' for sample '{sample_name}'")
    return Path(bam_dir) / f"{sample_name}.bam"


def slug(value: object) -> str:
    """Make a stable path-safe token for analysis ids."""
    text = str(value).strip()
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_") or "NA"


def extraction_token(extraction: str) -> str:
    if extraction == "Co-extraction":
        return "co"
    if extraction == "Single":
        return "single"
    return slug(extraction).lower()


def make_analysis_id(pair_id: object, extraction: str, sample: object, source: object) -> str:
    return "__".join([
        slug(pair_id),
        extraction_token(extraction),
        slug(sample),
        slug(source),
    ])


def build_sample_manifest(pair_table_path: Path) -> pd.DataFrame:
    """Return one row per analysis unit with source-aware analysis_id.

    `sample` is not unique in this project: a pair may contain the same visible
    sample name from different sources/extractions. `analysis_id` is therefore
    the workflow key and output-directory name.
    """
    pairs = load_pair_table(pair_table_path)
    rows = []
    for _, row in pairs.iterrows():
        for extraction, sample_col, source_col in [
            ("Single", "single_sample", "single_source"),
            ("Co-extraction", "co_sample", "co_source"),
        ]:
            sample = row[sample_col]
            source = row[source_col]
            pair_id = str(row["pair_id"])
            rows.append({
                "analysis_id": make_analysis_id(pair_id, extraction, sample, source),
                "sample": sample,
                "pair_id": pair_id,
                "extraction": extraction,
                "disease": row["disease"],
                "source": source,
                "bam": str(resolve_bam_path(sample, source)),
            })
    manifest = pd.DataFrame(rows)
    duplicates = manifest[manifest.duplicated("analysis_id", keep=False)]
    if len(duplicates):
        raise ValueError("duplicate analysis_id values:\n" + duplicates.to_string(index=False))
    return manifest.reset_index(drop=True)


def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge sorted/unsorted half-open intervals for safe point overlap queries."""
    merged: list[list[int]] = []
    for s, e in sorted(intervals):
        if e <= s:
            continue
        if not merged or s > merged[-1][1]:
            merged.append([s, e])
        else:
            merged[-1][1] = max(merged[-1][1], e)
    return [(s, e) for s, e in merged]


def load_feature_intervals(
    feature_names: list[str],
    path_overrides: dict[str, str] | None = None,
) -> dict[str, dict[str, tuple[list[int], list[int]]]]:
    features: dict[str, dict[str, tuple[list[int], list[int]]]] = {}
    overrides = path_overrides or {}
    for feature in feature_names:
        path_str = overrides.get(feature) or FEATURE_PATHS.get(feature)
        if path_str is None:
            raise ValueError(f"unknown feature '{feature}'")
        path = Path(path_str)
        by_chrom: dict[str, list[tuple[int, int]]] = defaultdict(list)
        with open_text(path) as handle:
            for line in handle:
                if not line.strip() or line.startswith("#"):
                    continue
                fields = line.rstrip("\n").split("\t")
                chrom = fields[0]
                if chrom not in STANDARD_CHROMS:
                    continue
                by_chrom[chrom].append((int(fields[1]), int(fields[2])))
        packed: dict[str, tuple[list[int], list[int]]] = {}
        raw_total = merged_total = 0
        for chrom in CHROMS:
            raw_intervals = by_chrom.get(chrom, [])
            intervals = merge_intervals(raw_intervals)
            raw_total += len(raw_intervals)
            merged_total += len(intervals)
            packed[chrom] = ([s for s, _ in intervals], [e for _, e in intervals])
        features[feature] = packed
        print(
            f"feature_loaded\t{feature}\traw_intervals={raw_total}\tmerged_intervals={merged_total}",
            flush=True,
        )
    return features


def threshold_from_hist(hist: np.ndarray, q: float) -> int:
    n = int(hist.sum())
    if n == 0:
        return -1
    return int(np.searchsorted(np.cumsum(hist), max(1, math.ceil(q * n)), side="left"))
