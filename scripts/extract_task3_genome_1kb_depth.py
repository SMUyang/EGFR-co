#!/usr/bin/env python3
"""Extract whole-genome fixed-bin fragment overlap depth for Task3 sources."""
from __future__ import annotations

import argparse
import gzip
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pysam
from dataclasses import dataclass


@dataclass
class FragmentBatch:
    starts: np.ndarray
    ends: np.ndarray

    @property
    def lengths(self) -> np.ndarray:
        return self.ends - self.starts

    def __len__(self) -> int:
        return len(self.starts)


def collect_chromosome_fragments(
    alignment: pysam.AlignmentFile,
    chromosome: str,
    chrom_length: int,
) -> FragmentBatch:
    """Collect fragment coordinates for one chromosome from high-quality proper-pair alignments."""
    starts: list[int] = []
    ends: list[int] = []
    for read in alignment.fetch(chromosome):
        if not getattr(read, "is_read1", False):
            continue
        if not getattr(read, "is_paired", False):
            continue
        if not getattr(read, "is_proper_pair", False):
            continue
        if getattr(read, "is_secondary", False):
            continue
        if getattr(read, "is_supplementary", False):
            continue
        if getattr(read, "is_qcfail", False):
            continue
        if getattr(read, "is_duplicate", False):
            continue
        if getattr(read, "is_unmapped", False):
            continue
        if getattr(read, "mate_is_unmapped", False):
            continue
        if getattr(read, "mapping_quality", 0) < 20:
            continue
        if read.reference_id != read.next_reference_id:
            continue
        tlen = read.template_length
        if tlen == 0:
            continue
        if tlen > 0:
            start = read.reference_start
        else:
            start = read.next_reference_start
        end = start + abs(tlen)
        if start < 0 or end > chrom_length or end <= start:
            continue
        starts.append(start)
        ends.append(end)
    return FragmentBatch(
        starts=np.array(starts, dtype=np.int64),
        ends=np.array(ends, dtype=np.int64),
    )

SOURCES = ("cf", "ev", "ec")
OUTPUT_COLUMNS = ["pair", "chrom", "bin_start", "bin_end", "bin_width", "cf_raw_depth", "ev_raw_depth", "ec_raw_depth"]
MANIFEST_COLUMNS = ["pair", "source", "bam"]


def _validate_bin_size(bin_size: int) -> int:
    if int(bin_size) <= 0:
        raise ValueError("bin_size must be positive")
    return int(bin_size)


def _bin_depths(starts: np.ndarray, ends: np.ndarray, chrom_length: int, bin_size: int) -> list[dict]:
    bin_size = _validate_bin_size(bin_size)
    chrom_length = int(chrom_length)
    if chrom_length <= 0:
        return []
    diff = np.zeros(chrom_length + 1, dtype=np.int64)
    if len(starts):
        np.add.at(diff, starts.astype(np.int64), 1)
        np.add.at(diff, ends.astype(np.int64), -1)
    coverage = np.cumsum(diff[:-1], dtype=np.int64)
    cumulative = np.concatenate(([0], np.cumsum(coverage, dtype=np.int64)))
    rows: list[dict] = []
    for bin_start in range(0, chrom_length, bin_size):
        bin_end = min(bin_start + bin_size, chrom_length)
        width = bin_end - bin_start
        overlap_sum = int(cumulative[bin_end] - cumulative[bin_start])
        rows.append({"bin_start": bin_start, "bin_end": bin_end, "bin_width": width, "raw_depth": overlap_sum / width})
    return rows


def collect_bam_contig_depth(bam_path: str | Path, chromosome: str, *, bin_size: int = 1000) -> list[dict]:
    """Extract every bin's mean overlap depth for one BAM contig."""
    bam_path = Path(bam_path)
    with pysam.AlignmentFile(str(bam_path), "rb") as alignment:
        if chromosome not in alignment.references:
            return []
        chrom_length = int(alignment.get_reference_length(chromosome))
        fragments = collect_chromosome_fragments(alignment, chromosome, chrom_length)
    rows = _bin_depths(fragments.starts, fragments.ends, chrom_length, bin_size)
    return [{"chrom": chromosome, **row} for row in rows]


def discover_shared_contigs(bam_paths: Iterable[str | Path]) -> list[str]:
    """Return sorted contig names present in every input BAM."""
    shared: set[str] | None = None
    for bam_path in bam_paths:
        with pysam.AlignmentFile(str(bam_path), "rb") as alignment:
            names = set(alignment.references)
        shared = names if shared is None else shared.intersection(names)
    return sorted(shared or ())


def _contig_lengths(bam_paths: Iterable[str | Path], contigs: Iterable[str]) -> dict[str, int]:
    wanted = set(contigs)
    lengths: dict[str, int] = {}
    for bam_path in bam_paths:
        with pysam.AlignmentFile(str(bam_path), "rb") as alignment:
            for chrom in wanted:
                if chrom in alignment.references:
                    length = int(alignment.get_reference_length(chrom))
                    lengths[chrom] = length if chrom not in lengths else min(lengths[chrom], length)
    return lengths


def _extract_source(source: str, bam_path: Path, contigs: list[str], lengths: dict[str, int], bin_size: int) -> dict[str, list[dict]]:
    result: dict[str, list[dict]] = {}
    with pysam.AlignmentFile(str(bam_path), "rb") as alignment:
        for chrom in contigs:
            chrom_length = lengths[chrom]
            fragments = collect_chromosome_fragments(alignment, chrom, chrom_length)
            result[chrom] = _bin_depths(fragments.starts, fragments.ends, chrom_length, bin_size)
    return result


def _normalize_pair(value: object) -> int | str:
    text = str(value).strip()
    try:
        number = int(text)
    except (TypeError, ValueError):
        return text
    return number if str(number) == text else text


def _normalize_manifest(manifest: str | Path | pd.DataFrame) -> pd.DataFrame:
    frame = pd.read_csv(manifest, sep="\t") if not isinstance(manifest, pd.DataFrame) else manifest.copy()
    missing = {"source", "bam"}.difference(frame.columns)
    if missing:
        raise ValueError(f"manifest missing required columns: {', '.join(sorted(missing))}")
    if "pair" not in frame.columns:
        frame.insert(0, "pair", 1)
    frame = frame[MANIFEST_COLUMNS].copy()
    frame["pair"] = frame["pair"].map(_normalize_pair)
    frame["source"] = frame["source"].astype(str).str.strip().str.lower()
    if not set(frame["source"]).issubset(SOURCES):
        raise ValueError(f"manifest source must be one of {SOURCES}")
    if frame.duplicated(["pair", "source"]).any():
        raise ValueError("manifest must contain at most one BAM per pair and source")
    for pair, group in frame.groupby("pair", sort=False):
        missing_sources = set(SOURCES).difference(group["source"])
        if missing_sources:
            raise ValueError(f"manifest pair {pair!r} missing sources: {', '.join(sorted(missing_sources))}")
    source_order = {name: index for index, name in enumerate(SOURCES)}
    return frame.sort_values(["pair", "source"], key=lambda values: values.map(source_order) if values.name == "source" else values.map(str), kind="stable").reset_index(drop=True)


def _run_source_job(job: tuple[object, str, Path, list[str], dict[str, int]], bin_size: int) -> tuple[object, str, dict[str, list[dict]]]:
    pair, source, bam_path, contigs, lengths = job
    return pair, source, _extract_source(source, bam_path, contigs, lengths, bin_size)


def extract_depth_table(manifest: str | Path | pd.DataFrame, *, bin_size: int = 1000, threads: int = 1) -> pd.DataFrame:
    """Extract deterministic all-shared-contig depth rows independently for each pair."""
    bin_size = _validate_bin_size(bin_size)
    threads = max(1, int(threads))
    frame = _normalize_manifest(manifest)
    pair_info: dict[object, tuple[list[str], dict[str, int]]] = {}
    jobs: list[tuple[object, str, Path, list[str], dict[str, int]]] = []
    for pair, group in frame.groupby("pair", sort=False):
        paths = {row.source: Path(row.bam) for row in group.itertuples(index=False)}
        contigs = discover_shared_contigs(paths.values())
        lengths = _contig_lengths(paths.values(), contigs)
        pair_info[pair] = (contigs, lengths)
        jobs.extend((pair, source, paths[source], contigs, lengths) for source in SOURCES)
    worker_count = min(threads, len(jobs)) if jobs else 1
    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        results = list(pool.map(lambda job: _run_source_job(job, bin_size), jobs))
    by_pair: dict[object, dict[str, dict[str, list[dict]]]] = {}
    for pair, source, source_result in results:
        by_pair.setdefault(pair, {})[source] = source_result
    rows: list[dict] = []
    for pair in frame["pair"].drop_duplicates().tolist():
        contigs, _ = pair_info[pair]
        source_results = by_pair[pair]
        for chrom in contigs:
            n_bins = len(source_results[SOURCES[0]][chrom])
            for index in range(n_bins):
                base = source_results[SOURCES[0]][chrom][index]
                row = {"pair": pair, "chrom": chrom, "bin_start": base["bin_start"], "bin_end": base["bin_end"], "bin_width": base["bin_width"]}
                for source in SOURCES:
                    row[f"{source}_raw_depth"] = source_results[source][chrom][index]["raw_depth"]
                rows.append(row)
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def _atomic_write_table(frame: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        with gzip.open(temporary, "wt", encoding="utf-8", newline="") as handle:
            frame.to_csv(handle, sep="\t", index=False, columns=OUTPUT_COLUMNS)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def write_genome_1kb_depth(manifest: str | Path | pd.DataFrame, output: str | Path, *, bin_size: int = 1000, threads: int = 1, manifest_output: str | Path | None = None) -> Path:
    """Extract and atomically write compressed depth table plus normalized manifest."""
    output = Path(output)
    frame = extract_depth_table(manifest, bin_size=bin_size, threads=threads)
    _atomic_write_table(frame, output)
    manifest_frame = _normalize_manifest(manifest)
    manifest_path = Path(manifest_output) if manifest_output is not None else output.with_name("genome_1kb_depth_manifest.tsv")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_frame.to_csv(manifest_path, sep="\t", index=False, columns=MANIFEST_COLUMNS)
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest-output", type=Path)
    parser.add_argument("--bin-size", type=int, default=1000)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    manifest_output = write_genome_1kb_depth(args.manifest, args.output, bin_size=args.bin_size, threads=args.threads, manifest_output=args.manifest_output)
    with gzip.open(args.output, "rt", encoding="utf-8") as handle:
        rows = max(0, sum(1 for _ in handle) - 1)
    print(f"depth_written\t{args.output}\trows={rows}\tmanifest={manifest_output}", flush=True)


if __name__ == "__main__":
    main()
