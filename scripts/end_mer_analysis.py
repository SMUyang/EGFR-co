#!/usr/bin/env python3
"""BAM end-mer analysis with reference genome context normalization.

This is a repo-local implementation used by bam_qc_endmer.smk so the workflow
does not depend on an external script path.

Normalization method (following Jiang et al., 2020):
1. Count observed (O) 4-mer frequencies at 5'/3' fragment ends from the BAM
2. Compute expected (E) 4-mer frequencies from a sliding-window scan of the
   reference genome (or target BED regions for capture data)
3. Calculate O/E ratio for each of the 256 motifs
4. Normalize: normalized_pct = (O/E) / sum(all O/E) * 100

Outputs:
- CSV with observed counts, raw %, O/E ratio, and normalized %
- PNG plot for the top normalized 5' end k-mers

The implementation uses integer-encoded k-mer counting for lower Python
overhead and better scalability on large BAMs.
"""

import argparse
import hashlib
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pysam


BASE2INT = {
    'A': 0, 'C': 1, 'G': 2, 'T': 3,
    'a': 0, 'c': 1, 'g': 2, 't': 3,
}
INT2BASE = ['A', 'C', 'G', 'T']

ASCII2INT = np.full(256, 255, dtype=np.uint8)
ASCII2INT[ord('A')] = 0
ASCII2INT[ord('C')] = 1
ASCII2INT[ord('G')] = 2
ASCII2INT[ord('T')] = 3
ASCII2INT[ord('a')] = 0
ASCII2INT[ord('c')] = 1
ASCII2INT[ord('g')] = 2
ASCII2INT[ord('t')] = 3


RC_TABLE = str.maketrans("ACGTacgtNn", "TGCAtgcaNn")


def reverse_complement(seq):
    return seq.translate(RC_TABLE)[::-1].upper()


def encode_mer_from_seq(seq, mer_length=4, from_start=True):
    """Encode a k-mer into an integer. Return None if invalid or too short."""
    if seq is None or len(seq) < mer_length:
        return None

    mer = seq[:mer_length] if from_start else seq[-mer_length:]
    code = 0
    for ch in mer:
        val = BASE2INT.get(ch)
        if val is None:
            return None
        code = (code << 2) | val
    return code


def decode_mer(code, mer_length=4):
    """Decode integer code back to a k-mer string."""
    chars = []
    for _ in range(mer_length):
        chars.append(INT2BASE[code & 0b11])
        code >>= 2
    return ''.join(reversed(chars))


def count_kmers_in_seq(sequence, mer_length=4):
    """Count valid k-mers in a sequence using vectorized 2-bit encoding."""
    n_mers = 4 ** mer_length
    counts = np.zeros(n_mers, dtype=np.int64)
    if sequence is None or len(sequence) < mer_length:
        return counts

    seq_bytes = np.frombuffer(sequence.encode('ascii'), dtype=np.uint8)
    values = ASCII2INT[seq_bytes]
    valid_mask = values != 255

    if mer_length != 4:
        for start in range(len(sequence) - mer_length + 1):
            window = values[start:start + mer_length]
            if np.any(window == 255):
                continue
            code = 0
            for value in window:
                code = (code << 2) | int(value)
            counts[code] += 1
        return counts

    window_valid = (
        valid_mask[:-3]
        & valid_mask[1:-2]
        & valid_mask[2:-1]
        & valid_mask[3:]
    )
    if not np.any(window_valid):
        return counts

    codes = (
        (values[:-3].astype(np.uint16) << 6)
        | (values[1:-2].astype(np.uint16) << 4)
        | (values[2:-1].astype(np.uint16) << 2)
        | values[3:].astype(np.uint16)
    )
    valid_codes = codes[window_valid]
    counts += np.bincount(valid_codes, minlength=n_mers)
    return counts


def analyze_end_mer(bam_file, mer_length=4, threads=1, progress_interval=1000000):
    print(f"Analyzing: {bam_file}")
    print(f"  mer length: {mer_length}, decompression threads: {threads}")

    n_mers = 4 ** mer_length
    mer_5prime = np.zeros(n_mers, dtype=np.int64)
    mer_3prime = np.zeros(n_mers, dtype=np.int64)
    total_reads = 0

    with pysam.AlignmentFile(bam_file, "rb", threads=max(1, threads)) as bam:
        for read in bam.fetch(until_eof=True):
            if read.is_secondary or read.is_supplementary or read.is_unmapped:
                continue

            seq = read.query_sequence
            if seq:
                code_5 = encode_mer_from_seq(seq, mer_length, from_start=True)
                if code_5 is not None:
                    mer_5prime[code_5] += 1

                code_3 = encode_mer_from_seq(seq, mer_length, from_start=False)
                if code_3 is not None:
                    mer_3prime[code_3] += 1

            total_reads += 1
            if progress_interval and total_reads % progress_interval == 0:
                print(f"  Processed: {total_reads/1e6:.1f}M reads")

    print(f"  Total processed: {total_reads} reads")
    return mer_5prime, mer_3prime, total_reads
def analyze_fragment_end_mer(
    bam_file,
    reference_fasta,
    mer_length=4,
    threads=1,
    progress_interval=1000000,
    min_mapq=20,
    max_insert_size=1000,
):
    """
    Count fragment end motifs from paired-end BAM using genomic fragment coordinates.

    This is preferred for cfDNA fragmentomics.
    """
    print(f"Analyzing fragment ends: {bam_file}")
    print(f"  reference: {reference_fasta}")
    print(f"  mer length: {mer_length}, decompression threads: {threads}")

    n_mers = 4 ** mer_length
    mer_left = np.zeros(n_mers, dtype=np.int64)
    mer_right = np.zeros(n_mers, dtype=np.int64)

    total_records = 0
    total_fragments = 0
    skipped = 0

    with pysam.AlignmentFile(bam_file, "rb", threads=max(1, threads)) as bam, \
            pysam.FastaFile(reference_fasta) as fa:

        for read in bam.fetch(until_eof=True):
            total_records += 1

            if read.is_secondary or read.is_supplementary:
                skipped += 1
                continue

            if read.is_unmapped or read.mate_is_unmapped:
                skipped += 1
                continue

            if not read.is_paired or not read.is_proper_pair:
                skipped += 1
                continue

            if read.mapping_quality < min_mapq:
                skipped += 1
                continue

            # Count each fragment once.
            # In a normal FR paired-end BAM, read1 and read2 both represent the same template.
            # Use only read1 to avoid double counting.
            if not read.is_read1:
                continue

            if read.reference_id != read.next_reference_id:
                skipped += 1
                continue

            tlen = abs(read.template_length)
            if tlen == 0 or tlen > max_insert_size:
                skipped += 1
                continue

            chrom = bam.get_reference_name(read.reference_id)

            # template_length is positive for the leftmost read in normal cases,
            # but using reference_start and tlen is safer after selecting read1 only
            # if read1 is not always leftmost. Therefore use template boundaries:
            if read.template_length > 0:
                frag_start = read.reference_start
                frag_end = read.reference_start + read.template_length
            else:
                frag_end = read.reference_end
                frag_start = frag_end + read.template_length

            ok = add_fragment_end_mers_from_reference(
                fa=fa,
                chrom=chrom,
                start=frag_start,
                end=frag_end,
                mer_length=mer_length,
                mer_5prime=mer_left,
                mer_3prime=mer_right,
            )

            if ok:
                total_fragments += 1
            else:
                skipped += 1

            if progress_interval and total_records % progress_interval == 0:
                print(
                    f"  Processed: {total_records/1e6:.1f}M records, "
                    f"{total_fragments/1e6:.1f}M fragments"
                )

    print(f"  Total records scanned: {total_records}")
    print(f"  Total fragments counted: {total_fragments}")
    print(f"  Skipped records/fragments: {skipped}")

    return mer_left, mer_right, total_fragments

def plot_mer_distribution(mer_counts, mer_length, title, output_file, top_n=20):
    nonzero_codes = np.nonzero(mer_counts)[0]
    if len(nonzero_codes) == 0:
        print(f"  No mer counts available for plot: {output_file}")
        return

    sorted_codes = nonzero_codes[np.argsort(mer_counts[nonzero_codes])[::-1]]
    top_codes = sorted_codes[:top_n]

    mers = [decode_mer(int(code), mer_length) for code in top_codes]
    counts = [int(mer_counts[code]) for code in top_codes]

    total = int(mer_counts.sum())
    percentages = [c / total * 100 if total > 0 else 0 for c in counts]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(range(len(mers)), percentages, color='steelblue')
    ax.set_xlabel(f'{mer_length}-mer sequence')
    ax.set_ylabel('Percentage (%)')
    ax.set_title(title)
    ax.set_xticks(range(len(mers)))
    ax.set_xticklabels(mers, rotation=45, ha='right')

    for bar, pct in zip(bars, percentages):
        height = bar.get_height()
        ax.annotate(
            f'{pct:.1f}%',
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords='offset points',
            ha='center',
            va='bottom',
            fontsize=8,
        )

    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"  Chart saved: {output_file}")

def encode_mer_direct(mer):
    """Encode an exact k-mer string into an integer. Return None if invalid."""
    code = 0
    for ch in mer:
        val = BASE2INT.get(ch)
        if val is None:
            return None
        code = (code << 2) | val
    return code


def add_fragment_end_mers_from_reference(
    fa,
    chrom,
    start,
    end,
    mer_length,
    mer_5prime,
    mer_3prime,
):
    """
    Count fragment genomic end motifs.

    left end:
        reference[start : start + k]

    right end:
        reverse_complement(reference[end - k : end])

    This treats both fragment ends as 5' end motifs of the two DNA strands.
    """
    if start < 0 or end <= start or end - start < mer_length:
        return False

    chrom_len = fa.get_reference_length(chrom)
    if start + mer_length > chrom_len or end - mer_length < 0:
        return False

    left_mer = fa.fetch(chrom, start, start + mer_length).upper()
    right_mer = reverse_complement(fa.fetch(chrom, end - mer_length, end)).upper()

    code_left = encode_mer_direct(left_mer)
    code_right = encode_mer_direct(right_mer)

    if code_left is not None:
        mer_5prime[code_left] += 1

    if code_right is not None:
        mer_3prime[code_right] += 1

    return code_left is not None or code_right is not None

def _expected_cache_path(reference_fasta, mer_length=4, regions_bed=None):
    cache_root = Path(reference_fasta).resolve().parent / ".end_mer_expected_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    key = f"{Path(reference_fasta).resolve()}|{mer_length}|{Path(regions_bed).resolve() if regions_bed else 'whole_genome'}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    return cache_root / f"expected_{digest}.csv"


def compute_expected_mers(reference_fasta, mer_length=4, regions_bed=None):
    """Compute expected 4-mer frequencies from reference genome via sliding window.

    For capture sequencing, provide regions_bed to restrict the scan to probe
    targeted regions.
    """
    n_mers = 4 ** mer_length
    cache_path = _expected_cache_path(reference_fasta, mer_length, regions_bed)
    if cache_path.exists():
        cached = pd.read_csv(cache_path)
        if {"mer", "expected_freq"}.issubset(cached.columns) and len(cached) == n_mers:
            print(f"  Using cached expected frequencies: {cache_path}")
            return cached["expected_freq"].to_numpy(dtype=np.float64)

    expected = np.zeros(n_mers, dtype=np.int64)

    print(f"  Scanning reference for expected {mer_length}-mer frequencies...")
    if regions_bed:
        print(f"  Using target regions: {regions_bed}")

    # Build set of target regions if BED provided
    target_regions = None
    if regions_bed and os.path.exists(regions_bed):
        target_regions = {}
        with open(regions_bed, 'r') as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith('#') or line.startswith('track'):
                    continue
                fields = line.split('\t')
                if len(fields) >= 3:
                    chrom = fields[0]
                    start = int(fields[1])
                    end = int(fields[2])
                    target_regions.setdefault(chrom, []).append((start, end))
        # Sort intervals per chromosome
        for chrom in target_regions:
            target_regions[chrom].sort()

    with pysam.FastaFile(reference_fasta) as fa:
        chromosomes = fa.references
        for chrom in chromosomes:
            # Determine which positions to scan
            if target_regions is not None:
                if chrom not in target_regions:
                    continue
                for start, end in target_regions[chrom]:
                    interval_start = max(0, start)
                    interval_end = max(interval_start, end)
                    seq = fa.fetch(chrom, interval_start, interval_end)
                    expected += count_kmers_in_seq(seq, mer_length)
            else:
                seq = fa.fetch(chrom)
                expected += count_kmers_in_seq(seq, mer_length)

    total = int(expected.sum())
    if total == 0:
        print("  WARNING: No valid k-mers found in reference scan")
        return np.ones(n_mers, dtype=np.float64) / n_mers

    print(f"  Reference scan: {total:,} valid {mer_length}-mers found")
    expected_freq = expected.astype(np.float64) / total
    pd.DataFrame(
        {
            "mer": [decode_mer(code, mer_length) for code in range(n_mers)],
            "expected_freq": expected_freq,
        }
    ).to_csv(cache_path, index=False)
    print(f"  Cached expected frequencies: {cache_path}")
    return expected_freq


def plot_mer_distribution_normalized(normalized_5prime, mer_length, title, output_file, top_n=20):
    """Plot normalized (O/E) motif frequencies."""
    nonzero_codes = np.nonzero(normalized_5prime)[0]
    if len(nonzero_codes) == 0:
        print(f"  No normalized values available for plot: {output_file}")
        return

    sorted_codes = nonzero_codes[np.argsort(normalized_5prime[nonzero_codes])[::-1]]
    top_codes = sorted_codes[:top_n]

    mers = [decode_mer(int(code), mer_length) for code in top_codes]
    values = [float(normalized_5prime[code]) for code in top_codes]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(range(len(mers)), values, color='steelblue')
    ax.axhline(y=100.0 / len(normalized_5prime), color='red', linestyle='--', alpha=0.5, label='Uniform (100/256)')
    ax.set_xlabel(f'{mer_length}-mer sequence')
    ax.set_ylabel('Normalized frequency (%)')
    ax.set_title(title)
    ax.set_xticks(range(len(mers)))
    ax.set_xticklabels(mers, rotation=45, ha='right')
    ax.legend()

    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.annotate(
            f'{val:.2f}%',
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords='offset points',
            ha='center',
            va='bottom',
            fontsize=8,
        )

    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"  Chart saved: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='BAM end-mer analysis with reference-genome O/E normalization'
    )
    parser.add_argument('--bam', required=True, help='Input BAM file')
    parser.add_argument('--output', required=True, help='Output CSV file')
    parser.add_argument('--plot', required=True, help='Output chart file (normalized)')
    parser.add_argument('--mer-length', type=int, default=4, help='mer length (default: 4)')
    parser.add_argument('--threads', type=int, default=1, help='pysam decompression threads')
    parser.add_argument('--top-n', type=int, default=20, help='Show top N 5prime mers')
    parser.add_argument('--reference', type=str, default='', help='Reference FASTA for expected frequency calculation')
    parser.add_argument('--regions', type=str, default='', help='BED file of target regions (for capture sequencing)')
    parser.add_argument(
    '--mode',
    choices=['read', 'fragment'],
    default='fragment',
    help='End-mer mode: read uses read.query_sequence; fragment uses paired-end genomic fragment ends'
)

    parser.add_argument(
        '--min-mapq',
        type=int,
        default=20,
        help='Minimum MAPQ for fragment mode'
    )

    parser.add_argument(
        '--max-insert-size',
        type=int,
        default=1000,
        help='Maximum template length for fragment mode'
    )
    args = parser.parse_args()

    if not os.path.exists(args.bam):
        print(f"Error: BAM file not found {args.bam}")
        sys.exit(1)

    # Step 1: Observed counts from BAM
    if args.mode == 'fragment':
        if not args.reference or not os.path.exists(args.reference):
            print("Error: --reference is required in fragment mode")
            sys.exit(1)

        mer_5prime, mer_3prime, total_units = analyze_fragment_end_mer(
            args.bam,
            args.reference,
            args.mer_length,
            threads=args.threads,
            min_mapq=args.min_mapq,
            max_insert_size=args.max_insert_size,
         )
        unit_name = "fragments"
    else:
        mer_5prime, mer_3prime, total_units = analyze_end_mer(
        args.bam,
        args.mer_length,
        threads=args.threads,
    )
    unit_name = "reads"

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    os.makedirs(os.path.dirname(args.plot), exist_ok=True)

    # Step 2: Expected frequencies from reference genome
    n_mers = 4 ** args.mer_length
    has_reference = bool(args.reference and os.path.exists(args.reference))
    if has_reference:
        regions_bed = args.regions if args.regions else None
        expected_freq = compute_expected_mers(args.reference, args.mer_length, regions_bed)
    else:
        # No reference provided — uniform expectation (no normalization)
        expected_freq = np.ones(n_mers, dtype=np.float64) / n_mers
        print("  WARNING: No reference provided; O/E normalization skipped")

    # Step 3: O/E ratio
    total_5 = float(mer_5prime.sum())
    total_3 = float(mer_3prime.sum())
    obs_freq_5 = mer_5prime.astype(np.float64) / total_5 if total_5 > 0 else np.zeros(n_mers, dtype=np.float64)
    obs_freq_3 = mer_3prime.astype(np.float64) / total_3 if total_3 > 0 else np.zeros(n_mers, dtype=np.float64)

    # Avoid division by zero
    safe_expected = np.where(expected_freq > 0, expected_freq, np.inf)
    oe_ratio_5 = obs_freq_5 / safe_expected
    oe_ratio_3 = obs_freq_3 / safe_expected

    # Step 4: Normalize so that sum = 100%
    oe_sum_5 = oe_ratio_5.sum()
    oe_sum_3 = oe_ratio_3.sum()
    normalized_5 = oe_ratio_5 / oe_sum_5 * 100 if oe_sum_5 > 0 else np.zeros(n_mers, dtype=np.float64)
    normalized_3 = oe_ratio_3 / oe_sum_3 * 100 if oe_sum_3 > 0 else np.zeros(n_mers, dtype=np.float64)

    # Build output table
    data = []
    for code in range(n_mers):
        count_5 = int(mer_5prime[code])
        count_3 = int(mer_3prime[code])
        if count_5 == 0 and count_3 == 0 and expected_freq[code] == 0:
            continue

        mer = decode_mer(code, args.mer_length)
        pct_5 = count_5 / total_units * 100 if total_units > 0 else 0
        pct_3 = count_3 / total_units * 100 if total_units > 0 else 0
        data.append({
            'mer': mer,
            'count_5prime': count_5,
            'percent_5prime': round(pct_5, 6),
            'oe_ratio_5prime': round(float(oe_ratio_5[code]), 6),
            'normalized_5prime': round(float(normalized_5[code]), 6),
            'count_3prime': count_3,
            'percent_3prime': round(pct_3, 6),
            'oe_ratio_3prime': round(float(oe_ratio_3[code]), 6),
            'normalized_3prime': round(float(normalized_3[code]), 6),
            'expected_freq': round(float(expected_freq[code]), 8),
            'mode': args.mode,'unit': unit_name,
        })

    df = pd.DataFrame(data)
    if not df.empty:
        df = df.sort_values('normalized_5prime', ascending=False)
    df.to_csv(args.output, index=False)
    print(f"  Results saved: {args.output}")

    plot_mer_distribution_normalized(
        normalized_5,
        args.mer_length,
        f"5'-end {args.mer_length}-mer (normalized O/E, top {args.top_n})",
        args.plot,
        args.top_n,
    )

    print("Done!")


if __name__ == '__main__':
    main()
