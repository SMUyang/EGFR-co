#!/usr/bin/env python3
"""Standard sample queue runner for EGFR-co fragmentomics pipeline.

Parses sample queue manifests (TSV format, see queue.template.tsv), validates paired
sample integrity (Single vs. Co-extraction), and generates or executes commands for
motif counting, matrix generation, and manuscript figure rendering.

Usage:
    # Print commands for downstream execution
    python scripts/queue_runner.py --queue queue.tsv --action all

    # Execute end-mer extraction directly for queued samples
    python scripts/queue_runner.py --queue queue.tsv --action endmer --run
"""
from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"
REQUIRED_COLUMNS = {"sample_id", "pair_id", "extraction_method"}


def load_queue(path: str | Path) -> pd.DataFrame:
    """Load and validate sample queue TSV table."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Queue file not found: {p}")
    df = pd.read_csv(p, sep="\t")
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Queue file {p} missing required columns: {sorted(missing)}")
    return df


def validate_pairs(df: pd.DataFrame) -> dict[str, set[str]]:
    """Verify that each pair has both Single and Co-extraction arms."""
    pairs = df.groupby("pair_id")["extraction_method"].agg(set).to_dict()
    incomplete = {pid: methods for pid, methods in pairs.items() if not {"Single", "Co-extraction"}.issubset(methods)}
    if incomplete:
        print(f"[WARN] Incomplete pairs found (missing Single or Co-extraction arm): {incomplete}", file=sys.stderr)
    return pairs


def cmd_endmer(df: pd.DataFrame, *, threads: int = 4, reference: str | None = None) -> list[str]:
    """Generate end_mer_analysis.py commands for samples with bam_path."""
    if "bam_path" not in df.columns:
        print("[WARN] No 'bam_path' column in queue file; skipping BAM processing.", file=sys.stderr)
        return []

    cmds = []
    outdir = REPO / "data" / "endmer"
    ref_arg = f"--reference {shlex.quote(reference)} " if reference else ""

    for _, row in df.dropna(subset=["bam_path"]).iterrows():
        sample = row["sample_id"]
        bam = row["bam_path"]
        csv_out = row.get("endmer_csv") or str(outdir / f"{sample}.recal.end_mer.csv")
        plot_out = str(outdir / f"{sample}.end_mer.png")
        cmd = (
            f"python {shlex.quote(str(SCRIPTS / 'end_mer_analysis.py'))} "
            f"--bam {shlex.quote(str(bam))} "
            f"--output {shlex.quote(str(csv_out))} "
            f"--plot {shlex.quote(plot_out)} "
            f"{ref_arg}--mer-length 4 --threads {threads} --mode fragment --min-mapq 20"
        )
        cmds.append(cmd)
    return cmds


def cmd_figures(out_dir: str | Path | None = None) -> list[str]:
    """Generate figure commands for Panels A, B, C, D, E, F, and the full composite figure."""
    out_base = Path(out_dir) if out_dir else REPO / "outputs"
    return [
        f"python {shlex.quote(str(SCRIPTS / 'panel_a_pca.py'))} --out {shlex.quote(str(out_base / 'panel_a'))}",
        f"python {shlex.quote(str(SCRIPTS / 'panel_b_insert_size.py'))} --out-dir {shlex.quote(str(out_base / 'panel_b'))}",
        f"python {shlex.quote(str(SCRIPTS / 'panel_c_fragment_bins.py'))}",
        f"python {shlex.quote(str(SCRIPTS / 'panel_d_cluster_heatmap.py'))} --out-dir {shlex.quote(str(out_base / 'cluster_heatmap'))}",
        f"python {shlex.quote(str(SCRIPTS / 'panel_e_mer_leading_bases.py'))} --out {shlex.quote(str(out_base / 'panel_e'))}",
        f"python {shlex.quote(str(SCRIPTS / 'panel_f_epigenomic_enrichment.py'))} --out {shlex.quote(str(out_base / 'panel_f'))}",
        f"python {shlex.quote(str(SCRIPTS / 'generate_main_figure.py'))} --out {shlex.quote(str(out_base / 'main_figure'))}",
    ]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Sample queue runner for EGFR-co analyses")
    ap.add_argument("--queue", default="queue.template.tsv", help="Path to sample queue TSV file")
    ap.add_argument("--action", choices=["validate", "endmer", "figures", "all"], default="validate",
                    help="Action to execute: validate queue, run endmer, generate figures, or all")
    ap.add_argument("--threads", type=int, default=4, help="Threads for BAM processing")
    ap.add_argument("--reference", help="Path to reference FASTA for O/E normalization")
    ap.add_argument("--out-dir", help="Output directory for generated figures")
    ap.add_argument("--run", action="store_true", help="Execute generated commands via subprocess directly")

    args = ap.parse_args(argv)
    df = load_queue(args.queue)
    validate_pairs(df)

    if args.action == "validate":
        print(f"[OK] Queue {args.queue} valid: {len(df)} samples across {df['pair_id'].nunique()} pairs.")
        return 0

    cmds: list[str] = []
    if args.action in ("endmer", "all"):
        cmds.extend(cmd_endmer(df, threads=args.threads, reference=args.reference))
    if args.action in ("figures", "all"):
        cmds.extend(cmd_figures(out_dir=args.out_dir))

    if not cmds:
        print("[INFO] No commands generated.", file=sys.stderr)
        return 0

    if args.run:
        import subprocess
        for c in cmds:
            print(f"Executing: {c}")
            subprocess.run(c, shell=True, check=True)
    else:
        print("\n".join(cmds))

    return 0


if __name__ == "__main__":
    sys.exit(main())
