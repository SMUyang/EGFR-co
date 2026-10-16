#!/usr/bin/env python3
"""Workflow manifest to manuscript figure scripts command bridge.

Reads workflow manifest JSON (schema: paranomics/manifest@1),
translates output artifact contracts (recal_bam / endmer_csv / endmer_matrix) into
ready-to-run command lines for upstream figure scripts, printing them to stdout (pipeable to bash).

Usage:
    python adapter/paranomics_bridge.py <manifest.json> [--what endmer|figures|heatmap]
    python adapter/paranomics_bridge.py <manifest.json> --all | bash
"""
from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
UP = REPO / "upstream"

SUPPORTED_CONTRACTS = {"recal_bam", "endmer_csv", "endmer_matrix", "frag_features"}


def load_manifest(path: str | Path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema") != "paranomics/manifest@1":
        print(f"[WARN] unexpected manifest schema: {data.get('schema')}", file=sys.stderr)
    return data


def contract_paths(manifest: dict, contract: str) -> dict[str, str]:
    """Map sample ID to artifact path following samples[*].artifacts[contract] convention."""
    out: dict[str, str] = {}
    for sample, rec in manifest.get("samples", {}).items():
        art = rec.get("artifacts", {}) if isinstance(rec, dict) else {}
        if contract in art:
            out[sample] = art[contract]
    if not out:
        raise KeyError(f"manifest has no artifact '{contract}' (algorithm={manifest.get('algorithm')})")
    return out


def common_dir(paths: dict[str, str]) -> Path:
    """Find common parent directory for artifact paths (scripts consume directory arguments)."""
    dirs = {Path(p).parent.resolve() for p in paths.values()}
    while len(dirs) > 1:
        dirs = {d.parent for d in dirs}
    return dirs.pop()


def cmd_endmer_analysis(manifest: dict, *, threads: int = 4, mer_len: int = 4) -> list[str]:
    """recal_bam -> per-sample 5' end-motif frequency analysis (generates 256-motif CSV)."""
    bams = contract_paths(manifest, "recal_bam")
    outdir = REPO / "data" / "endmer"
    cmds = []
    for sample, bam in sorted(bams.items()):
        cmds.append(
            f"python {shlex.quote(str(UP / 'end_mer_analysis.py'))} "
            f"--bam {shlex.quote(bam)} "
            f"--output {shlex.quote(str(outdir / f'{sample}.end_mer.csv'))} "
            f"--plot {shlex.quote(str(outdir / f'{sample}.end_mer.png'))} "
            f"--mer-length {mer_len} --threads {threads} --mode fragment --min-mapq 20"
        )
    return cmds


def cmd_endmer_matrix(manifest: dict) -> list[str]:
    """endmer_csv directory -> paired value matrix & differential table (Panel D input)."""
    csvs = contract_paths(manifest, "endmer_csv")
    d = common_dir(csvs)
    outdir = REPO / "outputs" / "end_mer_latest_paired"
    return [
        f"python {shlex.quote(str(UP / 'end_mer_latest_paired_analysis.py'))} "
        f"--raw03-dir {shlex.quote(str(d))} "
        f"--stage1-b2-dir {shlex.quote(str(d))} "
        f"--first-fq-dir {shlex.quote(str(d))} "
        f"--out-dir {shlex.quote(str(outdir))}"
    ]


def cmd_figures(manifest: dict) -> list[str]:
    """endmer_csv directory -> Panel E (gc_bias_4mer) + Panel F (paired_fold_change)."""
    csvs = contract_paths(manifest, "endmer_csv")
    d = common_dir(csvs)
    return [
        f"python {shlex.quote(str(UP / 'analyze_task3_gc_bias_4mer.py'))} "
        f"--endmer-dir {shlex.quote(str(d))} "
        f"--out {shlex.quote(str(REPO / 'outputs' / 'gc_bias_4mer'))}",
        f"python {shlex.quote(str(UP / 'analyze_task3_endmer_paired_fold_change.py'))} "
        f"--endmer-dir {shlex.quote(str(d))} "
        f"--out {shlex.quote(str(REPO / 'outputs' / 'endmer_fold_change'))}",
    ]


def cmd_heatmap(manifest: dict, top_n: int = 128) -> list[str]:
    """endmer_matrix -> Panel D batch-corrected hierarchical clustered heatmap."""
    matrix = manifest.get("artifacts", {}).get(
        "endmer_matrix", str(REPO / "outputs" / "end_mer_latest_paired"
        / "latest_paired_ordered_value_matrix_normalized_5prime.tsv"))
    pairs = manifest.get("artifacts", {}).get(
        "endmer_pairs", str(REPO / "outputs" / "end_mer_latest_paired"
        / "latest_end_mer_correct_pairs.tsv"))
    return [
        f"python {shlex.quote(str(UP / 'end_mer_corrected_cluster_heatmap.py'))} "
        f"--value-matrix {shlex.quote(matrix)} "
        f"--pairs {shlex.quote(pairs)} "
        f"--out-dir {shlex.quote(str(REPO / 'outputs' / 'cluster_heatmap'))} "
        f"--top-n {top_n}"
    ]


BUILDERS = {
    "endmer": cmd_endmer_analysis,
    "matrix": cmd_endmer_matrix,
    "figures": cmd_figures,
    "heatmap": cmd_heatmap,
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("manifest", help="paranomics manifest JSON (stage1 / bam_qc_endmer / endmer_figures)")
    ap.add_argument("--what", choices=[*BUILDERS, "all"], default="all")
    args = ap.parse_args(argv)

    manifest = load_manifest(args.manifest)
    algo = manifest.get("algorithm", "?")
    targets = BUILDERS if args.what == "all" else {args.what: BUILDERS[args.what]}

    lines: list[str] = []
    for name, builder in targets.items():
        try:
            cmds = builder(manifest)
        except KeyError as exc:
            print(f"# skip {name}: {exc}", file=sys.stderr)
            continue
        lines.append(f"# === {name} (from manifest algorithm={algo}) ===")
        lines.extend(cmds)
        lines.append("")
    print("\n".join(lines), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
