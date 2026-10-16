#!/usr/bin/env python3
"""Aggregate per-sample fragmentomics checkpoints into final tables."""
from __future__ import annotations
import argparse, math, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
try:
    from scripts.fragmentomics_common import CHROMS, GC_BINS, LENGTH_BINS, threshold_from_hist
except ImportError:
    try:
        from upstream.fragmentomics_common import CHROMS, GC_BINS, LENGTH_BINS, threshold_from_hist
    except ImportError:
        from fragmentomics_common import CHROMS, GC_BINS, LENGTH_BINS, threshold_from_hist

LOCAL_GC_BINS = 100
FIXED_60_BIN = 600
EXPECTED_CHROMS = set(CHROMS)


def parse_hist(text):
    return np.fromiter((int(x) for x in str(text).split(",")), dtype=np.int64, count=GC_BINS)


def parse_2d(text):
    arr = np.zeros((GC_BINS, LOCAL_GC_BINS), dtype=np.int64)
    for i, row in enumerate(str(text).split(";")):
        if i >= GC_BINS or not row:
            continue
        for j, v in enumerate(row.split(",")):
            if j < LOCAL_GC_BINS:
                arr[i, j] = int(v)
    return arr


def validate_sample_dir(sd: Path):
    chroms = {p.stem for p in sd.glob("chr*.tsv") if p.is_file() and p.stat().st_size > 0}
    missing = sorted(EXPECTED_CHROMS - chroms)
    extra = sorted(chroms - EXPECTED_CHROMS)
    if missing:
        raise RuntimeError(f"incomplete per-sample directory {sd}: missing chromosome files {missing}")
    if extra:
        print(f"warning\textra chromosome-like files in {sd}: {extra}", flush=True)


def load_sample(sd, feature_names):
    validate_sample_dir(sd)
    gc_hist = {lb: np.zeros(GC_BINS, dtype=np.int64) for lb in LENGTH_BINS}
    ov_hist = {f: {lb: np.zeros(GC_BINS, dtype=np.int64) for lb in LENGTH_BINS} for f in feature_names}
    loc2d = {lb: np.zeros((GC_BINS, LOCAL_GC_BINS), dtype=np.int64) for lb in LENGTH_BINS}
    loc_ov2d = {f: {lb: np.zeros((GC_BINS, LOCAL_GC_BINS), dtype=np.int64) for lb in LENGTH_BINS} for f in feature_names}
    flank2d = {lb: np.zeros((GC_BINS, LOCAL_GC_BINS), dtype=np.int64) for lb in LENGTH_BINS}
    flank_ov2d = {f: {lb: np.zeros((GC_BINS, LOCAL_GC_BINS), dtype=np.int64) for lb in LENGTH_BINS} for f in feature_names}
    assign = {lb: {} for lb in LENGTH_BINS}
    sum_gc = {lb: 0.0 for lb in LENGTH_BINS}
    sum_len = {lb: 0.0 for lb in LENGTH_BINS}
    meta = {}
    for cp in sorted(sd.glob("chr*.tsv")):
        try:
            df = pd.read_csv(cp, sep="\t", encoding="latin-1")
        except Exception:
            print(f"warn\tskip_unreadable\t{cp.name}", flush=True)
            continue
        for _, row in df.iterrows():
            lb = row.get("length_bin")
            if lb not in LENGTH_BINS:
                continue
            if not isinstance(row.get("gc_hist"), str):
                continue
            gc_hist[lb] += parse_hist(row["gc_hist"])
            sum_gc[lb] += float(row["sum_gc"])
            sum_len[lb] += float(row["sum_len"])
            for f in feature_names:
                c = f"overlap_hist_{f}"
                if c in row:
                    ov_hist[f][lb] += parse_hist(row[c])
                c2 = f"local_overlap_2d_{f}"
                if c2 in row and isinstance(row[c2], str):
                    loc_ov2d[f][lb] += parse_2d(row[c2])
                c3 = f"flank_overlap_2d_{f}"
                if c3 in row and isinstance(row[c3], str):
                    flank_ov2d[f][lb] += parse_2d(row[c3])
            if isinstance(row.get("local_2d"), str):
                loc2d[lb] += parse_2d(row["local_2d"])
            if isinstance(row.get("flank_2d"), str):
                flank2d[lb] += parse_2d(row["flank_2d"])
            for key in row.index:
                if str(key).startswith("assign_"):
                    a = str(key)[7:]
                    assign[lb][a] = assign[lb].get(a, 0) + int(row[key])
            if not meta:
                meta = {
                    "analysis_id": row.get("analysis_id", sd.name),
                    "sample": row.get("sample", sd.name),
                    "source": row.get("source", "Unknown"),
                    "extraction": row["extraction"],
                    "disease": row["disease"],
                    "pair_id": str(row["pair_id"]),
                }
    if not meta:
        raise RuntimeError(f"no rows loaded from {sd}")
    thr = {lb: threshold_from_hist(gc_hist[lb], 0.90) for lb in LENGTH_BINS}
    return {"meta": meta, "gc_hist": gc_hist, "ov_hist": ov_hist, "loc2d": loc2d, "loc_ov2d": loc_ov2d, "flank2d": flank2d, "flank_ov2d": flank_ov2d, "assign": assign, "sum_gc": sum_gc, "sum_len": sum_len, "thr": thr}


def mh_or(mask, loc2d, loc_ov2d):
    """Mantel-Haenszel OR stratified by local-GC bin, with Robins-Breslow-Greenland CI.

    Strata = local GC bins (columns). Within each stratum:
      a = high-GC & overlap, b = high-GC & no-overlap,
      c = background & overlap, d = background & no-overlap.
    OR_MH = sum(a*d/n) / sum(b*c/n) = R / S.
    Var(ln OR_MH) uses the RBG estimator (valid for sparse strata), replacing the
    earlier sqrt(1/R + 1/S) approximation which underestimates variance.
    """
    bg = ~mask
    a = loc_ov2d[mask, :].sum(axis=0).astype(float)
    b = loc2d[mask, :].sum(axis=0).astype(float) - a
    c = loc_ov2d[bg, :].sum(axis=0).astype(float)
    d = loc2d[bg, :].sum(axis=0).astype(float) - c
    n = a + b + c + d
    v = n > 0
    if v.sum() == 0:
        return {"or": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
                "se_log_or": float("nan"), "n_high": 0, "n_bg": 0,
                "high_overlap": 0, "bg_overlap": 0}
    a, b, c, d, n = a[v], b[v], c[v], d[v], n[v]
    r = a * d / n
    s = b * c / n
    R = float(np.sum(r))
    S = float(np.sum(s))
    or_val = R / S if S > 0 else float("inf")
    # Robins-Breslow-Greenland variance of ln(OR_MH)
    P = (a + d) / n
    Q = (b + c) / n
    if R > 0 and S > 0:
        var_log = (
            float(np.sum(P * r)) / (2 * R * R)
            + float(np.sum(P * s + Q * r)) / (2 * R * S)
            + float(np.sum(Q * s)) / (2 * S * S)
        )
        se = math.sqrt(var_log) if var_log > 0 else float("inf")
    else:
        se = float("inf")
    lo = math.log(max(or_val, 1e-10))
    return {
        "or": or_val,
        "ci_low": math.exp(lo - 1.96 * se) if se < float("inf") else 0.0,
        "ci_high": math.exp(lo + 1.96 * se) if se < float("inf") else float("inf"),
        "se_log_or": se if se < float("inf") else float("nan"),
        "n_high": int(a.sum() + b.sum()),
        "n_bg": int(c.sum() + d.sum()),
        "high_overlap": int(a.sum()),
        "bg_overlap": int(c.sum()),
    }


def base_row(meta):
    return {
        "analysis_id": meta["analysis_id"],
        "sample": meta["sample"],
        "source": meta["source"],
        "pair_id": meta["pair_id"],
        "extraction": meta["extraction"],
        "disease": meta["disease"],
    }


def strict_pair_rows(df: pd.DataFrame, value_col: str, out_col_prefix: str):
    rows = []
    keys = ["pair_id", "length_bin", "gc_definition", "feature"]
    for (pid, lb, gd, feat), g in df.groupby(keys):
        co = g[g["extraction"] == "Co-extraction"]
        si = g[g["extraction"] == "Single"]
        if len(co) != 1 or len(si) != 1:
            continue
        cv = co[value_col].iloc[0]
        sv = si[value_col].iloc[0]
        rows.append({
            "pair_id": pid,
            "disease": co["disease"].iloc[0],
            "length_bin": lb,
            "gc_definition": gd,
            "feature": feat,
            f"co_{out_col_prefix}": cv,
            f"single_{out_col_prefix}": sv,
            "delta": cv - sv,
        })
    return rows


def aggregate(pipeline_dir, out_dir, feature_names):
    out_dir.mkdir(parents=True, exist_ok=True)
    per_sample = pipeline_dir / "per_sample"
    sdirs = sorted(d for d in per_sample.iterdir() if d.is_dir() and (d / "done.flag").exists())
    print(f"samples\t{len(sdirs)}", flush=True)
    all_data = {}
    for sd in sdirs:
        print(f"loading\t{sd.name}", flush=True)
        data = load_sample(sd, feature_names)
        aid = data["meta"]["analysis_id"]
        if aid in all_data:
            raise ValueError(f"duplicate analysis_id while loading aggregate: {aid}")
        all_data[aid] = data

    idxs = np.arange(GC_BINS)
    thr_rows = []
    ov_rows = []
    mh_rows = []
    as_rows = []

    for aid, data in all_data.items():
        m = data["meta"]
        for lb in LENGTH_BINS:
            n = int(data["gc_hist"][lb].sum())
            q = data["thr"][lb]
            thr_rows.append({
                **base_row(m),
                "length_bin": lb,
                "n_fragments": n,
                "mean_gc": data["sum_gc"][lb] / n if n else float("nan"),
                "q90_gc": q / 1000 if q >= 0 else float("nan"),
            })
            f60m = idxs >= FIXED_60_BIN
            if q < 0:
                gd_masks = [("fixed_0p60", f60m)]
            else:
                gd_masks = [("q90", idxs >= q), ("fixed_0p60", f60m)]
            n_total = int(data["gc_hist"][lb].sum())
            for gd, mask in gd_masks:
                hn = int(data["gc_hist"][lb][mask].sum())
                bn = n_total - hn
                if gd == "fixed_0p60":
                    ta = sum(data["assign"][lb].values())
                    for feat, cnt in data["assign"][lb].items():
                        as_rows.append({
                            **base_row(m),
                            "length_bin": lb,
                            "gc_definition": gd,
                            "assigned_feature": feat,
                            "count": cnt,
                            "fraction": cnt / ta if ta else 0,
                        })
                for feat in feature_names:
                    oh = data["ov_hist"][feat][lb]
                    ho = int(oh[mask].sum())
                    bo = int(oh[~mask].sum())
                    hf = ho / hn if hn else 0
                    bf = bo / bn if bn else 0
                    ov_rows.append({
                        **base_row(m),
                        "length_bin": lb,
                        "gc_definition": gd,
                        "feature": feat,
                        "high_n": hn,
                        "high_overlap": ho,
                        "high_overlap_fraction": hf,
                        "background_n": bn,
                        "background_overlap": bo,
                        "background_overlap_fraction": bf,
                        "raw_enrichment_ratio": hf / bf if bf > 0 else float("inf"),
                        "raw_delta": hf - bf,
                    })
                    for control_name, c2d, cov2d in [
                        ("local_1kb", data["loc2d"][lb], data["loc_ov2d"][feat][lb]),
                        ("flank_500bp", data["flank2d"][lb], data["flank_ov2d"][feat][lb]),
                    ]:
                        mh = mh_or(mask, c2d, cov2d)
                        mh_rows.append({
                            **base_row(m),
                            "length_bin": lb,
                            "gc_definition": gd,
                            "feature": feat,
                            "local_gc_control": control_name,
                            **mh,
                        })

    thr_df = pd.DataFrame(thr_rows)
    ov_df = pd.DataFrame(ov_rows)
    mh_df = pd.DataFrame(mh_rows)
    as_df = pd.DataFrame(as_rows)

    thr_df.to_csv(out_dir / "gc_thresholds.tsv", sep="\t", index=False)
    ov_df.to_csv(out_dir / "raw_overlap_summary.tsv", sep="\t", index=False)
    as_df.to_csv(out_dir / "assignment_exclusive.tsv", sep="\t", index=False)
    mh_df.to_csv(out_dir / "matched_or_summary.tsv", sep="\t", index=False)

    delta_rows = strict_pair_rows(ov_df, "high_overlap_fraction", "fraction")
    pd.DataFrame(delta_rows).to_csv(out_dir / "paired_delta.tsv", sep="\t", index=False)

    mh_delta_rows = []
    for control_name, sub in mh_df.groupby("local_gc_control"):
        rows = strict_pair_rows(sub, "or", "or")
        for row in rows:
            row["local_gc_control"] = control_name
            co_or, single_or = row["co_or"], row["single_or"]
            row["log_or_delta"] = math.log(co_or) - math.log(single_or) if co_or > 0 and single_or > 0 and math.isfinite(co_or) and math.isfinite(single_or) else float("nan")
        mh_delta_rows.extend(rows)
    pd.DataFrame(mh_delta_rows).to_csv(out_dir / "matched_or_paired_delta.tsv", sep="\t", index=False)

    print(
        f"done\tsamples={len(all_data)}\toverlap_rows={len(ov_df)}\t"
        f"delta_rows={len(delta_rows)}\tmh_rows={len(mh_df)}\tmh_delta_rows={len(mh_delta_rows)}",
        flush=True,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pipeline-dir", default="data/fragmentomics_genome_wide_v2")
    p.add_argument("--out-dir", default="outputs/fragmentomics_final")
    p.add_argument("--features", default="ccre_all,cpg_all,cpg_high_gc70,promoter_500bp_cpg,promoter_500bp_non_cpg,dnase_footprint")
    a = p.parse_args()
    aggregate(Path(a.pipeline_dir), Path(a.out_dir), a.features.split(","))


if __name__ == "__main__":
    main()
