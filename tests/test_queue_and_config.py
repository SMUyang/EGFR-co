"""Test suite for EGFR-co pathconfig and queue runner."""
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import pathconfig
import queue_runner
import paranomics_bridge as bridge


# ── Queue runner tests ──

def test_queue_template_valid():
    tsv = REPO_ROOT / "queue.template.tsv"
    assert tsv.exists()
    df = queue_runner.load_queue(tsv)
    assert len(df) == 6
    pairs = queue_runner.validate_pairs(df)
    assert len(pairs) == 3
    assert set(pairs.keys()) == {"P01", "P02", "P03"}
    for pid, methods in pairs.items():
        assert {"Single", "Co-extraction"}.issubset(methods)


def test_queue_command_generation():
    tsv = REPO_ROOT / "queue.template.tsv"
    df = queue_runner.load_queue(tsv)
    endmer_cmds = queue_runner.cmd_endmer(df, threads=2)
    assert len(endmer_cmds) == 6
    assert "end_mer_analysis.py" in endmer_cmds[0]
    assert "--threads 2" in endmer_cmds[0]

    fig_cmds = queue_runner.cmd_figures()
    assert len(fig_cmds) == 7
    assert any("panel_a_pca.py" in c for c in fig_cmds)
    assert any("panel_b_insert_size.py" in c for c in fig_cmds)
    assert any("panel_c_fragment_bins.py" in c for c in fig_cmds)
    assert any("panel_d_cluster_heatmap.py" in c for c in fig_cmds)
    assert any("panel_e_mer_leading_bases.py" in c for c in fig_cmds)
    assert any("panel_f_epigenomic_enrichment.py" in c for c in fig_cmds)
    assert any("generate_main_figure.py" in c for c in fig_cmds)


# ── pathconfig reproducibility tests ──

def test_pathconfig_three_tier(tmp_path, monkeypatch):
    # 1. default fallback
    assert str(pathconfig.get("raw03_dir", "DEFAULT")) == "DEFAULT"
    # 2. environment variable overrides default
    monkeypatch.setenv("EGFR_CO_RAW03_DIR", "/their/env/path")
    assert str(pathconfig.get("raw03_dir", "DEFAULT")) == "/their/env/path"
    # 3. yaml overrides environment variable (write temporary yaml pointing to paths.yaml)
    yaml_path = pathconfig.REPO_ROOT / "paths.yaml"
    yaml_path.write_text("raw03_dir: /their/yaml/path\n")
    try:
        assert str(pathconfig.get("raw03_dir", "DEFAULT")) == "/their/yaml/path"
    finally:
        if yaml_path.exists():
            yaml_path.unlink()
    # 4. unset and no default -> KeyError
    monkeypatch.delenv("EGFR_CO_RAW03_DIR", raising=False)
    try:
        pathconfig.get("nonexistent_key_12345")
        raised = False
    except KeyError:
        raised = True
    assert raised


def test_pathconfig_new_template_aliases():
    """Verify paths.yaml using the standardized template keys resolves correctly."""
    yaml_path = pathconfig.REPO_ROOT / "paths.yaml"
    template_content = """
Validation_dir: /data/val
se: /data/se_dir
Discovery: /data/disc
endmer_manifest: /data/endmer/manifest.json
frag_v2_fSinal_dir: /data/frag_final
"""
    yaml_path.write_text(template_content)
    try:
        # Check both new key name and legacy aliases
        assert str(pathconfig.get("Validation_dir")) == "/data/val"
        assert str(pathconfig.get("raw03_dir")) == "/data/val"
        assert str(pathconfig.get("se")) == "/data/se_dir"
        assert str(pathconfig.get("stage1_b2_dir")) == "/data/se_dir"
        assert str(pathconfig.get("Discovery")) == "/data/disc"
        assert str(pathconfig.get("first_fq_dir")) == "/data/disc"
        assert str(pathconfig.get("endmer_manifest")) == "/data/endmer/manifest.json"
        assert str(pathconfig.get("raw03_manifest")) == "/data/endmer/manifest.json"
        assert str(pathconfig.get("frag_v2_final_dir")) == "/data/frag_final"
        assert str(pathconfig.get("frag_v2_fSinal_dir")) == "/data/frag_final"
    finally:
        if yaml_path.exists():
            yaml_path.unlink()


def test_fragmentomics_common_feature_override(monkeypatch):
    import importlib.util
    monkeypatch.setenv("EGFR_CO_FEATURES_ROOT", "/fx")
    spec = importlib.util.spec_from_file_location(
        "fc_test", str(REPO_ROOT / "scripts" / "fragmentomics_common.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    assert m.FEATURE_PATHS["ccre_all"] == "/fx/encode_ccre_v2_GRCh38_preferred_ENCFF420VPZ.bed.gz"
    assert m.FEATURE_PATHS["cpg_all"] == "/fx/generated/cpg_all.hg38.bed"


# ── Legacy manifest bridge compatibility tests ──

def _stage1_manifest(tmp_path):
    return {
        "schema": "paranomics/manifest@1",
        "algorithm": "stage1",
        "samples": {
            "cf1": {"artifacts": {"recal_bam": str(tmp_path / "cf1.recal.bam")}},
            "ec1": {"artifacts": {"recal_bam": str(tmp_path / "ec1.recal.bam")}},
        },
    }


def test_legacy_bridge_commands(tmp_path):
    m = _stage1_manifest(tmp_path)
    cmds = bridge.cmd_endmer_analysis(m)
    assert len(cmds) == 2
    assert "--bam" in cmds[0] and "end_mer_analysis.py" in cmds[0]
