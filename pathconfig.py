"""Centralized path resolution with a three-tier fallback mechanism for reproducibility.

Resolution priority:
  1. Explicit configuration in paths.yaml
  2. Environment variables with prefix EGFR_CO_<KEY>
  3. Built-in script default values

Any script importing pathconfig can use ``P.get("key", default)`` instead of hardcoded paths.

To reproduce:
    cp paths.yaml.example paths.yaml   # Edit paths for local machine
or:
    export EGFR_CO_RAW03_DIR=/path/to/data/raw03
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
ENV_PREFIX = "EGFR_CO_"

# Alias mappings between manuscript standardized names and legacy identifiers
KEY_ALIASES: dict[str, list[str]] = {
    "validation_dir": ["validation_dir", "raw03_dir"],
    "raw03_dir": ["validation_dir", "raw03_dir"],
    "se": ["se", "stage1_b2_dir"],
    "stage1_b2_dir": ["se", "stage1_b2_dir"],
    "discovery": ["discovery", "discovery_dir", "first_fq_dir"],
    "discovery_dir": ["discovery", "discovery_dir", "first_fq_dir"],
    "first_fq_dir": ["discovery", "discovery_dir", "first_fq_dir"],
    "endmer_manifest": ["endmer_manifest", "raw03_manifest"],
    "raw03_manifest": ["endmer_manifest", "raw03_manifest"],
    "frag_v2_final_dir": ["frag_v2_final_dir", "frag_v2_fsinal_dir"],
    "frag_v2_fsinal_dir": ["frag_v2_fsinal_dir", "frag_v2_final_dir"],
}


def _load_yaml() -> dict[str, str]:
    path = REPO_ROOT / "paths.yaml"
    if not path.exists():
        return {}
    try:
        import yaml
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if isinstance(data, dict):
            return {str(k).lower(): str(v) for k, v in data.items() if v is not None and str(v).strip()}
        return {}
    except ImportError:
        return {}


def get(key: str, default: str | os.PathLike | None = None) -> Path:
    """Resolve a path key via three-tier fallback: paths.yaml -> environment variable -> default."""
    cfg = _load_yaml()
    candidates = KEY_ALIASES.get(key.lower(), [key.lower()])

    # Tier 1: paths.yaml (case-insensitive and alias-aware)
    for cand in candidates:
        if cand in cfg and cfg[cand].strip():
            return Path(cfg[cand]).expanduser()

    # Tier 2: Environment variables
    for cand in candidates:
        env = os.environ.get(f"{ENV_PREFIX}{cand.upper()}")
        if env and env.strip():
            return Path(env).expanduser()

    # Tier 3: Built-in default
    if default is not None:
        return Path(default)

    raise KeyError(
        f"path key '{key}' unset: set it in paths.yaml (copy paths.yaml.example) "
        f"or export {ENV_PREFIX}{key.upper()}"
    )


def configured_keys() -> dict[str, str]:
    """Return all active configurations (paths.yaml union env vars) for --show-paths debugging."""
    out: dict[str, str] = {}
    for k, v in _load_yaml().items():
        if str(v).strip():
            out[str(k)] = str(v)
    for k, v in os.environ.items():
        if k.startswith(ENV_PREFIX) and v.strip():
            out[k[len(ENV_PREFIX):].lower()] = v
    return out
