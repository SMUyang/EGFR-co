# EGFR-co: Comparative Fragmentomics Analysis (cfDNA Single Extraction vs. Co-Extraction)

[![Repository](https://img.shields.io/badge/GitHub-SMUyang%2FEGFR--co-blue)](https://github.com/SMUyang/EGFR-co)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> Analytical and figure generation codebase for the manuscript comparing **cell-free DNA (cfDNA) single extraction** versus **co-extraction (CoDNA)** across multiple fragmentomic dimensions (Panels A–F).

---

## Overview & Manuscript Figure Panels

This repository provides self-contained, reproducible Python scripts in `scripts/` to generate each panel of the manuscript main figure with unified publication-standard styling:

| Panel | Analytical Dimension | Target Output | Primary Script |
| :---: | :--- | :--- | :--- |
| **A** | Coverage profile & PCA | Whole-genome 1-kb depth PCA with 95% confidence ellipses (PC1 29.6% / PC2 23.2%) | `scripts/panel_a_pca.py` |
| **B** | Fragment size profile | cfDNA vs. CoDNA insert size distribution curves (50–450 bp, 167 bp peak) | `scripts/panel_b_insert_size.py` |
| **C** | Fragment length bin fractions | Paired comparisons across 90–150 bp, 151–220 bp, and 221–500 bp | `scripts/panel_c_fragment_bins.py` |
| **D** | 5' 4-mer motif clustering | Clustered heatmap with Retrospective & Prospective cohort annotations | `scripts/panel_d_cluster_heatmap.py` |
| **E** | Leading base 4-mer frequencies | 2x2 faceted boxplots for 5' leading bases (`A---`, `T---`, `C---`, `G---`) | `scripts/panel_e_mer_leading_bases.py` |
| **F** | Epigenomic feature enrichment | 1x4 subplots of high-GC fragment enrichment $\ln(\text{OR})$ across 4 regulatory features | `scripts/panel_f_epigenomic_enrichment.py` |
| **Full**| **Composite Main Figure** | **Assembles the complete 6-panel publication figure (Panels A–F in one layout)** | `scripts/generate_main_figure.py` |

---

## Standardized Terminology & Unified Color Palette

All figures strictly follow a **100% unified Nature-grade color palette** across every panel:

| Entity | Display Label | Hex Color Code | Visual Role |
| :--- | :--- | :--- | :--- |
| **Single extraction (cfDNA)** | `cfDNA` | `#524682` | **Deep Violet / Purple** (Panels A, B, C, D, E, F) |
| **Co-extraction (CoDNA)** | `CoDNA` | `#BD507D` | **Berry Pink / Rose Magenta** (Panels A, B, C, D, E, F) |
| **Retrospective cohort** | `Retrospective cohort` | `#8ECAE6` | **Soft Sky Blue** (Panel D cohort track) |
| **Prospective cohort** | `Prospective cohort` | `#264653` | **Deep Slate Navy** (Panel D cohort track) |
| **Paired trajectories** | Matched pair lines | `#A0A0A0` | Thin connecting line (`alpha=0.35`) between matched patient arms |

---

## Project Structure

```
EGFR-co/
├── scripts/                               # Core figure generation scripts (Panels A–F)
│   ├── panel_a_pca.py                     # Panel A: 1-kb depth PCA with confidence ellipses
│   ├── panel_b_insert_size.py             # Panel B: Insert size distribution curves
│   ├── panel_c_fragment_bins.py           # Panel C: Three-bin length comparison boxplots
│   ├── panel_d_cluster_heatmap.py         # Panel D: 5' 4-mer motif hierarchical clustermap
│   ├── panel_e_mer_leading_bases.py       # Panel E: 2x2 leading bases (A---, T---, C---, G---)
│   ├── panel_f_epigenomic_enrichment.py   # Panel F: 1x4 regulatory feature enrichment ln(OR)
│   ├── generate_main_figure.py            # Master script: Assembles complete 6-panel composite figure
│   ├── end_mer_analysis.py                # Core 256-motif frequency extractor from BAMs
│   ├── end_mer_latest_paired_analysis.py  # Paired matrix builder (Discovery & Validation)
│   └── queue_runner.py                    # Batch sample queue executor
├── queue.template.tsv                     # Standard sample queue interface template
├── pathconfig.py                          # Centralized 3-tier path resolution library
├── paths.yaml.example                     # Local path configuration template (Two cohorts)
├── data/                                  # Input data directory (git-ignored)
├── outputs/                               # Generated figures and tables (git-ignored)
│   └── main_figure/                       # figure_1_composite.png & figure_1_composite.pdf
├── tests/                                 # Automated unit tests (pytest)
├── README.md                              # Self-contained documentation & guide
└── LICENSE                                # MIT License
```

---

## Quick Start (3 Steps)

```bash
# 1. Clone repository and install dependencies
git clone https://github.com/SMUyang/EGFR-co.git
cd EGFR-co
python3 -m venv .venv && source .venv/bin/activate
pip install pyyaml pandas numpy scipy matplotlib seaborn pysam scikit-learn pytest

# 2. Configure data paths (optional; default relative paths in data/ and outputs/ work out-of-the-box)
cp paths.yaml.example paths.yaml
$EDITOR paths.yaml

# 3. Run automated tests and generate the complete 6-panel main figure
python3 -m pytest tests/ -q                              # 6 tests passed
python scripts/generate_main_figure.py --out outputs/main_figure  # Outputs figure_1_composite.png/.pdf
```

---

## Sample Queue Interface (`queue.template.tsv`)

For batch sample processing across cohorts:

```bash
# 1. Validate sample queue pairs (ensures each biological pair has Single & Co-extraction arms)
python scripts/queue_runner.py --queue queue.template.tsv --action validate

# 2. Extract 5' 4-mer end motifs for all queued BAM files
python scripts/queue_runner.py --queue queue.template.tsv --action endmer

# 3. Generate all manuscript figures (Panels A–F and composite figure)
python scripts/queue_runner.py --queue queue.template.tsv --action figures
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
