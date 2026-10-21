# Stromal activation programme and C1QC⁺ macrophages in gastric cancer

Analysis code for the manuscript:

> **A stromal activation programme is associated with lower C1QC⁺ macrophage representation in gastric cancer: single-cell, spatial and bulk transcriptomic evidence.**
> Amir Arshia Beheshti, Meysam Motevasseli, Elham Safarzadeh. *Submitted to Frontiers in Genetics.*

The code re-analyses public single-cell RNA-seq (GSE150290, GSE183904), 10x Visium (GSE251950) and bulk microarray (GSE62254, GSE84437) data. It derives a tumour-level stromal activation programme (SAP) with multicellular factor analysis (MOFA+) of non-myeloid pseudobulk profiles and relates it to myeloid-state composition, transcriptional programmes, regulons, trajectories, ligand–receptor channels, spatial organisation and independent bulk cohorts.

---

## Repository structure

```
GC-SAP-TAM/
├── notebooks/
│   ├── 01_Figure1_atlas_SAP_composition.ipynb   # atlas, MOFA+/SAP, composition, DE, GSEA
│   ├── 02_Figure2_regulons_trajectories.ipynb   # pySCENIC, Palantir, CellRank
│   └── 05_Figure5_external_validation.ipynb     # bulk validation (ACRG, Yonsei) + Figure 5
├── scripts/
│   ├── cellphonedb/
│   │   └── GC_TAM_Fig4A_CPDB_SAP.py             # CellPhoneDB v5 + tumour-level channel tests
│   └── spatial/                                 # Visium pipeline, run in order 01 → 07
│       ├── fig4_01_download_qc_select.py
│       ├── fig4_02_cell2location.py             # GPU recommended
│       ├── fig4_03_scores_tests_niches.py
│       ├── fig4_04_channels.py
│       ├── fig4_05_niche_regression.py
│       ├── fig4_06_drug2cell.py
│       └── fig4_07_figures.py
├── env/                                         # exported conda environments
├── docs/
│   └── software_versions.md
├── CITATION.cff
├── LICENSE
└── README.md
```

## Code-to-figure map

Internal script and output names come from an earlier figure numbering. This table gives the mapping to the submitted manuscript.

| Manuscript | Code | Internal output names |
|---|---|---|
| Figure 1, Table 1, Supplementary Figures S1, S2, S5–S7 | `notebooks/01_…` | `GC_Fig1_outputs/`, `Fig1*`, `S1*`, `S2*` |
| Figure 2, Supplementary Figure S8 | `notebooks/02_…` | `GC_Fig3_outputs/`, `Fig3*` |
| Figure 3A | `scripts/cellphonedb/GC_TAM_Fig4A_CPDB_SAP.py` | `GC_Fig4_outputs/`, `Fig4A*` |
| Figures 3B–F, 4A–E, Table 2, Supplementary Figures S3, S4, S9, S10 | `scripts/spatial/fig4_01 … fig4_07` | `GC_Fig4_spatial_outputs/` |
| Figure 5 and bulk validation results | `notebooks/05_…` | `GC_SAP_validation_outputs/` |

## Data

All input data are public and are **not** included in this repository.

| Accession | Type | Use |
|---|---|---|
| GSE150290 | scRNA-seq, 10x 3′ v2 | discovery |
| GSE183904 | scRNA-seq, 10x 5′ | discovery |
| GSE251950 | 10x Visium | spatial analysis |
| GSE62254 (ACRG) | Affymetrix GPL570 | independent bulk validation |
| GSE84437 (Yonsei) | Illumina GPL6947 | independent bulk validation |

The spatial (step 01) and bulk-validation code download their GEO files automatically. The single-cell inputs and the pySCENIC resources (transcription-factor list, cisTarget ranking databases and motif annotation) are described in the header of each notebook.

## Setup

All paths are resolved relative to one project folder, set with the environment variable `GC_BASE`:

```bash
export GC_BASE=/path/to/project_folder
```

Outputs are written to sub-folders of `GC_BASE` (`GC_Fig1_outputs/`, `GC_Fig3_outputs/`, `GC_Fig4_outputs/`, `GC_Fig4_spatial_outputs/`, `GC_SAP_validation_outputs/`).

Two conda environments were used: `scvi-env` for the atlas, CellPhoneDB, spatial and validation analyses, and `pyscenic-env` for regulon and trajectory inference. The exported environment files are in `env/`. Package versions are listed in `docs/software_versions.md`. A GPU is strongly recommended for scVI and cell2location.

## Running the analysis

Run the steps in this order, because each step uses outputs of the previous ones:

```bash
# 1. Atlas, SAP and composition (Figure 1)
jupyter nbconvert --to notebook --execute notebooks/01_Figure1_atlas_SAP_composition.ipynb

# 2. Regulons and trajectories (Figure 2)
jupyter nbconvert --to notebook --execute notebooks/02_Figure2_regulons_trajectories.ipynb

# 3. Single-cell ligand–receptor analysis (Figure 3A)
python scripts/cellphonedb/GC_TAM_Fig4A_CPDB_SAP.py

# 4. Spatial pipeline (Figures 3B–F and 4)
for s in scripts/spatial/fig4_0{1..7}_*.py; do python "$s"; done

# 5. Independent bulk validation (Figure 5)
jupyter nbconvert --to notebook --execute notebooks/05_Figure5_external_validation.ipynb
```

Every step is checkpointed and resumes from disk when re-run. Spatial steps accept `--force` to recompute.

## Pre-specified analysis plans

The spatial pipeline writes its hypotheses, thresholds and section selection to `GC_Fig4_spatial_outputs/tables/S0_preregistration.json` before any test is run. The bulk validation writes `GC_SAP_validation_outputs/analysis_plan.json`, including SHA-256 hashes of its inputs, before testing, and `provenance.json` after the run. These plans were written into the code before testing and were not externally registered.

## Citation

If you use this code, please cite the article (see `CITATION.cff`). The archived release DOI will be added here.

## License

Code is released under the MIT License (see `LICENSE`). The public datasets are subject to the terms of their original sources.

## Contact

Corresponding author: Elham Safarzadeh (E.safarzadeh@arums.ac.ir)
