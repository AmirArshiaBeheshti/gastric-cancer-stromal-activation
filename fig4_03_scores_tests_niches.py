#!/usr/bin/env python
# coding: utf-8
"""
 fig4_03_scores_tests_niches.py  (step 3 of 7)
 S4  spot scores: SAP (Fig. 1 axis-weight signature, bidirectional, residualised on stromal
     abundance -> stromal STATE), neighbourhood SAP, myeloid-state CLR (cell2location),
     label-free myeloid signatures, regulon scores, Hallmark/TAM-programme AUCell
 S5  H1-H5 (Dutilleul-corrected partial Spearman per section, pooled across sections)
     + 500 random stromal signatures (specificity; resumable)
 S6  distance-to-SAP-high gradients (hexagonal rings)
 S7  NMF tissue niches (stability-selected k) + neighbourhood enrichment
 RUN: python fig4_03_scores_tests_niches.py      (--force to redo)
"""

# ==============================================================================
# 0. IMPORTS / DEPENDENCIES
# ==============================================================================
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")   # less GPU fragmentation
import re
import sys
import json
import time
import gzip
import shutil
import tarfile
import warnings
import importlib
import subprocess
import urllib.request
from pathlib import Path


def ensure_pkg(pkg_import, pip_name=None):
    try:
        return importlib.import_module(pkg_import)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pip_name or pkg_import], check=True)
        return importlib.import_module(pkg_import)


for _imp, _pip in [("scanpy", "scanpy"), ("anndata", "anndata"), ("h5py", "h5py"),
                   ("sklearn", "scikit-learn"), ("PIL", "pillow"), ("adjustText", "adjustText")]:
    ensure_pkg(_imp, _pip)

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import stats
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment
import scanpy as sc
import anndata as ad
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.collections import EllipseCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
BASE = Path(os.environ.get("GC_BASE", "/home/arshia/payanname"))                                # project folder
FIG1 = BASE / "GC_Fig1_outputs"
FIG3 = BASE / "GC_Fig3_outputs"
FIG4A = BASE / "GC_Fig4_outputs"                                     # CellPhoneDB Fig. 4A (already run)
OUT = BASE / "GC_Fig4_spatial_outputs"
RAW = OUT / "raw"
H5 = OUT / "h5ad"
TAB = OUT / "tables"
MOD = OUT / "models"
FIG_MAIN = OUT / "figures" / "main"
FIG_SUPP = OUT / "figures" / "supp"
for _d in [OUT, RAW, H5, TAB, MOD, FIG_MAIN, FIG_SUPP]:
    _d.mkdir(parents=True, exist_ok=True)
LOGFILE = OUT / "run_log.txt"


def locate(path, *alternatives):
    """Expected location first; then the alternative names; then the same file name anywhere
    under BASE (up to 4 folders deep, outputs of this pipeline excluded)."""
    path = Path(path)
    for p in [path] + [path.parent / a for a in alternatives]:
        if p.exists():
            return p
    for name in [path.name] + list(alternatives):
        for depth in range(5):
            for hit in sorted(BASE.glob("/".join(["*"] * depth + [name]))):
                if hit.is_file() and OUT not in hit.parents:
                    return hit
    return path


# --- inputs: scRNA-seq results only (Figs 1, 3, 4A) ---
ATLAS_H5AD = FIG1 / "h5ad" / "S1d_cnv_v3.h5ad"                       # all lineages, layers['counts']
MYELOID_H5AD = locate(FIG1 / "h5ad" / "S1e_myeloid_v3_m3.h5ad",     # reviewed myeloid states (m3 labels)
                      "S1e_myeloid_final_v3_m3_amb20_F4.h5ad")
F_AXIS_W = FIG1 / "tables" / "S2_axis_weights_amb20_F4.csv"          # SAP axis weights (MOFA Factor 4, SAP-oriented)
F_FIG1E = FIG1 / "tables" / "Fig1E_composition_vs_axis_m3_amb20_F4.csv"
F_FIG1G = FIG1 / "tables" / "Fig1G_H_DE_results_v3_m3_amb20_F4.csv"
F_GSEA_MO = FIG1 / "tables" / "Fig1I_GSEA_Monocytederived_TAM_table.csv"
F_GSEA_RES = FIG1 / "tables" / "Fig1J_GSEA_Residentlike_TAM_table.csv"
F_DRIVERS = FIG3 / "tables" / "Fig3_lineage_drivers.csv"             # CellRank drivers -> fate signatures
F_REGULONS = FIG3 / "tables" / "Fig3_consensus_regulons_v2.csv"      # SCENIC consensus regulons
F_FIG3J = FIG3 / "tables" / "Fig3J_regulon_vs_SAP_within_state.csv"
F_FIG4A = FIG4A / "tables" / "Fig4A_tumour_level_channels_vs_SAP.csv"
# optional: a full MSigDB Hallmark library (json {set: [genes]}) if you have one; otherwise it is
# downloaded once with gseapy, and if that fails the Fig. 1I/J leading-edge genes are used
HALLMARK_JSON_CANDIDATE = locate(FIG1 / "resources" / "MSigDB_Hallmark_2020.json")
GENCODE_TSV = locate(FIG1 / "resources" / "gencode_v44_genes.tsv")  # optional (else IDs mapped via the atlas)
# --- signatures built by step 01 from the files above (used by all later steps) ---
F_SIGS = BASE / "GC_Fig4_spatial_outputs" / "tables" / "S0_signatures_spatial.csv"
F_PATHWAY = BASE / "GC_Fig4_spatial_outputs" / "tables" / "S0_programme_sets.csv"
HALLMARK_JSON = BASE / "GC_Fig4_spatial_outputs" / "tables" / "S0_hallmark_sets.json"
# SAP signature rule (built here from the Fig. 1 SAP axis weights and the Fig. 1 atlas):
SAP_VIEWS = ["Fibroblast", "Pericyte_SMC", "Endothelial"]
SAP_MIN_REL_WEIGHT = 0.25    # |w| >= 25% of the largest |w|
SAP_MAX_PER_ARM = 100
MONO_PANEL = ["FCN1", "VCAN", "S100A8", "S100A9", "S100A12", "CD300E"]
MYELOID_PANEL = ["CD14", "LYZ", "CD68", "AIF1", "CSF1R", "FCER1G", "TYROBP", "CD163", "C1QA", "C1QB"]
TRANSFER_MIN_RATIO = 1.0     # gene kept if its target compartment is the highest-expressing atlas lineage
# genes never used in signatures: cell cycle (Seurat cc.genes 2019 + extended list), lncRNA, MT/RP/IG/TR/HB
CC_GENES = set("""MCM5 PCNA TYMS FEN1 MCM7 MCM4 RRM1 UNG GINS2 MCM6 CDCA7 DTL PRIM1 UHRF1 CENPU HELLS RFC2 POLR1B
NASP RAD51AP1 GMNN WDR76 SLBP CCNE2 UBR7 POLD3 MSH2 ATAD2 RAD51 RRM2 CDC45 CDC6 EXO1 TIPIN DSCC1 BLM CASP8AP2 USP1
CLSPN POLA1 CHAF1B MRPL36 E2F8 MCM2 MLF1IP HMGB2 CDK1 NUSAP1 UBE2C BIRC5 TPX2 TOP2A NDC80 CKS2 NUF2 CKS1B MKI67
TMPO CENPF TACC3 PIMREG SMC4 CCNB2 CKAP2L CKAP2 AURKB BUB1 KIF11 ANP32E TUBB4B GTSE1 KIF20B HJURP CDCA3 JPT1
CDC20 TTK CDC25C KIF2C RANGAP1 NCAPD2 DLGAP5 CDCA2 CDCA8 ECT2 KIF23 HMMR AURKA PSRC1 ANLN LBR CKAP5 CENPE CTCF
NEK2 G2E3 GAS2L3 CBX5 CENPA FAM64A HN1""".split())
CELL_CYCLE_RE = re.compile(r"^(MKI67|TOP2A|CDK1|BIRC5|TK1|PCLAF|CCNA2|CCNB1|CCNB2|UBE2C|CENP[AEFW]|KIF\d+[A-Z]?|"
                           r"MCM\d|PCNA|TYMS|RRM2|AURK[AB]|PLK1|BUB1B?|CDC20|CDKN3|NUSAP1|TPX2|ZWINT|"
                           r"FOXM1|E2F[1-8]|HMGB2|STMN1|TUBA1B|H2AZ1|MAD2L1|PBK|MELK|DLGAP5|ASPM|DEPDC1|"
                           r"NCAPG2?|SPC2[45]|SHCBP1)$")
LNC_RE = re.compile(r"^(LINC\d|MIR\d+.*HG|.*-AS\d|HIST\d|LOC\d)")
F_FIG4A_LONG = FIG4A / "tables" / "Fig4A_cpdb_long_displayed_rows.csv"
CPDB_DB = FIG4A / "cpdb_db"
FIG4A_Q = 0.05               # Fig. 4A channels with tumour-level q < 0.05 are tested in space
MECH_PAIRS = ["IL1B_IL1_receptor", "CSF1_CSF1R", "IL34_CSF1R", "CXCL12_CXCR4", "LGALS3_MERTK"]  # Fig. 4A trends

# --- GSE251950 ---
GSE = "GSE251950"
URL_RAW = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE251nnn/GSE251950/suppl/GSE251950_RAW.tar"
URL_MATRIX = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE251nnn/GSE251950/matrix/GSE251950_series_matrix.txt.gz"
SAMPLES = {  # GSM -> (sample, site); verified against the GEO series record
    "GSM7990473": ("GC1", "primary"), "GSM7990474": ("GC2", "primary"),
    "GSM7990475": ("GC3", "primary"), "GSM7990476": ("GC4", "primary"),
    "GSM7990477": ("GC5", "primary"), "GSM7990478": ("GC6", "primary"),
    "GSM7990479": ("GC6-PM", "peritoneal_metastasis"), "GSM7990480": ("GC7", "primary"),
    "GSM7990481": ("GC8", "primary"), "GSM7990482": ("GC9", "primary"),
}
PRIMARY_ALL = [v[0] for v in SAMPLES.values() if v[1] == "primary"]
SPOT_DIAMETER_UM = 55.0
SEED = 0

# --- spot QC ---
QC_MIN_COUNTS, QC_MIN_GENES, QC_MAX_PCT_MT = 500, 250, 40.0

# --- cell2location ---
STROMA = ["Fibroblast", "Pericyte_SMC", "Endothelial"]
NONMYELOID_REF = ["Malignant", "Epithelial_NM", "T_NK", "B", "Plasma", "Mast"] + STROMA
MYELOID_STATES = ["Mono_classical", "TAM_Transitional", "TAM_Inflammatory", "TAM_SPP1", "TAM_C1QC",
                  "TAM_IFN", "TAM_Prolif", "cDC1", "cDC2", "mregDC", "pDC", "Neutrophil"]
MOMAC = ["Mono_classical", "TAM_Transitional", "TAM_Inflammatory", "TAM_SPP1", "TAM_C1QC",
         "TAM_IFN", "TAM_Prolif"]
MONO_TAM = ["TAM_SPP1", "TAM_Inflammatory", "TAM_IFN"]                # Fig. 1G compartment
RES_TAM = ["TAM_C1QC"]                                               # Fig. 1H compartment
REF_CAP_PER_LABEL, REF_CAP_PER_SAMPLE = 5000, 400                    # stratified down-sampling
REG_EPOCHS = 250
C2L_EPOCHS = 30000           # full-batch epochs (= gradient steps); mini-batch runs keep the same number of steps
C2L_BATCH_SIZES = [2500, 1250, 600]   # tried in this order when the full data set does not fit on the GPU
C2L_MIN_EPOCHS = 3000
POSTERIOR_BATCH = 500        # spots per batch when sampling the posterior (small = less GPU memory)
N_CELLS_PER_SPOT = 10        # gastric tumour Visium (55 um): ~5-20 nuclei; see sensitivity flag
DETECTION_ALPHA = 20         # recommended for sections with variable RNA detection
SENSITIVITY_N_CELLS = []     # e.g. [5, 20] -> re-maps with other priors (S3b; slow)

# --- spot-level analysis ---
MIN_STROMA_CELLS = 1.0       # cell2location posterior mean, sum of the three stromal lineages
MIN_MYELOID_CELLS = 1.0      # sum of the 7 monocyte/macrophage states
CLR_PSEUDO = 0.05            # cells
NEIGHBOUR_RADIUS_UM = 250.0  # rings 1-2 of the Visium hexagonal lattice (100 um pitch)
KERNEL_BW_UM = 100.0
MAX_LAG_UM, N_LAG_CLASSES = 1500.0, 15     # Dutilleul correlogram
MIN_SPOTS_TEST = 50
N_RANDOM_NULL = 500
MAX_RING = 6
HYP_STATES_FALLBACK = {"Mono_classical": +1, "TAM_Inflammatory": +1, "TAM_C1QC": -1, "TAM_Prolif": -1}
NMF_K_RANGE, NMF_RESTARTS = range(5, 13), 10

# --- Fig. 4A channels (source lineage -> target compartment); expected sign along SAP ---
# Tier "Fig4A" = every channel with tumour-level q < 0.05 in Fig. 4A (frozen from
# Fig4A_tumour_level_channels_vs_SAP.csv, strongest row per pair); tier "mechanistic" =
# channels named in the text with only a trend in Fig. 4A (tested in a separate BH family).
# Myeloid sender/receiver states are pooled into the monocyte/macrophage compartment
# ("MoMac"), because Visium spots cannot resolve individual myeloid states.
CHANNELS = [
    # activated arm (Fig. 4A rho > 0)
    ("FN1", "ITGAV_ITGB1", "Pericyte_SMC", "MoMac", +1, "Fig4A"),
    ("FN1", "ITGA5_ITGB1", "Pericyte_SMC", "MoMac", +1, "Fig4A"),
    ("ITGAV_ITGB1", "ADGRE2", "Pericyte_SMC", "MoMac", +1, "Fig4A"),
    ("CXCL1", "ACKR1", "MoMac", "Endothelial", +1, "Fig4A"),
    ("FBN1", "ITGA5_ITGB1", "Pericyte_SMC", "MoMac", +1, "Fig4A"),
    ("FN1", "ITGA4_ITGB1", "Pericyte_SMC", "MoMac", +1, "Fig4A"),
    ("VEGFA", "NRP1", "MoMac", "Pericyte_SMC", +1, "Fig4A"),
    ("THBS2", "CD36", "Pericyte_SMC", "MoMac", +1, "Fig4A"),
    ("THY1", "ADGRE2", "Pericyte_SMC", "MoMac", +1, "Fig4A"),
    ("THY1", "ITGAX_ITGB2", "Pericyte_SMC", "MoMac", +1, "Fig4A"),
    ("FN1", "ITGAV_ITGB5", "Pericyte_SMC", "MoMac", +1, "Fig4A"),
    ("IL1RN", "IL1R1_IL1RAP", "MoMac", "Pericyte_SMC", +1, "Fig4A"),
    ("TNC", "ITGA4_ITGB1", "Pericyte_SMC", "MoMac", +1, "Fig4A"),
    ("MPZL1", "MPZL1", "MoMac", "Fibroblast", +1, "Fig4A"),
    ("CD44", "SELE", "MoMac", "Endothelial", +1, "Fig4A"),
    # quiescent arm (Fig. 4A rho < 0)
    ("HEBP1", "FPR3", "Fibroblast", "MoMac", -1, "Fig4A"),
    ("APP", "TREM2_TYROBP", "Fibroblast", "MoMac", -1, "Fig4A"),
    ("CD320", "JAML", "Endothelial", "MoMac", -1, "Fig4A"),
    ("APOE", "TREM2_TYROBP", "Fibroblast", "MoMac", -1, "Fig4A"),
    ("JAM2", "ITGA4_ITGB1", "Endothelial", "MoMac", -1, "Fig4A"),
    ("OSM", "LIFR_IL6ST", "MoMac", "Endothelial", -1, "Fig4A"),
    ("ANXA1", "FPR3", "Fibroblast", "MoMac", -1, "Fig4A"),
    ("HLA-E", "VSIR", "Endothelial", "MoMac", -1, "Fig4A"),
    ("LRPAP1", "SORT1", "MoMac", "Endothelial", -1, "Fig4A"),
    ("RARRES2", "CCRL2", "Pericyte_SMC", "MoMac", -1, "Fig4A"),
    ("SEMA6D", "PLXNA1", "Endothelial", "MoMac", -1, "Fig4A"),
    ("TNFSF10", "TNFRSF10B", "Endothelial", "MoMac", -1, "Fig4A"),
    ("OSM", "OSMR_IL6ST", "MoMac", "Endothelial", -1, "Fig4A"),
    ("GAS6", "MERTK", "Endothelial", "MoMac", -1, "Fig4A"),
    ("CD248", "IL6R", "Endothelial", "MoMac", -1, "Fig4A"),
    ("GAS6", "AXL", "Endothelial", "MoMac", -1, "Fig4A"),
    ("APP", "CD74", "Endothelial", "MoMac", -1, "Fig4A"),
    ("CLU", "TREM2_TYROBP", "Pericyte_SMC", "MoMac", -1, "Fig4A"),
    # mechanistic channels with a Fig. 4A trend only (separate family)
    ("IL1B", "IL1R1_IL1RAP", "MoMac", "Pericyte_SMC", +1, "mechanistic"),
    ("CSF1", "CSF1R", "Endothelial", "MoMac", -1, "mechanistic"),
    ("IL34", "CSF1R", "Fibroblast", "MoMac", -1, "mechanistic"),
    ("CXCL12", "CXCR4", "Fibroblast", "MoMac", -1, "mechanistic"),
    ("LGALS3", "MERTK", "Endothelial", "MoMac", -1, "mechanistic"),
]
# --- Fig. 3J: regulons with FDR < 0.05 along SAP within myeloid states (sign of mean beta) ---
REGULON_DIR = {"RXRA(+)": +1, "VDR(+)": +1, "NR3C1(+)": +1, "TCF7L2(+)": +1, "BACH1(+)": +1, "STAT4(+)": +1,
               "MAF(+)": -1, "CEBPA(+)": -1, "MITF(+)": -1, "PPARG(+)": -1, "USF2(+)": -1, "HES1(+)": -1,
               "STAT1(+)": -1}
# --- S9 niche regression ---
MYELOID_SPEC_RATIO = 3.0     # reference expression in myeloid states / max non-myeloid lineage
MIN_COMP_CELLS = 0.5
N_COMP_PCS = 6
# --- S11 drug2cell: pre-specified mechanistic targets (Figs 1, 3, 4A) ---
MECH_TARGETS = {"CCR2", "IL1R1", "IL1B", "IL1A", "IL1RAP", "CSF1R", "ITGAV", "ITGB1", "ITGA5", "ITGA4",
                "TREM2", "MERTK", "AXL", "VEGFA", "NRP1", "KDR", "FLT1", "IL6", "IL6R", "JAK1", "JAK2",
                "STAT3", "CXCR4", "CD36", "PTGS2", "TNF", "NR3C1", "VDR", "RXRA", "PPARG", "CD274"}
# --- section selection (step 01) ---
N_SELECTED = 3               # number of primary sections analysed
MIN_COOCCUR_SPOTS = 500      # a section must have >= 500 spots with stromal AND myeloid marker signal
SELECT_OVERRIDE = []         # e.g. ["GC2", "GC5", "GC8"] to fix the choice by hand (leave [] for automatic)
STROMA_MARKERS = ["COL1A1", "COL1A2", "COL3A1", "DCN", "LUM", "PDGFRB", "RGS5", "ACTA2", "PECAM1", "VWF"]
MYELOID_MARKERS = ["CD68", "CD163", "C1QA", "C1QB", "C1QC", "LYZ", "CSF1R", "AIF1", "TYROBP", "FCGR3A"]

# --- pooling across sections ---
# With 3 sections the between-section variance (tau^2) cannot be estimated reliably, so sections
# are pooled by fixed-effect inverse-variance weighting (inference is conditional on these
# sections); the DerSimonian-Laird random-effects P is reported alongside (p_meta_RE) with I^2.
META_METHOD = "FE"           # "FE" (fixed effect, default for 3 sections) or "RE" (DerSimonian-Laird)

# --- S11 drug2cell (simplified, two transparent tiers) ---
D2C_MIN_DETECT = 0.02        # a target must be detected in >= 2% of spots of a section
D2C_MIN_SECTIONS = 2         # drug tested in >= 2 of the selected sections
D2C_TIER1 = dict(q=0.05, q_myeloid=0.05, share=0.50)                  # strict
D2C_TIER2 = dict(q=0.10, p_myeloid=0.05, share=0.30, consistent=True) # candidate

# --- checkpoints / recomputation ---
# Every step skips work that is already on disk. To recompute a step run it with --force
# (e.g.  python fig4_06_drug2cell.py --force) or delete GC_Fig4_spatial_outputs/checkpoints/<step>.done
FORCE = set()

# ==============================================================================
# 2. STYLE (identical to Figure 1)
# ==============================================================================
def _pick_font():
    from matplotlib import font_manager
    avail = {f.name for f in font_manager.fontManager.ttflist}
    for f in ["Arial", "Helvetica", "Liberation Sans", "Nimbus Sans", "DejaVu Sans"]:
        if f in avail:
            return f
    return "DejaVu Sans"


FONT = _pick_font()
mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": [FONT],
    "font.size": 7, "axes.titlesize": 8, "axes.titleweight": "bold",
    "axes.labelsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6,
    "legend.fontsize": 6, "legend.title_fontsize": 6.5,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "lines.linewidth": 0.8,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "savefig.dpi": 600, "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
MYELOID_PALETTE = {
    "TAM_C1QC": "#6A3D9A", "TAM_FOLR2": "#33A02C", "TAM_SPP1": "#E31A1C",
    "TAM_Inflammatory": "#FF9E4A", "TAM_IFN": "#F2D400", "TAM_Prolif": "#1A1A1A",
    "TAM_Transitional": "#A6761D", "Mono_classical": "#8C2D19", "Mono_nonclassical": "#C98B63",
    "cDC1": "#1F5FA8", "cDC2": "#4F8FD6", "mregDC": "#9ECAE9", "pDC": "#8C8C8C", "Neutrophil": "#BCBD22",
}
LINEAGE_PALETTE = {
    "Malignant": "#1F4E9C", "Epithelial_NM": "#8FB3E0", "T_NK": "#D62728", "B": "#3A89C9",
    "Plasma": "#8E44AD", "Mast": "#8C564B", "Fibroblast": "#E377C2", "Pericyte_SMC": "#7F9C3B",
    "Endothelial": "#E6A117",
}
CT_PALETTE = {**LINEAGE_PALETTE, **MYELOID_PALETTE}
C_ACT, C_QUI, C_GREY, C_DARK = "#B2182B", "#2166AC", "#8C8C8C", "#262626"
SAP_CMAP = "RdBu_r"
ABUND_CMAP = "magma"
HALLMARK_CMAP = LinearSegmentedColormap.from_list(
    "hallmark", ["#FFFBEA", "#FDD5A0", "#F07C6C", "#B8337A", "#4B0C5E"])


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOGFILE, "a") as fh:
        fh.write(line + "\n")


# ==============================================================================
# 3. STATISTICS
# ==============================================================================
def bh(p):
    p = np.asarray(p, float)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if ok.sum() == 0:
        return q
    pv = p[ok]
    o = np.argsort(pv)
    r = pv[o] * len(pv) / np.arange(1, len(pv) + 1)
    r = np.minimum.accumulate(r[::-1])[::-1]
    out = np.empty_like(pv)
    out[o] = np.minimum(r, 1)
    q[ok] = out
    return q


def dl_meta(est, se):
    """DerSimonian-Laird random-effects meta-analysis (Fisher z or beta)."""
    est, se = np.asarray(est, float), np.asarray(se, float)
    ok = np.isfinite(est) & np.isfinite(se) & (se > 0)
    est, se = est[ok], se[ok]
    k = len(est)
    if k == 0:
        return dict(est=np.nan, lo=np.nan, hi=np.nan, se=np.nan, p=np.nan, I2=np.nan, k=0)
    w = 1 / se ** 2
    fe = np.sum(w * est) / w.sum()
    Q = np.sum(w * (est - fe) ** 2)
    dfq = k - 1
    c = w.sum() - np.sum(w ** 2) / w.sum()
    tau2 = max(0.0, (Q - dfq) / c) if (dfq > 0 and c > 0) else 0.0
    wr = 1 / (se ** 2 + tau2)
    re = np.sum(wr * est) / wr.sum()
    sre = np.sqrt(1 / wr.sum())
    I2 = max(0.0, (Q - dfq) / Q) if (Q > 0 and dfq > 0) else 0.0
    return dict(est=re, lo=re - 1.96 * sre, hi=re + 1.96 * sre, se=sre,
                p=2 * stats.norm.sf(abs(re / sre)), I2=I2, k=k)


def fe_meta(est, se):
    """Fixed-effect (inverse-variance) pooling; returns the same keys as dl_meta."""
    est, se = np.asarray(est, float), np.asarray(se, float)
    ok = np.isfinite(est) & np.isfinite(se) & (se > 0)
    est, se = est[ok], se[ok]
    k = len(est)
    if k == 0:
        return dict(est=np.nan, lo=np.nan, hi=np.nan, se=np.nan, p=np.nan, I2=np.nan, k=0)
    w = 1 / se ** 2
    fe = np.sum(w * est) / w.sum()
    sfe = np.sqrt(1 / w.sum())
    Q = np.sum(w * (est - fe) ** 2)
    I2 = max(0.0, (Q - (k - 1)) / Q) if (Q > 0 and k > 1) else 0.0
    return dict(est=fe, lo=fe - 1.96 * sfe, hi=fe + 1.96 * sfe, se=sfe,
                p=2 * stats.norm.sf(abs(fe / sfe)), I2=I2, k=k)


def pool_meta(est, se):
    return fe_meta(est, se) if META_METHOD == "FE" else dl_meta(est, se)


def residualise(y, Z):
    """OLS residuals of y on [1, Z]."""
    A = np.column_stack([np.ones(len(y))] + ([Z] if Z is not None and Z.size else []))
    b = np.linalg.lstsq(A, y, rcond=None)[0]
    return y - A @ b


class SpatialCorr:
    """Spearman (optionally partial) correlation between two spot-level variables with the
    Dutilleul / Clifford-Richardson-Hemon modified t-test. The variance of r under spatial
    autocorrelation is approximated from the product of the two correlograms over
    equal-frequency distance classes: var(r) = (n + sum_k n_k rho_x(k) rho_y(k)) / n^2,
    n_eff = 1 + 1/var(r) (Clifford et al. 1989, Biometrics; Dutilleul 1993, Biometrics)."""

    def __init__(self, coords_um, max_lag=MAX_LAG_UM, n_classes=N_LAG_CLASSES):
        coords_um = np.asarray(coords_um, float)
        self.n_all = len(coords_um)
        tree = cKDTree(coords_um)
        P = tree.query_pairs(max_lag, output_type="ndarray")
        self.classes = []
        if len(P):
            d = np.linalg.norm(coords_um[P[:, 0]] - coords_um[P[:, 1]], axis=1)
            edges = np.unique(np.quantile(d, np.linspace(0, 1, n_classes + 1)))
            cls = np.clip(np.searchsorted(edges, d, side="right") - 1, 0, len(edges) - 2)
            self.classes = [P[cls == k] for k in range(len(edges) - 1)]

    def test(self, x, y, mask=None, covars=None):
        x, y = np.asarray(x, float), np.asarray(y, float)
        ok = np.isfinite(x) & np.isfinite(y)
        C = None
        if covars is not None:
            C = np.asarray(covars, float).reshape(len(x), -1)
            ok &= np.isfinite(C).all(axis=1)
        if mask is not None:
            ok &= np.asarray(mask, bool)
        idx = np.where(ok)[0]
        n = len(idx)
        nan = dict(r=np.nan, p=np.nan, n=n, n_eff=np.nan, z=np.nan, se=np.nan)
        if n < MIN_SPOTS_TEST:
            return nan
        rx, ry = stats.rankdata(x[idx]), stats.rankdata(y[idx])
        kcov = 0
        if C is not None:
            Zr = np.column_stack([stats.rankdata(C[idx, j]) for j in range(C.shape[1])])
            Zr = Zr[:, Zr.std(axis=0) > 0]
            kcov = Zr.shape[1]
            rx, ry = residualise(rx, Zr), residualise(ry, Zr)
        else:
            rx, ry = rx - rx.mean(), ry - ry.mean()
        sx, sy = np.sum(rx ** 2) / n, np.sum(ry ** 2) / n
        if sx <= 0 or sy <= 0:
            return nan
        r = float(np.sum(rx * ry) / (n * np.sqrt(sx * sy)))
        pos = -np.ones(self.n_all, dtype=np.int64)
        pos[idx] = np.arange(n)
        acc = 0.0
        for Pk in self.classes:
            a, b = pos[Pk[:, 0]], pos[Pk[:, 1]]
            keep = (a >= 0) & (b >= 0)
            if keep.sum() < 10:
                continue
            a, b = a[keep], b[keep]
            nk = 2.0 * len(a)
            ix = (2.0 * np.sum(rx[a] * rx[b]) / nk) / sx
            iy = (2.0 * np.sum(ry[a] * ry[b]) / nk) / sy
            acc += nk * ix * iy
        var_r = (n + acc) / n ** 2
        n_eff = float(n) if var_r <= 0 else float(np.clip(1.0 + 1.0 / var_r, 5.0, n))
        df = max(n_eff - 2 - kcov, 1.0)
        r_c = np.clip(r, -0.999999, 0.999999)
        t = r_c * np.sqrt(df / (1 - r_c ** 2))
        return dict(r=r, p=float(2 * stats.t.sf(abs(t), df)), n=n, n_eff=n_eff,
                    z=float(np.arctanh(r_c)), se=float(1 / np.sqrt(max(n_eff - 3 - kcov, 1.0))))


def meta_rows(df, by, z="z", se="se", r="r"):
    """Pool per-section Fisher z by group (META_METHOD); adds pooled r, CI, P, I2, sign
    consistency and the random-effects P as a sensitivity column."""
    out = []
    for key, d in df.groupby(by, sort=False):
        m = pool_meta(d[z], d[se])
        m_re = dl_meta(d[z], d[se])
        dd = d.dropna(subset=[r])
        npos, nneg = int((dd[r] > 0).sum()), int((dd[r] < 0).sum())
        row = dict(zip(by, key if isinstance(key, tuple) else (key,))) if isinstance(by, list) else {by: key}
        row.update(r_meta=np.tanh(m["est"]), r_lo=np.tanh(m["lo"]), r_hi=np.tanh(m["hi"]), p_meta=m["p"],
                   p_meta_RE=m_re["p"],
                   I2=m["I2"], k_sections=m["k"], n_pos=npos, n_neg=nneg,
                   p_sign=stats.binomtest(max(npos, nneg), npos + nneg, 0.5).pvalue if npos + nneg else np.nan)
        out.append(row)
    return pd.DataFrame(out)


def clr_frame(A, pseudo=CLR_PSEUDO):
    L = np.log(A + pseudo)
    return L.sub(L.mean(axis=1), axis=0)


# ==============================================================================
# 4. GEOMETRY HELPERS (Visium hexagonal lattice)
# ==============================================================================
HEX_OFFSETS = [(0, 2), (0, -2), (1, 1), (1, -1), (-1, 1), (-1, -1)]


def hex_adjacency(rows, cols):
    key = {(int(r), int(c)): i for i, (r, c) in enumerate(zip(rows, cols))}
    nb = [[key[(r + dr, c + dc)] for dr, dc in HEX_OFFSETS if (r + dr, c + dc) in key]
          for r, c in zip(map(int, rows), map(int, cols))]
    return nb


def ring_distance(nb, anchor, max_ring=MAX_RING):
    """Hop distance on the hexagonal lattice to the nearest anchor spot (capped)."""
    n = len(nb)
    dist = np.full(n, np.inf)
    front = list(np.where(anchor)[0])
    dist[front] = 0
    d = 0
    while front and d < max_ring:
        d += 1
        nxt = []
        for i in front:
            for j in nb[i]:
                if not np.isfinite(dist[j]):
                    dist[j] = d
                    nxt.append(j)
        front = nxt
    dist[~np.isfinite(dist)] = max_ring
    return dist


def kernel_matrix(coords_um, radius, bw, include_self):
    tree = cKDTree(coords_um)
    P = tree.query_pairs(radius, output_type="ndarray")
    n = len(coords_um)
    if len(P):
        d = np.linalg.norm(coords_um[P[:, 0]] - coords_um[P[:, 1]], axis=1)
        w = np.exp(-d ** 2 / (2 * bw ** 2))
        i = np.r_[P[:, 0], P[:, 1]]
        j = np.r_[P[:, 1], P[:, 0]]
        W = sp.csr_matrix((np.r_[w, w], (i, j)), shape=(n, n))
    else:
        W = sp.csr_matrix((n, n))
    if include_self:
        W = W + sp.identity(n, format="csr")
    return W


def neighbour_mean(W, values, valid):
    v = np.where(valid & np.isfinite(values), values, 0.0)
    m = (valid & np.isfinite(values)).astype(float)
    num, den = W @ v, W @ m
    out = np.full(len(v), np.nan)
    ok = den > 0
    out[ok] = num[ok] / den[ok]
    return out


# ==============================================================================
# 5. GENE SETS / SCORING HELPERS
# ==============================================================================
EXCLUDE_GENE_RE = re.compile(r"^(MT-|RP[SL]\d|RPLP|HB[ABDEGMQZ]\d?$|MTRNR|IG[HKL][VJCDG]|TR[ABDG][VJC])")


def load_frozen_signatures():
    t = pd.read_csv(F_SIGS)
    sap = t[t.signature == "SAP"].set_index("gene")
    sigs = {k: d.set_index("gene")["weight"] for k, d in t[t.signature != "SAP"].groupby("signature")}
    return sap, sigs


def load_hallmark():
    lib = json.loads(Path(HALLMARK_JSON).read_text())
    return {k: list(v) for k, v in lib.items()}


def zscore_genes(E, genes):
    """E: AnnData (lognorm in .X). Returns genes x spots z-scored DataFrame."""
    genes = [g for g in genes if g in E.var_names]
    X = E[:, genes].X
    X = X.toarray() if sp.issparse(X) else np.asarray(X)
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd[sd == 0] = np.nan
    Z = (X - mu) / sd
    return pd.DataFrame(Z.T, index=genes, columns=E.obs_names).dropna(how="all")


def weighted_score(Z, w, min_cov=0.5):
    g = w.index.intersection(Z.index)
    if len(g) < 3 or len(g) / max(len(w), 1) < min_cov:
        return None
    ww = w[g]
    return (Z.loc[g].mul(ww, axis=0).sum(axis=0) / ww.abs().sum())


def aucell(X, gene_names, gene_sets, top_frac=0.05, chunk=2000, seed=SEED):
    """AUCell (Aibar et al. 2017), identical implementation to Figure 1F."""
    X = X.tocsr() if sp.issparse(X) else sp.csr_matrix(X)
    n_genes = X.shape[1]
    max_rank = int(np.ceil(top_frac * n_genes))
    perm = np.random.default_rng(seed).permutation(n_genes)
    gidx = pd.Index(gene_names)
    members, n_eff = {}, {}
    for name, genes in gene_sets.items():
        m = np.zeros(n_genes, dtype=np.float32)
        pos = gidx.get_indexer([g for g in genes if g in gidx])
        pos = pos[pos >= 0]
        m[pos] = 1.0
        members[name], n_eff[name] = m[perm], max(len(pos), 1)
    weights = (max_rank - np.arange(max_rank)).astype(np.float32)
    out = {k: np.zeros(X.shape[0], dtype=np.float32) for k in gene_sets}
    for s in range(0, X.shape[0], chunk):
        D = X[s:s + chunk].toarray()[:, perm]
        top = np.argpartition(-D, max_rank - 1, axis=1)[:, :max_rank]
        order = np.take_along_axis(D, top, axis=1)
        top = np.take_along_axis(top, np.argsort(-order, axis=1, kind="stable"), axis=1)
        for k, m in members.items():
            out[k][s:s + chunk] = m[top] @ weights / (min(n_eff[k], max_rank) * max_rank)
    return pd.DataFrame(out)


def read_section(s):
    return sc.read_h5ad(H5 / f"S1_{s}.h5ad")


def px_per_um(a):
    sf = next(iter(a.uns["spatial"].values()))["scalefactors"]
    return sf["spot_diameter_fullres"] / SPOT_DIAMETER_UM


def lognorm(a):
    b = a.copy()
    b.layers["counts"] = b.X.copy()
    sc.pp.normalize_total(b, target_sum=1e4)
    b.layers["cp10k"] = b.X.copy()
    sc.pp.log1p(b)
    return b


# ==============================================================================
# 6. CHECKPOINTS / SHARED LOADERS
# ==============================================================================
CKPT = OUT / "checkpoints"
CKPT.mkdir(parents=True, exist_ok=True)
STEP_SCRIPTS = {"step01": "fig4_01_download_qc_select.py", "step02": "fig4_02_cell2location.py",
                "step03": "fig4_03_scores_tests_niches.py", "step04": "fig4_04_channels.py",
                "step05": "fig4_05_niche_regression.py", "step06": "fig4_06_drug2cell.py",
                "step07": "fig4_07_figures.py"}


def step_done(step):
    return (CKPT / f"{step}.done").exists()


def mark_done(step, **info):
    (CKPT / f"{step}.done").write_text(json.dumps(dict(finished=time.strftime("%Y-%m-%d %H:%M:%S"), **info),
                                                  indent=2, default=str))
    log(f"CHECKPOINT {step} done")


def require(*steps):
    miss = [s for s in steps if not step_done(s)]
    if miss:
        sys.exit("Run these steps first: " + ", ".join(f"{s} ({STEP_SCRIPTS[s]})" for s in miss))


def force_from_argv(keys):
    """`python fig4_0X_*.py --force` recomputes this step (adds its keys to FORCE)."""
    if "--force" in sys.argv:
        FORCE.update(keys)
    return bool(FORCE & set(keys))


def selected_sections():
    f = TAB / "S1_selected_sections.json"
    if f.exists():
        return json.loads(f.read_text())["selected"]
    return list(PRIMARY_ALL)


PRIMARY = selected_sections()        # the sections analysed by steps 02-07


def load_channels():
    """Channels written by step 01 from YOUR Fig. 4A table (tables/S0_channels.csv); the list in the
    configuration above is only the fallback (it is the same list, transcribed from that table)."""
    f = TAB / "S0_channels.csv"
    if not f.exists():
        return CHANNELS
    t = pd.read_csv(f)
    return [(r.ligand, r.receptor, r.source, r.target, int(r.sign), r.tier) for r in t.itertuples()]


CHANNELS = load_channels()


def load_manifest(selected=True):
    m = json.loads((RAW / "manifest.json").read_text())
    return {k: v for k, v in m.items() if (k in PRIMARY or not selected)}


def load_hyp():
    return {k: int(v) for k, v in json.loads((TAB / "S0_preregistration.json").read_text())
            ["H1_H4_state_directions"].items()}


def load_inf():
    return pd.read_csv(TAB / "S2_reference_signatures.csv", index_col=0)


def load_means(kind="means", tag="main"):
    return pd.read_csv(TAB / f"S3_c2l_abundance_{kind}_{tag}.csv", index_col=0)


def build_testers(S):
    return {s: SpatialCorr(d[["x_um", "y_um"]].values) for s, d in S.groupby("sample", sort=False)}


# ==============================================================================
# STEP-SPECIFIC CODE
# ==============================================================================
def stromal_gene_pool(inf, sap_genes, all_genes):
    """Stroma-enriched non-SAP genes (same atlas transferability rule as the SAP signature)."""
    r = inf.loc[inf.index.intersection(all_genes)]
    st = r[[c for c in STROMA if c in r]].max(axis=1)
    ot = r[[c for c in r.columns if c not in STROMA]].max(axis=1)
    pool = r.index[(st >= ot) & ~r.index.isin(sap_genes) & ~r.index.str.match(EXCLUDE_GENE_RE)]
    return pd.Index(pool)



def spot_scores(manifest, means, sap, sigs, hall, pw):
    f = TAB / "S4_spot_scores.csv"
    if f.exists() and "scores" not in FORCE:
        return pd.read_csv(f, index_col=0)
    w_sap = sap["weight"]
    hm_sets = {}
    for h in ["Hypoxia", "Glycolysis", "Epithelial Mesenchymal Transition", "Oxidative Phosphorylation",
              "Inflammatory Response", "TNF-alpha Signaling via NF-kB", "IL-6/JAK/STAT3 Signaling",
              "Interferon Gamma Response", "Angiogenesis"]:
        hit = [k for k in hall if re.sub(r"[^a-z0-9]", "", k.lower()) == re.sub(r"[^a-z0-9]", "", h.lower())]
        if hit:
            hm_sets[f"HM_{h}"] = hall[hit[0]]
    if pw is not None:
        for k, d in pw.groupby("signature"):
            if k.startswith("PWm_"):
                hm_sets[k] = list(d.gene)
    allrows = []
    for s in manifest:
        a = lognorm(read_section(s))
        A = means.reindex(a.obs_names)
        if A.isna().all(axis=None):
            raise RuntimeError(f"{s}: no cell2location abundances")
        o = pd.DataFrame(index=a.obs_names)
        o["sample"], o["site"] = s, a.obs["site"].values
        o["array_row"], o["array_col"] = a.obs.array_row.values, a.obs.array_col.values
        xy = a.obsm["spatial"] / px_per_um(a)
        o["x_um"], o["y_um"] = xy[:, 0], xy[:, 1]
        o["stroma_cells"] = A[[c for c in STROMA if c in A]].sum(axis=1)
        o["momac_cells"] = A[[c for c in MOMAC if c in A]].sum(axis=1)
        o["myeloid_all_cells"] = A[[c for c in MYELOID_STATES if c in A]].sum(axis=1)
        o["total_cells"] = A.sum(axis=1)
        o["log_stroma"] = np.log1p(o.stroma_cells)
        o["log_momac"] = np.log1p(o.momac_cells)
        # --- SAP (signature from the Fig. 1 axis weights, bidirectional) ---
        Z = zscore_genes(a, list(w_sap.index))
        for nm, w in [("SAP", w_sap), ("SAP_up", w_sap[w_sap > 0]), ("SAP_down", -w_sap[w_sap < 0])]:
            v = weighted_score(Z, w)
            o[nm] = v.reindex(o.index).values if v is not None else np.nan
        o["SAP_cov"] = len(w_sap.index.intersection(Z.index)) / len(w_sap)
        st = (o.stroma_cells >= MIN_STROMA_CELLS).values
        o["stroma_spot"] = st
        o["SAP_state"] = np.nan
        if st.sum() >= MIN_SPOTS_TEST:
            y = o.loc[st, "SAP"].values
            res = residualise(y, o.loc[st, ["log_stroma"]].values)
            o.loc[st, "SAP_state"] = (res - res.mean()) / res.std()
        W = kernel_matrix(o[["x_um", "y_um"]].values, NEIGHBOUR_RADIUS_UM, KERNEL_BW_UM, include_self=False)
        o["SAP_nbr"] = neighbour_mean(W, o["SAP_state"].values, st)
        # --- myeloid states (cell2location, SAP-blind) ---
        o["myeloid_spot"] = (o.momac_cells >= MIN_MYELOID_CELLS).values
        states = [c for c in MYELOID_STATES if c in A]
        C = clr_frame(A[states])
        for c in states:
            o[f"CLR_{c}"] = np.where(o.myeloid_spot, C[c].values, np.nan)
            o[f"ab_{c}"] = A[c].values
        for c in A.columns:
            if c not in states:
                o[f"ab_{c}"] = A[c].values
        # --- label-free myeloid states (Fig. 3 CellRank drivers; residualised on myeloid abundance) ---
        lf_genes = set().union(*[set(sigs[k].index) for k in sigs if k.startswith(("Fate_", "Mono_", "Myeloid_"))])
        Zl = zscore_genes(a, list(lf_genes))
        mab = weighted_score(Zl, sigs["Myeloid_abundance"]) if "Myeloid_abundance" in sigs else None
        for k in [k for k in sigs if k.startswith(("Fate_", "Mono_"))]:
            v = weighted_score(Zl, sigs[k])
            if v is None:
                continue
            v = v.reindex(o.index).values
            if mab is not None:
                m = mab.reindex(o.index).values
                ok = np.isfinite(v) & np.isfinite(m)
                v2 = np.full(len(v), np.nan)
                v2[ok] = residualise(v[ok], m[ok][:, None])
                v = v2
            o[f"LF_{k}"] = v
        o["LF_Myeloid_abundance"] = mab.reindex(o.index).values if mab is not None else np.nan
        reg = [k for k in sigs if k.startswith("Regulon_")]
        if reg:
            Zr = zscore_genes(a, list(set().union(*[set(sigs[k].index) for k in reg])))
            for k in reg:
                v = weighted_score(Zr, sigs[k], min_cov=0.3)
                if v is None or mab is None:
                    continue
                v = v.reindex(o.index).values
                m = mab.reindex(o.index).values
                ok = np.isfinite(v) & np.isfinite(m)
                v2 = np.full(len(v), np.nan)
                v2[ok] = residualise(v[ok], m[ok][:, None])
                o[f"REG_{k[8:]}"] = v2
        # --- Hallmark / TAM-programme AUCell ---
        X = a.X
        expressed = np.asarray((X > 0).sum(axis=0)).ravel() > 0
        auc = aucell(X[:, expressed], a.var_names[expressed], hm_sets)
        auc.index = a.obs_names
        for k in auc.columns:
            o[f"AUC_{k}"] = auc[k].values
        allrows.append(o)
        log(f"  S4 {s}: {st.sum()} stromal spots, {o.myeloid_spot.sum()} myeloid spots, "
            f"SAP coverage {o.SAP_cov.iloc[0]:.0%}")
    S = pd.concat(allrows)
    S.to_csv(f)
    return S




def test_states(S, testers, hyp):
    f = TAB / "S5_state_tests_per_section.csv"
    if f.exists() and "tests" not in FORCE:
        return pd.read_csv(f), pd.read_csv(TAB / "S5_state_tests_meta.csv")
    rows = []
    states = [c[4:] for c in S.columns if c.startswith("CLR_")]
    hyp = {k: int(v) for k, v in hyp.items()}
    for s, d in S.groupby("sample", sort=False):
        T = testers[s]
        mask = (d.stroma_spot & d.myeloid_spot).values
        cov = d[["log_stroma", "log_momac"]].values
        for st in states:
            for analysis, x, m, cv in [
                ("H1-4 within-spot SAP state (c2l)", d.SAP_state, mask, cov),
                ("H5 neighbourhood SAP (c2l)", d.SAP_nbr, d.myeloid_spot.values,
                 np.column_stack([cov, np.nan_to_num(d.SAP_state.values, nan=0.0)])),
                ("Sens: activated arm only", d.SAP_up, mask, cov),
                ("Sens: quiescent arm only (reversed)", -d.SAP_down, mask, cov),
                ("Sens: raw SAP (no stromal residualisation)", d.SAP, mask, d[["log_momac"]].values)]:
                r = T.test(x.values, d[f"CLR_{st}"].values, mask=m, covars=cv)
                rows.append(dict(sample=s, site=d.site.iloc[0], analysis=analysis, state=st, **r))
        for k in [c for c in d.columns if c.startswith("LF_") and c != "LF_Myeloid_abundance"]:
            st = k[3:].replace("Fate_", "")
            r = T.test(d.SAP_state.values, d[k].values, mask=d.stroma_spot.values,
                       covars=d[["log_stroma", "LF_Myeloid_abundance"]].values)
            rows.append(dict(sample=s, site=d.site.iloc[0], analysis="Label-free signatures (no deconvolution)",
                             state=st, **r))
        for k in [c for c in d.columns if c.startswith("REG_")]:
            r = T.test(d.SAP_nbr.values, d[k].values, mask=d.myeloid_spot.values,
                       covars=d[["log_momac", "log_stroma", "LF_Myeloid_abundance"]].values)
            rows.append(dict(sample=s, site=d.site.iloc[0], analysis="Regulon activity vs neighbourhood SAP (Fig. 3J)",
                             state=k[4:], **r))
    per = pd.DataFrame(rows)
    per.to_csv(f, index=False)
    prim = per[per["sample"].isin(PRIMARY)]
    meta = meta_rows(prim, ["analysis", "state"])
    is_reg = meta.analysis.str.startswith("Regulon")
    meta["expected_sign"] = np.where(is_reg, meta.state.map(REGULON_DIR), meta.state.map(hyp))
    meta["family"] = np.where(meta.expected_sign.notna(), "pre-specified", "exploratory")
    meta["q"] = np.nan
    for (an, fam), d in meta.groupby(["analysis", "family"]):
        meta.loc[d.index, "q"] = bh(d.p_meta.values)
    meta["direction_as_expected"] = np.sign(meta.r_meta) == meta.expected_sign
    meta.to_csv(TAB / "S5_state_tests_meta.csv", index=False)
    log("S5 meta (pre-specified, within-spot):\n" + meta[(meta.analysis.str.startswith("H1")) &
        (meta.family == "pre-specified")][["state", "r_meta", "p_meta", "q", "I2", "n_pos", "n_neg"]]
        .round(4).to_string(index=False))
    return per, meta


def random_null(S, manifest, inf, sap, testers, hyp):
    f = TAB / "S5_random_stromal_null.csv"
    if f.exists() and "tests" not in FORCE:
        return pd.read_csv(f)
    w = sap["weight"]
    secs = {s: lognorm(read_section(s)) for s in PRIMARY if s in manifest}
    common = None
    for a in secs.values():
        common = a.var_names if common is None else common.intersection(a.var_names)
    pool = stromal_gene_pool(inf, set(w.index), common)
    mean_expr = pd.concat([pd.Series(np.asarray(a[:, pool].X.mean(axis=0)).ravel(), index=pool)
                           for a in secs.values()], axis=1).mean(axis=1)
    sap_g = w.index.intersection(common)
    sap_mean = pd.concat([pd.Series(np.asarray(a[:, sap_g].X.mean(axis=0)).ravel(), index=sap_g)
                          for a in secs.values()], axis=1).mean(axis=1)
    edges = np.unique(np.quantile(pd.concat([mean_expr, sap_mean]), np.linspace(0, 1, 11)))
    bin_pool = np.clip(np.searchsorted(edges, mean_expr.values, side="right") - 1, 0, len(edges) - 2)
    bin_sap = np.clip(np.searchsorted(edges, sap_mean.values, side="right") - 1, 0, len(edges) - 2)
    by_bin = {b: pool[bin_pool == b] for b in np.unique(bin_pool)}
    Zs = {s: zscore_genes(a, list(pool)) for s, a in secs.items()}
    part = TAB / "S5_random_stromal_null.partial.csv"            # checkpoint every 50 draws
    out = pd.read_csv(part).to_dict("records") if part.exists() and "tests" not in FORCE else []
    start = (max(r["draw"] for r in out) + 1) if out else 0
    if start:
        log(f"  random stromal null: resuming at draw {start}")
    for b in range(start, N_RANDOM_NULL):
        rng = np.random.default_rng(SEED + b)                      # per-draw seed -> resumable
        genes = []
        for bb in bin_sap:
            cand = by_bin.get(bb, pool)
            if len(cand) == 0:
                cand = pool
            genes.append(rng.choice(cand))
        wr = pd.Series(w[sap_g].values, index=pd.Index(genes)).groupby(level=0).mean()
        recs = []
        for s, Z in Zs.items():
            d = S[S["sample"] == s]
            v = weighted_score(Z, wr, min_cov=0.3)
            if v is None:
                continue
            v = v.reindex(d.index).values
            st = d.stroma_spot.values & np.isfinite(v)
            x = np.full(len(v), np.nan)
            if st.sum() < MIN_SPOTS_TEST:
                continue
            x[st] = residualise(v[st], d.loc[st, ["log_stroma"]].values)
            mask = st & d.myeloid_spot.values
            for state in hyp:
                if f"CLR_{state}" not in d:
                    continue
                r = testers[s].test(x, d[f"CLR_{state}"].values, mask=mask,
                                    covars=d[["log_stroma", "log_momac"]].values)
                recs.append(dict(state=state, **r))
        if not recs:
            continue
        R = pd.DataFrame(recs)
        for state, dd in R.groupby("state"):
            m = pool_meta(dd.z, dd.se)
            out.append(dict(draw=b, state=state, r_meta=np.tanh(m["est"]), z_meta=m["est"] / m["se"]))
        if (b + 1) % 50 == 0:
            pd.DataFrame(out).to_csv(part, index=False)
            log(f"  random stromal null: {b + 1}/{N_RANDOM_NULL}")
    null = pd.DataFrame(out)
    null.to_csv(f, index=False)
    if part.exists():
        part.unlink()
    return null


def null_summary(meta, null, hyp):
    rows = []
    obs = meta[meta.analysis.str.startswith("H1")].set_index("state")
    for st, sgn in hyp.items():
        if st not in obs.index or st not in set(null.state):
            continue
        r_obs = obs.loc[st, "r_meta"]
        nr = null.loc[null.state == st, "r_meta"].values
        p_emp = (1 + np.sum(sgn * nr >= sgn * r_obs)) / (1 + len(nr))
        rows.append(dict(state=st, r_obs=r_obs, null_median=np.median(nr), null_q025=np.quantile(nr, 0.025),
                         null_q975=np.quantile(nr, 0.975), p_emp_directional=p_emp, n_draws=len(nr)))
    t = pd.DataFrame(rows)
    t.to_csv(TAB / "S5_random_stromal_null_summary.csv", index=False)
    return t




def gradients(S, testers, hyp):
    f = TAB / "S6_gradient_profiles.csv"
    if f.exists() and "tests" not in FORCE:
        return pd.read_csv(f), pd.read_csv(TAB / "S6_gradient_tests_meta.csv")
    prof, tests = [], []
    for s, d in S.groupby("sample", sort=False):
        nb = hex_adjacency(d.array_row.values, d.array_col.values)
        st = d.SAP_state.values
        ok = np.isfinite(st)
        if ok.sum() < MIN_SPOTS_TEST:
            continue
        hi = ok & (st >= np.nanquantile(st, 0.75))
        lo = ok & (st <= np.nanquantile(st, 0.25))
        d_hi, d_lo = ring_distance(nb, hi), ring_distance(nb, lo)
        rel = d_hi - d_lo                               # >0: closer to SAP-low stroma
        for state in hyp:
            col = f"CLR_{state}"
            if col not in d:
                continue
            y = d[col].values
            for ring in range(MAX_RING + 1):
                m = (d_hi == ring) & np.isfinite(y)
                if m.sum() >= 5:
                    prof.append(dict(sample=s, state=state, anchor="SAP-high", ring=ring, n=int(m.sum()),
                                     mean_clr=float(np.mean(y[m]))))
                m = (d_lo == ring) & np.isfinite(y)
                if m.sum() >= 5:
                    prof.append(dict(sample=s, state=state, anchor="SAP-low", ring=ring, n=int(m.sum()),
                                     mean_clr=float(np.mean(y[m]))))
            r = testers[s].test(rel, y, mask=d.myeloid_spot.values, covars=d[["log_stroma", "log_momac"]].values)
            tests.append(dict(sample=s, state=state, **r))
    prof, tests = pd.DataFrame(prof), pd.DataFrame(tests)
    prof.to_csv(f, index=False)
    tests.to_csv(TAB / "S6_gradient_tests_per_section.csv", index=False)
    meta = meta_rows(tests[tests["sample"].isin(PRIMARY)], "state")
    meta["expected_sign"] = -meta.state.map(hyp)                  # relative distance is the inverse
    meta["q"] = bh(meta.p_meta.values)
    meta.to_csv(TAB / "S6_gradient_tests_meta.csv", index=False)
    return prof, meta




def niches(S, means):
    f = TAB / "S7_niche_assignment.csv"
    if f.exists() and "niches" not in FORCE:
        return (pd.read_csv(f, index_col=0), pd.read_csv(TAB / "S7_niche_loadings.csv", index_col=0),
                pd.read_csv(TAB / "S7_nhood_enrichment_mean_z.csv", index_col=0))
    from sklearn.decomposition import NMF
    P = means.loc[S.index]
    P = P.div(P.sum(axis=1), axis=0)
    train = P.loc[S["sample"].isin(PRIMARY).values].values
    stab = []
    for k in NMF_K_RANGE:
        Hs = []
        for r in range(NMF_RESTARTS):
            m = NMF(n_components=k, init="random", random_state=SEED + r, max_iter=2000, tol=1e-6)
            m.fit(train)
            H = m.components_ / np.linalg.norm(m.components_, axis=1, keepdims=True)
            Hs.append(H)
        sims = []
        for H in Hs[1:]:
            Cm = Hs[0] @ H.T
            ri, ci = linear_sum_assignment(-Cm)
            sims.append(Cm[ri, ci].mean())
        stab.append(dict(k=k, stability=float(np.mean(sims)), stability_sd=float(np.std(sims))))
    stab = pd.DataFrame(stab)
    stab.to_csv(TAB / "S7_nmf_stability.csv", index=False)
    best = stab.stability.max()
    k = int(stab[stab.stability >= best - 0.01].k.min())
    m = NMF(n_components=k, init="nndsvda", random_state=SEED, max_iter=5000, tol=1e-6)
    m.fit(train)
    Wall = m.transform(P.values)
    Wn = Wall / np.maximum(Wall.max(axis=0, keepdims=True), 1e-12)
    H = pd.DataFrame(m.components_, columns=P.columns)
    H = H.div(H.sum(axis=1), axis=0)
    names = []
    for i, row in H.iterrows():
        top = row.sort_values(ascending=False).index[:2]
        names.append(f"N{i + 1}: {top[0]}/{top[1]}")
    H.index = names
    lab = pd.Series(np.array(names)[Wn.argmax(axis=1)], index=P.index, name="niche")
    lab.to_frame().to_csv(f)
    H.to_csv(TAB / "S7_niche_loadings.csv")
    # neighbourhood enrichment (label permutation within section)
    rng = np.random.default_rng(SEED)
    Zs = []
    for s, d in S.groupby("sample", sort=False):
        if s not in PRIMARY:
            continue
        nb = hex_adjacency(d.array_row.values, d.array_col.values)
        ii = np.array([i for i, l in enumerate(nb) for _ in l])
        jj = np.array([j for l in nb for j in l])
        L = pd.Categorical(lab.loc[d.index].values, categories=names).codes
        K = len(names)

        def counts(lbl):
            M = np.zeros((K, K))
            np.add.at(M, (lbl[ii], lbl[jj]), 1)
            return M
        obs = counts(L)
        perm = np.stack([counts(rng.permutation(L)) for _ in range(1000)])
        z = (obs - perm.mean(0)) / np.maximum(perm.std(0), 1e-9)
        Zs.append(z)
    Zm = pd.DataFrame(np.nanmean(np.stack(Zs), axis=0), index=names, columns=names)
    Zm.to_csv(TAB / "S7_nhood_enrichment_mean_z.csv")
    log(f"S7 niches: k = {k} (stability {best:.3f}); {lab.value_counts().to_dict()}")
    return lab.to_frame(), H, Zm



def main():
    force_from_argv({"scores", "tests", "niches"})
    require("step01", "step02")
    if step_done("step03") and not FORCE & {"scores", "tests", "niches"}:
        log("step03 already done (use --force to redo)")
        return
    log("=" * 70)
    log(f"STEP 03: spot scores, H1-H5, null, gradients, niches on {PRIMARY}")
    manifest = load_manifest()
    hyp = load_hyp()
    sap, sigs = load_frozen_signatures()
    hall = load_hallmark()
    pw = pd.read_csv(F_PATHWAY) if F_PATHWAY.exists() else None
    inf, means = load_inf(), load_means()
    S = spot_scores(manifest, means, sap, sigs, hall, pw)                 # checkpoint: S4_spot_scores.csv
    testers = build_testers(S)
    per, meta = test_states(S, testers, hyp)                              # checkpoint: S5_state_tests_*.csv
    null = random_null(S, manifest, inf, sap, testers, hyp)              # checkpoint: partial every 50 draws
    null_summary(meta, null, hyp)
    gradients(S, testers, hyp)                                            # checkpoint: S6_*.csv
    niches(S, means)                                                      # checkpoint: S7_*.csv
    mark_done("step03", n_spots=len(S), stromal=int(S.stroma_spot.sum()), myeloid=int(S.myeloid_spot.sum()))


if __name__ == "__main__":
    main()
