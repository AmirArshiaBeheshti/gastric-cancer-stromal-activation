#!/usr/bin/env python
# coding: utf-8
"""
 fig4_07_figures.py  (step 7 of 7)
 Fig. 4B-K main panels, supplementary figures and tables/methods_numbers_spatial.txt from the
 tables written by steps 01-06 (nothing is recomputed; re-run freely after style changes).
 RUN: python fig4_07_figures.py
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


def savefig(fig, name, folder=FIG_MAIN):
    for ext in ["pdf", "svg", "png"]:
        fig.savefig(folder / f"{name}.{ext}")
    plt.close(fig)
    log(f"  saved figure {folder.name}/{name}")


def panel_label(ax, letter, x=-0.05, y=1.04):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="right")


def fmt_p(p):
    if p is None or not np.isfinite(p):
        return "n.a."
    if p < 1e-3:
        m, e = f"{p:.1e}".split("e")
        return f"{m}×10$^{{{int(e)}}}$"
    return f"{p:.3f}" if p < 0.1 else f"{p:.2f}"


# ==============================================================================
# 4. GEOMETRY HELPERS (Visium hexagonal lattice)
# ==============================================================================
HEX_OFFSETS = [(0, 2), (0, -2), (1, 1), (1, -1), (-1, 1), (-1, -1)]


# ==============================================================================
# 5. GENE SETS / SCORING HELPERS
# ==============================================================================
EXCLUDE_GENE_RE = re.compile(r"^(MT-|RP[SL]\d|RPLP|HB[ABDEGMQZ]\d?$|MTRNR|IG[HKL][VJCDG]|TR[ABDG][VJC])")


def read_section(s):
    return sc.read_h5ad(H5 / f"S1_{s}.h5ad")


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


def load_S():
    return pd.read_csv(TAB / "S4_spot_scores.csv", index_col=0)


# ==============================================================================
# STEP-SPECIFIC CODE
# ==============================================================================
def _img_and_scale(a):
    lib = next(iter(a.uns["spatial"].values()))
    img = lib["images"].get("hires")
    sf = lib["scalefactors"]
    return img, sf.get("tissue_hires_scalef", 1.0), sf["spot_diameter_fullres"]


def _ellipses(ax, d, xy, **kw):
    w = np.full(len(xy), d)
    try:
        return EllipseCollection(w, w, np.zeros(len(xy)), units="xy", offsets=xy,
                                 offset_transform=ax.transData, **kw)
    except TypeError:
        return EllipseCollection(w, w, np.zeros(len(xy)), units="xy", offsets=xy, transOffset=ax.transData, **kw)


def spatial_ax(ax, a, values, cmap=ABUND_CMAP, vmin=None, vmax=None, title=None, categorical=None,
               show_img=True, img_alpha=0.55, cbar=True, cbar_label=None):
    img, s, dia = _img_and_scale(a)
    xy = a.obsm["spatial"] * s
    d = dia * s * 1.05
    if show_img and img is not None:
        ax.imshow(img, alpha=img_alpha)
    if categorical is not None:
        cols = [categorical.get(v, "#BBBBBB") for v in values]
        ec = _ellipses(ax, d, xy, facecolors=cols, edgecolors="none")
    else:
        v = np.asarray(values, float)
        vmin = np.nanquantile(v, 0.01) if vmin is None else vmin
        vmax = np.nanquantile(v, 0.992) if vmax is None else vmax
        ec = _ellipses(ax, d, xy, cmap=cmap, edgecolors="none")
        ec.set_array(np.nan_to_num(v, nan=vmin))
        ec.set_clim(vmin, vmax)
    ax.add_collection(ec)
    pad = d * 2
    ax.set_xlim(xy[:, 0].min() - pad, xy[:, 0].max() + pad)
    ax.set_ylim(xy[:, 1].max() + pad, xy[:, 1].min() - pad)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=6.5, fontweight="normal")
    if categorical is None and cbar:
        cb = plt.colorbar(ec, ax=ax, fraction=0.035, pad=0.01)
        cb.ax.tick_params(labelsize=5)
        cb.outline.set_linewidth(0.3)
        if cbar_label:
            cb.set_label(cbar_label, fontsize=5.5)
    return ec


def representative_section(S):
    c = S[S["sample"].isin(PRIMARY)].groupby("sample").apply(lambda d: int((d.stroma_spot & d.myeloid_spot).sum()))
    return c.idxmax()


def fig4B_celltype_maps(S, q05, rep):
    a = read_section(rep)
    Q = q05.loc[a.obs_names]
    show = [c for c in ["Malignant", "Fibroblast", "Pericyte_SMC", "Endothelial", "Mono_classical",
                        "TAM_Inflammatory", "TAM_SPP1", "TAM_C1QC"] if c in Q]
    fig, axes = plt.subplots(3, 3, figsize=(7.1, 6.6), gridspec_kw={"wspace": 0.12, "hspace": 0.18})
    img, _, _ = _img_and_scale(a)
    axes[0, 0].imshow(img)
    axes[0, 0].axis("off")
    axes[0, 0].set_title(f"{rep} · H&E", fontsize=6.5, fontweight="normal")
    for ax, c in zip(axes.ravel()[1:], show):
        spatial_ax(ax, a, Q[c].values, title=f"{c.replace('_', ' ')} (q05 cells)")
    for ax in axes.ravel()[1 + len(show):]:
        ax.axis("off")
    panel_label(axes[0, 0], "B", x=0.0)
    savefig(fig, "Fig4B_cell2location_maps")


def fig4C_sap_niche_maps(S, niche_lab, rep):
    a = read_section(rep)
    d = S.loc[a.obs_names]
    labs = niche_lab.loc[a.obs_names, "niche"].values
    names = sorted(set(niche_lab.niche))
    pal = dict(zip(names, plt.get_cmap("tab20")(np.linspace(0, 1, max(len(names), 2)))))
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.6), gridspec_kw={"wspace": 0.1})
    spatial_ax(axes[0], a, d.SAP_state.values, cmap=SAP_CMAP, vmin=-2.5, vmax=2.5,
               title="SAP stromal state (z)", cbar_label="SAP (z)")
    spatial_ax(axes[1], a, d.SAP_nbr.values, cmap=SAP_CMAP, vmin=-2, vmax=2,
               title="Neighbourhood SAP (rings 1–2)", cbar_label="SAP (z)")
    spatial_ax(axes[2], a, labs, categorical=pal, title="Tissue niches (NMF)")
    axes[2].legend(handles=[Patch(color=pal[n], label=n) for n in names], loc="upper left",
                   bbox_to_anchor=(1.0, 1.0), frameon=False, fontsize=5)
    panel_label(axes[0], "C", x=0.0)
    savefig(fig, "Fig4C_SAP_and_niche_maps")


def _forest(ax, per, meta_row, title, color):
    d = per.dropna(subset=["r"]).sort_values("sample")
    y = np.arange(len(d))[::-1] + 1.5
    lo = np.tanh(d.z - 1.96 * d.se)
    hi = np.tanh(d.z + 1.96 * d.se)
    ax.hlines(y, lo, hi, color=C_GREY, lw=0.6)
    ax.plot(d.r, y, "s", color=C_DARK, ms=2.6)
    h = 0.35
    ax.fill([meta_row.r_lo, meta_row.r_meta, meta_row.r_hi, meta_row.r_meta], [0, h, 0, -h], color=color, lw=0)
    ax.axvline(0, color=C_GREY, lw=0.5, ls=(0, (2, 2)))
    ax.set_yticks(list(y) + [0])
    ax.set_yticklabels(list(d["sample"]) + ["Meta"], fontsize=5.5)
    ax.set_xlabel("Partial Spearman ρ", fontsize=6)
    ax.set_title(f"{title}\nρ = {meta_row.r_meta:.2f}, P = {fmt_p(meta_row.p_meta)}, I² = {100 * meta_row.I2:.0f}%",
                 fontsize=6, fontweight="normal")
    for sp_ in ["top", "right"]:
        ax.spines[sp_].set_visible(False)


def fig4D_forests(per, meta, null_sum, hyp):
    an = "H1-4 within-spot SAP state (c2l)"
    states = [s for s in hyp if ((meta.analysis == an) & (meta.state == s)).any()]
    fig, axes = plt.subplots(1, len(states) + 1, figsize=(7.6, 2.4),
                             gridspec_kw={"wspace": 0.95, "width_ratios": [1] * len(states) + [0.9]})
    for ax, st in zip(axes, states):
        m = meta[(meta.analysis == an) & (meta.state == st)].iloc[0]
        p = per[(per.analysis == an) & (per.state == st) & per["sample"].isin(PRIMARY)]
        _forest(ax, p, m, st.replace("_", " "), C_ACT if hyp[st] > 0 else C_QUI)
    ax = axes[-1]
    if len(null_sum):
        for i, r in enumerate(null_sum.itertuples()):
            ax.plot([i, i], [r.null_q025, r.null_q975], color=C_GREY, lw=3, alpha=0.45, solid_capstyle="butt")
            ax.plot(i, r.null_median, "_", color=C_GREY, ms=6)
            ax.plot(i, r.r_obs, "D", color=C_ACT if hyp[r.state] > 0 else C_QUI, ms=3.5)
            ax.text(i, max(r.r_obs, r.null_q975) + 0.02, f"P$_{{emp}}$={r.p_emp_directional:.3f}",
                    ha="center", va="bottom", fontsize=4.8, rotation=90)
        ax.set_xticks(range(len(null_sum)))
        ax.set_xticklabels([s.replace("_", " ") for s in null_sum.state], rotation=90, fontsize=5.5)
        ax.axhline(0, color=C_GREY, lw=0.4, ls=(0, (2, 2)))
        ax.set_ylabel("Meta ρ", fontsize=6)
        top = max(null_sum.r_obs.max(), null_sum.null_q975.max())
        ax.set_ylim(top=top + 0.45)
        ax.set_title(f"SAP vs {N_RANDOM_NULL} random\nstromal signatures", fontsize=6, fontweight="normal", pad=8)
        for sp_ in ["top", "right"]:
            ax.spines[sp_].set_visible(False)
    panel_label(axes[0], "D", x=-0.35)
    savefig(fig, "Fig4D_SAP_vs_myeloid_states_forest")


def fig4E_gradients(prof, gmeta, hyp):
    fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.2), gridspec_kw={"wspace": 0.45})
    for ax, anchor in zip(axes, ["SAP-high", "SAP-low"]):
        for st in hyp:
            d = prof[(prof.state == st) & (prof.anchor == anchor) & prof["sample"].isin(PRIMARY)]
            if d.empty:
                continue
            g = d.groupby("ring").mean_clr.agg(["mean", "sem", "count"])
            g = g[g["count"] >= 3]
            col = MYELOID_PALETTE.get(st, C_DARK)
            ax.plot(g.index, g["mean"], "-o", color=col, ms=2.2, lw=0.9, label=st.replace("_", " "))
            ax.fill_between(g.index, g["mean"] - g["sem"], g["mean"] + g["sem"], color=col, alpha=0.15, lw=0)
        ax.set_xlabel(f"Hexagonal rings from {anchor} stroma", fontsize=6)
        ax.set_ylabel("Myeloid-state CLR (mean ± s.e.m.)", fontsize=6)
        for sp_ in ["top", "right"]:
            ax.spines[sp_].set_visible(False)
    axes[1].legend(frameon=False, fontsize=5.5, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    txt = "; ".join(f"{r.state.replace('_', ' ')} ρ={r.r_meta:.2f} (q={fmt_p(r.q)})" for r in gmeta.itertuples())
    fig.text(0.0, -0.08, f"Relative distance (SAP-high − SAP-low) vs CLR, meta: {txt}", fontsize=5)
    panel_label(axes[0], "E", x=-0.25)
    savefig(fig, "Fig4E_distance_gradients")


def fig4F_niches(S, niche_lab, H, Zm):
    d = S.join(niche_lab)
    d = d[d["sample"].isin(PRIMARY)]
    names = list(H.index)
    fig = plt.figure(figsize=(7.1, 2.8))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.5, 0.8, 1.0], wspace=0.6)
    ax = fig.add_subplot(gs[0])
    im = ax.imshow(H.values, aspect="auto", cmap="Reds", vmin=0)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=5)
    ax.set_xticks(range(H.shape[1]))
    ax.set_xticklabels([c.replace("_", " ") for c in H.columns], rotation=90, fontsize=5)
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.01).set_label("Loading (row-normalised)", fontsize=5)
    ax.set_title("NMF tissue niches", fontsize=6.5, fontweight="normal")
    ax2 = fig.add_subplot(gs[1])
    med = d.groupby(["niche", "sample"]).SAP_state.median().reset_index()
    order = med.groupby("niche").SAP_state.median().reindex(names).sort_values().index
    for i, n in enumerate(order):
        v = med[med.niche == n].SAP_state.dropna().values
        ax2.scatter(np.full(len(v), i) + np.random.default_rng(SEED).uniform(-0.15, 0.15, len(v)), v, s=4,
                    color=C_DARK, lw=0)
        if len(v):
            ax2.plot([i - 0.3, i + 0.3], [np.median(v)] * 2, color=C_ACT, lw=1)
    ax2.set_xticks(range(len(order)))
    ax2.set_xticklabels([n.split(":")[0] for n in order], fontsize=5.5)
    ax2.axhline(0, color=C_GREY, lw=0.4, ls=(0, (2, 2)))
    ax2.set_ylabel("SAP state (section median)", fontsize=6)
    for sp_ in ["top", "right"]:
        ax2.spines[sp_].set_visible(False)
    ax3 = fig.add_subplot(gs[2])
    v = np.nanmax(np.abs(Zm.values))
    im = ax3.imshow(Zm.values, cmap="RdBu_r", vmin=-v, vmax=v)
    ax3.set_xticks(range(len(names)))
    ax3.set_xticklabels([n.split(":")[0] for n in names], fontsize=5)
    ax3.set_yticks(range(len(names)))
    ax3.set_yticklabels([n.split(":")[0] for n in names], fontsize=5)
    plt.colorbar(im, ax=ax3, fraction=0.04, pad=0.02).set_label("Neighbourhood z (mean)", fontsize=5)
    ax3.set_title("Niche adjacency", fontsize=6.5, fontweight="normal")
    panel_label(ax, "F", x=-0.15)
    savefig(fig, "Fig4F_niches")


def fig4G_channels(cmeta, S, rep):
    maps = pd.read_pickle(TAB / "S8_channel_score_maps.pkl")
    d = cmeta.sort_values("r_meta")
    fig = plt.figure(figsize=(7.1, 3.4))
    gs = gridspec.GridSpec(2, 3, width_ratios=[1.25, 1, 1], wspace=0.25, hspace=0.25)
    ax = fig.add_subplot(gs[:, 0])
    y = np.arange(len(d))
    cols = [C_ACT if a == "activated" else C_QUI for a in d.arm]
    ax.hlines(y, d.r_lo, d.r_hi, color=cols, lw=0.8)
    ax.scatter(d.r_meta, y, c=cols, s=10, zorder=3, marker="o",
               edgecolors=["black" if q < 0.05 else "none" for q in d.q], linewidths=0.5)
    ax.axvline(0, color=C_GREY, lw=0.5, ls=(0, (2, 2)))
    ax.set_yticks(y)
    ax.set_yticklabels([f"{c} {'✓' if ok else '✗'}" for c, ok in zip(d.channel, d.direction_as_expected)],
                       fontsize=5)
    ax.set_xlabel("Meta partial ρ with SAP (leave-channel-out)", fontsize=6)
    ax.legend(handles=[Patch(color=C_ACT, label="activated arm (Fig. 4A)"),
                       Patch(color=C_QUI, label="quiescent arm (Fig. 4A)"),
                       Line2D([], [], marker="o", ls="", mfc="white", mec="black", label="q < 0.05")],
              frameon=False, fontsize=5, loc="lower right")
    for sp_ in ["top", "right"]:
        ax.spines[sp_].set_visible(False)
    a = read_section(rep)
    top = list(cmeta[cmeta.arm == "activated"].sort_values("r_meta", ascending=False).channel[:2]) + \
        list(cmeta[cmeta.arm == "quiescent"].sort_values("r_meta").channel[:2])
    for k, ch in enumerate(top[:4]):
        axm = fig.add_subplot(gs[k // 2, 1 + k % 2])
        v = maps.get((rep, ch))
        if v is not None:
            spatial_ax(axm, a, v, cmap="magma", title=ch, cbar_label="score")
    panel_label(ax, "G", x=-0.5)
    savefig(fig, "Fig4G_spatial_channels")


def fig4H_hallmark_maps(S, rep):
    a = read_section(rep)
    d = S.loc[a.obs_names]
    cols = [c for c in d.columns if c.startswith("AUC_HM_")][:9]
    fig, axes = plt.subplots(3, 3, figsize=(6.2, 6.0), gridspec_kw={"wspace": 0.1, "hspace": 0.3})
    for ax, c in zip(axes.ravel(), cols):
        v = d[c].values
        v = (v - np.nanmin(v)) / (np.nanmax(v) - np.nanmin(v) + 1e-12)
        spatial_ax(ax, a, v, cmap=HALLMARK_CMAP, vmin=0, vmax=np.nanquantile(v, 0.992), cbar=False,
                   title="HALLMARK_\n" + c[7:].upper().replace(" ", "_").replace("-", "_").replace("/", "_"))
        ax.title.set_fontsize(5.2)
    for ax in axes.ravel()[len(cols):]:
        ax.axis("off")
    panel_label(axes[0, 0], "H", x=0.0)
    savefig(fig, "Fig4H_hallmark_AUCell_maps")


def fig4I_niche_regression(nmeta, conc):
    fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.4), gridspec_kw={"wspace": 0.45})
    ax = axes[0]
    for c, col in [("Monocyte-derived TAM", C_ACT), ("Resident-like TAM", C_QUI)]:
        v = nmeta[nmeta.compartment == c].median_dR2_niche.dropna().values * 100
        if len(v):
            ax.hist(v, bins=40, color=col, alpha=0.6, label=c, density=True)
    ax.set_xlabel("Variance explained by neighbourhood (%)", fontsize=6)
    ax.set_ylabel("Density", fontsize=6)
    ax.legend(frameon=False, fontsize=5)
    ax = axes[1]
    f = TAB / "S9_niche_vs_Fig1G_Monocyte-derived_TAM.csv"
    if f.exists():
        x = pd.read_csv(f)
        sig = (x.padj < 0.05) & (x.log2FoldChange.abs() >= 0.5)
        ax.scatter(x.log2FoldChange[~sig], x.beta_meta[~sig], s=3, color="#D3D3D3", lw=0)
        ax.scatter(x.log2FoldChange[sig], x.beta_meta[sig], s=6,
                   c=np.where(x.log2FoldChange[sig] > 0, C_ACT, C_QUI), lw=0)
        lab = x[sig].reindex(x[sig].beta_meta.abs().sort_values(ascending=False).index).head(10)
        for _, r in lab.iterrows():
            ax.text(r.log2FoldChange, r.beta_meta, r.gene, fontsize=4.8, fontstyle="italic")
        rr = conc[conc.compartment == "Monocyte-derived TAM"]
        if len(rr):
            r0 = rr.iloc[0]
            ax.text(0.02, 0.98, f"Spearman ρ = {r0.spearman_rho:.2f}, P = {fmt_p(r0.p)}\n"
                                f"sign agreement (Fig. 1G DEGs) = {100 * r0.sign_agreement_DE_sig:.0f}%",
                    transform=ax.transAxes, va="top", fontsize=5.5)
        ax.axhline(0, color=C_GREY, lw=0.4, ls=(0, (2, 2)))
        ax.axvline(0, color=C_GREY, lw=0.4, ls=(0, (2, 2)))
        ax.set_xlabel("scRNA-seq: log$_2$FC per SD of SAP (Fig. 1G)", fontsize=6)
        ax.set_ylabel("Spatial: β neighbourhood SAP (meta)", fontsize=6)
    for a_ in axes:
        for sp_ in ["top", "right"]:
            a_.spines[sp_].set_visible(False)
    panel_label(axes[0], "I", x=-0.2)
    savefig(fig, "Fig4I_microenvironment_induced_TAM_expression")


def fig4J_drugs(dmeta):
    """Drug candidates: Tier 1/2 first (both families), then the best remaining mechanistic drugs."""
    tiered = dmeta[dmeta.tier.fillna("") != ""]
    rest = dmeta[(dmeta.tier.fillna("") == "") & (dmeta.family == "pre-specified mechanistic")]
    d = pd.concat([tiered.sort_values("p_meta").head(18), rest.sort_values("p_meta").head(max(0, 22 - len(tiered)))])
    if d.empty:
        return
    d = d.iloc[::-1]
    fig, ax = plt.subplots(figsize=(4.4, 0.19 * len(d) + 1.0))
    y = np.arange(len(d))
    cols = [C_ACT if r > 0 else C_QUI for r in d.r_meta]
    ax.hlines(y + 0.12, d.r_lo, d.r_hi, color=cols, lw=0.8)
    ax.scatter(d.r_meta, y + 0.12, c=cols, s=11, zorder=3,
               edgecolors=["black" if q < 0.05 else "none" for q in d.q], linewidths=0.5)
    if d.r_myeloid_attr.notna().any():
        ax.hlines(y - 0.12, d.r_myeloid_attr_lo, d.r_myeloid_attr_hi, color=C_GREY, lw=0.6)
        ax.scatter(d.r_myeloid_attr, y - 0.12, marker="D", s=8, facecolors="white", edgecolors=C_DARK,
                   linewidths=0.5, zorder=3)
    ax.axvline(0, color=C_GREY, lw=0.5, ls=(0, (2, 2)))
    ax.set_yticks(y)
    mark = {"Tier 1 (strict)": "[T1] ", "Tier 2 (candidate)": "[T2] "}
    tg = [t if isinstance(t, str) and t else (str(tt).split(";")[0] if isinstance(tt, str) else "")
          for t, tt in zip(d.mech_targets_hit, d.targets)]
    ax.set_yticklabels([f"{mark.get(t, '')}{n.split('|')[-1].title()} ({g})"
                        for n, t, g in zip(d.drug, d.tier.fillna(""), tg)], fontsize=5)
    for t, tier in zip(ax.get_yticklabels(), d.tier.fillna("")):
        if tier:
            t.set_fontweight("bold")
    tr = ax.get_yaxis_transform()
    for yi, sh in zip(y, d.myeloid_share):
        ax.text(1.02, yi, "" if not np.isfinite(sh) else f"{100 * sh:.0f}%", transform=tr, fontsize=5, va="center")
    ax.text(1.02, len(d) - 0.4, "myeloid\nshare", transform=tr, fontsize=5, va="bottom")
    ax.set_xlabel("Partial ρ: drug-target score vs neighbourhood SAP\n(myeloid-rich spots, pooled sections)",
                  fontsize=6)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=C_DARK, ms=3, label="spot-level score"),
                       Line2D([], [], marker="D", ls="", mfc="white", mec=C_DARK, ms=3,
                              label="myeloid-attributed (cell2location)"),
                       Line2D([], [], ls="", label="[T1] Tier 1 (strict)   [T2] Tier 2 (candidate)")],
              fontsize=5, frameon=False, loc="lower right")
    for sp_ in ["top", "right"]:
        ax.spines[sp_].set_visible(False)
    panel_label(ax, "J", x=-0.9)
    savefig(fig, "Fig4J_drug2cell_candidates")


def fig4K_concordance(meta):
    """Cross-scale concordance: scRNA-seq beta (Fig. 1E) vs spatial (this study)."""
    if not F_FIG1E.exists():
        return
    sc_b = pd.read_csv(F_FIG1E).set_index("state")["beta_clr_per_SD"]
    rows = []
    for an, lab_ in [("H1-4 within-spot SAP state (c2l)", "GSE251950 Visium (c2l)"),
                     ("Label-free signatures (no deconvolution)", "GSE251950 Visium (label-free)")]:
        m = meta[meta.analysis == an]
        for st in ["Mono_classical", "TAM_Inflammatory", "TAM_SPP1", "TAM_C1QC"]:
            x = m[m.state == st]
            if len(x) and st in sc_b:
                rows.append(dict(cohort=lab_, state=st, rho=x.r_meta.iloc[0], sc_beta=sc_b[st]))
    d = pd.DataFrame(rows)
    d.to_csv(TAB / "Fig4K_cross_scale_concordance.csv", index=False)
    marks = {"GSE251950 Visium (c2l)": "*", "GSE251950 Visium (label-free)": "P"}
    fig, ax = plt.subplots(figsize=(3.0, 2.7))
    for _, r in d.iterrows():
        ax.scatter(r.sc_beta, r.rho, marker=marks.get(r.cohort, "D"), s=28 if "Visium" in r.cohort else 16,
                   color=MYELOID_PALETTE.get(r.state, C_DARK), edgecolor="black" if "Visium" in r.cohort else "white",
                   lw=0.3, zorder=3)
    ax.axhline(0, color=C_GREY, lw=0.4, ls=(0, (2, 2)))
    ax.axvline(0, color=C_GREY, lw=0.4, ls=(0, (2, 2)))
    agree = int((np.sign(d.sc_beta) == np.sign(d.rho)).sum())
    ax.text(0.03, 0.97, f"Direction agrees: {agree}/{len(d)}", transform=ax.transAxes, va="top", fontsize=5.8)
    ax.set_xlabel("scRNA-seq: state ~ SAP (β, Fig. 1E)", fontsize=6)
    ax.set_ylabel("Spatial: partial ρ(SAP, state)", fontsize=6)
    h = [Line2D([], [], marker=marks[c], ls="", color=C_GREY, ms=4, label=c) for c in marks if c in set(d.cohort)]
    h += [Line2D([], [], marker="o", ls="", color=MYELOID_PALETTE[s], ms=4, label=s.replace("_", " "))
          for s in ["Mono_classical", "TAM_Inflammatory", "TAM_SPP1", "TAM_C1QC"]]
    ax.legend(handles=h, frameon=False, fontsize=4.8, loc="lower right")
    for sp_ in ["top", "right"]:
        ax.spines[sp_].set_visible(False)
    panel_label(ax, "K", x=-0.2)
    savefig(fig, "Fig4K_cross_scale_concordance")


def supp_figures(manifest, S, means, q05, inf, per, meta):
    qc = pd.read_csv(TAB / "S1_spot_qc.csv")
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.0), gridspec_kw={"wspace": 0.5})
    for ax, c, lab_ in [(axes[0], "spots_after_qc", "Spots after QC"), (axes[1], "median_umi", "Median UMI/spot"),
                        (axes[2], "median_genes", "Median genes/spot")]:
        ax.bar(qc["sample"], qc[c], color=[C_ACT if s in PRIMARY else C_GREY for s in qc["sample"]])
        ax.set_ylabel(lab_, fontsize=6)
        ax.tick_params(axis="x", rotation=90, labelsize=5.5)
    axes[0].set_title("red = analysed sections", fontsize=5.5, fontweight="normal")
    savefig(fig, "FigS_spatial_QC", FIG_SUPP)
    # reference signatures: marker check
    mk = {"Malignant": ["EPCAM", "KRT8"], "Fibroblast": ["DCN", "LUM"], "Pericyte_SMC": ["RGS5", "ACTA2"],
          "Endothelial": ["PECAM1", "VWF"], "Mono_classical": ["FCN1", "VCAN"], "TAM_Inflammatory": ["IL1B", "CXCL8"],
          "TAM_SPP1": ["SPP1", "GPNMB"], "TAM_C1QC": ["C1QA", "C1QB"], "T_NK": ["CD3D", "NKG7"], "Plasma": ["MZB1", "JCHAIN"]}
    genes = [g for v in mk.values() for g in v if g in inf.index]
    R = inf.loc[genes]
    R = R.div(R.max(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    im = ax.imshow(R.values, aspect="auto", cmap="Reds")
    ax.set_yticks(range(len(genes)))
    ax.set_yticklabels(genes, fontsize=5, fontstyle="italic")
    ax.set_xticks(range(R.shape[1]))
    ax.set_xticklabels([c.replace("_", " ") for c in R.columns], rotation=90, fontsize=5)
    plt.colorbar(im, ax=ax, fraction=0.03).set_label("Signature (row-max scaled)", fontsize=5)
    savefig(fig, "FigS_c2l_reference_signatures", FIG_SUPP)
    # deconvolution sanity: c2l abundance vs label-free signatures
    rows = []
    for s, d in S.groupby("sample"):
        for lf, ct in [("LF_Fate_TAM_C1QC", "ab_TAM_C1QC"), ("LF_Fate_TAM_Inflammatory", "ab_TAM_Inflammatory"),
                       ("LF_Fate_TAM_SPP1", "ab_TAM_SPP1"), ("LF_Mono_classical", "ab_Mono_classical"),
                       ("LF_Myeloid_abundance", "momac_cells")]:
            if lf in d and ct in d:
                rows.append(dict(sample=s, pair=f"{lf} ~ {ct}",
                                 rho=stats.spearmanr(d[lf], d[ct], nan_policy="omit")[0]))
    t = pd.DataFrame(rows)
    t.to_csv(TAB / "S4_c2l_vs_labelfree_spearman.csv", index=False)
    # SAP maps for every section
    secs = list(manifest)
    nc = 5
    nr = int(np.ceil(len(secs) / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(7.1, 1.6 * nr + 0.3), squeeze=False)
    for ax, s in zip(axes.ravel(), secs):
        a = read_section(s)
        spatial_ax(ax, a, S.loc[a.obs_names, "SAP_state"].values, cmap=SAP_CMAP, vmin=-2.5, vmax=2.5,
                   title=s, cbar=False)
    for ax in axes.ravel()[len(secs):]:
        ax.axis("off")
    savefig(fig, "FigS_SAP_maps_all_sections", FIG_SUPP)
    # per-section heatmap of all states (within-spot analysis)
    an = "H1-4 within-spot SAP state (c2l)"
    p = per[per.analysis == an].pivot_table(index="state", columns="sample", values="r")
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    v = np.nanmax(np.abs(p.values))
    im = ax.imshow(p.values, cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto")
    ax.set_yticks(range(len(p)))
    ax.set_yticklabels([i.replace("_", " ") for i in p.index], fontsize=5.5)
    ax.set_xticks(range(p.shape[1]))
    ax.set_xticklabels(p.columns, rotation=90, fontsize=5.5)
    plt.colorbar(im, ax=ax, fraction=0.03).set_label("Partial ρ (SAP state, CLR)", fontsize=5)
    savefig(fig, "FigS_state_tests_per_section", FIG_SUPP)




def methods_numbers(S, meta, null_sum, cmeta, conc, dmeta, k_niche):
    qc = pd.read_csv(TAB / "S1_spot_qc.csv")
    qs = qc[qc["sample"].isin(PRIMARY)]
    sel = json.loads((TAB / "S1_selected_sections.json").read_text())
    L = [f"GSE251950: {len(qc)} sections QC'd; {len(PRIMARY)} primary sections analysed ({', '.join(PRIMARY)}), "
         f"selected by: {sel['rule']}; {int(qs.spots_after_qc.sum()):,} spots after QC "
         f"(median UMI {qs.median_umi.median():.0f}).",
         f"Pooling across sections: {META_METHOD} (inverse-variance fixed effect; DL random-effects P reported as "
         f"p_meta_RE). Inference is conditional on the analysed sections.",
         f"cell2location: N_cells_per_location={N_CELLS_PER_SPOT}, detection_alpha={DETECTION_ALPHA}, "
         f"{C2L_EPOCHS} epochs; SAP-blind gene panel ({len(pd.read_csv(TAB / 'S3_c2l_gene_panel_main.csv'))} genes).",
         f"Stromal spots: {int(S.stroma_spot.sum()):,}; myeloid spots: {int(S.myeloid_spot.sum()):,}; "
         f"both: {int((S.stroma_spot & S.myeloid_spot).sum()):,}.",
         f"SAP signature coverage on Visium: {S.groupby('sample').SAP_cov.first().min():.0%}-"
         f"{S.groupby('sample').SAP_cov.first().max():.0%}.",
         f"NMF niches: k = {k_niche}."]
    for _, r in meta[meta.family == "pre-specified"].iterrows():
        L.append(f"  [{r.analysis}] {r.state}: meta rho {r.r_meta:+.3f} ({r.r_lo:+.3f} to {r.r_hi:+.3f}), "
                 f"P = {r.p_meta:.2e}, q = {r.q:.2e}, I2 = {100 * r.I2:.0f}%, {r.n_pos}+/{r.n_neg}-")
    for _, r in null_sum.iterrows():
        L.append(f"  random stromal null {r.state}: observed {r.r_obs:+.3f}, null median {r.null_median:+.3f}, "
                 f"P_emp = {r.p_emp_directional:.3f}")
    if len(cmeta):
        L.append(f"Channels in scRNA-seq direction: {int(cmeta.direction_as_expected.sum())}/{len(cmeta)}; "
                 f"q<0.05: {int((cmeta.q < 0.05).sum())}")
    for _, r in conc.iterrows():
        L.append(f"  niche regression vs Fig1G [{r.compartment}]: rho = {r.spearman_rho:+.3f} (P = {r.p:.2e}), "
                 f"sign agreement {100 * r.sign_agreement_DE_sig:.0f}% of {int(r.n_DE_sig)} DEGs")
    if dmeta is not None and len(dmeta):
        for fam in ["pre-specified mechanistic", "exploratory"]:
            mm = dmeta[dmeta.family == fam]
            L.append(f"drug2cell [{fam}]: {len(mm)} drugs; Tier 1 (q<{D2C_TIER1['q']}, attributed q<"
                     f"{D2C_TIER1['q_myeloid']}, myeloid share>={D2C_TIER1['share']:.0%}): "
                     f"{int((mm.tier == 'Tier 1 (strict)').sum())}; Tier 2 (q<{D2C_TIER2['q']}, attributed P<"
                     f"{D2C_TIER2['p_myeloid']}, share>={D2C_TIER2['share']:.0%}, same sign in every section): "
                     f"{int((mm.tier == 'Tier 2 (candidate)').sum())}")
    for pkg in ["scanpy", "anndata", "cell2location", "scvi", "liana", "drug2cell", "sklearn", "scipy", "numpy"]:
        try:
            L.append(f"  {pkg} {importlib.import_module(pkg).__version__}")
        except Exception:
            pass
    (TAB / "methods_numbers_spatial.txt").write_text("\n".join(L))
    log("\n".join(L))



def main():
    force_from_argv({"figs"})
    require("step01", "step02", "step03", "step04", "step05", "step06")
    log("=" * 70)
    log("STEP 07: figures")
    manifest = load_manifest()
    hyp = load_hyp()
    S, inf, means, q05 = load_S(), load_inf(), load_means(), load_means("q05")
    per = pd.read_csv(TAB / "S5_state_tests_per_section.csv")
    meta = pd.read_csv(TAB / "S5_state_tests_meta.csv")
    null_sum = pd.read_csv(TAB / "S5_random_stromal_null_summary.csv")
    prof, gmeta = pd.read_csv(TAB / "S6_gradient_profiles.csv"), pd.read_csv(TAB / "S6_gradient_tests_meta.csv")
    niche_lab = pd.read_csv(TAB / "S7_niche_assignment.csv", index_col=0)
    H = pd.read_csv(TAB / "S7_niche_loadings.csv", index_col=0)
    Zm = pd.read_csv(TAB / "S7_nhood_enrichment_mean_z.csv", index_col=0)
    cmeta = pd.read_csv(TAB / "S8_channel_tests_meta.csv")
    nmeta, conc = pd.read_csv(TAB / "S9_niche_regression_meta.csv"), pd.read_csv(TAB / "S9_concordance_with_Fig1G.csv")
    dmeta = pd.read_csv(TAB / "S11_drug2cell_meta.csv")
    rep = representative_section(S)
    log(f"FIG representative section: {rep}")
    panels = [("B", lambda: fig4B_celltype_maps(S, q05, rep)),
              ("C", lambda: fig4C_sap_niche_maps(S, niche_lab, rep)),
              ("D", lambda: fig4D_forests(per, meta, null_sum, hyp)),
              ("E", lambda: fig4E_gradients(prof, gmeta, hyp)),
              ("F", lambda: fig4F_niches(S, niche_lab, H, Zm)),
              ("G", lambda: fig4G_channels(cmeta, S, rep)),
              ("H", lambda: fig4H_hallmark_maps(S, rep)),
              ("I", lambda: fig4I_niche_regression(nmeta, conc)),
              ("J", lambda: fig4J_drugs(dmeta)),
              ("K", lambda: fig4K_concordance(meta)),
              ("supp", lambda: supp_figures(manifest, S, means, q05, inf, per, meta))]
    failed = []
    for name, fn in panels:                     # one failing panel does not stop the others
        try:
            fn()
        except Exception as e:
            failed.append(name)
            log(f"  panel {name} FAILED: {type(e).__name__}: {e}")
    methods_numbers(S, meta, null_sum, cmeta, conc, dmeta, len(H))
    mark_done("step07", failed_panels=failed)
    log("DONE. Inspect FigS_c2l_*_ELBO (convergence), FigS_c2l_reference_signatures (markers) and "
        "S4_c2l_vs_labelfree_spearman.csv (deconvolution sanity) before interpreting Fig. 4B-K.")


if __name__ == "__main__":
    main()
