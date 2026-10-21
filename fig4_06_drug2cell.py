#!/usr/bin/env python
# coding: utf-8
"""
 fig4_06_drug2cell.py  (step 6 of 7)
 S11 drug2cell (ChEMBL drug-target sets) scored per spot, tested against neighbourhood SAP in
     myeloid-rich spots (adjusted for myeloid and stromal abundance), per section with a
     per-section checkpoint, pooled across the 3 sections.
     Cell of origin: the same test on the MoMac-attributed target expression (cell2location)
     and the share of target transcripts coming from monocytes/macrophages.
     Output: tables/S11_drug_candidates.csv  (Tier 1 = strict, Tier 2 = candidate)
 RUN: python fig4_06_drug2cell.py      (--force to redo)
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


# ==============================================================================
# 4. GEOMETRY HELPERS (Visium hexagonal lattice)
# ==============================================================================
HEX_OFFSETS = [(0, 2), (0, -2), (1, 1), (1, -1), (-1, 1), (-1, -1)]


# ==============================================================================
# 5. GENE SETS / SCORING HELPERS
# ==============================================================================
EXCLUDE_GENE_RE = re.compile(r"^(MT-|RP[SL]\d|RPLP|HB[ABDEGMQZ]\d?$|MTRNR|IG[HKL][VJCDG]|TR[ABDG][VJC])")


def load_frozen_signatures():
    t = pd.read_csv(F_SIGS)
    sap = t[t.signature == "SAP"].set_index("gene")
    sigs = {k: d.set_index("gene")["weight"] for k, d in t[t.signature != "SAP"].groupby("signature")}
    return sap, sigs


def read_section(s):
    return sc.read_h5ad(H5 / f"S1_{s}.h5ad")


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


def load_inf():
    return pd.read_csv(TAB / "S2_reference_signatures.csv", index_col=0)


def load_means(kind="means", tag="main"):
    return pd.read_csv(TAB / f"S3_c2l_abundance_{kind}_{tag}.csv", index_col=0)


def load_S():
    return pd.read_csv(TAB / "S4_spot_scores.csv", index_col=0)


# ==============================================================================
# STEP-SPECIFIC CODE
# ==============================================================================
def _targets_of(var):
    for col in ("genes", "targets"):
        if col in var:
            return var[col].apply(lambda g: list(g) if isinstance(g, (list, tuple, np.ndarray))
                                  else [x for x in re.split(r"[,;]", str(g)) if x])
    raise KeyError("drug2cell output has no 'genes' column")


def myeloid_attribution_matrix(a, A, inf, genes):
    """Spots x genes: fraction of each gene's expected expression contributed by the
    monocyte/macrophage states (cell2location: a_sc * r_gc), and the attributed log CP10k."""
    genes = [g for g in genes if g in inf.index and g in a.var_names]
    if not genes:
        return None, None, []
    cols = [c for c in A.columns if c in inf.columns]
    mom = [c for c in MOMAC if c in cols]
    R = inf.loc[genes, cols].values.astype(float)                       # G x C
    den = A[cols].values @ R.T                                          # S x G
    num = A[mom].values @ inf.loc[genes, mom].values.astype(float).T    # S x G
    frac = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    X = a[:, genes].layers["cp10k"]
    X = X.toarray() if sp.issparse(X) else np.asarray(X)
    return frac, np.log1p(X * frac), genes


def drug2cell_section(s, d2c, S, means, inf):
    """One section: drug2cell scores + myeloid-attributed target scores, tested against
    neighbourhood SAP in myeloid-rich spots. Written to its own file (checkpoint)."""
    fsec = TAB / f"S11_drug2cell_{s}.partial.csv"
    ftg = TAB / f"S11_drug_targets_{s}.partial.json"
    if fsec.exists() and ftg.exists() and "drug" not in FORCE:
        log(f"  S11 {s}: re-used checkpoint")
        return pd.read_csv(fsec), json.loads(ftg.read_text())
    a = lognorm(read_section(s))
    try:
        d2c.score(a, use_raw=False, method="seurat")
    except Exception:
        d2c.score(a, use_raw=False, method="seurat", nested=True)
    D = a.uns["drug2cell"]
    tg = _targets_of(D.var)
    det_frac = np.asarray((a.layers["counts"] > 0).sum(axis=0)).ravel() / a.n_obs
    detected = set(a.var_names[det_frac >= D2C_MIN_DETECT])
    d = S.loc[a.obs_names]
    A = means.loc[a.obs_names]
    Xd = D.X.toarray() if sp.issparse(D.X) else np.asarray(D.X)
    mask = d.myeloid_spot.values
    cov = d[["log_momac", "log_stroma"]].values
    T = SpatialCorr(d[["x_um", "y_um"]].values)
    all_t = sorted(set().union(*[set(t) for t in tg]) & detected)
    frac, Lm_, gl = myeloid_attribution_matrix(a, A, inf, all_t)
    gpos = {g: i for i, g in enumerate(gl)}
    if Lm_ is not None:
        Lm = Lm_[mask]
        mu, sd = Lm.mean(axis=0), Lm.std(axis=0)
        sd[sd == 0] = np.nan
        Zl = (Lm_ - mu) / sd
        Xg = a[:, gl].layers["cp10k"]
        Xg = Xg.toarray() if sp.issparse(Xg) else np.asarray(Xg)
    rows, targets = [], {}
    for j, drug in enumerate(D.var_names):
        genes = set(tg.iloc[j])
        targets[drug] = sorted(genes)
        if not (genes & detected):
            continue
        r = T.test(d.SAP_nbr.values, Xd[:, j], mask=mask, covars=cov)
        row = dict(sample=s, drug=drug, **r)
        idx = [gpos[g] for g in genes if g in gpos]
        if idx:
            zs = np.nanmean(Zl[:, idx], axis=1)
            ra = T.test(d.SAP_nbr.values, zs, mask=mask, covars=cov)
            row.update({f"{k}_myel": v for k, v in ra.items()})
            xm = Xg[mask][:, idx]
            tot = xm.sum()
            row["myeloid_share"] = float((xm * frac[mask][:, idx]).sum() / tot) if tot > 0 else np.nan
        rows.append(row)
    per = pd.DataFrame(rows)
    per.to_csv(fsec, index=False)
    ftg.write_text(json.dumps(targets))
    log(f"  S11 {s}: {len(per)} drugs with a detected target")
    return per, targets


def drug2cell_analysis(manifest, S, means, inf, sap):
    """drug2cell target scores vs neighbourhood SAP in myeloid-rich spots, with a cell-of-origin
    check (MoMac-attributed target score + myeloid share of target transcripts).
    Two transparent tiers (thresholds frozen in the pre-registration):
      Tier 1 (strict)    q < 0.05, myeloid-attributed q < 0.05, same sign, myeloid share >= 50%
      Tier 2 (candidate) q < 0.10, myeloid-attributed P < 0.05, same sign, myeloid share >= 30%,
                         same direction in every section tested
    Drugs whose targets are SAP-signature genes are excluded (trivially SAP-correlated)."""
    f = TAB / "S11_drug2cell_meta.csv"
    if f.exists() and "drug" not in FORCE:
        return pd.read_csv(f)
    d2c = ensure_pkg("drug2cell", "drug2cell")
    sap_genes = set(sap.index)
    mom_ref = [c for c in MOMAC if c in inf.columns]
    non_ref = [c for c in inf.columns if c not in MYELOID_STATES]
    ref_ratio = inf[mom_ref].max(axis=1) / inf[non_ref].max(axis=1).clip(lower=1e-6)
    pers, targets = [], {}
    for s in PRIMARY:
        if s not in manifest:
            continue
        p, t = drug2cell_section(s, d2c, S, means, inf)
        pers.append(p)
        targets.update({k: set(v) for k, v in t.items()})
    per = pd.concat(pers, ignore_index=True)
    per.to_csv(TAB / "S11_drug2cell_per_section.csv", index=False)
    meta = meta_rows(per, "drug")
    if "z_myel" in per:
        mm = meta_rows(per.dropna(subset=["z_myel"]), "drug", z="z_myel", se="se_myel", r="r_myel")
        mm = mm[["drug", "r_meta", "r_lo", "r_hi", "p_meta", "k_sections"]].rename(
            columns={"r_meta": "r_myeloid_attr", "r_lo": "r_myeloid_attr_lo", "r_hi": "r_myeloid_attr_hi",
                     "p_meta": "p_myeloid_attr", "k_sections": "k_myeloid_attr"})
        meta = meta.merge(mm, on="drug", how="left")
        meta = meta.merge(per.groupby("drug").myeloid_share.mean().rename("myeloid_share"),
                          left_on="drug", right_index=True, how="left")
    else:
        for c in ["r_myeloid_attr", "r_myeloid_attr_lo", "r_myeloid_attr_hi", "p_myeloid_attr",
                  "k_myeloid_attr", "myeloid_share"]:
            meta[c] = np.nan
    meta = meta[meta.k_sections >= D2C_MIN_SECTIONS].copy()
    meta["drug_name"] = meta.drug.astype(str).str.split("|").str[-1].str.title()
    meta["targets"] = meta.drug.map(lambda x: ";".join(sorted(targets.get(x, []))))
    meta["targets_myeloid_enriched_ref"] = meta.drug.map(
        lambda x: ";".join(sorted(g for g in targets.get(x, set())
                                  if g in ref_ratio.index and ref_ratio[g] >= MYELOID_SPEC_RATIO)))
    meta["mechanistic"] = meta.drug.map(lambda x: bool(targets.get(x, set()) & MECH_TARGETS))
    meta["mech_targets_hit"] = meta.drug.map(lambda x: ";".join(sorted(targets.get(x, set()) & MECH_TARGETS)))
    meta["circular_SAP_target"] = meta.drug.map(lambda x: bool(targets.get(x, set()) & sap_genes))
    meta["family"] = np.where(meta.circular_SAP_target, "excluded (target in SAP)",
                              np.where(meta.mechanistic, "pre-specified mechanistic", "exploratory"))
    meta["q"], meta["q_myeloid_attr"] = np.nan, np.nan
    for fam, dd in meta.groupby("family"):
        if not fam.startswith("excluded"):
            meta.loc[dd.index, "q"] = bh(dd.p_meta.values)
            meta.loc[dd.index, "q_myeloid_attr"] = bh(dd.p_myeloid_attr.values)
    same = np.sign(meta.r_meta) == np.sign(meta.r_myeloid_attr)
    consistent = np.maximum(meta.n_pos, meta.n_neg) == meta.k_sections
    ok = ~meta.circular_SAP_target
    t1 = ok & (meta.q < D2C_TIER1["q"]) & (meta.q_myeloid_attr < D2C_TIER1["q_myeloid"]) & same & \
        (meta.myeloid_share >= D2C_TIER1["share"])
    t2 = ok & ~t1 & (meta.q < D2C_TIER2["q"]) & (meta.p_myeloid_attr < D2C_TIER2["p_myeloid"]) & same & \
        (meta.myeloid_share >= D2C_TIER2["share"]) & (consistent if D2C_TIER2["consistent"] else True)
    meta["consistent_all_sections"] = consistent
    meta["tier"] = np.select([t1, t2], ["Tier 1 (strict)", "Tier 2 (candidate)"], default="")
    meta["direction"] = np.where(meta.r_meta > 0, "higher next to SAP-high stroma", "lower next to SAP-high stroma")
    sig = meta.q < D2C_TIER2["q"]
    meta["cell_of_origin"] = np.select(
        [~sig, meta.tier != "", sig & (meta.myeloid_share < D2C_TIER2["share"])],
        ["", "myeloid-supported", "SAP-associated, but target expressed mainly by non-myeloid cells"],
        default="not myeloid-resolved (spot-level only)")
    meta = meta.sort_values(["tier", "p_meta"], ascending=[False, True])
    meta = pd.concat([meta[meta.tier != ""].sort_values(["tier", "p_meta"]), meta[meta.tier == ""]])
    meta.to_csv(f, index=False)
    cand = meta[meta.tier != ""]
    cand[["drug_name", "drug", "tier", "family", "direction", "r_meta", "q", "r_myeloid_attr", "p_myeloid_attr",
          "myeloid_share", "targets", "mech_targets_hit", "targets_myeloid_enriched_ref", "n_pos", "n_neg",
          "k_sections"]].to_csv(TAB / "S11_drug_candidates.csv", index=False)
    for fam in ["pre-specified mechanistic", "exploratory"]:
        x = meta[meta.family == fam]
        log(f"S11 [{fam}]: {len(x)} drugs; Tier 1: {int((x.tier == 'Tier 1 (strict)').sum())}, "
            f"Tier 2: {int((x.tier == 'Tier 2 (candidate)').sum())}")
    if len(cand):
        log("S11 candidates:\n" + cand[["drug_name", "tier", "family", "r_meta", "q", "myeloid_share",
                                        "mech_targets_hit"]].head(30).round(3).to_string(index=False))
    return meta


def main():
    force_from_argv({"drug"})
    require("step01", "step02", "step03")
    if step_done("step06") and "drug" not in FORCE:
        log("step06 already done (use --force to redo)")
        return
    log("=" * 70)
    log(f"STEP 06: drug2cell on {PRIMARY}")
    manifest = load_manifest()
    sap, _ = load_frozen_signatures()
    S, inf, means = load_S(), load_inf(), load_means()
    dmeta = drug2cell_analysis(manifest, S, means, inf, sap)
    mark_done("step06", drugs=len(dmeta), tier1=int((dmeta.tier == "Tier 1 (strict)").sum()),
              tier2=int((dmeta.tier == "Tier 2 (candidate)").sum()))


if __name__ == "__main__":
    main()
