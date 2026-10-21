#!/usr/bin/env python
# coding: utf-8
"""
 fig4_01_download_qc_select.py  (step 1 of 7)
 S0  download + assembly of GSE251950 (10 Visium sections; GC6-PM = peritoneal metastasis)
 S1  Ensembl -> GENCODE v44 harmonisation, spot QC (all 10 sections, per-section checkpoint)
 S1b outcome-blind selection of the 3 best PRIMARY sections (quality, stroma-myeloid
     co-occurrence, SAP spread) -> tables/S1_selected_sections.json, used by all later steps
 S0  pre-registration (hypotheses, thresholds, selected sections) frozen BEFORE any test
 RUN: python fig4_01_download_qc_select.py      (--force to redo)
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


# ==============================================================================
# STEP-SPECIFIC CODE
# ==============================================================================
def preregister(selection):
    f = TAB / "S0_preregistration.json"
    if f.exists() and "select" not in FORCE:
        return json.loads(f.read_text())
    hyp = dict(HYP_STATES_FALLBACK)
    if F_FIG1E.exists():
        e = pd.read_csv(F_FIG1E)
        e = e[e.fdr < 0.05]
        hyp = {r.state: int(np.sign(r.beta_clr_per_SD)) for _, r in e.iterrows()}
    reg = dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), dataset=GSE,
               primary_sections_available=PRIMARY_ALL, excluded_from_inference=["GC6-PM"],
               analysed_sections=selection["selected"], selection_rule=selection["rule"],
               pooling=f"{META_METHOD} (fixed-effect inverse-variance; DL random-effects P as sensitivity)",
               drug2cell_tiers=dict(tier1=D2C_TIER1, tier2=D2C_TIER2, min_detect=D2C_MIN_DETECT,
                                    min_sections=D2C_MIN_SECTIONS),
               H1_H4_state_directions=hyp,
               H5="same directions for neighbourhood SAP (rings 1-2, own spot excluded)",
               H6_channels=[dict(ligand=l, receptor=r, source=s, target=t, expected_sign=e, tier=k)
                            for l, r, s, t, e, k in CHANNELS],
               H6b_regulons_vs_neighbourhood_SAP=REGULON_DIR,
               H7="meta beta(neighbourhood SAP) for monocyte-derived-TAM genes concordant with Fig1G LFC",
               unit="tissue section; Dutilleul-corrected spot correlations, DL random-effects pooling",
               thresholds=dict(MIN_STROMA_CELLS=MIN_STROMA_CELLS, MIN_MYELOID_CELLS=MIN_MYELOID_CELLS,
                               NEIGHBOUR_RADIUS_UM=NEIGHBOUR_RADIUS_UM, N_CELLS_PER_SPOT=N_CELLS_PER_SPOT))
    f.write_text(json.dumps(reg, indent=2))
    log(f"S0 pre-registration frozen: {hyp}")
    return reg


def fetch(url, dest, tries=4):
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    urls = [url] + ([url.replace("https://", "http://"), url.replace("https://", "ftp://")]
                    if "ftp.ncbi.nlm.nih.gov" in url else [])
    for k in range(tries):
        for u in urls:
            log(f"  downloading {u} (attempt {k + 1}/{tries})")
            try:
                req = urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=1800) as r, open(tmp, "wb") as f:
                    shutil.copyfileobj(r, f, 1 << 20)
                tmp.rename(dest)
                return dest
            except Exception as e:
                log(f"    urllib failed: {e}")
            for tool, cmd in (("curl", ["curl", "-L", "-sS", "--fail", "-C", "-", "--retry", "3", "-o", str(tmp), u]),
                              ("wget", ["wget", "-q", "-c", "--tries=3", "-O", str(tmp), u])):
                if shutil.which(tool):
                    try:
                        subprocess.run(cmd, check=True)
                        if tmp.exists() and tmp.stat().st_size > 0:
                            tmp.rename(dest)
                            return dest
                    except Exception as e:
                        log(f"    {tool} failed: {e}")
        time.sleep(15 * (k + 1))
    raise RuntimeError(f"Download failed (network). Download it in a browser/VPN and save it as:\n"
                       f"  {dest}\n  from {url}\nthen re-run (the file is picked up automatically).")


def safe_extract(tar_path, dest):
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:*") as tf:
        root = dest.resolve()
        for m in tf.getmembers():
            target = (dest / m.name).resolve()
            if not str(target).startswith(str(root)) or m.issym() or m.islnk():
                raise RuntimeError(f"unsafe path in {tar_path}: {m.name}")
        try:
            tf.extractall(dest, filter="data")
        except TypeError:
            tf.extractall(dest)


def download_gse251950():
    """GSE251950_RAW.tar -> one tar.gz of SpaceRanger outputs per GSM -> RAW/<sample>/"""
    manifest = RAW / "manifest.json"
    if manifest.exists():
        return json.loads(manifest.read_text())
    local = RAW / "GSE251950_RAW.tar"
    if not local.exists():
        log(f"  (if the download is blocked, put GSE251950_RAW.tar manually in {RAW})")
    raw_tar = fetch(URL_RAW, RAW / "GSE251950_RAW.tar")
    try:
        mtx = fetch(URL_MATRIX, RAW / "GSE251950_series_matrix.txt.gz")
        with gzip.open(mtx, "rt", errors="replace") as fh:
            meta = {l.split("\t", 1)[0]: [v.strip().strip('"') for v in l.rstrip("\n").split("\t")[1:]]
                    for l in fh if l.startswith(("!Sample_geo_accession", "!Sample_title"))}
        geo_titles = dict(zip(meta["!Sample_geo_accession"], meta["!Sample_title"]))
        for gsm, (name, _) in SAMPLES.items():
            if geo_titles.get(gsm, name).replace(" ", "") != name:
                log(f"  WARNING: GEO title of {gsm} is '{geo_titles[gsm]}', table says '{name}' - check SAMPLES")
    except Exception as e:
        log(f"  series matrix check skipped ({e}); using the verified GSM table")
    stage = RAW / "_gsm_tars"
    if not stage.exists():
        safe_extract(raw_tar, stage)
    out = {}
    for gsm, (name, site) in SAMPLES.items():
        hits = sorted(stage.glob(f"{gsm}*"))
        if not hits:
            raise FileNotFoundError(f"{gsm} not found inside GSE251950_RAW.tar")
        sdir = RAW / name
        if not sdir.exists():
            for h in hits:
                if tarfile.is_tarfile(h):
                    safe_extract(h, sdir)
                else:
                    sdir.mkdir(parents=True, exist_ok=True)
                    shutil.copy(h, sdir / h.name)
        files = {p.name: str(p) for p in sdir.rglob("*") if p.is_file()}

        def pick(*names):
            for n in names:
                for k, v in files.items():
                    if k == n or k.endswith(n):
                        return v
            return None
        out[name] = dict(gsm=gsm, site=site,
                         h5=pick("filtered_feature_bc_matrix.h5"),
                         mtx=pick("matrix.mtx.gz", "matrix.mtx"),
                         positions=pick("tissue_positions_list.csv", "tissue_positions.csv"),
                         scalefactors=pick("scalefactors_json.json"),
                         hires=pick("tissue_hires_image.png", "tissue_hires_image.jpg"),
                         lowres=pick("tissue_lowres_image.png", "tissue_lowres_image.jpg"))
        missing = [k for k in ("positions", "scalefactors") if out[name][k] is None]
        if missing or (out[name]["h5"] is None and out[name]["mtx"] is None):
            raise FileNotFoundError(f"{name}: SpaceRanger files missing ({missing}); found {list(files)}")
    manifest.write_text(json.dumps(out, indent=2))
    log(f"S0 GSE251950 assembled: {list(out)}")
    return out




def _read_counts(rec):
    if rec["h5"]:
        a = sc.read_10x_h5(rec["h5"])
    else:
        a = sc.read_10x_mtx(str(Path(rec["mtx"]).parent), var_names="gene_symbols", make_unique=False)
    if "gene_ids" not in a.var:
        a.var["gene_ids"] = a.var_names
    return a


def _read_positions(path):
    p = pd.read_csv(path, header=None)
    if str(p.iloc[0, 0]).lower() == "barcode":
        p = pd.read_csv(path)
    else:
        p.columns = ["barcode", "in_tissue", "array_row", "array_col", "pxl_row_in_fullres", "pxl_col_in_fullres"]
    return p.set_index("barcode")


def _read_image(path):
    if not path:
        return None
    from PIL import Image
    img = np.asarray(Image.open(path).convert("RGB")).astype(np.float32) / 255.0
    return img


def harmonise_genes(a, id2name):
    """Ensembl ID -> GENCODE v44 symbol (identical to the atlas, Fig 1), duplicates summed."""
    ens = a.var["gene_ids"].astype(str).str.split(".").str[0].values
    sym = np.array([id2name.get(e, s) for e, s in zip(ens, a.var_names.astype(str))])
    X = a.X.tocsr() if sp.issparse(a.X) else sp.csr_matrix(a.X)
    codes, uniq = pd.factorize(pd.Series(sym))
    G = sp.csr_matrix((np.ones(len(codes), np.float32), (np.arange(len(codes)), codes)),
                      shape=(len(codes), len(uniq)))
    b = ad.AnnData(X=(X @ G).tocsr().astype(np.float32), obs=a.obs.copy(), var=pd.DataFrame(index=uniq))
    b.uns = dict(a.uns)
    for k in a.obsm.keys():
        b.obsm[k] = np.asarray(a.obsm[k])
    return b


def load_and_qc(manifest):
    done = all((H5 / f"S1_{s}.h5ad").exists() for s in manifest) and (TAB / "S1_spot_qc.csv").exists()
    if done and "qc" not in FORCE:
        log("S1 QC: already done (checkpoint)")
        return
    id2name = gene_id_map()
    rows = []
    for s, rec in manifest.items():
        if (H5 / f"S1_{s}.h5ad").exists() and (TAB / f"S1_spot_qc_{s}.json").exists() and "qc" not in FORCE:
            rows.append(json.loads((TAB / f"S1_spot_qc_{s}.json").read_text()))     # per-section checkpoint
            continue
        a = _read_counts(rec)
        pos = _read_positions(rec["positions"])
        pos = pos[pos.in_tissue.astype(int) == 1]
        common = a.obs_names.intersection(pos.index)
        a = a[common].copy()
        pos = pos.loc[common]
        sf = json.loads(Path(rec["scalefactors"]).read_text())
        a.obs["array_row"], a.obs["array_col"] = pos.array_row.values, pos.array_col.values
        a.obsm["spatial"] = pos[["pxl_col_in_fullres", "pxl_row_in_fullres"]].to_numpy(float)
        a.uns["spatial"] = {s: dict(images={k: v for k, v in (("hires", _read_image(rec["hires"])),
                                                              ("lowres", _read_image(rec["lowres"])))
                                            if v is not None},
                                    scalefactors=sf, metadata=dict(source=GSE, gsm=rec["gsm"]))}
        a = harmonise_genes(a, id2name)
        a.obs_names = [f"{s}_{b}" for b in a.obs_names]
        a.obs["sample"], a.obs["site"], a.obs["gsm"] = s, rec["site"], rec["gsm"]
        mt = a.var_names.str.startswith("MT-")
        tot = np.asarray(a.X.sum(axis=1)).ravel()
        a.obs["total_counts"] = tot
        a.obs["n_genes"] = np.asarray((a.X > 0).sum(axis=1)).ravel()
        a.obs["pct_mt"] = 100 * np.asarray(a[:, mt].X.sum(axis=1)).ravel() / np.maximum(tot, 1)
        keep = (a.obs.total_counts >= QC_MIN_COUNTS) & (a.obs.n_genes >= QC_MIN_GENES) & \
               (a.obs.pct_mt <= QC_MAX_PCT_MT)
        rows.append(dict(sample=s, gsm=rec["gsm"], site=rec["site"], spots_in_tissue=a.n_obs,
                         spots_after_qc=int(keep.sum()), median_umi=float(np.median(tot)),
                         median_genes=float(np.median(a.obs.n_genes)), median_pct_mt=float(a.obs.pct_mt.median()),
                         px_per_um=sf["spot_diameter_fullres"] / SPOT_DIAMETER_UM))
        a = a[keep.values].copy()
        a.obsm["MT"] = a[:, mt].X.toarray()
        a = a[:, ~mt].copy()
        a.write_h5ad(H5 / f"S1_{s}.h5ad", compression="gzip")
        (TAB / f"S1_spot_qc_{s}.json").write_text(json.dumps(rows[-1], default=float))
        log(f"  S1 {s}: {rows[-1]['spots_after_qc']}/{rows[-1]['spots_in_tissue']} spots, "
            f"median UMI {rows[-1]['median_umi']:.0f}")
    pd.DataFrame(rows).to_csv(TAB / "S1_spot_qc.csv", index=False)



# ==============================================================================
# S0b -- SIGNATURES FROM THE scRNA-seq RESULTS (Figs 1 and 3)
# ==============================================================================
def gene_id_map():
    """Ensembl ID -> symbol used by the atlas. GENCODE v44 table if present; otherwise the
    atlas var table (S1d) if it carries Ensembl IDs; otherwise {} (SpaceRanger symbols kept)."""
    f = TAB / "S0_gene_id_map.csv"
    if f.exists():
        m = pd.read_csv(f)
        return dict(zip(m.gene_id.astype(str), m.gene_name.astype(str)))
    if Path(GENCODE_TSV).exists():
        g = pd.read_csv(GENCODE_TSV, sep="\t")
        m = pd.DataFrame({"gene_id": g.gene_id.astype(str).str.split(".").str[0], "gene_name": g.gene_name.astype(str)})
        src = GENCODE_TSV.name
    else:
        A = sc.read_h5ad(ATLAS_H5AD, backed="r")
        var = A.var.copy()
        A.file.close()
        col = next((c for c in var.columns if var[c].astype(str).str.startswith("ENSG").mean() > 0.5), None)
        if col is None and pd.Index(var.index.astype(str)).str.startswith("ENSG").mean() > 0.5:
            name_col = next((c for c in var.columns if c.lower() in ("gene_name", "gene_symbols", "symbol", "feature_name")), None)
            m = pd.DataFrame({"gene_id": var.index.astype(str), "gene_name": var[name_col].astype(str)}) if name_col else None
        elif col is not None:
            m = pd.DataFrame({"gene_id": var[col].astype(str), "gene_name": var.index.astype(str)})
        else:
            m = None
        src = f"{ATLAS_H5AD.name} var" if m is not None else "none (SpaceRanger symbols kept)"
        if m is None:
            m = pd.DataFrame(columns=["gene_id", "gene_name"])
        m["gene_id"] = m.gene_id.str.split(".").str[0]
    m.to_csv(f, index=False)
    log(f"S0b gene-ID map: {len(m):,} IDs from {src}")
    return dict(zip(m.gene_id.astype(str), m.gene_name.astype(str)))


def atlas_lineage_means(chunk=20000):
    """Pseudobulk per lineage over tumour cells of the atlas: summed raw counts -> CPM (h5py,
    chunked, one pass)."""
    f = TAB / "S0_atlas_lineage_pseudobulk_cpm.csv.gz"
    if f.exists():
        return pd.read_csv(f, index_col=0)
    import h5py
    A = sc.read_h5ad(ATLAS_H5AD, backed="r")
    obs, genes = A.obs[["tissue", "lineage"]].copy(), pd.Index(A.var_names.astype(str))
    A.file.close()
    lab = np.where(obs.tissue.astype(str).values == "Tumor", obs.lineage.astype(str).values, None)
    cats = sorted({x for x in lab if x is not None})
    codes = pd.Categorical(lab, categories=cats).codes
    tot = np.zeros((len(cats), len(genes)))
    with h5py.File(ATLAS_H5AD, "r") as h:
        g = h["layers"]["counts"] if "layers" in h and "counts" in h["layers"] else h["X"]
        indptr = g["indptr"][:]
        n = len(indptr) - 1
        for s0 in range(0, n, chunk):
            s1 = min(n, s0 + chunk)
            a0, a1 = indptr[s0], indptr[s1]
            M = sp.csr_matrix((g["data"][a0:a1].astype(np.float64), g["indices"][a0:a1], indptr[s0:s1 + 1] - a0),
                              shape=(s1 - s0, len(genes)))
            c = codes[s0:s1]
            ok = c >= 0
            G = sp.csr_matrix((np.ones(ok.sum()), (c[ok], np.where(ok)[0])), shape=(len(cats), s1 - s0))
            tot += np.asarray((G @ M).todense())
    L = pd.DataFrame(tot.T, index=genes, columns=cats).groupby(level=0).sum()
    L = L / L.sum(axis=0) * 1e6
    L.to_csv(f, compression="gzip")
    log(f"S0b atlas lineage pseudobulk (CPM): {L.shape[1]} lineages ({', '.join(L.columns)})")
    return L


def excluded_gene(g):
    g = str(g)
    return bool(g in CC_GENES or CELL_CYCLE_RE.match(g) or LNC_RE.match(g) or EXCLUDE_GENE_RE.match(g))


def transfer_ratio(genes, lin, target_cols):
    """max expression in the target compartment / max in any other atlas lineage."""
    L = lin.reindex(list(genes))
    tg = [c for c in target_cols if c in L.columns]
    other = [c for c in L.columns if c not in tg]
    return (L[tg].max(axis=1) / L[other].max(axis=1).clip(lower=1e-9)).fillna(0.0)


def build_sap_signature(lin):
    """SAP signature from the Fig. 1 axis weights (SAP-oriented MOFA Factor 4): stromal views,
    |w| >= 25% of max, no cell-cycle / MT / RP / IG genes, one sign per gene, and the atlas
    transferability filter (gene highest in a stromal lineage, so that in a mixed Visium spot it
    reports the stromal state); <= 100 genes per arm."""
    w = pd.read_csv(F_AXIS_W, index_col=[0, 1]).iloc[:, 0]
    w.index.names = ["view", "gene"]
    t = w.reset_index().rename(columns={w.name: "weight"})
    t = t[t.view.isin(SAP_VIEWS) & (t.weight.abs() >= SAP_MIN_REL_WEIGHT * w.abs().max())]
    t = t[~t.gene.map(excluded_gene)]
    nsg = t.groupby("gene").weight.apply(lambda x: np.sign(x).nunique())
    t = t[t.gene.isin(nsg.index[nsg == 1])]
    t = t.loc[t.groupby("gene").weight.apply(lambda x: x.abs().idxmax())]
    top = pd.concat([t[t.weight > 0].nlargest(SAP_MAX_PER_ARM, "weight"),
                     t[t.weight < 0].nsmallest(SAP_MAX_PER_ARM, "weight")])
    ratio = transfer_ratio(top.gene, lin, SAP_VIEWS).values
    keep = ratio >= TRANSFER_MIN_RATIO
    TRANSFER_LOG.extend(dict(signature="SAP", gene=g, ratio=r, kept=bool(k))
                        for g, r, k in zip(top.gene, ratio, keep))
    sig = top[keep]
    log(f"S0b SAP signature: top {len(top)} weights -> {len(sig)} after the stromal transferability filter "
        f"({(sig.weight > 0).sum()} activated / {(sig.weight < 0).sum()} quiescent)")
    return sig.set_index("gene")["weight"]


TRANSFER_LOG = []


def myeloid_filter(name, genes, lin):
    """Keep genes highest in the myeloid lineage of the atlas (same rule as the SAP filter)."""
    genes = list(dict.fromkeys(genes))
    ratio = transfer_ratio(genes, lin, ["Myeloid"])
    keep = {g: bool(ratio[g] >= TRANSFER_MIN_RATIO) for g in genes}
    TRANSFER_LOG.extend(dict(signature=name, gene=g, ratio=float(ratio[g]), kept=keep[g]) for g in genes)
    return [g for g in genes if keep[g]]


def build_signatures():
    """All gene sets used downstream, written in one table (signature, gene, weight)."""
    if F_SIGS.exists() and "select" not in FORCE:
        return
    for f in [ATLAS_H5AD, F_AXIS_W, F_DRIVERS, F_REGULONS]:
        if not Path(f).exists():
            sys.exit(f"Missing input: {f}")
    lin = atlas_lineage_means()
    sap = build_sap_signature(lin)
    sap_g = set(sap.index)
    rows = [("SAP", g, float(v)) for g, v in sap.items()]
    # label-free myeloid states: CellRank lineage drivers (Fig. 3), SAP and cell-cycle genes removed
    ld = pd.read_csv(F_DRIVERS)
    piv = ld.pivot_table(index="gene", columns="lineage", values="corr")
    qv = ld.pivot_table(index="gene", columns="lineage", values="qval")
    for lin_ in piv.columns:
        spec = piv[lin_] - piv.drop(columns=lin_).max(axis=1) if piv.shape[1] > 1 else piv[lin_]
        ok = (qv[lin_] < 0.01) & (piv[lin_] > 0.20) & (spec > 0.10)
        g = spec[ok].sort_values(ascending=False)
        g = [x for x in g.index if not excluded_gene(x) and x not in sap_g][:50]
        g2 = myeloid_filter(f"Fate_{lin_}", g, lin)
        rows += [(f"Fate_{lin_}", x, 1.0) for x in g2]
        log(f"S0b Fate_{lin_}: {len(g)} driver genes -> {len(g2)} myeloid-transferable")
    rows += [("Mono_classical", g, 1.0) for g in myeloid_filter("Mono_classical", MONO_PANEL, lin)]
    rows += [("Myeloid_abundance", g, 1.0) for g in MYELOID_PANEL]
    # regulons of Fig. 3J (consensus SCENIC targets), SAP genes removed
    reg = pd.read_csv(F_REGULONS).set_index("regulon")
    for r in REGULON_DIR:
        if r in reg.index:
            g = [x for x in str(reg.loc[r, "genes"]).split(";") if x and x not in sap_g and not excluded_gene(x)]
            rows += [(f"Regulon_{r}", x, 1.0) for x in myeloid_filter(f"Regulon_{r}", g, lin)]
    sig = pd.DataFrame(rows, columns=["signature", "gene", "weight"]).drop_duplicates(["signature", "gene"])
    sig.to_csv(F_SIGS, index=False)
    pd.DataFrame(TRANSFER_LOG).to_csv(TAB / "S0_signature_transferability.csv", index=False)
    log(f"S0b signatures: {sig.signature.value_counts().to_dict()}")
    # Hallmark sets + TAM programmes (Fig. 1I/J leading edges)
    hall = None
    if Path(HALLMARK_JSON_CANDIDATE).exists():
        hall = json.loads(Path(HALLMARK_JSON_CANDIDATE).read_text())
        log(f"S0b Hallmark: {HALLMARK_JSON_CANDIDATE}")
    else:
        try:
            import gseapy
            hall = gseapy.get_library("MSigDB_Hallmark_2020", organism="Human")
            log("S0b Hallmark: MSigDB_Hallmark_2020 downloaded with gseapy (Enrichr)")
        except Exception as e:
            log(f"S0b Hallmark library not available ({e}); Fig. 1I/J leading-edge genes used instead")
    pw_rows, le = [], {}
    for f, tag in [(F_GSEA_MO, "MoTAM"), (F_GSEA_RES, "ResTAM")]:
        if not Path(f).exists():
            continue
        g = pd.read_csv(f)
        g = g[pd.to_numeric(g["FDR q-val"], errors="coerce") < 0.05]
        for _, r in g.iterrows():
            genes = [x for x in str(r.Lead_genes).split(";") if x]
            key = re.sub(r"[^A-Za-z0-9]+", "_", str(r.Term)).strip("_")
            pw_rows += [(f"PWm_{tag}_{key}", x, 1.0) for x in genes]
            le.setdefault(str(r.Term), set()).update(genes)
    pd.DataFrame(pw_rows, columns=["signature", "gene", "weight"]).to_csv(F_PATHWAY, index=False)
    if hall is None:
        hall = {k: sorted(v) for k, v in le.items()}
    HALLMARK_JSON.write_text(json.dumps({k: list(v) for k, v in hall.items()}))
    log(f"S0b programmes: {len(set(r[0] for r in pw_rows))} TAM programme sets; {len(hall)} Hallmark sets")


def _cpdb_partner_genes():
    """CellPhoneDB partner name -> gene list, from YOUR cpdb_db folder (complex subunits resolved)."""
    genes, cplx = {}, {}
    gi = CPDB_DB / "gene_input.csv"
    ci = CPDB_DB / "complex_input.csv"
    if gi.exists():
        g = pd.read_csv(gi)
        sym = "hgnc_symbol" if "hgnc_symbol" in g else "gene_name"
        genes = dict(zip(g.uniprot.astype(str), g[sym].astype(str)))
    if ci.exists():
        c = pd.read_csv(ci)
        ucols = [x for x in c.columns if x.startswith("uniprot_")]
        for r in c.itertuples():
            subs = [genes.get(str(getattr(r, u)), None) for u in ucols if isinstance(getattr(r, u), str)]
            cplx[r.complex_name] = [s for s in subs if s]
    return genes, cplx


def build_channels():
    """Stroma<->myeloid channels to test in space, taken from YOUR Fig. 4A tumour-level table:
    every interacting pair with q < 0.05 (its strongest sender|receiver row), plus the named
    mechanistic pairs (best row). Sender/receiver from the CellPhoneDB row; monocyte/macrophage
    states pooled into MoMac (Visium cannot resolve them); genes from the CellPhoneDB database."""
    f = TAB / "S0_channels.csv"
    if f.exists() and "select" not in FORCE:
        return pd.read_csv(f)
    if not (F_FIG4A.exists() and F_FIG4A_LONG.exists()):
        log("S0c Fig. 4A tables not found -> transcribed channel list used")
        t = pd.DataFrame(CHANNELS, columns=["ligand", "receptor", "source", "target", "sign", "tier"])
        t.to_csv(f, index=False)
        return t
    tab = pd.read_csv(F_FIG4A)
    lg = pd.read_csv(F_FIG4A_LONG, usecols=lambda c: c in {"id_cp_interaction", "partner_a", "partner_b",
                                                             "gene_a", "gene_b"}).drop_duplicates("id_cp_interaction")
    lg = lg.set_index("id_cp_interaction")
    u2g, cplx = _cpdb_partner_genes()

    def partner_genes(partner, gene):
        if isinstance(gene, str) and gene and gene != "nan":
            return [gene]
        name = str(partner).split(":", 1)[-1]
        if str(partner).startswith("simple:"):
            return [u2g.get(name)] if u2g.get(name) else []
        return cplx.get(name, [])

    def pool(ct):
        return "MoMac" if ct in MOMAC else ct

    sig = tab[tab.q < FIG4A_Q].assign(absr=lambda d: d.rho_SAP.abs())
    best = sig.sort_values("absr", ascending=False).drop_duplicates("interacting_pair").assign(tier="Fig4A")
    mech = tab[tab.interacting_pair.isin(MECH_PAIRS) & ~tab.interacting_pair.isin(best.interacting_pair)]
    mech = mech.sort_values("p").drop_duplicates("interacting_pair").assign(tier="mechanistic")
    rows, miss = [], []
    for r in pd.concat([best, mech]).itertuples():
        if r.id_cp_interaction not in lg.index:
            miss.append(r.interacting_pair)
            continue
        x = lg.loc[r.id_cp_interaction]
        ga, gb = partner_genes(x.partner_a, x.gene_a), partner_genes(x.partner_b, x.gene_b)
        if not ga or not gb:
            miss.append(r.interacting_pair)
            continue
        src, tgt = r.row.split("|")
        rows.append(dict(ligand="_".join(ga), receptor="_".join(gb), source=pool(src), target=pool(tgt),
                         sign=int(np.sign(r.rho_SAP)), tier=r.tier, interacting_pair=r.interacting_pair,
                         fig4a_row=r.row, fig4a_rho=r.rho_SAP, fig4a_q=r.q))
    t = pd.DataFrame(rows)
    if miss:
        log(f"S0c WARNING: genes not resolved for {miss} (check {CPDB_DB}); these channels are skipped")
    t.to_csv(f, index=False)
    log(f"S0c channels from Fig. 4A: {int((t.tier == 'Fig4A').sum())} with q < {FIG4A_Q} + "
        f"{int((t.tier == 'mechanistic').sum())} mechanistic\n" +
        t[["interacting_pair", "ligand", "receptor", "source", "target", "sign", "tier"]].to_string(index=False))
    return t


# ==============================================================================
# S1b -- OUTCOME-BLIND SELECTION OF THE N_SELECTED BEST PRIMARY SECTIONS
# ==============================================================================
def _marker_score(a, genes):
    g = [x for x in genes if x in a.var_names]
    X = a[:, g].layers["cp10k"]
    X = X.toarray() if sp.issparse(X) else np.asarray(X)
    return np.log1p(X).mean(axis=1), len(g)


def select_sections(sap):
    """Ranks the primary sections on four criteria that do NOT use any SAP-myeloid association
    (so the choice cannot bias the hypothesis tests):
      1. spots after QC                      (statistical units)
      2. median genes per spot              (data quality / gene detection)
      3. spots with stromal AND myeloid marker signal (the co-localisation the tests need)
      4. spread (SD) of the SAP score among stromal-marker spots (the exposure must vary)
    Each criterion is ranked across sections; the mean rank decides. SELECT_OVERRIDE wins."""
    f = TAB / "S1_selected_sections.json"
    if f.exists() and "select" not in FORCE:
        sel = json.loads(f.read_text())
        log(f"S1b selection (checkpoint): {sel['selected']}")
        return sel
    qc = pd.read_csv(TAB / "S1_spot_qc.csv").set_index("sample")
    w = sap["weight"]
    rows, st_scores, my_scores = [], {}, {}
    for s in PRIMARY_ALL:
        a = lognorm(read_section(s))
        st_scores[s], n_st = _marker_score(a, STROMA_MARKERS)
        my_scores[s], n_my = _marker_score(a, MYELOID_MARKERS)
        g = w.index.intersection(a.var_names)
        X = a[:, g].X
        X = X.toarray() if sp.issparse(X) else np.asarray(X)
        rows.append(dict(sample=s, sap_raw=X @ w[g].values / w[g].abs().sum(), n_markers_stroma=n_st,
                         n_markers_myeloid=n_my, sap_coverage=len(g) / len(w)))
    thr_st = np.median(np.concatenate(list(st_scores.values())))      # pooled thresholds (same for all)
    thr_my = np.median(np.concatenate(list(my_scores.values())))
    out = []
    for r in rows:
        s = r["sample"]
        st, my = st_scores[s] > thr_st, my_scores[s] > thr_my
        out.append(dict(sample=s, spots_after_qc=int(qc.loc[s, "spots_after_qc"]),
                        median_genes=float(qc.loc[s, "median_genes"]), median_umi=float(qc.loc[s, "median_umi"]),
                        n_stroma_myeloid_spots=int((st & my).sum()), frac_stroma_myeloid=float((st & my).mean()),
                        sap_sd_in_stroma=float(np.std(r["sap_raw"][st])) if st.sum() > 10 else np.nan,
                        sap_coverage=r["sap_coverage"]))
    t = pd.DataFrame(out).set_index("sample")
    crit = ["spots_after_qc", "median_genes", "n_stroma_myeloid_spots", "sap_sd_in_stroma"]
    t["eligible"] = t.n_stroma_myeloid_spots >= MIN_COOCCUR_SPOTS     # the tests need stroma + myeloid together
    t["mean_rank"] = t[crit].rank(ascending=False).mean(axis=1)
    t = t.sort_values(["eligible", "mean_rank", "n_stroma_myeloid_spots"], ascending=[False, True, False])
    chosen = list(SELECT_OVERRIDE) if SELECT_OVERRIDE else list(t.index[:N_SELECTED])
    t["selected"] = t.index.isin(chosen)
    t.to_csv(TAB / "S1_section_selection.csv")
    sel = dict(selected=chosen, rule=("manual override" if SELECT_OVERRIDE else
                                      f">= {MIN_COOCCUR_SPOTS} stroma+myeloid spots, then top {N_SELECTED} by mean "
                                      f"rank of {crit} (outcome-blind)"),
               table=t.reset_index().to_dict("records"))
    f.write_text(json.dumps(sel, indent=2, default=float))
    log("S1b section ranking:\n" + t.round(3).to_string())
    log(f"S1b SELECTED: {chosen}")
    return sel


def main():
    force_from_argv({"select"})                     # --force: signatures, channels, selection (QC kept)
    if "--force-qc" in sys.argv:                    # --force-qc: also redo the spot QC
        FORCE.update({"qc", "select"})
    if step_done("step01") and not FORCE & {"qc", "select"}:
        log(f"step01 already done -> selected {selected_sections()} (use --force to redo)")
        return
    log("=" * 70)
    log("STEP 01: download + QC + section selection")
    build_signatures()
    global CHANNELS
    ch = build_channels()
    CHANNELS = [(r.ligand, r.receptor, r.source, r.target, int(r.sign), r.tier) for r in ch.itertuples()]
    manifest = download_gse251950()
    load_and_qc(manifest)
    sap, _ = load_frozen_signatures()
    sel = select_sections(sap)
    preregister(sel)
    mark_done("step01", selected=sel["selected"])


if __name__ == "__main__":
    main()
