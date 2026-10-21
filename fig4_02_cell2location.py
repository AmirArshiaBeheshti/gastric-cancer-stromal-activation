#!/usr/bin/env python
# coding: utf-8
"""
 fig4_02_cell2location.py  (step 2 of 7)  -- GPU strongly recommended
 S2  cell2location reference signatures (NB regression) from the scRNA-seq atlas:
     9 non-myeloid lineages + 12 myeloid states (Fig. 1); trained model is checkpointed
 S3  joint spatial mapping of the 3 selected sections on a SAP-BLIND gene panel (SAP
     genes removed, so myeloid abundances and SAP are computed from disjoint genes);
     trained model is checkpointed (a crash after training does not retrain)
     GPU memory: full-batch training is used only if the data fit on the GPU; otherwise mini-batch
     training (2500 -> 1250 -> 600 spots) with the same number of gradient steps, and CPU as the
     last resort. Posterior sampling is done in small batches.
 RUN: python fig4_02_cell2location.py      (--force to redo)
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


# ==============================================================================
# STEP-SPECIFIC CODE
# ==============================================================================
def _train(mod, **kw):
    """scvi-tools API changed (use_gpu -> accelerator); try both."""
    import torch
    gpu = torch.cuda.is_available()
    for extra in ({"accelerator": "gpu" if gpu else "cpu"}, {"use_gpu": gpu}, {}):
        try:
            return mod.train(**kw, **extra)
        except TypeError:
            continue
    raise RuntimeError("cell2location training failed with all accelerator conventions")


def _history_plot(mod, name, skip):
    try:
        fig, ax = plt.subplots(figsize=(3.2, 2.2))
        h = mod.history["elbo_train"]
        ax.plot(h.index[skip:], h.values[skip:].ravel(), color=C_DARK, lw=0.7)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("-ELBO (train)")
        ax.set_title(name, fontsize=7)
        savefig(fig, f"FigS_{name}_ELBO", FIG_SUPP)
        h.to_csv(TAB / f"S_{name}_elbo.csv")
    except Exception as e:
        log(f"  ELBO plot skipped ({e})")


def build_reference():
    f = TAB / "S2_reference_signatures.csv"
    if f.exists() and "ref" not in FORCE:
        return pd.read_csv(f, index_col=0)
    import cell2location
    from cell2location.models import RegressionModel
    from cell2location.utils.filtering import filter_genes
    A = sc.read_h5ad(ATLAS_H5AD, backed="r")
    obs = A.obs[["sample_id", "dataset", "tissue", "lineage"]].copy()
    obs["lineage"] = obs["lineage"].astype(str)
    M = sc.read_h5ad(MYELOID_H5AD, backed="r")
    col = next((c for c in ["myeloid_state", "myeloid_state_m3", "state", "cell_state"] if c in M.obs), None)
    if col is None:
        sys.exit(f"{MYELOID_H5AD.name}: no myeloid-state column in obs ({list(M.obs.columns)[:20]})")
    state = M.obs[col].astype(str)
    log(f"  myeloid states from {MYELOID_H5AD.name} [{col}]: {sorted(state.unique())}")
    M.file.close()
    lab = obs["lineage"].copy()
    is_my = lab == "Myeloid"
    lab[is_my] = state.reindex(lab.index[is_my]).values      # removed contaminants -> NaN
    keep = (obs.tissue.astype(str) == "Tumor") & lab.isin(NONMYELOID_REF + MYELOID_STATES)
    obs["ref_label"] = lab
    obs = obs[keep]
    rng = np.random.default_rng(SEED)
    idx = []
    for lbl, d in obs.groupby("ref_label"):
        take = []
        for _, ds in d.groupby("sample_id", observed=True):
            ii = ds.index.values
            take.extend(rng.choice(ii, min(len(ii), REF_CAP_PER_SAMPLE), replace=False))
        if len(take) > REF_CAP_PER_LABEL:
            take = rng.choice(np.array(take), REF_CAP_PER_LABEL, replace=False)
        idx.extend(take)
    pos = np.sort(A.obs_names.get_indexer(pd.Index(idx)))
    sub = A[pos].to_memory()
    A.file.close()
    ref = ad.AnnData(X=sp.csr_matrix(sub.layers["counts"]), obs=sub.obs[["sample_id", "dataset"]].copy(),
                     var=pd.DataFrame(index=sub.var_names))
    ref.obs["ref_label"] = obs.loc[ref.obs_names, "ref_label"].values
    for c in ["sample_id", "dataset", "ref_label"]:
        ref.obs[c] = ref.obs[c].astype(str).astype("category")
    ref = ref[:, ~ref.var_names.str.match(EXCLUDE_GENE_RE)].copy()
    ref.obs.ref_label.value_counts().to_csv(TAB / "S2_reference_cells_per_label.csv")
    selected = filter_genes(ref, cell_count_cutoff=5, cell_percentage_cutoff2=0.03, nonz_mean_cutoff=1.12)
    plt.close("all")
    ref = ref[:, selected].copy()
    log(f"S2 reference: {ref.n_obs:,} cells, {ref.n_vars:,} genes, {ref.obs.ref_label.nunique()} labels")
    RegressionModel.setup_anndata(adata=ref, batch_key="sample_id", labels_key="ref_label",
                                  categorical_covariate_keys=["dataset"])
    mdir = MOD / "c2l_reference"
    mod = None
    if mdir.exists() and "ref" not in FORCE:                       # checkpoint: trained model on disk
        try:
            mod = RegressionModel.load(str(mdir), ref)
            log("  S2 reference model re-loaded from checkpoint (training skipped)")
        except Exception as e:
            log(f"  S2 checkpoint not usable ({e}); retraining")
            mod = None
    if mod is None:
        mod = RegressionModel(ref)
        _train(mod, max_epochs=REG_EPOCHS, batch_size=2500, train_size=1, lr=0.002)
        mod.save(str(mdir), overwrite=True)
        _history_plot(mod, "c2l_reference", 20)
    ref = mod.export_posterior(ref, sample_kwargs={"num_samples": 1000, "batch_size": 2500})
    fac = ref.uns["mod"]["factor_names"]
    if "means_per_cluster_mu_fg" in ref.varm.keys():
        inf = ref.varm["means_per_cluster_mu_fg"][[f"means_per_cluster_mu_fg_{i}" for i in fac]].copy()
    else:
        inf = ref.var[[f"means_per_cluster_mu_fg_{i}" for i in fac]].copy()
    inf.columns = fac
    inf.to_csv(f)
    return inf




def _free_gpu():
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _is_oom(e):
    return "out of memory" in str(e).lower() or type(e).__name__ == "OutOfMemoryError"


def gpu_free_gb():
    try:
        import torch
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            return free / 1e9, total / 1e9
    except Exception:
        pass
    return 0.0, 0.0


def fit_cell2location(vis, cell_state_df, n_cells):
    """Train the mapping model within the available GPU memory. Plans tried in order:
    full batch (only if the estimated need fits), mini-batches of 2500 / 1250 / 600 spots with the
    same total number of gradient steps as full-batch training, and finally CPU (full batch)."""
    import cell2location
    import torch
    n_obs, n_genes = vis.n_obs, vis.n_vars
    free, total = gpu_free_gb()
    need_full = n_obs * n_genes * 4 * 14 / 1e9          # ~14 float32 spot x gene tensors (empirical)
    plans = []
    if torch.cuda.is_available():
        if need_full < 0.8 * free:
            plans.append(dict(device="gpu", batch_size=None, epochs=C2L_EPOCHS))
        for bs in C2L_BATCH_SIZES:
            if bs < n_obs:
                steps_per_epoch = int(np.ceil(n_obs / bs))
                plans.append(dict(device="gpu", batch_size=bs,
                                  epochs=max(C2L_MIN_EPOCHS, int(np.ceil(C2L_EPOCHS / steps_per_epoch)))))
    plans.append(dict(device="cpu", batch_size=None, epochs=C2L_EPOCHS))
    log(f"  GPU: {free:.1f} GB free of {total:.1f} GB; full-batch estimate {need_full:.1f} GB -> "
        f"plans: {[(p['device'], p['batch_size'], p['epochs']) for p in plans]}")
    for plan in plans:
        _free_gpu()
        mod = cell2location.models.Cell2location(vis, cell_state_df=cell_state_df,
                                                 N_cells_per_location=n_cells, detection_alpha=DETECTION_ALPHA)
        log(f"  S3 training: {plan['device']}, batch_size={plan['batch_size']}, epochs={plan['epochs']}")
        try:
            kw = dict(max_epochs=plan["epochs"], batch_size=plan["batch_size"], train_size=1)
            if plan["device"] == "cpu":
                try:
                    mod.train(**kw, accelerator="cpu")
                except TypeError:
                    mod.train(**kw, use_gpu=False)
            else:
                _train(mod, **kw)
            return mod, dict(plan, n_spots=n_obs, n_genes=n_genes)
        except BaseException as e:                      # pyro wraps the CUDA error in other types
            if not _is_oom(e):
                raise
            log(f"  GPU out of memory with batch_size={plan['batch_size']} -> next plan")
            del mod
            _free_gpu()
    raise RuntimeError("cell2location could not be trained with any plan")


def export_posterior_safe(mod, vis):
    """Posterior sampling in small batches; halves the batch on GPU out-of-memory."""
    bs = POSTERIOR_BATCH
    while True:
        try:
            _free_gpu()
            return mod.export_posterior(vis, sample_kwargs={"num_samples": 1000, "batch_size": min(bs, vis.n_obs)})
        except BaseException as e:
            if not _is_oom(e) or bs <= 50:
                raise
            bs //= 2
            log(f"  posterior sampling out of GPU memory -> batch_size {bs}")


def map_cell2location(manifest, inf, sap_genes, n_cells=N_CELLS_PER_SPOT, tag="main"):
    f_means = TAB / f"S3_c2l_abundance_means_{tag}.csv"
    if f_means.exists() and "c2l" not in FORCE:
        return pd.read_csv(f_means, index_col=0), pd.read_csv(TAB / f"S3_c2l_abundance_q05_{tag}.csv", index_col=0)
    import cell2location
    secs = [read_section(s) for s in manifest]
    common = secs[0].var_names
    for a in secs[1:]:
        common = common.intersection(a.var_names)
    genes = inf.index.intersection(common)
    genes = genes[~genes.isin(sap_genes)]                        # SAP-blind deconvolution
    vis = ad.concat([a[:, genes] for a in secs], join="inner", merge="first")
    vis.obs["sample"] = vis.obs["sample"].astype("category")
    vis.X = sp.csr_matrix(vis.X)
    pd.Series(genes).to_csv(TAB / f"S3_c2l_gene_panel_{tag}.csv", index=False)
    log(f"S3 cell2location[{tag}]: {vis.n_obs:,} spots x {len(genes):,} genes "
        f"({len(sap_genes.intersection(inf.index))} SAP genes excluded), N_cells/spot = {n_cells}")
    cell2location.models.Cell2location.setup_anndata(adata=vis, batch_key="sample")
    mdir = MOD / f"c2l_mapping_{tag}"
    mod = None
    if mdir.exists() and "c2l" not in FORCE:                       # checkpoint: trained model on disk
        try:
            mod = cell2location.models.Cell2location.load(str(mdir), vis)
            log(f"  S3 mapping model [{tag}] re-loaded from checkpoint (training skipped)")
        except Exception as e:
            log(f"  S3 checkpoint not usable ({e}); retraining")
            mod = None
    if mod is None:
        mod, plan = fit_cell2location(vis, inf.loc[genes], n_cells)
        mod.save(str(mdir), overwrite=True)
        (MOD / f"c2l_mapping_{tag}_training.json").write_text(json.dumps(plan, indent=2))
        _history_plot(mod, f"c2l_mapping_{tag}", min(1000, plan["epochs"] // 10))
    vis = export_posterior_safe(mod, vis)

    def grab(kind):
        df = vis.obsm[f"{kind}_cell_abundance_w_sf"].copy()
        df.columns = [re.sub(r"^.*cell_abundance_w_sf_", "", c) for c in df.columns]
        return df
    means, q05 = grab("means"), grab("q05")
    means.to_csv(f_means)
    q05.to_csv(TAB / f"S3_c2l_abundance_q05_{tag}.csv")
    return means, q05



def main():
    force_from_argv({"ref", "c2l"})
    require("step01")
    if step_done("step02") and not FORCE & {"ref", "c2l"}:
        log("step02 already done (use --force to redo)")
        return
    log("=" * 70)
    log(f"STEP 02: cell2location on {PRIMARY}")
    for f in [ATLAS_H5AD, MYELOID_H5AD, F_SIGS]:      # F_SIGS is written by step 01
        if not Path(f).exists():
            sys.exit(f"Missing input: {f}")
    manifest = load_manifest()
    sap, _ = load_frozen_signatures()
    sap_genes = pd.Index(sorted(set(sap.index)))
    inf = build_reference()
    means, q05 = map_cell2location(manifest, inf, sap_genes)
    for nc in SENSITIVITY_N_CELLS:
        map_cell2location(manifest, inf, sap_genes, n_cells=nc, tag=f"N{nc}")
    mark_done("step02", sections=list(manifest), n_spots=len(means), n_types=means.shape[1])


if __name__ == "__main__":
    main()
