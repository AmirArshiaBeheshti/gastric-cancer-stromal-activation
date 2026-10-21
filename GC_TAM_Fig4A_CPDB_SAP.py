#!/usr/bin/env python
# coding: utf-8
"""
================================================================================
 GC_TAM_Fig4A_CPDB_SAP.py -- Figure 4A: stroma <-> TAM communication along SAP

 QUESTION
   SAP (Figure 1) co-varies with myeloid-state composition across tumours.
   Which ligand-receptor (L-R) channels connect stroma and TAM states, and which
   of them scale with SAP across tumours?

 DESIGN (dot-plot layout: sender|receiver rows split by group, L-R columns,
 dot colour/size = scaled mean, red ring = significant)
   Cells     tumour cells of S1d_cnv_v3.h5ad (whole atlas, raw counts);
             stroma = Fibroblast, Pericyte_SMC, Endothelial (lineage);
             myeloid states from S1e_myeloid_final_v3_m3_amb20_F4.h5ad (obs_names
             matched): the four SAP-associated states of Fig. 1e.
   Groups    SAP-high vs SAP-low = top vs bottom SAP tertile, defined WITHIN each
             dataset (GSE150290 3'v2 / GSE183904 5'), so chemistry is balanced
             between groups; middle tertile not displayed.
   Pooled    CellPhoneDB v5 statistical method (1,000 permutations) run separately
             in each group; to stop large tumours dominating the pool, cells are
             capped at 200 per tumour x cell type. Gives the displayed means and
             significance (red ring, P < 0.05).
   Tumour    Because pooled cells from few patients are pseudoreplicates, every
   level     displayed L-R x sender|receiver channel is also scored PER TUMOUR
             (CellPhoneDB definition: mean of ligand and receptor means, complexes =
             min over subunits, 0 if < 10% of cells express), and tested against SAP
             across ALL tumours with >= 10 sender and receiver cells (Spearman,
             partial on dataset; BH-FDR over all candidate channels). This, not the
             pooled P, is the inferential statistic; it is shown in the top strip.
   Columns   candidate = significant in >= 1 displayed row of either group (non-
             directional); top N by tumour-level FDR; ordered by association with SAP.

 INPUT   GC_Fig1_outputs/h5ad/S1d_cnv_v3.h5ad, S1e_myeloid_final_v3_m3_amb20_F4.h5ad
         GC_Fig1_outputs/tables/S2_MCP_scores_v3_amb20_F4.csv
 OUTPUT  GC_Fig4_outputs/figures/Fig4A_CPDB_stroma_TAM_SAP.{pdf,png,tiff}
         GC_Fig4_outputs/tables/ (CPDB outputs per group, tumour-level channel table,
         selected interactions, group membership)
 RUN     python GC_TAM_Fig4A_CPDB_SAP.py          (CellPhoneDB >= 5.0)
================================================================================
"""
import os
import re
import sys
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import stats

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
BASE = Path(os.environ.get("GC_BASE", "/home/arshia/payanname"))
ap = argparse.ArgumentParser()
ap.add_argument("--atlas", default=str(BASE / "GC_Fig1_outputs/h5ad/S1d_cnv_v3.h5ad"))
ap.add_argument("--myeloid", default=str(BASE / "GC_Fig1_outputs/h5ad/S1e_myeloid_final_v3_m3_amb20_F4.h5ad"))
ap.add_argument("--mcp", default=str(BASE / "GC_Fig1_outputs/tables/S2_MCP_scores_v3_amb20_F4.csv"))
ap.add_argument("--weights", default=str(BASE / "GC_Fig1_outputs/tables/S2_axis_weights_amb20_F4.csv"))
ap.add_argument("--out", default=str(BASE / "GC_Fig4_outputs"))
ap.add_argument("--cpdb_db", default=None, help="folder with cellphonedb.zip (downloaded if absent)")
ap.add_argument("--iterations", type=int, default=1000)
ap.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 2) - 1))
ap.add_argument("--n_lr", type=int, default=60, help="number of L-R columns displayed")
ap.add_argument("--max_cells", type=int, default=200, help="cap per tumour x cell type in pooled CPDB")
ap.add_argument("--rerun_cpdb", action="store_true", help="ignore cached CPDB results")
ARGS, _ = ap.parse_known_args()

OUT = Path(ARGS.out)
FIG, TAB, WORK = OUT / "figures", OUT / "tables", OUT / "cpdb_work"
DB_DIR = Path(ARGS.cpdb_db) if ARGS.cpdb_db else OUT / "cpdb_db"
for d in (FIG, TAB, WORK, DB_DIR):
    d.mkdir(parents=True, exist_ok=True)

SEED = 0
CPDB_VERSION = "v5.0.0"
CPDB_THRESHOLD = 0.10            # min fraction of expressing cells (CellPhoneDB default)
CPDB_PVALUE = 0.05
STROMA = ["Fibroblast", "Pericyte_SMC", "Endothelial"]
MYELOID = ["Mono_classical", "TAM_Inflammatory", "TAM_SPP1", "TAM_C1QC"]   # SAP-linked states (Fig. 1e)
DIRECTIONS = ("stroma_to_myeloid", "myeloid_to_stroma")
MIN_CELLS_TUMOUR = 10            # per sender and receiver for the tumour-level score
MIN_TUMOURS = 15
FDR_STAR = (0.05, 0.01)
SAP_WEIGHT_FRAC = 0.25          # SAP genes: |Factor4 weight| >= 25% of max in a stromal view, capped at the
SAP_MAX_PER_DIR = 100           # top 100 per arm
EXCLUDE_SMALL_MOLECULE = True    # drop metabolite-mediated pairs (e.g. "GABA_byGAD1_and_SLC6A1_..."):
                                 # inferred from enzyme expression, not protein ligand-receptor binding

LABEL = {"Fibroblast": "Fibroblast", "Pericyte_SMC": "Pericyte/SMC", "Endothelial": "Endothelial",
         "Mono_classical": "Mono-classical", "TAM_Inflammatory": "TAM-Inflam", "TAM_SPP1": "TAM-SPP1",
         "TAM_C1QC": "TAM-C1QC"}

# ==============================================================================
# 2. STYLE (same family as Figs 1-3)
# ==============================================================================
MM = 1 / 25.4
W2 = 183 * MM


def _font():
    have = {f.name for f in font_manager.fontManager.ttflist}
    for f in ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"]:
        if f in have:
            return f
    return "sans-serif"


mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": [_font()],
    "font.size": 6, "axes.titlesize": 7, "axes.labelsize": 6,
    "xtick.labelsize": 5.4, "ytick.labelsize": 5.8, "legend.fontsize": 5.6,
    "axes.linewidth": 0.5, "xtick.major.width": 0.4, "ytick.major.width": 0.4,
    "xtick.major.size": 0, "ytick.major.size": 0, "xtick.major.pad": 1.5, "ytick.major.pad": 1.5,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "savefig.dpi": 600, "figure.dpi": 150,
})
C_SIG = "#D7191C"
C_GRID = "#E6E6E6"
C_HIGH, C_LOW, C_DARK = "#B2182B", "#2166AC", "#262626"


SAPG = pd.Series(dtype=float)


def log(msg):
    print(msg, flush=True)


def savefig(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(FIG / f"{name}.tiff", bbox_inches="tight", pad_inches=0.03, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    log(f"  saved figures/{name}.[pdf|png|tiff]")


def bh(p):
    p = np.asarray(p, float)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if ok.any():
        pv = p[ok]
        o = np.argsort(pv)
        r = np.minimum.accumulate((pv[o] * len(pv) / np.arange(1, len(pv) + 1))[::-1])[::-1]
        out = np.empty_like(pv)
        out[o] = np.minimum(r, 1)
        q[ok] = out
    return q


# ==============================================================================
# 3. DATA
# ==============================================================================
def sap_groups():
    m = pd.read_csv(ARGS.mcp, index_col=0)
    m.index = m.index.astype(str)
    g = pd.Series("SAP-mid", index=m.index)
    for ds, d in m.groupby("dataset"):          # tertiles WITHIN dataset (chemistry-balanced groups)
        lo, hi = d.MCP.quantile([1 / 3, 2 / 3])
        g[d.index[d.MCP <= lo]] = "SAP-low"
        g[d.index[d.MCP >= hi]] = "SAP-high"
    m["SAP_group"] = g
    log("  SAP groups (tertiles within dataset): " +
        ", ".join(f"{k} {v}" for k, v in m.SAP_group.value_counts().items()))
    log("  " + m.groupby(["SAP_group", "dataset"]).size().unstack(fill_value=0).to_string().replace("\n", "\n  "))
    m[["MCP", "dataset", "SAP_group"]].to_csv(TAB / "Fig4A_tumour_groups.csv")
    return m


def load_cells(mcp):
    import anndata as ad
    log("  reading atlas obs (backed)")
    a = ad.read_h5ad(ARGS.atlas, backed="r")
    obs = a.obs[["sample_id", "dataset", "tissue", "lineage"]].copy()
    obs["sample_id"] = obs.sample_id.astype(str)
    m = ad.read_h5ad(ARGS.myeloid, backed="r")
    state = m.obs["myeloid_state"].astype(str)
    m.file.close()
    obs["cell_type"] = np.where(obs.lineage.astype(str).isin(STROMA), obs.lineage.astype(str), None)
    hit = obs.index.intersection(state.index)
    obs.loc[hit, "cell_type"] = state[hit].values
    log(f"  myeloid states mapped onto atlas by obs_names: {len(hit):,}/{len(state):,} cells")
    keep = (obs.tissue.astype(str) == "Tumor") & obs.sample_id.isin(mcp.index) & obs.cell_type.isin(STROMA + MYELOID)
    obs = obs[keep]
    log(f"  cells kept: {len(obs):,} in {obs.sample_id.nunique()} tumours")
    log("  " + obs.cell_type.value_counts().to_string().replace("\n", "\n  "))
    idx = np.where(a.obs_names.isin(obs.index))[0]
    sub = a[idx].to_memory()
    a.file.close()
    X = sub.layers["counts"] if "counts" in sub.layers else sub.X
    X = sp.csr_matrix(X, dtype=np.float32)
    lib = np.asarray(X.sum(axis=1)).ravel()
    X = sp.diags(1e4 / np.maximum(lib, 1)) @ X
    X.data = np.log1p(X.data)                  # log-normalised (CellPhoneDB recommendation)
    obs = obs.loc[sub.obs_names]
    obs["SAP_group"] = mcp.loc[obs.sample_id, "SAP_group"].values
    return X.tocsr(), obs, pd.Index(sub.var_names.astype(str))


def sap_genes():
    """Stromal genes that DEFINE SAP (oriented Factor4 weights): +1 activated arm, -1 quiescent arm."""
    if not Path(ARGS.weights).exists():
        log(f"  axis weights not found ({ARGS.weights}) -> no pre-specified SAP family")
        return pd.Series(dtype=float)
    w = pd.read_csv(ARGS.weights, index_col=[0, 1]).iloc[:, 0]
    w.index.names = ["view", "gene"]
    w = w[w.index.get_level_values("view").isin(STROMA)]
    w = w[w.abs() >= SAP_WEIGHT_FRAC * w.abs().max()]
    g = w.groupby(level="gene").apply(lambda x: x.loc[x.abs().idxmax()])
    g = pd.concat([g[g > 0].nlargest(SAP_MAX_PER_DIR), g[g < 0].nsmallest(SAP_MAX_PER_DIR)])
    log(f"  SAP-defining stromal genes: {(g > 0).sum()} activated / {(g < 0).sum()} quiescent")
    return np.sign(g)


# ==============================================================================
# 4. CELLPHONEDB (pooled, per SAP group)
# ==============================================================================
def cpdb_database():
    zp = DB_DIR / "cellphonedb.zip"
    if not zp.exists():
        from cellphonedb.utils import db_utils
        log(f"  downloading CellPhoneDB database {CPDB_VERSION}")
        db_utils.download_database(str(DB_DIR), CPDB_VERSION)
    return str(zp)


def run_cpdb(X, obs, genes, group, db):
    out = WORK / group
    cache = out / "means.tsv"
    if cache.exists() and not ARGS.rerun_cpdb:
        log(f"  {group}: reusing cached CellPhoneDB output")
        return {k: pd.read_csv(out / f"{k}.tsv", sep="\t", low_memory=False)
                for k in ("means", "pvalues", "deconvoluted")}
    import anndata as ad
    from cellphonedb.src.core.methods import cpdb_statistical_analysis_method
    out.mkdir(parents=True, exist_ok=True)
    o = obs[obs.SAP_group == group]
    rng = np.random.default_rng(SEED)
    take = []
    for _, d in o.groupby(["sample_id", "cell_type"], observed=True):   # cap per tumour x cell type
        take.extend(d.index if len(d) <= ARGS.max_cells else rng.choice(d.index, ARGS.max_cells, replace=False))
    o = o.loc[take]
    pos = obs.index.get_indexer(o.index)
    A = ad.AnnData(X=X[pos], obs=pd.DataFrame(index=o.index), var=pd.DataFrame(index=genes))
    f_cnt, f_meta = out / "counts.h5ad", out / "meta.tsv"
    A.write_h5ad(f_cnt)
    pd.DataFrame({"Cell": o.index, "cell_type": o.cell_type.values}).to_csv(f_meta, sep="\t", index=False)
    log(f"  {group}: CellPhoneDB on {len(o):,} cells from {o.sample_id.nunique()} tumours "
        f"({ARGS.iterations} permutations, {ARGS.threads} threads)")
    res = cpdb_statistical_analysis_method.call(
        cpdb_file_path=db, meta_file_path=str(f_meta), counts_file_path=str(f_cnt), counts_data="hgnc_symbol",
        output_path=str(out), iterations=ARGS.iterations, threshold=CPDB_THRESHOLD, threads=ARGS.threads,
        debug_seed=SEED, pvalue=CPDB_PVALUE, separator="|", result_precision=4)
    keep = {}
    for k in ("means", "pvalues", "deconvoluted"):
        res[k].to_csv(out / f"{k}.tsv", sep="\t", index=False)
        keep[k] = res[k]
    f_cnt.unlink(missing_ok=True)
    return keep


# ==============================================================================
# 5. TUMOUR-LEVEL CHANNEL SCORES
# ==============================================================================
def interaction_genes(means, decon):
    """id_cp_interaction -> (genes of partner a, genes of partner b); complexes -> all subunits."""
    d = decon.copy()
    by_uniprot = d.drop_duplicates("uniprot").set_index("uniprot")["gene_name"]
    by_complex = d[d.is_complex.astype(str).str.lower() == "true"].groupby("complex_name")["gene_name"].apply(
        lambda x: sorted(set(x)))

    def genes(partner, gene_col):
        p = str(partner)
        if p.startswith("simple:"):
            u = p.split(":", 1)[1]
            return [by_uniprot[u]] if u in by_uniprot else ([gene_col] if isinstance(gene_col, str) else [])
        if p.startswith("complex:"):
            return by_complex.get(p.split(":", 1)[1], [])
        return []
    out = {}
    for _, r in means.iterrows():
        out[r.id_cp_interaction] = (genes(r.partner_a, r.get("gene_a")), genes(r.partner_b, r.get("gene_b")))
    return out


def tumour_expression(X, obs, genes, needed):
    """Per tumour x cell type: mean log-expression and fraction expressing, for the needed genes."""
    gidx = genes.get_indexer(sorted(needed))
    ok = gidx >= 0
    gl = np.array(sorted(needed))[ok]
    Xs = X[:, gidx[ok]]
    key = obs.sample_id + "§" + obs.cell_type
    cats = pd.Categorical(key)
    G = sp.csr_matrix((np.ones(len(key)), (cats.codes, np.arange(len(key)))), shape=(len(cats.categories), len(key)))
    n = np.asarray(G.sum(axis=1)).ravel()
    mean = np.asarray((G @ Xs).todense()) / n[:, None]
    frac = np.asarray((G @ (Xs > 0).astype(np.float32)).todense()) / n[:, None]
    ix = pd.MultiIndex.from_tuples([tuple(c.split("§")) for c in cats.categories], names=["sample_id", "cell_type"])
    return (pd.DataFrame(mean, index=ix, columns=gl), pd.DataFrame(frac, index=ix, columns=gl),
            pd.Series(n, index=ix))


def side_score(mean, frac, genes_):
    g = [x for x in genes_ if x in mean.columns]
    if not g or len(g) < len(genes_):
        return None, None
    return mean[g].min(axis=1), frac[g].min(axis=1)


def partial_spearman(x, y, ds):
    d = pd.DataFrame({"x": x, "y": y, "ds": ds}).dropna()
    if len(d) < MIN_TUMOURS or d.x.nunique() < 3:
        return np.nan, np.nan, len(d)
    rx, ry = d.x.rank(), d.y.rank()
    Z = np.column_stack([np.ones(len(d)), pd.get_dummies(d.ds, drop_first=True).values.astype(float)])
    ex = rx - Z @ np.linalg.lstsq(Z, rx, rcond=None)[0]
    ey = ry - Z @ np.linalg.lstsq(Z, ry, rcond=None)[0]
    if ex.std() == 0 or ey.std() == 0:
        return np.nan, np.nan, len(d)
    r = np.corrcoef(ex, ey)[0, 1]
    k = Z.shape[1] - 1
    t = r * np.sqrt((len(d) - 2 - k) / max(1e-12, 1 - r ** 2))
    return r, 2 * stats.t.sf(abs(t), len(d) - 2 - k), len(d)


def tumour_level(cands, igenes, X, obs, genes, mcp, rows):
    needed = set()
    for i in cands.id_cp_interaction.unique():
        a, b = igenes.get(i, ([], []))
        needed |= set(a) | set(b)
    mean, frac, ncell = tumour_expression(X, obs, genes, needed)
    sap = mcp.MCP
    recs = []
    for iid, cr in cands.groupby("id_cp_interaction", sort=False):
        r = cr.iloc[0]
        ga, gb = igenes.get(iid, ([], []))
        ma, fa = side_score(mean, frac, ga)
        mb, fb = side_score(mean, frac, gb)
        if ma is None or mb is None:
            continue
        for row in [x for x in cr.row if x in rows]:      # only channels CPDB calls significant (non-directional)
            s, t = row.split("|")
            vals = {}
            for tum in sap.index:
                if (tum, s) not in ncell.index or (tum, t) not in ncell.index:
                    continue
                if ncell[(tum, s)] < MIN_CELLS_TUMOUR or ncell[(tum, t)] < MIN_CELLS_TUMOUR:
                    continue
                pa, pb = fa[(tum, s)] >= CPDB_THRESHOLD, fb[(tum, t)] >= CPDB_THRESHOLD
                vals[tum] = (ma[(tum, s)] + mb[(tum, t)]) / 2 if (pa and pb) else 0.0
            v = pd.Series(vals, dtype=float)
            rho, p, n = partial_spearman(v, sap.reindex(v.index), mcp.dataset.reindex(v.index))
            stromal_genes = ga if s in STROMA else gb
            arm = [SAPG[g] for g in stromal_genes if g in SAPG.index]
            recs.append(dict(id_cp_interaction=r.id_cp_interaction, interacting_pair=r.interacting_pair,
                             row=row, n_tumours=n, n_nonzero=int((v > 0).sum()), rho_SAP=rho, p=p,
                             SAP_gene_arm=(np.sign(np.sum(arm)) if arm else 0.0),
                             family="SAP-encoded" if arm else "exploratory"))
    t = pd.DataFrame(recs)
    t.loc[t.n_nonzero < 5, ["rho_SAP", "p"]] = np.nan
    # two pre-specified families: channels whose STROMAL partner is a SAP-defining gene (primary) vs all others
    t["q"] = np.nan
    for fam, d in t.groupby("family"):
        t.loc[d.index, "q"] = bh(d.p.values)
    return t


# ==============================================================================
# 6. ASSEMBLE THE PLOT TABLE
# ==============================================================================
def display_rows():
    rows = []
    if "stroma_to_myeloid" in DIRECTIONS:
        rows += [f"{s}|{m}" for s in STROMA for m in MYELOID]
    if "myeloid_to_stroma" in DIRECTIONS:
        rows += [f"{m}|{s}" for m in MYELOID for s in STROMA]
    return rows


def long_table(res, group, rows):
    meta_cols = [c for c in res["means"].columns if "|" not in c]
    cols = [r for r in rows if r in res["means"].columns]
    m = res["means"].melt(id_vars=meta_cols, value_vars=cols, var_name="row", value_name="mean")
    p = res["pvalues"].melt(id_vars=[c for c in res["pvalues"].columns if "|" not in c], value_vars=cols,
                            var_name="row", value_name="pvalue")
    m = m.merge(p[["id_cp_interaction", "row", "pvalue"]], on=["id_cp_interaction", "row"], how="left")
    m["group"] = group
    return m


def select_columns(L, tl):
    t = tl.copy()
    agg = t.groupby(["id_cp_interaction", "interacting_pair"]).agg(
        q_min=("q", "min"), p_min=("p", "min"),
        sap_encoded=("family", lambda x: (x == "SAP-encoded").any()),
        sap_arm=("SAP_gene_arm", lambda x: x[x != 0].iloc[0] if (x != 0).any() else 0.0),
        rho_best=("rho_SAP", lambda x: x.loc[t.loc[x.index, "p"].idxmin()] if t.loc[x.index, "p"].notna().any() else np.nan),
        rho_median=("rho_SAP", "median")).reset_index()
    agg = agg.dropna(subset=["p_min"])
    prim = agg[agg.sap_encoded & (agg.q_min < 0.10)].sort_values("p_min")
    rest = agg.drop(prim.index).sort_values(["q_min", "p_min"])
    agg = pd.concat([prim, rest]).head(ARGS.n_lr)
    agg = agg.sort_values("rho_best", ascending=False).reset_index(drop=True)
    return agg


def pretty_pair(s):
    return s.replace("_", "-").replace("complex", "cx")


# ==============================================================================
# 7. FIGURE 4A
# ==============================================================================
def figure_4a(L, cols, rows, n_tum):
    groups = ["SAP-high", "SAP-low"]
    row_order = [(g, r) for g in groups for r in rows]
    nr, nc = len(row_order), len(cols)
    cell = 2.55 * MM
    lab_w, strip_h, top_pad = 30 * MM, 3.2 * MM, 3 * MM
    plot_w, plot_h = nc * cell, nr * cell
    leg_w = 22 * MM
    fw = lab_w + plot_w + leg_w
    col_lab_h = 38 * MM
    fh = top_pad + 1.7 * strip_h + 1.8 * MM + plot_h + col_lab_h
    fig = plt.figure(figsize=(fw, fh))
    axm = fig.add_axes([lab_w / fw, col_lab_h / fh, plot_w / fw, plot_h / fh])
    axs = fig.add_axes([lab_w / fw, (col_lab_h + plot_h + 1.2 * MM) / fh, plot_w / fw, strip_h / fh])
    axa = fig.add_axes([lab_w / fw, (col_lab_h + plot_h + 1.2 * MM + strip_h + 0.6 * MM) / fh, plot_w / fw,
                        strip_h * 0.7 / fh])

    # scaled means: per interaction (column), min-max over all displayed rows of both groups
    L = L[L.id_cp_interaction.isin(cols.id_cp_interaction)].copy()
    rng_ = L.groupby("id_cp_interaction")["mean"].agg(["min", "max"])
    L = L.join(rng_, on="id_cp_interaction")
    L["scaled"] = ((L["mean"] - L["min"]) / (L["max"] - L["min"]).replace(0, np.nan)).fillna(0)
    xi = {i: k for k, i in enumerate(cols.id_cp_interaction)}
    yi = {k: nr - 1 - j for j, k in enumerate(row_order)}
    L["x"] = L.id_cp_interaction.map(xi)
    L["y"] = [yi.get((g, r), np.nan) for g, r in zip(L.group, L.row)]
    L = L.dropna(subset=["x", "y"])
    smax = (cell / (1 / 72) * 0.92) ** 2                 # dot area so the largest fits one cell
    size = 2 + smax * L.scaled.values ** 1.0
    sig = (L.pvalue < CPDB_PVALUE) & (L["mean"] > 0)
    for k in range(nc + 1):
        axm.axvline(k - 0.5, color=C_GRID, lw=0.3, zorder=0)
    for k in range(nr + 1):
        axm.axhline(k - 0.5, color=C_GRID, lw=0.3, zorder=0)
    sc_ = axm.scatter(L.x, L.y, s=size, c=L.scaled, cmap="viridis", vmin=0, vmax=1, lw=0, zorder=2)
    axm.scatter(L.x[sig], L.y[sig], s=size[sig.values] + 4, facecolors="none", edgecolors=C_SIG, lw=0.55, zorder=3)
    axm.axhline(len(rows) - 0.5, color=C_DARK, lw=0.6)          # boundary between groups
    axm.set_xlim(-0.5, nc - 0.5)
    axm.set_ylim(-0.5, nr - 0.5)
    axm.set_xticks(range(nc))
    axm.set_xticklabels([pretty_pair(s) for s in cols.interacting_pair], rotation=90, ha="center", va="top")
    axm.set_yticks([yi[k] for k in row_order])
    axm.set_yticklabels([f"{LABEL[r.split('|')[0]]} | {LABEL[r.split('|')[1]]}" for _, r in row_order])
    for sp_ in axm.spines.values():
        sp_.set_linewidth(0.6)
    # bold column labels when tumour-level FDR < 0.05
    for tl_, q in zip(axm.get_xticklabels(), cols.q_min):
        if q < FDR_STAR[0]:
            tl_.set_fontweight("bold")
    # group brackets (left)
    tr = axm.get_yaxis_transform()
    for gi_, g in enumerate(groups):
        y0 = nr - 1 - gi_ * len(rows)
        y1 = y0 - len(rows) + 1
        x = -lab_w / plot_w * 0.97
        axm.plot([x + 0.012, x, x, x + 0.012], [y0 + 0.4, y0 + 0.4, y1 - 0.4, y1 - 0.4], transform=tr,
                 color=C_HIGH if g == "SAP-high" else C_LOW, lw=0.8, clip_on=False)
        axm.text(x - 0.012, (y0 + y1) / 2, f"{g}\n(n = {n_tum[g]} tumours)", transform=tr, rotation=90,
                 ha="right", va="center", fontsize=6.5, fontweight="bold",
                 color=C_HIGH if g == "SAP-high" else C_LOW, clip_on=False)
    # direction separators within each group
    if len(DIRECTIONS) == 2:
        k1 = len(STROMA) * len(MYELOID)
        for gi_ in range(2):
            axm.axhline(nr - 1 - gi_ * len(rows) - k1 + 0.5, color="#9A9A9A", lw=0.4, ls=(0, (2, 2)))
    # top strip: tumour-level association with SAP
    rho = cols.rho_best.values
    axs.imshow(rho[None, :], cmap="RdBu_r", vmin=-0.6, vmax=0.6, aspect="auto", extent=(-0.5, nc - 0.5, 0, 1))
    for k, q in enumerate(cols.q_min):
        star = "**" if q < FDR_STAR[1] else ("*" if q < FDR_STAR[0] else "")
        if star:
            axs.text(k, 0.45, star, ha="center", va="center", fontsize=5.5, color="white" if abs(rho[k]) > 0.35 else C_DARK)
    axs.set_xlim(-0.5, nc - 0.5)
    axs.set_xticks([]); axs.set_yticks([0.5]); axs.set_yticklabels(["ρ with SAP (tumours)"], fontsize=5.6)
    for sp_ in axs.spines.values():
        sp_.set_linewidth(0.5)
    arm = cols.sap_arm.values
    for k, v in enumerate(arm):
        if v != 0:
            axa.add_patch(mpl.patches.Rectangle((k - 0.42, 0.1), 0.84, 0.8, color=C_HIGH if v > 0 else C_LOW, lw=0))
    axa.set_xlim(-0.5, nc - 0.5); axa.set_ylim(0, 1)
    axa.set_xticks([]); axa.set_yticks([0.5]); axa.set_yticklabels(["SAP gene (stromal partner)"], fontsize=5.6)
    for sp_ in axa.spines.values():
        sp_.set_visible(False)
    # legends (right)
    x0 = (lab_w + plot_w + 3 * MM) / fw
    cax = fig.add_axes([x0, (col_lab_h + plot_h * 0.67) / fh, 2.2 * MM / fw, plot_h * 0.19 / fh])
    cb = fig.colorbar(sc_, cax=cax)
    cb.outline.set_linewidth(0.4); cb.ax.tick_params(labelsize=5, width=0.4, length=1.5)
    cb.set_ticks([0, 0.25, 0.5, 0.75, 1])
    cax.set_title("scaled_means", fontsize=5.8, loc="left", pad=3)
    lax = fig.add_axes([x0, (col_lab_h + plot_h * 0.31) / fh, 18 * MM / fw, plot_h * 0.28 / fh])
    lax.axis("off")
    hs = [Line2D([], [], marker="o", ls="", color="k", markersize=np.sqrt(2 + smax * v), label=f"{v:g}")
          for v in (0.25, 0.5, 0.75, 1.0)]
    hs.append(Line2D([], [], marker="o", ls="", markerfacecolor="none", markeredgecolor=C_SIG, mew=0.8,
                     markersize=np.sqrt(smax * 0.6), label=f"CPDB P < {CPDB_PVALUE:g}"))
    lax.legend(handles=hs, title="scaled_means", loc="upper left", frameon=False, fontsize=5.4,
               title_fontsize=5.8, labelspacing=0.55, handletextpad=0.6, alignment="left")
    sax = fig.add_axes([x0, (col_lab_h + plot_h * 0.06) / fh, 2.2 * MM / fw, plot_h * 0.22 / fh])
    sm = mpl.cm.ScalarMappable(cmap="RdBu_r", norm=mpl.colors.Normalize(-0.6, 0.6))
    cb2 = fig.colorbar(sm, cax=sax)
    cb2.outline.set_linewidth(0.4); cb2.ax.tick_params(labelsize=5, width=0.4, length=1.5)
    cb2.set_ticks([-0.6, -0.3, 0, 0.3, 0.6])
    sax.set_title("ρ with SAP", fontsize=5.8, loc="left", pad=3)
    fig.text(x0, (col_lab_h + plot_h * 0.02) / fh, f"* q < {FDR_STAR[0]}, ** q < {FDR_STAR[1]}\n(tumour level,\n"
             "partial on dataset;\nbold = q < 0.05)", fontsize=5, va="top", ha="left")
    gax = fig.add_axes([x0, (col_lab_h + plot_h * 0.995) / fh, 18 * MM / fw, 1 * MM / fh])
    gax.axis("off")
    gax.legend(handles=[mpl.patches.Patch(color=C_HIGH, label="activated arm"),
                        mpl.patches.Patch(color=C_LOW, label="quiescent arm")],
               title="SAP gene", loc="lower left", frameon=False, fontsize=5.4, title_fontsize=5.8,
               handlelength=0.9, handleheight=0.9, alignment="left", borderaxespad=0)
    fig.text(0.005, 1 - 0.4 * MM / fh, "A", fontsize=9, fontweight="bold", va="top", ha="left")
    savefig(fig, "Fig4A_CPDB_stroma_TAM_SAP")


# ==============================================================================
# 8. MAIN
# ==============================================================================
def main():
    log("[1] tumour groups")
    mcp = sap_groups()
    global SAPG
    SAPG = sap_genes()
    log("[2] cells")
    X, obs, genes = load_cells(mcp)
    log("[3] CellPhoneDB (pooled per group)")
    db = cpdb_database()
    res = {g: run_cpdb(X, obs, genes, g, db) for g in ("SAP-high", "SAP-low")}
    rows = display_rows()
    L = pd.concat([long_table(res[g], g, rows) for g in res], ignore_index=True)
    L.to_csv(TAB / "Fig4A_cpdb_long_displayed_rows.csv", index=False)
    log("[4] tumour-level channel scores vs SAP")
    if EXCLUDE_SMALL_MOLECULE:
        small = L.interacting_pair.astype(str).str.contains(r"_by[A-Z0-9]", regex=True)
        log(f"  metabolite-mediated pairs excluded: {L.loc[small, 'id_cp_interaction'].nunique()}")
        L = L[~small]
    sigL = L[(L.pvalue < CPDB_PVALUE) & (L["mean"] > 0)]
    cands = sigL[["id_cp_interaction", "interacting_pair", "row"]].drop_duplicates(["id_cp_interaction", "row"])
    log(f"  candidate channels (CPDB-significant in that sender|receiver row in >= 1 group): {len(cands)} "
        f"({cands.id_cp_interaction.nunique()} interactions)")
    igenes = interaction_genes(res["SAP-high"]["means"], pd.concat([res[g]["deconvoluted"] for g in res]))
    tl = tumour_level(cands, igenes, X, obs, genes, mcp, rows)
    tl.to_csv(TAB / "Fig4A_tumour_level_channels_vs_SAP.csv", index=False)
    for fam, d in tl.groupby("family"):
        log(f"  {fam}: channels tested {d.p.notna().sum()}, q < 0.05: {(d.q < 0.05).sum()}")
    cols = select_columns(L, tl)
    cols.to_csv(TAB / "Fig4A_selected_interactions.csv", index=False)
    log(f"  displayed interactions: {len(cols)} ({(cols.q_min < 0.05).sum()} with tumour-level q < 0.05)")
    n_tum = mcp.SAP_group.value_counts().to_dict()
    log("[5] figure")
    figure_4a(L, cols, rows, n_tum)
    top = tl.dropna(subset=["p"]).sort_values("p").head(15)
    lines = ["Figure 4A -- numbers for Methods/Results (auto-generated)",
             f"Groups: SAP tertiles within dataset; SAP-high n = {n_tum.get('SAP-high')}, SAP-low n = {n_tum.get('SAP-low')} tumours",
             f"CellPhoneDB {CPDB_VERSION}, {ARGS.iterations} permutations, threshold {CPDB_THRESHOLD}, "
             f"cap {ARGS.max_cells} cells per tumour x cell type",
             f"Candidates {len(cands)}; tumour-level channels tested {tl.p.notna().sum()}; q < 0.05: {(tl.q < 0.05).sum()}",
             "Top tumour-level channels:"]
    for fam, d in tl.groupby("family"):
        lines.append(f"Family {fam}: tested {d.p.notna().sum()}, q < 0.05: {(d.q < 0.05).sum()}")
    lines += [f"  {r.interacting_pair} [{r.row}] ({r.family}): rho = {r.rho_SAP:+.2f}, P = {r.p:.2g}, "
              f"q = {r.q:.2g}, n = {r.n_tumours}" for _, r in top.iterrows()]
    (OUT / "methods_Fig4A.txt").write_text("\n".join(lines))
    log("\n".join(lines))


if __name__ == "__main__":
    main()
