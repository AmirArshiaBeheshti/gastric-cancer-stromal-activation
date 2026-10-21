"""
GC_SAP_survival_step2_cox.py -- STEP 2: pre-specified overall-survival analysis of SAP (H3)
in the Yonsei cohort (GSE84437).

Inputs (all produced earlier by this project; nothing is simulated or imputed):
  * GC_SAP_validation_outputs/A_external_scores_per_sample.csv   frozen per-tumour scores (SAP, stromal abundance)
  * GC_SAP_validation_outputs/raw/GSE84437_family.soft(.gz)       GEO clinical fields (downloaded by the
                                                                   validation script; re-downloaded if absent)
Order of operations:
  1. check inputs and record their SHA-256
  2. write the analysis plan (H3_analysis_plan.json) BEFORE any survival model is fitted;
     if a plan from an earlier run exists, the run stops unless the specification is identical
  3. fit the pre-specified models, write tables, figure, provenance and log
  4. zip the results and copy them to ~/Downloads (or show a download link in Jupyter/Colab)

ACRG (GSE62254) is not analysed: GEO provides no survival information for it (step 1).

Run:   python GC_SAP_survival_step2_cox.py          (GC_BASE=/path to change the project folder)
Needs: numpy, pandas, scipy, statsmodels, matplotlib, GEOparse
"""
import datetime
import hashlib
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

# ------------------------------------------------------------------ paths and fixed settings
BASE = Path(os.environ.get("GC_BASE", "/home/arshia/payanname"))
VAL = BASE / "GC_SAP_validation_outputs"
F_SCORES = VAL / "A_external_scores_per_sample.csv"
RAW = VAL / "raw"
OUT = BASE / "GC_SAP_survival_outputs"
PLAN = OUT / "H3_analysis_plan.json"
LOGFILE = OUT / "H3_run_log.txt"

COHORT = "GSE84437"
F_TIME, F_EVENT = "duration overall survival", "death"          # GEO field names (step 1)
F_AGE, F_SEX, F_T, F_N = "age", "Sex", "ptstage", "pnstage"
T_MAP = {"T1": "T1-2", "T2": "T1-2", "T3": "T3", "T4": "T4"}     # pT1 has only 11 patients
N_LEVELS = ["N0", "N1", "N2", "N3"]
ZERO_TIME = 0.5                                                  # months; replaces follow-up of 0
KM_TICKS = [0, 24, 48, 72, 96, 120, 144]
MIN_PATIENTS = 300                                               # stop if fewer complete cases

COVARS = "age10 + C(sex, Treatment('female')) + C(pT, Treatment('T1-2')) + C(pN, Treatment('N0'))"
MODELS = {   # name: (formula right-hand side, role)
    "primary: SAP | stroma + clinical": (f"SAP_z + Stroma_z + {COVARS}", "primary"),
    "secondary: SAP | clinical (no stromal adjustment)": (f"SAP_z + {COVARS}", "secondary"),
    "secondary: SAP alone": ("SAP_z", "secondary"),
}

SPEC = {
    "analysis": "H3: SAP and overall survival in the Yonsei cohort (GSE84437)",
    "not_analysed": "ACRG (GSE62254): GEO provides no survival or clinical fields (only patient number and tissue)",
    "inclusion": "tumours analysed in the bulk validation with non-missing death, duration overall survival, "
                 "age, sex, pT and pN in GEO",
    "endpoint": f"overall survival; time = GEO field '{F_TIME}' (months); event = GEO field '{F_EVENT}' == 1; "
                f"time 0 set to {ZERO_TIME} month",
    "exposure": "SAP score from A_external_scores_per_sample.csv (frozen validation output, unchanged), "
                "standardised (mean 0, SD 1) within the analysed tumours; HR per SD",
    "covariates": {"Stroma_z": "stromal-abundance score from the same file, standardised within analysed tumours",
                   "age10": "age / 10 (years)", "sex": "female (reference) / male",
                   "pT": "T1-2 (reference), T3, T4", "pN": "N0 (reference), N1, N2, N3"},
    "H3": "In the primary model, higher SAP is associated with shorter overall survival (HR per SD > 1)",
    "primary_test": "Cox proportional-hazards model (Efron ties), primary formula below; one-sided Wald P for "
                    "HR > 1; alpha 0.05; a single primary test, no multiplicity correction",
    "models": {k: v[0] for k, v in MODELS.items()},
    "secondary": ["the two secondary Cox models above (two-sided Wald P, reported descriptively)",
                  "proportional-hazards check for SAP in the primary model: Pearson correlation of SAP Schoenfeld "
                  "residuals with the rank of event time",
                  "Kaplan-Meier curves for SAP above vs at/below the median with log-rank P (display only)"],
    "reporting": "all estimates are reported whatever their direction or significance",
    "params": {"ZERO_TIME": ZERO_TIME, "T_MAP": T_MAP, "MIN_PATIENTS": MIN_PATIENTS},
}


# ------------------------------------------------------------------ helpers
def log(msg):
    line = f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(LOGFILE, "a") as fh:
        fh.write(line + "\n")


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def soft_file():
    hits = sorted(RAW.glob(f"{COHORT}_family.soft*"))
    if hits:
        return hits[0]
    import GEOparse
    RAW.mkdir(parents=True, exist_ok=True)
    GEOparse.get_GEO(geo=COHORT, destdir=str(RAW), silent=True)
    hits = sorted(RAW.glob(f"{COHORT}_family.soft*"))
    if not hits:
        sys.exit(f"could not obtain the GEO SOFT file for {COHORT}")
    return hits[0]


def clinical(soft):
    import GEOparse
    gse = GEOparse.get_GEO(filepath=str(soft), silent=True)
    rows = {}
    for gsm, s in gse.gsms.items():
        r = {}
        for c in s.metadata.get("characteristics_ch1", []):
            if ":" in c:
                k, v = c.split(":", 1)
                r[k.strip()] = v.strip()
        rows[gsm] = r
    return pd.DataFrame.from_dict(rows, orient="index")


# ------------------------------------------------------------------ steps
def check_inputs():
    if not F_SCORES.exists():
        sys.exit(f"missing {F_SCORES} -- run GC_SAP_external_validation.py first")
    OUT.mkdir(parents=True, exist_ok=True)
    for f in OUT.glob("H3_*"):                      # results of earlier step-2 runs (the plan is kept)
        if f.is_file() and f != PLAN:
            f.unlink()
    soft = soft_file()
    return {str(F_SCORES): sha256(F_SCORES), str(soft): sha256(soft)}, soft


def write_plan(inputs):
    spec = dict(SPEC, inputs=inputs)
    if PLAN.exists():
        old = json.loads(PLAN.read_text())
        created = old.pop("created")
        if old != spec:
            sys.exit(f"{PLAN.name} from an earlier run ({created}) differs from this specification; "
                     "the frozen plan is not overwritten. Restore the original settings or inputs.")
        log(f"frozen plan from {created} found and unchanged")
        return
    PLAN.write_text(json.dumps({"created": datetime.datetime.now().isoformat(timespec="seconds"), **spec}, indent=2))
    log(f"analysis plan frozen -> {PLAN.name} (before any survival model was fitted)")


def build_data(soft):
    sc = pd.read_csv(F_SCORES)
    sc = sc[sc.cohort == COHORT].set_index("name")
    cl = clinical(soft)
    d = sc[["SAP", "Stroma_abundance"]].join(cl, how="left")
    need = [F_TIME, F_EVENT, F_AGE, F_SEX, F_T, F_N]
    d = d.replace("", np.nan)
    excluded = int(d[need].isna().any(axis=1).sum())
    d = d.dropna(subset=need).copy()
    d["time"] = pd.to_numeric(d[F_TIME], errors="raise").astype(float)
    d["event"] = pd.to_numeric(d[F_EVENT], errors="raise").astype(int)
    assert set(d.event.unique()) <= {0, 1}, "death must be 0/1"
    assert (d.time >= 0).all(), "negative follow-up"
    n_zero = int((d.time == 0).sum())
    d.loc[d.time == 0, "time"] = ZERO_TIME
    d["age10"] = pd.to_numeric(d[F_AGE], errors="raise") / 10
    d["sex"] = d[F_SEX].str.lower()
    assert set(d.sex) <= {"female", "male"}, set(d.sex)
    assert set(d[F_T]) <= set(T_MAP), set(d[F_T])
    assert set(d[F_N]) <= set(N_LEVELS), set(d[F_N])
    d["pT"] = d[F_T].map(T_MAP)
    d["pN"] = d[F_N]
    d["SAP_z"] = (d.SAP - d.SAP.mean()) / d.SAP.std(ddof=1)
    d["Stroma_z"] = (d.Stroma_abundance - d.Stroma_abundance.mean()) / d.Stroma_abundance.std(ddof=1)
    if len(d) < MIN_PATIENTS:
        sys.exit(f"only {len(d)} complete cases (< {MIN_PATIENTS}) -- stopped")
    log(f"{COHORT}: {len(sc)} scored tumours, {excluded} without complete clinical data, "
        f"{len(d)} analysed, {d.event.sum()} deaths, {n_zero} follow-up times of 0 set to {ZERO_TIME}")
    return d


def fit_cox(d, rhs):
    import statsmodels.formula.api as smf
    from scipy import stats
    m = smf.phreg(f"time ~ {rhs}", d, status=d["event"].values, ties="efron")
    r = m.fit()
    names = m.exog_names
    ci = np.asarray(r.conf_int())
    z = r.params / r.bse
    tab = pd.DataFrame({"term": names, "coef": r.params, "se": r.bse, "HR": np.exp(r.params),
                        "HR_lo": np.exp(ci[:, 0]), "HR_hi": np.exp(ci[:, 1]), "z": z,
                        "P_two_sided": r.pvalues, "P_one_sided_HR_gt_1": stats.norm.sf(z)})
    return r, m, tab


def ph_check(r, m, d):
    from scipy import stats
    res = np.asarray(r.schoenfeld_residuals)
    j = m.exog_names.index("SAP_z")
    ev = d["event"].values == 1
    x = res[ev, j]
    t = stats.rankdata(d["time"].values[ev])
    ok = np.isfinite(x)
    rr, p = stats.pearsonr(x[ok], t[ok])
    return dict(term="SAP_z", r_schoenfeld_vs_rank_time=rr, P=p, n_events=int(ok.sum()))


def km(d):
    from statsmodels.duration.survfunc import SurvfuncRight, survdiff
    grp = np.where(d.SAP > d.SAP.median(), "SAP high", "SAP low")
    chisq, p = survdiff(d.time.values, d.event.values, grp)
    curves, risk = {}, []
    for g in ["SAP low", "SAP high"]:
        m = grp == g
        sf = SurvfuncRight(d.time.values[m], d.event.values[m])
        curves[g] = sf
        risk.append({"group": g, **{f"at_risk_{t}m": int((d.time.values[m] >= t).sum()) for t in KM_TICKS},
                     "n": int(m.sum()), "deaths": int(d.event.values[m].sum()),
                     "median_OS_months": sf.quantile(0.5)})
    rev = SurvfuncRight(d.time.values, 1 - d.event.values)        # reverse KM: median follow-up
    return grp, curves, pd.DataFrame(risk), dict(logrank_chisq=chisq, logrank_P=p,
                                                  median_followup_months=rev.quantile(0.5))


def figure(d, grp, curves, kmstat, res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 8, "axes.linewidth": 0.6,
                         "savefig.dpi": 600, "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.1, 3.1), gridspec_kw={"width_ratios": [1.2, 1], "wspace": 0.75})
    col = {"SAP low": "#4D4D4D", "SAP high": "#B2182B"}
    for g, sf in curves.items():
        t = np.r_[0, sf.surv_times]
        s = np.r_[1, sf.surv_prob]
        a.step(t, s, where="post", color=col[g], lw=1.1, label=f"{g} (n = {int((grp == g).sum())})")
        m = (grp == g) & (d.event.values == 0)
        ct = d.time.values[m]
        cs = [s[np.searchsorted(t, x, side="right") - 1] for x in ct]
        a.plot(ct, cs, "|", color=col[g], ms=3, mew=0.6)
    a.set_xlim(0, max(KM_TICKS) + 6)
    a.set_ylim(0, 1.02)
    a.set_xticks(KM_TICKS)
    a.set_xlabel("Months")
    a.set_ylabel("Overall survival")
    a.legend(frameon=False, loc="lower left")
    a.text(0.98, 0.98, f"log-rank P = {kmstat['logrank_P']:.2g}\n(median split, display only)",
           transform=a.transAxes, ha="right", va="top", fontsize=7)
    for s_ in ["top", "right"]:
        a.spines[s_].set_visible(False)
    for k, g in enumerate(["SAP low", "SAP high"]):              # numbers at risk
        m = grp == g
        a.text(-0.07, -0.30 - 0.07 * k, g, transform=a.transAxes, ha="right", va="center", fontsize=6.5,
               color=col[g])
        for t in KM_TICKS:
            a.text(t, -0.30 - 0.07 * k, str(int((d.time.values[m] >= t).sum())), transform=a.get_xaxis_transform(),
                   ha="center", va="center", fontsize=6.5, color=col[g])
    a.text(-0.07, -0.23, "At risk", transform=a.transAxes, ha="right", va="center", fontsize=6.5)
    a.set_title("A", loc="left", fontweight="bold")
    rows = res[res.term == "SAP_z"].reset_index(drop=True)
    y = np.arange(len(rows))[::-1]
    b.axvline(1, color="grey", lw=0.6, ls="--")
    b.errorbar(rows.HR, y, xerr=[rows.HR - rows.HR_lo, rows.HR_hi - rows.HR], fmt="s", color="black",
               ms=4, capsize=2, lw=0.8)
    b.set_yticks(y)
    short = {"primary": "Adjusted: stroma +\nage, sex, pT, pN", "no stromal": "Adjusted: age, sex,\npT, pN",
             "alone": "Unadjusted"}
    b.set_yticklabels([next(v for k, v in short.items() if k in m) for m in rows.model], fontsize=7)
    b.set_ylim(-0.6, len(rows) - 0.4)
    b.set_xscale("log")
    lo, hi = min(rows.HR_lo.min(), 1) / 1.08, max(rows.HR_hi.max(), 1) * 1.08
    b.set_xlim(lo, hi)
    from matplotlib.ticker import FixedLocator, NullLocator, FuncFormatter
    ticks = [t for t in [0.25, 0.5, 0.67, 0.8, 1, 1.25, 1.5, 2, 3, 4] if lo <= t <= hi]
    b.xaxis.set_major_locator(FixedLocator(ticks))
    b.xaxis.set_minor_locator(NullLocator())
    b.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    b.set_xlabel("HR per SD of SAP (95% CI)")
    for yi, (_, r) in zip(y, rows.iterrows()):
        b.text(1.02, yi, f"{r.HR:.2f} ({r.HR_lo:.2f}-{r.HR_hi:.2f})", transform=b.get_yaxis_transform(),
               va="center", fontsize=7)
    for s_ in ["top", "right"]:
        b.spines[s_].set_visible(False)
    b.set_title("B", loc="left", fontweight="bold")
    for ext in ["pdf", "png", "svg"]:
        fig.savefig(OUT / f"H3_survival_figure.{ext}", bbox_inches="tight")
    plt.close(fig)


def provenance(inputs):
    import importlib.metadata as im
    vers = {}
    for p in ["numpy", "pandas", "scipy", "statsmodels", "matplotlib", "GEOparse"]:
        try:
            vers[p] = im.version(p)
        except Exception:
            vers[p] = None
    outs = {f.name: sha256(f) for f in sorted(OUT.glob("H3_*")) if f.is_file() and f.name != "H3_run_log.txt"}
    (OUT / "H3_provenance.json").write_text(json.dumps(
        {"python": sys.version.split()[0], "packages": vers, "inputs": inputs, "outputs": outs}, indent=2))


def package():
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    zp = OUT / f"GC_SAP_survival_results_{stamp}.zip"
    files = [f for f in OUT.glob("H3_*") if f.is_file()]
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, f.name)
    log(f"results packed: {zp} ({len(files)} files)")
    try:
        from google.colab import files as colab_files
        colab_files.download(str(zp))
        return
    except Exception:
        pass
    try:
        from IPython import get_ipython
        if get_ipython() is not None and "IPKernelApp" in get_ipython().config:
            from IPython.display import FileLink, display
            display(FileLink(os.path.relpath(zp), result_html_prefix="Download results: "))
    except Exception:
        pass
    dl = Path(os.environ.get("DOWNLOAD_DIR", Path.home() / "Downloads"))
    try:
        dl.mkdir(parents=True, exist_ok=True)
        shutil.copy2(zp, dl / zp.name)
        log(f"results copied to {dl / zp.name}")
    except Exception as e:
        log(f"could not copy to {dl} ({e}); the zip is at {zp}")


def main():
    inputs, soft = check_inputs()
    write_plan(inputs)                                   # frozen before any model is fitted
    d = build_data(soft)
    d.rename_axis("sample").reset_index()[
        ["sample", "SAP", "Stroma_abundance", "SAP_z", "Stroma_z", "time", "event", "age10", "sex", "pT", "pN"]
    ].to_csv(OUT / "H3_analysis_data.csv", index=False)

    # cohort description
    desc = {"n": len(d), "deaths": int(d.event.sum()), "age_median": float(d.age10.median() * 10),
            "male": int((d.sex == "male").sum()), **{f"pT_{k}": int((d.pT == k).sum()) for k in ["T1-2", "T3", "T4"]},
            **{f"pN_{k}": int((d.pN == k).sum()) for k in N_LEVELS}}

    res, prim = [], None
    for name, (rhs, role) in MODELS.items():
        r, m, tab = fit_cox(d, rhs)
        tab.insert(0, "model", name)
        tab.insert(1, "role", role)
        tab["n"], tab["events"] = len(d), int(d.event.sum())
        res.append(tab)
        if role == "primary":
            prim = (r, m)
        s = tab[tab.term == "SAP_z"].iloc[0]
        log(f"{name}: HR per SD of SAP {s.HR:.3f} (95% CI {s.HR_lo:.3f}-{s.HR_hi:.3f}), "
            f"P two-sided {s.P_two_sided:.3g}, one-sided (HR>1) {s.P_one_sided_HR_gt_1:.3g}")
    res = pd.concat(res, ignore_index=True)
    res.to_csv(OUT / "H3_cox_models.csv", index=False)

    ph = ph_check(*prim, d)
    log(f"PH check (SAP, primary model): r = {ph['r_schoenfeld_vs_rank_time']:.3f}, P = {ph['P']:.3g}")
    grp, curves, risk, kmstat = km(d)
    risk.to_csv(OUT / "H3_km_groups.csv", index=False)
    log(f"KM median split: log-rank P = {kmstat['logrank_P']:.3g}; median follow-up "
        f"{kmstat['median_followup_months']:.1f} months (reverse KM)")

    p1 = res[(res.role == "primary") & (res.term == "SAP_z")].iloc[0]
    summary = {"cohort": COHORT, **desc, "median_followup_months": kmstat["median_followup_months"],
               "primary_HR_per_SD": p1.HR, "primary_HR_lo": p1.HR_lo, "primary_HR_hi": p1.HR_hi,
               "primary_P_one_sided": p1.P_one_sided_HR_gt_1, "primary_P_two_sided": p1.P_two_sided,
               "H3_supported": bool(p1.HR > 1 and p1.P_one_sided_HR_gt_1 < 0.05),
               "PH_r": ph["r_schoenfeld_vs_rank_time"], "PH_P": ph["P"],
               "logrank_P_median_split": kmstat["logrank_P"]}
    pd.DataFrame([summary]).to_csv(OUT / "H3_summary.csv", index=False)
    figure(d, grp, curves, kmstat, res)
    provenance(inputs)
    log(f"H3 {'supported' if summary['H3_supported'] else 'not supported'} "
        f"(HR {p1.HR:.2f}, one-sided P {p1.P_one_sided_HR_gt_1:.3g}); outputs in {OUT}")
    package()


if __name__ == "__main__":
    main()
