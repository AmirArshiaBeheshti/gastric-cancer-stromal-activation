"""
GC_SAP_survival_step1_inspect.py -- STEP 1 of the survival analysis (no statistical test is run).

Lists the clinical fields that GEO provides for the two bulk validation cohorts, so that the
survival analysis plan (step 2) can name the exact fields before any SAP-survival test is run.

It reads the GEO SOFT files already downloaded by GC_SAP_external_validation.py
(GC_SAP_validation_outputs/raw/); if a file is missing it is downloaded again with GEOparse.
It only reports field names, missingness and the distribution of values; it does not relate
any field to SAP or to any signature.

Run:  python GC_SAP_survival_step1_inspect.py          (GC_BASE=/path to change the project folder)
Then send:  GC_SAP_survival_outputs/step1_clinical_fields.txt
"""
import os
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

BASE = Path(os.environ.get("GC_BASE", "/home/arshia/payanname"))
RAW = BASE / "GC_SAP_validation_outputs" / "raw"
SAMPLES = BASE / "GC_SAP_validation_outputs" / "A_samples_used.csv"
OUT = BASE / "GC_SAP_survival_outputs"
COHORTS = ["GSE62254", "GSE84437"]


def get_gse(acc):
    import GEOparse
    soft = sorted(RAW.glob(f"{acc}_family.soft*"))
    if soft:
        return GEOparse.get_GEO(filepath=str(soft[0]), silent=True)
    RAW.mkdir(parents=True, exist_ok=True)
    return GEOparse.get_GEO(geo=acc, destdir=str(RAW), silent=True)


def characteristics(gse):
    rows = {}
    for gsm, s in gse.gsms.items():
        md = s.metadata
        r = {"title": " ".join(md.get("title", [])), "source_name": " ".join(md.get("source_name_ch1", []))}
        unlabelled = []
        for c in md.get("characteristics_ch1", []):
            if ":" in c:
                k, v = c.split(":", 1)
                r[k.strip()] = v.strip()
            else:
                unlabelled.append(c.strip())
        if unlabelled:
            r["unlabelled"] = "; ".join(unlabelled)
        rows[gsm] = r
    return pd.DataFrame.from_dict(rows, orient="index")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    used = pd.read_csv(SAMPLES) if SAMPLES.exists() else None
    lines = []
    for acc in COHORTS:
        gse = get_gse(acc)
        parts = [(acc, gse)]
        if len(gse.gsms) == 0:
            subs = [r.split(":")[-1].strip() for r in gse.metadata.get("relation", []) if "SuperSeries of" in r]
            parts = [(s, get_gse(s)) for s in subs]
        df = pd.concat([characteristics(g).assign(subseries=s) for s, g in parts])
        if used is not None:
            keep = used.loc[used.cohort == acc, "sample"]
            df = df.loc[df.index.intersection(keep)]
        df.to_csv(OUT / f"step1_characteristics_{acc}.csv")
        lines += ["=" * 90, f"{acc}: {len(df)} samples (those analysed in the validation)", "=" * 90]
        sup = [x for x in gse.metadata.get("supplementary_file", [])]
        lines.append(f"series supplementary files: {sup}")
        for col in df.columns:
            v = df[col].dropna().astype(str)
            v = v[v.str.strip() != ""]
            uniq = v.unique()
            num = pd.to_numeric(v, errors="coerce")
            if len(uniq) <= 12:
                desc = "; ".join(f"{u} (n={int((v == u).sum())})" for u in sorted(uniq, key=str))
            elif num.notna().mean() > 0.9:
                desc = f"numeric: min {num.min():g}, median {num.median():g}, max {num.max():g}"
            else:
                desc = "examples: " + " | ".join(map(str, uniq[:5]))
            lines.append(f"- {col!r}: non-missing {len(v)}/{len(df)}, {len(uniq)} distinct -> {desc}")
        lines.append("")
    txt = "\n".join(lines)
    (OUT / "step1_clinical_fields.txt").write_text(txt)
    print(txt)
    print(f"\nwritten: {OUT / 'step1_clinical_fields.txt'}")


if __name__ == "__main__":
    main()
