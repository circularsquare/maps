"""Liberia: Afrobarometer first-language answers by county, on the 2022 census's county populations.

    python sources/wafr_afro.py lr     the respondents -> data/raw/lr/lr_afro.csv (read-only .sav)
    python sources/lr_afro.py          -> data/normalized/lr.csv (county x language, counts)

THE CENSUS. Liberia's censuses ask ethnicity ("tribe") and publish it only nationally: the 2008
Final Results (data/raw/lr/nphc_2008_final_report.pdf, Wayback copy of lisgis.net) has Table
4.4, ethnic affiliation by age and sex, Liberia only; religiondots found the 2022 Final Results
national-only throughout for the social items. No language question. So the county pattern is
the survey's (AGENT_BRIEF section 2's survey case), rows `modelled`, and the census's 2008
national ethnic shares are the check (sources/lr.md).

THE SURVEY. Afrobarometer R4 (2008), R5 (2012), R6 (2015), R7 (2018), ~1,200 adults each, all
15 counties: R4-R6 "language of respondent", R7 "mother tongue" (Q2A). Ask 018's first-language
reading; R8-R9 ("language spoken in home") put English and Liberian English at 39% and are
shown for comparison only (FIRST_ROUNDS). Weighted shares per county, pooled, unshrunk (a
county's own respondents only; River Cess has the fewest).

ENGLISH. "English", "Liberian English" and "Simple Liberian English" answers (7.9% in R4-R7,
nearly all from English or Liberian English interviews, the only interview languages) are moved
to the language of the respondent's own ethnic group where that group has one; respondents
whose ethnic answer is English, national identity, other or missing keep it, drawn as Liberian
English. MOVE_ENGLISH is the one-line switch.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent.parent
RD = HERE.parent / "religiondots"
EXTRACT = HERE / "data" / "raw" / "lr" / "lr_afro.csv"
LOOKUP = RD / "data" / "geo" / "lr" / "lr_lookup.csv"
OUT = HERE / "data" / "normalized" / "lr.csv"
SOURCE_ID = "afrobarometer_r4_r7_liberia"
CENSUS_2022 = 5_250_187

FIRST_ROUNDS = (4, 5, 6, 7)     # (8, 9) = "language spoken in home"
MOVE_ENGLISH = True
ENGLISH_AT_R7 = True     # ask 018 ruling, 2026-10-05: English at R7's Q2A share (english_at_r7)

ENGLISH = {"English", "Liberian English", "Simple Liberian English"}
LANGS = {"Kpelle", "Bassa", "Grebo", "Gio", "Mano", "Lorma", "Kru", "Gola", "Kissi", "Vai",
         "Krahn", "Mandingo", "Gbandi", "Mende", "Belle", "Dei", "Sarpo"}
DROP = {"Missing", "Don't know", "Refused To Answer", "Refused", "nan", ""}
# 2008 census, Table 4.4, national % by ethnic group (the check; transcribed in sources/lr.md)
COUNTY = {"capemount": "grandcapemount", "bassa": "grandbassa", "rivercess": "rivercess"}


def say(ok, msg):
    print(("  ok  " if ok else "  FAIL") + "  " + msg)
    if not ok:
        raise SystemExit("check failed: " + msg)


def key(s):
    k = "".join(c for c in str(s).casefold() if c.isalpha())
    return COUNTY.get(k, k)


def load(lut):
    a = pd.read_csv(EXTRACT, keep_default_na=False)
    units = {key(n): u for u, n in zip(lut["unit"], lut["name"])}
    a["unit"] = a["region"].map(lambda s: units.get(key(s)))
    say(a["unit"].notna().all(), f"every respondent's county is a unit "
        f"{sorted(a.loc[a['unit'].isna(), 'region'].unique())}")
    a = a[~a["lang"].isin(DROP)].copy()
    a["l"] = a["lang"].map(lambda s: "Liberian English" if s in ENGLISH
                           else s if s in LANGS else "Other")
    return a


def shares(a, rounds, move):
    s = a[a["round"].isin(rounds)].copy()
    if move:
        en = s["l"] == "Liberian English"
        own = en & s["eth"].isin(LANGS)
        print(f"  English answers in rounds {rounds}: {int(en.sum())}, interview language "
              f"{s.loc[en, 'interview'].value_counts().to_dict()}; {int(own.sum())} moved")
        s.loc[own, "l"] = s.loc[own, "eth"]
    sh = s.groupby(["unit", "l"])["w"].sum()
    return sh / sh.groupby(level=0).transform("sum"), s.groupby("unit").size()


def english_at_r7(sh, lut):
    """Ask 018 (Anita, 2026-10-05): English at R7's mother-tongue question (Q2A; "English" and
    "Liberian English" together, drawn as Liberian English), each county's share shrunk to the
    national one by wafr_afro.K_SHRINK respondents; the county's other answers (English answers
    already moved to the ethnic language) scaled to what is left."""
    sys.path.insert(0, str(HERE / "sources"))
    from wafr_afro import r7_mother, shrink
    t = r7_mother("Liberia", {"en": sorted(ENGLISH)})
    units = {key(n): u for u, n in zip(lut["unit"], lut["name"])}
    reg = {r: units.get(key(r)) for r in t.index if r != "_national"}
    say(None not in reg.values() and len(set(reg.values())) == len(units),
        f"R7's REGION labels are the 15 counties {reg}")
    en = shrink(t, "en").rename(reg)
    print(f"  R7 Q2A English: {t.loc['_national', 'en_A'] / t.loc['_national', 'n']:.2%} "
          "nationally; drawn " + ", ".join(f"{u} {v:.1%}" for u, v in en.items()))
    out = []
    for u, s in sh.groupby(level=0):
        s = s.droplevel(0).drop("Liberian English", errors="ignore")
        s = s / s.sum() * (1 - en[u])
        s["Liberian English"] = en[u]
        out.append(pd.concat({u: s}))
    return pd.concat(out)


def main():
    lut = pd.read_csv(LOOKUP, dtype={"unit": str})
    pop = lut.set_index("unit")["pop"].astype(int)
    names = lut.set_index("unit")["name"]
    say(int(pop.sum()) == CENSUS_2022, f"15 counties, {int(pop.sum()):,} people (2022 census)")
    a = load(lut)
    say(set(a["unit"]) == set(pop.index), "every county has respondents")
    natl = {}
    for lab, r, mv in (("R4-R7 drawn", (4, 5, 6, 7), True),
                       ("R4-R7 English as given", (4, 5, 6, 7), False),
                       ("R8-R9 home", (8, 9), False)):
        sh, _ = shares(a, r, mv)
        natl[lab] = (sh * pop.reindex(sh.index.get_level_values(0)).values).groupby(
            level=1).sum() / pop.sum() * 100
    print(pd.DataFrame(natl).fillna(0).sort_values(list(natl)[0], ascending=False)
          .round(2).to_string())

    sh, n = shares(a, FIRST_ROUNDS, MOVE_ENGLISH)
    if ENGLISH_AT_R7:
        sh = english_at_r7(sh, lut)
    rows = []
    for u in pop.index:
        s = sh.loc[u]
        f = (s * pop[u]).to_numpy()
        base = np.floor(f)
        k = int(round(pop[u] - base.sum()))
        base[np.argsort(-(f - base))[:k]] += 1
        for lang, share, c in zip(s.index, s.values, base.astype(int)):
            if c > 0:
                rows.append((u, names[u], lang, c, share, int(n[u])))
    df = pd.DataFrame(rows, columns=["unit", "name", "lang", "count", "share", "n"])
    say(int(df["count"].sum()) == CENSUS_2022, f"drawn total {int(df['count'].sum()):,}")
    for u in pop.index:
        top = df[df["unit"] == u].sort_values("count", ascending=False).head(4)
        print(f"  {names[u]:17s} n={int(n[u]):4d}  " + ", ".join(
            f"{l} {s:.0%}" for l, s in zip(top["lang"], top["share"])))
    res = pd.DataFrame({
        "geo_id": df["unit"], "geo_level": "county", "geo_name": df["name"],
        "source_category": df["lang"], "count": df["count"], "tier": "modelled",
        "source_id": SOURCE_ID, "year": "2008-2018 (shares), 2022 (population)",
        "note": [f"share {s:.5f}; {k} respondents" for s, k in zip(df["share"], df["n"])]})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT, index=False)
    print(f"wrote {OUT} ({len(res)} rows, {res['source_category'].nunique()} answers)")


if __name__ == "__main__":
    main()
