"""Gambia: 2013 census ethnic group by LGA, read as language, moved by Afrobarometer R7 retention.

    python sources/wafr_afro.py gm     the respondents -> data/raw/gm/gm_afro.csv (read-only .sav)
    python sources/gm_census.py        -> data/normalized/gm.csv (LGA x language, counts)

THE CENSUS. 2013 Population and Housing Census, "Spatial Distribution of Population and
Socio-Cultural Characteristics" (GBoS; religiondots' copy, read-only), Annex G: population by
ethnicity and age for each of the eight LGAs, in counts (Tables G.10 ... G.61, the Both-sexes
table of each LGA). Columns: total population, non-Gambians, nationality not stated, then ten
ethnic groups (ethnicity was asked of Gambians only). The Total row of each is read off the
page and asserted: the parts sum to the total, the total is religiondots' 2013 LGA figure,
and the shares agree with Table 3.2's percentages (one decimal).

READ AS LANGUAGE, WITH RETENTION. Afrobarometer R7 (2018) asked ethnic group and Q2A "mother
tongue" of 1,200 Gambians (the Gambia is in R7-R9 only; R8-R9 ask "language spoken in home",
which pulls towards Wolof and Mandinka, ask 018). Each census group is shared over the
mother tongues its R7 members named (national vectors, weighted; the R7 regions are the old
divisions, not the LGAs). Groups under MIN_N respondents are kept whole. English answers (21,
mostly English interviews) are moved to the respondent's own group's language first.
RETENTION_ROUNDS is the switch: () keeps every group whole; (8, 9) uses the home-language
rounds (the record shows all three).

NON-GAMBIANS (110,705) and "not stated" are drawn on their LGA's Gambian mix (as Niger's
foreign residents are), so each LGA sums to its 2013 census population.
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
PDF = RD / "data" / "raw" / "gm" / "census_2013_spatial_distribution_report.pdf"
EXTRACT = HERE / "data" / "raw" / "gm" / "gm_afro.csv"
LOOKUP = RD / "data" / "geo" / "gm" / "gm_lookup.csv"
OUT = HERE / "data" / "normalized" / "gm.csv"
SOURCE_ID = "gm_phc2013_annexG_afrobarometer_r7"

RETENTION_ROUNDS = (7,)
MIN_N = 30

# LGA -> (printed page of its Both-sexes Annex G table, 1-based PDF page)
PAGES = {"Banjul": 59, "Kanifing": 62, "Brikama": 65, "Mansakonko": 74, "Kerewan": 83,
         "Kuntaur": 92, "Janjanbureh": 101, "Basse": 110}
GROUPS = ["Mandinka/Jahanka", "Fula/Tukulor/Lorobo", "Wollof", "Jola/Karoninka", "Serahule",
          "Serere", "Creole/Aku Marabou", "Manjago", "Bambara", "Other"]
# Table 3.2, % of Gambians, same order, for the check
T32 = {"Banjul": [22.5, 20.9, 24.4, 6.8, 3.9, 11.8, 4.3, 1.1, 2.6, 1.7],
       "Kanifing": [32.2, 17.3, 14.6, 16.4, 7.5, 4.7, 1.4, 2.9, 0.9, 2.0],
       "Brikama": [39.8, 19.7, 10.7, 18.6, 2.3, 2.8, 0.3, 2.9, 0.7, 2.1],
       "Mansakonko": [55.7, 32.1, 4.7, 1.3, 3.6, 0.5, 0.1, 0.8, 0.7, 0.5],
       "Kerewan": [30.8, 21.9, 31.4, 1.0, 0.6, 7.3, 0.1, 1.0, 5.0, 1.0],
       "Kuntaur": [23.9, 41.1, 32.6, 0.3, 0.3, 0.4, 0.0, 0.1, 0.6, 0.7],
       "Janjanbureh": [24.6, 41.2, 25.4, 0.6, 6.7, 0.3, 0.0, 0.2, 0.5, 0.6],
       "Basse": [28.9, 29.8, 0.6, 0.3, 38.9, 0.1, 0.1, 0.1, 0.8, 0.4]}

# census group -> survey ethnic answers that are its members
MEMBERS = {"Mandinka/Jahanka": {"Mandinka", "Jahanka"},
           "Fula/Tukulor/Lorobo": {"Fula, Tukulor or Lorobo"}, "Wollof": {"Wolof"},
           "Jola/Karoninka": {"Jola"}, "Serahule": {"Serahuleh"}, "Serere": {"Serer"},
           "Creole/Aku Marabou": {"Aku", "Creole", "Creole/Aku Marabou"},
           "Manjago": {"Manjago"}, "Bambara": {"Bambara"}, "Other": {"Other"}}
# group kept whole -> its language
OWN = {"Mandinka/Jahanka": "Mandinka", "Fula/Tukulor/Lorobo": "Fula", "Wollof": "Wolof",
       "Jola/Karoninka": "Jola", "Serahule": "Serahuleh", "Serere": "Serer",
       "Creole/Aku Marabou": "Krio (Aku)", "Manjago": "Manjago", "Bambara": "Bambara",
       "Other": "Other"}
LANG_OF_ETH = {"Mandinka": "Mandinka", "Jahanka": "Jahanka", "Fula, Tukulor or Lorobo": "Fula",
               "Wolof": "Wolof", "Jola": "Jola", "Serahuleh": "Serahuleh", "Serer": "Serer",
               "Manjago": "Manjago"}


def say(ok, msg):
    print(("  ok  " if ok else "  FAIL") + "  " + msg)
    if not ok:
        raise SystemExit("check failed: " + msg)


def num(s):
    return int(s.strip().replace(",", ""))


def census():
    import fitz
    fitz.TOOLS.mupdf_display_errors(False)
    doc = fitz.open(PDF)
    lut = pd.read_csv(LOOKUP).set_index("unit")["census_2013"]
    rows = {}
    for lga, p in PAGES.items():
        lines = doc[p - 1].get_text().split("\n")
        say(f"({lga}" in " ".join(lines[:12]).replace("( ", "(") and "Both" in
            " ".join(lines[:12]), f"page {p} is {lga}'s Both-sexes table")
        i = [k for k, x in enumerate(lines) if x.strip() == "Total"][-1]
        v = [num(x) for x in lines[i + 1:i + 14]]
        total, nong, ns, eth = v[0], v[1], v[2], v[3:]
        say(total == nong + ns + sum(eth), f"{lga}: {total:,} = non-Gambian {nong:,} + not "
            f"stated {ns} + ten groups")
        say(total == int(lut[lga]), f"{lga}: total is the 2013 census figure")
        pct = np.array(eth) / sum(eth) * 100
        worst = float(np.max(np.abs(pct - np.array(T32[lga]))))
        say(worst <= 0.1, f"{lga}: shares agree with Table 3.2 (worst {worst:.2f} pt)")
        rows[lga] = dict(zip(GROUPS, eth), _total=total, _nong=nong)
    t = pd.DataFrame(rows).T
    print(f"  {int(t['_total'].sum()):,} people, {int(t['_nong'].sum()):,} non-Gambian")
    return t


def vectors(rounds):
    a = pd.read_csv(EXTRACT, keep_default_na=False)
    s = a[a["round"].isin(rounds)].copy()
    s["lang"] = s["lang"].replace({"Serahule": "Serahuleh"})
    s["eth"] = s["eth"].replace({"Serahule": "Serahuleh"})
    en = s["lang"] == "English"
    own = en & s["eth"].isin(LANG_OF_ETH)
    if rounds:
        print(f"  English answers in R{rounds}: {int(en.sum())} (interview "
              f"{s.loc[en, 'interview'].value_counts().to_dict()}); {int(own.sum())} moved "
              "to their group's language")
    s.loc[own, "lang"] = s.loc[own, "eth"].map(LANG_OF_ETH)
    out, used = {}, {}
    for g in GROUPS:
        sg = s[s["eth"].isin(MEMBERS[g])]
        if len(sg) < MIN_N:
            out[g] = pd.Series({OWN[g]: 1.0})
            used[g] = f"kept whole ({len(sg)} respondents)"
            continue
        v = sg.groupby("lang")["w"].sum()
        v = v[~v.index.isin(["Don't know", "Refused", "Missing"])]
        out[g] = v / v.sum()
        used[g] = (f"{len(sg)} respondents: " + ", ".join(
            f"{k} {x:.0%}" for k, x in out[g].sort_values(ascending=False).head(3).items()))
    return out, used


def draw(t, vec):
    rows = []
    for lga in t.index:
        eth = t.loc[lga, GROUPS].astype(float)
        share = pd.concat([(eth[g] / eth.sum()) * vec[g] for g in GROUPS])
        share = share.groupby(level=0).sum()
        say(abs(share.sum() - 1) < 1e-9, f"{lga}: language shares sum to 1")
        share = share[share > 0]
        n = int(t.loc[lga, "_total"])
        f = (share * n).to_numpy()
        base = np.floor(f)
        k = int(round(n - base.sum()))
        base[np.argsort(-(f - base))[:k]] += 1
        for lang, s, c in zip(share.index, share.values, base.astype(int)):
            if c > 0:
                rows.append((lga, lang, s, c))
    return pd.DataFrame(rows, columns=["unit", "lang", "share", "count"])


def main():
    global RETENTION_ROUNDS
    t = census()
    nat = {}
    for lab, r in (("no move", ()), ("R7 mother tongue (drawn)", (7,)),
                   ("R8-R9 home", (8, 9))):
        vec, used = vectors(r) if r else ({g: pd.Series({OWN[g]: 1.0}) for g in GROUPS}, {})
        if r == RETENTION_ROUNDS:
            print("\n  retention vectors:")
            for g, u in used.items():
                print(f"    {g:22s} {u}")
        d = draw(t, vec)
        nat[lab] = d.groupby("lang")["count"].sum() / d["count"].sum() * 100
    print(pd.DataFrame(nat).fillna(0).sort_values("no move", ascending=False).round(2)
          .to_string())

    vec, _ = vectors(RETENTION_ROUNDS) if RETENTION_ROUNDS else (
        {g: pd.Series({OWN[g]: 1.0}) for g in GROUPS}, {})
    df = draw(t, vec)
    total = int(t["_total"].sum())
    say(int(df["count"].sum()) == total, f"drawn total {total:,}")
    for lga in t.index:
        top = df[df["unit"] == lga].sort_values("count", ascending=False).head(4)
        print(f"  {lga:12s} " + ", ".join(f"{l} {s:.0%}" for l, s in zip(top["lang"],
                                                                          top["share"])))
    res = pd.DataFrame({
        "geo_id": df["unit"], "geo_level": "lga", "geo_name": df["unit"],
        "source_category": df["lang"], "count": df["count"], "tier": "derived",
        "source_id": SOURCE_ID, "year": 2013,
        "note": [f"share {s:.5f}" for s in df["share"]]})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT, index=False)
    print(f"wrote {OUT} ({len(res)} rows, {res['source_category'].nunique()} languages)")


if __name__ == "__main__":
    main()
