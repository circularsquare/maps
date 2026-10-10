"""Togo: Afrobarometer first-language answers by region, on the 2022 census's region populations.

    python sources/wafr_afro.py tg     the respondents -> data/raw/tg/tg_afro.csv (read-only .sav)
    python sources/tg_afro.py          -> data/normalized/tg_afro.csv (unit x language, counts)

SINCE 2026-10-09 (session 32a047f0) Togo is drawn by sources/tg_mics.py: MICS6 2017's language
groups per region, split inside each group by this survey's answers (load() and shares() below).
This script's own output is kept as the comparison and no longer writes tg.csv.

NO CENSUS TABLE. Togo's 2010 RGPH-4 asked ethnicity (IPUMS ETHNICTG; IPUMS account blocked)
and INSEED published no ethnic or language table from it or from RGPH-5 (2022; the coverage
sweep listed INSEED's published booklets). So this is AGENT_BRIEF section 2's survey case:
survey shares x a population base, rows `modelled` (as sources/bq.md).

THE SURVEY. Afrobarometer R5 (2012), R6 (2014), R7 (2017), about 1,200 adults each, every
region: R5-R6 "language of respondent", R7 "respondent's mother tongue" (Q2A) -- the
first-language reading of ask 018. R8-R9 ask "language spoken in home" and are shown for
comparison only (FIRST_ROUNDS is the one-line switch). Weighted shares per region, pooled.

FRENCH. A French answer is moved to the language of the respondent's own ethnic group
(Nigeria's rule, sources/ng.md): French is almost nobody's first language in Togo, and the
answers come from French-language interviews (printed below). MOVE_FRENCH switches it off.

UNITS. religiondots' six: the five regions, Maritime without Lomé, and Lomé (Golfe 1 to 5),
with 2022 census populations (tg_lookup.csv). The survey's own "Lomé commune" is that unit;
its "Maritime" is the rest of the region.
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
EXTRACT = HERE / "data" / "raw" / "tg" / "tg_afro.csv"
LOOKUP = RD / "data" / "geo" / "tg" / "tg_lookup.csv"
OUT = HERE / "data" / "normalized" / "tg_afro.csv"   # was tg.csv until 2026-10-09 (tg_mics.py)
SOURCE_ID = "afrobarometer_r5_r7_togo"

FIRST_ROUNDS = (5, 6, 7)     # (8, 9) = "language spoken in home"
MOVE_FRENCH = True
FRENCH_AT_R7 = True          # ask 018 ruling, 2026-10-05: French at R7's Q2A share (french_at_r7)

REGION = {"maritime": "TG03", "lomecommune": "TG0305", "lome": "TG0305", "plateaux": "TG04",
          "kara": "TG02", "centrale": "TG01", "savanes": "TG05", "savane": "TG05"}

# survey answer (language card, and ethnic card for the French move) -> one spelling
LANG = {
    "Ewé": "Ewe", "Ewe": "Ewe", "Mina (Guen)": "Mina (Gen)", "Mina, Guen": "Mina (Gen)",
    "Ouatchi": "Ouatchi (Waci)", "Adja": "Aja", "Fon": "Fon",
    "Kabyè": "Kabiyè", "Kabye": "Kabiyè", "Ben (Moba)": "Moba", "Ben, Moba": "Moba",
    "Tem (Kotokoli)": "Tem (Kotokoli)", "Tem, Kotokoli": "Tem (Kotokoli)",
    "Nawdem (Losso)": "Nawdm (Losso)", "Nawdem, Losso": "Nawdm (Losso)",
    "Lama (Lamba)": "Lama (Lamba)", "Lamba": "Lama (Lamba)", "Lama, Lamba": "Lama (Lamba)",
    "Ikposso (Akposso)": "Ikposo (Akposso)", "Akposso": "Ikposo (Akposso)",
    "Ikposso, Akposso": "Ikposo (Akposso)",
    "Gourma": "Gourmanchéma", "Konkomba": "Konkomba",
    "Ife (Ana)": "Ifè (Ana)", "Ifè (Ana)": "Ifè (Ana)", "Ana": "Ifè (Ana)", "Ifè, Ana": "Ifè (Ana)",
    "Akébou": "Akebu", "Akebou": "Akebu",
    "Bassar": "Ntcham (Bassar)", "N’tcha (Bassar)": "Ntcham (Bassar)",
    "N'tcha (Bassar)": "Ntcham (Bassar)", "N’Tcha (Bassar)": "Ntcham (Bassar)",
    "N'Tcha (Bassar)": "Ntcham (Bassar)", "N’Tcha, Bassar": "Ntcham (Bassar)",
    "Ngam-Gam": "Ngangam (Gangam)", "Ngam-gam": "Ngangam (Gangam)",
    "Tchamba": "Tchamba (Akaselem)",
    "Tchokossi (Anoufom)": "Anufo (Tchokossi)", "Tchpkossi (Anoufom)": "Anufo (Tchokossi)",
    "Tchokossi, Anoufom": "Anufo (Tchokossi)",
    "Haoussa": "Hausa", "Peulh": "Fulfulde", "Yoruba": "Yoruba", "French": "French",
    # one respondent in R7; no language of that name found in Glottolog
    "Aklobo": "Other", "Other": "Other", "Others": "Other",
}
DROP = {"Don't know", "Refused", "Missing", "nan", ""}


def say(ok, msg):
    print(("  ok  " if ok else "  FAIL") + "  " + msg)
    if not ok:
        raise SystemExit("check failed: " + msg)


def load():
    a = pd.read_csv(EXTRACT, keep_default_na=False)
    a["unit"] = a["region"].map(lambda s: REGION.get(
        "".join(c for c in s.casefold().replace("é", "e").replace("é", "e") if c.isalpha())))
    say(a["unit"].notna().all(), f"every respondent's region is a unit "
        f"{sorted(a.loc[a['unit'].isna(), 'region'].unique())}")
    a = a[~a["lang"].isin(DROP)].copy()
    miss = sorted(set(a["lang"]) - set(LANG))
    say(not miss, f"every answer has a spelling {miss}")
    a["l"] = a["lang"].map(LANG)
    a["e"] = a["eth"].map(LANG)   # ethnic card shares the language spellings
    return a


def shares(a, rounds, move_french):
    s = a[a["round"].isin(rounds)].copy()
    if move_french:
        fr = s["l"] == "French"
        own = fr & s["e"].notna() & ~s["e"].isin(["French", "Other"])
        print(f"  French answers in rounds {rounds}: {int(fr.sum())}, interview language "
              f"{s.loc[fr, 'interview'].value_counts().to_dict()}; {int(own.sum())} moved to "
              "their ethnic group's language")
        s.loc[own, "l"] = s.loc[own, "e"]
    sh = s.groupby(["unit", "l"])["w"].sum()
    return sh / sh.groupby(level=0).transform("sum"), s.groupby("unit").size()


def french_at_r7(sh):
    """Ask 018 (Anita, 2026-10-05): French at R7's mother-tongue question (Q2A), each region's
    share shrunk to the national one by wafr_afro.K_SHRINK respondents; the region's other
    answers (French answers already moved to the ethnic language) scaled to what is left."""
    sys.path.insert(0, str(HERE / "sources"))
    from wafr_afro import r7_mother, shrink
    t = r7_mother("Togo", {"French": ["French"]})
    key = lambda s: "".join(c for c in s.casefold().replace("é", "e") if c.isalpha())
    reg = {r: REGION.get(key(r)) for r in t.index if r != "_national"}
    say(None not in reg.values() and len(set(reg.values())) == len(reg),
        f"R7's REGION labels are one unit each {reg}")
    fr = shrink(t, "French").rename(reg)
    print(f"  R7 Q2A French: {t.loc['_national', 'French_A'] / t.loc['_national', 'n']:.2%} "
          "nationally; drawn " + ", ".join(f"{u} {v:.1%}" for u, v in fr.items()))
    out = []
    for u, s in sh.groupby(level=0):
        s = s.droplevel(0).drop("French", errors="ignore")
        s = s / s.sum() * (1 - fr[u])
        s["French"] = fr[u]
        out.append(pd.concat({u: s}))
    return pd.concat(out)


def main():
    global FIRST_ROUNDS, MOVE_FRENCH
    a = load()
    lut = pd.read_csv(LOOKUP, dtype={"unit": str})
    pop = lut.set_index("unit")["pop"].astype(int)
    names = lut.set_index("unit")["name"]
    say(set(pop.index) == set(a["unit"]), "six units, as religiondots draws Togo")
    total = int(pop.sum())
    say(total == 8_095_498, f"2022 census population {total:,}")

    # comparisons for the record
    natl = {}
    for lab, r, mv in (("R5-R7 first language", (5, 6, 7), True),
                       ("R5-R7, French as given", (5, 6, 7), False),
                       ("R8-R9 home language", (8, 9), False)):
        sh, _ = shares(a, r, mv)
        natl[lab] = (sh * pop.reindex(sh.index.get_level_values(0)).values).groupby(
            level=1).sum() / total * 100
    print(pd.DataFrame(natl).fillna(0).sort_values(list(natl)[0], ascending=False)
          .round(2).head(25).to_string())

    sh, n = shares(a, FIRST_ROUNDS, MOVE_FRENCH)
    if FRENCH_AT_R7:
        sh = french_at_r7(sh)
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
    say(int(df["count"].sum()) == total, f"drawn total {int(df['count'].sum()):,}")
    for u in pop.index:
        top = df[df["unit"] == u].sort_values("count", ascending=False).head(5)
        print(f"  {names[u]:22s} n={int(n[u]):4d}  " + ", ".join(
            f"{l} {s:.0%}" for l, s in zip(top["lang"], top["share"])))
    res = pd.DataFrame({
        "geo_id": df["unit"], "geo_level": "region", "geo_name": df["name"],
        "source_category": df["lang"], "count": df["count"], "tier": "modelled",
        "source_id": SOURCE_ID, "year": "2012-2017 (shares), 2022 (population)",
        "note": [f"share {s:.5f}; {k} respondents" for s, k in zip(df["share"], df["n"])]})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT, index=False)
    print(f"wrote {OUT} ({len(res)} rows, {res['source_category'].nunique()} answers)")


if __name__ == "__main__":
    main()
