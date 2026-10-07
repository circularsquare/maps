"""Niger: RGP/H 2001 ethnic group by département, read as language, with Afrobarometer retention.

    python sources/ne_census.py --fetch   Niger's rows from religiondots' Afrobarometer .sav files
                                          (read-only) -> data/raw/ne/ne_afro.csv
    python sources/ne_census.py           -> data/normalized/ne.csv (région x language, counts)

THE CENSUS. Niger's 2012 RGP/H asked no ethnicity or language (its socio-cultural chapter is
religion and nationality; sources/ne.md). The 2001 RGP/H asked "nationalité ou ethnie" (C07):
a Nigerien was asked their ethnic group. "État et structure de la population" (2001 analysis
volume, ireda.ceped.org copy in data/raw/ne/), Tableau 30: Nigerien residents by ethnic group
and département, in counts. The eight 2001 départements are today's eight régions, same names,
same boundaries (the 2002 decentralisation renamed départements to régions). Transcribed into
T30 below; the PDF's text layer runs some cells together, and every row and column sum is
asserted against the printed totals, and every cell against Tableau 31's percentages.

READ AS LANGUAGE (AGENT_BRIEF section 2, ethnicity only). Niger's census groups are
ethnolinguistic (the report's own word). Retention: Afrobarometer asks both ethnic group and home
language. Per ethnic group and région, the language vector of that group's respondents (weighted,
shrunk towards the group's national vector with weight K) moves the census group onto the
languages its members named. Groups with fewer than MIN_N respondents (Arabe, Gourmantché,
Toubou) are kept whole on their own language.

WHICH ROUNDS (ask 018, ruled 2026-10-05: RETENTION_ROUNDS = (7,), R7's mother-tongue question
Q2A, which the extract now holds for R7; the rest of this paragraph is the earlier call).
RETENTION_ROUNDS = (5, 6): the rounds that asked "Language of
respondent", the first-language reading. R7-R9 asked "Language spoken in home", which pulls
towards Hausa, the lingua franca. The one-line switch is RETENTION_ROUNDS; () draws every
group whole on its language. sources/ne.md section 2 has the national totals under each.

POPULATION. The 2001 shares (Nigeriens who declared a group; "Autres ethnies", 5,951
naturalised or undeclared, left out) are applied to each région's 2012 census population
(religiondots' ne_lookup.csv, 17,138,707 people), as religiondots draws Niger.
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
AFB = RD / "data" / "raw" / "afrobarometer"
RAW = HERE / "data" / "raw" / "ne"
EXTRACT = RAW / "ne_afro.csv"
LOOKUP = RD / "data" / "geo" / "ne" / "ne_lookup.csv"
OUT = HERE / "data" / "normalized" / "ne.csv"
SOURCE_ID = "rgph2001_t30_afrobarometer_r5_r6_niger"

# Anita's ruling on ask 018 (2026-10-05): retention, and with it Hausa's pull, from R7's
# mother-tongue question (Q2A, R7's "lang" in the extract since then). Before: (5, 6).
RETENTION_ROUNDS = (7,)     # () = no retention; (5, 6) = "language of respondent" rounds
K = 10.0
MIN_N = 30

REGIONS = ["Agadez", "Diffa", "Dosso", "Maradi", "Tahoua", "Tillaberi", "Zinder", "Niamey"]
# Tableau 30, RGP/H 2001, Nigerien residents by ethnic group and département, counts.
T30 = {
    "Arabe":          [6663, 8227, 799, 1153, 14118, 2778, 3377, 2970],
    "Djerma-Sonrai":  [15938, 3067, 721287, 9487, 16708, 1191229, 10114, 333044],
    "Haoussa":        [77969, 15246, 631711, 1960395, 1539279, 196236, 1424714, 224181],
    "Gourma":         [111, 79, 937, 149, 165, 35051, 532, 2773],
    "Kanouri-Manga":  [14963, 205268, 1703, 5413, 3610, 1794, 271167, 9198],
    "Peulh":          [7009, 83833, 128847, 185251, 49459, 236646, 195793, 48679],
    "Touareg":        [192058, 3326, 14719, 69352, 344887, 208492, 155101, 28948],
    "Toubou":         [4295, 21066, 182, 460, 147, 298, 14909, 815],
    "Autres ethnies": [504, 717, 235, 324, 304, 1420, 704, 1743],
}
T30_ROW_TOTAL = {"Arabe": 40085, "Djerma-Sonrai": 2300874, "Haoussa": 6069731, "Gourma": 39797,
                 "Kanouri-Manga": 513116, "Peulh": 935517, "Touareg": 1016883, "Toubou": 42172,
                 "Autres ethnies": 5951}
T30_COL_TOTAL = [319510, 340829, 1500420, 2231984, 1968677, 1873944, 2076411, 652351]
T30_TOTAL = 10_964_126
# Tableau 31, % of each département, for the cell check (one decimal; "Total" column left out)
T31 = {
    "Arabe": [2.1, 2.4, 0.1, 0.1, 0.7, 0.1, 0.2, 0.5],
    "Djerma-Sonrai": [5.0, 0.9, 48.1, 0.4, 0.8, 63.6, 0.5, 51.1],
    "Haoussa": [24.4, 4.5, 42.1, 87.8, 78.2, 10.5, 68.6, 34.4],
    "Gourma": [0.0, 0.0, 0.1, 0.0, 0.0, 1.9, 0.0, 0.4],
    "Kanouri-Manga": [4.7, 60.2, 0.1, 0.2, 0.2, 0.1, 13.1, 1.4],
    "Peulh": [2.2, 24.6, 8.6, 8.3, 2.5, 12.6, 9.4, 7.5],
    "Touareg": [60.1, 1.0, 1.0, 3.1, 17.5, 11.1, 7.5, 4.4],
    "Toubou": [1.3, 6.2, 0.0, 0.0, 0.0, 0.0, 0.7, 0.1],
    "Autres ethnies": [0.2, 0.2, 0.0, 0.0, 0.0, 0.1, 0.0, 0.3],
}
LEFT_OUT = "Autres ethnies"

# census group -> its own language (the answer it is drawn as when kept whole)
OWN = {"Haoussa": "Hausa", "Djerma-Sonrai": "Zarma", "Peulh": "Fulfulde",
       "Touareg": "Tamasheq", "Kanouri-Manga": "Kanuri", "Toubou": "Tubu", "Arabe": "Arabic",
       "Gourma": "Gourmanchéma"}

# Afrobarometer labels (ethnic group and language cards share them) -> one spelling
AFRO = {"Haoussa": "Hausa", "Zarrma/Songhaï": "Zarma", "Zarma/Songhaï": "Zarma",
        "Zarma/Songhay": "Zarma", "Zrama/Songhay": "Zarma", "Fulfuldé": "Fulfulde",
        "Fulfulde": "Fulfulde", "Fulfudé": "Fulfulde", "Peulh": "Fulfulde",
        "Touareg": "Tamasheq", "Tamasheq": "Tamasheq", "Béri béri": "Kanuri",
        "Béri béri (Kanouri)": "Kanuri", "Kanuri": "Kanuri", "Kanouri": "Kanuri",
        "Arabe": "Arabic", "Toubou": "Tubu", "French": "French",
        "Goumantchéma": "Gourmanchéma", "Gourmantchéma": "Gourmanchéma",
        "Gourmatché": "Gourmanchéma"}
ETH_OF = {v: k for k, v in OWN.items()}   # survey ethnic answer (one spelling) -> census group

# (round, file, home language, ethnic group, interview language, weight)
ROUNDS = [
    (5, "merged-round-5-data-34-countries-2011-2013-last-update-july-2015_0.sav",
     "Q2", "Q84", "Q103", "withinwt"),
    (6, "merged_r6_data_2016_36countries2.sav", "Q2", "Q87", "Q103", "withinwt"),
    # R7: Q2A "Respondent's mother tongue", not Q2B "Language spoken in home" (Anita's ruling
    # on ask 018, 2026-10-05)
    (7, "r7_merged_data_34ctry.release.sav", "Q2A", "Q84", "Q103", "withinwt"),
    (8, "afrobarometer_release-dataset_merge-34ctry_r8_en_2023-03-01.sav",
     "Q2", "Q81", "Q103", "withinwt_hh"),
    (9, "R9.Merge_39ctry.20Nov23.final_.release_Updated.4Jun25-3.sav",
     "Q2", "Q84A", "Q102", "withinwt_hh"),
]
N_RESP = {5: 1199, 6: 1200, 7: 1200, 8: 1199, 9: 1200}

# Arabic by place (spec section 3, a label whose meaning depends on place): the Arabs of Diffa
# and Zinder are Shuwa (Chadian Arabic, chad1249) speakers, the Mohamid and kin who came from
# Chad; those of Tahoua, Agadez and the west are Azawagh and Kounta Arabs speaking Hassaniya
# (hass1238).
SHUWA_REGIONS = {"Diffa", "Zinder"}


def say(ok, msg):
    print(("  ok  " if ok else "  FAIL") + "  " + msg)
    if not ok:
        raise SystemExit("check failed: " + msg)


def region_key(s):
    s = str(s).strip().upper().replace("É", "E")
    return {r.upper(): r for r in REGIONS}[s]


# ---------------------------------------------------------------- extract

def fetch():
    import pyreadstat

    def rd(p, **kw):
        try:
            return pyreadstat.read_sav(str(p), **kw)
        except Exception:  # noqa: BLE001  R6 is not valid UTF-8
            return pyreadstat.read_sav(str(p), encoding="LATIN1", **kw)

    out = []
    for rnd, f, q, e, il, w in ROUNDS:
        _, meta = rd(AFB / f, metadataonly=True)
        lab = str(meta.column_names_to_labels.get(q, "")).casefold()
        say("language" in lab or "mother tongue" in lab,
            f"R{rnd} {q} is the language question ({lab!r})")
        elab = str(meta.column_names_to_labels.get(e, "")).casefold()
        say("ethnic" in elab or "tribe" in elab, f"R{rnd} {e} is the ethnic group ({elab!r})")
        df, _ = rd(AFB / f, usecols=["COUNTRY", "REGION", q, e, il, w], apply_value_formats=True)
        s = df[df["COUNTRY"].astype(str).str.strip().str.casefold() == "niger"]
        say(len(s) == N_RESP[rnd], f"R{rnd}: {len(s):,} Nigerien respondents")
        out.append(pd.DataFrame({
            "round": rnd, "question": lab, "region": s["REGION"].astype(str),
            "lang": s[q].astype(str), "eth": s[e].astype(str), "interview": s[il].astype(str),
            "w": pd.to_numeric(s[w], errors="coerce")}))
    a = pd.concat(out, ignore_index=True)
    RAW.mkdir(parents=True, exist_ok=True)
    a.to_csv(EXTRACT, index=False)
    print(f"wrote {EXTRACT} ({len(a):,} respondents)")


# ---------------------------------------------------------------- census

def census():
    t = pd.DataFrame(T30, index=REGIONS).T
    for g, row in t.iterrows():
        say(int(row.sum()) == T30_ROW_TOTAL[g], f"T30 {g}: row sums to {T30_ROW_TOTAL[g]:,}")
    cols = t.sum(axis=0)
    say(list(cols.astype(int)) == T30_COL_TOTAL, "T30: every département column sums to its total")
    say(int(t.values.sum()) == T30_TOTAL, f"T30: {T30_TOTAL:,} Nigerien residents")
    pct = t.div(cols, axis=1) * 100
    worst = max(abs(pct.loc[g, r] - T31[g][i]) for g in t.index for i, r in enumerate(REGIONS))
    say(worst <= 0.1 + 1e-9, f"T30 cells agree with Tableau 31's percentages (worst {worst:.3f} pt)")
    return t.drop(index=LEFT_OUT)


# ---------------------------------------------------------------- survey

def survey():
    a = pd.read_csv(EXTRACT, keep_default_na=False)
    say(len(a) == sum(N_RESP.values()), f"{len(a):,} respondents in the extract")
    a["region"] = a["region"].map(region_key)
    a["lang"] = a["lang"].map(lambda x: AFRO.get(x.strip(), "?"))
    a["eth"] = a["eth"].map(lambda x: AFRO.get(x.strip(), "?"))
    a["w"] = a["w"].astype(float)
    a["w"] = a["w"] * a.groupby("round")["w"].transform(lambda s: len(s) / s.sum())
    return a


def vectors(a, groups):
    """(group, région) -> Series of language shares, from RETENTION_ROUNDS."""
    s = a[a["round"].isin(RETENTION_ROUNDS) & (a["lang"] != "?") & (a["eth"] != "?")].copy()
    s["group"] = s["eth"].map(ETH_OF)
    langs = sorted(set(s["lang"]) | set(OWN.values()))
    out, used = {}, {}
    for g in groups:
        own = pd.Series(0.0, index=langs)
        own[OWN[g]] = 1.0
        sg = s[s["group"] == g]
        if len(sg) < MIN_N:
            for r in REGIONS:
                out[(g, r)] = own
            used[g] = f"kept whole ({len(sg)} respondents)"
            continue
        nat = sg.groupby("lang")["w"].sum().reindex(langs, fill_value=0.0)
        nat = nat / nat.sum()
        used[g] = f"{len(sg)} respondents, {nat[OWN[g]]:.1%} answered {OWN[g]}"
        for r in REGIONS:
            loc = sg[sg["region"] == r].groupby("lang")["w"].sum().reindex(langs, fill_value=0.0)
            out[(g, r)] = (loc + K * nat) / (loc.sum() + K)
    return out, used


def compare(a, cen):
    """Survey ethnic group by région (all rounds) against the 2001 census, % of région."""
    cs = cen.div(cen.sum(axis=0), axis=1) * 100
    print("\n  ethnic group, % of région: census 2001 / Afrobarometer R5-R9 pooled (n)")
    ok = a[a["eth"] != "?"]
    for r in REGIONS:
        g = ok[ok["region"] == r]
        sv = g.groupby("eth")["w"].sum() / g["w"].sum() * 100
        cells = [f"{OWN[k][:8]} {cs.loc[k, r]:4.1f}/{sv.get(OWN[k], 0):4.1f}"
                 for k in ("Touareg", "Peulh", "Kanouri-Manga", "Toubou", "Djerma-Sonrai")]
        print(f"    {r:9s} n={len(g):4d}  " + "  ".join(cells))


def main():
    global RETENTION_ROUNDS
    if "--rounds" in sys.argv:   # for the record's comparison; re-run without it afterwards
        v = sys.argv[sys.argv.index("--rounds") + 1]
        RETENTION_ROUNDS = tuple(int(x) for x in v.split(",") if x and x != "none")
    if "--fetch" in sys.argv:
        fetch()
    cen = census()
    a = survey()
    print("  interview language by round:",
          a.groupby("round")["interview"].agg(lambda s: s.value_counts().to_dict()).to_dict())
    compare(a, cen)
    vec, used = vectors(a, list(cen.index))
    print(f"\n  retention from rounds {RETENTION_ROUNDS}, K = {K:g}:")
    for g, u in used.items():
        print(f"    {g:15s} {u}")

    lut = pd.read_csv(LOOKUP)
    pop = lut.set_index("unit")["census_2012"].astype(int)
    say(sorted(pop.index) == sorted(REGIONS), "ne_lookup.csv: the eight régions")
    total = int(pop.sum())
    print(f"  2012 census population {total:,}")
    rows = []
    for r in REGIONS:
        share = cen[r] / cen[r].sum()
        lang = sum(share[g] * vec[(g, r)] for g in cen.index)
        say(abs(lang.sum() - 1) < 1e-9, f"{r}: language shares sum to 1")
        lang = lang[lang > 0]
        f = (lang * pop[r]).to_numpy()
        base = np.floor(f)
        k = int(round(pop[r] - base.sum()))
        base[np.argsort(-(f - base))[:k]] += 1
        for name, s, c in zip(lang.index, lang.values, base.astype(int)):
            if name == "Arabic":
                name = "Arabic (Shuwa)" if r in SHUWA_REGIONS else "Arabic (Hassaniya)"
            rows.append((r, name, s, c))
    df = pd.DataFrame(rows, columns=["unit", "lang", "share", "count"])
    say(int(df["count"].sum()) == total, f"drawn total {int(df['count'].sum()):,}")
    nat = df.groupby("lang")["count"].sum().sort_values(ascending=False)
    print("\n  national, as drawn:")
    for k_, v in nat.items():
        print(f"    {k_:22s} {v:>11,}  {v / total:7.3%}")
    df = df[df["count"] > 0]
    res = pd.DataFrame({
        "geo_id": df["unit"], "geo_level": "region", "geo_name": df["unit"],
        "source_category": df["lang"], "count": df["count"], "tier": "derived",
        "source_id": SOURCE_ID, "year": "2001 (ethnic shares), 2012 (population)",
        "note": [f"share {s:.5f}" for s in df["share"]],
    })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT, index=False)
    print(f"wrote {OUT} ({len(res)} rows, {res['source_category'].nunique()} languages)")


if __name__ == "__main__":
    main()
