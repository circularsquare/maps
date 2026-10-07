"""Kenya: home language from five pooled Afrobarometer rounds (2008-2022), by county.

    python sources/ke_afro.py --fetch   extract Kenya's rows from religiondots' merged .sav
                                        files (read-only) -> data/raw/ke/ab_ke_language.csv
    python sources/ke_afro.py           -> data/normalized/ke.csv   (county x answer, counts)

WHY A SURVEY. Kenya's censuses ask ethnicity, never language, and since 1989 KNBS publishes
ethnicity for the nation only (2019 KPHC Volume IV, Table 2.31; no county table exists). So
the ethnicity route would give one unit; the Afrobarometer gives all 47 counties. This is the
survey route of AGENT_BRIEF §2: shares from the survey times a population base, every row
`modelled`. The census ethnicity table is used as a check (census_check below). The record is
sources/ke.md.

ROUNDS. R4 (2008, district labels, joined to counties by DISTRICT_COUNTY), R6 (2014,
LOCATION.LEVEL.1 = county), R7 (2016, LOCATION.LEVEL.1 = county), R8 (2019, REGION = county),
R9 (2021, REGION = county). R5 (2011) carries only the eight old provinces and is not used.
Every round's county is checked against its province (R4, R6, R7, whose REGION is the
province) so a shifted location column, Nigeria R7's trap, would fail here.

THE QUESTION CHANGED, AND SWAHILI WITH IT. R4-R6 ask "Which language is your home language?";
R7-R9 "Language spoken in home". Swahili was on the card throughout: 1.4% in R4, 3.6% in R6
(2014), then 28.5%, 27.8%, 27.1% in R7-R9 (2016-2021). Nothing on the ground moved eightfold
in two years; the wording did. Under the R4-R6 wording, 97-98% of Kikuyu, Luo and Kalenjin
respondents name their group's language; under R7-R9's, 64-79%. The R4-R6 question reads as
a first language, the R7-R9 one as the language used in the household, which in a mixed or
urban household is often Swahili. This is a first-language map, so:

  * Swahili's and English's share in each county comes from R7's mother-tongue question (Q2A)
    by old province (SE_SOURCE, Anita's ruling on ask 018, 2026-10-05); R4 and R6 by county
    (SE_ROUNDS) before that.
  * In R7-R9, a Swahili or English answer from a respondent who names a Kenyan ethnic group is
    drawn on that group's language (ETHNIC), as Nigeria's English was; one who names no group
    (national identity only, "other", Swahili) is left out of the shares.
  * Every other answer's share comes from all five rounds among the rest, scaled to what
    Swahili and English leave.
Flagged to the supervisor: R7-R9 put Swahili at 49-60% of Nairobi and 68-80% of Mombasa, and
for children of those households it may well be the first language. SE_ROUNDS = [7, 8, 9]
reverses the call in one line.

COMBINED LABELS. "Meru/Embu" and "Masai/Samburu" are one answer each on the card for two
languages. Each is split by the county the respondent lives in (COMBINED): Embu county ->
Embu, Meru/Tharaka-Nithi/Isiolo -> Meru; Narok/Kajiado -> Maasai, Samburu/Marsabit/Isiolo ->
Samburu; anywhere else shared by the 2019 census's national ethnic counts (Table 2.31).

POPULATION. KNBS 2019 county totals of the conventional household population (Volume IV
Table 2.30's universe, the base religiondots draws Kenya on), 47,213,282 people.
"""
import os
import sys
import unicodedata
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent.parent
RD = HERE.parent / "religiondots"
AB_DIR = RD / "data" / "raw" / "afrobarometer"
RAW = HERE / "data" / "raw" / "ke"
EXTRACT = RAW / "ab_ke_language.csv"
RD_KE = RD / "data" / "normalized" / "ke.csv"       # KNBS Vol IV Table 2.30, county totals
OUT = HERE / "data" / "normalized" / "ke.csv"

POP_2019 = 47_213_282
N_COUNTIES = 47
SOURCE_ID = "afrobarometer_r4_r9_kenya"

# (round, file, language, verbatim, weight, county column, ethnic group, its verbatim,
#  interview language)
ROUNDS = [
    (4, "merged_r4_data.sav", "Q3", "Q3OTHER", "Withinwt", "DISTRICT", "Q79", "Q79OTHER", "Q103"),
    (6, "merged_r6_data_2016_36countries2.sav", "Q2", "Q2OTHER", "withinwt", "LOCATION.LEVEL.1",
     "Q87", "Q87OTHER", "Q103"),
    (7, "r7_merged_data_34ctry.release.sav", "Q2B", "Q2BOTHER", "withinwt", "LOCATION.LEVEL.1",
     "Q84", "Q84OTHER", "Q103"),
    (8, "afrobarometer_release-dataset_merge-34ctry_r8_en_2023-03-01.sav",
     "Q2", "Q2OTHER", "withinwt_hh", "REGION", "Q81", "Q81OTHER", "Q103"),
    (9, "R9.Merge_39ctry.20Nov23.final_.release_Updated.4Jun25-3.sav",
     "Q2", "Q2OTHER", "withinwt_hh", "REGION", "Q84A", "Q84AOTHER", "Q102"),
]
N_RESP = {4: 1104, 6: 2397, 7: 1599, 8: 2400, 9: 2400}
LANG_LABEL = ("language of respondent", "language spoken in home")
ETH_LABEL = ("tribe or ethnic group", "ethnic community")

SE = ["Swahili", "English"]
SE_ROUNDS = [4, 6]   # used only for the split-half and printouts since 2026-10-05; see SE_SOURCE
# Anita's ruling on ask 018 (2026-10-05): lingua francas are drawn at R7's separate mother-
# tongue question (Q2A). Swahili and English shares come from it, by old province (8 units,
# 80-395 respondents; 31 Swahili and 7 English answers, too few for 47 counties). "r46" puts
# back the R4/R6 county shares.
SE_SOURCE = "r7_mother"
MOTHER = RAW / "ab_ke_r7_mother.csv"
NON_ANSWERS = {"Missing", "Don't know", "Refused", "Kenyan"}
OTHER_KE = "Other African language"

# The card's labels -> the answer they are counted as (spellings across rounds merged).
CODED = {
    "Kiswahili": "Swahili",
    "MijiKenda": "Mijikenda", "Giriama": "Mijikenda",   # R8 printed Giriama beside Mijikenda
    "Meru / Embu": "Meru/Embu",
    "Maasai/Samburu": "Masai/Samburu", "Maasai / Samburu": "Masai/Samburu",
    "Rendile": "Rendille",
    "Gabra": "Borana",                                   # a Borana dialect (Glottolog gabr1255)
    "Oroma": "Orma",
}
READ_VERBATIM = {"Other", "Others"}

# Every free-text answer (upper-cased) -> the answer it is counted as.
# Clusters: the card's Luhya, Kalenjin and Mijikenda are each one answer for a cluster of
# varieties; a variety named in free text is counted in its cluster, so the handful who named
# Maragoli or Digo are not drawn apart from the hundreds who answered Luhya or Mijikenda.
VERBATIM = {
    # Luhya varieties
    "BUKUSU": "Luhya", "BUNYORE": "Luhya", "KIMARAGOLI": "Luhya", "MARAGOLI": "Luhya",
    "LUGOLI": "Luhya", "KIDAHO": "Luhya", "KIWANGA": "Luhya", "SAMIA": "Luhya",
    "MARACHI": "Luhya", "TIRIKI": "Luhya",
    # Mijikenda varieties
    "DURUMA": "Mijikenda", "KIDURUMA": "Mijikenda", "DIGO": "Mijikenda", "KIDIGO": "Mijikenda",
    "KIDIGU": "Mijikenda", "WADIGO": "Mijikenda", "GIRIAMA": "Mijikenda",
    "KIGIRIAMA": "Mijikenda", "KAUMA": "Mijikenda", "KIKAUMA": "Mijikenda",
    "KICHONI": "Mijikenda",
    # Kalenjin varieties
    "NANDI": "Kalenjin", "KIPSIGIS": "Kalenjin", "TURGEN": "Kalenjin",   # Tugen
    # languages of their own
    "SABOAT": "Sabaot", "SABAOT": "Sabaot", "SABOTI": "Sabaot",
    "OGIEK": "Okiek",
    "TESO": "Teso", "KURIA": "Kuria", "SUBA": "Suba", "POKOMO": "Pokomo", "KISII": "Kisii",
    "RENDILE": "Rendille", "RENDILLE": "Rendille",
    "BORAN": "Borana", "BORANA": "Borana", "GABRA": "Borana",
    "WARDEI": "Orma",                                    # the census files Wardei under Orma
    "AJURAN": "Somali",                                  # a Somali clan
    "BAJUNI": "Bajuni", "BAJUN": "Bajuni",
    "ARABIC": "Arabic", "ARAB": "Arabic",
    "GUJARATI": "Gujarati", "GUJRATHI": "Gujarati", "GUJARAT": "Gujarati",
    "SHENG": "Sheng", "SHENG'": "Sheng",
    "KIRINYAGA": "Kikuyu",                               # a Kikuyu county
    "HALF TAITA, HALF ZAI": "Taita",
    "NUBI": "Nubi", "NUBIAN": "Nubi",
    "DINKA": "Dinka",
    "PUNJABI": OTHER_KE,                                 # replaced below by single_verbatims
    # South Asian, unspecified: no narrower node holds every language the answer can mean
    "INDIAN": "Other language", "HINDU": "Other language", "ASIAN SOUTH": "Other language",
    # one-off, unidentifiable, or a group whose language is unclear
    "BOKOM": OTHER_KE, "SHELSHEL": OTHER_KE, "MUNYAYA": OTHER_KE, "WATTA": OTHER_KE,
    "MALAKOTE": OTHER_KE, "KISAGALLA": OTHER_KE, "NYARWANDA": OTHER_KE, "NYASA": OTHER_KE,
    "KENYAN": "Kenyan",
}
VERBATIM["PUNJABI"] = "Other language"

# A respondent's ethnic group (the card's label) -> the language drawn for a Swahili or English
# answer in R7-R9. Groups with no language of their own, or none given, are not here.
ETHNIC = {
    "Kikuyu": "Kikuyu", "Luhya": "Luhya", "Luo": "Luo", "Kamba": "Kamba",
    "Kalenjin": "Kalenjin", "Kisii": "Kisii", "Somali": "Somali", "Meru/Embu": "Meru/Embu",
    "Turkana": "Turkana", "MijiKenda": "Mijikenda", "Mijikenda": "Mijikenda",
    "Giriama": "Mijikenda", "Masai/Samburu": "Masai/Samburu", "Pokot": "Pokot",
    "Taita": "Taita", "Teso": "Teso", "Kuria": "Kuria", "Borana": "Borana", "Orma": "Orma",
    "Rendille": "Rendille", "Sabaot": "Sabaot",
}

# Census 2019, Table 2.31 (national), the ethnic groups the combined labels name.
CENSUS_ETH = {"Meru": 1_975_869, "Embu": 404_801, "Maasai": 1_189_522, "Samburu": 333_471}
COMBINED = {
    "Meru/Embu": ({"EMBU": "Embu", "MERU": "Meru", "THARAKA-NITHI": "Meru", "ISIOLO": "Meru"},
                  ["Meru", "Embu"]),
    "Masai/Samburu": ({"NAROK": "Maasai", "KAJIADO": "Maasai", "SAMBURU": "Samburu",
                       "MARSABIT": "Samburu", "ISIOLO": "Samburu"}, ["Maasai", "Samburu"]),
}

# The 47 counties (religiondots' names) and their old province, for the location check.
PROVINCE = {
    "Coast": ["MOMBASA", "KWALE", "KILIFI", "TANA RIVER", "LAMU", "TAITA/TAVETA"],
    "North Eastern": ["GARISSA", "WAJIR", "MANDERA"],
    "Eastern": ["MARSABIT", "ISIOLO", "MERU", "THARAKA-NITHI", "EMBU", "KITUI", "MACHAKOS",
                "MAKUENI"],
    "Central": ["NYANDARUA", "NYERI", "KIRINYAGA", "MURANG'A", "KIAMBU"],
    "Rift Valley": ["TURKANA", "WEST POKOT", "SAMBURU", "TRANS NZOIA", "UASIN GISHU",
                    "ELGEYO/MARAKWET", "NANDI", "BARINGO", "LAIKIPIA", "NAKURU", "NAROK",
                    "KAJIADO", "KERICHO", "BOMET"],
    "Western": ["KAKAMEGA", "VIHIGA", "BUNGOMA", "BUSIA"],
    "Nyanza": ["SIAYA", "KISUMU", "HOMA BAY", "MIGORI", "KISII", "NYAMIRA"],
    "Nairobi": ["NAIROBI"],
}
COUNTY_PROV = {c: p for p, cs in PROVINCE.items() for c in cs}

# R4's 1999-2009 districts -> the 2010 county that holds each (every one lies in one county).
DISTRICT_COUNTY = {
    "maragua": "MURANG'A", "thika": "KIAMBU", "malindi": "KILIFI",
    "marsabit north": "MARSABIT", "meru central": "MERU", "meru north": "MERU",
    "mwingi": "KITUI", "nithi (meru south)": "THARAKA-NITHI", "bondo": "SIAYA",
    "gucha (south kisii)": "KISII", "kisii central": "KISII",
    "north kisii (nyamira": "NYAMIRA", "nyando": "KISUMU", "rachuonyo": "HOMA BAY",
    "suba": "HOMA BAY", "buret": "KERICHO", "koibatek": "BARINGO",
    "marakwet": "ELGEYO/MARAKWET", "trans mara": "NAROK", "butere mumias": "KAKAMEGA",
    "butere/mumias": "KAKAMEGA", "lugari": "KAKAMEGA", "mt elgon": "BUNGOMA",
    "teso": "BUSIA", "taita taveta": "TAITA/TAVETA",
}


def say(ok, msg):
    print(("  ok  " if ok else "  FAIL") + "  " + msg)
    if not ok:
        raise SystemExit("check failed: " + msg)


def ckey(s):
    s = unicodedata.normalize("NFKC", str(s)).replace("’", "'")
    return " ".join(s.split()).strip().casefold()


def county_of(rnd, label):
    """A round's location label -> one of religiondots' 47 county names, or None."""
    k = ckey(label)
    if rnd == 4 and k in DISTRICT_COUNTY:
        return DISTRICT_COUNTY[k]
    k = k.replace("-", " ").replace("/", " ")
    k = " ".join(k.split())
    alias = {"nairobi city": "nairobi", "taita taveta": "taita/taveta",
             "tharaka nithi": "tharaka-nithi", "elgeyo marakwet": "elgeyo/marakwet"}
    k = alias.get(k, k)
    for c in COUNTY_PROV:
        if c.casefold() == k:
            return c
    return None


# ---------------------------------------------------------------- extract

def fetch():
    import pyreadstat
    out = []
    for rnd, name, q, qo, wt, loc, eth, etho, il in ROUNDS:
        p = AB_DIR / name
        if not p.exists():
            raise SystemExit(f"{p} missing: religiondots keeps the merged rounds")
        try:
            _, meta = pyreadstat.read_sav(str(p), metadataonly=True)
            enc = {}
        except Exception:  # noqa: BLE001  R6 is not valid UTF-8
            _, meta = pyreadstat.read_sav(str(p), metadataonly=True, encoding="LATIN1")
            enc = {"encoding": "LATIN1"}
        up = {c.upper(): c for c in meta.column_names}
        want = list(dict.fromkeys(["COUNTRY", "REGION", "RESPNO", "URBRUR", q, qo, wt, loc,
                                   eth, etho, il]))
        lab = str(meta.column_names_to_labels.get(up[q.upper()], "")).casefold()
        say(any(t in lab for t in LANG_LABEL), f"R{rnd} {q} is the home-language question "
            f"({lab!r})")
        elab = str(meta.column_names_to_labels.get(up[eth.upper()], "")).casefold()
        say(any(t in elab for t in ETH_LABEL), f"R{rnd} {eth} is the ethnic group ({elab!r})")
        df, meta = pyreadstat.read_sav(str(p), usecols=[up[c.upper()] for c in want], **enc)
        c = {k: up[k.upper()] for k in want}
        vl = meta.variable_value_labels

        def lab_of(k):
            m = vl.get(c[k], {})
            return sub[c[k]].map(m) if m else sub[c[k]]
        sub = df[df[c["COUNTRY"]].map(vl.get(c["COUNTRY"], {})).astype(str).str.strip()
                 .str.casefold() == "kenya"]
        say(len(sub) == N_RESP[rnd], f"R{rnd}: {len(sub):,} Kenyan respondents")
        w = pd.to_numeric(sub[c[wt]], errors="coerce")
        say(0.98 <= w.sum() / len(sub) <= 1.02, f"R{rnd} {wt} is a within-country weight "
            f"(mean {w.sum() / len(sub):.3f})")
        o = pd.DataFrame({
            "round": rnd, "respno": sub[c["RESPNO"]].astype(str),
            "region": lab_of("REGION"), "loc": lab_of(loc), "urb": lab_of("URBRUR"),
            "lang": lab_of(q), "verbatim": sub[c[qo]].astype(str).str.strip(),
            "eth": lab_of(eth), "eth_verbatim": sub[c[etho]].astype(str).str.strip(),
            "intlang": lab_of(il), "w": w,
        })
        say(o["lang"].notna().all(), f"R{rnd}: every answer code has a label")
        out.append(o)
    a = pd.concat(out, ignore_index=True)
    RAW.mkdir(parents=True, exist_ok=True)
    a.to_csv(EXTRACT, index=False)
    print(f"wrote {EXTRACT} ({len(a):,} respondents)")


# ---------------------------------------------------------------- harmonise

def answer(row):
    lab = str(row["lang"]).strip()
    v = str(row["verbatim"]).strip()
    v = "" if v.lower() in ("", "nan", "none") else " ".join(v.split()).upper()
    if lab in READ_VERBATIM:
        if not v:
            return OTHER_KE
        if v not in VERBATIM:
            raise SystemExit(f"verbatim {v!r} (R{row['round']}, {row['loc']}) is not in "
                             "VERBATIM: decide it there")
        return VERBATIM[v]
    return CODED.get(lab, lab)


def load():
    a = pd.read_csv(EXTRACT, dtype=str, keep_default_na=False)
    a["round"] = a["round"].astype(int)
    a["w"] = a["w"].astype(float)
    say(len(a) == sum(N_RESP.values()), f"{len(a):,} respondents in the extract")
    a["county"] = [county_of(r, l) for r, l in zip(a["round"], a["loc"])]
    bad = sorted(set(a.loc[a["county"].isna(), "loc"]))
    say(not bad, f"every location label is a county ({bad})")
    # the county must lie in the province REGION names (R4, R6, R7), so a shifted column fails
    pv = a[a["round"].isin([4, 6, 7])]
    off = pv[pv["county"].map(COUNTY_PROV) != pv["region"].str.strip()]
    say(off.empty, f"R4/R6/R7: every county lies in its REGION's province "
        f"({off.groupby(['round', 'region', 'county']).size().to_dict()})")
    say(set(a["county"]) == set(COUNTY_PROV), "all 47 counties sampled")
    a["answer"] = a.apply(answer, axis=1)
    n0 = len(a)
    a = a[~a["answer"].isin(NON_ANSWERS) & ~a["lang"].isin(NON_ANSWERS)].copy()
    print(f"  {n0 - len(a)} non-answers dropped (missing, don't know, 'Kenyan')")
    return a


def se_by_ethnicity(a):
    """R7-R9: a Swahili or English answer is drawn on the respondent's ethnic group's language."""
    m = a["round"].isin([7, 8, 9]) & a["answer"].isin(SE)
    e = a.loc[m, "eth"].astype(str).str.strip()
    to = e.map(ETHNIC)
    print(f"  R7-R9 Swahili/English answers: {int(m.sum())}; drawn on their ethnic group's "
          f"language: {int(to.notna().sum())} (top {to.value_counts().head(6).to_dict()}); "
          f"no group with a language of its own, left out: {int(to.isna().sum())} "
          f"({e[to.isna()].value_counts().head(5).to_dict()})")
    a.loc[to.index[to.notna()], "answer"] = to[to.notna()]
    drop = to.index[to.isna()]
    return a.drop(index=drop)


def split_combined(a):
    """Meru/Embu and Masai/Samburu: by the respondent's county, else by the census's counts."""
    rows = []
    keep = ~a["answer"].isin(COMBINED)
    for lab, (home, parts) in COMBINED.items():
        tot = sum(CENSUS_ETH[p] for p in parts)
        sub = a[a["answer"] == lab]
        n_home = 0
        for _, r in sub.iterrows():
            if r["county"] in home:
                rr = r.copy()
                rr["answer"] = home[r["county"]]
                rows.append(rr)
                n_home += 1
                continue
            for p in parts:
                rr = r.copy()
                rr["answer"], rr["w"] = p, r["w"] * CENSUS_ETH[p] / tot
                rows.append(rr)
        print(f"  {lab}: {len(sub)} answers, {n_home} in a home county, the rest shared "
              f"{' : '.join(f'{p} {CENSUS_ETH[p] / tot:.0%}' for p in parts)}")
    return pd.concat([a[keep], pd.DataFrame(rows)], ignore_index=True)


def single_verbatims(a):
    """A language named only in free text, by one respondent in the pool: on OTHER_KE."""
    coded = {CODED.get(x, x) for x in a["lang"].unique()} | set(CODED.values())
    n = a.groupby("answer").size()
    one = [k for k, v in n.items() if v == 1 and k not in coded
           and k not in (OTHER_KE, "Other language")]
    a.loc[a["answer"].isin(one), "answer"] = OTHER_KE
    return one


def r7_mother():
    """R7 Q2A, "Respondent's mother tongue", Kenya: province x (Swahili, English) weighted
    shares among those who answered. Cached in MOTHER."""
    if not MOTHER.exists():
        import pyreadstat
        p = AB_DIR / ROUNDS[2][1]
        _, meta = pyreadstat.read_sav(str(p), metadataonly=True)
        lab = str(meta.column_names_to_labels.get("Q2A", "")).casefold()
        say("mother tongue" in lab, f"R7 Q2A is the mother-tongue question ({lab!r})")
        df, _ = pyreadstat.read_sav(str(p), usecols=["COUNTRY", "REGION", "Q2A", "withinwt"],
                                    apply_value_formats=True)
        df = df[df["COUNTRY"].astype(str).str.strip() == "Kenya"]
        say(len(df) == N_RESP[7], f"R7 mother tongue: {len(df):,} Kenyan respondents")
        df.to_csv(MOTHER, index=False)
    m = pd.read_csv(MOTHER, dtype=str, keep_default_na=False)
    m["w"] = m["withinwt"].astype(float)
    m = m[~m["Q2A"].isin(NON_ANSWERS)]
    m["answer"] = m["Q2A"].map(lambda x: CODED.get(x, x))
    say(set(m["REGION"]) == set(PROVINCE), "R7 Q2A: every province sampled")
    tot = m.groupby("REGION")["w"].sum()
    e = m[m["answer"].isin(SE)].groupby(["REGION", "answer"])["w"].sum().div(tot, level="REGION")
    print(f"  R7 mother tongue (Q2A), {len(m)} answers: Swahili "
          f"{m.loc[m['answer'] == 'Swahili', 'w'].sum() / m['w'].sum():.2%}, English "
          f"{m.loc[m['answer'] == 'English', 'w'].sum() / m['w'].sum():.2%}; by province "
          f"{e.round(3).to_dict()}")
    return e


def county_shares(a):
    """Swahili and English from R7's mother tongue by province (or SE_ROUNDS by county); every
    other answer from all rounds among the rest."""
    if SE_SOURCE == "r7_mother":
        pe = r7_mother()
        rows = {(c, k): v for (p, k), v in pe.items() for c in PROVINCE[p]}
        e = pd.Series(rows, dtype=float)
        e.index.names = ["county", "answer"]
    else:
        se = a[a["round"].isin(SE_ROUNDS)]
        tot = se.groupby("county")["w"].sum()
        e = se[se["answer"].isin(SE)].groupby(["county", "answer"])["w"].sum()
        e = e.div(tot, level="county")
    rest = a[~a["answer"].isin(SE)]
    r = rest.groupby(["county", "answer"])["w"].sum()
    r = r.div(r.groupby(level="county").sum(), level="county")
    left = 1 - e.groupby(level="county").sum().reindex(
        r.index.get_level_values(0).unique()).fillna(0)
    r = r.mul(left, level="county")
    return pd.concat([e, r]).sort_index()


def main():
    if "--fetch" in sys.argv:
        fetch()
    a = load()
    ct = pd.crosstab(a["answer"], a["round"])
    ct["all"] = ct.sum(axis=1)
    print(ct.sort_values("all", ascending=False).to_string())
    se_tab = a[a["answer"].isin(SE)].groupby(["round", "answer"])["w"].sum().unstack(fill_value=0)
    print("\n  Swahili and English, weighted share of each round:")
    print((se_tab.div(a.groupby("round")["w"].sum(), axis=0)).round(3).to_string())
    a = se_by_ethnicity(a)
    a = split_combined(a)
    one = single_verbatims(a)
    print(f"  {len(one)} languages named in free text by one respondent -> {OTHER_KE}: {one}")

    rd = pd.read_csv(RD_KE, dtype={"geo_id": str})
    rd = rd[(rd["geo_level"] == "county") & (rd["source_category"] == "Total")]
    say(len(rd) == N_COUNTIES and int(rd["count"].sum()) == POP_2019,
        f"KNBS Table 2.30 universe: {len(rd)} counties, {int(rd['count'].sum()):,} people")
    pop = dict(zip(rd["geo_name"], rd["count"].astype(float)))
    gid = dict(zip(rd["geo_name"], rd["geo_id"]))
    say(set(pop) == set(COUNTY_PROV), "the 47 county names match KNBS's, both ways")

    sh = county_shares(a)
    s = sh.groupby(level="county").sum()
    say((s - 1).abs().max() < 1e-9, "every county's shares sum to 1")
    n_c = a.groupby("county").size()
    n_se = a[a["round"].isin(SE_ROUNDS)].groupby("county").size()
    print(f"  respondents per county: min {n_c.min()} ({n_c.idxmin()}), median "
          f"{int(n_c.median())}, max {n_c.max()} ({n_c.idxmax()}); in {SE_ROUNDS} (Swahili, "
          f"English): min {n_se.min()} ({n_se.idxmin()}), median {int(n_se.median())}")

    df = sh.rename("share").reset_index()
    df["count"] = df["share"] * df["county"].map(pop)
    out = []
    for c, g in df.groupby("county"):
        f = g["count"].to_numpy()
        base = np.floor(f)
        k = int(round(pop[c] - base.sum()))
        base[np.argsort(-(f - base))[:k]] += 1
        out.append(g.assign(count=base.astype(int)))
    df = pd.concat(out)
    say(int(df["count"].sum()) == POP_2019, f"drawn total {int(df['count'].sum()):,}")
    nat = df.groupby("answer")["count"].sum().sort_values(ascending=False)
    print("\n  national, as drawn:")
    for k, v in nat.items():
        print(f"    {k:24s} {v:>12,}  {v / POP_2019:6.2%}")
    for c in ("NAIROBI", "MOMBASA"):
        g = df[df["county"] == c].sort_values("count", ascending=False)
        print(f"  {c}: " + ", ".join(f"{r.answer} {r.share:.1%}" for r in g.head(9).itertuples()))

    n_by = a.groupby(["county", "answer"]).size()
    df["note"] = [f"share {r.share:.4f}; {n_by.get((r.county, r.answer), 0)} respondents "
                  f"(of {n_c[r.county]})" for r in df.itertuples()]
    df = df[df["count"] > 0]
    res = pd.DataFrame({
        "geo_id": df["county"].map(gid), "geo_level": "county", "geo_name": df["county"],
        "source_category": df["answer"], "count": df["count"], "tier": "modelled",
        "source_id": SOURCE_ID, "year": "2008-2022", "note": df["note"],
    })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    res.sort_values(["geo_id", "count"], ascending=[True, False]).to_csv(OUT, index=False)
    print(f"wrote {OUT} ({len(res)} rows, {res['source_category'].nunique()} answers)")

    split_half(a)
    census_check(nat)


def split_half(a):
    """R4+R6 against R7-R9, county shares of every answer 1%+ nationally (Swahili and English
    left out): do the two halves put each language in the same counties?"""
    def shares(b):
        t = b.groupby(["county", "answer"])["w"].sum()
        return t.div(t.groupby(level="county").sum(), level="county").unstack(fill_value=0)
    x = a[~a["answer"].isin(SE)]
    h1, h2 = shares(x[x["round"] <= 6]), shares(x[x["round"] >= 7])
    h2 = h2.reindex(index=h1.index, columns=h1.columns.union(h2.columns), fill_value=0)
    h1 = h1.reindex(columns=h2.columns, fill_value=0)
    # The assertion is over counties with 40+ respondents in each half: in Isiolo, Marsabit and
    # Tana River one round's eight interviews land in a Somali or a Borana village and swing
    # the share 0 <-> 75%, which is sampling, not a shifted column.
    n1 = x[x["round"] <= 6].groupby("county").size()
    n2 = x[x["round"] >= 7].groupby("county").size()
    big = [c for c in h1.index if n1.get(c, 0) >= 40 and n2.get(c, 0) >= 40]
    nat = x.groupby("answer")["w"].sum() / x["w"].sum()
    print(f"\n  split-half, R4+R6 against R7-R9, Pearson r across all counties / the "
          f"{len(big)} with 40+ respondents in each half:")
    for k in nat[nat >= 0.01].sort_values(ascending=False).index:
        r = np.corrcoef(h1[k], h2[k])[0, 1]
        rb = np.corrcoef(h1.loc[big, k], h2.loc[big, k])[0, 1]
        print(f"    {k:22s} {nat[k]:6.1%}   r = {r:+.3f} / {rb:+.3f}")
        if nat[k] >= 0.03:
            say(rb > 0.9, f"{k}: the two halves agree on where it is (r {rb:+.3f})")


# KPHC 2019 Table 2.31 (Volume IV pp. 423-424), national: the census's ethnic groups against
# the language drawn for them. Not a test of the method (ethnicity is not language), a bound.
CENSUS_T231 = {
    "Kikuyu": 8_148_668, "Luhya": 6_823_842, "Kalenjin": 6_358_113 - 778_408 - 296_374 - 52_596,
    "Luo": 5_066_966, "Kamba": 4_663_910, "Somali": 2_780_502, "Kisii": 2_703_235,
    "Mijikenda": 2_488_691, "Meru": 1_975_869, "Maasai": 1_189_522, "Turkana": 1_016_174,
    "Pokot": 778_408, "Embu": 404_801, "Teso": 417_670, "Taita": 344_415, "Samburu": 333_471,
    "Kuria": 313_854, "Sabaot": 296_374, "Borana": 276_236 + 141_200, "Orma": 158_993,
    "Suba": 157_787, "Pokomo": 112_075, "Rendille": 96_313, "Bajuni": 91_422,
    "Swahili": 56_074, "Okiek": 52_596,
}
CENSUS_TOTAL = 47_564_296


def census_check(nat):
    print("\n  against the census's national ethnic counts (Table 2.31; Kalenjin less Pokot, "
          "Sabaot and Ogiek, drawn apart here):")
    for k, v in CENSUS_T231.items():
        d = nat.get(k, 0) / POP_2019
        c = v / CENSUS_TOTAL
        print(f"    {k:12s} census ethnicity {c:6.2%}   drawn language {d:6.2%}   "
              f"{d - c:+6.2%}")


if __name__ == "__main__":
    main()
