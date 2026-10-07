"""Sudan: home language from Afrobarometer R6-R9 and Arab Barometer III pooled, by state.

    python sources/sd_afro.py --fetch   Sudan's rows from religiondots' merged .sav files
                                        (read-only) -> data/raw/sd/sd_language.csv
    python sources/sd_afro.py           -> data/raw/sd/sd_survey_by_state.csv (state x answer)

RETIRED AS THE MAP'S SOURCE 2026-10-05 (ask 019): Sudan is drawn from sources/sd_estimate.py.
This script stays as the check on the Arabic share in the central states (sources/sd.md).

Sudan's censuses of 1973, 1983, 1993 and 2008 asked no language; the 1955-56 census did, but
its volumes are search-only on HathiTrust and no province table of it is online. So this is the
survey route of AGENT_BRIEF section 2: survey shares times a population base, every row
`modelled`. The record is sources/sd.md.

THE QUESTION. Afrobarometer R6 "Language of respondent" (home language), R7-R9 "Language spoken
in home"; Arab Barometer III q1019_1 "First language". One answer each. Afrobarometer's verbatim
of "Other" is read (VERBATIM below).

ROUNDS LEFT OUT. Afrobarometer R5 (2013: 1,194 Arabic, 3 English, 2 French of 1,199) and Arab
Barometer II (2011: 1,538 of 1,538 Arabic) record no Sudanese language but Arabic at all: their
cards closed the question. Arab Barometer IV and VIII have no Sudanese rows; V asks no language.

THE INTERVIEW LANGUAGE. Every interview in R6 and R7 was in Arabic, and all but 16 of 3,000 in
R8-R9 (the rest English). Every interviewer's home language was Arabic except in R9, which
sent Beja- and Nubian-speaking interviewers to the east and the Blue Nile. No interview was held
in Fur, Masalit, Zaghawa or any Nuba language. The pooled survey puts 98% of Sudanese on Arabic;
sources/sd.md section 2 says why that is far too high and what was tried instead.

GEOGRAPHY. A respondent is put in a 2022 state where the round says which: R9 (18 states), R7's
localities (LOCALITY below; R7's names are machine translations), R6 and Arab Barometer III's 15
pre-2012 states (one old state that is now two puts half the weight in each). R8 names only six
macro-regions. Per state: (weighted answers located there + K x the macro-region's pooled share)
/ (weight there + K); the macro-region pools every respondent in it, R8 included.

POPULATION. COD-PS 2022 state populations (the Central Bureau of Statistics' pre-war projection
from the 2008 census), from religiondots' sd_lookup.csv, 46,934,433 people; Abyei has no figure.
"""
import os
import re
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
AFB = RD / "data" / "raw" / "afrobarometer"
ARB = RD / "data" / "raw" / "arabbarometer"
RAW = HERE / "data" / "raw" / "sd"
EXTRACT = RAW / "sd_language.csv"
LOOKUP = RD / "data" / "geo" / "sd" / "sd_lookup.csv"
# Since 2026-10-05 (ask 019) the map is drawn from sources/sd_estimate.py; this survey build is
# kept as the check on central states' Arabic share and writes beside the raw extract, not to
# data/normalized/sd.csv.
OUT = RAW / "sd_survey_by_state.csv"

CODPS_2022 = 46_934_433
N_STATES = 18
K = 10.0
SOURCE_ID = "afrobarometer_r6_r9_arabbarometer_iii_sudan"

# (survey, round, file, language, verbatim, weight, location column, interview language,
#  interviewer's home language)
ROUNDS = [
    ("AF", 6, "merged_r6_data_2016_36countries2.sav", "Q2", "Q2OTHER", "withinwt",
     "LOCATION.LEVEL.1", "Q103", "Q116"),
    ("AF", 7, "r7_merged_data_34ctry.release.sav", "Q2B", "Q2BOTHER", "withinwt",
     "LOCATION.LEVEL.1", "Q103", "Q116"),
    ("AF", 8, "afrobarometer_release-dataset_merge-34ctry_r8_en_2023-03-01.sav", "Q2",
     "Q2OTHER", "withinwt_hh", None, "Q103", None),
    ("AF", 9, "R9.Merge_39ctry.20Nov23.final_.release_Updated.4Jun25-3.sav", "Q2", "Q2OTHER",
     "withinwt_hh", "LOCATION.LEVEL.1", "Q102", "Q115"),
]
N_RESP = {("AF", 6): 1200, ("AF", 7): 1200, ("AF", 8): 1800, ("AF", 9): 1200, ("AB", 3): 1200}
NON_ANSWERS = {"Don't know", "Refused To Answer", "Refused to answer", "Refused", "Missing"}

OTHER_SD = "Other Sudanese language"
# Coded answers -> the answer they are counted as.
CODED = {
    "Sudanese Arabic": "Sudanese Arabic", "Arabic": "Sudanese Arabic",
    "Nubian Language": "Nubian", "Nubian language": "Nubian",
    "Beja language": "Beja", "Bija": "Beja",
    "English": "English",
    "Masalit": "Masalit",
    # Arab Barometer III, one answer in Kassala: the Rashaida's Arabic, not Sudanese Arabic
    "Bedouin": "Arabic (Bedouin)",
}
# Afrobarometer verbatims of "Other" (folded) -> answer.
VERBATIM = {
    "fur": "Fur", "zaghawa": "Zaghawa", "zaghawa / rutana": "Zaghawa", "masalit": "Masalit",
    "tama": "Tama", "tunjur": "Tunjur", "hausa": "Hausa",
    # "Fellata" is the Sudanese name for people of West African descent and, as a language,
    # for Fulfulde; FULANI is named outright.
    "fulani": "Fula", "flata": "Fula", "fallata": "Fula", "language fellata": "Fula",
    # rutana (rotana) is Sudanese Arabic for any non-Arabic vernacular: a language not named
    "rutana (non-arabic language)": OTHER_SD, "arabic / rutana (non-arabic language)": OTHER_SD,
    "brown amer language": "Tigre",            # the Beni Amer; Tigre is their language
    "language inqasna": "Gaam (Ingessana)",
    "language almapat": OTHER_SD,              # unidentified (Mabaan?), Blue Nile/Sennar
    "sulihab": OTHER_SD, "bargo": OTHER_SD,    # unidentified
}

# State labels, folded to letters only, -> the COD p-codes they cover (two for an old state that
# is now two). religiondots' sources/sd.py STATE, plus R6's and R7's spellings.
STATE = {
    "khartoum": ["SD01"], "khartom": ["SD01"],
    "northdarfur": ["SD02"], "southdarfur": ["SD03"], "westdarfur": ["SD04"],
    "eastdarfur": ["SD05"], "centraldarfur": ["SD06"],
    "southkordofan": ["SD07"], "southkurdofan": ["SD07"], "westkordofan": ["SD18"],
    "westkurdofan": ["SD18"],
    "bluenile": ["SD08"], "whitenile": ["SD09"], "redsea": ["SD10"], "theredsea": ["SD10"],
    "kassala": ["SD11"], "gedaref": ["SD12"], "gedarif": ["SD12"], "alqadarif": ["SD12"],
    "algedarif": ["SD12"], "northkordofan": ["SD13"], "northkurdofan": ["SD13"],
    "northkurdufan": ["SD13"], "sennar": ["SD14"], "sinnar": ["SD14"],
    "gezira": ["SD15"], "algezira": ["SD15"], "aljazeera": ["SD15"],
    "nhralnil": ["SD16"], "rivernile": ["SD16"], "nile": ["SD16"],
    "north": ["SD17"], "northern": ["SD17"],
}
# The pre-2012 states, in R6 and Arab Barometer III: South Darfur held East Darfur, West Darfur
# held Central Darfur, South Kordofan held West Kordofan.
OLD_STATE = {"southdarfur": ["SD03", "SD05"], "westdarfur": ["SD04", "SD06"],
             "southkurdufan": ["SD07", "SD18"], "southkordofan": ["SD07", "SD18"]}

# R7's 36 localities, as the release spells them (machine translations of the Arabic) -> state.
LOCALITY = {
    "AL-DA'EEN": "SD05",            # Ed Daein
    "AN EGG": "SD04",               # al-Beida (the egg), West Darfur; its answers: Masalit, Tunjur
    "EL FASHER": "SD02", "NYALA": "SD03", "ZALINGEI": "SD06",
    "MOTHER SMOKED": "SD06",        # Umm Dukhun
    "THE GARDEN": "SD04",           # al-Geneina (the little garden)
    "GEDAREF": "SD12", "KASSALA": "SD11", "PORT SUDAN": "SD10", "SWAKIN": "SD10",
    "WESTERN SKIPS": "SD12",        # Western Galabat
    "EMPRESS": "SD01", "KHARTOUM": "SD01", "MOUNT OF OLIVES": "SD01", "NAUTICAL": "SD01",
    "OMDURMAN": "SD01", "EAST OF THE NILE": "SD01",
    "AL RAHAD": "SD13", "PARA": "SD13", "SHIKAN": "SD13", "SHIKAN ALBAN IS NEW": "SD13",
    "DILLING": "SD07", "KADUGLI": "SD07", "FOOLA": "SD18",
    "CIVILIAN": "SD15",             # Medani
    "COSTY": "SD09", "DAMAZIN": "SD08", "EAST OF THE ISLAND": "SD15", "SANJA": "SD14",
    "SENNAR": "SD14", "SOUTH OF THE ISLAND": "SD15",
    "ATBARA": "SD16", "DONGOLA": "SD17",
}
# Left at macro-region on purpose: "PIZZAS" (Darfur, 8, not identified) and one Kordofan label
# that is mojibake in the release (8).
MACRO_OF = {"SD01": "Khartoum", "SD02": "Darfur", "SD03": "Darfur", "SD04": "Darfur",
            "SD05": "Darfur", "SD06": "Darfur", "SD07": "Kordofan", "SD13": "Kordofan",
            "SD18": "Kordofan", "SD08": "Central", "SD09": "Central", "SD14": "Central",
            "SD15": "Central", "SD10": "East", "SD11": "East", "SD12": "East", "SD16": "North",
            "SD17": "North"}
MACRO_KEY = {"khartoum": "Khartoum", "darfur": "Darfur", "kurdufan": "Kordofan",
             "kordofan": "Kordofan", "central": "Central", "middleeast": "Central",
             "east": "East", "north": "North", "northern": "North"}


def say(ok, msg):
    print(("  ok  " if ok else "  FAIL") + "  " + msg)
    if not ok:
        raise SystemExit("check failed: " + msg)


def letters(s):
    s = unicodedata.normalize("NFKD", str(s))
    return re.sub(r"[^a-z]", "", "".join(c for c in s if not unicodedata.combining(c)).lower())


def vfold(s):
    s = str(s).strip()
    return "" if s.lower() in ("", "nan", "none") else " ".join(s.split()).lower()


# ---------------------------------------------------------------- extract

def _read(path, **kw):
    import pyreadstat
    try:
        return pyreadstat.read_sav(str(path), **kw)
    except Exception:  # noqa: BLE001  R6 is not valid UTF-8
        return pyreadstat.read_sav(str(path), encoding="LATIN1", **kw)


def fetch():
    out = []
    for sv, rnd, name, q, qo, wt, loc, il, ih in ROUNDS:
        p = AFB / name
        _, meta = _read(p, metadataonly=True)
        lab = str(meta.column_names_to_labels.get(q, "")).casefold()
        say("language" in lab and ("respondent" in lab or "home" in lab),
            f"R{rnd} {q} is the home-language question ({lab!r})")
        cols = ["COUNTRY", "REGION", q, qo, wt, il] + [c for c in (loc, ih) if c]
        df, _ = _read(p, usecols=cols, apply_value_formats=True)
        sub = df[df["COUNTRY"].astype(str).str.strip().str.casefold() == "sudan"]
        say(len(sub) == N_RESP[(sv, rnd)], f"R{rnd}: {len(sub):,} Sudanese respondents")
        w = pd.to_numeric(sub[wt], errors="coerce")
        say(0.98 <= w.sum() / len(sub) <= 1.02, f"R{rnd} {wt} is a within-country weight")
        out.append(pd.DataFrame({
            "survey": sv, "round": rnd, "region": sub["REGION"].astype(str),
            "loc": sub[loc].astype(str) if loc else "",
            "lang": sub[q].astype(str), "verbatim": sub[qo].astype(str).str.strip(),
            "interview": sub[il].astype(str), "interviewer": sub[ih].astype(str) if ih else "",
            "w": w}))
    p = ARB / "ABIII_English.sav"
    df, meta = _read(p, usecols=["country", "q1", "q1019_1", "wt"], apply_value_formats=True)
    sub = df[df["country"].astype(str).str.contains("Sudan")]
    say(len(sub) == N_RESP[("AB", 3)], f"Arab Barometer III: {len(sub):,} Sudanese respondents")
    w = pd.to_numeric(sub["wt"], errors="coerce")
    out.append(pd.DataFrame({
        "survey": "AB", "round": 3, "region": "", "loc": sub["q1"].astype(str),
        "lang": sub["q1019_1"].astype(str), "verbatim": "", "interview": "", "interviewer": "",
        "w": w * len(sub) / w.sum()}))
    a = pd.concat(out, ignore_index=True)
    RAW.mkdir(parents=True, exist_ok=True)
    a.to_csv(EXTRACT, index=False)
    print(f"wrote {EXTRACT} ({len(a):,} respondents)")


# ---------------------------------------------------------------- harmonise

def answer(r):
    lab = r["lang"].strip()
    if lab == "Other":
        v = vfold(r["verbatim"])
        if not v:
            return OTHER_SD
        if v not in VERBATIM:
            raise SystemExit(f"verbatim {r['verbatim']!r} (R{r['round']}) not in VERBATIM")
        return VERBATIM[v]
    if lab not in CODED:
        raise SystemExit(f"answer {lab!r} ({r['survey']} {r['round']}) not in CODED")
    return CODED[lab]


def states_of(r):
    """The 2022 states a respondent can be put in: one, two (an old state), or None."""
    if r["survey"] == "AB" or r["round"] in (6, 9):
        k = letters(r["loc"])
        if r["survey"] == "AB" or r["round"] == 6:
            if k in OLD_STATE:
                return OLD_STATE[k]
        if k not in STATE:
            raise SystemExit(f"{r['survey']}{r['round']} state label {r['loc']!r} not in STATE")
        return STATE[k]
    if r["round"] == 7:
        s = LOCALITY.get(r["loc"].strip())
        return [s] if s else None
    return None


def load():
    a = pd.read_csv(EXTRACT, dtype=str, keep_default_na=False)
    a["round"] = a["round"].astype(int)
    a["w"] = a["w"].astype(float)
    say(len(a) == sum(N_RESP.values()), f"{len(a):,} respondents in the extract")
    print("  interview language:", a.groupby(["survey", "round"])["interview"]
          .agg(lambda s: s.value_counts().to_dict()).to_dict())
    print("  interviewer's home language:", a[a["interviewer"] != ""].groupby("round")[
        "interviewer"].agg(lambda s: s.value_counts().to_dict()).to_dict())
    n0 = len(a)
    a = a[~a["lang"].isin(NON_ANSWERS)].copy()
    print(f"  {n0 - len(a)} non-answers dropped")
    a["answer"] = a.apply(answer, axis=1)
    a["states"] = a.apply(states_of, axis=1)
    m = a["states"].map(lambda s: s[0] if s else None).map(MACRO_OF)
    a["macro"] = m.fillna(a["region"].map(lambda s: MACRO_KEY.get(letters(s))))
    say(a["macro"].notna().all(), "every respondent has a macro-region")
    # weights sum to n within each round
    a["w"] = a["w"] * a.groupby(["survey", "round"])["w"].transform(lambda s: len(s) / s.sum())
    return a


def singletons(a):
    """A language named in free text by one respondent in the pool goes on OTHER_SD."""
    free = set(VERBATIM.values()) - set(CODED.values())
    n = a.groupby("answer").size()
    one = [k for k in free if n.get(k, 0) == 1 and k != OTHER_SD]
    a.loc[a["answer"].isin(one), "answer"] = OTHER_SD
    return sorted(one)


def state_shares(a, lut):
    rows = []
    for _, r in a.iterrows():
        if r["states"]:
            for s in r["states"]:
                rows.append((s, r["answer"], r["w"] / len(r["states"])))
    loc = pd.DataFrame(rows, columns=["state", "answer", "w"])
    print(f"  {a['states'].notna().sum():,} of {len(a):,} respondents put in a state "
          f"({a.loc[a['states'].isna()].groupby(['survey', 'round']).size().to_dict()} at "
          "macro-region only)")
    macro = a.groupby(["macro", "answer"])["w"].sum()
    macro = macro.div(macro.groupby(level=0).sum(), level=0)
    t = loc.groupby(["state", "answer"])["w"].sum().unstack(fill_value=0.0)
    t = t.reindex(lut["geo_id"], fill_value=0.0)
    out = {}
    for st in t.index:
        prior = macro.loc[MACRO_OF[st]].reindex(t.columns, fill_value=0.0)
        out[st] = (t.loc[st] + K * prior) / (t.loc[st].sum() + K)
    sh = pd.DataFrame(out).T
    nloc = t.sum(axis=1)
    return sh, nloc


def main():
    if "--fetch" in sys.argv:
        fetch()
    a = load()
    ct = pd.crosstab(a["answer"], [a["survey"], a["round"]])
    ct["all"] = ct.sum(axis=1)
    print(ct.sort_values("all", ascending=False).to_string())
    one = singletons(a)
    print(f"  named by one respondent -> {OTHER_SD}: {one}")

    lut = pd.read_csv(LOOKUP, dtype=str)
    lut["pop"] = lut["pop"].astype(int)
    say(len(lut) == N_STATES and int(lut["pop"].sum()) == CODPS_2022,
        f"sd_lookup.csv: {len(lut)} states, {int(lut['pop'].sum()):,} people (COD-PS 2022)")
    used = sorted({s for ss in a["states"].dropna() for s in ss})
    say(used == sorted(lut["geo_id"]), "respondents in all 18 states, and only those")

    sh, nloc = state_shares(a, lut)
    say((sh.sum(axis=1) - 1).abs().max() < 1e-9, "every state's shares sum to 1")
    pop = lut.set_index("geo_id")["pop"]
    name = lut.set_index("geo_id")["name"]
    print(f"  located weight per state: min {nloc.min():.0f} ({name[nloc.idxmin()]}), "
          f"median {nloc.median():.0f}, max {nloc.max():.0f} ({name[nloc.idxmax()]})")
    print("\n  non-Arabic share by state (located respondents, drawn):")
    for st in sh.index:
        na = 1 - sh.loc[st, "Sudanese Arabic"]
        top = sh.loc[st].drop("Sudanese Arabic").sort_values(ascending=False).head(3)
        print(f"    {name[st]:15s} n={nloc[st]:6.0f}  {na:6.2%}   "
              + ", ".join(f"{k} {v:.1%}" for k, v in top.items() if v > 0))

    out = []
    for st in sh.index:
        f = (sh.loc[st] * pop[st]).to_numpy()
        base = np.floor(f)
        k = int(round(pop[st] - base.sum()))
        base[np.argsort(-(f - base))[:k]] += 1
        out.append(pd.DataFrame({"state": st, "answer": sh.columns, "share": sh.loc[st].values,
                                 "count": base.astype(int)}))
    df = pd.concat(out)
    say(int(df["count"].sum()) == CODPS_2022, f"drawn total {int(df['count'].sum()):,}")
    nat = df.groupby("answer")["count"].sum().sort_values(ascending=False)
    print("\n  national, as drawn:")
    for k_, v in nat.items():
        print(f"    {k_:26s} {v:>12,}  {v / CODPS_2022:7.3%}")
    df = df[df["count"] > 0]
    res = pd.DataFrame({
        "geo_id": df["state"], "geo_level": "state", "geo_name": df["state"].map(name),
        "source_category": df["answer"], "count": df["count"], "tier": "modelled",
        "source_id": SOURCE_ID, "year": "2013-2022",
        "note": [f"share {s:.5f}; {nloc[st]:.0f} located respondents" for s, st in
                 zip(df["share"], df["state"])],
    })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    res.sort_values(["geo_id", "count"], ascending=[True, False]).to_csv(OUT, index=False)
    print(f"wrote {OUT} ({len(res)} rows, {res['source_category'].nunique()} answers)")


if __name__ == "__main__":
    main()
