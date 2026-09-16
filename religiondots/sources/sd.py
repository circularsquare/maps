"""Sudan: religion from two pooled surveys, one national share, on the 2022 state projection.

Reads the merged Afrobarometer and Arab Barometer files and data/geo/sd/sd_lookup.csv; writes
data/normalized/sd.csv. `sources/sd.md` is the record in prose; `ask/RULINGS.md` 2026-09-15 and
2026-09-16 (the Maghreb and Mauritania rulings) and ask 033 the rulings it builds on.

## WHY SUDAN WAS CLOSED, AND WHY IT IS NOT

The Presidency deleted the religion question from the 2008 census (UNSD's country report;
`sources.md` §11af), and no census since has been held. §11af and §11aq closed Sudan because a
99%-Muslim survey pool "draws nothing". Anita's Maghreb ruling (2026-09-15) retired that reason: a
near-uniformly Muslim country is drawn with the best evidence there is for where its non-Muslims
are. The instruments are the Afrobarometer, rounds 5 to 9 (2013 to 2022, 6,599 interviews) and the
Arab Barometer, waves V and VII (2018-19 and 2022, 4,111), both sampling citizens aged 18 and over.

## THE POOL

  * **Afrobarometer R5-R9.** Every answer is grouped by name (`AFRO_GROUP`, raising on a new one):
    the Muslim answers and brotherhoods to Muslim; every named church and `Christian only` to
    Christian; `None`, `Atheist` and `Agnostic` to no religion.
  * **Arab Barometer V and VII.** II and III are out on the card (`ab.card`): neither offers a box
    for having no religion. V's `Atheist` and VII's `No religion` are each card's one such box.

## FOUR DROPS, EACH COUNTED AND ASSERTED

  1. **Refusals and don't-knows** (both surveys).
  2. **Answers that contradict the follow-up in the same interview** (Arab Barometer V, Algeria's
     rule, `sources/dz.md` §3): six of V's seven atheists name a Muslim branch (`Just a Muslim`,
     `Sunni`), as do one Christian and the one Jewish answer. `contradictions`.
  3. **Round 8's `None` in Darfur.** 31 of R8's 32 `None` answers are in Darfur (30 rural), 7.2%
     of its Darfur interviews, while Darfur returns 2 in the other four rounds and 1 in the Arab
     Barometer, about 2,400 interviews. R8 is also the only round with no location below the
     region. One round, one region, a share nothing else reproduces: read as a fieldwork artefact
     (`None` recorded for "no particular sect"), not as 31 Darfuris with no religion.
     `R8_DARFUR_NONE`. The playbook's "two boxes can trade places" trap, in one region.
  4. **Three single answers** naming a religion with no community here to place (`Bahai`, `Jewish`
     and `Traditional / ethnic religion`, one each). `SINGLETONS`.

## WHAT IS DRAWN

One national non-Muslim share and mix, Christian and no religion, applied to every state's people in
the Central Bureau of Statistics' 2022 projection (COD-PS). The tests that could place anything are
printed and asserted:

  * **no macro-region stands apart** (`standouts`: the six regions every round can be read at,
    each against the rest in both halves, early 2013-2019 and late 2021-2022, Bonferroni);
  * **towns**: an urban excess must hold in both instruments (Morocco's pre-registered bar,
    `sources/ma.py`) before an urban split could be drawn (`urban_test`).

The state labels are tested without the religion column (`held_out`, the waves and round that carry
all 18 states), and the Arab Barometer's two waves against a quota (`assert_not_quota`).

Usage:
    python sources/sd.py            rebuild data/normalized/sd.csv
    python sources/sd.py --fetch    fetch any missing survey file first
"""

import os
import re
import sys
import unicodedata

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
os.environ.setdefault("OMP_NUM_THREADS", "6")

import numpy as np
import pandas as pd
from scipy import stats

import afrobarometer as afb
import arabbarometer as ab
from afrobarometer import round_within_rows
from ma import stratified_p, wshare          # generic; Morocco's copies, imported not copied

LOOKUP = os.path.join(ROOT, "data", "geo", "sd", "sd_lookup.csv")
OUT = os.path.join(ROOT, "data", "normalized", "sd.csv")

COUNTRY = "Sudan"
AFRO_ROUNDS = [5, 6, 7, 8, 9]
AB_WAVES = ["V", "VII"]
AB_OMIT = {"II": "the card has no box for having no religion (its codes are muslim, christian, "
                 "unspecific answer, khaki, jewish); 4 Christians of 1,538",
           "III": "the card has no box for having no religion; 2 Christians of 1,200"}
AB_RECODE = {"refused": "Refused to answer", "Atheist": "No religion"}
YEARS = "2013-2022"
SOURCE_ID = "sd_afrobarometer_arabbarometer_2013_2022"
CATS = ["Muslim", "Christian", "No religion"]
NON_MUSLIM = ["Christian", "No religion"]

# Afrobarometer answers, keyed by the fold of the text before any " (" (R8-R9 spell the catch-alls
# out in long form), -> group. A new answer stops the build.
AFRO_GROUP = {
    "muslim only": "Muslim", "sunni only": "Muslim", "shia": "Muslim", "ismaeli": "Muslim",
    "tijaniya brotherhood": "Muslim", "qadiriya brotherhood": "Muslim",
    "mouridiya brotherhood": "Muslim",
    "christian only": "Christian", "anglican": "Christian", "church of christ": "Christian",
    "independent": "Christian", "jehovah's witness": "Christian", "lutheran": "Christian",
    "methodist": "Christian", "mormon": "Christian", "orthodox": "Christian",
    "presbyterian": "Christian", "seventh day adventist": "Christian",
    "none": "No religion", "atheist": "No religion", "agnostic": "No religion",
    "bahai": "SINGLE", "jewish": "SINGLE", "traditional / ethnic religion": "SINGLE",
}
SINGLETONS = {"bahai": 1, "jewish": 1, "traditional / ethnic religion": 1}
R8_DARFUR_NONE = 31
DARFUR_NONE_ELSEWHERE_MAX = 3          # the other Afrobarometer rounds and the Arab Barometer

# Arab Barometer V: the one sect item every respondent answers. Folded.
SECT_ISLAM = {"just a muslim", "sunni", "shia", "alawi", "druze", "maliki", "ibadi", "sufi"}
SECT_CHRISTIAN = {"catholic", "orthodox", "just a christian", "armenian", "protestant",
                  "evangelical", "coptic", "coptic orthodox", "anglican"}
CONTRADICTIONS = {("V", "No religion"): 6, ("V", "Christian"): 1, ("V", "Jewish"): 1}

# State labels, folded to letters only, -> COD p-code. `north` is Northern state wherever a label
# names a state (Afrobarometer R5, Arab Barometer V): both lists also carry River Nile apart.
STATE = {
    "SD01": ["khartoum", "khartom"], "SD02": ["northdarfur"], "SD03": ["southdarfur"],
    "SD04": ["westdarfur"], "SD05": ["eastdarfur"], "SD06": ["centraldarfur", "centraldarfu"],
    "SD07": ["southkordofan", "southkordufan", "southkurdofan", "southkurdufan"],
    "SD08": ["bluenile"],
    "SD09": ["whitenile"], "SD10": ["redsea"], "SD11": ["kassala"],
    "SD12": ["gedaref", "algedarif", "gadarif", "gedarif"],
    "SD13": ["northkordofan", "northkordufan", "northkurdofan", "northkurdufan"],
    "SD14": ["sennar", "sinnar"],
    "SD15": ["theisland", "algezira", "gezira"],
    "SD16": ["nileriver", "rivernile", "nahralnil", "nhralnil"],
    "SD17": ["north", "northern"], "SD18": ["westkordofan", "westkurdofan"],
}
STATE_KEY = {k: pc for pc, ks in STATE.items() for k in ks}
MACRO_OF = {"SD01": "Khartoum", "SD02": "Darfur", "SD03": "Darfur", "SD04": "Darfur",
            "SD05": "Darfur", "SD06": "Darfur", "SD07": "Kordofan", "SD13": "Kordofan",
            "SD18": "Kordofan", "SD08": "Central", "SD09": "Central", "SD14": "Central",
            "SD15": "Central", "SD10": "East", "SD11": "East", "SD12": "East", "SD16": "North",
            "SD17": "North"}
# Afrobarometer R6-R8 REGION labels, folded -> macro-region (R6's LOCATION.LEVEL.1 witnesses it).
MACRO_KEY = {"khartoum": "Khartoum", "darfur": "Darfur", "kurdufan": "Kordofan",
             "kordofan": "Kordofan", "kurdofan": "Kordofan", "central": "Central",
             "middleeast": "Central", "east": "East", "north": "North", "northern": "North"}
# Afrobarometer R5's fifteen states are the pre-2012 ones: West Darfur held Central Darfur, South
# Darfur held East Darfur, South Kordufan held West Kordofan. Each maps to its region only.

HALVES = [["AF5", "AF6", "AF7", "AB-V"], ["AF8", "AF9", "AB-VII"]]
STANDOUTS = set()
URBAN_BAR = 0.05                       # Morocco's, pre-registered in sources/ma.py
NONMUSLIM_BAND = (0.002, 0.02)
GEO_WAVES = ["AB-V", "AB-VII", "AF9"]  # the waves and round that name all 18 states
ALLOC_BAR = 0.80

# Pew Research Center, Religious Composition 2010-2020, percentages file, Sudan 2020
# (data/raw/estimates/pew.zip). Printed beside the survey's level, never used to set it.
PEW_2020 = {"Muslims": 98.855515, "Christians": 0.487978, "Religiously_unaffiliated": 0.563213,
            "Other_religions": 0.092054, "Hindus": 0.001238}

# note_public's survey figures, measured 2026-09-15 and asserted in main.
NOTE = dict(afro=6557, arab=4100, named=10657, nonmuslim=86, christian=62, no_religion=24,
            share_pct=0.687, christian_pct=0.504)


def fold(s):
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(ch for ch in s if not unicodedata.combining(ch)).lower().replace("’", "'")
    return " ".join(s.split())


def letters(s):
    return re.sub(r"[^a-z]", "", fold(s))


def load_afro():
    print("\n=== Afrobarometer, Sudan ===")
    a = afb.load(COUNTRY, expect_rounds=AFRO_ROUNDS, regroup=True,
                 extra=["LOCATION.LEVEL.1", "URBRUR", "URBRUR_COND"])
    key = a["category"].map(lambda c: fold(str(c).split(" (")[0]))
    unknown = sorted(set(key) - set(AFRO_GROUP))
    if unknown:
        raise SystemExit(f"Afrobarometer answers with no group: {unknown}")
    a["group"] = key.map(AFRO_GROUP)
    single = key[a["group"] == "SINGLE"].value_counts().to_dict()
    print(f"  single answers dropped (no community to place): {single}")
    if single != SINGLETONS:
        raise SystemExit(f"single answers are {single}, not {SINGLETONS}")
    a = a[a["group"] != "SINGLE"].copy()
    a["wave"] = "AF" + a["round"].astype(str)

    # macro-region and, where the round names one, state
    a["geo_id"] = None
    r5 = a["round"] == 5
    a.loc[r5, "geo_id"] = a.loc[r5, "geo_raw"].map(lambda s: STATE_KEY.get(letters(s)))
    r9 = a["round"] == 9
    a.loc[r9, "geo_id"] = a.loc[r9, "LOCATION.LEVEL.1"].map(lambda s: STATE_KEY.get(letters(s)))
    if a.loc[r5 | r9, "geo_id"].isna().any():
        bad = sorted(set(a.loc[(r5 | r9) & a["geo_id"].isna(), "geo_raw"].astype(str))
                     | set(a.loc[r9 & a["geo_id"].isna(), "LOCATION.LEVEL.1"].astype(str)))
        raise SystemExit(f"Afrobarometer R5/R9 labels with no state: {bad}")
    r5_macro = {"westdarfur": "Darfur", "southdarfur": "Darfur", "southkordufan": "Kordofan"}
    a["macro"] = a["geo_id"].map(MACRO_OF)
    a.loc[r5, "macro"] = a.loc[r5, "geo_raw"].map(
        lambda s: r5_macro.get(letters(s), MACRO_OF.get(STATE_KEY.get(letters(s)))))
    a.loc[r5, "geo_id"] = None          # pre-2012 states: region only
    mid = a["round"].isin([6, 7, 8])
    a.loc[mid, "macro"] = a.loc[mid, "geo_raw"].map(lambda s: MACRO_KEY.get(letters(s)))
    r9_region = a.loc[r9, "geo_raw"].map(lambda s: MACRO_KEY.get(letters(s)))
    if (r9_region != a.loc[r9, "macro"]).any():
        raise SystemExit("Afrobarometer R9: a state's LOCATION.LEVEL.1 disagrees with its REGION")
    r6 = a["round"] == 6
    r6_state = a.loc[r6, "LOCATION.LEVEL.1"].map(lambda s: STATE_KEY.get(letters(s)))
    r6_old = {"westdarfur": "Darfur", "southdarfur": "Darfur", "southkurdufan": "Kordofan"}
    r6_witness = a.loc[r6, "LOCATION.LEVEL.1"].map(
        lambda s: r6_old.get(letters(s), MACRO_OF.get(STATE_KEY.get(letters(s)))))
    off = r6_witness != a.loc[r6, "macro"]
    if off.any():
        pairs = (a.loc[r6][off].groupby(["LOCATION.LEVEL.1", "geo_raw"]).size().to_dict())
        raise SystemExit(f"Afrobarometer R6: LOCATION.LEVEL.1 disagrees with REGION: {pairs}")
    if a["macro"].isna().any():
        raise SystemExit(f"Afrobarometer rows with no region: "
                         f"{sorted(set(a.loc[a['macro'].isna(), 'geo_raw'].astype(str)))}")

    u = a["URBRUR"].astype(str).str.lower()
    u8 = a["URBRUR_COND"].astype(str).str.lower()
    a["urban"] = np.where(a["round"] == 8, u8.str.startswith("urban"), u.str.startswith("urban"))
    known = np.where(a["round"] == 8, u8.str.match(r"^(urban|rural)"), u.isin(["urban", "rural"]))
    if not known.all():
        raise SystemExit("Afrobarometer urban/rural has a value that is neither")

    none_darfur = (a["group"] == "No religion") & (a["macro"] == "Darfur")
    by = none_darfur.groupby(a["round"]).sum().to_dict()
    print(f"  `None` answers in Darfur by round: {by}")
    if by.get(8, 0) != R8_DARFUR_NONE or sum(v for k, v in by.items() if k != 8) > DARFUR_NONE_ELSEWHERE_MAX:
        raise SystemExit(f"Darfur's `None` by round is {by}; read sources/sd.md §3 before dropping")
    a = a[~(none_darfur & (a["round"] == 8))].copy()
    print(f"  dropped R8's {R8_DARFUR_NONE} `None` answers in Darfur (sources/sd.md §3)")
    a["category"] = a["group"]
    return a[["wave", "category", "geo_id", "macro", "urban", "w"]]


def contradictions(df):
    sect = df["sect_m"].where(df["sect_m"].notna(), df["sect_c"]).map(
        lambda v: fold(v) if isinstance(v, str) else None)
    cat = df["category"]
    bad = ((cat != "Muslim") & sect.isin(SECT_ISLAM)) | ((cat == "Muslim") & sect.isin(SECT_CHRISTIAN))
    got = df[bad].groupby(["wave", "category"]).size().to_dict()
    print(f"  answers contradicting the follow-up, dropped: {got}")
    if got != CONTRADICTIONS:
        raise SystemExit(f"contradictions are {got}, not {CONTRADICTIONS}")
    return df[~bad].copy()


def load_ab():
    ab.card(AB_WAVES, AB_OMIT)
    print("\n=== Arab Barometer, Sudan ===")
    df = ab.load(COUNTRY, expect_waves=AB_WAVES, waves=AB_WAVES, omit=AB_OMIT, recode=AB_RECODE,
                 extra={"sect_m": ("Q1012A_MUSLIM", "q1012a"), "sect_c": ("Q1012A_CHRISTIAN",),
                        "urban_raw": ("q13",)})
    print(pd.crosstab(df["category"], df["wave"]).to_string())
    n = int((df["category"] == "Refused to answer").sum())
    df = df[df["category"] != "Refused to answer"].copy()
    print(f"  dropped {n} refusals")
    df = contradictions(df)
    odd = sorted(set(df["category"]) - set(CATS))
    if odd:
        raise SystemExit(f"Arab Barometer answers with no category: {odd}")
    df["geo_id"] = df["geo_raw"].map(lambda s: STATE_KEY.get(letters(s)))
    if df["geo_id"].isna().any():
        raise SystemExit(f"Arab Barometer Q1 labels with no state: "
                         f"{sorted(set(df.loc[df['geo_id'].isna(), 'geo_raw']))}")
    for w, x in df.groupby("wave"):
        if x["geo_id"].nunique() != 18 or (x.groupby("geo_raw")["geo_id"].nunique() > 1).any():
            raise SystemExit(f"wave {w}: the Q1 labels are not 18 states one to one")
    df["macro"] = df["geo_id"].map(MACRO_OF)
    u = df["urban_raw"].astype(str).str.lower()
    if not u.isin(["urban", "rural"]).all():
        raise SystemExit("Arab Barometer q13 has a value that is neither urban nor rural")
    df["urban"] = u.eq("urban")
    df["wave"] = "AB-" + df["wave"]
    return df[["wave", "category", "geo_id", "macro", "urban", "w"]]


def allocation(g, pop):
    share = pop / pop.sum()
    print("\n  weighted respondents per state against COD-PS 2022, by wave:")
    for w in GEO_WAVES:
        x = g[g["wave"] == w]
        ws = x.groupby("geo_id")["w"].sum().reindex(pop.index, fill_value=0.0)
        rho = stats.spearmanr(ws, pop).statistic
        rel = (ws / ws.sum()) / share
        print(f"    {w:<7} Spearman {rho:+.3f}; lowest {rel.idxmin()} {rel.min():.2f}, highest "
              f"{rel.idxmax()} {rel.max():.2f}")
        if rho < ALLOC_BAR:
            raise SystemExit(f"{w}'s states do not fit the projection ({rho:+.3f})")


def standouts(g):
    cand = g.groupby("macro")["nonm"].sum()
    cand = cand[cand >= 2]
    bar = 0.05 / len(cand)
    print(f"\n  standouts: {len(cand)} regions with two or more non-Muslims, each against the rest "
          f"in both halves ({' | '.join(','.join(h) for h in HALVES)}), bar P < {bar:.4f}:")
    found = set()
    for n in cand.index:
        ps = [stratified_p(d, d["macro"] == n) for d in (g[g["wave"].isin(h)] for h in HALVES)]
        pooled = stratified_p(g, g["macro"] == n)
        print(f"    {n:<9} " + "   ".join(f"{o} vs {e:.2f} P={p:.4f}" for o, e, p in ps)
              + f"   pooled {pooled[0]} vs {pooled[1]:.2f} P={pooled[2]:.4f}")
        if all(p < bar for _o, _e, p in ps):
            found.add(n)
    if found != STANDOUTS:
        raise SystemExit(f"regions standing apart: {sorted(found)}, not {sorted(STANDOUTS)}")


def urban_test(g):
    passes = []
    for label, d in (("Afrobarometer", g[g["wave"].str.startswith("AF")]),
                     ("Arab Barometer", g[g["wave"].str.startswith("AB")])):
        o, e, p = stratified_p(d, d["urban"])
        print(f"  urban excess, {label}: {o} urban of {int(d['nonm'].sum())} non-Muslims against "
              f"{e:.2f} expected, P = {p:.4f}")
        passes.append(p < URBAN_BAR)
    if all(passes):
        raise SystemExit("both surveys now show an urban excess; an urban split needs a state "
                         "urban/rural table (sources/sd.md §4) before it can be drawn")


def main():
    if "--fetch" in sys.argv:
        afb.fetch(rounds=AFRO_ROUNDS)
        ab.fetch(waves=AB_WAVES + list(AB_OMIT))

    a = load_afro()
    b = load_ab()
    g = pd.concat([a, b], ignore_index=True)
    g["nonm"] = g["category"] != "Muslim"

    print("\n  pooled answers by wave:")
    print(pd.crosstab(g["category"], g["wave"]).to_string())
    print("  weighted non-Muslim share by wave: " + ", ".join(
        f"{w} {wshare(x, x['nonm']):.3%}" for w, x in g.groupby("wave", sort=False)))
    print("  weighted Christian share by wave: " + ", ".join(
        f"{w} {wshare(x, x['category'] == 'Christian'):.3%}" for w, x in g.groupby("wave", sort=False)))
    print("\n  answers by region (non-Muslim / all):")
    t = g.groupby("macro").agg(n=("nonm", "size"), nonm=("nonm", "sum"),
                               christian=("category", lambda s: int((s == "Christian").sum())))
    print(t.to_string())

    lut = pd.read_csv(LOOKUP, dtype={"geo_id": str})
    if len(lut) != 18:
        raise SystemExit("sd_lookup.csv is not the 18 states")
    pop = lut.set_index("geo_id")["pop"]

    geo = g[g["wave"].isin(GEO_WAVES)].copy()
    allocation(geo, pop)
    ab.held_out(geo, pop, COUNTRY, pop_source="COD-PS 2022 (CBS projection)")
    abq = b.copy()
    abq["wave"] = abq["wave"].str.replace("AB-", "", regex=False)
    ab.assert_not_quota(abq, COUNTRY)
    standouts(g)
    urban_test(g)

    L = wshare(g, g["nonm"])
    print(f"\n  weighted non-Muslim share, pooled: {L:.4%}")
    if not NONMUSLIM_BAND[0] <= L <= NONMUSLIM_BAND[1]:
        raise SystemExit(f"{L:.4%} is outside {NONMUSLIM_BAND}")
    nm = g[g["nonm"]]
    comp = (nm.groupby("category")["w"].sum() / nm["w"].sum()).reindex(NON_MUSLIM, fill_value=0.0)
    print("    non-Muslim mix (weighted): " + ", ".join(
        f"{c} {v:.1%} (n={int((nm['category'] == c).sum())})" for c, v in comp.items()))
    late = g[g["wave"].isin(HALVES[1])]
    print(f"    late half alone (2021-2022): {wshare(late, late['nonm']):.4%} non-Muslim, "
          f"{wshare(late, late['category'] == 'Christian'):.4%} Christian")
    print("  outside level check, Pew Research Center 2020 (everyone in Sudan): "
          + ", ".join(f"{k} {v:.3f}%" for k, v in PEW_2020.items())
          + f"; the pool reads Christian {100 * comp['Christian'] * L:.3f}% and no religion "
            f"{100 * comp['No religion'] * L:.3f}%")

    got = dict(afro=int(g["wave"].str.startswith("AF").sum()),
               arab=int(g["wave"].str.startswith("AB").sum()), named=len(g), nonmuslim=len(nm),
               christian=int((nm["category"] == "Christian").sum()),
               no_religion=int((nm["category"] == "No religion").sum()),
               share_pct=round(100 * L, 3), christian_pct=round(100 * comp["Christian"] * L, 3))
    print(f"\n  note_public's survey figures: {got}")
    if NOTE and got != NOTE:
        raise SystemExit(f"note_public's figures are {NOTE}; the pool gives {got}")

    names = dict(zip(lut["geo_id"], lut["name"]))
    m = pd.DataFrame({"Muslim": pop * (1 - L), **{c: pop * L * comp[c] for c in NON_MUSLIM}})[CATS]
    counts = round_within_rows(m)
    if not (counts.sum(axis=1) == pop.reindex(counts.index)).all():
        raise SystemExit("a state's rounded counts do not sum to its projection")
    out = counts.stack().rename("count").reset_index()
    out.columns = ["geo_id", "source_category", "count"]
    n_by = g.dropna(subset=["geo_id"]).groupby("geo_id").size()
    out["geo_level"] = "state"
    out["geo_name"] = out["geo_id"].map(names)
    out["basis"] = "self_id"
    out["year"] = YEARS
    out["source_id"] = SOURCE_ID
    out["note"] = out["geo_id"].map(
        lambda i: f"Afrobarometer R5-R9 and Arab Barometer V and VII pooled, n={int(n_by.get(i, 0))} "
                  "named this state; the national non-Muslim share and mix, on COD-PS 2022")
    cols = ["geo_id", "geo_level", "geo_name", "source_category", "count", "basis", "year",
            "source_id", "note"]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out[cols].to_csv(OUT, index=False, encoding="utf-8")
    total = int(out["count"].sum())
    print(f"\nwrote {OUT} ({len(out)} rows, {total:,} people, 18 states)")
    drawn = out.groupby("source_category")["count"].sum()
    for c in CATS:
        print(f"    {drawn[c] / total:9.4%}  {c}  ({int(drawn[c]):,})")


if __name__ == "__main__":
    main()
