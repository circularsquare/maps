"""Yemen — Sunni, Zaydi and undifferentiated Muslims by governorate, from the Arab Barometer.

Reads data/raw/arabbarometer/*.sav and writes data/normalized/ye.csv.
`sources/ye.md` has the record in prose. `sources/arabbarometer.py` holds the construction and
`sources/iq.py` is the country this one is built on the shape of.

## WHAT THE FILES OFFER, AND WHY ONLY ONE WAVE IS DRAWN

Yemen is in four Arab Barometer files and has religion answers in two. Wave II (2011, 1,200
Yemenis) carries governorate and weight and an empty `q1012`, so `ab.wave_coverage` does not
offer it. That leaves:

  * **wave III**, Nov-Dec 2013, 1,200 respondents, card `Sunni / Shia`, nothing else;
  * **wave V**, 2018-2019, 2,400 respondents in 240 PSUs, an interviewer-logged item
    (`[DO NOT READ, LOG ANSWER]`) on a precoded list: 815 `Shafi'i`, 616 `Sunni`, 545 `Just a
    Muslim`, 410 on code 14 `Alawi`, and 14 others.

**Wave V is drawn and wave III is the witness.** Wave III's card has no box for a Muslim who
names no branch, so its 1,198 Muslims were all put into Sunni or Shia; pooled with wave V it
would put forced choices beside volunteered ones and the undifferentiated share would be
half-measured. That is Iraq's card rule (`sources/iq.py`, waves II and III omitted). What wave
III can do is independent replication: a different fieldwork, five years and a war earlier.

## CODE 14 `Alawi` IS ZAYDI, AND THE GEOGRAPHY IS WHAT SAYS SO

Arab Barometer's wave V list has no Zaydi code, and no Yemeni is logged under code 6 `Shia`.
The 410 on code 14 are Sa'dah 62%, Amran 50%, Raymah 38%, Dhamar 37% and Hajjah 34%, and 1 of
950 respondents across the nine southern and eastern governorates, including 1 of 120 in
Hadramawt, which rules out the Ba 'Alawi sayyids of the Wadi. That is the Zaydi highlands and
nothing else. Wave III's `Shia`, asked on a different card, lands in the same places (Sa'dah 39
of 40). `zaydi_geography` asserts both.

## THE SPLIT-HALF, REPLACED

Two waves on two cards cannot give §14.16's early-against-late split. Two tests replace it, and
a category is drawn on its own governorate shares only if it passes the ones that apply:

  1. **`replication`: wave III against wave V**, on the one quantity both cards measure, the
     Zaydi (Shia) share of the Muslims who name a branch. Spearman over the 21 governorates
     against `spearman_null`'s exact bar, plus a 20,000-draw unit permutation that handles the
     tied zeros, plus the chi-square in each wave. Decides Sunni and Zaydi.
  2. **`lits.stability` on wave V's 240 PSUs**: the median Spearman over 400 random PSU halves
     against a PSU-to-governorate regrouping null, with the chi-square and the one-PSU cap as
     vetoes. Decides every answer, and is the only test `Just a Muslim` gets.

**What neither can see**: the sect item is logged by the interviewer, and eight governorates
have nobody at all logged as `Just a Muslim` (Ta'iz 0 of 260, Lahj 0 of 100). A PSU split
inside a governorate replicates a team's logging habit perfectly. The interviewer column
`E2001B` is present and blank for every Yemeni, so it cannot be tested. `logging_zeros` prints
it and the note says it.

Usage:
    python sources/ye.py --fetch    download the Arab Barometer waves (~46 MB of zips)
    python sources/ye.py            rebuild data/normalized/ye.csv
"""

import os
import re
import sys
import unicodedata

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from scipy import stats

import arabbarometer as ab
import spearman_null

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LOOKUP = os.path.join(ROOT, "data", "geo", "ye", "ye_lookup.csv")
OUT = os.path.join(ROOT, "data", "normalized", "ye.csv")

COUNTRY = "Yemen"
FILE_WAVES = ["III", "V"]
DRAWN = "V"
WITNESS = "III"
SOURCE_ID = "ye_arabbarometer_2018_2019"
YEARS = "2018-2019"
SOCOTRA = "YE32"          # a governorate since December 2013; in neither wave's list
N_UNITS = 21

BLANK_WEIGHT_REASON = (
    "every one of the 32 also has no recorded gender, which the weight is post-stratified on; "
    "they are 29 respondents in five PSUs of Hajjah and Dhamar and 3 elsewhere, 22 of them "
    "logged Zaydi, so dropping them would take a tenth of both governorates' interviews out of "
    "the Zaydi column")

# ---------------------------------------------------------------------------------------
# THE COMPOSITION: one category per respondent, from `Q1012` and the logged sect.
# ---------------------------------------------------------------------------------------
# Every key must occur in the pool, or `compose` stops.
SECT_READ = {
    # wave V
    "Shafi’i": "Sunni",       # the Sunni school of Yemen; volunteered, not a separate answer
    "Alawi": "Zaydi",         # code 14; see the module docstring and zaydi_geography
    "refused": "Muslim, denomination not given",
    "don't know": "Muslim, denomination not given",
    "other": "Muslim, other denomination",
    # wave III
    "Refuse": "Muslim, denomination not given",
}
# Answers that leave the universe of the drawn wave.
DROPPED = {
    "Atheist": ("1 respondent in 2,400, in Sana'a governorate, who also refused the sect item. "
                "Drawn at the national rate it would place about 14,500 people with no religion "
                "evenly over all 21 governorates on one interview; Egypt and Iraq dropped "
                "theirs."),
}

# What is drawn on its own governorate shares, asserted.
CARRIES = ["Just a Muslim", "Sunni", "Zaydi"]
# Answers the PSU test cannot rank at all; they are under the 1% floor either way.
UNTESTED = {}
# Sub-floor answers drawn at their governorate shares anyway, because `ab.build` spreads a tail
# only into room the carried answers leave, and 15 of the 21 governorates returned nothing but
# the three carried answers (no room at all). Both sit on `islam` with `Just a Muslim`, so at
# the node the map is the same as folding them into it; the floor exists to stop a small
# answer's noise being drawn as its own colour, and these have no colour of their own.
# Asserted against the mapping below.
SAME_NODE = {"Muslim, denomination not given": "Just a Muslim",
             "Muslim, other denomination": "Just a Muslim"}

# The guard on the Zaydi share of the branch-namers in the drawn wave. The outside figures are
# the US government's 35% Zaydi of the population and ACLED's 45% of Muslims (State Department,
# 2023 Report on International Religious Freedom: Yemen). The survey reads well under both;
# the band only catches a re-release that re-levels the pool.
ZAYDI_OF_NAMED_BAND = (0.10, 0.50)

# ---------------------------------------------------------------------------------------
# THE GOVERNORATES
# ---------------------------------------------------------------------------------------
# Every `Q1` label, folded by `key()`, -> COD-AB p-code. `sanaa` is the governorate (YE23) in
# waves II and V. **Wave III labels both 10503 and 10513 `Sana'a`**, so wave III is decoded on
# the CODE and its names are the witness; see `decode_iii`.
NORM = {
    "ibb": "YE11", "abyan": "YE12", "amanatalasimah": "YE13", "albayda": "YE14",
    "taizz": "YE15", "aljawf": "YE16", "hajjah": "YE17", "alhudaydah": "YE18",
    "hadramaut": "YE19", "hadhramaut": "YE19", "dhamar": "YE20", "shabwah": "YE21",
    "sadah": "YE22", "saada": "YE22", "sanaa": "YE23", "aden": "YE24", "adan": "YE24",
    "lahij": "YE25", "lahj": "YE25", "marib": "YE26", "almahwit": "YE27", "almahrah": "YE28",
    "amran": "YE29", "addali": "YE30", "raymah": "YE31",
}
# Waves II and III number governorates in the CSO's own order, which is the p-code order:
# 10500 + n is `YE{n + 10}`.
CSO_CODE_BASE = 10500
WAVE_II_SAV = "ABII_English.sav"
# Wave V numbers them 220001-220021, the former Yemen Arab Republic first and the six
# governorates of the former PDRY last (220016-220021). A permutation of NORM across the 1990
# border breaks that.
FORMER_PDRY = {"YE24", "YE12", "YE28", "YE19", "YE25", "YE21"}
V_PDRY_CODES = range(220016, 220022)

# The bands `zaydi_geography` asserts, which partition the 21.
HIGHLANDS = ["YE22", "YE29", "YE17", "YE20", "YE23", "YE13", "YE16", "YE27", "YE31"]
SOUTH_EAST = ["YE24", "YE25", "YE12", "YE21", "YE19", "YE28", "YE30", "YE15", "YE14"]
MIDDLE = ["YE11", "YE18", "YE26"]
HADRAMAWT, SADAH = "YE19", "YE22"

_ORDINAL = re.compile(r"^\s*\d+\s*[.)]\s*")


def key(s):
    s = unicodedata.normalize("NFKC", str(s)).replace("’", "'")
    return re.sub(r"[^a-z]", "", _ORDINAL.sub("", s).lower())


def compose(df):
    """One category per respondent: a non-Muslim keeps the religion answer, a Muslim the sect."""
    rel = df["category"].astype(str).str.strip()
    sect = df["sect"].where(df["sect"].notna()).astype(object)
    muslim = rel.eq("Muslim")
    for raw in SECT_READ:
        if not (sect == raw).any():
            raise SystemExit(f"SECT_READ names {raw!r}, which this pool does not contain")
    out = rel.copy()
    out[muslim] = sect[muslim].map(lambda s: SECT_READ.get(s, s))
    out[muslim & sect.isna()] = "Muslim, denomination not given"
    out[~muslim & rel.eq("Other")] = "Other religion"
    if out.isna().any():
        raise SystemExit(f"{int(out.isna().sum())} respondents came out with no category")
    print("\n  composed category, by wave:")
    print(pd.crosstab(out, df["wave"]).to_string())
    return out


def decode_iii(g):
    """Wave III on its CODE, with its names and wave II's names as the two witnesses."""
    import pyreadstat

    sub = g["wave"] == WITNESS
    code = pd.to_numeric(g.loc[sub, "geo_code"], errors="coerce").astype(int)
    by_code = code.map(lambda c: f"YE{c - CSO_CODE_BASE + 10}")
    by_name = g.loc[sub, "geo_raw"].map(key).map(NORM)
    bad = g.loc[sub][by_code != by_name]
    bad_codes = sorted(set(pd.to_numeric(bad["geo_code"]).astype(int)))
    print(f"\n  wave III on its code: {int(sub.sum())} respondents; the names disagree on "
          f"{len(bad)}, all at codes {bad_codes}")
    if bad_codes != [10503] or set(bad["geo_raw"]) != {"Sana'a"}:
        raise SystemExit("wave III's names and codes disagree somewhere other than the known "
                         "duplicate `Sana'a` on 10503; STOP")
    _df, meta = pyreadstat.read_sav(os.path.join(ab.AB_DIR, WAVE_II_SAV), metadataonly=True)
    labels = meta.variable_value_labels.get(ab._col(pd.DataFrame(columns=meta.column_names),
                                                    "q1"), {})
    ii = {int(c): NORM.get(key(l)) for c, l in labels.items() if 10501 <= c <= 10521}
    wrong = {c: v for c, v in ii.items() if v != f"YE{c - CSO_CODE_BASE + 10}"}
    if len(ii) != N_UNITS or wrong:
        raise SystemExit(f"wave II's own labels for codes 10501-10521 do not decode to the CSO "
                         f"order: {wrong or len(ii)}")
    print("    wave II labels the same 21 codes by name, 10503 as `Amanat al Asimah`, and every "
          "one decodes to the CSO order")
    return by_code


def decode(g):
    g = g.copy()
    g["geo_id"] = g["geo_raw"].map(key).map(NORM)
    g.loc[g["wave"] == WITNESS, "geo_id"] = decode_iii(g)
    unmapped = sorted(g.loc[g["geo_id"].isna(), "geo_raw"].astype(str).unique())
    if unmapped:
        raise SystemExit(f"Q1 labels with no governorate: {unmapped}")
    v = g["wave"] == DRAWN
    dup = g[v].groupby("geo_id")["geo_raw"].nunique()
    if (dup > 1).any():
        raise SystemExit(f"wave V uses two labels for one governorate: {dup[dup > 1].to_dict()}")
    pdry = set(g.loc[v & pd.to_numeric(g["geo_code"]).isin(V_PDRY_CODES), "geo_id"])
    if pdry != FORMER_PDRY:
        raise SystemExit(f"wave V's codes 220016-220021 decode by name to {sorted(pdry)}, not the "
                         f"six former PDRY governorates {sorted(FORMER_PDRY)}")
    print("  wave V on its names: codes 220016-220021 are the six governorates of the former "
          "PDRY, as its code list orders them")
    for w, sub in g.groupby("wave"):
        if sub["geo_id"].nunique() != N_UNITS:
            raise SystemExit(f"wave {w} covers {sub['geo_id'].nunique()} governorates, not 21")
    return g


def zaydi_geography(g, names):
    """Is code 14 (and wave III's `Shia`) the Zaydi highlands? Asserted, per wave."""
    band = {**{u: "highlands" for u in HIGHLANDS}, **{u: "south/east" for u in SOUTH_EAST},
            **{u: "middle" for u in MIDDLE}}
    if sorted(band) != sorted(set(g["geo_id"])):
        raise SystemExit("the three bands do not partition the 21 governorates")
    print("\n  the Zaydi answer (wave V code 14 `Alawi`; wave III `Shia`) by governorate:")
    tabs = {}
    for w, label in ((DRAWN, "Zaydi"), (WITNESS, "Shia")):
        sub = g[g["wave"] == w]
        t = sub.assign(z=sub["category"].eq(label)).groupby("geo_id")["z"].agg(n="size", z="sum")
        tabs[w] = t
        by_band = t.groupby(t.index.map(band))[["n", "z"]].sum()
        share = by_band["z"] / by_band["n"]
        print(f"    wave {w:<4} highlands {share['highlands']:.1%}, middle {share['middle']:.1%}, "
              f"south and east {share['south/east']:.2%}; Hadramawt "
              f"{int(t.loc[HADRAMAWT, 'z'])} of {int(t.loc[HADRAMAWT, 'n'])}; highest "
              f"{names[(t['z'] / t['n']).idxmax()]}")
        if share["south/east"] > 0.02 or share["highlands"] < 0.25:
            raise SystemExit(f"wave {w}'s {label!r} is not concentrated in the Zaydi highlands; "
                             "the reading of it as Zaydi does not hold. STOP")
        if t.loc[HADRAMAWT, "z"] / t.loc[HADRAMAWT, "n"] > 0.02:
            raise SystemExit("code 14 reaches Hadramawt, so it may be the Ba 'Alawi sayyids")
        if (t["z"] / t["n"]).idxmax() != SADAH:
            raise SystemExit(f"Sa'dah is not the most Zaydi governorate in wave {w}")
    both = pd.DataFrame({"V": tabs[DRAWN]["z"] / tabs[DRAWN]["n"] * 100,
                         "III": tabs[WITNESS]["z"] / tabs[WITNESS]["n"] * 100,
                         "n V": tabs[DRAWN]["n"], "n III": tabs[WITNESS]["n"]})
    both.index = both.index.map(names)
    print(both.sort_values("V", ascending=False).round(1).to_string())


def branch_share(sub, zaydi_label):
    named = sub[sub["category"].isin([zaydi_label, "Sunni"])]
    w = named.assign(z=named["w"] * named["category"].eq(zaydi_label))
    t = w.groupby("geo_id").agg(z=("z", "sum"), w=("w", "sum"))
    counts = named.groupby("geo_id")["category"].agg(
        n="size", z=lambda s: int(s.eq(zaydi_label).sum()))
    return t["z"] / t["w"], counts


def replication(g, names, n_perm=20000, seed=0):
    """Wave III against wave V on the Zaydi share of the branch-namers. Decides Sunni and Zaydi."""
    s5, c5 = branch_share(g[g["wave"] == DRAWN], "Zaydi")
    s3, c3 = branch_share(g[g["wave"] == WITNESS], "Shia")
    j = pd.concat([s3.rename("III"), s5.rename("V")], axis=1)
    if len(j.dropna()) != N_UNITS:
        raise SystemExit(f"{len(j.dropna())} governorates have branch-namers in both waves")
    rho = stats.spearmanr(j["III"], j["V"]).statistic
    bar, how = spearman_null.critical_rho(N_UNITS)
    rng = np.random.default_rng(seed)
    a, b = j["III"].to_numpy(), j["V"].to_numpy()
    perm = np.array([stats.spearmanr(a, rng.permutation(b)).statistic for _ in range(n_perm)])
    p_perm = (1 + int((perm >= rho).sum())) / (1 + n_perm)
    chis = {w: stats.chi2_contingency(np.array([c["z"], c["n"] - c["z"]]))[1]
            for w, c in ((WITNESS, c3), (DRAWN, c5))}
    print(f"\n  REPLICATION, wave III (2013, Sunni/Shia card) against wave V (2018-19, logged):")
    print(f"    Zaydi share of the branch-namers over {N_UNITS} governorates: Spearman "
          f"{rho:+.3f}; exact bar +{bar:.4f} ({how}), exact p {spearman_null.exact_p(rho, N_UNITS):.2g}")
    print(f"    {n_perm:,}-draw permutation p {p_perm:.2g} (null 95th {np.quantile(perm, 0.95):+.3f}),"
          f" which reads the tied zeros correctly{spearman_null.ties_note(a, b)}")
    print(f"    chi-square across the governorates: wave III p={chis[WITNESS]:.2g}, "
          f"wave V p={chis[DRAWN]:.2g}")
    loo = sorted(((names[u], stats.spearmanr(j.drop(index=u)["III"],
                                             j.drop(index=u)["V"]).statistic) for u in j.index),
                 key=lambda t: t[1])
    bar20, _ = spearman_null.critical_rho(N_UNITS - 1)
    print(f"    leave-one-out: {loo[0][1]:+.3f} without {loo[0][0]} to {loo[-1][1]:+.3f} without "
          f"{loo[-1][0]}, against +{bar20:.4f} at 20 units")
    ok = rho >= bar and p_perm < 0.05 and max(chis.values()) < 0.05
    print(f"    verdict: {'REPLICATES' if ok else 'DOES NOT REPLICATE'}")
    return ok, rho, p_perm


def psu_test(g5, nat, units):
    import lits

    f = g5.rename(columns={"category": "code", "cluster": "PSU_number"})
    return lits.stability(f, nat, units, unit_col="geo_id", untested=UNTESTED)


def logging_zeros(g5, names):
    t = g5.assign(j=g5["category"].eq("Just a Muslim")).groupby("geo_id")["j"].agg(["size", "sum"])
    zero = t[t["sum"] == 0]
    print(f"\n  governorates where nobody was logged `Just a Muslim` in wave V: {len(zero)} of "
          f"{len(t)}, {int(zero['size'].sum()):,} respondents: "
          + ", ".join(f"{names[u]} 0/{int(n)}" for u, n in zero["size"].items()))
    return zero


def main():
    if "--fetch" in sys.argv:
        ab.fetch()
    ab.unzip()

    print("=== Arab Barometer, Yemen ===")
    df = ab.load(COUNTRY, expect_waves=FILE_WAVES, waves=FILE_WAVES,
                 extra={"sect": ("q1012a",)}, raw={"cluster": ("psu", "bid")},
                 blank_weights={DRAWN: BLANK_WEIGHT_REASON})
    print(f"\n  pooled: {len(df):,} respondents with a religion answer")
    print(pd.crosstab(df["category"], df["wave"]).to_string())
    print("\n  the sect item's RAW answers by wave:")
    print(pd.crosstab(df["sect"].fillna("(none)"), df["wave"]).to_string())

    df["category"] = compose(df)
    ab.assert_one_wording(df, COUNTRY)

    lut = pd.read_csv(LOOKUP, dtype={"geo_id": str})
    names = dict(zip(lut["geo_id"], lut["name"]))
    g = decode(df[df["geo_raw"].notna()])
    units = sorted(u for u in lut["geo_id"] if u != SOCOTRA)
    if sorted(set(g["geo_id"])) != units:
        raise SystemExit(f"the survey's governorates are not the lookup's 21 without Socotra: "
                         f"{sorted(set(g['geo_id']) ^ set(units))}")
    pop = lut.set_index("geo_id")["pop"]

    zaydi_geography(g, names)

    for cat, why in DROPPED.items():
        n = int(((g["wave"] == DRAWN) & (g["category"] == cat)).sum())
        if not n:
            raise SystemExit(f"DROPPED names {cat!r}, which wave V no longer contains")
        print(f"\n  dropping {n} wave V respondent(s) who answered {cat!r}: {why}")
        g = g[~((g["wave"] == DRAWN) & (g["category"] == cat))]

    g5 = g[g["wave"] == DRAWN].copy()
    named = g5["category"].isin(["Zaydi", "Sunni"])
    zon = g5.loc[g5["category"].eq("Zaydi"), "w"].sum() / g5.loc[named, "w"].sum()
    print(f"\n  wave V: of the Yemenis who name a branch, {zon:.1%} are Zaydi (weighted), against "
          "the US government's\n    35% of the population and ACLED's 45% of Muslims (State "
          "Department IRF 2023). Nobody has counted.")
    if not ZAYDI_OF_NAMED_BAND[0] <= zon <= ZAYDI_OF_NAMED_BAND[1]:
        raise SystemExit(f"the Zaydi share of the branch-namers is {zon:.1%}, outside the band")

    ab.held_out(g5, pop[units], COUNTRY, pop_source="the Task Force's 2025 estimate")
    print("  (wave III, the witness):")
    ab.held_out(g[g["wave"] == WITNESS], pop[units], COUNTRY,
                pop_source="the Task Force's 2025 estimate")

    # Lebanon's check, on the one pair there is, with wave III's `Shia` read as the same box.
    q = g.copy()
    q.loc[(q["wave"] == WITNESS) & q["category"].eq("Shia"), "category"] = "Zaydi"
    ab.assert_not_quota(q, COUNTRY, waves=FILE_WAVES)

    logging_zeros(g5, names)

    nat = ab.national(g5)
    replicates, _rho, _p = replication(g, names)
    psu_carries = psu_test(g5, nat, units)
    large = [c for c in psu_carries if nat[c] >= ab.ELIGIBLE_FLOOR
             and (c not in ("Sunni", "Zaydi") or replicates)]
    print(f"\n  carries its own governorate shares: {large}")
    if sorted(large) != sorted(CARRIES):
        raise SystemExit(f"the tests select {sorted(large)}, not {sorted(CARRIES)}; read the "
                         "tables above and change CARRIES deliberately")
    sys.path.insert(0, os.path.join(ROOT, "taxonomy"))
    from ye2019 import resolve
    for c, host in SAME_NODE.items():
        if resolve(c) is None or resolve(c) != resolve(host) or host not in large:
            raise SystemExit(f"SAME_NODE says {c!r} shares a node with the carried {host!r}; "
                             f"taxonomy/ye2019.py maps them to {resolve(c)} and {resolve(host)}")
    if sorted(set(nat.index) - set(large)) != sorted(SAME_NODE):
        raise SystemExit(f"the sub-floor answers are {sorted(set(nat.index) - set(large))}, not "
                         f"SAME_NODE's {sorted(SAME_NODE)}")
    large = large + list(SAME_NODE)
    small = []
    print(f"  and {list(SAME_NODE)} at their governorate shares too, on the same node as "
          "`Just a Muslim`")

    out = ab.build(g5, nat, large, small, pop, units, unit_noun="governorate")
    n_by = g5.groupby("geo_id").size()
    out["geo_level"] = "governorate"
    out["geo_name"] = out["geo_id"].map(names)
    out["basis"] = "self_id"
    out["year"] = YEARS
    out["source_id"] = SOURCE_ID
    out["note"] = out.apply(
        lambda r: (f"Arab Barometer wave V, n={int(n_by[r.geo_id])} in this governorate; "
                   f"{r.basis_note} applied to the Population Task Force's 2025 governorate "
                   "estimate"), axis=1)
    total = int(out["count"].sum())
    want = int(pop[units].sum())
    if total != want:
        raise SystemExit(f"drawn {total:,} against the Task Force's {want:,} without Socotra")

    cols = ["geo_id", "geo_level", "geo_name", "source_category", "count", "basis", "year",
            "source_id", "note"]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out[cols].to_csv(OUT, index=False, encoding="utf-8")
    print(f"\nwrote {OUT} ({len(out)} rows, {total:,} people; Socotra's "
          f"{int(pop[SOCOTRA]):,} not drawn)")
    drawn = out.groupby("source_category")["count"].sum().sort_values(ascending=False)
    for cat, n in drawn.items():
        print(f"    {n / total * 100:6.2f}%  {cat}  ({n:,})")
    wide = out.pivot_table(index="geo_name", columns="source_category", values="count",
                           aggfunc="sum").fillna(0)
    wide = (wide.div(wide.sum(axis=1), axis=0) * 100).round(1)
    wide["n"] = wide.index.map({names[u]: int(n_by[u]) for u in units})
    wide["people"] = wide.index.map({names[u]: int(pop[u]) for u in units})
    print(wide.sort_values("Zaydi", ascending=False).to_string())


if __name__ == "__main__":
    main()
