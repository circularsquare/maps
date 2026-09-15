"""Togo — the 2022 census's national religion table, given a six-unit geography by the Afrobarometer.

Reads data/raw/afrobarometer/*.sav, data/geo/tg/tg_lookup.csv and tg_prefectures.csv, and UNSD's
Demographic Yearbook table 28 through `tools/oracle.py`; writes data/normalized/tg.csv.
`sources/tg.md` is the record; `sources/lr.py` is the construction this follows.

## TOGO ASKS, AND THE ANSWER HAS REACHED THE UN AND NOBODY ELSE

The 2022 census (RGPH-5, 8,095,498 people) asked religion, and INSEED sent the national table to
UNSD, which holds it by sex and by urban and rural: 15 categories, Catholic 1,688,420 down to
Adventist 17,960. INSEED itself has published three population booklets from that census and no
religion table at any level (sources.md §11w, §11aq). So the national level is a census count and
the geography below it has never been published.

## WHAT IS DRAWN, AND WHERE EACH NUMBER COMES FROM

    row margin      unit populations        2022 census, Livret 01 Tableaux 2 and 4   EXACT
    column margin   the religion totals     2022 census, UNSD table 28                 EXACT
    the interaction each unit's pattern     Afrobarometer R5-R9 pooled                 measured

Both margins are the same census. The religion table's 15 rows sum to 8,058,172, which is 37,326
short of the census count (23,099 urban and 14,228 rural, against Livret 01's Tableau 1); that
difference is a sixteenth column, `Not in the religion table`, so the two margins meet to the
person. It, `Not Stated` and `Unknown` are not drawn (`taxonomy/tg2022.py` EXCLUDED).

The table is fitted to both margins by `lr.ipf`, so the survey decides only how a unit's people
divide between the columns. Every row is `modelled` (§7b: nobody counted the cell).

## WHICH SURVEY ANSWER GIVES EACH CENSUS ROW ITS PATTERN

`SEED_FROM` pairs each census row with a survey group. A group that passes `cab.stability` at the
six units and is at least 1% of respondents gives its row its own pattern (`CARRIES`); every other
row is seeded at its national rate, and the fit then spreads it by what the carried rows leave
(`sources/lr.md` §9: a flat seed comes out as a mirror of the carried rows, not flat).

Two pairings are calls, both in `taxonomy/tg2022.py` REVIEW:
  * The card offered `Assembly of God` in rounds 5 and 6 only (149 answers) and nobody chose it
    after; the census row is seeded from `Evangelical` and that box together. The pair fails the
    split-half, so the call moves nothing drawn today.
  * The census's `Other Christians` is seeded from the small named churches (Orthodox, Church of
    Christ, the Celestial and Zionist boxes). It fails too.

## `Christian only` IS SPREAD OVER THE NAMED CHURCHES OF ITS OWN UNIT

The share naming no church runs 5.4, 7.0, 16.5, 15.2 and 7.2% by round, which is the fieldwork
(`playbooks/afrobarometer.md`). The census sets every church's level, so the only question is
whether that box distorts a church's pattern. It is 12 to 18% of Christians in every unit, so a
carried church's share is divided by one minus its unit's unnamed share (`unnamed_scale`), which
assumes the unnamed are shared out like the named in the same unit. `UNNAMED_RANGE_MAX` asserts it
stays even.

Usage:
    python sources/tg.py --fetch    nothing of its own; the Afrobarometer files are shared
                                    (`python sources/afrobarometer.py --fetch`), UNSD's through
                                    `python tools/oracle.py --fetch`
    python sources/tg.py            rebuild data/normalized/tg.csv
"""

import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "6")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.join(ROOT, "tools"))

import pandas as pd

import afrobarometer as ab
import cab
from cm import gkey, key
from lr import ipf, round_within_rows

LOOKUP = os.path.join(ROOT, "data", "geo", "tg", "tg_lookup.csv")
PREFS = os.path.join(ROOT, "data", "geo", "tg", "tg_prefectures.csv")
OUT = os.path.join(ROOT, "data", "normalized", "tg.csv")

COUNTRY = "Togo"
ROUNDS = [5, 6, 7, 8, 9]
SOURCE_ID = "tg_census2022_afrobarometer_2012_2022"
YEARS = "2022"
N_UNITS = 6
CENSUS_2022 = 8_095_498

# UNSD Demographic Yearbook table 28, Togo 2022, both sexes: Total, Urban, Rural. No `Total` row
# is forwarded, so the sum is checked against Livret 01 instead. Asserted against the oracle file.
CENSUS_RELIGION = {
    "Catholic": (1_688_420, 968_235, 720_184),
    "Muslim": (1_499_867, 786_126, 713_741),
    "Animist": (1_367_266, 170_176, 1_197_090),
    "No Religion": (759_447, 221_764, 537_683),
    "Assembly of God": (692_659, 293_701, 398_957),
    "Other Christians": (545_770, 342_023, 203_747),
    "Pentecostal": (496_978, 175_801, 321_177),
    "Evangelical Presbyterian Church": (283_513, 145_139, 138_374),
    "Other Religions": (231_073, 107_534, 123_539),
    "Not Stated": (187_619, 95_222, 92_397),
    "Baptist": (162_458, 65_835, 96_624),
    "Jehovah's Witnesses": (51_087, 31_261, 19_826),
    "Unknown": (47_100, 25_194, 21_906),
    "Methodist": (26_955, 13_657, 13_298),
    "Adventist": (17_960, 9_025, 8_935),
}
# Livret 01 Tableau 1: urban 3,473,792, rural 4,621,706.
CENSUS_URBAN, CENSUS_RURAL = 3_473_792, 4_621_706
RESIDUAL = "Not in the religion table"

# Every Togolese answer over the five rounds -> the survey group, keyed through `cm.key`.
GROUP = {
    "roman catholic": "Catholic",
    "muslim only": "Muslim", "sunni only": "Muslim", "shia": "Muslim", "ismaeli": "Muslim",
    "traditional/ethnic religion": "Traditional",
    "none": "None", "atheist": "None",
    "evangelical": "Evangelical or Assembly of God", "assembly of god": "Evangelical or Assembly of God",
    "assemblies of god": "Evangelical or Assembly of God",
    "pentecostal": "Pentecostal",
    "presbyterian": "Presbyterian",
    "baptist": "Baptist",
    "methodist": "Methodist",
    "jehovah's witness": "Jehovah's Witness",
    "seventh day adventist": "Adventist",
    "christian only": "Christian only",
    "orthodox": "Other Christian", "church of christ": "Other Christian",
    "zionist christian church": "Other Christian", "independent": "Other Christian",
    "lutheran": "Other Christian", "anglican": "Other Christian", "apostolic": "Other Christian",
    "new apostolic church": "Other Christian", "quaker/friends": "Other Christian",
    "mennonite": "Other Christian", "coptic": "Other Christian", "dutch reformed": "Other Christian",
    "other": "Other", "hindu": "Other", "bahai": "Other",
}
GROUPS = ["Catholic", "Muslim", "Traditional", "None", "Evangelical or Assembly of God", "Pentecostal",
          "Presbyterian", "Baptist", "Christian only", "Other Christian", "Methodist",
          "Jehovah's Witness", "Adventist", "Other"]
CHRISTIAN = ["Catholic", "Evangelical or Assembly of God", "Pentecostal", "Presbyterian", "Baptist",
             "Christian only", "Other Christian", "Methodist", "Jehovah's Witness", "Adventist"]

# census row -> the survey group that gives it a pattern (None: seeded at the national rate)
SEED_FROM = {
    "Catholic": "Catholic",
    "Muslim": "Muslim",
    "Animist": "Traditional",
    "No Religion": "None",
    "Assembly of God": "Evangelical or Assembly of God",
    "Other Christians": "Other Christian",
    "Pentecostal": "Pentecostal",
    "Evangelical Presbyterian Church": "Presbyterian",
    "Other Religions": "Other",
    "Not Stated": None,
    "Baptist": "Baptist",
    "Jehovah's Witnesses": "Jehovah's Witness",
    "Unknown": None,
    "Methodist": "Methodist",
    "Adventist": "Adventist",
    RESIDUAL: None,
}

# REGION label (through `cm.gkey`) -> unit.
NORM = {"lomecommune": "TG0305", "lome": "TG0305", "maritime": "TG03", "plateaux": "TG04",
        "centrale": "TG01", "kara": "TG02", "savanes": "TG05", "savane": "TG05"}
# The survey's prefecture spellings that COD-AB spells otherwise, and Lomé's arrondissements.
PREF_ALIAS = {"sousprefecturedemo": "plainedumo", "sousprefecturedemô": "plainedumo"}
LOCATION_ROUNDS = [6, 7, 9]
# Round 7 has one sampling point of 8 respondents labelled region Centrale and prefecture Cinkassé
# (Savanes). The weights are per sampling point and cannot say which is right; Centrale's and
# Savanes's round 7 totals (128 and 112) are off their usual 120 by exactly 8 each way, which
# leans to Savanes, and the answers (3 Muslim, 2 traditional, 3 Christian) fit either. It stays in
# REGION's Centrale, the stratum the weights were built in; 8 of 5,987.
EXPECTED_DISAGREE = {(7, "TG01", "cinkasse"): 8}

# What carries its own pattern, asserted against the split-half; set from its output 2026-09-15.
# Methodist and Jehovah's Witness pass the test too and are under the 1% floor.
CARRIES = ["Muslim", "Traditional", "None", "Pentecostal", "Presbyterian"]
UNNAMED_RANGE_MAX = 0.08


def check_census():
    """UNSD's rows for Togo 2022 must equal the transcription, and must fall short of Livret 01's
    count by the residual this module draws as its own column."""
    import oracle

    got = oracle.oracle(COUNTRY, 2022)
    if got is None:
        raise SystemExit("the oracle has no Togo 2022 rows; run `python tools/oracle.py --fetch`")
    for i, area in enumerate(("Total", "Urban", "Rural")):
        want = {k: v[i] for k, v in CENSUS_RELIGION.items()}
        have = {k: v for k, v in got.get(area, {}).items() if k != "Total"}
        if have != want:
            raise SystemExit(f"UNSD's Togo 2022 {area} rows differ from the transcription: "
                             f"{sorted(set(have.items()) ^ set(want.items()))}")
    tot = sum(v[0] for v in CENSUS_RELIGION.values())
    urb = sum(v[1] for v in CENSUS_RELIGION.values())
    rur = sum(v[2] for v in CENSUS_RELIGION.values())
    # UNSD's own rows miss by a person in three places (rounding in the forwarded table): urban plus
    # rural is 1 short of the total for Catholic and Assembly of God and 1 over for Baptist. Every
    # other row closes. The totals are what is drawn.
    off = {k: v[0] - v[1] - v[2] for k, v in CENSUS_RELIGION.items() if v[0] != v[1] + v[2]}
    if off != {"Catholic": 1, "Assembly of God": 1, "Baptist": -1}:
        raise SystemExit(f"UNSD's urban and rural rows no longer miss their totals by the known "
                         f"one person each: {off}")
    print(f"  UNSD table 28, Togo 2022: 15 rows equal the transcription in all three areas; they sum "
          f"to {tot:,}, {CENSUS_2022 - tot:,} short of the census ({CENSUS_URBAN - urb:,} urban, "
          f"{CENSUS_RURAL - rur:,} rural; UNSD's urban and rural rows miss their own totals by one "
          f"person in {sorted(off)})")
    if abs((CENSUS_URBAN - urb) + (CENSUS_RURAL - rur) - (CENSUS_2022 - tot)) > 1 or min(CENSUS_URBAN - urb, CENSUS_RURAL - rur) < 0:
        raise SystemExit("the residual does not split into non-negative urban and rural parts")
    return {**{k: v[0] for k, v in CENSUS_RELIGION.items()}, RESIDUAL: CENSUS_2022 - tot}


def report_card():
    """The boxes the grouping relies on must be value labels in every pooled round."""
    import pyreadstat

    watch = ["roman catholic", "christian only", "evangelical", "pentecostal", "presbyterian",
             "baptist", "muslim only", "none", "traditional/ethnic religion", "other"]
    print("\n  boxes on each round's card (value labels, not responses):")
    missing = []
    for rnd, name, _url, relname, _wt in ab.ROUNDS:
        if rnd not in ROUNDS:
            continue
        path = os.path.join(ab.AB_DIR, name)
        try:
            _d, meta = pyreadstat.read_sav(path, metadataonly=True)
        except pyreadstat._readstat_parser.ReadstatError:
            _d, meta = pyreadstat.read_sav(path, metadataonly=True, encoding="LATIN1")
        col = next(c for c in meta.column_names if c.upper() == relname.upper())
        have = {key(v) for v in meta.variable_value_labels.get(col, {}).values()}
        gone = [w for w in watch if w not in have]
        aog = sorted(h for h in have if "assembl" in h and "god" in h)
        print(f"    R{rnd}: {'every box present' if not gone else 'MISSING ' + ', '.join(gone)}; "
              f"Assembly of God labels {aog}")
        missing += [(rnd, w) for w in gone]
    if missing:
        raise SystemExit(f"boxes missing from a pooled round's card: {missing}")


def check_locations(df):
    """Rounds 6, 7 and 9 carry the prefecture (`LOCATION.LEVEL.1`); it must agree with REGION."""
    p = pd.read_csv(PREFS, dtype=str)
    if len(p) != 40:
        raise SystemExit(f"{PREFS} has {len(p)} prefectures, expected 40; re-run tg_geo.py")
    region_of = dict(zip(p["adm2_name"].map(gkey), p["adm1_pcode"]))
    print("\n  unit from REGION against the prefecture column (LOCATION.LEVEL.1):")
    seen = {}
    for rnd in LOCATION_ROUNDS:
        sub = df[df["round"] == rnd]
        loc = sub["LOCATION.LEVEL.1"].map(gkey).map(lambda s: PREF_ALIAS.get(s, s))
        unit = loc.map(lambda s: "TG0305" if s.startswith("arrondissement") else region_of.get(s))
        if unit.isna().any():
            raise SystemExit(f"R{rnd}: prefectures with no COD-AB match: "
                             f"{sorted(sub.loc[unit.isna(), 'LOCATION.LEVEL.1'].astype(str).unique())}")
        bad = unit != sub["geo_id"]
        for (g, l), n in sub[bad].assign(l=loc[bad]).groupby(["geo_id", "l"]).size().items():
            seen[(rnd, g, l)] = int(n)
        print(f"    R{rnd}: {len(sub):,} respondents, {int(bad.sum())} disagree")
    if seen != EXPECTED_DISAGREE:
        raise SystemExit(f"REGION and prefecture disagree differently from EXPECTED_DISAGREE: {seen}")
    print(f"    the one known disagreement holds: {EXPECTED_DISAGREE}, kept under REGION")


def swap_table(df, nm):
    """Madagascar's trap: traditional and none, early rounds against late, per unit."""
    print("\n  Traditional and None by unit, weighted % (the swap check, playbooks/afrobarometer.md):")
    for lab, rr in (("R5-R6", [5, 6]), ("R7", [7]), ("R8-R9", [8, 9])):
        s = df[df["round"].isin(rr)]
        t = s.groupby(["geo_id", "category"])["w"].sum().unstack(fill_value=0.0)
        t = 100 * t.div(t.sum(axis=1), axis=0)
        print(f"    {lab:<6}" + "  ".join(f"{nm[u][:8]:>8} {t.loc[u, 'Traditional']:4.1f}/{t.loc[u, 'None']:4.1f}"
                                         for u in t.index))


def unnamed_scale(df, nm):
    """1 / (1 - the unit's share of Christians answering `Christian only`), and the check that it is
    even enough across units for the adjustment to be a small one."""
    c = df[df["category"].isin(CHRISTIAN)]
    tot = c.groupby("geo_id")["w"].sum()
    un = c[c["category"] == "Christian only"].groupby("geo_id")["w"].sum().reindex(tot.index, fill_value=0)
    frac = un / tot
    print("\n  `Christian only` as a share of Christians: national "
          f"{100 * un.sum() / tot.sum():.1f}%; " + ", ".join(f"{nm[u]} {100 * v:.1f}%" for u, v in frac.sort_values().items()))
    if frac.max() - frac.min() > UNNAMED_RANGE_MAX:
        raise SystemExit(f"the unnamed share now ranges {100 * (frac.max() - frac.min()):.1f} points "
                         "across units; the per-unit spread of `Christian only` is no longer a small "
                         "adjustment, decide again")
    return 1.0 / (1.0 - frac)


def urban_witness(df, census):
    """UNSD's urban and rural rows against the survey's own urban share per group, each as a
    multiple of its source's overall urban share. The census row most rural must be Animist and the
    survey's most rural group (n >= 100) Traditional."""
    urb = df["URBRUR"].astype(str).str.casefold().str.startswith("urban")
    s_all = float((df["w"] * urb).sum() / df["w"].sum())
    s = df.assign(u=urb).groupby("category").apply(lambda d: (d["w"] * d["u"]).sum() / d["w"].sum(),
                                                   include_groups=False) / s_all
    n = df.groupby("category").size()
    c_all = sum(v[1] for v in CENSUS_RELIGION.values()) / sum(v[0] for v in CENSUS_RELIGION.values())
    c = pd.Series({k: v[1] / v[0] / c_all for k, v in CENSUS_RELIGION.items()})
    print("\n  urban share as a multiple of the whole (census rows from UNSD; survey groups pooled):")
    for row, grp in SEED_FROM.items():
        if row not in CENSUS_RELIGION:
            continue
        right = f"{grp:<32}{s[grp]:5.2f}x  n={int(n[grp])}" if grp else "(no survey group)"
        print(f"    {row:<34}{c[row]:5.2f}x    {right}")
    big = s[n[n >= 100].index]
    if c.idxmin() != "Animist" or big.idxmin() != "Traditional":
        raise SystemExit(f"the most rural census row is {c.idxmin()} and survey group {big.idxmin()}, "
                         "not Animist and Traditional")
    print("    asserted: Animist is the census's most rural row and Traditional the survey's most rural group")


def main():
    if "--fetch" in sys.argv:
        ab.fetch()
    print("=== Togo: the 2022 census's religion table on an Afrobarometer pattern ===")
    census = check_census()

    raw = ab.load(COUNTRY, expect_rounds=ROUNDS, regroup=True, extra=["LOCATION.LEVEL.1", "URBRUR"])
    raw["k"] = raw["category"].map(key)
    ct = pd.crosstab(raw["k"], raw["round"])
    ct["all"] = ct.sum(axis=1)
    print(f"\n  pooled: {len(raw):,} respondents; every answer as it arrives, keyed:")
    print(ct.sort_values("all", ascending=False).to_string())
    report_card()
    unmapped = sorted(set(raw["k"]) - set(GROUP))
    if unmapped:
        raise SystemExit(f"answers with no group: {unmapped}; add them to GROUP deliberately")
    df = raw.copy()
    df["raw_category"] = raw["category"]
    df["category"] = raw["k"].map(GROUP)
    ab.assert_one_wording(df, COUNTRY)

    tot = df.groupby("round")["w"].sum()
    byr = df.groupby(["category", "round"])["w"].sum().unstack(fill_value=0.0).div(tot, axis=1)
    print("\n  weighted share of all respondents by round (%):")
    print((100 * byr).round(1).reindex(GROUPS).to_string())

    # ---- units ----
    lut = pd.read_csv(LOOKUP, dtype={"geo_id": str})
    if len(lut) != N_UNITS:
        raise SystemExit(f"{LOOKUP} has {len(lut)} units, expected {N_UNITS}; re-run tg_geo.py")
    nm = dict(zip(lut["geo_id"], lut["name"]))
    pop = lut.set_index("geo_id")["pop"].astype("int64")
    if int(pop.sum()) != CENSUS_2022:
        raise SystemExit(f"tg_lookup.csv sums to {int(pop.sum()):,}, not {CENSUS_2022:,}")
    units = sorted(lut["geo_id"])
    df["geo_id"] = df["geo_raw"].map(gkey).map(NORM)
    if df["geo_id"].isna().any():
        raise SystemExit(f"REGION labels with no unit: "
                         f"{sorted(df.loc[df['geo_id'].isna(), 'geo_raw'].astype(str).unique())}")
    per_round = df.groupby("round")["geo_id"].nunique()
    if (per_round != N_UNITS).any():
        raise SystemExit(f"a pooled round does not sample all 6 units: {per_round.to_dict()}")
    if (df.groupby(["round", "geo_raw"])["geo_id"].nunique() > 1).any():
        raise SystemExit("one REGION label names two units in a round")
    check_locations(df)
    # NOT `ab.held_out`, which stops here: six units allow 720 orderings and 14 reach the observed
    # r = +0.943, because Lomé is sampled at 1.40x its 2022 share (the sampling frame is the 2010
    # census, when the old commune was a larger share of Togo) and Savanes at 0.73x. The function
    # says itself that under seven units it cannot carry a join. The REGION labels are names, not
    # codes, and `check_locations` puts every respondent of rounds 6, 7 and 9 but one sampling
    # point in the region its label names; that is this decode's witness.
    share_s = df.groupby("geo_id")["w"].sum() / df["w"].sum()
    share_p = pop / pop.sum()
    print("\n  weighted share of respondents against the 2022 census share (printed, not asserted):")
    for u in units:
        print(f"    {nm[u]:<22}{100 * share_s[u]:6.1f}%{100 * share_p[u]:7.1f}%  "
              f"{share_s[u] / share_p[u]:5.2f}x")

    # ---- the survey against the census, nationally ----
    nat = ab.national(df).reindex(GROUPS).fillna(0.0)
    shown = sum(v for k, v in census.items() if SEED_FROM[k])
    print("\n  national: survey group against the census row it seeds (% of those with a group):")
    for row, grp in SEED_FROM.items():
        if grp:
            cs = census[row] / shown
            print(f"    {row:<34}{100 * cs:6.2f}%   {grp:<32}{100 * nat[grp]:6.2f}%   {nat[grp] / cs:5.2f}x")
    print(f"    {'':<34}{'':>7}   {'Christian only':<32}{100 * nat['Christian only']:6.2f}%   (spread over its unit's churches)")
    urban_witness(df, census)
    swap_table(df, nm)

    # ---- quota, then the split-half ----
    dfw = df.rename(columns={"round": "wave", "category": "code"})[["wave", "geo_id", "code", "w"]]
    cab.assert_not_quota(dfw, COUNTRY, ROUNDS, unit_col="geo_id", cat_col="code")
    passed, _table = cab.stability(dfw, GROUPS, units, f"{N_UNITS} units")
    carries = [c for c in passed if nat[c] >= ab.ELIGIBLE_FLOOR and c != "Christian only"]
    if sorted(carries) != sorted(CARRIES):
        raise SystemExit(f"the split-half now selects {sorted(carries)}, not {sorted(CARRIES)}. Read the "
                         "table above, then edit CARRIES and the docstring deliberately.")

    # ---- seed, then fit to both census margins ----
    scale = unnamed_scale(df, nm)
    by = df.groupby(["geo_id", "category"])["w"].sum().unstack(fill_value=0.0)
    by = by.reindex(index=units, columns=GROUPS, fill_value=0.0)
    own = by.div(by.sum(axis=1), axis=0)
    cols = list(SEED_FROM)
    seed = pd.DataFrame(index=units, columns=cols, dtype=float)
    basis = {}
    for row, grp in SEED_FROM.items():
        if grp in carries:
            s = own[grp] * (scale if grp in CHRISTIAN else 1.0)
            seed[row] = s.reindex(units).to_numpy()
            basis[row] = f"the unit's own pattern from the survey's {grp} answers"
        else:
            seed[row] = census[row] / CENSUS_2022
            basis[row] = "no pattern of its own, seeded at the national rate"
    zero = [(nm[u], r) for u in units for r in cols if seed.loc[u, r] <= 0]
    print(f"\n  seed cells at zero (they stay zero): {zero or 'none'}")
    print("  fitting the unit x religion table to the census's two margins:")
    fitted = ipf(seed.mul(pop.reindex(units), axis=0), pop.reindex(units), pd.Series(census))
    counts = round_within_rows(fitted)
    if not (counts.sum(axis=1) == pop.reindex(units)).all():
        raise SystemExit("a unit's drawn total is not its census population")
    drift = counts.sum(axis=0) - pd.Series(census).reindex(counts.columns)
    print("    rounding drift against the census rows: "
          + ", ".join(f"{c} {int(d):+d}" for c, d in drift.items() if d))
    if drift.abs().max() > N_UNITS:
        raise SystemExit(f"rounding drift {drift.to_dict()} exceeds one person per unit")

    # ---- write ----
    n_by = df.groupby("geo_id").size()
    out = counts.stack().rename("count").reset_index()
    out.columns = ["geo_id", "source_category", "count"]
    out["geo_level"] = "region"
    out["geo_name"] = out["geo_id"].map(nm)
    out["basis"] = "self_id"
    out["year"] = YEARS
    out["source_id"] = SOURCE_ID
    out["note"] = out.apply(
        lambda r: ("2022 census unit population and national religion row, fitted with the pattern of "
                   f"Afrobarometer rounds 5-9 pooled (n={int(n_by[r.geo_id])} here); "
                   f"{basis[r.source_category]}"), axis=1)
    total = int(out["count"].sum())
    if total != CENSUS_2022:
        raise SystemExit(f"drawn {total:,} against the census {CENSUS_2022:,}")
    keep = ["geo_id", "geo_level", "geo_name", "source_category", "count", "basis", "year",
            "source_id", "note"]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out[keep].to_csv(OUT, index=False, encoding="utf-8")
    print(f"\nwrote {OUT} ({len(out)} rows, {total:,} people, {out['source_category'].nunique()} "
          f"categories, {out['geo_id'].nunique()} units)")

    # ---- what the file says, for note_public ----
    share = counts.div(counts.sum(axis=1), axis=0)
    surv = own.copy()
    print("\n  as drawn by unit (%), most Muslim first; survey's own share in brackets for carried rows:")
    show = ["Catholic", "Muslim", "Animist", "No Religion", "Assembly of God", "Pentecostal",
            "Evangelical Presbyterian Church", "Other Christians", "Baptist", "Other Religions"]
    print(f"    {'':<22}" + "".join(f"{r[:9]:>16}" for r in show))
    for u in share.sort_values("Muslim", ascending=False).index:
        cells = []
        for r in show:
            g = SEED_FROM[r]
            extra = f"({100 * surv.loc[u, g]:4.1f})" if g in carries else "      "
            cells.append(f"{100 * share.loc[u, r]:6.1f}{extra:>10}")
        print(f"    {nm[u]:<22}" + "".join(cells) + f"  pop {int(pop[u]):,} n={int(n_by[u])}")
    stated = [c for c in cols if c not in ("Not Stated", "Unknown", RESIDUAL)]
    st = counts[stated].sum(axis=1)
    print("\n  as a share of people with a stated religion (%):")
    for u in units:
        print(f"    {nm[u]:<22}" + ", ".join(f"{r} {100 * counts.loc[u, r] / st[u]:.1f}" for r in show))
    print(f"  national, stated religion only: " + ", ".join(
        f"{r} {100 * counts[r].sum() / st.sum():.2f}" for r in show))


if __name__ == "__main__":
    main()
