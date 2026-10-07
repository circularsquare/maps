"""Namibia — religion in the 14 regions from six pooled Afrobarometer rounds, on the 2023 census.

Reads data/raw/afrobarometer/*.sav, data/geo/na/na_lookup.csv, and as witnesses
data/raw/na/FR298.pdf and FR204.pdf (the 2013 and 2006-07 Demographic and Health Survey reports);
writes data/normalized/na.csv. `sources/na.md` is the record; `sources/cm.py` is the construction
this follows, `sources/mg.py` the none-or-traditional pair.

## NAMIBIA DOES NOT ASK

The 2001, 2011 and 2023 census forms carry no religion item (`sources.md` §11aq), so this is the
Nigeria construction (ask/answered/010-ng): each unit at its own survey mix and its census
population, the national level computed and never fitted, every row `modelled`.

    row margin      region populations      2023 census, Main Report Table 2.2   EXACT
    the composition each unit's own mix     Afrobarometer R4-R9 (R5-R6 for two   measured
                                            churches, see below)
    the national level                      neither                              computed

## THIRTEEN MIXES ON FOURTEEN REGIONS

Rounds 4 and 5 sample Kavango whole (it was split into Kavango East and West in 2013) and call
Zambezi `Caprivi`; rounds 6 to 9 have the 14 regions. Round 4's `DISTRICT` column could place its
Kavango respondents, round 5 has nothing below the region, and `cab.stability` refuses an empty
(round, unit) cell. So the survey is read at 13 units with Kavango whole, and both Kavango regions
take that one mix, each at its own census population (`KAVANGO`). The two are alike (Catholic-led,
Lutheran second, in every round that splits them).

## THE LUTHERANS DO NOT HOLD THEIR LEVEL, AND TWO ROUNDS AGREE WITH THE DHS

`Lutheran` runs 21.5, 42.8, 41.9, 22.2, 20.3, 23.5% over rounds 4 to 9. Where the others go is
visible region by region (`levels_by_round` prints it): in round 4 the Evangelical Lutheran Church in
Namibia (ELCIN) is half coded `Evangelical` (Omusati 31%, Oshikoto 25%), in round 7 `Evangelical`
and `Christian only` take Ohangwena's Lutherans (13% Lutheran, 27% and 30%), and in round 8
`Anglican` jumps to 34% in Omusati and 30% in Oshikoto, where every other round puts it at 4-14%.
So the Lutheran level is set by the coding, and the playbook's rule applies: a church is drawn
only with an outside witness to its level.

The witness is the 2013 Namibia DHS (NSA and the Ministry of Health), whose respondent table
(Table 3.1, report p.30) names ELCIN apart: **44.0% of women and 43.4% of men aged 15-49**, 43.8%
together. Rounds 5 and 6 put all Lutherans at 42.8% and 41.9%; the other four rounds at 20-24%.
So the Lutheran and Anglican shares are taken from rounds 5 and 6 only, the two rounds whose coding
reproduces the DHS, as a share of the pool they trade with (`Christian, other`: every Christian
answer except Catholic, Adventist and Pentecostal), and that pool's level and geography come from
all six rounds. Within rounds 5 and 6, Lutheran ranks the 13 units alike in both (+0.725, p=0.007)
and so does Anglican (+0.578, p=0.02). `LUTHERAN_DHS_GAP_MAX` asserts the two rounds still sit by
the DHS, `LUTHERAN_SWING_MIN` that the other four still do not.

**Catholics, Adventists and Pentecostals hold their level and come from all six rounds.** Roman
Catholic runs 19.6-26.8% by round and the DHS has 21.5% (19.6% of women, 25.9% of men; the 2006-07
DHS 22.4%). Adventists run 1.9-3.9% and the DHS has 4.5%, so they are drawn short (`sources/na.md`
§4). Pentecostals run 2.4-5.1% and nothing outside names them. All three pass the split-half.

## NONE AND TRADITIONAL ARE ONE BOX, SPLIT AT THE POOLED RATIO

The card offers both in every round, and they trade places: rounds 4-6 give traditional religion
2.3% and no religion 1.1%, rounds 7-9 0.3% and 4.2%. Kunene, where Himba communities keep the
ancestral fire, is 43% traditional in rounds 5 and 6 and 16-18% none in rounds 7 and 9 with no
traditional at all. That is Madagascar's swap, so the two are placed as one box and split at one
national ratio. Madagascar's ratio came from its late rounds with two DHS surveys as witnesses;
Namibia's DHS offers no traditional box, and the early and late rounds disagree in opposite
directions (none 33.5% of the pair against 94.5%), so the ratio is the six rounds pooled, 68.1%
none (`NONE_FRACTION`).

## `Other` IS NOT PLACED

It passes the split-half (+0.446), but 105 of its 108 answers are in rounds 7 to 9 (3.7, 3.1, 1.8%
against 0.0, 0.2, 0.1%), largest in Kunene, Omaheke and Otjozondjupa where the early rounds had
traditional religion instead. A share that depends on which rounds are in is Cameroon's `Other`; it
goes in the tail with the five Muslim answers. The tail is flat (spec §12's 2x rule: the residual
would draw Muslims at 4.56x in Kunene), so both are drawn at their national share in every region
and the placed categories are scaled to fill what is left.

Usage:
    python sources/na.py --fetch    the two DHS reports (~7 MB); the Afrobarometer files are shared
                                    (`python sources/afrobarometer.py --fetch`)
    python sources/na.py            rebuild data/normalized/na.csv
"""

import os
import re
import sys

os.environ.setdefault("OMP_NUM_THREADS", "6")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

import afrobarometer as ab
import cab
from tz import key

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LOOKUP = os.path.join(ROOT, "data", "geo", "na", "na_lookup.csv")
RAW = os.path.join(ROOT, "data", "raw", "na")
OUT = os.path.join(ROOT, "data", "normalized", "na.csv")

COUNTRY = "Namibia"
ROUNDS = [4, 5, 6, 7, 8, 9]
CHURCH_ROUNDS = [5, 6]
RECENT = [8, 9]
SOURCE_ID = "na_afrobarometer_2008_2021_census2023"
YEARS = "2008-2021"

N_REGIONS = 14
CENSUS_2023 = 3_022_401
KAV = "KAV"
KAVANGO = ("NA05", "NA14")

PAIR = "None or traditional"
NONE = "None"
TRAD = "Traditional/ethnic religion"
POOL = "Christian, other"
# The six-round categories. Lutheran and Anglican are carved out of POOL afterwards.
CATEGORIES = ["Roman Catholic", "Seventh Day Adventist", "Pentecostal", POOL, PAIR, "Muslim", "Other"]
CARVED = ["Lutheran", "Anglican"]
OUT_CATEGORIES = ["Lutheran", "Roman Catholic", "Anglican", "Seventh Day Adventist", "Pentecostal",
                  "Other Christian", NONE, TRAD, "Muslim", "Other"]

# Every answer Namibians give over the six rounds, keyed through `tz.key()` (which drops the
# bracketed gloss rounds 8-9 add to `Christian only` and `Pentecostal`), -> its six-round
# category. Named one by one: an answer that fell through a default would be dropped silently.
GROUP = {
    "roman catholic": "Roman Catholic", "seventh day adventist": "Seventh Day Adventist",
    "pentecostal": "Pentecostal",
    "lutheran": POOL, "anglican": POOL, "christian only": POOL, "evangelical": POOL,
    "methodist": POOL, "baptist": POOL, "jehovah's witness": POOL, "church of christ": POOL,
    "dutch reformed": POOL, "orthodox": POOL, "zionist christian church": POOL,
    "independent": POOL, "coptic": POOL, "mennonite": POOL, "presbyterian": POOL,
    "calvinist": POOL, "mormon": POOL,
    "none": PAIR, "atheist": PAIR, "traditional/ethnic religion": PAIR,
    "muslim only": "Muslim", "sunni only": "Muslim",
    "other": "Other", "bahai": "Other",
}
PAIR_BOX = {"none": NONE, "atheist": NONE, "traditional/ethnic religion": TRAD}

# Every REGION label over the six rounds, as bare letters, -> the survey unit.
NORM = {
    "caprivi": "NA01", "zambezi": "NA01", "erongo": "NA02", "hardap": "NA03", "karas": "NA04",
    "kavango": KAV, "kavangoeast": KAV, "kavangowest": KAV, "khomas": "NA06", "kunene": "NA07",
    "ohangwena": "NA08", "omaheke": "NA09", "omusati": "NA10", "oshana": "NA11",
    "oshikoto": "NA12", "otjozondjupa": "NA13",
}

# What the six-round split-half places, asserted so a change in the data stops the build (2026-10-03).
CARRIES = ["Roman Catholic", "Seventh Day Adventist", "Pentecostal", POOL, PAIR]
NOT_PLACED = {"Other": "105 of its 108 answers are in rounds 7-9; a pooled share would measure "
                       "which rounds are in"}
OTHER_EARLY_MAX = 12            # `Other` answers allowed in rounds 4-6 before NOT_PLACED is re-read
# Spec §12's small-category rule, measured 2026-10-03: the residual would draw Muslim at 4.56x its
# national share in Kunene, where the survey found none (Kunene's remainder is its late-round
# `Other` answers), so the tail is flat: Muslim and Other at their national shares everywhere.
TAIL_FLAT = True
# The carved churches must pass the rounds-5-6 split-half.
CARVED_PASS = ["Lutheran", "Anglican"]
# Levels. The DHS 2013 ELCIN share, women and men aged 15-49 together, is the Lutheran witness.
LUTHERAN_DHS_GAP_MAX = 0.03     # rounds 5 and 6 each within 3 points of it
LUTHERAN_SWING_MIN = 0.15       # and every other round at least 15 points below it
CHURCH_RANGE_MAX = 0.08         # Catholic 7.2 points, Adventist 2.0, Pentecostal 2.7 by round
CATHOLIC_DHS_GAP_MAX = 0.03
# Lutheran as drawn (40.5% on 2026-10-03) against the DHS's ELCIN alone (43.9%): the pool's level is
# all six rounds', a little under rounds 5-6's, so the drawn share sits a few points short.
LUTHERAN_DRAWN_GAP_MAX = 0.05
LEVEL_GAP_MAX = 0.035           # spec §12 (Norway): pooled against rounds 8-9
NONE_FRACTION = 0.681

# The DHS reports, re-read from the PDFs: (file, PDF page, row label, women %, women n, men %, men n).
DHS = {
    "2013": ("FR298.pdf", 52, {
        "Roman Catholic": (19.6, 1802, 25.9, 1041), "Protestant/Anglican": (21.2, 1947, 12.7, 511),
        "ELCIN": (44.0, 4035, 43.4, 1745), "Seventh-Day Adventist": (4.8, 436, 4.0, 161),
        "No religion": (1.1, 105, 1.8, 72), "Other": (9.0, 827, 12.0, 483)}),
    "2006-07": ("FR204.pdf", 53, {
        "Roman Catholic": (20.9, 2053, 26.3, 1028), "Protestant": (77.0, 7547, 70.3, 2754),
        "No religion": (1.4, 142, 2.4, 94), "Other": (0.4, 35, 0.6, 24)}),
}
DHS_URLS = {"FR298.pdf": "https://dhsprogram.com/pubs/pdf/FR298/FR298.pdf",
            "FR204.pdf": "https://dhsprogram.com/pubs/pdf/FR204/FR204c.pdf"}


def gkey(s):
    return re.sub(r"[^a-z]", "", str(s).casefold())


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    for name, url in DHS_URLS.items():
        dst = os.path.join(RAW, name)
        if os.path.exists(dst) and os.path.getsize(dst) > 1_000_000:
            print(f"  have {name}")
            continue
        r = requests.get(url, headers={"User-Agent": ab.UA}, timeout=600)
        r.raise_for_status()
        if r.content[:4] != b"%PDF" or b"%%EOF" not in r.content[-2048:]:
            raise SystemExit(f"{name} is not a complete PDF ([[reference_pdf_truncated_at_source]])")
        with open(dst + ".part", "wb") as f:
            f.write(r.content)
        os.replace(dst + ".part", dst)
        print(f"  got  {name} ({len(r.content):,} bytes)")


def dhs_witness():
    """Both DHS respondent tables re-read from the PDFs; returns each row's share, sexes pooled
    by weighted number."""
    import fitz

    out = {}
    for year, (name, page, rows) in DHS.items():
        path = os.path.join(RAW, name)
        if not os.path.exists(path):
            raise SystemExit(f"missing {path}; run with --fetch")
        t = " ".join(fitz.open(path)[page - 1].get_text().split())
        if "Table 3.1 Background characteristics of respondents" not in t:
            raise SystemExit(f"{name} PDF page {page} is not Table 3.1")
        for lab, (wp, wn, mp, mn) in rows.items():
            want = f"{lab} {wp} {wn:,} " if year == "2013" else f"{lab} {wp} {wn:,} "
            got = re.search(re.escape(lab) + r" ([\d.]+) ([\d,]+) [\d,]+ ([\d.]+) ([\d,]+)", t)
            vals = (float(got.group(1)), int(got.group(2).replace(",", "")),
                    float(got.group(3)), int(got.group(4).replace(",", ""))) if got else None
            if vals != (wp, wn, mp, mn):
                raise SystemExit(f"DHS {year} {lab}: transcribed {(wp, wn, mp, mn)}, the PDF has {vals}")
        tot = sum(wn + mn for _wp, wn, _mp, mn in rows.values())
        out[year] = {lab: (wn + mn) / tot for lab, (_wp, wn, _mp, mn) in rows.items()}
    print("\n  DHS respondent tables re-read from both reports and equal; women and men 15-49 pooled:")
    for year, d in out.items():
        print(f"    {year}: " + ", ".join(f"{k} {100 * v:.1f}%" for k, v in d.items()))
    return out


def report_card():
    """The boxes the grouping relies on must be on every round's card (value labels)."""
    import pyreadstat

    watch = ["christian only", "roman catholic", "lutheran", "anglican", "seventh day adventist",
             "pentecostal", "evangelical", "none", "traditional/ethnic religion", "other"]
    print("\n  boxes on each round's showcard (value labels, not responses):")
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
        print(f"    R{rnd}: {'every box present' if not gone else 'MISSING ' + ', '.join(gone)}")
        missing += [(rnd, w) for w in gone]
    if missing:
        raise SystemExit(f"boxes missing from a round's card: {missing}")


def levels_by_round(df, nm, dhs):
    """The churches by round, nationally and in the Owambo regions, against the DHS."""
    tot = df.groupby("round")["w"].sum()
    t = df.groupby(["k", "round"])["w"].sum().unstack(fill_value=0.0).div(tot, axis=1)
    show = ["lutheran", "roman catholic", "anglican", "christian only", "evangelical",
            "seventh day adventist", "pentecostal", "none", "traditional/ethnic religion", "other"]
    print("\n  weighted share of all respondents by round (%), and the range:")
    print(f"    {'':<28}" + "".join(f"{'R' + str(r):>7}" for r in ROUNDS))
    for k in show:
        row = t.loc[k] if k in t.index else pd.Series(0.0, index=tot.index)
        print(f"    {k:<28}" + "".join(f"{100 * row.get(r, 0):7.1f}" for r in ROUNDS)
              + f"   range {100 * (row.max() - row.min()):5.1f}")
    owambo = ["NA08", "NA10", "NA11", "NA12"]
    sub = df[df["geo_id"].isin(owambo)]
    st = sub.groupby(["k", "round"])["w"].sum().unstack(fill_value=0.0).div(
        sub.groupby("round")["w"].sum(), axis=1)
    print("  the four Owambo regions together (Ohangwena, Omusati, Oshana, Oshikoto):")
    for k in ["lutheran", "anglican", "evangelical", "christian only", "roman catholic"]:
        row = st.loc[k] if k in st.index else pd.Series(0.0, index=st.columns)
        print(f"    {k:<28}" + "".join(f"{100 * row.get(r, 0):7.1f}" for r in ROUNDS))

    elcin = dhs["2013"]["ELCIN"]
    lut = t.loc["lutheran"]
    print(f"  DHS 2013 ELCIN, women and men 15-49: {100 * elcin:.1f}%")
    near = {r: float(lut[r]) for r in CHURCH_ROUNDS if abs(lut[r] - elcin) > LUTHERAN_DHS_GAP_MAX}
    if near:
        raise SystemExit(f"rounds {CHURCH_ROUNDS} no longer sit within "
                         f"{100 * LUTHERAN_DHS_GAP_MAX:.0f} points of the DHS's ELCIN: {near}")
    far = {r: float(lut[r]) for r in ROUNDS if r not in CHURCH_ROUNDS
           and elcin - lut[r] < LUTHERAN_SWING_MIN}
    if far:
        raise SystemExit(f"rounds outside {CHURCH_ROUNDS} now come within "
                         f"{100 * LUTHERAN_SWING_MIN:.0f} points of the DHS's Lutheran level: {far}; "
                         "the reason for taking the churches from two rounds is weaker, decide again")
    for c in ["roman catholic", "seventh day adventist", "pentecostal"]:
        rng = float(t.loc[c].max() - t.loc[c].min())
        if rng > CHURCH_RANGE_MAX:
            raise SystemExit(f"{c} now moves {100 * rng:.1f} points across rounds")
    cath = float(df.loc[df["k"] == "roman catholic", "w"].sum() / df["w"].sum())
    if abs(cath - dhs["2013"]["Roman Catholic"]) > CATHOLIC_DHS_GAP_MAX:
        raise SystemExit(f"the pooled Catholic share {100 * cath:.1f}% is no longer within "
                         f"{100 * CATHOLIC_DHS_GAP_MAX:.0f} points of the DHS's")
    return t


def carved_shares(df, units):
    """Lutheran and Anglican as a share of POOL in each unit, rounds 5 and 6, weighted; and the
    split-half on Christians in those two rounds."""
    sub = df[df["round"].isin(CHURCH_ROUNDS) & (df["category"] == POOL)]
    sub = sub.assign(c=sub["k"].map(lambda k: {"lutheran": "Lutheran", "anglican": "Anglican"}
                                    .get(k, "rest of the pool")))
    by = sub.groupby(["geo_id", "c"])["w"].sum().unstack(fill_value=0.0).reindex(units, fill_value=0.0)
    frac = by.div(by.sum(axis=1), axis=0)

    chr_ = df[df["round"].isin(CHURCH_ROUNDS)
              & df["category"].isin(["Roman Catholic", "Seventh Day Adventist", "Pentecostal", POOL])]
    chr_ = chr_.assign(code=chr_["k"].map(lambda k: {"lutheran": "Lutheran", "anglican": "Anglican"}
                                          .get(k, "other Christian")))
    dfw = chr_.rename(columns={"round": "wave"})[["wave", "geo_id", "code", "w"]]
    passed, _t = cab.stability(dfw, ["Lutheran", "Anglican", "other Christian"], units,
                               f"{len(units)} units, Christians in rounds 5 and 6")
    got = [c for c in CARVED if c in passed]
    if got != CARVED_PASS:
        raise SystemExit(f"rounds 5-6 now place {got}, not {CARVED_PASS}; decide again")
    return frac[CARVED]


def pair_ratio(df):
    """None's share of the pair, all six rounds pooled, and the early and late rounds beside it."""
    def frac(rounds):
        s = df[df["round"].isin(rounds) & (df["category"] == PAIR)]
        n = s.loc[s["box"] == NONE, "w"].sum()
        return float(n / s["w"].sum())
    allr, early, late = frac(ROUNDS), frac([4, 5, 6]), frac([7, 8, 9])
    print(f"\n  None's share of the none-or-traditional pair: rounds 4-6 {early:.3f}, rounds 7-9 "
          f"{late:.3f}, all six {allr:.3f} (used)")
    if abs(allr - NONE_FRACTION) > 0.005:
        raise SystemExit(f"None's share of the pair is now {allr:.3f}, not {NONE_FRACTION}; "
                         "edit NONE_FRACTION and the docstring deliberately")
    return allr


def main():
    if "--fetch" in sys.argv:
        fetch()
    dhs = dhs_witness()
    print(f"\n=== Afrobarometer, {COUNTRY} ===")
    raw = ab.load(COUNTRY, expect_rounds=ROUNDS, regroup=True)
    print(f"\n  pooled: {len(raw):,} respondents with a religion answer over six rounds")
    raw["k"] = raw["category"].map(key)
    ct = pd.crosstab(raw["k"], raw["round"])
    ct["all"] = ct.sum(axis=1)
    print("\n  every answer as it arrives, keyed (this is what GROUP collapses):")
    print(ct.sort_values("all", ascending=False).to_string())

    report_card()
    unmapped = sorted(set(raw["k"]) - set(GROUP))
    if unmapped:
        raise SystemExit(f"answers with no category: {unmapped}; add them to GROUP deliberately")
    df = raw.copy()
    df["raw_category"] = raw["category"]
    df["category"] = raw["k"].map(GROUP)
    df["box"] = raw["k"].map(PAIR_BOX)
    ab.assert_one_wording(df, COUNTRY)
    other_early = int(((df["k"] == "other") & df["round"].isin([4, 5, 6])).sum())
    if other_early > OTHER_EARLY_MAX:
        raise SystemExit(f"{other_early} chose `Other` in rounds 4-6; NOT_PLACED's reason is weaker")

    # ---- units ----
    lut = pd.read_csv(LOOKUP, dtype={"geo_id": str})
    if len(lut) != N_REGIONS or int(lut["pop"].sum()) != CENSUS_2023:
        raise SystemExit(f"{LOOKUP} is not the 14 regions summing to {CENSUS_2023:,}; re-run na_geo.py")
    nm14 = dict(zip(lut["geo_id"], lut["name"]))
    pop14 = lut.set_index("geo_id")["pop"].astype(float)
    pop = pop14.rename(lambda g: KAV if g in KAVANGO else g).groupby(level=0).sum()
    nm = {**{g: n for g, n in nm14.items() if g not in KAVANGO}, KAV: "Kavango (East and West)"}
    units = sorted(pop.index)

    df["geo_id"] = df["geo_raw"].map(gkey).map(NORM)
    if df["geo_id"].isna().any():
        raise SystemExit(f"REGION labels with no unit: "
                         f"{sorted(df.loc[df['geo_id'].isna(), 'geo_raw'].astype(str).unique())}")
    per_round = df.groupby("round")["geo_id"].nunique()
    print("\n  survey units present per round: " + ", ".join(f"R{r} {n}" for r, n in per_round.items()))
    if (per_round != len(units)).any():
        raise SystemExit("a pooled round does not sample all 13 survey units")
    clash = df.groupby(["round", "geo_raw"])["geo_id"].nunique()
    if (clash > 1).any():
        raise SystemExit("one REGION label names two units in a round")
    split = df[df["round"] >= 6].groupby("geo_raw").size()
    print("  rounds 6-9 name the two Kavango regions apart: "
          + ", ".join(f"{k} {v}" for k, v in split.items() if "avango" in str(k)))

    levels_by_round(df, nm, dhs)

    for rnd in ROUNDS:
        print(f"\n  R{rnd}:", end="")
        ab.held_out(df[df["round"] == rnd], pop, f"{COUNTRY} R{rnd}", pop_source="census 2023")
    ab.held_out(df, pop, COUNTRY, pop_source="census 2023")

    nat = ab.national(df).reindex(CATEGORIES).fillna(0.0)
    print(f"\n  the survey's national shares, pooled over R4-R9 (n={len(df):,}):")
    for c in CATEGORIES:
        print(f"    {c:<30}{nat[c]:8.3%}")
    rn = df.groupby(["round", "category"])["w"].sum().unstack(fill_value=0)
    print("\n  by round (survey weighting, %):")
    print((100 * rn.div(rn.sum(axis=1), axis=0)).round(1).reindex(columns=CATEGORIES).to_string())

    # ---- quota, then the split-half ----
    dfw = df.rename(columns={"round": "wave", "category": "code"})[["wave", "geo_id", "code", "w"]]
    cab.assert_not_quota(dfw, COUNTRY, ROUNDS, unit_col="geo_id", cat_col="code")
    passed, _table = cab.stability(dfw, CATEGORIES, units, f"{len(units)} units")
    pair_apart = df.assign(code=df["box"].fillna("rest")).rename(columns={"round": "wave"})
    cab.stability(pair_apart[["wave", "geo_id", "code", "w"]], [NONE, TRAD, "rest"], units,
                  f"{len(units)} units, the pair's two answers apart (printed, not used)")
    carries = [c for c in passed if nat[c] >= ab.ELIGIBLE_FLOOR and c not in NOT_PLACED]
    if sorted(carries) != sorted(CARRIES):
        raise SystemExit(f"the split-half now selects {sorted(carries)}, not {sorted(CARRIES)}. "
                         "Read the table above, then edit CARRIES and the docstring deliberately.")
    stale = sorted(set(NOT_PLACED) - set(passed))
    if stale:
        raise SystemExit(f"NOT_PLACED names categories that no longer pass: {stale}")

    # ---- compose the six-round frame, then carve the two churches out of the pool ----
    frame, own, nraw, flat = ab.compose(df, nat, units, CATEGORIES, carries)
    if flat != TAIL_FLAT:
        raise SystemExit(f"the small-category rule now gives flat={flat}, against TAIL_FLAT="
                         f"{TAIL_FLAT}; read the multiples above and decide deliberately")
    carved = carved_shares(df, units)
    print("\n  Lutheran and Anglican as a share of the pool in rounds 5-6, by unit (%):")
    for u in units:
        print(f"    {nm[u]:<26}pool {100 * frame.loc[u, POOL]:5.1f}%  Lutheran "
              f"{100 * carved.loc[u, 'Lutheran']:5.1f}  Anglican {100 * carved.loc[u, 'Anglican']:5.1f}")
    frac = pair_ratio(df)
    out_frame = pd.DataFrame(index=units)
    out_frame["Lutheran"] = frame[POOL] * carved["Lutheran"]
    out_frame["Anglican"] = frame[POOL] * carved["Anglican"]
    out_frame["Other Christian"] = frame[POOL] - out_frame["Lutheran"] - out_frame["Anglican"]
    for c in ["Roman Catholic", "Seventh Day Adventist", "Pentecostal", "Muslim", "Other"]:
        out_frame[c] = frame[c]
    out_frame[NONE] = frame[PAIR] * frac
    out_frame[TRAD] = frame[PAIR] * (1.0 - frac)
    out_frame = out_frame[OUT_CATEGORIES]
    if (out_frame < -1e-12).any().any() or (out_frame.sum(axis=1) - 1).abs().max() > 1e-9:
        raise SystemExit("the composed frame is not a closed partition of every unit")

    # ---- the 13 mixes on the 14 regions ----
    regions = sorted(pop14.index)
    mix14 = pd.DataFrame({g: out_frame.loc[KAV if g in KAVANGO else g] for g in regions}).T
    counts = ab.round_within_rows(mix14.mul(pop14.reindex(regions), axis=0))
    if not (counts.sum(axis=1) == pop14.reindex(regions).round().astype("int64")).all():
        raise SystemExit("a region's drawn total is not its 2023 census population")
    drawn = counts.sum(axis=0) / counts.sum().sum()

    # ---- level: pooled against rounds 8-9 recomposed on the census, and against the DHS ----
    rec = df[df["round"].isin(RECENT)]
    rb = rec.groupby(["geo_id", "category"])["w"].sum().unstack(fill_value=0.0)
    rb = rb.reindex(index=units, columns=CATEGORIES, fill_value=0.0)
    recent = rb.div(rb.sum(axis=1), axis=0).mul(pop, axis=0).sum() / pop.sum()
    pooled = frame.mul(pop, axis=0).sum() / pop.sum()
    print("\n  national level of the six-round categories, as drawn and rounds 8-9 recomposed alike:")
    for c in CATEGORIES:
        print(f"    {c:<30}{100 * pooled[c]:8.2f}%{100 * recent[c]:8.2f}%{100 * (pooled[c] - recent[c]):+8.2f}")
    stale = [c for c in carries if abs(pooled[c] - recent[c]) > LEVEL_GAP_MAX]
    if stale:
        raise SystemExit(f"the pooled level differs from rounds 8-9 by more than "
                         f"{100 * LEVEL_GAP_MAX:.1f} points for {stale}; spec §12 (Norway)")
    d13 = dhs["2013"]
    print("\n  as drawn against the DHS 2013 (women and men 15-49; the survey is adults 18+):")
    print(f"    Lutheran {100 * drawn['Lutheran']:.1f}% (ELCIN alone {100 * d13['ELCIN']:.1f}%); "
          f"Catholic {100 * drawn['Roman Catholic']:.1f}% ({100 * d13['Roman Catholic']:.1f}%); "
          f"Adventist {100 * drawn['Seventh Day Adventist']:.1f}% "
          f"({100 * d13['Seventh-Day Adventist']:.1f}%); no religion {100 * drawn[NONE]:.1f}% "
          f"({100 * d13['No religion']:.1f}%); Anglican + Pentecostal + other Christian + other "
          f"{100 * (drawn['Anglican'] + drawn['Pentecostal'] + drawn['Other Christian'] + drawn['Other'] + drawn[TRAD]):.1f}% "
          f"(Protestant/Anglican + Other {100 * (d13['Protestant/Anglican'] + d13['Other']):.1f}%)")

    if abs(drawn["Lutheran"] - d13["ELCIN"]) > LUTHERAN_DRAWN_GAP_MAX:
        raise SystemExit(f"Lutheran as drawn ({100 * drawn['Lutheran']:.1f}%) is more than "
                         f"{100 * LUTHERAN_DRAWN_GAP_MAX:.0f} points from the DHS's ELCIN")

    # ---- write ----
    n_by = df.groupby("geo_id").size()
    n56 = df[df["round"].isin(CHURCH_ROUNDS)].groupby("geo_id").size()
    basis_note = {
        "Lutheran": "the unit's own share of the pool in rounds 5-6, times the pool's six-round share",
        "Anglican": "the unit's own share of the pool in rounds 5-6, times the pool's six-round share",
        "Other Christian": "what the pool leaves once Lutheran and Anglican are taken out",
        NONE: f"the unit's own none-or-traditional share, {100 * frac:.1f}% of it",
        TRAD: f"the unit's own none-or-traditional share, {100 * (1 - frac):.1f}% of it",
    }
    for c in ["Roman Catholic", "Seventh Day Adventist", "Pentecostal", "Muslim", "Other"]:
        basis_note[c] = ("the unit's own measured share" if c in carries else
                         "the national proportion within the unit's remainder")
    out = counts.stack().rename("count").reset_index()
    out.columns = ["geo_id", "source_category", "count"]
    out["geo_level"] = "region"
    out["geo_name"] = out["geo_id"].map(nm14)

    def note(r):
        u = KAV if r.geo_id in KAVANGO else r.geo_id
        where = " (Kavango East and West sampled as one Kavango in rounds 4-5, one mix for both)" \
            if u == KAV else ""
        return (f"2023 census population composed with the unit's own mix from Afrobarometer rounds "
                f"4-9 pooled (n={int(n_by[u])} here, {int(n56[u])} in rounds 5-6){where}; "
                f"{basis_note[r.source_category]}")
    out["basis"] = "self_id"
    out["year"] = YEARS
    out["source_id"] = SOURCE_ID
    out["note"] = out.apply(note, axis=1)
    total = int(out["count"].sum())
    if total != CENSUS_2023:
        raise SystemExit(f"drawn {total:,} against the census {CENSUS_2023:,}")
    cols = ["geo_id", "geo_level", "geo_name", "source_category", "count",
            "basis", "year", "source_id", "note"]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out[cols].to_csv(OUT, index=False, encoding="utf-8")
    print(f"\nwrote {OUT} ({len(out)} rows, {total:,} people, "
          f"{out['source_category'].nunique()} categories, {out['geo_id'].nunique()} regions)")

    print("\n  national, as drawn:")
    for c, n in counts.sum(axis=0).sort_values(ascending=False).items():
        print(f"    {100 * n / total:6.2f}%  {c}  ({n:,})")
    share = counts.div(counts.sum(axis=1), axis=0)
    print("\n  as drawn, by region, most Lutheran first:")
    print(f"    {'':<14}" + "".join(f"{c[:7]:>8}" for c in OUT_CATEGORIES))
    for g in share.sort_values("Lutheran", ascending=False).index:
        s = share.loc[g]
        print(f"    {nm14[g]:<14}" + "".join(f"{100 * s[c]:7.1f}%" for c in OUT_CATEGORIES)
              + f"{int(pop14[g]):>10,}")
    zero = [(nm14[g], c) for g in regions for c in OUT_CATEGORIES
            if counts.loc[g, c] == 0]
    print(f"  drawn at zero: {zero or 'none'}")


if __name__ == "__main__":
    main()
