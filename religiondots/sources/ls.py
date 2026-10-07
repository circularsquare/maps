"""Lesotho — religion in the 10 districts from six pooled Afrobarometer rounds, on the 2016 census.

Reads data/raw/afrobarometer/*.sav, data/geo/ls/ls_lookup.csv, and as witnesses
data/raw/ls/FR309.pdf and FR391.pdf (the 2014 and 2023-24 Demographic and Health Survey reports);
writes data/normalized/ls.csv. `sources/ls.md` is the record; `sources/na.py` is the module this
follows, without its carved churches.

## LESOTHO DOES NOT ASK

The 2016 census dictionary (161 variables) and IPUMS 1996 and 2006 carry no religion item
(`sources.md` §11aq), and the 2026 census has published nothing. So this is the Nigeria construction
(ask/answered/010-ng): each district at its own survey mix and its census population, the national
level computed and never fitted, every row `modelled`.

    row margin      district populations    2016 census, Key Findings Table 2.1.2  EXACT
    the composition each unit's own mix     Afrobarometer R4-R9, 10 districts      measured
    the national level                      neither                                computed

## THE CHURCHES HOLD THEIR LEVEL, AND TWO DHS REPORTS SAY SO

Unlike Namibia, `Christian only` stays at 0.5-4.3% by round, and the three mission churches sit by
the two open DHS respondent tables (Table 3.1, women and men aged 15-49, pooled by weighted number):

    by round, %             R4    R5    R6    R7    R8    R9     DHS 2014   DHS 2023-24
    Roman Catholic        43.7  38.8  42.4  42.2  41.5  39.1       39.3        35.8
    LEC (see below)       18.1  22.5  19.0  18.2  17.0  22.8       17.3        15.3
    Anglican              11.5   9.2  11.2   8.1   6.9   6.0        7.4         6.3

(Round 6 after the drop described below.)

The survey's adults (18+) run a few points more Catholic and LEC than the DHS's 15-49s; the DHS's
own 2014-to-2023 fall is of the same size. So the churches are drawn, which is Madagascar's case
(`playbooks/afrobarometer.md`), with the DHS asserted as the level witness (`levels_by_round`).

**The Lesotho Evangelical Church is one box under two names.** It is `Evangelical` 17-22% through
round 8; in round 9 Basotho also choose `Calvinist` (a label since round 5), and LEC members split
between them (13.1% and 9.6%; every
district has both, Mafeteng 19.3% and 8.6%). Round 5 already has 0.9% `Calvinist`. The church is the
Paris Mission's Reformed church, so both answers are grouped to `Lesotho Evangelical Church`, and
the two together hold their level (17.0-22.8%).

**Zionist, independent and apostolic churches are one box.** In round 4 nobody chose `Zionist
Christian Church` (it is a value label in every round, but the labels are the merged file's, not
Lesotho's card) and 9.6% chose `Independent` (Butha-Buthe 42%); rounds 5-9 have `Zionist Christian Church` at 5-10%
and `Independent` at 0-7%, and round 8 alone adds `Apostolic church` (2.3%). Together they hold at
9.6-14.0% by round (round 6 aside, below). Drawn on `christianity.africaninstituted`, as Eswatini's
Zionists and Apostles are.

## ROUND 6'S TRADITIONAL RELIGION IN THE NORTH IS DROPPED

Round 6 has `Traditional/ethnic religion` at 25.3% in Butha-Buthe, 17.9% in Leribe and 10.3% in
Berea, against 1.7% or less in those districts in every other round and in the rest of round 6. In
the same three districts round 6 has the fewest Zionists and no `Independent` at all. No other round
or instrument reproduces it (neither DHS has a traditional row; the 2014 DHS puts every non-Christian
religion at 1.4%), so the block is dropped, counted and asserted, as Sudan's round 8 Darfur `None`
was (`playbooks/afrobarometer.md`). Those respondents' real answers are probably Zionist or
independent; they are left out rather than guessed. `R6_NORTH_TRAD` is the count.

## WHAT THE SPLIT-HALF PLACES

Catholic (+0.758), Anglican (+0.812), Methodist (+0.770), Pentecostal (+0.461, p=0.02) and the
Zionist box (+0.745) carry their own district shares. The LEC fails (+0.267 against a null of
+0.412), as do Other Christian, None, traditional, Muslim and Other; they share what the placed
five leave in each district at their national proportions (the residual; the worst small-category
multiple is 1.13x, so not flat). Anglican falls by round (11.5% to 6.0%) and Pentecostal rises
(4.3% to 10.4%); both pooled levels sit 2.4 points from rounds 8-9, inside the 3.5-point bar.

Usage:
    python sources/ls.py --fetch    the two DHS reports (~25 MB); the Afrobarometer files are shared
                                    (`python sources/afrobarometer.py --fetch`)
    python sources/ls.py            rebuild data/normalized/ls.csv
"""

import os
import re
import sys

os.environ.setdefault("OMP_NUM_THREADS", "6")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: F401
import pandas as pd

import afrobarometer as ab
import cab
from tz import key

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LOOKUP = os.path.join(ROOT, "data", "geo", "ls", "ls_lookup.csv")
RAW = os.path.join(ROOT, "data", "raw", "ls")
OUT = os.path.join(ROOT, "data", "normalized", "ls.csv")

COUNTRY = "Lesotho"
ROUNDS = [4, 5, 6, 7, 8, 9]
RECENT = [8, 9]
SOURCE_ID = "ls_afrobarometer_2008_2022_census2016"
YEARS = "2008-2022"

N_DISTRICTS = 10
CENSUS_2016 = 2_007_201

LEC = "Lesotho Evangelical Church"
AIC = "Zionist and independent churches"
CATEGORIES = ["Roman Catholic", LEC, "Anglican", "Methodist", "Pentecostal", AIC,
              "Other Christian", "None", "Traditional/ethnic religion", "Muslim", "Other"]

# Every answer Basotho give over the six rounds, keyed through `tz.key()`, -> its category.
# Named one by one: an answer that fell through a default would be dropped silently.
GROUP = {
    "roman catholic": "Roman Catholic",
    "evangelical": LEC, "calvinist": LEC,
    "anglican": "Anglican", "methodist": "Methodist", "pentecostal": "Pentecostal",
    "zionist christian church": AIC, "independent": AIC, "apostolic church": AIC,
    "christian only": "Other Christian", "baptist": "Other Christian",
    "church of christ": "Other Christian", "jehovah's witness": "Other Christian",
    "seventh day adventist": "Other Christian", "presbyterian": "Other Christian",
    "lutheran": "Other Christian", "dutch reformed": "Other Christian",
    "orthodox": "Other Christian", "coptic": "Other Christian", "mormon": "Other Christian",
    "quaker/friends": "Other Christian",
    "none": "None", "atheist": "None", "agnostic": "None",
    "traditional/ethnic religion": "Traditional/ethnic religion",
    "muslim only": "Muslim", "ismaeli": "Muslim", "shia": "Muslim",
    "other": "Other", "bahai": "Other",
}

# Every REGION label over the six rounds, as bare letters, -> the COD-AB district pcode.
NORM = {
    "maseru": "LSA", "buthabuthe": "LSB", "bothabothe": "LSB", "buthebuthe": "LSB",
    "leribe": "LSC", "berea": "LSD", "mafeteng": "LSE", "mohaleshoek": "LSF", "quthing": "LSG",
    "qachasnek": "LSH", "mokhotlong": "LSJ", "thabatseka": "LSK",
}

# The round 6 block (see the docstring): traditional answers in these three districts.
R6_NORTH = ["LSB", "LSC", "LSD"]
R6_NORTH_TRAD = 62              # answers, measured 2026-10-03; asserted
# Every other (round, district) cell of traditional; the highest is Qacha's Nek's 3.9% in round 9.
TRAD_ELSEWHERE_MAX = 0.045

# What the split-half places, asserted so a change in the data stops the build (2026-10-03).
# The LEC fails (+0.267 against a null of +0.412) and goes in the tail with Other Christian, None,
# traditional, Muslim and Other.
CARRIES = ["Roman Catholic", "Anglican", "Methodist", "Pentecostal", AIC]
NOT_PLACED = {}
# Spec §12's small-category rule, measured 2026-10-03: the residual's worst multiple is Muslim at
# 1.13x in Mafeteng, under 2x, so the tail is the residual.
TAIL_FLAT = False

# Levels against the DHS (see the docstring's table).
CHURCH_RANGE_MAX = {"Roman Catholic": 0.06, LEC: 0.07, "Anglican": 0.065}
DHS_GAP_MAX = 0.05              # each church pooled within 5 points of the DHS 2014
LEVEL_GAP_MAX = 0.035           # spec §12 (Norway): pooled against rounds 8-9

# The DHS reports, re-read from the PDFs: (file, PDF page, {row label: (women %, women n, men %, men n)}).
DHS = {
    "2014": ("FR309.pdf", 70, {
        "Roman Catholic": (38.6, 2558, 40.9, 1088), "Lesotho Evangelical": (17.1, 1133, 17.9, 476),
        "Anglican": (7.2, 477, 7.8, 207), "Pentecostal": (24.9, 1646, 18.8, 499),
        "Other Christian": (10.1, 668, 6.8, 180), "Other non-Christian": (1.4, 90, 1.6, 42),
        "No religion": (0.7, 49, 6.3, 168)}),
    "2023-24": ("FR391.pdf", 82, {
        "Roman Catholic": (34.7, 2225, 38.4, 1097), "Lesotho Evangelical Church": (14.6, 934, 17.0, 484),
        "Methodist": (1.5, 94, 0.9, 25), "Anglican Church": (6.2, 398, 6.6, 188),
        "Seventh Day Adventist": (1.2, 76, 0.9, 27), "Pentecostal": (16.8, 1074, 12.5, 356),
        "Other Christian": (23.1, 1482, 13.3, 381), "Islam": (0.2, 13, 0.6, 16),
        "Other": (0.2, 13, 1.5, 42), "None": (1.6, 104, 8.3, 238)}),
}
DHS_URLS = {"FR309.pdf": "https://dhsprogram.com/pubs/pdf/FR309/FR309.pdf",
            "FR391.pdf": "https://dhsprogram.com/pubs/pdf/FR391/FR391.pdf"}
# The survey category each DHS row witnesses.
DHS_ROW = {"2014": {"Roman Catholic": "Roman Catholic", LEC: "Lesotho Evangelical",
                    "Anglican": "Anglican"},
           "2023-24": {"Roman Catholic": "Roman Catholic", LEC: "Lesotho Evangelical Church",
                       "Anglican": "Anglican Church", "Methodist": "Methodist"}}


def gkey(s):
    return re.sub(r"[^a-z]", "", str(s).casefold().replace("’", "'"))


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    for name, url in DHS_URLS.items():
        dst = os.path.join(RAW, name)
        if os.path.exists(dst) and os.path.getsize(dst) > 1_000_000:
            print(f"  have {name}")
            continue
        r = requests.get(url, headers={"User-Agent": ab.UA}, timeout=900)
        r.raise_for_status()
        if r.content[:4] != b"%PDF" or b"%%EOF" not in r.content[-2048:]:
            raise SystemExit(f"{name} is not a complete PDF ([[reference_pdf_truncated_at_source]])")
        with open(dst + ".part", "wb") as f:
            f.write(r.content)
        os.replace(dst + ".part", dst)
        print(f"  got  {name} ({len(r.content):,} bytes)")


def dhs_witness():
    """Both DHS respondent tables re-read from the PDFs; returns each row's share, sexes pooled by
    weighted number."""
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
            got = re.search(r"(?<![A-Za-z-])" + re.escape(lab)
                            + r" ([\d.]+) ([\d,]+) [\d,]+ ([\d.]+) ([\d,]+)", t)
            vals = (float(got.group(1)), int(got.group(2).replace(",", "")),
                    float(got.group(3)), int(got.group(4).replace(",", ""))) if got else None
            if vals != (wp, wn, mp, mn):
                raise SystemExit(f"DHS {year} {lab}: transcribed {(wp, wn, mp, mn)}, the PDF has {vals}")
        tot = sum(wn + mn for _wp, wn, _mp, mn in rows.values())
        out[year] = {lab: (wn + mn) / tot for lab, (_wp, wn, _mp, mn) in rows.items()}
    print("\n  DHS respondent tables re-read from both reports; women and men 15-49 pooled:")
    for year, d in out.items():
        print(f"    {year}: " + ", ".join(f"{k} {100 * v:.1f}%" for k, v in d.items()))
    return out


def report_card():
    """The boxes the grouping relies on must be on every round's card (value labels)."""
    import pyreadstat

    watch = ["christian only", "roman catholic", "anglican", "evangelical", "pentecostal",
             "methodist", "independent", "none", "traditional/ethnic religion", "other"]
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
        extra = [w for w in ["zionist christian church", "calvinist", "apostolic church"] if w in have]
        print(f"    R{rnd}: {'every box present' if not gone else 'MISSING ' + ', '.join(gone)}"
              f"; also {', '.join(extra) or 'none of zionist/calvinist/apostolic'}")
        missing += [(rnd, w) for w in gone]
    if missing:
        raise SystemExit(f"boxes missing from a round's card: {missing}")


def drop_r6_north_traditional(df):
    """Round 6's traditional answers in Butha-Buthe, Leribe and Berea: counted, asserted, dropped."""
    trad = df["category"] == "Traditional/ethnic religion"
    block = trad & (df["round"] == 6) & df["geo_id"].isin(R6_NORTH)
    cell = (df.assign(t=trad * df["w"]).groupby(["round", "geo_id"])[["t", "w"]].sum())
    cell = cell["t"] / cell["w"]
    print("\n  traditional religion by round and district (%), before the drop:")
    print((100 * cell.unstack(0)).round(1).to_string())
    others = cell.drop([(6, g) for g in R6_NORTH])
    print(f"  round 6 in {R6_NORTH}: {int(block.sum())} answers dropped; the highest other "
          f"(round, district) cell is {100 * others.max():.1f}%")
    if int(block.sum()) != R6_NORTH_TRAD:
        raise SystemExit(f"the round 6 northern block is {int(block.sum())} answers, not "
                         f"{R6_NORTH_TRAD}; read the table above and decide again")
    if others.max() > TRAD_ELSEWHERE_MAX:
        raise SystemExit("another round or district now reaches the round 6 block's level; "
                         "the block is no longer unique")
    return df[~block].copy()


def levels_by_round(df, dhs):
    """The churches by round against both DHS reports."""
    tot = df.groupby("round")["w"].sum()
    t = df.groupby(["category", "round"])["w"].sum().unstack(fill_value=0.0).div(tot, axis=1)
    print("\n  weighted share of all respondents by round (%), the range, and the DHS:")
    print(f"    {'':<34}" + "".join(f"{'R' + str(r):>7}" for r in ROUNDS) + "   range   2014  2023-24")
    for c in CATEGORIES:
        row = t.loc[c] if c in t.index else pd.Series(0.0, index=tot.index)
        w = [dhs[y].get(DHS_ROW[y].get(c, ""), float("nan")) for y in ["2014", "2023-24"]]
        print(f"    {c:<34}" + "".join(f"{100 * row.get(r, 0):7.1f}" for r in ROUNDS)
              + f"   {100 * (row.max() - row.min()):5.1f}  {100 * w[0]:5.1f}  {100 * w[1]:5.1f}")
    ko = df[df["k"] == "christian only"].groupby("round")["w"].sum() / tot
    print("  `Christian only` by round: " + ", ".join(f"R{r} {100 * v:.1f}%" for r, v in ko.items()))
    for c, mx in CHURCH_RANGE_MAX.items():
        rng = float(t.loc[c].max() - t.loc[c].min())
        if rng > mx:
            raise SystemExit(f"{c} now moves {100 * rng:.1f} points across rounds")
    pooled = ab.national(df)
    for c in CHURCH_RANGE_MAX:
        gap = abs(pooled[c] - dhs["2014"][DHS_ROW["2014"][c]])
        if gap > DHS_GAP_MAX:
            raise SystemExit(f"{c} pooled {100 * pooled[c]:.1f}% is {100 * gap:.1f} points from the "
                             "DHS 2014")
    return t


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
    ab.assert_one_wording(df, COUNTRY)

    # ---- units ----
    lut = pd.read_csv(LOOKUP, dtype={"geo_id": str})
    if len(lut) != N_DISTRICTS or int(lut["pop"].sum()) != CENSUS_2016:
        raise SystemExit(f"{LOOKUP} is not the 10 districts summing to {CENSUS_2016:,}; re-run ls_geo.py")
    nm = dict(zip(lut["geo_id"], lut["name"]))
    pop = lut.set_index("geo_id")["pop"].astype(float)
    units = sorted(pop.index)

    df["geo_id"] = df["geo_raw"].map(gkey).map(NORM)
    if df["geo_id"].isna().any():
        raise SystemExit(f"REGION labels with no unit: "
                         f"{sorted(df.loc[df['geo_id'].isna(), 'geo_raw'].astype(str).unique())}")
    per_round = df.groupby("round")["geo_id"].nunique()
    print("\n  districts present per round: " + ", ".join(f"R{r} {n}" for r, n in per_round.items()))
    if (per_round != len(units)).any():
        raise SystemExit("a pooled round does not sample all 10 districts")
    if (df.groupby(["round", "geo_raw"])["geo_id"].nunique() > 1).any():
        raise SystemExit("one REGION label names two districts in a round")
    if (df.groupby(["round", "geo_id"])["geo_raw"].nunique() > 1).any():
        raise SystemExit("one district has two REGION labels in a round")

    df = drop_r6_north_traditional(df)
    levels_by_round(df, dhs)

    for rnd in ROUNDS:
        print(f"\n  R{rnd}:", end="")
        ab.held_out(df[df["round"] == rnd], pop, f"{COUNTRY} R{rnd}", pop_source="census 2016")
    ab.held_out(df, pop, COUNTRY, pop_source="census 2016")

    nat = ab.national(df).reindex(CATEGORIES).fillna(0.0)
    print(f"\n  the survey's national shares, pooled over R4-R9 (n={len(df):,}):")
    for c in CATEGORIES:
        print(f"    {c:<34}{nat[c]:8.3%}")

    # ---- quota, then the split-half ----
    dfw = df.rename(columns={"round": "wave", "category": "code"})[["wave", "geo_id", "code", "w"]]
    cab.assert_not_quota(dfw, COUNTRY, ROUNDS, unit_col="geo_id", cat_col="code")
    passed, _table = cab.stability(dfw, CATEGORIES, units, f"{len(units)} districts")
    carries = [c for c in passed if nat[c] >= ab.ELIGIBLE_FLOOR and c not in NOT_PLACED]
    if CARRIES is None or sorted(carries) != sorted(CARRIES):
        raise SystemExit(f"the split-half now selects {sorted(carries)}, not {CARRIES}. "
                         "Read the table above, then edit CARRIES and the docstring deliberately.")
    stale = sorted(set(NOT_PLACED) - set(passed))
    if stale:
        raise SystemExit(f"NOT_PLACED names categories that no longer pass: {stale}")

    # ---- compose ----
    frame, own, nraw, flat = ab.compose(df, nat, units, CATEGORIES, carries)
    if flat != TAIL_FLAT:
        raise SystemExit(f"the small-category rule now gives flat={flat}, against TAIL_FLAT="
                         f"{TAIL_FLAT}; read the multiples above and decide deliberately")
    if (frame < -1e-12).any().any() or (frame.sum(axis=1) - 1).abs().max() > 1e-9:
        raise SystemExit("the composed frame is not a closed partition of every district")
    counts = ab.round_within_rows(frame.mul(pop.reindex(units), axis=0))
    if not (counts.sum(axis=1) == pop.reindex(units).round().astype("int64")).all():
        raise SystemExit("a district's drawn total is not its 2016 census population")
    drawn = counts.sum(axis=0) / counts.sum().sum()

    # ---- level: pooled against rounds 8-9 recomposed on the census, and against the DHS ----
    rec = df[df["round"].isin(RECENT)]
    rb = rec.groupby(["geo_id", "category"])["w"].sum().unstack(fill_value=0.0)
    rb = rb.reindex(index=units, columns=CATEGORIES, fill_value=0.0)
    recent = rb.div(rb.sum(axis=1), axis=0).mul(pop, axis=0).sum() / pop.sum()
    print("\n  national level, as drawn and rounds 8-9 recomposed alike, and the two DHS reports:")
    for c in CATEGORIES:
        w = [dhs[y].get(DHS_ROW[y].get(c, ""), float("nan")) for y in ["2014", "2023-24"]]
        print(f"    {c:<34}{100 * drawn[c]:8.2f}%{100 * recent[c]:8.2f}%{100 * (drawn[c] - recent[c]):+8.2f}"
              f"   DHS {100 * w[0]:5.1f}  {100 * w[1]:5.1f}")
    stale = [c for c in carries if abs(drawn[c] - recent[c]) > LEVEL_GAP_MAX]
    if stale:
        raise SystemExit(f"the pooled level differs from rounds 8-9 by more than "
                         f"{100 * LEVEL_GAP_MAX:.1f} points for {stale}; spec §12 (Norway)")

    # ---- write ----
    n_by = df.groupby("geo_id").size()
    out = counts.stack().rename("count").reset_index()
    out.columns = ["geo_id", "source_category", "count"]
    out["geo_level"] = "district"
    out["geo_name"] = out["geo_id"].map(nm)

    def note(r):
        how = ("the district's own measured share" if r.source_category in carries else
               "the national proportion" + (" in every district" if flat else
                                            " within the district's remainder"))
        return (f"2016 census population composed with the district's own mix from Afrobarometer "
                f"rounds 4-9 pooled (n={int(n_by[r.geo_id])} here); {how}")
    out["basis"] = "self_id"
    out["year"] = YEARS
    out["source_id"] = SOURCE_ID
    out["note"] = out.apply(note, axis=1)
    total = int(out["count"].sum())
    if total != CENSUS_2016:
        raise SystemExit(f"drawn {total:,} against the census {CENSUS_2016:,}")
    cols = ["geo_id", "geo_level", "geo_name", "source_category", "count",
            "basis", "year", "source_id", "note"]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out[cols].to_csv(OUT, index=False, encoding="utf-8")
    print(f"\nwrote {OUT} ({len(out)} rows, {total:,} people, "
          f"{out['source_category'].nunique()} categories, {out['geo_id'].nunique()} districts)")

    print("\n  national, as drawn:")
    for c, n in counts.sum(axis=0).sort_values(ascending=False).items():
        print(f"    {100 * n / total:6.2f}%  {c}  ({n:,})")
    share = counts.div(counts.sum(axis=1), axis=0)
    print("\n  as drawn, by district, most Catholic first:")
    print(f"    {'':<14}" + "".join(f"{c[:7]:>8}" for c in CATEGORIES))
    for g in share.sort_values("Roman Catholic", ascending=False).index:
        s = share.loc[g]
        print(f"    {nm[g]:<14}" + "".join(f"{100 * s[c]:7.1f}%" for c in CATEGORIES)
              + f"{int(pop[g]):>10,}")
    zero = [(nm[g], c) for g in units for c in CATEGORIES if counts.loc[g, c] == 0]
    print(f"  drawn at zero: {zero or 'none'}")


if __name__ == "__main__":
    main()
