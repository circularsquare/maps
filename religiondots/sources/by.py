"""Belarus: religion by oblast from the EBRD Life in Transition Survey III, with Catholics placed
by the Catholic Church's own diocesan counts.

Reads data/raw/lits/lits_iii.dta and data/geo/by/by_lookup.csv, writes data/normalized/by.csv.
`sources/lits.py` holds the survey machinery and `sources/by.md` the record.

No Belarusian census asks about religion (2019 Form 2N, 25 questions, none on religion;
`sources.md` §scout-2026-09-14-taiwan-belarus-gabon), so this survey is the count, and every row
is `modelled`.

## LiTS III GIVES BELARUS A NATIONAL COMPOSITION AND NO OBLAST GEOGRAPHY

75 PSUs over all seven units. The split-half on PSUs (`lits.stability`) passes nothing a reader
could use:

    73.55%  ORTHODOX CHRISTIAN                      median rho +0.11, p 0.41
    14.59%  ATHEISTIC / AGNOSTIC / NONE             +0.21, p 0.28
     9.21%  CATHOLIC                                +0.37, p 0.10
     1.02%  BUDDHIST                                +0.56, p 0.037   passes; OVERRIDE below
     0.76%  OTHER                                   +0.15, p 0.45
     0.50%  OTHER CHRISTIAN, INCLUDING PROTESTANT   +0.28, p 0.27
     0.27%  JEWISH                                  -0.52, p 1.00
     0.11%  MUSLIM                                  one respondent, no test

## ITS CATHOLICS ARE IN THE WRONG OBLASTS, AND ITS POLES SAY WHY

Weighted Catholic share comes out Gomel 18.1%, Brest 14.2%, Grodno 11.2%, which inverts the one
thing known in advance: Grodno is Belarus's Catholic and Polish oblast. The survey's ethnicity item
(`q923`) fails the same way against the 2019 census, and the census is a count: LiTS has Poles at
8.2% of Gomel against the census's 0.19%, and 3.9% of Grodno against 21.7% (rank correlation over
the seven units -0.21). Of Gomel's 41 Catholic respondents, 17 also answer Pole, and they sit with
five interviewers of one Gomel team. So this is fieldwork in a few PSUs, not a community the census
missed, and it is asserted rather than described (`ethnicity_witness`).

## SO CATHOLICS FOLLOW THE CHURCH'S DIOCESAN COUNTS, AT THE SURVEY'S NATIONAL LEVEL

Spec §3.5a's construction, a survey's total over a church's structure. The level is LiTS's 9.21%;
the share of it each diocese gets is the Annuario Pontificio's Catholics per diocese (read at
catholic-hierarchy.org): Grodno 548,125, Minsk-Mohilev 652,300, Vitebsk 167,516, Pinsk 54,140.
A diocese spanning several units gives each the same share (Pinsk is Brest and Gomel oblasts;
Minsk-Mohilev is Minsk city, Minsk oblast and Mogilev oblast), because nothing here measures the
split between them. Checked: the drawn Catholic share of every unit is at least its census share of
Poles, who are close to all Catholic (`poles_floor`).

The other seven answers go at the national rate inside each unit's non-Catholic remainder, which
is what failing the split-half earns them.

Usage:
    python sources/by.py            rebuild data/normalized/by.csv
"""

import os
import sys

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "6")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import lits
from by_geo import LITS_RAW, lits_label

GEO = os.path.join(ROOT, "data", "geo", "by")
LOOKUP = os.path.join(GEO, "by_lookup.csv")
OUT = os.path.join(ROOT, "data", "normalized", "by.csv")

COUNTRY = "Belarus"
SOURCE_ID = "by_lits3_2016"
N_UNITS = 7

CATHOLIC = "CATHOLIC"

# What the split-half passes, and what of that is drawn on its own unit shares. Both asserted, so a
# change in the data or the test is a stop here rather than a silent redraw.
PASSES = ["BUDDHIST"]
CARRIES = []

OVERRIDE = {
    "BUDDHIST": (
        "passes at p=0.037, one pass in seven tests where about one in three runs would show a "
        "false pass somewhere. Thirteen respondents, 1.02% weighted, about 92,000 people, in a "
        "country whose registered Buddhist communities are a handful and whose EVS 2017 card has "
        "no Buddhist box. Five of the thirteen are one each in five Minsk city PSUs. Read as the "
        "keying slip Kyrgyzstan showed on the same card (code 2 beside code 1, none), spread at "
        "the national rate as a ceiling, as kg.py does."),
}

UNTESTED = {"MUSLIM": "one respondent in the whole sample, in Minsk city, so no halving can rank it"}

EXCLUDE = [lits.REFUSAL]

# The Catholic Church's own count of Catholics per diocese, Annuario Pontificio, as tabulated on
# catholic-hierarchy.org (dgrod, dmins, dvitb, dpins), read 2026-09-14. The Church's population
# figures are stale for Vitebsk (1,412,489 against Belstat's 1,060,687) and are not used; only the
# Catholic counts are, as shares of their sum.
DIOCESES = {
    "Grodno":        (548_125, "Annuario 2024, 31 Dec 2023", ["BY003"]),
    "Minsk-Mohilev": (652_300, "Annuario 2024, 31 Dec 2023", ["BY005", "BY004", "BY006"]),
    "Vitebsk":       (167_516, "Annuario 2023, 31 Dec 2022", ["BY007"]),
    "Pinsk":         (54_140,  "Annuario 2024, 31 Dec 2023", ["BY001", "BY002"]),
}

# 2019 census, Belstat bulletin "Национальный состав населения Республики Беларусь", the table
# of nationalities by oblast and Minsk city (pp. 17-19): everyone, and Poles.
CENSUS_2019 = {
    "BY001": (1_348_115, 14_893), "BY007": (1_135_731, 9_806), "BY002": (1_388_512, 2_572),
    "BY003": (1_026_816, 223_119), "BY005": (2_018_281, 19_397), "BY004": (1_471_240, 15_785),
    "BY006": (1_024_751, 2_121),
}
CENSUS_2019_TOTAL = (9_413_446, 287_693)

# `q923` codes are `Option N`; `q923_ethnicity` carries the words. Asserted, not assumed.
POLE_CODE = "Option 3"
# The survey's Pole geography has to fail against the census for the reasoning above to stand. If
# a re-release fixes it, this stops and the Catholic placement should be looked at again.
POLE_RANK_CEILING = 0.5


def read_lits(lut):
    df = lits.load(COUNTRY)
    raw = set(df["region_name"].astype(str))
    if raw != LITS_RAW:
        raise SystemExit(f"LiTS Belarus region strings changed: {sorted(raw)}")
    to_geo = dict(zip(lut["lits_region"], lut.index))
    df["region"] = df["region_name"].map(lits_label)
    df["geo_id"] = df["region"].map(to_geo)
    if df["geo_id"].isna().any():
        raise SystemExit(f"LiTS regions with no unit: {sorted(set(df.loc[df['geo_id'].isna(), 'region']))}")
    return df


def ethnicity_witness(lut, units):
    """LiTS's Poles against the census's, per unit. The evidence that the survey misplaces the
    Catholic and Polish minority, asserted."""
    d = pd.read_stata(lits.DTA, columns=["country", "region_name", "weight_population", "q922",
                                         "q923", "q923_ethnicity", "interviewerID"],
                      convert_categoricals=True)
    d = d[d["country"].astype(str).str.contains(COUNTRY, case=False, na=False)].copy()
    words = d.loc[d["q923"].astype(str) == POLE_CODE, "q923_ethnicity"].astype(str).unique()
    if list(words) != ["Pole"]:
        raise SystemExit(f"q923 {POLE_CODE} reads {list(words)}, not ['Pole']")
    d["geo_id"] = d["region_name"].map(lits_label).map(dict(zip(lut["lits_region"], lut.index)))
    w = d["weight_population"].astype(float)
    pole = (d["q923"].astype(str) == POLE_CODE)
    lits_pole = (w * pole).groupby(d["geo_id"]).sum() / w.groupby(d["geo_id"]).sum()
    census = pd.Series({u: p / n for u, (n, p) in CENSUS_2019.items()})
    rho = float(spearmanr(lits_pole.reindex(units), census.reindex(units))[0])
    print("\n  ethnicity witness, LiTS q923 `Pole` against the 2019 census (share of each unit):")
    print(f"    {'unit':<14}{'LiTS':>8}{'census':>9}")
    for u in units:
        print(f"    {lut.loc[u, 'name']:<14}{lits_pole[u] * 100:7.2f}%{census[u] * 100:8.2f}%")
    print(f"    rank correlation over the {len(units)} units: {rho:+.2f}")
    if not rho < POLE_RANK_CEILING:
        raise SystemExit(f"LiTS's Poles now rank with the census's at {rho:+.2f}; the reason the "
                         "survey's Catholic geography is set aside no longer holds. Re-read "
                         "sources/by.md before building.")
    gomel = d[(d["geo_id"] == "BY002") & (d["q922"].astype(str) == CATHOLIC)]
    per_iv = gomel.groupby(gomel["interviewerID"].astype(str)).agg(
        catholics=("q922", "size"), poles=("q923", lambda s: int((s.astype(str) == POLE_CODE).sum())))
    per_iv = per_iv.sort_values("catholics", ascending=False)
    print(f"    Gomel's {len(gomel)} Catholic respondents by interviewer (catholics, of them poles): "
          + ", ".join(f"{i} {r.catholics}/{r.poles}" for i, r in per_iv.iterrows()))
    return rho


def catholic_shares(nat_cath, pop):
    """Each unit's Catholic share: the survey's national figure, split by diocesan counts, even
    within a diocese."""
    placed = [u for _, _, us in DIOCESES.values() for u in us]
    if sorted(placed) != sorted(pop.index) or len(set(placed)) != len(placed):
        raise SystemExit(f"DIOCESES must place each unit exactly once: {placed}")
    church = sum(c for c, _, _ in DIOCESES.values())
    national = nat_cath * float(pop.sum())
    share = {}
    print(f"\n  Catholics: the survey's {nat_cath:.2%} of {int(pop.sum()):,} is {national:,.0f} "
          f"people; the Church counts {church:,} ({church / pop.sum():.1%}), used only as shape")
    print(f"    {'diocese':<15}{'Church':>10}{'of all':>8}{'drawn':>10}{'share of its units':>20}")
    for name, (cath, edition, us) in DIOCESES.items():
        dpop = float(pop[us].sum())
        dcath = national * cath / church
        for u in us:
            share[u] = dcath / dpop
        print(f"    {name:<15}{cath:>10,}{cath / church:>8.1%}{dcath:>10,.0f}{dcath / dpop:>19.1%}")
    return pd.Series(share)


def poles_floor(share, lut):
    """Poles in Belarus are close to all Catholic, so no unit may be drawn with fewer Catholics
    than its census share of Poles."""
    print("\n  Poles floor, drawn Catholic share against the 2019 census share of Poles:")
    bad = []
    for u in share.index:
        n, p = CENSUS_2019[u]
        ok = share[u] >= p / n
        print(f"    {lut.loc[u, 'name']:<14}{share[u] * 100:7.2f}%  Poles {p / n * 100:6.2f}%"
              + ("" if ok else "   BELOW"))
        if not ok:
            bad.append(lut.loc[u, "name"])
    if bad:
        raise SystemExit(f"drawn Catholics fall below the census's Poles in {bad}")


def lits_by_diocese(kept, lut):
    """The survey's own Catholics by diocese, for the record: how far it is from the Church's."""
    w = kept.assign(c=(kept["code"] == CATHOLIC) * kept["w"])
    pop = lut["pop"]
    cath = (w.groupby("geo_id")["c"].sum() / w.groupby("geo_id")["w"].sum()) * pop
    church = sum(c for c, _, _ in DIOCESES.values())
    print("\n  the survey's own Catholics by diocese, against the Church's distribution:")
    for name, (c, _, us) in DIOCESES.items():
        print(f"    {name:<15} LiTS {cath[us].sum() / cath.sum():6.1%}   Church {c / church:6.1%}")


def build(nat, share, pop, units):
    others = [c for c in nat.index if c != CATHOLIC]
    rest = float(nat[others].sum())
    rows = []
    for u in units:
        p = int(pop[u])
        rows.append((u, CATHOLIC, share[u] * p,
                     "the survey's national Catholic share, placed by diocesan counts"))
        for c in others:
            rows.append((u, c, (1 - share[u]) * float(nat[c]) / rest * p,
                         "national share within the oblast's non-Catholic remainder"))
    out = pd.DataFrame(rows, columns=["geo_id", "source_category", "count", "basis_note"])
    out["count"] = out["count"].round().astype("int64")
    drift = int(pop[units].sum()) - int(out["count"].sum())
    if abs(drift) > len(out):
        raise SystemExit(f"rounding drift {drift} is larger than one person per row")
    if drift:
        out.loc[out["count"].idxmax(), "count"] += drift
    print(f"  rounding drift {drift:+d} people, absorbed into the largest cell")
    return out


def main():
    lut = pd.read_csv(LOOKUP).set_index("geo_id")
    pop = lut["pop"].astype("int64")
    units = sorted(lut.index)
    if len(units) != N_UNITS:
        raise SystemExit(f"{len(units)} units in the lookup, expected {N_UNITS}")

    df = read_lits(lut)
    missing = lits.coverage(df, pop)
    if missing:
        raise SystemExit(f"LiTS III does not reach {missing}")
    lits.held_out(df, pop, "Belarus", pop_source="Belstat's 1 January 2026 figures",
                  names=[lut.loc[u, "name"] for u in units])

    kept = df[~df["code"].isin(EXCLUDE)]
    ref_w = float(df.loc[df["code"].isin(EXCLUDE), "w"].sum() / df["w"].sum())
    print(f"\n  {len(df) - len(kept)} of {len(df):,} respondents excluded ({', '.join(EXCLUDE)}), "
          f"{ref_w:.3%} of the weighted sample; countries/by.py's gap states it")
    nat = lits.national(kept)

    passes = lits.stability(kept, nat, units, untested=UNTESTED)
    if sorted(passes) != sorted(PASSES):
        raise SystemExit(f"the split-half now passes {sorted(passes)}, not {sorted(PASSES)}; read "
                         "the table above and update PASSES, OVERRIDE and the docstring deliberately")
    large = [c for c in passes if c not in OVERRIDE]
    for c in passes:
        if c in OVERRIDE:
            print(f"    OVERRIDE {c}: {OVERRIDE[c]}")
    if sorted(large) != sorted(CARRIES):
        raise SystemExit(f"drawn on own unit shares: {large}, expected {CARRIES}")

    ethnicity_witness(lut, units)
    lits_by_diocese(kept, lut)
    share = catholic_shares(float(nat[CATHOLIC]), pop)
    poles_floor(share, lut)

    out = build(nat, share, pop, units)
    total = int(out["count"].sum())
    if total != int(pop.sum()):
        raise SystemExit(f"drawn {total:,} against Belstat's {int(pop.sum()):,}")
    drawn_cath = out.loc[out["source_category"] == CATHOLIC, "count"].sum() / total
    if abs(drawn_cath - nat[CATHOLIC]) > 1e-4:
        raise SystemExit(f"drawn Catholic share {drawn_cath:.4%} is not the survey's {nat[CATHOLIC]:.4%}")

    lits.lean(df, out, EXCLUDE, units)

    n_unit = kept.groupby("geo_id").size()
    out["geo_level"] = "oblast"
    out["geo_name"] = out["geo_id"].map(lut["name"])
    out["basis"] = "self_id"
    out["year"] = "2015-2016"
    out["source_id"] = SOURCE_ID
    out["note"] = [
        (f"EBRD Life in Transition Survey III, n={int(n_unit[g])} in this unit; {b} applied to "
         "Belstat's 1 January 2026 population")
        for g, b in zip(out["geo_id"], out["basis_note"])]
    cols = ["geo_id", "geo_level", "geo_name", "source_category", "count", "basis", "year",
            "source_id", "note"]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out[cols].to_csv(OUT, index=False, encoding="utf-8")
    print(f"\nwrote {OUT} ({len(out)} rows, {total:,} people, "
          f"{out['source_category'].nunique()} categories, {out['geo_id'].nunique()} units)")

    print("\n  national, as drawn:")
    nat_drawn = out.groupby("source_category")["count"].sum().sort_values(ascending=False)
    for c, n in nat_drawn.items():
        print(f"    {n / total * 100:7.3f}%  {n:>9,}  {c}")
    piv = out.pivot_table(index="geo_id", columns="source_category", values="count", aggfunc="sum")
    sh = piv.div(piv.sum(axis=1), axis=0)
    print("\n  by unit, Catholic and Orthodox share as drawn, with the survey's n:")
    for u in sh[CATHOLIC].sort_values(ascending=False).index:
        print(f"    {lut.loc[u, 'name']:<14} n={int(n_unit[u]):>4}  Catholic {sh.loc[u, CATHOLIC]:6.1%}"
              f"  Orthodox {sh.loc[u, 'ORTHODOX CHRISTIAN']:6.1%}  Catholics {int(piv.loc[u, CATHOLIC]):,}")


if __name__ == "__main__":
    main()
