"""Turkmenistan — religion by velayat: how each nationality answers the Central Asia Barometer
(waves 4-6, 2018-2019), applied to where the 2022 census counts each nationality.

Reads the three Turkmen `.dta` files out of `data/raw/cab/*.zip`, wave 14's Turkmen `.dta` as a
witness, and the census tables `sources/tm_geo.py` writes; writes data/normalized/tm.csv.
`sources/cab.py` holds the reader and the tests; `sources/tm.md` is the record and `sources.md`
§11ao and §9do the write-up. Anita approved building the country on 2026-09-14, knowing the
split-half has no power on six units.

**NO TURKMEN CENSUS SINCE INDEPENDENCE HAS PUBLISHED RELIGION.** The 2022 census's eleven
published sections carry nationality and language and no religion table. Every row this file
writes is `modelled` (§7).

## THE SURVEY'S SAMPLE IS A 1995 COUNTRY, SO ITS REGIONAL SHARES ARE NOT WHAT IS DRAWN

The barometer's sampling frame and weighting targets for Turkmenistan are the statistics
agency's 1995 figures (wave 4 methods report, p. 14 and Table 6; p. 45). A face-to-face
random-route sample on that frame reaches far more of the Soviet-era Russian population than
is left:

    respondents, weighted            waves 4-6    2022 census (tables 4.1, 4.3)
    Russian, Ukrainian or Armenian       6.4%        1.87% of the country
    ... in Ashgabat                     (printed)    7.71% of the city
    Christian in Ashgabat               35.3%        (no census figure)

Wave 14 (autumn 2023, by mobile phone, 90% Turkmen against the census's 86.7%) reads Ashgabat
at 8.4% Christian. Applying the pooled regional shares to the 2022 population would draw about
four times as many Christians in the capital as there are Russians, Ukrainians and Armenians
living in it.

So the survey is used for what it can say, **how each nationality answers**, and the census for
**where each nationality lives**:

    Christian share of a velayat = sum over nationality cells of
        (the cell's share of the velayat, census 2022) x (Christian share of the cell, survey, national)

The cells are Turkmen, Uzbek, Russian/Ukrainian/Armenian (the three nationalities whose
respondents answer Christian), and everyone else. The within-cell rate is NATIONAL, not per
velayat: there are 248 Russian, Ukrainian and Armenian respondents in all, and a per-velayat
rate would be read off a few dozen. A per-velayat version is printed beside it for the record.
It also means a Turkmen Christian is drawn at the same rate in every velayat: the survey has
two of them, both in Ashgabat, and cannot say where the converts live (§14, `sources/tm.md`).

Muslim is each velayat's residual; the four answers with one to seven respondents share it at
their national proportions, post-stratified the same way.

## THE TESTS, WHICH DO NOT DECIDE ANYTHING HERE AND ARE RUN ANYWAY

Split-half on waves (three waves, all three 1-against-2 halvings since `cab.stability` was
fixed for odd counts, per-wave permutation null over six units), plus the spatial chi-square.
Printed, and what they select is asserted so a change in the data is a failure here. Run
2026-09-14: Muslim and Christian +0.600 against a null 95th of +0.600, p=0.0665, which is the
no-power result §11ao predicted; `A non-believer` passes on seven respondents and is refused in
`REFUSED` (four are one wave's Ashgabat interviews).

The decode is pinned without names by the wave 4 methods report's Table 6, which allocates
each named region a different number of PSUs (`frame_witness`), and `lits.held_out` is run
against that 1995 frame (r=+0.994, none of 719 orderings reaches it). Against the 2022 census
six orderings beat the truth, because Ashgabat took in part of Ahal in 2013. No answer
is placed from its own regional share, so a pass would change nothing that is drawn; the
Christian geography's witnesses are the census's nationality counts and wave 14's ordering,
both printed with an exact permutation p over all 720 orderings.

Usage:
    python sources/tm.py            rebuild data/normalized/tm.csv (under a minute)
"""

import io
import itertools
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "6")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

import cab
import lits

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LOOKUP = os.path.join(ROOT, "data", "geo", "tm", "tm_lookup.csv")
NATIONALITY = os.path.join(ROOT, "data", "geo", "tm", "tm_nationality.csv")
OUT = os.path.join(ROOT, "data", "normalized", "tm.csv")
W14 = os.path.join(cab.RAW, "central-asia-barometer-survey-wave-14-stata-turkmenistan-2023-autumn.dta")
W14_RAR = "CAB-Survey-Wave-14-All-Countries-And-Files-2023-Autumn.rar"

COUNTRY = "turkmenistan"
WAVES = [4, 5, 6]
SOURCE_ID = "tm_cab_2018_2019"
N_RESPONDENTS = 4_500

EXCLUDE = ["Don't Know (vol.)", "Refused (vol.)"]

RUA = "Russian, Ukrainian or Armenian"
CELLS = ["Turkmen", "Uzbek", RUA, "other"]
SURVEY_CELL = {
    "Turkmen (Tur)": "Turkmen", "Uzbek (Tur)": "Uzbek",
    "Russian (Tur)": RUA, "Ukrainian (Tur)": RUA, "Armenian (Tur)": RUA,
    "Azerbaijani (Tur)": "other", "Tatar (Tur)": "other", "Uygur (Tur)": "other",
    "Balochi (Tur)": "other", "Kazakh (Tur)": "other", "Other (vol.)": "other",
}
CENSUS_CELL = {"Turkmens": "Turkmen", "Uzbeks": "Uzbek", "Russians": RUA, "Ukrainians": RUA,
               "Armenians": RUA}          # every other census row is "other"
MIN_CELL_N = 30
MIN_CELL_N_VELAYAT = 30                   # for the per-velayat version, printed only

MODELLED = ["Christian"]
# What cab.stability selects on waves 4-6, asserted (see the module docstring), and why a
# pass is refused. Nothing is placed from its own regional share here, so a refusal is a record.
EXPECTED_PASS = ["A non-believer"]
REFUSED = {
    "A non-believer":
        "passes at p=0.0285 on seven respondents, four of them wave 6's Ashgabat interviews "
        "(largest wave-velayat cell 57%; Christian's is 28%). Spec §12, Uzbekistan: the "
        "chi-square cannot see a cluster. Spread at the national rate inside the residual.",
}


# Wave 4 methods report, Table 6, "Sample Distribution - Turkmenistan": the frame population
# (urban, rural) and the PSUs allocated (urban, rural), per named region. Ten interviews a PSU.
FRAME_1995 = {
    "TM-A": (203_385, 367_876, 8, 13),
    "TM-S": (530_575, 0, 19, 0),
    "TM-B": (270_318, 83_266, 10, 4),
    "TM-D": (301_712, 562_040, 11, 20),
    "TM-L": (388_992, 451_775, 14, 16),
    "TM-M": (254_671, 703_734, 10, 25),
}


def frame_witness(df, names):
    """THE DECODE WITNESS THAT USES NO NAMES. Table 6 allocates each NAMED region a number of
    PSUs, and the six numbers are all different; wave 4's file has to hold ten interviews per
    PSU under the code its label gives that region, and as many distinct sampling points.

    The held-out check against the 2022 census does not pin this decode and is not asked to:
    the sample was allocated on 1995 figures, and Ashgabat took in part of Ahal in 2013, so the
    survey reads Ashgabat thin and Ahal full against 2022 (0.82x and 1.18x) and six orderings
    beat the true one. `lits.held_out` is run against the frame the sample was drawn on.
    """
    w4 = df[df["wave"] == 4]
    got = w4.groupby("geo_id").size()
    psus = w4.groupby("geo_id")["psu"].nunique()
    bad = {u: (int(got[u]), int(psus[u]), 10 * (v[2] + v[3]))
           for u, v in FRAME_1995.items()
           if got[u] != 10 * (v[2] + v[3]) or psus[u] != v[2] + v[3]}
    if bad:
        raise SystemExit(f"wave 4 respondents / sampling points per velayat against Table 6's "
                         f"PSU allocation x 10: {bad}")
    alloc = sorted(v[2] + v[3] for v in FRAME_1995.values())
    if len(set(alloc)) != len(alloc):
        raise SystemExit("Table 6's PSU counts are not all different, so they pin nothing")
    print("  decode witness: wave 4's interviews and sampling points per velayat equal Table 6's "
          "PSU allocation for the region of that name, and the six allocations all differ: "
          + ", ".join(f"{names[u]} {v[2] + v[3]}" for u, v in FRAME_1995.items()))


def extra_columns(wave, cols):
    """Columns cab.read_wave does not return, from the same file in the same row order."""
    _, blob = cab._dta(wave, COUNTRY)
    with pd.io.stata.StataReader(io.BytesIO(blob)) as rd:
        df = rd.read(convert_categoricals=False)
        sets = rd.value_labels()
        lbl = dict(zip(rd._varlist, rd._lbllist))
    out = pd.DataFrame({"_region_code": df[cab.REGION].astype(int).to_numpy()})
    for c in cols:
        labels = sets.get(lbl.get(c))
        if labels:
            out[c] = [cab._norm(labels.get(int(x), x)) for x in df[c]]
        else:
            out[c] = df[c].to_numpy()
    return out


def load():
    df = cab.load(COUNTRY, WAVES)
    ex = pd.concat([extra_columns(w, ["Ethnic_M", "SamPt", "IntCode"]) for w in WAVES],
                   ignore_index=True)
    if len(ex) != len(df) or (ex["_region_code"].to_numpy() != df["region_code"].to_numpy()).any():
        raise SystemExit("the ethnicity/PSU columns do not line up with cab.load's rows")
    df["ethnic"] = ex["Ethnic_M"].to_numpy()
    df["psu"] = df["wave"].astype(str) + ":" + ex["SamPt"].astype(int).astype(str).to_numpy()
    df["interviewer"] = df["wave"].astype(str) + ":" + ex["IntCode"].astype(str).to_numpy()
    unknown = sorted(set(df["ethnic"]) - set(SURVEY_CELL))
    if unknown:
        raise SystemExit(f"Ethnic_M labels with no cell: {unknown}")
    df["cell"] = df["ethnic"].map(SURVEY_CELL)
    return df


def wave14(lut):
    """Wave 14's Turkmen answers: region, answer, weight. A witness, never pooled."""
    if not os.path.exists(W14):
        raise SystemExit(f"{W14} is missing. Extract the Turkmen Stata file from {W14_RAR} "
                         "(Windows' own tar reads rar: tar -xf <rar> -C <dir>) and copy it "
                         "into data/raw/cab/.")
    with pd.io.stata.StataReader(W14) as rd:
        d = rd.read(convert_categoricals=False)
        sets = rd.value_labels()
        lbl = dict(zip(rd._varlist, rd._lbllist))

    def lab(c):
        s = sets[lbl[c]]
        return [cab._norm(s[int(x)]) for x in d[c]]

    out = pd.DataFrame({"region": lab("MM10"), "code": lab("DD13"),
                        "w": d["FinalWgt1"].astype(float).to_numpy()})
    by_label = dict(zip(lut["cab_region"], lut.index))
    unknown = sorted(set(out["region"]) - set(by_label))
    if unknown:
        raise SystemExit(f"wave 14 region labels not in tm_lookup.csv: {unknown}")
    out["geo_id"] = out["region"].map(by_label)
    return out


def exact_rank(x, y):
    """Spearman and its exact one-sided p over every ordering of `y`."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    r = cab._rho(x, y)
    null = [cab._rho(x, y[list(p)]) for p in itertools.permutations(range(len(y)))]
    return r, sum(v >= r - 1e-12 for v in null) / len(null)


def concentration(kept, cats, names):
    """For each answer, the share of its respondents in its largest (wave, velayat) cell, its
    largest sampling point and its largest interviewer (§12, Uzbekistan)."""
    print("\n  where each answer's respondents bunch (largest cell, sampling point, interviewer):")
    for c in cats:
        sub = kept[kept["code"] == c]
        n = len(sub)
        cell = sub.groupby(["wave", "geo_id"]).size()
        w, g = cell.idxmax()
        psu = sub.groupby("psu").size()
        iv = sub.groupby("interviewer").size()
        print(f"    {c[:34]:<36}n={n:>5,}  cell {cell.max() / n:6.1%} (wave {int(w)}, "
              f"{names[g]})  PSU {psu.max() / n:6.1%} of {len(psu)}  interviewer "
              f"{iv.max() / n:6.1%} of {len(iv)}")


def main():
    lut = pd.read_csv(LOOKUP).set_index("geo_id")
    pop = lut["pop"].astype("int64")
    units = sorted(lut.index)
    names = lut["name"].to_dict()

    df = load()
    if len(df) != N_RESPONDENTS:
        raise SystemExit(f"{len(df):,} respondents, expected {N_RESPONDENTS:,}")
    pairs = set(zip(df["region_code"], df["region"]))
    written = set(zip(lut["cab_code"], lut["cab_region"]))
    if pairs != written:
        raise SystemExit(f"CAB region code/label pairs differ from tm_lookup.csv: "
                         f"{sorted(pairs ^ written)}")
    df["geo_id"] = df["region_code"].map(dict(zip(lut["cab_code"], lut.index)))
    per = df.groupby(["wave", "geo_id"]).size().unstack(fill_value=0)
    print(f"  respondents per velayat per wave: min {per.to_numpy().min()} "
          f"({per.stack().idxmin()}), max {per.to_numpy().max()}; every velayat in every wave")

    frame_witness(df, names)
    frame = pd.Series({u: FRAME_1995[u][0] + FRAME_1995[u][1] for u in units})
    lits.held_out(df, frame, "Turkmenistan",
                  pop_source="the barometer's own 1995 frame (wave 4 methods report, Table 6)",
                  names=[names[u] for u in units])
    s = df.groupby("geo_id")["w"].sum() / df["w"].sum()
    k = pop / pop.sum()
    print(f"    against the 2022 census instead (not a test, see the docstring): r = "
          f"{np.corrcoef(s.loc[units], k.loc[units])[0, 1]:+.3f}; "
          + ", ".join(f"{names[u]} {s[u] / k[u]:.2f}x" for u in units))

    # The quota test on the whole answer column (spec §12, Lebanon).
    cab.assert_not_quota(df, "Turkmenistan", WAVES)

    kept = df[~df["code"].isin(EXCLUDE)].copy()
    if len(kept) != len(df):
        raise SystemExit(f"{len(df) - len(kept)} don't-knows or refusals; waves 4-6 had none "
                         "when this was written, so the gap= field and §3.5 lean need writing")
    raw_nat = cab.national(kept)
    cats = list(raw_nat.index)

    passed, _ = cab.stability(kept, cats, units, "6 velayats", cell_refused=REFUSED)
    concentration(kept, cats, names)
    if sorted(passed) != sorted(EXPECTED_PASS):
        raise SystemExit(f"the split-half now passes {sorted(passed)}, not {EXPECTED_PASS}. "
                         "Nothing here is placed from its own regional share, so read the "
                         "tables and update EXPECTED_PASS and the docstring.")
    for c in passed:
        print(f"\n  REFUSED on a written reason: {c}: {REFUSED[c]}")

    # ---- where each nationality lives: the census
    nat_tab = pd.read_csv(NATIONALITY)
    nat_tab["cell"] = nat_tab["nationality"].map(CENSUS_CELL).fillna("other")
    census_cell = (nat_tab.groupby(["geo_id", "cell"])["persons"].sum().unstack(fill_value=0)
                   .reindex(index=units, columns=CELLS, fill_value=0))
    if (census_cell.sum(axis=1) != pop.reindex(units)).any():
        raise SystemExit("tm_nationality.csv does not sum to tm_lookup.csv's populations")
    census_share = census_cell.div(census_cell.sum(axis=1), axis=0)
    survey_cell = (kept.groupby(["geo_id", "cell"])["w"].sum().unstack(fill_value=0)
                   .reindex(index=units, columns=CELLS, fill_value=0))
    survey_share = survey_cell.div(survey_cell.sum(axis=1), axis=0)
    print("\n  nationality cells, survey (weighted, waves 4-6) against the 2022 census:")
    print(f"    {'velayat':<10}" + "".join(f"{c[:12]:>26}" for c in CELLS))
    for u in units + ["national"]:
        if u == "national":
            s = survey_cell.sum() / survey_cell.to_numpy().sum()
            k = census_cell.sum() / census_cell.to_numpy().sum()
        else:
            s, k = survey_share.loc[u], census_share.loc[u]
        print(f"    {names.get(u, u):<10}" + "".join(
            f"{100 * s[c]:11.2f}% vs {100 * k[c]:6.2f}%    " for c in CELLS))

    # ---- how each nationality answers: the survey, nationally
    n_cell = kept.groupby("cell").size().reindex(CELLS, fill_value=0)
    if (n_cell < MIN_CELL_N).any():
        raise SystemExit(f"nationality cells under {MIN_CELL_N} respondents: "
                         f"{n_cell[n_cell < MIN_CELL_N].to_dict()}")
    rate = (kept.groupby(["cell", "code"])["w"].sum().unstack(fill_value=0)
            .reindex(index=CELLS, columns=cats, fill_value=0))
    rate = rate.div(rate.sum(axis=1), axis=0)
    print("\n  answers within each nationality cell, weighted, all velayats pooled:")
    print(f"    {'cell':<32}{'n':>6}" + "".join(f"{c[:14]:>16}" for c in cats))
    for c in CELLS:
        print(f"    {c:<32}{int(n_cell[c]):>6,}" + "".join(
            f"{100 * rate.loc[c, a]:15.2f}%" for a in cats))

    model = census_share.dot(rate)                          # velayat x answer
    nat_cells = census_cell.sum(axis=0)
    nat_model = (nat_cells / nat_cells.sum()).dot(rate)     # answer, post-stratified nationally

    # The per-velayat version, for the record only.
    pv = pd.DataFrame(index=units, columns=cats, dtype=float)
    for u in units:
        acc = pd.Series(0.0, index=cats)
        for c in CELLS:
            sub = kept[(kept["geo_id"] == u) & (kept["cell"] == c)]
            if len(sub) >= MIN_CELL_N_VELAYAT:
                r = sub.groupby("code")["w"].sum().reindex(cats, fill_value=0.0)
                r = r / r.sum()
            else:
                r = rate.loc[c]
            acc += census_share.loc[u, c] * r
        pv.loc[u] = acc

    raw = cab.shares(kept, units, cats)
    w14 = wave14(lut)
    t14 = w14.groupby(["geo_id", "code"])["w"].sum().unstack(fill_value=0.0).reindex(units)
    w14_share = t14.div(t14.sum(axis=1), axis=0)
    print("\n  Christian, by velayat:")
    print(f"    {'velayat':<10}{'waves 4-6':>11}{'per-velayat':>13}{'DRAWN':>9}{'wave 14':>9}"
          f"{'census RUA':>12}")
    for u in sorted(units, key=lambda u: -model.loc[u, "Christian"]):
        print(f"    {names[u]:<10}{100 * raw.loc[u, 'Christian']:10.2f}%"
              f"{100 * pv.loc[u, 'Christian']:12.2f}%{100 * model.loc[u, 'Christian']:8.2f}%"
              f"{100 * w14_share.loc[u, 'Christian']:8.2f}%{100 * census_share.loc[u, RUA]:11.2f}%")
    w14_nat = w14.groupby("code")["w"].sum() / w14["w"].sum()
    print(f"    {'national':<10}{100 * raw_nat['Christian']:10.2f}%{'':>13}"
          f"{100 * nat_model['Christian']:8.2f}%{100 * w14_nat.get('Christian', 0):8.2f}%"
          f"{100 * nat_cells[RUA] / nat_cells.sum():11.2f}%")
    print("\n  witnesses on the drawn Christian ordering, exact one-sided p over 720 orderings:")
    for label, other in (("waves 4-6, own regional shares", raw["Christian"]),
                         ("wave 14 (2023, phone), never pooled", w14_share["Christian"])):
        r, p = exact_rank(model.loc[units, "Christian"], other.loc[units])
        print(f"    against {label:<40} rho {r:+.3f}  p {p:.4f}")
    if model["Christian"].idxmax() != "TM-S":
        raise SystemExit(f"Christian is highest in {names[model['Christian'].idxmax()]}, "
                         "not Ashgabat")

    small = [c for c in cats if c not in MODELLED]
    comp = cab.compose(model, None, {}, nat_model, MODELLED, [], small)
    level_of = {c: ("national share within each census nationality, applied to the velayat's "
                    "2022 nationality counts" if c in MODELLED else
                    "national share, post-stratified on nationality, within the velayat's "
                    "residual") for c in cats}
    out = cab.counts(comp, pop, level_of)
    total = int(out["count"].sum())
    if total != int(pop.sum()):
        raise SystemExit(f"drawn {total:,} against the census's {int(pop.sum()):,}")
    print("\n  no respondent in waves 4-6 refused or said they did not know, so §3.5 has "
          "nothing to lean")

    n_reg = df.groupby("geo_id").size()
    out["geo_level"] = "velayat"
    out["geo_name"] = out["geo_id"].map(names)
    out["basis"] = "self_id"
    out["year"] = "2018-2019"
    out["source_id"] = SOURCE_ID
    out["note"] = out.apply(
        lambda r: (f"Central Asia Barometer waves 4-6, n={int(n_reg[r.geo_id])} in this "
                   f"velayat; {r.basis_note}; 2022 census population"), axis=1)
    cols = ["geo_id", "geo_level", "geo_name", "source_category", "count",
            "basis", "year", "source_id", "note"]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out[cols].to_csv(OUT, index=False, encoding="utf-8")
    print(f"\nwrote {OUT} ({len(out)} rows, {total:,} people, "
          f"{out['source_category'].nunique()} answers, {out['geo_id'].nunique()} velayats)")

    print("\n  national, as drawn:")
    drawn_nat = (out.groupby("source_category")["count"].sum() / total).sort_values(
        ascending=False)
    for c, s in drawn_nat.items():
        print(f"    {s * 100:7.3f}%  {int(out.loc[out['source_category'] == c, 'count'].sum()):>12,}"
              f"  {c}")
    ch = out[out["source_category"] == "Christian"].set_index("geo_id")["count"]
    print(f"\n    Christians drawn: {int(ch.sum()):,}, of whom {int(ch['TM-S']):,} "
          f"({ch['TM-S'] / ch.sum():.1%}) in Ashgabat; by velayat: "
          + ", ".join(f"{names[u]} {int(ch[u]):,}" for u in ch.sort_values(ascending=False).index))


if __name__ == "__main__":
    main()
