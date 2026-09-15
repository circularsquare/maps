"""Uzbekistan — religion by region, from the Central Asia Barometer (2017-2019) WITHIN ETHNIC
GROUP, applied to the 2026 census's ethnic groups in each region.

Reads the six Uzbek `.dta` files out of `data/raw/cab/*.zip` and the National Statistics
Committee's 2026 census compilation, and writes data/normalized/uz.csv. `sources/cab.py` holds
the survey reader and the tests; `sources/uz.md` has the source hunt, the review (§8) and this
redraw (§9); `sources.md` §9di and §9dp the write-up.

**NO UZBEK CENSUS HAS EVER ASKED ABOUT RELIGION**, the 2026 one included; `uz.md` §1 settles
that against the printed questionnaire. Every row this file writes is `modelled` in §7's sense.

## Why by ethnic group (second build, 2026-09-14)

The first build put each region's survey shares on its population, and Tashkent city came out
21.7% Christian: the survey's city sample is 23% Russian and Russians are three quarters
Christian. The 2026 census counts the city **9.29% Russian (299,725 of 3,224,838)**. Where a
census counts a group whose religion differs sharply, the map draws by group (the citizenship
splits in `be`, `se`, `no`, `dk`; Latvia's non-citizens, `sources/lv.md` §4). So religion within
ethnic group comes from the survey, and each group's size per region from the census.

## The census table

"Preliminary results of the population and agriculture census conducted in the Republic of
Uzbekistan in 2026", National Statistics Committee, Tashkent 2026, English edition, CENSUS_URL.
The same numbers are printed twice: "Ethnic composition of the population, by region" (printed
p. 62, PDF p. 64) and "Distribution of the population by ethnic composition and sex, by region"
(printed pp. 63-70, PDF pp. 65-72). Both are parsed and have to agree cell for cell; every
region's groups have to sum to its total, every group's males and females to the group, and
the fourteen regions to the national row. Census moment 15 January 2026, 39,047,321 people.

## The groups, sized to what both sides carry

    group     census rows                                   survey `Ethnic_M`                answered n
    central   Uzbeks Karakalpaks Kazakhs Tajiks Kyrgyz      Uzbek Karakalpak Kazakh Tajik         8,391
              Turkmens                                      Kyrgyz
    russian   Russians                                      Russian                                 261
    other     Other                                         Tatar, Other (vol.)                     235

The census has no Ukrainian, Belarusian, Korean, Tatar or Armenian row, so "other Slavic" cannot
be its own group: they are inside Other. The survey has no Turkmen code, so its Turkmens answer
Other (vol.); Karakalpakstan's 37 of them are 35 Muslim. Census Turkmens go to `central`, where
their religion belongs, and the survey's few sit in `other`, which leans that group's national
mix Muslim by a few thousand people (the run prints it). 44 answered respondents gave no
ethnicity and are left out of the within-group shares; they are 0.5% of the sample.

## What carries its own geography, per group

`central` gets the usual construction: Sweden's split-half on waves plus the spatial chi-square,
at the 14 regions and at the five DHS 1996 survey regions, Muslim kept as the residual.

`russian` and `other` cannot take a split-half at 14 or 5 units: most (wave, region) cells hold
nobody. They are tested on two units, **Tashkent city against the rest of the country**, where
both have respondents. The test shuffles the city label across whole sampling points (`SamPt`)
within each wave, 2,000 draws, on the weighted share, and requires the chi-square as well. An
answer that passes takes the city's or the rest's share; the others share the remainder at the
group's national proportions, the group's largest answer included.

`SamPt` is the sampling point, in every wave (the first build's "no cluster in the public file"
was wrong; `uz.md` §8). The split-half stays on waves, which a PSU re-run found gives the same
verdicts, and the two-unit test uses the PSU because on two units it is the only clustered
resampling available.

Run 2026-09-14:

    central, 14 regions               n  median rho  null 95th       p    chi2 p
    Muslim                        8,309      +0.645     +0.353  0.0005   7e-22   KEPT AS RESIDUAL
    no particular faith              29      +0.462     +0.360  0.0145   5e-09   region share
    Christian                        17      +0.421     +0.395  0.0425   1e-04   region share
    non-believer                     19      +0.567     +0.373  0.0095   4e-12   region share
    Other (vol.)                     14      +0.411     +0.500  0.1849   2e-26   national rate
    Jewish                            3                                          national rate
    (5 DHS regions: the same four pass and Other (vol.) fails, so nothing is placed only there)

    russian, city 199 / rest 62     city    rest   perm p   chi2 p
    Christian                      77.7%   63.3%   0.069    0.033    THE GROUP'S RESIDUAL
    non-believer                   11.3%    8.6%   0.60     0.92     national rate
    Muslim                          1.3%   26.4%   0.0005   1e-07    city or rest
    no particular faith             9.4%    1.7%   0.14     0.86     national rate

    other, city 94 / rest 141       city    rest   perm p   chi2 p
    Muslim                         41.4%   88.1%   0.0005   2e-10    THE GROUP'S RESIDUAL
    Christian                      20.5%    7.4%   0.034    2e-04    city or rest
    non-believer                   19.2%    2.3%   0.0005   0.030    city or rest
    no particular faith            18.2%    1.2%   0.0005   3e-04    city or rest

Central Christian is the likeliest false pass (17 respondents, p = 0.0425 against 0.05); it is
0.20% of the group, so either way it moves few people.

Usage:
    python sources/uz.py            rebuild data/normalized/uz.csv (about two minutes)
"""

import io
import os
import re
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
LOOKUP = os.path.join(ROOT, "data", "geo", "uz", "uz_lookup.csv")
OUT = os.path.join(ROOT, "data", "normalized", "uz.csv")

COUNTRY = "uzbekistan"
WAVES = [1, 2, 3, 4, 5, 6]
SOURCE_ID = "uz_cab_2017_2019"
N_RESPONDENTS = 9_000
N_ANSWERED = 8_931            # after the 50 don't-knows and 19 refusals on religion
N_GROUPED = 8_887             # of those, the ones who gave an ethnicity

EXCLUDE = ["Don't Know (vol.)", "Refused (vol.)"]

CENSUS_PDF = os.path.join(ROOT, "data", "raw", "uz", "uz_census2026_results_en.pdf")
CENSUS_URL = "https://stat.uz/img/news/english_natija_merged-2_p42445.pdf"
CENSUS_PAGES = 89
CENSUS_TOTAL = 39_047_321
CENSUS_NATION = "Republic of Uzbekistan"
# Census row -> COD pcode, in the census's own row order, which is the office's region order
# that the barometer's codes 4001-4014 and uz_lookup.csv also follow.
CENSUS_REGION = {
    "Republic of Karakalpakstan": "UZ35", "Andijan Region": "UZ03", "Bukhara Region": "UZ06",
    "Jizzakh Region": "UZ08", "Kashkadarya Region": "UZ10", "Navoi Region": "UZ12",
    "Namangan Region": "UZ14", "Samarkand Region": "UZ18", "Surkhandarya Region": "UZ22",
    "Syrdarya Region": "UZ24", "Tashkent Region": "UZ27", "Fergana Region": "UZ30",
    "Khorezm Region": "UZ33", "Tashkent city": "UZ26",
}
CENSUS_GROUPS = ["Uzbeks", "Karakalpaks", "Kazakhs", "Tajiks", "Kyrgyz", "Russians", "Turkmens",
                 "Other"]
CENSUS_TO_GROUP = {"Uzbeks": "central", "Karakalpaks": "central", "Kazakhs": "central",
                   "Tajiks": "central", "Kyrgyz": "central", "Turkmens": "central",
                   "Russians": "russian", "Other": "other"}
SURVEY_TO_GROUP = {"Uzbek (Uzb)": "central", "Karakalpak (Uzb)": "central",
                   "Kazakh (Uzb)": "central", "Tajik (Uzb)": "central", "Kyrgyz (Uzb)": "central",
                   "Russian (Uzb)": "russian", "Tatar (Uzb)": "other", "Other (vol.)": "other"}
NO_GROUP = ["Don't Know (vol.)", "Refused (vol.)"]
GROUPS = ["central", "russian", "other"]
MINORITY = ["russian", "other"]
CITY = "UZ26"

# The DHS 1996 survey regions (Uzbekistan DHS 1996 final report, chapter 1, section 1.6 and
# Figure 1.1): Tashkent City alone, and four groups of contiguous oblasts. A published design
# grouping rather than one drawn for this map, which is the point of using it.
DHS_REGION = {
    "UZ35": "W", "UZ33": "W",
    "UZ12": "S", "UZ06": "S", "UZ10": "S", "UZ22": "S",
    "UZ18": "C", "UZ08": "C", "UZ24": "C", "UZ27": "C",
    "UZ14": "E", "UZ30": "E", "UZ03": "E",
    "UZ26": "T",
}
DHS_NAME = {
    "W": "Karakalpakstan and Khorezm",
    "S": "Navoi, Bukhara, Kashkadarya and Surkhandarya",
    "C": "Samarkand, Jizzakh, Syrdarya and Tashkent region",
    "E": "Namangan, Fergana and Andijan",
    "T": "Tashkent city",
}

KEEP_AS_RESIDUAL = {"Muslim"}

OVERRIDE = {
    "Other (vol.)":
        "refused if it passes: 11 of its 18 respondents are wave 4's Bukhara interviews, in five "
        "sampling points, and all eleven were recorded by one interviewer; the other Bukhara "
        "interviewer that wave recorded none in 46 (uz.md §8). An interviewer effect, which "
        "neither a wave split nor a PSU split can see. Spread at the national rate.",
}

TWO_UNIT_PERM = 2000

# What the tests select, asserted so a change in the data or the tests is a failure here
# rather than a silent re-drawing of the country. Filled from the 2026-09-14 run.
CENTRAL_FINE = ["A believer of no particular faith", "A non-believer", "Christian"]
CENTRAL_COARSE = []
MINORITY_PLACED = {"russian": ["Muslim"],
                   "other": ["A believer of no particular faith", "A non-believer", "Christian"]}


# ----------------------------------------------------------------------------- the census

def _count(s):
    t = re.sub(r"\s", "", s)
    if not t.isdigit():
        raise SystemExit(f"census table: expected a count, found {s!r}")
    return int(t)


def _lines(page):
    raw = [ln.strip() for ln in page.get_text().splitlines()]
    out = []
    for ln in raw:
        if not ln:
            continue
        if ln == "Karakalpakstan" and out and out[-1] == "Republic of":
            out[-1] = "Republic of Karakalpakstan"         # wraps in the long table
            continue
        out.append("Republic of Karakalpakstan" if ln == "Rep. of Karakalpakstan" else ln)
    return out


def census_groups():
    """The 2026 census's eight groups for the 14 regions, from two tables that must agree."""
    import fitz
    if not os.path.exists(CENSUS_PDF):
        raise SystemExit(f"{CENSUS_PDF} is missing: fetch {CENSUS_URL} (stat.uz needs curl -k)")
    doc = fitz.open(CENSUS_PDF)
    if doc.page_count != CENSUS_PAGES:
        raise SystemExit(f"census PDF has {doc.page_count} pages, expected {CENSUS_PAGES}")
    rows = [CENSUS_NATION] + list(CENSUS_REGION)

    wide_lines = _lines(doc[63])
    if "Ethnic composition of the population, by region" not in wide_lines:
        raise SystemExit("PDF page 64 is not the ethnic composition table")
    wide = {}
    for i, ln in enumerate(wide_lines):
        if ln in rows:
            wide[ln] = [_count(x) for x in wide_lines[i + 1:i + 10]]

    long, cur = {}, None
    for p in range(64, 72):
        ls = _lines(doc[p])
        for i, ln in enumerate(ls):
            if ln in rows:
                cur = ln
                t, m, f = (_count(x) for x in ls[i + 1:i + 4])
                if t != m + f:
                    raise SystemExit(f"census long table: {ln} {t} != {m} + {f}")
                long[cur] = {"Total": t}
            elif ln in CENSUS_GROUPS and cur is not None:
                t, m, f = (_count(x) for x in ls[i + 1:i + 4])
                if t != m + f:
                    raise SystemExit(f"census long table: {cur} {ln} {t} != {m} + {f}")
                if ln in long[cur]:
                    raise SystemExit(f"census long table: {cur} {ln} read twice")
                long[cur][ln] = t

    for r in rows:
        if r not in wide or r not in long:
            raise SystemExit(f"census: row {r} missing (wide {r in wide}, long {r in long})")
        lv = [long[r]["Total"]] + [long[r].get(g) for g in CENSUS_GROUPS]
        if wide[r] != lv:
            raise SystemExit(f"census: the two tables disagree on {r}: {wide[r]} against {lv}")
        if sum(lv[1:]) != lv[0]:
            raise SystemExit(f"census: {r}'s groups sum to {sum(lv[1:]):,}, not {lv[0]:,}")
    for j, col in enumerate(["Total"] + CENSUS_GROUPS):
        s = sum(wide[r][j] for r in CENSUS_REGION)
        if s != wide[CENSUS_NATION][j]:
            raise SystemExit(f"census: the 14 regions' {col} sum to {s:,}, the nation to "
                             f"{wide[CENSUS_NATION][j]:,}")
    if wide[CENSUS_NATION][0] != CENSUS_TOTAL:
        raise SystemExit(f"census national total {wide[CENSUS_NATION][0]:,}, expected "
                         f"{CENSUS_TOTAL:,}")
    out = pd.DataFrame([wide[r] for r in CENSUS_REGION], columns=["Total"] + CENSUS_GROUPS,
                       index=[CENSUS_REGION[r] for r in CENSUS_REGION])
    print(f"  census 2026, two tables agree on all {len(rows)} rows x 9 columns; 14 regions sum "
          f"to {CENSUS_TOTAL:,}")
    return out


# ----------------------------------------------------------------------------- the survey

def survey_extras(waves):
    """`Ethnic_M` and `SamPt` for the same rows in the same order as `cab.load`, with the region
    and religion codes read again so the alignment can be asserted rather than assumed."""
    frames = []
    for w in waves:
        _, blob = cab._dta(w, COUNTRY)
        with pd.io.stata.StataReader(io.BytesIO(blob)) as rd:
            d = rd.read(convert_categoricals=False)
            sets = rd.value_labels()
            lbl = dict(zip(rd._varlist, rd._lbllist))
        labels = sets.get(lbl.get("Ethnic_M"))
        if not labels:
            raise SystemExit(f"wave {w}: Ethnic_M has no value labels")
        if d["Ethnic_M"].isna().any() or d["SamPt"].isna().any():
            raise SystemExit(f"wave {w}: Ethnic_M or SamPt has missing values")
        codes = d["Ethnic_M"].astype(int)
        unknown = sorted(set(codes) - set(labels))
        if unknown:
            raise SystemExit(f"wave {w}: Ethnic_M codes with no label: {unknown}")
        frames.append(pd.DataFrame({
            "ethnic": codes.map(lambda c: cab._norm(labels[c])).to_numpy(),
            "psu": d["SamPt"].astype(int).to_numpy(),
            "region_check": d["Region_M"].astype(int).to_numpy(),
            "answer_check": d["Religion_M"].astype(int).to_numpy(),
        }))
    return pd.concat(frames, ignore_index=True)


def two_unit_test(sub, cats, residual, label):
    """Tashkent city against the rest of the country for one small group.

    Statistic: |weighted share in the city - weighted share elsewhere|, per answer. Null: the
    city flag shuffled across whole sampling points within each wave. Veto: the chi-square on
    the unweighted 2x2. Returns the answers that pass both, the residual excluded.
    """
    from scipy.stats import chi2_contingency
    p = sub.groupby(["wave", "psu"]).agg(city=("city", "first"), sides=("city", "nunique"))
    if (p["sides"] != 1).any():
        raise SystemExit(f"{label}: a sampling point has respondents on both sides of the city")
    wt = (sub.pivot_table(index=["wave", "psu"], columns="code", values="w", aggfunc="sum",
                          fill_value=0.0)
          .reindex(index=p.index, columns=cats, fill_value=0.0).to_numpy())
    tot = wt.sum(axis=1)
    city = p["city"].to_numpy(bool)
    waves = p.index.get_level_values(0).to_numpy()

    def stat(flag):
        return np.abs(wt[flag].sum(axis=0) / tot[flag].sum()
                      - wt[~flag].sum(axis=0) / tot[~flag].sum())

    obs = stat(city)
    rng = np.random.default_rng(cab.STAB_SEED)
    idx = [np.where(waves == w)[0] for w in np.unique(waves)]
    null = np.empty((TWO_UNIT_PERM, len(cats)))
    for k in range(TWO_UNIT_PERM):
        f = city.copy()
        for ii in idx:
            f[ii] = city[ii][rng.permutation(len(ii))]
        null[k] = stat(f)
    pp = (1 + (null >= obs - 1e-12).sum(axis=0)) / (1 + TWO_UNIT_PERM)

    n = sub.groupby(["city", "code"]).size().unstack(fill_value=0).reindex(
        index=[True, False], columns=cats, fill_value=0)
    share = sub.groupby(["city", "code"])["w"].sum().unstack(fill_value=0.0).reindex(
        index=[True, False], columns=cats, fill_value=0.0)
    share = share.div(share.sum(axis=1), axis=0)
    print(f"\n  {label}: Tashkent city ({int(n.loc[True].sum())} answered, "
          f"{int(p['city'].sum())} sampling points) against the rest ({int(n.loc[False].sum())}, "
          f"{int((~p['city']).sum())}); {TWO_UNIT_PERM}-draw PSU shuffle within wave + chi-square")
    print(f"    {'answer':<36}{'n':>5}{'city':>8}{'rest':>8}{'perm p':>9}{'chi2 p':>10}  verdict")
    placed = []
    for j, c in enumerate(cats):
        tab = np.array([[n.loc[True, c], n.loc[True].sum() - n.loc[True, c]],
                        [n.loc[False, c], n.loc[False].sum() - n.loc[False, c]]])
        try:
            chi = float(chi2_contingency(tab)[1])
        except ValueError:
            chi = float("nan")
        if c == residual:
            verdict = "the group's residual"
        elif pp[j] < cab.STAB_ALPHA and np.isfinite(chi) and chi < cab.STAB_ALPHA:
            verdict = "city and rest apart"
            placed.append(c)
        else:
            verdict = "group national rate"
        print(f"    {c[:34]:<36}{int(n[c].sum()):>5}{share.loc[True, c]:8.1%}"
              f"{share.loc[False, c]:8.1%}{pp[j]:9.4f}{chi:10.2e}  {verdict}")
    return placed


# ----------------------------------------------------------------------------- the build

def main():
    lut = pd.read_csv(LOOKUP).set_index("geo_id")
    units = list(lut.index)
    names = lut["name"].to_dict()

    census = census_groups()
    if list(census.index) != units:
        raise SystemExit(f"census row order {list(census.index)} is not uz_lookup.csv's {units}")
    ratio = census["Total"] / lut["pop"]
    print("  census 15 January 2026 against SIAT 1 January 2026, per region: " + ", ".join(
        f"{names[u]} {ratio[u]:.3f}" for u in units))
    if ratio.min() < 0.9 or ratio.max() > 1.25:
        raise SystemExit("a census region is more than 10% below or 25% above SIAT's: check the "
                         "join before trusting it")
    pop = census["Total"].astype("int64")
    grp_pop = pd.DataFrame({g: census[[c for c in CENSUS_GROUPS if CENSUS_TO_GROUP[c] == g]]
                           .sum(axis=1) for g in GROUPS})

    df = cab.load(COUNTRY, WAVES)
    if len(df) != N_RESPONDENTS:
        raise SystemExit(f"{len(df):,} respondents, expected {N_RESPONDENTS:,}")
    pairs = set(zip(df["region_code"], df["region"]))
    written = set(zip(lut["cab_code"], lut["cab_region"]))
    if pairs != written:
        raise SystemExit(f"CAB region code/label pairs differ from uz_lookup.csv: "
                         f"{sorted(pairs ^ written)}")
    df["geo_id"] = df["region_code"].map(dict(zip(lut["cab_code"], lut.index)))
    df["coarse"] = df["geo_id"].map(DHS_REGION)

    ex = survey_extras(WAVES)
    if (len(ex) != len(df) or (ex["region_check"].to_numpy() != df["region_code"].to_numpy()).any()
            or (ex["answer_check"].to_numpy() != df["answer_code"].to_numpy()).any()):
        raise SystemExit("Ethnic_M/SamPt rows do not line up with cab.load's rows")
    df["ethnic"] = ex["ethnic"].to_numpy()
    df["psu"] = ex["psu"].to_numpy()
    stray = sorted(set(df["ethnic"]) - set(SURVEY_TO_GROUP) - set(NO_GROUP))
    if stray:
        raise SystemExit(f"Ethnic_M answers with no group: {stray}")
    spans = df.groupby(["wave", "psu"])["geo_id"].nunique()
    if spans.max() != 1:
        raise SystemExit(f"{int((spans > 1).sum())} sampling points span two regions")
    print(f"  {df.groupby(['wave', 'psu']).ngroups} sampling points, each inside one region")

    per = df.groupby(["wave", "geo_id"]).size().unstack(fill_value=0)
    print(f"  respondents per region per wave: min {per.to_numpy().min()} "
          f"({per.stack().idxmin()}), max {per.to_numpy().max()}; every region in every wave")

    lits.held_out(df, pop, "Uzbekistan",
                  pop_source="the 2026 census's region counts",
                  names=[names[u] for u in units])

    # Quota test on the whole answer column, don't-knows and refusals included: the quota is a
    # property of the fieldwork and not of the answers this map draws (spec §12, Lebanon).
    # The wave list is passed explicitly (cab.py docstring, the trap it fixes).
    cab.assert_not_quota(df, "Uzbekistan", WAVES)

    answered = df[~df["code"].isin(EXCLUDE)].copy()
    if len(answered) != N_ANSWERED:
        raise SystemExit(f"{len(answered):,} answered respondents, expected {N_ANSWERED:,}")
    wex = df.loc[df["code"].isin(EXCLUDE), "w"].sum() / df["w"].sum()
    print(f"\n  {len(df) - len(answered)} of {len(df):,} respondents excluded "
          f"({', '.join(EXCLUDE)}), {wex:.2%} weighted")
    kept = answered[~answered["ethnic"].isin(NO_GROUP)].copy()
    if len(kept) != N_GROUPED:
        raise SystemExit(f"{len(kept):,} answered respondents with an ethnicity, expected "
                         f"{N_GROUPED:,}")
    nog = answered.loc[answered["ethnic"].isin(NO_GROUP), "w"].sum() / answered["w"].sum()
    print(f"  {len(answered) - len(kept)} more gave no ethnicity and sit out of the within-group "
          f"shares ({nog:.2%} weighted)")
    kept["group"] = kept["ethnic"].map(SURVEY_TO_GROUP)

    # Survey against census, group shares, nationally and in the two Tashkent units.
    print(f"\n    {'group':<10}{'answered':>9}{'survey nat.':>12}{'census nat.':>12}"
          f"{'survey city':>12}{'census city':>12}{'survey Tash.reg':>16}{'census Tash.reg':>16}")
    for g in GROUPS:
        def sh(frame):
            return frame.loc[frame["group"] == g, "w"].sum() / frame["w"].sum()
        print(f"    {g:<10}{int((kept['group'] == g).sum()):>9,}{sh(kept):12.2%}"
              f"{grp_pop[g].sum() / pop.sum():12.2%}{sh(kept[kept['geo_id'] == CITY]):12.2%}"
              f"{grp_pop.loc[CITY, g] / pop[CITY]:12.2%}"
              f"{sh(kept[kept['geo_id'] == 'UZ27']):16.2%}{grp_pop.loc['UZ27', g] / pop['UZ27']:16.2%}")

    comp, level = {}, {}
    all_cats = list(cab.national(kept).index)

    # --- central: the usual construction
    cen = kept[kept["group"] == "central"]
    nat_c = cab.national(cen)
    cats_c = list(nat_c.index)
    print(f"\n  central group, weighted national: " + ", ".join(
        f"{c} {s:.2%}" for c, s in nat_c.items()))
    fine_pass, _ = cab.stability(cen, cats_c, units, "14 regions, central group")
    print("\n  largest single (wave, region) cell, as a share of each answer's respondents:")
    for c in cats_c:
        cell = cen[cen["code"] == c].groupby(["wave", "geo_id"]).size()
        top = cell.idxmax()
        print(f"    {c[:34]:<36}{int(cell.max()):>5} of {int(cell.sum()):>5} = "
              f"{cell.max() / cell.sum():6.1%}  (wave {top[0]}, {names[top[1]]})")
    coarse_pass, _ = cab.stability(cen, cats_c, sorted(DHS_NAME),
                                   "5 DHS 1996 survey regions, central group", unit_col="coarse")
    stale = sorted(set(OVERRIDE) - set(all_cats))
    if stale:
        raise SystemExit(f"OVERRIDE names answers this country does not have: {stale}")
    fine = [c for c in cats_c if c in fine_pass and c not in KEEP_AS_RESIDUAL and c not in OVERRIDE]
    coarse = [c for c in cats_c if c in coarse_pass and c not in fine
              and c not in KEEP_AS_RESIDUAL and c not in OVERRIDE]
    small = [c for c in cats_c if c not in fine and c not in coarse]
    for c in OVERRIDE:
        if c in fine_pass or c in coarse_pass:
            print(f"\n  REFUSED on a written reason: {c}: {OVERRIDE[c]}")
    print(f"\n  central: {fine} at the region, {coarse} at the DHS survey region, {small} at the "
          "group's national rate inside each region's residual")
    problems = []
    if CENTRAL_FINE is None or sorted(fine) != sorted(CENTRAL_FINE) \
            or sorted(coarse) != sorted(CENTRAL_COARSE):
        problems.append(
            f"the tests now select {sorted(fine)} at the region and {sorted(coarse)} at the survey "
            f"region for the central group, not {CENTRAL_FINE} and {CENTRAL_COARSE}")
    share_f = cab.shares(cen, units, cats_c)
    share_c = cab.shares(cen, sorted(DHS_NAME), cats_c, unit_col="coarse")
    comp["central"] = cab.compose(share_f, share_c, DHS_REGION, nat_c, fine, coarse, small)
    level["central"] = {c: ("region" if c in fine else "DHS survey region" if c in coarse
                            else "national") for c in cats_c}

    # --- russian, other: city against the rest
    placed_all = {}
    for g in MINORITY:
        s = kept[kept["group"] == g].copy()
        nat_g = cab.national(s)
        cats_g = list(nat_g.index)
        cells = s.groupby(["wave", "geo_id"]).size().reindex(
            pd.MultiIndex.from_product([WAVES, units]), fill_value=0)
        dcells = s.groupby(["wave", "coarse"]).size().reindex(
            pd.MultiIndex.from_product([WAVES, sorted(DHS_NAME)]), fill_value=0)
        print(f"\n  {g} group: {len(s)} answered; weighted national " + ", ".join(
            f"{c} {v:.2%}" for c, v in nat_g.items()))
        print(f"    split-half impossible: {int((cells == 0).sum())} of {len(cells)} (wave, region) "
              f"and {int((dcells == 0).sum())} of {len(dcells)} (wave, DHS region) cells are empty")
        s["city"] = s["geo_id"] == CITY
        s["side"] = np.where(s["city"], "city", "rest")
        residual = cats_g[0]
        placed = two_unit_test(s, cats_g, residual, g)
        placed_all[g] = sorted(placed)
        two = cab.shares(s, ["city", "rest"], cats_g, unit_col="side")
        rest_small = [c for c in cats_g if c not in placed]
        c2 = cab.compose(two, two, {}, nat_g, placed, [], rest_small)
        comp[g] = pd.DataFrame([c2.loc["city" if u == CITY else "rest"] for u in units],
                               index=units)
        level[g] = {c: ("city or rest" if c in placed else "national") for c in cats_g}
        if g == "other":
            kk = s[~s["geo_id"].isin(["UZ35", "UZ33"]) & ~s["city"]]
            rr = s[~s["city"]]
            for c in ("Muslim", "Christian"):
                a = rr.loc[rr["code"] == c, "w"].sum() / rr["w"].sum()
                b = kk.loc[kk["code"] == c, "w"].sum() / kk["w"].sum()
                print(f"    other, rest of country, {c}: {a:.1%}; without Karakalpakstan and "
                      f"Khorezm, where Other (vol.) is the survey's Turkmens: {b:.1%}; "
                      f"x {int(grp_pop.loc[grp_pop.index != CITY, 'other'].sum()):,} people = "
                      f"{(b - a) * grp_pop.loc[grp_pop.index != CITY, 'other'].sum():+,.0f}")
    if MINORITY_PLACED is None or placed_all != MINORITY_PLACED:
        problems.append(f"the two-unit test now places {placed_all}, not {MINORITY_PLACED}")
    if problems:
        raise SystemExit("; ".join(problems) + ". That is a change in what this country claims to "
                         "know: read the tables above, then update CENTRAL_FINE, CENTRAL_COARSE, "
                         "MINORITY_PLACED and the docstring.")

    # --- combine: people per (region, answer) = sum over groups of share x census group size
    expected = pd.DataFrame(0.0, index=units, columns=all_cats)
    for g in GROUPS:
        expected += comp[g].reindex(columns=all_cats, fill_value=0.0).mul(grp_pop[g], axis=0)
    for g in GROUPS:
        err = (comp[g].sum(axis=1) - 1).abs().max()
        if err > 1e-9:
            raise SystemExit(f"{g} composition does not close, worst {err:.2e}")
    region_comp = expected.div(pop, axis=0)
    level_of = {c: "religion within ethnic group" for c in all_cats}
    out = cab.counts(region_comp, pop, level_of)
    total = int(out["count"].sum())
    if total != CENSUS_TOTAL:
        raise SystemExit(f"drawn {total:,} against the census's {CENSUS_TOTAL:,}")

    lits.lean(df, out, EXCLUDE, units)

    # Spec §12, Tajikistan: the category with the sharpest known geography has to come back in
    # its home region.
    top = region_comp["Christian"].idxmax()
    if top != CITY:
        raise SystemExit(f"Christian share is highest in {names[top]}, not Tashkent city")

    old = None
    if os.path.exists(OUT):
        old = pd.read_csv(OUT, dtype={"geo_id": str})

    n_reg = df.groupby("geo_id").size()
    out["geo_level"] = "region"
    out["geo_name"] = out["geo_id"].map(names)
    out["basis"] = "self_id"
    out["year"] = "2017-2019"
    out["source_id"] = SOURCE_ID
    out["note"] = out["geo_id"].map(
        lambda u: (f"Central Asia Barometer waves 1-6, n={int(n_reg[u])} in this region; religion "
                   "within three ethnic groups (Uzbek and other Central Asian, Russian, other) "
                   "applied to the 2026 census's ethnic groups in this region, 15 January 2026"))
    cols = ["geo_id", "geo_level", "geo_name", "source_category", "count",
            "basis", "year", "source_id", "note"]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out[cols].to_csv(OUT, index=False, encoding="utf-8")
    print(f"\nwrote {OUT} ({len(out)} rows, {total:,} people, "
          f"{out['source_category'].nunique()} answers, {out['geo_id'].nunique()} regions)")

    print("\n  national, as drawn:")
    drawn_nat = (out.groupby("source_category")["count"].sum() / total).sort_values(
        ascending=False)
    for c, sh in drawn_nat.items():
        print(f"    {sh * 100:7.3f}%  {int(out.loc[out['source_category'] == c, 'count'].sum()):>12,}"
              f"  {c}")

    drawn = out.pivot_table(index="geo_id", columns="source_category", values="count",
                            aggfunc="sum")
    drawn = drawn.div(drawn.sum(axis=1), axis=0)
    oldc = None
    if old is not None:
        op = old.pivot_table(index="geo_id", columns="source_category", values="count",
                             aggfunc="sum")
        oldc = op["Christian"] / op.sum(axis=1)
    print(f"\n    {'region':<16}{'census':>11}{'Russian':>9}{'Muslim':>9}{'Christ.':>9}"
          f"{'non-bel.':>9}{'no part.':>9}{'first build Christ.':>21}")
    for u in drawn["Christian"].sort_values(ascending=False).index:
        prev = f"{oldc[u] * 100:8.2f}%" if oldc is not None and u in oldc else ""
        print(f"    {names[u]:<16}{int(pop[u]):>11,}{grp_pop.loc[u, 'russian'] / pop[u] * 100:8.2f}%"
              f"{drawn.loc[u, 'Muslim'] * 100:8.2f}%{drawn.loc[u, 'Christian'] * 100:8.2f}%"
              f"{drawn.loc[u, 'A non-believer'] * 100:8.2f}%"
              f"{drawn.loc[u, 'A believer of no particular faith'] * 100:8.2f}%{prev:>21}")
    c_n = int(out.loc[out["source_category"] == "Christian", "count"].sum())
    c_t = int(out.loc[(out["source_category"] == "Christian") & (out["geo_id"] == CITY),
                      "count"].sum())
    print(f"\n    Christians drawn: {c_n:,}, of whom {c_t:,} ({c_t / c_n:.1%}) in Tashkent city")
    if old is not None:
        o_n = int(old.loc[old["source_category"] == "Christian", "count"].sum())
        o_t = int(old.loc[(old["source_category"] == "Christian") & (old["geo_id"] == CITY),
                          "count"].sum())
        print(f"    first build: {o_n:,}, of whom {o_t:,} in Tashkent city")
    print("\n    Tashkent city by group (people):")
    for g in GROUPS:
        row = comp[g].loc[CITY] * grp_pop.loc[CITY, g]
        print(f"      {g:<9}{int(grp_pop.loc[CITY, g]):>10,}  " + ", ".join(
            f"{c} {int(round(v)):,}" for c, v in row.items() if v >= 0.5))


if __name__ == "__main__":
    main()
