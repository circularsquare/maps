"""The Central Asia Barometer — the shared machinery for every country drawn from it.

`sources/uz.py` is the first. `sources.md` §11ak opened the source for Tajikistan (where the
religion item is excluded by design), §11ao scanned the other countries in the same archives
and found Uzbekistan (waves 1-6) and Turkmenistan (waves 4-6, and 14 by phone).

## What the files are

`data/raw/cab/CAB-Survey-Wave-<N>-All-Countries-And-Files-<year>-<season>.zip` (waves 1-9) and
`.rar` (10-14), from the open Discuss Data mirror, no form. Each zip holds one Stata file per
country, `central-asia-barometer-survey-wave-<N>-stata-<country>-<year>-<season>.dta`, plus the
questionnaires and the methods report. This module reads the `.dta` straight out of the zip.

Three columns matter and they have the same names in waves 1-6:

    Region_M     region, coded 4001-4014 for Uzbekistan
    Religion_M   101 Christian / 102 Muslim / 103 Jewish / 104 a believer of another faith /
                 105 a believer of no particular faith / 106 a non-believer /
                 996 Other (vol.) / 998 Refused (vol.) / 999 Don't Know (vol.)
    totwt        "Total Weight CAB OTS W1-W6", mean 1.0 within each wave, post-stratified on
                 region x urban/rural, age and gender (wave 4 methods report, p. 20)

**Read the codes and map each column through its OWN label set.** Wave 1's `TypeProb2` has a
duplicated value label, so `convert_categoricals=True` raises on the whole file. `load` reads
integers, looks each code up in the label set Stata attached to that column in that file, and
asserts that a code means the same words in every wave before anything is pooled.

## THE SPLIT-HALF IS ON WAVES (AND THE PSU IS IN THE FILE AFTER ALL)

The design is settlements as PSUs, ten interviews each, stratified by region x urban/rural (27
strata in Uzbekistan). **Correction, 2026-09-14 review (`sources/uz.md` §8): `SamPt` is the
sampling point, in all six Uzbek waves, 902 PSUs each inside one region; `IntCode` is the
interviewer.** This module was written believing no cluster column existed. A PSU re-run of
Uzbekistan's split-half kept every verdict, so the construction below stands; `sources/uz.py`
reads `SamPt` itself for its two-unit tests. Splitting rows would split clusters and count
each one twice, so the resampling unit here is the WAVE. Six waves give ten distinct
three-against-three halvings, and `stability` is Sweden's construction (`sources/se.py`,
spec §12 "ONE SPLIT-HALF IS A DRAW"): the median Spearman over every halving, against a null
that permutes the unit labels independently within each wave (§9cy's rule: a single global
relabelling moves no rank correlation and returns p = 1 everywhere), plus the spatial
chi-square as a veto (spec §12 "A RANK TEST CAN BE PASSED BY A COLUMN THAT IS MOSTLY ZERO").

## THE QUOTA TEST HAS TO BE TOLD THE WAVES

`arabbarometer.quota_agreement` walks pairs of waves in the order it is given. Before
2026-09-14 that order was always `ab.WAVE_NAMES`, so a frame whose waves are `1..6` matched no
Arab Barometer wave, compared nothing, returned `None`, and `assert_not_quota` passed without
raising. `assert_not_quota` here passes `waves=` explicitly and requires every pair to have
been compared.

Usage: imported. `sources/uz.py` is the worked example.
"""

import glob
import io
import os
import re
import sys
import zipfile

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "cab")

REGION = "Region_M"
RELIGION = "Religion_M"
WEIGHT = "totwt"

STAB_ALPHA = 0.05
STAB_PERM = 2000
STAB_SEED = 0


def _norm(label):
    """Collapse whitespace; the religion prompt ends in `?` in one wave and `…?` in another,
    but these are VALUE labels, which carry no such punctuation drift."""
    return re.sub(r"\s+", " ", str(label)).strip()


def _dta(wave, country):
    zips = glob.glob(os.path.join(RAW, f"CAB-Survey-Wave-{wave}-All-Countries-And-Files-*.zip"))
    if len(zips) != 1:
        raise SystemExit(f"wave {wave}: expected one zip in {RAW}, found {zips} (waves 10-14 "
                         "are .rar and are not read here)")
    z = zipfile.ZipFile(zips[0])
    names = [n for n in z.namelist() if f"-stata-{country.lower()}-" in n.lower()]
    if len(names) != 1:
        raise SystemExit(f"wave {wave}: expected one {country} .dta, found {names}")
    return names[0], z.read(names[0])


def read_wave(wave, country):
    """One wave's respondents: `wave`, `region_code`, `region`, `answer_code`, `code`, `w`.

    `code` is the religion answer's label text, which is what the mapping keys on.
    """
    name, blob = _dta(wave, country)
    with pd.io.stata.StataReader(io.BytesIO(blob)) as rd:
        # The whole file, then the three columns: `read(columns=...)` subsets `_varlist`
        # without `_lbllist`, and the label lookup below then finds no label set at all.
        df = rd.read(convert_categoricals=False)
        sets = rd.value_labels()
        lbl = dict(zip(rd._varlist, rd._lbllist))
    out = pd.DataFrame(index=range(len(df)))
    for col, code_col, text_col in ((REGION, "region_code", "region"),
                                    (RELIGION, "answer_code", "code")):
        if col not in df.columns:
            raise SystemExit(f"wave {wave}: {name} has no {col} column. Columns are renamed in "
                             "the phone waves (wave 14: DD13 religion, MM10 region), and the "
                             "religion item is not asked of every country in every wave "
                             "(Tajikistan never, Uzbekistan not in waves 7-13); scan the country's "
                             "files and write the result down before choosing waves "
                             "(playbooks/cab.md).")
        if df[col].isna().any():
            raise SystemExit(f"wave {wave}: {int(df[col].isna().sum())} respondents with no "
                             f"{col}; §11ao found it answered by all of them")
        labels = sets.get(lbl.get(col))
        if not labels:
            raise SystemExit(f"wave {wave}: {col} has no value label set attached")
        codes = df[col].astype(int)
        unknown = sorted(set(codes) - set(labels))
        if unknown:
            raise SystemExit(f"wave {wave}: {col} codes with no label: {unknown}")
        out[code_col] = codes.to_numpy()
        out[text_col] = codes.map(lambda c: _norm(labels[c])).to_numpy()
    # AN ITEM CAN BE IN THE FILE AND NOT ASKED: wave 14 codes all 1,500 Tajiks and every Uzbek
    # `Not Asked` (sources/tj.md §4). Added 2026-09-14: any `Not Asked` religion answer stops.
    not_asked = int(out["code"].str.casefold().str.contains("not asked").sum())
    if not_asked:
        raise SystemExit(f"wave {wave}: {not_asked} of {len(out)} {country} respondents are coded "
                         f"`Not Asked` on {RELIGION}; this wave did not ask this country the item.")
    out["w"] = df[WEIGHT].astype(float).to_numpy()
    if out["w"].isna().any() or (out["w"] <= 0).any():
        raise SystemExit(f"wave {wave}: {WEIGHT} has missing or non-positive values")
    out["wave"] = wave
    return out, name


# Waves 1-6 were face to face. Wave 7 moved to mobile phones, and the religion item and its level
# moved with it: Turkmenistan's wave 14 reads Ashgabat 8.44% Christian against 35.32% face to face
# (sources.md §11ao, sources/tm.md §4). A phone wave can witness an ordering, never a level.
PHONE_FROM_WAVE = 7


def load(country, waves, phone_witness=False):
    """Every respondent in `waves`, with each code's label asserted identical across waves.

    Added 2026-09-14: a list mixing face-to-face and phone waves always stops, and phone waves on
    their own stop unless `phone_witness=True` says they are read as a witness, not a pool.
    """
    phone = [w for w in waves if int(w) >= PHONE_FROM_WAVE]
    if phone and len(phone) < len(list(waves)):
        raise SystemExit(f"waves {list(waves)} mix face-to-face waves with the phone waves {phone}. "
                         "The mode change moved the item and its level; never pool across it.")
    if phone and not phone_witness:
        raise SystemExit(f"waves {phone} were fielded by mobile phone. Load them apart with "
                         "phone_witness=True and use them to witness an ordering, never a level "
                         "(tm.py::wave14 is the pattern).")
    frames = []
    print(f"Central Asia Barometer, {country}, waves {list(waves)}:")
    for w in waves:
        d, name = read_wave(w, country)
        print(f"  wave {w}: {len(d):,} respondents, {d['region'].nunique()} regions, "
              f"weight mean {d['w'].mean():.3f}   ({name})")
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    for code_col, text_col in (("region_code", "region"), ("answer_code", "code")):
        per = df.groupby(code_col)[text_col].unique()
        drift = {int(k): list(v) for k, v in per.items() if len(v) != 1}
        if drift:
            raise SystemExit(f"a {code_col} means different words in different waves: {drift}")
        per = df.groupby(text_col)[code_col].unique()
        drift = {k: list(v) for k, v in per.items() if len(v) != 1}
        if drift:
            raise SystemExit(f"one {text_col} label sits on two codes: {drift}")
    return df


def national(df):
    """Weighted national shares, indexed by answer label. Each wave's weights have mean 1.0
    over the same 1,500 respondents, so pooling them is pooling equal-sized waves."""
    return (df.groupby("code")["w"].sum() / df["w"].sum()).sort_values(ascending=False)


def assert_not_quota(df, country, waves, unit_col="geo_id", cat_col="code"):
    """Arab Barometer's quota test, TOLD WHICH WAVES TO PAIR, and required to have paired them.

    See the module docstring for the trap. Every pair of the `waves` has to be comparable;
    anything less is printed and stops the build, because a quota test that compared fewer
    pairs than it was handed has quietly tested less than it says.
    """
    import arabbarometer as ab
    worst, n_pairs = ab.assert_not_quota(df, country, unit_col, cat_col, waves=list(waves),
                                         min_pairs=1)
    want = len(waves) * (len(waves) - 1) // 2
    if n_pairs != want:
        raise SystemExit(f"the quota test compared {n_pairs} of the {want} wave pairs; read "
                         "the pairs it printed before trusting a pass")
    print(f"    compared {n_pairs} of {want} wave pairs, so the pass is a pass")
    return worst, n_pairs


def _rho(x, y):
    """Spearman with average ranks. se.py's and be.py's, so all three agree exactly."""
    from scipy.stats import rankdata
    rx, ry = rankdata(x).astype(float), rankdata(y).astype(float)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    sx, sy = np.sqrt((rx ** 2).sum()), np.sqrt((ry ** 2).sum())
    if sx <= 0 or sy <= 0:
        return np.nan
    return float((rx * ry).sum() / (sx * sy))


def stability(df, cats, units, label, unit_col="geo_id", cell_refused=()):
    """WHICH CATEGORIES CARRY THEIR OWN GEOGRAPHY AT THIS LEVEL — se.py::_stability, on waves.

    Unweighted respondent counts, as Sweden. A category passes only if BOTH the median
    split-half Spearman beats the per-wave permutation null at 0.05 AND the spatial chi-square
    (this category against everyone else, across the units, pooled) is under 0.05. A failure
    is a failure to demonstrate signal: the people stay drawn, at a coarser level or at the
    national rate inside the residual.

    Returns `(carries, table)`; `table` is one dict per category for the record.
    """
    import stability as shared      # this function is named `stability` too
    waves = sorted(df["wave"].unique())
    units = list(units)
    ui = {u: i for i, u in enumerate(units)}
    ci = {c: i for i, c in enumerate(cats)}
    wi = {w: i for i, w in enumerate(waves)}
    cube = np.zeros((len(waves), len(units), len(cats)))
    for (w, u, k), n in df.groupby(["wave", unit_col, "code"]).size().items():
        if k in ci:
            cube[wi[w], ui[u], ci[k]] += n
    empty = [(waves[a], units[b]) for a, b in zip(*np.where(cube.sum(axis=2) == 0))]
    if empty:
        raise SystemExit(f"(wave, unit) cells with no respondent: {empty}")

    # Every distinct halving, once, odd wave counts included (spec §12, Colombia and
    # Turkmenistan); `sources/stability.py` has the enumeration and its brute-force test.
    splits = shared.halvings(len(waves))
    obs = shared.median_rho(cube, splits)
    null = shared.wave_null(cube, splits, STAB_PERM, STAB_SEED)

    nraw = cube.sum(axis=0)                                  # (unit, category)
    tot = nraw.sum()
    print(f"\n  split-half stability at the {label}: {len(waves)} waves, median of "
          f"{len(splits)} halvings, against a {STAB_PERM}-draw per-wave unit-label permutation "
          "null (§14.16, §9cy), plus the spatial chi-square (§9bi):")
    print(f"    {'answer':<36}{'n':>6}{'share':>8}{'median rho':>12}{'null 95th':>11}"
          f"{'p':>8}{'chi2 p':>10}  verdict")
    carries, table = [], []
    for j, c in enumerate(cats):
        n = int(nraw[:, j].sum())
        chi = shared.chi2_p(nraw[:, j], nraw.sum(axis=1))
        p, q95 = shared.permutation_p(obs[j], null[:, j], STAB_ALPHA)
        row = dict(category=c, n=n, share=n / tot, rho=float(obs[j]), null95=float("nan"),
                   p=float("nan"), chi_p=chi, level=label, passed=False)
        if not np.isfinite(p):
            print(f"    {c[:34]:<36}{n:>6,}{100 * n / tot:7.2f}%{'':>12}{'':>11}{'':>8}"
                  f"{chi:10.2e}  no test possible")
            table.append(row)
            continue
        ok = p < STAB_ALPHA and np.isfinite(chi) and chi < STAB_ALPHA
        row.update(null95=q95, p=p, passed=ok)
        if ok:
            carries.append(c)
            verdict = "own geography"
        elif p < STAB_ALPHA:
            verdict = "REFUSED: passes the rank test, but the units do not differ"
        else:
            verdict = "NOT distinguishable from chance"
        print(f"    {c[:34]:<36}{n:>6,}{100 * n / tot:7.2f}%{obs[j]:+12.3f}"
              f"{row['null95']:+11.3f}{p:8.4f}{chi:10.2e}  {verdict}")
        table.append(row)

    # A CHI-SQUARE CANNOT VETO A CLUSTER (spec §12, Uzbekistan). Added 2026-09-14 as a stop and
    # not a veto, so no verdict moves: a pass with more than `stability.CELL_CAP` of its
    # respondents in one (wave, unit) cell must be refused by the caller, by name, in
    # `cell_refused` (`tm.py::REFUSED`). Answers placed in uz and tm hold at most 28% in one cell;
    # Turkmenistan's refused non-believers hold 57%.
    unknown = sorted(set(cell_refused) - set(cats))
    if unknown:
        raise SystemExit(f"cell_refused names answers that are not tested here: {unknown}")
    heavy = []
    for c in carries:
        col = cube[:, :, ci[c]]
        top = float(col.max() / col.sum())
        if top > shared.CELL_CAP and c not in cell_refused:
            a, b = np.unravel_index(int(col.argmax()), col.shape)
            heavy.append(f"{c}: {top:.0%} of its respondents are wave {waves[a]}, {units[b]}")
    if heavy:
        raise SystemExit(
            f"at the {label}, answers pass with over {shared.CELL_CAP:.0%} of their respondents in "
            "one (wave, unit) cell:\n    " + "\n    ".join(heavy)
            + "\n  A pass that rests on one cell is one interviewer or one sampling point, which "
            "neither a wave split nor the chi-square can see. Refuse it in the country module with "
            "the reason written down and pass its name in `cell_refused` (tm.py::REFUSED).")
    return carries, table


def shares(df, units, cats, unit_col="geo_id"):
    """Weighted share of each answer in each unit, pooled over the waves."""
    t = df.groupby([unit_col, "code"])["w"].sum().unstack(fill_value=0.0)
    t = t.reindex(index=list(units), columns=list(cats), fill_value=0.0)
    return t.div(t.sum(axis=1), axis=0)


def compose(fine_share, coarse_share, parent, nat, fine, coarse, small):
    """Per-unit composition as a closed partition — se.py::_compose with the parent explicit.

    A fine-level category takes its own unit's share; a coarse-level category takes its
    coarse unit's share in every fine unit inside it (`parent` maps fine -> coarse); what is
    left of each unit is divided among `small` at their NATIONAL relative proportions.
    """
    small_total = float(nat[small].sum()) if small else 0.0
    out = pd.DataFrame(index=fine_share.index, columns=fine_share.columns, dtype=float)
    for c in fine:
        out[c] = fine_share[c]
    for c in coarse:
        out[c] = [float(coarse_share.loc[parent[u], c]) for u in fine_share.index]
    residual = 1.0 - out[fine + coarse].sum(axis=1)
    if small:
        if (residual <= 0).any():
            raise SystemExit(f"units with no room for the tail: "
                             f"{sorted(residual[residual <= 0].index)}")
        print(f"  the tail is {residual.min():.2%} of {residual.idxmin()} and "
              f"{residual.max():.2%} of {residual.idxmax()}, against {small_total:.2%} "
              "nationally")
        for c in small:
            out[c] = residual * (nat[c] / small_total)
    elif residual.abs().max() > 1e-9:
        raise SystemExit("no tail, but the placed categories do not close every unit")
    err = (out.sum(axis=1) - 1.0).abs().max()
    if err > 1e-9:
        raise SystemExit(f"composition does not sum to 1, worst {err:.2e}")
    return out


def counts(comp, pop, level_of):
    """Composition x population -> integer rows, rounding drift into the largest cell."""
    rows = []
    for u in comp.index:
        p = int(pop[u])
        for c in comp.columns:
            rows.append((u, c, float(comp.loc[u, c]) * p, level_of[c]))
    out = pd.DataFrame(rows, columns=["geo_id", "source_category", "count", "basis_note"])
    out["count"] = out["count"].round().astype("int64")
    drift = int(sum(int(pop[u]) for u in comp.index)) - int(out["count"].sum())
    if abs(drift) > len(out):
        raise SystemExit(f"rounding drift {drift} is larger than one person per row")
    if drift:
        out.loc[out["count"].idxmax(), "count"] += drift
    print(f"  rounding drift {drift:+d} people, absorbed into the largest cell")
    return out
