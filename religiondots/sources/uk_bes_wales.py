"""The British Election Study internet panel -> Wales's Christian denominations, self-ID, by area.

`sources/uk_bes_wales.py` -> `data/normalized/uk_bes_wales.csv`

WHY THIS EXISTS. England and Wales publish no Christian denomination at any geography
(sources/uk.md §2). England has been split since 2026-09-07 by uk_split.py, from a church
register, the English Church Census 2005 and this same survey. **None of the first two crosses
the border**: the English Church Census stopped at it and Wales has had no church attendance
census since 1995. So Wales stayed one undifferentiated `Christian` node, which Anita called
out on 2026-10-04: "its the most prominent undifferentiated christianity left in europe."

WHAT THIS FILE DOES DIFFERENTLY FROM ENGLAND'S. England's survey only supplies the national
mix; the placement comes from churches. Wales has no church source, so **the survey supplies
the placement too**, and that is only allowed where the survey's own geography is stable:
the split-half test of spec §9bi/§9bl (playbooks/ess.md), run here on respondent folds.

WHICH RESPONDENTS. Every Welsh respondent in waves 1-31, each counted ONCE, using the answer
from the wave nearest to wave 21 (May 2021, closest to census day). England uses wave 21 alone
because 25,630 English respondents answered it; Wales had 1,557, of whom 581 were Christian,
and that is too thin for twelve areas. Taking each person once across the panel gives 8,630
distinct Welsh adults and 2,970 Christians without counting anybody twice, which is the
objection uk_bes.py raises against pooling waves. `p_religion` is a profile variable with the
same nineteen codes in every wave (checked: identical value labels W1-W31).

`oslaua` (local authority from the respondent's full postcode) is filled for every Welsh row,
so every respondent has one of the 22 principal areas.

THE GEOGRAPHY: ITL3, TWELVE AREAS, NOT THE 22 AUTHORITIES. The 22 principal areas carry as few
as 35 Christian respondents (Merthyr Tydfil); ITL3's twelve groups of whole authorities carry at
least 67 (Anglesey). The legs that pass at authority level also pass at ITL3, with higher
split-half correlations (Anglican +0.81 against +0.59), so the coarser grain loses nothing the
test could see and halves the noise. ITL3 is the ONS's own grouping and nests exactly in the
authorities, so no boundary join is involved.

THE LEGS AND THE OPTION LIST, WHICH WAS WRITTEN FOR ENGLAND AND SCOTLAND.

    2  Church of England/Anglican/Episcopal   -> anglican (the Church in Wales)
    3  Roman Catholic                         -> catholic
    4  Presbyterian/Church of Scotland        -> reformed, with 7 and 8, as in England
    7  United Reformed Church                 -> reformed
    8  Free Presbyterian                      -> reformed
    5  Methodist                              -> methodist
    6  Baptist                                -> baptist
    17 Orthodox, 18 Pentecostal, 19 Evangelical independent, 9 Brethren
                                              -> too few here; stay on `Christian`

**Two Welsh churches have no box of their own, and both distort a leg.** The Presbyterian
Church of Wales was the Calvinistic Methodist church until 1928 and is still called y Methodistiaid
Calvinaidd; a member offered "Presbyterian/Church of Scotland" or "Methodist" may well take the
second. That is the likely reason `methodist` peaks in Anglesey (19%), Conwy (19%) and Gwynedd
(13%), where Wesleyan Methodism was always weak and the Calvinistic Methodists strong. The leg is
drawn as Methodist because that is the word the respondents chose. The Union of Welsh
Independents (Annibynwyr, Congregationalist) has no box at all; its people are inside BES's
generic "Other" (3.8% of Welsh adults against England's 2.4%), which also holds non-Christians
and cannot be separated. Both are stated in sources/uk.md and the note, not corrected.

THE DECISION PER LEG, applied as written (playbooks/ess.md):

    PLACE   at least MIN_N respondents and the split-half passes at ITL3 (permutation p and the
            chi-square both under 0.05): each area gets its own share
    FLAT    at least MIN_N respondents and the test fails: Wales's share everywhere
    PARENT  fewer than MIN_N respondents: not drawn; those people stay on the census's own
            `Christian`, which is what the census said about them

MIN_N is 100. Below that a leg's Wales-wide share has a standard error over a fifth of itself,
and four of the five legs under it fail the test anyway.

THE UNSPECIFIED SHARE, WHICH ENGLAND DOES NOT HAVE. BES wave 21, weighted, finds Christians to
be 33.0% of Welsh adults (1,557 respondents); the census finds 47.1% (1,171,591 of 2,489,882
adults). The level is taken from wave 21 alone, like England's, because the pool is not a sample
of one date and waves 1 and 19 carry no weights (pooled, the figure would be 29.7%). BES has no
"Christian, no denomination" box, so the census's bare Christians who would not pick a church
are not in it. Their share of Welsh Christians, 1 - BES/census, is left on the parent
(`unspecified` here). That follows "unspecified is fine" (draw the named split, leave the rest on
the parent). England's split was built before that ruling and applies its mix to every
Christian; the two countries therefore differ in method at the border, and sources/uk.md says so.

Run: python sources/uk_bes_wales.py            # -> data/normalized/uk_bes_wales.csv, with the test
     python sources/uk_bes_wales.py --report   # print everything, write nothing
"""
import argparse
import csv
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np        # noqa: E402
import pandas as pd       # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import stability as st    # noqa: E402

RAW = os.path.join(ROOT, "data", "raw", "uk")
SAV = os.path.join(RAW, "BES2024_W31_Panel_v31.05.sav")
OUT = os.path.join(ROOT, "data", "normalized", "uk_bes_wales.csv")

SOURCE_ID = "uk_wa_bes_w1_31"
BASIS = "self_id"
YEAR = 2021
WALES = 3
CENTRE_WAVE = 21
N_WAVES = 31

# Census 2021, Wales, usual residents aged 18+ and the Christians among them. Sum over
# religion_tb x resident_age_7a categories 4-7 at country level, from
#   api.beta.ons.gov.uk/v1/population-types/UR/census-observations
#     ?dimensions=religion_tb,resident_age_7a&area-type=ctry
WALES_ADULTS = 2_489_882
WALES_ADULT_CHRISTIANS = 1_171_591

LEG = {2: "anglican", 3: "catholic", 4: "reformed", 7: "reformed", 8: "reformed",
       5: "methodist", 6: "baptist", 9: "brethren", 17: "orthodox", 18: "pentecostal",
       19: "newchurch"}
LEGS = ["anglican", "catholic", "methodist", "baptist", "reformed", "newchurch", "orthodox",
        "pentecostal", "brethren"]
MIN_N = 100
ALPHA = 0.05
FOLDS = 6                 # respondent folds for the split-half; 10 distinct halvings
SEED = 20261004
N_PERM = 999

# The 22 principal areas, by the ONS code BES writes in `oslaua`, and their ITL3 (2021) group.
LA = {
    "W06000001": ("Isle of Anglesey", "Isle of Anglesey"),
    "W06000002": ("Gwynedd", "Gwynedd"),
    "W06000003": ("Conwy", "Conwy and Denbighshire"),
    "W06000004": ("Denbighshire", "Conwy and Denbighshire"),
    "W06000005": ("Flintshire", "Flintshire and Wrexham"),
    "W06000006": ("Wrexham", "Flintshire and Wrexham"),
    "W06000008": ("Ceredigion", "South West Wales"),
    "W06000009": ("Pembrokeshire", "South West Wales"),
    "W06000010": ("Carmarthenshire", "South West Wales"),
    "W06000011": ("Swansea", "Swansea"),
    "W06000012": ("Neath Port Talbot", "Bridgend and Neath Port Talbot"),
    "W06000013": ("Bridgend", "Bridgend and Neath Port Talbot"),
    "W06000014": ("Vale of Glamorgan", "Cardiff and Vale of Glamorgan"),
    "W06000015": ("Cardiff", "Cardiff and Vale of Glamorgan"),
    "W06000016": ("Rhondda Cynon Taf", "Central Valleys"),
    "W06000024": ("Merthyr Tydfil", "Central Valleys"),
    "W06000018": ("Caerphilly", "Gwent Valleys"),
    "W06000019": ("Blaenau Gwent", "Gwent Valleys"),
    "W06000020": ("Torfaen", "Gwent Valleys"),
    "W06000021": ("Monmouthshire", "Monmouthshire and Newport"),
    "W06000022": ("Newport", "Monmouthshire and Newport"),
    "W06000023": ("Powys", "Powys"),
}
COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count",
           "basis", "year", "source_id", "note"]


def respondents():
    """One row per distinct Welsh respondent: answer nearest wave 21, LA, ITL3, weight."""
    import pyreadstat
    if not os.path.exists(SAV):
        sys.exit(f"missing {SAV}\n  see sources/uk_bes.md for the download")
    _, meta = pyreadstat.read_sav(SAV, metadataonly=True)
    have = set(meta.column_names)
    labels = {w: meta.variable_value_labels.get(f"p_religionW{w}") for w in range(1, N_WAVES + 1)}
    ref = labels[CENTRE_WAVE]
    for w, lab in labels.items():
        if lab is not None and lab != ref:
            sys.exit(f"p_religionW{w} has different value labels from W{CENTRE_WAVE}; "
                     "the codes are not comparable across waves any more")
    want = ["id"]
    for w in range(1, N_WAVES + 1):
        want += [c for c in (f"p_religionW{w}", f"countryW{w}", f"oslauaW{w}", f"wt_new_W{w}")
                 if c in have]
    df, _ = pyreadstat.read_sav(SAV, usecols=want)
    parts = []
    for w in range(1, N_WAVES + 1):
        c, r, la, wt = f"countryW{w}", f"p_religionW{w}", f"oslauaW{w}", f"wt_new_W{w}"
        if c not in df or r not in df:
            continue
        sub = df[(df[c] == WALES) & df[r].notna()]
        part = pd.DataFrame({"id": sub["id"], "wave": w, "code": sub[r].astype(int),
                             "la": sub[la] if la in sub else None,
                             "wt": sub[wt] if wt in sub else np.nan})
        # weights rescaled to mean 1 within the wave among the Welsh, so waves without one
        # (wave 1) can sit beside waves with one at weight 1
        m = part["wt"].mean()
        part["wt"] = (part["wt"] / m).fillna(1.0) if np.isfinite(m) and m > 0 else 1.0
        parts.append(part)
    long = pd.concat(parts, ignore_index=True)
    long["dist"] = (long["wave"] - CENTRE_WAVE).abs() + 0.1 * (long["wave"] < CENTRE_WAVE)
    one = long.sort_values(["id", "dist"]).drop_duplicates("id").copy()
    bad = sorted(set(one["la"].dropna()) - set(LA))
    if bad or one["la"].isna().any():
        sys.exit(f"Welsh respondents with no or a non-Welsh authority: {bad}, "
                 f"{int(one['la'].isna().sum())} missing")
    one["la_name"] = one["la"].map(lambda c: LA[c][0])
    one["itl3"] = one["la"].map(lambda c: LA[c][1])
    one["leg"] = one["code"].map(LEG)
    return one, long


def test(chr_, level):
    """(frame per leg: n, rho, null95, p, chi2_p, passed) at `level`, on respondent folds."""
    units = sorted(chr_[level].unique())
    rng = np.random.default_rng(SEED)
    fold = rng.integers(0, FOLDS, len(chr_))
    cube = np.zeros((FOLDS, len(units), len(LEGS)))
    for (f, u, l), n in pd.Series(1, index=pd.MultiIndex.from_arrays(
            [fold, chr_[level], chr_["leg"]])).groupby(level=[0, 1, 2]).sum().items():
        cube[f, units.index(u), LEGS.index(l)] = n
    tot = cube.sum(axis=2)
    splits = st.halvings(FOLDS)
    obs = st.median_rho(cube, splits, tot=tot)
    null = st.wave_null(cube, splits, n_perm=N_PERM, seed=1, tot=tot)
    allc = cube.sum(axis=0)
    rows = []
    for j, leg in enumerate(LEGS):
        p, q = st.permutation_p(obs[j], null[:, j])
        chi = st.chi2_p(allc[:, j], allc.sum(axis=1))
        rows.append({"leg": leg, "n": int(allc[:, j].sum()), "rho": obs[j], "null95": q,
                     "p": p, "chi2_p": chi, "passed": bool(p < ALPHA and chi < ALPHA)})
    return pd.DataFrame(rows).set_index("leg"), int(tot.sum(axis=0).min())


def decide(t):
    """leg -> PLACE / FLAT / PARENT, from the ITL3 test."""
    out = {}
    for leg, r in t.iterrows():
        if r["n"] < MIN_N:
            out[leg] = "PARENT"
        elif r["passed"]:
            out[leg] = "PLACE"
        else:
            out[leg] = "FLAT"
    return out


def build(report=False):
    one, long = respondents()
    chr_ = one[one["leg"].notna()].copy()
    print(f"Welsh respondents, waves 1-{N_WAVES}, each once: {len(one):,}"
          f"  (answer rows read: {len(long):,})")
    print(f"  of them Christian: {len(chr_):,}")
    # The LEVEL comes from wave 21 alone, weighted, as England's does (uk_bes.py): 1,557 Welsh
    # respondents is plenty for one Wales-wide share, and the pool is not a sample of any one
    # date (wave 1 and wave 19 carry no weights, and people who left the panel early answered
    # in 2014-2017). The pooled, mixed-weight figure is printed beside it for comparison.
    w21 = long[long["wave"] == CENTRE_WAVE]
    w21_chr = w21["code"].isin(LEG.keys())
    bes_pct = 100 * w21.loc[w21_chr, "wt"].sum() / w21["wt"].sum()
    pooled_pct = 100 * one.loc[one["leg"].notna(), "wt"].sum() / one["wt"].sum()
    print(f"  BES Christian % of Welsh adults, pooled (mixed weights, not used) {pooled_pct:.1f}%")
    census_pct = 100 * WALES_ADULT_CHRISTIANS / WALES_ADULTS
    classified = min(1.0, bes_pct / census_pct)
    print(f"  BES Christian % of Welsh adults, wave {CENTRE_WAVE} weighted (n={len(w21):,}) "
          f"{bes_pct:.1f}%")
    print(f"  census Christian % of Welsh adults         {census_pct:.1f}%")
    print(f"  -> share of census Christians the survey names: {100 * classified:.1f}%; "
          f"the rest stays on `Christian`")
    other = 100 * one.loc[one["code"] == 15, "wt"].sum() / one["wt"].sum()
    print(f"  BES 'Other' (pools Welsh Independents with non-Christians) {other:.1f}% of adults")

    results = {}
    for level in ("la_name", "itl3"):
        t, minn = test(chr_, level)
        results[level] = t
        print(f"\n  split-half at {level}: {chr_[level].nunique()} units, "
              f"fewest Christians in a unit {minn}")
        for leg, r in t.iterrows():
            print(f"    {leg:12s} n={r['n']:5d} rho={r['rho']:+.3f} null95={r['null95']:+.3f} "
                  f"p={r['p']:.3f} chi2_p={r['chi2_p']:.2g} {'PASS' if r['passed'] else ''}")
    dec = decide(results["itl3"])
    print("\n  decision (ITL3):", ", ".join(f"{k} {v}" for k, v in dec.items()))

    nat = chr_.groupby("leg")["wt"].sum()
    nat = nat / nat.sum()
    by = chr_.groupby(["itl3", "leg"])["wt"].sum().unstack(fill_value=0.0)
    n_by = chr_.groupby("itl3").size()
    by = by.div(by.sum(axis=1), axis=0)
    print("\n  weighted share of each area's Christians (BES), %")
    print((100 * by[LEGS]).round(1).assign(n=n_by).to_string())
    print("\n  Wales:", ", ".join(f"{l} {100 * nat.get(l, 0):.1f}" for l in LEGS))

    rows = []
    for leg in LEGS:
        rows.append({"geo_id": "W92000004", "geo_level": "country", "geo_name": "Wales",
                     "source_category": leg, "count": round(float(nat.get(leg, 0.0)), 6),
                     "basis": BASIS, "year": YEAR, "source_id": SOURCE_ID,
                     "note": (f"share of Welsh adult Christians naming a church; "
                              f"{int(results['itl3'].loc[leg, 'n'])} respondents; "
                              f"decision={dec[leg]}")})
    rows.append({"geo_id": "W92000004", "geo_level": "country", "geo_name": "Wales",
                 "source_category": "classified", "count": round(classified, 6),
                 "basis": BASIS, "year": YEAR, "source_id": SOURCE_ID,
                 "note": (f"BES Christian {bes_pct:.2f}% of Welsh adults / census "
                          f"{census_pct:.2f}%; the share of census Christians a church is "
                          "named for")})
    for unit in by.index:
        for leg in LEGS:
            rows.append({"geo_id": unit, "geo_level": "itl3", "geo_name": unit,
                         "source_category": leg, "count": round(float(by.loc[unit, leg]), 6),
                         "basis": BASIS, "year": YEAR, "source_id": SOURCE_ID,
                         "note": f"share of the area's Christian respondents; n={int(n_by[unit])}"})
    if report:
        return 0
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {len(rows)} rows -> {OUT}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    return build(report=args.report)


if __name__ == "__main__":
    sys.exit(main())
