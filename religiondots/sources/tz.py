"""Tanzania — the regional pattern of Christianity and Islam, from five pooled Afrobarometer rounds.

Reads data/raw/afrobarometer/*.sav, data/geo/tz/tz_lookup.csv and tz_districts.csv, and (as a
witness only) data/raw/gfs/gfs_all_countries_wave2.csv; writes data/normalized/tz.csv.
`sources/tz.md` has the scouting record; `sources/afrobarometer.py` the shared construction;
`sources/ng.py` is the precedent this follows (spec §9cv, `ask/answered/010-ng`).

## TANZANIA HAS NOT ASKED SINCE 1967

The 1967 census was the last to ask religion. Every census since (1978, 1988, 2002, 2012, 2022)
left it off, on the post-independence policy that the state does not count its citizens by
religion, and Tanzania is absent from the UNSD oracle. The Christian and Muslim shares are a
standing argument in Tanzanian politics, which is Nigeria's situation, and Anita's ruling on
Nigeria covers it: draw the regional picture from a pooled survey and say why the state does not
count. Every row is `modelled`.

## WHAT IS DRAWN, AND WHERE EACH NUMBER COMES FROM

    row margin      region populations     NBS 2022 census, Table 1        EXACT
    the composition each unit's own mix    Afrobarometer R4, R6-R9 pooled  measured
    the national level                     neither                         computed

Nothing is fitted to a column margin (`sources/ng.py`'s reason: fitting the columns back to the
survey's own national shares undoes the population reweighting the row margin did).

## ROUND 5 IS NOT USED, AND ROUND 4 IS PLACED BY DISTRICT

Geita, Katavi, Njombe and Simiyu were created in March 2012 out of Mwanza, Kagera, Shinyanga,
Iringa and Rukwa. Round 4 (June 2008) and round 5 (May 2012) both still use the 26 old regions.
**Round 4 carries a DISTRICT column and round 5 carries nothing below the region**, so round 4's
respondents are placed by district, checked against COD-AB's own district list and against the
region label, and round 5's respondents in the five split regions cannot be placed at all.
Rather than measure nine units on fewer rounds than the other twenty-one (Nigeria's round 6 gap,
made on purpose), round 5 is left out: 2,400 respondents. Round 4's 16 respondents in Magu are
dropped too, because Magu district was split in 2012 and its Busega half went to Simiyu.

## ROUND 8 HAS REGION CODES AND NO REGION LABELS

The R8 merged file's `REGION` value-label set (`labels7`) has no entries for Tanzania's codes
740-770, so `ab.load` returns no label for any of its 2,398 respondents. The codes mean the same
region in every other round (asserted), and round 8 is decoded by code and then checked: its
sample shares per unit against rounds 7 and 9, and its weighted shares against the census.

## MBEYA AND SONGWE ARE ONE UNIT

`sources/tz_geo.py` has why. 30 units: 25 mainland regions, Mbeya with Songwe, and Zanzibar's 5.

## THE DENOMINATIONS ARE NOT DRAWN

§11ai's reason. The share answering `Christian only` runs from 3.6% (R5) to 21.3% (R9), and round
7's card is a different card: Methodist takes 7.1% there and 0.0-0.2% in every other round while
Anglican falls from about 4.5% to 0.5%, and `Tanzania Assemblies of God`, `Pentecoste` and
`Evangelical Assemblies of God` appear in that round only. Grouping to the five categories below
makes every one of those harmless.

Usage:
    python sources/tz.py        rebuild data/normalized/tz.csv (the .sav files are shared; fetch
                                them with `python sources/ng.py --fetch` if absent)
"""

import os
import re
import sys
import unicodedata

os.environ.setdefault("OMP_NUM_THREADS", "6")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

import afrobarometer as ab
import cab
import stability

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LOOKUP = os.path.join(ROOT, "data", "geo", "tz", "tz_lookup.csv")
DISTRICTS = os.path.join(ROOT, "data", "geo", "tz", "tz_districts.csv")
GFS_CSV = os.path.join(ROOT, "data", "raw", "gfs", "gfs_all_countries_wave2.csv")
OUT = os.path.join(ROOT, "data", "normalized", "tz.csv")

COUNTRY = "Tanzania"
ROUNDS_READ = [4, 5, 6, 7, 8, 9]
ROUNDS = [4, 6, 7, 8, 9]
RECENT = [8, 9]
SOURCE_ID = "tz_afrobarometer_2008_2022_census2022"
YEARS = "2008-2022"

N_UNITS = 30
CENSUS_2022 = 61_741_120

CATEGORIES = ["Christian", "Muslim", "Traditional/ethnic religion", "None", "Other"]

# Every answer Tanzanians give across the six rounds -> the category it is drawn as, keyed through
# `key()`. Named one by one, because an answer that falls through a default is silently dropped.
#   * `Independent` is the card's African independent church box; Christian.
#   * `Coptic`, `Dutch Reformed`, `Zionist Christian Church` are other countries' boxes on a
#     continental card; 19 Tanzanians between them, all Christian.
#   * `Qadiriya` and `Mouridiya` brotherhoods are Muslim.
#   * `Atheist` and `Agnostic` join the card's own `None` (18 respondents in five rounds).
#   * `Hindu` (2) and `Bahai` (1) go to `Other`: Tanzania's Hindu community is real, and 2 of
#     13,157 respondents cannot place it anywhere.
GROUP = {
    "christian only": "Christian", "roman catholic": "Christian", "lutheran": "Christian",
    "anglican": "Christian", "seventh day adventist": "Christian", "pentecostal": "Christian",
    "pentecoste": "Christian", "methodist": "Christian", "independent": "Christian",
    "evangelical": "Christian", "baptist": "Christian", "moravian": "Christian",
    "morovian": "Christian", "other christian": "Christian", "mennonite": "Christian",
    "orthodox": "Christian", "jehovah's witness": "Christian",
    "tanzania assemblies of god": "Christian", "evangelical assemblies of god": "Christian",
    "church of christ": "Christian", "dutch reformed": "Christian", "coptic": "Christian",
    "quaker/friends": "Christian", "presbyterian": "Christian", "mormon": "Christian",
    "calvinist": "Christian", "zionist christian church": "Christian",
    "muslim only": "Muslim", "sunni only": "Muslim", "shia only": "Muslim", "shia": "Muslim",
    "ismaeli": "Muslim", "qadiriya brotherhood": "Muslim", "mouridiya brotherhood": "Muslim",
    "traditional/ethnic religion": "Traditional/ethnic religion",
    "none": "None", "atheist": "None", "agnostic": "None",
    "other": "Other", "hindu": "Other", "bahai": "Other",
}

# Every `REGION` label the rounds use -> the unit name in tz_lookup.csv, keyed through `ckey()`.
# `Mrwara` (R5, R6, code 748) and `Unfuja Kusini` (R5, R6, code 762) are typing slips; the code
# consistency check in `decode_codes()` is what proves them, not the resemblance.
NORM = {
    "arusha": "Arusha", "coast(pwani)": "Pwani", "pwani": "Pwani",
    "dar es salaam": "Dar es Salaam", "dar-es-salaam": "Dar es Salaam",
    "dares salaam": "Dar es Salaam",
    "dodoma": "Dodoma", "geita": "Geita", "iringa": "Iringa", "kagera": "Kagera",
    "kaskazini pemba": "Kaskazini Pemba", "north pemba": "Kaskazini Pemba",
    "pemba kaskazini": "Kaskazini Pemba",
    "kaskazini unguja": "Kaskazini Unguja", "north unguja": "Kaskazini Unguja",
    "unguja kaskazini": "Kaskazini Unguja",
    "katavi": "Katavi", "kigoma": "Kigoma", "kilimanjaro": "Kilimanjaro",
    "kusini pemba": "Kusini Pemba", "south pemba": "Kusini Pemba", "pemba kusini": "Kusini Pemba",
    "kusini unguja": "Kusini Unguja", "south unguja": "Kusini Unguja",
    "unfuja kusini": "Kusini Unguja",
    "lindi": "Lindi", "manyara": "Manyara", "mara": "Mara",
    "mbeya": "Mbeya and Songwe", "songwe": "Mbeya and Songwe",
    "mjini magharibi": "Mjini Magharibi", "urban west": "Mjini Magharibi",
    "morogoro": "Morogoro", "mrwara": "Mtwara", "mtwara": "Mtwara", "mwanza": "Mwanza",
    "njombe": "Njombe", "rukwa": "Rukwa", "ruvuma": "Ruvuma", "shinyanga": "Shinyanga",
    "simiyu": "Simiyu", "singida": "Singida", "tabora": "Tabora", "tanga": "Tanga",
}

# Round 4's old regions that lost territory in March 2012, and the units their districts can
# now be in. A district decode outside this set is an error, not a placement.
SPLIT_CHILDREN = {
    "Mwanza": {"Mwanza", "Geita", "Simiyu"},
    "Shinyanga": {"Shinyanga", "Geita", "Simiyu"},
    "Kagera": {"Kagera", "Geita"},
    "Iringa": {"Iringa", "Njombe"},
    "Rukwa": {"Rukwa", "Katavi"},
}
# Round 4 district labels that do not start with a COD-AB 2018 district name. Arumeru was divided
# into Arusha and Meru districts, both in Arusha region.
DISTRICT_ALIAS = {"ARUMERU": "Meru"}
# Sampled 2008 districts that were divided ACROSS a new regional line, so the label cannot say
# which side a respondent lived on. Dropped, not guessed.
R4_AMBIGUOUS = {"MAGU": "Busega District was split off Magu in 2012 and put in Simiyu Region"}

# What is drawn on its own unit shares, asserted against the split-half so a change in the data is
# a failure here rather than a quiet redrawing. Set from the test's output, 2026-09-14: Christian
# +0.923, Muslim +0.967, None +0.852, each at p 0.0005 against a null 95th of about +0.24, all
# with chi-square p under 1e-200. `Other` also clears the rank test (+0.523) and is 0.93% of the
# pool, under ab.ELIGIBLE_FLOOR, so it is not placed; `Traditional/ethnic religion` fails.
CARRIES = ["Christian", "Muslim", "None"]
# Failing categories kept in one unit (spec §12, Honduras), asserted likewise.
STANDOUT_AGREE = 0.95
STANDOUTS = {}
# Spec §12's small-category rule sends the tail FLAT: under the residual, Traditional/ethnic
# religion would be drawn at 4.03x its national share in Mbeya and Songwe, where none of the 606
# pooled respondents gave it. Asserted, so a change in the data re-opens the call.
TAIL_FLAT = True
# Spec §12 (Norway): the pooled level against the recent rounds. Measured 2026-09-14 at under
# 0.9 points for every carried category, so no §3.4 rescale; asserted under Sweden's 3.5.
LEVEL_GAP_MAX = 0.035
# The GFS witness: both drawn-on categories must order the shared units the same way.
GFS_RHO_MIN = 0.80
GFS_P_MAX = 0.01

# The Global Flourishing Study, wave 1, Tanzania: `REGION1_Y1` labels from the GFS codebook
# (`gfs_sample_data_variables`, OSF c8hbk), mapped to this map's units. The codebook has no
# Songwe, so 1816 Mbeya is taken as the two together; 1811 South Pemba has no respondents in
# wave 1. `REL2_Y1`: 1 Christianity, 2 Islam, 12 primal/animist/folk, 97 none, 98/99 unanswered.
GFS_COUNTRY = 18
GFS_N = 9_075
GFS_REGION = {
    1801: "Dar es Salaam", 1802: "Dodoma", 1803: "Tabora", 1804: "Singida", 1805: "Shinyanga",
    1806: "Ruvuma", 1807: "Rukwa", 1808: "Pwani", 1809: "Kusini Unguja",
    1810: "Kaskazini Unguja", 1811: "Kusini Pemba", 1812: "Kaskazini Pemba", 1813: "Mwanza",
    1814: "Mtwara", 1815: "Morogoro", 1816: "Mbeya and Songwe", 1817: "Mara", 1818: "Manyara",
    1820: "Kilimanjaro", 1821: "Kigoma", 1822: "Kagera", 1823: "Iringa",
    1824: "Mjini Magharibi", 1825: "Arusha", 1826: "Tanga", 1827: "Simiyu", 1828: "Katavi",
    1829: "Geita", 1830: "Njombe", 1831: "Lindi",
}
GFS_REL = {1: "Christian", 2: "Muslim", 12: "Traditional/ethnic religion", 97: "None"}
GFS_UNANSWERED = {98, 99, -98}


def key(s):
    """An Afrobarometer religion label reduced to what identifies the answer (`sources/ng.py`)."""
    s = unicodedata.normalize("NFKC", str(s)).replace("’", "'").replace("‘", "'")
    s = s.split("(")[0]
    s = re.sub(r"\s*/\s*", "/", s)
    return " ".join(s.split()).strip().casefold()


def ckey(s):
    return " ".join(str(s).split()).strip().casefold()


def letters(s):
    return re.sub(r"[^a-z]", "", unicodedata.normalize("NFKD", str(s)).casefold())


def round_within_rows(m):
    """Largest-remainder rounding inside each row, so every unit total stays exact."""
    out = np.zeros(m.shape, dtype="int64")
    for i in range(m.shape[0]):
        row = m.iloc[i].to_numpy(dtype=float)
        target = int(round(row.sum()))
        base = np.floor(row).astype("int64")
        short = target - int(base.sum())
        if short:
            base[np.argsort(-(row - base))[:short]] += 1
        out[i] = base
    return pd.DataFrame(out, index=m.index, columns=m.columns)


def district_unit(label, cod):
    """The unit a district label is in today, by COD-AB 2018's district names, or None.

    A COD-AB district matches when its name starts with the label's first word, or (for names of
    five letters or more) the label's first word starts with it (`BUTIAMA` against `Butiam`).
    Several districts of one unit may match; districts of two units may not.
    """
    lab = str(label).strip()
    if not lab or lab.lower() == "nan":
        return None
    word = letters(DISTRICT_ALIAS.get(lab.upper(), lab.replace("'", " ").split()[0]))
    if not word:
        return None
    hits = set()
    for d, u in cod:
        first = letters(str(d).split()[0])
        if letters(d).startswith(word) or (len(first) >= 5 and word.startswith(first)):
            hits.add(u)
    return next(iter(hits)) if len(hits) == 1 else None


def decode_codes(raw):
    """Each REGION code -> one unit name, from every round that labels it. Asserted unique."""
    lab = raw.dropna(subset=["geo_raw"]).copy()
    lab["unit_name"] = lab["geo_raw"].map(ckey).map(NORM)
    bad = sorted(lab.loc[lab["unit_name"].isna(), "geo_raw"].astype(str).unique())
    if bad:
        raise SystemExit(f"REGION labels with no unit: {bad}; add them to NORM deliberately")
    per = lab.groupby("geo_code")["unit_name"].unique()
    clash = {int(c): list(v) for c, v in per.items() if len(v) != 1}
    if clash:
        raise SystemExit(f"a REGION code names different units in different rounds: {clash}")
    table = {int(c): v[0] for c, v in per.items()}
    print(f"\n  {lab['geo_raw'].nunique()} REGION labels over the labelled rounds -> "
          f"{len(set(table.values()))} units; every code names one unit in every round")
    return table


def check_r8(df, units, nm):
    """Round 8 is decoded by code; its sample shares must look like rounds 7 and 9's."""
    share = pd.crosstab(df["geo_id"], df["round"], normalize="columns").reindex(units).fillna(0)
    ref = (share[7] + share[9]) / 2
    r = np.corrcoef(share[8], ref)[0, 1]
    worst = (share[8] - ref).abs().sort_values(ascending=False)
    print(f"\n  round 8 (decoded by code) against rounds 7 and 9, unweighted sample share per unit: "
          f"r = {r:+.3f}; largest gap {nm[worst.index[0]]} {100 * worst.iloc[0]:.2f} points")
    if r < 0.95:
        raise SystemExit("round 8's code decode does not reproduce rounds 7 and 9's sample "
                         "design; the codes may mean something else in round 8")


def check_locations(df, cod, nm):
    """Rounds 6, 7 and 9 carry a district (`LOCATION.LEVEL.1`); it must agree with the region."""
    print("\n  region label against the district column (LOCATION.LEVEL.1):")
    for rnd in (6, 7, 9):
        d = df[df["round"] == rnd]
        du = d["LOCATION.LEVEL.1"].map(lambda s: district_unit(s, cod))
        matched = du.notna()
        disagree = matched & (du != d["geo_id"])
        print(f"    R{rnd}: {int(matched.sum()):,} of {len(d):,} respondents' districts match a "
              f"COD-AB name; {int(disagree.sum())} disagree with the region label")
        if disagree.any():
            pairs = (d[disagree].assign(du=du[disagree])
                     .groupby(["geo_id", "LOCATION.LEVEL.1", "du"]).size())
            for (g, loc, u), n in pairs.items():
                print(f"      {nm[g]} / {loc} -> {nm[u]}  ({n})")
        if disagree.sum() > 0.01 * matched.sum():
            raise SystemExit(f"R{rnd}: more than 1% of district-matched respondents sit in a "
                             "different unit from their region label")


def decode_r4(df, cod, name_to_id, nm):
    """Round 4, placed by DISTRICT and checked against its old-region label."""
    r4 = df["round"] == 4
    old = df.loc[r4, "geo_raw"].map(ckey).map(NORM)
    dist = df.loc[r4, "DISTRICT"].astype(str).str.strip()
    amb = dist.str.upper().isin(R4_AMBIGUOUS)
    for lab, why in R4_AMBIGUOUS.items():
        print(f"    R4 {lab}: {int((dist.str.upper() == lab).sum())} respondents dropped; {why}")
    du = dist.map(lambda s: district_unit(s, cod))
    none = r4.copy()
    none.loc[r4] = du.isna() & ~amb
    if none.any():
        raise SystemExit(f"R4 districts with no single COD-AB unit: "
                         f"{sorted(df.loc[none, 'DISTRICT'].astype(str).unique())}")
    ok = []
    for idx in du.index[~amb]:
        o, u = old[idx], nm[du[idx]]
        allowed = SPLIT_CHILDREN.get(o, {o})
        ok.append(u in allowed)
        if u not in allowed:
            raise SystemExit(f"R4 district {df.at[idx, 'DISTRICT']!r} decodes to {u}, which the "
                             f"old region {o} cannot contain")
    moved = sum(1 for idx in du.index[~amb] if nm[du[idx]] != old[idx])
    print(f"    R4: {len(ok):,} respondents placed by district, every one inside its old region's "
          f"2012 successors; {moved} of them in a region created after the round")
    out = df.copy()
    out.loc[du.index[~amb], "geo_id"] = du[~amb]
    return out.drop(index=du.index[amb])


def report_card():
    """Which of the grouped categories' boxes each used round's showcard offered at all."""
    import pyreadstat

    watch = ["christian only", "muslim only", "none", "traditional/ethnic religion", "other",
             "atheist", "agnostic"]
    print("\n  boxes on each round's showcard (value labels, not responses):")
    missing_none = []
    for rnd, name, _url, relname, _wt in ab.ROUNDS:
        if rnd not in ROUNDS:
            continue
        _d, meta = pyreadstat.read_sav(os.path.join(ab.AB_DIR, name), metadataonly=True)
        col = next(c for c in meta.column_names if c.upper() == relname.upper())
        have = {key(v) for v in meta.variable_value_labels.get(col, {}).values()}
        print(f"    R{rnd}: " + ", ".join(f"{w} {'yes' if w in have else 'NO'}" for w in watch))
        if "none" not in have:
            missing_none.append(rnd)
    if missing_none:
        raise SystemExit(f"the None box is not on the card in R{missing_none}")


def standouts(dfw, cats, units, failing, table):
    """Spec §12 (Honduras): does one unit top a failing category in both halves of the halvings?"""
    waves = sorted(dfw["wave"].unique())
    ui = {u: i for i, u in enumerate(units)}
    cube = np.zeros((len(waves), len(units), len(cats)))
    for (w, u, k), n in dfw.groupby(["wave", "geo_id", "code"]).size().items():
        cube[waves.index(w), ui[u], cats.index(k)] += n
    sp = stability.halvings(len(waves))
    sa, sb = zip(*(stability.halves(cube, a, b) for a, b in sp))
    top_unit, top_share = stability.top_both_halves(np.array(sa), np.array(sb))
    chi = {row["category"]: row["chi_p"] for row in table}
    found = {}
    print(f"\n  failing categories: does one unit top both halves (needs {STANDOUT_AGREE:.0%} of "
          f"{len(sp)} halvings, plus chi-square < 0.05)?")
    for c in failing:
        j = cats.index(c)
        if top_unit[j] >= 0:
            best, frac = int(top_unit[j]), float(top_share[j])
        else:
            best, frac = None, 0.0
        ok = frac >= STANDOUT_AGREE and np.isfinite(chi[c]) and chi[c] < 0.05
        if ok:
            found[c] = units[best]
        print(f"    {c:<30} {units[best] if best is not None else '-':<6} {frac:6.1%}  "
              f"chi2 p {chi[c]:.2e}  {'kept in that unit' if ok else 'no standout'}")
    return found


def compose(df, nat, units, carried, stand):
    """Per-unit shares as a closed partition: carried, standouts, then spec §12's residual/2x rule."""
    by = df.groupby(["geo_id", "category"])["w"].sum().unstack(fill_value=0.0)
    by = by.reindex(index=units, columns=CATEGORIES, fill_value=0.0)
    own = by.div(by.sum(axis=1), axis=0)
    nraw = df.groupby(["geo_id", "category"]).size().unstack(fill_value=0)
    nraw = nraw.reindex(index=units, columns=CATEGORIES, fill_value=0)

    fixed = pd.DataFrame(0.0, index=units, columns=CATEGORIES)
    for c in carried:
        fixed[c] = own[c]
    for c, u in stand.items():
        rest = [x for x in units if x != u]
        fixed[c] = float(by.loc[rest, c].sum() / by.loc[rest].sum().sum())
        fixed.loc[u, c] = own.loc[u, c]
    tail = [c for c in CATEGORIES if c not in carried and c not in stand]
    tail_nat = float(sum(nat[c] for c in tail))

    # the residual, as it would ship
    frame = fixed.copy()
    remainder = 1.0 - fixed[carried + list(stand)].sum(axis=1)
    for c in tail:
        frame[c] = remainder * nat[c] / tail_nat
    mult = remainder / tail_nat
    print("\n  spec §12 small-category rule: the residual's multiple of national share, in units "
          "where the survey found none of that category:")
    rows, worst = stability.residual_multiples(mult, nraw == 0, tail)
    for c, u, m in rows:
        print(f"    {c:<30} worst {m:.2f}x in {u} (found none there; national "
              f"{100 * nat[c]:.2f}%)")
    flat = worst is not None and worst[2] >= stability.SMALL_CATEGORY_MULTIPLE
    if flat:
        print(f"    {worst[0]} at {worst[2]:.2f}x in {worst[1]}: 2x or more, so the tail goes FLAT")
        scale = (1.0 - tail_nat) / fixed[carried + list(stand)].sum(axis=1)
        frame = fixed.mul(scale, axis=0)
        for c in tail:
            frame[c] = nat[c]
    else:
        print("    under 2x everywhere, so the tail stays the residual")
    if (frame.sum(axis=1) - 1.0).abs().max() > 1e-9:
        raise SystemExit("a unit's shares do not sum to 1")
    return frame, own, nraw, flat


def gfs_witness(own_ab, units, pop, nm):
    """The Global Flourishing Study, 2023, as an independent second sample. Prints; asserts the
    decode and the direction of agreement, never used to draw."""
    if not os.path.exists(GFS_CSV):
        print("\n  (GFS witness skipped: data/raw/gfs/gfs_all_countries_wave2.csv is absent; "
              "`python sources/jp_gfs.py --fetch`)")
        return None
    g = pd.read_csv(GFS_CSV, usecols=["COUNTRY", "REGION1_Y1", "REL2_Y1", "ANNUAL_WEIGHT_C1"],
                    low_memory=False)
    g = g[pd.to_numeric(g["COUNTRY"], errors="coerce") == GFS_COUNTRY].copy()
    if len(g) != GFS_N:
        raise SystemExit(f"GFS Tanzania has {len(g):,} rows, this was written against {GFS_N:,}")
    g["reg"] = pd.to_numeric(g["REGION1_Y1"], errors="coerce").astype("Int64")
    g["rel"] = pd.to_numeric(g["REL2_Y1"], errors="coerce")
    g["w"] = pd.to_numeric(g["ANNUAL_WEIGHT_C1"], errors="coerce")
    stray = sorted(set(g["reg"].dropna().astype(int)) - set(GFS_REGION))
    if stray:
        raise SystemExit(f"GFS REGION1 codes with no label here: {stray}")
    name_to_id = {v: k for k, v in nm.items()}
    g["geo_id"] = g["reg"].astype(int).map(GFS_REGION).map(name_to_id)
    g = g[~g["rel"].isin(GFS_UNANSWERED)].copy()
    g["category"] = g["rel"].map(GFS_REL).fillna("Other")
    shared = sorted(set(g["geo_id"]))
    print(f"\n  GFS wave 1 (2023), Tanzania: {len(g):,} answered, {len(shared)} of {N_UNITS} units")
    ab.held_out(g, pop.reindex(shared), "Tanzania (GFS)", pop_source="NBS 2022 census")

    by = g.groupby(["geo_id", "category"])["w"].sum().unstack(fill_value=0.0)
    gs = by.div(by.sum(axis=1), axis=0)
    gnat = g.groupby("category")["w"].sum() / g["w"].sum()
    print("    GFS's own weighted national shares: "
          + ", ".join(f"{c} {100 * gnat.get(c, 0):.1f}%" for c in CATEGORIES))
    rng = np.random.default_rng(0)
    out = {}
    for c in ("Christian", "Muslim"):
        a = own_ab.loc[shared, c].to_numpy()
        b = gs.reindex(shared)[c].fillna(0).to_numpy()
        ra, rb = pd.Series(a).rank(), pd.Series(b).rank()
        rho = float(np.corrcoef(ra, rb)[0, 1])
        null = np.array([np.corrcoef(ra, rb.sample(frac=1, random_state=int(s)).to_numpy())[0, 1]
                         for s in rng.integers(0, 2**31, 5000)])
        p = (1 + int((null >= rho).sum())) / (1 + len(null))
        gap = pd.Series(100 * (a - b), index=shared)
        print(f"    {c}: Spearman {rho:+.3f} across {len(shared)} units against the Afrobarometer "
              f"pool (permutation p {p:.4f}); median gap {gap.median():+.1f} points, largest "
              f"{nm[gap.abs().idxmax()]} {gap[gap.abs().idxmax()]:+.1f}")
        out[c] = (rho, p)
    wp = pop.reindex(shared)
    recomposed = {c: float((gs.reindex(shared)[c].fillna(0) * wp).sum() / wp.sum())
                  for c in ("Christian", "Muslim")}
    print(f"    GFS recomposed on the census over its {len(shared)} units: Christian "
          f"{100 * recomposed['Christian']:.1f}%, Muslim {100 * recomposed['Muslim']:.1f}%")
    return out, gnat, recomposed


def main():
    print("=== Afrobarometer, Tanzania ===")
    raw = ab.load(COUNTRY, expect_rounds=ROUNDS_READ, regroup=True,
                  extra=["DISTRICT", "LOCATION.LEVEL.1"], unlabelled_rounds=[8])
    print(f"\n  pooled: {len(raw):,} respondents with a religion answer over six rounds")

    ct = pd.crosstab(raw["category"], raw["round"])
    ct["all"] = ct.sum(axis=1)
    print("\n  every answer as it arrives (this is what GROUP collapses):")
    print(ct.sort_values("all", ascending=False).to_string(max_colwidth=44))

    conly = raw[raw["category"].map(key) == "christian only"]
    by_round = conly.groupby("round")["w"].sum() / raw.groupby("round")["w"].sum()
    print("\n  share answering `Christian only` rather than naming a denomination, by round:")
    for r, v in by_round.items():
        print(f"    R{r}  {v:6.1%}")
    used = by_round.reindex(ROUNDS)
    if used.max() - used.min() < 0.10:
        raise SystemExit("the `Christian only` share across the drawn rounds now moves less than "
                         "10 points, so the argument for collapsing the denominations is weaker; "
                         "re-read the docstring and decide deliberately")
    report_card()

    # ---- group ----
    k = raw["category"].map(key)
    unmapped = sorted(set(k) - set(GROUP))
    if unmapped:
        raise SystemExit(f"answers with no category: {unmapped}; add them to GROUP deliberately")
    df = raw.copy()
    df["raw_category"] = raw["category"]
    df["category"] = k.map(GROUP)
    ab.assert_one_wording(df, COUNTRY)

    # ---- units ----
    lut = pd.read_csv(LOOKUP, dtype={"geo_id": str})
    if len(lut) != N_UNITS:
        raise SystemExit(f"{LOOKUP} has {len(lut)} units, expected {N_UNITS}; re-run tz_geo.py")
    nm = dict(zip(lut["geo_id"], lut["name"]))
    name_to_id = dict(zip(lut["name"], lut["geo_id"]))
    pop = lut.set_index("geo_id")["pop"].astype(float)
    if int(pop.sum()) != CENSUS_2022:
        raise SystemExit(f"tz_lookup.csv sums to {int(pop.sum()):,}, not {CENSUS_2022:,}")
    units = sorted(lut["geo_id"])
    dl = pd.read_csv(DISTRICTS, dtype=str)
    cod = list(zip(dl["district"], dl["unit"]))

    code_unit = decode_codes(df)
    df["geo_id"] = df["geo_code"].map(lambda c: name_to_id.get(code_unit.get(int(c))))
    if df["geo_id"].isna().any():
        raise SystemExit(f"REGION codes with no unit: {sorted(df.loc[df['geo_id'].isna(), 'geo_code'].unique())}")

    n5 = int((df["round"] == 5).sum())
    df = df[df["round"].isin(ROUNDS)].copy()
    print(f"\n  round 5 left out ({n5:,} respondents): its 26 old regions cannot place anyone in "
          "the five regions split in 2012 and it has no district column; see the docstring")
    df = decode_r4(df, cod, name_to_id, nm)
    check_r8(df, units, nm)
    check_locations(df, cod, nm)

    per_round = df.groupby("round")["geo_id"].nunique()
    print("    units present per round: " + ", ".join(f"R{r} {n}" for r, n in per_round.items()))
    if (per_round != N_UNITS).any():
        raise SystemExit("a drawn round does not sample all 30 units")

    # ---- the decode, per round, without touching the religion column ----
    for rnd in ROUNDS:
        print(f"\n  R{rnd}:", end="")
        ab.held_out(df[df["round"] == rnd], pop, f"{COUNTRY} R{rnd}", pop_source="NBS 2022 census")
    ab.held_out(df, pop, COUNTRY, pop_source="NBS 2022 census")

    nat = ab.national(df).reindex(CATEGORIES).fillna(0.0)
    print(f"\n  the survey's national shares, pooled over R{', R'.join(map(str, ROUNDS))} "
          f"(n={len(df):,}):")
    for c in CATEGORIES:
        print(f"    {c:<30}{nat[c]:8.3%}")
    rn = df.groupby(["round", "category"])["w"].sum().unstack(fill_value=0)
    print("\n  by round (survey weighting, %):")
    print((100 * rn.div(rn.sum(axis=1), axis=0)).round(1).reindex(columns=CATEGORIES).to_string())

    # ---- quota, then the split-half ----
    dfw = df.rename(columns={"round": "wave", "category": "code"})[["wave", "geo_id", "code", "w"]]
    cab.assert_not_quota(dfw, COUNTRY, ROUNDS, unit_col="geo_id", cat_col="code")
    carries, table = cab.stability(dfw, CATEGORIES, units, f"{N_UNITS} units")
    carries = [c for c in carries if nat[c] >= ab.ELIGIBLE_FLOOR]
    if sorted(carries) != sorted(CARRIES):
        raise SystemExit(f"the split-half now selects {sorted(carries)}, not {sorted(CARRIES)}. "
                         "Read the table above, then edit CARRIES and the docstring deliberately.")
    failing = [c for c in CATEGORIES if c not in carries and nat[c] >= ab.ELIGIBLE_FLOOR]
    stand = standouts(dfw, CATEGORIES, units, failing, table)
    if stand != STANDOUTS:
        raise SystemExit(f"standouts are now {stand}, against STANDOUTS={STANDOUTS}")

    # ---- compose ----
    frame, own, nraw, flat = compose(df, nat, units, carries, stand)
    if flat != TAIL_FLAT:
        raise SystemExit(f"the small-category rule now gives flat={flat}, against TAIL_FLAT="
                         f"{TAIL_FLAT}; read the multiples above and decide deliberately")
    counts = round_within_rows(frame.mul(pop.reindex(units), axis=0))
    if not (counts.sum(axis=1) == pop.reindex(units).round().astype("int64")).all():
        raise SystemExit("a unit's drawn total is not its census population")
    drawn = counts.sum(axis=0) / counts.sum().sum()

    # ---- level: pooled against the recent rounds, both recomposed on the census ----
    rec = df[df["round"].isin(RECENT)]
    rb = rec.groupby(["geo_id", "category"])["w"].sum().unstack(fill_value=0.0)
    rb = rb.reindex(index=units, columns=CATEGORIES, fill_value=0.0)
    rshare = rb.div(rb.sum(axis=1), axis=0)
    recent = rshare.mul(pop.reindex(units), axis=0).sum() / pop.sum()
    print("\n  national level: survey pool, as drawn, and rounds 8-9 alone recomposed the same way:")
    print(f"    {'category':<30}{'survey':>9}{'drawn':>9}{'R8-R9':>9}{'drawn-R8R9':>12}")
    for c in CATEGORIES:
        print(f"    {c:<30}{100 * nat[c]:8.2f}%{100 * drawn[c]:8.2f}%{100 * recent[c]:8.2f}%"
              f"{100 * (drawn[c] - recent[c]):+11.2f}")
    stale = [c for c in carries if abs(drawn[c] - recent[c]) > LEVEL_GAP_MAX]
    if stale:
        raise SystemExit(f"the pooled level differs from rounds 8-9 by more than "
                         f"{100 * LEVEL_GAP_MAX:.1f} points for {stale}; spec §12 (Norway) says "
                         "consider §3.4 before drawing")
    gw = gfs_witness(own, units, pop, nm)
    if gw is not None:
        weak = {c: v for c, v in gw[0].items() if v[0] < GFS_RHO_MIN or v[1] > GFS_P_MAX}
        if weak:
            raise SystemExit(f"the GFS no longer orders the units like the Afrobarometer: {weak}")

    # A smoke test on the decode, not corroboration of the fine pattern: Zanzibar is Muslim to
    # a degree no mainland region is, and a join that had crossed the channel would fail this.
    zan = [name_to_id[n] for n in ("Kaskazini Unguja", "Kusini Unguja", "Mjini Magharibi",
                                   "Kaskazini Pemba", "Kusini Pemba")]
    share = counts.div(counts.sum(axis=1), axis=0)
    mainland_max = share.drop(index=zan)["Muslim"].max()
    print(f"\n  Zanzibar's five units, Muslim share as drawn: "
          + ", ".join(f"{nm[z]} {100 * share.loc[z, 'Muslim']:.1f}%" for z in zan)
          + f"; the most Muslim mainland unit is {nm[share.drop(index=zan)['Muslim'].idxmax()]} "
          f"at {100 * mainland_max:.1f}%")
    if share.loc[zan, "Muslim"].min() <= mainland_max:
        raise SystemExit("a mainland unit is drawn more Muslim than a Zanzibar one; check the "
                         "Zanzibar decode before drawing")

    # ---- write ----
    n_by = df.groupby("geo_id").size()
    basis_note = {c: ("the unit's own measured share" if c in carries else
                      "the national share" if flat else
                      "the national proportion within the unit's remainder") for c in CATEGORIES}
    out = counts.stack().rename("count").reset_index()
    out.columns = ["geo_id", "source_category", "count"]
    out["geo_level"] = "region"
    out["geo_name"] = out["geo_id"].map(nm)
    out["basis"] = "self_id"
    out["year"] = YEARS
    out["source_id"] = SOURCE_ID
    out["note"] = out.apply(
        lambda r: ("NBS 2022 census population composed with the unit's own mix from "
                   f"Afrobarometer rounds 4 and 6-9 pooled (n={int(n_by[r.geo_id])} here); "
                   f"{basis_note[r.source_category]}"), axis=1)
    total = int(out["count"].sum())
    if total != CENSUS_2022:
        raise SystemExit(f"drawn {total:,} against the census {CENSUS_2022:,}")
    cols = ["geo_id", "geo_level", "geo_name", "source_category", "count",
            "basis", "year", "source_id", "note"]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out[cols].to_csv(OUT, index=False, encoding="utf-8")
    print(f"\nwrote {OUT} ({len(out)} rows, {total:,} people, "
          f"{out['source_category'].nunique()} categories, {out['geo_id'].nunique()} units)")

    print("\n  national, as drawn:")
    for c, n in counts.sum(axis=0).sort_values(ascending=False).items():
        print(f"    {100 * n / total:6.2f}%  {c}  ({n:,})")
    print("\n  as drawn, by unit, most Muslim first (pooled n in brackets):")
    for u in share.sort_values("Muslim", ascending=False).index:
        s = share.loc[u]
        print(f"    {nm[u]:<20}" + "".join(f"{100 * s[c]:7.1f}%" for c in CATEGORIES)
              + f"{int(pop[u]):>12,}  n={int(n_by[u])}")
    print(f"    columns: {', '.join(CATEGORIES)}")
    zero = [(nm[u], c) for u in units for c in carries if counts.loc[u, c] == 0]
    print(f"  drawn at zero in a carried category (no pooled respondent gave it): {zero or 'none'}")
    print(f"  thinnest unit {nm[n_by.idxmin()]} n={int(n_by.min())}; median n={int(n_by.median())}; "
          f"total n={int(n_by.sum()):,}")


if __name__ == "__main__":
    main()
