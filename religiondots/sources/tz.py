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

## ZANZIBAR'S CHRISTIANS ARE ONE SHARE ACROSS ITS FIVE REGIONS

Anita's ruling, 2026-09-14 night (`ask/RULINGS.md`, `sources/tz.md` §8): Zanzibar's Christians
were attacked in 2012-13, and region by region the figures rest on single respondents (Kusini
Pemba's one Christian of 144) or on none (three regions). So the five regions' respondents are
pooled, survey-weighted, for the Christian share only, and every Zanzibar region is drawn at it.
Muslim and None keep each region's own measurement, scaled to fill what is left; the flat tail
is untouched, and every region keeps its census total. `pool_zanzibar()`.

## THE AFROBAROMETER'S CHURCH ANSWERS ARE GROUPED

§11ai's reason. The share answering `Christian only` runs from 3.6% (R5) to 21.3% (R9), and round
7's card is a different card: Methodist takes 7.1% there and 0.0-0.2% in every other round while
Anglican falls from about 4.5% to 0.5%, and `Tanzania Assemblies of God`, `Pentecoste` and
`Evangelical Assemblies of God` appear in that round only. Grouping to the five categories below
makes every one of those harmless, and the Afrobarometer's Christians are one category here.

## THE CHURCHES: THE GFS SETS THE LEVELS, BOTH SURVEYS SAY WHERE (2026-10-03, `sources/tz.md` §9)

Each mainland unit's Christians (the Afrobarometer's count, unchanged) are split into Catholic,
Lutheran, Anglican, Pentecostal, Adventist and other Christian. The Global Flourishing Study 2023
asked every Christian which church they most identify with, and 0.5% named none; the
Afrobarometer's `Christian only` is 10-35% of Christians by round. So:

  * the GFS's church shares per unit, its few unnamed spread at its national proportions;
  * the Afrobarometer's (rounds 4, 6, 8, 9; round 7's card is a different card) with its
    `Christian only` spread at one set of national proportions inside every unit, chosen so each
    church lands on the GFS's level (Pentecostal takes 61.5% of the unnamed, Catholic none);
  * the two averaged per unit by respondents.

Two tests per church: the surveys must rank the 25 mainland units alike (`CHURCH_P_MAX`), and the
GFS's level must sit between the Afrobarometer's named share (a floor) and that share plus every
`Christian only` answer. All five pass. Spreading the unnamed at the Afrobarometer's own named
proportions would put Pentecostals at about 10% of Christians against the GFS's 21.6%: the
unnamed are mostly not Catholic. Zanzibar's pooled Christians stay one category; the GFS has two
Christians there. `church_shares()`.

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
# Anita's ruling, 2026-09-14 night: one Christian share across Zanzibar's five regions, pooled from
# their respondents with the survey's weights (13 Christians of 1,072 interviews, 0.945%; the
# weights put 50.6% of the pool in Mjini Magharibi against its 47.3% of the census). Asserted
# against `tz_lookup.csv`'s `zanzibar` column so a renamed region cannot drop out silently.
ZANZIBAR = ("Kaskazini Unguja", "Kusini Unguja", "Mjini Magharibi", "Kaskazini Pemba",
            "Kusini Pemba")
ZANZIBAR_POOLED = ["Christian"]
# The GFS witness: both drawn-on categories must order the shared units the same way.
GFS_RHO_MIN = 0.80
GFS_P_MAX = 0.01

# THE CHURCHES (2026-10-03, `fafd1067-chea`; docstring "THE CHURCHES COME FROM THE GLOBAL
# FLOURISHING STUDY"). `REL3_Y1`, asked of everyone who gave Christianity in `REL2_Y1` (GFS codebook
# wave 2, OSF 285w7, p.21): "Which of the following denominations or churches do you most identify
# with, if any?". Codes not named here (2 Orthodox, 4 Presbyterian/Reformed, 6 Methodist, 7 Baptist,
# 9 Independent/Holiness/Evangelical, 10 Latter-day Saints, 11 Jehovah's Witness, 13 African
# Initiated, 96 other) are `Other Christian`. 97 (no denomination in particular), 98 and 99 are the
# unnamed pool, spread at the national proportions inside each unit.
GFS_CHURCH = {1: "Roman Catholic", 3: "Anglican", 5: "Lutheran", 8: "Pentecostal",
              12: "Seventh Day Adventist"}
GFS_CHURCH_UNNAMED = {97, 98, 99, -98}
CHURCHES = ["Roman Catholic", "Lutheran", "Anglican", "Pentecostal", "Seventh Day Adventist"]
OTHER_CHRISTIAN = "Other Christian"
# The Afrobarometer's boxes for the same churches, keyed through `key()`. Round 7 is left out of
# the church witness: its card is a different card (Methodist 11.0% of Christians there, 0.0-0.3%
# in every other round; Anglican 0.7% against 7.4%; Pentecostal 0.1% against 4.9-11.3%).
AB_CHURCH = {"roman catholic": "Roman Catholic", "lutheran": "Lutheran", "anglican": "Anglican",
             "pentecostal": "Pentecostal", "seventh day adventist": "Seventh Day Adventist"}
AB_UNNAMED = "christian only"
AB_CHURCH_ROUNDS = [4, 6, 8, 9]
# Each church must order the 25 mainland units alike in the two surveys (Spearman, permutation
# p under 0.05). Measured 2026-10-03: Catholic +0.69, Lutheran +0.77, Anglican +0.57, Pentecostal
# +0.66, Adventist +0.40 (null 95th about +0.34).
CHURCH_P_MAX = 0.05
# And the GFS's national level must lie inside the Afrobarometer's bounds: no lower than the
# church's named share of Christians (a floor) and no higher than that plus everyone answering
# `Christian only`.
GFS_CHURCH_N = 5_937
# Weighted by the drawn Christians, the GFS's Catholic level sits 0.13 points of Christians under
# the Afrobarometer's named Catholic share (2026-10-03): the unnamed hold essentially no Catholics.
# A church that far under its floor takes none of the unnamed; any further and the build stops.
SPREAD_NEG_TOL = 0.02

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


def pool_zanzibar(frame, df, zan, carried, nm):
    """Anita's ruling: one Christian share across Zanzibar's five regions (docstring).

    The pooled share is the survey-weighted share among all of Zanzibar's respondents, taken
    through the same flat-tail scaling `compose()` gave every unit (the carried categories fill
    1 minus the national tail). In each region the other carried categories (Muslim, None) keep
    their own proportions and are scaled to fill what is left. The tail is not touched.
    """
    if not TAIL_FLAT:
        raise SystemExit("pool_zanzibar() assumes the flat tail; re-read it before pooling")
    z = df[df["geo_id"].isin(zan)]
    w = z.groupby("category")["w"].sum().reindex(CATEGORIES).fillna(0.0)
    own_pool = w / w.sum()
    rest = [c for c in carried if c not in ZANZIBAR_POOLED]
    out = frame.copy()
    print(f"\n  Zanzibar pooled (Anita's ruling): {len(z):,} respondents, "
          + ", ".join(f"{int((z['category'] == c).sum())} {c}" for c in ZANZIBAR_POOLED))
    for u in zan:
        budget = float(frame.loc[u, carried].sum())
        pooled = {c: float(own_pool[c] / own_pool[carried].sum() * budget) for c in ZANZIBAR_POOLED}
        left = budget - sum(pooled.values())
        have = float(frame.loc[u, rest].sum())
        if have <= 0:
            raise SystemExit(f"{nm[u]}: nothing left to scale after pooling {ZANZIBAR_POOLED}")
        for c in rest:
            out.loc[u, c] = frame.loc[u, c] * left / have
        for c, v in pooled.items():
            out.loc[u, c] = v
        print(f"    {nm[u]:<18} n={int((z['geo_id'] == u).sum()):4d}  "
              + ", ".join(f"{c} {100 * frame.loc[u, c]:.3f}% -> {100 * out.loc[u, c]:.3f}%"
                          for c in ZANZIBAR_POOLED + rest))
    if (out.sum(axis=1) - 1.0).abs().max() > 1e-9:
        raise SystemExit("a unit's shares do not sum to 1 after pooling Zanzibar")
    if not out.drop(index=zan).equals(frame.drop(index=zan)):
        raise SystemExit("pooling Zanzibar moved a mainland unit")
    for c in ZANZIBAR_POOLED:
        if out.loc[zan, c].max() - out.loc[zan, c].min() > 1e-12:
            raise SystemExit(f"Zanzibar's {c} share is not one share across the five regions")
    return out


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


def church_shares(df, mainland, nm, christians):
    """Each mainland unit's Christians split into churches (docstring): the GFS 2023 sets each
    church's level, both surveys say where. Returns a units x (CHURCHES + Other Christian) frame
    of shares among Christians, rows summing to 1, and the pieces for the record.

    `christians` is each mainland unit's drawn Christian count, the weight for every national
    figure here, so the levels compared are the levels drawn.
    """
    from scipy.stats import rankdata

    if not os.path.exists(GFS_CSV):
        raise SystemExit("the churches need data/raw/gfs/gfs_all_countries_wave2.csv; "
                         "`python sources/jp_gfs.py --fetch`")
    g = pd.read_csv(GFS_CSV, usecols=["COUNTRY", "REGION1_Y1", "REL2_Y1", "REL3_Y1",
                                      "ANNUAL_WEIGHT_C1"], low_memory=False)
    g = g[pd.to_numeric(g["COUNTRY"], errors="coerce") == GFS_COUNTRY]
    g = g[pd.to_numeric(g["REL2_Y1"], errors="coerce") == 1].copy()
    if len(g) != GFS_CHURCH_N:
        raise SystemExit(f"GFS Tanzania has {len(g):,} Christians, this was written against "
                         f"{GFS_CHURCH_N:,}")
    g["d"] = pd.to_numeric(g["REL3_Y1"], errors="coerce")
    if g["d"].isna().any():
        raise SystemExit("a GFS Christian has no REL3 answer")
    g["w"] = pd.to_numeric(g["ANNUAL_WEIGHT_C1"], errors="coerce")
    name_to_id = {v: k for k, v in nm.items()}
    g["geo_id"] = g["REGION1_Y1"].astype(int).map(GFS_REGION).map(name_to_id)
    unnamed = g["d"].isin(GFS_CHURCH_UNNAMED)
    g["c"] = g["d"].map(GFS_CHURCH).fillna(OTHER_CHRISTIAN)
    cats = CHURCHES + [OTHER_CHRISTIAN]
    print(f"\n  churches: GFS wave 1 (2023), {len(g):,} Christians; "
          f"{int(unnamed.sum())} name no church ({100 * g.loc[unnamed, 'w'].sum() / g['w'].sum():.2f}% "
          "weighted), spread at the national proportions inside each unit")
    named = g[~unnamed]
    gnat = named.groupby("c")["w"].sum().reindex(cats, fill_value=0.0)
    gnat = gnat / gnat.sum()
    gu = g[g["geo_id"].isin(mainland)]
    missing = sorted(set(mainland) - set(gu["geo_id"]))
    if missing:
        raise SystemExit(f"mainland units with no GFS Christian: {[nm[u] for u in missing]}")
    by = gu[~gu["d"].isin(GFS_CHURCH_UNNAMED)].groupby(["geo_id", "c"])["w"].sum().unstack(
        fill_value=0.0).reindex(index=mainland, columns=cats, fill_value=0.0)
    un = gu[gu["d"].isin(GFS_CHURCH_UNNAMED)].groupby("geo_id")["w"].sum().reindex(
        mainland, fill_value=0.0)
    by = by + un.to_numpy()[:, None] * gnat.to_numpy()[None, :]
    share = by.div(by.sum(axis=1), axis=0)
    n_g = gu.groupby("geo_id").size().reindex(mainland)

    # The Afrobarometer's answers for the same churches: named shares per unit (the rank test)
    # and, nationally, each church's floor and ceiling among Christians.
    a = df[(df["category"] == "Christian") & df["round"].isin(AB_CHURCH_ROUNDS)].copy()
    a["c"] = a["raw_category"].map(key).map(lambda k: AB_CHURCH.get(k, "UNNAMED" if k == AB_UNNAMED
                                                                    else OTHER_CHRISTIAN))
    an = a[a["c"] != "UNNAMED"]
    ab_u = an[an["geo_id"].isin(mainland)].groupby(["geo_id", "c"])["w"].sum().unstack(
        fill_value=0.0).reindex(index=mainland, columns=cats, fill_value=0.0)
    ab_u = ab_u.div(ab_u.sum(axis=1), axis=0)
    floor = a.groupby("c")["w"].sum() / a["w"].sum()
    unnamed_ab = float(floor.get("UNNAMED", 0.0))

    # The Afrobarometer's unit shares with its unnamed pool spread at one set of national
    # proportions inside every unit, the proportions chosen so that, weighted by the drawn
    # Christians, each church lands on the GFS's own level. A negative proportion would mean the
    # GFS level is under the Afrobarometer's floor; asserted not.
    am = a[a["geo_id"].isin(mainland)]
    full = am.groupby(["geo_id", "c"])["w"].sum().unstack(fill_value=0.0).reindex(
        index=mainland, columns=cats + ["UNNAMED"], fill_value=0.0)
    full = full.div(full.sum(axis=1), axis=0)
    cw = christians.reindex(mainland).astype(float)
    # `Other Christian` is not a church and is not spread into: the Afrobarometer names more of
    # it (Independent, Evangelical, Mennonite, Baptist, 6.1% of Christians) than the GFS (5.0%),
    # so it keeps its own named share, and the five churches take the unnamed in the GFS's
    # proportions among them.
    target = share.mul(cw, axis=0).sum()
    room = float(cw.sum() - (full[OTHER_CHRISTIAN] * cw).sum())
    target = target[CHURCHES] * room / float(target[CHURCHES].sum())
    spread_p = ((target - full[CHURCHES].mul(cw, axis=0).sum())
                / float((full["UNNAMED"] * cw).sum())).reindex(cats, fill_value=0.0)
    print("    the Afrobarometer's `Christian only` answers, spread so each church meets the GFS level: "
          + ", ".join(f"{c} {100 * spread_p[c]:.1f}%" for c in cats))
    if (spread_p < -SPREAD_NEG_TOL).any() or abs(float(spread_p.sum()) - 1.0) > 1e-9:
        raise SystemExit(f"the GFS's church levels cannot be reached by spreading the unnamed: "
                         f"{spread_p.round(4).to_dict()}")
    if (spread_p < 0).any():
        print(f"    {', '.join(spread_p[spread_p < 0].index)} is under the Afrobarometer's floor by "
              "a hair; it takes none of the unnamed and sits at that floor")
        spread_p = spread_p.clip(lower=0.0)
        spread_p = spread_p / spread_p.sum()
    ab_spread = full[cats] + full["UNNAMED"].to_numpy()[:, None] * spread_p.to_numpy()[None, :]
    n_ab = am.groupby("geo_id").size().reindex(mainland).astype(float)
    pooled = (share.mul(n_g, axis=0) + ab_spread.mul(n_ab, axis=0)).div(n_g + n_ab, axis=0)

    def rho(x, y):
        return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])

    rng = np.random.default_rng(0)
    print(f"    {'church':<24}{'GFS':>7}{'AB floor':>10}{'AB ceil':>9}{'rho':>8}{'null95':>8}"
          f"{'p':>8}   (share of Christians; rank over {len(mainland)} mainland units, "
          f"Afrobarometer R{', R'.join(map(str, AB_CHURCH_ROUNDS))})")
    bad = []
    for c in CHURCHES:
        r = rho(share[c], ab_u[c])
        null = np.array([rho(share[c], rng.permutation(ab_u[c].to_numpy())) for _ in range(5000)])
        p = (1 + int((null >= r).sum())) / (1 + len(null))
        lo, hi = float(floor.get(c, 0.0)), float(floor.get(c, 0.0)) + unnamed_ab
        print(f"    {c:<24}{100 * gnat[c]:6.1f}%{100 * lo:9.1f}%{100 * hi:8.1f}%{r:+8.3f}"
              f"{np.quantile(null, 0.95):+8.3f}{p:8.4f}")
        if p > CHURCH_P_MAX:
            bad.append(f"{c} ranks the units differently in the two surveys (p {p:.3f})")
        if not lo <= gnat[c] <= hi:
            bad.append(f"{c}'s GFS level {100 * gnat[c]:.1f}% is outside the Afrobarometer's "
                       f"{100 * lo:.1f}-{100 * hi:.1f}%")
    if bad:
        raise SystemExit("the churches no longer pass: " + "; ".join(bad))
    print(f"    Afrobarometer `Christian only`: {100 * unnamed_ab:.1f}% of Christians in those rounds")
    print("    per unit, % of Christians: drawn (both surveys, by respondents) / GFS / Afrobarometer "
          "spread; n GFS + Afrobarometer:")
    for u in mainland:
        print(f"      {nm[u]:<18}" + "".join(
            f"{100 * pooled.loc[u, c]:5.1f}/{100 * share.loc[u, c]:4.0f}/{100 * ab_spread.loc[u, c]:<4.0f}"
            for c in cats) + f"  {int(n_g[u])}+{int(n_ab[u])}")
    print(f"      columns: {', '.join(cats)}")
    drawn = pooled.mul(cw, axis=0).sum() / cw.sum()
    print("    national, % of mainland Christians: drawn " + ", ".join(
        f"{c} {100 * drawn[c]:.1f}" for c in cats))
    if (pooled.sum(axis=1) - 1).abs().max() > 1e-9 or (pooled < -1e-12).any().any():
        raise SystemExit("the pooled church shares are not a partition of each unit's Christians")
    return pooled, gnat, int(len(g)), int(n_ab.sum())


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
    zan = [name_to_id[n] for n in ZANZIBAR]
    flagged = sorted(lut.loc[lut["zanzibar"].astype(str).str.lower() == "true", "geo_id"])
    if sorted(zan) != flagged:
        raise SystemExit(f"ZANZIBAR names {sorted(zan)}, tz_lookup.csv flags {flagged}")
    frame = pool_zanzibar(frame, df, zan, carries, nm)
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
    share =counts.div(counts.sum(axis=1), axis=0)
    mainland_max = share.drop(index=zan)["Muslim"].max()
    print(f"\n  Zanzibar's five units, Muslim share as drawn: "
          + ", ".join(f"{nm[z]} {100 * share.loc[z, 'Muslim']:.1f}%" for z in zan)
          + f"; the most Muslim mainland unit is {nm[share.drop(index=zan)['Muslim'].idxmax()]} "
          f"at {100 * mainland_max:.1f}%")
    if share.loc[zan, "Muslim"].min() <= mainland_max:
        raise SystemExit("a mainland unit is drawn more Muslim than a Zanzibar one; check the "
                         "Zanzibar decode before drawing")

    # ---- the churches: each mainland unit's Christians split at the GFS's church shares ----
    mainland = [u for u in units if u not in zan]
    csh, gnat_church, n_gfs_chr, n_ab_chr = church_shares(df, mainland, nm,
                                                           counts.loc[mainland, "Christian"])
    church_cats = CHURCHES + [OTHER_CHRISTIAN]
    split = round_within_rows(csh.mul(counts.loc[mainland, "Christian"].astype(float), axis=0))
    if not (split.sum(axis=1) == counts.loc[mainland, "Christian"]).all():
        raise SystemExit("a unit's churches do not sum to its Christians")
    counts = counts.join(pd.DataFrame(0, index=units, columns=church_cats).add(
        split.reindex(units, fill_value=0), fill_value=0).astype("int64"))
    counts.loc[mainland, "Christian"] = 0
    if not (counts.sum(axis=1) == pop.reindex(units).round().astype("int64")).all():
        raise SystemExit("a unit's drawn total moved when its Christians were split")
    share = counts.div(counts.sum(axis=1), axis=0)
    chr_total = float(counts[church_cats + ["Christian"]].sum().sum())
    print("\n  churches as drawn, % of Tanzania (and of its Christians):")
    for c in church_cats + ["Christian"]:
        n = float(counts[c].sum())
        print(f"    {c:<24}{100 * n / counts.sum().sum():6.2f}%  ({100 * n / chr_total:5.1f}%)")

    # ---- write ----
    n_by = df.groupby("geo_id").size()
    basis_note = {c: ("the unit's own measured share" if c in carries else
                      "the national share" if flat else
                      "the national proportion within the unit's remainder") for c in CATEGORIES}
    for c in church_cats:
        basis_note[c] = (f"the unit's Christians (its own measured share) split at the unit's own "
                         f"church shares from the Global Flourishing Study 2023 ({n_gfs_chr:,} "
                         f"Christians nationally) and Afrobarometer rounds 4, 6, 8 and 9 "
                         f"({n_ab_chr:,} mainland Christians), pooled by respondents, at the GFS's "
                         "national level")
    out = counts.stack().rename("count").reset_index()
    out.columns = ["geo_id", "source_category", "count"]
    out["geo_level"] = "region"
    out["geo_name"] = out["geo_id"].map(nm)
    out["basis"] = "self_id"
    out["year"] = YEARS
    out["source_id"] = SOURCE_ID
    n_zan = int(n_by.reindex(zan).sum())

    out = out[~((out["source_category"] == "Christian") & out["geo_id"].isin(mainland))
              & ~(out["source_category"].isin(church_cats) & out["geo_id"].isin(zan))]

    def basis(r):
        if r.geo_id in zan and r.source_category in ZANZIBAR_POOLED:
            return (f"one share for Zanzibar's five regions, pooled from their {n_zan:,} "
                    "respondents (Anita's ruling, 2026-09-14)")
        if r.geo_id in zan and r.source_category in carries:
            return "the unit's own measured share, scaled around Zanzibar's pooled Christian share"
        return basis_note[r.source_category]

    out["note"] = out.apply(
        lambda r: ("NBS 2022 census population composed with the unit's own mix from "
                   f"Afrobarometer rounds 4 and 6-9 pooled (n={int(n_by[r.geo_id])} here); "
                   f"{basis(r)}"), axis=1)
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
        print(f"    {nm[u]:<20}" + "".join(f"{100 * s[c]:7.1f}%" for c in CATEGORIES + church_cats)
              + f"{int(pop[u]):>12,}  n={int(n_by[u])}")
    print(f"    columns: {', '.join(CATEGORIES + church_cats)}")
    zero = [(nm[u], c) for u in units for c in carries if c != "Christian" and counts.loc[u, c] == 0]
    zero += [(nm[u], c) for u in mainland for c in CHURCHES if counts.loc[u, c] == 0]
    print(f"  drawn at zero in a carried category (no pooled respondent gave it): {zero or 'none'}")
    print(f"  thinnest unit {nm[n_by.idxmin()]} n={int(n_by.min())}; median n={int(n_by.median())}; "
          f"total n={int(n_by.sum()):,}")


if __name__ == "__main__":
    main()
