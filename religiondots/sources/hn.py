"""Honduras — religion by department, from INE's own ENDESA-MICS 2019.

Reads data/raw/hn/endesa/Bases de datos/hh.sav and writes data/normalized/hn.csv.
`sources/hn.md` is this country's record; `sources.md` §11ap is the scouting that found it.

## THE SOURCE

The *Encuesta Nacional de Demografía y Salud* 2019 was fielded as MICS6 with UNICEF, and INE
publishes its microdata as an open zip on `ine.gob.hn` (`BasesdatosENDESA2019.zip`, seven SPSS
files). The household questionnaire carries `HC1`, *"Religión del jefe del hogar"*, the same
MICS item the Dominican Republic is drawn from (§9cf, `sources/do.py`), with a card that names
Mormons as well as Adventists and Witnesses:

    CATOLICA · EVANGELICA · TESTIGOS DE JEHOVA · MORMON · ADVENTISTA ·
    OTRO(ESPECIFIQUE) · NINGUNA RELIGION · NO RESPONDE

**20,669 completed households, 80,439 people in them, 1,221 clusters, all 18 departments.**
The report prints the item and tabulates none of it. No Honduran census has asked religion
(the Censo 2026 form does not either, §11ap), so every row is `modelled`.

## IT IS THE HOUSEHOLD HEAD'S RELIGION

Everyone in a household is drawn in its head's column, which is `sources/do.py`'s ceiling.
Nothing in this survey prices the error (no module asks a person their own religion), and
the Dominican measurement of it is not transferred here.

## INE'S COPY HAS NO WEIGHTS, SO THEY ARE REBUILT FROM TWO TABLES IN THE REPORT

None of the seven files carries a weight, PSU or stratum variable. The MICS copy that does is
behind the account `ask/006-pa` already asks for. `rebuild_weights()` makes an approximate
household weight out of:

  * **Tabla SR.3.1** (printed p.70): weighted and unweighted households for each of the 20
    sampling domains, and nationally by urban and rural. Every unweighted figure is asserted
    against the microdata, which proves the transcription and the domain codes together.
  * **Tabla SD.1** (printed p.765): census enumeration areas in the frame and in the sample,
    by department and area. The frame counts are asserted against the table's own totals.

The design drew PSUs with probability proportional to size inside department x area, so the
weight is roughly constant inside a stratum. Each department's SR.3.1 household total is
split between urban and rural in proportion to its frame enumeration areas, with ONE national
factor for how many more households a rural area holds, fitted so that the national urban
share comes out at SR.3.1's printed 47.5%. San Pedro Sula and the Distrito Central take their
own SR.3.1 totals out of their department's urban part. Three other SR.3.1 rows the fit never
saw (household size, sex of the head, ethnicity of the head) are the witnesses, and the
rebuilt weights have to reproduce them better than no weights at all.

**The weights barely matter for the map, and this says why.** Department TOTALS come from
INE's population projection, not from the survey. The weights only move a department's
shares through its urban/rural mix, and the national rate of any category drawn flat. Both
effects are printed.

## WHICH CATEGORIES CARRY THEIR OWN GEOGRAPHY

§14.16's split-half, on CLUSTERS (the sampling unit, `sources/do.py`'s argument), taken as the
MEDIAN over 400 random halvings drawn inside each department rather than one parity split
(§9cy: one halving is a draw, the median is a statistic; the Dominican reviewer showed parity
alone moved the Witnesses by 0.18). Three requirements, each of which can only make a
category fail:

  1. the median rank correlation clears the §14.16 bar, 1.96/sqrt(17) = +0.475;
  2. **Sweden's spatial chi-square** (§9bi, `sources/se.py`) at 0.05, here on unweighted
     HOUSEHOLDS, the unit that answered, rather than on persons;
  3. **no single cluster holds half of the category's households** (spec §12, Uzbekistan).

    CATOLICA             +0.907   own geography
    EVANGELICA           +0.886   own geography
    NINGUNA RELIGION     +0.841   own geography
    TESTIGOS DE JEHOVA   +0.522   own geography, the narrowest pass (63.5% of halvings clear)
    ADVENTISTA           +0.358   fails; Islas de la Bahía stands apart in 400 of 400 halvings
    MORMON               +0.263   fails
    OTRO(ESPECIFIQUE)    +0.122   fails

**There is no coarser published tier to fall back to.** ENDESA's domains are the departments
and the two cities and the report tabulates nothing in between, so `sources/se.py`'s second
level does not exist here. What the Adventists show instead is a coarser partition of a
different kind: **one department apart from the rest.** The rank test fails because the other
seventeen are indistinguishable from each other at 205 households, not because nothing is
there; Islas de la Bahía is the highest Adventist department in both halves of every one of
the 400 halvings (75 households, 27 of its 42 clusters, 8.3% against 0.6% elsewhere). So a
failing category keeps its measured share in a department that stands apart in at least 95%
of halvings and takes its share across the other departments everywhere else. The rule is
applied to every failing category and only the Adventists meet it (Mormons 23%, `OTRO` 25%).

**The other failing categories take their NATIONAL share, and the carried shares are scaled
to fill the rest** (`sources/do.py`'s construction, applied to several categories). §9bi's
residual construction was run first and rejected on its own reversal check: it drew
Latter-day Saints at 3.85% of the Bay Islands against 1.26% measured, and at 0.94% of Gracias
a Dios where the survey found none, because those two departments' tails are Adventist and
`OTRO`, not a national mix (spec §12, Latvia).

## THE CROSS-CHECK

LAPOP's AmericasBarometer 2012, 2014, 2018 and 2023, decoded per §11ap. In 2012-2018 `prov`
is LAPOP's own department order and the labels the merged file prints are 2023's, so a label
join misplaces most respondents; `lapop_decode()` rebuilds the decode from nothing but the
`municipio` names and COD-AB's municipality table, and asserts it against §11ap's table.
2010 has no `municipio` and 2016's answer codes are shifted, so both are left out.

**That assertion is not independent**: §11ap read the same names the same way. So
`population_witness()` also pins the decode with no names and no religion, LAPOP's weighted
share of respondents per department against INE's 2024 projection, and IS asserted on every
build that has the LAPOP extract (sources/hn.md §9). The religion comparison itself is
reported, never asserted.

Usage:
    python sources/hn.py --fetch    INE's zip (13 MB) and a Honduras extract of the LAPOP merge
    python sources/hn.py            rebuild data/normalized/hn.csv
"""

import os
import sys
import unicodedata
import urllib.request
import zipfile

os.environ.setdefault("OMP_NUM_THREADS", "6")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "hn")
ZIP = os.path.join(RAW, "BasesdatosENDESA2019.zip")
HH_SAV = os.path.join(RAW, "endesa", "Bases de datos", "hh.sav")
LOOKUP = os.path.join(ROOT, "data", "geo", "hn", "hn_lookup.csv")
MUNIS = os.path.join(ROOT, "data", "geo", "hn", "hn_municipios.csv")
LAPOP_DTA = os.path.join(ROOT, "data", "raw", "lapop",
                         "Grand_Merge_2004-2023_LAPOP_AmericasBarometer_v1.0_FREE.dta")
LAPOP_HN = os.path.join(ROOT, "data", "raw", "lapop", "lapop_hn_municipio.feather")
OUT = os.path.join(ROOT, "data", "normalized", "hn.csv")

SOURCE_ID = "hn_ine_endesa_mics_2019"
ZIP_URL = "https://ine.gob.hn/wp-content/uploads/2025/02/BasesdatosENDESA2019.zip"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) religiondots/1.0"}

N_DEPT = 18

# HC1's value labels, verbatim from hh.sav's label set. What reaches `source_category`.
CATEGORY = {
    1: "CATOLICA",
    2: "EVANGELICA",
    3: "TESTIGOS DE JEHOVA",
    4: "MORMON",
    5: "ADVENTISTA",
    6: "OTRO(ESPECIFIQUE)",
    7: "NINGUNA RELIGION",
    9: "NO RESPONDE",
}
NONRESPONSE = 9
RELIGIONS = [c for c in CATEGORY if c != NONRESPONSE]

COMPLETED = 1
CITY_INTO_DEPT = {19: 5, 20: 8}       # San Pedro Sula -> Cortés, Distrito Central -> F. Morazán

N_HOUSEHOLDS = 20_669
N_PEOPLE = 80_439

# ---------------------------------------------------------------------------------------
# Tabla SR.3.1, printed p.70 (PDF p.86 of the UNAH mirror): households by domain,
# (weighted, unweighted), keyed by HH7. 5 and 8 are "Resto Cortés" and "Resto Francisco
# Morazán"; 19 and 20 the two cities. The weighted column sums to 20,668 against a printed
# total of 20,669, which is the table's own rounding.
SR31_DOMAIN = {
    1: (1064, 1053), 2: (742, 991), 3: (1267, 1151), 4: (941, 1077), 5: (2475, 1331),
    6: (1115, 1151), 7: (1134, 1108), 8: (1525, 1135), 9: (161, 675), 10: (586, 957),
    11: (202, 803), 12: (498, 988), 13: (791, 1067), 14: (405, 843), 15: (1207, 1075),
    16: (1112, 1103), 17: (472, 935), 18: (1414, 1213), 19: (1364, 986), 20: (2193, 1027),
}
SR31_AREA = {1: (9812, 8653), 2: (10857, 12016)}                # HH6: urbana, rural
# The three witness rows. Household size (HH48), sex of the head (HHSEX), ethnicity of the
# head (HC2, grouped the way the table groups it).
SR31_SIZE = {1: (2144, 2215), 2: (3162, 3139), 3: (4248, 4214), 4: (4373, 4256),
             5: (3061, 3078), 6: (1795, 1781), 7: (1885, 1986)}
SR31_SEX = {1: (13802, 13820), 2: (6867, 6849)}
SR31_ETHNIC = {"Garífuna": (253, 406), "Lenca": (1274, 1616), "Maya Chortí": (245, 285),
               "Misquito": (208, 645), "Otros Pueblos": (771, 768),
               "Ninguno / No sabe / No responde": (17918, 16949)}
HC2_GROUP = {1: "Garífuna", 7: "Lenca", 9: "Maya Chortí", 5: "Misquito",
             2: "Otros Pueblos", 3: "Otros Pueblos", 4: "Otros Pueblos", 6: "Otros Pueblos",
             8: "Otros Pueblos", 96: "Otros Pueblos",
             98: "Ninguno / No sabe / No responde", 99: "Ninguno / No sabe / No responde"}

# Tabla SD.1, printed p.765 (PDF p.781): enumeration areas (UPM) by department, as
# (frame urban, frame rural, sample urban, sample rural). Cortés and Francisco Morazán
# include their cities. Totals printed: frame 3,026 / 2,656, sample 592 / 634.
SD1 = {
    1: (217, 104, 43, 22), 2: (102, 119, 30, 29), 3: (160, 180, 32, 34), 4: (103, 160, 23, 39),
    5: (880, 215, 130, 21), 6: (106, 177, 27, 36), 7: (94, 206, 22, 42), 8: (715, 262, 96, 51),
    9: (18, 35, 17, 24), 10: (33, 119, 15, 39), 11: (29, 25, 24, 18), 12: (38, 94, 14, 38),
    13: (23, 192, 5, 53), 14: (32, 75, 12, 38), 15: (119, 231, 22, 45), 16: (107, 215, 23, 42),
    17: (48, 71, 20, 31), 18: (202, 176, 37, 32),
}
SD1_ROW_TOTALS = {1: (321, 65), 2: (221, 59), 3: (340, 66), 4: (263, 62), 5: (1095, 151),
                  6: (283, 63), 7: (300, 64), 8: (977, 147), 9: (53, 41), 10: (152, 54),
                  11: (54, 42), 12: (132, 52), 13: (215, 58), 14: (107, 50), 15: (350, 67),
                  16: (322, 65), 17: (119, 51), 18: (378, 69)}
SD1_TOTALS = (3026, 2656, 592, 634)

# ---------------------------------------------------------------------------------------
# The split-half, §14.16, on clusters.
STABILITY_BAR = 1.96 / np.sqrt(N_DEPT - 1)
N_SPLITS = 400
SPLIT_SEED = 20260914
CHI_ALPHA = 0.05
CLUSTER_CAP = 0.5

# Categories drawn on their own department shares. Asserted against `stability()`, so a change
# in the data stops the build rather than quietly redrawing the country.
CARRIES = [1, 2, 3, 7]

# A category that fails may still keep its measured share in ONE department, if that department
# is the category's highest in BOTH halves of at least STANDOUT_AGREE of the halvings (and the
# chi-square and cluster requirements hold). Everywhere else it takes its share across the other
# departments. Asserted, like CARRIES. See the module docstring.
STANDOUT_AGREE = 0.95
STANDOUTS = {5: 11}          # ADVENTISTA in Islas de la Bahía

# LAPOP's own department order in 2012, 2014 and 2018, as §11ap read it off the `municipio`
# names. `lapop_decode()` re-derives it and asserts it equals this. In 2023 `prov` is 400 plus
# INE's official number, which is also ENDESA's HH7.
LAPOP_1218 = {401: 8, 402: 3, 403: 12, 404: 5, 405: 1, 406: 2, 407: 18, 408: 11, 409: 4,
              410: 10, 411: 13, 412: 14, 413: 16, 418: 7, 419: 15, 420: 9, 421: 6, 422: 17}
LAPOP_WAVES = [2012, 2014, 2018, 2023]


def fold(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return "".join(ch for ch in s.lower() if ch.isalnum())


# =======================================================================================
# fetch and load
# =======================================================================================

def fetch():
    os.makedirs(RAW, exist_ok=True)
    if not (os.path.exists(ZIP) and os.path.getsize(ZIP) > 1_000_000):
        req = urllib.request.Request(ZIP_URL, headers=UA)
        with urllib.request.urlopen(req, timeout=900) as r, open(ZIP + ".part", "wb") as f:
            f.write(r.read())
        os.replace(ZIP + ".part", ZIP)
    print(f"  have {os.path.basename(ZIP)} ({os.path.getsize(ZIP):,} bytes)")
    with zipfile.ZipFile(ZIP) as z:
        z.extractall(os.path.join(RAW, "endesa"))
    if not os.path.exists(LAPOP_HN):
        fetch_lapop()


def fetch_lapop():
    """Slim the LAPOP grand merge to Honduras with `prov` and `municipio` and their labels.

    `sources/lapop.py`'s slim file leaves `municipio` out, and the municipality names are the
    only thing that decodes Honduras. Reading the 1.1 GB .dta takes a few minutes; this runs
    once and the extract is kept.
    """
    import pyreadstat

    if not os.path.exists(LAPOP_DTA):
        print(f"  {LAPOP_DTA} not on disk; the LAPOP cross-check will be skipped")
        return
    print("  slimming the LAPOP merge to Honduras (a few minutes)...")
    cols = ["pais", "year", "prov", "municipio", "q3c", "q3cn", "weight1500"]
    df, meta = pyreadstat.read_dta(LAPOP_DTA, usecols=cols, apply_value_formats=False)
    df = df[df["pais"] == 4].copy()
    labs = meta.variable_value_labels

    def label(var, v):
        try:
            return labs.get(var, {}).get(float(v), labs.get(var, {}).get(int(float(v)), ""))
        except (TypeError, ValueError):
            return ""

    df["prov_lab"] = [label("prov", v) for v in df["prov"]]
    df["muni_lab"] = [label("municipio", v) for v in df["municipio"]]
    for c in ("prov", "municipio", "q3c", "q3cn"):
        df[c] = df[c].astype(str)
    df.reset_index(drop=True).to_feather(LAPOP_HN)
    print(f"  wrote {LAPOP_HN} ({len(df):,} Honduran respondents)")


def check_labels(meta):
    got = {int(k): v for k, v in meta.variable_value_labels.get("HC1", {}).items()}
    if got != CATEGORY:
        raise SystemExit(f"HC1's value labels have changed: {got}")
    print(f"  HC1: {meta.column_names_to_labels.get('HC1')!r}, {len(got)} answers matching "
          "hh.sav's label set exactly")


def load():
    import pyreadstat

    if not os.path.exists(HH_SAV):
        raise SystemExit(f"{HH_SAV} missing, run with --fetch")
    _, meta = pyreadstat.read_sav(HH_SAV, metadataonly=True)
    check_labels(meta)
    hh, _ = pyreadstat.read_sav(HH_SAV, usecols=["HH1", "HH2", "HH6", "HH7", "HH46", "HH48",
                                                "HC1", "HC2", "HHSEX"])
    done = hh["HH46"] == COMPLETED
    if int(done.sum()) != N_HOUSEHOLDS:
        raise SystemExit(f"{int(done.sum()):,} completed households, expected {N_HOUSEHOLDS:,}")
    if hh.loc[done, "HC1"].isna().any() or hh.loc[~done, "HC1"].notna().any():
        raise SystemExit("HC1 is not answered for exactly the completed households")
    hh = hh[done].copy()
    for c in ("HH1", "HH2", "HH6", "HH7", "HH48", "HC1"):
        hh[c] = hh[c].astype(int)
    unknown = sorted(set(hh["HC1"]) - set(CATEGORY))
    if unknown:
        raise SystemExit(f"HC1 codes with no label: {unknown}")
    if int(hh["HH48"].sum()) != N_PEOPLE:
        raise SystemExit(f"HH48 sums to {int(hh['HH48'].sum()):,}, expected {N_PEOPLE:,}")

    # hl.sav as a witness on HH48: one roster line per member, keyed (HH1, HH2).
    hl_path = os.path.join(os.path.dirname(HH_SAV), "hl.sav")
    hl, _ = pyreadstat.read_sav(hl_path, usecols=["HH1", "HH2"])
    n_hl = hl.groupby(["HH1", "HH2"]).size()
    hh = hh.join(n_hl.rename("n_hl"), on=["HH1", "HH2"])
    if (hh["n_hl"] != hh["HH48"]).any():
        raise SystemExit("HH48 differs from the member roster's line count in some households")

    hh["domain"] = hh["HH7"]
    hh["dept"] = hh["HH7"].replace(CITY_INTO_DEPT)
    if sorted(hh["dept"].unique()) != list(range(1, N_DEPT + 1)):
        raise SystemExit("departments after merging the two cities are not 1..18")
    if (hh.loc[hh["domain"].isin(CITY_INTO_DEPT), "HH6"] != 1).any():
        raise SystemExit("a San Pedro Sula or Distrito Central household is coded rural")
    if (hh.groupby("HH1")["domain"].nunique() > 1).any():
        raise SystemExit("a cluster spans two sampling domains")
    print(f"Honduras: {N_HOUSEHOLDS:,} completed households, {N_PEOPLE:,} people (HH48 equals "
          f"the roster in every one), {hh['HH1'].nunique():,} clusters, {N_DEPT} departments")
    return hh


# =======================================================================================
# the weights
# =======================================================================================

def rebuild_weights(hh):
    """Approximate household weights out of Tablas SR.3.1 and SD.1. See the module docstring."""
    # ---- the transcriptions, proven against the microdata and against themselves
    unw = hh.groupby("domain").size()
    bad = {d: (int(unw.get(d, 0)), u) for d, (_, u) in SR31_DOMAIN.items() if unw.get(d, 0) != u}
    if bad:
        raise SystemExit(f"SR.3.1's unweighted households disagree with hh.sav: {bad}")
    area = hh.groupby("HH6").size()
    if any(int(area[a]) != u for a, (_, u) in SR31_AREA.items()):
        raise SystemExit("SR.3.1's unweighted urban/rural split disagrees with hh.sav")
    for d, (fu, fr, su, sr) in SD1.items():
        if (fu + fr, su + sr) != SD1_ROW_TOTALS[d]:
            raise SystemExit(f"SD.1 row {d} does not add to its own total")
    if tuple(int(sum(v[i] for v in SD1.values())) for i in range(4)) != SD1_TOTALS:
        raise SystemExit("SD.1's columns do not add to the printed totals")
    print("  SR.3.1: all 20 unweighted domain counts and both area counts match hh.sav; "
          "SD.1: every row and column adds to its printed total")

    # sampled clusters in the data against SD.1's sampled UPMs
    cl = hh.drop_duplicates("HH1").groupby(["dept", "HH6"]).size().unstack(fill_value=0)
    short = {d: (int(cl.loc[d, 1]), SD1[d][2], int(cl.loc[d, 2]), SD1[d][3]) for d in SD1
             if (cl.loc[d, 1], cl.loc[d, 2]) != (SD1[d][2], SD1[d][3])}
    print(f"  clusters with a completed household: {hh['HH1'].nunique():,} of SD.1's "
          f"{sum(SD1_TOTALS[2:]):,} sampled; departments that differ (urban got/SD.1, "
          f"rural got/SD.1): {short or 'none'}")

    dept_total = {d: float(SR31_DOMAIN[d][0]) for d in SD1}
    for city, d in CITY_INTO_DEPT.items():
        dept_total[d] += SR31_DOMAIN[city][0]
    target_urban = SR31_AREA[1][0] / (SR31_AREA[1][0] + SR31_AREA[2][0])

    def urban_share(k):
        return {d: fu / (fu + k * fr) for d, (fu, fr, _, _) in SD1.items()}

    def national_urban(k):
        u = urban_share(k)
        return sum(dept_total[d] * u[d] for d in SD1) / sum(dept_total.values())

    lo, hi = 0.2, 5.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if national_urban(mid) > target_urban:
            lo = mid
        else:
            hi = mid
    k = (lo + hi) / 2
    us = urban_share(k)
    print(f"  one national factor: a rural enumeration area holds {k:.3f}x the households of "
          f"an urban one, which puts the country {national_urban(k):.2%} urban against "
          f"SR.3.1's {target_urban:.2%}")

    stratum_w = {}
    for d in SD1:
        urban = dept_total[d] * us[d]
        rural = dept_total[d] - urban
        city = next((c for c, dd in CITY_INTO_DEPT.items() if dd == d), None)
        if city is not None:
            urban -= SR31_DOMAIN[city][0]
            if urban <= 0:
                raise SystemExit(f"department {d}'s urban households are all in its city; "
                                 "the frame proportions do not fit")
            stratum_w[(city, 1)] = SR31_DOMAIN[city][0] / int(unw[city])
        n_u = int(((hh["domain"] == d) & (hh["HH6"] == 1)).sum())
        n_r = int(((hh["domain"] == d) & (hh["HH6"] == 2)).sum())
        stratum_w[(d, 1)] = urban / n_u
        stratum_w[(d, 2)] = rural / n_r
    hh["hw"] = [stratum_w[(d, a)] for d, a in zip(hh["domain"], hh["HH6"])]
    hh["hw"] *= len(hh) / hh["hw"].sum()
    hh["pw"] = hh["hw"] * hh["HH48"]
    print(f"  household weights run {hh['hw'].min():.3f} to {hh['hw'].max():.3f} "
          f"(Gracias a Dios lowest, the Distrito Central highest is expected)")

    # ---- the witnesses the fit never saw
    def miss(col, table, keyfun):
        keys = hh[col].map(keyfun)
        w = hh.groupby(keys)["hw"].sum()
        n = hh.groupby(keys).size()
        if any(int(n.get(g, 0)) != u for g, (_, u) in table.items()):
            raise SystemExit(f"SR.3.1's unweighted {col} row disagrees with hh.sav: "
                             f"{ {g: (int(n.get(g, 0)), u) for g, (_, u) in table.items()} }")
        scale = sum(v[0] for v in table.values()) / w.sum()
        rebuilt = sum(abs(w.get(g, 0) * scale - wt) for g, (wt, _) in table.items())
        none = sum(abs(n.get(g, 0) - wt) for g, (wt, _) in table.items())
        return rebuilt, none

    print("  witnesses (households misplaced against SR.3.1's weighted column, summed):")
    tot_rebuilt = tot_none = 0.0
    for label, col, table, keyfun in [
            ("household size", "HH48", SR31_SIZE, lambda v: min(int(v), 7)),
            ("sex of the head", "HHSEX", SR31_SEX, lambda v: int(v)),
            ("ethnicity of the head", "HC2", SR31_ETHNIC,
             lambda v: HC2_GROUP.get(int(v), "Ninguno / No sabe / No responde")
             if pd.notna(v) else "Ninguno / No sabe / No responde")]:
        r, n = miss(col, table, keyfun)
        tot_rebuilt += r
        tot_none += n
        print(f"    {label:<24} rebuilt weights {r:7,.0f}   no weights {n:7,.0f}")
    if tot_rebuilt >= tot_none:
        raise SystemExit("the rebuilt weights reproduce SR.3.1's witness rows no better than "
                         "no weights at all; the reconstruction is not doing its job")
    return hh


# =======================================================================================
# checks
# =======================================================================================

def held_out(hh, pop, n_perm=20000, seed=0):
    """Department share of people in the survey against INE's projection, nothing religious."""
    print("\n  held-out check (nothing here touches the religion column):")
    s = hh.groupby("geo_id")["pw"].sum()
    s = s / s.sum()
    p = pop / pop.sum()
    j = pd.concat([s.rename("survey"), p.rename("pop")], axis=1)
    if j.isna().any().any():
        raise SystemExit("a department is in one of the two and not the other")
    r = np.corrcoef(j["survey"], j["pop"])[0, 1]
    rng = np.random.default_rng(seed)
    a, b = j["survey"].to_numpy(), j["pop"].to_numpy()
    perm = np.array([np.corrcoef(a, rng.permutation(b))[0, 1] for _ in range(n_perm)])
    beaten = int((perm >= r).sum())
    ratio = (j["survey"] / j["pop"]).sort_values()
    print(f"    share of people, ENDESA 2019 (rebuilt weights) vs INE 2024: r = {r:+.4f}, "
          f"{beaten} of {n_perm:,} random pairings reach it (best {perm.max():+.3f})")
    print(f"    ratio runs {ratio.iloc[0]:.3f} ({ratio.index[0]}) to {ratio.iloc[-1]:.3f} "
          f"({ratio.index[-1]})")
    if beaten:
        raise SystemExit("the population check does not pin the department join")


def stability(hh, names):
    """§14.16 on clusters, median of N_SPLITS halvings, with Sweden's chi-square and the
    largest-cluster check beside it. Returns the categories that pass all three."""
    from scipy.stats import rankdata

    import stability as shared      # this function is named `stability` too
    rel = hh[hh["HC1"] != NONRESPONSE]
    depts = sorted(rel["dept"].unique())
    clusters = rel.groupby("HH1").agg(dept=("dept", "first")).reset_index()
    ci = {c: i for i, c in enumerate(clusters["HH1"])}
    di = {d: i for i, d in enumerate(depts)}
    M = np.zeros((len(clusters), len(RELIGIONS)))
    for (c, code), w in rel.groupby(["HH1", "HC1"])["pw"].sum().items():
        M[ci[c], RELIGIONS.index(code)] = w
    T = M.sum(axis=1)
    D = np.zeros((len(depts), len(clusters)))
    D[[di[d] for d in clusters["dept"]], np.arange(len(clusters))] = 1.0

    def rho_for(mask):
        a_num, a_den = D @ (M * mask[:, None]), D @ (T * mask)
        b_num, b_den = D @ (M * (1 - mask)[:, None]), D @ (T * (1 - mask))
        sa, sb = a_num / a_den[:, None], b_num / b_den[:, None]
        out = np.full(len(RELIGIONS), np.nan)
        for j in range(len(RELIGIONS)):
            ra, rb = rankdata(sa[:, j]), rankdata(sb[:, j])
            if ra.std() > 0 and rb.std() > 0:
                out[j] = np.corrcoef(ra, rb)[0, 1]
        return out, sa, sb

    rng = np.random.default_rng(SPLIT_SEED)
    by_dept = [np.flatnonzero(clusters["dept"].to_numpy() == d) for d in depts]
    draws = np.empty((N_SPLITS, len(RELIGIONS)))
    halves_a = np.empty((N_SPLITS, len(depts), len(RELIGIONS)))
    halves_b = np.empty_like(halves_a)
    for s in range(N_SPLITS):
        mask = np.zeros(len(clusters))
        for idx in by_dept:
            pick = rng.permutation(idx)[: len(idx) // 2]
            mask[pick] = 1.0
        draws[s], halves_a[s], halves_b[s] = rho_for(mask)
    top_unit, top_share = shared.top_both_halves(halves_a, halves_b)
    parity = rho_for((clusters["HH1"].to_numpy() % 2 == 0).astype(float))[0]
    median = np.nanmedian(draws, axis=0)
    passrate = np.mean(draws >= STABILITY_BAR, axis=0)

    nat = rel.groupby("HC1")["pw"].sum() / rel["pw"].sum()
    n_dept = rel.groupby("dept").size().reindex(depts)
    print(f"\n  split-half on clusters (§14.16), median of {N_SPLITS} halvings inside each "
          f"department, bar +{STABILITY_BAR:.3f} on {N_DEPT} departments; Sweden's chi-square "
          f"on households; the largest single cluster's share of each answer:")
    print(f"    {'category':<22}{'national':>9}{'hh':>6}{'median':>8}{'parity':>8}"
          f"{'pass %':>8}{'chi2 p':>10}{'top cl':>8}  verdict")
    carries, eligible = [], {}
    for j, code in enumerate(RELIGIONS):
        hit = rel[rel["HC1"] == code].groupby("dept").size().reindex(depts).fillna(0)
        p = shared.chi2_p(hit.to_numpy(), n_dept.to_numpy())
        top = rel[rel["HC1"] == code].groupby("HH1").size().max() / max(int(hit.sum()), 1)
        ok_rank = np.isfinite(median[j]) and median[j] >= STABILITY_BAR
        ok_chi = p < CHI_ALPHA
        ok_cl = top < CLUSTER_CAP
        eligible[code] = ok_chi and ok_cl
        if ok_rank and ok_chi and ok_cl:
            verdict = "own geography"
            carries.append(code)
        elif ok_rank and not ok_chi:
            verdict = "REFUSED: passes the rank test, the departments do not differ"
        elif ok_rank and not ok_cl:
            verdict = "REFUSED: passes, but one cluster is most of the answer"
        else:
            verdict = "fails; see the standout test below"
        print(f"    {CATEGORY[code]:<22}{nat[code] * 100:8.2f}%{int(hit.sum()):>6,}"
              f"{median[j]:+8.3f}{parity[j]:+8.3f}{passrate[j] * 100:7.1f}%{p:10.1e}"
              f"{top:8.2f}  {verdict}")
    if carries != CARRIES:
        raise SystemExit(
            f"the split-half now says {[CATEGORY[c] for c in carries]} carry their own "
            f"geography, against CARRIES={[CATEGORY[c] for c in CARRIES]}. Read the table "
            "above, then edit CARRIES deliberately. Do not move the bar.")

    # ---- standouts: a failing category whose highest department is the same in both halves
    standouts = {}
    print(f"\n  categories that fail, and whether one department stands apart in both halves "
          f"(needs {STANDOUT_AGREE:.0%} of {N_SPLITS}):")
    for j, code in enumerate(RELIGIONS):
        if code in carries:
            continue
        if top_unit[j] >= 0:
            dept, share = depts[top_unit[j]], top_share[j]
        else:
            dept, share = None, 0.0
        ok = share >= STANDOUT_AGREE and eligible[code]
        if ok:
            standouts[code] = dept
        print(f"    {CATEGORY[code]:<22} {names.get(dept, '-') if dept else '-':<20} "
              f"{share:6.1%}   {'kept in that department' if ok else 'flat'}")
    if standouts != STANDOUTS:
        raise SystemExit(f"standouts are now {standouts}, against STANDOUTS={STANDOUTS}; read "
                         "the table above and edit STANDOUTS deliberately")
    return carries, standouts


def lapop_decode(names):
    """Re-derive §11ap's decode from `municipio` names alone, assert it, and return the frame."""
    if not os.path.exists(LAPOP_HN):
        if os.path.exists(LAPOP_DTA):
            fetch_lapop()
        else:
            return None
    lp = pd.read_feather(LAPOP_HN)
    munis = pd.read_csv(MUNIS, dtype=str)
    muni_depts = {}
    for n, p in zip(munis["adm2_name"], munis["adm1_pcode"]):
        muni_depts.setdefault(fold(n), set()).add(int(p[2:]))

    lp = lp[lp["year"].astype(int).isin(LAPOP_WAVES)].copy()
    lp["year"] = lp["year"].astype(int)
    lp["prov_i"] = pd.to_numeric(lp["prov"], errors="coerce")
    lp = lp[lp["prov_i"].notna()].copy()
    lp["prov_i"] = lp["prov_i"].astype(int)

    # For each 2012-2018 prov code: the department every one of its municipality names shares.
    early = lp[lp["year"] < 2023]
    derived, unplaced = {}, []
    for p, g in early.groupby("prov_i"):
        sets = [muni_depts.get(fold(m)) for m in g["muni_lab"].unique()]
        if any(s is None for s in sets):
            unplaced += [m for m in g["muni_lab"].unique() if fold(m) not in muni_depts]
            continue
        common = set.intersection(*sets)
        if len(common) == 1:
            derived[p] = common.pop()
    if unplaced:
        raise SystemExit(f"LAPOP municipality names COD-AB does not have: {sorted(set(unplaced))}")
    if derived != LAPOP_1218:
        raise SystemExit(f"the municipio names give {derived}, §11ap's table says "
                         f"{LAPOP_1218}; do not pool until they agree")
    late = lp[lp["year"] == 2023]
    wrong23 = [(m, p) for m, p in zip(late["muni_lab"], late["prov_i"])
               if (p - 400) not in muni_depts.get(fold(m), set())]
    if wrong23:
        raise SystemExit(f"2023 respondents whose municipio is not in prov-400: {wrong23[:5]}")
    lp["dept"] = np.where(lp["year"] == 2023, lp["prov_i"] - 400, lp["prov_i"].map(LAPOP_1218))
    naive = lp["prov_lab"].map(lambda s: next((c for c, n in names.items() if fold(n) == fold(s)),
                                              None))
    misplaced = float((naive != lp["dept"]).mean())
    print(f"\n  LAPOP decode: the {len(derived)} codes of 2012-2018 each resolve to exactly one "
          f"department from their municipio names alone, equal to §11ap's table; every 2023 "
          f"municipio sits in prov-400. The printed labels would misplace {misplaced:.1%}.")
    return lp


def population_witness(lp, pop, unit_of, names, n_perm=20000, seed=0):
    """Pin the LAPOP decode with something that reads no municipality name and no answer.

    `lapop_decode()` asserts that the `municipio` names give §11ap's table, but §11ap read the
    same names the same way, so that check proves the transcription and not the decode
    (sources/hn.md §8). A national sample is allocated roughly by population, so LAPOP's
    weighted share of respondents per department has to track INE's 2024 projection under the
    right decode and should not under a wrong one. Asserted: the decoded shares beat every one
    of `n_perm` random pairings, and beat the printed labels both pooled and in each 2012-2018
    wave (in 2023 the labels are correct, so the two are the same). The review measured
    r = +0.939 decoded against +0.300 on the labels. Nothing here reaches a count.
    """
    ps = pop / pop.sum()
    w = pd.to_numeric(lp["weight1500"], errors="coerce").fillna(1.0)
    decoded = lp["dept"].map(unit_of)
    labelled = lp["prov_lab"].map(
        lambda s: next((unit_of[c] for c, n in names.items() if fold(n) == fold(s)), None))
    if decoded.isna().any():
        raise SystemExit("population witness: a decoded LAPOP department has no geo_id")

    def share_r(dept, mask):
        s = w[mask & dept.notna()].groupby(dept[mask & dept.notna()]).sum()
        s = s.reindex(ps.index).fillna(0.0)
        s = s / s.sum()
        return float(np.corrcoef(s.to_numpy(), ps.to_numpy())[0, 1]), s

    everyone = pd.Series(True, index=lp.index)
    r_dec, s_dec = share_r(decoded, everyone)
    r_lab, _ = share_r(labelled, everyone)
    rng = np.random.default_rng(seed)
    a, b = s_dec.to_numpy(), ps.to_numpy()
    perm = np.array([np.corrcoef(a, rng.permutation(b))[0, 1] for _ in range(n_perm)])
    beaten = int((perm >= r_dec).sum())
    print(f"\n  population witness on the decode (no names, no religion): LAPOP's weighted share "
          f"of respondents per department against INE 2024, r = {r_dec:+.3f} decoded "
          f"({beaten} of {n_perm:,} random pairings reach it), {r_lab:+.3f} on the printed "
          f"labels; n={len(lp):,}")
    problems = []
    if beaten:
        problems.append(f"{beaten} of {n_perm:,} random pairings reach the decoded r")
    if not r_dec > r_lab:
        problems.append(f"the printed labels ({r_lab:+.3f}) do as well as the decode")
    for y in LAPOP_WAVES:
        m = lp["year"] == y
        ry, ly = share_r(decoded, m)[0], share_r(labelled, m)[0]
        print(f"    {y}: decoded {ry:+.3f}, labels {ly:+.3f}"
              + ("   (labels are correct in this wave)" if y >= 2023 else ""))
        if y < 2023 and not ry > ly:
            problems.append(f"in {y} the printed labels ({ly:+.3f}) do as well as the decode "
                            f"({ry:+.3f})")
    if problems:
        raise SystemExit("the LAPOP decode fails its population witness: " + "; ".join(problems)
                         + ". Do not pool LAPOP by department until this is understood.")


def cross_check(hh, names, pop, unit_of, n_perm=20000, seed=0):
    """LAPOP 2012, 2014, 2018, 2023 against ENDESA, by department. The decode is asserted,
    twice (names in `lapop_decode`, population in `population_witness`); the religion
    comparison is reported, never asserted."""
    lp = lapop_decode(names)
    if lp is None:
        print("\n  cross-check against the AmericasBarometer SKIPPED, no LAPOP file on disk")
        return
    population_witness(lp, pop, unit_of, names, n_perm=n_perm, seed=seed)
    rel = lp["q3c"].where(~lp["q3c"].str.strip().str.lower().isin(
        {"a", "b", "c", "z", "", ".", "nan", "none"}), lp["q3cn"])
    lp["code"] = pd.to_numeric(rel, errors="coerce")
    lp = lp[lp["code"].notna()].copy()
    lp["code"] = lp["code"].astype(int)
    lp["w"] = pd.to_numeric(lp["weight1500"], errors="coerce").fillna(1.0)
    print(f"  cross-check: LAPOP {', '.join(map(str, LAPOP_WAVES))} (n={len(lp):,}) on "
          f"{lp['dept'].nunique()} departments; 2010 has no municipio and 2016's codes shift")
    e = hh[hh["HC1"] != NONRESPONSE]
    etot = e.groupby("dept")["pw"].sum()
    ltot = lp.groupby("dept")["w"].sum()
    pairs = [("Catholic", [1], [1]), ("non-Catholic Christian", [2, 3, 4, 5], [2, 5, 6, 12]),
             ("no religion", [7], [4, 11])]
    rng = np.random.default_rng(seed)
    for label, ec, lc in pairs:
        a = e[e["HC1"].isin(ec)].groupby("dept")["pw"].sum().reindex(etot.index).fillna(0) / etot
        b = lp[lp["code"].isin(lc)].groupby("dept")["w"].sum().reindex(ltot.index).fillna(0) / ltot
        j = pd.concat([a.rename("e"), b.rename("l")], axis=1).dropna()
        r = np.corrcoef(j["e"], j["l"])[0, 1]
        x, y = j["e"].to_numpy(), j["l"].to_numpy()
        perm = np.array([np.corrcoef(x, rng.permutation(y))[0, 1] for _ in range(n_perm)])
        print(f"    {label:<24} r = {r:+.2f}   {int((perm >= r).sum()):>5} of {n_perm:,} reach "
              f"it   ENDESA {a.mul(etot).sum() / etot.sum() * 100:5.1f}%   LAPOP "
              f"{b.mul(ltot).sum() / ltot.sum() * 100:5.1f}%   on {len(j)} departments")


# =======================================================================================
# build
# =======================================================================================

def main():
    if "--fetch" in sys.argv:
        fetch()

    hh = load()
    lut = pd.read_csv(LOOKUP, dtype={"geo_id": str, "unit": str})
    if len(lut) != N_DEPT:
        raise SystemExit(f"{len(lut)} departments in the lookup, re-run sources/hn_geo.py")
    unit_of = dict(zip(lut["endesa_hh7"].astype(int), lut["geo_id"]))
    names = dict(zip(lut["endesa_hh7"].astype(int), lut["name"]))
    gname = dict(zip(lut["geo_id"], lut["name"]))
    pop = pd.Series(dict(zip(lut["geo_id"], lut["pop_2024"].astype(int))))
    hh["geo_id"] = hh["dept"].map(unit_of)

    raw_share = (hh[hh["HC1"] != NONRESPONSE].groupby(["geo_id", "HC1"]).size()
                 .unstack(fill_value=0))
    raw_share = raw_share.div(raw_share.sum(axis=1), axis=0)
    hh = rebuild_weights(hh)
    held_out(hh, pop)

    rel = hh[hh["HC1"] != NONRESPONSE]
    by_unit = rel.groupby(["geo_id", "HC1"])["pw"].sum().unstack(fill_value=0.0)
    unit_share = by_unit.div(by_unit.sum(axis=1), axis=0)
    unw = (rel.groupby(["geo_id", "HC1"])["HH48"].sum().unstack(fill_value=0))
    unw = unw.div(unw.sum(axis=1), axis=0)
    diff = (unit_share - unw).abs().max()
    print("  what the weights move, largest department-share change against unweighted "
          "people: " + ", ".join(f"{CATEGORY[c]} {diff[c] * 100:.2f} pt" for c in RELIGIONS))

    nat = rel.groupby("HC1")["pw"].sum() / rel["pw"].sum()
    nr_share = (hh.assign(nr=(hh["HC1"] == NONRESPONSE) * hh["pw"]).groupby("geo_id")["nr"].sum()
                / hh.groupby("geo_id")["pw"].sum())
    print(f"  NO RESPONDE: {nr_share.mul(pop).sum() / pop.sum():.3%} of people as drawn, "
          f"{int((hh['HC1'] == NONRESPONSE).sum())} households; not drawn")

    carries, standouts = stability(hh, names)
    cross_check(hh, names, pop, {d: g for d, g in unit_of.items()})

    # ---- the composition. Carried categories keep their department shares. A standout keeps
    # its measured share in its one department and takes its pooled share across the other
    # seventeen everywhere else. Every other failing category takes its NATIONAL share, and the
    # carried shares are scaled, in their measured proportions, to fill what is left.
    #
    # NOT §9bi's residual construction, and the reason was measured on this data: splitting each
    # department's measured tail at national proportions drew Latter-day Saints at 3.85% of the
    # Bay Islands against 1.26% measured, and at 0.94% of Gracias a Dios where the survey found
    # none, because those two departments' tails are Adventist and "other" rather than a
    # national mix (spec §12, Latvia). A national share cannot inflate a category anywhere.
    small = [c for c in RELIGIONS if c not in carries]
    flat = [c for c in small if c not in standouts]
    comp = pd.DataFrame(index=unit_share.index, columns=RELIGIONS, dtype=float)
    for c in flat:
        comp[c] = float(nat[c])
    for c, d in standouts.items():
        g = unit_of[d]
        rest = rel[rel["geo_id"] != g]
        pooled = float(rest.loc[rest["HC1"] == c, "pw"].sum() / rest["pw"].sum())
        comp[c] = pooled
        comp.loc[g, c] = unit_share.loc[g, c]
        print(f"\n  {CATEGORY[c]}: {unit_share.loc[g, c]:.2%} in {gname[g]} as measured, "
              f"{pooled:.2%} (its share across the other {N_DEPT - 1}) everywhere else")
    fixed = comp[small].sum(axis=1)
    carried = unit_share[carries].sum(axis=1)
    for k in carries:
        comp[k] = unit_share[k] / carried * (1.0 - fixed)
    if (comp.sum(axis=1) - 1).abs().max() > 1e-9:
        raise SystemExit("composition does not sum to 1")
    print("  spec §12's reversal check, every category not carried, drawn beside measured:")
    for c in small:
        d = comp[c] - unit_share[c]
        print(f"    {CATEGORY[c]:<20} drawn {comp[c].min():.2%} to {comp[c].max():.2%}; "
              f"measured {unit_share[c].min():.2%} to {unit_share[c].max():.2%}; "
              f"biggest gap {gname[d.abs().idxmax()]} {d[d.abs().idxmax()] * 100:+.2f} pt")
    scale = (1.0 - fixed) / carried
    print(f"  the carried shares are scaled by {scale.min():.3f} ({gname[scale.idxmin()]}) to "
          f"{scale.max():.3f} ({gname[scale.idxmax()]})")

    rows = []
    for g in sorted(pop.index):
        p = int(pop[g])
        for c in RELIGIONS:
            rows.append((g, CATEGORY[c], comp.loc[g, c] * (1 - nr_share[g]) * p))
        rows.append((g, CATEGORY[NONRESPONSE], nr_share[g] * p))
    out = pd.DataFrame(rows, columns=["geo_id", "source_category", "count"])
    out["count"] = out["count"].round().astype("int64")
    target = int(pop.sum())
    drift = target - int(out["count"].sum())
    if abs(drift) > len(out):
        raise SystemExit(f"rounding drift {drift} is larger than one person per row")
    if drift:
        out.loc[out["count"].idxmax(), "count"] += drift
    print(f"\n  rounding drift {drift:+d} people, absorbed into the largest cell")

    n_hh = hh.groupby("geo_id").size()
    n_cl = hh.groupby("geo_id")["HH1"].nunique()
    code_of = {v: k for k, v in CATEGORY.items()}

    def how_drawn(g, cat):
        c = code_of[cat]
        if c == NONRESPONSE:
            return "not answered, not drawn"
        if c in carries:
            return "department share"
        if c in standouts:
            if unit_of[standouts[c]] == g:
                return "department share, this department standing apart in every halving"
            return f"its share across the departments other than {names[standouts[c]]}"
        return "national share, having failed the split-half"

    out["geo_level"] = "departamento"
    out["geo_name"] = out["geo_id"].map(gname)
    out["basis"] = "self_id"
    out["year"] = "2019"
    out["source_id"] = SOURCE_ID
    out["note"] = [
        (f"INE ENDESA-MICS 2019, religion of the household head, n={int(n_hh[g]):,} households "
         f"in {int(n_cl[g]):,} clusters; " + how_drawn(g, cat)
         + "; weights rebuilt from Tablas SR.3.1 and SD.1; applied to INE's 2024 projection")
        for g, cat in zip(out["geo_id"], out["source_category"])]
    if int(out["count"].sum()) != target:
        raise SystemExit("drawn total is not the population")

    cols = ["geo_id", "geo_level", "geo_name", "source_category", "count",
            "basis", "year", "source_id", "note"]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out[cols].to_csv(OUT, index=False, encoding="utf-8")
    print(f"wrote {OUT} ({len(out)} rows, {target:,} people, {N_DEPT} departments)")

    drawn = out[out["source_category"] != CATEGORY[NONRESPONSE]]
    tot = drawn["count"].sum()
    print("\n  national, as drawn (share of people drawn):")
    for cat, v in (drawn.groupby("source_category")["count"].sum() / tot).sort_values(
            ascending=False).items():
        print(f"    {v * 100:6.2f}%  {cat}")
    show = drawn.pivot_table(index="geo_id", columns="source_category", values="count",
                             aggfunc="sum")
    show = show.div(show.sum(axis=1), axis=0) * 100
    print(f"\n    {'department':<20}{'hh':>6}{'Cath':>7}{'Evang':>7}{'None':>7}{'Adv':>6}"
          f"{'JW':>6}{'LDS':>6}{'Oth':>6}   unweighted Cath/Evang")
    for g in show[CATEGORY[1]].sort_values().index:
        print(f"    {gname[g]:<20}{int(n_hh[g]):>6,}{show.loc[g, CATEGORY[1]]:7.1f}"
              f"{show.loc[g, CATEGORY[2]]:7.1f}{show.loc[g, CATEGORY[7]]:7.1f}"
              f"{show.loc[g, CATEGORY[5]]:6.1f}{show.loc[g, CATEGORY[3]]:6.1f}"
              f"{show.loc[g, CATEGORY[4]]:6.1f}{show.loc[g, CATEGORY[6]]:6.1f}"
              f"   {raw_share.loc[g, 1] * 100:5.1f}/{raw_share.loc[g, 2] * 100:4.1f}")


if __name__ == "__main__":
    main()
