"""Norway — a roll of every state-funded faith community, and a survey that asks what people are.

Writes data/normalized/no.csv (citizens) and no_foreign.csv (foreign residents).

Usage:
    python sources/no.py --fetch     # ESS tabulations, Eurostat census, Pew, SSB's rolls  (~6 MB)
    python sources/no.py             # rebuild from data/raw/no/

THE OFFICE WAS CHECKED FIRST (§9cu), AND WHAT IT PUBLISHES IS A ROLL, NOT A QUESTION. SSB has no
public shelf of commissioned (`oppdrag`) tables; its site search for `religion` returns 1,067
items and not one is a commissioned religion table or a regional self-identification figure.
What SSB does publish is membership, because the state pays every faith and life-stance community
a grant per member and the county governors check the lists against the population register:

    12025   Church of Norway members and affiliated, % of inhabitants, per kommune and county,
            2015-2025 (67.7% nationally in 2020, 60.9% in 2025)
    08531   members of every OTHER grant-receiving community, by county and five groups
            (Buddhism, Islam, Christianity, other religion, life stance), 2010-2020. The series
            ENDS IN 2020: from 2021 the communities no longer report members' home municipality.
    06326   the same, national only, ten groups, to 2026

**That is a `roll` (spec §3.1) and this map does not draw it for Norway.** Finland (§9by) and
Sweden (§9cz) are drawn from self-identification with the register printed beside it, and Norway
on a roll would put a 30-point step in the Lutheran share at both borders that is a step between
two questions, which is the US/Canada problem §3.5a was built to remove. SSB's own
living-conditions survey found 47% of adults saying they belong to a religion in 2020, the year
the roll put 80.6% on one. build() prints the roll beside the survey, at the same 11 counties.

SO THE TWO HALVES ARE GREECE'S (§9z), FINLAND'S (§9by), BELGIUM'S (§9cy) AND SWEDEN'S (§9cz):

    Norwegian citizens   4.79M   ESS rounds 5-11 pooled, `ctzcntr = Yes`
    foreign residents    0.60M   Eurostat `cens_21ctz_r3` x Pew's origin compositions

THE VARIABLE IS `rlgdnno`, AND IT EXISTS IN ALL SEVEN ROUNDS. `rlgdnano` and `rlgdnbno` do not
(E201VariableNotFound). Belgium's lesson was to read both cards: `rlgdnno` is `rlgdnm` with one
answer split, `Protestant` into `Den norske kirke` and `Andre protestantiske trossamfunn`, which is
the distinction that matters most in Norway. `_check_card` proves the nesting from the fetched
cross-tab on every build.

THE LEVEL IS NUTS 2 IN EVERY ROUND, IN TWO VINTAGES THAT DO NOT NEST IN EACH OTHER.

    rounds 5-9     NUTS 2016   7 regions   NO01 Oslo og Akershus, NO02 Innlandet, NO03 Sor-
                                           Ostlandet, NO04 Agder og Rogaland, NO05 Vestlandet,
                                           NO06 Trondelag, NO07 Nord-Norge
    rounds 10-11   NUTS 2021   6 regions   NO02, NO06, NO07, NO08 Oslo og Viken, NO09 Agder og
                                           Sor-Ostlandet, NO0A Vestlandet

NO08 takes Ostfold and Buskerud out of NO03 and Akershus out of NO01; NO09 takes Agder out of
NO04; NO0A takes Rogaland out of NO04. The only geography both vintages are unions of is FOUR
units: Innlandet, Trondelag, Nord-Norge, and everything else (`SOR`). So Sweden's two-level
construction (spec §12, nested units) runs over:

    fine     the 7 NUTS 2016 regions, rounds 5-9
    coarse   the 4 shared units, all seven rounds
    residual the national rate inside each county's residual

and a category takes the finest level at which it passes both of Sweden's tests.

COUNTED AT THE 11 COUNTIES OF 2020-2023, because that is where the census counts citizenship and
so where the foreign half is measured. Ten of the eleven sit inside one NUTS 2016 region. **Viken
does not**: it is Akershus (NO01), Ostfold and Buskerud (NO03), and Jevnaker and Lunner from
Oppland (NO02), so its citizen composition is those regions' shares blended by their 1 January
2019 populations, `VIKEN_2019`. Its placement inside the county is by kommune population, so
Baerum and Halden draw the same blend.

WHAT THE TEST SELECTS, AND TWO CALLS ON TOP OF IT.

    7 regions        No religion (kept as the residual), Den norske kirke, the free churches
    4 shared units   No religion only; four units give a rank test almost no resolution
    OVERRIDE         Islam, drawn at the 7 regions: rank p = 0.0555, chi-square 1.3e-10, and
                     SSB's own roll orders the same 7 regions nearly the same way at +0.929
    residual         Catholic, Orthodox, other Christian, Eastern, other non-Christian, Jewish

**The level is the 2020-2023 rounds, the pattern is 2010-2018's** (RESCALE_TO_LATE, spec §3.4).
The Church of Norway lost twelve points of self-identified citizens between the two pools, which
is Norway's fastest-moving fact and too large to leave a 2014 midpoint on a map beside a 2021
census half. Both the roll and the survey say the decline is proportional rather than a uniform
shift, so it is one factor per category.
"""

import argparse
import io
import json
import os
import ssl
import sys
import urllib.parse
import urllib.request
import zipfile

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "taxonomy"))
sys.path.insert(0, HERE)
RAW = os.path.join(ROOT, "data", "raw", "no")
OUT = os.path.join(ROOT, "data", "normalized", "no.csv")
OUT_FOREIGN = os.path.join(ROOT, "data", "normalized", "no_foreign.csv")

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) religiondots"}

# THE STATISTIC IS IMPORTED, NOT COPIED, which is tools/ess_split_half.py's rule: the halvings,
# the median and the null are sources/stability.py's, and alpha, draw count and seed are
# sources/be.py's, so Norway cannot quietly run a different test from the one Belgium and Sweden
# draw with.
import be  # noqa: E402
import stability  # noqa: E402

# --- ESS --------------------------------------------------------------------------------
ESS_API = "https://api.nsd.no/graphql"
ESS_FINE_ROUNDS = {
    5: ("0189b86b-8aa4-4be3-88ad-39c58b02f19f", 89),
    6: ("450fa78e-68ab-493f-b169-dbc7ab8ffec2", 85),
    7: ("9c96a1b2-b027-43c1-8c74-e883f892d0bb", 91),
    8: ("ffc43f48-e15a-4a1c-8813-47eda377c355", 98),
    9: ("b2b0bf39-176b-4eca-8d26-3c05ea83d2cb", 280),
}
ESS_LATE_ROUNDS = {
    10: ("f37d014a-6958-42d4-b03b-17c29e481d3d", 286),
    11: ("242aaa39-3bbb-40f5-98bf-bfb1ce53d8ef", 179),
}
ESS_ROUNDS = {**ESS_FINE_ROUNDS, **ESS_LATE_ROUNDS}

ESS_BREAK = ["region", "ctzcntr", "rlgblg", "rlgdnno"]
CARD_BREAK = ["rlgdnno", "rlgdnm"]

_TAB = """query($id:ID!,$v:Int!,$bv:[String!]!){analysis{
 frequencyTabulationByVariables(input:{
   datafile:{id:$id,version:$v}, breakVariables:$bv, byVariables:["cntry"],
   instance:PUBLISHED, agencyId:INT_ESSERIC, includeMissing:true,%s
   metadataLanguage:"en"}){
 responses{by{value} response{
   variableValues{name values codeList{value label isMissing}} table{path count}}}}}}"""
ESS_TAB_W = _TAB % ' weightVariable:"pspwght",'
ESS_TAB_N = _TAB % ""

# --- Eurostat and Pew ---------------------------------------------------------------------
EU_DATA = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/"
EU_CTZ = "cens_21ctz_r3"
PEW_ZIP = ("https://www.pewresearch.org/wp-content/uploads/sites/20/2025/06/"
           "Religious-Composition-2010-2020-dataset.zip")

# --- SSB, the roll, which is printed and not drawn -----------------------------------------
SSB_API = "https://data.ssb.no/api/v0/no/table/"
SSB_YEAR = 2020          # the last year 08531 has a county dimension

# --- the geography ------------------------------------------------------------------------
# The counting unit: NUTS 3 2021, the 11 counties of 2020-2023. Norwegian names are Latin
# script already; kept with their letters because these are data values.
NUTS3 = {
    "NO081": "Oslo", "NO082": "Viken", "NO020": "Innlandet",
    "NO091": "Vestfold og Telemark", "NO092": "Agder", "NO0A1": "Rogaland",
    "NO0A2": "Vestland", "NO0A3": "Møre og Romsdal", "NO060": "Trøndelag",
    "NO071": "Nordland", "NO074": "Troms og Finnmark",
}
# SSB's county number for the same 11, for the roll tables.
SSB_FYLKE = {
    "03": "NO081", "30": "NO082", "34": "NO020", "38": "NO091", "42": "NO092", "11": "NO0A1",
    "46": "NO0A2", "15": "NO0A3", "50": "NO060", "18": "NO071", "54": "NO074",
}
assert set(SSB_FYLKE.values()) == set(NUTS3)

# ESS rounds 5-9.
NUTS2016 = {
    "NO01": "Oslo og Akershus", "NO02": "Innlandet (Hedmark og Oppland)",
    "NO03": "Sør-Østlandet", "NO04": "Agder og Rogaland", "NO05": "Vestlandet",
    "NO06": "Trøndelag", "NO07": "Nord-Norge",
}
# ESS rounds 10-11.
NUTS2021_2 = {
    "NO02": "Innlandet", "NO06": "Trøndelag", "NO07": "Nord-Norge",
    "NO08": "Oslo og Viken", "NO09": "Agder og Sør-Østlandet", "NO0A": "Vestlandet",
}
MISSING_REGION = "99999"    # round 8 carries one respondent as `Not available`

# The coarsest geography both vintages are unions of. Jevnaker and Lunner (15,897 people) moved
# from Oppland to Viken in 2020, so NO02 is 4.3% larger in rounds 5-9 than in 10-11; that is the
# one place the two definitions of a shared unit differ, and it is small enough to name rather
# than model.
COMMON = {"NO02": "Innlandet", "NO06": "Trøndelag", "NO07": "Nord-Norge",
          "SOR": "the rest of Norway"}
TO_COMMON_2016 = {"NO01": "SOR", "NO02": "NO02", "NO03": "SOR", "NO04": "SOR", "NO05": "SOR",
                  "NO06": "NO06", "NO07": "NO07"}
TO_COMMON_2021 = {"NO02": "NO02", "NO06": "NO06", "NO07": "NO07", "NO08": "SOR", "NO09": "SOR",
                  "NO0A": "SOR"}

# VIKEN, the one county that straddles NUTS 2016 regions. SSB table 07459, population at 1
# January 2019, the last date the old counties existed: Akershus 624,055 (NO01); Ostfold 297,520
# + Buskerud 283,148 + Svelvik 6,685, which went from Vestfold into Drammen (NO03); Jevnaker 6,846
# + Lunner 9,051 from Oppland (NO02). Asker's Royken and Hurum were Buskerud kommuner and are
# inside Buskerud's figure. Total population rather than citizens, because nothing publishes
# citizens at old-county level; the blend is of shares, so this only matters if Akershus and
# Ostfold differ in their foreign share by a lot, and the census puts all of Viken at 12.2%.
VIKEN_2019 = {"NO01": 624_055, "NO03": 297_520 + 283_148 + 6_685, "NO02": 6_846 + 9_051}
UNIT_REGIONS = {
    "NO081": {"NO01": 1.0}, "NO082": VIKEN_2019, "NO020": {"NO02": 1.0},
    "NO091": {"NO03": 1.0}, "NO092": {"NO04": 1.0}, "NO0A1": {"NO04": 1.0},
    "NO0A2": {"NO05": 1.0}, "NO0A3": {"NO05": 1.0}, "NO060": {"NO06": 1.0},
    "NO071": {"NO07": 1.0}, "NO074": {"NO07": 1.0},
}
assert set(UNIT_REGIONS) == set(NUTS3)


def _weights(level):
    """{county: {region: weight summing to 1}} at the fine or the coarse level."""
    out = {}
    for u, parts in UNIT_REGIONS.items():
        acc = {}
        for r, w in parts.items():
            k = r if level == "fine" else TO_COMMON_2016[r]
            acc[k] = acc.get(k, 0.0) + float(w)
        tot = sum(acc.values())
        out[u] = {k: v / tot for k, v in acc.items()}
    return out


# --- the categories -----------------------------------------------------------------------
NO_RELIGION = "No religion"          # rlgblg = No
REFUSAL = "__refused__"

# The nesting `_check_card` asserts: every rlgdnno answer sits inside exactly this rlgdnm answer.
CARD_NEST = {
    "Den norske kirke": "Protestant",
    "Andre protestantiske trossamfunn (f.eks. frikirker, anglikanske kirke, pinsevenner og andre)":
        "Protestant",
    "Katolsk kirke": "Roman Catholic",
    "Ortodoks kirke (gresk, russisk, andre)": "Eastern Orthodox",
    "Andre kristne trossamfunn (f.eks. Jehovas vitner, mormonerne)":
        "Other Christian denomination",
    "Det mosaiske trossamfunn (jødisk)": "Jewish",
    "Islam (muslimsk)": "Islam",
    "Østlige religioner (f.eks. buddhisme, hinduisme, sikh, shintoisme, taoisme, konfutsianisme)":
        "Eastern religions",
    "Andre ikke-kristne religioner": "Other Non-Christian religions",
}

COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count",
           "basis", "year", "source_id", "note"]

# Norway's own totals, asserted so a re-fetch against a new ESS release or census vintage fails
# here rather than quietly redrawing the map.
POP_2021 = 5_391_370           # cens_21ctz_r3 TOTAL = NAT + FOR + STLS 1,700 + UNK 30
# Unweighted, ctzcntr = Yes, region present, refusals dropped. note_public quotes these, so they
# are asserted rather than printed.
N_CITIZENS_FINE = 7_045        # rounds 5-9, the 7 NUTS 2016 regions
N_CITIZENS_ALL = 9_611         # all seven rounds

# `No religion` stays the residual at every level, spec §12 (nested units): it is the big
# category that closes the partition, and fixed at a region's share the small tail has to absorb
# every county's own departure in the fine categories. _compose prints the check.
KEEP_AS_RESIDUAL = {NO_RELIGION}

ISLAM = "Islam (muslimsk)"
FREE_CHURCHES = ("Andre protestantiske trossamfunn (f.eks. frikirker, anglikanske kirke, "
                 "pinsevenner og andre)")
LUTHERAN = "Den norske kirke"

# What the test selects at the 7 regions, asserted, so a re-fetch that changes a verdict stops the
# build rather than quietly redrawing the country.
EXPECT_FINE_PASS = {NO_RELIGION, LUTHERAN, FREE_CHURCHES}

# ONE CATEGORY IS DRAWN AGAINST THE RANK TEST, AND IT IS SAID HERE (sources/gt.py's convention).
OVERRIDE = {
    ISLAM: (
        "rank test p = 0.0555 at the 7 NUTS 2016 regions, a hair over 0.05, with a spatial "
        "chi-square of 1.3e-10, so the regions plainly differ and seven units give the ordering "
        "little power. An independent instrument settles it: SSB's 2019 roll of members of "
        "grant-receiving Muslim communities, summed from the old counties to the same 7 regions, "
        "orders them nearly the same way at Spearman +0.929 (p = 0.003), Oslo og Akershus top on "
        "both, with two neighbouring pairs swapped (Innlandet and Agder og Rogaland, Vestlandet "
        "and Trøndelag). Drawn at the national rate instead, "
        "Oslo came out 4.4% Muslim against 9.6% on the roll. sources/no.md §6."),
}

# §3.4, STRUCTURE FROM THE DETAILED POOL AND TOTALS FROM THE RECENT ONE. Rounds 10-11 (2020-2023)
# cannot be placed at the 7 regions, and between them and rounds 5-9 (2010-2018) the Church of
# Norway fell from 44.14% to 32.24% of citizens and No religion rose from 48.74% to 60.01%. So a
# category drawn at the 7 regions keeps rounds 5-9's regional pattern and is scaled, by one factor
# per category, to its rounds 10-11 national share, weighted by the census's citizen population
# per county (not by the survey pool's own unit mix, spec §12's Nigeria entry). ONE FACTOR, NOT A
# SHIFT, because both witnesses say the decline is proportional: the roll's county ratios
# 2025/2020 have a coefficient of variation of 0.018 against 0.077 for the point differences,
# and the survey's own four shared units fall by ratios of 0.69 to 0.75 against drops of 9.9 to
# 15.6 points. The residual categories share what is left at the all-seven-round proportions,
# because two rounds are too thin for the smallest answers. sources/no.md §7.
RESCALE_TO_LATE = True


def _key(s):
    return " ".join(str(s).split())


# =======================================================================================
# fetch
# =======================================================================================

def _ess(query, variables):
    body = json.dumps({"query": query, "variables": variables}).encode()
    req = urllib.request.Request(ESS_API, data=body, headers={
        "Content-Type": "application/json", **UA})
    r = json.load(urllib.request.urlopen(req, timeout=900))
    if "errors" in r:
        sys.exit(f"!! ESS API: {r['errors'][0].get('message')} "
                 f"{r['errors'][0].get('extensions', {}).get('code', '')}")
    return r["data"]


def _save(obj, dest):
    tmp = dest + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False)
    os.replace(tmp, dest)


def _ess_fetch(fid, ver, bv, query, dest, rnd):
    if os.path.exists(dest):
        return
    d = _ess(query, {"id": fid, "v": ver, "bv": bv})
    hit = [x for x in d["analysis"]["frequencyTabulationByVariables"]["responses"]
           if x["by"][0]["value"] == "NO"]
    if not hit:
        sys.exit(f"!! ESS round {rnd} has no NO response")
    _save(hit[0]["response"], dest)


def _eurostat(dataset, **params):
    """Eurostat needs certifi's CURRENT bundle — sources/gr.py records why."""
    import certifi
    ctx = ssl.create_default_context(cafile=certifi.where())
    u = EU_DATA + dataset + "?" + urllib.parse.urlencode(params, doseq=True)
    return json.load(urllib.request.urlopen(
        urllib.request.Request(u, headers=UA), timeout=900, context=ctx))


def _ssb(table, query):
    body = json.dumps({"query": query, "response": {"format": "json-stat2"}}).encode()
    req = urllib.request.Request(SSB_API + table, data=body, headers={
        "Content-Type": "application/json", **UA})
    return json.load(urllib.request.urlopen(req, timeout=300))


def fetch():
    os.makedirs(RAW, exist_ok=True)

    print("ESS…")
    for rnd, (fid, ver) in sorted(ESS_ROUNDS.items()):
        for tag, q in (("w", ESS_TAB_W), ("n", ESS_TAB_N)):
            _ess_fetch(fid, ver, ESS_BREAK, q, os.path.join(RAW, f"ess_r{rnd}_{tag}.json"), rnd)
        _ess_fetch(fid, ver, CARD_BREAK, ESS_TAB_N, os.path.join(RAW, f"ess_r{rnd}_card.json"),
                   rnd)
        n = sum(c["count"] for c in json.load(
            open(os.path.join(RAW, f"ess_r{rnd}_n.json"), encoding="utf-8"))["table"])
        print(f"  round {rnd}: {n:,.0f} respondents")

    print("Eurostat census…")
    dest = os.path.join(RAW, "cens_21ctz_r3_no.json")
    if not os.path.exists(dest):
        d = _eurostat(EU_CTZ, format="JSON", lang="EN", age="TOTAL", sex="T",
                      geo=sorted(NUTS3) + ["NO"])
        _save(d, dest)
        print(f"  {os.path.getsize(dest):,} bytes")
    else:
        print("  already on disk")

    print("Pew…")
    dest = os.path.join(RAW, "pew.zip")
    if not os.path.exists(dest):
        with urllib.request.urlopen(urllib.request.Request(PEW_ZIP, headers=UA),
                                    timeout=600) as r, open(dest + ".tmp", "wb") as f:
            f.write(r.read())
        os.replace(dest + ".tmp", dest)
        print(f"  {os.path.getsize(dest):,} bytes")
    else:
        print("  already on disk")

    print("SSB rolls (for the comparison, not for the map)…")
    fylker = sorted(SSB_FYLKE)
    jobs = {
        "ssb_08531.json": ("08531", [
            {"code": "Region", "selection": {"filter": "item", "values": ["0"] + fylker}},
            {"code": "ReligionLivs", "selection": {"filter": "item",
                                                   "values": ["999", "200", "400", "600",
                                                              "902", "900"]}},
            {"code": "Tid", "selection": {"filter": "item", "values": [str(SSB_YEAR)]}}]),
        "ssb_12025.json": ("12025", [
            {"code": "KOKkommuneregion0000", "selection": {
                "filter": "item", "values": ["EAK"] + [f"EKA{f}" for f in fylker]}},
            {"code": "ContentsCode", "selection": {
                "filter": "item", "values": ["KOSmedldnkinnb0000", "KOSmedltrolinnb0000"]}},
            {"code": "Tid", "selection": {"filter": "item", "values": [str(SSB_YEAR), "2025"]}}]),
        "ssb_07459.json": ("07459", [
            {"code": "Region", "selection": {"filter": "item", "values": fylker}},
            {"code": "Tid", "selection": {"filter": "item", "values": [str(SSB_YEAR)]}}]),
    }
    for name, (table, query) in jobs.items():
        dest = os.path.join(RAW, name)
        if os.path.exists(dest):
            print(f"  {table}: already on disk")
            continue
        _save(_ssb(table, query), dest)
        print(f"  {table}: {os.path.getsize(dest):,} bytes")


# =======================================================================================
# reading
# =======================================================================================

def _ess_table(rnd, tag):
    """One saved ESS response as a tidy frame: region CODE, everything else as a LABEL.

    `path` IS A LIST OF INDICES INTO `codeList`, NOT CODE VALUES, fi.py's trap. Norway's
    `rlgdnno` codes run 1-9 and then 6666/7777/9999, so reading them as codes would drop every
    missing value and shift nothing visibly, which is the silent version.
    """
    d = json.load(open(os.path.join(RAW, f"ess_r{rnd}_{tag}.json"), encoding="utf-8"))
    order = [v["name"] for v in d["variableValues"]]
    codes = {v["name"]: v["codeList"] for v in d["variableValues"]}
    rows = []
    for cell in d["table"]:
        rec = {"round": rnd, "count": float(cell["count"])}
        for i, n in enumerate(order):
            c = codes[n][cell["path"][i]]
            rec[n] = c["value"] if n == "region" else _key(c["label"])
            rec[n + "_miss"] = bool(c["isMissing"])
        rows.append(rec)
    return pd.DataFrame(rows)


def _pool(rounds, tag):
    """Citizens with a region, category resolved (refusals kept as REFUSAL)."""
    df = pd.concat([_ess_table(r, tag) for r in sorted(rounds)], ignore_index=True)
    df = df[(df["ctzcntr"] == "Yes") & (~df["region_miss"])
            & (df["region"] != MISSING_REGION)].copy()
    df["cat"] = np.where(
        df["rlgblg"] == "No", NO_RELIGION,
        np.where((df["rlgblg"] != "Yes") | df["rlgdnno_miss"], REFUSAL, df["rlgdnno"]))
    return df


def _check_level():
    """The region code set of every round, asserted. Italy's lesson (§9as): a round that changes
    level or vintage does not error, it returns different regions and pooling them averages."""
    for rnd in sorted(ESS_ROUNDS):
        d = _ess_table(rnd, "n")
        codes = set(d.loc[~d["region_miss"] & (d["region"] != MISSING_REGION), "region"])
        want = NUTS2016 if rnd in ESS_FINE_ROUNDS else NUTS2021_2
        if codes != set(want):
            sys.exit(f"!! round {rnd}: regions {sorted(codes)}, expected {sorted(want)}")
        print(f"  round {rnd}: {len(codes)} regions, NUTS "
              f"{'2016' if rnd in ESS_FINE_ROUNDS else '2021'}")


def _check_card():
    """`rlgdnno` nests inside `rlgdnm` answer for answer, proved from the data every build.

    Belgium's `_check_be_card` is the model: if ESS ever changes the Norwegian card, the build
    stops rather than drawing a Protestant cell under a Lutheran name.
    """
    seen = set()
    for rnd in sorted(ESS_ROUNDS):
        d = _ess_table(rnd, "card")
        d = d[d["count"] > 0]
        for _, r in d.iterrows():
            a, b = r["rlgdnno"], r["rlgdnm"]
            if r["rlgdnno_miss"] or r["rlgdnm_miss"]:
                if r["rlgdnno_miss"] != r["rlgdnm_miss"]:
                    sys.exit(f"!! round {rnd}: `{a}` / `{b}` missing on one card only "
                             f"({r['count']:.0f} respondents)")
                continue
            if CARD_NEST.get(a) != b:
                sys.exit(f"!! round {rnd}: rlgdnno `{a}` sits in rlgdnm `{b}` for "
                         f"{r['count']:.0f} respondents; CARD_NEST says `{CARD_NEST.get(a)}`")
            seen.add(a)
    missing = sorted(set(CARD_NEST) - seen)
    if missing:
        sys.exit(f"!! rlgdnno answers no round produced: {missing}")
    print(f"  rlgdnno nests in rlgdnm in all {len(ESS_ROUNDS)} rounds; "
          f"{len(seen)} substantive answers")


# =======================================================================================
# the test
# =======================================================================================

def _stability(raw, cats, rounds, units, label):
    """WHICH CATEGORIES CARRY THEIR OWN GEOGRAPHY at `units`: §9cy's test plus §9cz's gate.

    The median Spearman over every round split against a permutation of the unit labels drawn
    PER ROUND (§9cy's bug is a global relabelling, which is the identity), and a spatial
    chi-square of the category against the rest of the base over the same units (§9cz: a
    mostly-zero column can pass the rank test). Both at be.STAB_ALPHA, unweighted counts.
    """
    rounds, units = sorted(rounds), sorted(units)
    stray = sorted(set(raw["region"]) - set(units))
    if stray:
        sys.exit(f"!! {label}: units outside the tested geography: {stray}")
    ri = {r: i for i, r in enumerate(rounds)}
    ui = {u: i for i, u in enumerate(units)}
    ci = {c: i for i, c in enumerate(cats)}
    cube = np.zeros((len(rounds), len(units), len(cats)))
    for (r, u, c), v in raw.groupby(["round", "region", "cat"])["count"].sum().items():
        cube[ri[r], ui[u], ci[c]] += v
    # A unit with no respondent in a round makes `stability.median_rho` skip halvings, and the
    # per-round relabelling skips different ones in the null (spec §12, "A ROUND THAT SKIPS A
    # UNIT"). Refused here as `cab.stability` refuses it (added 2026-09-14): pass the units
    # sampled in every round, as `ua.py::TEST_UNITS` does, or drop the round.
    empty = [(rounds[a], units[b]) for a, b in zip(*np.where(cube.sum(axis=2) == 0))]
    if empty:
        sys.exit(f"!! {label}: (round, unit) cells with no respondent: {empty}. Test on the units "
                 "sampled in every round (ua.py::EXPECT_ABSENT, TEST_UNITS) or leave the round out")

    splits = stability.halvings(len(rounds))
    obs = stability.median_rho(cube, splits)
    null = stability.wave_null(cube, splits, be.STAB_PERM, be.STAB_SEED)

    pooled = cube.sum(axis=0)
    per_unit = pooled.sum(axis=1)
    k = len(splits[0][0])
    print(f"\n  split-half at the {label}: rounds {rounds}, {len(splits)} splits of {k} against "
          f"{len(rounds) - k}, {be.STAB_PERM}-draw per-round permutation null, plus the chi-square")
    print(f"    respondents per unit: " + ", ".join(
        f"{u} {per_unit[i]:,.0f}" for i, u in enumerate(units)))
    print(f"    {'category':<52}{'n':>7}{'median rho':>12}{'null 95th':>11}{'p':>8}"
          f"{'chi2 p':>10}  verdict")
    carries = []
    for j, c in enumerate(cats):
        n = int(round(pooled[:, j].sum()))
        chi = stability.chi2_p(pooled[:, j], per_unit)
        p, q95 = stability.permutation_p(obs[j], null[:, j], be.STAB_ALPHA)
        if not np.isfinite(p):
            print(f"    {c[:50]:<52}{n:>7,}{'':>12}{'':>11}{'':>8}{chi:10.2e}  no test possible")
            continue
        ok = p < be.STAB_ALPHA and np.isfinite(chi) and chi < be.STAB_ALPHA
        if ok:
            carries.append(c)
            verdict = "own geography"
        elif p < be.STAB_ALPHA:
            verdict = "REFUSED: passes the rank test, but the units do not differ"
        else:
            verdict = "not distinguishable from chance"
        print(f"    {c[:50]:<52}{n:>7,}{obs[j]:+12.3f}{q95:+11.3f}"
              f"{p:8.4f}{chi:10.2e}  {verdict}")
    return carries


# =======================================================================================
# the citizen half
# =======================================================================================

def _assert_source(pool, label):
    import no2024
    unknown = sorted(set(pool["cat"]) - no2024.SOURCE)
    if unknown:
        sys.exit(f"!! {label}: source categories with no mapping: {unknown}")
    vanished = sorted(no2024.SOURCE - set(pool["cat"]))
    if vanished:
        sys.exit(f"!! {label}: no2024.SOURCE categories nobody answered: {vanished} — the "
                 "denomination axis moved under the parse")


def _citizen_shares():
    # ---- fine: 7 NUTS 2016 regions, rounds 5-9
    raw = _pool(ESS_FINE_ROUNDS, "n")
    wtd = _pool(ESS_FINE_ROUNDS, "w")
    answered = 1.0 - wtd.loc[wtd["cat"] == REFUSAL, "count"].sum() / wtd["count"].sum()
    print(f"  {raw['count'].sum():,.0f} citizens with a region in rounds 5-9; "
          f"{100 * (1 - answered):.2f}% declined (weighted) and are not drawn")
    raw, wtd = raw[raw["cat"] != REFUSAL], wtd[wtd["cat"] != REFUSAL]
    n_fine = int(round(raw["count"].sum()))
    if N_CITIZENS_FINE is None:
        print(f"  !! N_CITIZENS_FINE is unset; this build has {n_fine:,}")
    elif n_fine != N_CITIZENS_FINE:
        sys.exit(f"!! {n_fine:,} answered citizens in rounds 5-9, expected {N_CITIZENS_FINE:,}")
    _assert_source(wtd, "rounds 5-9")

    tab7 = wtd.groupby(["region", "cat"])["count"].sum().unstack(fill_value=0.0)
    tab7 = tab7.reindex(index=sorted(NUTS2016), fill_value=0.0)
    share7 = tab7.div(tab7.sum(axis=1), axis=0)
    nat = tab7.sum() / tab7.sum().sum()
    cats = sorted(nat.index, key=lambda c: -nat[c])
    share7 = share7.reindex(columns=cats, fill_value=0.0)
    fine_pass = _stability(raw, cats, ESS_FINE_ROUNDS, NUTS2016, "7 NUTS 2016 regions")

    # ---- coarse: the 4 shared units, all seven rounds
    def to_common(df):
        df = df.copy()
        m = np.where(df["round"].isin(list(ESS_FINE_ROUNDS)),
                     df["region"].map(TO_COMMON_2016), df["region"].map(TO_COMMON_2021))
        df["region"] = m
        if df["region"].isna().any():
            sys.exit("!! a region with no shared-unit mapping")
        return df

    raw4 = to_common(_pool(ESS_ROUNDS, "n"))
    wtd4 = to_common(_pool(ESS_ROUNDS, "w"))
    raw4, wtd4 = raw4[raw4["cat"] != REFUSAL], wtd4[wtd4["cat"] != REFUSAL]
    n_all = int(round(raw4["count"].sum()))
    if N_CITIZENS_ALL is None:
        print(f"  !! N_CITIZENS_ALL is unset; this build has {n_all:,}")
    elif n_all != N_CITIZENS_ALL:
        sys.exit(f"!! {n_all:,} answered citizens over seven rounds, expected {N_CITIZENS_ALL:,}")
    _assert_source(wtd4, "rounds 5-11")
    tab4 = wtd4.groupby(["region", "cat"])["count"].sum().unstack(fill_value=0.0)
    tab4 = tab4.reindex(index=sorted(COMMON), columns=cats, fill_value=0.0)
    share4 = tab4.div(tab4.sum(axis=1), axis=0)
    coarse_pass = _stability(raw4, cats, ESS_ROUNDS, COMMON, "4 units both NUTS vintages share")

    if set(fine_pass) != EXPECT_FINE_PASS:
        sys.exit(f"!! the 7-region test now selects {sorted(fine_pass)}, not "
                 f"{sorted(EXPECT_FINE_PASS)}; a round has been reissued or the test changed, and "
                 "OVERRIDE, note_public and sources/no.md all quote the old verdicts")
    for c, why in OVERRIDE.items():
        if c in fine_pass:
            sys.exit(f"!! `{c}` now passes on its own; take it out of OVERRIDE")
        print(f"\n  OVERRIDE, drawn at the 7 regions against the rank test: `{c}`\n    {why}")
    fine = [c for c in cats if (c in fine_pass or c in OVERRIDE) and c not in KEEP_AS_RESIDUAL]
    coarse = [c for c in cats if c in coarse_pass and c not in fine and c not in KEEP_AS_RESIDUAL]
    small = [c for c in cats if c not in fine and c not in coarse]
    print(f"\n  {len(fine)} categories at the 7 regions, {len(coarse)} at the 4 shared units, "
          f"{len(small)} at the national rate inside each county's residual")
    nraw7 = raw.groupby("cat")["count"].sum()
    for c in cats:
        tag = ("7 regions" if c in fine else "4 shared units" if c in coarse
               else "national rate (residual)")
        print(f"    {100 * nat[c]:>6.2f}%  {int(nraw7.get(c, 0)):>6,}  {c}   [{tag}]")

    # ---- rounds 10 and 11 nationally, which is the level the fine categories are scaled to
    late = _pool(ESS_LATE_ROUNDS, "w")
    late = late[late["cat"] != REFUSAL]
    ln = late.groupby("cat")["count"].sum()
    ln = ln / ln.sum()
    nat_all = wtd4.groupby("cat")["count"].sum()
    nat_all = nat_all / nat_all.sum()
    print(f"\n  ROUNDS 10-11 (2020-2023) AGAINST ROUNDS 5-9 (2010-2018), national, weighted:")
    for c in cats:
        print(f"    {c[:50]:<52} 5-9 {100 * nat[c]:6.2f}%   10-11 {100 * ln.get(c, 0.0):6.2f}%   "
              f"{100 * (ln.get(c, 0.0) - nat[c]):+6.2f}")
    counts = {"fine": raw.groupby(["region", "cat"])["count"].sum(),
              "coarse": raw4.groupby(["region", "cat"])["count"].sum()}
    return share7, share4, nat, fine, coarse, small, answered, counts, n_fine, n_all, ln, nat_all


def _compose(share7, share4, nat, fine, coarse, small, t_late, nat_all, cit_pop):
    """Per-county composition as a closed partition.

    A fine category takes its NUTS 2016 region's share (a blend of regions for Viken); a coarse
    one its shared unit's share; both are then scaled to their rounds 10-11 national level
    (RESCALE_TO_LATE, §3.4). What is left of the county is divided among the rest at their
    all-seven-round national proportions, §9bi's residual, so the tail still moves with each
    county's measured shares.
    """
    w7, w4 = _weights("fine"), _weights("coarse")
    out = pd.DataFrame(index=sorted(NUTS3), columns=list(nat.index), dtype=float)
    for u in out.index:
        for c in fine:
            out.loc[u, c] = sum(w * float(share7.loc[r, c]) for r, w in w7[u].items())
        for c in coarse:
            out.loc[u, c] = sum(w * float(share4.loc[k, c]) for k, w in w4[u].items())
    fixed = fine + coarse
    w_cit = cit_pop.reindex(out.index).astype(float)
    w_cit = w_cit / w_cit.sum()
    residual = 1.0 - out[fixed].sum(axis=1) if fixed else pd.Series(1.0, index=out.index)
    for c in sorted(KEEP_AS_RESIDUAL & set(small)):
        for lvl, sh, w in (("7 regions", share7, w7), ("4 shared units", share4, w4)):
            own = pd.Series({u: sum(x * float(sh.loc[r, c]) for r, x in w[u].items())
                             for u in out.index})
            alt = residual - own
            print(f"  if `{c}` were fixed at its {lvl} share (before rescaling), the tail left "
                  f"would be negative in {int((alt < 0).sum())} of {len(alt)} counties (worst "
                  f"{alt.min():+.2%}, {NUTS3[alt.idxmin()]})")
    if RESCALE_TO_LATE:
        print("  scaling the fixed categories to their rounds 10-11 national share (§3.4), "
              "weighted by citizens per county:")
        for c in fixed:
            old = float((out[c] * w_cit).sum())
            f = float(t_late.get(c, 0.0)) / old
            out[c] = out[c] * f
            print(f"    {c[:50]:<52} {100 * old:6.2f}% -> {100 * float(t_late.get(c, 0.0)):6.2f}%  "
                  f"x{f:.4f}   ({100 * out[c].min():.2f}% to {100 * out[c].max():.2f}% by county)")
        residual = 1.0 - out[fixed].sum(axis=1) if fixed else pd.Series(1.0, index=out.index)
    if (residual <= 0).any():
        sys.exit(f"!! counties with no room for the tail: {sorted(residual[residual <= 0].index)}")
    small_total = float(nat_all[small].sum())
    print(f"  the tail is {residual.min():.2%} of {NUTS3[residual.idxmin()]} and "
          f"{residual.max():.2%} of {NUTS3[residual.idxmax()]}")
    for c in small:
        out[c] = residual * (float(nat_all[c]) / small_total)
    err = (out.sum(axis=1) - 1.0).abs().max()
    if err > 1e-9:
        sys.exit(f"!! composition does not sum to 1, worst {err:.2e}")
    drawn_nat = (out.mul(w_cit, axis=0)).sum()
    print("  national citizen composition as drawn, against rounds 10-11 and all seven rounds:")
    for c in drawn_nat.sort_values(ascending=False).index:
        print(f"    {c[:50]:<52} drawn {100 * drawn_nat[c]:6.2f}%   10-11 "
              f"{100 * float(t_late.get(c, 0.0)):6.2f}%   5-11 {100 * float(nat_all.get(c, 0.0)):6.2f}%")
    return out


# =======================================================================================
# the foreign half
# =======================================================================================

def _census():
    d = json.load(open(os.path.join(RAW, "cens_21ctz_r3_no.json"), encoding="utf-8"))
    dims = d["id"]
    cats = [list(d["dimension"][x]["category"]["index"]) for x in dims]
    sizes = d["size"]
    rows = []
    for k, v in d["value"].items():
        i, idx = int(k), []
        for s in reversed(sizes):
            idx.append(i % s)
            i //= s
        idx = list(reversed(idx))
        rows.append([cats[j][idx[j]] for j in range(len(dims))] + [v])
    df = pd.DataFrame(rows, columns=dims + ["value"])
    # Every NUTS level is a row, so only the 11 leaves are read.
    df = df[df["geo"].isin(NUTS3)].copy()
    df["unit"] = df["geo"]
    return df


def _foreign_half(cen):
    """[geo_id, node, count] for foreign citizens, counted at NUTS 3 (se.py's, other.no)."""
    import origin_religion as origin

    with zipfile.ZipFile(os.path.join(RAW, "pew.zip")) as z:
        name = [n for n in z.namelist() if n.endswith("(percentages).csv")][0]
        pew = pd.read_csv(io.BytesIO(z.read(name)))
    pew = pew[(pew["Year"] == 2020) & (pew["Level"] == 1)].set_index("Country")
    fams = origin.FAMILIES

    leaf = cen[cen["citizen"].str.fullmatch(r"[A-Z]{2}") & (cen["citizen"] != "NO")]
    total_for = cen[cen["citizen"] == "FOR"]["value"].sum()
    covered = leaf["value"].sum()
    print(f"  {leaf['citizen'].nunique()} named citizenships cover {covered:,.0f} of "
          f"{total_for:,.0f} foreign residents ({100 * covered / total_for:.2f}%)")
    # EUROSTAT'S `FOR` CAN HOLD THE COUNTRY'S OWN RECOGNISED NON-CITIZENS (Latvia's 190,544
    # `RNC`), and scaling the named citizenships up to `FOR` would hand them to the named
    # countries (spec §12, "EUROSTAT'S `FOR`"). Norway's `RNC` is 0 (2026-09-14); asserted, with
    # `lv.py::_foreign_half`'s coverage band.
    rnc = cen[cen["citizen"] == "RNC"]["value"].sum()
    if rnc:
        sys.exit(f"!! cens_21ctz_r3 has {rnc:,.0f} recognised non-citizens (RNC) inside FOR; scale "
                 "to FOR - RNC and draw RNC separately, as lv.py::_foreign_half does")
    if not 0.9 < covered / total_for <= 1.0001:
        sys.exit("!! the named citizenships do not match FOR; read the census file before scaling")

    comp, unmapped = {}, []
    for iso in sorted(leaf["citizen"].unique()):
        pn = origin.PEW_BY_ISO.get(iso, "MISSING")
        if pn == "MISSING":
            unmapped.append(iso)
            continue
        row = origin.REGIONAL.get(iso) if pn is None else (
            {f: float(pew.loc[pn, f]) for f in fams} if pn in pew.index else None)
        if row is None:
            unmapped.append(f"{iso} ({pn})")
            continue
        comp[iso] = origin.composition(iso, row, "other.no")
    if unmapped:
        sys.exit(f"!! {len(unmapped)} citizenships have no composition: {unmapped}")

    rows = []
    for unit, g in leaf.groupby("unit"):
        base = g["value"].sum()
        if base <= 0:
            continue
        target = cen[(cen["unit"] == unit) & (cen["citizen"] == "FOR")]["value"].sum()
        scale = target / base
        acc = {}
        for iso, v in g.groupby("citizen")["value"].sum().items():
            if v <= 0:
                continue
            for node, share in comp[iso].items():
                acc[node] = acc.get(node, 0.0) + v * scale * share
        for node, c in acc.items():
            rows.append((unit, node, c))
    df = pd.DataFrame(rows, columns=["geo_id", "node", "count"])
    print(f"  foreign half: {df['count'].sum():,.0f} people, {df['node'].nunique()} nodes")
    return df


# =======================================================================================
# the roll, printed beside the survey
# =======================================================================================

def _jsonstat(name):
    d = json.load(open(os.path.join(RAW, name), encoding="utf-8"))
    ids, sizes = d["id"], d["size"]
    order = {x: sorted(d["dimension"][x]["category"]["index"],
                       key=lambda k: d["dimension"][x]["category"]["index"][k]) for x in ids}
    rows = []
    for i, v in enumerate(d["value"]):
        j, pos = i, []
        for s in reversed(sizes):
            pos.append(j % s)
            j //= s
        pos = list(reversed(pos))
        rows.append({x: order[x][pos[k]] for k, x in enumerate(ids)} | {"value": v})
    return pd.DataFrame(rows)


def _roll():
    """{county: {...}} from 12025, 08531 and 07459, all at SSB_YEAR."""
    pop = _jsonstat("ssb_07459.json").groupby("Region")["value"].sum()
    k = _jsonstat("ssb_12025.json")
    o = _jsonstat("ssb_08531.json")
    y = str(SSB_YEAR)
    out = {}
    for f, u in SSB_FYLKE.items():
        dnk = k[(k["KOKkommuneregion0000"] == f"EKA{f}") & (k["ContentsCode"] == "KOSmedldnkinnb0000")
                & (k["Tid"] == y)]["value"]
        g = o[o["Region"] == f].set_index("ReligionLivs")["value"]
        out[u] = {"pop": float(pop[f]), "dnk_pct": float(dnk.iloc[0]),
                  "islam": float(g["400"]), "christian": float(g["600"]),
                  "buddhism": float(g["200"]), "other": float(g["902"]),
                  "lifestance": float(g["900"]), "outside": float(g["999"])}
    nat = k[(k["KOKkommuneregion0000"] == "EAK") & (k["ContentsCode"] == "KOSmedldnkinnb0000")]
    nat = {t: float(v) for t, v in zip(nat["Tid"], nat["value"])}
    return out, nat


# =======================================================================================
# build
# =======================================================================================

def build():
    import no2024

    print("checking the region set of every round…")
    _check_level()
    print("checking the Norwegian card against the harmonised one…")
    _check_card()

    print("\nESS…")
    (share7, share4, nat, fine, coarse, small, answered, counts, n_fine, n_all,
     t_late, nat_all) = _citizen_shares()

    print("\nEurostat census…")
    cen = _census()
    pop = cen[cen["citizen"] == "TOTAL"].groupby("unit")["value"].sum()
    nat_cit = cen[cen["citizen"] == "NAT"].groupby("unit")["value"].sum()
    for_ = cen[cen["citizen"] == "FOR"].groupby("unit")["value"].sum()
    print(f"  {pop.sum():,.0f} people = {nat_cit.sum():,.0f} Norwegian citizens "
          f"+ {for_.sum():,.0f} foreign, over {len(pop)} counties")
    if sorted(pop.index) != sorted(NUTS3):
        sys.exit(f"!! census counties {sorted(pop.index)}")
    if int(pop.sum()) != POP_2021:
        sys.exit(f"!! census total {pop.sum():,.0f} is not Norway's {POP_2021:,}")

    print("\ncomposing the citizen half…")
    comp = _compose(share7, share4, nat, fine, coarse, small, t_late, nat_all, nat_cit)
    split = no2024.ORTHODOX_SPLIT
    if abs(sum(split.values()) - 1.0) > 1e-9:
        sys.exit(f"!! the Orthodox split does not sum to 1: {sum(split.values())}")
    print(f"  splitting `{no2024.ORTHODOX_ANSWER}` ({100 * nat_all[no2024.ORTHODOX_ANSWER]:.2f}% of "
          f"citizens) on the {no2024.MEMBERS_YEAR} grant-counted members:")
    for k, v in split.items():
        print(f"    {100 * v:5.2f}%  {k}  -> {no2024.MAP[k]}")

    w7, w4 = _weights("fine"), _weights("coarse")
    rows = []
    for u in sorted(NUTS3):
        n_cit = float(nat_cit[u]) * answered
        for cat, s in comp.loc[u].items():
            if s <= 0:
                continue
            late_note = ("; national level from rounds 10-11 (2020-2023), §3.4"
                         if RESCALE_TO_LATE else "")
            if cat in fine:
                parts = ", ".join(f"{r} {100 * w:.1f}%" for r, w in w7[u].items())
                note = (f"ESS rounds 5-9 at NUTS 2016 ({parts}); §12 nested units{late_note}"
                        + ("; drawn by OVERRIDE, see sources/no.py" if cat in OVERRIDE else ""))
            elif cat in coarse:
                parts = ", ".join(f"{k} {100 * w:.1f}%" for k, w in w4[u].items())
                note = (f"ESS rounds 5-11 at the 4 units both NUTS vintages share ({parts})"
                        f"{late_note}")
            else:
                note = ("share of the county's residual at the rounds 5-11 national proportions "
                        "(§9bi, split-half not passed)")
            if cat == no2024.ORTHODOX_ANSWER:
                for sub, w in split.items():
                    rows.append((u, NUTS3[u], sub, s * w * n_cit,
                                 note + f"; communion split on {no2024.MEMBERS_YEAR} grant-counted "
                                        "members (§3.11)"))
            else:
                rows.append((u, NUTS3[u], cat, s * n_cit, note))
    cit = pd.DataFrame(rows, columns=["geo_id", "geo_name", "source_category", "count", "note"])
    cit["geo_level"] = "nuts3"
    cit["basis"] = "self_id"
    cit["year"] = 2021
    cit["source_id"] = "ess_r5_r9_nuts2016_x_r5_r11_shared4"
    cit = cit[COLUMNS]
    print(f"  citizen half: {cit['count'].sum():,.0f} people "
          f"({100 * cit['count'].sum() / nat_cit.sum():.2f}% of citizens; the rest declined)")
    unknown = sorted(set(cit["source_category"]) - set(no2024.MAP))
    if unknown:
        sys.exit(f"!! unmapped source categories: {unknown}")

    print("\nforeign half…")
    ext = _foreign_half(cen)
    ext["geo_level"] = "nuts3"
    ext["geo_name"] = ext["geo_id"].map(NUTS3)
    ext["tier"] = "modelled"
    ext["basis"] = "nationality_derived"
    ext["year"] = 2021
    ext["source_id"] = "cens_21ctz_r3_x_pew2020"
    ext = ext[["geo_id", "geo_level", "geo_name", "node", "count", "tier", "basis", "year",
               "source_id"]]

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    cit.to_csv(OUT + ".tmp", index=False)
    os.replace(OUT + ".tmp", OUT)
    ext.to_csv(OUT_FOREIGN + ".tmp", index=False)
    os.replace(OUT_FOREIGN + ".tmp", OUT_FOREIGN)

    drawn = cit["count"].sum() + ext["count"].sum()
    print(f"\nwrote {OUT}  ({len(cit):,} rows, {cit['source_category'].nunique()} source categories)")
    print(f"wrote {OUT_FOREIGN}  ({len(ext):,} rows, {ext['node'].nunique()} nodes)")
    print(f"drawn {drawn:,.0f} of {pop.sum():,.0f} — {100 * drawn / pop.sum():.2f}%")

    both = pd.concat([
        cit.assign(node=cit["source_category"].map(no2024.resolve))[["geo_id", "node", "count"]],
        ext[["geo_id", "node", "count"]]], ignore_index=True)
    top = both.groupby("node")["count"].sum().sort_values(ascending=False)
    print("\nnational totals:")
    for node, c in top.head(22).items():
        print(f"  {c:>11,.0f}  {100 * c / drawn:5.2f}%  {node}")

    byu = both.groupby(["geo_id", "node"])["count"].sum().unstack(fill_value=0.0)
    byu_tot = byu.sum(axis=1)
    fam = lambda pre: byu[[c for c in byu.columns if c == pre or c.startswith(pre + ".")]].sum(axis=1)
    print("\nper county (drawn, both halves):")
    print(f"    {'county':<22}{'lutheran':>10}{'unaffil':>9}{'islam':>8}{'catholic':>9}"
          f"{'orth+or':>9}{'protest':>9}{'foreign':>9}")
    for u in sorted(NUTS3, key=lambda x: NUTS3[x]):
        t = byu_tot[u]
        orth = (byu.loc[u].get("christianity.orthodox", 0) + byu.loc[u].get("christianity.oriental", 0)
                + sum(byu.loc[u][c] for c in byu.columns if c.startswith("christianity.orthodox.")
                      or c.startswith("christianity.oriental.")))
        print(f"    {NUTS3[u]:<22}{100 * byu.loc[u].get('christianity.lutheran', 0) / t:9.2f}%"
              f"{100 * byu.loc[u].get('unaffiliated', 0) / t:8.2f}%{100 * fam('islam')[u] / t:7.2f}%"
              f"{100 * fam('christianity.catholic')[u] / t:8.2f}%{100 * orth / t:8.2f}%"
              f"{100 * byu.loc[u].get('christianity.protestant', 0) / t:8.2f}%"
              f"{100 * for_[u] / pop[u]:8.2f}%")

    # ---- THE ROLL BESIDE THE SURVEY. Not a validation: the roll counts members, whom the state
    # pays for, and the survey asks what people consider themselves. sources/no.md reads it.
    from scipy.stats import spearmanr
    roll, rnat = _roll()
    for u, r in roll.items():
        if abs(r["pop"] / pop[u] - 1.0) > 0.02:
            sys.exit(f"!! {u}: SSB's {SSB_YEAR} population {r['pop']:,.0f} against the census's "
                     f"{pop[u]:,.0f} — the county code table is joined to the wrong unit")
    rp = pd.DataFrame(roll).T
    print(f"\nTHE ROLL ({SSB_YEAR}) BESIDE THE SURVEY, at the same 11 counties:")
    print(f"  Church of Norway members and affiliated, national: "
          + ", ".join(f"{t} {v:.1f}%" for t, v in sorted(rnat.items())))
    out_pct = 100 * rp["outside"].sum() / rp["pop"].sum()
    dnk_n = (rp["dnk_pct"] / 100 * rp["pop"]).sum()
    print(f"  members of another grant-receiving community: {100 * rp['outside'].sum() / rp['pop'].sum():.2f}%; "
          f"on no roll: {100 - 100 * dnk_n / rp['pop'].sum() - out_pct:.2f}%")
    lut = byu.get("christianity.lutheran", pd.Series(0.0, index=byu.index))
    print(f"  on this map: christianity.lutheran {100 * lut.sum() / drawn:.2f}%, "
          f"unaffiliated {100 * byu.get('unaffiliated', pd.Series(0.0)).sum() / drawn:.2f}%")
    survey_dnk = comp["Den norske kirke"] if "Den norske kirke" in comp.columns else None
    cmp = pd.DataFrame({
        "roll_dnk": rp["dnk_pct"],
        "survey_dnk_cit": 100 * survey_dnk.reindex(rp.index),
        "roll_islam": 100 * rp["islam"] / rp["pop"],
        "map_islam": 100 * fam("islam").reindex(rp.index) / byu_tot.reindex(rp.index),
        "roll_chr": 100 * rp["christian"] / rp["pop"],
        "map_chr": 100 * (byu.sum(axis=1) * 0 + sum(
            byu[c] for c in byu.columns
            if c.startswith("christianity") and c != "christianity.lutheran")).reindex(rp.index)
            / byu_tot.reindex(rp.index),
        "roll_life": 100 * rp["lifestance"] / rp["pop"],
    }).sort_values("roll_dnk", ascending=False)
    print(f"    {'county':<22}{'roll DNK':>9}{'survey':>8}{'roll Isl':>9}{'map Isl':>8}"
          f"{'roll oChr':>10}{'map oChr':>9}{'roll life':>10}")
    for u, r in cmp.iterrows():
        print(f"    {NUTS3[u]:<22}{r['roll_dnk']:8.1f}%{r['survey_dnk_cit']:7.2f}%"
              f"{r['roll_islam']:8.2f}%{r['map_islam']:7.2f}%{r['roll_chr']:9.2f}%"
              f"{r['map_chr']:8.2f}%{r['roll_life']:9.2f}%")
    for a, b, what in (("roll_dnk", "survey_dnk_cit", "Church of Norway (roll) vs its survey share"),
                       ("roll_islam", "map_islam", "Islam (roll) vs Islam on this map"),
                       ("roll_chr", "map_chr", "other Christian (roll) vs this map")):
        sp = spearmanr(cmp[a].values, cmp[b].values)
        print(f"  Spearman {what}: {sp.correlation:+.3f} (p={sp.pvalue:.3f}) over {len(cmp)} counties")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    a = ap.parse_args()
    if a.fetch:
        fetch()
    else:
        build()
