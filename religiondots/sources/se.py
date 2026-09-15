"""Sweden — a register that counts one church exactly, and a survey that asks everybody.

Writes data/normalized/se.csv (citizens) and se_foreign.csv (foreign residents).

Usage:
    python sources/se.py --fetch     # ESS tabulations, Eurostat census, Pew, the church  (~6 MB)
    python sources/se.py             # rebuild from data/raw/se/

THE OFFICE'S CUSTOM-TABLE SHELF WAS CHECKED FIRST (§9cu's instruction) AND THE ANSWER IS
INTERESTING. SCB has no browsable archive of past commissioned tabulations the way CBS has
`maatwerk`: ordering statistics there is bespoke and paid, and a site search for `trossamfund`
returns civil-society accounts and occupational medians. But SCB does hold a religion variable
and does tabulate it on commission, and **the customer publishes the result**: the Church of
Sweden's `Medlemsutveckling 2020-2021, per forsamling, kommun och lan samt riket` says in its
own header that the population, membership and deceased-member figures are *"framtagna av SCB
pa uppdrag av Svenska kyrkan"*. That is 5,627,932 members at 31 December 2021, exact, for
every parish, all 290 kommuner and all 21 lan. §11k's line *"Sweden. SCB carries nothing"* is
true of SCB's own catalogue and false of what SCB produces.

**It is one denomination, so it cannot BE the source, and it is used here as the witness.**
build() prints the comparison and sources/se.md §4 reads it.

SO THE TWO HALVES ARE GREECE'S (§9z), FRANCE'S (§9ab), ITALY'S (§9as) AND FINLAND'S (§9by):

    Swedish citizens     9.57M   ESS rounds 5-9 and 11 pooled, `ctzcntr = Yes`
    foreign residents    0.86M   Eurostat `cens_21ctz_r3` x Pew's origin compositions

BOTH COME OUT OF THE SAME CENSUS TABLE, so they partition by construction: `cens_21ctz_r3`
publishes `NAT` and `FOR` beside its named citizenships.

THE VARIABLE IS `rlgdnase`. `rlgdnse` DOES NOT EXIST IN ANY ROUND and raises
E201VariableNotFound, which is Finland's `rlgdnafi` trap again and the loud version of §11ai's
warning about `a`-suffixed revisions.

TWO LEVELS, SPLIT BY CATEGORY, WHICH IS ITALY'S CONSTRUCTION (sources.md §9as). Rebuilt so
on 2026-09-14, on Anita's approval; the first build drew everything at the 21 lan.
Sweden is in ESS rounds 1-9 and 11 and is ABSENT FROM ROUND 10. Rounds 1-4 have no `region`
variable, the wall Greece, France, Italy and Finland all hit. That leaves six, and `regunit`
splits them:

    rounds 5, 6, 7, 8   NUTS level 3   21 lan          6,449 citizens
    rounds 9, 11        NUTS level 2    8 riksomraden  2,657 citizens

Every lan sits inside one riksomrade, so rounds 5-8 can be cut at either level and all six
rounds at the coarser one. **A category takes the finest level at which it passes both tests
below**, and `_compose` applies a riksomrade's share inside each of its lan, the way it.py
applies a ripartizione's shares inside each regione:

                             21 lan, rounds 5-8     8 riksomraden, all six rounds
    Svenska kyrkan              own                    passes too
    Islam                       own                    passes too
    Annan protestantisk         own                    fails
    Katolska kyrkan             fails                  own
    Ortodoxa kyrkan             fails                  own
    Judisk                      fails                  own, on 10 respondents
    No religion                 fails                  passes, KEPT AS THE RESIDUAL
    the other three             fail                   fail

The free churches are why this is not one whole build at NUTS 2: that level puts Jonkoping
(6.25%) in `Smaland med oarna` with Kalmar (0.69%), and on rounds 5-8 the signal reverses to
-0.119. `No religion` is 68% of citizens and the complement of the lan categories, so fixing
it at the riksomrade would leave the small tail negative in 6 lan; `KEEP_AS_RESIDUAL` says so
and _compose prints the check. The riksomrade pool takes all six rounds because the only
reason rounds 9 and 11 were ever left out is that they have no NUTS 3.

**The Jewish pass is the thinnest on the map.** Seven of its ten respondents are in Stockholm,
which is what the chi-square (0.011, and 0.013 on a Monte Carlo version) is detecting. The rest
of the ordering is two respondents in Smaland, one in Sydsverige and none in Vastsverige, so
Kalmar and Gotland draw more Jewish citizens per head than Gothenburg. Applied as written;
adding `Judisk` to KEEP_AS_RESIDUAL reverses it.

**One halving is a draw from a statistic, not the statistic.** An earlier version of this
docstring drew the level conclusion from one chronological halving; four rounds admit three,
and `Svenska kyrkan` at 21 lan scores +0.125, +0.434 and +0.458 across them against a bar of
+0.3701. Spec §12 carries it.

WHICH CATEGORIES CARRY THEIR OWN GEOGRAPHY: §9cy's TEST, WITH A SECOND REQUIREMENT.
`_stability` is be.py's round-split permutation test, unchanged, because two countries on one
instrument must not use two tests. **Sweden adds a spatial chi-square at 0.05, which can only
make a category fail.** At 21 lan `Annan icke-kristen religion` (21 respondents) and
`Osterlandsk religion` (23) both CLEAR the permutation test, at p = 0.018 and 0.022, with
chi-squares of 0.32 and 0.40: a rank correlation over a column that is zero in most units is
decided by how the ties break, so a small category can be PASSED by one.

The failing categories are still DRAWN, at the national rate inside each lan's residual, so
the partition stays closed. `No religion` is 95% of that residual, so it still moves with each
lan's measured shares and runs from 50.79% of Kronoberg to 76.43% of Gavleborg.
"""

import argparse
import io
import json
import os
import re
import ssl
import sys
import urllib.parse
import urllib.request
import zipfile

import numpy as np
import pandas as pd

import stability

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "taxonomy"))
sys.path.insert(0, HERE)
RAW = os.path.join(ROOT, "data", "raw", "se")
OUT = os.path.join(ROOT, "data", "normalized", "se.csv")
OUT_FOREIGN = os.path.join(ROOT, "data", "normalized", "se_foreign.csv")

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) religiondots"}

# --- ESS --------------------------------------------------------------------------------
ESS_API = "https://api.nsd.no/graphql"

# (datafile id, version), the same ids sources/fr.py, it.py and fi.py use.
ESS_NUTS3_ROUNDS = {
    5: ("0189b86b-8aa4-4be3-88ad-39c58b02f19f", 89),
    6: ("450fa78e-68ab-493f-b169-dbc7ab8ffec2", 85),
    7: ("9c96a1b2-b027-43c1-8c74-e883f892d0bb", 91),
    8: ("ffc43f48-e15a-4a1c-8813-47eda377c355", 98),
}
# NUTS 2 only. Pooled with rounds 5-8 (aggregated to NUTS 2) for the categories drawn at the
# riksomrade, and for nothing else; see `_citizen_shares`.
ESS_NUTS2_ROUNDS = {
    9: ("b2b0bf39-176b-4eca-8d26-3c05ea83d2cb", 280),
    11: ("242aaa39-3bbb-40f5-98bf-bfb1ce53d8ef", 179),
}
ESS_ROUNDS = {**ESS_NUTS3_ROUNDS, **ESS_NUTS2_ROUNDS}

ESS_BREAK = ["region", "ctzcntr", "rlgblg", "rlgdnase"]

# `%s` is the weight clause. pspwght is normalised to the sample size, so both passes total
# the same n — but the WEIGHTED per-cell values are fractional and are not respondent counts,
# so the unweighted pass is what the n floors and the chi-squares are read off (de_ess.py).
_TAB = """query($id:ID!,$v:Int!,$bv:[String!]!){analysis{
 frequencyTabulationByVariables(input:{
   datafile:{id:$id,version:$v}, breakVariables:$bv, byVariables:["cntry"],
   instance:PUBLISHED, agencyId:INT_ESSERIC, includeMissing:true,%s
   metadataLanguage:"en"}){
 responses{by{value} response{
   variableValues{name values codeList{value label isMissing}} table{path count}}}}}}"""
ESS_TAB_W = _TAB % ' weightVariable:"pspwght",'
ESS_TAB_N = _TAB % ""

# --- Eurostat ---------------------------------------------------------------------------
EU_DATA = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/"
EU_CTZ = "cens_21ctz_r3"        # population by citizenship, age, NUTS 3 — census 2021

PEW_ZIP = ("https://www.pewresearch.org/wp-content/uploads/sites/20/2025/06/"
           "Religious-Composition-2010-2020-dataset.zip")

# --- the register, which is not a source here and is fetched anyway -----------------------
# The Church of Sweden's own membership publication, produced for it by SCB. It draws nothing:
# this build does not use one number from it. It is pulled because the claim that Sweden
# measures its largest church twice and gets 54% and 23% should be reproducible from the repo
# rather than quoted from a note. build() prints the comparison at the end.
#
# THE URL HAS A LITERAL `(1)` IN IT and `?id=` after the extension; both are part of the path
# the site serves and neither is a mistake. There are two editions on that host,
# `MedlemsutvecklingLKF.pdf` (2018-2019) and this one (2020-2021), and the newer one is the
# last at this geography.
CHURCH_PDF = ("https://www.svenskakyrkan.se/filer/1374643/MedlemsutvecklingLKF(1).pdf"
              "?id=2398412")
CHURCH_YEAR = 2021

# --- the geography ------------------------------------------------------------------------
# NUTS 2021, the 21 lan, which is what data/geo/se/se_lau.gpkg rolls up to. Swedish is already
# Latin script; the names are the GISCO workbook's own spellings with the diacritics kept,
# because these are data values rather than prose.
NUTS3 = {
    "SE110": "Stockholms län", "SE121": "Uppsala län", "SE122": "Södermanlands län",
    "SE123": "Östergötlands län", "SE124": "Örebro län", "SE125": "Västmanlands län",
    "SE211": "Jönköpings län", "SE212": "Kronobergs län", "SE213": "Kalmar län",
    "SE214": "Gotlands län", "SE221": "Blekinge län", "SE224": "Skåne län",
    "SE231": "Hallands län", "SE232": "Västra Götalands län", "SE311": "Värmlands län",
    "SE312": "Dalarnas län", "SE313": "Gävleborgs län", "SE321": "Västernorrlands län",
    "SE322": "Jämtlands län", "SE331": "Västerbottens län", "SE332": "Norrbottens län",
}
# The 8 riksomraden. Every lan is inside exactly one and its NUTS 3 code starts with it.
NUTS2 = {
    "SE11": "Stockholm", "SE12": "Östra Mellansverige", "SE21": "Småland med öarna",
    "SE22": "Sydsverige", "SE23": "Västsverige", "SE31": "Norra Mellansverige",
    "SE32": "Mellersta Norrland", "SE33": "Övre Norrland",
}
assert {u[:4] for u in NUTS3} == set(NUTS2)

# The Church of Sweden's PDF spells two lan differently from ESS and from GISCO — `Dalarna
# län` for Dalarnas and `Kalmars län` for Kalmar — so the join is by this explicit table and
# not by name. [[reference_name_join_wrong_neighbour]]: a name join picks the wrong twin
# silently, and a 21-row genitive difference is exactly the shape that slips through. The
# table is then CHECKED AGAINST POPULATION rather than trusted, which is a check that does not
# depend on names at all: see `_register`.
CHURCH_LAN = {
    "Stockholms län": "SE110", "Uppsala län": "SE121", "Södermanlands län": "SE122",
    "Östergötlands län": "SE123", "Örebro län": "SE124", "Västmanlands län": "SE125",
    "Jönköpings län": "SE211", "Kronobergs län": "SE212", "Kalmars län": "SE213",
    "Gotlands län": "SE214", "Blekinge län": "SE221", "Skåne län": "SE224",
    "Hallands län": "SE231", "Västra Götalands län": "SE232", "Värmlands län": "SE311",
    "Dalarna län": "SE312", "Gävleborgs län": "SE313", "Västernorrlands län": "SE321",
    "Jämtlands län": "SE322", "Västerbottens län": "SE331", "Norrbottens län": "SE332",
}

# --- the categories -----------------------------------------------------------------------
NO_RELIGION = "No religion"          # rlgblg = No; the source has no atheist/agnostic split
# rlgblg answers that are neither Yes nor No, and rlgdnase answers from a respondent who said
# Yes and then declined the denomination. spec §3.5: refusals are marked, not filled.
REFUSAL = "__refused__"

# NO SIZE FLOOR. `lapop.ELIGIBLE_FLOOR`'s 1% gate was here and is gone: `_stability`'s
# chi-square requirement refuses the same categories on evidence rather than on size, and a
# floor would ALSO have refused the free churches, which are 2.0% of citizens on 130
# respondents and are the strongest geographic result in the country. Size is eligibility, the
# chi-square is evidence, and only the second is about whether there is a geography there.

# The split-half's null, and these are be.py's values unchanged — §9cy built this test for
# ESS the same day and two countries on one instrument must not use two tests.
STAB_ALPHA = 0.05
STAB_PERM = 2000
STAB_SEED = 0

COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count",
           "basis", "year", "source_id", "note"]

# Sweden's own totals, asserted so a re-fetch against a new ESS release or census vintage
# fails here rather than quietly redrawing the map.
POP_2021 = 10_452_325          # cens_21ctz_r3 TOTAL
N_ROUNDS = 4
N_UNITS = 21
# Unweighted, ctzcntr = Yes, region present, refusals dropped, rounds 5-8. This is the figure
# note_public quotes as "people interviewed", so it is asserted rather than printed.
N_CITIZENS = 6_449
# The same over all six rounds at the riksomrade (rounds 5-8 aggregated, plus 9 and 11).
N_CITIZENS_ALL = 9_106

# NO RELIGION PASSES AT THE RIKSOMRADE AND STAYS THE RESIDUAL, on arithmetic. It is 68% of
# citizens and the complement of the lan-level categories, so fixing it at its riksomrade's
# share leaves the small tail to absorb each lan's own departure in Svenska kyrkan, Islam and
# the free churches, and the tail goes negative in 6 of 21 lan (Kronoberg -5.27%). _compose
# prints that check on every build. As the residual it still varies by lan.
# A category added here goes back into the residual even if it passes; `Judisk` is the one
# to add if its 10-respondent pass is judged too thin (the docstring has its cost).
KEEP_AS_RESIDUAL = {NO_RELIGION}
CHURCH_MEMBERS = 5_627_932     # 31/12/2021, the PDF's own lan rows summed


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


def _eurostat(dataset, **params):
    """Eurostat needs certifi's CURRENT bundle — sources/gr.py records why."""
    import certifi
    ctx = ssl.create_default_context(cafile=certifi.where())
    u = EU_DATA + dataset + "?" + urllib.parse.urlencode(params, doseq=True)
    return json.load(urllib.request.urlopen(
        urllib.request.Request(u, headers=UA), timeout=900, context=ctx))


def fetch():
    os.makedirs(RAW, exist_ok=True)

    print("ESS…")
    for rnd, (fid, ver) in sorted(ESS_ROUNDS.items()):
        for tag, q in (("w", ESS_TAB_W), ("n", ESS_TAB_N)):
            dest = os.path.join(RAW, f"ess_r{rnd}_{tag}.json")
            if os.path.exists(dest):
                continue
            d = _ess(q, {"id": fid, "v": ver, "bv": ESS_BREAK})
            se = [x for x in d["analysis"]["frequencyTabulationByVariables"]["responses"]
                  if x["by"][0]["value"] == "SE"]
            if not se:
                sys.exit(f"!! ESS round {rnd} has no SE response")
            json.dump(se[0]["response"], open(dest, "w", encoding="utf-8"),
                      ensure_ascii=False)
        n = sum(c["count"] for c in json.load(
            open(os.path.join(RAW, f"ess_r{rnd}_n.json"), encoding="utf-8"))["table"])
        print(f"  round {rnd}: {n:,.0f} respondents")

    print("Eurostat census…")
    dest = os.path.join(RAW, "cens_21ctz_r3_se.json")
    if not os.path.exists(dest):
        d = _eurostat(EU_CTZ, format="JSON", lang="EN", age="TOTAL", sex="T")
        json.dump(d, open(dest, "w", encoding="utf-8"))
        print(f"  {os.path.getsize(dest):,} bytes")
    else:
        print("  already on disk")

    print("Pew…")
    dest = os.path.join(RAW, "pew.zip")
    if not os.path.exists(dest):
        with urllib.request.urlopen(urllib.request.Request(PEW_ZIP, headers=UA),
                                    timeout=600) as r, open(dest, "wb") as f:
            f.write(r.read())
        print(f"  {os.path.getsize(dest):,} bytes")
    else:
        print("  already on disk")

    print("Church of Sweden membership, per lan (for the comparison, not for the map)…")
    dest = os.path.join(RAW, "medlemsutveckling_lkf.pdf")
    if not os.path.exists(dest):
        with urllib.request.urlopen(urllib.request.Request(CHURCH_PDF, headers=UA),
                                    timeout=600) as r, open(dest, "wb") as f:
            f.write(r.read())
        # [[reference_pdf_truncated_at_source]]: Content-Length can match a damaged file, so
        # the trailer is what says the download is whole.
        with open(dest, "rb") as f:
            f.seek(-64, os.SEEK_END)
            if b"%%EOF" not in f.read():
                sys.exit(f"!! {dest} has no %%EOF trailer — truncated at source")
        print(f"  {os.path.getsize(dest):,} bytes")
    else:
        print("  already on disk")


# =======================================================================================
# the citizen half
# =======================================================================================

def _ess_table(rnd, tag):
    """One round as a tidy frame: region CODE, everything else as a LABEL.

    `path` IS A LIST OF INDICES INTO `values`, NOT A LIST OF CODE VALUES — fi.py's trap,
    which for Finland silently emptied every denomination above code 9. Sweden's nine codes
    are all valid indices too, so the same misreading here would shift the whole card by one
    and put Catholics on `Svenska kyrkan` with no error anywhere.
    """
    d = json.load(open(os.path.join(RAW, f"ess_r{rnd}_{tag}.json"), encoding="utf-8"))
    vv = {v["name"]: v for v in d["variableValues"]}
    order = [v["name"] for v in d["variableValues"]]
    labels = {n: [c["label"] for c in vv[n]["codeList"]] for n in order}
    missing = {n: [c["isMissing"] for c in vv[n]["codeList"]] for n in order}
    values = {n: [c["value"] for c in vv[n]["codeList"]] for n in order}
    rows = []
    for cell in d["table"]:
        rec = {"round": rnd}
        for i, n in enumerate(order):
            j = cell["path"][i]
            rec[n] = values[n][j] if n == "region" else labels[n][j]
            rec[n + "_miss"] = missing[n][j]
        rec["count"] = cell["count"]
        rows.append(rec)
    return pd.DataFrame(rows)


def _pool(rounds, tag):
    """Citizens with a region, refusals dropped, category resolved."""
    df = pd.concat([_ess_table(r, tag) for r in rounds], ignore_index=True)
    df = df[(df["ctzcntr"] == "Yes") & (~df["region_miss"])].copy()
    df["cat"] = np.where(
        df["rlgblg"] == "No", NO_RELIGION,
        np.where((df["rlgblg"] != "Yes") | df["rlgdnase_miss"], REFUSAL, df["rlgdnase"]))
    return df


def _check_level():
    """`regunit` says which NUTS level each round is, and it is asserted rather than inferred.

    Italy's lesson (sources.md §9as): a round that drops a level does not error, it just returns fewer,
    bigger regions, and pooling the two silently averages a fine country into a coarse one.
    """
    for rnd in sorted(ESS_ROUNDS):
        d = _ess_table(rnd, "n")
        codes = sorted(set(d.loc[~d["region_miss"], "region"]))
        want3 = rnd in ESS_NUTS3_ROUNDS
        n = len(codes)
        if want3 and (n != N_UNITS or set(codes) != set(NUTS3)):
            sys.exit(f"!! round {rnd} should be the {N_UNITS} lan and has {n}: {codes[:6]}")
        if not want3 and set(codes) != set(NUTS2):
            sys.exit(f"!! round {rnd} should be the 8 riksomraden and has {n}: {codes[:8]}")
        print(f"  round {rnd}: {n} regions, NUTS {'3' if want3 else '2'}")


def _stability(raw, cats, nraw, rounds, units, label):
    """WHICH CATEGORIES CARRY THEIR OWN LAN GEOGRAPHY — §14.16, split by ROUND, be.py's test.

    **TWO REQUIREMENTS, AND THE SECOND ONE IS SWEDEN'S ADDITION.** A category carries its own
    lan shares only if it passes BOTH the round-split permutation test AND a spatial
    chi-square at 0.05 over the same 21 lan. §9bi states the second one and states it in the
    direction of overrides: *"Before proposing one, run the chi-square: if the units do not
    differ, there is nothing to draw."* The logic is symmetric and Sweden is where the other
    direction bites. `Annan icke-kristen religion` (21 respondents) and `Osterlandsk religion`
    (23) both CLEAR the permutation test, at p = 0.018 and 0.022, with chi-squares of 0.32 and
    0.40 — the survey cannot tell the 21 lan apart for either of them at all. The mechanism is
    that a rank correlation over a column that is zero in most units is decided by how the
    ties break, so a small category does not merely lose power, it can be PASSED by chance;
    ten tests at alpha 0.05 expect half a false pass and these are two candidates sitting
    right where one would look for them.

    **This adds a requirement and can only make a category fail**, which is the safe direction
    and is not the forbidden move of moving a bar until something passes. It also replaces
    `lapop.ELIGIBLE_FLOOR`'s 1% size gate, which would have refused the same two categories
    for a worse reason: size is eligibility and the chi-square is evidence, and only the
    second one is about whether there is a geography there.

    **This replaced a single chronological halving against `spearman_null`'s fixed bar, and
    Sweden is the country that shows why one halving is not a statistic.** Four rounds admit
    three distinct halvings, and on `Svenska kyrkan` at 21 lan they give +0.125, +0.434 and
    +0.458 against a bar of +0.3701 — one verdict of "no" and two of "yes" from the same data,
    decided by which two rounds you happened to put together. §9cy's construction takes the
    MEDIAN over every split instead, which is a statistic rather than a draw, and compares it
    to a permutation null of the lan labels rather than to a bar computed for an untied
    Spearman that these mostly-zero columns do not have.

    Sweden's fine pool is four rounds, so the splits are 2-against-2 and there are three of
    them; Belgium's is seven rounds and 35. Three is few, and it is the whole reason the
    median is taken rather than one of them being picked.

    It runs twice: at the 21 lan on rounds 5-8, and at the 8 riksomraden on all six rounds,
    where the splits are 3-against-3 and there are ten.

    A failure is a failure to demonstrate signal and not a demonstration of noise. The people
    stay drawn, at the national rate inside each lan's residual; what is withdrawn is the
    claim to know which lan they are in.
    """
    rounds = sorted(rounds)
    units = sorted(units)
    cube = np.zeros((len(rounds), len(units), len(cats)), dtype=float)
    for ri, rnd in enumerate(rounds):
        t = (raw[raw["round"] == rnd].groupby(["region", "cat"])["count"].sum()
             .unstack(fill_value=0.0))
        for ui, u in enumerate(units):
            for cj, c in enumerate(cats):
                if u in t.index and c in t.columns:
                    cube[ri, ui, cj] = t.loc[u, c]
    # A unit with no respondent in a round makes `stability.median_rho` skip halvings, and the
    # per-round relabelling skips different ones in the null (spec §12, "A ROUND THAT SKIPS A
    # UNIT"). Refused, as `no._stability` and `cab.stability` refuse it (added 2026-09-14).
    empty = [(rounds[a], units[b]) for a, b in zip(*np.where(cube.sum(axis=2) == 0))]
    if empty:
        sys.exit(f"!! {label}: (round, unit) cells with no respondent: {empty}. Test on the units "
                 "sampled in every round (ua.py::TEST_UNITS) or leave the round out")

    # Every distinct halving, each counted once. This used to keep only combinations holding
    # round 0, which is right for Sweden's even pools (4 and 6 rounds) and wrong for an odd one
    # (spec §12, Colombia); `stability.halvings` keeps both right.
    splits = stability.halvings(len(rounds))
    obs = stability.median_rho(cube, splits)
    null = stability.wave_null(cube, splits, STAB_PERM, STAB_SEED)

    print(f"\n  split-half stability at the {label}, {len(rounds)} rounds, median of "
          f"{len(splits)} splits, against a {STAB_PERM}-draw unit-label "
          f"permutation null (§14.16, §9cy), plus the spatial chi-square:")
    print(f"    {'category':<52}{'n':>7}{'national':>10}{'median rho':>12}"
          f"{'null 95th':>11}{'p':>8}{'chi2 p':>10}  verdict")
    carries = []
    tot = raw["count"].sum()
    for j, c in enumerate(cats):
        n = int(raw[raw["cat"] == c]["count"].sum())
        chi = stability.chi2_p(nraw[c].values, nraw.sum(axis=1).values)
        p, q95 = stability.permutation_p(obs[j], null[:, j], STAB_ALPHA)
        if not np.isfinite(p):
            print(f"    {c[:50]:<52}{n:>7,}{100 * n / tot:9.3f}%{'':>12}{'':>11}{'':>8}"
                  f"{chi:10.2e}  no test possible")
            continue
        ok = p < STAB_ALPHA and np.isfinite(chi) and chi < STAB_ALPHA
        if ok:
            carries.append(c)
        if ok:
            verdict = "own geography"
        elif p < STAB_ALPHA:
            verdict = "REFUSED: passes the rank test, but the lan do not differ"
        else:
            verdict = "NOT distinguishable from chance"
        print(f"    {c[:50]:<52}{n:>7,}{100 * n / tot:9.3f}%{obs[j]:+12.3f}"
              f"{q95:+11.3f}{p:8.4f}{chi:10.2e}  {verdict}")
    return carries


def _citizen_shares():
    """The shares at both levels, and which categories earned geography at which level.

    Italy's splice (sources.md §9as): a category takes the finest level at which it passes
    both tests. The lan pool is rounds 5-8, the only rounds with NUTS 3. The riksomrade pool
    is all six rounds, 5-8 aggregated to NUTS 2 plus 9 and 11, because the only reason 9 and
    11 were ever left out is that they have no NUTS 3.
    """
    raw = _pool(ESS_NUTS3_ROUNDS, "n")
    pool = _pool(ESS_NUTS3_ROUNDS, "w")
    answered = 1.0 - pool.loc[pool["cat"] == REFUSAL, "count"].sum() / pool["count"].sum()
    print(f"  {raw['count'].sum():,.0f} citizens with a region over {len(ESS_NUTS3_ROUNDS)} "
          f"rounds; {100 * (1 - answered):.2f}% refused and are not drawn")
    raw = raw[raw["cat"] != REFUSAL]
    pool = pool[pool["cat"] != REFUSAL]
    if int(raw["count"].sum()) != N_CITIZENS:
        sys.exit(f"!! {raw['count'].sum():,.0f} answered citizens, expected {N_CITIZENS:,} — "
                 "a round has been reissued and note_public quotes this figure")

    import se2024
    # THE ASSERTION IS SYMMETRIC, which is fi.py's §8 finding: `set(pool) - set(SOURCE)`
    # catches a category that APPEARED and cannot catch one that VANISHED, and a vanished
    # category is what an off-by-one on the denomination axis looks like. `SOURCE` and not
    # `MAP`, because `Ortodoxa kyrkan` is one answer in the survey and three rows in the CSV.
    unknown = sorted(set(pool["cat"]) - se2024.SOURCE)
    if unknown:
        sys.exit(f"!! source categories with no mapping: {unknown}")
    vanished = sorted(se2024.SOURCE - set(pool["cat"]))
    if vanished:
        sys.exit(f"!! se2024.SOURCE categories that no round produced: {vanished} — a mapped "
                 "answer with nobody on it means the denomination axis moved under the "
                 "parse, not that Sweden lost a religion")

    tab = pool.groupby(["region", "cat"])["count"].sum().unstack(fill_value=0.0)
    if len(tab) != N_UNITS:
        sys.exit(f"!! {len(tab)} lan have respondents, expected {N_UNITS}")
    share = tab.div(tab.sum(axis=1), axis=0)
    nraw = raw.groupby(["region", "cat"])["count"].sum().unstack(fill_value=0.0)
    nraw = nraw.reindex(index=share.index, columns=share.columns, fill_value=0.0)
    nat = tab.sum() / tab.sum().sum()

    per_unit = nraw.sum(axis=1)
    print(f"  respondents per lan: min {per_unit.min():,.0f} ({per_unit.idxmin()}), "
          f"median {per_unit.median():,.0f}, max {per_unit.max():,.0f} "
          f"({per_unit.idxmax()})")

    # ---- the split-half, §14.16 and §9cy, on the rounds, plus §9bi's chi-square
    cats = sorted(nat.index, key=lambda k: -nat[k])
    lan = _stability(raw, cats, nraw, ESS_NUTS3_ROUNDS, NUTS3, "21 lan")

    # ---- the riksomrade pool: all six rounds, region cut to its NUTS 2 prefix
    raw2 = _pool(ESS_ROUNDS, "n")
    pool2 = _pool(ESS_ROUNDS, "w")
    raw2 = raw2[raw2["cat"] != REFUSAL].copy()
    pool2 = pool2[pool2["cat"] != REFUSAL].copy()
    for d in (raw2, pool2):
        d["region"] = d["region"].str[:4]
    if int(raw2["count"].sum()) != N_CITIZENS_ALL:
        sys.exit(f"!! {raw2['count'].sum():,.0f} answered citizens over six rounds, expected "
                 f"{N_CITIZENS_ALL:,}")
    unknown = sorted(set(pool2["cat"]) - se2024.SOURCE)
    if unknown:
        sys.exit(f"!! rounds 9/11 produce source categories with no mapping: {unknown}")
    tab2 = pool2.groupby(["region", "cat"])["count"].sum().unstack(fill_value=0.0)
    if set(tab2.index) != set(NUTS2):
        sys.exit(f"!! riksomraden with respondents: {sorted(tab2.index)}")
    tab2 = tab2.reindex(index=sorted(NUTS2), columns=share.columns, fill_value=0.0)
    share2 = tab2.div(tab2.sum(axis=1), axis=0)
    nraw2 = raw2.groupby(["region", "cat"])["count"].sum().unstack(fill_value=0.0)
    nraw2 = nraw2.reindex(index=sorted(NUTS2), columns=share.columns, fill_value=0.0)
    riks_pass = _stability(raw2, cats, nraw2, ESS_ROUNDS, NUTS2, "8 riksomraden")
    riks = [c for c in cats if c in riks_pass and c not in lan and c not in KEEP_AS_RESIDUAL]
    small = [c for c in nat.index if c not in lan and c not in riks]
    print(f"\n  {len(lan)} categories at the lan, {len(riks)} at the riksomrade, "
          f"{len(small)} at the national rate inside each lan's residual")

    print("\n  national citizen shares (weighted, rounds 5-8), with unweighted respondent counts:")
    for k, v in nat.sort_values(ascending=False).items():
        tag = "lan" if k in lan else "riksomrade" if k in riks else "national rate"
        print(f"    {100 * v:>6.2f}%  {int(nraw[k].sum()):>6,}  {k}   [{tag}]")
    return share, share2, nat, lan, riks, small, answered, nraw, nraw2


def _compose(share, share2, nat, lan, riks, small):
    """Per-lan composition as a closed partition: Italy's splice closed by §9bi's residual.

    A lan-level category takes its own lan's share. A riksomrade-level category takes its
    riksomrade's share, the same figure in every lan inside it, which is how it.py applies a
    ripartizione's shares inside each regione (sources.md §9as). What is left of the lan is
    divided among the rest at their NATIONAL relative proportions, so the tail still moves
    with each lan's measured shares. Guatemala's §9bi has that argument.
    """
    small_total = float(nat[small].sum())
    out = pd.DataFrame(index=share.index, columns=share.columns, dtype=float)
    for c in lan:
        out[c] = share[c]
    for c in riks:
        out[c] = [float(share2.loc[u[:4], c]) for u in share.index]
    residual = 1.0 - out[lan + riks].sum(axis=1)
    if (residual <= 0).any():
        sys.exit(f"!! lan with no room for the tail: {sorted(residual[residual <= 0].index)}")
    print(f"  the tail is {residual.min():.2%} of {residual.idxmin()} and "
          f"{residual.max():.2%} of {residual.idxmax()}, against {small_total:.2%} nationally")
    # The check behind KEEP_AS_RESIDUAL, printed rather than asserted so a new vintage shows it.
    for c in sorted(KEEP_AS_RESIDUAL & set(small)):
        alt = residual - pd.Series([float(share2.loc[u[:4], c]) for u in share.index],
                                   index=share.index)
        print(f"  if `{c}` were fixed at its riksomrade's share, the tail left would be "
              f"negative in {int((alt < 0).sum())} of {len(alt)} lan "
              f"(worst {alt.min():+.2%}, {NUTS3[alt.idxmin()]})")
    for c in small:
        out[c] = residual * (nat[c] / small_total)
    err = (out.sum(axis=1) - 1.0).abs().max()
    if err > 1e-9:
        sys.exit(f"!! composition does not sum to 1, worst {err:.2e}")
    return out


# =======================================================================================
# the foreign half, and the population both halves are scaled to
# =======================================================================================

def _census():
    """NUTS 3 x citizenship from cens_21ctz_r3, as a tidy frame."""
    d = json.load(open(os.path.join(RAW, "cens_21ctz_r3_se.json"), encoding="utf-8"))
    dims = d["id"]
    cats = [list(d["dimension"][x]["category"]["index"]) for x in dims]
    sizes = d["size"]
    rows = []
    for k, v in d["value"].items():
        i = int(k)
        idx = []
        for s in reversed(sizes):
            idx.append(i % s)
            i //= s
        idx = list(reversed(idx))
        rows.append([cats[j][idx[j]] for j in range(len(dims))] + [v])
    df = pd.DataFrame(rows, columns=dims + ["value"])
    # The geo dimension holds EVERY NUTS level at once — SE, SE1, SE11, SE110 are all rows —
    # so a prefix filter sums the same people four times. Only the 21 leaves are read, and the
    # SEZ/SEZZ/SEZZZ "extra-regio" rows (all zero here) are excluded by the same list.
    df = df[df["geo"].isin(NUTS3)]
    df["unit"] = df["geo"]
    return df


def _register():
    """The Church of Sweden's lan rows: {NUTS 3 -> (population, members)}.

    THE NAME TABLE IS CHECKED AGAINST THE CENSUS'S OWN POPULATION rather than eyeballed. The
    PDF prints each lan's total folkmangd beside its membership, so a swapped pair shows up as
    a population that does not match the census's for that code — and Swedish lan populations
    differ by enough that no two of them could be confused. That is a join check that does not
    look at names at all, which is the only kind worth having
    ([[reference_name_join_wrong_neighbour]]).
    """
    import fitz
    d = fitz.open(os.path.join(RAW, "medlemsutveckling_lkf.pdf"))
    lines = []
    for i in range(d.page_count):
        lines.extend(d[i].get_text().split("\n"))
    num = re.compile(r"^-?[\d  ]+$")
    out = {}
    for i, ln in enumerate(lines):
        s = ln.strip()
        if not s.endswith(" län") or s not in CHURCH_LAN:
            continue
        # The two numbers after the lan's name are `Total folkmangd 31/12` and `Medlemmar i
        # Svenska Kyrkan 31/12`, in that order, and the header on page 1 says so.
        vals, j = [], i + 1
        while j < len(lines) and len(vals) < 2:
            t = lines[j].strip()
            if num.match(t):
                vals.append(int(t.replace(" ", "").replace(" ", "")))
            elif t:
                break
            j += 1
        if len(vals) == 2:
            out[CHURCH_LAN[s]] = (vals[0], vals[1])
    if len(out) != N_UNITS:
        sys.exit(f"!! the church PDF gave {len(out)} lan rows, expected {N_UNITS}")
    if abs(sum(v[1] for v in out.values()) - CHURCH_MEMBERS) > 0:
        sys.exit(f"!! church membership sums to {sum(v[1] for v in out.values()):,}, not the "
                 f"expected {CHURCH_MEMBERS:,} — the edition on disk has changed")
    return out


def _foreign_half(cen):
    """[geo_id, node, count] for foreign nationals, counted at NUTS 3."""
    import origin_religion as origin

    with zipfile.ZipFile(os.path.join(RAW, "pew.zip")) as z:
        name = [n for n in z.namelist() if n.endswith("(percentages).csv")][0]
        pew = pd.read_csv(io.BytesIO(z.read(name)))
    pew = pew[(pew["Year"] == 2020) & (pew["Level"] == 1)].set_index("Country")
    fams = origin.FAMILIES

    leaf = cen[cen["citizen"].str.fullmatch(r"[A-Z]{2}") & (cen["citizen"] != "SE")]
    total_for = cen[cen["citizen"] == "FOR"]["value"].sum()
    covered = leaf["value"].sum()
    print(f"  {leaf['citizen'].nunique()} named citizenships cover {covered:,.0f} of "
          f"{total_for:,.0f} foreign residents ({100 * covered / total_for:.2f}%)")
    # EUROSTAT'S `FOR` CAN HOLD THE COUNTRY'S OWN RECOGNISED NON-CITIZENS (Latvia's 190,544
    # `RNC`), and scaling the named citizenships up to `FOR` would hand them to the named
    # countries (spec §12, "EUROSTAT'S `FOR`"). Sweden's `RNC` is 0 (2026-09-14); asserted, with
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
        comp[iso] = origin.composition(iso, row, "other.se")
    if unmapped:
        sys.exit(f"!! {len(unmapped)} citizenships have no composition: {unmapped}")

    # The unnamed remainder — stateless, recognised non-citizen, unknown and rounding — is
    # spread over the named citizenships of its own lan rather than dropped, so each unit's
    # foreign total is the census's own.
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
# build
# =======================================================================================

def build():
    import se2024

    print("checking the NUTS level of every round…")
    _check_level()

    print("\nESS…")
    share, share2, nat, lan, riks, small, answered, nraw, nraw2 = _citizen_shares()

    print("\nEurostat census…")
    cen = _census()
    pop = cen[cen["citizen"] == "TOTAL"].groupby("unit")["value"].sum()
    nat_cit = cen[cen["citizen"] == "NAT"].groupby("unit")["value"].sum()
    for_ = cen[cen["citizen"] == "FOR"].groupby("unit")["value"].sum()
    print(f"  {pop.sum():,.0f} people = {nat_cit.sum():,.0f} Swedish citizens "
          f"+ {for_.sum():,.0f} foreign, over {len(pop)} lan")
    missing = sorted(set(NUTS3) - set(pop.index))
    if missing:
        sys.exit(f"!! census has no rows for {missing}")
    if abs(pop.sum() - POP_2021) > 2_000:
        sys.exit(f"!! census total {pop.sum():,.0f} is not Sweden's {POP_2021:,}")

    # ---- citizen half
    print("\ncomposing the citizen half…")
    comp = _compose(share, share2, nat, lan, riks, small)
    # ANSWER 4 BECOMES THREE ROWS, on MUCF's `betjanade` counts. se2024's docstring argues it;
    # the split is done HERE rather than inside resolve() so that it is visible in se.csv and
    # tools/check_mapping.py can see all three, and AFTER the split-half so the test and the
    # respondent counts are on the answer the survey actually collected.
    split = se2024.ORTHODOX_SPLIT
    if abs(sum(split.values()) - 1.0) > 1e-9:
        sys.exit(f"!! the Orthodox split does not sum to 1: {sum(split.values())}")
    print(f"  splitting `{se2024.ORTHODOX_ANSWER}` ({100 * nat[se2024.ORTHODOX_ANSWER]:.2f}% "
           "of citizens) on MUCF betjanade:")
    for k, v in split.items():
        print(f"    {100 * v:5.2f}%  {k}  -> {se2024.MAP[k]}")

    rows = []
    for unit in sorted(NUTS3):
        n_cit = float(nat_cit.get(unit, 0.0)) * answered
        if unit not in comp.index:
            sys.exit(f"!! no ESS respondents in {unit}")
        for cat, s in comp.loc[unit].items():
            if s <= 0:
                continue
            r = unit[:4]
            if cat in lan:
                note = f"ESS rounds 5-8 at this lan, {int(nraw.loc[unit, cat])} respondents"
            elif cat in riks:
                note = (f"ESS rounds 5-9 and 11 at {r} ({NUTS2[r]}), "
                        f"{int(nraw2.loc[r, cat])} respondents; that riksomrade's share, "
                        "applied inside this lan (§9as)")
            else:
                note = "national share within the lan's residual (§9bi, split-half not passed)"
            if cat == se2024.ORTHODOX_ANSWER:
                for sub, w in split.items():
                    rows.append((unit, NUTS3[unit], sub, s * w * n_cit,
                                 note + "; communion split on MUCF betjanade 2024 (§3.11)"))
            else:
                rows.append((unit, NUTS3[unit], cat, s * n_cit, note))
    cit = pd.DataFrame(rows, columns=["geo_id", "geo_name", "source_category", "count",
                                      "note"])
    cit["geo_level"] = "nuts3"
    cit["basis"] = "self_id"
    cit["year"] = CHURCH_YEAR
    cit["source_id"] = "ess_r5_r8_nuts3_x_r5_r11_nuts2"
    cit = cit[COLUMNS]
    print(f"  citizen half: {cit['count'].sum():,.0f} people "
          f"({100 * cit['count'].sum() / nat_cit.sum():.2f}% of citizens; the rest declined)")

    unknown = sorted(set(cit["source_category"]) - set(se2024.MAP))
    if unknown:
        sys.exit(f"!! unmapped source categories: {unknown}")

    # ---- foreign half
    print("\nforeign half…")
    ext = _foreign_half(cen)
    ext["geo_level"] = "nuts3"
    ext["geo_name"] = ext["geo_id"].map(NUTS3)
    ext["tier"] = "modelled"
    ext["basis"] = "nationality_derived"
    ext["year"] = 2021
    ext["source_id"] = "cens_21ctz_r3_x_pew2020"
    ext = ext[["geo_id", "geo_level", "geo_name", "node", "count", "tier", "basis",
               "year", "source_id"]]

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    cit.to_csv(OUT, index=False)
    ext.to_csv(OUT_FOREIGN, index=False)

    drawn = cit["count"].sum() + ext["count"].sum()
    print(f"\nwrote {OUT}  ({len(cit):,} rows, "
          f"{cit['source_category'].nunique()} source categories)")
    print(f"wrote {OUT_FOREIGN}  ({len(ext):,} rows, {ext['node'].nunique()} nodes)")
    print(f"drawn {drawn:,.0f} of {pop.sum():,.0f} — {100 * drawn / pop.sum():.2f}%")

    both = pd.concat([
        cit.assign(node=cit["source_category"].map(se2024.resolve))[["node", "count"]],
        ext[["node", "count"]]], ignore_index=True)
    top = both.groupby("node")["count"].sum().sort_values(ascending=False)
    print("\nnational totals:")
    for node, c in top.head(20).items():
        print(f"  {c:>11,.0f}  {100 * c / drawn:5.2f}%  {node}")

    # ---- WHAT ROUNDS 9 AND 11 WOULD HAVE SAID, which is what dropping them costs.
    late = _pool(ESS_NUTS2_ROUNDS, "w")
    late = late[late["cat"] != REFUSAL]
    ln = late.groupby("cat")["count"].sum()
    ln = ln / ln.sum()
    print(f"\nROUNDS 9 AND 11 (2018 and 2023, NUTS 2 only, {_pool(ESS_NUTS2_ROUNDS, 'n')['count'].sum():,.0f} "
          f"respondents) AGAINST ROUNDS 5-8 (the riksomrade categories pool all six):")
    for k in nat.sort_values(ascending=False).index:
        print(f"  {k[:50]:<52} rounds 5-8 {100 * nat[k]:6.2f}%   "
              f"rounds 9+11 {100 * ln.get(k, 0.0):6.2f}%   "
              f"{100 * (ln.get(k, 0.0) - nat[k]):+6.2f} points")

    # ---- THE CROSS-CHECK THAT MATTERS, and it is the whole point of the country.
    # The Church of Sweden's own membership, tabulated for it by SCB, at the SAME 21 lan.
    # This is NOT a validation: the two instruments ask different questions and the gap and
    # its geography are the finding. sources/se.md §4 reads it.
    from scipy.stats import spearmanr
    reg = _register()
    for u, (p, _) in reg.items():
        if abs(p - pop[u]) / pop[u] > 0.02:
            sys.exit(f"!! {u}: the church PDF says {p:,} people and the census says "
                     f"{pop[u]:,.0f} — the lan name table is joined to the wrong code")
    tot_p = sum(v[0] for v in reg.values())
    tot_m = sum(v[1] for v in reg.values())
    svk = both[both["node"] == "christianity.lutheran"]["count"].sum()
    print(f"\nREGISTER ({CHURCH_YEAR}) vs SURVEY, the thing this country is about:")
    print(f"  Church of Sweden membership, counted   {tot_m:>10,}   {100 * tot_m / tot_p:5.2f}%")
    print(f"  'Svenska kyrkan' on this map           {svk:>10,.0f}   {100 * svk / drawn:5.2f}%")
    print(f"  the difference                         {tot_m - svk:>10,.0f}   "
          f"{100 * (tot_m / tot_p - svk / drawn):+5.2f} points")
    rs = pd.Series({k: v[1] / v[0] for k, v in reg.items()})
    es = share["Svenska kyrkan"].reindex(rs.index)
    sp = spearmanr(rs.values, es.values)
    print(f"  and the two do not order the lan the same way: Spearman {sp.correlation:+.3f} "
          f"(p={sp.pvalue:.3f}) over {len(rs)} lan.")
    cmp = pd.DataFrame({"register": 100 * rs, "survey": 100 * es}).sort_values(
        "register", ascending=False)
    print(f"    {'lan':<24}{'register %':>12}{'survey %':>10}")
    for u in list(cmp.index[:4]) + list(cmp.index[-4:]):
        print(f"    {NUTS3[u][:24]:<24}{cmp.loc[u, 'register']:12.2f}"
              f"{cmp.loc[u, 'survey']:10.2f}")
    print("  The register's top is Norrbotten and the survey's is Kronoberg: nominal "
          "membership\n  is highest in the north and stated belonging is highest in the "
          "Smaland free-church\n  belt, which are two different geographies of the same "
          "church rather than an error.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    a = ap.parse_args()
    if a.fetch:
        fetch()
    else:
        build()
