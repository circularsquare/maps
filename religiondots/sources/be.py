"""Belgium — European Social Survey rounds 5-11 (citizens) + Eurostat census x Pew (residents).

    python sources/be.py --fetch     -> data/raw/be/, then
    python sources/be.py             -> data/normalized/be.csv + be_foreign.csv

Belgium has never asked religion in a modern census and its statistical office publishes
nothing on the subject at all: Statbel's own site search returns zero results for `religie`,
`godsdienst`, `levensbeschouwing`, `moslim`, `religion`, `culte` and `confession` in both
its Dutch and its French index. sources/be.md §1 has the record, including how to read the
site at all, which is not free.

So the country is the Greece construction (§9z), and it is two halves of one census table:

  * **Belgian citizens, 10.08M.** ESS rounds 5 to 11 pooled and restricted to
    `ctzcntr = Yes`, 11,015 people over the eleven NUTS 2 provinces.
  * **Foreign residents, 1.45M.** Eurostat's 2021 census table `cens_21ctz_r3`, 200 named
    citizenships, crossed with Pew's religious composition for each origin country.

**The foreign half matters more in Belgium than in any other country drawn this way.**
Foreign citizens are 12.58% of the country and 34.99% of the Brussels-Capital Region, and
ESS's Belgian sample is 7.9% non-citizen — so the survey alone would draw Brussels roughly a
third wrong, in a direction nothing in the sample could reveal. Greece's foreign share is
7.2% and Finland's is 5.2%.

TWO THINGS THAT WILL BITE, both written down because they cost time here:

1. **Belgium has no country-specific denomination variable in rounds 10 and 11.** `rlgdnbe`
   exists in rounds 5 to 9 and is absent from the later two, where the API answers a bare
   HTTP 400 rather than naming the variable. Pooling on `rlgdnbe` therefore silently drops
   the two most recent rounds. What saves Belgium is that `rlgdnbe` is only the Dutch and
   French rendering of the harmonised card — its eight codes are the harmonised `rlgdnm`'s
   eight codes, value for value, in all five rounds that carry both — so `rlgdnm` is the
   variable to pool and nothing is lost by using it. The Netherlands is the opposite case:
   `rlgdnanl` splits hervormd, gereformeerd and PKN, which `rlgdnm` flattens to `Protestant`.
2. **`table.path` is a list of INDICES into `values`, not a list of code values.** Reading it
   as codes silently zeroes every category whose code is not also a valid index. fi.py
   records this; it is repeated here because it is the single most dangerous thing about
   this API.

WHAT IS NEW HERE: the split-half stability test is run on an ESS country for the first time.
Greece, Finland, France, Germany and Italy all draw every category at its measured
geography. Belgium's card has eight denominations and three of them are reached by fewer
than a hundred respondents in seven rounds pooled, so the question §14.16 asks — does this
category's ranking across the units replicate — is worth asking, and the answer changes the
map. The test and the reason are in `_stability` below.
"""

import argparse
import io
import json
import os
import ssl
import sys
import urllib.error
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
RAW = os.path.join(ROOT, "data", "raw", "be")
OUT = os.path.join(ROOT, "data", "normalized", "be.csv")
OUT_FOREIGN = os.path.join(ROOT, "data", "normalized", "be_foreign.csv")

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) religiondots"}

# --- ESS --------------------------------------------------------------------------------
ESS_API = "https://api.nsd.no/graphql"

# Belgium is in every ESS round. Rounds 1-4 have no `region` variable, the wall Greece,
# France, Italy and Finland all hit, so SEVEN of eleven are usable. (datafile id, version),
# the same ids sources/fr.py, sources/it.py and sources/fi.py use.
ESS_ROUNDS = {
    5: ("0189b86b-8aa4-4be3-88ad-39c58b02f19f", 89),
    6: ("450fa78e-68ab-493f-b169-dbc7ab8ffec2", 85),
    7: ("9c96a1b2-b027-43c1-8c74-e883f892d0bb", 91),
    8: ("ffc43f48-e15a-4a1c-8813-47eda377c355", 98),
    9: ("b2b0bf39-176b-4eca-8d26-3c05ea83d2cb", 280),
    10: ("f37d014a-6958-42d4-b03b-17c29e481d3d", 286),
    11: ("242aaa39-3bbb-40f5-98bf-bfb1ce53d8ef", 179),
}

# `rlgdnm`, not `rlgdnbe` — see the module docstring, trap 1. `_check_be_card` re-proves the
# equivalence from the data in every round that carries both, rather than asking you to
# believe this comment.
ESS_BREAK = ["region", "ctzcntr", "rlgblg", "rlgdnm"]
ESS_BREAK_BE = ["region", "rlgdnbe"]      # rounds 5-9 only, for the card check
# NOT DRAWN. ESS asks everyone who says they do not belong to a religion whether they ever
# did, and then which one. No count in this build comes from it; it is fetched because the
# thing worth saying about Belgium is the size of the lapsed-Catholic population, and a
# claim that size should be reproducible from the repo rather than quoted from a note.
ESS_BREAK_PAST = ["ctzcntr", "rlgblge", "rlgdnme"]

# `%s` is the weight clause. pspwght is normalised to the sample size, so both passes total
# the same n — but the WEIGHTED per-cell values are fractional and are not respondent counts,
# so the unweighted pass is what the n floors are read off (de_ess.py's note).
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

# --- the geography ------------------------------------------------------------------------
# NUTS 2021, the ten provinces plus Brussels, which is what data/geo/be/be_lau.gpkg rolls up
# to and what ESS's `region` is in all seven usable rounds. NO RECODE IS NEEDED: unlike
# Finland's three NUTS vintages in seven rounds and Greece's two in three, Belgium's eleven
# codes and their labels are byte-identical from round 5 to round 11. `_check_regions`
# asserts that rather than assuming it.
NUTS2 = {
    "BE10": "Brussels / Bruxelles",
    "BE21": "Antwerpen",
    "BE22": "Limburg",
    "BE23": "Oost-Vlaanderen",
    "BE24": "Vlaams-Brabant",
    "BE25": "West-Vlaanderen",
    "BE31": "Brabant wallon",
    "BE32": "Hainaut",
    "BE33": "Liege",
    "BE34": "Luxembourg",
    "BE35": "Namur",
}

# --- the categories -----------------------------------------------------------------------
NO_RELIGION = "No religion"          # rlgblg = No; the source has no atheist/agnostic split
# rlgblg answers that are neither Yes nor No, and rlgdnm answers from a respondent who said
# Yes and then declined the denomination. spec §3.5: refusals are marked, not filled.
REFUSAL = "__refused__"

COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count",
           "basis", "year", "source_id", "note"]

# Belgium's own totals, asserted so a re-fetch against a new ESS release or census vintage
# fails here rather than quietly redrawing the map.
POP_2021 = 11_554_767       # cens_21ctz_r3's own Belgian total
N_ROUNDS = 7
N_UNITS = 11
# Unweighted, ctzcntr = Yes, rounds 5-11. This is the figure note_public quotes as "people
# interviewed", so it is asserted rather than printed: a reissued round would otherwise move
# it silently and leave the reader-facing text wrong.
N_CITIZENS = 10_877

# §14.16's bar, applied round-wise. See `_stability`.
STAB_ALPHA = 0.05
STAB_PERM = 2000
STAB_SEED = 0


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


def _ess_soft(query, variables):
    """Like `_ess`, but returns None when the variable simply is not in this round.

    ESS answers a missing variable with HTTP 400 and `E201VariableNotFound` in the body
    rather than with an empty result, so "Belgium has no card in rounds 10 and 11" and "the
    query is malformed" arrive looking identical. Only the first is tolerated here.
    """
    body = json.dumps({"query": query, "variables": variables}).encode()
    req = urllib.request.Request(ESS_API, data=body, headers={
        "Content-Type": "application/json", **UA})
    try:
        r = json.load(urllib.request.urlopen(req, timeout=900))
    except urllib.error.HTTPError as e:
        r = json.loads(e.read().decode("utf-8", "replace"))
    if "errors" in r:
        code = r["errors"][0].get("extensions", {}).get("code", "")
        if code == "E201VariableNotFound":
            return None
        sys.exit(f"!! ESS API: {r['errors'][0].get('message')} {code}")
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
            be = [x for x in d["analysis"]["frequencyTabulationByVariables"]["responses"]
                  if x["by"][0]["value"] == "BE"]
            if not be:
                sys.exit(f"!! ESS round {rnd} has no BE response")
            json.dump(be[0]["response"], open(dest, "w", encoding="utf-8"),
                      ensure_ascii=False)
        n = sum(c["count"] for c in json.load(
            open(os.path.join(RAW, f"ess_r{rnd}_n.json"), encoding="utf-8"))["table"])
        print(f"  round {rnd}: {n:,.0f} respondents")

    # The country-specific card, rounds 5-9. Not drawn from; it is the evidence for the
    # docstring's claim that `rlgdnm` loses nothing in Belgium, and `_check_be_card` reads it.
    print("ESS rlgdnbe (rounds 5-9, the card check)…")
    for rnd, (fid, ver) in sorted(ESS_ROUNDS.items()):
        dest = os.path.join(RAW, f"ess_r{rnd}_be.json")
        if os.path.exists(dest):
            continue
        d = _ess_soft(ESS_TAB_N, {"id": fid, "v": ver, "bv": ESS_BREAK_BE})
        if d is None:
            # Rounds 10 and 11: Belgium has no country-specific card at all.
            json.dump({"__absent__": True}, open(dest, "w", encoding="utf-8"))
            print(f"  round {rnd}: rlgdnbe absent (E201VariableNotFound)")
            continue
        be = [x for x in d["analysis"]["frequencyTabulationByVariables"]["responses"]
              if x["by"][0]["value"] == "BE"]
        json.dump(be[0]["response"] if be else {"__absent__": True},
                  open(dest, "w", encoding="utf-8"), ensure_ascii=False)

    print("ESS rlgblge / rlgdnme (the lapsed half, not drawn from)…")
    for rnd, (fid, ver) in sorted(ESS_ROUNDS.items()):
        dest = os.path.join(RAW, f"ess_r{rnd}_past.json")
        if os.path.exists(dest):
            continue
        d = _ess_soft(ESS_TAB_N, {"id": fid, "v": ver, "bv": ESS_BREAK_PAST})
        if d is None:
            json.dump({"__absent__": True}, open(dest, "w", encoding="utf-8"))
            continue
        be = [x for x in d["analysis"]["frequencyTabulationByVariables"]["responses"]
              if x["by"][0]["value"] == "BE"]
        json.dump(be[0]["response"] if be else {"__absent__": True},
                  open(dest, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"  {N_ROUNDS} rounds on disk")

    print("Eurostat census…")
    dest = os.path.join(RAW, "cens_21ctz_r3_be.json")
    if not os.path.exists(dest):
        d = _eurostat(EU_CTZ, format="JSON", lang="EN", age="TOTAL", sex="T")
        json.dump(d, open(dest, "w", encoding="utf-8"), ensure_ascii=False)
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


# =======================================================================================
# the citizen half
# =======================================================================================

def _ess_table(rnd, tag):
    """One round as a tidy frame: region CODE, everything else as a LABEL.

    `path` IS A LIST OF INDICES INTO `values`, NOT A LIST OF CODE VALUES — the module
    docstring's trap 2, and fi.py's. Reading it as codes silently returns zero for every cell
    whose code is not also a valid index, which on this card means Islam and both
    non-Christian buckets, and Belgium would have drawn with no Muslims and no error.
    """
    d = json.load(open(os.path.join(RAW, f"ess_r{rnd}_{tag}.json"), encoding="utf-8"))
    vv = {v["name"]: v for v in d["variableValues"]}
    order = [v["name"] for v in d["variableValues"]]
    labels = {n: [c["label"] for c in vv[n]["codeList"]] for n in order}
    missing = {n: [c["isMissing"] for c in vv[n]["codeList"]] for n in order}
    values = {n: [c["value"] for c in vv[n]["codeList"]] for n in order}
    rows = []
    for cell in d["table"]:
        rec = {}
        for i, n in enumerate(order):
            j = cell["path"][i]
            rec[n] = values[n][j] if n == "region" else labels[n][j]
            rec[n + "_miss"] = missing[n][j]
        rec["count"] = cell["count"]
        rows.append(rec)
    return pd.DataFrame(rows)


def _check_regions():
    """The eleven NUTS 2 codes and their labels are identical in all seven rounds.

    Greece pools two NUTS vintages and Finland three, and in both the failure is silent: a
    region splits in two and both halves come out undersized with no error anywhere
    ([[reference_name_join_wrong_neighbour]]). Belgium does not have that problem, and this
    asserts it rather than trusting it, because "no recode needed" is exactly the sort of
    claim that stops being true when ESS reissues a round.
    """
    seen = {}
    for rnd in ESS_ROUNDS:
        d = json.load(open(os.path.join(RAW, f"ess_r{rnd}_n.json"), encoding="utf-8"))
        vv = [v for v in d["variableValues"] if v["name"] == "region"][0]
        codes = {c["value"] for c in vv["codeList"] if not c["isMissing"]}
        if codes != set(NUTS2):
            sys.exit(f"!! round {rnd} regions are not the eleven provinces: "
                     f"extra {sorted(codes - set(NUTS2))}, "
                     f"missing {sorted(set(NUTS2) - codes)}")
        for c in vv["codeList"]:
            if not c["isMissing"]:
                seen.setdefault(c["value"], set()).add(c["label"])
    split = {k: v for k, v in seen.items() if len(v) > 1}
    if split:
        sys.exit(f"!! a region carries two labels across rounds: {split}")
    print(f"  regions: the same {len(seen)} NUTS 2 codes with one label each in all "
          f"{N_ROUNDS} rounds, no recode needed")


def _check_be_card():
    """`rlgdnm` loses nothing in Belgium, proved from the rounds that carry both variables.

    The claim the whole build rests on is that the country-specific `rlgdnbe` is the
    harmonised card in Dutch and French rather than a finer one. That is checkable: rounds 5
    to 9 carry both variables, so their per-round totals must agree category by category in
    code order. If ESS ever revises `rlgdnbe` into a longer card the way it did `rlgdnanl`,
    this fails and Belgium should be rebuilt on the country variable for the rounds that have
    one.
    """
    checked = 0
    for rnd in sorted(ESS_ROUNDS):
        d = json.load(open(os.path.join(RAW, f"ess_r{rnd}_be.json"), encoding="utf-8"))
        if d.get("__absent__"):
            continue
        vv = [v for v in d["variableValues"] if v["name"] == "rlgdnbe"][0]
        vals = [c["value"] for c in vv["codeList"]]
        miss = [c["isMissing"] for c in vv["codeList"]]
        axis = [v["name"] for v in d["variableValues"]].index("rlgdnbe")
        be = {}
        for cell in d["table"]:
            j = cell["path"][axis]
            if not miss[j]:
                be[vals[j]] = be.get(vals[j], 0) + cell["count"]

        m = _ess_table(rnd, "n")
        m = m[~m["rlgdnm_miss"]]
        codes = {c["label"]: c["value"] for c in
                 [x for x in json.load(open(os.path.join(RAW, f"ess_r{rnd}_n.json"),
                                            encoding="utf-8"))["variableValues"]
                  if x["name"] == "rlgdnm"][0]["codeList"]}
        harm = {codes[k]: v for k, v in m.groupby("rlgdnm")["count"].sum().items()}
        if be != harm:
            sys.exit(f"!! round {rnd}: rlgdnbe and rlgdnm disagree — {be} vs {harm}. "
                     f"Belgium's country card is no longer the harmonised one and this "
                     f"build is using the wrong variable.")
        checked += 1
    if checked < 5:
        sys.exit(f"!! only {checked} rounds could be card-checked, expected 5")
    print(f"  rlgdnbe == rlgdnm, code for code, in all {checked} rounds that carry both; "
          f"rounds 10 and 11 have no Belgian card at all")


def _category(row):
    """rlgblg x rlgdnm -> one source category, or REFUSAL."""
    if row["rlgblg_miss"]:
        return REFUSAL
    if row["rlgblg"] == "No":
        return NO_RELIGION
    if row["rlgblg"] != "Yes":
        return REFUSAL
    # said yes and then did not name one, or the variable is Not applicable for them
    return REFUSAL if row["rlgdnm_miss"] else row["rlgdnm"]


# ---------------------------------------------------------------------------------------
# §14.16, applied to ESS for the first time
# ---------------------------------------------------------------------------------------

def _stability(raw, cats):
    """WHICH CATEGORIES CARRY THEIR OWN PROVINCE GEOGRAPHY — §14.16, split by ROUND.

    The barometer modules split by PSU because they have microdata. This API hands back
    cross-tabs and no PSU, so the unit of resampling is the round: seven independent fielding
    waves, ten years apart end to end, split 35 ways into a 3-set and its complementary
    4-set. The statistic is the median Spearman across those 35 splits of a category's share
    across the eleven provinces.

    **The null is the same statistic with the province labels shuffled**, which is what makes
    this a test rather than a threshold ([[reference_check_needs_power]]): eleven units is
    few, most of these categories are small, and a fixed bar like 1.96/sqrt(n-1) would be
    reading a correlation whose sampling distribution nobody has looked at. A category passes
    when fewer than STAB_ALPHA of the permuted medians reach its observed one.

    **What a failure means, and does not.** It is a failure to demonstrate signal, not a
    demonstration of noise. The people stay drawn; what is withdrawn is the claim to know
    which province they are in, and they go at the national rate inside each province's
    residual. Belgium's Jewish cell is the one where that costs something visible and
    note_public says so: the community is concentrated in Antwerp and 16 respondents in
    seven rounds cannot show it.

    **The quota check §14.16's Lebanon entry asks for is not needed and could not run.** ESS
    is a probability sample with no sect or region quota — its own sampling documentation
    requires random selection at every stage — and `quota_agreement` needs the two waves to
    offer identical rational shares, which 35 overlapping splits of one pool cannot produce.
    Said here rather than silently skipped.
    """
    rounds = sorted(ESS_ROUNDS)
    cube = np.zeros((len(rounds), N_UNITS, len(cats)), dtype=float)
    units = sorted(NUTS2)
    for ri, rnd in enumerate(rounds):
        g = raw[raw["round"] == rnd]
        t = g.groupby(["region", "cat"])["count"].sum().unstack(fill_value=0.0)
        for ui, u in enumerate(units):
            for cj, c in enumerate(cats):
                if u in t.index and c in t.columns:
                    cube[ri, ui, cj] = t.loc[u, c]
    # A province with no respondent in a round makes `stability.median_rho` skip halvings, and the
    # per-round relabelling skips different ones in the null (spec §12, "A ROUND THAT SKIPS A
    # UNIT"). Refused, as `no._stability` and `cab.stability` refuse it (added 2026-09-14).
    empty = [(rounds[a], units[b]) for a, b in zip(*np.where(cube.sum(axis=2) == 0))]
    if empty:
        sys.exit(f"!! (round, province) cells with no respondent: {empty}. Test on the provinces "
                 "sampled in every round (ua.py::TEST_UNITS) or leave the round out")

    splits = stability.halvings(len(rounds))
    obs = stability.median_rho(cube, splits)
    null = stability.wave_null(cube, splits, STAB_PERM, STAB_SEED)

    print(f"\n  split-half stability across the {len(rounds)} rounds, median of "
          f"{len(splits)} three-against-four splits, against a {STAB_PERM}-draw "
          f"province-label permutation null (§14.16):")
    print(f"    {'category':<32}{'n':>7}{'national':>10}{'median rho':>12}"
          f"{'null 95th':>11}{'p':>8}  verdict")
    carries = []
    tot = raw["count"].sum()
    for j, c in enumerate(cats):
        n = int(raw[raw["cat"] == c]["count"].sum())
        p, q95 = stability.permutation_p(obs[j], null[:, j], STAB_ALPHA)
        if not np.isfinite(p):
            print(f"    {c[:30]:<32}{n:>7,}{100 * n / tot:9.3f}%{'':>12}{'':>11}{'':>8}  "
                  f"no test possible")
            continue
        if p < STAB_ALPHA:
            carries.append(c)
        print(f"    {c[:30]:<32}{n:>7,}{100 * n / tot:9.3f}%{obs[j]:+12.3f}"
              f"{q95:+11.3f}{p:8.4f}  "
              + ("own geography" if p < STAB_ALPHA else "NOT distinguishable from chance"))
    return carries


def _citizen_shares():
    """NUTS 2 x denomination shares among Belgian citizens, pooled over seven rounds."""
    import be2024

    _check_regions()
    _check_be_card()

    frames, unweighted = [], []
    for rnd in sorted(ESS_ROUNDS):
        w = _ess_table(rnd, "w")
        n = _ess_table(rnd, "n")
        for df, bag in ((w, frames), (n, unweighted)):
            df = df[(df["ctzcntr"] == "Yes")].copy()
            df["cat"] = df.apply(_category, axis=1)
            bag.append(df.assign(round=rnd))
        # RESPONDENTS ARE COUNTED UNWEIGHTED. pspwght is a post-stratification weight and its
        # citizen subtotal is NOT the number of citizens interviewed.
        a, b = frames[-1], unweighted[-1]
        print(f"  round {rnd}: {b['count'].sum():>6,.0f} citizen respondents "
              f"({a['count'].sum():>6,.0f} weighted), "
              f"{b['region'].nunique():>2} provinces, "
              f"{b[~b['cat'].isin([REFUSAL])]['cat'].nunique():>2} categories used")
    if len(frames) != N_ROUNDS:
        sys.exit(f"!! expected {N_ROUNDS} rounds, pooled {len(frames)}")

    pool = pd.concat(frames, ignore_index=True)
    raw = pd.concat(unweighted, ignore_index=True)

    total = pool["count"].sum()
    ref = pool[pool["cat"] == REFUSAL]["count"].sum()
    answered = 1.0 - ref / total
    n_people = raw["count"].sum()
    print(f"  pooled {n_people:,.0f} citizens interviewed over {N_ROUNDS} rounds "
          f"({total:,.0f} weighted); declined to answer {ref:,.0f} weighted "
          f"({100 * ref / total:.2f}%)")
    if n_people != N_CITIZENS:
        sys.exit(f"!! {n_people:,.0f} citizen respondents, expected {N_CITIZENS:,} — "
                 f"ESS has reissued a round and every figure in note_public needs re-reading")

    pool = pool[pool["cat"] != REFUSAL]
    raw = raw[raw["cat"] != REFUSAL]

    # Symmetric, for fi.py's reason: `set(pool) - set(MAP)` catches a category that APPEARED
    # and cannot catch one that VANISHED, which is what an off-by-one on the codeList axis
    # looks like. The pooled set over seven rounds is all nine MAP keys, so requiring
    # equality costs nothing and fails loudly on the regression.
    unknown = sorted(set(pool["cat"]) - set(be2024.MAP))
    if unknown:
        sys.exit(f"!! source categories with no mapping: {unknown}")
    vanished = sorted(set(be2024.MAP) - set(pool["cat"]))
    if vanished:
        sys.exit(f"!! be2024.MAP categories that no round produced: {vanished} — a mapped "
                 "answer with nobody on it means the denomination axis moved under the "
                 "parse, not that Belgium lost a religion")

    tab = pool.groupby(["region", "cat"])["count"].sum().unstack(fill_value=0.0)
    if len(tab) != N_UNITS:
        sys.exit(f"!! {len(tab)} provinces have respondents, expected {N_UNITS}")
    nat = tab.sum() / tab.sum().sum()
    cats = list(nat.sort_values(ascending=False).index)

    nraw = raw.groupby(["region", "cat"])["count"].sum().unstack(fill_value=0.0)
    per_unit = nraw.sum(axis=1)
    print(f"  respondents per province: min {per_unit.min():,.0f} ({per_unit.idxmin()}), "
          f"median {per_unit.median():,.0f}, max {per_unit.max():,.0f} "
          f"({per_unit.idxmax()})")

    carries = _stability(raw, cats)
    small = [c for c in cats if c not in carries]

    # The measured province shares, then the §14.16 construction: a category that carries its
    # own geography passes through, and what is left of each province is divided among the
    # rest at their NATIONAL relative proportions. So the tail's geography is the residual of
    # the stable measurements rather than a flat national rate. lits.py's build(), by hand,
    # because this file works in shares rather than in counts.
    share = tab.div(tab.sum(axis=1), axis=0)
    if small:
        small_total = float(sum(nat[c] for c in small))
        residual = 1.0 - share[carries].sum(axis=1)
        if (residual <= 0).any():
            sys.exit(f"!! provinces with no room for the tail: "
                     f"{sorted(residual[residual <= 0].index)}")
        print(f"    the tail is {residual.min():.2%} of {residual.idxmin()} and "
              f"{residual.max():.2%} of {residual.idxmax()}, against "
              f"{small_total:.2%} nationally")
        for c in small:
            share[c] = residual * (nat[c] / small_total)

    print("  national citizen shares (weighted), with unweighted respondent counts:")
    for k, v in nat.sort_values(ascending=False).items():
        flag = "" if k in carries else "   (drawn in the residual, not measured per province)"
        print(f"    {100 * v:>6.2f}%  {int(nraw[k].sum()):>6,}  {k}{flag}")
    return share, answered, nraw, carries


# =======================================================================================
# the foreign half, and the population both halves are scaled to
# =======================================================================================

def _lapsed():
    """How many of the unaffiliated used to belong, and to what. Printed, never drawn.

    `rlgblge` is asked of everyone who answered No to `rlgblg`, and `rlgdnme` of everyone who
    answers Yes to that. It is the only thing in this instrument that can tell a country
    which was Catholic and stopped apart from one which never was, and for Belgium it is the
    whole shape of the map: the 54% on `unaffiliated` is not a neutral cell.

    Finland prints a register against a survey here because it has two instruments; Belgium
    has one, so what it prints instead is the same instrument's own memory.
    """
    ever_yes = ever_no = 0
    past = {}
    for rnd in sorted(ESS_ROUNDS):
        d = json.load(open(os.path.join(RAW, f"ess_r{rnd}_past.json"), encoding="utf-8"))
        if d.get("__absent__"):
            continue
        order = [v["name"] for v in d["variableValues"]]
        vv = {v["name"]: v for v in d["variableValues"]}
        lab = {n: [c["label"] for c in vv[n]["codeList"]] for n in order}
        mis = {n: [c["isMissing"] for c in vv[n]["codeList"]] for n in order}
        ic, ie, ip = (order.index(x) for x in ("ctzcntr", "rlgblge", "rlgdnme"))
        for cell in d["table"]:
            if lab["ctzcntr"][cell["path"][ic]] != "Yes" or mis["ctzcntr"][cell["path"][ic]]:
                continue
            je, jp = cell["path"][ie], cell["path"][ip]
            if mis["rlgblge"][je]:
                continue
            if lab["rlgblge"][je] == "No":
                ever_no += cell["count"]
                continue
            if lab["rlgblge"][je] != "Yes":
                continue
            ever_yes += cell["count"]
            if not mis["rlgdnme"][jp]:
                k = lab["rlgdnme"][jp]
                past[k] = past.get(k, 0) + cell["count"]
    named = sum(past.values())
    print("\nTHE LAPSED HALF of `unaffiliated`, which is what this country is about "
          "(printed, not drawn):")
    print(f"  of {ever_yes + ever_no:,} citizens who say they do NOT belong to a religion, "
          f"{ever_yes:,} say they once did ({100 * ever_yes / (ever_yes + ever_no):.1f}%)")
    for k, v in sorted(past.items(), key=lambda kv: -kv[1]):
        print(f"    {100 * v / named:>6.1f}%  {v:>5,}  {k}")
    return ever_yes, ever_no, past


def _census():
    """NUTS 2 x citizenship from cens_21ctz_r3, as a tidy frame."""
    d = json.load(open(os.path.join(RAW, "cens_21ctz_r3_be.json"), encoding="utf-8"))
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
    # The geo dimension holds EVERY NUTS level at once — BE, BE2, BE21, BE211 are all rows —
    # so a prefix filter sums the same people four times. Only the eleven NUTS 2 leaves are
    # read. BEZ/BEZZ/BEZZZ, Eurostat's "extra-regio", are all zero for Belgium and excluded
    # by the same key list.
    df = df[df["geo"].isin(NUTS2)]
    df["unit"] = df["geo"]
    return df


def _foreign_half(cen):
    """[geo_id, node, count] for foreign nationals, counted at NUTS 2."""
    import origin_religion as origin

    with zipfile.ZipFile(os.path.join(RAW, "pew.zip")) as z:
        name = [n for n in z.namelist() if n.endswith("(percentages).csv")][0]
        pew = pd.read_csv(io.BytesIO(z.read(name)))
    pew = pew[(pew["Year"] == 2020) & (pew["Level"] == 1)].set_index("Country")
    fams = origin.FAMILIES

    leaf = cen[cen["citizen"].str.fullmatch(r"[A-Z]{2}") & (cen["citizen"] != "BE")]
    total_for = cen[cen["citizen"] == "FOR"]["value"].sum()
    covered = leaf["value"].sum()
    print(f"  {leaf['citizen'].nunique()} named citizenships cover {covered:,.0f} of "
          f"{total_for:,.0f} foreign residents ({100 * covered / total_for:.2f}%)")
    # EUROSTAT'S `FOR` CAN HOLD THE COUNTRY'S OWN RECOGNISED NON-CITIZENS (Latvia's 190,544
    # `RNC`), and scaling the named citizenships up to `FOR` would hand them to the named
    # countries (spec §12, "EUROSTAT'S `FOR`"). Belgium's `RNC` is 0 (2026-09-14); asserted, with
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
        comp[iso] = origin.composition(iso, row, "other.be")
    if unmapped:
        sys.exit(f"!! {len(unmapped)} citizenships have no composition: {unmapped}")

    # The unnamed remainder — STLS, RNC, UNK and rounding — is spread over the named
    # citizenships of its own province rather than dropped, so each unit's foreign total is
    # the census's own.
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
    import be2024

    print("ESS…")
    share, answered, nraw, carries = _citizen_shares()

    print("\nEurostat census…")
    cen = _census()
    pop = cen[cen["citizen"] == "TOTAL"].groupby("unit")["value"].sum()
    nat = cen[cen["citizen"] == "NAT"].groupby("unit")["value"].sum()
    for_ = cen[cen["citizen"] == "FOR"].groupby("unit")["value"].sum()
    print(f"  {pop.sum():,.0f} people = {nat.sum():,.0f} Belgian citizens "
          f"+ {for_.sum():,.0f} foreign, over {len(pop)} provinces")
    missing = sorted(set(NUTS2) - set(pop.index))
    if missing:
        sys.exit(f"!! census has no rows for {missing}")
    if abs(pop.sum() - POP_2021) > 2_000:
        sys.exit(f"!! census total {pop.sum():,.0f} is not Belgium's {POP_2021:,}")
    print(f"  foreign share runs {100 * (for_ / pop).min():.2f}% "
          f"({(for_ / pop).idxmin()}) to {100 * (for_ / pop).max():.2f}% "
          f"({(for_ / pop).idxmax()}), against {100 * for_.sum() / pop.sum():.2f}% "
          f"nationally; ESS's own non-citizen share is 7.9%, which is why this half exists")

    # ---- citizen half
    rows = []
    for unit in sorted(NUTS2):
        n_cit = float(nat.get(unit, 0.0)) * answered
        if unit not in share.index:
            sys.exit(f"!! no ESS respondents in {unit}")
        for cat, s in share.loc[unit].items():
            if s <= 0:
                continue
            how = ("pooled ESS rounds 5-11, {} respondents".format(
                int(nraw.loc[unit, cat])) if cat in carries else
                "pooled ESS rounds 5-11, national rate in this province's residual "
                "(did not clear the split-half)")
            rows.append((unit, NUTS2[unit], cat, s * n_cit, how))
    cit = pd.DataFrame(rows, columns=["geo_id", "geo_name", "source_category", "count",
                                      "note"])
    cit["geo_level"] = "nuts2"
    cit["basis"] = "self_id"
    cit["year"] = 2024
    cit["source_id"] = "ess_r5_r11_pooled"
    cit = cit[COLUMNS]
    print(f"\n  citizen half: {cit['count'].sum():,.0f} people "
          f"({100 * cit['count'].sum() / nat.sum():.2f}% of citizens; the rest declined)")

    unknown = sorted(set(cit["source_category"]) - set(be2024.MAP))
    if unknown:
        sys.exit(f"!! unmapped source categories: {unknown}")

    # ---- foreign half
    print("foreign half…")
    ext = _foreign_half(cen)
    ext["geo_level"] = "nuts2"
    ext["geo_name"] = ext["geo_id"].map(NUTS2)
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
        cit.assign(node=cit["source_category"].map(be2024.resolve))[["geo_id", "node",
                                                                    "count"]],
        ext[["geo_id", "node", "count"]]], ignore_index=True)
    top = both.groupby("node")["count"].sum().sort_values(ascending=False)
    print("\nnational totals:")
    for node, c in top.items():
        print(f"  {c:>11,.0f}  {100 * c / drawn:5.2f}%  {node}")

    _lapsed()

    print("\nby province, the four largest nodes:")
    per = both.groupby(["geo_id", "node"])["count"].sum().unstack(fill_value=0.0)
    per = per.div(per.sum(axis=1), axis=0)
    cols = list(top.head(4).index)
    print("        " + "".join(f"{c.split('.')[-1][:14]:>16}" for c in cols))
    for u in sorted(NUTS2):
        print(f"  {u} {NUTS2[u][:14]:<16}"
              + "".join(f"{100 * per.loc[u, c]:15.2f}%" for c in cols))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    a = ap.parse_args()
    if a.fetch:
        fetch()
    else:
        build()
