"""Latvia — a survey that asks, a ministry that counts members, and a tenth of the country without citizenship.

Writes data/normalized/lv.csv (citizens and recognised non-citizens) and lv_foreign.csv (foreign
citizens).

Usage:
    python sources/lv.py --fetch     # ESS tabulations, Eurostat census, Pew
    python sources/lv.py             # rebuild from data/raw/lv/

THE OFFICE WAS CHECKED FIRST (§9cu), AND IT COUNTS CONGREGATIONS, NOT PEOPLE. The Central
Statistical Bureau's PxWeb search for `reli*` (2026-09-14) finds one religion table, KUR010,
registered congregations by denomination, national. Its unpublished statistics are a paid
individual service with no public shelf of past orders. Latvia's censuses do not ask religion
(sources.md §11a). The Ministry of Justice publishes the members each religious organisation reports
every year, nationally, by church, with no geography (2025: Lutheran 701,118, Catholic 304,759,
Orthodox 250,240, Old Believers 2,060). That is a `roll` (spec §3.1); build() prints it beside the
survey and it is not drawn. sources/lv.md §1.

SO THE COUNTRY IS THREE PARTS, NOT TWO:

    Latvian citizens          1.64M   ESS rounds 4, 9 and 11 pooled, `ctzcntr = Yes`
    recognised non-citizens   0.19M   ESS rounds 4 and 9, holders of an alien's passport; their
                                      national composition inside each region's count
    foreign citizens          0.06M   Eurostat `cens_21ctz_r3` x Pew's origin compositions

THE NON-CITIZENS ARE INSIDE EUROSTAT'S `FOR`. The census reports 190,544 recognised non-citizens as
`RNC`, and `FOR` (252,305) includes them. Every other two-half country scales its named citizenships
up to `FOR`, which here would multiply Russia, Ukraine and Belarus by four and put Latvia's own
non-citizens on Pew's Russia. So the foreign half's target is `FOR - RNC`, asserted, and the
non-citizens are drawn from the survey that sampled them. Only rounds 4 (`ctzshipb`) and 9
(`ctzshipd`) say which non-citizens hold an alien's passport; rounds 10 and 11 carry `ctzcntr` alone.

WHICH ROUNDS. ESS's country page lists Latvia in rounds 3, 4, 7, 9, 10 and 11.

    3   2006-07  a separate file (`ess3lv`) with the harmonised card only: no Orthodox split, no
                 non-Christian answer, no design weight. Not used.
    4   2008-09  NO `region`, BUT `regionlv`: the same six statistical regions by name. Rounds 1-4
                 carry country-specific region variables, which every earlier ESS country missed by
                 testing `region` alone. ESS publishes no `pspwght` for Latvia, so `dweight`.
    7   2014-15  no Latvian respondent in the integrated file and no `ess7lv` DOI. Not usable.
    9   2018-20  `region`, NUTS 3.
   10   2020-22  the self-completion file, `scrlgblg`. ITS `region` CARRIES NO GEOGRAPHY: Riga is 37%
                 big city (95% in round 11) and every region is about 70% Latvian-speaking and 12%
                 Catholic, Latgale included. `_check_labels` proves it on every build. Printed
                 nationally, not pooled.
   11   2023-24  `region`, NUTS 3 (`regunit` says NUTS level 2, which for Latvia is the country).

THE VARIABLE IS `rlgdnlv`, carried by rounds 4, 9, 10 and 11 and finer than `rlgdnm`: it separates
Lutheran from Baptist and other Protestants, and Russian or Greek Orthodox from other Orthodox, who
in Latvia are the Old Believers. `_check_card` proves it nests in `rlgdnm`.

COUNTED AT THE SIX NUTS 3 REGIONS, which is both where ESS places people and where the census counts
citizenship, so every unit takes its own region's composition and nothing is blended.
"""

import argparse
import io
import json
import os
import sys
import unicodedata
import urllib.request
import zipfile

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "4")

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "taxonomy"))
sys.path.insert(0, HERE)
RAW = os.path.join(ROOT, "data", "raw", "lv")
OUT = os.path.join(ROOT, "data", "normalized", "lv.csv")
OUT_FOREIGN = os.path.join(ROOT, "data", "normalized", "lv_foreign.csv")

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) religiondots"}

# THE STATISTIC IS IMPORTED, NOT COPIED: Norway's `_stability` (Belgium's median split-half Spearman,
# per-round permutation null, Sweden's chi-square gate). Latvia must not run a different test.
import be  # noqa: E402,F401
import no as _no  # noqa: E402

# --- ESS --------------------------------------------------------------------------------
ESS_API = _no.ESS_API
ESS_ROUNDS = {                       # pooled for the citizen half
    4: ("99ba8b91-a921-4a2a-9436-52c536d7ec9d", 70),
    9: ("b2b0bf39-176b-4eca-8d26-3c05ea83d2cb", 280),
    11: ("242aaa39-3bbb-40f5-98bf-bfb1ce53d8ef", 179),
}
ESS_FILES = {**ESS_ROUNDS, 10: ("178d1c16-db15-466e-b1a5-cea36109e089", 146)}
# ROUND 3 IS A WITNESS, NEVER POOLED. A separate Latvia file (`ess3lv`) with the harmonised card only,
# so it sees Protestant and Eastern Orthodox but not Lutheran or the Old Believers, and no design
# weight. It is the one Latvian round outside both the pool and the test, which is what makes it
# evidence about the pool's regional ordering. sources/lv.md §5.
WITNESS_FILES = {3: ("ce8adb47-937b-4f6b-8a47-8b5c5bc89a7f", 8)}
ALL_FILES = {**ESS_FILES, **WITNESS_FILES}
ESS_YEARS = {3: "2006-07", 4: "2008-09", 9: "2018-20", 10: "2020-22", 11: "2023-24"}

# §3.4: the regional pattern comes from ESS_ROUNDS, the national level from these. Round 10 carries a
# usable national composition although its region does not.
LATE_ROUNDS = [9, 10, 11]
RESCALE_TO_LATE = True

REGION_VAR = {3: "regionlv", 4: "regionlv", 9: "region", 10: "region", 11: "region"}
BLG_VAR = {3: "rlgblg", 4: "rlgblg", 9: "rlgblg", 10: "scrlgblg", 11: "rlgblg"}
LANG_VAR = {3: "lnghoma", 4: "lnghoma", 9: "lnghom1", 10: "lnghom1", 11: "lnghom1"}
WEIGHT_VAR = {3: None, 4: "dweight", 9: "pspwght", 10: "pspwght", 11: "pspwght"}
CTZSHIP_VAR = {4: "ctzshipb", 9: "ctzshipd"}
ALIEN_CODE = {4: "65", 9: "6500"}     # "Alien's passport", Latvia's recognised non-citizens
NONCIT_ROUNDS = sorted(CTZSHIP_VAR)

# Round 4's regionlv codes, by name, onto NUTS 3.
REGION_CODE = {r: {"1": "LV003", "2": "LV005", "3": "LV006", "4": "LV007", "5": "LV008",
                   "6": "LV009"} for r in (3, 4)}

# --- Eurostat and Pew ---------------------------------------------------------------------
EU_CTZ = "cens_21ctz_r3"
PEW_ZIP = _no.PEW_ZIP

# --- CSP's population by ethnicity per region, 1 January 2021: a WITNESS for OVERRIDE, never drawn ---
CSP_IRE031 = "https://data.stat.gov.lv/api/v1/lv/OSP_PUB/POP/IR/IRE/IRE031"
CSP_QUERY = {"query": [
    {"code": "ETHNICITY", "selection": {"filter": "item", "values": [
        "TOTAL", "E_LAT", "E_RUS", "E_BRU", "E_UKR", "E_POL"]}},
    {"code": "AREA", "selection": {"filter": "item", "values": [
        "LV003", "LV005", "LV006", "LV007", "LV008", "LV009"]}},
    {"code": "ContentsCode", "selection": {"filter": "item", "values": ["IRE031"]}},
    {"code": "TIME", "selection": {"filter": "item", "values": ["2021"]}}],
    "response": {"format": "csv"}}
# CSP's labels for the six regions as they were until 1 January 2024. On that date Riga and Pieriga
# became one statistical region and Vidzeme and Kurzeme were redrawn under new codes (LV00A, LV00C,
# LV00B). ESS round 11 (2023-24) and the 2021 census both use the earlier six, so no vintage crosses
# in this build; a later round will.
CSP_REGION = {
    "Rīgas statistiskais reģions (Rīga) (līdz 01.01.2024.)": "LV006",
    "Pierīgas statistiskais reģions (līdz 01.01.2024.)": "LV007",
    "Vidzemes statistiskais reģions (līdz 01.01.2024.)": "LV008",
    "Kurzemes statistiskais reģions (līdz 01.01.2024.)": "LV003",
    "Zemgales statistiskais reģions": "LV009",
    "Latgales statistiskais reģions": "LV005",
}

# --- the geography ------------------------------------------------------------------------
NUTS3 = {
    "LV003": "Kurzeme", "LV005": "Latgale", "LV006": "Rīga", "LV007": "Pierīga",
    "LV008": "Vidzeme", "LV009": "Zemgale",
}
RIGA, LATGALE, VIDZEME = "LV006", "LV005", "LV008"

# --- the categories -----------------------------------------------------------------------
NO_RELIGION = "No religion"
REFUSAL = "__refused__"
CATHOLIC = "Catholic"

# rlgdnlv against rlgdnm, the nesting `_check_card` asserts.
CARD_NEST = {
    "Christian, denomination not specified": "Other Christian denomination",
    "Catholic": "Roman Catholic",
    "Lutheran": "Protestant",
    "Other Protestant Denominations": "Protestant",
    "Baptist": "Protestant",
    "Russian or Greek Orthodox": "Eastern Orthodox",
    "Other Orthodox Denominations": "Eastern Orthodox",
    "Other Christian Denominations": "Other Christian denomination",
    "Jewish": "Jewish",
    "Islam": "Islam",
    "Eastern religions": "Eastern religions",
    "Other Non-Christian Religions": "Other Non-Christian religions",
}

COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count",
           "basis", "year", "source_id", "note"]

# Latvia's own totals, asserted so a re-fetch against a new release fails here rather than quietly
# redrawing the map. None prints the value instead.
POP_2021 = 1_893_223          # cens_21ctz_r3 TOTAL = NAT + FOR (incl. RNC) + STLS
RNC_2021 = 190_544
N_CITIZENS = 3_681            # unweighted answered citizens with a region, rounds 4, 9, 11
N_NONCIT = 252                # unweighted answered alien's-passport holders, rounds 4 (184) and 9 (68)

# `No religion` stays the residual, spec §12 (nested units), as in Sweden, Norway and Denmark. Fixed
# at its region's share the tail would not go negative anywhere, so the rule costs nothing.
KEEP_AS_RESIDUAL = {NO_RELIGION}

# What the 6-region test selects, asserted, so a re-fetch that changes a verdict stops the build.
# `Other Christian Denominations` passes the chi-square at 0.0499 on 19 respondents and is the likeliest
# false pass of the eleven tested (Belgium's `Eastern religions`); it maps to the `christianity` root
# and its geography carries nothing.
EXPECT_PASS = {NO_RELIGION, CATHOLIC, "Other Orthodox Denominations", "Baptist",
               "Other Christian Denominations"}

# CATEGORIES DRAWN AGAINST THE RANK TEST, WITH THE REASON PRINTED ON EVERY BUILD (sources/gt.py).
# Both are Denmark's shape, one unit standing apart that a rank correlation over few units cannot
# see, and here the residual construction does not soften that contrast, it reverses it: Latgale's
# non-Catholic remainder is Orthodox and Kurzeme's is Lutheran, so the national-rate tail moves each
# church's people into the other's region. spec §12's rule is to leave such a category alone unless
# an independent witness orders the units; `_witness()` prints the one this build has.
OVERRIDE = {
    "Lutheran": (
        "rank test p = 0.43 at the 6 regions with a spatial chi-square of 4.0e-14. Latgale is "
        "2.0-3.5% Lutheran among citizens in each of rounds 4, 9 and 11 against 6.8% or more "
        "everywhere else, and the other five shuffle between rounds. An independent sample orders "
        "all six the same way: round 3 (2006-07), outside the pool and the test, ranks the regions "
        "by its Protestant answer at Spearman +0.886 (exact p 0.017) against the pool's Lutheran, "
        "Baptist and other Protestant shares. Drawn at the national rate inside the residual, "
        "Latgale came out 7.0% Lutheran against 2.7% measured. sources/lv.md §5."),
    "Russian or Greek Orthodox": (
        "rank test p = 0.0755 at the 6 regions, a near miss, with a spatial chi-square of 3.1e-15. "
        "Riga and Latgale are 11.4% and 11.0% Orthodox among citizens and the other four 2.9-4.9%. "
        "The population register orders the regions nearly the same way: Russians, Belarusians and "
        "Ukrainians as a share of each region on 1 January 2021 (CSP IRE031; Riga 43.1%, Latgale "
        "42.3%, Vidzeme 10.0%) against the pool's Orthodox share, Spearman +0.943 (exact p 0.008), "
        "only Kurzeme and Vidzeme swapped at the bottom. Ethnicity is not religion and counts "
        "non-citizens too, so it is evidence about where the Orthodox are and not how many. Round 3's "
        "Eastern Orthodox answer agrees more weakly (+0.657). Drawn at the national rate inside the "
        "residual, Latgale came out 3.9% Orthodox against 11.0% measured and Kurzeme 7.1% against "
        "2.9%. sources/lv.md §5."),
}

# THE ROLL, printed beside the survey and not drawn. Ministry of Justice, "Ziņojums par Tieslietu
# ministrijā iesniegtajiem reliģisko organizāciju pārskatiem par darbību 2025. gadā", §2.5 and its
# chart of confessions over 1,000 members (tm.gov.lv/lv/media/29203). Members as each organisation
# counts them; the report says so itself.
ROLL_2025 = {
    "christianity.lutheran": 701_118,
    "christianity.catholic": 304_759,
    "christianity.orthodox.canonical": 250_240,
    "christianity.orthodox.oldbeliever": 2_060,
    "christianity.baptist": 6_544,
}
ROLL_TOTAL_2025 = 1_286_431

# The one roll with any geography: the Catholic dioceses' members, from the ministry's 2013 public
# report (§2.5, "kopumā Latvijā ir 389 670 katoļticīgo locekļu"). The dioceses are not the
# statistical regions; the pairing below is approximate (Riga archdiocese is Riga, Pieriga and
# Vidzeme; Jelgava diocese reaches into Selija) and it is printed, never used.
CATHOLIC_DIOCESES_2013 = {
    "Riga archdiocese": (222_910, ["LV006", "LV007", "LV008"]),
    "Liepaja diocese": (28_000, ["LV003"]),
    "Jelgava diocese": (50_760, ["LV009"]),
    "Rezekne-Aglona diocese": (88_000, ["LV005"]),
}


def _key(s):
    return " ".join(str(s).split())


def _fold(s):
    return "".join(ch for ch in unicodedata.normalize("NFKD", str(s))
                   if not unicodedata.combining(ch)).lower()


def _p(name):
    return os.path.join(RAW, name)


# =======================================================================================
# fetch
# =======================================================================================

def _q(weight):
    return _no._TAB % (f' weightVariable:"{weight}",' if weight else "")


def _ess(query, variables):
    body = json.dumps({"query": query, "variables": variables}).encode()
    req = urllib.request.Request(ESS_API, data=body, headers={
        "Content-Type": "application/json", **UA})
    r = json.load(urllib.request.urlopen(req, timeout=900))
    if "errors" in r:
        sys.exit(f"!! ESS API: {r['errors'][0].get('message')} "
                 f"{r['errors'][0].get('extensions', {}).get('code', '')}")
    return r["data"]


def _ess_fetch(rnd, bv, weight, dest):
    if os.path.exists(dest):
        return
    fid, ver = ALL_FILES[rnd]
    d = _ess(_q(weight), {"id": fid, "v": ver, "bv": bv})
    hit = [x for x in d["analysis"]["frequencyTabulationByVariables"]["responses"]
           if x["by"][0]["value"] == "LV"]
    if not hit:
        sys.exit(f"!! ESS round {rnd} has no LV response for {bv}")
    _no._save(hit[0]["response"], dest)


def fetch():
    os.makedirs(RAW, exist_ok=True)

    print("ESS…")
    for rnd in sorted(ESS_FILES):
        reg, blg = REGION_VAR[rnd], BLG_VAR[rnd]
        main = [reg, "ctzcntr", blg, "rlgdnlv"]
        _ess_fetch(rnd, main, None, _p(f"ess_r{rnd}_n.json"))
        _ess_fetch(rnd, main, WEIGHT_VAR[rnd], _p(f"ess_r{rnd}_w.json"))
        _ess_fetch(rnd, [reg, "domicil"], None, _p(f"ess_r{rnd}_domicil.json"))
        _ess_fetch(rnd, [reg, LANG_VAR[rnd]], None, _p(f"ess_r{rnd}_lang.json"))
        if rnd in ESS_ROUNDS:
            _ess_fetch(rnd, ["rlgdnlv", "rlgdnm"], None, _p(f"ess_r{rnd}_card.json"))
        if rnd in CTZSHIP_VAR:
            nb = [CTZSHIP_VAR[rnd], blg, "rlgdnlv"]
            _ess_fetch(rnd, nb, None, _p(f"ess_r{rnd}_noncit_n.json"))
            _ess_fetch(rnd, nb, WEIGHT_VAR[rnd], _p(f"ess_r{rnd}_noncit_w.json"))
        n = sum(c["count"] for c in json.load(
            open(_p(f"ess_r{rnd}_n.json"), encoding="utf-8"))["table"])
        print(f"  round {rnd}: {n:,.0f} respondents")
    for rnd in sorted(WITNESS_FILES):
        reg = REGION_VAR[rnd]
        _ess_fetch(rnd, [reg, "ctzcntr", BLG_VAR[rnd], "rlgdnm"], None, _p(f"ess_r{rnd}_n.json"))
        _ess_fetch(rnd, [reg, "domicil"], None, _p(f"ess_r{rnd}_domicil.json"))
        _ess_fetch(rnd, [reg, LANG_VAR[rnd]], None, _p(f"ess_r{rnd}_lang.json"))
        print(f"  round {rnd} (witness): fetched")

    print("Eurostat census…")
    dest = _p("cens_21ctz_r3_lv.json")
    if not os.path.exists(dest):
        d = _no._eurostat(EU_CTZ, format="JSON", lang="EN", age="TOTAL", sex="T",
                          geo=sorted(NUTS3) + ["LV"])
        _no._save(d, dest)
        print(f"  {os.path.getsize(dest):,} bytes")
    else:
        print("  already on disk")

    print("Pew…")
    dest = _p("pew.zip")
    if not os.path.exists(dest):
        with urllib.request.urlopen(urllib.request.Request(PEW_ZIP, headers=UA),
                                    timeout=600) as r, open(dest + ".tmp", "wb") as f:
            f.write(r.read())
        os.replace(dest + ".tmp", dest)
        print(f"  {os.path.getsize(dest):,} bytes")
    else:
        print("  already on disk")

    print("CSP IRE031, population by ethnicity per region, 2021 (a witness, not drawn)…")
    dest = _p("csp_ire031_2021.csv")
    if not os.path.exists(dest):
        req = urllib.request.Request(CSP_IRE031, data=json.dumps(CSP_QUERY).encode(), headers={
            "Content-Type": "application/json", **UA})
        raw = urllib.request.urlopen(req, timeout=300).read()
        with open(dest + ".tmp", "wb") as f:
            f.write(raw)
        os.replace(dest + ".tmp", dest)
        print(f"  {os.path.getsize(dest):,} bytes")
    else:
        print("  already on disk")


# =======================================================================================
# reading
# =======================================================================================

def _read(path, rnd):
    """One saved ESS response as a tidy frame, with the per-round variable names normalised.

    region -> NUTS 3 code (round 4 recoded) plus `region_label`; the belonging question -> rlgblg;
    the home language -> lang (label); the citizenship -> ctzship (code) plus `ctzship_label`;
    everything else as a LABEL. `path` indexes into `codeList` (fi.py's trap).
    """
    d = json.load(open(path, encoding="utf-8"))
    order = [v["name"] for v in d["variableValues"]]
    codes = {v["name"]: v["codeList"] for v in d["variableValues"]}
    rename = {REGION_VAR[rnd]: "region", BLG_VAR[rnd]: "rlgblg", LANG_VAR[rnd]: "lang"}
    if rnd in CTZSHIP_VAR:
        rename[CTZSHIP_VAR[rnd]] = "ctzship"
    rows = []
    for cell in d["table"]:
        rec = {"round": rnd, "count": float(cell["count"])}
        for i, n in enumerate(order):
            c = codes[n][cell["path"][i]]
            k = rename.get(n, n)
            if k == "region":
                rec[k] = REGION_CODE.get(rnd, {}).get(c["value"], c["value"])
                rec["region_label"] = _key(c["label"])
            elif k == "ctzship":
                rec[k] = str(c["value"])
                rec["ctzship_label"] = _key(c["label"])
            elif k == "rlgblg":
                # Round 10's self-completion `scrlgblg` splits No into "No, never" and "No, but did
                # in the past"; both are No to the question every other round asks.
                lab = _key(c["label"])
                rec[k] = "No" if (not c["isMissing"] and lab.startswith("No,")) else lab
            else:
                rec[k] = _key(c["label"])
            rec[k + "_miss"] = bool(c["isMissing"])
        rows.append(rec)
    return pd.DataFrame(rows)


def _cat(df):
    return np.where(
        df["rlgblg"] == "No", NO_RELIGION,
        np.where((df["rlgblg"] != "Yes") | df["rlgdnlv_miss"], REFUSAL, df["rlgdnlv"]))


def _pool(rounds, tag):
    """Citizens with a region, category resolved (refusals kept as REFUSAL)."""
    df = pd.concat([_read(_p(f"ess_r{r}_{tag}.json"), r) for r in sorted(rounds)],
                   ignore_index=True)
    df = df[(df["ctzcntr"] == "Yes") & (~df["region_miss"])].copy()
    df["cat"] = _cat(df)
    return df


def _check_level():
    """The region code set of every round, and its labels, asserted (Italy's lesson, §9as)."""
    for rnd in sorted(ALL_FILES):
        d = _read(_p(f"ess_r{rnd}_n.json"), rnd)
        blg = sorted(set(d.loc[~d["rlgblg_miss"], "rlgblg"]))
        if blg != ["No", "Yes"]:
            sys.exit(f"!! round {rnd}: {BLG_VAR[rnd]} answers {blg}")
        d = d[~d["region_miss"]]
        if set(d["region"]) != set(NUTS3):
            sys.exit(f"!! round {rnd}: regions {sorted(set(d['region']))}, expected {sorted(NUTS3)}")
        for code, lab in sorted(set(zip(d["region"], d["region_label"]))):
            if _fold(NUTS3[code]) not in _fold(lab) and _fold(lab) not in _fold(NUTS3[code]):
                sys.exit(f"!! round {rnd}: {code} is labelled {lab!r}, expected {NUTS3[code]}")
    print(f"  rounds {sorted(ALL_FILES)}: the 6 regions in every round, labels agree with NUTS 3 "
          "(rounds 3 and 4's regionlv recoded by name)")


def _profile(rnd):
    def share(fname, col, value):
        d = _read(_p(fname), rnd)
        d = d[~d["region_miss"] & ~d[col + "_miss"]]
        t = d.groupby("region")["count"].sum()
        s = d[d[col] == value].groupby("region")["count"].sum() / t
        return s.reindex(sorted(NUTS3)).fillna(0.0), d
    big, _ = share(f"ess_r{rnd}_domicil.json", "domicil", "A big city")
    rus, lang = share(f"ess_r{rnd}_lang.json", "lang", "Russian")
    if rnd in WITNESS_FILES:
        cath, main = share(f"ess_r{rnd}_n.json", "rlgdnm", "Roman Catholic")
    else:
        cath, main = share(f"ess_r{rnd}_n.json", "rlgdnlv", CATHOLIC)
    # rlgdnlv's missing flag marks everyone who does not belong; the Catholic share above is
    # therefore of people who named a denomination, which is what separates Latgale.
    t = main.groupby("region")["count"].sum()
    return big, rus, cath, t / t.sum(), lang


def _check_labels():
    """EVERY ROUND'S REGIONS CHECKED AGAINST THE PEOPLE BEHIND THEM (Denmark's round 9, §9dg).

    Riga is the region most often called a big city; Latgale is the most Russian-speaking outside
    Riga and the most Catholic; Vidzeme is the least Russian-speaking. Every pooled round must pass,
    and ROUND 10 MUST FAIL: its `region` carries no geography, so if a re-fetch makes it pass, ESS
    has fixed the file and the round belongs in the pool.
    """
    from scipy.stats import chi2_contingency

    def ok(big, rus, cath):
        return (big.idxmax() == RIGA and rus.drop(RIGA).idxmax() == LATGALE
                and rus.idxmin() == VIDZEME and cath.idxmax() == LATGALE)

    for rnd in sorted(ALL_FILES):
        big, rus, cath, sample, lang = _profile(rnd)
        tab = lang.assign(r=lang["lang"] == "Russian").groupby(["region", "r"])["count"] \
            .sum().unstack(fill_value=0.0)
        chi = chi2_contingency(tab.values)[1]
        print(f"  round {rnd} ({ESS_YEARS[rnd]}): big city " + ", ".join(
            f"{NUTS3[r]} {100 * big[r]:.0f}%" for r in big.index))
        print("      Russian at home " + ", ".join(f"{NUTS3[r]} {100 * rus[r]:.0f}%" for r in rus.index)
              + f"  (chi-square across regions p = {chi:.1e})")
        print("      Catholic, of denominations named " + ", ".join(
            f"{NUTS3[r]} {100 * cath[r]:.0f}%" for r in cath.index))
        print("      sample " + ", ".join(f"{NUTS3[r]} {100 * sample[r]:.0f}%" for r in sample.index))
        if (rnd in ESS_ROUNDS or rnd in WITNESS_FILES) and not ok(big, rus, cath):
            sys.exit(f"!! round {rnd}: the regions no longer look like Latvia's; check the labels")
        if rnd not in ESS_ROUNDS and rnd not in WITNESS_FILES:
            if ok(big, rus, cath):
                sys.exit(f"!! round {rnd}'s regions now pass the label check; ESS has fixed the "
                         "file and it can be pooled")
            print(f"      round {rnd} FAILS, as expected: its region carries no geography")


def _check_card():
    """`rlgdnlv` nests in `rlgdnm`, proved from each pooled round's cross-tab."""
    for rnd in sorted(ESS_ROUNDS):
        d = _read(_p(f"ess_r{rnd}_card.json"), rnd)
        d = d[d["count"] > 0]
        seen = set()
        for _, r in d.iterrows():
            a, b = r["rlgdnlv"], r["rlgdnm"]
            if r["rlgdnlv_miss"] or r["rlgdnm_miss"]:
                if r["rlgdnlv_miss"] != r["rlgdnm_miss"]:
                    sys.exit(f"!! round {rnd}: `{a}` / `{b}` missing on one card only")
                continue
            if CARD_NEST.get(a) != b:
                sys.exit(f"!! round {rnd}: rlgdnlv `{a}` sits in rlgdnm `{b}` for "
                         f"{r['count']:.0f} respondents")
            seen.add(a)
        print(f"  round {rnd}: rlgdnlv nests in rlgdnm, {len(seen)} answers given")


# =======================================================================================
# the citizen part
# =======================================================================================

def _assert_source(pool, label, both_ways=True):
    import lv2024
    unknown = sorted(set(pool["cat"]) - lv2024.SOURCE)
    if unknown:
        sys.exit(f"!! {label}: source categories with no mapping: {unknown}")
    vanished = sorted(lv2024.SOURCE - set(pool["cat"]))
    if both_ways and vanished:
        sys.exit(f"!! {label}: lv2024.SOURCE categories nobody answered: {vanished}")


def _composition(df):
    t = df.groupby("cat")["count"].sum()
    return t / t.sum()


def _citizen_shares():
    raw = _pool(ESS_ROUNDS, "n")
    wtd = _pool(ESS_ROUNDS, "w")
    answered = 1.0 - wtd.loc[wtd["cat"] == REFUSAL, "count"].sum() / wtd["count"].sum()
    print(f"  {raw['count'].sum():,.0f} citizens with a region; {100 * (1 - answered):.2f}% "
          "declined (weighted) and are not drawn")
    raw, wtd = raw[raw["cat"] != REFUSAL], wtd[wtd["cat"] != REFUSAL]
    n = int(round(raw["count"].sum()))
    if N_CITIZENS is None:
        print(f"  !! N_CITIZENS is unset; this build has {n:,}")
    elif n != N_CITIZENS:
        sys.exit(f"!! {n:,} answered citizens, expected {N_CITIZENS:,}")
    _assert_source(wtd, "rounds 4, 9, 11")

    print("  by round, citizens, weighted (round 10 nationally only):")
    r10 = _pool([10], "w")
    r10 = r10[r10["cat"] != REFUSAL]
    per = {r: _composition(wtd[wtd["round"] == r]) for r in sorted(ESS_ROUNDS)}
    per[10] = _composition(r10)
    per = dict(sorted(per.items()))
    nat0 = _composition(wtd)
    print("    " + " " * 40 + "".join(f"{'r' + str(r) + ' ' + ESS_YEARS[r]:>13}" for r in per)
          + f"{'pooled':>10}")
    for c in nat0.sort_values(ascending=False).index:
        print(f"    {c[:38]:<40}" + "".join(f"{100 * per[r].get(c, 0):12.2f}%" for r in per)
              + f"{100 * nat0[c]:9.2f}%")
    nraw = {r: int(raw[raw["round"] == r]["count"].sum()) for r in sorted(ESS_ROUNDS)}
    print("    answered citizens: " + ", ".join(f"r{r} {v:,}" for r, v in nraw.items())
          + f", r10 {int(_pool([10], 'n').query('cat != @REFUSAL')['count'].sum()):,}")

    tab = wtd.groupby(["region", "cat"])["count"].sum().unstack(fill_value=0.0)
    tab = tab.reindex(index=sorted(NUTS3), fill_value=0.0)
    share = tab.div(tab.sum(axis=1), axis=0)
    nat = tab.sum() / tab.sum().sum()
    cats = sorted(nat.index, key=lambda c: -nat[c])
    share = share.reindex(columns=cats, fill_value=0.0)
    passed = _no._stability(raw, cats, ESS_ROUNDS, NUTS3, "6 statistical regions")

    if EXPECT_PASS is not None and set(passed) != EXPECT_PASS:
        sys.exit(f"!! the 6-region test now selects {sorted(passed)}, not {sorted(EXPECT_PASS)}; "
                 "OVERRIDE, note_public and sources/lv.md quote the old verdicts")
    for c, why in OVERRIDE.items():
        if c in passed:
            sys.exit(f"!! `{c}` now passes on its own; take it out of OVERRIDE")
        print(f"\n  OVERRIDE, drawn at the 6 regions against the test: `{c}`\n    {why}")
    fine = [c for c in cats if (c in passed or c in OVERRIDE) and c not in KEEP_AS_RESIDUAL]
    small = [c for c in cats if c not in fine]
    nraw_c = raw.groupby("cat")["count"].sum()
    print(f"\n  {len(fine)} categories at the 6 regions, {len(small)} at the national rate inside "
          "each region's residual")
    for c in cats:
        print(f"    {100 * nat[c]:>6.2f}%  {int(nraw_c.get(c, 0)):>6,}  {c}   "
              f"[{'6 regions' if c in fine else 'national rate (residual)'}]")
    print("\n  shares by region, citizens, weighted:")
    print("    " + " " * 34 + "".join(f"{NUTS3[r][:9]:>11}" for r in share.index))
    for c in cats:
        print(f"    {c[:32]:<34}" + "".join(f"{100 * share.loc[r, c]:10.2f}%" for r in share.index))
    return share, nat, fine, small, answered, n


def _late_level():
    """The national citizen composition of LATE_ROUNDS, weighted, refusals out."""
    late = pd.concat([_pool([r], "w") for r in LATE_ROUNDS], ignore_index=True)
    late = late[late["cat"] != REFUSAL]
    return _composition(late)


def _rescale(share, fine, nat_late, cit_pop):
    """§3.4, NORWAY'S CONSTRUCTION: each fine category keeps the pool's regional pattern and is
    scaled by one factor to its LATE_ROUNDS national share, weighted by the census's citizens per
    region (not by the survey's own regional mix, spec §12's Nigeria entry). A factor and not a
    shift, because between round 4 and rounds 9+11 the regional ratios vary less than the point
    differences (coefficient of variation 0.24 against 0.54 for No religion, 0.24 against 1.11 for
    Catholic, 0.34 against 0.91 for Lutheran; sources/lv.md §6)."""
    w = cit_pop.reindex(share.index) / cit_pop.reindex(share.index).sum()
    out = share.copy()
    print(f"  §3.4: fine categories scaled to rounds {LATE_ROUNDS}' national level "
          "(implied by the pool at census weights -> late):")
    for c in fine:
        implied = float((share[c] * w).sum())
        target = float(nat_late.get(c, 0.0))
        f = target / implied if implied > 0 else 1.0
        out[c] = share[c] * f
        print(f"    {c[:38]:<40}{100 * implied:7.2f}% -> {100 * target:6.2f}%   x{f:.4f}")
    if (out[fine].sum(axis=1) >= 1.0).any():
        sys.exit("!! the scaled fine categories fill a whole region")
    return out


def _witness(share):
    """ROUND 3 AGAINST THE POOL, region by region. Printed evidence for OVERRIDE, not a gate.

    Round 3 (2006-07) is outside the pool and the test and carries only the harmonised card, so the
    pool's answers are summed to what round 3 can see. Spearman over the 6 regions, with an exact
    permutation p over all 720 orderings.
    """
    import itertools
    from scipy.stats import spearmanr
    d = _read(_p("ess_r3_n.json"), 3)
    d = d[(d["ctzcntr"] == "Yes") & ~d["region_miss"]].copy()
    d["cat"] = np.where(d["rlgblg"] == "No", NO_RELIGION,
                        np.where((d["rlgblg"] != "Yes") | d["rlgdnm_miss"], REFUSAL, d["rlgdnm"]))
    d = d[d["cat"] != REFUSAL]
    t = d.groupby("region")["count"].sum().reindex(sorted(NUTS3))
    groups = {
        "Protestant": ["Lutheran", "Baptist", "Other Protestant Denominations"],
        "Eastern Orthodox": ["Russian or Greek Orthodox", "Other Orthodox Denominations"],
        "Roman Catholic": [CATHOLIC],
        NO_RELIGION: [NO_RELIGION],
    }
    print(f"\n  WITNESS, round 3 (2006-07, {int(t.sum()):,} answered citizens, unweighted) against the "
          "pool's regional shares:")
    out = {}
    for r3cat, members in groups.items():
        a = (d[d["cat"] == r3cat].groupby("region")["count"].sum().reindex(sorted(NUTS3)).fillna(0)
             / t)
        b = share[[m for m in members if m in share.columns]].sum(axis=1).reindex(sorted(NUTS3))
        rho = spearmanr(a, b).correlation
        perms = [spearmanr(a.values, np.array(p)).correlation
                 for p in itertools.permutations(b.values)]
        p = sum(1 for x in perms if x >= rho - 1e-12) / len(perms)
        out[r3cat] = (rho, p)
        print(f"    {r3cat:<18} rho {rho:+.3f}  exact p {p:.4f}   r3: " + ", ".join(
            f"{NUTS3[u][:7]} {100 * a[u]:.1f}%" for u in a.index))

    # The population register: Russians, Belarusians and Ukrainians as a share of each region. Not
    # religion, and it counts non-citizens too; it says where the Orthodox are, not how many.
    eth = pd.read_csv(_p("csp_ire031_2021.csv"), encoding="utf-8")
    eth.columns = ["ethnicity", "area", "value"]
    eth["unit"] = eth["area"].map(CSP_REGION)
    if eth["unit"].isna().any():
        sys.exit(f"!! CSP region labels not recognised: {sorted(set(eth.loc[eth['unit'].isna(), 'area']))}")
    piv = eth.pivot_table(index="unit", columns="ethnicity", values="value", aggfunc="sum") \
        .reindex(sorted(NUTS3))
    slav = (piv["Krievi"] + piv["Baltkrievi"] + piv["Ukraiņi"]) / piv["Pavisam"]
    orth = share["Russian or Greek Orthodox"].reindex(sorted(NUTS3))
    rho = spearmanr(slav, orth).correlation
    perms = [spearmanr(slav.values, np.array(q)).correlation for q in itertools.permutations(orth.values)]
    p = sum(1 for x in perms if x >= rho - 1e-12) / len(perms)
    out["ethnicity"] = (rho, p)
    print(f"\n  WITNESS, CSP IRE031 (1 January 2021, {int(piv['Pavisam'].sum()):,} people): Russians, "
          "Belarusians and Ukrainians per region against the pool's `Russian or Greek Orthodox`:")
    print(f"    rho {rho:+.3f}  exact p {p:.4f}   " + ", ".join(
        f"{NUTS3[u][:7]} {100 * slav[u]:.1f}% / {100 * orth[u]:.2f}%" for u in slav.index))
    return out


def _compose(share, nat, fine, small):
    """Per region: a fine category takes its region's share; the rest share the residual at the
    national proportions (§9bi)."""
    out = pd.DataFrame(index=sorted(NUTS3), columns=list(nat.index), dtype=float)
    for c in fine:
        out[c] = share[c].astype(float)
    residual = 1.0 - out[fine].sum(axis=1) if fine else pd.Series(1.0, index=out.index)
    for c in sorted(KEEP_AS_RESIDUAL & set(small)):
        alt = residual - share[c].astype(float)
        print(f"  if `{c}` were fixed at its region's share, the tail left would be negative in "
              f"{int((alt < 0).sum())} of {len(alt)} regions (worst {alt.min():+.2%})")
    if (residual <= 0).any():
        sys.exit(f"!! regions with no room for the tail: {sorted(residual[residual <= 0].index)}")
    tot = float(nat[small].sum())
    for c in small:
        out[c] = residual * (float(nat[c]) / tot)
    print("  residual categories as drawn among citizens (the survey's own regional share in "
          "brackets):")
    for c in small[:4]:
        print(f"    {c[:30]:<32}" + "  ".join(
            f"{NUTS3[u][:7]} {100 * out.loc[u, c]:.2f}% ({100 * float(share.loc[u, c]):.2f}%)"
            for u in out.index))
    if (out.sum(axis=1) - 1.0).abs().max() > 1e-9:
        sys.exit("!! composition does not sum to 1")
    return out


# =======================================================================================
# the non-citizen part
# =======================================================================================

def _noncitizens():
    """National composition of Latvia's recognised non-citizens, rounds 4 and 9.

    ESS asks non-citizens which citizenship they hold, and Latvia's non-citizens answer with the
    code for an alien's passport. Rounds 10 and 11 do not ask, so they cannot contribute. The
    respondents are too few for any regional test, so the composition is national.
    """
    frames = {"n": [], "w": []}
    for r in NONCIT_ROUNDS:
        for tag in frames:
            d = _read(_p(f"ess_r{r}_noncit_{tag}.json"), r)
            labels = set(d.loc[d["ctzship"] == ALIEN_CODE[r], "ctzship_label"])
            if not labels or not all("alien" in _fold(x) for x in labels):
                sys.exit(f"!! round {r}: code {ALIEN_CODE[r]} is labelled {labels}, not an "
                         "alien's passport")
            d = d[d["ctzship"] == ALIEN_CODE[r]].copy()
            d["cat"] = _cat(d)
            frames[tag].append(d)
    raw = pd.concat(frames["n"], ignore_index=True)
    wtd = pd.concat(frames["w"], ignore_index=True)
    answered = 1.0 - wtd.loc[wtd["cat"] == REFUSAL, "count"].sum() / wtd["count"].sum()
    raw, wtd = raw[raw["cat"] != REFUSAL], wtd[wtd["cat"] != REFUSAL]
    n = int(round(raw["count"].sum()))
    if N_NONCIT is None:
        print(f"  !! N_NONCIT is unset; this build has {n:,}")
    elif n != N_NONCIT:
        sys.exit(f"!! {n:,} answered non-citizens, expected {N_NONCIT:,}")
    _assert_source(wtd, "non-citizens, rounds 4 and 9", both_ways=False)
    comp = _composition(wtd)
    per = {r: int(raw[raw["round"] == r]["count"].sum()) for r in NONCIT_ROUNDS}
    print(f"  {n:,} answered holders of an alien's passport ({per}); "
          f"{100 * (1 - answered):.2f}% declined (weighted)")
    return comp, answered, n


# =======================================================================================
# the foreign part
# =======================================================================================

def _census():
    d = json.load(open(_p("cens_21ctz_r3_lv.json"), encoding="utf-8"))
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
    df = df[df["geo"].isin(NUTS3)].copy()
    df["unit"] = df["geo"]
    return df


def _foreign_half(cen):
    """[geo_id, node, count] for foreign citizens at NUTS 3, WITH THE NON-CITIZENS TAKEN OUT."""
    import origin_religion as origin

    with zipfile.ZipFile(_p("pew.zip")) as z:
        name = [n for n in z.namelist() if n.endswith("(percentages).csv")][0]
        pew = pd.read_csv(io.BytesIO(z.read(name)))
    pew = pew[(pew["Year"] == 2020) & (pew["Level"] == 1)].set_index("Country")
    fams = origin.FAMILIES

    by = lambda code: cen[cen["citizen"] == code].groupby("unit")["value"].sum()
    target = (by("FOR") - by("RNC")).reindex(sorted(NUTS3)).fillna(0.0)
    leaf = cen[cen["citizen"].str.fullmatch(r"[A-Z]{2}") & (cen["citizen"] != "LV")]
    covered = leaf["value"].sum()
    print(f"  {leaf['citizen'].nunique()} named citizenships cover {covered:,.0f} of "
          f"{target.sum():,.0f} foreign citizens once the {by('RNC').sum():,.0f} recognised "
          f"non-citizens are taken out of FOR ({100 * covered / target.sum():.2f}%)")
    if not 0.9 < covered / target.sum() <= 1.0001:
        sys.exit("!! the named citizenships do not match FOR - RNC; the non-citizens are not where "
                 "this build thinks they are")

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
        comp[iso] = origin.composition(iso, row, "other.lv")
    if unmapped:
        sys.exit(f"!! {len(unmapped)} citizenships have no composition: {unmapped}")

    rows = []
    for unit, g in leaf.groupby("unit"):
        base = g["value"].sum()
        if base <= 0:
            continue
        scale = float(target[unit]) / base
        acc = {}
        for iso, v in g.groupby("citizen")["value"].sum().items():
            if v <= 0:
                continue
            for node, s in comp[iso].items():
                acc[node] = acc.get(node, 0.0) + v * scale * s
        for node, c in acc.items():
            rows.append((unit, node, c))
    df = pd.DataFrame(rows, columns=["geo_id", "node", "count"])
    print(f"  foreign part: {df['count'].sum():,.0f} people, {df['node'].nunique()} nodes")
    return df


# =======================================================================================
# build
# =======================================================================================

def build():
    import lv2024

    print("checking the regions, their labels against the people, and the Latvian card…")
    _check_level()
    _check_labels()
    _check_card()

    print("\nESS, citizens…")
    share, nat, fine, small, answered, n = _citizen_shares()

    print("\nESS, recognised non-citizens…")
    nc_comp, nc_answered, nc_n = _noncitizens()
    print(f"    {'':<40}{'non-citizens':>13}{'citizens':>10}")
    for c in nc_comp.sort_values(ascending=False).index.union(nat.index, sort=False):
        print(f"    {c[:38]:<40}{100 * nc_comp.get(c, 0):12.2f}%{100 * nat.get(c, 0):9.2f}%")

    print("\nEurostat census…")
    cen = _census()
    by = lambda code: cen[cen["citizen"] == code].groupby("unit")["value"].sum() \
        .reindex(sorted(NUTS3)).fillna(0.0)
    pop, nat_cit, for_, rnc, stls = by("TOTAL"), by("NAT"), by("FOR"), by("RNC"), by("STLS")
    unk = by("UNK")
    print(f"  {pop.sum():,.0f} people = {nat_cit.sum():,.0f} citizens + {rnc.sum():,.0f} recognised "
          f"non-citizens + {for_.sum() - rnc.sum():,.0f} foreign citizens + {stls.sum():,.0f} "
          f"stateless + {unk.sum():,.0f} unknown")
    if (pop - nat_cit - for_ - stls - unk).abs().max() > 1:
        sys.exit("!! NAT + FOR + STLS + UNK is not TOTAL; FOR may no longer include RNC")
    if POP_2021 is not None and int(pop.sum()) != POP_2021:
        sys.exit(f"!! census total {pop.sum():,.0f} is not Latvia's {POP_2021:,}")
    if RNC_2021 is not None and int(rnc.sum()) != RNC_2021:
        sys.exit(f"!! RNC {rnc.sum():,.0f} is not {RNC_2021:,}")
    print("    region        citizens  non-citizens  foreign")
    for u in sorted(NUTS3):
        print(f"    {NUTS3[u]:<10}{100 * nat_cit[u] / pop[u]:10.2f}%{100 * (rnc[u] + stls[u]) / pop[u]:12.2f}%"
              f"{100 * (for_[u] - rnc[u]) / pop[u]:9.2f}%")

    _witness(share)

    print("\ncomposing the citizen part…")
    nat_late = _late_level().reindex(nat.index).fillna(0.0)
    if RESCALE_TO_LATE:
        share = _rescale(share, fine, nat_late, nat_cit)
        comp = _compose(share, nat_late, fine, small)
    else:
        comp = _compose(share, nat, fine, small)
    w_cit = nat_cit / nat_cit.sum()
    drawn_nat = comp.mul(w_cit, axis=0).sum()
    print("  citizen composition as drawn, nationally, against the late rounds:")
    for c in drawn_nat.sort_values(ascending=False).index[:8]:
        print(f"    {c[:38]:<40}{100 * drawn_nat[c]:7.2f}%  {100 * nat_late.get(c, 0):7.2f}%")

    rows = []
    pool_note = ("ESS rounds 4, 9 and 11 at NUTS 3 (round 4's regionlv recoded by name)"
                 + (f"; scaled to rounds {LATE_ROUNDS}' national level (§3.4)"
                    if RESCALE_TO_LATE else ""))
    for u in sorted(NUTS3):
        n_cit = float(nat_cit[u]) * answered
        for cat, s in comp.loc[u].items():
            if s <= 0:
                continue
            if cat in fine:
                note = pool_note + ("; drawn by OVERRIDE, see sources/lv.py" if cat in OVERRIDE else "")
            else:
                note = ("share of the region's residual at the national proportions "
                        "(§9bi, split-half not passed)")
            for sub, w in lv2024.split(cat).items():
                rows.append((u, NUTS3[u], sub, s * w * n_cit, note, "ess_r4_r9_r11_nuts3",
                             "citizens"))
        n_nc = float(rnc[u] + stls[u]) * nc_answered
        for cat, s in nc_comp.items():
            if s <= 0:
                continue
            note = ("recognised non-citizens and stateless people: the national composition of "
                    "alien's-passport holders in ESS rounds 4 and 9")
            for sub, w in lv2024.split(cat).items():
                rows.append((u, NUTS3[u], sub, s * w * n_nc, note, "ess_r4_r9_noncitizens",
                             "noncitizens"))
    cit = pd.DataFrame(rows, columns=["geo_id", "geo_name", "source_category", "count", "note",
                                      "source_id", "part"])
    cit["geo_level"] = "nuts3"
    cit["basis"] = "self_id"
    cit["year"] = 2021
    parts = cit.groupby("part")["count"].sum()
    cit = cit[COLUMNS]
    print(f"  citizen part: {parts.get('citizens', 0):,.0f} people "
          f"({100 * parts.get('citizens', 0) / nat_cit.sum():.2f}% of citizens; the rest declined)")
    print(f"  non-citizen part: {parts.get('noncitizens', 0):,.0f} people "
          f"({100 * parts.get('noncitizens', 0) / (rnc.sum() + stls.sum()):.2f}% of non-citizens "
          "and stateless)")
    unknown = sorted(set(cit["source_category"]) - set(lv2024.MAP))
    if unknown:
        sys.exit(f"!! unmapped source categories: {unknown}")

    print("\nforeign part…")
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
    print(f"drawn {drawn:,.0f} of {pop.sum():,.0f}, {100 * drawn / pop.sum():.2f}%")

    both = pd.concat([
        cit.assign(node=cit["source_category"].map(lv2024.resolve))[["geo_id", "node", "count"]],
        ext[["geo_id", "node", "count"]]], ignore_index=True)
    top = both.groupby("node")["count"].sum().sort_values(ascending=False)
    print("\nnational totals:")
    for node, c in top.head(22).items():
        print(f"  {c:>11,.0f}  {100 * c / drawn:5.2f}%  {node}")

    byu = both.groupby(["geo_id", "node"])["count"].sum().unstack(fill_value=0.0)
    tot = byu.sum(axis=1)
    col = lambda k: byu[k] if k in byu.columns else pd.Series(0.0, index=byu.index)
    fam = lambda pre: byu[[c for c in byu.columns if c == pre or c.startswith(pre + ".")]].sum(axis=1)
    print("\nper region (drawn, all three parts):")
    print(f"    {'region':<10}{'unaffil':>9}{'catholic':>9}{'lutheran':>9}{'orthodox':>9}"
          f"{'oldbel':>8}{'islam':>7}{'noncit':>8}{'foreign':>8}")
    for u in sorted(NUTS3):
        t = tot[u]
        print(f"    {NUTS3[u]:<10}{100 * col('unaffiliated')[u] / t:8.2f}%"
              f"{100 * fam('christianity.catholic')[u] / t:8.2f}%"
              f"{100 * col('christianity.lutheran')[u] / t:8.2f}%"
              f"{100 * col('christianity.orthodox.canonical')[u] / t:8.2f}%"
              f"{100 * col('christianity.orthodox.oldbeliever')[u] / t:7.2f}%"
              f"{100 * fam('islam')[u] / t:6.2f}%"
              f"{100 * (rnc[u] + stls[u]) / pop[u]:7.2f}%{100 * (for_[u] - rnc[u]) / pop[u]:7.2f}%")

    # ---- THE ROLL BESIDE THE SURVEY. Not a validation (sources/se.md §4, sources/no.md §8).
    print(f"\nTHE ROLL (Ministry of Justice, members reported for 2025) BESIDE THE MAP, as shares of "
          f"the census's {pop.sum():,.0f}:")
    print(f"  all reported members: {ROLL_TOTAL_2025:,}, {100 * ROLL_TOTAL_2025 / pop.sum():.1f}%")
    for node, m in ROLL_2025.items():
        drawn_n = fam(node).sum() if node == "christianity.catholic" else col(node).sum()
        print(f"  {node:<36} roll {m:>9,} {100 * m / pop.sum():5.1f}%   map "
              f"{drawn_n:>9,.0f} {100 * drawn_n / drawn:5.1f}%")

    from scipy.stats import spearmanr
    print("\n  Catholic dioceses, 2013 roll, against the survey's Catholic share among citizens "
          "(population from the 2021 census; the pairing of dioceses to regions is approximate):")
    xs, ys = [], []
    for name, (m, regions) in CATHOLIC_DIOCESES_2013.items():
        p = float(pop[regions].sum())
        w = nat_cit[regions] / nat_cit[regions].sum()
        s = float((share.loc[regions, CATHOLIC] * w).sum()) if CATHOLIC in share.columns else 0.0
        xs.append(m / p)
        ys.append(s)
        print(f"    {name:<24} roll {m:>8,}  {100 * m / p:5.1f}% of {p:>9,.0f}   survey {100 * s:5.1f}%")
    print(f"    Spearman over 4 dioceses: {spearmanr(xs, ys).correlation:+.3f} (4 units; an ordering, "
          "not a test)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    a = ap.parse_args()
    if a.fetch:
        fetch()
    else:
        build()
