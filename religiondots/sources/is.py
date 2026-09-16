"""Iceland: a register of every faith body, and a survey that asks what people are.

Writes data/normalized/is.csv (citizens) and is_foreign.csv (foreign residents).

Usage:
    python sources/is.py --fetch     # ESS tabulations, Eurostat census, Pew, Hagstofa's register
    python sources/is.py             # rebuild from data/raw/is/

THE OFFICE WAS CHECKED FIRST, AND WHAT IT PUBLISHES IS A REGISTER (sources.md
§scout-2026-09-15-europe). Hagstofa Islands tabulates Registers Iceland's record of the faith or
life-stance body each resident is registered in:

    MAN10001   every registered body, national only, 1998-2026: about sixty bodies, plus
               `Önnur trúfélög og ótilgreint` (other bodies and unspecified) and `Utan trú- og
               lífsskoðunarfélaga` (outside every body)
    MAN10289   population by parish, clergy district and deanery, 1 December 2023, split only
    ...10302   into in and not in the Church of Iceland, for people 16 and over (2010-2023)

**That is a `roll` (spec §3.1) and this map does not draw it for Iceland**, for Norway's and
Denmark's reasons (sources/no.md §1, sources/dk.md §1): one church by place and every other body
only nationally, and a membership count beside neighbours drawn from what people say. build()
prints the roll beside the survey, nationally and at the same two units.

SO THE TWO HALVES ARE DENMARK'S AND NORWAY'S:

    Icelandic citizens   312k   ESS rounds 6, 8, 10 and 11 pooled, `ctzcntr = Yes`
    foreign residents     47k   Eurostat `cens_21ctz_r3` x Pew's origin compositions

ICELAND IS IN ESS ROUNDS 2, 6, 8, 9, 10 AND 11, AND ITS `region` IS TWO UNITS. Round 2 carries
`regionis` with the single value Iceland; rounds 3-5 and 7 are absent. Rounds 6-11 give NUTS 3
2016/2021, which for Iceland is IS001 Höfuðborgarsvæði (the capital area, seven municipalities) and
IS002 Landsbyggð (everything else). Nothing finer exists in any round.

**ROUND 9's `region` IS NOT THE NUTS 3 SPLIT, AND IT IS LEFT OUT.** Its IS001 is 79% of the sample
where the other rounds put 59-67% there (the capital area is 64% of Iceland), and its IS002 is 30%
village or farm and 22% town, where every other round's IS002 is 7-16% village or farm and 48-62%
town. So the towns outside the capital area were coded IS001. Two codes cannot be recoded the way
Denmark's round 9 was (sources/dk.md §3), so round 9 is in no pool; `_check_round9` asserts the
published file still fails the domicile profile, and its national shares are printed.

THE CARD IS ICELANDIC AND CHANGES BETWEEN ROUNDS. `rlgdnis` (rounds 6, 8) and `rlgdnais` (9-11)
both split the harmonised `Protestant` into the Church of Iceland, the Free Church and other
Lutheran bodies, and name Asatru. The later card replaces `Russian Orthodox Church` with
`Orthodox Church` and `Muslim Association of Iceland` with `Islam`; those two are harmonised to the
later label. `_check_card` proves every answer's nesting in `rlgdnm` from the fetched cross-tab.

TWO UNITS, SO THE RANK SPLIT-HALF CANNOT DECIDE ANYTHING. A Spearman over two units is +1 or -1.
It is run and printed (no.py's `_stability`) and decides nothing. The deciding test is Uzbekistan's
two-unit test (spec §12, `uz.py::two_unit_test`): the absolute difference in a category's share
between the two units, against the region labels shuffled among respondents within each round, plus
the chi-square, both at 0.05. Shuffling respondents rather than sampling points is right for
Iceland because its ESS sample is nearly unclustered: round 8's sample design file gives its 880
respondents 705 PSUs in 4 strata (ESS8 SDDF user guide, Table 1).
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
RAW = os.path.join(ROOT, "data", "raw", "is")
OUT = os.path.join(ROOT, "data", "normalized", "is.csv")
OUT_FOREIGN = os.path.join(ROOT, "data", "normalized", "is_foreign.csv")

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) religiondots"}

# Norway's query template, save and Eurostat helpers, and its split-half, imported so Iceland
# prints the same statistic its neighbours draw with.
import be  # noqa: E402
import no as _no  # noqa: E402
import stability  # noqa: E402
import is2024 as tax  # noqa: E402

# --- ESS --------------------------------------------------------------------------------
ESS_API = _no.ESS_API
ESS_ROUNDS = {
    6: ("450fa78e-68ab-493f-b169-dbc7ab8ffec2", 85),
    8: ("ffc43f48-e15a-4a1c-8813-47eda377c355", 98),
    10: ("f37d014a-6958-42d4-b03b-17c29e481d3d", 286),
    11: ("242aaa39-3bbb-40f5-98bf-bfb1ce53d8ef", 179),
}
# Round 9 is fetched for the check that keeps it out and for its national shares, nothing else.
ROUND9 = {9: ("b2b0bf39-176b-4eca-8d26-3c05ea83d2cb", 280)}
ALL_FILES = {**ESS_ROUNDS, **ROUND9}
LATE_ROUNDS = (10, 11)

CARD_VAR = {6: "rlgdnis", 8: "rlgdnis", 9: "rlgdnais", 10: "rlgdnais", 11: "rlgdnais"}
FETCH_WEIGHTS = {"n": None, "w": "pspwght"}
YEAR_VARS = ("inwyys", "inwyr", "inwyye")

UNITS = {"IS001": "Höfuðborgarsvæði", "IS002": "Landsbyggð"}
MISSING_REGION = "99999"       # round 10 carries two respondents as `Not available`

# The later card's label for the two answers the earlier card named differently.
HARMONISE = {
    "rlgdnis": {"Rússnesku rétttrúnaðarkirkjunni": tax.ORTHODOX,
                "Félags múslima á Íslandi": tax.ISLAM},
    "rlgdnais": {},
}

# Every answer's `rlgdnm` answer, asserted from each round's cross-tab by `_check_card`.
CARD_NEST = {
    tax.NATIONAL: "Protestant",
    tax.FREE: "Protestant",
    tax.OTHER_LUTHERAN: "Protestant",
    tax.CATHOLIC: "Roman Catholic",
    tax.ORTHODOX: "Eastern Orthodox",
    # Rounds 10 and 11 put all ten of these in `Other Christian denomination`; round 6's one is in
    # `Eastern Orthodox`, which is ESS's recode of that round's code and not the respondent's answer.
    # The map reads the Icelandic label, so the one respondent stays with the answer they gave.
    tax.OTHER_CHRISTIAN: {"Other Christian denomination", "Eastern Orthodox"},
    tax.ISLAM: "Islam",
    tax.EASTERN: "Eastern religions",
    tax.ASATRU: "Other Non-Christian religions",
    tax.OTHER_NONCHRISTIAN: "Other Non-Christian religions",
    tax.OTHER: "Other Non-Christian religions",
}

# --- Eurostat and Pew ---------------------------------------------------------------------
EU_CTZ = "cens_21ctz_r3"
PEW_ZIP = _no.PEW_ZIP

# --- Hagstofa, the register, which is printed and not drawn ---------------------------------
HAG = "https://px.hagstofa.is/pxis/api/v1/is/Samfelag/menning/5_trufelog/trufelog/"
HAG_REGISTER = "MAN10001"
HAG_PARISH = {"2021": "MAN10291", "2023": "MAN10289"}
REGISTER_YEARS = ["2012", "2016", "2018", "2020", "2021", "2023", "2026"]

# Deanery 03, Kjalarnessprófastsdæmi, straddles the NUTS 3 line; 01 and 02 are all capital area,
# 04-09 all outside it. Parish -> municipality from the parishes' own names: Hafnarfjörður
# (Hafnarfjarðar-, Víðistaða-, Ástjarnarsókn), Garðabær (Garða-, Bessastaðasókn), Mosfellsbær
# (Lágafellssókn), Kjalarnes in Reykjavík (Brautarholtssókn), Kjósarhreppur (Reynivallasókn);
# and outside, Grindavík, Suðurnesjabær (Hvalsnes-, Útskálasókn), Reykjanesbær (Keflavíkur-,
# Njarðvíkur-, Kirkjuvogssókn) and Vogar (Kálfatjarnarsókn).
CAPITAL_03 = {"Hafnarfjarðarsókn", "Víðistaðasókn", "Ástjarnarsókn", "Garðasókn", "Bessastaðasókn",
              "Lágafellssókn", "Brautarholtssókn", "Reynivallasókn"}
REST_03 = {"Grindavíkursókn", "Hvalsnessókn", "Útskálasókn", "Keflavíkursókn", "Njarðvíkursókn",
           "Kirkjuvogssókn", "Kálfatjarnarsókn",
           # 2021 still lists Njarðvík as two parishes, both in Reykjanesbær; 2023 has one.
           "Ytri-Njarðvíkursókn", "Innri-Njarðvíkursókn"}

COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count",
           "basis", "year", "source_id", "note"]

# Iceland's own totals, asserted so a re-fetch against a new release fails here. None prints.
POP_2021 = 359_122             # cens_21ctz_r3 TOTAL = NAT + FOR + STLS 48 + UNK 29
N_CITIZENS = 3_222             # unweighted answered citizens with a region, four rounds

KEEP_AS_RESIDUAL = {tax.NO_RELIGION}

# The two-unit test (uz.py::two_unit_test's construction, respondents shuffled within round).
TWO_UNIT_PERM = 2000
TWO_UNIT_SEED = 0
ALPHA = be.STAB_ALPHA
# What it selects, asserted, so a re-fetch that changes a verdict stops the build. The Church of
# Iceland: 36.0% of the capital area's answers against 44.4% elsewhere, p 0.0005, chi-square
# 2.9e-06, the same side in rounds 8, 10 and 11 and level in round 6. The Free Church: 2.5% against
# 0.3%, the same side in all four rounds, which is where its congregations are (Reykjavík and
# Hafnarfjörður). `No religion` would pass and stays the residual. sources/is.md §4.
EXPECT_PASS = {tax.NATIONAL, tax.FREE}

# spec §12, "A SURVEY POOL THAT SPANS A FAST CHANGE": a drawn category is scaled to the late
# rounds' national share when it moved more than this; ask 022 keeps the rule for new ESS countries.
# The Church of Iceland moved 1.92 points (34.80% pooled, 32.87% in rounds 10-11): no rescale.
DRIFT_BAR = 0.035
EXPECT_RESCALE = False


def _key(s):
    return unicodedata.normalize("NFC", " ".join(str(s).split()))


# =======================================================================================
# fetch
# =======================================================================================

def _q(weight):
    return _no._TAB % (f' weightVariable:"{weight}",' if weight else "")


def _ess(query, variables, soft=False):
    body = json.dumps({"query": query, "variables": variables}).encode()
    req = urllib.request.Request(ESS_API, data=body, headers={
        "Content-Type": "application/json", **UA})
    r = json.load(urllib.request.urlopen(req, timeout=900))
    if "errors" in r:
        if soft:
            return None
        sys.exit(f"!! ESS API: {r['errors'][0].get('message')} "
                 f"{r['errors'][0].get('extensions', {}).get('code', '')}")
    return r["data"]


def _ess_fetch(rnd, bv, weight, dest, soft=False):
    if os.path.exists(dest):
        return True
    fid, ver = ALL_FILES[rnd]
    d = _ess(_q(weight), {"id": fid, "v": ver, "bv": bv}, soft=soft)
    if d is None:
        return False
    hit = [x for x in d["analysis"]["frequencyTabulationByVariables"]["responses"]
           if x["by"][0]["value"] == "IS"]
    if not hit:
        sys.exit(f"!! ESS round {rnd} has no IS response")
    _no._save(hit[0]["response"], dest)
    return True


def _hag_get(table):
    return json.load(urllib.request.urlopen(
        urllib.request.Request(HAG + table + ".px", headers=UA), timeout=120))


def _hag_fetch(table, pick, dest):
    """All values of every variable, except those in `pick` {variable text: [value texts]}."""
    if os.path.exists(dest):
        print(f"  {table}: already on disk")
        return
    meta = _hag_get(table)
    query = []
    for v in meta["variables"]:
        if v["text"] in pick:
            codes = [c for c, t in zip(v["values"], v["valueTexts"]) if t in pick[v["text"]]]
            if len(codes) != len(pick[v["text"]]):
                sys.exit(f"!! {table}: `{v['text']}` has no value among {pick[v['text']]}")
        else:
            codes = list(v["values"])
        query.append({"code": v["code"], "selection": {"filter": "item", "values": codes}})
    body = json.dumps({"query": query, "response": {"format": "json-stat2"}}).encode()
    req = urllib.request.Request(HAG + table + ".px", data=body, headers={
        "Content-Type": "application/json", **UA})
    _no._save(json.load(urllib.request.urlopen(req, timeout=300)), dest)
    print(f"  {table}: {os.path.getsize(dest):,} bytes")


def fetch():
    os.makedirs(RAW, exist_ok=True)

    print("ESS…")
    for rnd in sorted(ALL_FILES):
        card = CARD_VAR[rnd]
        for tag, w in FETCH_WEIGHTS.items():
            _ess_fetch(rnd, ["region", "ctzcntr", "rlgblg", card], w,
                       os.path.join(RAW, f"ess_r{rnd}_{tag}.json"))
        _ess_fetch(rnd, ["region", "domicil"], None, os.path.join(RAW, f"ess_r{rnd}_domicil.json"))
        _ess_fetch(rnd, [card, "rlgdnm"], None, os.path.join(RAW, f"ess_r{rnd}_card.json"))
        dest = os.path.join(RAW, f"ess_r{rnd}_year.json")
        if not any(_ess_fetch(rnd, [v], None, dest, soft=True) for v in YEAR_VARS):
            print(f"  round {rnd}: no interview-year variable among {YEAR_VARS}")
        n = sum(c["count"] for c in json.load(
            open(os.path.join(RAW, f"ess_r{rnd}_n.json"), encoding="utf-8"))["table"])
        print(f"  round {rnd}: {n:,.0f} respondents")

    print("Eurostat census…")
    dest = os.path.join(RAW, "cens_21ctz_r3_is.json")
    if not os.path.exists(dest):
        d = _no._eurostat(EU_CTZ, format="JSON", lang="EN", age="TOTAL", sex="T",
                          geo=sorted(UNITS) + ["IS"])
        _no._save(d, dest)
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

    print("Hagstofa, the register (for the comparison, not for the map)…")
    _hag_fetch(HAG_REGISTER, {"Skipting": ["Alls", "18 ára og eldri"]},
               os.path.join(RAW, "hag_man10001.json"))
    for year, table in HAG_PARISH.items():
        _hag_fetch(table, {"Kyn": ["Alls"]}, os.path.join(RAW, f"hag_parish_{year}.json"))


# =======================================================================================
# reading
# =======================================================================================

def _read(rnd, tag):
    """One saved ESS response as a tidy frame: region CODE, everything else as a LABEL, and the
    round's card variable renamed `card`. `path` indexes `codeList` (fi.py's trap)."""
    d = json.load(open(os.path.join(RAW, f"ess_r{rnd}_{tag}.json"), encoding="utf-8"))
    order = [v["name"] for v in d["variableValues"]]
    codes = {v["name"]: v["codeList"] for v in d["variableValues"]}
    rename = {CARD_VAR[rnd]: "card"}
    rows = []
    for cell in d["table"]:
        rec = {"round": rnd, "count": float(cell["count"])}
        for i, n in enumerate(order):
            c = codes[n][cell["path"][i]]
            k = rename.get(n, n)
            rec[k] = c["value"] if n == "region" else _key(c["label"])
            rec[k + "_miss"] = bool(c["isMissing"])
        rows.append(rec)
    return pd.DataFrame(rows)


def _pool(rounds, tag):
    """Citizens with a region, category resolved and harmonised (refusals kept as REFUSAL)."""
    frames = []
    for r in sorted(rounds):
        df = _read(r, tag)
        h = HARMONISE[CARD_VAR[r]]
        df["card"] = df["card"].map(lambda x: h.get(x, x))
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df = df[(df["ctzcntr"] == "Yes") & (~df["region_miss"]) & (df["region"] != MISSING_REGION)].copy()
    df["cat"] = np.where(
        df["rlgblg"] == "No", tax.NO_RELIGION,
        np.where((df["rlgblg"] != "Yes") | df["card_miss"], tax.REFUSAL, df["card"]))
    return df


def _check_level():
    """Every pooled round has exactly the two NUTS 3 codes."""
    for rnd in sorted(ALL_FILES):
        d = _read(rnd, "n")
        codes = set(d.loc[~d["region_miss"] & (d["region"] != MISSING_REGION) & (d["count"] > 0),
                          "region"])
        if codes != set(UNITS):
            sys.exit(f"!! round {rnd}: regions {sorted(codes)}, expected {sorted(UNITS)}")
    print(f"  rounds {sorted(ALL_FILES)}: IS001 and IS002 in every round")


def _profile(rnd):
    d = _read(rnd, "domicil")
    d = d[~d["region_miss"] & ~d["domicil_miss"] & (d["region"] != MISSING_REGION)]
    t = d.groupby("region")["count"].sum()
    share = lambda labels: (d[d["domicil"].isin(labels)].groupby("region")["count"].sum()
                            / t).reindex(sorted(UNITS)).fillna(0.0)
    return (share(["A big city", "Suburbs or outskirts of big city"]),
            share(["Town or small city"]),
            share(["Country village", "A country village", "Farm or home in countryside"]),
            (t / t.sum()).reindex(sorted(UNITS)).fillna(0.0))


def _check_round9():
    """ROUND 9's REGION, CHECKED AGAINST DOMICILE RATHER THAN TRUSTED (sources/is.md §2).

    In every pooled round the capital area is at least half big city or suburbs and the rest of
    the country is at least 40% town. Round 9 as published must FAIL the second, so the build
    stops if ESS reissues the file and the round could be pooled.
    """
    ok = lambda big, town: big["IS001"] >= 0.50 and town["IS002"] >= 0.40
    for rnd in sorted(ALL_FILES):
        big, town, vil, share = _profile(rnd)
        print(f"  round {rnd}: sample IS001 {100 * share['IS001']:.1f}%; big city or suburbs "
              f"IS001 {100 * big['IS001']:.1f}% IS002 {100 * big['IS002']:.1f}%; town IS001 "
              f"{100 * town['IS001']:.1f}% IS002 {100 * town['IS002']:.1f}%; village or farm "
              f"IS001 {100 * vil['IS001']:.1f}% IS002 {100 * vil['IS002']:.1f}%")
        if rnd in ESS_ROUNDS and not ok(big, town):
            sys.exit(f"!! round {rnd} fails the domicile profile the pooled rounds share")
        if rnd in ROUND9 and ok(big, town):
            sys.exit(f"!! round {rnd} now passes the domicile profile; ESS may have fixed its "
                     "region, and ROUND9 should be reconsidered for the pool")
    print("  round 9 as published fails it, so it stays out of every pool")


def _check_card():
    """Every harmonised answer sits in exactly its CARD_NEST `rlgdnm` answer, in every round."""
    seen, bad = set(), []
    for rnd in sorted(ALL_FILES):
        d = _read(rnd, "card")
        d = d[d["count"] > 0]
        h = HARMONISE[CARD_VAR[rnd]]
        print(f"  round {rnd}, {CARD_VAR[rnd]} x rlgdnm (all respondents, unweighted):")
        for _, r in d.sort_values(["card_miss", "card"]).iterrows():
            if not r["card_miss"]:
                print(f"    {r['count']:>5.0f}  {r['card']:<42} -> {r['rlgdnm']}")
        for _, r in d.iterrows():
            a, b = h.get(r["card"], r["card"]), r["rlgdnm"]
            if r["card_miss"] or r["rlgdnm_miss"]:
                if r["card_miss"] != r["rlgdnm_miss"]:
                    bad.append(f"round {rnd}: `{a}` / `{b}` missing on one card only "
                               f"({r['count']:.0f} respondents)")
                continue
            allowed = CARD_NEST.get(a)
            allowed = {allowed} if isinstance(allowed, str) else (allowed or set())
            if b not in allowed:
                bad.append(f"round {rnd}: {CARD_VAR[rnd]} `{r['card']}` sits in rlgdnm `{b}` for "
                           f"{r['count']:.0f} respondents; CARD_NEST allows {sorted(allowed)}")
            seen.add(a)
    if bad:
        for line in bad:
            print(f"  !! {line}")
        sys.exit(1)
    missing = sorted(set(CARD_NEST) - seen)
    if missing:
        sys.exit(f"!! card answers no round produced: {missing}")
    print(f"  rlgdnis and rlgdnais nest in rlgdnm in all {len(ALL_FILES)} rounds; {len(seen)} "
          "harmonised answers")


def _years():
    out = {}
    for rnd in sorted(ALL_FILES):
        p = os.path.join(RAW, f"ess_r{rnd}_year.json")
        if not os.path.exists(p):
            continue
        d = json.load(open(p, encoding="utf-8"))
        codes = d["variableValues"][0]["codeList"]
        acc = {}
        for cell in d["table"]:
            c = codes[cell["path"][0]]
            if cell["count"] and not c["isMissing"]:
                acc[c["value"]] = acc.get(c["value"], 0) + cell["count"]
        out[rnd] = acc
    return out


# =======================================================================================
# the tests
# =======================================================================================

def _two_unit_test(raw, cats):
    """The capital area against the rest, per category (uz.py::two_unit_test's construction).

    Statistic: |share in IS001 - share in IS002| on pooled unweighted counts. Null: each round's
    region labels shuffled among that round's respondents, which for one category is a
    hypergeometric draw per round with the round's margins fixed. Veto: the 2 x 2 chi-square.
    """
    rounds, units = sorted(ESS_ROUNDS), sorted(UNITS)
    cube = np.zeros((len(rounds), 2, len(cats)))
    ri = {r: i for i, r in enumerate(rounds)}
    ui = {u: i for i, u in enumerate(units)}
    ci = {c: i for i, c in enumerate(cats)}
    for (r, u, c), v in raw.groupby(["round", "region", "cat"])["count"].sum().items():
        cube[ri[r], ui[u], ci[c]] += v
    tot = cube.sum(axis=2)
    if (tot == 0).any():
        sys.exit(f"!! (round, unit) cells with no respondent: {np.argwhere(tot == 0).tolist()}")
    cube_i = np.rint(cube).astype(np.int64)
    if np.abs(cube - cube_i).max() > 1e-9:
        sys.exit("!! the unweighted cube holds a fractional count; the n files are not respondents")
    tot_i = cube_i.sum(axis=2)
    n0, n1 = tot_i[:, 0].sum(), tot_i[:, 1].sum()
    obs = np.abs(cube_i[:, 0, :].sum(axis=0) / n0 - cube_i[:, 1, :].sum(axis=0) / n1)

    k = cube_i.sum(axis=1)                                       # (round, cat)
    rng = np.random.default_rng(TWO_UNIT_SEED)
    x0 = rng.hypergeometric(k, tot_i.sum(axis=1)[:, None] - k,
                            np.broadcast_to(tot_i[:, 0][:, None], k.shape),
                            size=(TWO_UNIT_PERM,) + k.shape)      # (perm, round, cat)
    null = np.abs(x0.sum(axis=1) / n0 - (k[None] - x0).sum(axis=1) / n1)
    pp = (1 + (null >= obs - 1e-12).sum(axis=0)) / (1 + TWO_UNIT_PERM)

    pooled = cube_i.sum(axis=0)
    per_unit = pooled.sum(axis=1)
    print(f"\n  two-unit test: capital area ({n0:,} answered citizens) against the rest ({n1:,}); "
          f"rounds {rounds}; {TWO_UNIT_PERM}-draw respondent shuffle within round + chi-square")
    print(f"    {'category':<42}{'n':>6}{'IS001':>8}{'IS002':>8}{'perm p':>9}{'chi2 p':>10}"
          f"  same side every round   verdict")
    passed = []
    for j, c in enumerate(cats):
        chi = stability.chi2_p(pooled[:, j], per_unit)
        s0 = cube_i[:, 0, j] / tot_i[:, 0]
        s1 = cube_i[:, 1, j] / tot_i[:, 1]
        sign = np.sign(pooled[0, j] / per_unit[0] - pooled[1, j] / per_unit[1])
        agree = int((np.sign(s0 - s1) == sign).sum())
        if c in KEEP_AS_RESIDUAL:
            verdict = "the residual"
        elif pp[j] < ALPHA and np.isfinite(chi) and chi < ALPHA:
            verdict = "two units apart"
            passed.append(c)
        else:
            verdict = "national rate (residual)"
        if c in KEEP_AS_RESIDUAL and pp[j] < ALPHA and chi < ALPHA:
            verdict += " (would pass)"
        print(f"    {c[:40]:<42}{int(pooled[:, j].sum()):>6,}{pooled[0, j] / per_unit[0]:8.2%}"
              f"{pooled[1, j] / per_unit[1]:8.2%}{pp[j]:9.4f}{chi:10.2e}  {agree} of {len(rounds)}"
              f"{'':>17}{verdict}")
    print("    per round, IS001 / IS002: " + "; ".join(
        f"r{r} " + ", ".join(f"{c[:12]} {cube_i[i, 0, ci[c]] / tot_i[i, 0]:.1%}/"
                             f"{cube_i[i, 1, ci[c]] / tot_i[i, 1]:.1%}" for c in cats[:2])
        for i, r in enumerate(rounds)))
    return passed


# =======================================================================================
# the citizen half
# =======================================================================================

def _assert_source(pool, label):
    unknown = sorted(set(pool["cat"]) - tax.SOURCE - {tax.REFUSAL})
    if unknown:
        sys.exit(f"!! {label}: source categories with no mapping: {unknown}")
    vanished = sorted(tax.SOURCE - set(pool["cat"]))
    if vanished:
        sys.exit(f"!! {label}: is2024.SOURCE categories nobody answered: {vanished}")


def _composition(df):
    t = df.groupby("cat")["count"].sum()
    return t / t.sum()


def _citizen_shares():
    raw = _pool(ESS_ROUNDS, "n")
    wtd = _pool(ESS_ROUNDS, "w")
    answered = 1.0 - wtd.loc[wtd["cat"] == tax.REFUSAL, "count"].sum() / wtd["count"].sum()
    print(f"  {raw['count'].sum():,.0f} citizens with a region; {100 * (1 - answered):.2f}% "
          "declined (weighted) and are not drawn")
    raw, wtd = raw[raw["cat"] != tax.REFUSAL], wtd[wtd["cat"] != tax.REFUSAL]
    n = int(round(raw["count"].sum()))
    if N_CITIZENS is None:
        print(f"  !! N_CITIZENS is unset; this build has {n:,}")
    elif n != N_CITIZENS:
        sys.exit(f"!! {n:,} answered citizens, expected {N_CITIZENS:,}")
    _assert_source(wtd, "rounds 6, 8, 10, 11")

    r9 = _pool(ROUND9, "w")
    r9 = r9[r9["cat"] != tax.REFUSAL]
    per = {r: _composition(wtd[wtd["round"] == r]) for r in sorted(ESS_ROUNDS)}
    per[9] = _composition(r9)
    nat = _composition(wtd)
    cats = sorted(nat.index, key=lambda c: -nat[c])
    years = _years()
    print("  by round, weighted, citizens (round 9 printed, not pooled):")
    print("    " + " " * 42 + "".join(f"{'r' + str(r):>9}" for r in sorted(per)))
    for c in cats:
        print(f"    {c[:40]:<42}" + "".join(f"{100 * per[r].get(c, 0):8.2f}%" for r in sorted(per)))
    for r in sorted(years):
        print(f"    round {r} interviews by year: {dict(sorted(years[r].items()))}")

    tab = wtd.groupby(["region", "cat"])["count"].sum().unstack(fill_value=0.0)
    tab = tab.reindex(index=sorted(UNITS), columns=cats, fill_value=0.0)
    share = tab.div(tab.sum(axis=1), axis=0)

    # The rank test, printed for the record. On two units it has nothing to rank.
    print("\n  the rank split-half, printed and not deciding (two units):")
    _no._stability(raw, cats, ESS_ROUNDS, UNITS, "2 NUTS 3 units")
    passed = _two_unit_test(raw, cats)
    if EXPECT_PASS is None:
        print(f"  !! EXPECT_PASS is unset; this build passes {sorted(passed)}")
    elif set(passed) != EXPECT_PASS:
        sys.exit(f"!! the two-unit test now selects {sorted(passed)}, not {sorted(EXPECT_PASS)}")
    fine = [c for c in cats if c in passed and c not in KEEP_AS_RESIDUAL]
    small = [c for c in cats if c not in fine]

    late = _pool(LATE_ROUNDS, "w")
    late = late[late["cat"] != tax.REFUSAL]
    reach = late.groupby(["round", "region"])["count"].sum()
    for r in LATE_ROUNDS:
        for u in UNITS:
            if reach.get((r, u), 0) <= 0:
                sys.exit(f"!! late round {r} does not reach {u}; the rescale cannot be used")
    t_late = _composition(late)
    print("\n  rounds 10-11 against the pool, national, weighted, citizens:")
    for c in cats:
        print(f"    {c[:40]:<42} pool {100 * nat[c]:6.2f}%   10-11 {100 * t_late.get(c, 0):6.2f}%"
              f"   {100 * (t_late.get(c, 0) - nat[c]):+6.2f}")
    drift = max([abs(t_late.get(c, 0) - nat[c]) for c in fine] or [0.0])
    rescale = drift > DRIFT_BAR
    print(f"  largest drift among the categories drawn at two units: {100 * drift:.2f} points, "
          f"bar {100 * DRIFT_BAR:.1f}; rescale {'YES' if rescale else 'no'}")
    if EXPECT_RESCALE is None:
        print(f"  !! EXPECT_RESCALE is unset; this build says {rescale}")
    elif rescale != EXPECT_RESCALE:
        sys.exit(f"!! rescale is now {rescale}, expected {EXPECT_RESCALE}")

    nraw = raw.groupby("cat")["count"].sum()
    print(f"\n  {len(fine)} categories at the two units, {len(small)} at the national rate inside "
          "each unit's residual")
    for c in cats:
        print(f"    {100 * nat[c]:>6.2f}%  {int(nraw.get(c, 0)):>6,}  {c}   "
              f"[{'two units' if c in fine else 'national rate (residual)'}]")
    print("\n  shares by unit, weighted:")
    for c in cats:
        print(f"    {c[:40]:<42}" + "".join(f"{UNITS[u][:14]:>16} {100 * share.loc[u, c]:6.2f}%"
                                            for u in share.index))
    return share, nat, fine, small, answered, n, t_late, rescale


def _compose(share, nat, fine, small, t_late, rescale, cit_pop):
    """Per unit: a fine category takes its unit's share (scaled to rounds 10-11 if `rescale`);
    the rest share the residual at the pool's national proportions (§9bi)."""
    out = pd.DataFrame(index=sorted(UNITS), columns=list(nat.index), dtype=float)
    for u in out.index:
        for c in fine:
            out.loc[u, c] = float(share.loc[u, c])
    w_cit = cit_pop.reindex(out.index).astype(float)
    w_cit = w_cit / w_cit.sum()
    residual = 1.0 - out[fine].sum(axis=1) if fine else pd.Series(1.0, index=out.index)
    for c in sorted(KEEP_AS_RESIDUAL & set(small)):
        alt = residual - share[c].reindex(out.index)
        print(f"  if `{c}` were fixed at its unit's share, the tail left would be "
              f"{', '.join(f'{u} {v:+.2%}' for u, v in alt.items())}")
    if rescale:
        print("  scaling the fixed categories to their rounds 10-11 national share, weighted by "
              "citizens per unit:")
        for c in fine:
            old = float((out[c] * w_cit).sum())
            f = float(t_late.get(c, 0.0)) / old
            out[c] = out[c] * f
            print(f"    {c[:40]:<42} {100 * old:6.2f}% -> {100 * float(t_late.get(c, 0.0)):6.2f}%"
                  f"  x{f:.4f}")
        residual = 1.0 - out[fine].sum(axis=1) if fine else pd.Series(1.0, index=out.index)
    if (residual <= 0).any():
        sys.exit(f"!! units with no room for the tail: {sorted(residual[residual <= 0].index)}")
    tot = float(nat[small].sum())
    for c in small:
        out[c] = residual * (float(nat[c]) / tot)
    print("  residual categories as drawn among citizens (the survey's own unit share in brackets):")
    for c in small[:6]:
        print(f"    {c[:40]:<42}" + "  ".join(
            f"{u} {100 * out.loc[u, c]:.2f}% ({100 * float(share.loc[u, c]):.2f}%)"
            for u in out.index))
    if (out.sum(axis=1) - 1.0).abs().max() > 1e-9:
        sys.exit("!! composition does not sum to 1")
    drawn_nat = out.mul(w_cit, axis=0).sum()
    print("  national citizen composition as drawn, against rounds 10-11 and the pool:")
    for c in drawn_nat.sort_values(ascending=False).index:
        print(f"    {c[:40]:<42} drawn {100 * drawn_nat[c]:6.2f}%   10-11 "
              f"{100 * float(t_late.get(c, 0.0)):6.2f}%   pool {100 * float(nat.get(c, 0.0)):6.2f}%")
    return out, drawn_nat


# =======================================================================================
# the foreign half (dk.py's, other.is)
# =======================================================================================

def _census():
    d = json.load(open(os.path.join(RAW, "cens_21ctz_r3_is.json"), encoding="utf-8"))
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
    df = df[df["geo"].isin(UNITS)].copy()
    df["unit"] = df["geo"]
    return df


def _foreign_half(cen):
    """[geo_id, node, count] for foreign citizens at NUTS 3."""
    import origin_religion as origin

    with zipfile.ZipFile(os.path.join(RAW, "pew.zip")) as z:
        name = [n for n in z.namelist() if n.endswith("(percentages).csv")][0]
        pew = pd.read_csv(io.BytesIO(z.read(name)))
    pew = pew[(pew["Year"] == 2020) & (pew["Level"] == 1)].set_index("Country")
    fams = origin.FAMILIES

    leaf = cen[cen["citizen"].str.fullmatch(r"[A-Z]{2}") & (cen["citizen"] != "IS")]
    total_for = cen[cen["citizen"] == "FOR"]["value"].sum()
    covered = leaf["value"].sum()
    print(f"  {leaf['citizen'].nunique()} named citizenships cover {covered:,.0f} of "
          f"{total_for:,.0f} foreign residents ({100 * covered / total_for:.2f}%)")
    # spec §12, "EUROSTAT'S `FOR`": Iceland's RNC is 0 (2026-09-15); asserted.
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
        comp[iso] = origin.composition(iso, row, "other.is")
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
            for node, s in comp[iso].items():
                acc[node] = acc.get(node, 0.0) + v * scale * s
        for node, c in acc.items():
            rows.append((unit, node, c))
    df = pd.DataFrame(rows, columns=["geo_id", "node", "count"])
    print(f"  foreign half: {df['count'].sum():,.0f} people, {df['node'].nunique()} nodes")
    return df


# =======================================================================================
# the register, printed beside the survey
# =======================================================================================

def _jsonstat(name):
    """A json-stat2 file as a frame of dimension LABELS and `value`."""
    d = json.load(open(os.path.join(RAW, name), encoding="utf-8"))
    ids, sizes = d["id"], d["size"]
    order = {x: sorted(d["dimension"][x]["category"]["index"],
                       key=lambda k: d["dimension"][x]["category"]["index"][k]) for x in ids}
    label = {x: d["dimension"][x]["category"]["label"] for x in ids}
    vals = d["value"]
    if isinstance(vals, dict):
        vals = [vals.get(str(i)) for i in range(int(np.prod(sizes)))]
    if len(vals) != int(np.prod(sizes)):
        sys.exit(f"!! {name}: {len(vals)} values for a declared size {sizes} (li.py's trap)")
    rows = []
    for i, v in enumerate(vals):
        j, pos = i, []
        for s in reversed(sizes):
            pos.append(j % s)
            j //= s
        pos = list(reversed(pos))
        rows.append({x: _key(label[x][order[x][pos[k]]]) for k, x in enumerate(ids)} | {"value": v})
    return pd.DataFrame(rows), ids


def _register_national():
    df, ids = _jsonstat("hag_man10001.json")
    year, body, split = ids
    df["value"] = pd.to_numeric(df["value"], errors="coerce").fillna(0.0)
    alls = df[df[split] == "Alls"].pivot_table(index=body, columns=year, values="value",
                                                aggfunc="sum")
    names = {"Þjóðkirkjan": "Church of Iceland", "Kaþólska kirkjan": "Catholic Church",
             "Fríkirkjan í Reykjavík": "Free Church, Reykjavík",
             "Fríkirkjan í Hafnarfirði": "Free Church, Hafnarfjörður",
             "Óháði söfnuðurinn": "Independent Congregation", "Ásatrúarfélagið": "Ásatrú",
             "Siðmennt": "Siðmennt (humanist)",
             "Önnur trúfélög og ótilgreint": "other bodies and unspecified",
             "Utan trú- og lífsskoðunarfélaga": "outside every body"}
    missing = sorted(set(names) - set(alls.index))
    if missing:
        sys.exit(f"!! MAN10001 has no row for {missing}")
    print(f"    {'':<32}" + "".join(f"{y:>8}" for y in REGISTER_YEARS))
    print(f"    {'population':<32}" + "".join(f"{alls.loc['Alls', y] / 1000:7.1f}k"
                                                 for y in REGISTER_YEARS))
    for k, v in names.items():
        print(f"    {v:<32}" + "".join(f"{100 * alls.loc[k, y] / alls.loc['Alls', y]:7.2f}%"
                                         for y in REGISTER_YEARS))
    return {y: float(alls.loc["Þjóðkirkjan", y] / alls.loc["Alls", y]) for y in REGISTER_YEARS}


def _register_parish(year, census_share):
    """The Church of Iceland's parish roll summed to the two units, with the join witnessed by
    population: the parishes' own head count must put the capital area's share of Iceland within
    1.5 points of the census's."""
    df, ids = _jsonstat(f"hag_parish_{year}.json")
    parish, member, sex = ids
    df["value"] = pd.to_numeric(df["value"], errors="coerce").fillna(0.0)
    labels = list(dict.fromkeys(df[parish]))
    code = {t: t.split(" ", 1)[0] for t in labels}
    leaves = [t for t in labels if code[t] != "0"
              and not any(o != code[t] and o.startswith(code[t] + ".") for o in code.values())]
    unit_of, unknown = {}, []
    for t in leaves:
        dean = code[t][:2]
        name = t.split(" ", 1)[1].split(", ")[-1].strip()
        if dean in ("01", "02"):
            unit_of[t] = "IS001"
        elif dean == "03":
            if name in CAPITAL_03:
                unit_of[t] = "IS001"
            elif name in REST_03:
                unit_of[t] = "IS002"
            else:
                unknown.append(t)
        elif dean in ("04", "05", "06", "07", "08", "09"):
            unit_of[t] = "IS002"
        else:
            unknown.append(t)
    if unknown:
        sys.exit(f"!! {year}: parishes in neither CAPITAL_03 nor REST_03, or in no known deanery: "
                 f"{unknown}")
    # Parish names change between years (Njarðvík), so each year need only place parishes on both
    # sides of the line; the population witness below is what catches a wrong assignment.
    if {unit_of[t] for t in leaves if code[t][:2] == "03"} != {"IS001", "IS002"}:
        sys.exit(f"!! {year}: deanery 03 does not straddle the two units as expected")
    piv = df.pivot_table(index=parish, columns=member, values="value", aggfunc="sum")
    want = {"Yngri en 16 ára": "under16", "Alls 16 ára og eldri": "adults",
            "Í Þjóðkirkjunni": "members", "Ekki í Þjóðkirkjunni": "nonmembers"}
    if set(want) - set(piv.columns):
        sys.exit(f"!! {year}: membership columns {list(piv.columns)}")
    piv = piv.rename(columns=want)
    leaf = piv.loc[leaves].copy()
    leaf["unit"] = [unit_of[t] for t in leaf.index]
    tot = piv.loc[[t for t in labels if code[t] == "0"][0]]
    for c in want.values():
        if abs(leaf[c].sum() - tot[c]) > 0.5:
            sys.exit(f"!! {year}: parishes sum to {leaf[c].sum():,.0f} {c}, the total row says "
                     f"{tot[c]:,.0f}")
    if (leaf["members"] + leaf["nonmembers"] - leaf["adults"]).abs().max() > 0.5:
        sys.exit(f"!! {year}: members + non-members is not the 16-and-over count in some parish")
    by = leaf.groupby("unit")[list(want.values())].sum()
    pop = by["under16"] + by["adults"]
    cap = float(pop["IS001"] / pop.sum())
    print(f"  {year} (1 December): {len(leaves)} parishes, {pop.sum():,.0f} people; capital area "
          f"{100 * cap:.2f}% of them against the census's {100 * census_share:.2f}%")
    if abs(cap - census_share) > 0.015:
        sys.exit(f"!! {year}: the parish-to-unit assignment puts the capital area {100 * cap:.2f}% "
                 "of Iceland; the join is wrong")
    return by["members"] / by["adults"]


# =======================================================================================
# build
# =======================================================================================

def build():
    print("checking the region set, round 9's region and the Icelandic cards…")
    _check_level()
    _check_round9()
    _check_card()

    print("\nESS…")
    share, nat, fine, small, answered, n, t_late, rescale = _citizen_shares()

    print("\nEurostat census…")
    cen = _census()
    pop = cen[cen["citizen"] == "TOTAL"].groupby("unit")["value"].sum()
    nat_cit = cen[cen["citizen"] == "NAT"].groupby("unit")["value"].sum()
    for_ = cen[cen["citizen"] == "FOR"].groupby("unit")["value"].sum()
    print(f"  {pop.sum():,.0f} people = {nat_cit.sum():,.0f} Icelandic citizens + {for_.sum():,.0f} "
          f"foreign + {pop.sum() - nat_cit.sum() - for_.sum():,.0f} other, over {len(pop)} units")
    for code in ("STLS", "UNK"):
        if code in set(cen["citizen"]):
            print(f"    {code}: {cen[cen['citizen'] == code]['value'].sum():,.0f}")
    if sorted(pop.index) != sorted(UNITS):
        sys.exit(f"!! census units {sorted(pop.index)}")
    if POP_2021 is None:
        print(f"  !! POP_2021 is unset; this census has {pop.sum():,.0f}")
    elif int(pop.sum()) != POP_2021:
        sys.exit(f"!! census total {pop.sum():,.0f} is not Iceland's {POP_2021:,}")

    print("\ncomposing the citizen half…")
    comp, drawn_nat = _compose(share, nat, fine, small, t_late, rescale, nat_cit)

    rows = []
    for u in sorted(UNITS):
        n_cit = float(nat_cit[u]) * answered
        for cat, s in comp.loc[u].items():
            if s <= 0:
                continue
            if cat in fine:
                note = (f"ESS rounds 6, 8, 10, 11 at NUTS 3 ({u} {UNITS[u]}); two-unit test passed"
                        + ("; national level from rounds 10-11, spec §12" if rescale else ""))
            else:
                note = ("share of the unit's residual at the pool's national proportions (§9bi, "
                        "two-unit test not passed)")
            rows.append((u, UNITS[u], cat, s * n_cit, note))
    cit = pd.DataFrame(rows, columns=["geo_id", "geo_name", "source_category", "count", "note"])
    cit["geo_level"] = "nuts3"
    cit["basis"] = "self_id"
    cit["year"] = 2021
    cit["source_id"] = "ess_r6_r8_r10_r11_nuts3"
    cit = cit[COLUMNS]
    print(f"  citizen half: {cit['count'].sum():,.0f} people "
          f"({100 * cit['count'].sum() / nat_cit.sum():.2f}% of citizens; the rest declined)")
    unknown = sorted(set(cit["source_category"]) - set(tax.MAP))
    if unknown:
        sys.exit(f"!! unmapped source categories: {unknown}")

    print("\nforeign half…")
    ext = _foreign_half(cen)
    ext["geo_level"] = "nuts3"
    ext["geo_name"] = ext["geo_id"].map(UNITS)
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
    declined = float(nat_cit.sum()) * (1.0 - answered)
    other = float(pop.sum() - nat_cit.sum() - for_.sum())
    print(f"  not drawn: {declined:,.0f} citizens who declined ({100 * declined / pop.sum():.3f}%), "
          f"{other:,.0f} stateless or unknown citizenship ({100 * other / pop.sum():.3f}%); "
          f"gap_share {(declined + other) / pop.sum():.6f}")

    both = pd.concat([
        cit.assign(node=cit["source_category"].map(tax.resolve))[["geo_id", "node", "count"]],
        ext[["geo_id", "node", "count"]]], ignore_index=True)
    top = both.groupby("node")["count"].sum().sort_values(ascending=False)
    print("\nnational totals:")
    for node, c in top.head(22).items():
        print(f"  {c:>11,.0f}  {100 * c / drawn:5.2f}%  {node}")

    byu = both.groupby(["geo_id", "node"])["count"].sum().unstack(fill_value=0.0)
    tot = byu.sum(axis=1)
    fam = lambda pre: byu[[c for c in byu.columns if c == pre or c.startswith(pre + ".")]].sum(axis=1)
    print("\nper unit (drawn, both halves):")
    print(f"    {'unit':<20}{'lutheran':>10}{'unaffil':>9}{'catholic':>9}{'islam':>8}"
          f"{'orthodox':>9}{'pagan':>8}{'foreign':>9}")
    for u in sorted(UNITS):
        t = tot[u]
        print(f"    {UNITS[u]:<20}{100 * byu.loc[u].get('christianity.lutheran', 0) / t:9.2f}%"
              f"{100 * byu.loc[u].get('unaffiliated', 0) / t:8.2f}%"
              f"{100 * fam('christianity.catholic')[u] / t:8.2f}%{100 * fam('islam')[u] / t:7.2f}%"
              f"{100 * (fam('christianity.orthodox')[u] + fam('christianity.oriental')[u]) / t:8.2f}%"
              f"{100 * fam('paganism')[u] / t:7.2f}%{100 * for_[u] / pop[u]:8.2f}%")

    # ---- THE REGISTER BESIDE THE SURVEY. Not a validation: it counts registration, which for a
    # child is the parents' and for most adults was never revisited; the survey asks belonging.
    print("\nTHE REGISTER (Hagstofa MAN10001, 1 January) BESIDE THE SURVEY:")
    reg = _register_national()
    print(f"  on this map: christianity.lutheran {100 * top.get('christianity.lutheran', 0) / drawn:.2f}%"
          f", unaffiliated {100 * top.get('unaffiliated', 0) / drawn:.2f}%; the survey's Church of "
          f"Iceland answer among citizens, as drawn {100 * drawn_nat.get(tax.NATIONAL, 0):.2f}%")
    print("\nTHE CHURCH OF ICELAND'S PARISH ROLL AT THE SAME TWO UNITS (members of those 16 and over):")
    cap_share = float(pop["IS001"] / pop.sum())
    for year in HAG_PARISH:
        roll = _register_parish(year, cap_share)
        print(f"    roll {year}: " + ", ".join(f"{UNITS[u]} {100 * roll[u]:.1f}%" for u in roll.index)
              + f";  survey, citizens: " + ", ".join(
                  f"{UNITS[u]} {100 * share.loc[u, tax.NATIONAL]:.1f}%" for u in share.index))
        same = (roll["IS001"] < roll["IS002"]) == (share.loc["IS001", tax.NATIONAL]
                                                   < share.loc["IS002", tax.NATIONAL])
        print(f"    the roll and the survey order the two units {'the same way' if same else 'OPPOSITE WAYS'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    a = ap.parse_args()
    if a.fetch:
        fetch()
    else:
        build()
