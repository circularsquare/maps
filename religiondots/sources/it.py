"""Italy — two populations, no census question, and a survey that gave up its geography.

Writes data/normalized/it.csv (citizens) and it_foreign.csv (foreign residents).

Usage:
    python sources/it.py --fetch     # ESS tabulations, Eurostat census, Pew  (~6 MB)
    python sources/it.py             # rebuild from data/raw/it/

Italy has never asked about religion and ISTAT does not collect it at all — it is treated as
sensitive data and is absent from the census, the permanent census and every multiscopo. The
two halves are Greece's (§9z) and France's (§9ab):

    Italian citizens    54.0M   ESS rounds 6, 8, 9, 10, 11 pooled, `ctzcntr = Yes`
    foreign residents    5.0M   Eurostat `cens_21ctz_r3` x Pew's origin compositions

BOTH COME OUT OF THE SAME CENSUS TABLE, so they partition by construction: `cens_21ctz_r3`
publishes `NAT` and `FOR` beside its 221 named citizenships.

**THE THING THIS COUNTRY DOES THAT NO OTHER ONE HERE DOES: IT IS DRAWN AT THREE RESOLUTIONS
AT ONCE, AND THAT IS ANITA'S CALL (§14).** ESS gave Italy NUTS 2 in rounds 6 and 8 and then
took it away — rounds 9, 10 and 11, which hold two thirds of the sample and all of the recent
vintage, carry only the five ripartizioni. So:

    foreign residents          NUTS 3, 107 province      measured citizenship counts
    Catholic / unaffiliated    NUTS 2, 20 regioni        rounds 6+8, 3,368 respondents
    everything else            NUTS 1, 5 ripartizioni    rounds 9+10+11, 7,663 respondents

The alternative was one level for the whole country, and both versions of that are worse. At
NUTS 2 the median region holds **three** minority respondents and the thinnest holds one; at
NUTS 1 the whole country is five units of 11.8M and Italy becomes the coarsest thing on the
map. Splitting by category costs an explanation and keeps both the Catholic/secular gradient,
which the sample supports at 20 units, and a minority magnitude that is not built on n=1.

**AND THE SPLIT IS CHEAPER THAN IT LOOKS BECAUSE OF WHERE ITALY'S MINORITIES ARE.** The
citizen minorities are 1.38M people, 2.55% of citizens. The foreign residents are 5.03M and
are drawn at 107 units. So roughly four fifths of everyone this map exists to show is in the
half with the finest geography in Europe, and the coarse level is applied to the fifth that
no Italian instrument can locate anyway.

HOW THE THREE LEVELS BECOME ONE. scatter.py takes a single `unit` column, so counting happens
at NUTS 3 and the coarse rows are spread down to it **proportional to each province's own
citizen population**, which the census gives exactly. That is not the invention it might look
like: the dots would be placed by population inside the coarse unit regardless (§8.2), so
disaggregating by population first changes nothing spatially — it changes only which column
the pipeline reads. What it *does* buy is that the citizen half is spread by CITIZEN
population rather than total, which matters in Italy because the foreign share runs from 2% in
the south to 13% in Emilia-Romagna. `note` on every citizen row names the level its
composition actually came from, so nothing here is recoverable only from this docstring.

THE CROSS-CHECK, AND IT IS PARTLY BAD NEWS. CESNUR puts non-Catholic Italian citizens at 4.2%;
this build gets 2.55%. ESS finds Muslims well (494,000 against CESNUR's 417,900) and Orthodox
at about half (212,000 against ~400,000), and it is wrong about Jews by roughly double. The
Protestant cell is the interesting one and taxonomy/it2024.py has it: ~71,000 Protestants
against CESNUR's 378,000 historic-plus-Pentecostal, with the gap almost exactly filling
`Other Christian denomination` — Italian Pentecostals and Jehovah's Witnesses do not tick
"Protestant". See sources/it.md §5.
"""

import argparse
import io
import json
import os
import shutil
import ssl
import sys
import urllib.parse
import urllib.request
import zipfile

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "taxonomy"))
RAW = os.path.join(ROOT, "data", "raw", "it")
OUT = os.path.join(ROOT, "data", "normalized", "it.csv")
OUT_FOREIGN = os.path.join(ROOT, "data", "normalized", "it_foreign.csv")

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) religiondots"}

# --- ESS --------------------------------------------------------------------------------
ESS_API = "https://api.nsd.no/graphql"

# Italy is in rounds 1, 2, 6, 8, 9, 10 and 11. Rounds 1 and 2 have no `region` variable at
# all (E201VariableNotFound), the same wall Greece and France hit, so five are usable.
# **AND THE FIVE SPLIT IN TWO**, which is the whole shape of this country: `region` is NUTS 2
# in rounds 6 and 8 and NUTS 1 in rounds 9, 10 and 11. It is not a coding drift that can be
# mapped across — the finer level is simply not in the later files. `regunit` says which is
# which and is checked below rather than trusted. (datafile id, version).
ESS_NUTS2_ROUNDS = {
    6: ("450fa78e-68ab-493f-b169-dbc7ab8ffec2", 85),
    8: ("ffc43f48-e15a-4a1c-8813-47eda377c355", 98),
}
ESS_NUTS1_ROUNDS = {
    9: ("b2b0bf39-176b-4eca-8d26-3c05ea83d2cb", 280),
    10: ("f37d014a-6958-42d4-b03b-17c29e481d3d", 286),
    11: ("242aaa39-3bbb-40f5-98bf-bfb1ce53d8ef", 179),
}
ESS_ROUNDS = {**ESS_NUTS2_ROUNDS, **ESS_NUTS1_ROUNDS}

ESS_TAB = """query($id:ID!,$v:Int!,$bv:[String!]!){analysis{
 frequencyTabulationByVariables(input:{
   datafile:{id:$id,version:$v}, breakVariables:$bv, byVariables:["cntry"],
   instance:PUBLISHED, agencyId:INT_ESSERIC, includeMissing:true,
   weightVariable:"pspwght", metadataLanguage:"en"}){
 responses{by{value} response{
   variableValues{name values codeList{value label isMissing}} table{path count}}}}}}"""

# --- Eurostat ---------------------------------------------------------------------------
EU_DATA = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/"
EU_CTZ = "cens_21ctz_r3"        # population by citizenship, age, NUTS 3 — census 2021

PEW_ZIP = ("https://www.pewresearch.org/wp-content/uploads/sites/20/2025/06/"
           "Religious-Composition-2010-2020-dataset.zip")

# --- the geography -------------------------------------------------------------------------
NUTS1 = {
    "ITC": "Nord-Ovest", "ITF": "Sud", "ITG": "Isole",
    "ITH": "Nord-Est", "ITI": "Centro",
}

# 19 regioni, with Trentino-Alto Adige split into its two autonomous provinces the way NUTS
# splits it. ITF2 (Molise) is the one unit NO ESS ROUND CONTAINS — see `_composition`.
NUTS2 = {
    "ITC1": "Piemonte", "ITC2": "Valle d'Aosta", "ITC3": "Liguria", "ITC4": "Lombardia",
    "ITF1": "Abruzzo", "ITF2": "Molise", "ITF3": "Campania", "ITF4": "Puglia",
    "ITF5": "Basilicata", "ITF6": "Calabria",
    "ITG1": "Sicilia", "ITG2": "Sardegna",
    "ITH1": "Provincia Autonoma di Bolzano", "ITH2": "Provincia Autonoma di Trento",
    "ITH3": "Veneto", "ITH4": "Friuli-Venezia Giulia", "ITH5": "Emilia-Romagna",
    "ITI1": "Toscana", "ITI2": "Umbria", "ITI3": "Marche", "ITI4": "Lazio",
}

# The two categories with enough sample to survive being cut twenty ways. Everything else in
# taxonomy/it2024.py's list is drawn at NUTS 1. `Not applicable` is not a non-response — it is
# everyone who said they belong to no religion, 23% of citizens.
BIG_CATS = ("Roman Catholic", "Not applicable")

# Asserted, not assumed — see the guard in build(). 107 province is the same count
# sources/it_geo.py gets from the GISCO LAU workbook, which is a different file.
N_NUTS3 = 107
POP_FLOOR, POP_CEIL = 55_000_000, 62_000_000

# The smallest pooled NUTS 2 sample a regione may speak for; below it, its ripartizione does.
# 100 puts the standard error on the Catholic share at about 4.3 points (p ~ 0.75); 23, which
# is what Trento actually has, puts it near 9. See `_composition`.
N_FLOOR = 100


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
    """Eurostat needs certifi's CURRENT bundle — sources/gr.py has why that is not a
    server-side fault, and why `pip install -U certifi` is the fix."""
    import certifi
    ctx = ssl.create_default_context(cafile=certifi.where())
    u = EU_DATA + dataset + "?" + urllib.parse.urlencode(params, doseq=True)
    return json.load(urllib.request.urlopen(
        urllib.request.Request(u, headers=UA), timeout=900, context=ctx))


def _reuse(dest, *candidates):
    """Copy a shared asset another country already fetched, rather than re-downloading.

    fr.py's reason, and it is about vintage rather than bandwidth: a rebuild that silently
    picked up a newer Pew or a revised census would change this country's numbers without
    changing this country's code.
    """
    for src in candidates:
        if os.path.exists(src):
            shutil.copyfile(src, dest)
            print(f"  copied from {os.path.relpath(src, ROOT)} "
                  f"({os.path.getsize(dest):,} bytes)")
            return True
    return False


def fetch():
    os.makedirs(RAW, exist_ok=True)

    print("ESS…")
    for rnd, (fid, ver) in sorted(ESS_ROUNDS.items()):
        dest = os.path.join(RAW, f"ess_r{rnd}.json")
        if os.path.exists(dest):
            print(f"  round {rnd} already on disk")
            continue
        d = _ess(ESS_TAB, {"id": fid, "v": ver,
                           "bv": ["region", "regunit", "ctzcntr", "rlgdnm"]})
        it = [x for x in d["analysis"]["frequencyTabulationByVariables"]["responses"]
              if x["by"][0]["value"] == "IT"]
        if not it:
            sys.exit(f"!! ESS round {rnd} has no IT response")
        json.dump(it[0]["response"], open(dest, "w", encoding="utf-8"),
                  ensure_ascii=False)
        n = sum(c["count"] for c in it[0]["response"]["table"])
        print(f"  round {rnd}: {n:,.0f} weighted respondents -> {os.path.basename(dest)}")

    print("Eurostat census…")
    dest = os.path.join(RAW, "cens_21ctz_r3_it.json")
    if os.path.exists(dest):
        print("  already on disk")
    elif not _reuse(dest, os.path.join(ROOT, "data", "raw", "fr", "cens_21ctz_r3_fr.json")):
        # THE FILE IS EU-WIDE AND THE COUNTRY SUFFIX IN ITS NAME IS A LIE OF CONVENIENCE.
        # `cens_21ctz_r3` is fetched with no `geo` filter, so France's copy already holds all
        # 1,687 NUTS codes including Italy's 108. Copying it is not a shortcut — it is what
        # guarantees fr.csv and it.csv were built against the same census vintage.
        d = _eurostat(EU_CTZ, format="JSON", lang="EN", age="TOTAL", sex="T")
        json.dump(d, open(dest, "w", encoding="utf-8"))
        print(f"  {os.path.getsize(dest):,} bytes")

    print("Pew…")
    dest = os.path.join(RAW, "pew.zip")
    if os.path.exists(dest):
        print("  already on disk")
    elif not _reuse(dest,
                    os.path.join(ROOT, "data", "raw", "gr", "pew.zip"),
                    os.path.join(ROOT, "data", "raw", "fr", "pew.zip")):
        with urllib.request.urlopen(urllib.request.Request(PEW_ZIP, headers=UA),
                                    timeout=600) as r, open(dest, "wb") as f:
            f.write(r.read())
        print(f"  {os.path.getsize(dest):,} bytes")


# =======================================================================================
# the citizen half
# =======================================================================================

def _ess_table(rnd):
    """(region, regunit, ctzcntr, rlgdnm) -> weighted count, one round, Italian CITIZENS."""
    d = json.load(open(os.path.join(RAW, f"ess_r{rnd}.json"), encoding="utf-8"))
    vv = {v["name"]: v for v in d["variableValues"]}
    order = [v["name"] for v in d["variableValues"]]
    labels = {n: [c["label"] for c in vv[n]["codeList"]] for n in order}
    # `region` is read as a CODE and everything else as a LABEL — gr.py's finding, and Italy
    # needs it for a third reason after Greece's alphabet switch and France's reletttering:
    # the labels are the only place an apostrophe lives ("Valle d'Aosta/Vallée d'Aoste"),
    # they are mojibake in some rounds, and pooling on them would split the region in two.
    values = {n: [c["value"] for c in vv[n]["codeList"]] for n in order}
    rows = []
    for cell in d["table"]:
        rec = {}
        for i, n in enumerate(order):
            rec[n] = (values if n == "region" else labels)[n][cell["path"][i]]
        rec["count"] = cell["count"]
        rows.append(rec)
    return pd.DataFrame(rows)


def _pooled_shares(rounds, level, expect):
    """Region x denomination shares among Italian citizens, pooled over `rounds`.

    Returns (shares indexed by region code, answered-fraction). `level` is only used for
    reporting; `expect` is the set of region codes the pool is allowed to contain.
    """
    import it2024

    frames = []
    for rnd in sorted(rounds):
        df = _ess_table(rnd)
        cit = df[df["ctzcntr"] == "Yes"]
        stray = sorted(set(cit["region"]) - expect)
        if stray:
            sys.exit(f"!! round {rnd} has unrecognised {level} codes: {stray}")
        # `regunit` names the NUTS level the round actually used. Asserting it is what stops
        # a future ESS release quietly moving Italy back to regioni — or on to province —
        # and being pooled as though nothing had changed.
        units = sorted(set(cit["regunit"]))
        print(f"  round {rnd}: {cit['count'].sum():,.0f} citizen respondents, "
              f"{cit['region'].nunique()} {level} units, regunit={units}")
        frames.append(cit)
    pool = pd.concat(frames, ignore_index=True)

    # spec §3.5: refusals are marked, not filled.
    nc = pool[pool["rlgdnm"].isin(it2024.EXCLUDED)]["count"].sum()
    total = pool["count"].sum()
    print(f"  pooled {total:,.0f} citizen respondents over {len(frames)} rounds; "
          f"refusals {nc:,.0f} ({100 * nc / total:.2f}%)")
    pool = pool[~pool["rlgdnm"].isin(it2024.EXCLUDED)]

    unknown = sorted(set(pool["rlgdnm"]) - set(it2024.MAP))
    if unknown:
        sys.exit(f"!! ESS denominations with no mapping: {unknown}")

    tab = pool.groupby(["region", "rlgdnm"])["count"].sum().unstack(fill_value=0.0)
    share = tab.div(tab.sum(axis=1), axis=0)
    nat = tab.sum() / tab.sum().sum()
    print(f"  national ({level}): "
          + "  ".join(f"{k}:{100 * v:.2f}%"
                      for k, v in nat.sort_values(ascending=False).items()))
    return share, 1.0 - nc / total, tab.sum(axis=1)


def _composition(s2, s1, n2_size):
    """{nuts2 code: {category: share}} — the three-level splice, one regione at a time.

    The minority LEVELS come from the ripartizione (NUTS 1, recent, 7,663 respondents); the
    Catholic-to-unaffiliated RATIO comes from the regione (NUTS 2, older, 3,368). The second
    is rescaled into whatever the first leaves, so every vector still sums to one and no
    category is counted twice.
    """
    minority = [c for c in s1.columns if c not in BIG_CATS]
    out, fell_back = {}, []
    for u in sorted(NUTS2):
        n1 = u[:3]
        if n1 not in s1.index:
            sys.exit(f"!! no ESS respondents in ripartizione {n1}")
        vec = {c: float(s1.loc[n1, c]) for c in minority if s1.loc[n1, c] > 0}
        m = sum(vec.values())

        # MOLISE IS IN NO ESS ROUND AT ALL — not round 6, not round 8 — so its
        # Catholic/unaffiliated ratio has to come from its ripartizione like everything else.
        # 294,000 people, 0.5% of Italy, and the same shape as France's Corsica except that
        # here the surrounding NUTS 1 unit is sampled and can carry it honestly.
        #
        # AND THE SAME FALLBACK CATCHES THE REGIONI THAT ARE SAMPLED BUT NOT SAMPLED ENOUGH,
        # which is a bigger set and was found by disbelieving the output. Trento has 23
        # pooled respondents and Bolzano 29, and at that size the Catholic share carries a
        # standard error near 9 points — enough that the first build made **South Tyrol the
        # least Catholic region in Italy at 48% unaffiliated**, which is the opposite of
        # every other thing known about it. spec §3.9b withdrew the floor on how many UNITS
        # a country may have; it says nothing about how few PEOPLE may stand behind one, and
        # a headline claim about a real place should not rest on 23 of them. `N_FLOOR` is
        # the smallest pooled sample a regione may speak for; below it the ripartizione
        # speaks instead. It is a knob and it is Anita's — sources/it.md §4 has what moving
        # it does.
        thin = u in s2.index and float(n2_size.get(u, 0.0)) < N_FLOOR
        src = s2 if (u in s2.index and not thin) else s1
        key = u if (u in s2.index and not thin) else n1
        if key != u:
            fell_back.append(f"{u} ({NUTS2[u]}, n={float(n2_size.get(u, 0.0)):.0f})")
        big = {c: float(src.loc[key, c]) for c in BIG_CATS if c in src.columns}
        b = sum(big.values())
        if b <= 0:
            sys.exit(f"!! {u} has no Catholic or unaffiliated respondents")
        for c, v in big.items():
            vec[c] = v / b * (1.0 - m)
        out[u] = vec
    if fell_back:
        print(f"  composition from the ripartizione instead of the regione "
              f"({len(fell_back)} of {len(NUTS2)}):")
        for u in fell_back:
            print(f"    {u}")
    return out, {u.split(" ")[0] for u in fell_back}


# =======================================================================================
# the census, and the foreign half
# =======================================================================================

def _census():
    """NUTS 3 x citizenship from cens_21ctz_r3, as a tidy frame."""
    d = json.load(open(os.path.join(RAW, "cens_21ctz_r3_it.json"), encoding="utf-8"))
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
    # THE GEO DIMENSION HOLDS EVERY NUTS LEVEL AT ONCE — `IT`, `ITC`, `ITC1`, `ITC11` are all
    # rows of the same table — so a prefix filter sums the same people four times. Only the
    # NUTS 3 leaves are read.
    #
    # AND THE LAST CHARACTER OF AN ITALIAN NUTS 3 CODE IS NOT ALWAYS A DIGIT. Lombardia has
    # twelve province and NUTS ran out of numerals, so Mantova is `ITC4A`, Lodi `ITC4B`,
    # **Milano `ITC4C`** and Monza e Brianza `ITC4D`; Sardegna and Sicilia do the same. A
    # `\d{2}` tail drops ten province and 6.6M people — including the largest one in the
    # country — and drops them SILENTLY, because every remaining unit still balances and the
    # national total is never asserted. The guard is the LAU count below, not this regex.
    df = df[df["geo"].str.fullmatch(r"IT[A-Z]\d[0-9A-Z]")].copy()
    df["unit"] = df["geo"]
    return df


def _foreign_half(cen):
    """[geo_id, node, count] for foreign nationals, counted at NUTS 3."""
    import origin_religion as origin

    with zipfile.ZipFile(os.path.join(RAW, "pew.zip")) as z:
        name = [n for n in z.namelist() if n.endswith("(percentages).csv")][0]
        pew = pd.read_csv(io.BytesIO(z.read(name)))
    pew = pew[(pew["Year"] == 2020) & (pew["Level"] == 1)].set_index("Country")
    fams = origin.FAMILIES

    leaf = cen[cen["citizen"].str.fullmatch(r"[A-Z]{2}") & (cen["citizen"] != "IT")]
    total_for = cen[cen["citizen"] == "FOR"]["value"].sum()
    covered = leaf["value"].sum()
    print(f"  {leaf['citizen'].nunique()} named citizenships cover {covered:,.0f} of "
          f"{total_for:,.0f} foreign residents ({100 * covered / total_for:.2f}%)")

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
        comp[iso] = origin.composition(iso, row, "other.it")
    if unmapped:
        sys.exit(f"!! {len(unmapped)} citizenships have no composition: {unmapped}")

    # The unnamed remainder — STLS, RNC, UNK and rounding — is spread over the named
    # citizenships of its own province rather than dropped, so each province's foreign total
    # is the census's own.
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
    print(f"  foreign half: {df['count'].sum():,.0f} people, {df['node'].nunique()} nodes "
          f"over {df['geo_id'].nunique()} province")
    return df


# =======================================================================================
# build
# =======================================================================================

def build():
    import it2024

    print("ESS, NUTS 1 (rounds 9, 10, 11) — the minority levels…")
    s1, answered1, _ = _pooled_shares(ESS_NUTS1_ROUNDS, "nuts1", set(NUTS1))
    print("ESS, NUTS 2 (rounds 6, 8) — the Catholic/unaffiliated ratio…")
    s2, _, n2_size = _pooled_shares(ESS_NUTS2_ROUNDS, "nuts2", set(NUTS2))
    comp, coarse = _composition(s2, s1, n2_size)

    print("Eurostat census…")
    cen = _census()
    pop = cen[cen["citizen"] == "TOTAL"].groupby("unit")["value"].sum()
    nat = cen[cen["citizen"] == "NAT"].groupby("unit")["value"].sum()
    for_ = cen[cen["citizen"] == "FOR"].groupby("unit")["value"].sum()
    print(f"  {pop.sum():,.0f} people = {nat.sum():,.0f} Italian citizens "
          f"+ {for_.sum():,.0f} foreign, over {len(pop)} NUTS 3 units")

    # The census carries one NUTS 3 code the LAU file does not: an extra-regio unit with no
    # comune and no population. It is dropped rather than special-cased, and asserted to be
    # the only one, which is how a future NUTS revision gets caught (§8.1).
    empty = sorted(u for u in pop.index if pop.get(u, 0) <= 0)
    if empty:
        print(f"  dropping {len(empty)} NUTS 3 unit(s) with no population: {empty}")
        pop, nat, for_ = pop.drop(empty), nat.drop(empty), for_.drop(empty)
        cen = cen[~cen["unit"].isin(empty)]
    bad = sorted(u for u in pop.index if u[:4] not in NUTS2)
    if bad:
        sys.exit(f"!! NUTS 3 units outside the expected regioni: {bad}")

    # THE CHECK THAT CATCHES A SILENTLY-TRUNCATED GEOGRAPHY, and it exists because the first
    # build of this file was truncated and looked perfectly healthy: 97 province, every unit
    # internally consistent, 99.83% of "the country" drawn. The percentage was of the wrong
    # denominator. Assert the province count and the population against the placement layer,
    # which is built from a different file by a different script (§8.1).
    if len(pop) != N_NUTS3:
        sys.exit(f"!! expected {N_NUTS3} province with population, got {len(pop)} — "
                 f"the geo filter in _census() is dropping units")
    if not (POP_FLOOR < pop.sum() < POP_CEIL):
        sys.exit(f"!! Italy's census population is {pop.sum():,.0f}, outside the "
                 f"{POP_FLOOR:,}-{POP_CEIL:,} this country should be in")
    missing = sorted(set(NUTS2) - {u[:4] for u in pop.index})
    if missing:
        sys.exit(f"!! regioni with no province in the census: {missing}")

    # ---- citizen half, disaggregated to NUTS 3 by CITIZEN population
    rows = []
    for unit in sorted(pop.index):
        n2, n1 = unit[:4], unit[:3]
        n_cit = float(nat.get(unit, 0.0)) * answered1
        if n_cit <= 0:
            continue
        note = (f"composition: {n2} ({NUTS2[n2]}) for Catholic and unaffiliated, "
                f"{n1} ({NUTS1[n1]}) for every other category; spread to this provincia "
                f"by its own citizen population")
        for cat, s in sorted(comp[n2].items()):
            if s <= 0:
                continue
            rows.append((unit, cat, s * n_cit, note))
    cit = pd.DataFrame(rows, columns=["geo_id", "source_category", "count", "note"])
    cit["geo_level"] = "nuts3"
    cit["geo_name"] = cit["geo_id"].map(lambda u: NUTS2[u[:4]])
    cit["basis"] = "self_id"
    cit["year"] = 2024
    cit["source_id"] = "ess_r6_r8_nuts2_x_r9_r10_r11_nuts1"
    cit = cit[["geo_id", "geo_level", "geo_name", "source_category", "count", "basis",
               "year", "source_id", "note"]]
    print(f"  citizen half: {cit['count'].sum():,.0f} people over "
          f"{cit['geo_id'].nunique()} province "
          f"({100 * cit['count'].sum() / nat.sum():.2f}% of citizens; the rest declined)")

    # THE HEADLINE NUMBER FOR THIS COUNTRY, and it is population-weighted for a reason: ten
    # of twenty-one regioni fall back to their ripartizione, which sounds like half the map
    # and is not, because the ten are the small ones.
    fine = float(pop[[u for u in pop.index if u[:4] not in coarse]].sum())
    print(f"  Catholic/unaffiliated split at the REGIONE for {100 * fine / pop.sum():.1f}% "
          f"of the population ({len(NUTS2) - len(coarse)} of {len(NUTS2)} regioni), at the "
          f"ripartizione for the rest")

    unknown = sorted(set(cit["source_category"]) - set(it2024.MAP))
    if unknown:
        sys.exit(f"!! unmapped source categories: {unknown}")

    # ---- foreign half
    print("foreign half…")
    ext = _foreign_half(cen)
    ext["geo_level"] = "nuts3"
    ext["geo_name"] = ext["geo_id"].map(lambda u: NUTS2[u[:4]])
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
        cit.assign(node=cit["source_category"].map(it2024.resolve))[["node", "count"]],
        ext[["node", "count"]]], ignore_index=True)
    top = both.groupby("node")["count"].sum().sort_values(ascending=False)
    print("\nnational totals:")
    for node, c in top.head(16).items():
        print(f"  {c:>11,.0f}  {100 * c / drawn:5.2f}%  {node}")

    # ---- the cross-checks, all against numbers no part of this build used
    cit_only = cit.assign(node=cit["source_category"].map(it2024.resolve))
    cit_tot = cit_only.groupby("node")["count"].sum()
    minority = cit_tot.drop(["christianity.catholic.latin", "unaffiliated"],
                            errors="ignore").sum()
    print("\nCROSS-CHECKS (CESNUR, Le religioni in Italia — used by nothing above):")
    print(f"  non-Catholic citizens   {100 * minority / cit_tot.sum():5.2f}%  "
          f"against CESNUR's 4.2%")
    for node, cesnur, what in (
            ("islam.sunni", 417_900, "Muslim citizens"),
            ("christianity.orthodox.canonical", 400_000, "Orthodox citizens"),
            ("judaism", 24_000, "Jews (UCEI registered members)")):
        got = float(cit_tot.get(node, 0.0))
        print(f"  {what:<24}{got:>10,.0f}  against {cesnur:>9,.0f}  "
              f"({got / cesnur:.2f}x)")
    prot = float(cit_tot.get("christianity.protestant", 0.0))
    other_chr = float(cit_tot.get("christianity", 0.0))
    print(f"  Protestant {prot:,.0f} + other Christian {other_chr:,.0f} = "
          f"{prot + other_chr:,.0f} against CESNUR's 378,000 "
          f"(Pentecostal 313,000 + historic ~65,000) — see taxonomy/it2024.py")
    # Pew's own Italy row, read from the same zip the foreign half uses — but Pew's ITALY
    # figure is used by nothing in this build (only its per-ORIGIN-COUNTRY rows are), so this
    # is a genuine second opinion rather than a restatement. gr.py's check, same shape.
    with zipfile.ZipFile(os.path.join(RAW, "pew.zip")) as z:
        name = [n for n in z.namelist() if n.endswith("(percentages).csv")][0]
        pw = pd.read_csv(io.BytesIO(z.read(name)))
    pw = pw[(pw["Year"] == 2020) & (pw["Level"] == 1)].set_index("Country").loc["Italy"]
    muslim = top[[n for n in top.index if n.startswith("islam")]].sum()
    unaff = float(top.get("unaffiliated", 0.0))
    print(f"  Muslims, BOTH halves      {100 * muslim / pop.sum():5.2f}%  against Pew's "
          f"{float(pw['Muslims']):.2f}%")
    print(f"  unaffiliated, BOTH halves {100 * unaff / pop.sum():5.2f}%  against Pew's "
          f"{float(pw['Religiously_unaffiliated']):.2f}%  <- SEE sources/it.md §5")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    a = ap.parse_args()
    if a.fetch:
        fetch()
    else:
        build()
