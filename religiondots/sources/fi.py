"""Finland — a register that counts membership, and a survey that asks about belonging.

Writes data/normalized/fi.csv (citizens) and fi_foreign.csv (foreign residents).

Usage:
    python sources/fi.py --fetch     # ESS tabulations, Eurostat census, Pew  (~6 MB)
    python sources/fi.py             # rebuild from data/raw/fi/

WHY THIS IS NOT BUILT FROM THE REGISTER, WHICH IS THE WHOLE STORY OF THE COUNTRY.
Finland records religious-community membership for every resident in the population
information system, and Statistics Finland publishes it as StatFin `vaerak/11rx.px`:
twenty-six named communities, 1990-2025, exact to the person. It has no geography at all.
Its dimensions are community, age, sex and year, and there is no seventh table hiding
anywhere: all twelve databases at pxdata.stat.fi were walked in full on 2026-09-08, 5,634
nodes, and the only other things matching a religion word are parish payroll and a leisure
survey's "read religious or devotional books". sources/fi.md §2 has the record.

So Finland offers a register with no map and a survey with one, and they are not measuring
the same thing. **The register counts formal membership of a registered religious community,
which is a tax and records status you leave by filing a form. The survey asks whether you
consider yourself as belonging to a religion.** In round 11 those are 63% and 43% of
Finland, and that twenty-point gap is real rather than error: it is the people who have not
resigned from the Evangelical Lutheran Church and also do not describe themselves as
belonging to it. Germany (§3.9a) is the register-basis country on this map and it publishes
its register by Gemeinde; Finland publishes the same kind of number and withholds the
geography, so this build takes the other quantity rather than a worse version of Germany's.
`basis` says which one, and note_public leads with the difference.

THE TWO HALVES are Greece's (§9z), France's (§9ab) and Italy's (§9bp):

    Finnish citizens     5.25M   ESS rounds 5-11 pooled, `ctzcntr = Yes`
    foreign residents    0.28M   Eurostat `cens_21ctz_r3` x Pew's origin compositions

BOTH COME OUT OF THE SAME CENSUS TABLE, so they partition by construction: `cens_21ctz_r3`
publishes `NAT` and `FOR` beside its named citizenships.

THE DENOMINATION VARIABLE IS `rlgdnafi` AND NOT `rlgdnfi`, WHICH DOES NOT EXIST. §11ai's
warning is that several countries carry `a`/`b`-suffixed revisions and that pooling on the
bare name silently drops rounds; Finland is the case where the bare name drops ALL of them
and raises E201VariableNotFound instead, which at least fails loudly. What the suffixed one
buys is the entire point of mapping Finland: `rlgdnm`, the harmonised variable, offers seven
families and calls 43% of the country "Protestant". `rlgdnafi` names the Evangelical
Lutheran Church, the Orthodox Church, Pentecostals, the Free Church, Adventists, Jehovah's
Witnesses and Mormons separately, which is the shape of Finnish religion rather than a
European average of it.
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

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "taxonomy"))
RAW = os.path.join(ROOT, "data", "raw", "fi")
OUT = os.path.join(ROOT, "data", "normalized", "fi.csv")
OUT_FOREIGN = os.path.join(ROOT, "data", "normalized", "fi_foreign.csv")

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) religiondots"}

# --- ESS --------------------------------------------------------------------------------
ESS_API = "https://api.nsd.no/graphql"

# Finland is in every ESS round. Rounds 1-4 have no `region` variable (E201VariableNotFound),
# the wall Greece, France and Italy all hit, so SEVEN of eleven are usable — and unlike Italy
# (§9bp) the level never drops: `region` is NUTS 3, all 19 maakunnat, in every one of them.
# (datafile id, version), the same ids sources/fr.py and sources/it.py use.
ESS_ROUNDS = {
    5: ("0189b86b-8aa4-4be3-88ad-39c58b02f19f", 89),
    6: ("450fa78e-68ab-493f-b169-dbc7ab8ffec2", 85),
    7: ("9c96a1b2-b027-43c1-8c74-e883f892d0bb", 91),
    8: ("ffc43f48-e15a-4a1c-8813-47eda377c355", 98),
    9: ("b2b0bf39-176b-4eca-8d26-3c05ea83d2cb", 280),
    10: ("f37d014a-6958-42d4-b03b-17c29e481d3d", 286),
    11: ("242aaa39-3bbb-40f5-98bf-bfb1ce53d8ef", 179),
}

ESS_BREAK = ["region", "ctzcntr", "rlgblg", "rlgdnafi"]

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

# --- the register, which is not a source here and is fetched anyway -----------------------
# StatFin `vaerak/11rx.px`, the table §11k closed Finland on. It draws nothing: it has no
# geography and this build does not use one number from it. It is pulled because the whole
# claim of this country is that a register and a survey disagree, and a claim that size
# should be reproducible from the repo rather than quoted from a note. build() prints the
# comparison at the end.
#
# ITS ID IS `11rx.px` AND NOT `statfin_vaerak_pxt_11rx.px`, which is what the PxWeb web UI's
# own URL shows you and which 400s against the API. The API listing's `id` is the only id
# that works, and that is worth knowing for every Finnish table.
STATFIN_11RX = "https://pxdata.stat.fi/PxWeb/api/v1/en/StatFin/vaerak/11rx.px"
REGISTER_YEAR = "2024"
REG_LUTHERAN = "F09"     # Evangelical Lutheran Church of Finland
REG_ORTHODOX = "F08"     # Greek Orthodox Church, i.e. the Orthodox Church of Finland
REG_ISLAM = "D00"
REG_NONE = "H00"         # persons not members of any religious community
REG_TOTAL = "SSS"

# --- the geography ------------------------------------------------------------------------
# NUTS 2021, the 19 maakunnat, which is what data/geo/fi/fi_lau.gpkg rolls up to. Names as
# the GISCO workbook spells them; Finnish is already Latin script and is left alone.
NUTS3 = {
    "FI1B1": "Helsinki-Uusimaa", "FI1C1": "Varsinais-Suomi", "FI1C2": "Kanta-Hame",
    "FI1C3": "Paijat-Hame", "FI1C4": "Kymenlaakso", "FI1C5": "Etela-Karjala",
    "FI1D1": "Etela-Savo", "FI1D2": "Pohjois-Savo", "FI1D3": "Pohjois-Karjala",
    "FI1D5": "Keski-Pohjanmaa", "FI1D7": "Lappi", "FI1D8": "Kainuu",
    "FI1D9": "Pohjois-Pohjanmaa", "FI193": "Keski-Suomi", "FI194": "Etela-Pohjanmaa",
    "FI195": "Pohjanmaa", "FI196": "Satakunta", "FI197": "Pirkanmaa", "FI200": "Aland",
}

# THREE NUTS VINTAGES IN SEVEN ROUNDS, which is worse than Greece's two (§9z) and is silent
# in exactly the same way: pooling on the raw code splits a region in two and both halves
# come out undersized with no error anywhere.
#   round 5      NUTS 2006   FI13x / FI18x / FI1Ax
#   rounds 6-10  NUTS 2013   FI1D4 Kainuu, FI1D6 Pohjois-Pohjanmaa
#   round 11     NUTS 2021   FI1D8 Kainuu, FI1D9 Pohjois-Pohjanmaa
# Every pair is checked against ESS's OWN labels below rather than trusted, and the one pair
# whose labels legitimately differ is named there.
RECODE = {
    # NUTS 2006 -> NUTS 2021
    "FI181": "FI1B1", "FI182": "FI1B1", "FI183": "FI1C1", "FI184": "FI1C2",
    "FI185": "FI1C3", "FI186": "FI1C4", "FI187": "FI1C5", "FI131": "FI1D1",
    "FI132": "FI1D2", "FI133": "FI1D3", "FI134": "FI1D8", "FI1A1": "FI1D5",
    "FI1A2": "FI1D9", "FI1A3": "FI1D7",
    # NUTS 2013 -> NUTS 2021
    "FI1D4": "FI1D8", "FI1D6": "FI1D9",
}

# The one merge, and it is the reason this check exists rather than a nuisance it raised.
# Itä-Uusimaa (FI182) was abolished on 1 January 2011 and absorbed into Uusimaa (FI181), and
# NUTS then RENAMED the enlarged region Helsinki-Uusimaa (FI1B1). So round 5 carries two
# regions whose labels are `Uusimaa` and `Itä-Uusimaa` and every later round carries one
# called `Helsinki-Uusimaa`: neither old label matches the new one, and the label check
# correctly refuses both until they are named here. Territorially FI181 + FI182 = FI1B1
# exactly, so pooling loses nobody; what it loses is the ability to tell the eastern fringe
# apart from Helsinki in round 5, which is 2,000 respondents out of 12,982.
RECODE_LABEL_EXEMPT = {"FI181", "FI182"}

# --- the categories -----------------------------------------------------------------------
NO_RELIGION = "No religion"          # rlgblg = No; the source has no atheist/agnostic split
# rlgblg answers that are neither Yes nor No, and rlgdnafi answers from a respondent who said
# Yes and then declined the denomination. spec §3.5: refusals are marked, not filled.
REFUSAL = "__refused__"

COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count",
           "basis", "year", "source_id", "note"]

# Finland's own totals, asserted so a re-fetch against a new ESS release or census vintage
# fails here rather than quietly redrawing the map.
POP_2021 = 5_533_793
N_ROUNDS = 7
N_UNITS = 19
# Unweighted, ctzcntr = Yes, rounds 5-11. This is the figure note_public quotes as "people
# interviewed", so it is asserted rather than printed: a reissued round would otherwise move
# it silently and leave the reader-facing text wrong.
N_CITIZENS = 12_741


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
            fi = [x for x in d["analysis"]["frequencyTabulationByVariables"]["responses"]
                  if x["by"][0]["value"] == "FI"]
            if not fi:
                sys.exit(f"!! ESS round {rnd} has no FI response")
            json.dump(fi[0]["response"], open(dest, "w", encoding="utf-8"),
                      ensure_ascii=False)
        n = sum(c["count"] for c in json.load(
            open(os.path.join(RAW, f"ess_r{rnd}_n.json"), encoding="utf-8"))["table"])
        print(f"  round {rnd}: {n:,.0f} respondents")

    print("Eurostat census…")
    dest = os.path.join(RAW, "cens_21ctz_r3_fi.json")
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

    print("StatFin register (for the comparison, not for the map)…")
    dest = os.path.join(RAW, "statfin_11rx.json")
    if not os.path.exists(dest):
        q = {"query": [
            {"code": "uskontokunta_10_20190101",
             "selection": {"filter": "all", "values": ["*"]}},
            {"code": "timeperiod_y",
             "selection": {"filter": "item", "values": [REGISTER_YEAR]}}],
            "response": {"format": "json-stat2"}}
        req = urllib.request.Request(
            STATFIN_11RX, data=json.dumps(q).encode(),
            headers={"Content-Type": "application/json", **UA})
        d = json.load(urllib.request.urlopen(req, timeout=300))
        json.dump(d, open(dest, "w", encoding="utf-8"), ensure_ascii=False)
        print(f"  {os.path.getsize(dest):,} bytes, "
              f"{len(d['dimension']['uskontokunta_10_20190101']['category']['index'])} "
              f"religious communities, no region dimension")
    else:
        print("  already on disk")


# =======================================================================================
# the citizen half
# =======================================================================================

def _ess_table(rnd, tag):
    """One round as a tidy frame: region CODE, everything else as a LABEL.

    `path` IS A LIST OF INDICES INTO `values`, NOT A LIST OF CODE VALUES. Reading it as
    codes silently returns zero for every cell whose code is not also a valid index, which
    for Finland means every denomination above 9 — Islam, the two Other Christian buckets
    and both non-Christian buckets — and the map would have come out with no Muslims and no
    error. Greece's gr.py already indexes correctly; this is that convention written down.
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


def _region_labels():
    """code -> the set of labels ESS gives it, pooled over every round."""
    seen = {}
    for rnd in ESS_ROUNDS:
        d = json.load(open(os.path.join(RAW, f"ess_r{rnd}_n.json"), encoding="utf-8"))
        vv = [v for v in d["variableValues"] if v["name"] == "region"][0]
        for c in vv["codeList"]:
            if not c["isMissing"]:
                seen.setdefault(c["value"], set()).add(c["label"])
    return seen


def _check_recode():
    """Assert the NUTS recode against ESS's own region labels, per §12's join rule.

    A recode table is a name join wearing a code's clothes: it looks authoritative and it
    fails silently, which is [[reference_name_join_wrong_neighbour]]'s whole subject. So
    every pair is required to carry the same label on both sides, and the single pair that
    legitimately does not is named in RECODE_LABEL_EXEMPT with its reason.
    """
    seen = _region_labels()
    bad = []
    for old, new in sorted(RECODE.items()):
        if old in RECODE_LABEL_EXEMPT:
            continue
        lo, ln = seen.get(old, set()), seen.get(new, set())
        if not lo or not ln:
            bad.append(f"{old}->{new}: one side never appears ({lo} / {ln})")
        elif not (lo & ln):
            bad.append(f"{old}->{new}: labels disagree ({sorted(lo)} vs {sorted(ln)})")
    if bad:
        sys.exit("!! NUTS recode does not hold:\n  " + "\n  ".join(bad))
    after = {RECODE.get(c, c) for c in seen}
    if after != set(NUTS3):
        sys.exit(f"!! recoded regions are not the 19 maakunnat: "
                 f"extra {sorted(after - set(NUTS3))}, missing {sorted(set(NUTS3) - after)}")
    print(f"  recode holds: {len(seen)} raw codes over {N_ROUNDS} rounds -> "
          f"{len(after)} maakunnat, every pair label-matched"
          f" ({len(RECODE_LABEL_EXEMPT)} exempt: Uusimaa and Ita-Uusimaa became "
          f"Helsinki-Uusimaa in 2011)")


def _category(row):
    """rlgblg x rlgdnafi -> one source category, or REFUSAL."""
    if row["rlgblg_miss"]:
        return REFUSAL
    if row["rlgblg"] == "No":
        return NO_RELIGION
    if row["rlgblg"] != "Yes":
        return REFUSAL
    # said yes and then did not name one, or the variable is Not applicable for them
    return REFUSAL if row["rlgdnafi_miss"] else row["rlgdnafi"]


def _citizen_shares():
    """NUTS 3 x denomination shares among Finnish citizens, pooled over seven rounds."""
    import fi2024

    _check_recode()

    frames, unweighted = [], []
    for rnd in sorted(ESS_ROUNDS):
        w = _ess_table(rnd, "w")
        n = _ess_table(rnd, "n")
        for df, bag in ((w, frames), (n, unweighted)):
            df = df[(df["ctzcntr"] == "Yes")].copy()
            df["region"] = df["region"].map(lambda r: RECODE.get(r, r))
            df["cat"] = df.apply(_category, axis=1)
            bag.append(df.assign(round=rnd))
        # RESPONDENTS ARE COUNTED UNWEIGHTED. pspwght is a post-stratification weight and
        # its citizen subtotal is NOT the number of citizens interviewed: over seven rounds
        # the weighted figure is 12,718 and the people are 12,741. Anything reported as a
        # count of people has to come off the unweighted pass or it is quietly 23 short.
        a, b = frames[-1], unweighted[-1]
        print(f"  round {rnd}: {b['count'].sum():>6,.0f} citizen respondents "
              f"({a['count'].sum():>6,.0f} weighted), "
              f"{b['region'].nunique():>2} maakunnat, "
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

    unknown = sorted(set(pool["cat"]) - set(fi2024.MAP))
    if unknown:
        sys.exit(f"!! source categories with no mapping: {unknown}")

    tab = pool.groupby(["region", "cat"])["count"].sum().unstack(fill_value=0.0)
    if len(tab) != N_UNITS:
        sys.exit(f"!! {len(tab)} maakunnat have respondents, expected {N_UNITS}")
    share = tab.div(tab.sum(axis=1), axis=0)

    # THE N FLOOR IS READ OFF THE UNWEIGHTED PASS, because a weighted cell is fractional and
    # a 0.4 there is not "someone was counted". Reported, not enforced: a survey's thin cells
    # are a property of the instrument and the note says so rather than the code hiding them.
    nraw = raw.groupby(["region", "cat"])["count"].sum().unstack(fill_value=0.0)
    per_unit = nraw.sum(axis=1)
    print(f"  respondents per maakunta: min {per_unit.min():,.0f} ({per_unit.idxmin()}), "
          f"median {per_unit.median():,.0f}, max {per_unit.max():,.0f} "
          f"({per_unit.idxmax()})")

    nat = tab.sum() / tab.sum().sum()
    print("  national citizen shares (weighted), with unweighted respondent counts:")
    for k, v in nat.sort_values(ascending=False).items():
        print(f"    {100 * v:>6.2f}%  {int(nraw[k].sum()):>6,}  {k}")
    return share, answered, nraw


# =======================================================================================
# the foreign half, and the population both halves are scaled to
# =======================================================================================

def _census():
    """NUTS 3 x citizenship from cens_21ctz_r3, as a tidy frame."""
    d = json.load(open(os.path.join(RAW, "cens_21ctz_r3_fi.json"), encoding="utf-8"))
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
    # The geo dimension holds EVERY NUTS level at once — FI, FI1, FI1B, FI1B1 are all rows —
    # so a prefix filter sums the same people four times. Only the 19 leaves are read, and
    # Finland's leaves are not a single regex shape: FI193 is five characters of digits and
    # FI1B1 is not, so the recoded key list is the filter.
    df = df[df["geo"].isin(NUTS3)]
    df["unit"] = df["geo"]
    return df


def _register():
    """StatFin 11rx as {community code -> people} for REGISTER_YEAR. Comparison only."""
    d = json.load(open(os.path.join(RAW, "statfin_11rx.json"), encoding="utf-8"))
    dims, sizes = d["id"], d["size"]
    cats = [list(d["dimension"][x]["category"]["index"]) for x in dims]
    ri = dims.index("uskontokunta_10_20190101")
    out = {}
    for i, v in enumerate(d["value"]):
        idx, k = [], i
        for s in reversed(sizes):
            idx.append(k % s)
            k //= s
        idx = list(reversed(idx))
        out[cats[ri][idx[ri]]] = v
    return out


def _foreign_half(cen):
    """[geo_id, node, count] for foreign nationals, counted at NUTS 3."""
    import origin_religion as origin

    with zipfile.ZipFile(os.path.join(RAW, "pew.zip")) as z:
        name = [n for n in z.namelist() if n.endswith("(percentages).csv")][0]
        pew = pd.read_csv(io.BytesIO(z.read(name)))
    pew = pew[(pew["Year"] == 2020) & (pew["Level"] == 1)].set_index("Country")
    fams = origin.FAMILIES

    leaf = cen[cen["citizen"].str.fullmatch(r"[A-Z]{2}") & (cen["citizen"] != "FI")]
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
        comp[iso] = origin.composition(iso, row, "other.fi")
    if unmapped:
        sys.exit(f"!! {len(unmapped)} citizenships have no composition: {unmapped}")

    # The unnamed remainder — STLS, RNC, UNK and rounding — is spread over the named
    # citizenships of its own maakunta rather than dropped, so each unit's foreign total is
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
    import fi2024

    print("ESS…")
    share, answered, nraw = _citizen_shares()

    print("Eurostat census…")
    cen = _census()
    pop = cen[cen["citizen"] == "TOTAL"].groupby("unit")["value"].sum()
    nat = cen[cen["citizen"] == "NAT"].groupby("unit")["value"].sum()
    for_ = cen[cen["citizen"] == "FOR"].groupby("unit")["value"].sum()
    print(f"  {pop.sum():,.0f} people = {nat.sum():,.0f} Finnish citizens "
          f"+ {for_.sum():,.0f} foreign, over {len(pop)} maakunnat")
    missing = sorted(set(NUTS3) - set(pop.index))
    if missing:
        sys.exit(f"!! census has no rows for {missing}")
    if abs(pop.sum() - POP_2021) > 2_000:
        sys.exit(f"!! census total {pop.sum():,.0f} is not Finland's {POP_2021:,}")

    # ---- citizen half
    rows = []
    for unit in sorted(NUTS3):
        n_cit = float(nat.get(unit, 0.0)) * answered
        if unit not in share.index:
            sys.exit(f"!! no ESS respondents in {unit}")
        for cat, s in share.loc[unit].items():
            if s <= 0:
                continue
            rows.append((unit, NUTS3[unit], cat, s * n_cit,
                         f"pooled ESS rounds 5-11, {int(nraw.loc[unit, cat])} respondents"))
    cit = pd.DataFrame(rows, columns=["geo_id", "geo_name", "source_category", "count",
                                      "note"])
    cit["geo_level"] = "nuts3"
    cit["basis"] = "self_id"
    cit["year"] = 2024
    cit["source_id"] = "ess_r5_r11_pooled"
    cit = cit[COLUMNS]
    print(f"  citizen half: {cit['count'].sum():,.0f} people "
          f"({100 * cit['count'].sum() / nat.sum():.2f}% of citizens; the rest declined)")

    unknown = sorted(set(cit["source_category"]) - set(fi2024.MAP))
    if unknown:
        sys.exit(f"!! unmapped source categories: {unknown}")

    # ---- foreign half
    print("foreign half…")
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
        cit.assign(node=cit["source_category"].map(fi2024.resolve))[["node", "count"]],
        ext[["node", "count"]]], ignore_index=True)
    top = both.groupby("node")["count"].sum().sort_values(ascending=False)
    print("\nnational totals:")
    for node, c in top.head(20).items():
        print(f"  {c:>11,.0f}  {100 * c / drawn:5.2f}%  {node}")

    # ---- THE CROSS-CHECK THAT MATTERS, and it is the whole point of the country.
    # StatFin vaerak/11rx.px: the register's own membership counts, read from the file rather
    # than quoted, so a new vintage moves this print rather than making it quietly wrong.
    # This is NOT a validation. The two instruments ask different questions and the gap is
    # the finding, not the error.
    reg = _register()
    drawn_tot = both["count"].sum()

    def _s(prefix):
        return both[both["node"].str.startswith(prefix)]["count"].sum() / drawn_tot

    print(f"\nREGISTER ({REGISTER_YEAR}) vs SURVEY, the thing this country is about:")
    for label, prefix, code in (
            ("Evangelical Lutheran", "christianity.lutheran", REG_LUTHERAN),
            ("Orthodox", "christianity.orthodox", REG_ORTHODOX),
            ("Islam", "islam", REG_ISLAM),
            ("no religion / no community", "unaffiliated", REG_NONE)):
        s, r = 100 * _s(prefix), 100 * reg[code] / reg[REG_TOTAL]
        print(f"  {label:<28} survey {s:5.2f}%   register {r:5.2f}%   {s - r:+6.2f} points")
    print("  Islam and Orthodoxy go the OTHER way, and that is the register's limit rather "
          "than\n  the survey's: it counts members of a registered congregation, which most "
          "Finnish\n  Muslims never join, and it cannot see an Orthodox resident who belongs "
          "to no\n  Finnish parish.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    a = ap.parse_args()
    if a.fetch:
        fetch()
    else:
        build()
