"""Greece — two populations, one census, and a survey that cannot see a minority.

Writes data/normalized/gr.csv (citizens) and gr_foreign.csv (foreign residents).

Usage:
    python sources/gr.py --fetch     # ESS tabulations, Eurostat census, Pew  (~1 MB)
    python sources/gr.py             # rebuild from data/raw/gr/

Greece has not asked about religion in a census since 1951, and the two halves are Spain's
(§9y) with better sources on both sides:

    Greek citizens     9.72M   ESS rounds 5, 10 and 11 pooled, restricted to `ctzcntr = Yes`
    foreign residents  0.76M   Eurostat `cens_21ctz_r3` x Pew's origin compositions
    Mount Athos        1,811   authored, and see taxonomy/gr2024.py

BOTH POPULATIONS COME OUT OF THE SAME CENSUS TABLE, which is why they partition exactly:
`cens_21ctz_r3` publishes `NAT` and `FOR` alongside its 221 named citizenships, so the
denominator of the citizen half and the numerator of the foreign half are the same file's own
rows and cannot drift apart.

THE PART THAT NEEDED A DECISION, AND IT WAS ANITA'S. ESS samples non-citizens — unlike CIS,
which does not (§9y) — so `ctzcntr` is what stops the halves double-counting. But ESS reaches
them badly and shrinkingly: 203 of 2,713 respondents in round 5, 83 of 2,800 in round 10, 87
of 2,757 in round 11, against a true 7.2%. And among CITIZENS it finds **nine** Muslims in
round 5 and none at all in rounds 10 and 11, against a recognised minority of 100,000-120,000
people who are 29% of their own region. A Greek-language national sample does not reach a
Turkish- and Pomak-speaking minority.

So the Thracian minority is split out of Anatoliki Makedonia-Thraki's citizen population using
the Council of Europe ECRI figure quoted in the US State Department's religious-freedom
reports. That is spec §3.1's permitted SPLIT — the region's citizen total is unchanged, only
its composition — and §14.9 permits the basis. Without it the map would say the historically
Muslim region of Greece is its least Muslim one, which is a sharper falsehood than the silence
it replaces.

THE CROSS-CHECK, and it is the reason to believe the whole construction. Two independent
routes to Greece's Muslim share: Pew's own country estimate for Greece is **5.12%**, and this
build comes out at **5.08%** — from Eurostat citizenship counts, Pew origin compositions and
one minority figure, none of which is Pew's Greece row. Nothing was tuned to make that happen,
and it is what settles the Albanian coefficient (taxonomy/origin_religion.py).
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
RAW = os.path.join(ROOT, "data", "raw", "gr")
OUT = os.path.join(ROOT, "data", "normalized", "gr.csv")
OUT_FOREIGN = os.path.join(ROOT, "data", "normalized", "gr_foreign.csv")

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) religiondots"}

# --- ESS --------------------------------------------------------------------------------
# The ESS data portal is an SPA that returns the same 1,070-byte shell for every path; its
# /env.js names the real backend in one line, and that backend answers ANONYMOUS queries.
# `analysis.frequencyTabulationByVariables` cross-tabulates server-side, so a country's
# religion-by-region table is one request and no microdata moves. See sources/gr.md.
ESS_API = "https://api.nsd.no/graphql"

# Greece is in rounds 1, 2, 4, 5, 10 and 11. Rounds 1, 2 and 4 have no `region` variable at
# all (E201VariableNotFound), so three are usable. (datafile id, version).
ESS_ROUNDS = {
    5: ("0189b86b-8aa4-4be3-88ad-39c58b02f19f", 89),
    10: ("f37d014a-6958-42d4-b03b-17c29e481d3d", 286),
    11: ("242aaa39-3bbb-40f5-98bf-bfb1ce53d8ef", 179),
}

# breakVariables takes variable NAMES, not the UUIDs the search returns, and its GraphQL type
# is [String!]! — a variable declared [String] or [String!] is rejected outright.
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

# --- the two authored cells ---------------------------------------------------------------
# Council of Europe ECRI, quoted in the US State Department's International Religious Freedom
# reports: "approximately 100,000-120,000" people in Thrace descend from the Muslim minority
# recognised by the 1923 Treaty of Lausanne. The midpoint is used; the range is in note_public.
THRACE_MINORITY = 110_000
THRACE_UNIT = "EL51"
ATHOS_UNIT = "ELZZ"

NUTS2 = {
    "EL30": "Attiki", "EL41": "Voreio Aigaio", "EL42": "Notio Aigaio", "EL43": "Kriti",
    "EL51": "Anatoliki Makedonia, Thraki", "EL52": "Kentriki Makedonia",
    "EL53": "Dytiki Makedonia", "EL54": "Ipeiros", "EL61": "Thessalia",
    "EL62": "Ionia Nisia", "EL63": "Dytiki Ellada", "EL64": "Sterea Ellada",
    "EL65": "Peloponnisos", "ELZZ": "Agion Oros (Mount Athos)",
}

# NUTS 2006 -> NUTS 2016. Greece was recoded from GR to EL in 2016 AND renumbered; the
# mapping is 1:1 and round 5 is the only pooled round that needs it.
GR_TO_EL = {
    "GR11": "EL51", "GR12": "EL52", "GR13": "EL53", "GR14": "EL61", "GR21": "EL54",
    "GR22": "EL62", "GR23": "EL63", "GR24": "EL64", "GR25": "EL65", "GR30": "EL30",
    "GR41": "EL41", "GR42": "EL42", "GR43": "EL43",
}

EU_AGGREGATES = {"TOTAL", "NAT", "FOR", "EU_FOR", "NEU", "EUR_NEU", "EUR_OTH", "AFR",
                 "AFR_OTH", "AME_N", "AME_N_OTH", "AME_X_N", "AME_X_N_OTH", "ASI",
                 "ASI_OTH", "OCE", "OCE_OTH", "RNC", "STLS", "UNK"}


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
    """Eurostat needs certifi's CURRENT bundle.

    The chain is GlobalSign Atlas R46 -> GlobalSign Root R46, and a `certifi` older than
    about 2021 does not carry that root — which fails as `unable to get local issuer
    certificate` on curl, urllib AND certifi at once, i.e. exactly like a server that omits
    its intermediate (spec §9h's test). It is not one. `pip install -U certifi` is the fix;
    sources/ru.py's note about Rosstat is a genuine case of the other thing and still stands.
    """
    import certifi
    ctx = ssl.create_default_context(cafile=certifi.where())
    u = EU_DATA + dataset + "?" + urllib.parse.urlencode(params, doseq=True)
    return json.load(urllib.request.urlopen(
        urllib.request.Request(u, headers=UA), timeout=900, context=ctx))


def fetch():
    os.makedirs(RAW, exist_ok=True)

    print("ESS…")
    for rnd, (fid, ver) in ESS_ROUNDS.items():
        dest = os.path.join(RAW, f"ess_r{rnd}.json")
        if os.path.exists(dest):
            print(f"  round {rnd} already on disk")
            continue
        d = _ess(ESS_TAB, {"id": fid, "v": ver,
                           "bv": ["region", "ctzcntr", "rlgdnm"]})
        gr = [x for x in d["analysis"]["frequencyTabulationByVariables"]["responses"]
              if x["by"][0]["value"] == "GR"]
        if not gr:
            sys.exit(f"!! ESS round {rnd} has no GR response")
        json.dump(gr[0]["response"], open(dest, "w", encoding="utf-8"),
                  ensure_ascii=False)
        n = sum(c["count"] for c in gr[0]["response"]["table"])
        print(f"  round {rnd}: {n:,.0f} weighted respondents -> {os.path.basename(dest)}")

    print("Eurostat census…")
    dest = os.path.join(RAW, "cens_21ctz_r3_el.json")
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


# =======================================================================================
# the citizen half
# =======================================================================================

def _ess_table(rnd):
    """(region, ctzcntr, rlgdnm) -> weighted count, for one round, Greek CITIZENS only."""
    d = json.load(open(os.path.join(RAW, f"ess_r{rnd}.json"), encoding="utf-8"))
    vv = {v["name"]: v for v in d["variableValues"]}
    order = [v["name"] for v in d["variableValues"]]
    labels = {n: [c["label"] for c in vv[n]["codeList"]] for n in order}
    # `region` is read as a CODE and everything else as a LABEL, and the difference matters:
    # ESS labels the Greek regions in Latin in round 5 ("Anatoliki Makedonia & Thraki") and
    # in Greek in rounds 10 and 11 ("Aνατολική Μακεδονία, Θράκη"), so pooling on labels
    # would silently split all thirteen regions in two and every one of them would look
    # half-sized. The codeList VALUES are NUTS codes and are stable apart from the 2016
    # GR->EL recode, which GR_TO_EL handles.
    values = {n: [c["value"] for c in vv[n]["codeList"]] for n in order}
    rows = []
    for cell in d["table"]:
        rec = {}
        for i, n in enumerate(order):
            rec[n] = (values if n == "region" else labels)[n][cell["path"][i]]
        rec["count"] = cell["count"]
        rows.append(rec)
    df = pd.DataFrame(rows)
    df["region"] = df["region"].map(lambda r: GR_TO_EL.get(r, r))
    return df


def _citizen_shares():
    """NUTS 2 x ESS denomination shares among Greek citizens, pooled over three rounds."""
    import gr2024

    frames = []
    for rnd in sorted(ESS_ROUNDS):
        df = _ess_table(rnd)
        cit = df[df["ctzcntr"] == "Yes"]
        n = cit["count"].sum()
        print(f"  round {rnd}: {n:,.0f} citizen respondents, "
              f"{cit['rlgdnm'].nunique()} denominations used, "
              f"{cit['region'].nunique()} regions")
        frames.append(cit.assign(round=rnd))
    pool = pd.concat(frames, ignore_index=True)

    # spec §3.5: refusals are marked, not filled.
    nc = pool[pool["rlgdnm"].isin(gr2024.EXCLUDED)]["count"].sum()
    total = pool["count"].sum()
    print(f"  pooled {total:,.0f} citizen respondents over {len(frames)} rounds; "
          f"refusals {nc:,.0f} ({100 * nc / total:.2f}%)")
    pool = pool[~pool["rlgdnm"].isin(gr2024.EXCLUDED)]

    unknown = sorted(set(pool["rlgdnm"]) - set(gr2024.MAP))
    if unknown:
        sys.exit(f"!! ESS denominations with no mapping: {unknown}")

    tab = pool.groupby(["region", "rlgdnm"])["count"].sum().unstack(fill_value=0.0)
    share = tab.div(tab.sum(axis=1), axis=0)
    nat = tab.sum() / tab.sum().sum()
    print("  national citizen shares: "
          + "  ".join(f"{k}:{100 * v:.2f}%" for k, v in nat.sort_values(
              ascending=False).items()))
    return share, 1.0 - nc / total


# =======================================================================================
# the foreign half, and the population both halves are scaled to
# =======================================================================================

def _census():
    """NUTS 3 x citizenship from cens_21ctz_r3, as a tidy frame."""
    d = json.load(open(os.path.join(RAW, "cens_21ctz_r3_el.json"), encoding="utf-8"))
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
    # THE GEO DIMENSION HOLDS EVERY NUTS LEVEL AT ONCE — `EL`, `EL3`, `EL30`, `EL301` are
    # all rows of the same table — so a prefix filter sums the same people four times. Only
    # the NUTS 3 leaves are read, and `ELZZZ` is one of them: Mount Athos, extra-regio.
    df = df[df["geo"].str.fullmatch(r"EL(\d{3}|ZZZ)")]
    df["unit"] = df["geo"].str[:4]
    return df


def _foreign_half(cen):
    """[geo_id, node, count] for foreign nationals, counted at NUTS 2."""
    import origin_religion as origin

    with zipfile.ZipFile(os.path.join(RAW, "pew.zip")) as z:
        name = [n for n in z.namelist() if n.endswith("(percentages).csv")][0]
        pew = pd.read_csv(io.BytesIO(z.read(name)))
    pew = pew[(pew["Year"] == 2020) & (pew["Level"] == 1)].set_index("Country")
    fams = origin.FAMILIES

    # Mount Athos is authored whole (taxonomy/gr2024.py) and its 125 "foreign residents" are
    # foreign-BORN MONKS — the Russian, Serbian, Romanian and Bulgarian houses — not a migrant
    # population. Running the origin model over them put 17.6 Muslims and 13 irreligious
    # people on the Holy Mountain, which is the sort of thing a model does when it is applied
    # one unit past where it makes sense.
    cen = cen[cen["unit"] != ATHOS_UNIT]
    leaf = cen[cen["citizen"].str.fullmatch(r"[A-Z]{2}") & (cen["citizen"] != "EL")]
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
        comp[iso] = origin.composition(iso, row, "other.gr")
    if unmapped:
        sys.exit(f"!! {len(unmapped)} citizenships have no composition: {unmapped}")

    # The unnamed remainder — STLS, RNC, UNK and rounding — is spread over the named
    # citizenships of its own region rather than dropped, so each region's foreign total is
    # the census's own. It is 0.16% of the foreign population.
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
    import gr2024

    print("ESS…")
    share, answered = _citizen_shares()

    print("Eurostat census…")
    cen = _census()
    pop = cen[cen["citizen"] == "TOTAL"].groupby("unit")["value"].sum()
    nat = cen[cen["citizen"] == "NAT"].groupby("unit")["value"].sum()
    for_ = cen[cen["citizen"] == "FOR"].groupby("unit")["value"].sum()
    print(f"  {pop.sum():,.0f} people = {nat.sum():,.0f} Greek citizens "
          f"+ {for_.sum():,.0f} foreign, over {len(pop)} NUTS 2 units")
    missing = sorted(set(NUTS2) - set(pop.index))
    if missing:
        sys.exit(f"!! census has no rows for {missing}")

    # ---- citizen half
    rows = []
    for unit in sorted(NUTS2):
        n_cit = float(nat.get(unit, 0.0)) * answered
        if unit == ATHOS_UNIT:
            # taxonomy/gr2024.py: an extra-regio monastic community no sample contains.
            rows.append((unit, NUTS2[unit], "Monastic community of Mount Athos",
                         float(pop.get(unit, 0.0)),
                         "authored, whole unit; only Orthodox monks may reside on Athos"))
            continue
        if unit not in share.index:
            sys.exit(f"!! no ESS respondents in {unit}")
        vec = share.loc[unit]
        minority = 0.0
        if unit == THRACE_UNIT:
            # spec §3.1's permitted SPLIT: the region's citizen total is unchanged, only its
            # composition. Everything else in the region is scaled down to make room.
            minority = min(float(THRACE_MINORITY), n_cit)
            if minority < THRACE_MINORITY:
                print(f"  !! Thrace minority capped at the region's citizen population")
            n_rest = n_cit - minority
        else:
            n_rest = n_cit
        for cat, s in vec.items():
            if s <= 0:
                continue
            rows.append((unit, NUTS2[unit], cat, s * n_rest, ""))
        if minority > 0:
            rows.append((unit, NUTS2[unit],
                         "Muslim minority of Thrace (Treaty of Lausanne)", minority,
                         "Council of Europe ECRI, 100,000-120,000; midpoint used"))
    cit = pd.DataFrame(rows, columns=["geo_id", "geo_name", "source_category", "count",
                                      "note"])
    cit["geo_level"] = "nuts2"
    cit["basis"] = "self_id"
    cit["year"] = 2024
    cit["source_id"] = "ess_r5_r10_r11_pooled"
    cit = cit[["geo_id", "geo_level", "geo_name", "source_category", "count", "basis",
               "year", "source_id", "note"]]
    print(f"  citizen half: {cit['count'].sum():,.0f} people "
          f"({100 * cit['count'].sum() / nat.sum():.2f}% of citizens; the rest declined)")

    unknown = sorted(set(cit["source_category"]) - set(gr2024.MAP))
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
        cit.assign(node=cit["source_category"].map(gr2024.resolve))[["node", "count"]],
        ext[["node", "count"]]], ignore_index=True)
    top = both.groupby("node")["count"].sum().sort_values(ascending=False)
    print("\nnational totals:")
    for node, c in top.head(16).items():
        print(f"  {c:>11,.0f}  {100 * c / drawn:5.2f}%  {node}")
    muslim = top[[n for n in top.index if n.startswith("islam")]].sum()
    print(f"\nCROSS-CHECK  Muslims {muslim:,.0f} = {100 * muslim / pop.sum():.2f}% of "
          f"Greece, against Pew's own 2020 country estimate of 5.12%")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    a = ap.parse_args()
    if a.fetch:
        fetch()
    else:
        build()
