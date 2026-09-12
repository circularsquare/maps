"""France — two populations, one census, and the country spec §14.3 was written against.

Writes data/normalized/fr.csv (citizens) and fr_foreign.csv (foreign residents).

Usage:
    python sources/fr.py --fetch     # ESS tabulations, Eurostat census, Pew  (~7 MB)
    python sources/fr.py             # rebuild from data/raw/fr/

France has never asked about religion in a census and is forbidden by law from doing so, and
this is Greece's §9z pattern with both halves larger:

    French citizens     60.42M   ESS rounds 5-11 pooled, restricted to `ctzcntr = Yes`
    foreign residents    4.74M   Eurostat `cens_21ctz_r3` x Pew's origin compositions
    NOT DRAWN            2.28M   Corsica and the four overseas régions — see below

BOTH POPULATIONS COME OUT OF THE SAME CENSUS TABLE, which is why they partition exactly:
`cens_21ctz_r3` publishes `NAT` and `FOR` alongside its 201 named citizenships, so the
denominator of the citizen half and the numerator of the foreign half are rows of one file and
cannot drift apart. France's named citizenships cover **100.00%** of its foreign population,
which is better than Greece's 99.84% and means the unnamed-remainder rescale below is a
no-op here — it is kept because the next country will need it.

THE GEOGRAPHY IS THE FINDING, AND IT REVERSES A SCOUTING NOTE. `sources.md` §11l judged France
on the post-2016 régions — 13 units, 5.26M each, "fails badly" — and closed the country.
**ESS does not use those.** It carries the *anciennes régions*: NUTS-2010 `FR10`/`FR21`...`FR82`
in rounds 5-7, recoded to NUTS-2016 `FR10`/`FRB0`...`FRL0` in rounds 8-11, a clean 1:1 exactly
like Greece's GR->EL. That is 21 units at 3.10M people each. Still the coarsest counting
geography on this map — Russia's federal subjects are 1.82M — and spec §3.9b says that is not
a gate. It is stated in `grain` and in note_public rather than solved.

WHAT IS NOT DRAWN, AND WHY IT IS NOT BORROWED. ESS's French frame excludes Corsica and the
overseas régions, so 21 of the census's 27 NUTS-2 units have a citizen composition and six do
not. **Nothing is borrowed for the missing six.** Applying a metropolitan 49%-no-religion,
39%-Catholic composition to Martinique or La Réunion would be a plainly false statement about
two of the most religious parts of the republic, and applying it to Corsica would draw the
national average and tell the reader nothing true. They are left undrawn and spec §6.12's
coverage wash says so. Mayotte is a further step out: it is declared in the Eurostat geo
dimension and carries **no values at all**, so France's most Muslim territory is absent from
the source rather than from the decision.

THE CROSS-CHECK, per spec §14.10's fifth condition. Two routes sharing no input: this build
comes out **7.96% Muslim** against Pew's own independent 2020 estimate for France of **9.10%**.
Looser than Greece's 5.08 against 5.12, and the gap runs the direction a self-identification
survey always runs against a composite estimate. Nothing is tuned to close it.
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
RAW = os.path.join(ROOT, "data", "raw", "fr")
OUT = os.path.join(ROOT, "data", "normalized", "fr.csv")
OUT_FOREIGN = os.path.join(ROOT, "data", "normalized", "fr_foreign.csv")

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) religiondots"}

# --- ESS --------------------------------------------------------------------------------
# Open, keyless, and it cross-tabulates server-side, so no microdata moves — which is the
# whole reason this country is drawable at all under spec §14.3. See sources/gr.md §1 for the
# three argument-convention traps; they are unchanged here.
ESS_API = "https://api.nsd.no/graphql"

# France is in every round. Rounds 1-4 have no `region` variable (E201VariableNotFound), the
# same wall Greece hit, so SEVEN of eleven are usable — Greece got three. (datafile id, ver).
# Ids come from `search.searchDatafiles`, whose `instance` and `agencyId` are arguments of the
# FIELD and not members of `SearchInput`; passing them inside the input is rejected outright.
ESS_ROUNDS = {
    5: ("0189b86b-8aa4-4be3-88ad-39c58b02f19f", 89),
    6: ("450fa78e-68ab-493f-b169-dbc7ab8ffec2", 85),
    7: ("9c96a1b2-b027-43c1-8c74-e883f892d0bb", 91),
    8: ("ffc43f48-e15a-4a1c-8813-47eda377c355", 98),
    9: ("b2b0bf39-176b-4eca-8d26-3c05ea83d2cb", 280),
    10: ("f37d014a-6958-42d4-b03b-17c29e481d3d", 286),
    11: ("242aaa39-3bbb-40f5-98bf-bfb1ce53d8ef", 179),
}

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

# The 21 anciennes régions ESS samples, in NUTS 2016 codes. Romanised and unaccented, because
# spec's audience and the legend read Latin characters.
NUTS2 = {
    "FR10": "Ile-de-France",
    "FRB0": "Centre-Val de Loire",
    "FRC1": "Bourgogne", "FRC2": "Franche-Comte",
    "FRD1": "Basse-Normandie", "FRD2": "Haute-Normandie",
    "FRE1": "Nord-Pas-de-Calais", "FRE2": "Picardie",
    "FRF1": "Alsace", "FRF2": "Champagne-Ardenne", "FRF3": "Lorraine",
    "FRG0": "Pays de la Loire", "FRH0": "Bretagne",
    "FRI1": "Aquitaine", "FRI2": "Limousin", "FRI3": "Poitou-Charentes",
    "FRJ1": "Languedoc-Roussillon", "FRJ2": "Midi-Pyrenees",
    "FRK1": "Auvergne", "FRK2": "Rhone-Alpes",
    "FRL0": "Provence-Alpes-Cote d'Azur",
}

# THE FIVE OVERSEAS REGIONS, DRAWN FROM PEW — added 2026-09-07, and this is a THIRD instrument.
#
# ESS's French frame is metropolitan and excludes these, so the first build left 2.2M people
# undrawn rather than borrow a metropolitan composition for Martinique. Pew turns out to publish
# **all five as separate countries** in the same file `origin_religion.py` already reads, which
# is the one source that makes them drawable, and the reason it is legitimate is geometric
# rather than statistical:
#
#     EACH DOM IS EXACTLY ONE NUTS 2 UNIT, so Pew's territory estimate IS a unit estimate.
#
# There is no downscaling at all. spec §14.3's rule — never model at a finer resolution than the
# source publishes its magnitude at — is satisfied by identity, which is the cleanest this map
# ever gets to be about a modelled figure.
#
# `estimate` is §3.1's own basis for exactly this ("a compiler's judgement | Pew, WRD/WCD,
# ARDA"), and each DOM unit is built on that basis ALONE: the foreign half is deliberately not
# run over them, because Pew's estimate already covers every resident whatever passport they
# hold, and adding it would double-count. `_foreign_half` filters on NUTS2 for that reason.
#
# THE POPULATION CHECK PASSED BEFORE THIS WAS WIRED. Pew's own populations land within 1-2% of
# the 2021 census for the four units the census carries — 407,394/415,792, 356,615/360,748,
# 289,056/286,617, 861,446/871,156 — which is an independent confirmation of the magnitude from
# a source that is not the census. Per spec §3.4 the SHARES are Pew's and the TOTALS are the
# census's, for those four. Mayotte has no census row at all and is Pew's on both.
DOM = {
    "FRY1": ("Guadeloupe", "Guadeloupe"),
    "FRY2": ("Martinique", "Martinique"),
    "FRY3": ("Guyane", "French Guiana"),
    "FRY4": ("La Reunion", "Reunion"),
    "FRY5": ("Mayotte", "Mayotte"),
}

# Outside every instrument. Named so the build log states what it is dropping rather than
# discovering it as a set difference, and so a future round that adds one is noticed.
#
# CORSICA IS THE ONE THING STILL NOT DRAWN, and it was looked for rather than assumed. ESS's
# `region` is the only geography variable in the file (709 variables; the others are `cntry`,
# `regunit` and `domicil`), so no ESS round reaches it. Pew publishes the five DOM because they
# are territories with their own ISO codes and does NOT publish Corsica, because it is part of
# metropolitan France. What is left is a general-population survey with ~10 Corsican respondents
# in 2,000, or the Annuario Pontificio's diocese of Ajaccio — and that second one is a `roll`
# against a `self_id` map, which §3.1 forbids and which would draw Corsica far more Catholic
# than the mainland purely as an artefact of the instrument. It stays undrawn, at 0.51%.
NOT_DRAWN = {
    "FRM0": "Corse",
    "FRZZ": "Extra-Regio",
}

# NUTS 2010 -> NUTS 2016. The régions themselves did not change; France was RELETTERED in the
# 2016 revision, when the 22 anciennes régions became sub-units of the 13 new ones. The map is
# 1:1 and rounds 5-7 are the ones that need it. FR83/FRM0 (Corse) is listed for completeness
# and never appears, because ESS does not sample it.
FR10_TO_16 = {
    "FR10": "FR10", "FR21": "FRF2", "FR22": "FRE2", "FR23": "FRD2", "FR24": "FRB0",
    "FR25": "FRD1", "FR26": "FRC1", "FR30": "FRE1", "FR41": "FRF3", "FR42": "FRF1",
    "FR43": "FRC2", "FR51": "FRG0", "FR52": "FRH0", "FR53": "FRI3", "FR61": "FRI1",
    "FR62": "FRJ2", "FR63": "FRI2", "FR71": "FRK2", "FR72": "FRK1", "FR81": "FRJ1",
    "FR82": "FRL0", "FR83": "FRM0",
}

EU_AGGREGATES = {"TOTAL", "NAT", "FOR", "EU_FOR", "NEU", "EUR_NEU", "EUR_OTH", "AFR",
                 "AFR_OTH", "AME_N", "AME_N_OTH", "AME_X_N", "AME_X_N_OTH", "ASI",
                 "ASI_OTH", "OCE", "OCE_OTH", "RNC", "STLS", "UNK"}

PEW_FRANCE_MUSLIM = 9.10        # Pew, Religious Composition by Country 2020, for the check


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
    """Eurostat needs certifi's CURRENT bundle — see sources/gr.py for why that is not a
    server-side fault, and why `pip install -U certifi` is the fix."""
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
        fr = [x for x in d["analysis"]["frequencyTabulationByVariables"]["responses"]
              if x["by"][0]["value"] == "FR"]
        if not fr:
            sys.exit(f"!! ESS round {rnd} has no FR response")
        json.dump(fr[0]["response"], open(dest, "w", encoding="utf-8"),
                  ensure_ascii=False)
        n = sum(c["count"] for c in fr[0]["response"]["table"])
        print(f"  round {rnd}: {n:,.0f} weighted respondents -> {os.path.basename(dest)}")

    print("Eurostat census…")
    dest = os.path.join(RAW, "cens_21ctz_r3_fr.json")
    if not os.path.exists(dest):
        d = _eurostat(EU_CTZ, format="JSON", lang="EN", age="TOTAL", sex="T")
        json.dump(d, open(dest, "w", encoding="utf-8"))
        print(f"  {os.path.getsize(dest):,} bytes")
    else:
        print("  already on disk")

    print("Pew…")
    dest = os.path.join(RAW, "pew.zip")
    if not os.path.exists(dest):
        src = os.path.join(ROOT, "data", "raw", "gr", "pew.zip")
        if os.path.exists(src):
            # The same file Greece and Spain already read. Copied rather than re-fetched so
            # a rebuild cannot silently pick up a different Pew vintage than gr.csv used.
            with open(src, "rb") as a, open(dest, "wb") as b:
                b.write(a.read())
            print(f"  copied from data/raw/gr/pew.zip ({os.path.getsize(dest):,} bytes)")
        else:
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
    """(region, ctzcntr, rlgdnm) -> weighted count, for one round, French CITIZENS only."""
    d = json.load(open(os.path.join(RAW, f"ess_r{rnd}.json"), encoding="utf-8"))
    vv = {v["name"]: v for v in d["variableValues"]}
    order = [v["name"] for v in d["variableValues"]]
    labels = {n: [c["label"] for c in vv[n]["codeList"]] for n in order}
    # `region` is read as a CODE and everything else as a LABEL. Greece needed this because
    # ESS switched from Latin to Greek labels mid-series; France needs it for a quieter
    # reason — rounds 5-7 and 8-11 label the same twenty-one régions identically but code
    # them in different NUTS vintages, and the labels have drifted in punctuation anyway
    # ("Nord - Pas-de-Calais" vs "Nord-Pas de Calais", "Centre" vs "Centre — Val de Loire").
    # Pooling on labels would split six régions in two and every one would look half-sized.
    values = {n: [c["value"] for c in vv[n]["codeList"]] for n in order}
    rows = []
    for cell in d["table"]:
        rec = {}
        for i, n in enumerate(order):
            rec[n] = (values if n == "region" else labels)[n][cell["path"][i]]
        rec["count"] = cell["count"]
        rows.append(rec)
    df = pd.DataFrame(rows)
    df["region"] = df["region"].map(lambda r: FR10_TO_16.get(r, r))
    stray = sorted(set(df["region"]) - set(NUTS2) - set(DOM) - set(NOT_DRAWN))
    if stray:
        sys.exit(f"!! round {rnd} has unrecognised region codes: {stray}")
    return df


def _citizen_shares():
    """NUTS 2 x ESS denomination shares among French citizens, pooled over seven rounds."""
    import fr2024

    frames = []
    for rnd in sorted(ESS_ROUNDS):
        df = _ess_table(rnd)
        cit = df[df["ctzcntr"] == "Yes"]
        n = cit["count"].sum()
        used = cit[cit["count"] > 0]["rlgdnm"].nunique()
        print(f"  round {rnd:2}: {n:7,.0f} citizen respondents, {used:2} denominations "
              f"used, {cit['region'].nunique()} regions")
        frames.append(cit.assign(round=rnd))
    pool = pd.concat(frames, ignore_index=True)

    # spec §3.5: refusals are marked, not filled.
    nc = pool[pool["rlgdnm"].isin(fr2024.EXCLUDED)]["count"].sum()
    total = pool["count"].sum()
    print(f"  pooled {total:,.0f} citizen respondents over {len(frames)} rounds; "
          f"refusals {nc:,.0f} ({100 * nc / total:.2f}%)")
    pool = pool[~pool["rlgdnm"].isin(fr2024.EXCLUDED)]

    unknown = sorted(set(pool["rlgdnm"]) - set(fr2024.MAP))
    if unknown:
        sys.exit(f"!! ESS denominations with no mapping: {unknown}")

    covered = sorted(set(pool["region"]))
    if set(covered) != set(NUTS2):
        sys.exit(f"!! ESS region set is not the expected 21: "
                 f"{sorted(set(covered) ^ set(NUTS2))}")

    tab = pool.groupby(["region", "rlgdnm"])["count"].sum().unstack(fill_value=0.0)
    share = tab.div(tab.sum(axis=1), axis=0)
    nat = tab.sum() / tab.sum().sum()
    print("  national citizen shares: "
          + "  ".join(f"{k}:{100 * v:.2f}%" for k, v in nat.sort_values(
              ascending=False).items()))

    # The smallest cell a régional share rests on, printed rather than asserted: it is the
    # honest bound on how far down the legend a reader should trust (spec §3.9b's "say what
    # the country cannot show").
    n_per = tab.sum(axis=1).sort_values()
    print(f"  thinnest regions: "
          + ", ".join(f"{NUTS2[u]} {v:,.0f}" for u, v in n_per.head(3).items()))
    return share, 1.0 - nc / total


# =======================================================================================
# the foreign half, and the population both halves are scaled to
# =======================================================================================

def _census():
    """NUTS 3 x citizenship from cens_21ctz_r3, as a tidy frame."""
    d = json.load(open(os.path.join(RAW, "cens_21ctz_r3_fr.json"), encoding="utf-8"))
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
    # THE GEO DIMENSION HOLDS EVERY NUTS LEVEL AT ONCE — `FR`, `FR1`, `FR10`, `FR101` are all
    # rows of the same table — so a prefix filter sums the same people four times. Only the
    # NUTS 3 leaves are read. France's are five characters and the last three are not all
    # digits (`FRB01`, `FRY10`, `FRZZZ`), which is why this is a length test and not the
    # digit test gr.py could get away with.
    df = df[df["geo"].str.fullmatch(r"FR\w{3}")]
    df["unit"] = df["geo"].str[:4]
    return df


def _foreign_half(cen):
    """[geo_id, node, count] for foreign nationals, counted at NUTS 3 — 94 départements.

    **AT NUTS 3 SINCE 2026-09-08, AND THIS REVERSES WHAT THIS FUNCTION USED TO SAY.** It
    used to count at NUTS 2 on Greece's rule — *mixing would put the sharper geography on
    the half with the weaker claim to it* — reasoning that drawing a model five times finer
    than the survey would make the most-inferred part of France also the most
    precise-looking part of it.

    Italy (§9as) showed that rule has an unstated premise: **that the fine half is the small
    half.** It is not, here. The foreign half is 4.9M people and it holds most of what a
    religion map of France is for — and the price of the old rule was that the map could say
    Île-de-France is 16% Muslim and nothing whatever about Seine-Saint-Denis, which
    `sources/fr.md` §8 named as the single biggest thing wrong with this country.

    At NUTS 3 the foreign half alone puts **Seine-Saint-Denis at 12.9% Muslim against
    Seine-et-Marne's 3.7%** — a 3.5x spread inside one région that the old geography
    flattened to a single number. The basis does not change and neither does the model; only
    the resolution, and Eurostat publishes at this resolution, so §14.3's *never model finer
    than the source publishes* is satisfied by the source rather than by an argument.

    The citizen half stays at the région because ESS has nothing finer (§2), so France is
    now mixed-resolution the way Italy is: `build()` spreads the citizen composition down to
    these same units by each département's own citizen population, which is spatially a
    no-op (§8.2) and keeps one `unit` column for scatter.py.
    """
    import origin_religion as origin

    with zipfile.ZipFile(os.path.join(RAW, "pew.zip")) as z:
        name = [n for n in z.namelist() if n.endswith("(percentages).csv")][0]
        pew = pd.read_csv(io.BytesIO(z.read(name)))
    pew = pew[(pew["Year"] == 2020) & (pew["Level"] == 1)].set_index("Country")
    fams = origin.FAMILIES

    cen = cen[cen["unit"].isin(NUTS2)]
    leaf = cen[cen["citizen"].str.fullmatch(r"[A-Z]{2}") & (cen["citizen"] != "FR")]
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
        comp[iso] = origin.composition(iso, row, "other.fr")
    if unmapped:
        sys.exit(f"!! {len(unmapped)} citizenships have no composition: {unmapped}")

    # The unnamed remainder — STLS, RNC, UNK and rounding — is spread over the named
    # citizenships of its own DÉPARTEMENT rather than dropped, so each département's foreign
    # total is the census's own. In France it is nothing at all: the named list covers
    # 100.00%. Kept because Greece's was 0.16% and the next country's will not be zero.
    for_by_geo = cen[cen["citizen"] == "FOR"].groupby("geo")["value"].sum()
    rows = []
    for geo, g in leaf.groupby("geo"):
        base = g["value"].sum()
        if base <= 0:
            continue
        scale = float(for_by_geo.get(geo, 0.0)) / base
        acc = {}
        for iso, v in g.groupby("citizen")["value"].sum().items():
            if v <= 0:
                continue
            for node, share in comp[iso].items():
                acc[node] = acc.get(node, 0.0) + v * scale * share
        for node, c in acc.items():
            rows.append((geo, node, c))
    df = pd.DataFrame(rows, columns=["geo_id", "node", "count"])
    print(f"  foreign half: {df['count'].sum():,.0f} people, {df['node'].nunique()} nodes "
          f"over {df['geo_id'].nunique()} départements")
    return df


# =======================================================================================
# the overseas half — Pew, one estimate per unit
# =======================================================================================

def _pew_2020():
    with zipfile.ZipFile(os.path.join(RAW, "pew.zip")) as z:
        name = [n for n in z.namelist() if n.endswith("(percentages).csv")][0]
        pew = pd.read_csv(io.BytesIO(z.read(name)))
    return pew[(pew["Year"] == 2020) & (pew["Level"] == 1)].set_index("Country")


def _dom_half(cen):
    """[geo_id, geo_name, source_category, count, note] for the five overseas régions.

    Pew's seven families, at the one geography where a Pew country row is also a NUTS 2 unit.
    The families are carried through as SOURCE CATEGORIES rather than resolved here, so that
    taxonomy/fr2024.py holds every depth call in one place and tools/check_mapping.py sees
    them — which is the arrangement gr2024.py uses for the two cells sources/gr.py invents.
    """
    import origin_religion as origin

    pew = _pew_2020()
    missing = [n for _, (_, n) in DOM.items() if n not in pew.index]
    if missing:
        sys.exit(f"!! Pew has no 2020 Level-1 row for {missing} — it had all five on "
                 f"2026-09-07 and a vintage that drops one cannot be silently skipped")

    pop_cen = cen[cen["citizen"] == "TOTAL"].groupby("unit")["value"].sum()
    rows = []
    for unit, (name, pew_name) in sorted(DOM.items()):
        row = pew.loc[pew_name]
        shares = {f: float(row[f]) for f in origin.FAMILIES}
        tot = sum(shares.values())
        if not 99.0 <= tot <= 101.0:
            sys.exit(f"!! Pew's families for {pew_name} sum to {tot:.2f}, not 100")

        # spec §3.4: structure from the source that has it, total from the recent one. Four of
        # the five have a 2021 census row and it is preferred; Mayotte has none and is Pew's.
        census_total = float(pop_cen.get(unit, 0.0))
        if census_total > 0:
            total, note = census_total, "Pew 2020 shares on the 2021 census total"
            ratio = float(row["Population"]) / census_total
            print(f"  {unit} {name:<12} census {census_total:>9,.0f}  "
                  f"Pew {float(row['Population']):>9,.0f}  ratio {ratio:.3f}")
            if not 0.85 <= ratio <= 1.15:
                sys.exit(f"!! Pew and the census disagree by more than 15% on {name} — "
                         f"that is a join or vintage problem, not an estimate difference")
        else:
            total = float(row["Population"])
            note = "Pew 2020 shares AND total; the census table has no row for this unit"
            print(f"  {unit} {name:<12} census        —  "
                  f"Pew {total:>9,.0f}  (Pew supplies the total too)")

        # EACH DOM IS ONE NUTS 2 UNIT AND ALSO EXACTLY ONE NUTS 3 UNIT — `FRY1` holds only
        # `FRY10` — so moving the country to NUTS 3 costs these five nothing: the code gains
        # a digit and the estimate is unchanged. §9ag's "a Pew country row IS a unit row"
        # still holds by identity. Mayotte is the one that has to be written down rather
        # than read off the census, because the census has no row for it at either level.
        geo = unit + "0"
        for fam, pct in shares.items():
            if pct <= 0:
                continue
            rows.append((geo, name, fam.replace("_", " "), total * pct / tot, note))
    df = pd.DataFrame(rows, columns=["geo_id", "geo_name", "source_category", "count",
                                     "note"])
    print(f"  overseas half: {df['count'].sum():,.0f} people over {df['geo_id'].nunique()} "
          f"units, {df['source_category'].nunique()} Pew families")
    return df


# =======================================================================================
# build
# =======================================================================================

def build():
    import fr2024

    print("ESS…")
    share, answered = _citizen_shares()

    print("Eurostat census…")
    cen = _census()
    pop = cen[cen["citizen"] == "TOTAL"].groupby("unit")["value"].sum()
    nat = cen[cen["citizen"] == "NAT"].groupby("unit")["value"].sum()
    for_ = cen[cen["citizen"] == "FOR"].groupby("unit")["value"].sum()
    print(f"  {pop.sum():,.0f} people = {nat.sum():,.0f} French citizens "
          f"+ {for_.sum():,.0f} foreign, over {len(pop)} NUTS 2 units")
    missing = sorted(set(NUTS2) - set(pop.index))
    if missing:
        sys.exit(f"!! census has no rows for {missing}")

    # Everything the census has that no instrument reaches. Reported unit by unit, because a
    # silently dropped region is exactly the kind of hole spec §6.12 exists to make visible
    # and this is where the map learns about it.
    outside = sorted(set(pop.index) - set(NUTS2) - set(DOM))
    dropped = float(pop[pop.index.isin(outside)].sum())
    print(f"  reached by no instrument — NOT DRAWN, and not borrowed (see the docstring):")
    for u in outside:
        if pop.get(u, 0) <= 0:
            continue
        print(f"    {u}  {NOT_DRAWN.get(u, '?'):<14} {pop[u]:>10,.0f}")
    unknown_units = [u for u in outside if u not in NOT_DRAWN]
    if unknown_units:
        sys.exit(f"!! census has NUTS 2 units this build has never heard of: "
                 f"{unknown_units} — a new region cannot be dropped silently")
    if pop.get("FRY5", 0) > 0:
        sys.exit("!! Mayotte now carries census values — it was empty when this was written "
                 "and is drawn from Pew's total instead. A census row is better, but it also "
                 "means the foreign half would now reach the unit and double-count against "
                 "Pew's estimate, so this is a decision for a person and not a rescale.")

    # ---- citizen half, spread from the région to its départements
    # ESS carries nothing below the région, so the COMPOSITION here is the région's and says
    # so in `note`. What the split buys is that each département takes its own citizen
    # count, and the drawn unit is then the same one the foreign half uses — scatter.py
    # takes a single `unit` column, so the two halves have to meet at one level. Spreading a
    # coarse composition down by population is spatially a no-op (§8.2): the dots would have
    # been placed by population inside the région anyway.
    nat3 = cen[cen["citizen"] == "NAT"].groupby("geo")["value"].sum()
    rows = []
    for unit in sorted(NUTS2):
        vec = share.loc[unit]
        kids = sorted(g for g in nat3.index if g.startswith(unit))
        if not kids:
            sys.exit(f"!! {unit} has no NUTS 3 children in the census")
        for geo in kids:
            n_cit = float(nat3.get(geo, 0.0)) * answered
            if n_cit <= 0:
                continue
            note = (f"composition is {unit} ({NUTS2[unit]})'s — ESS carries nothing below "
                    f"the région; spread to this département by its own citizen population")
            for cat, s in vec.items():
                if s <= 0:
                    continue
                rows.append((geo, NUTS2[unit], cat, s * n_cit, note))
    cit = pd.DataFrame(rows, columns=["geo_id", "geo_name", "source_category", "count",
                                      "note"])
    cit["basis"] = "self_id"
    cit["year"] = 2024
    cit["source_id"] = "ess_r5_to_r11_pooled"
    drawn_nat = float(nat[nat.index.isin(NUTS2)].sum())
    print(f"  citizen half: {cit['count'].sum():,.0f} people "
          f"({100 * cit['count'].sum() / drawn_nat:.2f}% of the citizens in these 21 "
          f"regions; the rest declined)")

    # ---- the overseas half, which shares the citizen file's schema and not its basis
    print("overseas régions (Pew)…")
    dom = _dom_half(cen)
    dom["basis"] = "estimate"
    dom["year"] = 2020
    dom["source_id"] = "pew_2020"

    cit = pd.concat([cit, dom], ignore_index=True)
    cit["geo_level"] = "nuts3"
    cit = cit[["geo_id", "geo_level", "geo_name", "source_category", "count", "basis",
               "year", "source_id", "note"]]

    unknown = sorted(set(cit["source_category"]) - set(fr2024.MAP))
    if unknown:
        sys.exit(f"!! unmapped source categories: {unknown}")

    # ---- foreign half
    print("foreign half…")
    ext = _foreign_half(cen)
    ext["geo_level"] = "nuts3"
    # The name is still the RÉGION's — NUTS 3 codes are all this table carries and Eurostat's
    # département labels are not read anywhere else, so naming the parent is honest and
    # matches what the citizen half writes for the same unit.
    ext["geo_name"] = ext["geo_id"].str[:4].map(NUTS2)
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
    # France's denominator is the census PLUS Mayotte, which the census table omits. Leaving
    # Mayotte out of the denominator as well as the numerator would report a coverage figure
    # that quietly agreed the territory does not exist.
    mayotte = float(_pew_2020().loc[DOM["FRY5"][1], "Population"])
    france = float(pop.sum()) + mayotte
    print(f"\nwrote {OUT}  ({len(cit):,} rows, "
          f"{cit['source_category'].nunique()} source categories)")
    print(f"wrote {OUT_FOREIGN}  ({len(ext):,} rows, {ext['node'].nunique()} nodes)")
    print(f"drawn {drawn:,.0f} of {france:,.0f} — {100 * drawn / france:.2f}% "
          f"(census {pop.sum():,.0f} + Mayotte {mayotte:,.0f} from Pew; "
          f"{dropped:,.0f} in Corsica, {100 * dropped / france:.2f}%, reached by nothing)")
    units = sorted(set(cit["geo_id"]) | set(ext["geo_id"]))
    n_metro = len([u for u in units if not u.startswith("FRY")])
    print(f"  units drawn: {n_metro} metropolitan départements + "
          f"{len(units) - n_metro} overseas = {len(units)}, at "
          f"{drawn / len(units):,.0f} people each — the citizen composition is still the "
          f"{len(NUTS2)} régions', and every citizen row's `note` says so")

    both = pd.concat([
        cit.assign(node=cit["source_category"].map(fr2024.resolve))[["node", "count"]],
        ext[["node", "count"]]], ignore_index=True)
    top = both.groupby("node")["count"].sum().sort_values(ascending=False)
    print("\nnational totals:")
    for node, c in top.head(16).items():
        print(f"  {c:>11,.0f}  {100 * c / drawn:5.2f}%  {node}")

    # spec §14.10's fifth condition: a fractional-share model states what its output was
    # checked against. Pew's France row is not an input to anything above.
    muslim = top[[n for n in top.index if n.startswith("islam")]].sum()
    got = 100 * muslim / drawn
    print(f"\nCROSS-CHECK  Muslims {muslim:,.0f} = {got:.2f}% of the drawn population, "
          f"against Pew's own 2020 country estimate of {PEW_FRANCE_MUSLIM:.2f}%")
    if not 0.6 * PEW_FRANCE_MUSLIM <= got <= 1.4 * PEW_FRANCE_MUSLIM:
        sys.exit(f"!! the two routes have diverged past 40% — that is a broken join or a "
                 f"changed source, not a modelling difference")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    a = ap.parse_args()
    if a.fetch:
        fetch()
    else:
        build()
