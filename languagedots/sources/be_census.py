"""Belgium: build data/normalized/be.csv (arrondissements) and be_communes.csv (placement).

    python sources/be_census.py --fetch     Eurostat census 2021 country of birth -> data/raw/be/
    python sources/be_census.py             build

Belgium has asked no language since the 1947 census. Anita's 2026-10-05 ruling for rich countries
with no language question (AGENT_BRIEF Â§2): the language of each area, plus surveys where they
exist, plus immigrant languages by country of birth. Every row is `derived`. Built per commune
(581, GISCO LAU 2021 with its 2021 population), summed to the 44 NUTS 3 arrondissements (Verviers
in two, its German-speaking communes being their own NUTS 3), which are the counting units; the
commune figures are kept only to place dots (be_communes.csv).

Per commune:
  * Brussels-Capital (19 communes): BRIO Taalbarometer 5 (2024), original home language.
  * the Vlaamse Rand (19 communes round Brussels): BRIO Taalbarometer Rand 2 (2018); the six
    facility communes from its cluster figure, varied between them by Le Soir's 2005 shares.
  * Voeren, Comines-Warneton, Mouscron: a minority share from what is printed (sources/be.md Â§3).
  * everywhere else: immigrants (Eurostat cens_21cob_r3, census 2021, by NUTS 3, spread over the
    arrondissement's communes by population) on their country's main language, less TeO2's
    share who speak only the national language at home (France's survey; Belgium has none,
    sources/be.md Â§2b), which goes to the area's language; the Belgian-born on the area's
    language: Dutch in Flanders, French in Wallonia, German in the German-speaking Community.
  * survey zones: the survey's "other" (neither French nor Dutch) is split over languages by the
    arrondissement's foreign-born, mapped to languages, French- and Dutch-speaking origins left
    out; no retention step (the survey already measured it).
"""
import argparse
import json
import os
import sys
import urllib.request

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "taxonomy"))
from fr_build import COUNTRY_LANG as FR_COUNTRY_LANG, TEO_FRENCH, teo_region  # noqa: E402

RD = os.path.join(os.path.dirname(ROOT), "religiondots")
LAU_XLSX = os.path.join(RD, "data", "geo", "lau2021", "EU-27-LAU-2021-NUTS-2021.xlsx")
RAW = os.path.join(ROOT, "data", "raw", "be")
COB = os.path.join(RAW, "cens_21cob_r3_be.json")
OUT = os.path.join(ROOT, "data", "normalized", "be.csv")
OUT_COM = os.path.join(ROOT, "data", "normalized", "be_communes.csv")
YEAR = 2021
SOURCE_ID = "be_cens21_cob_x_brio"

# country of birth -> language: France's table (sources/fr.md Â§2), plus France itself, which is
# the reporting country there. Belgium is the reporting country here and never appears.
COUNTRY_LANG = dict(FR_COUNTRY_LANG)
COUNTRY_LANG["FR"] = "French"
# territories Eurostat lists for Belgium that France's table has no row for (a few hundred
# people in all): their official language. Faroe and Greenland go on Danish, the language
# their Belgian-resident natives are likeliest to share with the census's country list.
COUNTRY_LANG.update({c: "English" for c in ("AI", "BM", "CK", "CQ", "FK", "GG", "GI", "IM",
                                            "JE", "KY", "MS", "PN", "SH", "TC", "VG")})
COUNTRY_LANG.update({c: "French" for c in ("BL", "MF", "NC", "PF", "PM", "TF", "WF")})
COUNTRY_LANG.update({"FO": "Danish", "GL": "Danish"})
LOCAL = {"French", "Dutch", "German"}
EAST_SUCCESSORS = ["RU", "UA", "BY", "MD", "GE", "AM", "AZ", "KZ", "KG", "TJ", "TM", "UZ", "EE",
                   "LV", "LT", "RS", "HR", "BA", "SI", "MK", "ME", "XK", "CZ", "SK"]

# ------------------------------------------------------------------------------------------
# Surveys. Pair answers ("Dutch/French") count half to each.
# ------------------------------------------------------------------------------------------
# BRIO Taalbarometer 5 (2024, 1,627 adults, Brussels-Capital Region), table 3 "oorspronkelijke
# thuistaal" (the language(s) spoken at home in the family one grew up in), factsheet at
# briobrussel.be/node/19094, image tabel 3.png. TB4 (2018) for comparison: 52.2 / 5.6 / 10.7 /
# 10.1 / 21.4.
TB5 = {"French": 41.3, "Dutch": 7.5, "Dutch/French": 4.3, "French/other": 18.0, "other": 28.8}
# BRIO Taalbarometer Rand 2 (Rudi Janssens, 2019), briobrussel.be/node/14829. Table 1,
# original home language, all 19 Rand communes, TBR2.
RAND = {"Dutch": 45.0, "Dutch/French": 10.2, "Dutch/other": 0.7, "French": 20.4,
        "French/other": 6.8, "other": 17.0}
# Figure 1, CURRENT home language by commune cluster, the facility cluster, TBR2 bars read off
# the chart (to about half a point; the page prints no numbers): the only per-cluster figure.
RAND_FAC = {"Dutch": 18.5, "Dutch/French": 14.6, "Dutch/other": 1.2, "French": 47.2,
            "French/other": 10.5, "other": 7.8}
# Le Soir 2005 survey of French speakers in the six facility communes, as cited by English
# Wikipedia ("Municipalities of Belgium with language facilities"). Used only for the pattern
# between the six: the level is BRIO's.
LESOIR = {"23098": 55, "23099": 78, "23100": 79, "23101": 58, "23102": 54, "23103": 72}
RAND_OTHER = {"23002", "23003", "23016", "23025", "23033", "23047", "23050", "23052", "23062",
              "23077", "24104", "23088", "23094"}
RAND_NAMES = {"23098": "Drogenbos", "23099": "Kraainem", "23100": "Linkebeek",
              "23101": "Sint-Genesius-Rode", "23102": "Wemmel", "23103": "Wezembeek-Oppem",
              "23002": "Asse", "23003": "Beersel", "23016": "Dilbeek", "23025": "Grimbergen",
              "23033": "Hoeilaart", "23047": "Machelen", "23050": "Meise", "23052": "Merchtem",
              "23062": "Overijse", "23077": "Sint-Pieters-Leeuw", "24104": "Tervuren",
              "23088": "Vilvoorde", "23094": "Zaventem"}

# The language-border facility communes with a printed figure (minority share of the
# non-immigrant population and of immigrants folded onto the local language):
#   Voeren: about 40% call themselves francophone (electoral lists; CEFAN, Universite Laval,
#     "La commune des Fourons"); 1947 gave French majorities in five of six villages.
#   Comines-Warneton: 7-8% of identity cards issued in Dutch, October 2012 (nl.wikipedia,
#     Komen-Waasten); 1947, language mostly spoken: 14.3% Dutch.
#   Mouscron: 1947, language mostly spoken: 23.0% Dutch (nl.wikipedia, Moeskroen), scaled by
#     Comines-Warneton's measured fall from 1947 to 2012 (7.5 / 14.3): no later figure exists.
MINORITY = {"73109": ("Voeren", "French", 0.40),
            "57097": ("Komen-Waasten", "Dutch", 0.075),          # Comines-Warneton
            "57096": ("Moeskroen", "Dutch", 0.230 * 0.075 / 0.143)}  # Mouscron


def shares(d):
    """Survey answers -> French / Dutch / other shares, pairs halved, normalised to 1."""
    out = {"French": 0.0, "Dutch": 0.0, "other": 0.0}
    for k, v in d.items():
        parts = k.split("/")
        for p in parts:
            out[p] += v / len(parts)
    t = sum(out.values())
    return {k: v / t for k, v in out.items()}


def fetch():
    os.makedirs(RAW, exist_ok=True)
    x = pd.read_excel(LAU_XLSX, sheet_name="BE")
    geos = sorted(x["NUTS 3 CODE"].unique())
    q = "&".join(f"geo={g}" for g in geos)
    url = ("https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/cens_21cob_r3"
           f"?sex=T&age=TOTAL&{q}&lang=en")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    d = json.load(urllib.request.urlopen(req, timeout=300))
    tmp = COB + ".tmp"
    json.dump(d, open(tmp, "w", encoding="utf-8"), ensure_ascii=False)
    os.replace(tmp, COB)
    print(f"wrote {COB}: {len(d['value']):,} values, {len(geos)} NUTS 3")


def communes():
    x = pd.read_excel(LAU_XLSX, sheet_name="BE")
    x["lau"] = (x["LAU CODE"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
                .str.zfill(5))
    x["unit"] = x["NUTS 3 CODE"].astype(str).str.strip()
    x["pop"] = pd.to_numeric(x["POPULATION"], errors="coerce").fillna(0.0)
    x["name"] = x["LAU NAME NATIONAL"].astype(str)
    if len(x) != 581 or x["lau"].duplicated().any():
        sys.exit(f"!! expected 581 distinct communes, got {len(x)}")
    return x[["lau", "unit", "name", "pop"]]


def cob():
    d = json.load(open(COB, encoding="utf-8"))
    dims, sizes = d["id"], d["size"]
    cats = {k: list(d["dimension"][k]["category"]["index"]) for k in dims}
    rows = []
    for k, v in d["value"].items():
        i, idx = int(k), []
        for s in reversed(sizes):
            idx.append(i % s)
            i //= s
        idx = list(reversed(idx))
        rec = {x: cats[x][idx[j]] for j, x in enumerate(dims)}
        rows.append((rec["geo"], rec["c_birth"], float(v)))
    df = pd.DataFrame(rows, columns=["unit", "cob", "n"])
    # continent block of each named country, from Eurostat's own ordering
    blocks, cur = {}, None
    starts = {"BG": "EU", "IS": "EUR", "AO": "AFR", "CA": "AME", "KZ": "ASI", "AU": "OCE"}
    for c in cats["c_birth"]:
        cur = starts.get(c, cur)
        if len(c) == 2 and c != "BE":
            blocks[c] = cur
    return df, blocks


_LOCAL_NODE = {"indoeuropean.romance.french": "French",
               "indoeuropean.germanic.continental.dutch": "Dutch",
               "indoeuropean.germanic.continental.german": "German"}


def language_mix(iso):
    """The shared origin table (sources/origin_mix.py, 2026-10-05), Belgium's three languages
    as labels. COUNTRY_LANG above is no longer read for the mixes, only to check coverage."""
    from origin_mix import mix
    return [(_LOCAL_NODE.get(n, n), s) for n, s in mix(iso, "be").items()]


def main():
    com = communes()
    df, blocks = cob()
    w = df.pivot_table(index="unit", columns="cob", values="n", aggfunc="sum").fillna(0.0)
    named = [c for c in w.columns if len(c) == 2 and c != "BE"]
    missing = sorted(set(named) - set(COUNTRY_LANG))
    if missing:
        sys.exit(f"!! countries of birth with no language: {missing}")
    if set(w.index) != set(com["unit"]):
        sys.exit(f"!! NUTS 3 mismatch: {sorted(set(w.index) ^ set(com['unit']))}")

    # checks: TOTAL = NAT + FOR + UNK; named countries against FOR
    print("Eurostat census 2021, country of birth, by NUTS 3:")
    tot = w["TOTAL"].sum()
    print(f"  total {tot:,.0f}; born in Belgium {w['NAT'].sum():,.0f}; abroad "
          f"{w['FOR'].sum():,.0f}; unknown {w.get('UNK', pd.Series(0)).sum():,.0f}")
    gap = (w["TOTAL"] - w["NAT"] - w["FOR"] - w.get("UNK", 0)).abs().max()
    if gap > 1:
        sys.exit(f"!! TOTAL != NAT + FOR + UNK (worst {gap:,.0f})")
    resid = w["FOR"] - w[named].sum(axis=1)
    # The residual is exactly Eurostat's EUR_OTH, "other European countries": people born in a
    # European state Belgium's register does not name (the USSR, Yugoslavia, Czechoslovakia
    # before they split, most likely). Spread over the arrondissement's own births in the
    # successor states, which is where those countries' people are.
    if (resid - w["EUR_OTH"]).abs().max() > 1:
        sys.exit("!! FOR minus the named countries is not EUR_OTH")
    east = [c for c in EAST_SUCCESSORS if c in named]
    print(f"  named countries {w[named].sum().sum():,.0f}; EUR_OTH {resid.sum():,.0f} "
          f"spread over the successor states {east}")
    if (resid < -1).any():
        sys.exit("!! named countries exceed FOR somewhere")
    lau_tot = com.groupby("unit")["pop"].sum()
    rel = (lau_tot / w["TOTAL"] - 1).abs()
    print(f"  LAU 2021 population against the census, worst arrondissement {rel.max():.3%}")
    if rel.max() > 0.01:
        sys.exit("!! LAU population and census total disagree by over 1%")

    # immigrants per arrondissement x language, before retention; FOR scaled up to cover the
    # unnamed residual
    imm = {}  # unit -> {(iso, label): n}
    for u in w.index:
        e = w.loc[u, east]
        if e.sum() <= 0:
            e = w[east].sum()
        extra = e / e.sum() * resid[u]
        d = {}
        for iso in named:
            n = w.loc[u, iso] + extra.get(iso, 0.0)
            if n <= 0:
                continue
            for lab, sh in language_mix(iso):
                d[(iso, lab)] = d.get((iso, lab), 0.0) + n * sh
        imm[u] = d

    def other_mix(u):
        """Foreign-born languages that are neither French nor Dutch, as shares."""
        m = {}
        for (iso, lab), n in imm[u].items():
            if lab in ("French", "Dutch"):
                continue
            m[lab] = m.get(lab, 0.0) + n
        t = sum(m.values())
        return {k: v / t for k, v in m.items()}

    # the census total, not the LAU one, is what each arrondissement holds; communes take
    # their LAU share of it
    com["P"] = com["pop"] / com["unit"].map(lau_tot) * com["unit"].map(w["TOTAL"])
    com["share"] = com["pop"] / com["unit"].map(lau_tot)

    def area_lang(u):
        return "German" if u == "BE336" else "Dutch" if u.startswith("BE2") else "French"

    # survey shares
    bru = shares(TB5)
    rand = shares(RAND)
    fac = shares(RAND_FAC)
    pop = dict(zip(com["lau"], com["P"]))
    for c in list(LESOIR) + sorted(RAND_OTHER):
        name = com.loc[com["lau"] == c, "name"].iloc[0]
        if RAND_NAMES[c] not in name:
            sys.exit(f"!! {c} is {name}, not {RAND_NAMES[c]}")
    for c, (nm, _, _) in MINORITY.items():
        name = com.loc[com["lau"] == c, "name"].iloc[0]
        if nm.split("-")[0] not in name:
            sys.exit(f"!! {c} is {name}, not {nm}")
    fac_pop = sum(pop[c] for c in LESOIR)
    k = fac["French"] * fac_pop / sum(LESOIR[c] / 100 * pop[c] for c in LESOIR)
    fac_c = {}
    for c in LESOIR:
        f = LESOIR[c] / 100 * k
        rest = 1 - f
        nd = fac["Dutch"] / (fac["Dutch"] + fac["other"])
        fac_c[c] = {"French": f, "Dutch": rest * nd, "other": rest * (1 - nd)}
    oth_pop = sum(pop[c] for c in RAND_OTHER)
    rand_rest = {l: (rand[l] * (fac_pop + oth_pop) - sum(fac_c[c][l] * pop[c] for c in LESOIR))
                 / oth_pop for l in rand}
    print("Brussels (TB5 2024, original home language): " +
          ", ".join(f"{k_} {v:.2%}" for k_, v in bru.items()))
    print("Rand, all 19 (TBR2, original): " + ", ".join(f"{k_} {v:.2%}" for k_, v in rand.items()))
    print("Rand facility cluster (TBR2, current): " +
          ", ".join(f"{k_} {v:.2%}" for k_, v in fac.items()))
    for c in LESOIR:
        print(f"  {RAND_NAMES[c]:20} French {fac_c[c]['French']:.1%}  Dutch "
              f"{fac_c[c]['Dutch']:.1%}  other {fac_c[c]['other']:.1%}  (Le Soir "
              f"{LESOIR[c]}%, scaled x{k:.3f})")
    print("Rand, the other 13 (residual): " +
          ", ".join(f"{k_} {v:.2%}" for k_, v in rand_rest.items()))
    if min(rand_rest.values()) < 0:
        sys.exit("!! negative residual share in the Rand")

    rows = []  # lau, unit, label, n, part
    moved = {}
    for r in com.itertuples(index=False):
        u, c, P = r.unit, r.lau, r.P
        if u == "BE100" or c in LESOIR or c in RAND_OTHER:
            sh = bru if u == "BE100" else fac_c[c] if c in LESOIR else rand_rest
            src = ("BRIO Taalbarometer 5" if u == "BE100" else "BRIO Taalbarometer Rand 2")
            rows.append((c, u, "French", P * sh["French"], src))
            rows.append((c, u, "Dutch", P * sh["Dutch"], src))
            for lab, s in other_mix(u).items():
                rows.append((c, u, lab, P * sh["other"] * s, src + ", other by country of birth"))
            continue
        local = area_lang(u)
        if c in MINORITY:
            _, mlang, msh = MINORITY[c]
            split = {local: 1 - msh, mlang: msh}
        else:
            split = {local: 1.0}
        n_imm = 0.0
        for (iso, lab), n in imm[u].items():
            n = n * r.share
            n_imm += n
            if lab in LOCAL:
                rows.append((c, u, lab, n, "immigrants by country of birth"))
                continue
            f = TEO_FRENCH[teo_region(iso, blocks.get(iso))]
            rows.append((c, u, lab, n * (1 - f), "immigrants by country of birth"))
            for l2, s2 in split.items():
                rows.append((c, u, l2, n * f * s2, "immigrants, national language at home"))
            moved[lab] = moved.get(lab, 0.0) + n * f
        for l2, s2 in split.items():
            rows.append((c, u, l2, (P - n_imm) * s2, "born in Belgium"))

    out = pd.DataFrame(rows, columns=["lau", "unit", "label", "count", "part"])
    if (out["count"] < -1e-6).any():
        sys.exit("!! negative count")
    chk = out.groupby("lau")["count"].sum()
    bad = (chk - com.set_index("lau")["P"]).abs().max()
    if bad > 0.5:
        sys.exit(f"!! communes do not sum to their population (worst {bad:,.1f})")
    print(f"TeO2 retention moved {sum(moved.values()):,.0f} immigrants onto the area's language; "
          "largest: " + ", ".join(f"{k_} {v:,.0f}" for k_, v in
                                   sorted(moved.items(), key=lambda x: -x[1])[:8]))

    import be2021
    for lab in out["label"].unique():
        be2021.resolve(lab)

    comm = out.groupby(["lau", "unit", "label"], as_index=False)["count"].sum()
    comm = comm[comm["count"] > 0]
    tmp = OUT_COM + ".tmp"
    comm.to_csv(tmp, index=False)
    os.replace(tmp, OUT_COM)

    arr = out.groupby(["unit", "label", "part"], as_index=False)["count"].sum()
    arr = arr[arr["count"] > 0].rename(columns={"unit": "geo_id", "label": "source_category"})
    arr["geo_level"] = "nuts3"
    arr["tier"] = "derived"
    arr["year"] = YEAR
    arr["source_id"] = SOURCE_ID
    arr = arr[["geo_id", "geo_level", "source_category", "count", "tier", "part", "year",
               "source_id"]]
    tmp = OUT + ".tmp"
    arr.to_csv(tmp, index=False)
    os.replace(tmp, OUT)
    if abs(arr["count"].sum() - tot) > 5:
        sys.exit(f"!! be.csv sums to {arr['count'].sum():,.0f}, census {tot:,.0f}")

    nat = arr.groupby("source_category")["count"].sum().sort_values(ascending=False)
    print(f"\nwrote {OUT}: {len(arr):,} rows, {arr['geo_id'].nunique()} arrondissements, "
          f"{arr['count'].sum():,.0f} people, {len(nat)} languages; {OUT_COM}: {len(comm):,} rows")
    for lab, n in nat.head(25).items():
        print(f"  {lab:22} {n:12,.0f}  {n / nat.sum():6.2%}")
    print("by region:")
    reg = arr.assign(r=arr["geo_id"].str[:3]).groupby(["r", "source_category"])["count"].sum()
    for rg in ("BE1", "BE2", "BE3"):
        s = reg.loc[rg].sort_values(ascending=False)
        print(f"  {rg}: " + ", ".join(f"{k_} {v / s.sum():.1%}" for k_, v in s.head(6).items()))
    g = arr[arr["geo_id"] == "BE336"].groupby("source_category")["count"].sum()
    print(f"  German-speaking Community (BE336): German {g.get('German', 0) / g.sum():.1%}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    a = ap.parse_args()
    if a.fetch:
        fetch()
    main()
