"""Norway: no language statistics. Norwegian, plus North Sami from a cited estimate, plus
immigrant languages by country background, per kommune (Anita's 2026-10-05 rule for rich
countries; Sweden's method, sources/se.md).

    python sources/no_ssb.py --fetch     SSB tables 09817 and 07459, KLASS 91->552 (data/raw/no/)
    python sources/no_ssb.py             -> data/normalized/no.csv, data/geo/no/no_hexes.gpkg

SOURCES (SSB PxWeb API v0 and KLASS, open, no key; 1 January 2023, the last year on the 2020-2023
kommune codes that religiondots' GISCO LAU 2021 polygons carry):
  07459  population per kommune
  09817  immigrants (B) and Norwegian-born to two immigrant parents (C) per kommune by country
         background (landbakgrunn: own birth country for B, the parents' for C)
  KLASS 91 -> 552  SSB's 3-digit country codes to ISO alpha-3; queue.csv's iso3 -> cc

LANGUAGES. Each country background through origin_mix.mix(iso, "no"): the origin's drawn home
mix on this map, else its main language. Immigrants keep it; Norwegian-born to immigrant parents
keep it at 78% (Parkvall 2009 pp. 83-84 for Sweden's second generation with two foreign-born
parents, borrowed: no Norwegian figure was found), the rest Norwegian. Everyone else Norwegian
(Bokmal and Nynorsk are written standards of one language), less North Sami.

NORTH SAMI 10,000: Samisk sprakundersokelse 2012 (Solstad, ed., NIBR), "narmere 20.000"
North Sami speakers in Norway, Sweden and Finland, "om lag halvparten i Norge". Placed: half in
Kautokeino and Karasjok, where nearly everyone speaks it (the same report), split by population;
the other half over the rest of the Sami language administrative area in Troms og Finnmark (Tana,
Nesseby, Porsanger, Kafjord, Lavangen, Tjeldsund) by population. A
placement only; carved out of those kommuner's Norwegian. Lule and South Sami (a few hundred
speakers each) are not drawn.
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "taxonomy"))
from rdlink import RD_GEO  # noqa: E402

RAW = ROOT / "data" / "raw" / "no"
OUT = ROOT / "data" / "normalized" / "no.csv"
HEX = ROOT / "data" / "geo" / "no" / "no_hexes.gpkg"
LAU = RD_GEO / "no" / "no_lau.gpkg"
YEAR = "2023"
API = "https://data.ssb.no/api/v0/no/table/"
SECOND_GEN_KEEP = 0.78
NORWEGIAN = "indoeuropean.germanic.north.norwegian"
NORTH_SAMI = "uralic.saami_north"
SAMI_TOTAL = 10_000
SAMI_CORE = {"5430": "Kautokeino", "5437": "Karasjok"}      # 2020-2023 codes
SAMI_REST = {"5441": "Tana", "5442": "Nesseby", "5436": "Porsanger", "5426": "Kafjord",
             "5415": "Lavangen", "5412": "Tjeldsund"}
# SSB country backgrounds with no ISO code of their own
SPECIAL = {"161": "XK", "139": "GB"}       # Kosovo, United Kingdom: KLASS gives no single ISO match


def post(table, query, name):
    import requests
    r = requests.post(API + table, json={"query": query, "response": {"format": "json-stat2"}},
                      timeout=300)
    r.raise_for_status()
    (RAW / name).write_bytes(r.content)
    print(f"  {table} -> {name}: {len(r.content):,} bytes")


def kommuner():
    import geopandas as gpd
    g = gpd.read_file(LAU)
    g = g[g["lau"] != "NO-21"]
    k = g.dissolve(by="lau", as_index=False)[["lau", "name", "geometry"]]
    if len(k) != 356:
        raise SystemExit(f"{LAU}: {len(k)} kommuner, expected 356")
    return k


def fetch():
    import requests
    RAW.mkdir(parents=True, exist_ok=True)
    codes = sorted(kommuner()["lau"])
    item = lambda code, vals: {"code": code, "selection": {"filter": "item", "values": vals}}
    post("09817", [item("Region", codes), item("InnvandrKat", ["B", "C"]),
                   {"code": "Landbakgrunn", "selection": {"filter": "all", "values": ["*"]}},
                   item("ContentsCode", ["Personer1"]), item("Tid", [YEAR])], "ssb_09817.json")
    post("07459", [item("Region", codes), item("ContentsCode", ["Personer1"]),
                   item("Tid", [YEAR])], "ssb_07459.json")
    r = requests.get("https://data.ssb.no/api/klass/v1/classifications/91/corresponds",
                     params={"targetClassificationId": 552, "from": "2023-01-01"},
                     headers={"Accept": "application/json"}, timeout=60)
    r.raise_for_status()
    (RAW / "klass_91_552.json").write_bytes(r.content)


def jsonstat(name):
    d = json.loads((RAW / name).read_text(encoding="utf-8"))
    dims = d["id"]
    sizes = d["size"]
    cats = [list(d["dimension"][k]["category"]["index"]) for k in dims]
    if isinstance(d["dimension"][dims[0]]["category"]["index"], dict):
        cats = [sorted(d["dimension"][k]["category"]["index"],
                       key=d["dimension"][k]["category"]["index"].get) for k in dims]
    idx = pd.MultiIndex.from_product(cats, names=dims)
    v = pd.Series(d["value"], index=idx).fillna(0)
    assert len(v) == int(pd.Series(sizes).prod())
    return v, {k: d["dimension"][k]["category"].get("label", {}) for k in dims}


def iso2_map():
    corr = json.loads((RAW / "klass_91_552.json").read_text(encoding="utf-8"))
    a3 = {}
    for it in corr["correspondenceItems"]:
        a3.setdefault(it["sourceCode"], set()).add(it["targetCode"])
    q = pd.read_csv(ROOT / "queue.csv", usecols=["cc", "iso3"], keep_default_na=False)
    to2 = dict(zip(q["iso3"], q["cc"].str.upper()))
    to2.update({"KOS": "XK", "XKX": "XK", "SCG": "RS", "PSE": "PS", "SSD": "SS", "HKG": "HK",
                "MAC": "MO", "TWN": "TW", "ESH": "EH"})
    out = {}
    for code, s in a3.items():
        s = {t for t in s if t not in ("BVT", "SJM", "UMI", "HMD")} or s
        hits = {to2[t] for t in s if t in to2}
        if len(hits) == 1:
            out[code] = hits.pop()
    out.update(SPECIAL)
    return out


def main():
    from origin_mix import mix

    pop_s, _ = jsonstat("ssb_07459.json")
    pop = pop_s.groupby(level="Region").sum()
    imm, labels = jsonstat("ssb_09817.json")
    imm = imm.droplevel(["ContentsCode", "Tid"])
    lb = labels["Landbakgrunn"]
    k = kommuner()
    if set(pop.index) != set(k["lau"]):
        raise SystemExit("07459's kommuner differ from the LAU polygons")
    tot = int(pop.sum())
    print(f"07459: {len(pop)} kommuner, {tot:,} people on 1 January {YEAR}")

    # country rows only: SSB's aggregates (999 all, VES/IVE groups) are not countries
    countries = [c for c in imm.index.get_level_values("Landbakgrunn").unique()
                 if c.isdigit() and c != "999"]
    agg = imm.xs("999", level="Landbakgrunn")
    sumc = imm[imm.index.get_level_values("Landbakgrunn").isin(countries)] \
        .groupby(level=["Region", "InnvandrKat"]).sum()
    gap = (agg - sumc.reindex(agg.index)).abs().max()
    print(f"09817: {len(countries)} country backgrounds; immigrants {int(agg.xs('B', level='InnvandrKat').sum()):,}, "
          f"Norwegian-born to immigrant parents {int(agg.xs('C', level='InnvandrKat').sum()):,}; "
          f"countries against 'all' per kommune, worst gap {gap:.0f}")

    to2 = iso2_map()
    by_country = imm[imm.index.get_level_values("Landbakgrunn").isin(countries)]
    nat = by_country.groupby(level="Landbakgrunn").sum()
    unknown = {c: (lb.get(c, c), int(nat[c])) for c in countries if c not in to2 and nat[c] > 0}
    print(f"  country backgrounds with no ISO code ({sum(v for _, v in unknown.values()):,} people, "
          f"on `other`): {unknown}")

    rows = []
    mixes = {}
    for (reg, cat, c), n in by_country.items():
        if n <= 0:
            continue
        iso = to2.get(c)
        if iso and iso not in mixes:
            try:
                mixes[iso] = mix(iso, "no")
            except KeyError:
                mixes[iso] = {"other": 1.0}
        m = mixes[iso] if iso else {"other": 1.0}
        keep = 1.0 if cat == "B" else SECOND_GEN_KEEP
        for node, s in m.items():
            rows.append((reg, node, n * keep * s, cat))
        if keep < 1:
            rows.append((reg, NORWEGIAN, n * (1 - keep), cat))
    df = pd.DataFrame(rows, columns=["geo_id", "node", "count", "cat"])
    foreign = df.groupby(["geo_id", "node"])["count"].sum()

    # North Sami, carved from Norwegian in the Sami administrative area
    sami = {}
    core = sum(pop[c] for c in SAMI_CORE)
    rest = sum(pop[c] for c in SAMI_REST)
    for c in SAMI_CORE:
        sami[c] = SAMI_TOTAL / 2 * pop[c] / core
    for c in SAMI_REST:
        sami[c] = SAMI_TOTAL / 2 * pop[c] / rest
    print("  North Sami: " + ", ".join(f"{(SAMI_CORE | SAMI_REST)[c]} {v:,.0f} "
                                      f"({v / pop[c]:.0%})" for c, v in sami.items()))

    out = []
    for reg in pop.index:
        f = foreign.xs(reg, level="geo_id") if reg in foreign.index.get_level_values(0) else pd.Series(dtype=float)
        imm_people = agg.xs(reg, level="Region").sum()
        rowd = f.to_dict()
        rowd[NORWEGIAN] = rowd.get(NORWEGIAN, 0) + pop[reg] - imm_people - sami.get(reg, 0)
        if sami.get(reg):
            rowd[NORTH_SAMI] = sami[reg]
        if rowd[NORWEGIAN] < 0:
            raise SystemExit(f"{reg}: negative Norwegian")
        s = pd.Series(rowd)
        s = s * (pop[reg] / s.sum())          # the 'all' vs countries gap, tiny
        fl = s.apply(int)
        fl[(s - fl).sort_values(ascending=False).index[:int(pop[reg]) - int(fl.sum())]] += 1
        for node, c in fl.items():
            if c > 0:
                out.append(dict(geo_id=reg, geo_level="kommune",
                                geo_name=k.set_index("lau").loc[reg, "name"],
                                source_category=node, count=int(c), tier="derived",
                                year=int(YEAR), source_id="ssb_09817_07459"))
    out = pd.DataFrame(out)
    if int(out["count"].sum()) != tot:
        raise SystemExit("total moved")
    t = out.groupby("source_category")["count"].sum().sort_values(ascending=False)
    print(f"every kommune sums to 07459; {tot:,} people, {len(t)} nodes")
    for node, v in t.head(20).items():
        print(f"   {v:>9,}  {v / tot:6.2%}  {node}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False, encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}: {len(out)} rows")

    if not HEX.exists():
        from _grid import hex_layer
        units = k.rename(columns={"lau": "unit"})
        HEX.parent.mkdir(parents=True, exist_ok=True)
        hex_layer("no", units, census={u: int(pop[u]) for u in pop.index}, out=HEX)


if __name__ == "__main__":
    if "--fetch" in sys.argv:
        fetch()
    main()
