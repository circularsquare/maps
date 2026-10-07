"""Denmark: no language statistics. Danish, plus German in Sonderjylland from a cited estimate,
plus immigrant languages by country of origin, per kommune (Anita's 2026-10-05 rule for rich
countries; Norway's and Sweden's method, sources/no.md, sources/se.md).

    python sources/dk_dst.py --fetch     Statistics Denmark FOLK1C (data/raw/dk/)
    python sources/dk_dst.py             -> data/normalized/dk.csv, data/geo/dk/dk_hexes.gpkg

SOURCE (Statistics Denmark StatBank API, open, no key): FOLK1C, population on 1 January 2026
(2026Q1) per kommune (98, plus Christianso 411) by ancestry (HERKOMST: 5 Danish origin,
4 immigrants, 3 descendants) and country of origin (IELAND: for immigrants the birth country,
for descendants the parents'). The kommune codes are the LAU codes of religiondots'
`dk_lau.gpkg` (99 polygons, unchanged since 2007).

COUNTRY -> LANGUAGE. DST's English country names -> alpha-2 by queue.csv's names plus the
NAME_FIX table below; each through origin_mix.mix(iso, "dk") (the origin's drawn home mix on this
map, else its main language). Immigrants keep it; descendants keep it at 78% (Parkvall 2009 for
Sweden's second generation with two foreign-born parents, borrowed as Norway did), the rest
Danish. Stateless and "not stated" origins on `other`. Everyone of Danish origin is Danish.

GERMAN 5,000: the German minority in Sonderjylland is put at about 15,000 people, 6% of the four
Sonderjylland kommuner (Graenseforeningen's leksikon, "Tyske mindretal, Det", 2022 figure);
about two-thirds of them speak Danish, often Sonderjysk, at home (Den Store Danske
Encyclopaedi vol. 4, 1996, as cited by da.wikipedia "Det tyske mindretal i Nordslesvig").
So one third, 5,000, are drawn on German, over Aabenraa, Haderslev, Sonderborg and Tonder by
population, carved out of those kommuner's Danish.
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

RAW = ROOT / "data" / "raw" / "dk"
RAWF = RAW / "dst_folk1c_2026K1.csv"
OUT = ROOT / "data" / "normalized" / "dk.csv"
HEX = ROOT / "data" / "geo" / "dk" / "dk_hexes.gpkg"
LAU = RD_GEO / "dk" / "dk_lau.gpkg"
TID = "2026K1"
SECOND_GEN_KEEP = 0.78
DANISH = "indoeuropean.germanic.north.danish"
GERMAN = "indoeuropean.germanic.continental.german"
GERMAN_TOTAL = 5_000
SONDERJYLLAND = {"580": "Aabenraa", "510": "Haderslev", "540": "Sonderborg", "550": "Tonder"}
OTHER = "other"
REALM = {"BEF5G": "GL", "BEF5F": "FO"}   # born in the Realm's other parts: Danish origin in FOLK1C
# DST English names queue.csv does not carry; None = no single country (drawn on `other`)
NAME_FIX = {
    "Stateless": None, "Europe not stated": None, "Africa not stated": None,
    "Asia not stated": None, "America not stated": None, "Not stated": None,
    "Oceania not stated": None, "Unknown country": None, "Stateless/unknown country": None,
    "Czechoslovakia": "QT", "GDR": "DE", "Former Yugoslavia": "YU", "Yugoslavia": "YU",
    "Soviet Union": "SU", "USSR": "SU",
    "United Kingdom": "GB", "USA": "US", "Republic of North Macedonia": "MK",
    "Congo, Democratic Republic": "CD", "Czech Republic": "CZ",
    "Yugoslavia, Federal Republic": "YU", "Serbia and Montenegro": "YU", "Gambia, The": "GM",
    "Ivory Coast": "CI", "Congo, Republic": "CG", "Gaza": "PS", "West Bank": "PS",
    "East Jerusalem": "PS", "Equatorial Guinea": "GQ", "Sao Tome and Principe": "ST",
    "East Timor": "TL", "Middle East not stated": None, "West Indies": None,
    "South and central America not stated": None, "Pacific Islands": None,
}


def fetch():
    import requests
    RAW.mkdir(parents=True, exist_ok=True)
    q = {"table": "FOLK1C", "format": "BULK", "lang": "en", "valuePresentation": "CodeAndValue",
         "variables": [{"code": "OMRÅDE", "values": ["*"]}, {"code": "KØN", "values": ["TOT"]},
                       {"code": "ALDER", "values": ["IALT"]}, {"code": "HERKOMST", "values": ["*"]},
                       {"code": "IELAND", "values": ["*"]}, {"code": "Tid", "values": [TID]}]}
    r = requests.post("https://api.statbank.dk/v1/data", json=q, timeout=300)
    r.raise_for_status()
    RAWF.write_bytes(r.content)
    print(f"  FOLK1C {TID} -> {RAWF.name}: {len(r.content):,} bytes")
    for t in REALM:
        q = {"table": t, "format": "CSV", "lang": "en", "valuePresentation": "CodeAndValue",
             "variables": [{"code": "FF", "values": ["*"]}, {"code": "Tid", "values": ["2026"]}]}
        r = requests.post("https://api.statbank.dk/v1/data", json=q, timeout=120)
        r.raise_for_status()
        (RAW / f"dst_{t.lower()}_2026.csv").write_bytes(r.content)


def realm_born():
    """BEF5G / BEF5F: national counts of Greenland- and Faroe-born residents by parents' birth
    place. Everyone with a parent born in Denmark is left on Danish; the rest are drawn on the
    territory's home mix."""
    out = {}
    for t, iso in REALM.items():
        d = pd.read_csv(RAW / f"dst_{t.lower()}_2026.csv", sep=";", dtype=str, encoding="utf-8-sig")
        d.columns = ["ff", "tid", "n"]
        d["code"] = d["ff"].str.split(" ", n=1).str[0]
        d["n"] = d["n"].astype(int)
        keep = d[~d["code"].str.startswith(("BDK", "DK"))]["n"].sum()
        print(f"  {t}: {d['n'].sum():,} born in {iso}, {keep:,} with no parent born in Denmark")
        out[iso] = int(keep)
    return out


def load():
    d = pd.read_csv(RAWF, sep=";", dtype=str, encoding="utf-8-sig")
    d.columns = ["area", "sex", "age", "herk", "land", "tid", "n"]
    for c in ("area", "herk", "land"):
        d[c + "_code"] = d[c].str.split(" ", n=1).str[0]
        d[c + "_name"] = d[c].str.split(" ", n=1).str[1].str.strip()
    d["n"] = pd.to_numeric(d["n"], errors="coerce").fillna(0).astype(int)
    return d


def iso2_map(names):
    q = pd.read_csv(ROOT / "queue.csv", usecols=["cc", "country"], keep_default_na=False)
    by = {n.lower(): c.upper() for c, n in zip(q["cc"], q["country"])}
    out, miss = {}, []
    for n in names:
        if n in NAME_FIX:
            out[n] = NAME_FIX[n]
        elif n.lower() in by:
            out[n] = by[n.lower()]
        else:
            miss.append(n)
    return out, miss


def kommuner():
    import geopandas as gpd
    g = gpd.read_file(LAU)
    g["lau"] = g["lau"].astype(str)
    if len(g) != 99 or g["lau"].nunique() != 99:
        raise SystemExit(f"{LAU}: {len(g)} LAUs, expected 99")
    return g


def main():
    from origin_mix import mix

    d = load()
    k = kommuner()
    d = d[d["area_code"].isin(set(k["lau"]))]
    if set(d["area_code"]) != set(k["lau"]):
        raise SystemExit(f"FOLK1C areas differ from the LAU polygons: "
                         f"{set(k['lau']) ^ set(d['area_code'])}")
    tot_rows = d[(d["herk"].str.startswith("TOT")) & (d["land_code"] == "0000")]
    pop = tot_rows.set_index("area_code")["n"]
    tot = int(pop.sum())
    by_herk = d[d["land_code"] == "0000"].pivot_table(index="area_code", columns="herk_code",
                                                     values="n", aggfunc="sum")
    gap = (by_herk[["3", "4", "5"]].sum(axis=1) - by_herk["TOT"]).abs().max()
    print(f"FOLK1C {TID}: {len(pop)} areas, {tot:,} people; Danish origin {int(by_herk['5'].sum()):,}, "
          f"immigrants {int(by_herk['4'].sum()):,}, descendants {int(by_herk['3'].sum()):,}; "
          f"ancestry parts vs total, worst gap {gap}")
    assert gap == 0

    f = d[d["herk_code"].isin(["3", "4"]) & (d["land_code"] != "0000") & (d["n"] > 0)]
    # countries against the 0000 row per kommune and ancestry
    sumc = f.groupby(["area_code", "herk_code"])["n"].sum()
    agg = d[d["herk_code"].isin(["3", "4"]) & (d["land_code"] == "0000")] \
        .set_index(["area_code", "herk_code"])["n"]
    cgap = (agg - sumc.reindex(agg.index).fillna(0)).abs().max()
    print(f"  {f['land_code'].nunique()} origins named; countries vs 'total' per kommune, worst gap {cgap}")
    assert cgap == 0
    to2, miss = iso2_map(sorted(f["land_name"].unique()))
    if miss:
        nat = f[f["land_name"].isin(miss)].groupby("land_name")["n"].sum().sort_values(ascending=False)
        raise SystemExit(f"origins with no ISO code -- add to NAME_FIX:\n{nat.to_string()}")

    mixes = {}
    rows = []
    for r in f.itertuples():
        iso = to2[r.land_name]
        if iso is None:
            m = {OTHER: 1.0}
        else:
            if iso not in mixes:
                try:
                    mixes[iso] = mix(iso, "dk")
                except KeyError as e:
                    print(f"  !! no mix for {iso} ({r.land_name}): {e}; on other")
                    mixes[iso] = {OTHER: 1.0}
            m = mixes[iso]
        keep = 1.0 if r.herk_code == "4" else SECOND_GEN_KEEP
        for node, s in m.items():
            rows.append((r.area_code, node, r.n * keep * s))
        if keep < 1:
            rows.append((r.area_code, DANISH, r.n * (1 - keep)))
    foreign = pd.DataFrame(rows, columns=["geo_id", "node", "count"]) \
        .groupby(["geo_id", "node"])["count"].sum()
    unk = f[f["land_name"].map(to2).isna()].groupby("land_name")["n"].sum()
    print(f"  origins on `other`: {unk.to_dict()}")

    sj = sum(pop[c] for c in SONDERJYLLAND)
    german = {c: GERMAN_TOTAL * pop[c] / sj for c in SONDERJYLLAND}
    print("  German: " + ", ".join(f"{SONDERJYLLAND[c]} {v:,.0f} ({v / pop[c]:.1%})"
                                    for c, v in german.items()))

    # Greenland- and Faroe-born: national counts, spread by Danish-origin population (no kommune
    # figure is published); carved out of Danish
    realm = realm_born()
    dan = by_herk["5"].astype(float)
    realm_rows = {}
    for iso, n in realm.items():
        for node, s in mix(iso, "dk").items():
            for reg in pop.index:
                realm_rows.setdefault(reg, {}).setdefault(node, 0.0)
                realm_rows[reg][node] += n * s * dan[reg] / dan.sum()

    names = k.set_index("lau")["name"]
    out = []
    for reg in pop.index:
        rowd = foreign.xs(reg, level="geo_id").to_dict() if reg in foreign.index.get_level_values(0) else {}
        rr = realm_rows.get(reg, {})
        for node, v in rr.items():
            rowd[node] = rowd.get(node, 0) + v
        rowd[DANISH] = rowd.get(DANISH, 0) + by_herk.at[reg, "5"] - german.get(reg, 0) \
            - sum(rr.values())
        if german.get(reg):
            rowd[GERMAN] = rowd.get(GERMAN, 0) + german[reg]
        s = pd.Series(rowd)
        if abs(s.sum() - pop[reg]) > 1e-6 * max(pop[reg], 1) + 1e-6:
            raise SystemExit(f"{reg}: {s.sum()} != {pop[reg]}")
        fl = s.apply(int)
        fl[(s - fl).sort_values(ascending=False).index[:int(pop[reg]) - int(fl.sum())]] += 1
        for node, c in fl.items():
            if c > 0:
                out.append(dict(geo_id=reg, geo_level="kommune", geo_name=names[reg],
                                source_category=node, count=int(c), tier="derived",
                                year=2026, source_id="dst_folk1c_2026k1"))
    out = pd.DataFrame(out)
    if int(out["count"].sum()) != tot:
        raise SystemExit("total moved")
    t = out.groupby("source_category")["count"].sum().sort_values(ascending=False)
    print(f"every kommune sums to FOLK1C; {tot:,} people, {len(t)} nodes")
    for node, v in t.head(20).items():
        print(f"   {v:>9,}  {v / tot:6.2%}  {node}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False, encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}: {len(out)} rows")

    if not HEX.exists():
        from _grid import hex_layer
        units = k[["lau", "geometry"]].rename(columns={"lau": "unit"})
        HEX.parent.mkdir(parents=True, exist_ok=True)
        hex_layer("dk", units, census={u: int(pop[u]) for u in pop.index}, out=HEX)


if __name__ == "__main__":
    if "--fetch" in sys.argv:
        fetch()
    main()
