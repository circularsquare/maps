"""Turkmenistan population.csv: 2022 census carried onto today's etraps, plus 2026.

Needs census.py and prep_boundaries.py to have run. Reads
  data/turkmenistan/census_settlements.csv     1,810 leaf settlements (census.py)
  data/turkmenistan/census_units.csv           48 census etraps and cities
  data/turkmenistan/boundaries/adm2.gpkg       53 current units (prep_boundaries.py)
  data/turkmenistan/raw/turkmenistan-261005.osm.pbf     OSM place nodes
  data/turkmenistan/raw/kontur_boundaries_TM_20230628.gpkg   the census-time etraps
  religiondots/data/geo/tm/tm_hexes.gpkg        Kontur 400 m population (read only)
  data/turkmenistan/raw/WPP2024_TotalPopulationBySex.csv.gz  UN WPP 2024
Writes
  data/turkmenistan/population.csv             code,level,year,pop
  data/turkmenistan/settlement_match.csv       where each census settlement went

2022: the census (17 December 2022). The census has 48 etraps and cities; OSM
now has 53, after five etraps were created from parts of old ones (Altyn asyr,
Döwletli, Farap, Garabekewül, Oguzhan) and Ashgabat's four etraps are listed
separately. Each census settlement is placed on an OSM place node of the same
name inside its census etrap (Kontur's June 2023 copy of the OSM boundaries,
which still has the census's 48 units), and counted in whichever current unit
the node falls in. Settlements with no node follow their gengeshlik's matched
villages; any still left are spread over the current units that overlap the
census etrap by Kontur population. Every census etrap total is kept.

2026: the census times UN WPP 2024's medium-variant national growth from census
day to mid-2026, the same factor for every unit. There is no post-census
official series by velayat or etrap (see README).
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "6")
os.environ.setdefault("MKL_NUM_THREADS", "6")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "6")
os.environ["OSM_USE_CUSTOM_INDEXING"] = "NO"

import difflib
import re
import sys
from collections import defaultdict
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pyogrio

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prep_boundaries import english  # noqa: E402

HELPER = Path(__file__).resolve().parents[2]
REPO = HELPER.parent
DATA = HELPER / "data" / "turkmenistan"
RAW = DATA / "raw"
HEXES = REPO / "religiondots" / "data" / "geo" / "tm" / "tm_hexes.gpkg"

CENSUS_YEAR = 2022
EST_YEAR = 2026
CENSUS_DAY = 2022 + 350 / 365      # 17 December 2022 as a fraction of the year
BUFFER_M = 5000                    # a node may sit this far outside its census etrap

VELAYAT_CODE = {"Ashgabat": "TM-S", "Ahal": "TM-A", "Balkan": "TM-B", "Dashoguz": "TM-D",
                "Lebap": "TM-L", "Mary": "TM-M"}

GENERIC = re.compile(
    r"\b(obasy|oba|shaherchesi|shaheri|gengeshligi|gengeshlik|geneshlik|etraby|etrap|town|city|"
    r"station|village|settlement|named after|of the|duralgasy|posyolok|poselok|imeni|"
    r"razyezd|aul)\b")


def key(s):
    """Match key: Turkmen and English spellings of one name come out the same."""
    if not s:
        return ""
    s = s.lower().replace("ňň", "ň")  # OSM often doubles it (Toraňňyly = census Torangyly)
    s = GENERIC.sub(" ", s)          # English words first: english() turns town into tovn
    s = english(s)
    s = s.replace("’", "").replace("'", "").replace("`", "")
    s = GENERIC.sub(" ", s)
    s = re.sub(r"[^a-z0-9]", "", s)
    return s.replace("ng", "n")      # Turkmen ň is n after english(); the census writes ng


def tag(other, k):
    m = re.search(rf'"{re.escape(k)}"=>"([^"]*)"', other or "")
    return m.group(1) if m else None


def osm_places():
    p = pyogrio.read_dataframe(RAW / "turkmenistan-261005.osm.pbf", layer="points",
                               where="place IS NOT NULL")
    p = p[p.place.isin(["city", "town", "village", "hamlet", "isolated_dwelling",
                        "suburb", "neighbourhood", "locality", "quarter", "farm"])]
    rows = []
    for r in p.itertuples():
        names = {r.name}
        for k in ("name:tk", "name:en", "alt_name", "alt_name:tk", "old_name", "official_name",
                  "name:tk-Latn", "loc_name", "short_name"):
            v = tag(r.other_tags, k)
            if v:
                names.update(v.split(";"))
        pop = tag(r.other_tags, "population")
        try:
            pop = int(pop) if pop else None
        except ValueError:
            pop = None
        keys = {key(n) for n in names if n} - {""}
        if keys:
            rows.append({"osm": int(r.osm_id), "name": r.name, "place": r.place, "keys": keys,
                         "pop": pop, "gengesh": key(tag(r.other_tags, "addr:city")),
                         "geometry": r.geometry})
    return gpd.GeoDataFrame(rows, crs=4326)


def census_polygons():
    """Kontur's OSM boundaries of 2023-06-28: the census's 48 etraps and cities."""
    k = gpd.read_file(RAW / "kontur_boundaries_TM_20230628.gpkg")
    k = k[k.admin_level == 6].copy()
    k["unit"] = k.name.map(english)
    ash = gpd.read_file(RAW / "kontur_boundaries_TM_20230628.gpkg")
    ash = ash[(ash.admin_level == 4) & (ash.name == "Aşgabat")]
    return k[["unit", "geometry"]], ash.geometry.iloc[0]


def wpp_factor():
    d = pd.read_csv(RAW / "WPP2024_TotalPopulationBySex.csv.gz", low_memory=False,
                    usecols=["ISO3_code", "Variant", "Time", "PopTotal"])
    d = d[(d.ISO3_code == "TKM") & (d.Variant == "Medium")].set_index("Time").PopTotal * 1000
    # WPP is 1 July of each year; interpolate to census day
    at_census = d[2022] + (d[2023] - d[2022]) * (CENSUS_DAY - 2022.5)
    return d[EST_YEAR] / at_census, at_census, d[EST_YEAR]


def main():
    sett = pd.read_csv(DATA / "census_settlements.csv")
    units = pd.read_csv(DATA / "census_units.csv")
    adm2 = gpd.read_file(DATA / "boundaries" / "adm2.gpkg")
    cpoly, ash_poly = census_polygons()
    cmap = dict(zip(cpoly.unit, cpoly.geometry))
    missing = set(units[units.velayat != "Ashgabat"].unit) - set(cmap)
    if missing:
        sys.exit(f"census units with no Kontur 2023 polygon: {missing}")

    places = osm_places()
    places = gpd.sjoin(places, adm2[["code", "geometry"]], predicate="within", how="left")
    places = places.drop(columns="index_right")
    places_m = places.to_crs(3857)

    out = []            # rows: census unit, settlement, pop, current code, how
    gmap = {}           # (census unit, gengeshlik key) -> current codes of OSM places that name it
    for (vel, unit), grp in sett.groupby(["velayat", "unit"], sort=False):
        if vel == "Ashgabat":
            code = adm2.loc[adm2.name == unit, "code"]
            assert len(code) == 1, unit
            for s in grp.itertuples():
                out.append({**s._asdict(), "code": code.iloc[0], "how": "unit", "osm": None})
            continue
        if unit == "Arkadag city":
            for s in grp.itertuples():
                out.append({**s._asdict(), "code": "TM-AR", "how": "unit", "osm": None})
            continue
        area = gpd.GeoSeries([cmap[unit]], crs=4326).to_crs(3857).buffer(BUFFER_M).iloc[0]
        cand = places_m[places_m.within(area)]
        # people stay in their census velayat: the velayat totals are census rows,
        # and Ashgabat and Arkadag are their own units
        cand = cand[cand.code.fillna("").str.startswith(VELAYAT_CODE[vel] + "-")]
        bykey = defaultdict(list)
        for c in cand.itertuples():
            for kk in c.keys:
                bykey[kk].append(c)
        for c in cand.itertuples():
            if pd.isna(c.code):
                continue
            if c.gengesh:                       # the village's addr:city is its gengeshlik
                gmap.setdefault((unit, c.gengesh), []).append(c.code)
            for kk in c.keys:                   # a gengeshlik is named after its main village
                gmap.setdefault((unit, kk), []).append(c.code)
        used = set()
        for s in grp.itertuples():
            k = key(s.name)
            hits = bykey.get(k, [])
            how = "name"
            if not hits and len(k) >= 5:
                close = difflib.get_close_matches(k, list(bykey), n=3, cutoff=0.85)
                hits = [c for kk in close for c in bykey[kk]]
                how = "fuzzy"
            # several nodes: prefer the census population, then the gengeshlik, then a node not used yet
            if len(hits) > 1:
                same_pop = [c for c in hits if c.pop == s.pop]
                if len(same_pop) >= 1:
                    hits = same_pop
            if len(hits) > 1 and isinstance(s.parent, str) and s.parent:
                pk = key(s.parent)
                same_g = [c for c in hits if c.gengesh and c.gengesh == pk]
                if same_g:
                    hits = same_g
            if len(hits) > 1:
                fresh = [c for c in hits if c.osm not in used]
                if fresh:
                    hits = fresh
            codes = {c.code for c in hits}
            if hits and len(codes) == 1 and pd.notna(next(iter(codes))):
                used.add(hits[0].osm)
                out.append({**s._asdict(), "code": next(iter(codes)), "how": how,
                            "osm": hits[0].osm})
            else:
                out.append({**s._asdict(), "code": None,
                            "how": "ambiguous" if hits else "none", "osm": None})

    m = pd.DataFrame(out).drop(columns="Index")
    # second pass: unmatched settlements follow their gengeshlik's matched villages
    for (unit, parent), grp in m.groupby(["unit", m.parent.fillna("")]):
        if not parent:
            continue
        got = grp[grp.code.notna()]
        if got.empty:
            continue
        best = got.groupby("code")["pop"].sum().idxmax()
        idx = grp.index[grp.code.isna()]
        m.loc[idx, "code"] = best
        m.loc[idx, "how"] = "gengeshlik"

    # second pass, b: a gengeshlik with no matched village goes where OSM's villages
    # tagged with that gengeshlik (addr:city), or its namesake village, lie
    for (unit, parent), grp in m[m.code.isna()].groupby(["unit", m.parent.fillna("")]):
        if not parent:
            continue
        codes = pd.Series(gmap.get((unit, key(parent)), []), dtype=object)
        if codes.empty:
            continue
        top = codes.value_counts()
        if top.iloc[0] / top.sum() >= 0.8:
            m.loc[grp.index, "code"] = top.index[0]
            m.loc[grp.index, "how"] = "gengeshlik-osm"

    # third pass: spread what is left over the current units by Kontur population
    hexes = gpd.read_file(HEXES)
    hexes["geometry"] = hexes.geometry.to_crs(3857).centroid.to_crs(4326)
    hexes = gpd.sjoin(hexes[["pop", "geometry"]], adm2[["code", "geometry"]],
                      predicate="within", how="inner").drop(columns="index_right")
    spread = []
    for unit, grp in m[m.code.isna()].groupby("unit"):
        poly = cmap[unit]
        h = hexes[hexes.within(poly)]
        share = h.groupby("code")["pop"].sum()
        # only onto units that took named villages from this census etrap, so the
        # boundary slivers between the 2023 and current outlines get nothing
        took = set(m[(m.unit == unit) & m.code.notna()].code)
        if took:
            share = share[share.index.isin(took)]
        else:
            share = share[share == share.max()]
        share = share / share.sum()
        left = int(grp["pop"].sum())
        for code, sh in share.items():
            spread.append({"velayat": grp.velayat.iloc[0], "unit": unit, "parent": "",
                           "name": f"(unmatched, {len(grp)} settlements)", "kind": "",
                           "pop": left * sh, "code": code, "how": "kontur", "osm": None})
    m_matched = m[m.code.notna()]
    allrows = pd.concat([m_matched.astype({"osm": object}), pd.DataFrame(spread).astype({"osm": object})], ignore_index=True)
    m.to_csv(DATA / "settlement_match.csv", index=False)

    # sanity: every census unit total kept
    tot = allrows.groupby("unit")["pop"].sum().round().astype(int)
    for u in units.itertuples():
        assert abs(tot[u.unit] - u.total) <= 1, (u.unit, tot[u.unit], u.total)

    # report
    print("how settlements were placed (people):")
    print(allrows.groupby("how")["pop"].sum().map("{:,.0f}".format).to_string())
    print(f"settlements: {len(m)}, by method: {m.how.value_counts().to_dict()}")
    flow = allrows.groupby(["unit", "code"])["pop"].sum().round().astype(int).reset_index()
    print("\ncensus unit -> current unit (only splits shown):")
    for unit, g in flow.groupby("unit"):
        if len(g) > 1:
            print(f"  {unit}: " + ", ".join(f"{r.code} {r.pop:,}" for r in g.itertuples()))

    # integer counts per current unit, largest remainder, national total exact
    raw = allrows.groupby("code")["pop"].sum()
    base = raw.apply(int)
    short = int(round(raw.sum())) - int(base.sum())
    for c in (raw - base).sort_values(ascending=False).index[:short]:
        base[c] += 1
    c2022 = base.reindex(adm2.code).fillna(0).astype(int)
    assert c2022.sum() == 7057841, c2022.sum()

    f, at_c, at_e = wpp_factor()
    print(f"\nWPP 2024 medium: {at_c:,.0f} at census day, {at_e:,.0f} mid-{EST_YEAR}, factor {f:.4f}")
    c2026 = (c2022 * f).round().astype(int)

    rows = []
    for code in adm2.code:
        rows.append((code, 2, CENSUS_YEAR, int(c2022[code])))
        rows.append((code, 2, EST_YEAR, int(c2026[code])))
    reg = adm2.set_index("code").group
    for g in sorted(reg.unique()):
        cs = reg[reg == g].index
        rows.append((g, 1, CENSUS_YEAR, int(c2022[cs].sum())))
        rows.append((g, 1, EST_YEAR, int(c2026[cs].sum())))
    pd.DataFrame(rows, columns=["code", "level", "year", "pop"]).to_csv(
        DATA / "population.csv", index=False)

    names = adm2.set_index("code").name
    print(f"\n{'code':8s} {'unit':32s} {CENSUS_YEAR:>10} {EST_YEAR:>10}")
    for code in adm2.code:
        print(f"{code:8s} {names[code]:32s} {c2022[code]:>10,} {c2026[code]:>10,}")
    print(f"wrote {DATA / 'population.csv'}")


if __name__ == "__main__":
    main()
