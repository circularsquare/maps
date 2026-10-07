"""Belarus: the 2019 census's raions and cities as polygons, and a Kontur placement layer.

    python sources/by_geo.py --fetch    Nominatim: 118 searches and 4 lookups (raions), 1 lookup
                                        (ten cities); ~3 minutes at Nominatim's pace
    python sources/by_geo.py            -> data/geo/by/by_units.gpkg, by_lookup.csv, by_hexes.gpkg

UNITS. The census cube (sources/by_census.py) has 137 units: 118 raions, 10 cities of oblast
subordination and Minsk's 9 city districts. They are drawn as 129 polygons, all OpenStreetMap
boundary relations fetched through Nominatim:

  * 118 raions: one Nominatim search per census raion ("<adjective> район", Belarus only), the one
    admin_level-6 boundary (place_rank 12) whose oblast is the census oblast; every search had
    exactly one (asserted). OSM's raion polygons already leave the cities of oblast subordination
    and Minsk out, as the census does.
  * 10 cities of oblast subordination (Brest, Baranovichi, Pinsk, Vitebsk, Novopolotsk, Gomel,
    Grodno, Zhodino, Mogilev, Bobruisk): OSM city boundary relations, ids pinned in CITY_OSM.
    Polotsk city is not one of them in 2019 (the cube's Polotsk city row is empty; its people are
    in Polotsk raion), and OSM's Polotsk raion holds it.
  * Minsk city: OSM relation 59195, religiondots' copy (data/raw/by/osm_minsk_r59195.geojson),
    353 km2. The census's nine city districts are drawn as one unit (2.02 million people): their
    mixes hardly differ (46.7-49.8% Belarusian native, 32-37% Belarusian at home), so a district
    layer would move almost nothing.

WHY OSM AND NOT COD-AB. COD-AB Belarus admin 2 (religiondots/data/raw/by/shp) is a coarse
drawing: against OSM the same-named raion has a median IoU of 0.70 and as low as 0.31
(Beshenkovichi), its Orsha/Dubrovno line runs through Orsha city (Dubrovno raion read 2.83x its
census people in Kontur), its Kirovsk line through Bobruisk, and it draws each city inside its
raion. On COD the Kontur/census fit per unit was p10 0.71, p90 1.48, log r 0.951; on OSM it is
p10 0.83, p90 1.30, log r 0.978, with no unit outside a factor of 3.

THE JOIN AND ITS WITNESS. OSM's raion is picked by the census's own Russian name and oblast. COD
is kept as the witness neither name decides: the census raion is name-joined to COD (its Russian
adjective transliterated in COD's style, ц c, х h, я ja, й j, matched within the oblast on the
longest shared prefix, at least four letters or all but the last of a shorter COD name, unique;
Starye Dorogi, Goretsky and Ivye pinned in ALIAS; 1:1 both ways), and the COD raion holding the
most of each OSM polygon must be that same raion, for all 118 (asserted). Unit ids are COD
pcodes, `<raion pcode>-city` for a city, and `BY005` for Minsk (religiondots' id, so its
kontur_cap.csv row for Minsk applies).

CHECKS: Kontur 2023 over the census per unit, normalised nationally, and the shuffled-join control
that `_grid.hex_layer` prints; the ten cities banded on their own; no two units overlapping;
Belarus's land outside every unit printed (about 980 km2 of slivers along the border, where OSM's
and COD's national lines differ; 9,011 Kontur people).

PLACEMENT. Kontur 2023 r8 hexes keyed by centroid (`sources/_grid.py`). The extract is
religiondots' (data/raw/by/), copied into languagedots/data/geo/kontur/ once.
"""
import json
import os
import re
import shutil
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from rdlink import RD  # noqa: E402

SHP = RD / "data" / "raw" / "by" / "shp"
RD_MINSK_OSM = RD / "data" / "raw" / "by" / "osm_minsk_r59195.geojson"
RD_KONTUR_GZ = RD / "data" / "raw" / "by" / "kontur_population_BY_20231101.gpkg.gz"
OUR_KONTUR = ROOT / "data" / "geo" / "kontur"
RAW = ROOT / "data" / "raw" / "by"
OSM_CITIES = RAW / "osm_cities.geojson"
OSM_RAION_SEARCH = RAW / "osm_raions_search.json"
OSM_RAIONS = RAW / "osm_raions.geojson"
OBLAST_BE = {1: "Брэсцкая вобласць", 2: "Віцебская вобласць", 3: "Гомельская вобласць",
             4: "Гродзенская вобласць", 6: "Мінская вобласць", 7: "Магілёўская вобласць"}
GEO = ROOT / "data" / "geo" / "by"
NORM = ROOT / "data" / "normalized" / "by.csv"
AREA_CRS = 32635
UA_NOMINATIM = {"User-Agent": "languagedots/1.0 (language dot map research)"}

# SOATO (census geo_id // 10^6) -> OSM relation of the city boundary (Nominatim search, 2026-10-05:
# each the one boundary=administrative city/town result for the Russian name inside Belarus)
CITY_OSM = {1401: 72615, 1410: 3629362, 1445: 1749248, 2401: 6825777, 2418: 6825778,
            3401: 163244, 4401: 130921, 6413: 79911, 7401: 62145, 7410: 167857}
# the raion each city sits in, by name (Novopolotsk in Polotsk raion, Zhodino in Smolevichi)
CITY_HOST = {1401: "BY001003", 1410: "BY001001", 1445: "BY001013", 2401: "BY007021",
             2418: "BY007012", 3401: "BY002006", 4401: "BY003003", 6413: "BY004016",
             7401: "BY006017", 7410: "BY006002"}
CITY_EN = {1401: "Brest city", 1410: "Baranovichi city", 1445: "Pinsk city", 2401: "Vitebsk city",
           2418: "Novopolotsk city", 3401: "Gomel city", 4401: "Grodno city", 6413: "Zhodino city",
           7401: "Mogilev city", 7410: "Bobruisk city"}
OBLAST_COD = {1: "BY001", 2: "BY007", 3: "BY002", 4: "BY003", 5: "BY005", 6: "BY004", 7: "BY006"}
ALIAS = {"стародорожский": "St.Dorogi", "горецкий": "Gorki", "ивьевский": "Ive"}
MINSK_RAION = "BY004010"
MINSK_UNIT = "BY005"          # religiondots' id for the city, so its kontur_cap.csv row applies
EXPECTED_UNITS = 129
NATIONAL = 9_413_446

_TR = dict(zip("абвгдеёжзийклмнопрстуфхцчшщъыьэюя",
               ["a", "b", "v", "g", "d", "e", "e", "zh", "z", "i", "j", "k", "l", "m", "n", "o",
                "p", "r", "s", "t", "u", "f", "h", "c", "ch", "sh", "sch", "", "y", "", "e",
                "ju", "ja"]))


def translit(s):
    return "".join(_TR.get(ch, ch) for ch in s.lower())


def cod_skeleton(name):
    """COD mixes two romanisations (Gancevichi but Kletsk, Ljahovichi but Lyuban): fold both
    onto the one translit() writes."""
    s = name.lower().replace("'", "")
    for a, b in (("ts", "c"), ("kh", "h"), ("ya", "ja"), ("yu", "ju")):
        s = s.replace(a, b)
    return s


def lcp(a, b):
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return n


def _get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA_NOMINATIM), timeout=120) as r:
        return r.read()


def _lookup(rel_ids):
    """Nominatim polygons for OSM relations, 40 per request, as one FeatureCollection."""
    feats = []
    ids = list(rel_ids)
    for i in range(0, len(ids), 40):
        part = ",".join(f"R{r}" for r in ids[i:i + 40])
        doc = json.loads(_get("https://nominatim.openstreetmap.org/lookup?format=geojson"
                              f"&polygon_geojson=1&osm_ids={part}"))
        feats += doc["features"]
        time.sleep(1.5)
    return json.dumps({"type": "FeatureCollection", "features": feats}, ensure_ascii=False)


def fetch():
    RAW.mkdir(parents=True, exist_ok=True)
    if not OSM_CITIES.exists():
        OSM_CITIES.write_text(_lookup(CITY_OSM.values()), encoding="utf-8")
        print(f"  got {OSM_CITIES.name}")
    if not OSM_RAION_SEARCH.exists():
        # one Nominatim search per census raion, "<adjective> район"; every candidate kept
        found = {}
        for name in census_units().query("geo_level == 'raion'")["geo_name"]:
            q = re.sub(r"\s*р-н$", " район", name)
            url = ("https://nominatim.openstreetmap.org/search?format=jsonv2&addressdetails=1"
                   "&countrycodes=by&limit=10&q=" + urllib.parse.quote(q))
            found[name] = json.loads(_get(url))
            time.sleep(1.2)
        OSM_RAION_SEARCH.write_text(json.dumps(found, ensure_ascii=False), encoding="utf-8")
        print(f"  searched {len(found)} raions")
    if not OSM_RAIONS.exists():
        ids = sorted(set(pick_raion_relations().values()))
        OSM_RAIONS.write_text(_lookup(ids), encoding="utf-8")
        print(f"  got {OSM_RAIONS.name} ({len(ids)} relations)")


def pick_raion_relations():
    """census raion name -> the OSM relation: the one admin_level-6 boundary (place_rank 12)
    among the search results whose oblast (address.state) is the census oblast."""
    found = json.loads(OSM_RAION_SEARCH.read_text(encoding="utf-8"))
    cen = census_units().query("geo_level == 'raion'").set_index("geo_name")
    out, bad = {}, []
    for name, res in found.items():
        want = OBLAST_BE[cen.loc[name, "ob"]]
        hits = [x for x in res if x["osm_type"] == "relation" and x["category"] == "boundary"
                and x["type"] == "administrative" and x.get("place_rank") == 12
                and x.get("address", {}).get("state") == want]
        if len(hits) != 1:
            bad.append(f"{name}: {[(x['osm_id'], x['display_name'][:50]) for x in hits]}")
            continue
        out[name] = int(hits[0]["osm_id"])
    if bad:
        raise SystemExit("raion search not one boundary each:\n  " + "\n  ".join(bad))
    if len(set(out.values())) != len(out):
        raise SystemExit("two census raions picked one OSM relation")
    return out


def census_units():
    df = pd.read_csv(NORM)
    df = df[df["question"] == "native"]
    u = df.groupby(["geo_id", "geo_level", "geo_name", "oblast"], as_index=False)["count"].sum()
    if len(u) != 137 or u["count"].sum() != NATIONAL:
        raise SystemExit(f"by.csv: {len(u)} units, {u['count'].sum():,} people")
    u["soato"] = u["geo_id"] // 1_000_000
    u["ob"] = u["geo_id"] // 1_000_000_000
    return u


def main():
    import geopandas as gpd
    from shapely.ops import unary_union

    if "--fetch" in sys.argv:
        fetch()
    a2 = gpd.read_file(SHP / "blr_admin2.shp", engine="fiona")
    if len(a2) != 119 or a2.crs.to_epsg() != 4326:
        raise SystemExit(f"COD admin 2: {len(a2)} features, {a2.crs}")
    a2 = a2.set_index("adm2_pcode", drop=False)
    cen = census_units()

    # --- raion join
    rows = []
    raions = cen[cen["geo_level"] == "raion"]
    for _, r in raions.iterrows():
        stem = re.sub(r"\s*р-н$", "", r["geo_name"]).strip().lower()
        cod = a2[(a2["adm1_pcode"] == OBLAST_COD[r["ob"]]) & (a2["adm2_name"] != "Minsk City")]
        if stem in ALIAS:
            hit = cod.index[cod["adm2_name"] == ALIAS[stem]].tolist()
            score = None
        else:
            t = translit(stem)
            scores = {p: lcp(t, cod_skeleton(n)) for p, n in cod["adm2_name"].items()}
            best = max(scores.values())
            hit = [p for p, s in scores.items() if s == best]
            score = best
            # four letters, or all but the last letter of a shorter COD name (Lida, Mosty)
            if len(hit) == 1 and best < min(4, len(cod.loc[hit[0], "adm2_name"]) - 1):
                hit = []
        if len(hit) != 1:
            raise SystemExit(f"{r['geo_name']} ({r['oblast']}): COD matches {hit} (prefix {score})")
        rows.append((r["geo_id"], r["geo_name"], hit[0], a2.loc[hit[0], "adm2_name"], score))
    lut = pd.DataFrame(rows, columns=["geo_id", "geo_name", "unit", "cod_name", "prefix"])
    if lut["unit"].duplicated().any() or len(lut) != 118:
        raise SystemExit(f"raion join not 1:1: {lut[lut['unit'].duplicated(keep=False)]}")
    if set(lut["unit"]) != set(a2.index) - {"BY005001"}:
        raise SystemExit("a COD raion has no census raion")
    weak = lut[lut["prefix"].fillna(99) < 6]
    print(f"name join: 118 census raions <-> 118 COD raions, 1:1; pinned {sorted(ALIAS)}; "
          f"shortest prefixes: " + ", ".join(f"{g}->{c} ({int(p)})" for g, c, p in
                                              zip(weak["geo_name"], weak["cod_name"], weak["prefix"])))

    metric = lambda g: float(gpd.GeoSeries([g], crs=4326).to_crs(AREA_CRS).area.iloc[0]) / 1e6  # noqa: E731

    # --- raion polygons: OSM's, one relation per census raion; COD's same-named raion witnesses
    rel_of = pick_raion_relations()
    osm_r = gpd.read_file(OSM_RAIONS)
    osm_r["rel"] = osm_r["osm_id"].astype(int)
    osm_r = osm_r.set_index("rel")
    if set(osm_r.index) != set(rel_of.values()):
        raise SystemExit(f"Nominatim lookup returned {len(osm_r)} raions, not the {len(rel_of)} picked")
    geom, ious, bad = {}, {}, []
    for _, r in lut.iterrows():
        g = osm_r.loc[rel_of[r["geo_name"]], "geometry"].buffer(0)
        cod = a2.loc[r["unit"], "geometry"].buffer(0)
        if r["unit"] == MINSK_RAION:            # COD splits the city off at 87 km2; compare like with like
            cod = unary_union([cod, a2.loc["BY005001", "geometry"].buffer(0)])
        ious[r["unit"]] = metric(g.intersection(cod)) / metric(g.union(cod))
        # the COD raion holding most of the OSM polygon must be the name-joined one
        over = {p: metric(g.intersection(c.buffer(0))) for p, c in a2.geometry.items()
                if c.intersects(g)}
        top = max(over, key=over.get)
        if top != r["unit"] and not (r["unit"] == MINSK_RAION and top == "BY005001"):
            bad.append(f"{r['geo_name']}: OSM r{rel_of[r['geo_name']]} lies mostly in COD {top}, "
                       f"not {r['unit']}")
        geom[r["unit"]] = g
    if bad:
        raise SystemExit("OSM raion against COD:\n  " + "\n  ".join(bad))
    iou = pd.Series(ious).sort_values()
    print(f"  OSM raions against COD's same-named raion: IoU median {iou.median():.3f}, lowest "
          + ", ".join(f"{a2.loc[u, 'adm2_name']} {v:.3f}" for u, v in iou.head(5).items()))
    if iou.min() < 0.25:
        raise SystemExit("an OSM raion and COD's same-named raion share under 25% of their union")
    lut["osm_rel"] = lut["geo_name"].map(rel_of)

    # --- Minsk city (OSM relation 59195, religiondots' copy) and the ten cities, cut out of
    # whatever raion polygons hold them
    city_geo = {MINSK_UNIT: gpd.read_file(RD_MINSK_OSM).geometry.iloc[0].buffer(0)}
    osm_c = gpd.read_file(OSM_CITIES)
    osm_c["rel"] = osm_c["osm_id"].astype(int)
    osm_c = osm_c.set_index("rel")
    if set(osm_c.index) != set(CITY_OSM.values()):
        raise SystemExit(f"Nominatim returned {sorted(osm_c.index)}")
    city_rows = []
    for soato, rel in CITY_OSM.items():
        row = cen[cen["soato"] == soato]
        if len(row) != 1 or row["geo_level"].iloc[0] != "city":
            raise SystemExit(f"SOATO {soato}: {len(row)} census city rows")
        unit = f"{CITY_HOST[soato]}-city"
        city_geo[unit] = osm_c.loc[rel, "geometry"].buffer(0)
        city_rows.append((row["geo_id"].iloc[0], row["geo_name"].iloc[0], unit,
                          CITY_EN[soato], None, rel))
    for unit, g in city_geo.items():
        km2 = metric(g)
        inside = {p: metric(g.intersection(geom[p])) for p in lut["unit"] if geom[p].intersects(g)}
        inside = {p: round(a, 1) for p, a in inside.items() if a > 0.05}
        host = MINSK_RAION if unit == MINSK_UNIT else unit[:-5]
        stray = {p: a for p, a in inside.items() if p != host}
        if not (15 <= km2 <= 400) or sum(stray.values()) > 1.0:
            raise SystemExit(f"{unit}: {km2:.1f} km2, overlapping raions {inside}")
        for p in inside:
            geom[p] = geom[p].difference(g)
        geom[unit] = g
        print(f"  {unit:<15} {km2:6.1f} km2; inside OSM raion polygons: {inside or 'none (excluded already)'}")
    mins = cen[cen["geo_level"] == "minsk_district"]
    extra = pd.DataFrame(city_rows, columns=lut.columns)
    extra = pd.concat([extra, pd.DataFrame({"geo_id": mins["geo_id"], "geo_name": mins["geo_name"],
                                            "unit": MINSK_UNIT, "cod_name": "Minsk city",
                                            "prefix": None, "osm_rel": 59195})])
    lut = pd.concat([lut, extra.astype(lut.dtypes.to_dict())], ignore_index=True)

    # --- coverage: Belarus (COD admin 0) less every unit; a hole here is land nobody is drawn on
    a0 = gpd.read_file(SHP / "blr_admin0.shp", engine="fiona").geometry.iloc[0].buffer(0)
    allu = unary_union(list(geom.values()))
    hole = a0.difference(allu)
    pieces = [p for p in getattr(hole, "geoms", [hole]) if metric(p) > 1.0]
    print(f"  Belarus outside every unit: {metric(hole):.1f} km2, {len(pieces)} pieces over 1 km2"
          + "".join(f"\n    {metric(p):.1f} km2 at {p.representative_point().x:.3f} "
                    f"{p.representative_point().y:.3f}" for p in pieces[:8]))
    ovl = sum(metric(geom[a].intersection(geom[b])) for i, a in enumerate(geom)
              for b in list(geom)[i + 1:] if geom[a].intersects(geom[b]))
    print(f"  units overlapping each other: {ovl:.1f} km2 in all")
    if ovl > 20:
        raise SystemExit("unit polygons overlap")
    lut = lut.merge(cen[["geo_id", "count"]].rename(columns={"count": "pop"}), on="geo_id")
    if len(lut) != 137 or lut["pop"].sum() != NATIONAL or set(lut["unit"]) != set(geom):
        raise SystemExit(f"lookup: {len(lut)} rows, {lut['pop'].sum():,}, "
                         f"units {len(set(lut['unit']))} against polygons {len(geom)}")
    units = gpd.GeoDataFrame({"unit": list(geom)}, geometry=list(geom.values()), crs=4326)
    if len(units) != EXPECTED_UNITS:
        raise SystemExit(f"{len(units)} units, expected {EXPECTED_UNITS}")
    names = lut.groupby("unit")["cod_name"].first()
    units["name"] = units["unit"].map(names)

    GEO.mkdir(parents=True, exist_ok=True)
    lut.to_csv(GEO / "by_lookup.csv", index=False, encoding="utf-8")
    units.to_file(GEO / "by_units.gpkg", driver="GPKG")
    print(f"wrote {GEO / 'by_units.gpkg'} ({len(units)} units) and by_lookup.csv")

    OUR_KONTUR.mkdir(parents=True, exist_ok=True)
    gz = OUR_KONTUR / RD_KONTUR_GZ.name
    if not gz.exists() and not (OUR_KONTUR / RD_KONTUR_GZ.name[:-3]).exists():
        shutil.copyfile(RD_KONTUR_GZ, gz)
    from _grid import hex_layer
    census = lut.groupby("unit")["pop"].sum().to_dict()
    layer = hex_layer("by", units, census=census)

    # the ten new city polygons, on their own: Kontur over census, normalised nationally
    per = layer.groupby("unit")["pop"].sum()
    ratio = per.sum() / NATIONAL
    print("  cities, Kontur / census normalised: " + ", ".join(
        f"{names[u]} {per.get(u, 0) / census[u] / ratio:.2f}" for u in sorted(census)
        if u.endswith("-city")))
    # Zhodino reads 1.85: Kontur puts 121,000 people in OSM's 25 km2 against the census's 64,841,
    # its top hexes on the town's own blocks (28.29-28.36 E, 54.08-54.11 N), so the polygon is the
    # town and Kontur is dense there; it only shapes placement inside the town.
    band = {"BY004016-city": 2.0}
    bad = [u for u in census if u.endswith("-city")
           and not 0.6 <= per.get(u, 0) / census[u] / ratio <= band.get(u, 1.6)]
    if bad:
        raise SystemExit(f"cities outside 0.6-1.6 of the census: {bad}")


if __name__ == "__main__":
    main()
