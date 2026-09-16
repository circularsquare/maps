"""Faroe Islands: the census's seven districts, as Kontur hexes labelled by district and weighted
by the register's village populations in the census month.

Writes
    data/geo/fo/fo_hexes.gpkg     Kontur 400 m hexagons (2023-11-01): `unit` (the district as MT325
                                  prints it), `village`, `pop` (the village's November 2011 register
                                  population shared over its hexes in Kontur's proportions)

No polygon layer for the districts exists: geoBoundaries has only the Faroes' outline. So the
districts are built from GADM 4.1's 30 municipalities, by a rule the office's own tables check.

## WHAT A DISTRICT IS

MT325's district codes (`4100` Norðoyar ... `4700` S-streymoy) are Hagstova's register regions
(IB01035), and a region is a list of villages (IB01031). VILLAGES below sums to IB01035's seven
regions to the person in November 2011 and in November 2023, so it is the office's definition
(witness 1). It is island groups, with Streymoy cut in two: N-streymoy is Kvívík and Vestmanna
municipalities, the Streymoy side of Sunda municipality, and Kollafjørður, Signabøur and
Oyrareingir; S-streymoy is the rest of Tórshavn municipality with Nólsoy, Hestur and Koltur.

## THE RULE FOR A HEX

  1. Its centroid's GADM municipality (the nearest within 3 km for a centroid over the sea).
  2. The municipality's district, except for the two municipalities the districts cut:
  3. Sunda, by island: land is cut out of OSM's sea polygons (`water.WATER`) around Sunda, the
     two land pieces the named Sunda villages (`SUNDA`) stand on are Streymoy and Eysturoy, and
     a hex goes to the nearer. GADM's own Sunda polygon was tried first and is one part across
     the sound, holding villages from both sides.
  4. Tórshavn: a hex goes to the district of its nearest named Tórshavn village (`TORSHAVN`).

## THE WEIGHT: KONTUR, CALIBRATED TO VILLAGES

Kontur alone was tried first and moves Tórshavn's people into the countryside: within 2.5 km of
each named village it puts 7,289 people at Tórshavn against the register's 13,999, and 753 at
Hvítanes against 106. N-streymoy came out 1.158x the register over the national ratio, outside
the 0.85-1.15 band set beforehand, and moving Sunda's Streymoy side to Eysturoy only dropped it
to 0.853, so the band could not have caught that error. So the placement is the office's count:
each hex goes to the nearest village of its own district, and each village's register population
in November 2011 (IB01031) is shared over its hexes in Kontur's proportions. A village with
people and no hex nearer to it than to another village gets a 200 m disc at its point. This is
`sources/ir_geo.py`'s calibration one level finer (geography playbook, Iran).

Village points are GeoNames' (`FO.zip`), joined on the name, with GEONAMES_ID for the names that
differ or repeat. GeoNames' own admin codes put Oyri, Saltnes and Kolbeinagjógv on the wrong
island, so only its coordinates are used.

## THE WITNESSES

  1. VILLAGES against IB01035, both months, exactly; and IB01035's November 2011 regions against
     the census's own district populations (MT1), within 3%.
  2. Every village with a point lands in its VILLAGES district by the rule: 115 positions, none
     of them used to make the rule except the 25 in SUNDA and TORSHAVN.
  3. The calibrated weight sums to each district's November 2011 register population, less the
     few people in villages GeoNames has no point for (NO_POINT, asserted small).

Usage:
    python sources/fo_geo.py --fetch    Kontur FO, GADM FRO level 2, GeoNames FO, IB01031, IB01035
    python sources/fo_geo.py            build data/geo/fo/fo_hexes.gpkg
"""

import csv
import gzip
import io
import os
import shutil
import sys
import zipfile

os.environ.setdefault("OMP_NUM_THREADS", "6")          # [[feedback_cap_cpu]]
os.environ.setdefault("OPENBLAS_NUM_THREADS", "6")
os.environ.setdefault("MKL_NUM_THREADS", "6")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import fo                                                        # noqa: E402  the census tables
from geo_checks import read_layer                                # noqa: E402  shared, not copied

sys.path.insert(0, ROOT)
import water                                                     # noqa: E402  WATER, OSM's sea

RAW = fo.RAW
GEO = os.path.join(ROOT, "data", "geo", "fo")
HEXES = os.path.join(GEO, "fo_hexes.gpkg")

KONTUR_URL = ("https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
              "kontur_population_FO_20231101.gpkg.gz")
KONTUR_GZ = os.path.join(RAW, "kontur_population_FO_20231101.gpkg.gz")
KONTUR = KONTUR_GZ[:-3]
GADM_URL = "https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_FRO_2.json"
GADM = os.path.join(RAW, "gadm41_FRO_2.json")
GEONAMES_URL = "https://download.geonames.org/export/dump/FO.zip"
GEONAMES = os.path.join(RAW, "FO.zip")
IB_VILLAGE = ("IB/IB01/fo_abgd_md_t.px",
              [{"code": "village/city", "selection": {"filter": "all", "values": ["*"]}},
               {"code": "month", "selection": {"filter": "item", "values": ["2011M11", "2023M11"]}}])
IB_REGION = ("IB/IB01/fo_tr_md_t.px",
             [{"code": "value mode", "selection": {"filter": "item", "values": ["OBS"]}},
              {"code": "region", "selection": {"filter": "all", "values": ["*"]}},
              {"code": "month", "selection": {"filter": "item", "values": ["2011M11", "2023M11"]}}])
MONTHS = ["2011M11", "2023M11"]

UTM = "EPSG:32629"
NEAREST_MUNI_M = 3000
CENSUS_REGISTER_BAND = (0.97, 1.03)
DISC_M = 200
NO_POINT_MAX = 10                   # people in November 2011, per village without a point

REGION_LABEL = {"Norðoyggjar region": "Norðoyar", "Eysturoy region": "Eysturoy",
                "Streymoy region, north": "N-streymoy", "Streymoy region, south": "S-streymoy",
                "Vágar region": "Vágar", "Sandoy region": "Sandoy", "Suðuroy region": "Suðuroy"}

# IB01031's villages by district. Witness 1 holds this to IB01035 to the person.
VILLAGES = {
    "Norðoyar": ["Ánirnar", "Árnafjørður", "Depil", "Haraldssund", "Hattarvík", "Húsar",
                 "Hvannasund", "Kirkja", "Klaksvík", "Kunoy", "Mikladalur", "Múli", "Norðdepil",
                 "Norðoyri", "Norðtoftir", "Svínoy", "Syðradalur, Kalsoy", "Trøllanes",
                 "Viðareiði"],
    "Eysturoy": ["Eiði", "Ljósá", "Oyri", "Oyrarbakki", "Norðskáli", "Svínáir", "Gjógv",
                 "Funningur", "Funningsfjørður", "Elduvík", "Oyndarfjørður", "Hellurnar",
                 "Fuglafjørður", "Kolbeinagjógv", "Undir Gøtueiði", "Leirvík", "Norðragøta",
                 "Syðrugøta", "Gøtugjógv", "Selatrað", "Strendur", "Innan Glyvur", "Skála",
                 "Skálafjørður, Runavík municipality", "Skálafjørður, Eystur municipality",
                 "Morskranes", "Glyvrar", "Saltangará", "Runavík", "Lamba", "Lambareiði",
                 "Rituvík", "Æðuvík", "Toftir", "Nes, Eysturoy", "Saltnes", "Søldarfjørður",
                 "Skipanes"],
    "N-streymoy": ["Kvívík", "Leynar", "Skælingur", "Stykkið", "Vestmanna", "Válur", "Hósvík",
                   "Hvalvík", "Streymnes", "Saksun", "Nesvík", "Langasandur", "Haldórsvík",
                   "Tjørnuvík", "Kollafjørður", "Oyrareingir", "Signabøur"],
    "S-streymoy": ["Tórshavn", "Argir", "Hoyvík", "Kirkjubøur", "Velbastaður",
                   "Syðradalur, Streymoy", "Norðradalur", "Kaldbak", "Kaldbaksbotnur", "Sund",
                   "Hvítanes", "Nólsoy", "Hestur", "Koltur", "Mjørkadalur"],
    "Vágar": ["Bøur", "Gásadalur", "Mykines", "Miðvágur", "Sandavágur", "Sørvágur", "Vatnsoyrar"],
    "Sandoy": ["Dalur", "Húsavík", "Sandur", "Skarvanes", "Skopun", "Skálavík", "Skúgvoy",
               "Stóra Dímun"],
    "Suðuroy": ["Akrar", "Fámjin", "Froðba", "Hov", "Hvalba", "Lopra", "Porkeri", "Sandvík",
                "Sumba", "Trongisvágur", "Tvøroyri", "Vágur", "Víkarbyrgi", "Ørðavík",
                "Ørðavíkslíð"],
}

# Hagstova's name -> GeoNames id where GeoNames spells it differently or has two places of the
# name. Every other village must match exactly one GeoNames populated place by name.
GEONAMES_ID = {
    "Ánirnar": 2624689,             # Ánir
    "Haldórsvík": 2620913,          # Haldarsvík
    "Hellurnar": 2620482,           # Hellur
    "Lamba": 2618005,               # Lambi
    "Skála": 2613915,               # Skáli (not Ytri Skáli)
    "Skúgvoy": 2613477,             # Skúvoy
    "Ørðavík": 2615720,             # Øravík
    "Ørðavíkslíð": 2617758,         # Líðin, beside Øravík
    "Undir Gøtueiði": 2621296,      # Gøtueiði
    "Nes, Eysturoy": 2616498,       # three places named Nes
    "Syðradalur, Kalsoy": 2611964,
    "Syðradalur, Streymoy": 2611965,
    "Skálafjørður, Runavík municipality": 2613929,
    "Hvítanes": 2619504,            # not the Suðuroy one
    "Langasandur": 2617970,         # not the one GeoNames files on Eysturoy
    "Vágur": 2610806,               # Suðuroy, not the Klaksvík neighbourhood
}
# No GeoNames point; asserted at or under NO_POINT_MAX people in November 2011.
NO_POINT = {"Kaldbaksbotnur", "Stóra Dímun", "Mjørkadalur", "Skálafjørður, Eystur municipality"}

# GADM (NAME_1, NAME_2) -> district; None for the two municipalities the districts cut.
MUNICIPALITY = {}
for _m in ("Fugloy", "Hvannasund", "Húsar", "Klaksvík", "Kunoy", "Viðareiði"):
    MUNICIPALITY[("Norderøerne", _m)] = "Norðoyar"
for _m in ("Eiði", "Eystur", "Fuglafjørður", "Nes", "Runavík", "Sjógv"):
    MUNICIPALITY[("Eysturoyar", _m)] = "Eysturoy"
MUNICIPALITY[("Eysturoyar", "Sunda")] = None
MUNICIPALITY[("Streymoyar", "Kvívík")] = "N-streymoy"
MUNICIPALITY[("Streymoyar", "Vestmanna")] = "N-streymoy"
MUNICIPALITY[("Streymoyar", "Tórshavn")] = None
for _m in ("Húsavík", "Sand", "Skopun", "Skálavík", "Skúvoy"):
    MUNICIPALITY[("Sandoyar", _m)] = "Sandoy"
for _m in ("Fámjin", "Hov", "Hvalba", "Porkeri", "Sumba", "Tvøroyri", "Vágur"):
    MUNICIPALITY[("Suðuroyar", _m)] = "Suðuroy"
for _m in ("Sørvágs", "Vágur"):
    MUNICIPALITY[("Vågø", _m)] = "Vágar"

# The villages the two cuts are made by: Sunda's on each side of the sound, and every Tórshavn
# municipality village GeoNames has. Their GeoNames coordinates are copied here so the rule does
# not depend on the name join it is witnessed by.
SUNDA = {
    "Hósvík": (62.15187, -6.94527, "N-streymoy"),
    "Hvalvík": (62.18778, -7.03278, "N-streymoy"),
    "Streymnes": (62.19153, -7.03150, "N-streymoy"),
    "Saksun": (62.24821, -7.17241, "N-streymoy"),
    "Haldórsvík": (62.27555, -7.09331, "N-streymoy"),
    "Tjørnuvík": (62.28896, -7.14555, "N-streymoy"),
    "Oyri": (62.19032, -6.97760, "Eysturoy"),
    "Oyrarbakki": (62.20995, -7.00001, "Eysturoy"),
    "Norðskáli": (62.22187, -7.00513, "Eysturoy"),
    "Svínáir": (62.23164, -7.02842, "Eysturoy"),
}
TORSHAVN = {
    "Kollafjørður": (62.11167, -6.91067, "N-streymoy"),
    "Signabøur": (62.09619, -6.92727, "N-streymoy"),
    "Oyrareingir": (62.09921, -6.94553, "N-streymoy"),
    "Kaldbak": (62.06057, -6.82808, "S-streymoy"),
    "Hvítanes": (62.04547, -6.76297, "S-streymoy"),
    "Hoyvík": (62.03407, -6.78127, "S-streymoy"),
    "Tórshavn": (62.00973, -6.77164, "S-streymoy"),
    "Argir": (61.99595, -6.77154, "S-streymoy"),
    "Velbastaður": (61.98084, -6.85259, "S-streymoy"),
    "Kirkjubøur": (61.95306, -6.79078, "S-streymoy"),
    "Norðradalur": (62.03889, -6.92554, "S-streymoy"),
    "Syðradalur, Streymoy": (62.01882, -6.90860, "S-streymoy"),
    "Nólsoy": (62.00937, -6.67019, "S-streymoy"),
    "Hestur": (61.95581, -6.88260, "S-streymoy"),
    "Koltur": (61.98647, -6.96242, "S-streymoy"),
}


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    for url, path, magic in ((KONTUR_URL, KONTUR_GZ, b"\x1f\x8b"), (GADM_URL, GADM, b"{"),
                             (GEONAMES_URL, GEONAMES, b"PK\x03\x04")):
        if os.path.exists(path) and os.path.getsize(path) > 1000:
            print("already have", path)
            continue
        print("GET", url)
        r = requests.get(url, timeout=300, headers={"User-Agent": fo.UA})
        r.raise_for_status()
        if not r.content.startswith(magic):
            raise SystemExit(f"{url}: does not start with {magic!r}: {r.content[:80]!r}")
        fo._write(path, r.content)
        print(f"  {len(r.content):,} bytes")
    for name, (path, query) in (("IB01031", IB_VILLAGE), ("IB01035", IB_REGION)):
        print("POST", fo.API + path)
        body = fo.post_csv(path, query, name)
        fo._write(os.path.join(RAW, name + ".csv"), body)
        print(f"  {len(body):,} bytes")


def unpack():
    if os.path.exists(KONTUR) and os.path.getsize(KONTUR) > 0:
        return
    if not os.path.exists(KONTUR_GZ):
        raise SystemExit(f"missing {KONTUR_GZ}; run with --fetch")
    with gzip.open(KONTUR_GZ, "rb") as src, open(KONTUR + ".part", "wb") as dst:
        shutil.copyfileobj(src, dst)
    os.replace(KONTUR + ".part", KONTUR)
    with open(KONTUR, "rb") as fh:
        if fh.read(4) != b"SQLi":
            raise SystemExit(f"{KONTUR} is not a GeoPackage")


def register_count(tok, where):
    return 0 if tok == "-" else fo.cell(tok, where)


def read_villages():
    head, body = fo.read_rows("IB01031")
    if head != ["village/city"] + MONTHS:
        raise SystemExit(f"IB01031: unexpected columns {head}")
    return {r[0]: [register_count(v, f"IB01031 {r[0]}") for v in r[1:]] for r in body}


def read_regions():
    head, body = fo.read_rows("IB01035")
    if len(body) != 1 or body[0][0] != "Observed":
        raise SystemExit(f"IB01035: expected one Observed row, got {[r[0] for r in body]}")
    out = {}
    for lab, v in zip(head[1:], body[0][1:]):
        region, month = lab.rsplit(" ", 1)
        key = "Total" if region == "Total Faroe Islands" else REGION_LABEL[region]
        out.setdefault(key, {})[month] = fo.cell(v, f"IB01035 {lab}")
    return out


def geonames_points():
    """{Hagstova village: (lat, lon)} for every village not in NO_POINT; stops on a name that
    matches no GeoNames populated place or more than one without a GEONAMES_ID."""
    with zipfile.ZipFile(GEONAMES) as z:
        rows = list(csv.reader(io.TextIOWrapper(z.open("FO.txt"), encoding="utf-8"),
                               delimiter="\t", quoting=csv.QUOTE_NONE))
    by_id = {int(r[0]): r for r in rows}
    by_name = {}
    for r in rows:
        if r[6] == "P":
            by_name.setdefault(fo.despace(r[1]), []).append(r)
    out, bad = {}, []
    for d, vs in VILLAGES.items():
        for v in vs:
            if v in NO_POINT:
                continue
            if v in GEONAMES_ID:
                r = by_id.get(GEONAMES_ID[v])
                if r is None or r[6] != "P":
                    bad.append((v, "id is not a populated place"))
                    continue
            else:
                hits = by_name.get(v, [])
                if len(hits) != 1:
                    bad.append((v, f"{len(hits)} GeoNames places"))
                    continue
                r = hits[0]
            out[v] = (float(r[4]), float(r[5]))
    if bad:
        raise SystemExit(f"GeoNames join: {bad}")
    return out


def points(table, crs):
    """GeoDataFrame from {name: (lat, lon, district)}."""
    import geopandas as gpd
    from shapely.geometry import Point

    return gpd.GeoDataFrame(
        {"name": list(table), "district": [v[2] for v in table.values()]},
        geometry=[Point(v[1], v[0]) for v in table.values()], crs="EPSG:4326").to_crs(crs)


def first(frame):
    return frame[~frame.index.duplicated(keep="first")]


def assign(cent, munis, sunda_land, torshavn_pts):
    """District for each point in `cent` (a GeoDataFrame in UTM), by the module's rule."""
    import geopandas as gpd

    j = first(gpd.sjoin_nearest(cent[["geometry"]], munis[["key", "geometry"]], how="left",
                                max_distance=NEAREST_MUNI_M))
    if j["key"].isna().any():
        raise SystemExit(f"{int(j['key'].isna().sum())} points more than {NEAREST_MUNI_M} m from "
                         "any municipality")
    district = j["key"].map(MUNICIPALITY)
    for key, target in ((("Eysturoyar", "Sunda"), sunda_land),
                        (("Streymoyar", "Tórshavn"), torshavn_pts)):
        idx = j.index[j["key"] == key]
        if len(idx):
            near = first(gpd.sjoin_nearest(cent.loc[idx, ["geometry"]],
                                           target[["district", "geometry"]], how="left"))
            district.loc[idx] = near["district"]
    return district


def main():
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import box
    from shapely.ops import unary_union

    if "--fetch" in sys.argv:
        fetch()
    unpack()

    ok = True

    def say(good, msg):
        nonlocal ok
        ok &= bool(good)
        print(f"  {'OK ' if good else 'BAD'} {msg}")

    print("Faroe Islands: districts and placement\n")

    # ---- witness 1: what a district is
    villages = read_villages()
    regions = read_regions()
    listed = [v for vs in VILLAGES.values() for v in vs]
    names = set(villages) - {"Total Faroe Islands"}
    say(len(listed) == len(set(listed)) and set(listed) == names,
        f"VILLAGES names each of IB01031's {len(names)} villages once"
        + ("" if set(listed) == names else f"; not listed {sorted(names - set(listed))}, "
                                          f"not in IB01031 {sorted(set(listed) - names)}"))
    for i, month in enumerate(MONTHS):
        sums = {d: sum(villages[v][i] for v in vs) for d, vs in VILLAGES.items()}
        bad = [(d, sums[d], regions[d][month]) for d in VILLAGES if sums[d] != regions[d][month]]
        say(not bad and sum(sums.values()) == regions["Total"][month]
            == villages["Total Faroe Islands"][i],
            f"{month}: the villages sum to IB01035's seven regions to the person "
            f"(total {regions['Total'][month]:,})" + (f"; differ {bad}" if bad else ""))
    _ages, pop = fo.read_population()
    census = {d: pop[d]["SUM (age)"] for d in fo.DISTRICTS}
    ratio = {d: regions[d]["2011M11"] / census[d] for d in fo.DISTRICTS}
    say(all(CENSUS_REGISTER_BAND[0] <= r <= CENSUS_REGISTER_BAND[1] for r in ratio.values()),
        "IB01035's November 2011 regions / the census's district populations (MT1): "
        + ", ".join(f"{d} {r:.3f}" for d, r in ratio.items()) + f", inside {CENSUS_REGISTER_BAND}")

    # ---- municipalities, and the two cuts
    g = read_layer(GADM, "GADM FRO level 2").to_crs(UTM)
    g["key"] = list(zip(g["NAME_1"], g["NAME_2"]))
    say(len(g) == 30 and set(g["key"]) == set(MUNICIPALITY),
        f"GADM has the {len(MUNICIPALITY)} municipalities MUNICIPALITY names ({len(g)} features)")
    if not ok:
        raise SystemExit("district definitions FAILED")

    if not os.path.exists(water.WATER):
        raise SystemExit(f"missing {water.WATER}, the OSM water polygons water.py reads")
    w0, s0, e0, n0 = g[g["key"] == ("Eysturoyar", "Sunda")].to_crs("EPSG:4326").total_bounds
    frame = (w0 - 0.05, s0 - 0.03, e0 + 0.05, n0 + 0.03)
    sea = gpd.read_file(water.WATER, bbox=frame)
    land = box(*frame).difference(unary_union(list(sea.geometry)))
    pieces = (gpd.GeoDataFrame(geometry=[land], crs="EPSG:4326").explode(index_parts=False)
              .to_crs(UTM).reset_index(drop=True))
    pieces["piece"] = range(len(pieces))
    lab = first(gpd.sjoin_nearest(points(SUNDA, UTM), pieces[["piece", "geometry"]], how="left",
                                  distance_col="m"))
    per_piece = lab.groupby("piece")["district"].agg(lambda s: sorted(set(s)))
    say(len(per_piece) == 2 and all(len(v) == 1 for v in per_piece)
        and {v[0] for v in per_piece} == {"N-streymoy", "Eysturoy"} and lab["m"].max() < 1000,
        f"OSM's land around Sunda is {len(pieces)} pieces; the named Sunda villages stand on "
        f"{len(per_piece)} of them, one per side of the sound (village to land: max "
        f"{lab['m'].max():.0f} m)")
    pieces["district"] = pieces["piece"].map(per_piece.map(lambda v: v[0]))
    sunda_land = pieces[pieces["district"].notna()].reset_index(drop=True)
    torshavn_pts = points(TORSHAVN, UTM)
    if not ok:
        raise SystemExit("cuts FAILED")

    # ---- witness 2: every village lands in its district by the rule
    where = geonames_points()
    district_of = {v: d for d, vs in VILLAGES.items() for v in vs}
    vil = points({v: (lat, lon, district_of[v]) for v, (lat, lon) in where.items()}, UTM)
    vil["register"] = vil["name"].map(lambda v: villages[v][0])
    got = assign(vil, g, sunda_land, torshavn_pts)
    wrong = vil[got.values != vil["district"].values]
    say(len(wrong) == 0,
        f"all {len(vil)} villages with a GeoNames point land in their district by the rule"
        + (f"; wrong (village, rule, VILLAGES): "
           f"{list(zip(wrong['name'], got[wrong.index], wrong['district']))}" if len(wrong) else ""))
    nopt = {v: villages[v][0] for v in NO_POINT}
    say(all(n <= NO_POINT_MAX for n in nopt.values()),
        f"the {len(NO_POINT)} villages GeoNames has no point for hold {sum(nopt.values())} people "
        f"in November 2011: {nopt}")
    if not ok:
        raise SystemExit("village checks FAILED")

    # ---- Kontur, labelled by the rule
    hexes = read_layer(KONTUR, "Kontur FO")
    popcol = next((c for c in hexes.columns if c.lower() == "population"), None)
    if popcol is None:
        raise SystemExit(f"no population column in {list(hexes.columns)}")
    hexes = hexes[hexes[popcol] > 0].to_crs(UTM).reset_index(drop=True)
    k_all = float(hexes[popcol].sum())
    cent = gpd.GeoDataFrame(geometry=hexes.centroid, crs=UTM)
    hexes["unit"] = assign(cent, g, sunda_land, torshavn_pts)
    print(f"\n  Kontur FO 2023-11-01: {len(hexes):,} populated hexes, {k_all:,.0f} people")
    reg23 = {d: regions[d]["2023M11"] for d in fo.DISTRICTS}
    k = hexes.groupby("unit")[popcol].sum()
    national = k_all / regions["Total"]["2023M11"]
    print("  Kontur alone / register November 2023, over the national "
          f"{national:.3f} (why it is calibrated): "
          + ", ".join(f"{d} {k[d] / reg23[d] / national:.3f}" for d in fo.DISTRICTS))

    # ---- calibrate to villages
    hexes["village"] = None
    for d in fo.DISTRICTS:
        hi = hexes.index[hexes["unit"] == d]
        vd = vil[vil["district"] == d]
        near = first(gpd.sjoin_nearest(cent.loc[hi, ["geometry"]], vd[["name", "geometry"]],
                                       how="left"))
        hexes.loc[hi, "village"] = near["name"]
    kv = hexes.groupby("village")[popcol].sum()
    reg = vil.set_index("name")["register"]
    hexes["pop"] = hexes[popcol] * hexes["village"].map(reg / kv).fillna(0.0)
    moved = {}
    for d in fo.DISTRICTS:
        m = hexes["unit"] == d
        a = hexes.loc[m, popcol] / hexes.loc[m, popcol].sum()
        b = hexes.loc[m, "pop"] / hexes.loc[m, "pop"].sum()
        moved[d] = 0.5 * float((a - b).abs().sum())
    print("  weight moved between hexes by the calibration, share of each district: "
          + ", ".join(f"{d} {100 * v:.0f}%" for d, v in moved.items()))
    for v in ("Tórshavn", "Hvítanes", "Kaldbak", "Haldórsvík", "Klaksvík", "Sandur"):
        print(f"     {v}: Kontur {kv.get(v, 0):,.0f} (2023) -> register {reg[v]:,} (Nov 2011)")

    lonely = vil[(vil["register"] > 0) & ~vil["name"].isin(kv.index)]
    discs = gpd.GeoDataFrame({"unit": lonely["district"].values, "village": lonely["name"].values,
                              "pop": lonely["register"].astype(float).values},
                             geometry=lonely.buffer(DISC_M).values, crs=UTM)
    print(f"  {len(discs)} villages with people and no hex of their own get a {DISC_M} m disc: "
          + ", ".join(f"{v} {int(p)}" for v, p in zip(discs["village"], discs["pop"])))
    out = pd.concat([hexes.loc[hexes["pop"] > 0, ["unit", "village", "pop", "geometry"]], discs],
                    ignore_index=True)
    out = gpd.GeoDataFrame(out, geometry="geometry", crs=UTM)

    # ---- witness 3
    wsum = out.groupby("unit")["pop"].sum()
    nopt_d = {d: sum(villages[v][0] for v in NO_POINT if district_of[v] == d) for d in fo.DISTRICTS}
    bad = [(d, round(float(wsum[d]), 3), regions[d]["2011M11"] - nopt_d[d]) for d in fo.DISTRICTS
           if abs(float(wsum[d]) - (regions[d]["2011M11"] - nopt_d[d])) > 0.5]
    say(not bad, "the calibrated weight sums to each district's November 2011 register "
                 "population, less the villages with no point" + (f"; differ {bad}" if bad else ""))

    out = out.to_crs("EPSG:4326")
    w, s, e, n = out.total_bounds
    say(-7.8 < w < e < -6.1 and 61.3 < s < n < 62.45,
        f"bbox {w:.3f} {s:.3f} {e:.3f} {n:.3f} is the Faroes")
    say(set(out["unit"]) == set(fo.DISTRICTS), "the layer covers all seven districts")
    if not ok:
        raise SystemExit("placement checks FAILED")
    os.makedirs(GEO, exist_ok=True)
    out.to_file(HEXES + ".part.gpkg", driver="GPKG", layer="fo_hexes")
    os.replace(HEXES + ".part.gpkg", HEXES)
    print(f"\n  wrote {HEXES} ({len(out):,} features)")


if __name__ == "__main__":
    main()
