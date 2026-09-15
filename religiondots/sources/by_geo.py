"""Belarus: boundaries and populations for the six oblasts and Minsk city.

Writes data/geo/by/by_oblasts.gpkg and data/geo/by/by_lookup.csv.

OCHA COD-AB Belarus (`cod-ab-blr`, v01, valid from 2022-07-27), the shapefile bundle: seven ADM1
features, six oblasts and Minsk City, the same seven units as Belstat's tables and LiTS III's
sample frame.

## COD'S MINSK CITY IS A QUARTER OF THE CITY, SO IT IS REPLACED

COD draws Minsk City at 86.8 km². The city is 353.64 km² (ru.wikipedia's infobox; the 2012
boundary decree gave 348.84 km²), and five of LiTS III's nine Minsk PSUs (both Frunzenskij,
Moskovskij, Pervomajskij, Oktjabr'skij) fall outside COD's polygon, in Minsk oblast. Left as
delivered, Minsk oblast's dots would be drawn over the capital's outer districts and the city's two
million people squeezed into its centre, with every total still correct. So Minsk City is
OpenStreetMap relation 59195 (fetched from Nominatim) clipped to COD's Minsk oblast plus Minsk
City, joined with COD's own city polygon, and Minsk oblast is what is left of the two. The join
keeps a 1.02 km² south-eastern patch that COD puts in the city and OSM does not. Asserted: the OSM
polygon leaves at most 2 km² of COD's city out, the result is within 2% of 353.64 km², and all
nine city PSUs fall inside it and no Minsk oblast PSU does. geoBoundaries was no better: its Belarus ADM1 is CIESIN's 2005 layer, smallest unit 217 km².

## POPULATION IS BELSTAT'S 1 JANUARY 2026 TABLE

`belstat.gov.by`, "Численность населения на 1 января 2026 г. по областям и г.Минску", read out of
the page's own HTML table. Seven figures summing exactly to the national 9,056,080. COD-PS is not
used: Belarus counted in 2019 and publishes every year since.

## WITNESSES, ONE PER JOIN

- Belstat to COD: the Russian label, transliterated, begins the COD English name (first five
  letters) and exactly one of the seven, with `г.` pairing only with `Minsk City`.
- LiTS to COD, by name: the label's first word is the COD name's first word, and `region` pairs
  only with a unit that is not the city. Both Minsks start `Minsk`, so the name alone cannot
  pin the twin; that is the next witness's job.
- LiTS to COD, by place: each PSU's own latitude and longitude falls inside the polygon of the
  unit its label decodes to. One PSU (28, `Vitebski Senno`) sits 3 km over the Mogilev line
  and is allowed by name, within `PSU_SLACK_KM`; any other stops the build.

Usage:
    python sources/by_geo.py --fetch    one ~1.3 MB zip from HDX, one page from belstat.gov.by,
                                        one GeoJSON from Nominatim
    python sources/by_geo.py            rebuild from data/raw/by/
"""

import html
import os
import re
import sys
import urllib.request
import zipfile

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import geopandas as gpd
import pandas as pd
from shapely.ops import unary_union

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
RAW = os.path.join(ROOT, "data", "raw", "by")
SHP_DIR = os.path.join(RAW, "shp")
OUT_DIR = os.path.join(ROOT, "data", "geo", "by")
OUT = os.path.join(OUT_DIR, "by_oblasts.gpkg")
LOOKUP = os.path.join(OUT_DIR, "by_lookup.csv")

UA_BROWSER = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) religiondots/1.0"}
UA_NOMINATIM = {"User-Agent": "religiondots/1.0 (religion dot map research)"}

DOWNLOADS = {
    "blr_admin_boundaries.shp.zip": (
        "https://data.humdata.org/dataset/d64db592-f54c-4ae4-9b68-2fd81d7a4175/resource/"
        "e5dec16b-89d4-4a4b-9f3e-bb40b7269eb5/download/blr_admin_boundaries.shp.zip", UA_BROWSER),
    "belstat_population_2026.html": (
        "https://www.belstat.gov.by/ofitsialnaya-statistika/ssrd-mvf_2/"
        "natsionalnaya-stranitsa-svodnyh-dannyh/naselenie_6/chislennost-naseleniya1_yan_poobl/",
        UA_BROWSER),
    "osm_minsk_r59195.geojson": (
        "https://nominatim.openstreetmap.org/lookup?osm_ids=R59195&format=geojson"
        "&polygon_geojson=1", UA_NOMINATIM),
}

N_UNITS = 7
BELSTAT_TOTAL = 9_056_080
BELSTAT_DATE = "1 January 2026"
MINSK_CITY_KM2 = 353.64          # ru.wikipedia infobox for Минск
AREA_CRS = 32635                 # UTM 35N; Belarus runs 23-33 E

COD_NAME = {
    "BY001": "Brest", "BY002": "Gomel", "BY003": "Grodno", "BY004": "Minsk",
    "BY005": "Minsk City", "BY006": "Mogilev", "BY007": "Vitebsk",
}
EN_NAME = {
    "BY001": "Brest", "BY002": "Gomel", "BY003": "Grodno", "BY004": "Minsk region",
    "BY005": "Minsk city", "BY006": "Mogilev", "BY007": "Vitebsk",
}

# Belstat's row labels, verbatim, against the COD pcode.
BELSTAT_LABEL = {
    "Брестская": "BY001", "Витебская": "BY007", "Гомельская": "BY002", "Гродненская": "BY003",
    "г.Минск": "BY005", "Минская": "BY004", "Могилевская": "BY006",
}

# LiTS III `region_name`, verbatim. Two oblasts are spelled two ways in the delivered file (a
# double space in `Gomel'  region`, a trailing space on `Grodno region `), and the city is
# `Minsk ` with a trailing space, so `lits.load`'s end-strip alone would split Gomel' in two.
LITS_RAW = {"Brest region", "Gomel' region", "Gomel'  region", "Grodno region", "Grodno region ",
            "Minsk ", "Minsk region", "Mogilev region", "Vitebsk region"}
LITS_REGION = {
    "Brest region": "BY001", "Gomel' region": "BY002", "Grodno region": "BY003",
    "Minsk region": "BY004", "Minsk": "BY005", "Mogilev region": "BY006",
    "Vitebsk region": "BY007",
}

# PSUs whose coordinates fall outside their labelled unit, by name, and how far they may be.
PSU_OUTSIDE = {28: "Vitebski Senno"}
PSU_SLACK_KM = 15.0

_TR = dict(zip("абвгдеёжзийклмнопрстуфхцчшщъыьэюя",
               ["a", "b", "v", "g", "d", "e", "e", "zh", "z", "i", "i", "k", "l", "m", "n", "o",
                "p", "r", "s", "t", "u", "f", "kh", "ts", "ch", "sh", "shch", "", "y", "", "e",
                "yu", "ya"]))


def lits_label(s):
    """Collapse LiTS's internal and trailing spaces: `Gomel'  region` -> `Gomel' region`."""
    return re.sub(r"\s+", " ", str(s)).strip()


def translit(s):
    return "".join(_TR.get(ch, ch) for ch in str(s).lower())


def fetch():
    os.makedirs(RAW, exist_ok=True)
    for name, (url, headers) in DOWNLOADS.items():
        dst = os.path.join(RAW, name)
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            print(f"  have {name} ({os.path.getsize(dst):,} bytes)")
            continue
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=300) as r:
            body = r.read()
        with open(dst + ".part", "wb") as f:
            f.write(body)
        os.replace(dst + ".part", dst)
        print(f"  got  {name} ({len(body):,} bytes)")


def belstat_population():
    """The seven unit figures and the national one, read out of the page's own table."""
    raw = open(os.path.join(RAW, "belstat_population_2026.html"), encoding="utf-8").read()
    text = html.unescape(re.sub(r"<[^>]+>", "\n", raw)).replace("\xa0", " ")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    number = re.compile(r"^\d{1,3}(?: \d{3})+$")
    try:
        anchor = lines.index("Области и г.Минск:")
    except ValueError:
        raise SystemExit("Belstat's page has no `Области и г.Минск:` row; the layout changed")

    def first_number(i):
        for ln in lines[i + 1:i + 20]:
            if number.match(ln):
                return int(ln.replace(" ", ""))
        raise SystemExit(f"no figure after Belstat's row {lines[i]!r}")

    rb = max(i for i in range(anchor) if lines[i] == "Республика Беларусь")
    national = first_number(rb)
    rows = {}
    for label, pcode in BELSTAT_LABEL.items():
        hits = [i for i in range(anchor, len(lines)) if lines[i] == label]
        if not hits:
            raise SystemExit(f"Belstat's table has no row {label!r}")
        rows[pcode] = first_number(hits[0])
    if national != BELSTAT_TOTAL:
        raise SystemExit(f"Belstat's national figure is now {national:,}, not {BELSTAT_TOTAL:,}; "
                         "the page was reissued, so update BELSTAT_TOTAL and BELSTAT_DATE")
    if sum(rows.values()) != national:
        raise SystemExit(f"the seven units sum to {sum(rows.values()):,} against {national:,}")
    print(f"  Belstat, {BELSTAT_DATE}: {national:,}, and the seven units sum to it exactly")
    return pd.Series(rows, name="pop")


def name_witnesses():
    """Belstat's Russian labels and LiTS's English ones, each against COD's names."""
    bad = []
    for label, pcode in BELSTAT_LABEL.items():
        city = label.startswith("г.")
        stem = translit(label.replace("г.", ""))
        hits = sorted(p for p, nm in COD_NAME.items()
                      if stem.startswith(nm.lower()[:5]) and (("City" in nm) == city))
        if hits != [pcode]:
            bad.append(f"Belstat {label!r} is written against {pcode} and matches {hits}")
    for label, pcode in LITS_REGION.items():
        region = label.endswith(" region")
        first = label.split()[0].replace("'", "").lower()
        hits = sorted(p for p, nm in COD_NAME.items()
                      if nm.split()[0].lower() == first and (("City" in nm) != region))
        if hits != [pcode]:
            bad.append(f"LiTS {label!r} is written against {pcode} and matches {hits}")
    if bad:
        raise SystemExit("name witnesses failed:\n  " + "\n  ".join(bad))
    print(f"  names: each Belstat and LiTS label matches exactly one of COD's {N_UNITS} names, "
          "and it is the one it is written against")


def minsk_surgery(g):
    """Replace COD's 86.8 km² Minsk City with OSM relation 59195, and give the rest to the oblast."""
    osm = gpd.read_file(os.path.join(RAW, "osm_minsk_r59195.geojson"))
    if len(osm) != 1 or osm.geometry.iloc[0].geom_type not in ("Polygon", "MultiPolygon"):
        raise SystemExit(f"Nominatim returned {len(osm)} features, expected one polygon")
    city_osm = osm.to_crs(g.crs).geometry.iloc[0].buffer(0)
    cod_city = g.loc["BY005", "geometry"].buffer(0)
    cod_obl = g.loc["BY004", "geometry"].buffer(0)
    both = unary_union([cod_city, cod_obl])

    def km2(geom):
        return float(gpd.GeoSeries([geom], crs=g.crs).to_crs(AREA_CRS).area.iloc[0]) / 1e6

    outside = km2(city_osm.difference(both))
    uncovered = km2(cod_city.difference(city_osm))
    # The two sources disagree about one patch: COD's city polygon runs 1.02 km² past OSM's line in
    # the south-east (27.69 E, 53.87 N, up to about 750 m; measured 2026-09-14, plus a 0.04 km²
    # sliver). Nothing here says which is right, so the city keeps both claims: OSM's polygon
    # plus COD's. Two square kilometres is the ceiling on that disagreement; more is a different
    # boundary, not a patch.
    city = unary_union([city_osm.intersection(both), cod_city])
    oblast = both.difference(city)
    print(f"  Minsk City: COD {km2(cod_city):.1f} km², OSM relation 59195 {km2(city_osm):.1f} km²; "
          f"OSM outside COD's Minsk pair {outside:.2f} km², COD city outside OSM {uncovered:.2f} km²")
    if outside > 0.5 or uncovered > 2.0:
        raise SystemExit("OSM's Minsk does not sit inside COD's Minsk oblast plus city, or leaves "
                         "more than a patch of COD's city polygon out; look before replacing anything")
    if abs(km2(city) / MINSK_CITY_KM2 - 1) > 0.02:
        raise SystemExit(f"Minsk City is {km2(city):.1f} km² against {MINSK_CITY_KM2}")
    print(f"  Minsk City drawn at {km2(city):.1f} km² (city's own figure {MINSK_CITY_KM2}), "
          f"Minsk oblast at {km2(oblast):,.0f} km² (COD's was {km2(cod_obl):,.0f})")
    g.loc["BY005", "geometry"] = city
    g.loc["BY004", "geometry"] = oblast
    return g


def psu_witness(g):
    """Each LiTS PSU's own coordinates against the polygon its label decodes to."""
    import lits
    d = pd.read_stata(lits.DTA, columns=["country", "PSU_number", "PSU_name", "region_name",
                                         "latitude", "longitude"], convert_categoricals=True)
    d = d[d["country"].astype(str).str.contains("Belarus", case=False, na=False)]
    raw = set(d["region_name"].astype(str))
    if raw != LITS_RAW:
        raise SystemExit(f"LiTS III's Belarus region strings are now {sorted(raw)}, not "
                         f"{sorted(LITS_RAW)}")
    d = d.assign(label=d["region_name"].map(lits_label))
    psu = d.groupby("PSU_number").agg(label=("label", "first"), name=("PSU_name", "first"),
                                      lat=("latitude", "first"), lon=("longitude", "first"),
                                      labels=("label", "nunique")).reset_index()
    if (psu["labels"] != 1).any():
        raise SystemExit("a LiTS PSU carries two region labels")
    psu["unit"] = psu["label"].map(LITS_REGION)
    pts = gpd.GeoDataFrame(psu, geometry=gpd.points_from_xy(psu["lon"], psu["lat"]), crs=4326)
    j = gpd.sjoin(pts, g[["pcode", "geometry"]].reset_index(drop=True), how="left",
                  predicate="within")
    j = j[~j.index.duplicated(keep="first")]
    wrong = j[j["pcode"] != j["unit"]]
    gm = g.to_crs(AREA_CRS)
    bad = []
    for _, r in wrong.iterrows():
        pt = gpd.GeoSeries([r.geometry], crs=4326).to_crs(AREA_CRS).iloc[0]
        dist = pt.distance(gm.loc[r["unit"], "geometry"]) / 1000
        line = (f"PSU {int(r['PSU_number'])} {r['name']!r} labelled {r['label']!r} lies in "
                f"{r['pcode']} ({COD_NAME.get(r['pcode'], 'nothing')}), {dist:.1f} km from "
                f"{COD_NAME[r['unit']]}")
        if int(r["PSU_number"]) in PSU_OUTSIDE and dist <= PSU_SLACK_KM:
            print(f"    allowed by name: {line}")
        else:
            bad.append(line)
    if bad:
        raise SystemExit("LiTS PSUs outside the unit their label decodes to:\n  " + "\n  ".join(bad))
    in_city = j[j["pcode"] == "BY005"]
    if set(in_city["unit"]) != {"BY005"} or len(in_city) != int((psu["unit"] == "BY005").sum()):
        raise SystemExit("the Minsk twin is not separated: city PSUs and oblast PSUs mix")
    print(f"  places: {len(psu) - len(wrong)} of {len(psu)} LiTS PSUs fall inside the unit their "
          f"label decodes to, all {len(in_city)} Minsk city PSUs inside the city polygon and no "
          "Minsk region PSU inside it")


def main():
    if "--fetch" in sys.argv:
        fetch()
    zpath = os.path.join(RAW, "blr_admin_boundaries.shp.zip")
    if not os.path.exists(zpath):
        raise SystemExit(f"{zpath} missing; run with --fetch")
    os.makedirs(SHP_DIR, exist_ok=True)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(SHP_DIR)
    g = gpd.read_file(os.path.join(SHP_DIR, "blr_admin1.shp"), engine="fiona")
    if len(g) != N_UNITS:
        raise SystemExit(f"{len(g)} ADM1 features, expected {N_UNITS}")
    if g.crs is None or g.crs.to_epsg() != 4326:
        raise SystemExit(f"unexpected CRS {g.crs}")
    g["pcode"] = g["adm1_pcode"].astype(str).str.strip()
    got = dict(zip(g["pcode"], g["adm1_name"]))
    if got != COD_NAME:
        raise SystemExit(f"COD's pcodes or names changed: {got}")
    g = g.set_index("pcode", drop=False)
    print(f"read COD-AB Belarus ADM1: {len(g)} units")

    pop = belstat_population()
    name_witnesses()
    g = minsk_surgery(g)
    psu_witness(g)

    g["pop"] = g["pcode"].map(pop).astype("int64")
    g["area_km2"] = g.to_crs(AREA_CRS).area / 1e6
    g["density"] = g["pop"] / g["area_km2"]
    order = g.sort_values("density", ascending=False)["pcode"].map(EN_NAME).tolist()
    print(f"    densest to sparsest: {', '.join(order)}")
    if order[0] != "Minsk city" or g.loc["BY005", "density"] < 3000 or \
            g.drop(index="BY005")["density"].max() > 100:
        raise SystemExit("density does not put Minsk city far above every oblast; the population "
                         "join is permuted or the city polygon is wrong")

    g["unit"] = g["pcode"]
    g["geo_id"] = g["pcode"]
    g["name"] = g["pcode"].map(EN_NAME)
    os.makedirs(OUT_DIR, exist_ok=True)
    out = g.reset_index(drop=True)[["unit", "name", "pcode", "geo_id", "pop", "geometry"]]
    out = gpd.GeoDataFrame(out, geometry="geometry", crs=g.crs)
    out.to_file(OUT, layer="oblasts", driver="GPKG")
    print(f"\nwrote {OUT} ({len(out)} polygons)")

    by_pcode = {v: k for k, v in LITS_REGION.items()}
    ru = {v: k for k, v in BELSTAT_LABEL.items()}
    codes = sorted(g["pcode"])
    lut = pd.DataFrame({
        "geo_id": codes, "unit": codes, "name": [EN_NAME[p] for p in codes],
        "name_ru": [ru[p] for p in codes], "lits_region": [by_pcode[p] for p in codes],
        "pop": [int(pop[p]) for p in codes],
        "area_km2": [round(float(g.loc[p, "area_km2"]), 1) for p in codes],
    })
    lut.to_csv(LOOKUP, index=False, encoding="utf-8")
    print(f"wrote {LOOKUP}")
    print(lut.to_string(index=False))


if __name__ == "__main__":
    main()
