"""Kazakhstan: the 2021 census's 218 rayons and cities as polygons, and a Kontur placement layer.

    python sources/kz_geo.py --fetch   also fetch the three OSM relations (polygons.openstreetmap.fr)
    python sources/kz_geo.py           -> data/geo/kz/kz_units.gpkg, kz_lookup.csv, kz_hexes.gpkg

THE CENSUS SIDE is `data/normalized/kz.csv`'s `rayon` rows (sources/kz_census.py): 218 units, each
an engine rayon number (`R144`) with its Russian name (OWNER3_NAME) and its oblast's KATO code.
The engine has no rayon KATO code (its `РАЙОН КАТО` field is another, half-filled name string).

NO KATO-CODED RAYON LAYER WAS FOUND. religiondots stopped at 17 oblasts and joined nothing finer
(its `карта_район` find has 190 rows, not 218); OSM's 227 admin_level=6 relations in Kazakhstan
carry no `ref:kato` (Overpass, 2026-10-05); COD-AB's ADM2 is English with `KAZ###` pcodes.

THE BOUNDARY FILE is COD-AB Kazakhstan 2023 ADM2 (religiondots/data/raw/kz, read-only), 218
polygons, which is the census's count by coincidence: four changes between them cancel.
  * After the census (2022), in COD: Aksuat district carved from Tarbagatay, and Samar district
    from Kokpekti. Undone: each is dissolved back (the census puts Aksuat village in Tarbagatay
    and Samarskoye in Kokpekti; asserted below).
  * Before the census (2021), not in COD: Kosshy became a city administration out of
    Tselinograd district, and Sauran district was made from Turkestan city administration's
    rural okrugs. COD's "Turkestan" is the city's 282 km2 and its "Kentau" (7,699 km2) holds
    Sauran's ground as well as Kentau's. Rebuilt from OSM's own relations of those units:
    Kosshy (15594335) is cut out of Tselinograd; Turkestan city (5496366) and Kentau
    (17322798) are cut out of the union of COD's Turkestan and Kentau, and Sauran is the rest.
The 2022 oblast reform (Abai, Jetisu, Ulytau) moved whole rayons, so dissolving COD's ADM1 back to
17 regions (religiondots' kz_geo.py MERGE) gives every rayon its 2021 oblast.

THE JOIN is by name inside each 2021 oblast: the census's Russian name transliterated and
stripped of `район`, `Г.А.` and the adjectival ending, against COD's English stripped of
`District`, scored by string similarity and assigned one-to-one best first. Every pair whose
normalised names are not identical is in PINNED, written and read pair by pair (a rename such as
Beimbet Mailin = Taran, Baiterek = Zelenov, Akkuly = Lebyazhye, Kapchagay = Qonayev, or a
spelling like Termitau); the run stops on any unpinned inexact pair, and on any pin the scorer
would not have reached. Asserted 218 both ways, and the census population on them.

THE WITNESS neither key decides is Kontur: per-rayon Kontur over census, normalised nationally,
with two shuffle controls: across the whole country (`_grid.hex_layer`) and, the discriminating
one for a name join made inside each oblast, shuffled only within oblasts.
"""
import difflib
import json
import math
import os
import random
import re
import shutil
import sys
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

COD_ZIP = RD / "data" / "raw" / "kz" / "kaz_adm_unhcr_2023_shp.zip"
COD_DIR = "kaz_adm_unhcr_2023_SHP"
RD_KONTUR_GZ = RD / "data" / "raw" / "kz" / "kontur_population_KZ_20231101.gpkg.gz"
OUR_KONTUR = ROOT / "data" / "geo" / "kontur"
RAW = ROOT / "data" / "raw" / "kz"
GEO = ROOT / "data" / "geo" / "kz"
NORM = ROOT / "data" / "normalized" / "kz.csv"
NATIONAL = 19_186_015
EXPECTED = 218

# COD's 2023 ADM1 -> the census's 2021 oblast KATO (the 2022 reform undone, as religiondots does)
ADM1_KATO = {
    "Akmola Region": "110000000", "Aktobe Region": "150000000", "Almaty Region": "190000000",
    "Jetisu Region": "190000000", "Atyrau Region": "230000000",
    "West Kazakhstan Region": "270000000", "Jambyl Region": "310000000",
    "Karaganda Region": "350000000", "Ulytau Region": "350000000",
    "Kostanay Region": "390000000", "Kyzylorda Region": "430000000",
    "Mangystau Region": "470000000", "Pavlodar Region": "550000000",
    "North Kazakhstan Region": "590000000", "Turkistan Region": "610000000",
    "East Kazakhstan Region": "630000000", "Abay Region": "630000000",
    "Astana": "710000000", "Almaty": "750000000", "Shymkent": "790000000",
}
# 2022 districts dissolved back into the rayon they were carved from (COD pcode -> COD pcode)
MERGE_BACK = {"KAZ001002": "KAZ008008",    # Aksuat -> Tarbagatay
              "KAZ008006": "KAZ001006"}    # Samar -> Kokpekti
MERGE_CHECK = {"KAZ008008": ("R016", "С.АКСУАТ"), "KAZ001006": ("R019", "С.САМАРСКОЕ")}

OSM = {"kosshy": 15594335, "turkestan_city": 5496366, "kentau": 17322798, "sauran": 3407078}
TSELINOGRAD, COD_TURKESTAN, COD_KENTAU = "KAZ002016", "KAZ018015", "KAZ018005"
# the three rebuilt units, pinned to their census rows
REBUILT = {"kosshy": "R180", "turkestan_city": "R031", "kentau": "R199", "sauran": "R198"}

# census name -> COD English, for every pair whose normalised names are not identical. Each was
# read by hand; the scorer must also reach it (asserted), so a pin cannot hide a wrong twin.
PINNED = {
    # renames
    "R039": "Zelenov District",        # РАЙОН БӘЙТЕРЕК, Zelenov renamed Baiterek 2018
    "R174": "Lebyazhye District",      # РАЙОН АҚҚУЛЫ, Lebyazhye renamed Akkuly
    "R182": "Qonayev",                 # КАПЧАГАЙ Г.А., renamed Konaev 2022
    "R099": "Taran District",          # РАЙОН БЕИМБЕТА МАЙЛИНА, Taran renamed
    # Russian against Kazakh spelling of the same name
    "R035": "Oiyl District",           # УИЛСКИЙ
    "R057": "Borili District",         # БУРЛИНСКИЙ
    "R052": "Zhanakala District",      # ЖАНГАЛИНСКИЙ (the last WKO district left; seat Zhangala)
    "R069": "Karakiya District",       # КАРАКИЯНСКИЙ
    "R153": "Ile District",            # ИЛИЙСКИЙ
    "R104": "Nauyrzym District",       # НАУРЗУМСКИЙ
    "R030": "Sozak District",          # СУЗАКСКИЙ
    "R200": "Termitau",                # ТЕМИРТАУ Г.А. (COD's spelling)
    "R084": "Alga District",           # АЛГИНСКИЙ
    "R209": "Satbayev",                # САТПАЕВ Г.А.
    "R041": "Nura District",           # НУРИНСКИЙ (Karaganda)
    "R021": "Kurshim District",        # КУРЧУМСКИЙ
    "R212": "Baikonur",                # БАЙКОНЫР Г.А. (Kyzylorda)
    "R173": "Sharbaqty District",      # ЩЕРБАКТИНСКИЙ
    "R142": "Tulkibas District",       # ТЮЛЬКУБАССКИЙ
    "R001": "Baikonur District",       # РАЙОН БАЙҚОҢЫР (Astana)
    "R190": "Turar Ryskulov District",
    "R093": "Fyodorovsky District",    # ФЕДОРОВСКИЙ
    "R163": "Ertis District",          # ИРТЫШСКИЙ
    "R032": "Otyrar District",         # ОТРАРСКИЙ
    "R013": "Urzhar District",         # УРДЖАРСКИЙ
    "R126": "Gabit Musirepov District",
    "R063": "Kyzylkoga District",      # КЗЫЛКОГИНСКИЙ
    "R123": "Magzhan Zhumabaev District",
}

TR = dict(zip("абвгдеёжзийклмнопрстуфхцчшщъыьэюяәғқңөұүһі",
              ["a", "b", "v", "g", "d", "e", "e", "zh", "z", "i", "y", "k", "l", "m", "n", "o",
               "p", "r", "s", "t", "u", "f", "kh", "ts", "ch", "sh", "shch", "", "y", "", "e",
               "yu", "ya", "a", "g", "k", "n", "o", "u", "u", "h", "i"]))


def fold(s):
    """Latin letters only, with the spellings that vary between transliterations merged."""
    s = s.lower()
    for a, b in (("shch", "sh"), ("ch", "sh"), ("kh", "k"), ("q", "k"), ("zh", "j"), ("yu", "u"),
                 ("ya", "a"), ("y", "i"), ("w", "u"), ("ö", "o"), ("ü", "u"), ("ï", "i"),
                 ("e", "i")):
        s = s.replace(a, b)
    s = re.sub(r"[^a-z]", "", s)
    s = re.sub(r"(.)\1+", r"\1", s)
    # Russian adjectival and genitive endings against Kazakh/English bare stems:
    # algin/alga, kobdin/kobda, abaiski/abai, gabitamusripova/gabitmusiripov
    prev = None
    while prev != s:
        prev = s
        s = re.sub(r"(ski|in|ov|a|i|o|u)$", "", s) if len(s) > 4 else s
    return s


def census_key(name):
    """`c:` marks a city administration (Г.А.), so Kostanay city and Kostanay district,
    Pavlodar city and district, cannot trade places."""
    city = "c:" if re.search(r"г\.а\.?", name.lower()) else ""
    return city + _census_stem(name)


def _census_stem(name):
    s = name.lower().replace("ё", "е")
    s = re.sub(r"\bг\.а\.?", " ", s)
    s = re.sub(r"\bрайон\b", " ", s)
    s = re.sub(r"\bим\.", " ", s)
    s = " ".join(s.split())
    # adjectival rayon names: алакольский -> алаколь; айтекебийский -> айтекеби
    s = re.sub(r"(ий|ый|ой)?(с|ц)кий$", "", s) if re.search(r"(с|ц)кий$", s) else s
    return fold("".join(TR.get(c, c) for c in s))


def cod_key(name):
    s = re.sub(r"\b(district|public administration)\b", " ", name, flags=re.I)
    s = s.split(",")[0]
    city = "" if re.search(r"\bdistrict\b", name, flags=re.I) else "c:"
    return city + fold(s)


def fetch():
    import requests
    RAW.mkdir(parents=True, exist_ok=True)
    h = {"User-Agent": "languagedots-boundary-check/1.0"}
    for name, rid in OSM.items():
        dest = RAW / f"osm_rel_{rid}.geojson"
        if dest.exists() and dest.stat().st_size > 1000:
            continue
        requests.get(f"https://polygons.openstreetmap.fr/index.py?id={rid}", headers=h, timeout=300)
        r = requests.get(f"https://polygons.openstreetmap.fr/get_geojson.py?id={rid}&params=0",
                         headers=h, timeout=300)
        r.raise_for_status()
        json.loads(r.text)
        dest.write_text(r.text, encoding="utf-8")
        print(f"  OSM relation {rid} ({name}): {len(r.text):,} bytes")


def osm_shape(rid):
    import geopandas as gpd
    g = gpd.read_file(RAW / f"osm_rel_{rid}.geojson")
    g = g.set_crs(4326, allow_override=True)
    return g.union_all()


def build_units(a2):
    """COD ADM2 -> 2021 units: undo the 2022 splits, rebuild Kosshy, Turkestan, Kentau, Sauran."""
    import geopandas as gpd
    from shapely.ops import unary_union

    a2 = a2.copy()
    a2["oblast"] = a2["ADM1_EN"].map(ADM1_KATO)
    if a2["oblast"].isna().any():
        raise SystemExit(f"COD ADM1 not in ADM1_KATO: {sorted(a2.loc[a2['oblast'].isna(), 'ADM1_EN'].unique())}")
    a2["pc"] = a2["ADM2_PCODE"].replace(MERGE_BACK)
    for child, parent in MERGE_BACK.items():
        a, b = a2.set_index("ADM2_PCODE").loc[[child, parent], "oblast"]
        if a != b:
            raise SystemExit(f"{child} and {parent} are in different 2021 oblasts")
    u = a2.dissolve(by="pc", as_index=False, aggfunc="first")[["pc", "oblast", "geometry"]]
    u["ADM2_EN"] = u["pc"].map(dict(zip(a2["ADM2_PCODE"], a2["ADM2_EN"])))
    print(f"  COD {len(a2)} ADM2 -> {len(u)} after dissolving Aksuat and Samar back")

    # equal-area metres for the cuts, then back
    u = u.to_crs(6933)
    shp = {k: gpd.GeoSeries([osm_shape(v)], crs=4326).to_crs(6933).iloc[0] for k, v in OSM.items()}
    idx = {p: i for i, p in enumerate(u["pc"])}
    km2 = lambda g: g.area / 1e6  # noqa: E731

    # Kosshy out of Tselinograd
    tsel = u.geometry.iloc[idx[TSELINOGRAD]]
    kos = shp["kosshy"].intersection(tsel)
    if km2(kos) / km2(shp["kosshy"]) < 0.98:
        raise SystemExit(f"OSM Kosshy only {km2(kos) / km2(shp['kosshy']):.2f} inside COD Tselinograd")
    new = [{"pc": "OSM15594335", "oblast": "110000000", "ADM2_EN": "Kosshy", "geometry": kos}]
    u.loc[idx[TSELINOGRAD], "geometry"] = tsel.difference(kos)

    # Turkestan, Kentau, Sauran out of COD's Turkestan + Kentau
    block = unary_union([u.geometry.iloc[idx[COD_TURKESTAN]], u.geometry.iloc[idx[COD_KENTAU]]])
    city = shp["turkestan_city"].intersection(block)
    ken = shp["kentau"].intersection(block).difference(city)
    sau = block.difference(city).difference(ken)
    so = shp["sauran"].intersection(block)
    print(f"  Turkestan block {km2(block):,.0f} km2 (COD Turkestan "
          f"{km2(u.geometry.iloc[idx[COD_TURKESTAN]]):,.0f} + Kentau "
          f"{km2(u.geometry.iloc[idx[COD_KENTAU]]):,.0f}) -> city {km2(city):,.0f}, Kentau "
          f"{km2(ken):,.0f}, Sauran {km2(sau):,.0f}")
    print(f"    OSM's own Sauran: {km2(shp['sauran']):,.0f} km2, {km2(so):,.0f} inside the block; "
          f"it overlaps OSM Kentau by {km2(shp['sauran'].intersection(shp['kentau'])):,.0f} and "
          f"Turkestan city by {km2(shp['sauran'].intersection(shp['turkestan_city'])):,.0f}")
    for nm, g, want in (("city", shp["turkestan_city"], city), ("Kentau", shp["kentau"], ken)):
        if km2(want) / km2(g) < 0.95:
            raise SystemExit(f"OSM {nm} only {km2(want) / km2(g):.2f} inside COD's block")
    if km2(sau) < 5000:
        raise SystemExit(f"Sauran remainder {km2(sau):,.0f} km2, too small")
    u.loc[idx[COD_TURKESTAN], "geometry"] = city
    u.loc[idx[COD_TURKESTAN], "ADM2_EN"] = "Turkestan"
    u.loc[idx[COD_KENTAU], "geometry"] = ken
    new.append({"pc": "OSM3407078", "oblast": "610000000", "ADM2_EN": "Sauran",
                "geometry": sau})
    u = pd.concat([u, gpd.GeoDataFrame(new, geometry="geometry", crs=6933)], ignore_index=True)
    u = gpd.GeoDataFrame(u, geometry="geometry", crs=6933)
    if len(u) != EXPECTED:
        raise SystemExit(f"{len(u)} units after the rebuild, expected {EXPECTED}")
    return u.to_crs(4326)


def join(u, cen):
    """census rayon rows (geo_id, name, parent, pop) <-> COD units (pc, oblast, ADM2_EN)."""
    pinned_pc = {"R180": "OSM15594335", "R198": "OSM3407078", "R031": COD_TURKESTAN,
                 "R199": COD_KENTAU}
    pairs, report = {}, []
    for ob in sorted(cen["parent"].unique()):
        c = cen[cen["parent"] == ob]
        p = u[u["oblast"] == ob]
        if len(c) != len(p):
            raise SystemExit(f"oblast {ob}: {len(c)} census rayons, {len(p)} polygons")
        cands = []
        for _i, r in c.iterrows():
            for _j, q in p.iterrows():
                a, b = census_key(r["unit_name"]), cod_key(q["ADM2_EN"])
                sc = difflib.SequenceMatcher(None, a, b).ratio()
                if r["geo_id"] in pinned_pc:
                    sc = 2.0 if q["pc"] == pinned_pc[r["geo_id"]] else -1.0
                # PINNED does not steer the scorer: an inexact pair passes only when the
                # scorer reached it unaided AND it is pinned (checked in main)
                cands.append((sc, r["geo_id"], q["pc"], r["unit_name"], q["ADM2_EN"], a, b))
        taken_c, taken_p = set(), set()
        for sc, g, pc, cn, en, a, b in sorted(cands, reverse=True):
            if g in taken_c or pc in taken_p:
                continue
            taken_c.add(g)
            taken_p.add(pc)
            pairs[g] = pc
            report.append((ob, g, cn, pc, en, sc, a, b))
    return pairs, report


def main():
    import geopandas as gpd

    if "--fetch" in sys.argv:
        fetch()
    a2 = gpd.read_file(f"zip://{COD_ZIP.as_posix()}!{COD_DIR}/kaz_admbnda_adm2_unhcr_2023.shp")
    if len(a2) != 218:
        raise SystemExit(f"COD ADM2 has {len(a2)} polygons, expected 218")
    u = build_units(a2)

    df = pd.read_csv(NORM, dtype={"geo_id": str, "parent": str})
    ray = df[df["geo_level"] == "rayon"]
    cen = ray.groupby(["geo_id", "unit_name", "parent"], as_index=False)["count"].sum()
    if len(cen) != EXPECTED or cen["count"].sum() != NATIONAL:
        raise SystemExit(f"census: {len(cen)} rayons, {cen['count'].sum():,} people")

    # the merge-back witnesses: the 2022 district's seat is in the 2021 rayon it is merged into
    sel = pd.read_csv(RAW / "kz_qlik_rayon_villages.csv", dtype=str) \
        if (RAW / "kz_qlik_rayon_villages.csv").exists() else None
    if sel is not None:
        for pc, (g, village) in MERGE_CHECK.items():
            hit = sel[(sel["rayon"] == g[1:].lstrip("0")) & (sel["village"] == village)]
            if hit.empty:
                raise SystemExit(f"{village} is not in census rayon {g}")
        print("  Aksuat village is in census Tarbagatay and Samarskoye in census Kokpekti")

    pairs, report = join(u, cen)
    bad, inexact = [], []
    for ob, g, cn, pc, en, sc, a, b in report:
        if sc >= 2.0:
            continue
        if a == b:
            continue
        inexact.append((ob, g, cn, en, sc, a, b))
        if PINNED.get(g) != en:
            bad.append((ob, g, cn, en, round(sc, 2)))
    print(f"\n  join: {len(pairs)} pairs, {len(inexact)} on inexact names")
    for ob, g, cn, en, sc, a, b in sorted(inexact, key=lambda r: r[4]):
        print(f"    {g} {cn:<28} -> {en:<30} {sc:.2f}  ({a} / {b})")
    unused = set(PINNED) - {r[1] for r in inexact}
    if unused:
        raise SystemExit(f"pins never used (exact already, or the scorer went elsewhere): {unused}")
    if bad:
        raise SystemExit(f"{len(bad)} inexact pairs not in PINNED; read each and pin it")
    if len(set(pairs.values())) != EXPECTED or set(pairs) != set(cen["geo_id"]) \
            or set(pairs.values()) != set(u["pc"]):
        raise SystemExit("the join is not one-to-one over all 218 both ways")
    print(f"  218 census rayons <-> 218 polygons, one-to-one both ways, {NATIONAL:,} people")

    pc2g = {v: k for k, v in pairs.items()}
    u["unit"] = u["pc"].map(pc2g)
    name = dict(zip(cen["geo_id"], cen["unit_name"]))
    u["name"] = u["unit"].map(name)
    lut = cen.rename(columns={"count": "pop"})
    lut["unit"] = lut["geo_id"]
    lut["cod_pcode"] = lut["geo_id"].map(pairs)
    lut["cod_name"] = lut["cod_pcode"].map(dict(zip(u["pc"], u["ADM2_EN"])))
    GEO.mkdir(parents=True, exist_ok=True)
    lut[["geo_id", "unit", "unit_name", "parent", "cod_pcode", "cod_name", "pop"]].to_csv(
        GEO / "kz_lookup.csv", index=False, encoding="utf-8")
    u[["unit", "name", "pc", "ADM2_EN", "oblast", "geometry"]].to_file(
        GEO / "kz_units.gpkg", driver="GPKG")
    a = u.to_crs(6933)
    a["km2"] = a.area / 1e6
    print("  smallest units, km2: " + ", ".join(
        f"{n} {k:.0f}" for n, k in a.sort_values("km2")[["name", "km2"]].head(5).values))

    # placement layer and the Kontur witness
    OUR_KONTUR.mkdir(parents=True, exist_ok=True)
    gz = OUR_KONTUR / RD_KONTUR_GZ.name
    if not gz.exists() and not (OUR_KONTUR / RD_KONTUR_GZ.name[:-3]).exists():
        shutil.copyfile(RD_KONTUR_GZ, gz)
    from _grid import hex_layer
    census = dict(zip(lut["unit"], lut["pop"]))
    layer = hex_layer("kz", u[["unit", "geometry"]], census=census)
    within_oblast_control(layer, lut)
    layer = calibrate_towns(layer, u, census)
    layer.to_file(GEO / "kz_hexes.gpkg", layer="hexes", driver="GPKG")
    print(f"  rewrote {GEO / 'kz_hexes.gpkg'} with town-calibrated `pop` (Kontur kept as `kontur`)")


GEONAMES = ROOT.parent / "data" / "geonamescities.csv"     # maps/data, read-only
# census towns GeoNames' cities file lacks, placed by hand; each must fall in its unit and have
# Kontur people within the disc (asserted like the rest)
TOWN_XY = {"Г.КАСКЕЛЕН": (76.620, 43.200), "Г.УШАРАЛ": (80.940, 46.170),
           "Г.СЕРЕБРЯНСК": (83.290, 49.690)}
TOWN_SKIP_SHARE = 0.95     # a town that is (nearly) its whole unit needs no calibration


def calibrate_towns(layer, u, census):
    """Kontur gives Kazakhstan's towns too small a share of their own rayon.

    The census counts every town (`Село КАТО` rows `Г.<name>`, 86 of them, in
    data/raw/kz/kz_qlik_rayon_villages.csv). In a disc round each town (radius for 2,000 people
    per km2, at least 3 km), Kontur's share of the rayon against the census town's share was a
    median 0.6 and as low as 0.22 (Lisakovsk; Aksu 0.25, Shakhtinsk 0.26, Stepnogorsk 0.34):
    without this, a mining town's dots would go to its villages. Inside each rayon that holds a
    town and other people, the disc's hexes are scaled to the town's census share and the rest
    to the remainder; the rayon's total and every count are untouched (brief §4.4: placement
    only). Rayons that are one city are left alone.
    """
    import geopandas as gpd
    from shapely.geometry import Point

    vil = pd.read_csv(RAW / "kz_qlik_rayon_villages.csv", dtype=str)
    towns = vil[vil["village"].str.startswith("Г.")].copy()
    towns["unit"] = "R" + towns["rayon"].str.zfill(3)
    towns["n"] = towns["n"].astype(int)
    gn = pd.read_csv(GEONAMES, sep=";", dtype=str)
    gn = gn[gn["Country Code"] == "KZ"]
    alt = gn["Alternate Names"].fillna("").str.lower().str.split(",")

    m = layer.to_crs(6933)
    cen = m.geometry.centroid
    kontur = m["pop"].to_numpy(dtype=float).copy()
    w = kontur.copy()
    units = u.set_index("unit")
    done, skipped, missing = [], [], []
    for un, ts in towns.groupby("unit"):
        share = ts["n"].sum() / census[un]
        if share >= TOWN_SKIP_SHARE:
            skipped.append(un)
            continue
        idx_u = (m["unit"] == un).to_numpy()
        k_u = kontur[idx_u].sum()
        in_disc = None
        for _i, t in ts.iterrows():
            nm = t["village"][2:].strip().lower()
            if t["village"] in TOWN_XY:
                cands = [Point(*TOWN_XY[t["village"]])]
            else:
                hit = gn[alt.apply(lambda a: nm in a) | (gn["Name"].str.lower() == nm)]
                cands = [Point(float(c.split(",")[1]), float(c.split(",")[0]))
                         for c in hit["Coordinates"]]
            cands = [p for p in cands if units.loc[un, "geometry"].contains(p)]
            if not cands:
                missing.append(f"{un} {t['village']} ({t['n']:,})")
                continue
            p = gpd.GeoSeries([cands[0]], crs=4326).to_crs(6933).iloc[0]
            r = max(3000.0, math.sqrt(t["n"] / 2000 / math.pi) * 1000)
            disc = idx_u & (cen.distance(p) < r).to_numpy()
            if in_disc is not None and (disc & in_disc).any():
                raise SystemExit(f"{un}: town discs overlap")
            k_t = kontur[disc].sum()
            if k_t <= 0:
                raise SystemExit(f"{un} {t['village']}: no Kontur people within {r:,.0f} m")
            w[disc] = kontur[disc] * (t["n"] / census[un]) * k_u / k_t
            done.append((un, t["village"], t["n"], (k_t / k_u) / (t["n"] / census[un])))
            in_disc = disc if in_disc is None else (in_disc | disc)
        if in_disc is None:
            continue
        rest = idx_u & ~in_disc
        k_rest = kontur[rest].sum()
        town_share = sum(n for u2, _v, n, _q in done if u2 == un) / census[un]
        if k_rest > 0:
            w[rest] = kontur[rest] * (1 - town_share) * k_u / k_rest
        if abs(w[idx_u].sum() - k_u) > 1e-6 * k_u + 1e-6:
            raise SystemExit(f"{un}: calibration changed the unit's weight total")
    qs = sorted(q for *_x, q in done)
    print(f"  TOWNS: {len(done)} census towns calibrated in their rayons (Kontur share / census "
          f"share before: min {qs[0]:.2f}, median {qs[len(qs) // 2]:.2f}, max {qs[-1]:.2f}); "
          f"{len(skipped)} one-city rayons left alone; not placed: {missing}")
    dens = w / (m.geometry.area.to_numpy() / 1e6)
    top = int(dens.argmax())
    print(f"  densest hex after calibration {dens[top]:,.0f} weight/km2 in {m['unit'].iloc[top]} "
          f"(Kontur's own densest {(kontur / (m.geometry.area.to_numpy() / 1e6)).max():,.0f})")
    if len(missing) > 2:
        raise SystemExit("more census towns than expected could not be placed")
    out = layer.copy()
    out["kontur"] = kontur
    out["pop"] = w
    return out


def within_oblast_control(layer, lut):
    """The join is made inside each oblast, so the null that tests it shuffles only there."""
    k = layer.groupby("unit")["pop"].sum()
    rows = [(r["parent"], r["pop"], float(k.get(r["unit"], 0.0))) for _i, r in lut.iterrows()]
    ratio = sum(x[2] for x in rows) / sum(x[1] for x in rows)

    def stats(rs):
        lc = [math.log(c) for _o, c, _k in rs]
        lk = [math.log(max(kk, 1.0)) for _o, _c, kk in rs]
        n = len(lc)
        ma, mb = sum(lc) / n, sum(lk) / n
        num = sum((x - ma) * (y - mb) for x, y in zip(lc, lk))
        r = num / math.sqrt(sum((x - ma) ** 2 for x in lc) * sum((y - mb) ** 2 for y in lk))
        out = sum(1 for _o, c, kk in rs if not (1 / 2 <= kk / c / ratio <= 2))
        return r, out

    r0, out0 = stats(rows)
    rng = random.Random(0)
    by = {}
    for x in rows:
        by.setdefault(x[0], []).append(x)
    rs, outs = [], []
    for _ in range(1000):
        sh = []
        for ob, xs in by.items():
            ks = [x[2] for x in xs]
            rng.shuffle(ks)
            sh += [(ob, x[1], kk) for x, kk in zip(xs, ks)]
        r, o = stats(sh)
        rs.append(r)
        outs.append(o)
    rs.sort()
    outs.sort()
    print(f"  WITHIN-OBLAST shuffle control: log r = {r0:.3f} against a best of {rs[-1]:.3f} "
          f"(median {rs[500]:.3f}) over 1,000 shuffles inside oblasts; units outside a factor "
          f"of 2: {out0} against a shuffled median of {outs[500]} (min {outs[0]})")
    if r0 <= rs[-1] or out0 >= outs[0]:
        raise SystemExit("the within-oblast shuffle does as well as the join")


if __name__ == "__main__":
    main()
