"""Qatar -- the ten municipalities of 2004, rebuilt from COD-AB's zones, and the placement layer.

Writes:
    data/geo/qa/qa_units.gpkg     the 10 counted units, dissolved from zones (`unit`, census_pop)
    data/geo/qa/qa_zones.gpkg     COD-AB's zones with their 2004 municipality and 2004 population
    data/geo/qa/qa_hexes.gpkg     Kontur 400 m hexes cut to the zones: `unit`, `zone`, `pop` (`place`)
    data/geo/qa/qa_lookup.csv     zone -> 2004 municipality, both names, 2004 people, Kontur 2023

Usage:
    python sources/qa_geo.py --fetch    COD-AB shapefile zip (~24 MB) + Kontur QA (~0.6 MB)
    python sources/qa_geo.py            rebuild from data/raw/qa/

## THE UNITS ARE 2004's, REBUILT FROM COD-AB ZONES BY ZONE NUMBER

Qatar had ten municipalities in 2004 and has eight now: Al Ghuwairiya, Al Jemailya, Jeryan Al
Batna and Mesaieed were folded into their neighbours, and Al Daayen and Al Sheehaniya were created.
The zones underneath kept their numbers. The census's Table 2 lists the 87 zones of 2004 under
their municipality, with each zone's population (sources/qa.py `read_zones`), and COD-AB Qatar v02
(zones valid 30 November 2015, from the Planning and Statistics Authority) draws 91 zones whose
pcode ends in the zone number. So each 2004 municipality is the union of the COD-AB zones carrying
its 2004 zone numbers. Zone 57, the Industrial Area, is Al Rayyan's in 2004 and Doha's now; zones
69 and 70 are Doha's and Umm Salal's in 2004 and Al Daayen's now.

Two 2004 zones have no COD-AB polygon and six COD-AB zones had nobody in 2004; both lists are
asserted exactly (ZONES_MERGED, ZONES_EMPTY_2004), and the choices are in sources/qa.md §5.

## A WITNESS THE NUMBER DOES NOT DECIDE: THE ZONE NAMES, AND KONTUR BY ZONE

Table 2 prints a name for every zone and COD-AB prints its own. For each zone number in both, the
two names must share a word (fuzzy, after dropping words like `North` and `Al`), unless the zone
is in ZONES_RENAMED, which pins every one that does not with the reason. Then Kontur 2023 summed
per zone must rank the zones like Table 2's 2004 populations, above SPEARMAN_MIN and above every
one of SHUFFLES random reassignments of the populations to zones. Qatar has grown about fourfold
since, so this measures a pattern, not a level.

## THE PLACEMENT WEIGHT IS THE 2004 ZONE POPULATION, SPREAD INSIDE EACH ZONE BY KONTUR

Kontur's hexes are cut to the zones and their people shared by area (Doha's inner zones are under
one hex, spec §8.2e), then every zone's pieces are scaled to that zone's 2004 population. So the
weights inside a municipality are 2004's zone by zone, and Kontur 2023 only says where inside a
zone. A zone with people in 2004 and no Kontur piece is placed on its own polygon. Each unit's
weights are asserted to sum to its Table 1 population.
"""

import csv
import difflib
import glob
import gzip
import os
import re
import shutil
import sys
import unicodedata
import zipfile

os.environ.setdefault("OMP_NUM_THREADS", "6")          # [[feedback_cap_cpu]]
os.environ.setdefault("OPENBLAS_NUM_THREADS", "6")
os.environ.setdefault("MKL_NUM_THREADS", "6")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import qa                                                        # noqa: E402  the census tables
from geo_checks import read_layer                                # noqa: E402

RAW = os.path.join(ROOT, "data", "raw", "qa")
SHP_DIR = os.path.join(RAW, "codab_shp")
GEO = os.path.join(ROOT, "data", "geo", "qa")
KONTUR = os.path.join(ROOT, "data", "geo", "kontur")

ZIP = os.path.join(RAW, "qat_admin_boundaries.shp.zip")
ZIP_URL = ("https://data.humdata.org/dataset/6a84f3b8-41cd-4769-a61f-6dbd5f61bd05/resource/"
           "5e996f7c-42b5-4335-871c-3a78ffb158f7/download/qat_admin_boundaries.shp.zip")
GZ_URL = ("https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
          "kontur_population_QA_20231101.gpkg.gz")
GZ = os.path.join(KONTUR, "kontur_population_QA_20231101.gpkg.gz")
GPKG = GZ[:-3]

OUT_UNITS = os.path.join(GEO, "qa_units.gpkg")
OUT_ZONES = os.path.join(GEO, "qa_zones.gpkg")
OUT_HEXES = os.path.join(GEO, "qa_hexes.gpkg")
OUT_LOOKUP = os.path.join(GEO, "qa_lookup.csv")

COD_ZONES = 91
# 2004 zone -> the COD-AB zone that holds it now, same municipality, the name checked below.
ZONES_MERGED = {10: 20, 11: 21}          # Wadi Al Sail (East) and Al Rumeila (East)
MERGED_NAMES = {20: "Wadi Al Sail", 21: "Rumaila"}
# COD-AB zones with no row in Table 2, so nobody lived there in March 2004 (a zone of 9 people is
# listed), and the 2004 municipality each polygon is given. None carries a 2004 weight.
ZONES_EMPTY_2004 = {
    46: "Doha",        # Al Thumama, the same COD name as zone 47, which is Doha's in 2004
    49: "Doha",        # Hamad International Airport, Banana Island, Ras Bu Funtas: beside zone 48
    50: "Doha",        # Al Thumama, as 46
    58: "Al Rayyan",   # Wholesale Market, beside zone 57, which is Al Rayyan's in 2004
    98: "Al Wakra",    # Al Adaid, beside zone 95 Al Kharrara; Al Wakra's in COD-AB too
    99: "Al Shamal",   # 2.7 km2 with no name, inside Al Shamal in COD-AB
}
# Zone numbers whose Table 2 and COD-AB names share no word, each read on the first run
# (2026-09-15); the run asserts this set exactly. None is explained by a source here: they are
# pinned so that a changed file fails loudly, and the Kontur rank witness below is what says the
# numbers still point at the right polygons.
_BY_NUMBER_2004 = "2004 names it only `New District Of Doha` and its number"
ZONES_RENAMED = {
    2: "2004 `Al Diwan`, COD-AB `Al Bidda`: neighbouring names in central Doha",
    4: "2004 `Al Asmakh`, COD-AB `Mushaireb`: neighbouring names in central Doha",
    7: "2004 `New Markets`, COD-AB `Al Souq`: both name a market",
    19: "2004 `Doha Port`, COD-AB names it only `Zone 19`",
    24: "2004 `Al Muntazah`, COD-AB `Rawdat Al Khail`",
    31: "2004 `Al Duhail South`, COD-AB `Umm Lekhba`",
    41: "2004 `Al Hilal (West)`, COD-AB `Nuaija`; zone 42 is `Al Hilal` in both",
    45: "2004 `Al Matar Al Qadeem` is Arabic for COD-AB's `Old Airport`",
    47: "2004 `Al Rawda`, COD-AB `Al Thumama`",
    60: _BY_NUMBER_2004 + "; COD-AB `Zone 60`",
    61: "2004 `Diplomatic District`, COD-AB `Al Dafna - Al Qassar`",
    62: _BY_NUMBER_2004 + "; COD-AB `Zone 62`",
    63: _BY_NUMBER_2004 + "; COD-AB `Onaiza`",
    64: _BY_NUMBER_2004 + "; COD-AB `Lejbailat`",
    65: _BY_NUMBER_2004 + "; COD-AB `Onaiza`",
    66: _BY_NUMBER_2004 + "; COD-AB `Onaiza - Leqtaifiya - Al Qassar`",
    67: _BY_NUMBER_2004 + "; COD-AB `Hazm Al Markhiya`",
    68: "2004 `Qatar University`, COD-AB `Jelaiah - Al Tarfa - Jeryan Nejaima`",
    69: _BY_NUMBER_2004 + "; COD-AB `Jabal Thuaileb - Al Kharayej - Lusail - Al Egla - Wadi Al Banat`",
    81: "2004 `Abu Nakhla / Mukainess`, COD-AB `Mebaireek`",
}

STOP = {"zone", "north", "south", "east", "west", "new", "old", "district", "doha", "area",
        "fareeq", "fereej", "qadeem", "jadeed", "jadeeda", "town", "madinat", "city"}
TOKEN_RATIO = 0.75
SPEARMAN_MIN = 0.50
SHUFFLES = 1000
KONTUR_RATIO = (2.5, 5.0)         # Kontur 2023-11 over the 2004 census, nationally
KONTUR_OUTSIDE_MAX = 0.03         # Kontur people in no zone
EQ = "EPSG:6933"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) religiondots/1.0"}


def _get(url, dst, magic, min_size):
    import requests

    if os.path.exists(dst) and os.path.getsize(dst) > min_size:
        print(f"  have {os.path.basename(dst)} ({os.path.getsize(dst):,} bytes)")
        return
    print("GET", url)
    r = requests.get(url, timeout=1800, stream=True, headers=UA)
    r.raise_for_status()
    with open(dst + ".part", "wb") as fh:
        for chunk in r.iter_content(1 << 20):
            fh.write(chunk)
    with open(dst + ".part", "rb") as fh:
        head = fh.read(len(magic))
    if head != magic:
        raise SystemExit(f"{dst}: starts {head!r}, expected {magic!r}")
    os.replace(dst + ".part", dst)
    print(f"  got {os.path.getsize(dst):,} bytes")


def fetch():
    os.makedirs(RAW, exist_ok=True)
    os.makedirs(KONTUR, exist_ok=True)
    _get(ZIP_URL, ZIP, b"PK", 5_000_000)
    _get(GZ_URL, GZ, b"\x1f\x8b", 100_000)


def unpack():
    if not os.path.exists(ZIP):
        raise SystemExit(f"missing {ZIP}; run with --fetch")
    if not glob.glob(os.path.join(SHP_DIR, "**", "*.shp"), recursive=True):
        with zipfile.ZipFile(ZIP) as zf:
            zf.extractall(SHP_DIR)
    if not os.path.exists(GPKG):
        if not os.path.exists(GZ):
            raise SystemExit(f"missing {GZ}; run with --fetch")
        with gzip.open(GZ, "rb") as src, open(GPKG + ".part", "wb") as dst:
            shutil.copyfileobj(src, dst)
        os.replace(GPKG + ".part", GPKG)
    with open(GPKG, "rb") as fh:
        if fh.read(4) != b"SQLi":
            raise SystemExit(f"{GPKG} is not a GeoPackage")


def admin2_shp():
    shps = glob.glob(os.path.join(SHP_DIR, "**", "*.shp"), recursive=True)
    hits = [p for p in shps if re.search(r"adm(in)?2(?!.*_em)", os.path.basename(p).lower())]
    if len(hits) != 1:
        raise SystemExit(f"expected one admin 2 shapefile (not edge-matched) in {SHP_DIR}: "
                         f"{[os.path.basename(p) for p in shps]}")
    return hits[0]


def tokens(name):
    s = unicodedata.normalize("NFKD", str(name))
    s = "".join(c for c in s if not unicodedata.combining(c)).casefold()
    return {t for t in re.sub(r"[^a-z]+", " ", s).split() if len(t) >= 4 and t not in STOP}


def whole(name):
    """The name folded to its letters, every word kept, the zone number dropped."""
    s = unicodedata.normalize("NFKD", str(name))
    s = "".join(c for c in s if not unicodedata.combining(c)).casefold()
    return re.sub(r"[^a-z]+", "", s)


def names_agree(a, b):
    """True when two zone names share a word, or are the same name spelled nearly alike. The
    second test is for names made only of short or common words (`Umm Bab`, `Al Doha Al
    Jadeeda`) and for transliterations (`Al Shahhniya`, `Al Sheehaniya`)."""
    if difflib.SequenceMatcher(None, whole(a), whole(b)).ratio() >= TOKEN_RATIO:
        return True
    return any(difflib.SequenceMatcher(None, x, y).ratio() >= TOKEN_RATIO
               for x in tokens(a) for y in tokens(b))


def main():
    import geopandas as gpd
    import numpy as np
    import pandas as pd

    if "--fetch" in sys.argv:
        fetch()
    unpack()

    ok = True

    def say(good, msg):
        nonlocal ok
        ok &= bool(good)
        print(f"  {'OK ' if good else 'BAD'} {msg}")

    print("Qatar: 2004 municipalities from COD-AB zones, and placement\n")
    t1 = qa.read_t01()
    zones_2004, _, _ = qa.read_zones()

    # ---- 1. COD-AB zones
    cod = read_layer(admin2_shp(), "COD-AB QAT admin 2")
    say(len(cod) == COD_ZONES and not cod["adm2_pcode"].duplicated().any(),
        f"COD-AB admin 2 has {len(cod)} zones with distinct pcodes (expected {COD_ZONES}), "
        f"crs {cod.crs}")
    cod["zone"] = cod["adm2_pcode"].str[-3:].astype(int)
    tail = cod["adm2_name"].astype(str).str.extract(r"(\d+)\s*$")[0]
    wrong = cod[tail.isna() | (tail.fillna("-1").astype(int) != cod["zone"])]
    say(len(wrong) == 0 and not cod["zone"].duplicated().any(),
        "every COD-AB zone's name ends in the zone number its pcode carries"
        + (f"; not: {list(zip(wrong['adm2_pcode'], wrong['adm2_name']))[:6]}" if len(wrong) else ""))

    in_2004 = set(zones_2004)
    in_cod = set(cod["zone"])
    say(in_2004 - in_cod == set(ZONES_MERGED),
        f"the 2004 zones with no COD-AB polygon are exactly {sorted(ZONES_MERGED)}: "
        f"{sorted(in_2004 - in_cod)}")
    say(in_cod - in_2004 == set(ZONES_EMPTY_2004),
        f"the COD-AB zones with no 2004 row are exactly {sorted(ZONES_EMPTY_2004)}: "
        f"{sorted(in_cod - in_2004)}")
    cod_name = dict(zip(cod["zone"], cod["adm2_name"]))
    for old, new in ZONES_MERGED.items():
        say(zones_2004[old][0] == zones_2004[new][0]
            and str(cod_name.get(new, "")).startswith(MERGED_NAMES[new])
            and names_agree(zones_2004[old][1], cod_name.get(new, "")),
            f"2004 zone {old} {zones_2004[old][1]!r} folds into zone {new} "
            f"{cod_name.get(new)!r}, both in {zones_2004[old][0]}")
    if not ok:
        raise SystemExit("zone sets FAILED")

    # 2004 people per COD-AB zone, and the 2004 municipality of every COD-AB zone
    pop04 = {z: v[2][0] for z, v in zones_2004.items() if z not in ZONES_MERGED}
    for old, new in ZONES_MERGED.items():
        pop04[new] += zones_2004[old][2][0]
    unit_of = {z: v[0] for z, v in zones_2004.items() if z not in ZONES_MERGED}
    unit_of.update(ZONES_EMPTY_2004)
    cod["unit"] = cod["zone"].map(unit_of)
    cod["pop_2004"] = cod["zone"].map(pop04).fillna(0).astype("int64")
    cod["name_2004"] = cod["zone"].map(lambda z: zones_2004[z][1] if z in zones_2004 else "")
    per_unit = cod.groupby("unit")["pop_2004"].sum()
    say(all(per_unit.get(u, -1) == t1[u][0] for u in qa.UNITS),
        "COD-AB zones carrying Table 2's populations sum to Table 1 in all ten municipalities")

    # ---- 2. witness: the names
    disagree = sorted(z for z in in_2004 & in_cod if not names_agree(zones_2004[z][1], cod_name[z]))
    for z in disagree:
        print(f"      zone {z:>2}: 2004 {zones_2004[z][1]!r}  COD-AB {cod_name[z]!r}"
              f"{'' if z in ZONES_RENAMED else '   <-- not pinned'}")
    agree = len(in_2004 & in_cod) - len(disagree)
    say(set(disagree) == set(ZONES_RENAMED),
        f"{agree} of {len(in_2004 & in_cod)} shared zone numbers carry names sharing a word; "
        f"the {len(disagree)} that do not are exactly the pinned ZONES_RENAMED")

    # ---- 3. Kontur, cut to the zones
    hexes = read_layer(GPKG, "Kontur QA")
    popcol = next((c for c in hexes.columns if c.lower() == "population"), None)
    if popcol is None:
        raise SystemExit(f"no population column in {list(hexes.columns)}")
    hexes = hexes[hexes[popcol] > 0].to_crs(EQ).reset_index(drop=True)
    hexes["hex_km2"] = hexes.area / 1e6
    k_all = float(hexes[popcol].sum())
    z_eq = cod.to_crs(EQ)
    zkm2 = z_eq.set_index("zone").area / 1e6
    print(f"\n  Kontur QA: {len(hexes):,} populated hexes, {k_all:,.0f} people; zones median "
          f"{zkm2.median():.2f} km2 ({zkm2.median() / 0.74:.1f} hexes), "
          f"{int((zkm2 < 0.74).sum())} smaller than one hex")
    pieces = gpd.overlay(hexes[[popcol, "hex_km2", "geometry"]], z_eq[["zone", "unit", "geometry"]],
                         how="intersection", keep_geom_type=True)
    pieces["kontur"] = pieces[popcol] * (pieces.area / 1e6) / pieces["hex_km2"]
    pieces = pieces[pieces["kontur"] > 0].reset_index(drop=True)
    kept = float(pieces["kontur"].sum())
    say(1 - kept / k_all <= KONTUR_OUTSIDE_MAX,
        f"{kept:,.0f} of Kontur's {k_all:,.0f} people fall inside a zone "
        f"({100 * (1 - kept / k_all):.2f}% outside, bar {100 * KONTUR_OUTSIDE_MAX:.0f}%)")
    ratio = kept / qa.TOTAL
    say(KONTUR_RATIO[0] <= ratio <= KONTUR_RATIO[1],
        f"Kontur 2023 inside the zones is {ratio:.2f}x the 2004 census, inside {KONTUR_RATIO}")

    kz = pieces.groupby("zone")["kontur"].sum()
    cod["kontur_2023"] = cod["zone"].map(kz).fillna(0.0)

    # ---- 4. witness: Kontur ranks the zones like 2004 does
    lived = cod[cod["pop_2004"] > 0]
    rk = lived["pop_2004"].rank().to_numpy()
    kk = lived["kontur_2023"].rank().to_numpy()
    rho = float(np.corrcoef(rk, kk)[0, 1])
    rng = np.random.default_rng(2004)
    null = np.array([np.corrcoef(rng.permutation(rk), kk)[0, 1] for _ in range(SHUFFLES)])
    say(rho >= SPEARMAN_MIN and rho > null.max(),
        f"Spearman of Kontur 2023 against Table 2 over {len(lived)} zones is {rho:.3f} "
        f"(bar {SPEARMAN_MIN}); {SHUFFLES} shuffles reach at most {null.max():.3f}")

    print("\n  per 2004 municipality, Kontur 2023 share against the census 2004 share:")
    ku = cod.groupby("unit")["kontur_2023"].sum()
    for u in qa.UNITS:
        s04, s23 = t1[u][0] / qa.TOTAL, ku[u] / kept
        print(f"    {u:<16}{t1[u][0]:>9,}  {100 * s04:5.1f}%   Kontur {ku[u]:>11,.0f}  "
              f"{100 * s23:5.1f}%   {s23 / s04:5.2f}x")

    # ---- 5. the placement layer: 2004 zone people, spread by Kontur inside each zone
    pieces["pop"] = pieces["zone"].map(pop04).fillna(0) * pieces["kontur"] / pieces["zone"].map(kz)
    place = pieces[pieces["pop"] > 0][["unit", "zone", "pop", "geometry"]]
    no_piece = sorted(z for z, p in pop04.items() if p > 0 and z not in set(kz.index))
    if no_piece:
        fill = z_eq[z_eq["zone"].isin(no_piece)][["unit", "zone", "geometry"]].copy()
        fill["pop"] = fill["zone"].map(pop04).astype(float)
        place = pd.concat([place, fill[["unit", "zone", "pop", "geometry"]]], ignore_index=True)
    print(f"\n  placement layer: {len(place):,} pieces; zones with 2004 people and no Kontur "
          f"piece, placed on their polygon: "
          + (", ".join(f"{z} {cod_name[z]!r} ({pop04[z]:,})" for z in no_piece) or "none"))
    wsum = place.groupby("unit")["pop"].sum()
    say(all(abs(wsum.get(u, 0) - t1[u][0]) < 0.5 for u in qa.UNITS),
        "every municipality's placement weights sum to its Table 1 population")
    npieces = place.groupby("unit").size()
    print("  pieces per municipality: " + ", ".join(f"{u} {npieces.get(u, 0):,}" for u in qa.UNITS))

    place = gpd.GeoDataFrame(place, geometry="geometry", crs=EQ).to_crs("EPSG:4326")
    w, s, e, n = place.total_bounds
    say(50.6 < w < e < 51.8 and 24.4 < s < n < 26.3, f"bbox {w:.3f} {s:.3f} {e:.3f} {n:.3f} is Qatar")
    if not ok:
        raise SystemExit("placement checks FAILED")

    # ---- write
    os.makedirs(GEO, exist_ok=True)
    zones_out = cod[["zone", "adm2_pcode", "adm2_name", "name_2004", "unit", "pop_2004",
                     "kontur_2023", "geometry"]].to_crs("EPSG:4326")
    units = zones_out.dissolve(by="unit", as_index=False)[["unit", "geometry"]]
    units["census_pop"] = units["unit"].map(lambda u: t1[u][0]).astype("int64")
    units["area_km2"] = units.to_crs(EQ).area / 1e6
    print("\n  2004 municipalities rebuilt: " + ", ".join(
        f"{r.unit} {r.area_km2:,.0f} km2" for r in units.itertuples()))
    for gdf, dst, layer in ((units, OUT_UNITS, "qa_units"), (zones_out, OUT_ZONES, "qa_zones"),
                            (place, OUT_HEXES, "qa_hexes")):
        gdf.to_file(dst + ".part.gpkg", driver="GPKG", layer=layer)
        os.replace(dst + ".part.gpkg", dst)
    with open(OUT_LOOKUP, "w", encoding="utf-8", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["zone", "adm2_pcode", "unit_2004", "name_2004", "name_codab", "pop_2004",
                     "kontur_2023"])
        for r in zones_out.sort_values("zone").itertuples():
            wr.writerow([r.zone, r.adm2_pcode, r.unit, r.name_2004, r.adm2_name, r.pop_2004,
                         round(r.kontur_2023)])
    print(f"\nwrote {OUT_UNITS}\nwrote {OUT_ZONES}\nwrote {OUT_HEXES} ({len(place):,} pieces)\n"
          f"wrote {OUT_LOOKUP}")


if __name__ == "__main__":
    main()
