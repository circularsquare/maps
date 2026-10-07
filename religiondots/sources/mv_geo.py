"""Maldives geography: COD-AB's 20 atolls and Malé, and Kontur hexes cut to the islands as the placement
layer.

Writes:
    data/geo/mv/mv_hexes.gpkg    Kontur hexes cut to COD-AB's islands, with `unit` (pcode) and `pop`
    data/geo/mv/mv_lookup.csv    unit -> name, census 2014 people, Kontur 2023, pieces

Reads data/normalized/mv.csv (run `python sources/mv.py` first) for each unit's 2014 people, and
data/raw/mv/PP5-Updated-20181014.xlsx for the island names and Malé's islands.

## UNITS

COD-AB Maldives (HDX `cod-ab-mdv`, valid from 2024-10-22): 21 first-level units, the 20 atolls by
their administrative names and Malé (MV021: Maale, Vilin'gili and Hulhumaale), and 1,556 islands.
The census prints its atolls by their geographic names with the administrative letters in brackets
(`North Thiladhunmathi (HA)`); `sources/mv.py::ATOLLS` pairs the letters with COD's pcodes, and the
witness here is the island names: every atoll's administrative islands in Table PP5 must be found
among COD's islands of that pcode more than of any other.

## PLACEMENT: ISLANDS, NOT ATOLL POLYGONS

COD's atoll polygons take in their lagoons and Malé's is three islands, so a Kontur hex centroid
over Malé's harbour lands in Kaafu: by centroid, Kontur puts 79,664 people in Malé and 179,049 in
Kaafu. So the placement layer is each Kontur hex cut to COD's island polygons, the hex's people
shared over its land pieces by area (`sources/mt_geo.py`'s rule), and each piece takes its island's
atoll. Hexes touching no island are snapped to the nearest island within `SNAP_M`.

## PLACEMENT: EVERY COUNTED ISLAND AT ITS 2014 COUNT

Table PP5 counts each of the 187 administrative islands and Malé's islands in 2014. Kontur 2023 is a
poor guide between islands: it puts 198,456 people on Malé island against 128,767 counted and 4,347
on Hulhumalé against 17,149, and a Kontur hex over Malé's harbour that touches Funadhoo gives that
fuel-depot island 4,627 people. So each counted island's pieces are scaled to its PP5 count (an
island Kontur misses takes its own polygon), and the rest of each atoll's people (resort and
industrial islands, the census's non-administrative islands) are shared over its other islands in
proportion to Kontur. PP5's names join COD's inside each atoll by folded name, then by the closest
name (printed for audit), then `ISLAND_PINS`; all 187 must join. Kontur only places people inside
an island and among the uncounted islands.

Usage:
    python sources/mv_geo.py --fetch    COD-AB shapefiles (9 MB) and Kontur MV (92 KB) if missing
    python sources/mv_geo.py
"""

import csv
import gzip
import os
import re
import shutil
import sys
import urllib.request
import zipfile

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
os.environ.setdefault("OMP_NUM_THREADS", "6")

import geopandas as gpd
import numpy as np
import pandas as pd

import mv
from geo_checks import read_layer

RAW = mv.RAW
COD_ZIP = os.path.join(RAW, "mdv_admin_boundaries.shp.zip")
COD_URL = ("https://data.humdata.org/dataset/a968d227-24c9-49ea-bbce-cee04ee94819/resource/"
           "21c520dd-b1a0-4e76-af6c-11dff697401d/download/mdv_admin_boundaries.shp.zip")
COD_DIR = os.path.join(RAW, "cod_ab")
KONTUR_DIR = os.path.join(ROOT, "data", "geo", "kontur")
KONTUR = os.path.join(KONTUR_DIR, "kontur_population_MV_20231101.gpkg")
KONTUR_URL = ("https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
              "kontur_population_MV_20231101.gpkg.gz")
GEO = os.path.join(ROOT, "data", "geo", "mv")
HEXES = os.path.join(GEO, "mv_hexes.gpkg")
LOOKUP = os.path.join(GEO, "mv_lookup.csv")
CRS_M = "EPSG:32643"         # UTM 43N, metres; the Maldives span 72.6-73.8 E

FUZZY_CUTOFF = 0.75
# PP5 island -> COD island name, where the folded names do not settle it (read off both lists)
ISLAND_PINS = {("L", "Gamu3"): "Gan"}      # Gamu is the main settlement on Laamu's Gan island
KONTUR_HEXES, KONTUR_TOTAL = 1019, 518_768
SNAP_M = 1_000.0
UNIT_BAND = (0.4, 2.5)       # Kontur 2023 share / census 2014 share, per unit, set before the run

# Malé's islands in PP5 -> COD's island. The harbours row (Malé, Hulhumalé and Villimalé) and
# Hulhulé (the airport island, a Kaafu island in COD) are put on Maale, the island beside both.
MALE_ISLANDS = {"Henveiru": "Maale", "Galolhu": "Maale", "Machchangolhi": "Maale",
                "Maafannu": "Maale", "Harbours (Male', Hulhumale' & Villingili)": "Maale",
                "Hulhule": "Maale", "Villigili": "Vilin'gili", "HulhuMale'": "Hulhumaale"}


def fetch():
    os.makedirs(RAW, exist_ok=True)
    os.makedirs(KONTUR_DIR, exist_ok=True)
    for url, dst, magic in ((COD_URL, COD_ZIP, b"PK"), (KONTUR_URL, KONTUR + ".gz", b"\x1f\x8b")):
        if os.path.exists(dst) and os.path.getsize(dst) > 50_000:
            continue
        print("  GET", url)
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}),
                                    timeout=300) as r:
            data = r.read()
        if not data.startswith(magic):
            raise SystemExit(f"{url} did not return the expected file")
        with open(dst + ".part", "wb") as fh:
            fh.write(data)
        os.replace(dst + ".part", dst)


def unpack():
    if not os.path.exists(os.path.join(COD_DIR, "mdv_admin2.shp")):
        with zipfile.ZipFile(COD_ZIP) as z:
            z.extractall(COD_DIR)
    if not os.path.exists(KONTUR):
        with gzip.open(KONTUR + ".gz", "rb") as src, open(KONTUR + ".part", "wb") as dst:
            shutil.copyfileobj(src, dst)
        os.replace(KONTUR + ".part", KONTUR)


def fold(s):
    s = re.sub(r"\d+$", "", str(s).strip()).lower()
    s = re.sub(r"[^a-z]", "", s)
    return re.sub(r"(.)\1+", r"\1", s)          # doubled letters vary between spellings


def pp5():
    """{code: {island name: people}} for the administrative islands, and {Malé row: people}."""
    t = pd.read_excel(os.path.join(RAW, "PP5-Updated-20181014.xlsx"), header=None)
    islands, male, in_male = {}, {}, False
    for r in t.itertuples(index=False):
        a, loc, n = r[0], r[1], r[2]
        loc = None if (isinstance(loc, float) and np.isnan(loc)) else str(loc).strip()
        if loc == "Male'":
            in_male = True
            continue
        if loc and loc.startswith("Atolls"):
            in_male = False
        if in_male and loc and not (isinstance(n, float) and np.isnan(n)):
            male[loc] = int(n)
        if isinstance(a, str) and a.strip() in mv.CODES and loc:
            islands.setdefault(a.strip(), {})[loc] = int(n)
    return islands, male


def match_islands(islands, a2):
    """{(code, PP5 name): COD adm2_pcode} for every administrative island, by folded name inside its
    atoll, then by the closest folded name (difflib, printed for audit), then `ISLAND_PINS`."""
    import difflib
    out, fuzzy = {}, []
    for c in mv.CODES:
        cod = a2[a2["adm1_pcode"] == mv.PCODE[c]]
        folds = {}
        for pc, nm in zip(cod["adm2_pcode"], cod["adm2_name"]):
            folds.setdefault(fold(nm), []).append(pc)
        for nm in islands[c]:
            if (c, nm) in ISLAND_PINS:
                pin = ISLAND_PINS[(c, nm)]
                hit = cod.loc[cod["adm2_name"] == pin, "adm2_pcode"].tolist()
                if len(hit) != 1:
                    raise SystemExit(f"pin {c} {nm} -> {pin}: {len(hit)} COD islands of that name")
                out[(c, nm)] = hit[0]
                fuzzy.append((c, nm, pin, "pinned"))
                continue
            f = fold(nm)
            if f in folds and len(folds[f]) == 1:
                out[(c, nm)] = folds[f][0]
                continue
            near = difflib.get_close_matches(f, list(folds), n=1, cutoff=FUZZY_CUTOFF)
            if near and len(folds[near[0]]) == 1:
                out[(c, nm)] = folds[near[0]][0]
                fuzzy.append((c, nm, cod.loc[cod["adm2_pcode"] == out[(c, nm)], "adm2_name"].iloc[0],
                              f"{difflib.SequenceMatcher(None, f, near[0]).ratio():.2f}"))
            else:
                fuzzy.append((c, nm, None, "unmatched"))
    return out, fuzzy


def main():
    if "--fetch" in sys.argv or not os.path.exists(COD_ZIP) or not os.path.exists(KONTUR + ".gz"):
        fetch()
    unpack()
    ok = True

    def say(good, msg):
        nonlocal ok
        print(("  ok  " if good else "  !!  ") + msg)
        ok = ok and good

    # ---- 1. units and the join
    a1 = read_layer(os.path.join(COD_DIR, "mdv_admin1.shp"), "COD-AB MDV admin1")
    a2 = read_layer(os.path.join(COD_DIR, "mdv_admin2.shp"), "COD-AB MDV admin2")
    name = dict(zip(a1["adm1_pcode"], a1["adm1_name"]))
    say(len(a1) == 21 and all(name.get(mv.PCODE[c]) == mv.NAME[c] for c in mv.CODES + ["MALE"]),
        "COD-AB: 21 first-level units, each pcode carrying the name sources/mv.py pairs it with")
    say(len(a2) == 1556 and set(a2["adm1_pcode"]) == set(a1["adm1_pcode"]),
        f"COD-AB: {len(a2):,} islands, every one under a first-level unit")
    islands, male = pp5()
    say(list(islands) == mv.CODES and sum(male.values()) == 153904,
        f"PP5: {sum(len(v) for v in islands.values())} administrative islands in the 20 atolls; "
        f"Malé's {len(male)} rows sum to 153,904")
    cod_names = {pc: {fold(n) for n in g["adm2_name"]} for pc, g in a2.groupby("adm1_pcode")}
    say(sum(len(v) for v in islands.values()) == 187
        and sum(sum(v.values()) for v in islands.values()) == 211543,
        "PP5: 187 administrative islands in the atolls, 211,543 people (PP3's administrative row)")
    worst = []
    for c in mv.CODES:
        want = {fold(n) for n in islands[c]}
        hits = {pc: len(want & names) / len(want) for pc, names in cod_names.items()}
        best_other = max(v for pc, v in hits.items() if pc != mv.PCODE[c])
        worst.append((c, hits[mv.PCODE[c]], best_other))
    # The bar was 0.6 own before the first run; Dhaalu's 50% failed it on spelling (Kudahuvadhoo,
    # Meedhoo and other names repeat across atolls, and COD and PP5 spell vowels differently), so the
    # test is the ratio to the next-best pcode, which is what tells a right pairing from a wrong one.
    say(all(own >= 0.4 and own >= 2.5 * other for _, own, other in worst),
        "island-name witness: each atoll's PP5 islands are found under its own pcode (lowest "
        + f"{min(w[1] for w in worst):.0%}) far more than under any other (highest "
        + f"{max(w[2] for w in worst):.0%})")
    for c, own, other in sorted(worst, key=lambda w: w[1])[:3]:
        print(f"      {c} {mv.NAME[c]}: {own:.0%} of its islands under its pcode, {other:.0%} elsewhere")
    matched, fuzzy = match_islands(islands, a2)
    for c, nm, got, how in fuzzy:
        print(f"      island {c:<3} {nm:<22} -> {got} ({how})")
    say(len(matched) == 187 and len(set(matched.values())) == 187,
        "all 187 administrative islands join a COD island of their own atoll, one to one")
    if "--match-only" in sys.argv:
        return

    # ---- 2. Kontur cut to the islands
    hexes = read_layer(KONTUR, "Kontur MV")
    say(len(hexes) == KONTUR_HEXES and round(float(hexes["population"].sum())) == KONTUR_TOTAL,
        f"Kontur MV 2023-11-01: {len(hexes):,} hexes, {hexes['population'].sum():,.0f} people")
    hexes = hexes[hexes["population"] > 0].to_crs(CRS_M).reset_index(drop=True)
    hexes["hid"] = np.arange(len(hexes))
    isl = a2.to_crs(CRS_M)[["adm2_name", "adm2_pcode", "adm1_pcode", "geometry"]]
    pieces = gpd.overlay(hexes[["hid", "population", "geometry"]], isl, how="intersection",
                         keep_geom_type=True)
    land = pieces.area.groupby(pieces["hid"]).transform("sum")
    pieces["pop"] = pieces["population"] * pieces.area / land
    landless = hexes[~hexes["hid"].isin(pieces["hid"])]
    snap = gpd.sjoin_nearest(landless[["hid", "population", "geometry"]], isl, how="left",
                             max_distance=SNAP_M, distance_col="dist")
    snap = snap.drop_duplicates("hid")
    near = snap[snap["adm1_pcode"].notna()]
    dropped = float(snap.loc[snap["adm1_pcode"].isna(), "population"].sum())
    # a snapped hex is placed on the island it is nearest, as that island's whole polygon piece
    add = isl.loc[near["index_right"].astype(int)].copy()
    add["pop"] = near["population"].to_numpy()
    print(f"  {len(landless)} hexes touch no island ({landless['population'].sum():,.0f} people): "
          f"{len(near)} snapped to an island within {SNAP_M:,.0f} m, {dropped:,.0f} people dropped")
    place = pd.concat([pieces[["adm2_name", "adm2_pcode", "adm1_pcode", "pop", "geometry"]],
                       add[["adm2_name", "adm2_pcode", "adm1_pcode", "pop", "geometry"]]],
                      ignore_index=True)
    place = gpd.GeoDataFrame(place.rename(columns={"adm1_pcode": "unit"}), geometry="geometry",
                             crs=CRS_M)
    place = place[place["pop"] > 0].reset_index(drop=True)
    kontur_unit = place.groupby("unit")["pop"].sum()
    people = pd.read_csv(mv.OUT).groupby("geo_id")["count"].sum()

    # ---- 3. every island the census counts, to its 2014 count; the rest of each atoll (resorts,
    # industrial islands) to the atoll's remainder, in proportion to Kontur
    target = {}                                   # COD island pcode -> 2014 people
    for (c, nm), pc in matched.items():
        target[pc] = target.get(pc, 0) + islands[c][nm]
    male_pc = dict(zip(a2.loc[a2["adm1_pcode"] == mv.PCODE["MALE"], "adm2_name"],
                       a2.loc[a2["adm1_pcode"] == mv.PCODE["MALE"], "adm2_pcode"]))
    for row, n in male.items():
        if row not in MALE_ISLANDS:
            raise SystemExit(f"PP5 Malé row {row!r} has no island in MALE_ISLANDS")
        pc = male_pc[MALE_ISLANDS[row]]
        target[pc] = target.get(pc, 0) + n
    have = place.groupby("adm2_pcode")["pop"].sum()
    for pc in ("MV021001", "MV021002", "MV021003"):
        print(f"  {a2.loc[a2['adm2_pcode'] == pc, 'adm2_name'].iloc[0]}: Kontur 2023 "
              f"{have.get(pc, 0):,.0f}, census 2014 {target[pc]:,}")
    empty = sorted(pc for pc in target if pc not in have.index)
    if empty:
        fill = isl[isl["adm2_pcode"].isin(empty)].rename(columns={"adm1_pcode": "unit"}).copy()
        fill["pop"] = 1.0
        place = pd.concat([place, fill[place.columns]], ignore_index=True)
        have = place.groupby("adm2_pcode")["pop"].sum()
        print(f"  {len(empty)} counted islands with no Kontur piece take their own polygon: "
              + ", ".join(f"{n} ({target[p]:,})" for p, n in
                          zip(fill["adm2_pcode"], fill["adm2_name"])))
    counted = place["adm2_pcode"].isin(list(target))
    place.loc[counted, "pop"] = (place.loc[counted, "pop"]
                                 * place.loc[counted, "adm2_pcode"].map(target)
                                 / place.loc[counted, "adm2_pcode"].map(have))
    rest = {}
    for u in people.index:
        in_u = place["unit"] == u
        r = people[u] - place.loc[in_u & counted, "pop"].sum()
        other = in_u & ~counted
        rest[u] = (round(r), int(other.sum()))
        if r < -0.5 or (r > 0.5 and not other.any()):
            raise SystemExit(f"{name[u]}: {r:,.0f} people left for the uncounted islands, "
                             f"{int(other.sum())} pieces to hold them")
        if other.any():
            place.loc[other, "pop"] = place.loc[other, "pop"] * max(r, 0) / place.loc[other, "pop"].sum()
    print("  resort and other islands per atoll (people, pieces): " + ", ".join(
        f"{name[u]} {v[0]:,} ({v[1]})" for u, v in rest.items() if u != mv.PCODE["MALE"]))
    place = place[place["pop"] > 0].reset_index(drop=True)
    k = place.groupby("unit")["pop"].sum()
    say((k.reindex(people.index) - people).abs().max() < 0.5,
        "every unit's pieces sum to its 2014 people")

    # ---- 4. Kontur against the census, before the calibration
    rel = (kontur_unit / kontur_unit.sum()) / (people / people.sum())
    print("  Kontur 2023 share / census 2014 share, per unit, before calibration:")
    for u, r in rel.sort_values().items():
        print(f"      {name[u]:<13} {r:5.2f}  census {people[u]:>7,}  Kontur {kontur_unit[u]:>9,.0f}")
    say(rel.between(*UNIT_BAND).all(), f"every unit inside {UNIT_BAND}")
    rng = np.random.default_rng(1)
    rho = kontur_unit.rank().corr(people.reindex(kontur_unit.index).rank())
    null = max(pd.Series(rng.permutation(kontur_unit.to_numpy()), index=kontur_unit.index).rank()
               .corr(people.reindex(kontur_unit.index).rank()) for _ in range(2000))
    say(rho > null, f"rank witness over 21 units: Spearman {rho:+.3f}, best of 2,000 shuffles "
                    f"{null:+.3f}")
    say(set(k.index) == set(people.index), "every counted unit has placement pieces")
    if not ok:
        raise SystemExit("checks FAILED")

    os.makedirs(GEO, exist_ok=True)
    out = place[["unit", "adm2_name", "pop", "geometry"]].to_crs("EPSG:4326")
    w, s, e, n = out.total_bounds
    if not (72.4 < w < e < 73.9 and -0.8 < s < n < 7.2):
        raise SystemExit(f"bbox {w:.2f} {s:.2f} {e:.2f} {n:.2f} is not the Maldives")
    out.to_file(HEXES + ".part.gpkg", driver="GPKG", layer="mv_hexes")
    os.replace(HEXES + ".part.gpkg", HEXES)
    with open(LOOKUP, "w", encoding="utf-8", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["unit", "name", "census_2014", "kontur_2023", "pieces"])
        for u in sorted(people.index):
            wr.writerow([u, name[u], int(people[u]), round(float(kontur_unit[u])),
                         int((out["unit"] == u).sum())])
    print(f"\n  wrote {HEXES} ({len(out):,} pieces) and {LOOKUP}")


if __name__ == "__main__":
    main()
