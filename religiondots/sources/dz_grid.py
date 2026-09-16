"""Algeria: the placement layer, Kontur 400 m population hexagons keyed to wilaya.

Writes data/geo/dz/dz_hexes.gpkg.

Algeria is 2.38 million km2 and about nine in ten of its people live on the northern tenth of it.
Tamanrasset, Illizi, Adrar and Tindouf are 1.49 million km2 between them and held 677,833 people
in 2008 (under 2%). Spread flat, a wilaya's dots would sit on the Sahara; with Kontur an empty hex
takes no dots.

THE JOIN IS ON HEX CENTROIDS, so a hex on a wilaya line belongs to one side and nobody is counted
twice. Hexes whose centroid is outside every wilaya (the `DZ` extract overruns into Morocco,
Tunisia, Libya, Niger, Mali and Western Sahara, and the coast) are dropped and reported.

Kontur is 2023 and the population table is the 2008 census, so the national ratio reads about
1.33. It is used only inside a wilaya. The per-wilaya ratio is printed because it shows how far
each wilaya's share of the country has moved since 2008, which the population table cannot.

## TWO PLACES WHERE KONTUR IS WRONG FOR THIS MAP, BOTH CHECKED AGAINST GEONAMES

`seat_check` sums Kontur people within 5 km of every wilaya seat GeoNames lists with 50,000 or more
people (feature codes PPLA and PPLC in GeoNames' `DZ` extract, CC BY 4.0) and divides by GeoNames'
population for the town.

  * **Béchar is a hole.** Kontur has 593 people within about 20 km of the town, and the wilaya
    reads 0.31 of its 2008 share; its heaviest hexes are Abadla, Beni Abbès and the Aïn Sefra
    road. Left alone, Béchar's 270,061 census people would be drawn on those. `KONTUR_HOLES`
    asserts Béchar is the only seat under `HOLE_RATIO`, and `fill_holes` adds a 3 km disc on
    GeoNames' point for the town carrying the wilaya's shortfall against the national ratio.
  * **Tindouf's heaviest hexes are the refugee camps**, 25 to 50 km south-east of the town. The
    2008 census counts 49,149 people in the wilaya and Kontur 109,299, 1.67 of its share. The
    census figure is matched by Kontur's hexes near the town (asserted in `town_only`), so only
    the hexes within `TOWN_ONLY` km of GeoNames' Tindouf are kept and the camps take no dots.

Usage:
    python sources/dz_grid.py --fetch    one gzipped gpkg from Kontur (14.7 MB); GeoNames DZ.zip
    python sources/dz_grid.py            rebuild from data/raw/dz/
"""

import gzip
import os
import shutil
import sys
import urllib.request
import zipfile

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "dz")
GEO = os.path.join(ROOT, "data", "geo", "dz")
WILAYAS = os.path.join(GEO, "dz_wilayas.gpkg")
OUT = os.path.join(GEO, "dz_hexes.gpkg")

GZ_URL = ("https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
          "kontur_population_DZ_20231101.gpkg.gz")
GZ_NAME = "kontur_population_DZ_20231101.gpkg.gz"
GPKG_NAME = "kontur_population_DZ_20231101.gpkg"
GEONAMES_URL = "https://download.geonames.org/export/dump/DZ.zip"
GEONAMES = os.path.join(RAW, "geonames_DZ.zip")

EXPECTED_WILAYAS = 48
KABYLIE = {"DZ06", "DZ10", "DZ15"}

# Kontur 2023 over the 2008 census. The expectation was written before the first run: ~1.36, and
# a weight only needs the shape, so the band is wide.
EXPECTED_RATIO = 1.36
KONTUR_TOLERANCE = 0.35

SEAT_MIN_POP = 50_000
HOLE_KM = 5.0
HOLE_RATIO = 0.10          # Kontur within HOLE_KM of a seat, over GeoNames' population
KONTUR_HOLES = {"DZ08"}    # Béchar; asserted
DISC_KM = 3.0
TOWN_ONLY = {"DZ37": 15.0}  # Tindouf: keep hexes within this many km of the seat
TOWN_ONLY_BAND = (0.5, 2.0)

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/128.0 Safari/537.36")

GEONAMES_COLS = ["geonameid", "name", "asciiname", "alternatenames", "lat", "lon", "fclass",
                 "fcode", "cc", "cc2", "admin1", "admin2", "admin3", "admin4", "population",
                 "elevation", "dem", "timezone", "modified"]


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    if not os.path.exists(GEONAMES):
        req = urllib.request.Request(GEONAMES_URL, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=300) as r:
            data = r.read()
        if data[:2] != b"PK":
            raise SystemExit(f"{GEONAMES_URL} is not a zip")
        with open(GEONAMES, "wb") as fh:
            fh.write(data)
    gz = os.path.join(RAW, GZ_NAME)
    gpkg = os.path.join(RAW, GPKG_NAME)
    if os.path.exists(gpkg) and os.path.getsize(gpkg) > 5_000_000:
        print("already have", gpkg)
        return
    if not os.path.exists(gz):
        print("GET", GZ_URL)
        r = requests.get(GZ_URL, timeout=1800, stream=True, headers={"User-Agent": UA})
        r.raise_for_status()
        with open(gz + ".part", "wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
        os.replace(gz + ".part", gz)
        print(f"  {os.path.getsize(gz):,} bytes")
    with gzip.open(gz, "rb") as src, open(gpkg + ".part", "wb") as dst:
        shutil.copyfileobj(src, dst)
    os.replace(gpkg + ".part", gpkg)
    with open(gpkg, "rb") as fh:
        magic = fh.read(16)
    if magic[:4] != b"SQLi":
        raise SystemExit(f"{gpkg} is not a GeoPackage; starts {magic!r}")
    print(f"  unpacked {os.path.getsize(gpkg):,} bytes")


def km(lat0, lon0, lat, lon):
    p = np.radians
    a = (np.sin(p(lat - lat0) / 2) ** 2
         + np.cos(p(lat0)) * np.cos(p(lat)) * np.sin(p(lon - lon0) / 2) ** 2)
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def seats(wil):
    """The largest GeoNames PPLA or PPLC inside each wilaya, as (unit -> row)."""
    import geopandas as gpd

    with zipfile.ZipFile(GEONAMES) as zf:
        t = pd.read_csv(zf.open("DZ.txt"), sep="\t", header=None, names=GEONAMES_COLS,
                        quoting=3, dtype=str, keep_default_na=False)
    s = t[t["fcode"].isin(["PPLA", "PPLC"])].copy()
    s["lat"], s["lon"] = s["lat"].astype(float), s["lon"].astype(float)
    s["population"] = pd.to_numeric(s["population"], errors="coerce").fillna(0).astype(int)
    pts = gpd.GeoDataFrame(s, geometry=gpd.points_from_xy(s["lon"], s["lat"]), crs=4326)
    j = gpd.sjoin(pts, wil[["unit", "geometry"]], how="inner", predicate="within")
    j = j.sort_values("population", ascending=False).drop_duplicates("unit")
    print(f"  GeoNames: {len(s)} PPLA/PPLC places, a seat found in {j['unit'].nunique()} of "
          f"{len(wil)} wilayas")
    return j.set_index("unit")


def seat_check(out, seat, names):
    """Kontur people within HOLE_KM of every seat of SEAT_MIN_POP or more, over GeoNames."""
    rows = []
    for u, r in seat[seat["population"] >= SEAT_MIN_POP].iterrows():
        h = out[out["unit"] == u]
        near = km(r["lat"], r["lon"], h["lat"].to_numpy(), h["lon"].to_numpy()) <= HOLE_KM
        rows.append((u, r["name"], int(r["population"]), float(h.loc[near, "pop"].sum())))
    t = pd.DataFrame(rows, columns=["unit", "seat", "geonames", "kontur"])
    t["ratio"] = t["kontur"] / t["geonames"]
    t = t.sort_values("ratio")
    print(f"\n  Kontur within {HOLE_KM:g} km of each wilaya seat of {SEAT_MIN_POP:,}+ (GeoNames), "
          "lowest five:")
    for _i, r in t.head(5).iterrows():
        print(f"      {names[r['unit']]:<18} {r['seat']:<14} GeoNames {r['geonames']:>9,}   "
              f"Kontur {r['kontur']:>9,.0f}   {r['ratio']:.2f}")
    holes = set(t.loc[t["ratio"] < HOLE_RATIO, "unit"])
    if holes != KONTUR_HOLES:
        raise SystemExit(f"wilaya seats with under {HOLE_RATIO:.0%} of their people in Kontur: "
                         f"{sorted(holes)}, not {sorted(KONTUR_HOLES)}")


def fill_holes(out, seat, census, ratio, names):
    import geopandas as gpd

    add = []
    for u in sorted(KONTUR_HOLES):
        have = float(out.loc[out["unit"] == u, "pop"].sum())
        short = census[u] * ratio - have
        if short <= 0:
            raise SystemExit(f"{names[u]} has no shortfall against the national ratio")
        r = seat.loc[u]
        pt = gpd.GeoSeries(gpd.points_from_xy([r["lon"]], [r["lat"]]), crs=4326)
        local = pt.to_crs(f"+proj=aeqd +lat_0={r['lat']} +lon_0={r['lon']} +units=m")
        disc = local.buffer(DISC_KM * 1000, 32).to_crs(4326).iloc[0]
        add.append({"unit": u, "pop": short, "geometry": disc, "lat": r["lat"], "lon": r["lon"]})
        print(f"  {names[u]}: Kontur holds {have:,.0f} against {census[u] * ratio:,.0f} at the "
              f"national ratio; a {DISC_KM:g} km disc on GeoNames' {r['name']} takes the "
              f"{short:,.0f}")
    return pd.concat([out, gpd.GeoDataFrame(add, crs=out.crs)], ignore_index=True)


def town_only(out, seat, census, ratio, names):
    for u, radius in TOWN_ONLY.items():
        r = seat.loc[u]
        m = out["unit"] == u
        d = km(r["lat"], r["lon"], out.loc[m, "lat"].to_numpy(), out.loc[m, "lon"].to_numpy())
        keep = pd.Series(d <= radius, index=out.index[m])
        kept = float(out.loc[keep.index[keep], "pop"].sum())
        dropped = float(out.loc[keep.index[~keep], "pop"].sum())
        rel = kept / (census[u] * ratio)
        print(f"  {names[u]}: {kept:,.0f} Kontur people within {radius:g} km of GeoNames' "
              f"{r['name']} ({rel:.2f} of the census at the national ratio), {dropped:,.0f} "
              "beyond it dropped")
        if not TOWN_ONLY_BAND[0] <= rel <= TOWN_ONLY_BAND[1]:
            raise SystemExit(f"{names[u]}'s hexes near the town do not match its census count")
        out = out.drop(index=keep.index[~keep])
    return out.reset_index(drop=True)


def main():
    import geopandas as gpd
    from geo_checks import read_layer

    gpkg = os.path.join(RAW, GPKG_NAME)
    if "--fetch" in sys.argv or not os.path.exists(gpkg) or not os.path.exists(GEONAMES):
        fetch()
    if not os.path.exists(WILAYAS):
        raise SystemExit(f"missing {WILAYAS}; run sources/dz_geo.py first")

    hexes = read_layer(gpkg, "Kontur DZ")
    popcol = next((c for c in hexes.columns if c.lower() == "population"), None)
    if popcol is None:
        raise SystemExit(f"no population column in {list(hexes.columns)}")
    print(f"Kontur hexes: {len(hexes):,}, crs={hexes.crs}, population {hexes[popcol].sum():,.0f}")

    wil = gpd.read_file(WILAYAS)
    if len(wil) != EXPECTED_WILAYAS:
        raise SystemExit(f"{WILAYAS} has {len(wil)} wilayas, expected {EXPECTED_WILAYAS}")

    cent = hexes.geometry.centroid
    pts = gpd.GeoDataFrame({popcol: hexes[popcol].to_numpy()}, geometry=cent,
                           crs=hexes.crs).to_crs(wil.crs)
    hexes = hexes.to_crs(wil.crs)
    joined = gpd.sjoin(pts, wil[["unit", "geometry"]], how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")].reindex(pts.index)

    outside = joined["unit"].isna()
    lost = float(pts.loc[outside, popcol].sum())
    print(f"\n  hexes whose centroid is outside every wilaya: {int(outside.sum()):,} "
          f"({lost:,.0f} people, {100.0 * lost / pts[popcol].sum():.3f}%); dropped")

    keep = ~outside
    out = gpd.GeoDataFrame({"unit": joined.loc[keep, "unit"].to_numpy(),
                            "pop": pts.loc[keep, popcol].to_numpy(dtype=float),
                            "lat": pts.loc[keep].geometry.y.to_numpy(),
                            "lon": pts.loc[keep].geometry.x.to_numpy()},
                           geometry=hexes.geometry[keep.to_numpy()].to_numpy(), crs=wil.crs)

    per = out.groupby("unit")["pop"].agg(["size", "sum"])
    missing = sorted(set(wil["unit"]) - set(per.index))
    if missing:
        raise SystemExit(f"wilayas with no populated hex: {missing}")
    if (per["sum"] <= 0).any():
        raise SystemExit(f"wilayas whose hexes sum to zero: {sorted(per.index[per['sum'] <= 0])}")
    print(f"  all {EXPECTED_WILAYAS} wilayas have hexes: {per['size'].min():,} to "
          f"{per['size'].max():,} each")

    census = dict(zip(wil["unit"], wil["pop"]))
    names = dict(zip(wil["unit"], wil["name"]))
    tot = float(out["pop"].sum())
    ratio = tot / sum(census.values())
    print(f"\n  Kontur {tot:,.0f} vs RGPH 2008 {sum(census.values()):,}: ratio {ratio:.3f} "
          f"(expected about {EXPECTED_RATIO})")
    if abs(ratio - EXPECTED_RATIO) > KONTUR_TOLERANCE:
        raise SystemExit("Kontur and the 2008 census disagree beyond the band; check the download")

    print("\n  per-wilaya Kontur 2023 / RGPH 2008, divided by the national ratio (1.00 = the "
          "wilaya kept its 2008 share):")
    rel = {u: (r["sum"] / census[u]) / ratio for u, r in per.iterrows()}
    for u in sorted(rel, key=rel.get):
        print(f"      {names[u]:<20} {int(per.loc[u, 'size']):>7,} hexes   {rel[u]:5.2f}")
    kab_c = sum(census[u] for u in KABYLIE) / sum(census.values())
    kab_k = float(per.loc[sorted(KABYLIE), "sum"].sum()) / tot
    print(f"\n  Kabylie (Tizi Ouzou, Béjaïa, Bouira): {kab_c:.2%} of Algeria in the 2008 census, "
          f"{kab_k:.2%} of Kontur 2023")

    seat = seats(wil)
    seat_check(out, seat, names)
    out = fill_holes(out, seat, census, ratio, names)
    out = town_only(out, seat, census, ratio, names)

    os.makedirs(GEO, exist_ok=True)
    out[["unit", "pop", "geometry"]].to_file(OUT, layer="hexes", driver="GPKG")
    print(f"\nwrote {OUT} ({len(out):,} cells)")


if __name__ == "__main__":
    main()
