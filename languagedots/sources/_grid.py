"""A placement layer for units religiondots has no layer for: Kontur 400 m hexes keyed to units.

    from _grid import hex_layer
    hex_layer("zm", units, census={"ZM1001": 23511, ...})   # -> data/geo/zm/zm_hexes.gpkg

`units` is a GeoDataFrame with a `unit` column (your counts' unit ids). Each Kontur hex goes to the
unit its CENTROID falls in (no hex split between two units), with `pop` = Kontur's population.
The file name ends `_hexes.gpkg` on purpose: religiondots' kontur_cap check recognises Kontur
layers by that suffix and runs on it at scatter time (rdlink.py; new cap blocks are registered in
languagedots/kontur_cap.csv, never religiondots').

WHERE THE KONTUR FILE COMES FROM, read-only: religiondots/data/geo/kontur/ has most countries'
extracts. A .gpkg there is read in place; a .gz only there is unpacked into languagedots'
data/geo/kontur/; a country neither has is downloaded into languagedots'. Nothing is written into
religiondots.

THE CHECKS, from religiondots/playbooks/geography.md "Placement", which every agent building
geography reads: people in hexes outside every unit (a border overlap or a boundary that stops
short), every unit holding at least one populated hex, Kontur against the census per unit
(normalised by the national ratio, printed at both ends) and a shuffled-join control on the log
correlation, which is what says the join carries information. A band failure is often Kontur's
fault, not the join's (the playbook's examples); this helper reports, it does not decide.
"""
import gzip
import math
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RD_KONTUR = ROOT.parent / "religiondots" / "data" / "geo" / "kontur"
OUR_KONTUR = ROOT / "data" / "geo" / "kontur"
URL = ("https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
       "kontur_population_{CC}_20231101.gpkg.gz")


def kontur_path(cc):
    CC = cc.upper()
    name = f"kontur_population_{CC}_20231101.gpkg"
    for d in (RD_KONTUR, OUR_KONTUR):
        p = d / name
        if p.exists() and p.stat().st_size > 10_000:
            return p
    OUR_KONTUR.mkdir(parents=True, exist_ok=True)
    gz = RD_KONTUR / (name + ".gz")
    if not gz.exists():
        gz = OUR_KONTUR / (name + ".gz")
        if not gz.exists():
            import requests
            r = requests.get(URL.format(CC=CC), timeout=3600, stream=True,
                             headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            with open(gz, "wb") as fh:
                for chunk in r.iter_content(1 << 20):
                    fh.write(chunk)
    out = OUR_KONTUR / name
    with gzip.open(gz, "rb") as src, open(out, "wb") as dst:
        shutil.copyfileobj(src, dst)
    with open(out, "rb") as fh:
        if fh.read(4) != b"SQLi":
            raise SystemExit(f"{out} is not a GeoPackage")
    return out


def hex_layer(cc, units, census=None, out=None, band=3.0):
    import geopandas as gpd

    hexes = gpd.read_file(kontur_path(cc))
    if len(hexes) == 0:
        raise SystemExit(f"Kontur {cc.upper()} extract has ZERO features")
    popcol = next(c for c in hexes.columns if c.lower() == "population")
    # centroid in Kontur's own CRS (3857), then reproject the points: reprojecting first moves it
    pts = gpd.GeoDataFrame({"pop": hexes[popcol].to_numpy(dtype=float)},
                           geometry=hexes.geometry.centroid, crs=hexes.crs).to_crs(units.crs)
    j = gpd.sjoin(pts, units[["unit", "geometry"]], how="left", predicate="within")
    j = j[~j.index.duplicated(keep="first")].reindex(pts.index)
    outside = j["unit"].isna()
    print(f"  Kontur {cc.upper()}: {len(hexes):,} hexes, {pts['pop'].sum():,.0f} people; "
          f"{int(outside.sum()):,} hexes ({pts.loc[outside, 'pop'].sum():,.0f} people) outside every unit")
    keep = ~outside
    layer = gpd.GeoDataFrame({"unit": j.loc[keep, "unit"].astype(str).to_numpy(),
                              "pop": pts.loc[keep, "pop"].to_numpy()},
                             geometry=hexes.geometry[keep.to_numpy()].to_crs(units.crs).to_numpy(),
                             crs=units.crs).to_crs(4326)
    per = layer.groupby("unit")["pop"].sum()
    empty = sorted(set(units["unit"].astype(str)) - set(per.index[per > 0]))
    if empty:
        print(f"  !! {len(empty)} units with no populated hex (they will draw nothing unless "
              f"given their own polygon): {empty[:8]}")

    if census:
        rows = [(u, census[u], float(per.get(u, 0.0))) for u in census if census[u] > 0]
        ratio = sum(k for _, _, k in rows) / sum(c for _, c, _ in rows)
        norm = sorted(((k / c / ratio), u) for u, c, k in rows)
        print(f"  Kontur / census nationally {ratio:.3f}; per unit, normalised: "
              f"p10 {norm[len(norm) // 10][0]:.2f}  median {norm[len(norm) // 2][0]:.2f}  "
              f"p90 {norm[9 * len(norm) // 10][0]:.2f}")
        print("  lowest: " + ", ".join(f"{u} {r:.2f}" for r, u in norm[:5]))
        print("  highest: " + ", ".join(f"{u} {r:.2f}" for r, u in norm[-5:]))
        outside_band = sum(1 for r, _ in norm if not (1 / band <= r <= band))
        print(f"  {outside_band} of {len(norm)} units outside a factor of {band:g}")
        if len(rows) >= 8 and all(k > 0 for _, _, k in rows):
            lc = [math.log(c) for _, c, _ in rows]
            lk = [math.log(k) for _, _, k in rows]

            def pear(a, b):
                ma, mb = sum(a) / len(a), sum(b) / len(b)
                num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
                return num / math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))
            r = pear(lc, lk)
            rng = random.Random(0)
            best = max(abs(pear(lc, rng.sample(lk, len(lk)))) for _ in range(500))
            print(f"  log correlation r = {r:.3f} against a best of {best:.3f} over 500 shuffles"
                  + ("" if r > best else "   !! THE JOIN IS NOT CARRYING INFORMATION"))

    out = Path(out) if out else ROOT / "data" / "geo" / cc / f"{cc}_hexes.gpkg"
    out.parent.mkdir(parents=True, exist_ok=True)
    layer.to_file(out, layer="hexes", driver="GPKG")
    print(f"  wrote {out} ({len(layer):,} hexes)")
    return layer
