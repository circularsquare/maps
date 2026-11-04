"""Sample the CUDEM riverbed (raw/bathy, from fetch_bathy.py) along every OSM tunnel way where the
build treats the ground as water, and write evidence/riverbed.csv.

Points are the build's own 25 m points (data/tracks.json, so run build.py first) where the way
is a tunnel and its smoothed ground is missing or below 1 m (the same `wet` rule as build.py).

Columns: way_id, lon, lat, bed_navd88_m (CUDEM at the point, nearest cell), notes,
bed_min_15m_m (deepest cell within 15 m: the tunnel is tens of metres wide and cut through a
dredged channel), way_name, track.
`notes` says "water" where the bed is below -0.5 m NAVD88, "shore" between -0.5 and +1 m,
"land" above +1 m (the build's ground said wet but CUDEM says dry: banks, seawalls, fill),
"nodata" where CUDEM has no value.
"""
import glob
import json
import pathlib

import numpy as np
import pandas as pd
import rasterio

ROOT = pathlib.Path(__file__).parent
RAW = ROOT / "raw" / "bathy"
NODATA_BELOW = -500.0


def main():
    d = json.loads((ROOT / "data" / "tracks.json").read_text(encoding="utf-8"))
    recs = []
    for t in d["tracks"]:
        if t["cls"] != "tunnel":
            continue
        for k, (lon, lat, g) in enumerate(zip(t["lon"], t["lat"], t["ground"])):
            if g is None or g < 1.0:
                recs.append({"way_id": t["id"], "lon": lon, "lat": lat, "way_name": t["name"],
                             "track": t["track"], "k": k})
    df = pd.DataFrame(recs)
    lon = df["lon"].to_numpy(); lat = df["lat"].to_numpy()
    bed = np.full(len(df), np.nan); bmin = np.full(len(df), np.nan)
    for f in sorted(glob.glob(str(RAW / "cudem_*.tif"))):
        with rasterio.open(f) as ds:
            b = ds.bounds
            m = (lon >= b.left) & (lon < b.right) & (lat > b.bottom) & (lat <= b.top) & np.isnan(bed)
            if not m.any():
                continue
            arr = ds.read(1).astype(float)
            arr[arr < NODATA_BELOW] = np.nan
            r, c = rasterio.transform.rowcol(ds.transform, lon[m], lat[m])
            r = np.asarray(r); c = np.asarray(c)
            ok = (r >= 0) & (r < arr.shape[0]) & (c >= 0) & (c < arr.shape[1])
            idx = np.where(m)[0][ok]; r = r[ok]; c = c[ok]
            bed[idx] = arr[r, c]
            # 15 m is ~5 rows (3.4 m) and ~6 columns (2.6 m)
            for i_, rr, cc in zip(idx, r, c):
                w = arr[max(0, rr - 5):rr + 6, max(0, cc - 6):cc + 7]
                if not np.all(np.isnan(w)):
                    bmin[i_] = np.nanmin(w)
    notes = []
    for v in bed:
        if np.isnan(v):
            notes.append("nodata")
        elif v < -0.5:
            notes.append("water")
        elif v <= 1.0:
            notes.append("shore")
        else:
            notes.append("land")
    df["bed_navd88_m"] = np.round(bed, 2)
    df["bed_min_15m_m"] = np.round(bmin, 2)
    df["notes"] = notes
    out = df[["way_id", "lon", "lat", "bed_navd88_m", "notes", "bed_min_15m_m", "way_name", "track"]]
    (ROOT / "evidence").mkdir(exist_ok=True)
    out.to_csv(ROOT / "evidence" / "riverbed.csv", index=False)
    print(len(out), "points on", out["way_id"].nunique(), "ways")
    print(out["notes"].value_counts().to_string())
    w = out[out["notes"] == "water"]
    print("water bed NAVD88 m: min %.1f, median %.1f, max %.1f" % (w["bed_navd88_m"].min(),
          w["bed_navd88_m"].median(), w["bed_navd88_m"].max()))


if __name__ == "__main__":
    main()
