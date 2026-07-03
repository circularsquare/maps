"""One-time split of the global GEBCO GeoTIFF into regional tiles.

Windowed reads (rasterio) crop each region straight off disk -- the 7.5 GB grid
is never loaded whole. Each region gets a few degrees of overlap margin so the
basins/divides near a tile seam (which are unreliable, being cut) sit outside the
area you actually map.

  # named regions:
  python split_gebco.py ~/Downloads/GEBCO_2026.tif ../data/ascentshed/gebco

  # one custom box (lon0 lat0 lon1 lat1 name):
  python split_gebco.py ~/Downloads/GEBCO_2026.tif ../data/ascentshed/gebco \
        --bbox 5 43 17 48 alps

Then:  python run_dem.py ../data/ascentshed/gebco/<name>.tif 1 <N>
"""
import sys, os
import rasterio
from rasterio.windows import from_bounds

# lon0, lat0, lon1, lat1  -- interior extent (overlap is added on top)
REGIONS = {
    "alps":         (4, 43, 17, 49),
    "japan":        (128, 30, 146, 46),
    "newzealand":   (166, -47, 179, -34),
    "hawaii":       (-161, 18, -154, 23),
    "taiwan_luzon": (118, 12, 124, 26),
    "sunda":        (95, -9, 120, 6),     # Sumatra/Java/Borneo, islands + shelf
    "iceland":      (-25, 63, -13, 67),
    "himalaya_w":   (72, 28, 88, 37),     # full Himalaya is wide -> west/east halves
    "himalaya_e":   (86, 25, 100, 35),
    "andes_c":      (-78, -28, -64, -10),
    "cascades":     (-124, 40, -118, 49),
    "alaska":       (-153, 58, -135, 64),
    "caucasus":     (40, 40, 50, 45),
}

OVERLAP = 2.0   # degrees of margin added on every side


def cut(ds, bbox, name, outdir, overlap=OVERLAP):
    lon0, lat0, lon1, lat1 = bbox
    L = ds.bounds
    b = (max(lon0 - overlap, L.left), max(lat0 - overlap, L.bottom),
         min(lon1 + overlap, L.right), min(lat1 + overlap, L.top))
    win = from_bounds(*b, transform=ds.transform).round_offsets().round_lengths()
    arr = ds.read(1, window=win)
    transform = ds.window_transform(win)
    meta = ds.meta.copy()
    meta.update(driver="GTiff", height=arr.shape[0], width=arr.shape[1],
                transform=transform, count=1, compress="deflate")
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, f"{name}.tif")
    with rasterio.open(out, "w", **meta) as dst:
        dst.write(arr, 1)
    px = max(arr.shape)
    flag = "  <-- big, consider downsample" if px > 5200 else ""
    print(f"{name:14s} {arr.shape[1]}x{arr.shape[0]} px  "
          f"elev {int(arr.min())}..{int(arr.max())} m{flag}")


def main():
    src, outdir = sys.argv[1], sys.argv[2]
    with rasterio.open(src) as ds:
        deg = 1.0 / abs(ds.transform.a)
        print(f"source {ds.width}x{ds.height}  {deg:.0f} cells/deg  crs {ds.crs}")
        if "--bbox" in sys.argv:
            i = sys.argv.index("--bbox")
            box = tuple(float(v) for v in sys.argv[i + 1:i + 5])
            name = sys.argv[i + 5]
            cut(ds, box, name, outdir)
        else:
            for name, box in REGIONS.items():
                cut(ds, box, name, outdir)


if __name__ == "__main__":
    main()
