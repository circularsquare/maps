"""
Convert CROPGRIDS NetCDF harvested-area grids into dot-map binary data.
Each dot represents HA_PER_DOT hectares of harvested crop area.

Usage:
  python make_cropgrids_dots.py          # process all .nc files
  python make_cropgrids_dots.py wheat    # process just one crop

Expects: data/cropgrids/CROPGRIDSv1.08_NC_maps/CROPGRIDSv1.08_*.nc
Outputs: data/all_dots.bin and data/legend.json
"""
import numpy as np
import struct
import json
import os
import sys
import glob
import time

try:
    import netCDF4
except ImportError:
    sys.exit("Need netCDF4: pip install netCDF4")

HA_PER_DOT = 10_000
NC_DIR = "data/cropgrids/CROPGRIDSv1.08_NC_maps"
OUT_DIR = "data"

# Merge fodder variants into their parent crop (dots combined at grid level)
# target code -> list of source file codes that get merged into it
MERGE = {
    "maize":      ["maizefor"],
    "sorghum":    ["sorghumfor"],
    "rye":        ["ryefor"],
    "cabbage":    ["cabbagefor"],
    "carrot":     ["carrotfor"],
    "oilseednes": ["oilseedfor"],
}
# Rename-only: fodder crops with no non-fodder counterpart (just drop "for" suffix)
RENAME = {
    "beetfor":  "beet",
    "swedefor": "swede",
    "turnipfor": "turnip",
}

# Names that deserve nicer formatting than just .title()
PRETTY_NAMES = {
    "aniseetc": "Anise & Similar",
    "beet": "Beet",
    "berrynes": "Berries NES",
    "brazil": "Brazil Nut",
    "broadbean": "Broad Bean",
    "canaryseed": "Canary Seed",
    "cashewapple": "Cashew Apple",
    "cerealnes": "Cereals NES",
    "chilleetc": "Chilli & Peppers",
    "citrusnes": "Citrus NES",
    "cocoa": "Cocoa",
    "coconut": "Coconut",
    "coffee": "Coffee",
    "cowpea": "Cowpea",
    "cranberry": "Cranberry",
    "cucumberetc": "Cucumber & Gherkin",
    "eggplant": "Eggplant",
    "fibrenes": "Fibre NES",
    "fornes": "Fodder NES",
    "fruitnes": "Fruit NES",
    "grassnes": "Grass NES",
    "grapefruitetc": "Grapefruit & Pomelo",
    "greenbean": "Green Bean",
    "greenbroadbean": "Green Broad Bean",
    "greencorn": "Green Corn",
    "greenonion": "Green Onion",
    "greenpea": "Green Pea",
    "groundnut": "Groundnut",
    "hempseed": "Hemp Seed",
    "jutelikefiber": "Jute-like Fibre",
    "kapokfiber": "Kapok Fibre",
    "kapokseed": "Kapok Seed",
    "karite": "Shea Nut",
    "kolanut": "Kola Nut",
    "legumenes": "Legumes NES",
    "lemonlime": "Lemon & Lime",
    "mate": "Mate",
    "melonetc": "Melon",
    "melonseed": "Melon Seed",
    "mixedgrain": "Mixed Grain",
    "mixedgrass": "Mixed Grass",
    "nutnes": "Nuts NES",
    "oilpalm": "Oil Palm",
    "oilseednes": "Oilseeds NES",
    "peachetc": "Peach & Nectarine",
    "peppermint": "Peppermint",
    "pigeonpea": "Pigeon Pea",
    "popcorn": "Popcorn",
    "pulsenes": "Pulses NES",
    "pumpkinetc": "Pumpkin & Squash",
    "rootnes": "Roots NES",
    "safflower": "Safflower",
    "sourcherry": "Sour Cherry",
    "spicenes": "Spices NES",
    "stonefruitnes": "Stone Fruit NES",
    "stringbean": "String Bean",
    "sugarbeet": "Sugar Beet",
    "sugarcane": "Sugar Cane",
    "sugarnes": "Sugar NES",
    "sunflower": "Sunflower",
    "swede": "Swede",
    "sweetpotato": "Sweet Potato",
    "tangetc": "Tangerine & Mandarin",
    "tobacco": "Tobacco",
    "tropicalnes": "Tropical Fruit NES",
    "turnip": "Turnip",
    "vegetablenes": "Vegetables NES",
    "vegfor": "Vegetables (Fodder)",
    "watermelon": "Watermelon",
    "rasberry": "Raspberry",
}


def hilbert_distances_vectorized(ix, iy, order):
    """Compute Hilbert curve distances for arrays of grid coords."""
    x = ix.copy().astype(np.int64)
    y = iy.copy().astype(np.int64)
    d = np.zeros(len(x), dtype=np.int64)
    for s in range(order - 1, -1, -1):
        n = 1 << s
        rx = ((x & n) > 0).astype(np.int64)
        ry = ((y & n) > 0).astype(np.int64)
        d += n * n * ((3 * rx) ^ ry)
        # Rotate: where ry == 0
        mask = ry == 0
        # Sub-mask: ry==0 and rx==1
        flip = mask & (rx == 1)
        x[flip] = n - 1 - x[flip]
        y[flip] = n - 1 - y[flip]
        # Swap x,y where ry==0
        x[mask], y[mask] = y[mask].copy(), x[mask].copy()
    return d


def process_crop(nc_path):
    """Read one NetCDF file, return list of (lon, lat) dots."""
    ds = netCDF4.Dataset(nc_path, "r")
    data = ds.variables["harvarea"][:]
    lats = ds.variables["lat"][:]
    lons = ds.variables["lon"][:]
    ds.close()

    if hasattr(data, "filled"):
        data = data.filled(0.0)
    data = np.nan_to_num(data, nan=0.0)
    data[data < 0] = 0  # ocean = -1

    cell_size = abs(float(lats[1] - lats[0]))

    # Non-zero pixels
    rows, cols = np.where(data > 0)
    if len(rows) == 0:
        return [], 0.0

    pixel_lats = lats[rows]
    pixel_lons = lons[cols]
    vals = data[rows, cols].astype(np.float64)
    total_ha = vals.sum()

    # Hilbert sort
    ix = np.round((pixel_lons + 180) / cell_size).astype(np.int64)
    iy = np.round((pixel_lats + 90) / cell_size).astype(np.int64)
    distances = hilbert_distances_vectorized(ix, iy, order=13)
    sort_idx = np.argsort(distances)
    pixel_lons = np.asarray(pixel_lons[sort_idx], dtype=np.float64)
    pixel_lats = np.asarray(pixel_lats[sort_idx], dtype=np.float64)
    vals = vals[sort_idx]

    # Accumulate and drop dots
    rng = np.random.default_rng(42)
    dots = []
    accum = 0.0
    half = cell_size / 2
    for i in range(len(vals)):
        accum += vals[i]
        while accum >= HA_PER_DOT:
            dx = rng.uniform(-half, half)
            dy = rng.uniform(-half, half)
            dots.append((float(pixel_lons[i] + dx), float(pixel_lats[i] + dy)))
            accum -= HA_PER_DOT

    return dots, total_ha


def process_merged_crops(nc_paths):
    """Read multiple NetCDF files, add their grids together, then generate dots."""
    combined = None
    lats = lons = None
    for nc_path in nc_paths:
        ds = netCDF4.Dataset(nc_path, "r")
        data = ds.variables["harvarea"][:]
        if lats is None:
            lats = ds.variables["lat"][:]
            lons = ds.variables["lon"][:]
        ds.close()
        if hasattr(data, "filled"):
            data = data.filled(0.0)
        data = np.nan_to_num(data, nan=0.0)
        data[data < 0] = 0
        if combined is None:
            combined = data.astype(np.float64)
        else:
            combined += data.astype(np.float64)

    cell_size = abs(float(lats[1] - lats[0]))
    rows, cols = np.where(combined > 0)
    if len(rows) == 0:
        return [], 0.0

    pixel_lats = lats[rows]
    pixel_lons = lons[cols]
    vals = combined[rows, cols]
    total_ha = vals.sum()

    ix = np.round((pixel_lons + 180) / cell_size).astype(np.int64)
    iy = np.round((pixel_lats + 90) / cell_size).astype(np.int64)
    distances = hilbert_distances_vectorized(ix, iy, order=13)
    sort_idx = np.argsort(distances)
    pixel_lons = np.asarray(pixel_lons[sort_idx], dtype=np.float64)
    pixel_lats = np.asarray(pixel_lats[sort_idx], dtype=np.float64)
    vals = vals[sort_idx]

    rng = np.random.default_rng(42)
    dots = []
    accum = 0.0
    half = cell_size / 2
    for i in range(len(vals)):
        accum += vals[i]
        while accum >= HA_PER_DOT:
            dx = rng.uniform(-half, half)
            dy = rng.uniform(-half, half)
            dots.append((float(pixel_lons[i] + dx), float(pixel_lats[i] + dy)))
            accum -= HA_PER_DOT
    return dots, total_ha


def prettify_name(code):
    if code in PRETTY_NAMES:
        return PRETTY_NAMES[code]
    return code.replace("_", " ").title()


def main():
    nc_files = sorted(glob.glob(os.path.join(NC_DIR, "CROPGRIDSv1.08_*.nc")))
    if not nc_files:
        sys.exit(f"No CROPGRIDSv1.08_*.nc files found in {NC_DIR}/")

    # Filter to single crop if specified
    if len(sys.argv) > 1:
        query = sys.argv[1].lower()
        nc_files = [f for f in nc_files if query in os.path.basename(f).lower()]
        if not nc_files:
            sys.exit(f"No .nc file matching '{query}'")

    # Build reverse merge map: source_code -> target_code
    merge_into = {}
    for target, sources in MERGE.items():
        for src in sources:
            merge_into[src] = target

    # Skip files that will be merged (process them with their parent)
    skip_codes = set(merge_into.keys())

    print(f"Found {len(nc_files)} crop files")

    # First pass: load all grids so we can merge
    raw_grids = {}  # code -> nc_path
    for nc_path in nc_files:
        stem = os.path.splitext(os.path.basename(nc_path))[0]
        code = stem.replace("CROPGRIDSv1.08_", "")
        raw_grids[code] = nc_path

    # Build processing list: skip merged sources, add merged grids to parent
    proc_list = []  # (final_code, [nc_paths])
    seen = set()
    for nc_path in nc_files:
        stem = os.path.splitext(os.path.basename(nc_path))[0]
        code = stem.replace("CROPGRIDSv1.08_", "")
        if code in skip_codes:
            continue
        if code in seen:
            continue
        seen.add(code)
        paths = [nc_path]
        # Add any sources that merge into this code
        if code in MERGE:
            for src in MERGE[code]:
                if src in raw_grids:
                    paths.append(raw_grids[src])
        # Apply rename
        final_code = RENAME.get(code, code)
        proc_list.append((final_code, paths))

    crops = []
    all_dot_data = []
    t0 = time.time()

    for i, (code, paths) in enumerate(proc_list):
        name = prettify_name(code)

        t1 = time.time()
        if len(paths) == 1:
            dots, total_ha = process_crop(paths[0])
        else:
            dots, total_ha = process_merged_crops(paths)
        elapsed = time.time() - t1

        merge_note = f" (merged {len(paths)} files)" if len(paths) > 1 else ""
        print(f"[{i+1:3d}/{len(proc_list)}] {name:30s}  {total_ha:>12,.0f} ha  ->  {len(dots):>6,} dots  ({elapsed:.1f}s){merge_note}")

        if len(dots) > 0:
            crops.append({"code": code, "name": name, "dots": len(dots)})
            all_dot_data.append(dots)

    if not crops:
        sys.exit("No dots generated!")

    # Placeholder colors (frontend overrides via group HSL)
    import colorsys
    for i, crop in enumerate(crops):
        hue = i / len(crops)
        r, g, b = colorsys.hls_to_rgb(hue, 0.45, 0.7)
        crop["color"] = f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"

    legend_path = os.path.join(OUT_DIR, "legend.json")
    with open(legend_path, "w") as f:
        json.dump(crops, f, indent=2)
    print(f"\nWrote {legend_path} ({len(crops)} crops)")

    # Binary: float32 lon, float32 lat, uint16 crop_id (supports >255 crops)
    bin_path = os.path.join(OUT_DIR, "all_dots.bin")
    total = 0
    with open(bin_path, "wb") as out:
        for crop_idx, dots in enumerate(all_dot_data):
            for lon, lat in dots:
                out.write(struct.pack("<ffH", lon, lat, crop_idx))
            total += len(dots)

    size_mb = os.path.getsize(bin_path) / 1024 / 1024
    elapsed_total = time.time() - t0
    print(f"Wrote {bin_path}: {total:,} dots, {size_mb:.1f} MB ({elapsed_total:.0f}s total)")


if __name__ == "__main__":
    main()
