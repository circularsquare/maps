"""
Download cartographic boundary tract shapefiles + optional AREAWATER files.

Tract files are cartographic boundary (cb_) versions, which are pre-clipped to the
coastline — so dots won't scatter into the ocean/harbor. The full TIGER (tl_) files
extend into water and cause dots to appear offshore.

AREAWATER files cover inland water bodies (rivers, lakes, bays within tracts).
Pass --areawater to download them too; scatter_dots.py will subtract them when scattering.

Usage:
    python download_shapefiles.py --state 36               # NY tracts only
    python download_shapefiles.py --state 36 --areawater   # NY tracts + inland water
    python download_shapefiles.py --all --areawater        # everything
"""

import argparse
import time
import zipfile
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import geopandas as gpd

from fetch_data import STATE_FIPS

TRACT_URL  = "https://www2.census.gov/geo/tiger/GENZ2024/shp"
NHD_S3     = "https://prd-tnm.s3.amazonaws.com/StagedProducts/Hydrography/NHD/State/Shape"
HEADERS    = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.census.gov/",
}
DELAY      = 0.5  # seconds between requests

# Maps Census FIPS codes to the state name used in NHD filenames
FIPS_TO_NHD = {
    "01": "Alabama", "02": "Alaska", "04": "Arizona", "05": "Arkansas",
    "06": "California", "08": "Colorado", "09": "Connecticut", "10": "Delaware",
    "11": "District_of_Columbia", "12": "Florida", "13": "Georgia", "15": "Hawaii",
    "16": "Idaho", "17": "Illinois", "18": "Indiana", "19": "Iowa",
    "20": "Kansas", "21": "Kentucky", "22": "Louisiana", "23": "Maine",
    "24": "Maryland", "25": "Massachusetts", "26": "Michigan", "27": "Minnesota",
    "28": "Mississippi", "29": "Missouri", "30": "Montana", "31": "Nebraska",
    "32": "Nevada", "33": "New_Hampshire", "34": "New_Jersey", "35": "New_Mexico",
    "36": "New_York", "37": "North_Carolina", "38": "North_Dakota", "39": "Ohio",
    "40": "Oklahoma", "41": "Oregon", "42": "Pennsylvania", "44": "Rhode_Island",
    "45": "South_Carolina", "46": "South_Dakota", "47": "Tennessee", "48": "Texas",
    "49": "Utah", "50": "Vermont", "51": "Virginia", "53": "Washington",
    "54": "West_Virginia", "55": "Wisconsin", "56": "Wyoming",
    "72": "Puerto_Rico",
}


def download_and_extract(url: str, out_path: Path):
    zip_path = out_path.parent / (out_path.name + ".zip")
    print(f"  Downloading {url}...")
    time.sleep(DELAY)
    resp = requests.get(url, stream=True, timeout=120, headers=HEADERS)
    resp.raise_for_status()
    with open(zip_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 64):
            f.write(chunk)
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(out_path)
    except zipfile.BadZipFile:
        body = zip_path.read_text(errors="replace")[:500]
        zip_path.unlink(missing_ok=True)
        raise RuntimeError(f"Server returned non-zip response for {url}\nResponse body: {body}")
    zip_path.unlink()
    print(f"  Extracted to {out_path}")


def download_tracts(state_fips: str, out_dir: Path):
    name = f"cb_2024_{state_fips}_tract_500k"
    out_path = out_dir / name
    if out_path.exists():
        print(f"  {name}: already exists, skipping")
        return
    download_and_extract(f"{TRACT_URL}/{name}.zip", out_path)


def download_nhd_water(state_fips: str, out_dir: Path) -> Path:
    """
    Download NHDWaterbody and NHDArea shapefiles for the state from the USGS S3 bucket
    using HTTP Range requests (remotezip) — only the layers we need are downloaded,
    not the full 700 MB+ state archive.
    Returns the directory the shapefiles were extracted to.
    """
    from remotezip import RemoteZip

    state_name = FIPS_TO_NHD.get(state_fips)
    if not state_name:
        raise ValueError(f"No NHD name mapping for FIPS {state_fips}")

    nhd_dir = out_dir / f"nhd_{state_fips}"
    nhd_dir.mkdir(exist_ok=True)

    # Check if already extracted
    if (nhd_dir / "NHDWaterbody.shp").exists() and (nhd_dir / "NHDArea.shp").exists():
        print(f"    NHD water already downloaded, skipping")
        return nhd_dir

    url = f"{NHD_S3}/NHD_H_{state_name}_State_Shape.zip"
    print(f"  Fetching NHD layers from {url} (range requests, no full download)...")

    layers = ["NHDWaterbody", "NHDArea"]
    exts = [".shp", ".dbf", ".prj", ".shx"]

    with RemoteZip(url) as zf:
        for layer in layers:
            for ext in exts:
                member = f"Shape/{layer}{ext}"
                try:
                    data = zf.read(member)
                    (nhd_dir / f"{layer}{ext}").write_bytes(data)
                except KeyError:
                    pass  # some states may lack NHDArea entries

    print(f"    Extracted NHDWaterbody + NHDArea to {nhd_dir}")
    return nhd_dir


def download_areawater_and_clip(state_fips: str, out_dir: Path, min_water_area: float = 40_000):
    """
    Download NHD water polygons for the state (via USGS S3, range requests),
    subtract them from tract polygons, and save the result as a ready-to-use
    clipped shapefile. scatter_dots.py loads this directly.
    """
    tract_shp = out_dir / f"cb_2024_{state_fips}_tract_500k" / f"cb_2024_{state_fips}_tract_500k.shp"
    if not tract_shp.exists():
        print(f"  Tract shapefile not found — download tracts first")
        return

    clipped_path = out_dir / f"cb_2024_{state_fips}_tract_clipped.shp"
    if clipped_path.exists():
        print(f"  Clipped shapefile already exists, skipping")
        return

    nhd_dir = download_nhd_water(state_fips, out_dir)

    water_parts = []
    for layer in ["NHDWaterbody", "NHDArea"]:
        shp = nhd_dir / f"{layer}.shp"
        if shp.exists():
            water_parts.append(gpd.read_file(shp))

    if not water_parts:
        print(f"  No NHD water data found, skipping clip")
        return

    tracts = gpd.read_file(tract_shp).to_crs(epsg=4326)
    print(f"  Clipping tracts against NHD water (spatial index)...")
    water = gpd.pd.concat(water_parts, ignore_index=True).to_crs(epsg=4326)
    if min_water_area > 0:
        water_m = water.to_crs(epsg=3857)
        keep = water_m.geometry.area >= min_water_area
        print(f"  Kept {keep.sum()} / {len(water)} water polygons >= {min_water_area:.0f} m²")
        water = water[keep]
    # Simplify water geometry in meters (30m tolerance) before reprojecting back —
    # reduces vertex count dramatically on complex river polygons, speeds up overlay
    water = water.to_crs(epsg=3857)
    water["geometry"] = water.geometry.simplify(30, preserve_topology=True)
    water = water.to_crs(epsg=4326)
    # overlay uses an STR-tree spatial index — only tracts that actually intersect
    # a water polygon get clipped; the rest pass through untouched
    clipped = gpd.overlay(tracts, water[["geometry"]], how="difference", keep_geom_type=True)
    clipped["geometry"] = clipped["geometry"].make_valid()
    clipped = clipped[~clipped.geometry.is_empty]
    clipped.to_file(clipped_path)
    print(f"  Saved clipped tracts to {clipped_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", help="Single state FIPS code (e.g. 36 for NY)")
    parser.add_argument("--all", action="store_true", help="Download all 50 states + DC")
    parser.add_argument("--areawater", action="store_true", help="Also download inland water shapefiles")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel states to process (default: 4)")
    parser.add_argument(
        "--min-water-area", type=float, default=5_000,
        help="Minimum water polygon area in sq meters to include (default: 5000). "
             "Filters out tiny ditches/streams. Set 0 to disable.",
    )
    args = parser.parse_args()

    out_dir = Path("data/shapefiles")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.state:
        states = [args.state.zfill(2)]
    elif args.all:
        states = STATE_FIPS
    else:
        parser.print_help()
        return

    def process_state(fips):
        print(f"State {fips}:")
        download_tracts(fips, out_dir)
        if args.areawater:
            download_areawater_and_clip(fips, out_dir, min_water_area=args.min_water_area)

    if args.workers == 1 or len(states) == 1:
        for fips in states:
            process_state(fips)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(process_state, fips): fips for fips in states}
            for fut in as_completed(futures):
                fips = futures[fut]
                if fut.exception():
                    print(f"State {fips} ERROR: {fut.exception()}")


if __name__ == "__main__":
    main()
