"""Build a city pack (format 1, notes/T-004.md) and its boundary (notes/T-003.md).

    python pipeline/build_city.py nyc

Downloads are cached in data/raw/; outputs go to data/packs/<city>.json, .bin and
.boundary.geojson. Safe to re-run: cached files are reused, outputs are replaced whole.
The zone gravity is solved by the sim crate (cargo, `pack_gravity`), so Rust must be installed.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")  # before numpy; Anita is using the machine

import datetime
import gzip
import io
import json
import shutil
import struct
import subprocess
import sys
import time
import zipfile

import h3
import numpy as np
import pandas as pd
import requests
import shapefile  # pyshp
from shapely.geometry import mapping, shape
from shapely import make_valid, union_all

import spread
import water
import wfh

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
WORK = os.path.join(ROOT, "data", "work")
PACKS = os.path.join(ROOT, "data", "packs")
UA = {"User-Agent": "trainworld-pipeline/0.1 (rail game data build; python-requests)"}
R_EARTH = 6371008.8
FORMAT = 1

TIGER_BLOCKS = "https://www2.census.gov/geo/tiger/TIGER2020/TABBLOCK20/tl_2020_{fips}_tabblock20.zip"
LODES_WAC = "https://lehd.ces.census.gov/data/lodes/LODES8/{st}/wac/{st}_wac_S000_JT00_{year}.csv.gz"
LODES_RAC = "https://lehd.ces.census.gov/data/lodes/LODES8/{st}/rac/{st}_rac_S000_JT00_{year}.csv.gz"
# 2020 county polygons: they match the 2020 block GEOIDs (Connecticut's old counties, which
# LODES 8 also uses; the 2022+ files have Connecticut's planning regions instead).
COUNTIES = "https://www2.census.gov/geo/tiger/GENZ2020/shp/cb_2020_us_county_500k.zip"
# Checks only, not inputs.
POPEST = "https://www2.census.gov/programs-surveys/popest/datasets/2020-2021/counties/totals/co-est2021-alldata.csv"
QCEW_TOTAL = "https://data.bls.gov/cew/data/api/{year}/a/industry/10.csv"

STATES = {"09": "ct", "34": "nj", "36": "ny", "42": "pa"}

CITIES = {
    "nyc": {
        "origin": (-73.985, 40.758),  # Times Square / Midtown
        "lodes_years": (2023, 2022),
        # T-030: 25 m pixels; water under 2 pixels (~50 m) wide is dropped (creeks, ditches,
        # ponds a railway crosses on a culvert or a short bridge at grade)
        "water": {"cell_m": 25.0, "min_width_px": 2},
        # T-014, every case argued in notes/T-014.md
        "job_adjustments": {
            "moves": [
                # North Shore University Hospital campus, Manhasset: 41,521 jobs, about Northwell's
                # whole non-campus staff; keep 10,000 (NSUH ~7,000 staff plus the Feinstein
                # Institutes beside it), the rest over the population of the counties where
                # Northwell ran hospitals in 2023
                {"name": "Northwell at NSUH", "block": "360593018002016", "keep": 10000, "sector": "CNS16",
                 "to_counties": ["36059", "36103", "36081", "36061", "36085", "36119"]},
                # JetBlue's head office, 27-01 Queens Plaza North: ~1,300 staff there, ~10,300
                # transport jobs on the block; the rest are crew based at JFK (Terminal 5's block)
                {"name": "JetBlue crew at the LIC head office", "block": "360810033011005",
                 "sector": "CNS08", "keep_sector": 1300, "to_block": "360810716001015"},
            ],
            # home care agencies: aides work in clients' homes, the agency reports them at its
            # office. Health care jobs of such blocks go over their county's population.
            "home_care": {"min_jobs": 1000, "min_health": 0.8, "min_low_pay": 0.0, "max_high_pay": 0.35},
        },
        # notes/T-003.md: the 2023 New York-Newark-Jersey City MSA (22 counties) plus
        # Fairfield CT, Dutchess NY and Orange NY.
        "counties": {
            # MSA, New York
            "36005": "Bronx", "36047": "Kings", "36061": "New York", "36081": "Queens",
            "36085": "Richmond", "36059": "Nassau", "36103": "Suffolk", "36119": "Westchester",
            "36087": "Rockland", "36079": "Putnam",
            # MSA, New Jersey
            "34003": "Bergen", "34013": "Essex", "34017": "Hudson", "34019": "Hunterdon",
            "34023": "Middlesex", "34025": "Monmouth", "34027": "Morris", "34029": "Ocean",
            "34031": "Passaic", "34035": "Somerset", "34037": "Sussex", "34039": "Union",
            # added
            "09001": "Fairfield CT", "36027": "Dutchess", "36071": "Orange NY",
        },
    },
}


# ---------------------------------------------------------------- downloads

def fetch(url, name=None):
    """Download url into data/raw once; return the local path."""
    path = os.path.join(RAW, name or url.rsplit("/", 1)[1])
    if os.path.exists(path):
        return path
    print(f"  downloading {url}", flush=True)
    tmp = path + ".part"
    with requests.get(url, headers=UA, timeout=300, stream=True) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    os.replace(tmp, path)
    return path


class HttpRangeFile(io.RawIOBase):
    """Read-only seekable file over HTTP Range requests, so zipfile can pull one member."""

    def __init__(self, url):
        self.url = url
        h = requests.head(url, headers=UA, timeout=60, allow_redirects=True)
        h.raise_for_status()
        if h.headers.get("Accept-Ranges") != "bytes":
            raise OSError("no range support")
        self.size = int(h.headers["Content-Length"])
        self.pos = 0
        self.fetched = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b""
        hdr = dict(UA, Range=f"bytes={self.pos}-{self.pos + n - 1}")
        r = requests.get(self.url, headers=hdr, timeout=300)
        if r.status_code != 206:
            raise OSError(f"range request got {r.status_code}")
        self.pos += len(r.content)
        self.fetched += len(r.content)
        return r.content

    def readinto(self, b):
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)


def fetch_zip_member(url, suffix):
    """Fetch only the member ending in `suffix` from a remote zip (cached in data/raw)."""
    base = url.rsplit("/", 1)[1][: -len(".zip")]
    path = os.path.join(RAW, base + suffix)
    if os.path.exists(path):
        return path
    print(f"  fetching {base}{suffix} out of {url}", flush=True)
    tmp = path + ".part"
    try:
        remote = HttpRangeFile(url)
        src = io.BufferedReader(remote, buffer_size=1 << 22)
        with zipfile.ZipFile(src) as z:
            member = next(m for m in z.namelist() if m.endswith(suffix))
            with z.open(member) as zin, open(tmp, "wb") as out:
                while True:
                    chunk = zin.read(1 << 22)
                    if not chunk:
                        break
                    out.write(chunk)
        print(f"    {remote.fetched / 1e6:.1f} MB transferred of a {remote.size / 1e6:.0f} MB zip")
    except OSError as e:  # no range support: take the whole zip
        print(f"    range read failed ({e}); downloading the whole zip")
        with zipfile.ZipFile(fetch(url)) as z:
            member = next(m for m in z.namelist() if m.endswith(suffix))
            with z.open(member) as zin, open(tmp, "wb") as out:
                out.write(zin.read())
    os.replace(tmp, path)
    return path


# ---------------------------------------------------------------- readers

def read_dbf(path, fields):
    """Read named fields of a dBASE III file into numpy arrays of bytes (fixed width)."""
    with open(path, "rb") as f:
        head = f.read(32)
        nrec, hlen, rlen = struct.unpack("<IHH", head[4:12])
        desc = f.read(hlen - 32)
        f.seek(hlen)
        data = f.read(nrec * rlen)
    names, formats, offsets = [], [], []
    off = 1  # deletion flag
    for i in range(0, len(desc) - 1, 32):
        d = desc[i : i + 32]
        if d[0] == 0x0D:
            break
        name = d[:11].split(b"\0")[0].decode()
        length = d[16]
        names.append(name)
        formats.append(f"S{length}")
        offsets.append(off)
        off += length
    assert off == rlen, (off, rlen)
    dt = np.dtype({"names": names, "formats": formats, "offsets": offsets, "itemsize": rlen})
    rec = np.frombuffer(data, dtype=dt, count=nrec)
    return {k: rec[k] for k in fields}


def blocks_for_state(fips, county_set):
    dbf = fetch_zip_member(TIGER_BLOCKS.format(fips=fips), ".dbf")
    d = read_dbf(dbf, ["GEOID20", "POP20", "INTPTLAT20", "INTPTLON20"])
    geoid = np.char.strip(d["GEOID20"]).astype("U15")
    keep = np.isin(geoid.astype("U5"), list(county_set))  # astype to U5 truncates: state+county
    return pd.DataFrame({
        "geoid": geoid[keep],
        "pop": np.char.strip(d["POP20"][keep]).astype(np.int64),
        "lat": np.char.strip(d["INTPTLAT20"][keep]).astype(np.float64),
        "lon": np.char.strip(d["INTPTLON20"][keep]).astype(np.float64),
    }), len(geoid)


def jobs_for_state(st, years, county_set):
    for y in years:
        url = LODES_WAC.format(st=st, year=y)
        cached = os.path.exists(os.path.join(RAW, url.rsplit("/", 1)[1]))
        if cached or requests.head(url, headers=UA, timeout=60).status_code == 200:
            # sector and earnings columns are for the T-014 adjustments
            w = pd.read_csv(fetch(url), usecols=["w_geocode", "C000", "CE01", "CE03"] + wfh.CNS,
                            dtype={"w_geocode": str})
            w = w[w["w_geocode"].str[:5].isin(county_set)]
            return w.rename(columns={"w_geocode": "geoid"}).assign(jobs=w["C000"].to_numpy()), y, url
    raise RuntimeError(f"no LODES WAC for {st} in {years}")


def residents_for_state(st, year, county_set):
    """T-054: LODES RAC S000 JT00, jobs by the block their holder lives in (C000): the home end
    of the same jobs WAC counts at work."""
    url = LODES_RAC.format(st=st, year=year)
    r = pd.read_csv(fetch(url), usecols=["h_geocode", "C000"], dtype={"h_geocode": str})
    r = r[r["h_geocode"].str[:5].isin(county_set)]
    return r.rename(columns={"h_geocode": "geoid", "C000": "rac"}), url


# ---------------------------------------------------------------- build

def build_boundary(city, cfg):
    path = fetch(COUNTIES)
    r = shapefile.Reader(path)
    fields = [f[0] for f in r.fields[1:]]
    gi = fields.index("GEOID")
    geoms = [make_valid(shape(sr.shape.__geo_interface__)) for sr in r.iterShapeRecords()
             if sr.record[gi] in cfg["counties"]]
    found = len(geoms)
    assert found == len(cfg["counties"]), f"found {found} of {len(cfg['counties'])} counties"
    merged = union_all(geoms)

    def rnd(c):
        if isinstance(c, (list, tuple)) and c and isinstance(c[0], (int, float)):
            return [round(c[0], 6), round(c[1], 6)]
        return [rnd(x) for x in c]

    geom = mapping(merged)
    geom = {"type": geom["type"], "coordinates": rnd(geom["coordinates"])}
    fc = {"type": "FeatureCollection", "features": [{
        "type": "Feature",
        "properties": {"city": city, "counties": sorted(cfg["counties"]),
                       "source": "Census cartographic boundary file cb_2020_us_county_500k, dissolved"},
        "geometry": geom,
    }]}
    out = os.path.join(PACKS, f"{city}.boundary.geojson")
    with open(out + ".part", "w") as f:
        json.dump(fc, f, separators=(",", ":"))
    os.replace(out + ".part", out)
    print(f"boundary: {geom['type']}, {len(geom['coordinates'])} part(s), "
          f"{os.path.getsize(out) / 1e6:.2f} MB -> {out}")
    return merged


def adjust_jobs(df, cfg, rates=None):
    """T-014: move LODES jobs reported at an office that are worked elsewhere (notes/T-014.md).
    Works on the block table (jobs as float) before spreading; totals are unchanged. With
    `rates` (T-042: work-from-home share per CNS sector), the `work_c` column (commuting jobs)
    moves with them: n jobs of sector s carry n x (1 - rate_s) commuters."""
    adj = cfg.get("job_adjustments", {})
    jobs = df["jobs"].to_numpy(np.float64).copy()
    work = df["work_c"].to_numpy(np.float64).copy() if rates else None
    pop = df["pop"].to_numpy(np.float64)
    county = df["geoid"].str[:5].to_numpy()
    row = {g: i for i, g in enumerate(df["geoid"])}
    log = []

    def move(i, n, sector, to_block=None, to_counties=None):
        nc = n * (1.0 - rates[sector]) if rates else 0.0
        jobs[i] -= n
        if rates:
            work[i] -= nc
        if to_block is not None:
            jobs[to_block] += n
            if rates:
                work[to_block] += nc
        else:
            w = np.where(np.isin(county, list(to_counties)), pop, 0.0)
            jobs[:] += n * w / w.sum()
            if rates:
                work[:] += nc * w / w.sum()

    for a in adj.get("moves", []):
        i = row[a["block"]]
        if "keep" in a:
            n = jobs[i] - a["keep"]
        else:  # one sector's jobs, less what stays at the office
            n = float(df[a["sector"]].iat[i]) - a["keep_sector"]
        assert 0 < n <= jobs[i], (a["name"], n, jobs[i])
        move(i, n, a["sector"], row.get(a.get("to_block")), a.get("to_counties"))
        log.append((a["name"], a["block"], n))
    hc = adj.get("home_care")
    if hc:
        c0 = df["C000"].to_numpy(np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            sel = ((c0 >= hc["min_jobs"]) & (df["CNS16"] / c0 >= hc["min_health"])
                   & (df["CE01"] / c0 >= hc["min_low_pay"]) & (df["CE03"] / c0 <= hc["max_high_pay"])).to_numpy()
        for i in np.nonzero(sel)[0]:
            n = float(df["CNS16"].iat[i])
            move(i, n, "CNS16", to_counties=[county[i]])
            log.append(("home care", df["geoid"].iat[i], n))
    moved = sum(n for *_, n in log)
    print(f"job adjustments (T-014): {len(log)} blocks, {moved:,.0f} jobs moved")
    for name, g, n in log:
        if name != "home care":
            print(f"    {name}: {n:,.0f} jobs off block {g}")
    nh = [n for name, _, n in log if name == "home care"]
    if nh:
        print(f"    home care: {len(nh)} agency blocks, {sum(nh):,.0f} jobs spread over their county's population")
    assert abs(jobs.sum() - df["jobs"].sum()) < 1e-3 * df["jobs"].sum()
    out = df.copy()
    out["jobs"] = jobs
    if rates:
        assert abs(work.sum() - df["work_c"].sum()) < 1e-3 * df["work_c"].sum()
        out["work_c"] = work
    return out, log


def build_water(cfg, boundary):
    """The water mask over the boundary's bounding box (T-030): TIGER water at `cell_m`, then
    water narrower than `min_width_px` pixels dropped (pipeline/water.py)."""
    t = time.time()
    lon0, lat0 = cfg["origin"]
    wc = cfg["water"]
    mask, info = water.build_mask(lon0, lat0, boundary.bounds, wc["cell_m"], fetch, fetch(COUNTIES))
    raw_runs = len(mask.xs) // 2
    if wc["min_width_px"] > 1:
        mask = mask.without_narrow(wc["min_width_px"])
    info["min_width_m"] = wc["min_width_px"] * wc["cell_m"]
    print(f"water: {mask.W} x {mask.H} pixels of {mask.cell:.0f} m, {info['counties']} counties' AREAWATER "
          f"({info['areawater_polygons']:,} polygons); {raw_runs:,} runs, {len(mask.xs) // 2:,} after dropping "
          f"water under {info['min_width_m']:.0f} m wide; {(mask.row_off.nbytes + mask.xs.nbytes) / 1e6:.2f} MB; "
          f"water {mask.water_pixels() / (mask.W * mask.H):.1%} of the extent; {time.time() - t:.0f} s")
    return mask, info


def commute_by_county(city, cfg, df):
    """T-042: the gravity's two margins by county, before (workers = population scaled to all
    jobs; all LODES jobs) and after (commuters only). Printed and kept in data/work."""
    c = df["geoid"].str[:5]
    g = pd.DataFrame({"pop": df["pop"], "jobs": df["jobs"], "home_c": df["home_c"], "work_c": df["work_c"]}).groupby(c).sum()
    J, P, W, H = g["jobs"].sum(), g["pop"].sum(), g["work_c"].sum(), g["home_c"].sum()
    t = pd.DataFrame({"county": [cfg["counties"][k] for k in g.index],
                      "from_home_before": g["pop"] * J / P, "from_home_after": g["home_c"] * W / H,
                      "to_work_before": g["jobs"], "to_work_after": g["work_c"]}, index=g.index)
    t.round(0).to_csv(os.path.join(WORK, f"{city}.commute_by_county.csv"))
    print(f"commutes per day (gravity margins), before -> after work from home: {J:,.0f} -> {W:,.0f} ({W / J - 1:+.1%})")
    for k, r in t.sort_values("to_work_before", ascending=False).iterrows():
        print(f"    {r['county']:<14} from home {r['from_home_before']:>10,.0f} -> {r['from_home_after']:>10,.0f} "
              f"({r['from_home_after'] / r['from_home_before'] - 1:+6.1%})   to work {r['to_work_before']:>10,.0f} -> "
              f"{r['to_work_after']:>10,.0f} ({r['to_work_after'] / r['to_work_before'] - 1:+6.1%})")


def write_cells(city, cfg, cells, totals, sources, mask=None, mask_info=None, commute=None):
    """The cell arrays (and the water mask) as a format-1 pack without the gravity, in
    data/work/<city>.cells.*; pack_gravity keeps every array it does not know."""
    order = np.argsort(cells["h3"])
    arrays = [
        ("h3", "u64", "cell", cells["h3"][order].astype("<u8")),
        ("x_m", "f32", "cell", cells["x_m"][order].astype("<f4")),
        ("y_m", "f32", "cell", cells["y_m"][order].astype("<f4")),
        ("pop", "f32", "cell", cells["pop"][order].astype("<f4")),
        ("jobs", "f32", "cell", cells["jobs"][order].astype("<f4")),
    ]
    if "commute_home" in cells:  # T-042: what the gravity balances on (notes/T-004.md)
        arrays += [("commute_home", "f32", "cell", cells["commute_home"][order].astype("<f4")),
                   ("commute_work", "f32", "cell", cells["commute_work"][order].astype("<f4"))]
    if mask is not None:
        arrays += [("water_row", "u32", "water", mask.row_off.astype("<u4")),
                   ("water_x", "u16", "water", mask.xs.astype("<u2"))]
    buf = bytearray()
    entries = []
    for name, dtype, per, a in arrays:
        buf += b"\0" * (-len(buf) % 8)
        entries.append({"name": name, "dtype": dtype, "per": per, "offset": len(buf), "count": int(len(a))})
        buf += a.tobytes()
    lon0, lat0 = cfg["origin"]
    header = {
        "format": FORMAT,
        "city": city,
        "built": datetime.date.today().isoformat(),
        "origin": {"lon": lon0, "lat": lat0},
        "h3_res": 9,
        "cells": int(len(order)),
        "arrays": entries,
        "totals": totals,
        "sources": sources,
    }
    if mask is not None:
        header["water"] = mask.header(mask_info)
    if commute is not None:
        header["commute"] = commute
    stem = os.path.join(WORK, f"{city}.cells")
    with open(stem + ".bin.part", "wb") as f:
        f.write(buf)
    os.replace(stem + ".bin.part", stem + ".bin")
    with open(stem + ".json.part", "w") as f:
        json.dump(header, f, indent=2)
    os.replace(stem + ".json.part", stem + ".json")
    return stem + ".json"


def add_gravity(cells_json, city):
    """Solve the 2 km zone gravity with the sim crate's own code (sim/src/bin/pack_gravity.rs)
    and write the finished pack to data/packs/. One implementation of the gravity: the game
    rebuilds zone trips from these factors with the same decay table (notes/T-019.md)."""
    env = dict(os.environ)
    cargo_bin = os.path.join(os.path.expanduser("~"), ".cargo", "bin")
    env["PATH"] = cargo_bin + os.pathsep + env.get("PATH", "")
    env.setdefault("CARGO_BUILD_JOBS", "2")
    # its own target dir, so the pipeline never waits on (or blocks) the app's wasm builds
    env["CARGO_TARGET_DIR"] = os.path.join(WORK, "target")
    out_json = os.path.join(PACKS, f"{city}.json")
    cargo = shutil.which("cargo", path=env["PATH"])  # Windows does not search env's PATH itself
    if not cargo:
        sys.exit("cargo not found: install Rust (rustup) or put ~/.cargo/bin on PATH")
    cmd = [cargo, "run", "--quiet", "--release", "--features", "demand", "--bin", "pack_gravity",
           "--manifest-path", os.path.join(ROOT, "sim", "Cargo.toml"), "--", cells_json, out_json]
    subprocess.run(cmd, env=env, check=True)
    with open(os.path.join(PACKS, f"{city}.bin"), "rb") as f:
        buf = f.read()
    gz = len(gzip.compress(buf, 6))
    print(f"pack: bin {len(buf) / 1e6:.2f} MB raw, {gz / 1e6:.2f} MB gzipped -> {out_json}")
    return len(buf), gz


def checks(cfg, county_pop, county_jobs, lodes_year):
    """Compare against published county figures; prints, returns nothing."""
    counties = cfg["counties"]
    try:
        pe = pd.read_csv(fetch(POPEST), encoding="latin-1", dtype={"STATE": str, "COUNTY": str})
        pe["fips"] = pe["STATE"] + pe["COUNTY"]
        base = pe.set_index("fips")["ESTIMATESBASE2020"]
        pub = sum(int(base[c]) for c in counties)
        ours = sum(county_pop.get(c, 0) for c in counties)
        print(f"check pop: blocks {ours:,} vs Census estimates base 2020 {pub:,} "
              f"({(ours - pub) / pub:+.3%}); worst counties:")
        diffs = sorted(((county_pop.get(c, 0) - int(base[c]), c) for c in counties), key=lambda t: -abs(t[0]))
        for d, c in diffs[:4]:
            print(f"    {counties[c]:<14} blocks {county_pop.get(c, 0):>10,}  base {int(base[c]):>10,}  {d:+,}")
    except Exception as e:  # a check, never a build blocker
        print(f"check pop: skipped ({e})")
    try:
        q = pd.read_csv(fetch(QCEW_TOTAL.format(year=lodes_year), f"qcew_{lodes_year}_a_industry10.csv"),
                        dtype={"area_fips": str})
        q = q[(q["own_code"] == 0) & q["area_fips"].isin(counties)]
        pub = int(q["annual_avg_emplvl"].sum())
        ours = sum(county_jobs.get(c, 0) for c in counties)
        print(f"check jobs: LODES {lodes_year} {ours:,} vs BLS QCEW {lodes_year} annual average "
              f"{pub:,} ({(ours - pub) / pub:+.1%}; {len(q)} of {len(counties)} counties in QCEW)")
    except Exception as e:
        print(f"check jobs: skipped ({e})")


def main(city):
    t0 = time.time()
    cfg = CITIES[city]
    for d in (RAW, WORK, PACKS):
        os.makedirs(d, exist_ok=True)
    county_set = set(cfg["counties"])
    state_fips = sorted({c[:2] for c in county_set})

    boundary = build_boundary(city, cfg)
    mask, mask_info = build_water(cfg, boundary)

    blocks, jobs, years, sources = [], [], set(), []
    for fips in state_fips:
        b, n_all = blocks_for_state(fips, county_set)
        print(f"blocks {STATES[fips]}: {len(b):,} of {n_all:,} in the county set, pop {b['pop'].sum():,}")
        blocks.append(b)
        w, y, url = jobs_for_state(STATES[fips], cfg["lodes_years"], county_set)
        print(f"jobs {STATES[fips]}: LODES {y}, {len(w):,} blocks, {w['jobs'].sum():,} jobs")
        jobs.append(w)
        years.add(y)
    blocks = pd.concat(blocks, ignore_index=True)
    jobs = pd.concat(jobs, ignore_index=True)
    lodes_year = max(years)
    if len(years) > 1:
        print(f"WARNING: mixed LODES years {sorted(years)}")

    m = jobs.merge(blocks[["geoid"]], on="geoid", how="left", indicator=True)
    lost = m[m["_merge"] == "left_only"]
    print(f"LODES blocks with no 2020 block centroid: {len(lost):,} blocks, {lost['jobs'].sum():,} jobs lost")
    df = blocks.merge(jobs, on="geoid", how="left")
    # T-054: where the job holders live (LODES RAC), the home end of the commute
    rac = pd.concat([residents_for_state(STATES[f], lodes_year, county_set)[0] for f in state_fips], ignore_index=True)
    df = df.merge(rac, on="geoid", how="left")
    print(f"residents (LODES RAC {lodes_year}): {rac['rac'].sum():,} jobs by home block, "
          f"{df['rac'].sum():,.0f} on blocks with a centroid")
    for k in ["jobs", "C000", "CE01", "CE03", "rac"] + wfh.CNS:
        df[k] = df[k].fillna(0).astype(np.int64)
    df = df[(df["pop"] > 0) | (df["jobs"] > 0) | (df["rac"] > 0)]

    county_pop = blocks.groupby(blocks["geoid"].str[:5])["pop"].sum().to_dict()
    county_jobs = df.groupby(df["geoid"].str[:5])["jobs"].sum().to_dict()

    df = df.reset_index(drop=True)
    df.to_csv(os.path.join(WORK, f"{city}.blocks.csv.gz"), index=False)  # for T-014/T-006 analysis
    lon0, lat0 = cfg["origin"]
    # before T-013: each block whole in the cell of its internal point (kept for the comparison)
    cell_str = [h3.latlng_to_cell(la, lo, 9) for la, lo in zip(df["lat"].to_numpy(), df["lon"].to_numpy())]
    ids = np.array([h3.str_to_int(c) for c in cell_str], dtype=np.uint64)
    pt_uniq, inv = np.unique(ids, return_inverse=True)
    pt_pop = np.bincount(inv, weights=df["pop"].to_numpy(), minlength=len(pt_uniq))
    pt_job = np.bincount(inv, weights=df["jobs"].to_numpy(), minlength=len(pt_uniq))
    print(f"internal points: {len(pt_uniq):,} cells, {int((pt_pop > 0).sum()):,} with people, "
          f"{int((pt_job > 0).sum()):,} with jobs")
    # T-013: spread over the block polygon's land
    # T-042: commuters only (work from home makes no commute), ACS 2019-2023
    home_c, work_c, rates, wstats = wfh.commuters(df, RAW, fetch)
    df["home_c"], df["work_c"] = home_c, work_c
    ts = wstats["tracts"]
    print(f"work from home: {ts['tracts']:,} tracts, {ts['worked_from_home']:,.0f} of {ts['workers']:,.0f} "
          f"resident workers ({ts['region_share']:.1%}); blocks without a tract {ts['blocks_without_tract']} "
          f"({ts['pop_without_tract']:,.0f} people, given their county's share)")
    for r in wstats["industries"]:
        print(f"    {r['industry']:<42} {r['share']:6.1%} of {r['workers']:>9,}")
    adj_df, adj_log = adjust_jobs(df, cfg, rates)
    uniq, vals, st = spread.spread(adj_df, county_set, fetch, mask, lon0, lat0,
                                   cols=("pop", "jobs", "home_c", "work_c"))
    pop, job, hc, wc = vals["pop"], vals["jobs"], vals["home_c"], vals["work_c"]
    print(f"spread: {st}")
    keep = (pop > 0) | (job > 0)
    uniq, pop, job, hc, wc = uniq[keep], pop[keep], job[keep], hc[keep], wc[keep]
    commute_by_county(city, cfg, adj_df)
    print(f"spread: {len(uniq):,} cells, {int((pop > 0).sum()):,} with people, {int((job > 0).sum()):,} with jobs")
    ll = np.array([h3.cell_to_latlng(h3.int_to_str(int(c))) for c in uniq])
    x = R_EARTH * np.radians(ll[:, 1] - lon0) * np.cos(np.radians(lat0))
    y = R_EARTH * np.radians(ll[:, 0] - lat0)

    totals = {"pop": int(round(pop.sum())), "jobs": int(round(job.sum())),
              "commute_home": int(round(hc.sum())), "commute_work": int(round(wc.sum()))}
    sources = [
        {"what": "population", "name": "2020 Census, TIGER/Line TABBLOCK20 POP20, spread over each block polygon (TIGER2020PL) by land area",
         "url": "https://www2.census.gov/geo/tiger/TIGER2020/TABBLOCK20/"},
        {"what": "jobs", "name": f"LEHD LODES 8 WAC S000 JT00 {lodes_year}, C000 (all jobs)",
         "url": "https://lehd.ces.census.gov/data/lodes/LODES8/"},
        {"what": "boundary", "name": "Census cb_2020_us_county_500k, counties in notes/T-003.md",
         "url": COUNTIES},
    ]
    sources.append({"what": "water", "name": "TIGER/Line 2020 AREAWATER by county, and the sea outside "
                    "the TIGER 2020 state polygons", "url": "https://www2.census.gov/geo/tiger/TIGER2020/AREAWATER/"})
    sources.append({"what": "work from home", "name": "ACS 2019-2023 5-year, B08301 by tract (home end) and "
                    "B08126 by industry over the city's tracts (work end), table-based summary files",
                    "url": "https://www2.census.gov/programs-surveys/acs/summary_file/2023/table-based-SF/"})
    sources.append({"what": "home end of commutes", "name": f"LEHD LODES 8 RAC S000 JT00 {lodes_year}, C000 (jobs by home block)",
                    "url": "https://lehd.ces.census.gov/data/lodes/LODES8/"})
    commute = {"home": "LODES RAC jobs by home block x (1 - tract work-from-home share)",
               "work": "sum over LODES sectors of jobs x (1 - industry work-from-home share)",
               "region_share": round(ts["region_share"], 4),
               "industries": wstats["industries"]}
    cells_json = write_cells(city, cfg, {"h3": uniq, "x_m": x, "y_m": y, "pop": pop, "jobs": job,
                                         "commute_home": hc, "commute_work": wc},
                             totals, sources, mask, mask_info, commute)
    nbytes, gz = add_gravity(cells_json, city)

    print(f"totals: pop {totals['pop']:,}, jobs {totals['jobs']:,}")
    checks(cfg, county_pop, county_jobs, lodes_year)
    n = len(uniq)
    both = int(((pop > 0) & (job > 0)).sum())
    print(f"cells: {n:,}; pop only {int(((pop > 0) & (job == 0)).sum()) / n:.1%}, "
          f"jobs only {int(((pop == 0) & (job > 0)).sum()) / n:.1%}, both {both / n:.1%}")
    for label, v in (("pop", pop), ("jobs", job)):
        print(f"top cells by {label}:")
        for i in np.argsort(-v)[:8]:
            print(f"    {h3.int_to_str(int(uniq[i]))}  {ll[i, 0]:.4f},{ll[i, 1]:.4f}  "
                  f"pop {pop[i]:>8,.0f}  jobs {job[i]:>8,.0f}")
    print(f"done in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "nyc")
