"""T-1016: the United States's inputs for a planned row, a metro or a countryside zone, through
tier0's hooks (notes/T-1004.md section 2), from the sources and readers the 13 built US packs use
(build_city's `blocks_for_state` and `jobs_for_state`, group_quarters, the ACS summary file):

- **Outline**: the drawn boundary is the row's TIGER/Line 2020 counties, water out to the state lines,
  simplified topologically once for the country (tiger_counties.py), so it meets a built pack's border
  exactly; the footprint the build filters by is the cartographic counties (cb_2020_us_county_500k,
  clipped to the shore), as build_city's.
- **People**: 2020 census blocks (TIGER/Line TABBLOCK20 POP20, the .dbf only, by HTTP range) less the
  people in institutions (PL 94-171 P5, T-234). Each block's people go over the cells nearer its
  internal point than any other block's, by GHSL; a block whose cells hold no GHSL people keeps them
  in its own cell (every block with people or jobs gets one).
- **Jobs and everyday destinations**: LEHD LODES 8 WAC by block, C000, the newest year a state has
  from 2023 (the built packs' year) back to 2016 (Michigan 2021, Alaska 2016), spread the same way by
  tier 0's jobs proxy; errands CNS07 + CNS19, leisure CNS17 + CNS18, as build_city.
- **Work from home**: ACS 2019-23 B08301, worked from home over workers, over the row's counties.
- **Curve and regions**: the US curve d^-0.8 exp(-d / 30 km) (T-198) and the state line factor 0.23
  (T-162); each cell's region is the state of its nearest block.
- **Puerto Rico** (rows of unit set PRI): outline, people and work from home; jobs, curve and one
  region are tier 0's (no LODES there).

The blocks and LODES of a state are cached together in data/work/usa/blocks_<fips>.npz (every block
of the state); the downloads stay in data/raw as build_city keeps them (the PL zip can go once its
pl2020_p5_<fips>.csv.gz exists).

    python pipeline/usa.py state <fips> [...]          make the states' block caches ahead of a run
    python pipeline/usa.py check <id> [<plan.json>]    a built pack's people and jobs by county
"""
import json
import os
import sys

import h3
import numpy as np
import pandas as pd
import shapefile  # pyshp
import shapely
from shapely import make_valid, union_all
from shapely.geometry import mapping, shape

import build_city as b
import group_quarters
import tiger_counties
import wfh as acs

CACHE = os.path.join(b.WORK, "usa")
LODES_YEARS = tuple(range(2023, 2015, -1))
SECTORS = {"errand": ["CNS07", "CNS19"], "leisure": ["CNS17", "CNS18"]}
ACS_COUNTY = "acs2023_b08301_county.csv"
POP_SOURCE = ("2020 Census, TIGER/Line TABBLOCK20 POP20 less institutional group quarters (PL 94-171 P5, T-234), by "
              "block, each block's people over the cells nearer its internal point than any other block's by GHSL "
              "GHS-POP E2020 (T-1016)")
REGION_SOURCE = "US state: the state of the 2020 census block whose internal point is nearest the cell's centre (T-1016)"
_rows = {}


# ---------------------------------------------------------------- blocks and LODES, per state

def state_blocks(fips):
    """Every 2020 block of the state: geoid, pop (less institutions), pop_census, gq_inst, lat, lon,
    and its LODES WAC jobs and sector jobs (zero in Puerto Rico); and the cache's meta."""
    path = os.path.join(CACHE, f"blocks_{fips}.npz")
    if not os.path.exists(path):
        os.makedirs(CACHE, exist_ok=True)
        # the downloads this makes go once the cache holds them (a 36 MB .dbf and a 36 MB zip for Georgia)
        name, ab = group_quarters.PL_STATES[fips]
        big = [os.path.join(b.RAW, f"tl_2020_{fips}_tabblock20.dbf"), os.path.join(b.RAW, f"{ab}2020.pl.zip")]
        big = [p for p in big if not os.path.exists(p)]
        dbf = b.fetch_zip_member(b.TIGER_BLOCKS.format(fips=fips), ".dbf")
        counties = set(np.unique(np.char.strip(b.read_dbf(dbf, ["GEOID20"])["GEOID20"]).astype("U5")).tolist())
        bl, n_all = b.blocks_for_state(fips, counties)
        meta = {"fips": fips, "blocks": int(n_all), "pop_census": int(bl["pop_census"].sum()), "gq_inst": int(bl["gq_inst"].sum())}
        cols = {"jobs": np.zeros(len(bl), np.int32), "errand": np.zeros(len(bl), np.int32), "leisure": np.zeros(len(bl), np.int32)}
        if fips != "72":
            w, year, url = b.jobs_for_state(b.STATES[fips], LODES_YEARS, counties)
            m = bl[["geoid"]].merge(w, on="geoid", how="left")
            cols["jobs"] = m["C000"].fillna(0).to_numpy(np.int32)
            for k, cs in SECTORS.items():
                cols[k] = m[cs].fillna(0).sum(axis=1).to_numpy(np.int32)
            lost = int(w["C000"].sum()) - int(cols["jobs"].sum())
            meta.update(lodes_year=int(year), lodes_url=url, lodes_jobs=int(w["C000"].sum()), lodes_lost_jobs=lost)
        np.savez_compressed(path + ".part.npz", geoid=bl["geoid"].to_numpy().astype("U15"),
                            pop=bl["pop"].to_numpy(np.int32), pop_census=bl["pop_census"].to_numpy(np.int32),
                            gq_inst=bl["gq_inst"].to_numpy(np.int32), lat=bl["lat"].to_numpy(np.float64),
                            lon=bl["lon"].to_numpy(np.float64), meta=np.array(json.dumps(meta)), **cols)
        os.replace(path + ".part.npz", path)
        for p in big:
            if os.path.exists(p):
                os.remove(p)
        print(f"  usa: state {fips} cached: {meta}", flush=True)
    z = np.load(path)
    return {k: z[k] for k in z.files if k != "meta"}, json.loads(str(z["meta"]))


class RowBlocks:
    """The blocks of a row's counties, with the cells they spread over once the signals exist."""

    def __init__(self, counties):
        parts, self.meta = [], {}
        want = np.array(sorted(counties), dtype="U5")
        for st in sorted({c[:2] for c in counties}):
            d, meta = state_blocks(st)
            keep = np.isin(d["geoid"].astype("U5"), want)
            parts.append({k: v[keep] for k, v in d.items()})
            self.meta[st] = meta
        self.d = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        found = set(np.unique(self.d["geoid"].astype("U5")).tolist())
        missing = sorted(set(counties) - found)
        if missing:
            raise ValueError(f"no 2020 blocks for counties {missing}")
        self.lat0 = float(np.mean(self.d["lat"]))
        self.k = np.cos(np.radians(self.lat0))
        self.tree = None
        self.cells = None

    def xy(self, lat, lon):
        return np.c_[np.asarray(lon) * self.k, np.asarray(lat)]

    def nearest_block(self, lat, lon):
        from scipy.spatial import cKDTree
        if self.tree is None:
            self.tree = cKDTree(self.xy(self.d["lat"], self.d["lon"]))
        return self.tree.query(self.xy(lat, lon))[1]

    def assign(self, sig):
        """(each signal cell's nearest block, each block's own cell): the block's res-9 cell where
        the signals have it, else the nearest cell centre (a block point outside the footprint)."""
        if self.cells is not None and len(self.cells[0]) == len(sig["h3"]):
            return self.cells
        from scipy.spatial import cKDTree
        near = self.nearest_block(sig["lat"], sig["lon"])
        idx = {int(c): i for i, c in enumerate(sig["h3"])}
        lat, lon = self.d["lat"], self.d["lon"]
        own = np.array([idx.get(h3.str_to_int(h3.latlng_to_cell(a, o, 9)), -1) for a, o in zip(lat, lon)])
        out = own < 0
        if out.any():
            own[out] = cKDTree(self.xy(sig["lat"], sig["lon"])).query(self.xy(lat[out], lon[out]))[1]
        self.cells = (near, own, int((out & ((self.d["pop"] > 0) | (self.d["jobs"] > 0))).sum()))
        return self.cells

    def spread(self, col, sig, weight):
        """Per signal cell: each block's `col` over the cells nearest its point, by weight; a block
        whose cells weigh nothing keeps it in its own cell. Returns (per cell, share kept whole)."""
        near, own, _ = self.assign(sig)
        v = self.d[col].astype(np.float64)
        w = np.asarray(weight, np.float64)
        ws = np.bincount(near, weights=w, minlength=len(v))
        out = np.zeros(len(near))
        ok = ws[near] > 0
        out[ok] = v[near[ok]] * w[ok] / ws[near[ok]]
        whole = (ws <= 0) & (v > 0)
        np.add.at(out, own[whole], v[whole])
        return out, float(v[whole].sum() / max(v.sum(), 1.0))


def row_blocks(cid, counties):
    key = (cid, tuple(sorted(counties)))
    if key not in _rows:
        _rows[key] = RowBlocks(counties)
    return _rows[key]


# ---------------------------------------------------------------- the hooks

def check_units(counties):
    bad = [c for c in counties if not (len(c) == 5 and c.isdigit())]
    if bad:
        raise ValueError(f"usa: plan units must be 2020 county GEOIDs, not {bad[:5]}")


def peopled(cid, counties, geom):
    """geom without its parts that hold no block with people or jobs (the Northwestern Hawaiian
    Islands in Honolulu County, empty islets), as osm_units drops pieces with nobody in them."""
    parts = list(getattr(geom, "geoms", [geom]))
    if len(parts) == 1:
        return geom
    lat, lon = block_points(cid, counties)
    keep = [p for p in parts if shapely.contains_xy(p, lon, lat).any()]
    return union_all(keep) if keep else geom


def footprint_file(cid, counties):
    """The row's cartographic counties (cb_2020_us_county_500k, clipped to the shore) as GeoJSON, for
    tier0's `boundary_file`; its path. Rewritten each build, so a replanned row never reads old units."""
    r = shapefile.Reader(b.fetch(b.COUNTIES))
    fields = [f[0] for f in r.fields[1:]]
    gi, ni = fields.index("GEOID"), fields.index("NAME")
    feats = [{"type": "Feature", "properties": {"geoid": sr.record[gi], "name": sr.record[ni]},
              "geometry": mapping(peopled(cid, counties, make_valid(shape(sr.shape.__geo_interface__))))}
             for sr in r.iterShapeRecords() if sr.record[gi] in counties]
    missing = sorted(set(counties) - {f["properties"]["geoid"] for f in feats})
    if missing:
        raise ValueError(f"counties not in {b.COUNTIES}: {missing}")
    out = os.path.join(b.WORK, f"{cid}.counties.geojson")
    os.makedirs(b.WORK, exist_ok=True)
    with open(out + ".part", "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": feats}, f)
    os.replace(out + ".part", out)
    return out


def drawn(cid, counties):
    """The drawn outline: the row's simplified TIGER/Line counties, dissolved (as build_city's), less
    parts with nobody in them."""
    geoms = []
    for st in sorted({c[:2] for c in counties}):
        tiger = tiger_counties.counties(b.fetch, st if st in tiger_counties.OWN_STATES else None)
        absent = [c for c in counties if c[:2] == st and c not in tiger]
        assert not absent, f"TIGER counties missing {absent}"
        geoms += [tiger[c] for c in counties if c[:2] == st]
    return peopled(cid, counties, union_all(geoms)), (f"Census TIGER/Line 2020 counties tl_2020_us_county, dissolved, simplified "
                              f"{tiger_counties.TOLERANCE_DEG} degrees (T-352): {len(counties)} counties")


def block_points(cid, counties):
    rb = row_blocks(cid, counties)
    sel = (rb.d["pop"] > 0) | (rb.d["jobs"] > 0)
    return rb.d["lat"][sel], rb.d["lon"][sel]


def population(cid, counties, sig):
    rb = row_blocks(cid, counties)
    out, whole = rb.spread("pop", sig, sig["pop_ghsl"])
    d = rb.d
    # the block table build_city keeps beside a US pack, which nocar.py reads for each cell's tract
    sel = (d["pop"] > 0) | (d["jobs"] > 0)
    pd.DataFrame({k: d[k][sel] for k in ("geoid", "pop", "pop_census", "gq_inst", "jobs", "lat", "lon")}).to_csv(
        os.path.join(b.WORK, f"{cid}.blocks.csv.gz"), index=False)
    print(f"  usa: {len(d['geoid']):,} blocks in {len(counties)} counties, census {int(d['pop_census'].sum()):,}, "
          f"{int(d['gq_inst'].sum()):,} in institutions -> {out.sum():,.0f} people (GHSL {sig['pop_ghsl'].sum():,.0f}); "
          f"{whole:.1%} kept whole in their block's cell; {rb.assign(sig)[2]:,} blocks with people or jobs outside the "
          f"footprint, in their nearest cell", flush=True)
    return out, POP_SOURCE


def lodes_years(rb):
    ys = {st: m["lodes_year"] for st, m in rb.meta.items() if "lodes_year" in m}
    top = max(ys.values())
    odd = [f"{b.STATES[st].upper()} {y}" for st, y in sorted(ys.items()) if y != top]
    return f"{top}" + (f" ({', '.join(odd)})" if odd else "")


def jobs(cid, counties, sig, weight):
    rb = row_blocks(cid, counties)
    out, whole = rb.spread("jobs", sig, weight)
    sectors = {k: rb.spread(k, sig, weight)[0] for k in SECTORS}
    np.savez_compressed(os.path.join(b.WORK, f"{cid}.usa_sectors.npz"), h3=sig["h3"], jobs=out, **sectors)
    years = lodes_years(rb)
    lost = sum(m.get("lodes_lost_jobs", 0) for m in rb.meta.values())
    print(f"  usa: LODES {years}: {out.sum():,.0f} jobs; {whole:.1%} kept whole in their block's cell; "
          f"{lost:,} jobs in the states' WAC on no 2020 block", flush=True)
    return out, (f"LEHD LODES 8 WAC S000 JT00 {years}, C000 (all jobs) by 2020 block, each block's over the cells "
                 f"nearer its internal point than any other block's (T-1016)")


def everyday(cid, lat, lon, jobs_cell):
    """tier0's `everyday` hook: LODES's customer-facing sectors per pack cell, from the jobs step."""
    z = np.load(os.path.join(b.WORK, f"{cid}.usa_sectors.npz"))
    idx = {int(c): i for i, c in enumerate(z["h3"])}
    rows = np.array([idx.get(h3.str_to_int(h3.latlng_to_cell(a, o, 9)), -1) for a, o in zip(lat, lon)])
    out = {}
    for k, cs in SECTORS.items():
        v = np.where(rows >= 0, z[k][np.maximum(rows, 0)], 0.0)
        out[k] = (v, f"LEHD LODES 8 WAC by block, {' + '.join(cs)}", "LODES, used as is by every built US pack")
    return out


def acs_county():
    """ACS 2019-23 B08301 workers (E001) and worked from home (E021) for every county, cut once from
    the nationwide table-based summary file."""
    path = os.path.join(CACHE, ACS_COUNTY)
    if not os.path.exists(path):
        os.makedirs(CACHE, exist_ok=True)
        src = b.fetch(acs.ACS_SF.format(t="b08301"))
        parts = []
        for ch in pd.read_csv(src, sep="|", usecols=["GEO_ID", "B08301_E001", "B08301_E021"], dtype={"GEO_ID": str}, chunksize=200_000):
            parts.append(ch[ch["GEO_ID"].str.startswith("0500000US")])
        t = pd.concat(parts, ignore_index=True)
        t.insert(0, "fips", t.pop("GEO_ID").str[9:14])
        t.to_csv(path + ".part", index=False)
        os.replace(path + ".part", path)
    return pd.read_csv(path, dtype={"fips": str}).set_index("fips")


def wfh_share(counties):
    t = acs_county()
    have = [c for c in counties if c in t.index]
    if len(have) < len(counties):
        print(f"  usa: no ACS 2019-23 B08301 for {sorted(set(counties) - set(have))}", flush=True)
    s = t.loc[have]
    share = float(s["B08301_E021"].sum() / max(s["B08301_E001"].sum(), 1))
    print(f"  usa: work from home {share:.1%} (ACS 2019-23 B08301 over {len(have)} counties, "
          f"{int(s['B08301_E001'].sum()):,} workers)", flush=True)
    return share


def regions(cid, counties, lat, lon):
    """tier0's `regions` hook: each pack cell's state, by its nearest block."""
    rb = row_blocks(cid, counties)
    st = rb.d["geoid"][rb.nearest_block(lat, lon)].astype("U2")
    codes = sorted(set(st.tolist()))
    region = np.searchsorted(np.array(codes), st).astype(np.uint8)
    return region, {"level": "admin-1", "ids": codes, "names": [b.STATES[c].upper() for c in codes], "source": REGION_SOURCE}


def plan_hooks(row):
    """T-1016: the country keys for any planned US row (unit set USA) or Puerto Rico row (PRI)."""
    cid, units = row["id"], row["units"]
    counties = sorted(units.get("USA", []) + units.get("PRI", []))
    check_units(counties)
    keys = {
        "boundary_file": lambda: footprint_file(cid, counties),
        "boundary_drawn": lambda: drawn(cid, counties),
        "cell_points": lambda boundary: block_points(cid, counties),
        "population": lambda city, boundary, sig: population(cid, counties, sig),
        "wfh": lambda: wfh_share(counties),
    }
    if "USA" not in units:
        return keys
    return {**keys,
            "jobs": lambda city, boundary, sig, weight: jobs(cid, counties, sig, weight),
            "jobs_url": "https://lehd.ces.census.gov/data/lodes/LODES8/",
            "everyday": lambda city, lat, lon, j: everyday(cid, lat, lon, j),
            "regions": lambda lat, lon: regions(cid, counties, lat, lon),
            "decay_pow": b.DECAY_POW_US, "decay_km": b.DECAY_KM_US, "border_factor": b.BORDER_FACTOR_US}


# ---------------------------------------------------------------- check

def check(cid, plan=None):
    """A built pack's people and jobs by county (each res-9 cell's centre in the cartographic
    counties) against its blocks' census and LODES figures, and the row's totals."""
    import plan_config
    from read_pack import load_pack
    rows, _ = plan_config.load(plan or plan_config.PLAN)
    row = next(r for r in rows if r["id"] == cid)
    counties = sorted(row["units"].get("USA", []) + row["units"].get("PRI", []))
    header, a = load_pack(cid)
    if "merged" in header:  # merged cells span counties: the res-9 copy merge_cells keeps
        header, a = load_pack(cid, os.path.join(b.WORK, "res9"))
    rb = row_blocks(cid, counties)
    d = rb.d
    cty = d["geoid"].astype("U5")
    with open(footprint_file(cid, counties), encoding="utf-8") as f:
        feats = json.load(f)["features"]
    ll = np.array([h3.cell_to_latlng(h3.int_to_str(int(c))) for c in a["h3"]])
    pts = shapely.points(ll[:, 1], ll[:, 0])
    where = np.full(len(pts), "", dtype="U5")
    for ft in feats:
        g = shape(ft["geometry"])
        shapely.prepare(g)
        where[shapely.contains(g, pts)] = ft["properties"]["geoid"]
    names = {ft["properties"]["geoid"]: ft["properties"]["name"] for ft in feats}
    rows_out = []
    for c in counties:
        rows_out.append((c, names[c], int(d["pop"][cty == c].sum()), float(a["pop"][where == c].sum(dtype=np.float64)),
                         int(d["jobs"][cty == c].sum()), float(a["jobs"][where == c].sum(dtype=np.float64))))
    t = pd.DataFrame(rows_out, columns=["county", "name", "census_pop", "pack_pop", "lodes_jobs", "pack_jobs"])
    P, J = float(a["pop"].sum(dtype=np.float64)), float(a["jobs"].sum(dtype=np.float64))
    print(f"{cid}: people {P:,.0f} against blocks {int(d['pop'].sum()):,} less institutions "
          f"({P / max(d['pop'].sum(), 1) - 1:+.2%}; census {int(d['pop_census'].sum()):,}); jobs {J:,.0f} against LODES "
          f"{int(d['jobs'].sum()):,} ({J / max(d['jobs'].sum(), 1) - 1:+.2%}); cells in no county "
          f"{int((where == '').sum())} holding {float(a['pop'][where == ''].sum()):,.0f} people")
    for col, ref in (("pack_pop", "census_pop"), ("pack_jobs", "lodes_jobs")):
        t[col + "_diff"] = t[col] / t[ref].clip(lower=1) - 1
        worst = t.reindex(t[col + "_diff"].abs().sort_values(ascending=False).index).head(5)
        print(f"  worst counties by {col[5:]}: " + "; ".join(
            f"{r['name']} {r[ref]:,.0f} -> {r[col]:,.0f} ({r[col + '_diff']:+.1%})" for _, r in worst.iterrows()))
    return t


if __name__ == "__main__":
    if sys.argv[1] == "state":
        for st in sys.argv[2:]:
            print(st, state_blocks(st)[1])
    elif sys.argv[1] == "check":
        check(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
