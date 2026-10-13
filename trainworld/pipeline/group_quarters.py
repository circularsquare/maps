"""T-234: people in institutional group quarters (prisons and jails, juvenile facilities, nursing
homes, other institutions) are not homes: the census counts them where they are held or cared for,
but they do not commute, shop or take children to school. Their number is taken off each block's
population before anything is built on it.

    python pipeline/group_quarters.py <city> [...]     apply to finished packs in place

Source: the 2020 Census redistricting file (PL 94-171) table P5, group quarters population by
major type, per block, from each state's summary file on www2.census.gov (no login or key; the
Census API refused keyless requests on 2026-10-13). Every US state has it; a small per-state table
is cached in data/raw/pl2020_p5_<st>.csv.gz and the state zip can go after. Removed: P5_003N correctional facilities for adults,
P5_004N juvenile facilities, P5_005N nursing facilities/skilled-nursing facilities, P5_006N other
institutional facilities. Kept: the non-institutional types (P5_008N college/university student
housing, P5_009N military quarters, P5_010N other non-institutional), whose residents travel.

`build_city.py` does this as it reads the blocks (`blocks_for_state`). For a finished pack this
script does the same in place: the block table's `pop` becomes the census population less the
institutional group quarters (`pop_census` and `gq_inst` kept beside it), the pack's `pop` array is
re-spread from it exactly as build_city.py spreads it, and every other array is copied bit for bit.
A cell left with no people and no jobs is dropped, as build_city.py drops it. The inputs built on
`pop` are then rebuilt (school children, university adults; the park draw is T-235) and the gravity
re-solved, so the pack is what a fresh build would give.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")

import io
import json
import sys
import zipfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PL_ZIP = "https://www2.census.gov/programs-surveys/decennial/2020/data/01-Redistricting_File--PL_94-171/{name}/{ab}2020.pl.zip"
PL_STATES = {"09": ("Connecticut", "ct"), "10": ("Delaware", "de"), "11": ("District_of_Columbia", "dc"),
             "24": ("Maryland", "md"), "25": ("Massachusetts", "ma"), "33": ("New_Hampshire", "nh"),
             "34": ("New_Jersey", "nj"), "36": ("New_York", "ny"), "42": ("Pennsylvania", "pa"),
             "44": ("Rhode_Island", "ri"), "51": ("Virginia", "va"), "54": ("West_Virginia", "wv")}
# P5 group quarters population by major type (2020 PL 94-171), and the block total for a check
ALL = ["P1_001N", "P5_001N", "P5_002N", "P5_003N", "P5_004N", "P5_005N", "P5_006N",
       "P5_007N", "P5_008N", "P5_009N", "P5_010N"]
REMOVED = {"P5_003N": "correctional facilities for adults", "P5_004N": "juvenile facilities",
           "P5_005N": "nursing and skilled-nursing facilities", "P5_006N": "other institutional facilities"}
KEPT = {"P5_008N": "college and university student housing", "P5_009N": "military quarters",
        "P5_010N": "other non-institutional facilities"}
SOURCE = ("2020 Census redistricting file (PL 94-171), table P5 group quarters population by major "
          "type, per block (state summary files): institutional types (adult correctional, juvenile, "
          "nursing, other institutional) are taken off each block's population")
SOURCE_URL = "https://www2.census.gov/programs-surveys/decennial/2020/data/01-Redistricting_File--PL_94-171/"


def state_table(st, fetch):
    """Every 2020 block of a state (2-digit FIPS) with P1_001N and P5, from the state's PL 94-171
    summary file: the geographic header (SUMLEV 750 = block, LOGRECNO, GEOCODE = block GEOID),
    segment 1 (P1) and segment 3 (P5). Kept as a small table in data/raw/pl2020_p5_<st>.csv.gz;
    the state zip is needed only to make it and can be deleted after."""
    import build_city as bc
    out = os.path.join(bc.RAW, f"pl2020_p5_{st}.csv.gz")
    if os.path.exists(out):
        return pd.read_csv(out, dtype={"geoid": str})
    name, ab = PL_STATES[st]
    zpath = fetch(PL_ZIP.format(name=name, ab=ab))
    with zipfile.ZipFile(zpath) as z:
        def member(suffix):
            m = next(m for m in z.namelist() if m.lower().endswith(suffix))
            return io.TextIOWrapper(z.open(m), encoding="latin-1")
        geo = pd.read_csv(member("geo2020.pl"), sep="|", header=None, dtype=str, usecols=[2, 7, 9],
                          names=None, quoting=3)
        geo.columns = ["sumlev", "logrecno", "geoid"]
        geo = geo[geo["sumlev"] == "750"]
        s1 = pd.read_csv(member("000012020.pl"), sep="|", header=None, dtype=str, usecols=[4, 5], quoting=3)
        s1.columns = ["logrecno", "P1_001N"]
        s3 = pd.read_csv(member("000032020.pl"), sep="|", header=None, dtype=str, usecols=[4] + list(range(5, 15)), quoting=3)
        s3.columns = ["logrecno"] + [f"P5_{i:03d}N" for i in range(1, 11)]
    d = geo.merge(s1, on="logrecno").merge(s3, on="logrecno")
    assert len(d) == len(geo), "PL segments do not cover every block"
    d = d[["geoid"] + ALL]
    for k in ALL:
        d[k] = d[k].astype(np.int64)
    assert (d["geoid"].str.len() == 15).all()
    d.to_csv(out + ".part", index=False, compression="gzip")
    os.replace(out + ".part", out)
    print(f"  PL 94-171 {ab.upper()}: {len(d):,} blocks, population {d['P1_001N'].sum():,}, "
          f"group quarters {d['P5_001N'].sum():,} ({d['P5_002N'].sum():,} institutional)", flush=True)
    return d


def by_block(counties, fetch):
    """One row per 2020 block of the counties (5-digit FIPS): geoid, every ALL column, and
    gq_inst, the sum of the REMOVED types."""
    frames = []
    for st in sorted({c[:2] for c in counties}):
        d = state_table(st, fetch)
        frames.append(d[d["geoid"].str[:5].isin(counties)])
    d = pd.concat(frames, ignore_index=True)
    d["gq_inst"] = d[list(REMOVED)].sum(axis=1)
    assert (d["P5_002N"] == d["gq_inst"]).all(), "P5 institutional types do not add up to P5_002N"
    assert (d["gq_inst"] <= d["P1_001N"]).all(), "more people in institutions than on the block"
    return d


def remove(blocks, counties, fetch):
    """`blocks` (geoid, pop = census population): returns it with pop less the institutional group
    quarters, and pop_census, gq_inst beside it; the census counts must match P1 exactly."""
    gq = by_block(counties, fetch)
    m = blocks.merge(gq[["geoid", "P1_001N", "gq_inst"]], on="geoid", how="left")
    missing = m["P1_001N"].isna()
    assert not missing.any(), f"{int(missing.sum())} blocks missing from the PL file"
    assert (m["P1_001N"].astype(np.int64).to_numpy() == blocks["pop"].to_numpy()).all(), "TIGER POP20 differs from PL P1_001N"
    out = blocks.copy()
    out["pop_census"] = blocks["pop"].to_numpy()
    out["gq_inst"] = m["gq_inst"].astype(np.int64).to_numpy()
    out["pop"] = out["pop_census"] - out["gq_inst"]
    return out


def cap_job_holders(df):
    """On a block with people in institutions, its resident job holders (LODES RAC, `rac`) can be
    no more than its remaining residents (`pop`, after `remove`). LODES puts job holders on such
    blocks (Rikers Island: 676 on blocks with 27 residents outside the jails), from administrative
    addresses and its noise; inmates and nursing-home residents hold almost no jobs. Other blocks
    keep LODES as it is. `rac_lodes` keeps the LODES figure."""
    out = df.copy()
    out["rac_lodes"] = df["rac"].to_numpy()
    inst = df["gq_inst"].to_numpy() > 0
    out["rac"] = np.where(inst, np.minimum(df["rac"].to_numpy(), df["pop"].to_numpy()), df["rac"].to_numpy())
    return out


def summary(counties, fetch, names=None):
    """Per county: census population, each group-quarters type, and what is removed."""
    gq = by_block(counties, fetch)
    g = gq.groupby(gq["geoid"].str[:5])[ALL + ["gq_inst"]].sum()
    if names:
        g.insert(0, "name", [names.get(k, k) for k in g.index])
    return g


POP_SOURCE = ("2020 Census, TIGER/Line TABBLOCK20 POP20 less institutional group quarters (PL 94-171 P5, "
              "T-234), spread over each block polygon (TIGER2020PL) by land area")
HOME_RULE = ("LODES RAC jobs by home block, capped on blocks with institutional group quarters at the "
             "block's remaining residents (T-234), x (1 - tract work-from-home share)")


def apply(city):
    """Take the institutional group quarters off a finished pack in place: `pop`, and
    `commute_home` where job holders are capped. Every other array is kept byte for byte."""
    import build_city as bc
    import regions
    import spread
    import water
    import wfh
    from read_pack import load_pack

    cfg = bc.CITIES[city]
    counties = set(cfg["counties"])
    path = os.path.join(bc.WORK, f"{city}.blocks.csv.gz")
    df = pd.read_csv(path, dtype={"geoid": str})
    if "pop_census" in df:
        print(f"{city}: block table already has the group quarters taken off; restore the pack and "
              f"data/work/pre-T234/blocks/ to apply again", flush=True)
        return False
    new = cap_job_holders(remove(df, counties, bc.fetch))
    share, _ = wfh.tract_shares(df, bc.RAW, bc.fetch)
    # old and new side by side on the old rows, spread together (the commuter home end exactly as
    # build_city.py computes it: wfh.commuters)
    o = df.assign(pop_new=new["pop"].to_numpy(),
                  home_c=df["rac"].to_numpy(np.float64) * (1.0 - share),
                  home_new=new["rac"].to_numpy(np.float64) * (1.0 - share))
    # build_city keeps a block only if it has people, jobs or resident job holders
    df = new[(new["pop"] > 0) | (new["jobs"] > 0) | (new["rac"] > 0)].reset_index(drop=True)

    header, a = load_pack(city)
    wb = header["water"]
    (x0, y0), (W, H) = wb["origin_m"], wb["size"]
    mask = water.WaterMask(x0, y0, wb["cell_m"], W, H, np.array(a["water_row"]), np.array(a["water_x"]))
    lon0, lat0 = header["origin"]["lon"], header["origin"]["lat"]
    ids, vals, stats = spread.spread(o, counties, bc.fetch, mask, lon0, lat0, cols=("pop", "pop_new", "home_c", "home_new"))
    cells = a["h3"]
    i = np.searchsorted(ids, cells)
    ok = (i < len(ids)) & (ids[np.minimum(i, len(ids) - 1)] == cells)
    assert ok.all(), f"{int((~ok).sum())} pack cells got no block; rebuild the pack"
    for col, arr in (("pop", "pop"), ("home_c", "commute_home")):
        mine = vals[col][i].astype("<f4")
        if not np.array_equal(mine, a[arr]):
            raise SystemExit(f"{city}: the block table no longer spreads to the pack's {arr} "
                             f"({int((mine != a[arr]).sum())} cells differ); rebuild the pack")
    pop0, pop1 = a["pop"].copy(), vals["pop_new"][i].astype("<f4")
    home0, home1 = a["commute_home"].copy(), vals["home_new"][i].astype("<f4")
    # a cell left with no people and no jobs goes, as build_city.py drops it
    empty = (pop1 <= 0) & (a["jobs"] <= 0)

    # a cell's region is chosen by its blocks' people, jobs and job holders: check it holds
    rdf, codes, cols = regions.add_columns(df)
    rids, rvals, _ = spread.spread(rdf, counties, bc.fetch, mask, lon0, lat0, cols=tuple(cols))
    reg = regions.of_cells(rvals, cols)[np.searchsorted(rids, cells)]
    moved = int((reg[~empty] != a["region"][~empty]).sum()) if "region" in a else 0

    stem = os.path.join(bc.PACKS, city)
    with open(stem + ".bin", "rb") as f:
        buf = bytearray(f.read())
    new_arrays = {"pop": pop1, "commute_home": home1}
    if moved:
        new_arrays["region"] = reg.astype(np.uint8)
    if empty.any():
        buf = drop_cells(header, buf, a, ~empty, new_arrays)
        dropped = [f"{int(c):x}" for c in cells[empty]]
    else:
        dropped = []
        for name, v in new_arrays.items():
            e = next(e for e in header["arrays"] if e["name"] == name)
            assert e["count"] == len(cells)
            raw = v.tobytes()
            buf[e["offset"]:e["offset"] + len(raw)] = raw
    keep = ~empty
    header["totals"]["pop"] = int(round(float(pop1[keep].sum(dtype=np.float64))))
    header["totals"]["commute_home"] = int(round(float(home1[keep].sum(dtype=np.float64))))
    for s in header["sources"]:
        if s["what"] == "population":
            s["name"] = POP_SOURCE
    header["sources"] = [s for s in header["sources"] if s["what"] != "group quarters"]
    header["sources"].append({"what": "group quarters", "name": SOURCE, "url": SOURCE_URL})
    header["commute"]["home"] = HOME_RULE
    with open(stem + ".bin.part", "wb") as f:
        f.write(buf)
    with open(stem + ".json.part", "w") as f:
        json.dump(header, f, indent=2)
    os.replace(stem + ".bin.part", stem + ".bin")
    os.replace(stem + ".json.part", stem + ".json")
    df.to_csv(path + ".part", index=False, compression="gzip")
    os.replace(path + ".part", path)
    f64 = lambda v: float(v.sum(dtype=np.float64))
    print(f"{city}: {int(new['gq_inst'].sum()):,} people in institutions off; pop {f64(pop0):,.0f} -> {f64(pop1):,.0f} "
          f"({int((pop1 != pop0).sum()):,} cells); job holders {int(new['rac_lodes'].sum()):,} -> {int(new['rac'].sum()):,}, "
          f"commute home end {f64(home0):,.0f} -> {f64(home1):,.0f} ({int((home1 != home0).sum()):,} cells); "
          f"cells whose region moved {moved}; cells dropped (no people or jobs left) {dropped}; spread {stats['seconds']} s", flush=True)
    return True


# arrays rebuilt after apply() by the steps that follow it (main), so they may differ at a dropped cell
REBUILT = {"pop", "commute_home", "school_children", "university_adults", "school_enrollment", "university_enrollment"}


def drop_cells(header, buf, a, keep, new_arrays):
    """The pack without the cells where `keep` is False: every per-cell array cut, other arrays
    copied, `new_arrays` (full length) in place of the pack's. The zone gravity is removed (the
    caller re-solves it; its zones are over the cells). A dropped cell must carry nothing but what
    is rebuilt: its jobs, sectors, medical, airport and park weights must all be zero."""
    sizes = {"f32": 4, "f64": 8, "u8": 1, "u16": 2, "u32": 4, "u64": 8, "i32": 4}
    n = int(keep.sum())
    out, arrays = bytearray(), []
    for e in header["arrays"]:
        name = e["name"]
        if name.startswith("zone"):
            continue
        data = bytes(buf[e["offset"]:e["offset"] + e["count"] * sizes[e["dtype"]]])
        if e["per"] == "cell":
            v = new_arrays[name] if name in new_arrays else np.frombuffer(data, dtype=read_dtype(e["dtype"]))
            gone = v[~keep].astype(np.float64)
            if name == "park_draw" and gone.sum() > 0:
                # relative draw of a park over an institution's grounds: it goes with the cell (a
                # fresh park fetch would put it on the park's other cells)
                print(f"  dropped cells carried park draw {gone.sum():.4f} of {v.astype(np.float64).sum():,.1f}", flush=True)
            elif name not in REBUILT and name not in ("h3", "x_m", "y_m", "region"):
                assert np.all(np.abs(gone) < 1e-3), f"dropped cell carries {name} {gone}"
            data = np.ascontiguousarray(v[keep]).tobytes()
            e = dict(e, count=n)
        out += b"\0" * (-len(out) % 8)
        arrays.append(dict(e, offset=len(out)))
        out += data
    header["arrays"] = arrays
    header["cells"] = n
    header.pop("gravity", None)
    header.pop("zones", None)
    for k in ("jobs", "commute_work"):
        if k in header["totals"] and k in a:
            header["totals"][k] = int(round(float(a[k][keep].astype(np.float64).sum())))
    return out


def read_dtype(d):
    return {"f32": "<f4", "f64": "<f8", "u8": "u1", "u16": "<u2", "u32": "<u4", "u64": "<u8", "i32": "<i4"}[d]


def main(cities):
    import build_city as bc
    import enrich_schools
    import enrich_universities
    for city in cities:
        if not apply(city):
            continue
        # the inputs built on population, then the gravity (the commute home end changed). The
        # park draw's nearby-residents term (enrich_parks.py) also reads `pop`, but it needs the
        # Overpass tiles again (an hour for nine packs); it is left to T-235, `enrich_parks.py`
        # on each pack, which reads the new `pop`.
        enrich_schools.main(city)
        enrich_universities.main(city)
        bc.add_gravity(os.path.join(bc.PACKS, f"{city}.json"), city)


if __name__ == "__main__":
    main(sys.argv[1:])

