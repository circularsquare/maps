"""Public-school attendance inputs for every US city pack: NCES public schools (T-142; New York too
since T-181, which dropped its NYC DOE building input under the "no city-specific data" rule).

`enrich_schools.py <city>`; replaces the pack's school arrays in place, preserving H3 and every
other array bit for bit. NCES EDGE/CCD 2023-24 operational, non-full-virtual public schools in the
pack's boundary. Origins use 2020 block residents times ACS 2023 ages 3-17 shares (ages 3-4
approximated by two fifths of under-fives) over the pack's counties.
"""
import json
import os
import h3
import numpy as np
import pandas as pd
import build_city as b
import spread
import education_sources as education


def snap_to_cell(r, cell_index):
    """The pack cell holding a school's point, or the nearest within 500 m (None past that)."""
    hid = h3.latlng_to_cell(r["lat"], r["lon"], 9)
    i = cell_index.get(h3.str_to_int(hid))
    if i is not None:
        return i, None
    def metres(c):
        lat, lon = h3.cell_to_latlng(c)
        return float(b.R_EARTH * np.hypot(np.radians(lat-r["lat"]), np.cos(np.radians(lat))*np.radians(lon-r["lon"])))
    candidates = [c for c in h3.grid_disk(hid, 3) if h3.str_to_int(c) in cell_index]
    nearest = min(candidates, key=metres) if candidates else None
    if nearest is None or metres(nearest) > 500:
        return None, None
    return cell_index[h3.str_to_int(nearest)], round(metres(nearest), 1)


def main(city="nyc"):
    """Any US city pack, from the national NCES layer alone (no city source)."""
    cfg = b.CITIES[city]
    boundary = b.build_boundary(city, cfg)
    stem = os.path.join(b.PACKS, city)
    header = json.load(open(stem + ".json"))
    buf = bytearray(open(stem + ".bin", "rb").read())
    a = next(a for a in header["arrays"] if a["name"] == "h3")
    cells = np.frombuffer(buf, dtype="<u8", count=a["count"], offset=a["offset"]).copy()
    cell_index = {int(h): i for i, h in enumerate(cells)}
    school_enrollment = np.zeros(len(cells), dtype="<f4")
    records, excluded = education.public_schools(boundary)
    matched, omitted, snaps = [], [], []
    for r in records:
        i, snap = snap_to_cell(r, cell_index)
        if i is None:
            omitted.append(r | {"reason": "no_existing_cell_within_500m"})
            continue
        if snap is not None:
            snaps.append({"building": r["building"], "distance_m": snap})
        school_enrollment[i] += r["enrollment"]
        matched.append(r | {"cell": i})
    total = sum(r["enrollment"] for r in records)
    share = float(school_enrollment.sum(dtype=np.float64)) / total if total else 0.0
    if share < 0.95:
        raise ValueError(f"NCES location match coverage {share:.1%} below 95%")
    print(f"NCES schools: {len(matched)} matched, enrollment {school_enrollment.sum(dtype=np.float64):,.0f} ({share:.2%}), {len(snaps)} snapped, {len(omitted)} omitted", flush=True)
    df = pd.read_csv(os.path.join(b.WORK, city + ".blocks.csv.gz"), dtype={"geoid": str})
    df = df[df.geoid.str[:5].isin(cfg["counties"])].copy()
    df["school_children"], fallback_blocks = education.child_weights(df)
    mask, _ = b.build_water(cfg, boundary)
    ids, vals, stats = spread.spread(df, set(cfg["counties"]), b.fetch, mask, *cfg["origin"], cols=("school_children",))
    index = np.searchsorted(cells, ids)
    hit = (index < len(cells)) & (cells[np.minimum(index, len(cells)-1)] == ids)
    if vals["school_children"][~hit].sum() > 0.01:
        raise ValueError("child population outside existing cell pack")
    children = np.zeros(len(cells), dtype="<f4")
    children[index[hit]] = vals["school_children"][hit]
    buf = education.replace_cell_arrays(header, buf, {"school_children": children, "school_enrollment": school_enrollment})
    header["school"] = {
        "coverage": "NCES public schools in the city pack's counties; no private schools or universities",
        "nces_school_year": "2023-2024",
        "enrolled": float(school_enrollment.sum(dtype=np.float64)), "matched_buildings": len(matched),
        "nces_enrolled": float(school_enrollment.sum(dtype=np.float64)), "nces_schools": len(matched),
        "nces_location_match_share": share, "location_match_share": share,
        "child_weight": float(children.sum(dtype=np.float64)),
        "child_origin": "2020 block population x ACS 2023 tract ages 5-17 plus 2/5 of under-5; missing tracts use full-FIPS county share, or the pack's share for Connecticut",
        "fallback_blocks": fallback_blocks, "attendance_factor": 0.9,
        "sources": [education.NCES, "https://api.census.gov/data/2023/acs/acs5/groups/B01001.html"]}
    with open(stem + ".bin.part", "wb") as f:
        f.write(buf)
    with open(stem + ".json.part", "w") as f:
        json.dump(header, f, indent=2)
    os.replace(stem + ".bin.part", stem + ".bin")
    os.replace(stem + ".json.part", stem + ".json")
    with open(os.path.join(b.WORK, city + ".schools.audit.json"), "w") as f:
        json.dump({"metadata": header["school"], "snaps": snaps, "schools": matched, "spread": stats, "nces_excluded": excluded, "nces_omitted": omitted}, f, indent=2)
    print(header["school"], flush=True)


if __name__ == "__main__":
    import sys
    main(*sys.argv[1:2])
