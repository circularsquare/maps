"""NYC metro public-school attendance inputs: DOE buildings and NCES schools.

Append to the existing NYC pack, preserving H3 and all established demand arrays.
DOE locations: 2024 school point file. Enrollment: dated rows in the current building report.
NCES 2023-24 operational, non-full-virtual public schools extend coverage outside
the five boroughs. Origins use 2020 block residents times ACS 2023 ages 3-17 shares
(ages 3-4 approximated by two fifths of under-fives) throughout the metro.
"""
import json
import os
import zipfile
from urllib.parse import urlencode
import h3
import numpy as np
import pandas as pd
import shapefile
import build_city as b
import spread
import education_sources as education
from shapely.geometry import shape
from shapely import union_all

BORO_COUNTIES = {"36005", "36047", "36061", "36081", "36085"}
LOCATION_ID = "jfju-ynrr"
ENROLLMENT_ID = "gkd7-3vk7"


def inputs():
    metadata = json.load(open(b.fetch(f"https://data.cityofnewyork.us/api/views/{LOCATION_ID}.json", "nyc.school-locations.metadata.json")))
    blob = metadata["blobId"]
    points = b.fetch(f"https://data.cityofnewyork.us/api/views/{LOCATION_ID}/files/{blob}", "nyc.school-locations.2024.zip")
    with zipfile.ZipFile(points) as z:
        files = {os.path.splitext(n)[1].lower(): n for n in z.namelist()}
        import io
        reader = shapefile.Reader(shp=io.BytesIO(z.read(files[".shp"])), shx=io.BytesIO(z.read(files[".shx"])), dbf=io.BytesIO(z.read(files[".dbf"])))
        fields = [f[0] for f in reader.fields[1:]]
        records = [dict(zip(fields, r.record)) | {"point": r.shape.points[0]} for r in reader.iterShapeRecords()]
    url = f"https://data.cityofnewyork.us/resource/{ENROLLMENT_ID}.json?" + urlencode({"$limit": 50000})
    enroll = pd.DataFrame(json.load(open(b.fetch(url, "nyc.school-enrollment.json"))))
    return records, enroll


def main():
    cfg = b.CITIES["nyc"]
    boundary = b.build_boundary("nyc", cfg)
    reader = shapefile.Reader(b.fetch(b.COUNTIES))
    boroughs = union_all([shape(r.shape.__geo_interface__) for r in reader.iterShapeRecords()
                         if r.record.as_dict()["GEOID"] in BORO_COUNTIES])
    records, enroll = inputs()
    date = enroll.data_as_of.max()
    current = enroll[enroll.data_as_of == date].copy()
    current["bldg_enroll"] = pd.to_numeric(current.bldg_enroll, errors="raise")
    unknown_enrollment = int(current.bldg_enroll.isna().sum())
    current = current[current.bldg_enroll.notna() & (current.bldg_enroll > 0)].copy()
    # A shared building appears once per organization. Never sum its repeated building total.
    if (current.groupby("bldg_id").bldg_enroll.nunique() > 1).any():
        raise ValueError("conflicting enrollment totals within a building")
    buildings = current.drop_duplicates("bldg_id")
    locations = {}
    for r in records:
        lat, lon = float(r["Latitude"]), float(r["Longitude"])
        if not (40 < lat < 42 and -75 < lon < -73):
            continue
        key = r["Building_C"]
        if key in locations:
            a = locations[key]
            if abs(a["Latitude"]-lat)+abs(a["Longitude"]-lon) > 0.002:
                raise ValueError(f"conflicting coordinates for {key}")
        locations[key] = r
    stem = os.path.join(b.PACKS, "nyc")
    header = json.load(open(stem + ".json"))
    buf = bytearray(open(stem + ".bin", "rb").read())
    a = next(a for a in header["arrays"] if a["name"] == "h3")
    cells = np.frombuffer(buf, dtype="<u8", count=a["count"], offset=a["offset"]).copy()
    cell_index = {int(h): i for i, h in enumerate(cells)}
    school_enrollment = np.zeros(len(cells), dtype="<f4")
    missing, matched, snaps = [], [], []
    for r in buildings.itertuples():
        if r.bldg_enroll <= 0:
            continue
        p = locations.get(r.bldg_id)
        if p is None:
            missing.append({"building": r.bldg_id, "enrollment": int(r.bldg_enroll)})
            continue
        hid = h3.latlng_to_cell(p["Latitude"], p["Longitude"], 9)
        i = cell_index.get(h3.str_to_int(hid))
        if i is None:
            candidates = [c for c in h3.grid_disk(hid, 3) if h3.str_to_int(c) in cell_index]
            if not candidates:
                raise ValueError(f"no city cell near school {r.bldg_id}")
            hid = min(candidates, key=lambda c: sum((v-w)**2 for v,w in zip(h3.cell_to_latlng(c), (p["Latitude"],p["Longitude"]))))
            lat,lon = h3.cell_to_latlng(hid)
            distance = b.R_EARTH*np.sqrt(np.radians(lat-p["Latitude"])**2+(np.cos(np.radians(lat))*np.radians(lon-p["Longitude"]))**2)
            if distance > 500:
                raise ValueError(f"school {r.bldg_id} needs {distance:.0f} m snap")
            snaps.append({"building": r.bldg_id, "distance_m": round(float(distance), 1)})
            i = cell_index[h3.str_to_int(hid)]
        school_enrollment[i] += r.bldg_enroll
        matched.append({"building": r.bldg_id, "name": r.bldg_name, "enrollment": int(r.bldg_enroll), "lat": p["Latitude"], "lon": p["Longitude"], "cell": i})
    coverage = float(school_enrollment.sum()) / float(buildings.bldg_enroll.sum())
    if coverage < 0.90:
        raise ValueError(f"only {coverage:.1%} of enrollment has matched locations")
    print(f"school buildings: {len(matched)} matched, enrollment {school_enrollment.sum():,.0f}, coverage {coverage:.1%}, {len(snaps)} snapped", flush=True)
    doe_enrolled = float(school_enrollment.sum())
    doe_buildings = len(matched)
    for r in matched:
        r["source"] = "NYC DOE"
    additional, excluded = education.public_schools(boundary, boroughs)
    omitted = []
    for r in additional:
        hid = h3.latlng_to_cell(r["lat"], r["lon"], 9)
        i = cell_index.get(h3.str_to_int(hid))
        if i is None:
            candidates = [c for c in h3.grid_disk(hid, 3) if h3.str_to_int(c) in cell_index]
            def metres(c):
                lat, lon = h3.cell_to_latlng(c)
                return float(b.R_EARTH * np.hypot(np.radians(lat-r["lat"]), np.cos(np.radians(lat))*np.radians(lon-r["lon"])))
            nearest = min(candidates, key=metres) if candidates else None
            if nearest is None or metres(nearest) > 500:
                omitted.append(r | {"reason": "no_existing_cell_within_500m"})
                continue
            i = cell_index[h3.str_to_int(nearest)]
            snaps.append({"building": r["building"], "distance_m": round(metres(nearest), 1)})
        school_enrollment[i] += r["enrollment"]
        matched.append(r | {"cell": i})
    nces_total = sum(r["enrollment"] for r in additional)
    nces_matched = sum(r["enrollment"] for r in matched if r["source"] != "NYC DOE")
    if not nces_total or nces_matched / nces_total < 0.95:
        raise ValueError("NCES location match coverage below 95%")
    df = pd.read_csv(os.path.join(b.WORK, "nyc.blocks.csv.gz"), dtype={"geoid": str})
    df = df[df.geoid.str[:5].isin(cfg["counties"])].copy()
    df["school_children"], fallback_blocks = education.child_weights(df)
    mask, _ = b.build_water(cfg, boundary)
    ids, vals, stats = spread.spread(df, set(cfg["counties"]), b.fetch, mask, *cfg["origin"], cols=("school_children",))
    index = np.searchsorted(cells, ids)
    hit = (index < len(cells)) & (cells[np.minimum(index,len(cells)-1)] == ids)
    if vals["school_children"][~hit].sum() > 0.01:
        raise ValueError("child population outside existing cell pack")
    children = np.zeros(len(cells), dtype="<f4")
    children[index[hit]] = vals["school_children"][hit]
    buf = education.replace_cell_arrays(header, buf, {"school_children": children, "school_enrollment": school_enrollment})
    header["school"] = {
        "coverage": "Matched NYC DOE buildings in five boroughs; NCES public schools elsewhere in NYC metro; no private schools or universities",
        "enrollment_as_of": date, "nces_school_year": "2023-2024",
        "enrolled": float(school_enrollment.sum()), "matched_buildings": len(matched),
        "doe_enrolled": doe_enrolled, "doe_buildings": doe_buildings,
        "doe_location_match_share": coverage, "nces_enrolled": nces_matched,
        "nces_schools": len(matched)-doe_buildings, "nces_location_match_share": nces_matched/nces_total,
        "location_match_share": float(school_enrollment.sum()) / (float(buildings.bldg_enroll.sum())+nces_total),
        "child_weight": float(children.sum()),
        "child_origin": "2020 block population x ACS 2023 tract ages 5-17 plus 2/5 of under-5; all NYC metro counties; missing tracts use full-FIPS county share, or metro share for Connecticut",
        "fallback_blocks": fallback_blocks, "attendance_factor": 0.9,
        "sources": [f"https://data.cityofnewyork.us/Education/School-Point-Locations/{LOCATION_ID}", f"https://data.cityofnewyork.us/Education/Enrollment-Capacity-And-Utilization-Reports/{ENROLLMENT_ID}", education.NCES, "https://api.census.gov/data/2023/acs/acs5/groups/B01001.html"]}
    with open(stem + ".bin.part", "wb") as f:
        f.write(buf)
    with open(stem + ".json.part", "w") as f:
        json.dump(header, f, indent=2)
    os.replace(stem + ".bin.part", stem + ".bin")
    os.replace(stem + ".json.part", stem + ".json")
    with open(os.path.join(b.WORK,"nyc.schools.audit.json"), "w") as f:
        json.dump({"metadata": header["school"], "missing": missing, "unknown_enrollment_rows": unknown_enrollment, "snaps": snaps, "schools": matched, "spread": stats, "nces_excluded": excluded, "nces_omitted": omitted}, f, indent=2)
    print(header["school"], flush=True)


if __name__ == "__main__":
    main()
