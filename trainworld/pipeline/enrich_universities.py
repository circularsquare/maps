"""T-152: optional university attendance inputs from IPEDS 2023, for any US city pack
(New York first; the other registered cities T-142). `enrich_universities.py <city>`.

Active degree-granting reporting institutions; degree/certificate-seeking
undergraduates plus graduates, excluding exclusively distance-education students.
One enrollment budget per UNITID at its directory coordinate, never per child campus.
"""
import hashlib
import json
import os
import zipfile

import h3
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import Point

import build_city as b
import education_sources as education
import spread
import wfh

YEAR = 2023
BASE = "https://nces.ed.gov/ipeds/datacenter/data/"


def source(name):
    path = b.fetch(BASE + name + ".zip", name + ".zip")
    with zipfile.ZipFile(path) as z:
        name = next(n for n in z.namelist() if n.lower().endswith(".csv"))
        with z.open(name) as f:
            result = pd.read_csv(f, encoding="utf-8-sig", encoding_errors="replace")
    result.columns = result.columns.str.strip().str.upper()
    return result


def eligible(enrollment):
    """Eligible students by UNITID (degree-seeking undergraduates + graduates, less exclusively
    online), and the UNITIDs whose counts are invalid."""
    levels = enrollment[enrollment.EFDELEV.isin([3, 12])].copy()
    # A structural blank (A=not applicable) is zero ONLY when the reported non-distance
    # cohort already equals the total. Other missing/negative counts remain unknown.
    if {"XEFDEEXC", "EFDENON"} <= set(levels.columns):
        proven_in_person = (levels.EFDEEXC.isna() & (levels.XEFDEEXC == "A") &
                            (levels.EFDENON == levels.EFDETOT) & (levels.EFDETOT >= 0))
        levels.loc[proven_in_person, "EFDEEXC"] = 0
    # A missing/negative online count without that evidence is not zero enrollment.
    valid = (levels.EFDETOT.notna() & levels.EFDEEXC.notna() &
             (levels.EFDETOT >= 0) & (levels.EFDEEXC >= 0) & (levels.EFDEEXC <= levels.EFDETOT))
    invalid_ids = set(levels.loc[~valid, "UNITID"])
    levels = levels[valid].copy()
    totals = levels.groupby("UNITID")[["EFDETOT", "EFDEEXC"]].sum()
    totals["eligible"] = totals.EFDETOT - totals.EFDEEXC
    return totals, invalid_ids


def merged_sites(directory, previous):
    """T-142: an institution that absorbed others this year (their ACT "C", NEWID pointing at it)
    reports one total at its own directory point, which put all twelve Connecticut community
    colleges (2023) at Hartford. Split its total over its own and the absorbed campuses' points
    by each one's own eligible enrollment the year before. {UNITID: [(unitid, name, lat, lon,
    share)]}; an institution without prior counts stays whole."""
    if previous is None:
        return {}
    prior, _ = eligible(previous)
    absorbed = directory[(directory.NEWID > 0) & (directory.ACT.astype(str).str.strip() == "C")]
    sites = {}
    for target, group in absorbed.groupby("NEWID"):
        rows = pd.concat([directory[directory.UNITID == target], group])
        rows = rows[np.isfinite(rows.LATITUDE) & np.isfinite(rows.LONGITUD)]
        weights = np.array([max(0.0, float(prior.eligible.get(u, 0.0))) for u in rows.UNITID])
        if len(rows) < 2 or weights.sum() <= 0:
            continue
        sites[int(target)] = [(int(r.UNITID), r.INSTNM, float(r.LATITUDE), float(r.LONGITUD), float(w / weights.sum()))
                              for r, w in zip(rows.itertuples(), weights) if w > 0]
    return sites


def institutions(directory, enrollment, boundary, previous=None):
    """Select non-overlapping level rows and exclude online-only pupils, not campuses.
    `previous`: the year before's enrollment, to split this year's mergers (`merged_sites`)."""
    if directory.UNITID.duplicated().any() or enrollment.duplicated(["UNITID", "EFDELEV"]).any():
        raise ValueError("duplicate IPEDS institution/level rows")
    totals, invalid_ids = eligible(enrollment)
    sites = merged_sites(directory, previous)
    shapely.prepare(boundary)  # T-425: the whole IPEDS directory is tested against the outline
    matched, excluded = [], []
    for r in directory.itertuples():
        if not np.isfinite(r.LATITUDE) or not np.isfinite(r.LONGITUD):
            excluded.append({"unitid": int(r.UNITID), "reason": "missing_coordinates"})
            continue
        pieces = sites.get(int(r.UNITID), [(int(r.UNITID), r.INSTNM, float(r.LATITUDE), float(r.LONGITUD), 1.0)])
        inside = [p for p in pieces if boundary.covers(Point(p[3], p[2]))]
        if not inside:
            continue
        reason = None
        if str(r.ACT).strip() not in {"A", "N", "R"} or r.CYACTIVE != 1 or r.NEWID > 0:
            reason = "child_merged_or_inactive"
        elif r.DEGGRANT != 1:
            reason = "not_degree_granting"
        elif r.UNITID in invalid_ids:
            reason = "invalid_enrollment_or_online_count"
        elif r.UNITID not in totals.index:
            reason = "missing_enrollment"
        elif totals.loc[r.UNITID, "eligible"] <= 0:
            reason = "no_in_person_eligible_enrollment"
        if reason:
            excluded.append({"unitid": int(r.UNITID), "name": r.INSTNM, "reason": reason})
            continue
        t = totals.loc[r.UNITID]
        for unitid, name, lat, lon, share in inside:
            row = dict(unitid=int(r.UNITID), name=r.INSTNM, lat=lat, lon=lon,
                       enrollment=int(round(t.eligible * share)),
                       total_degree_seeking_and_grad=int(round(t.EFDETOT * share)),
                       exclusively_online=int(round(t.EFDEEXC * share)), system=str(r.F1SYSNAM).strip())
            if share < 1.0:
                row |= dict(campus_unitid=unitid, campus=name, share=share)
            matched.append(row)
    return matched, excluded


def adult_weights(blocks):
    cols = ["B01001_E001"] + [f"B01001_E{i:03d}" for i in [3, 4, 5, 6, 27, 28, 29, 30]]
    keys = wfh.tract_key(blocks.geoid.str[:11].to_numpy())
    age = wfh.extract("b01001", cols, keys, b.RAW, b.fetch).set_index("key")
    for c in cols:
        age[c] = pd.to_numeric(age[c], errors="raise").clip(lower=0)
    age["adults"] = age.B01001_E001 - age[cols[1:]].sum(axis=1, min_count=8)
    age["share"] = age.adults / age.B01001_E001.replace(0, np.nan)
    ordinary = age[~age.index.str.startswith("09-")]
    county = ordinary.groupby(ordinary.index.str[:5])[["adults", "B01001_E001"]].sum()
    county_share = (county.adults / county.B01001_E001.replace(0, np.nan)).dropna().to_dict()
    metro = age.adults.sum() / age.B01001_E001.sum()
    shares = age.share.reindex(keys).to_numpy()
    missing = ~np.isfinite(shares)
    shares = np.where(missing, [county_share.get(g, metro) for g in blocks.geoid.str[:5]], shares)
    if not np.isfinite(shares).all() or ((shares < 0) | (shares > 1)).any():
        raise ValueError("invalid adult origin shares")
    return blocks["pop"].to_numpy() * shares, int(missing.sum())


def main(city="nyc"):
    cfg = b.CITIES[city]
    boundary = b.build_boundary(city, cfg)
    records, excluded = institutions(source(f"hd{YEAR}"), source(f"ef{YEAR}a_dist"), boundary, source(f"ef{YEAR - 1}a_dist"))
    stem = os.path.join(b.PACKS, city)
    with open(stem + ".json") as f:
        header = json.load(f)
    with open(stem + ".bin", "rb") as f:
        buf = f.read()
    a = next(a for a in header["arrays"] if a["name"] == "h3")
    cells = np.frombuffer(buf, dtype="<u8", count=a["count"], offset=a["offset"])
    index = {int(c): i for i, c in enumerate(cells)}
    enrollment = np.zeros(len(cells), dtype="<f4")
    matched, omitted, snaps = [], [], []
    for r in records:
        hid = h3.latlng_to_cell(r["lat"], r["lon"], 9)
        cell = index.get(h3.str_to_int(hid))
        if cell is None:
            def metres(c):
                lat, lon = h3.cell_to_latlng(c)
                return float(b.R_EARTH * np.hypot(np.radians(lat-r["lat"]), np.cos(np.radians(lat))*np.radians(lon-r["lon"])))
            candidates = [c for c in h3.grid_disk(hid, 3) if h3.str_to_int(c) in index]
            nearest = min(candidates, key=metres) if candidates else None
            if nearest is None or metres(nearest) > 500:
                omitted.append(r | {"reason": "no_existing_cell_within_500m"})
                continue
            cell = index[h3.str_to_int(nearest)]
            snaps.append(dict(unitid=r["unitid"], distance_m=metres(nearest)))
        enrollment[cell] += r["enrollment"]
        matched.append(r | {"cell": cell})
    source_total = sum(r["enrollment"] for r in records)
    if source_total <= 0 or float(enrollment.sum()) / source_total < 0.95:
        raise ValueError("university location coverage below 95%")
    blocks = pd.read_csv(os.path.join(b.WORK, city + ".blocks.csv.gz"), dtype={"geoid": str})
    blocks["university_adults"], fallback = adult_weights(blocks)
    mask, _ = b.build_water(cfg, boundary)
    ids, values, stats = spread.spread(blocks, set(cfg["counties"]), b.fetch, mask, *cfg["origin"], cols=("university_adults",))
    positions = np.searchsorted(cells, ids)
    hit = (positions < len(cells)) & (cells[np.minimum(positions, len(cells)-1)] == ids)
    if values["university_adults"][~hit].sum() > 0.01:
        raise ValueError("adult origins outside existing pack")
    adults = np.zeros(len(cells), dtype="<f4")
    adults[positions[hit]] = values["university_adults"][hit]
    buf = education.replace_cell_arrays(header, buf, {"university_adults": adults, "university_enrollment": enrollment})
    header["university"] = dict(
        coverage="Matched active degree-granting IPEDS reporting institutions in the city pack's counties; undergraduate degree/certificate-seeking and graduate students, excluding exclusively online enrollment",
        year=YEAR, enrolled=float(enrollment.sum(dtype=np.float64)), institutions=len({r["unitid"] for r in matched}),
        sites=len(matched), merged_campus_sites=sum(1 for r in matched if "campus_unitid" in r),
        exclusively_online_excluded=sum(r["exclusively_online"] for r in records),
        location_match_share=float(enrollment.sum())/source_total, adult_weight=float(adults.sum(dtype=np.float64)),
        adult_origin="2020 block population x ACS 2023 tract age 18+ share; full county FIPS fallback, metro fallback for Connecticut",
        fallback_blocks=fallback, attendance_factor=0.6, decay_m=6000,
        limitations=f"One directory point per reporting UNITID, not campus-specific allocation, except institutions that absorbed others in {YEAR} (split over the absorbed campuses by their {YEAR - 1} enrollment); adult homes and attendance/timing estimated; no nondegree undergraduates, staff or global coverage",
        sources=[BASE+f"hd{YEAR}.zip", BASE+f"ef{YEAR}a_dist.zip", BASE+f"ef{YEAR - 1}a_dist.zip", "https://api.census.gov/data/2023/acs/acs5/groups/B01001.html"],
        source_sha256={name: hashlib.sha256(open(os.path.join(b.RAW, name), "rb").read()).hexdigest() for name in [f"hd{YEAR}.zip", f"ef{YEAR}a_dist.zip", f"ef{YEAR - 1}a_dist.zip"]})
    with open(stem + ".bin.part", "wb") as f:
        f.write(buf)
    with open(stem + ".json.part", "w") as f:
        json.dump(header, f, indent=2)
    os.replace(stem + ".bin.part", stem + ".bin")
    os.replace(stem + ".json.part", stem + ".json")
    with open(os.path.join(b.WORK, city + ".universities.audit.json"), "w") as f:
        json.dump(dict(metadata=header["university"], institutions=matched, excluded=excluded,
                       omitted=omitted, snaps=snaps, spread=stats,
                       adult_origin_input_total=float(blocks.university_adults.sum())), f, indent=2)
    print(json.dumps(header["university"], indent=2), flush=True)


if __name__ == "__main__":
    import sys
    main(*sys.argv[1:2])
