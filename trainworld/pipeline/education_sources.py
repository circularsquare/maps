"""Reusable US public-school source (NCES EDGE/CCD 2023-24).

Coordinates and MEMBER are from the same official administrative layer. Query pages
are cached; unique NCES school IDs, not shared coordinates, identify enrollments.
"""
import hashlib
import json
from urllib.parse import urlencode

import numpy as np
import pandas as pd
from shapely.geometry import Point

import build_city as b
import wfh

NCES = "https://nces.ed.gov/opengis/rest/services/K12_School_Locations/EDGE_ADMINDATA_PUBLICSCH_2324/MapServer/1"
FIELDS = "NCESSCH,SCH_NAME,SURVYEAR,STATUS,VIRTUAL,MEMBER,LATCOD,LONCOD"


def public_schools(boundary):
    """Eligible physical schools inside a polygon.

    Do not use this layer's CNTY field (not a reliable full county FIPS). Spatial
    filtering also respects the game's 2020 Connecticut county boundaries.
    """
    west, south, east, north = boundary.bounds
    where = f"LONCOD >= {west:.8f} AND LONCOD <= {east:.8f} AND LATCOD >= {south:.8f} AND LATCOD <= {north:.8f}"
    digest = hashlib.sha256(where.encode()).hexdigest()[:16]
    records, seen, rejected = [], {}, []
    offset = 0
    while True:
        query = dict(f="json", where=where, outFields=FIELDS, returnGeometry="false",
                     orderByFields="NCESSCH", resultOffset=offset, resultRecordCount=1000)
        path = b.fetch(NCES + "/query?" + urlencode(query), f"nces-schools-2324-{digest}-{offset}.json")
        with open(path) as f:
            page = json.load(f)
        if "error" in page:
            raise ValueError(f"NCES query failed: {page['error']}")
        rows = page["features"]
        for feature in rows:
            r = feature["attributes"]
            key = r["NCESSCH"]
            if key in seen:
                # Repeated pages must fail instead of looping or doubling demand.
                raise ValueError(f"duplicate NCES school ID: {key}")
            seen[key] = True
            point = Point(r["LONCOD"], r["LATCOD"])
            if not boundary.covers(point):
                reason = "outside_city"
            elif str(r["STATUS"]).strip() not in {"1", "3", "4", "5", "8"}:
                reason = "not_operational"
            elif r["VIRTUAL"] not in {"Not Virtual", "Supplemental Virtual"}:
                reason = "virtual_or_unknown"
            elif r["MEMBER"] is None or r["MEMBER"] <= 0:
                reason = "missing_or_nonpositive_enrollment"
            else:
                records.append(dict(building="NCES:" + key, name=r["SCH_NAME"],
                                    enrollment=int(r["MEMBER"]), lat=r["LATCOD"],
                                    lon=r["LONCOD"], source="NCES CCD 2023-24",
                                    virtual=r["VIRTUAL"]))
                continue
            rejected.append(dict(id=key, reason=reason))
        if not page.get("exceededTransferLimit", False):
            break
        if not rows:
            raise ValueError("NCES indicated more data but returned an empty page")
        offset += len(rows)
    return records, rejected


def child_weights(blocks):
    """ACS age shares by tract; full-state county fallbacks, metro for CT gaps."""
    columns = ["B01001_E001"] + [f"B01001_E{i:03d}" for i in [3, 4, 5, 6, 27, 28, 29, 30]]
    keys = wfh.tract_key(blocks.geoid.str[:11].to_numpy())
    age = wfh.extract("b01001", columns, keys, b.RAW, b.fetch).set_index("key")
    for c in columns:
        age[c] = pd.to_numeric(age[c], errors="raise").clip(lower=0)
    age["children"] = age[[f"B01001_E{i:03d}" for i in [4, 5, 6, 28, 29, 30]]].sum(axis=1, min_count=6) + 0.4 * (age.B01001_E003 + age.B01001_E027)
    age["share"] = age.children / age.B01001_E001.replace(0, np.nan)
    ordinary = age[~age.index.str.startswith("09-")]
    county = ordinary.groupby(ordinary.index.str[:5])[["children", "B01001_E001"]].sum()
    county_shares = (county.children / county.B01001_E001.replace(0, np.nan)).dropna().to_dict()
    region_share = age.children.sum() / age.B01001_E001.sum()
    shares = age.share.reindex(keys).to_numpy()
    absent = ~np.isfinite(shares)
    fallback = np.array([county_shares.get(g, region_share) for g in blocks.geoid.str[:5]])
    shares = np.where(absent, fallback, shares)
    if not np.isfinite(shares).all() or ((shares < 0) | (shares > 1)).any():
        raise ValueError("invalid child population shares")
    return blocks["pop"].to_numpy() * shares, int(absent.sum())


def replace_cell_arrays(header, buf, replacements):
    """Compact the directory's live arrays, preserving all unrelated bytes exactly."""
    sizes = {"f32": 4, "f64": 8, "u8": 1, "u16": 2, "u32": 4, "u64": 8, "i32": 4}
    out, arrays = bytearray(), []
    for a in header["arrays"]:
        if a["name"] in replacements:
            continue
        out += b"\0" * (-len(out) % 8)
        data = buf[a["offset"]:a["offset"] + a["count"] * sizes[a["dtype"]]]
        if len(data) != a["count"] * sizes[a["dtype"]]:
            raise ValueError(f"truncated array {a['name']}")
        arrays.append(dict(a, offset=len(out)))
        out += data
    for name, weights in replacements.items():
        out += b"\0" * (-len(out) % 8)
        arrays.append(dict(name=name, dtype="f32", per="cell", offset=len(out), count=len(weights)))
        out += np.asarray(weights, dtype="<f4").tobytes()
    header["arrays"] = arrays
    return out
