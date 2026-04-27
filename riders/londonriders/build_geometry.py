"""
Extract station coords + line topology from NUMBAT definitions.xlsx.

Outputs geometry.json with:
  stations:  { mnlc: { name, lat, lon, asc, lines: [...] } }
  branches:  [ { line, dir, branch_id, branch_name, mnlcs: [...], distances_m: [...] } ]
  lines:     [ { line, color, coords: [[lon, lat], ...] } ]   (for drawing on map)
"""
import json
import os
from collections import defaultdict

import openpyxl

DEF_XLSX = "data/numbat_definitions.xlsx"
OUT_JSON = "geometry.json"

# Official-ish TfL line colors (from tfl.gov.uk style guide).
# Keys are NUMBAT line codes from the Lines sheet.
LINE_COLORS = {
    "BAK": "#B36305",  # Bakerloo
    "CEN": "#E32017",  # Central
    "CIR": "#FFD300",  # Circle
    "DIS": "#00782A",  # District
    "HAM": "#F3A9BB",  # Hammersmith & City (and combined HAM/CIR)
    "JUB": "#A0A5A9",  # Jubilee
    "MET": "#9B0056",  # Metropolitan
    "NOR": "#000000",  # Northern
    "PIC": "#003688",  # Piccadilly
    "VIC": "#0098D4",  # Victoria
    "WAC": "#95CDBA",  # Waterloo & City
    "DLR": "#00A4A7",  # DLR
    "EZL": "#6950A1",  # Elizabeth line
    # Overground branches (all the new line names launched 2024). For v1 just
    # paint them all with the canonical orange family — we can refine later.
    "NLL": "#FA7B05",  # North London (Mildmay)
    "ELL": "#FA7B05",  # East London (Windrush)
    "WEL": "#FA7B05",  # West London (Lioness etc)
    "GOB": "#FA7B05",  # Gospel Oak - Barking (Suffragette)
    "URL": "#FA7B05",  # Liberty
    "WAG": "#FA7B05",  # Watford DC (Lioness)
    "TRM": "#84B817",  # Trams
}
DEFAULT_COLOR = "#808183"


def load_stations():
    wb = openpyxl.load_workbook(DEF_XLSX, read_only=True, data_only=True)
    ws = wb["Stations"]
    rows = ws.iter_rows(values_only=True)
    header = list(next(rows))
    idx = {h: i for i, h in enumerate(header)}
    stations = {}
    for row in rows:
        mnlc = row[idx["MasterNLC"]]
        if mnlc is None:
            continue
        active = row[idx["Active"]]
        lat = row[idx["Latitude"]]
        lon = row[idx["Longitude"]]
        if not active or lat is None or lon is None:
            continue
        stations[int(mnlc)] = {
            "name": row[idx["UniqueStationName"]],
            "asc": row[idx["MasterASC"]],
            "lat": float(lat),
            "lon": float(lon),
        }
    print(f"  {len(stations)} active stations")
    return stations


def load_lines():
    wb = openpyxl.load_workbook(DEF_XLSX, read_only=True, data_only=True)
    ws = wb["Lines"]
    rows = ws.iter_rows(values_only=True)
    header = list(next(rows))
    idx = {h: i for i, h in enumerate(header)}
    lines = {}
    for row in rows:
        code = row[idx["LineCode"]]
        if not code:
            continue
        lines[code] = {
            "code": code,
            "description": row[idx["LineDescription"]],
            "mode": row[idx["ChildMode"]],
            "is_tfl": bool(row[idx["IsTfLLine"]]),
            "plot_dir": row[idx["PlotDir"]],
        }
    print(f"  {len(lines)} lines defined")
    return lines


def load_branches():
    """Branches table: line, dir, branch_seq, branch_name."""
    wb = openpyxl.load_workbook(DEF_XLSX, read_only=True, data_only=True)
    ws = wb["Branches"]
    rows = ws.iter_rows(values_only=True)
    header = list(next(rows))
    idx = {h: i for i, h in enumerate(header)}
    branches = {}
    for row in rows:
        bseq = row[idx["BranchSeq"]]
        if bseq is None:
            continue
        branches[int(bseq)] = {
            "branch_seq": int(bseq),
            "line": row[idx["Line"]],
            "dir": row[idx["Dir"]],
            "name": row[idx["Branch"]],
        }
    print(f"  {len(branches)} branches")
    return branches


def load_links_rail():
    """Read LinksRail. Filter to canonical sub-seq=0, valid rows."""
    wb = openpyxl.load_workbook(DEF_XLSX, read_only=True, data_only=True)
    ws = wb["LinksRail"]
    rows = ws.iter_rows(values_only=True)
    header = list(next(rows))
    idx = {h: i for i, h in enumerate(header)}
    out = []
    for row in rows:
        if row[idx["Valid"]] is not True:
            continue
        sub = row[idx["LinkSubSeq"]]
        if sub != 0:
            continue
        line = row[idx["Line"]]
        i_mnlc = row[idx["i_mnlc"]]
        j_mnlc = row[idx["j_mnlc"]]
        if line is None or i_mnlc in (None, -1) or j_mnlc in (None, -1):
            continue
        out.append({
            "line": line,
            "dir": row[idx["Dir"]],
            "branch_id": int(row[idx["BranchID"]]) if row[idx["BranchID"]] else None,
            "branch_link_seq": int(row[idx["BranchLinkSeq"]]) if row[idx["BranchLinkSeq"]] else 0,
            "i_mnlc": int(i_mnlc),
            "j_mnlc": int(j_mnlc),
            "distance_m": float(row[idx["DistanceM"]]) if row[idx["DistanceM"]] else 0.0,
            "mode": row[idx["Mode"]],
            "is_plot_dir": bool(row[idx["IsPlotDir"]]),
        })
    print(f"  {len(out)} valid rail links (sub_seq=0)")
    return out


def build_branches(links_rail, branches_meta):
    """Group links by (line, dir, branch_id) and order by branch_link_seq.
    Returns list of branches with ordered station sequence + per-link distances."""
    grouped = defaultdict(list)
    for lk in links_rail:
        if lk["branch_id"] is None:
            continue
        grouped[(lk["line"], lk["dir"], lk["branch_id"])].append(lk)

    out = []
    for (line, dir_, bid), lks in grouped.items():
        lks.sort(key=lambda l: l["branch_link_seq"])
        mnlcs = [lks[0]["i_mnlc"]]
        distances = []
        for lk in lks:
            # skip if link doesn't connect (out-of-order branch scrap)
            if mnlcs[-1] != lk["i_mnlc"]:
                # gap — start fresh chain only if longer than current
                # for now, just append as-is
                pass
            mnlcs.append(lk["j_mnlc"])
            distances.append(lk["distance_m"])
        meta = branches_meta.get(bid, {})
        out.append({
            "line": line,
            "dir": dir_,
            "branch_id": bid,
            "branch_name": meta.get("name", f"branch {bid}"),
            "mode": lks[0]["mode"],
            "is_plot_dir": lks[0]["is_plot_dir"],
            "mnlcs": mnlcs,
            "distances_m": distances,
        })
    return out


def build_line_geometries(branches, stations, lines_meta):
    """For each branch in plot direction, emit a polyline (straight between stations)."""
    out = []
    for br in branches:
        if not br["is_plot_dir"]:
            continue
        line = br["line"]
        meta = lines_meta.get(line, {})
        if not meta.get("is_tfl"):
            continue
        coords = []
        for mnlc in br["mnlcs"]:
            s = stations.get(mnlc)
            if s:
                coords.append([round(s["lon"], 5), round(s["lat"], 5)])
        if len(coords) < 2:
            continue
        out.append({
            "line": line,
            "branch_id": br["branch_id"],
            "branch_name": br["branch_name"],
            "color": LINE_COLORS.get(line, DEFAULT_COLOR),
            "coords": coords,
        })
    return out


def main():
    print("Loading stations...")
    stations = load_stations()
    print("Loading lines...")
    lines_meta = load_lines()
    print("Loading branches...")
    branches_meta = load_branches()
    print("Loading LinksRail...")
    links_rail = load_links_rail()

    print("Building branch sequences...")
    branches = build_branches(links_rail, branches_meta)
    print(f"  {len(branches)} branches built")

    # filter to TfL-served branches only
    tfl_branches = [b for b in branches if lines_meta.get(b["line"], {}).get("is_tfl")]
    print(f"  {len(tfl_branches)} TfL branches")

    line_geos = build_line_geometries(branches, stations, lines_meta)
    print(f"  {len(line_geos)} line geometries (plot direction)")

    # only keep stations that appear on at least one TfL branch
    used_mnlcs = set()
    for br in tfl_branches:
        used_mnlcs.update(br["mnlcs"])

    stations_out = {}
    for mnlc in used_mnlcs:
        s = stations.get(mnlc)
        if s:
            stations_out[str(mnlc)] = {
                "name": s["name"],
                "asc": s["asc"],
                "lat": round(s["lat"], 6),
                "lon": round(s["lon"], 6),
            }

    out = {
        "stations": stations_out,
        "branches": [
            {
                "line": b["line"],
                "dir": b["dir"],
                "branch_id": b["branch_id"],
                "branch_name": b["branch_name"],
                "mode": b["mode"],
                "is_plot_dir": b["is_plot_dir"],
                "mnlcs": b["mnlcs"],
                "distances_m": [round(d, 1) for d in b["distances_m"]],
            }
            for b in tfl_branches
        ],
        "lines": line_geos,
        "line_colors": LINE_COLORS,
    }

    print(f"Writing {OUT_JSON}...")
    with open(OUT_JSON, "w") as f:
        json.dump(out, f)
    size_mb = os.path.getsize(OUT_JSON) / 1024 / 1024
    print(f"Done! {OUT_JSON} is {size_mb:.2f} MB")
    print(f"  {len(stations_out)} stations, {len(tfl_branches)} branches, "
          f"{len(line_geos)} drawn line segments")


if __name__ == "__main__":
    main()
