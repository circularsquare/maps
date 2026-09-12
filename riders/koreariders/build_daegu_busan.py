"""Estimate Daegu Lines 1-3 and Busan-Gimhae LRT segment flows.

The inputs publish station gates but not OD pairs. Distance-decay seeds are
balanced to weekday station entries/exits, then assigned to shortest paths.
Daegu is fitted as one three-line network; Busan-Gimhae remains separate from
the differently dated Busan Lines 1-4 model.
"""
import argparse
import datetime as dt
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import openpyxl

from build_busan import (assign, calibrate_decay, fit_od, path_km,
                         paths_between, read_csv, trip_length_note)
import city_track as CT
import yearbook_extra as YX


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
STATIONS = DATA / "city_stations.xlsx"

# Which row of the yearbook's 도시철도 sheet 10 measures each system's trip
# length. The decay is no longer assumed: it is whatever reproduces this.
TRIP_LENGTH_ROW = {
    "daegu": ("대구교통공사", "계"),
    "busan_gimhae": ("부산-김해 경전철㈜", "부산-김해"),
}

# How far either side of the published mean the scenario band reaches. What is
# still unknown after the mean is fixed is the *shape* of the decay, route
# choice, and that the published distribution is 2022 against 2026 gates. 15 %
# covers that; it is still an assumption range and not a confidence interval,
# but it is now a range around a measurement rather than around a guess.
TRIP_LENGTH_SPREAD = 0.15
COLORS = {"대구 1호선": "#D93F5C", "대구 2호선": "#00AA80",
          "대구 3호선": "#FFB100", "부산김해경전철": "#8652A1"}


def compact(value):
    return re.sub(r"\s+", "", str(value or "").strip())


def haversine_km(a, b):
    lon1, lat1 = map(math.radians, a)
    lon2, lat2 = map(math.radians, b)
    dlat, dlon = lat2 - lat1, lon2 - lon1
    q = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 6371.0088 * 2 * math.asin(math.sqrt(q))


def catalog_rows(path, operator):
    book = openpyxl.load_workbook(path, read_only=True, data_only=True)
    sheet = book["표준데이터 역사"]
    rows = sheet.iter_rows(values_only=True)
    header = next(rows)
    ix = {name: i for i, name in enumerate(header)}
    out = []
    for row in rows:
        if str(row[ix["운영기관명"]] or "").strip() != operator:
            continue
        out.append({"code": str(row[ix["역번호"]]).strip(),
                    "name": str(row[ix["역사명"]]).strip(),
                    "line_name": str(row[ix["노선명"]]).strip(),
                    "en": compact(row[ix["영문역사명"]]) or str(row[ix["역사명"]]).strip(),
                    "coord": [float(row[ix["역경도"]]), float(row[ix["역위도"]])]})
    return out


def make_network(platforms, transfer_km=0.5):
    nodes, edges, graph, groups, lines = {}, [], defaultdict(list), defaultdict(list), defaultdict(list)
    for row in platforms:
        key = row["key"]
        nodes[key] = row
        groups[row["name"]].append(key)
        lines[row["line"]].append(key)
    for line, keys in lines.items():
        keys.sort(key=lambda key: nodes[key]["seq"])
        for a, b in zip(keys, keys[1:]):
            km = haversine_km(nodes[a]["coord"], nodes[b]["coord"])
            if km <= 0:
                raise ValueError("Invalid station distance")
            edge = len(edges)
            edges.append({"a": a, "b": b, "line": line, "km": km})
            graph[a].append((b, km, edge, 0))
            graph[b].append((a, km, edge, 1))
    for keys in groups.values():
        for a in keys:
            for b in keys:
                if a != b:
                    graph[a].append((b, transfer_km, None, None))
    return nodes, edges, graph, groups


def daegu_network(path=STATIONS):
    source = catalog_rows(path, "대구교통공사")
    count_rows = read_csv(DATA / "daegu" / "counts.csv")
    count_name = {}
    for row in count_rows:
        raw = int(row["역번호"])
        code = str(raw // 10).zfill(4)
        name = row["역명"].strip()
        if name[:-1] in {"명덕", "반월당", "청라언덕"} and name[-1:] in "123":
            name = name[:-1]
        count_name[code] = name
    platforms = []
    for row in source:
        if row["code"] not in count_name:
            raise ValueError(f"Daegu station absent from gate data: {row['name']}")
        line = int(row["code"][1])
        row.update({"key": "D" + row["code"], "seq": int(row["code"]),
                    "line": f"대구 {line}호선", "name": count_name[row["code"]]})
        platforms.append(row)
    return make_network(platforms)


def gimhae_network(path=STATIONS):
    source = catalog_rows(path, "부산-김해경전철㈜")
    count_rows = read_csv(DATA / "busan_gimhae" / "counts.csv")
    count_names = {compact(row["역사명"]): row["역사명"].strip() for row in count_rows}
    platforms = []
    for row in source:
        key = compact(row["name"])
        if key not in count_names:
            raise ValueError(f"Busan-Gimhae station absent from gate data: {row['name']}")
        row.update({"key": "G" + row["code"], "seq": int(row["code"]),
                    "line": "부산김해경전철", "name": row["name"]})
        platforms.append(row)
    return make_network(platforms)


def expected_weekdays(month):
    start = dt.date.fromisoformat(month + "-01")
    return {start + dt.timedelta(days=i) for i in range(32)
            if (start + dt.timedelta(days=i)).strftime("%Y-%m") == month
            and (start + dt.timedelta(days=i)).weekday() < 5}


def daegu_demand(nodes, groups, month):
    totals, seen, days = defaultdict(float), set(), set()
    for row in read_csv(DATA / "daegu" / "counts.csv"):
        day = dt.date(2026, int(row["월"]), int(row["일"]))
        if day.strftime("%Y-%m") != month or day.weekday() >= 5:
            continue
        code = "D" + str(int(row["역번호"]) // 10).zfill(4)
        kind = row["승하차"].strip()
        if code not in nodes or kind not in ("승차", "하차"):
            raise ValueError("Unknown Daegu station/count type")
        key = day, code, kind
        if key in seen:
            raise ValueError("Duplicate Daegu station-day count")
        seen.add(key); days.add(day)
        totals[nodes[code]["name"], kind] += int(row["일계"].replace(",", ""))
    return finish_demand(groups, totals, seen, days, month)


def gimhae_demand(nodes, groups, month):
    by_compact = {compact(node["name"]): node["name"] for node in nodes.values()}
    totals, seen, days = defaultdict(float), set(), set()
    for row in read_csv(DATA / "busan_gimhae" / "counts.csv"):
        day = dt.date.fromisoformat(row["영업일자"])
        if day.strftime("%Y-%m") != month or day.weekday() >= 5:
            continue
        name = by_compact.get(compact(row["역사명"]))
        kind = row["분류"].strip()
        if not name or kind not in ("승차", "하차"):
            raise ValueError("Unknown Busan-Gimhae station/count type")
        key = day, name, kind
        if key in seen:
            raise ValueError("Duplicate Busan-Gimhae station-day count")
        seen.add(key); days.add(day)
        totals[name, kind] += int(row["합계"].replace(",", ""))
    return finish_demand(groups, totals, seen, days, month)


def finish_demand(groups, totals, seen, days, month):
    if days != expected_weekdays(month):
        raise ValueError("Incomplete reference month")
    names = sorted(groups)
    if len(seen) != sum(len(codes) for codes in groups.values()) * len(days) * 2:
        raise ValueError("Incomplete station-day coverage")
    if any((name, kind) not in totals for name in names for kind in ("승차", "하차")):
        raise ValueError("Missing station-complex counts")
    board = np.array([totals[name, "승차"] / len(days) for name in names])
    alight = np.array([totals[name, "하차"] / len(days) for name in names])
    return names, board, alight, {"days": len(days), "boardings": float(board.sum()),
                                  "alightings": float(alight.sum()),
                                  "destination_scale": float(board.sum() / alight.sum())}


def build_system(key, nodes, edges, graph, groups, demand_fn, month):
    names, costs, paths = paths_between(graph, groups)
    demand_names, board, alight, report = demand_fn(nodes, groups, month)
    if names != demand_names:
        raise ValueError("Network and demand station order differ")
    # The decay is fitted to the published mean trip length rather than
    # assumed. Bands longer than the network itself are dropped from the
    # published figure first -- see yearbook_extra.urban_mean_trip.
    km = path_km(paths, edges, len(names))
    operator, ylabel = TRIP_LENGTH_ROW[key]
    published = YX.urban_mean_trip(operator, ylabel, max_km=float(km.max()))
    if published is None:
        raise ValueError("No published trip length for %s" % key)
    target_km, dropped = published
    fits = [(target_km * f,) + calibrate_decay(costs, board, alight, km,
                                               target_km * f)
            for f in (1.0 - TRIP_LENGTH_SPREAD, 1.0, 1.0 + TRIP_LENGTH_SPREAD)]
    decays = [d for _, d, _ in fits]

    values, scenario_report = [], []
    for decay in decays:
        od, error, iterations = fit_od(costs, board, alight, decay)
        loads = assign(od, paths, len(edges))
        net = defaultdict(float)
        for i, edge in enumerate(edges):
            flow = loads[i, 0] - loads[i, 1]
            net[nodes[edge["a"]]["name"]] += flow
            net[nodes[edge["b"]]["name"]] -= flow
        scaled_alight = alight * board.sum() / alight.sum()
        flow_error = max(abs(net[name] - (board[i] - scaled_alight[i])) for i, name in enumerate(names))
        if flow_error > 1e-4:
            raise ValueError("Routed flow violates station conservation")
        values.append(loads)
        scenario_report.append({"decay_km": decay, "max_gate_residual": error,
                                "iterations": iterations,
                                "mean_trip_km": float((od * km).sum() / od.sum()),
                                "max_station_flow_residual": flow_error})
    values = np.stack(values)
    center = values[1]
    sums = values.sum(axis=2)
    # Real track where OSM has it; straight hops where it does not.
    by_line = {}
    for i, edge in enumerate(edges):
        by_line.setdefault(edge["line"], []).append(
            (i, nodes[edge["a"]]["coord"], nodes[edge["b"]]["coord"]))
    track, track_report = CT.fit_edges(by_line)

    features = []
    for i, edge in enumerate(edges):
        a, b = nodes[edge["a"]], nodes[edge["b"]]
        shape = track.get(i) or [a["coord"], b["coord"]]
        features.append({"type": "Feature",
            "geometry": {"type": "LineString", "coordinates": shape},
            "properties": {"line": edge["line"], "from": a["name"], "to": b["name"],
                "daily_down": round(center[i, 0], 1), "daily_up": round(center[i, 1], 1),
                "daily": round(center[i].sum(), 1), "daily_low": round(sums[:, i].min(), 1),
                "daily_high": round(sums[:, i].max(), 1),
                "daily_down_low": round(values[:, i, 0].min(), 1),
                "daily_down_high": round(values[:, i, 0].max(), 1),
                "daily_up_low": round(values[:, i, 1].min(), 1),
                "daily_up_high": round(values[:, i, 1].max(), 1),
                "km": round(edge["km"], 3), "source": key + "_gravity",
                "period": month + " Mon-Fri mean", "geometry_source": ("osm" if i in track else "straight"), "estimated": True}})
    sensitivity = (sums.max(axis=0) - sums.min(axis=0)) / sums[1]
    report.update({"station_complexes": len(names), "platforms": len(nodes), "segments": len(edges),
                   "month": month, "day_filter": "Monday-Friday (no holiday exclusion)",
                   "track_geometry": track_report,
                   "scenarios": scenario_report,
                   "trip_length_source": "철도통계연보 2022 도시철도 sheet 10 (통행거리별 여객 승차실적), row %s / %s" % (operator, ylabel),
                   "published_mean_trip_km": target_km,
                   "published_pkm_dropped_beyond_network": dropped,
                   "network_longest_trip_km": float(km.max()),
                   "fitted_decay_km": decays[1],
                   "scenario_spread_fraction": TRIP_LENGTH_SPREAD,
                   # A scenario that could not reach its target is at the
                   # network's own ceiling: with every pair weighted alike the
                   # gates and the geometry decide the mean trip, and no decay
                   # can lengthen it further.
                   "scenario_targets_km": [t for t, _, _ in fits],
                   "scenario_reached_km": [m for _, _, m in fits],
                   "scenario_target_unreachable": [abs(m - t) > 0.01
                                                   for t, _, m in fits],
                   "central_trip_km_shortfall": round(target_km - fits[1][2], 4),
                   "trip_length_note": trip_length_note(target_km, fits),
                   "sensitivity_median_fraction": float(np.median(sensitivity)),
                   "sensitivity_max_fraction": float(np.max(sensitivity)),
                   "limits": "Distance-gravity OD whose decay is fitted to the published mean trip length, not the OD itself. Straight station geometry. Scenario bounds are not confidence intervals."})
    station_names = {node["name"]: node["en"] for node in nodes.values()}
    return features, station_names, report


def build(daegu_month="2026-06", gimhae_month="2025-12", station_path=STATIONS):
    dn, de, dg, dgroups = daegu_network(station_path)
    gn, ge, gg, ggroups = gimhae_network(station_path)
    df, dnames, dr = build_system("daegu", dn, de, dg, dgroups, daegu_demand, daegu_month)
    gf, gnames, gr = build_system("busan_gimhae", gn, ge, gg, ggroups, gimhae_demand, gimhae_month)
    meta = {}
    for line in sorted({f["properties"]["line"] for f in df + gf}):
        if line.startswith("대구"):
            en, city = "Daegu Line " + line.split()[1].replace("호선", "") + " (est.)", "대구"
        else:
            en, city = "Busan–Gimhae LRT (est.)", "부산·김해"
        meta[line] = {"en": en, "color": COLORS[line], "cls": "metro", "estimated": True,
                      "period": daegu_month if city == "대구" else gimhae_month, "city": city}
    result = {"type": "FeatureCollection", "features": df + gf, "line_meta": meta,
              "station_names": {**dnames, **gnames}, "model_report": {"daegu": dr, "busan_gimhae": gr}}
    result["source_sha256"] = {str(path.relative_to(HERE)): hashlib.sha256(path.read_bytes()).hexdigest()
                                for path in (DATA / "daegu" / "counts.csv",
                                             DATA / "busan_gimhae" / "counts.csv", station_path)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--daegu-month", default="2026-06")
    parser.add_argument("--gimhae-month", default="2025-12")
    parser.add_argument("--stations", type=Path, default=STATIONS)
    parser.add_argument("--output", type=Path, default=DATA / "daegu_busan_segments.geojson")
    args = parser.parse_args()
    result = build(args.daegu_month, args.gimhae_month, args.stations)
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(result["model_report"], indent=2))


if __name__ == "__main__":
    main()
