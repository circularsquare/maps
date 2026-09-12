"""Build intracity station bubbles without rerunning any routing model.

Seoul metropolitan totals come from the already-built weekday OD matrix: each
trip contributes once at its origin and once at its destination, so transfers
are not double-counted. The other cities use the same weekday gate totals and
reference months as their segment models.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from build_busan import demand as busan_demand
from build_busan import load_network, paths_between
from build_daegu_busan import daegu_demand, daegu_network, gimhae_demand, gimhae_network
from build_small_cities import CITY, DATA, STATIONS, station_catalog, weekday_demand


HERE = Path(__file__).resolve().parent


def station_margins(pairs, flows, count):
    totals = flows.sum(axis=1) if flows.ndim == 2 else flows
    board = np.zeros(count)
    alight = np.zeros(count)
    np.add.at(board, pairs[:, 0], totals)
    np.add.at(alight, pairs[:, 1], totals)
    return board, alight


def feature(name, en, coord, lines, board, alight, source, period, coverage="measured gates"):
    return {
        "type": "Feature",
        "properties": {
            "station": name,
            "station_en": en,
            "line": lines[0] if lines else "",
            "lines": lines,
            "riders": round(float(board + alight)),
            "boardings": round(float(board)),
            "alightings": round(float(alight)),
            "kind": "metro",
            "source": source,
            "period": period,
            "coverage": coverage,
        },
        "geometry": {"type": "Point", "coordinates": [round(coord[0], 6), round(coord[1], 6)]},
    }


def seoul_features(folder):
    stations = json.loads((folder / "stations.json").read_text(encoding="utf-8"))
    stats = json.loads((folder / "stats.json").read_text(encoding="utf-8"))
    matrix = np.load(folder / "od_hourly.npz", allow_pickle=False)
    if str(matrix["day"]) != "weekday" or stats.get("day") != "weekday":
        raise ValueError("Expected matching weekday Seoul sources")
    names = matrix["names"].tolist()
    board, alight = station_margins(matrix["pairs"], matrix["x"], len(names))
    by_name = {c["name"]: c for c in stations["complexes"]}
    if set(names) != set(by_name):
        raise ValueError("Seoul OD and station rosters differ")
    line_names = {key: "수도권 " + value["display"] for key, value in stats["line_meta"].items()}
    out = []
    for i, name in enumerate(names):
        if round(board[i] + alight[i]) < 1:
            continue
        row = by_name[name]
        lines = sorted({line_names[p["line"]] for p in row["platforms"] if p["line"] in line_names})
        out.append(feature(name, row.get("name_en", name), [row["lon"], row["lat"]], lines,
                           board[i], alight[i], "seoul_weekday_od", "2023-11 weekday mean",
                           "weekday OD station margins; coverage varies by station"))
    return out, {
        "stations": len(out), "boardings": float(board.sum()), "alightings": float(alight.sum()),
        "source": "seoulriders/data/od_hourly.npz and stations.json",
    }


def busan_features(folder, month):
    nodes, _, graph, groups = load_network(folder)
    names, _, _ = paths_between(graph, groups)
    board, alight, report = busan_demand(folder, nodes, names, month)
    out = []
    for i, name in enumerate(names):
        codes = groups[name]
        coord = [sum(nodes[c]["coord"][0] for c in codes) / len(codes),
                 sum(nodes[c]["coord"][1] for c in codes) / len(codes)]
        lines = [f"부산 {line}호선" for line in sorted({nodes[c]["line"] for c in codes})]
        en = next((nodes[c]["en"] for c in codes if nodes[c]["en"]), name)
        out.append(feature(name, en, coord, lines, board[i], alight[i],
                           "busan_gates", month + " Mon-Fri mean"))
    return out, {**report, "stations": len(out)}


def small_city_features(month):
    catalogs = station_catalog(STATIONS)
    out, reports = [], {}
    for key, config in CITY.items():
        _, names, board, alight, report = weekday_demand(DATA / key, config, month)
        for i, name in enumerate(names):
            row = catalogs[key][config["aliases"].get(name, name)]
            out.append(feature(name, row["en"], row["coord"], [config["line"]],
                               board[i], alight[i], key + "_gates", month + " Mon-Fri mean"))
        reports[key] = {**report, "stations": len(names)}
    return out, reports


def network_features(network_fn, demand_fn, month, source):
    nodes, _, _, groups = network_fn()
    names, board, alight, report = demand_fn(nodes, groups, month)
    out = []
    for i, name in enumerate(names):
        codes = groups[name]
        coord = [sum(nodes[c]["coord"][0] for c in codes) / len(codes),
                 sum(nodes[c]["coord"][1] for c in codes) / len(codes)]
        lines = sorted({nodes[c]["line"] for c in codes})
        en = next((nodes[c]["en"] for c in codes if nodes[c]["en"]), name)
        out.append(feature(name, en, coord, lines, board[i], alight[i],
                           source, month + " Mon-Fri mean"))
    return out, {**report, "stations": len(out)}


def build(seoul_folder, month="2026-07", daegu_month="2026-06", gimhae_month="2025-12"):
    seoul, seoul_report = seoul_features(seoul_folder)
    busan, busan_report = busan_features(DATA / "busan", month)
    small, small_reports = small_city_features(month)
    daegu, daegu_report = network_features(daegu_network, daegu_demand, daegu_month, "daegu_gates")
    gimhae, gimhae_report = network_features(gimhae_network, gimhae_demand, gimhae_month,
                                              "busan_gimhae_gates")
    features = seoul + busan + small + daegu + gimhae
    features.sort(key=lambda row: row["properties"]["riders"])
    return {
        "type": "FeatureCollection",
        "features": features,
        "report": {"stations": len(features), "seoul": seoul_report,
                   "busan": busan_report, **small_reports,
                   "daegu": daegu_report, "busan_gimhae": gimhae_report},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seoul-source", type=Path, default=HERE.parent / "seoulriders" / "data")
    parser.add_argument("--month", default="2026-07")
    parser.add_argument("--daegu-month", default="2026-06")
    parser.add_argument("--gimhae-month", default="2025-12")
    parser.add_argument("--output", type=Path, default=DATA / "metro_stations.geojson")
    args = parser.parse_args()
    result = build(args.seoul_source, args.month, args.daegu_month, args.gimhae_month)
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(result["report"], indent=2))


if __name__ == "__main__":
    main()
