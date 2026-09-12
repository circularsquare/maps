"""Estimate Daejeon and Gwangju Line 1 flows from measured station gates.

Both systems are single, unbranched lines, so every OD pair has exactly one
route.  IPF balances a distance-decay seed to average weekday entries/exits;
5/10/15 km decay scenarios expose how much the unknown trip-length distribution
changes each segment.  Run only after checking that nobody else is building the
rail data.
"""
import argparse
import datetime as dt
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import openpyxl

from build_busan import calibrate_decay, fit_od, read_csv, trip_length_note
import city_track as CT
import yearbook_extra as YX


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
STATIONS = DATA / "city_stations.xlsx"

# Which row of the yearbook's 도시철도 sheet 10 measures each line's trip
# length, and how far either side of it the scenario band reaches. See
# build_daegu_busan.py for why 15 %.
TRIP_LENGTH_ROW = {
    "daejeon": ("대전교통공사", "1호선"),
    "gwangju": ("광주광역시", "1호선"),
}
TRIP_LENGTH_SPREAD = 0.15

CITY = {
    "daejeon": {
        "ko": "대전",
        "operator": "대전교통공사",
        "line": "대전 1호선",
        "en": "Daejeon Line 1 (est.)",
        "color": "#007448",
        "date_col": "날짜",
        "published_km": 20.5,
        "aliases": {},
    },
    "gwangju": {
        "ko": "광주",
        "operator": "광주교통공사",
        "line": "광주 1호선",
        "en": "Gwangju Line 1 (est.)",
        "color": "#009088",
        "date_col": "일자",
        "published_km": 20.5,
        "aliases": {
            "학동증심사": "학동증심사입구",
            "문화전당": "문화전당(구도청)",
            "컨벤션센터": "김대중컨벤션센터(마륵)",
            "광주송정": "광주송정역",
        },
    },
}


def haversine_km(a, b):
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    dlat, dlon = lat2 - lat1, lon2 - lon1
    q = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 6371.0088 * 2 * math.asin(math.sqrt(q))


def station_catalog(path):
    book = openpyxl.load_workbook(path, read_only=True, data_only=True)
    sheet = book["표준데이터 역사"]
    rows = sheet.iter_rows(values_only=True)
    header = next(rows)
    index = {name: i for i, name in enumerate(header)}
    required = {"역사명", "영문역사명", "역위도", "역경도", "운영기관명"}
    if not required.issubset(index):
        raise ValueError("Unexpected national station workbook columns")
    result = defaultdict(dict)
    for row in rows:
        operator = str(row[index["운영기관명"]] or "").strip()
        for key, config in CITY.items():
            if operator == config["operator"]:
                name = str(row[index["역사명"]]).strip()
                result[key][name] = {
                    "en": str(row[index["영문역사명"]] or name).strip(),
                    "coord": [float(row[index["역경도"]]), float(row[index["역위도"]])],
                }
    return result


def weekday_demand(folder, config, month):
    rows = read_csv(folder / "counts.csv")
    date_col = config["date_col"]
    fixed = {date_col, "역번호", "역명", "구분"}
    hour_cols = [name for name in rows[0] if name not in fixed]
    selected, seen, days = [], set(), set()
    for row in rows:
        day = dt.date.fromisoformat(row[date_col])
        if not day.isoformat().startswith(month) or day.weekday() >= 5:
            continue
        code, kind = int(row["역번호"]), row["구분"].strip()
        if kind not in ("승차", "하차"):
            raise ValueError(f"Unknown count type {kind}")
        key = (day, code, kind)
        if key in seen:
            raise ValueError("Duplicate station-day count")
        seen.add(key)
        days.add(day)
        selected.append((code, row["역명"].strip(), kind,
                         sum(int((row[name] or "0").replace(",", "")) for name in hour_cols)))
    if not days:
        raise ValueError("No matching days")
    start = dt.date.fromisoformat(month + "-01")
    expected = {start + dt.timedelta(days=i) for i in range(32)
                if (start + dt.timedelta(days=i)).strftime("%Y-%m") == month
                and (start + dt.timedelta(days=i)).weekday() < 5}
    if days != expected:
        raise ValueError("Incomplete reference month")
    codes = sorted({code for code, _, _, _ in selected})
    if codes != list(range(codes[0], codes[-1] + 1)):
        raise ValueError("Station codes do not form one line")
    if len(seen) != len(codes) * len(days) * 2:
        raise ValueError("Incomplete station-day coverage")
    names = {}
    totals = defaultdict(float)
    for code, name, kind, count in selected:
        if code in names and names[code] != name:
            raise ValueError("Station code changed name")
        names[code] = name
        totals[code, kind] += count
    board = np.array([totals[code, "승차"] / len(days) for code in codes])
    alight = np.array([totals[code, "하차"] / len(days) for code in codes])
    return codes, [names[code] for code in codes], board, alight, {
        "days": len(days),
        "boardings": float(board.sum()),
        "alightings": float(alight.sum()),
        "destination_scale": float(board.sum() / alight.sum()),
    }


def line_assignment(od):
    """Return increasing-code and decreasing-code load for each adjacent edge."""
    n = od.shape[0]
    loads = np.zeros((n - 1, 2))
    for edge in range(n - 1):
        loads[edge, 0] = od[:edge + 1, edge + 1:].sum()
        loads[edge, 1] = od[edge + 1:, :edge + 1].sum()
    return loads


def build_city(key, config, catalog, month):
    codes, names, board, alight, report = weekday_demand(DATA / key, config, month)
    joined = []
    for name in names:
        catalog_name = config["aliases"].get(name, name)
        if catalog_name not in catalog:
            raise ValueError(f"{config['ko']} station missing from national catalog: {name}")
        joined.append(catalog[catalog_name])
    direct_km = np.array([haversine_km(a["coord"][::-1], b["coord"][::-1])
                          for a, b in zip(joined, joined[1:])])
    edge_km = direct_km * (config["published_km"] / direct_km.sum())
    chainage = np.r_[0, np.cumsum(edge_km)]
    costs = np.abs(chainage[:, None] - chainage[None, :])

    # Fit the decay to the published mean trip length rather than assuming it.
    # These are single unbranched lines, so `costs` is already the routed
    # distance and doubles as the km matrix.
    operator, ylabel = TRIP_LENGTH_ROW[key]
    published = YX.urban_mean_trip(operator, ylabel, max_km=float(costs.max()))
    if published is None:
        raise ValueError("No published trip length for %s" % key)
    target_km, dropped = published
    fits = [(target_km * f,) + calibrate_decay(costs, board, alight, costs,
                                               target_km * f)
            for f in (1.0 - TRIP_LENGTH_SPREAD, 1.0, 1.0 + TRIP_LENGTH_SPREAD)]
    decays = [d for _, d, _ in fits]

    scenario_loads, scenario_report = [], []
    for decay in decays:
        od, error, iterations = fit_od(costs, board, alight, decay)
        loads = line_assignment(od)
        scaled_alight = alight * board.sum() / alight.sum()
        net = np.r_[loads[0, 0] - loads[0, 1],
                    loads[:-1, 1] - loads[:-1, 0] + loads[1:, 0] - loads[1:, 1],
                    loads[-1, 1] - loads[-1, 0]]
        flow_error = float(np.max(np.abs(net - (board - scaled_alight))))
        if flow_error > 1e-4:
            raise ValueError("Routed flow violates station conservation")
        scenario_loads.append(loads)
        scenario_report.append({
            "decay_km": decay,
            "max_gate_residual": error,
            "iterations": iterations,
            "mean_trip_km": float((od * costs).sum() / od.sum()),
            "max_station_flow_residual": flow_error,
        })

    values = np.stack(scenario_loads)
    center = values[1]
    sums = values.sum(axis=2)
    # Real track where OSM has it; straight hops where it does not.
    track, track_report = CT.fit_edges({config["line"]: [
        (i, x["coord"], y["coord"])
        for i, (x, y) in enumerate(zip(joined, joined[1:]))]})

    features = []
    for edge, (a, b) in enumerate(zip(joined, joined[1:])):
        shape = track.get(edge) or [a["coord"], b["coord"]]
        features.append({
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": shape},
            "properties": {
                "line": config["line"], "from": names[edge], "to": names[edge + 1],
                "daily_down": round(center[edge, 0], 1),
                "daily_up": round(center[edge, 1], 1),
                "daily": round(center[edge].sum(), 1),
                "daily_low": round(sums[:, edge].min(), 1),
                "daily_high": round(sums[:, edge].max(), 1),
                "daily_down_low": round(values[:, edge, 0].min(), 1),
                "daily_down_high": round(values[:, edge, 0].max(), 1),
                "daily_up_low": round(values[:, edge, 1].min(), 1),
                "daily_up_high": round(values[:, edge, 1].max(), 1),
                "km": round(float(edge_km[edge]), 3),
                "source": "small_city_gravity", "period": month + " Mon-Fri mean",
                "geometry_source": ("osm" if edge in track else "straight"), "estimated": True,
            },
        })
    sensitivity = (sums.max(axis=0) - sums.min(axis=0)) / sums[1]
    report.update({
        "city": config["ko"], "stations": len(names), "segments": len(features),
        "month": month, "day_filter": "Monday-Friday (no holiday exclusion)",
        "track_geometry": track_report,
        "published_line_km": config["published_km"],
        "unscaled_straight_line_km": float(direct_km.sum()),
        "scenarios": scenario_report,
        "trip_length_source": "철도통계연보 2022 도시철도 sheet 10 (통행거리별 여객 승차실적), row %s / %s" % (operator, ylabel),
        "published_mean_trip_km": target_km,
        "published_pkm_dropped_beyond_network": dropped,
        "network_longest_trip_km": float(costs.max()),
        "fitted_decay_km": decays[1],
        "scenario_spread_fraction": TRIP_LENGTH_SPREAD,
        # A scenario that could not reach its target is at the network's own
        # ceiling -- with every pair weighted alike, the gates and the geometry
        # decide the mean trip and no decay can lengthen it further.
        "scenario_targets_km": [t for t, _, _ in fits],
        "scenario_reached_km": [m for _, _, m in fits],
        "scenario_target_unreachable": [abs(m - t) > 0.01 for t, _, m in fits],
        "central_trip_km_shortfall": round(target_km - fits[1][2], 4),
        "trip_length_note": trip_length_note(target_km, fits),
        "sensitivity_median_fraction": float(np.median(sensitivity)),
        "sensitivity_max_fraction": float(np.max(sensitivity)),
        "limits": "Gravity OD whose decay is fitted to the published mean trip length; the OD itself is still inferred, not observed. Gate fit is an accounting check, not independent validation. Scenario bounds are not confidence intervals. Straight station geometry is scaled to the published line length for distance costs.",
        "source_sha256": {
            "counts.csv": hashlib.sha256((DATA / key / "counts.csv").read_bytes()).hexdigest(),
            "city_stations.xlsx": hashlib.sha256(STATIONS.read_bytes()).hexdigest(),
        },
    })
    station_names = {name: joined[i]["en"] for i, name in enumerate(names)}
    meta = {config["line"]: {"en": config["en"], "color": config["color"],
                             "cls": "metro", "estimated": True, "period": month,
                             "city": config["ko"]}}
    return features, station_names, meta, report


def build(month="2026-07", station_path=STATIONS):
    catalogs = station_catalog(station_path)
    features, station_names, line_meta, reports = [], {}, {}, {}
    for key, config in CITY.items():
        city_features, city_names, city_meta, report = build_city(
            key, config, catalogs[key], month)
        features.extend(city_features)
        station_names.update(city_names)
        line_meta.update(city_meta)
        reports[key] = report
    return {"type": "FeatureCollection", "features": features,
            "line_meta": line_meta, "station_names": station_names,
            "model_report": reports}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--month", default="2026-07")
    parser.add_argument("--stations", type=Path, default=STATIONS)
    parser.add_argument("--output", type=Path, default=DATA / "small_city_segments.geojson")
    args = parser.parse_args()
    result = build(args.month, args.stations)
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    temporary.replace(args.output)
    for key, report in result["model_report"].items():
        (DATA / key / "model_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(result["model_report"], indent=2))


if __name__ == "__main__":
    main()
