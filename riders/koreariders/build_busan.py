"""Experimental Busan Lines 1-4 flows from measured gates, not measured OD.

Fit exponential travel-time seeds to station-complex entries/exits with IPF,
then assign trips to shortest paths with a five-minute transfer penalty.
Three decay scales expose sensitivity; they are scenarios, not confidence bounds.
Run only after checking that nobody else is building the rail data.
"""
import argparse
import csv
import datetime as dt
import hashlib
import heapq
import io
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

import city_track as CT

HERE = Path(__file__).resolve().parent
D = HERE / "data" / "busan"
COLORS = {1: "#F06A00", 2: "#81BF48", 3: "#BB8C00", 4: "#217DCB"}


def read_csv(path):
    raw = path.read_bytes()
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        text = raw.decode("utf-16")
    else:
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = raw.decode("cp949")
    delimiter = "\t" if "\t" in text.splitlines()[0] else ","
    return list(csv.DictReader(io.StringIO(text), delimiter=delimiter))


def fit_od(cost, board, alight, decay):
    """Keep diagonal zero; match origins and proportionately balanced destinations."""
    if decay <= 0 or np.any(board < 0) or np.any(alight < 0):
        raise ValueError("Invalid demand or decay")
    if not np.all(np.isfinite(cost)) or min(board.sum(), alight.sum()) <= 0:
        raise ValueError("Disconnected network or empty demand")
    target = alight * (board.sum() / alight.sum())
    od = np.exp(-cost / decay)
    np.fill_diagonal(od, 0)
    for iteration in range(10000):
        od *= np.divide(board, od.sum(axis=1), out=np.zeros_like(board), where=od.sum(axis=1)>0)[:, None]
        od *= np.divide(target, od.sum(axis=0), out=np.zeros_like(target), where=od.sum(axis=0)>0)[None, :]
        error = max(np.max(np.abs(od.sum(axis=1)-board)), np.max(np.abs(od.sum(axis=0)-target)))
        if error < 1e-5:
            return od, float(error), iteration + 1
    raise ValueError("OD balancing did not converge")


def path_km(paths, edges, n):
    """Routed distance in km for every OD pair, from the assigned paths.

    Busan's costs are minutes rather than kilometres, so the km a trip covers
    is not the matrix the fit runs on. This is what the published trip-length
    distribution has to be compared against, and what caps it.
    """
    km = np.zeros((n, n))
    for (i, j), route in paths.items():
        km[i, j] = sum(edges[e]["km"] for e, _ in route)
    return km


def calibrate_decay(cost, board, alight, km, target_km, lo=0.5, hi=400.0,
                    tol=1e-3):
    """Find the decay whose fitted OD reproduces a published mean trip length.

    The decay used to be a stated assumption -- README.md called it "a
    provisional assumption, not a fitted parameter" -- because nothing published
    said how far a rider actually goes. The yearbook's 도시철도 sheet 10 does say,
    per line, so the parameter can be measured instead of guessed.

    Mean trip length rises monotonically with the decay scale (a longer scale
    puts more weight on distant pairs), so plain bisection finds it. Each step
    is a full IPF, which is why the bracket is wide but the tolerance is loose:
    a decay accurate to a millimetre of mean trip buys nothing.

    A target can be unreachable, and that is a result rather than an error. At
    an infinite decay every pair is weighted alike and the mean trip is
    whatever the gates and the geometry give on their own -- there is no way to
    make riders go further than a network that short allows. Daejeon's
    published 6.69 km is already 95 % of its own ceiling. So return the nearest
    achievable fit and let the caller record that it did not reach the target.

    Returns (decay, achieved_mean_km).
    """
    def mean_at(d):
        od, _, _ = fit_od(cost, board, alight, d)
        return float((od * km).sum() / od.sum())

    m_lo, m_hi = mean_at(lo), mean_at(hi)
    if target_km <= m_lo:
        return lo, m_lo
    if target_km >= m_hi:
        return hi, m_hi
    for _ in range(60):
        mid = math.sqrt(lo * hi)          # geometric: decay spans two decades
        m = mean_at(mid)
        if abs(m - target_km) < tol:
            return mid, m
        if m < target_km:
            lo = mid
        else:
            hi = mid
    mid = math.sqrt(lo * hi)
    return mid, mean_at(mid)


def trip_length_note(target_km, fits):
    """Say in words whether the calibration reached the published figure.

    A collapsed scenario band is not confidence. Where the central target
    cannot be reached the model is pinned against the most uniform OD its own
    gates allow, and the band narrows because it is up against a boundary
    rather than because the answer is well determined -- so the note has to say
    which of the two it is.
    """
    _, _, mid = fits[1]
    if abs(mid - target_km) <= 0.01:
        return ("Decay fitted to the published mean trip length; the central "
                "run reproduces it.")
    return ("The published mean trip of %.2f km is longer than this network's "
            "gates can produce under any distance decay (ceiling %.2f km), so "
            "the central run is pinned at the most uniform OD available and "
            "still falls %.2f km short. Distance-decay gravity is the wrong "
            "model family here, not merely the wrong parameter: decay can only "
            "ever shorten trips relative to uniform. Mid-line segments are "
            "therefore still understated, and the narrow scenario band reflects "
            "a boundary rather than confidence."
            % (target_km, mid, target_km - mid))


def load_network(folder):
    stations = {int(r["역번호"]): r for r in read_csv(folder / "stations.csv")}
    rows = read_csv(folder / "distances.csv")
    nodes, edges, graph, groups = {}, [], defaultdict(list), defaultdict(list)
    lines = defaultdict(list)
    for r in rows:
        code, line = int(r["역번호"]), int(r["호선"])
        st = stations[code]
        name = r["역명"].strip()
        nodes[code] = {"name": name, "en": st["영문역사명"], "line": line,
                       "coord": [float(st["역경도"]), float(st["역위도"])]}
        groups[name].append(code)
        lines[line].append((code, r))
    for line, stops in lines.items():
        stops.sort()
        for (a, _), (b, r) in zip(stops, stops[1:]):
            if b != a+1:
                raise ValueError(f"Non-consecutive station codes {a}, {b}")
            mm, ss = map(float, r["소요시간(분)"].split(":"))
            minutes, km = mm + ss/60, float(r["역간거리(km)"])
            if minutes <= 0 or km <= 0:
                raise ValueError("Invalid segment distance/time")
            idx = len(edges)
            edges.append({"a": a, "b": b, "line": line, "km": km, "minutes": minutes})
            graph[a].append((b, minutes, idx, 0))
            graph[b].append((a, minutes, idx, 1))
    for codes in groups.values():
        for a in codes:
            for b in codes:
                if a != b:
                    graph[a].append((b, 5.0, None, None))
    return nodes, edges, graph, groups


def paths_between(graph, groups):
    names = sorted(groups)
    costs = np.zeros((len(names), len(names)))
    paths = {}
    for i, name in enumerate(names):
        best = {c: 0.0 for c in groups[name]}
        heap, parent = [(0.0, c) for c in groups[name]], {}
        heapq.heapify(heap)
        while heap:
            cost, a = heapq.heappop(heap)
            if cost != best[a]:
                continue
            for b, weight, edge, direction in graph[a]:
                new = cost + weight
                if new < best.get(b, float("inf")):
                    best[b] = new
                    parent[b] = (a, edge, direction)
                    heapq.heappush(heap, (new, b))
        for j, dest in enumerate(names):
            end = min(groups[dest], key=lambda c: best.get(c, float("inf")))
            costs[i, j] = best.get(end, float("inf"))
            route = []
            while end in parent:
                end, edge, direction = parent[end]
                if edge is not None:
                    route.append((edge, direction))
            paths[i, j] = route[::-1]
    return names, costs, paths


def demand(folder, nodes, names, month):
    totals, seen, days = defaultdict(float), set(), set()
    for r in read_csv(folder / "counts.csv"):
        day = dt.date.fromisoformat(r["년월일"])
        if not day.isoformat().startswith(month) or day.weekday() >= 5:
            continue
        code, kind = int(r["역번호"]), r["구분"]
        if code not in nodes or kind not in ("승차", "하차"):
            raise ValueError("Unknown station or count type")
        key = (day, code, kind)
        if key in seen:
            raise ValueError("Duplicate station-day count")
        seen.add(key)
        days.add(day)
        count = int(r["합계"].replace(",", ""))
        if count < 0:
            raise ValueError("Negative gate count")
        totals[nodes[code]["name"], kind] += count
    if not days:
        raise ValueError("No matching days")
    expected_days = {dt.date.fromisoformat(month + "-01") + dt.timedelta(days=k) for k in range(31)}
    expected_days = {d for d in expected_days if d.isoformat().startswith(month) and d.weekday()<5}
    if days != expected_days:
        raise ValueError("Incomplete reference month")
    # Gate-less platforms at shared interchanges may be wholly absent, but a
    # reporting platform must have both count types on every selected day.
    codes = {code for day, code, kind in seen}
    if len(seen) != len(codes) * len(days) * 2:
        raise ValueError("Incomplete station-day coverage")
    if any((name, kind) not in totals for name in names for kind in ("승차", "하차")):
        raise ValueError("Missing station-complex counts")
    b = np.array([totals[name, "승차"] / len(days) for name in names])
    a = np.array([totals[name, "하차"] / len(days) for name in names])
    return b, a, {"days": len(days), "gate_platforms": len(codes),
                  "boardings": float(b.sum()), "alightings": float(a.sum()),
                  "destination_scale": float(b.sum()/a.sum())}


def assign(od, paths, edge_count):
    loads = np.zeros((edge_count, 2))
    for (i, j), route in paths.items():
        for edge, direction in route:
            loads[edge, direction] += od[i, j]
    return loads


def build(folder, month):
    nodes, edges, graph, groups = load_network(folder)
    names, costs, paths = paths_between(graph, groups)
    b, a, report = demand(folder, nodes, names, month)
    scenarios, scenario_report = [], []
    for decay in (10, 20, 30):
        od, error, iterations = fit_od(costs, b, a, decay)
        loads = assign(od, paths, len(edges))
        net = defaultdict(float)
        for k, edge in enumerate(edges):
            flow = loads[k, 0] - loads[k, 1]
            net[nodes[edge["a"]]["name"]] += flow
            net[nodes[edge["b"]]["name"]] -= flow
        flow_error = max(abs(net[name] - (b[i] - a[i]*b.sum()/a.sum())) for i, name in enumerate(names))
        if flow_error > 1e-4:
            raise ValueError("Routed flow violates station conservation")
        scenarios.append(loads)
        passenger_km = sum(e["km"] * loads[k].sum() for k, e in enumerate(edges))
        scenario_report.append({"decay_minutes": decay, "max_gate_residual": error,
                                "iterations": iterations, "mean_trip_km": passenger_km/b.sum(),
                                "max_station_flow_residual": flow_error})
    values = np.stack(scenarios)
    center = values[1]
    sums = values.sum(axis=2)
    # Real track where OSM has it; straight hops where it does not.
    by_line = {}
    for k, e in enumerate(edges):
        by_line.setdefault(f'부산 {e["line"]}호선', []).append(
            (k, nodes[e["a"]]["coord"], nodes[e["b"]]["coord"]))
    track, track_report = CT.fit_edges(by_line)

    features = []
    for k, e in enumerate(edges):
        sa, sb = nodes[e["a"]], nodes[e["b"]]
        shape = track.get(k) or [sa["coord"], sb["coord"]]
        features.append({"type": "Feature", "geometry": {"type": "LineString", "coordinates": shape},
            "properties": {"line": f'부산 {e["line"]}호선', "from": sa["name"], "to": sb["name"],
                "daily_down": round(center[k, 0], 1), "daily_up": round(center[k, 1], 1),
                "daily": round(center[k].sum(), 1), "daily_low": round(sums[:, k].min(), 1),
                "daily_high": round(sums[:, k].max(), 1), "km": e["km"],
                "daily_down_low": round(values[:, k, 0].min(), 1), "daily_down_high": round(values[:, k, 0].max(), 1),
                "daily_up_low": round(values[:, k, 1].min(), 1), "daily_up_high": round(values[:, k, 1].max(), 1),
                "source": "busan_gravity", "period": month + " Mon-Fri mean",
                "geometry_source": ("osm" if k in track else "straight"), "estimated": True}})
    report.update({"station_complexes": len(names), "platforms": len(nodes), "segments": len(edges),
        "month": month, "day_filter": "Monday-Friday (no holiday exclusion)", "scenarios": scenario_report,
        "track_geometry": track_report,
        "sensitivity_median_fraction": float(np.median((sums.max(axis=0)-sums.min(axis=0))/sums[1])),
        "sensitivity_max_fraction": float(np.max((sums.max(axis=0)-sums.min(axis=0))/sums[1])),
        "limits": "Uncalibrated gravity OD, not observed trips. Gate fit is an accounting check, not independent validation. Lines 1-4 only; external rail transfers are treated as gate demand. Scenario bounds are not confidence intervals."})
    report["source_sha256"] = {name: hashlib.sha256((folder/name).read_bytes()).hexdigest()
                               for name in ("stations.csv", "distances.csv", "counts.csv")}
    return {"type": "FeatureCollection", "features": features,
        "line_meta": {f"부산 {i}호선": {"en": f"Busan Line {i} (est.)", "color": COLORS[i], "cls": "metro", "estimated": True, "period": month, "city": "부산"} for i in range(1, 5)},
        "station_names": {n["name"]: n["en"] for n in nodes.values()}, "model_report": report}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--month", default="2026-07")
    parser.add_argument("--source", type=Path, default=D)
    parser.add_argument("--output", type=Path, default=HERE / "data" / "busan_segments.geojson")
    args = parser.parse_args()
    result = build(args.source, args.month)
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    temporary.replace(args.output)
    (args.source / "model_report.json").write_text(json.dumps(result["model_report"], indent=2), encoding="utf-8")
    print(json.dumps(result["model_report"], indent=2))


if __name__ == "__main__":
    main()
