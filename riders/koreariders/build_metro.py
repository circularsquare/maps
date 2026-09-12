"""Import the Seoul metropolitan network from seoulriders' full weekday build.

No routing is rerun. Sum hourly passenger loads, combine reverse service links,
and retain the existing track shapes. Express lines require an intermediate-stop
allocation before their service links can be presented as physical segments.
"""
import argparse
import heapq
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent


def reverse(s):
    s = dict(s)
    for a, b in [("ca", "cb"), ("a", "b"), ("ae", "be")]:
        s[a], s[b] = s[b], s[a]
    s["p"] = s["p"][2:] + s["p"][:2]
    return s


def distribute_express(stats):
    """Put express loads on the local stop graph, separately for each line.

    h includes hx, so subtract the express subset before allocating it. A rider
    passing three local segments contributes once to each, not a third to each.
    Fail if the local topology cannot explain an express link; never drop it.
    """
    graphs, templates, rows = {}, {}, []
    for s in stats["segments"]:
        h, hx = s["h"], s.get("hx", [0] * len(s["h"]))
        if len(h) != len(stats["hours"]) or len(hx) != len(h):
            raise ValueError("Invalid hourly array length")
        if any(not math.isfinite(v) or v < 0 for v in h + hx):
            raise ValueError("Invalid hourly passenger loads")
        # Independently rounded source arrays can disagree by 0.1.
        if any(x > v + 0.11 for v, x in zip(h, hx)):
            raise ValueError("Express loads exceed total loads")
        local = [max(0, v-x) for v, x in zip(h, hx)]
        n, nx = s.get("n"), s.get("nx")
        has_local = any(local) if n is None else sum(n) > sum(nx or [])
        if has_local:
            row = dict(s, h=local)
            rows.append(row)
            line, a, b = s["line"], s["ca"], s["cb"]
            p = s["p"]
            length = distance([[p[1], p[0]], [p[3], p[2]]])
            graph = graphs.setdefault(line, {})
            graph.setdefault(a, {})[b] = length
            graph.setdefault(b, {})[a] = length
            templates[(line, a, b)] = s
            templates.setdefault((line, b, a), reverse(s))
        elif any(local):
            raise ValueError("Passenger load without local service")
    allocated, extra = 0, 0
    for s in stats["segments"]:
        hx = s.get("hx", [])
        if not any(hx):
            continue
        line, start, end = s["line"], s["ca"], s["cb"]
        graph = graphs.get(line, {})
        heap, best, parent = [(0, start)], {start: 0}, {}
        while heap:
            cost, a = heapq.heappop(heap)
            if cost != best[a]:
                continue
            if a == end:
                break
            for b, weight in graph.get(a, {}).items():
                new = cost + weight
                if new < best.get(b, math.inf):
                    best[b], parent[b] = new, a
                    heapq.heappush(heap, (new, b))
        if end not in best:
            raise ValueError(f"No local path for express {line}: {start} -> {end}")
        path, b = [], end
        while b != start:
            a = parent[b]
            path.append((a, b))
            b = a
        for a, b in reversed(path):
            rows.append(dict(templates[(line, a, b)], h=hx))
        allocated += 1
        extra += sum(hx) * (len(path)-1)
    out = dict(stats, segments=rows)
    expected = sum(sum(s["h"]) for s in stats["segments"]) + extra
    actual = sum(sum(s["h"]) for s in rows)
    if abs(expected-actual) > 0.11 * len(stats["segments"]):
        raise ValueError("Express allocation did not conserve passenger traversals")
    return out, {"express_links": allocated, "extra_passenger_segments": round(extra, 1)}


def distance(coords):
    total = 0
    for (x1, y1), (x2, y2) in zip(coords, coords[1:]):
        a, b = math.radians(y1), math.radians(y2)
        h = math.sin((b-a)/2)**2 + math.cos(a)*math.cos(b)*math.sin(math.radians(x2-x1)/2)**2
        total += 12742 * math.asin(min(1, math.sqrt(h)))
    return total


def build(stats, shapes):
    if stats.get("day") != "weekday" or stats.get("build", {}).get("sample") != 1:
        raise ValueError("Expected a full weekday stats.json (sample=1)")
    stats, allocation = distribute_express(stats)
    pairs, names, source_total = {}, {}, 0
    for s in stats["segments"]:
        values = s["h"]
        if len(values) != len(stats["hours"]) or any(not math.isfinite(v) or v < 0 for v in values):
            raise ValueError("Invalid hourly passenger loads")
        # Increasing platform code, except the closing edge of Line 2's ring.
        forward = s["ca"] < s["cb"]
        if s["line"] == "2" and {s["ca"], s["cb"]} == {"0201", "0243"}:
            forward = not forward
        a, b = (s["ca"], s["cb"]) if forward else (s["cb"], s["ca"])
        key = (s["line"], a, b)
        p = s["p"]
        shape_key = f'{s["line"]}|{p[0]:.5f},{p[1]:.5f}|{p[2]:.5f},{p[3]:.5f}'
        raw = shapes.get(shape_key)
        if not raw:
            reverse_key = f'{s["line"]}|{p[2]:.5f},{p[3]:.5f}|{p[0]:.5f},{p[1]:.5f}'
            if shapes.get(reverse_key):
                raw = list(reversed(shapes[reverse_key]))
        coords = [[v[1], v[0]] for v in raw] if raw else [[p[1], p[0]], [p[3], p[2]]]
        if not forward:
            coords.reverse()
        if key not in pairs:
            pairs[key] = {"type": "Feature", "geometry": {"type": "LineString", "coordinates": coords},
                          "properties": {"line": "수도권 " + stats["line_meta"][s["line"]]["display"],
                                         "from": s["a"] if forward else s["b"],
                                         "to": s["b"] if forward else s["a"],
                                         "daily_down": 0, "daily_up": 0,
                                         "source": "seoulriders", "period": "modelled weekday",
                                         "geometry_source": "track" if raw and len(raw) > 2 else "straight"}}
        prop = pairs[key]["properties"]
        if raw and len(raw) > 2 and prop["geometry_source"] == "straight":
            pairs[key]["geometry"]["coordinates"] = coords
            prop["geometry_source"] = "track"
        value = sum(values)
        prop["daily_down" if forward else "daily_up"] += value
        source_total += value
        for ko, en in [(s["a"], s["ae"]), (s["b"], s["be"])]:
            if en:
                names[ko] = en
    features = list(pairs.values())
    for f in features:
        p = f["properties"]
        p["daily_down"], p["daily_up"] = round(p["daily_down"], 1), round(p["daily_up"], 1)
        p["daily"] = round(p["daily_down"] + p["daily_up"], 1)
        p["km"] = round(distance(f["geometry"]["coordinates"]), 4)
    assert abs(sum(f["properties"]["daily"] for f in features) - source_total) < 0.1
    meta = {"수도권 " + m["display"]: {"en": ("Seoul " if key.isdigit() else "") + m["display_en"], "color": m["color"], "cls": "metro"}
            for key, m in stats["line_meta"].items()}
    return {"type": "FeatureCollection", "features": features, "line_meta": meta,
            "station_names": names, "source_build": stats["build"], "day": stats["day"],
            "allocation": allocation,
            "note": "Modelled weekday from seoulriders; 2023 OD reweighted to weekday counts. Intercity is a separate 2022 annual average."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=HERE.parent / "seoulriders" / "data")
    parser.add_argument("--output", type=Path, default=HERE / "data" / "metro_segments.geojson")
    args = parser.parse_args()
    paths = [args.source / "stats.json", args.source / "link_shapes.json"]
    before = [(p.stat().st_mtime_ns, p.stat().st_size) for p in paths]
    raw_stats, raw_shapes = [p.read_bytes() for p in paths]
    stats, shapes = json.loads(raw_stats), json.loads(raw_shapes)
    result = build(stats, shapes)
    if before != [(p.stat().st_mtime_ns, p.stat().st_size) for p in paths]:
        raise RuntimeError("Source files changed during import; wait for the source build to finish")
    result["source_sha256"] = {"stats.json": hashlib.sha256(raw_stats).hexdigest(),
                               "link_shapes.json": hashlib.sha256(raw_shapes).hexdigest()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    temporary.replace(args.output)
    straight = sum(f["properties"]["geometry_source"] == "straight" for f in result["features"])
    print(f'{len(result["features"])} segments, {len(result["line_meta"])} lines; {straight} straight-line shapes')


if __name__ == "__main__":
    main()
