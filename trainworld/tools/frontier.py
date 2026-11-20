"""T-3887: the mode frontier. Builds one straight test line of each mode along real corridors in a city pack, lets
the game calculate its trips, reads its day's money as the game shows it, and writes a CSV; `chart` draws it.

    python trainworld/tools/frontier.py corridors                       the corridors and their density
    python trainworld/tools/frontier.py sweep --port 6196 [--out f.csv] [--only dense:10:metro,...] [--fare-bus 0.8]
    python trainworld/tools/frontier.py chart [f.csv] [--png out.png]
    python trainworld/tools/frontier.py table [f.csv]                   every point's modes, best first
    python trainworld/tools/frontier.py shares --port 6196 [--out f.csv]  trip shares by trip length and stop spacing
    python trainworld/tools/frontier.py sharechart [f.csv] [--png out.png]

`sweep` needs a Vite on the port and opens its own game there (play.mjs, sandbox). Each corridor x length x mode
is a fresh game (`play.mjs reload`), so lines never share riders; each headway set is one more calculation on it.
The CSV is appended row by row and a rerun skips rows already in it, so a stopped sweep resumes. Method and
findings: notes/T-3887.md.
"""
import argparse, csv, json, math, os, subprocess, sys, time

TREE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # the maps root of this tree
TW = os.path.join(TREE, "trainworld")
PLAY = os.path.join(TW, "checks", "play.mjs")
PAGE_LIB = os.path.join(TW, "tools", "frontier.js")
ART = os.path.join(TREE, "artifacts", "trainworld")
DEFAULT_CSV = os.path.join(ART, "t3887-frontier.csv")
SCRATCH = os.environ.get("TW_FRONTIER_SCRATCH", os.path.join(os.environ.get("TEMP", "."), "tw-frontier"))

CITY = "cic"  # Chicago: flat, inland on three sides, a grid of straight arterials, US prices
# name: (centre lon, lat), bearing in degrees (0 north). Straight lines centred on the point, so a longer corridor
# reaches out past the place it is named for.
CORRIDORS = {
    "dense": ((-87.638, 41.882), -12.5),   # through the Loop, north along the lakefront, south to the South Side
    "medium": ((-87.6866, 41.90), 0.0),    # Western Avenue, through the North and West Sides
    "thin": ((-88.10, 41.86), 0.0),        # DuPage County suburbs, through Wheaton
}
LENGTHS_KM = [2, 5, 10, 20, 40]
CATCH_M = 800  # people and jobs within this of the line are its corridor's demand (the denominator)

# mode: stop spacing (m; None = the bus's own stops), level, track flags, cars, platform (m), shortest length (km)
MODES = {
    "bus":         dict(spacing=None, level=0, cars=None, platform=None, min_km=0),
    "bus_1km":     dict(spacing=None, bus_spacing=1000, level=0, cars=None, platform=None, min_km=0),  # its own stops 1 km apart
    "street_tram": dict(spacing=500, level=0, tram=True, street=True, cars=None, platform=40, min_km=0),
    "tram":        dict(spacing=700, level=0, tram=True, cars=None, platform=40, min_km=0),
    "metro":       dict(spacing=1000, level=-2, cars=6, platform=120, min_km=0),
    "rail":        dict(spacing=3000, level=1, cars=8, platform=160, min_km=0),
    "hsr":         dict(spacing=20000, level=1, hsr=True, fast=True, cars=8, platform=160, min_km=40),
}
# headways at high, medium and low demand, seconds: the bus's new-line default, twice as often, and four times
# S only where A was the best of A-C (a thin corridor), D only where C was and its trains were still full. Levels run
# in this order on one build, so the cash spent on trains so far is always the fleet of the level just run.
SERVICES = {"S": [900, 1800, 3600], "A": [480, 900, 1800], "B": [240, 480, 900], "C": [120, 240, 480], "D": [60, 120, 240]}
FULL_FOR_D = 90


def wants_s(rows):
    return all(s in rows for s in "ABC") and max("ABC", key=lambda s: float(rows[s]["net_day"])) == "A"



def wants_d(rows):
    """rows: service -> row (dicts with net_day, fullest_pct) of one corridor, length and mode."""
    if not all(s in rows for s in "ABC"):
        return False
    best = max("ABC", key=lambda s: float(rows[s]["net_day"]))
    return best == "C" and float(rows["C"]["fullest_pct"] or 0) >= FULL_FOR_D
AMORTIZE_DAYS = 60  # building cost spread over 60 game days (6,000 real days at the x100 economy; the loan's term)


def centred(c, bearing_deg, L_km):
    lo, la = c
    kx, ky = 111320 * math.cos(math.radians(la)), 110540
    b = math.radians(bearing_deg)
    dx, dy = math.sin(b) * L_km * 500, math.cos(b) * L_km * 500
    return (lo - dx / kx, la - dy / ky), (lo + dx / kx, la + dy / ky)


def along(p0, p1, n):
    return [[p0[0] + (p1[0] - p0[0]) * k / n, p0[1] + (p1[1] - p0[1]) * k / n] for k in range(n + 1)]


# a build the game refuses (a stop on a river, say) is tried again with the stops between the ends moved along the
# line by this share of the spacing
PHASES = [0.0, 0.3, -0.3, 0.45]


def stations_for(mode, L_km, p0, p1, phase=0.0, spacing=None):
    sp = spacing or MODES[mode]["spacing"]
    n = max(1, round(L_km * 1000 / sp))
    if sp >= 2000 and n > 1:
        # stops this far apart: an even count of gaps, so one stop is at the corridor's centre (the Loop) and the result
        # does not swing on whether the centre happens to fall between two stops
        n = max(2, 2 * round(n / 2))
    ts = [0.0] + [(k + phase) / n for k in range(1, n)] + [1.0]
    return [[p0[0] + (p1[0] - p0[0]) * t, p0[1] + (p1[1] - p0[1]) * t] for t in ts]


# ---- corridor demand from the pack

_pack = None


def pack_cells():
    global _pack
    if _pack is None:
        os.environ.setdefault("OMP_NUM_THREADS", "2")
        sys.path.insert(0, os.path.join(TW, "pipeline"))
        import numpy as np, h3
        from read_pack import load_pack
        _, a = load_pack(CITY)
        ll = np.array([h3.cell_to_latlng(h3.int_to_str(int(c))) for c in a["h3"]])
        _pack = (ll[:, 1], ll[:, 0], a["pop"].astype(float), a["jobs"].astype(float))
    return _pack


def catchment(p0, p1):
    """People and jobs within CATCH_M of the segment p0-p1, and their density over the band's area (km²)."""
    import numpy as np
    lon, lat, pop, jobs = pack_cells()
    la0 = (p0[1] + p1[1]) / 2
    kx, ky = 111320 * math.cos(math.radians(la0)), 110540
    x, y = (lon - p0[0]) * kx, (lat - p0[1]) * ky
    ex, ey = (p1[0] - p0[0]) * kx, (p1[1] - p0[1]) * ky
    L2 = ex * ex + ey * ey
    t = np.clip((x * ex + y * ey) / L2, 0, 1)
    d = np.hypot(x - t * ex, y - t * ey)
    m = d < CATCH_M
    L = math.sqrt(L2)
    area = (2 * CATCH_M * L + math.pi * CATCH_M ** 2) / 1e6
    people, work = float(pop[m].sum()), float(jobs[m].sum())
    return people, work, (people + work) / area


def cmd_corridors(_a):
    for name, (c, b) in CORRIDORS.items():
        for L in LENGTHS_KM:
            p0, p1 = centred(c, b, L)
            pe, jo, dens = catchment(p0, p1)
            print(f"{name:7s} {L:3d} km  {p0[0]:.4f},{p0[1]:.4f} -> {p1[0]:.4f},{p1[1]:.4f}  people {pe:9.0f}  jobs {jo:9.0f}  {dens:7.0f} a km²")


# ---- the game

def play(*args, timeout=200):
    r = subprocess.run(["node", PLAY, *args], cwd=TREE, capture_output=True, text=True, timeout=timeout,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    line = (r.stdout.strip().splitlines() or [""])[-1]
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return {"ok": False, "error": (r.stdout + r.stderr)[-800:]}


def page_call(port, call, arg):
    os.makedirs(SCRATCH, exist_ok=True)
    f = os.path.join(SCRATCH, f"call-{port}.js")
    with open(PAGE_LIB, encoding="utf-8") as src, open(f, "w", encoding="utf-8") as out:
        out.write(src.read() + f"\nreturn await FRONTIER.{call}({json.dumps(arg)});\n")
    r = play("do", "--file", f, "--port", str(port), "--timeout", "170", "--full")
    if not r.get("ok"):
        raise RuntimeError(f"{call}: {r.get('error')}")
    return r["result"]


def spec_for(mode, corridor, L, phase=0.0, spacing=None):
    m = MODES[mode]
    p0, p1 = centred(*CORRIDORS[corridor], L)
    s = dict(mode=mode, name=f"{corridor}-{L}-{mode}", level=m["level"], cars=m["cars"], platform=m["platform"],
             hsr=m.get("hsr", False), tram=m.get("tram", False), street=m.get("street", False), fast=m.get("fast", False))
    if mode.startswith("bus"):
        s["mode"] = "bus"
        s["busSpacing"] = m.get("bus_spacing") if spacing is None else (spacing if spacing > 400 else None)
        s["path"] = along(p0, p1, max(1, round(L * 10)))
    else:
        s["stations"] = stations_for(mode, L, p0, p1, phase, spacing)
    return s


FIELDS = ["city", "corridor", "length_km", "mode", "service", "headways_s", "stations", "people", "jobs", "density_km2",
          "line_km", "cars", "trains_needed", "round_trip_min", "riders_day", "rides_day", "peak_per_hour", "fullest_pct",
          "fares_day", "running_day", "upkeep_day", "built", "fleet", "amortized_day", "profit_day", "net_day",
          "net_per_1000_catchment", "profit_per_rider", "net_per_rider", "payback_days", "fare_bus", "issues"]


def done_rows(path):
    """(corridor, length, mode) -> service -> row, of the rows already written"""
    out = {}
    if os.path.exists(path):
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                out.setdefault((r["corridor"], r["length_km"], r["mode"]), {})[r["service"]] = r
    return out


def cmd_sweep(a):
    port = a.port
    out = a.out
    os.makedirs(os.path.dirname(out), exist_ok=True)
    have = done_rows(out)
    only = set(a.only.split(",")) if a.only else None
    new = not os.path.exists(out)
    f = open(out, "a", newline="", encoding="utf-8")
    wr = csv.DictWriter(f, FIELDS)
    if new:
        wr.writeheader()
    for corridor in CORRIDORS:
        for L in LENGTHS_KM:
            p0, p1 = centred(*CORRIDORS[corridor], L)
            people, jobs, dens = catchment(p0, p1)
            for mode in MODES:
                if L < MODES[mode]["min_km"]:
                    continue
                if only and f"{corridor}:{L}:{mode}" not in only:
                    continue
                rows = have.get((corridor, str(L), mode), {})
                if all(s in rows for s in "ABC") and ("D" in rows or not wants_d(rows)) and ("S" in rows or not wants_s(rows)):
                    continue
                t0 = time.time()
                b = None
                for phase in PHASES if not mode.startswith("bus") else [0.0]:
                    r = play("reload", CITY, "--port", str(port), "--timeout", "170")
                    if not r.get("ok"):
                        print("reload failed", r, flush=True)
                        break
                    play("do", "wait.final()", "--port", str(port), "--timeout", "170")
                    if a.fare_bus is not None:
                        page_call_raw(port, {"op": "fareBus", "mul": a.fare_bus})
                    b = page_call(port, "build", spec_for(mode, corridor, L, phase))
                    if b.get("line") is not None and not b.get("issues"):
                        break
                    print(f"{corridor} {L} {mode}: build at phase {phase}: {b.get('issues')}", flush=True)
                    b = None
                if b is None:
                    continue
                fleet = 0.0
                for s in SERVICES:
                    if s in rows or (s == "S" and not wants_s(rows)) or (s == "D" and not wants_d(rows)):
                        continue
                    m = page_call(port, "service", {"line": b["line"], "seconds": SERVICES[s]})
                    if not m.get("ok"):
                        print(f"{corridor} {L} {mode} {s}: {m}", flush=True)
                        break
                    fleet += m["bought"] * 1e6
                    built = b["built"] * 1e6
                    am = (built + fleet) / AMORTIZE_DAYS
                    profit = m["fares"] - m["running"] - m["upkeep"]
                    net = profit - am
                    row = dict(city=CITY, corridor=corridor, length_km=L, mode=mode, service=s,
                               headways_s="/".join(map(str, SERVICES[s])), stations=b["stops"], people=round(people),
                               jobs=round(jobs), density_km2=round(dens), line_km=round(m["km"], 2), cars=m["cars"],
                               trains_needed="/".join(str(round(x)) for x in m["need"]), round_trip_min=round(m["roundTripS"] / 60, 1),
                               riders_day=round(m["riders"]), rides_day=round(m["rides"]), peak_per_hour=round(m["peakPerHour"]),
                               fullest_pct=None if m["fullestPct"] is None else round(m["fullestPct"]),
                               fares_day=round(m["fares"]), running_day=round(m["running"]), upkeep_day=round(m["upkeep"]),
                               built=round(built), fleet=round(fleet), amortized_day=round(am), profit_day=round(profit),
                               net_day=round(net), net_per_1000_catchment=round(net / max(1.0, people + jobs) * 1000, 1),
                               profit_per_rider=round(profit / m["riders"], 2) if m["riders"] > 0 else None,
                               net_per_rider=round(net / m["riders"], 2) if m["riders"] > 0 else None,
                               payback_days=round((built + fleet) / profit, 1) if profit > 0 else None,
                               fare_bus=a.fare_bus, issues="; ".join((b.get("issues") or []) + ([f"{b['dropped']} stops over water left out"] if b.get("dropped") else [])))
                    rows[s] = row
                    wr.writerow(row)
                    f.flush()
                    print(f"{corridor:6s} {L:3d} {mode:11s} {s} riders {row['riders_day']:>8} profit {profit/1e6:8.2f}M net {net/1e6:8.2f}M "
                          f"peak {row['peak_per_hour']:>6}/h full {row['fullest_pct']}  ({time.time() - t0:.0f} s)", flush=True)
    f.close()


def page_call_raw(port, op):
    os.makedirs(SCRATCH, exist_ok=True)
    f = os.path.join(SCRATCH, f"op-{port}.js")
    with open(f, "w", encoding="utf-8") as out:
        out.write(f"return await page.evaluate((op) => window.tw.client.edit(op), {json.dumps(op)});\n")
    return play("do", "--file", f, "--port", str(port), "--timeout", "60")


# ---- the share view: trip length against stop spacing, each mode alone against the car and walking

SHARE_CORRIDOR, SHARE_LINE_KM = "medium", 28
SHARE_MODES = ["bus", "street_tram", "tram", "metro"]  # rail runs the same trains as the metro: the same trips
SHARE_SPACINGS = [400, 800, 1500, 3000]  # m; the bus at 400 is its own placing (stops on the 400 m street grid)
SHARE_LENGTHS = [1, 2, 3, 5, 8, 12, 18]  # km, a trip along the line
SHARE_ORIGINS = 25  # trips of each length from this many starts spread along the line, averaged
SHARE_FIELDS = ["corridor", "mode", "spacing_m", "trip_km", "offset_m", "origin", "p_walk", "p_train", "p_drive", "rail_offered", "why",
                "rail_perceived", "road_perceived", "issues"]


def share_pairs(corridor, line_km, trip_km, n, offset_m=0):
    """n trips of trip_km along the corridor's line, their starts spread over it (the ends 1 km inside the line), both
    ends moved offset_m off the line to its right."""
    c, b = CORRIDORS[corridor]
    p0, p1 = centred(c, b, line_km)
    kx, ky = 111320 * math.cos(math.radians(c[1])), 110540
    ox, oy = math.cos(math.radians(b)) * offset_m / kx, -math.sin(math.radians(b)) * offset_m / ky
    p0, p1 = (p0[0] + ox, p0[1] + oy), (p1[0] + ox, p1[1] + oy)
    span = line_km - 2 - trip_km
    out = []
    for k in range(n):
        s0 = 1 + span * (k + 0.5) / n
        t0, t1 = s0 / line_km, (s0 + trip_km) / line_km
        out.append(([p0[0] + (p1[0] - p0[0]) * t0, p0[1] + (p1[1] - p0[1]) * t0],
                    [p0[0] + (p1[0] - p0[0]) * t1, p0[1] + (p1[1] - p0[1]) * t1]))
    return out


def cmd_shares(a):
    port, out = a.port, a.out
    have = set()
    if os.path.exists(out):
        with open(out, newline="", encoding="utf-8") as f:
            have = {(r["mode"], r["spacing_m"]) for r in csv.DictReader(f)}
    new = not os.path.exists(out)
    f = open(out, "a", newline="", encoding="utf-8")
    wr = csv.DictWriter(f, SHARE_FIELDS)
    if new:
        wr.writeheader()
    offsets = [int(x) for x in a.offsets.split(",")]
    pairs = [(L, o, k, pr) for o in offsets for L in SHARE_LENGTHS
             for k, pr in enumerate(share_pairs(SHARE_CORRIDOR, SHARE_LINE_KM, L, SHARE_ORIGINS, o))]
    for mode in a.modes.split(","):
        for sp in [int(x) for x in a.spacings.split(",")] if mode != "none" else [0]:
            if (mode, str(sp)) in have:
                continue
            t0 = time.time()
            b = None
            for phase in PHASES if mode not in ("bus", "none") else [0.0]:
                play("reload", CITY, "--port", str(port), "--timeout", "170")
                play("do", "wait.final()", "--port", str(port), "--timeout", "170")
                if mode == "none":
                    b = {"line": None, "issues": []}
                    break
                b = page_call(port, "build", spec_for(mode, SHARE_CORRIDOR, SHARE_LINE_KM, phase, sp))
                if b.get("line") is not None and not b.get("issues"):
                    break
                print(f"{mode} {sp}: build at phase {phase}: {b.get('issues')}", flush=True)
                b = None
            if b is None:
                continue
            if b["line"] is not None:
                m = page_call(port, "service", {"line": b["line"], "seconds": SERVICES["A"]})
                if not m.get("ok"):
                    print(f"{mode} {sp}: {m}", flush=True)
                    continue
            res = page_call(port, "trips", {"pairs": [pr for *_, pr in pairs]})
            for (L, o, k, _), r in zip(pairs, res):
                if r is None:
                    continue
                wr.writerow(dict(corridor=SHARE_CORRIDOR, mode=mode, spacing_m=sp, trip_km=L, offset_m=o, origin=k, p_walk=round(r["pWalk"], 4),
                                 p_train=round(r["pTrain"], 4), p_drive=round(r["pDrive"], 4), rail_offered=r["rail"], why=r["why"],
                                 rail_perceived=None if r["railPerceived"] is None else round(r["railPerceived"], 1),
                                 road_perceived=round(r["roadPerceived"], 1), issues="; ".join(b.get("issues") or [])))
            f.flush()
            tr = [r["pTrain"] for r in res if r]
            print(f"{mode:11s} {sp:5d}  mean train share {sum(tr) / max(1, len(tr)):.3f}  ({time.time() - t0:.0f} s)", flush=True)
    f.close()


def cmd_table(a):
    """Each corridor and length: every mode at its best service, best first (net a day, US$M)."""
    from frontier_chart import best_rows, ORDER
    best = best_rows(a.csv)
    keys = sorted({(k[0], k[1]) for k in best}, key=lambda k: (list(CORRIDORS).index(k[0]), k[1]))
    for cor, L in keys:
        rs = sorted((best[(cor, L, m)] for m in ORDER if (cor, L, m) in best), key=lambda r: -float(r["net_day"]))
        print(f"{cor:6s} {L:3d} km  " + "  ".join(
            f"{r['mode']}:{float(r['net_day']) / 1e6:.1f}({r['service']},{int(r['riders_day']) // 1000}k)" for r in rs))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("corridors")
    s = sub.add_parser("sweep")
    s.add_argument("--port", type=int, required=True)
    s.add_argument("--out", default=DEFAULT_CSV)
    s.add_argument("--only", default=None, help="corridor:length:mode,... (e.g. medium:5:bus)")
    s.add_argument("--fare-bus", type=float, default=None, help="the bus fare multiple (fareBus) for the whole sweep")
    sh = sub.add_parser("shares")
    sh.add_argument("--port", type=int, required=True)
    sh.add_argument("--out", default=os.path.join(ART, "t3887-shares.csv"))
    sh.add_argument("--modes", default=",".join(["none"] + SHARE_MODES))
    sh.add_argument("--spacings", default=",".join(map(str, SHARE_SPACINGS)))
    sh.add_argument("--offsets", default="0", help="m off the line for both trip ends, a list")
    sc = sub.add_parser("sharechart")
    sc.add_argument("csv", nargs="?", default=os.path.join(ART, "t3887-shares.csv"))
    sc.add_argument("--png", default=os.path.join(ART, "t3887-shares.png"))
    t = sub.add_parser("table")
    t.add_argument("csv", nargs="?", default=DEFAULT_CSV)
    c = sub.add_parser("chart")
    c.add_argument("csv", nargs="?", default=DEFAULT_CSV)
    c.add_argument("--png", default=os.path.join(ART, "t3887-frontier.png"))
    a = ap.parse_args()
    if a.cmd == "corridors":
        cmd_corridors(a)
    elif a.cmd == "sweep":
        cmd_sweep(a)
    elif a.cmd == "shares":
        cmd_shares(a)
    elif a.cmd == "sharechart":
        from frontier_chart import share_chart
        share_chart(a.csv, a.png)
    elif a.cmd == "table":
        cmd_table(a)
    else:
        from frontier_chart import chart
        chart(a.csv, a.png)


if __name__ == "__main__":
    main()
