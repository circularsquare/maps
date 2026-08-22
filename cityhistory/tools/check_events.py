"""Validate data/events.json -- the hand-curated history notes the viewer pops up on the map.

The notes are written by hand (and by agents working a region or an era at a time), so the
things that go wrong are typos in coordinates, a note pinned to a year the map cannot show,
overlong text that runs off the screen, and -- the one you cannot see by reading the file --
six notes all firing inside the same two seconds of playback.

    python tools/check_events.py             # errors + warnings
    python tools/check_events.py --coverage  # + a century x region table of what is covered
    python tools/check_events.py --file X.json   # check a draft before merging it in

Errors are things the viewer will get wrong. Warnings are judgement calls; several of the seed
notes trip them on purpose (Columbus points at open ocean, and that is the point).
"""
import json, math, sys, pathlib

sys.stdout.reconfigure(encoding='utf-8', errors='replace')   # city names are not cp1252

ROOT = pathlib.Path(__file__).resolve().parent.parent
EV = ROOT / "data" / "events.json"
CITIES = ROOT / "data" / "cities.json"

IMPACTS = {-3: "abandoned", -2: "heavy loss, recovers", -1: "mild decline",
           0: "marker, nothing yet", 1: "steady growth", 2: "strong", 3: "explosive"}
REQUIRED = {"y", "la", "lo", "t", "im"}
OPTIONAL = {"d", "w", "p", "pri"}

T_MAX = 34      # headline chars. Longer and it starts crowding the leaderboard at 1400px.
D_MAX = 66      # detail chars, ~400px at 11.5px Nunito -- a quarter of the map's width.

# Viewer constants, mirrored -- keep in step with index.html. The viewer fires a note when the
# run crosses its year and holds it for NOTE_SECONDS of wall clock; during playback at 1x the
# scrub fraction advances at a constant PLAY_SPEED, so that is exactly a window of
# NOTE_SECONDS * PLAY_SPEED in s, which is what the crowding and density checks below measure.
NOTE_SECONDS = 3.0     # NOTE_IN is inside the hold, so it is (NOTE_HOLD + NOTE_OUT) / 1000
PLAY_SPEED = 0.016
NOTE_MAX = 5   # a note that cannot find screen room near its anchor is also dropped, which
               # this cannot model -- so the draw counts below are an optimistic bound


# ---------- the timeline warp, ported from index.html ----------
def seg_y(ymin, ymax):
    return [max(ymin, -3500), 1, 1400, 1900, ymax]

def seg_s(Y):
    a, b = (Y[1] - Y[0]) / 2, Y[2] - Y[1]     # pre-AD 1 counted at half rate
    return [0, 0.5 * a / (a + b), 0.5, 0.75, 1]

def year_to_s(y, Y, S):
    y = max(Y[0], min(Y[4], y))
    for i in range(4):
        if y <= Y[i + 1]:
            return S[i] + (y - Y[i]) / (Y[i + 1] - Y[i]) * (S[i + 1] - S[i])
    return 1.0

def half_window(e):
    return 0.5 * NOTE_SECONDS * PLAY_SPEED * e.get("w", 1)


# ---------- the display floor, ported from index.html ----------
FLOOR_KNOTS = [(600, 5000), (700, 10000), (1400, 10000), (1500, 20000)]

def map_floor(y):
    K = FLOOR_KNOTS
    if y <= K[0][0]:
        return K[0][1]
    for i in range(1, len(K)):
        if y <= K[i][0]:
            a, b = K[i - 1], K[i]
            if a[1] == b[1]:
                return a[1]
            f = (y - a[0]) / (b[0] - a[0])
            return 10 ** (math.log10(a[1]) + (math.log10(b[1]) - math.log10(a[1])) * f)
    return K[-1][1]


# ---------- region buckets, ported from index.html ----------
def region_of(la, lo):
    if lo <= -120 and -55 <= la <= 30:
        return "Oceania"
    if lo <= -34:
        return "North America" if (la >= 13 or (la >= 7 and lo <= -77)) else "South America"
    if (la <= -11 and lo >= 112) or (lo >= 141 and la <= 25) or (lo >= 155 and la < 30):
        return "Oceania"
    if 26 <= lo <= 63 and 12 <= la <= 42:
        red_sea = 33 + (30 - la) * 0.59
        if not (la < 22 and lo < red_sea):
            return "Middle East"
    if -34 <= lo <= 60:
        if la < 38 and 6 <= lo <= 11.5:
            return "Africa"
        if la >= 37:
            return "Europe"
        if la >= 36 and lo <= -0.5:
            return "Europe"
        if la >= 35.8 and 11 <= lo <= 19:
            return "Europe"
        if la >= 34 and 20 <= lo <= 26:
            return "Europe"
        if la >= -38:
            return "Africa"
    if 60 <= lo <= 92 and 4 <= la <= 37:
        return "South Asia"
    if 92 <= lo <= 95 and 22 <= la <= 30:
        return "South Asia"
    if 92 <= lo <= 141 and -11 <= la < 22:
        return "Southeast Asia"
    if 95 <= lo <= 98 and 22 <= la <= 28:
        return "Southeast Asia"
    if 98 <= lo <= 146 and 22 <= la <= 54:
        return "East Asia"
    if 130 <= lo <= 180 and 22 <= la <= 60:
        return "East Asia"
    return "Central Asia"


def fmt_pop(p):
    if p >= 1e6:
        return f"{p / 1e6:.1f}M"
    if p >= 1e3:
        return f"{p / 1e3:.0f}k"
    return str(int(p))


def fmt_year(y):
    return f"{-y}BC" if y < 0 else str(int(y))


def haversine(la1, lo1, la2, lo2):
    p = math.pi / 180
    a = (0.5 - math.cos((la2 - la1) * p) / 2
         + math.cos(la1 * p) * math.cos(la2 * p) * (1 - math.cos((lo2 - lo1) * p)) / 2)
    return 12742 * math.asin(math.sqrt(max(0.0, a)))


def pop_at(pts, y):
    if not pts or y < pts[0][0] or y > pts[-1][0]:
        return 0.0
    lo, hi = 0, len(pts) - 1
    while hi - lo > 1:
        m = (lo + hi) // 2
        if pts[m][0] <= y:
            lo = m
        else:
            hi = m
    a, b = pts[lo], pts[hi]
    if b[0] == a[0]:
        return float(a[1])
    f = (y - a[0]) / (b[0] - a[0])
    return 10 ** (math.log10(a[1]) + (math.log10(b[1]) - math.log10(a[1])) * f)


def main():
    coverage = "--coverage" in sys.argv
    # --file lets one lane of a fan-out validate its own draft before it is merged in. Accepts
    # either the full wrapper or a bare array of notes.
    src = EV
    if "--file" in sys.argv:
        src = pathlib.Path(sys.argv[sys.argv.index("--file") + 1])
    raw = json.loads(src.read_text(encoding="utf-8"))
    events = raw["events"] if isinstance(raw, dict) else raw
    data = json.loads(CITIES.read_text(encoding="utf-8"))
    cities = data["cities"]
    Y = seg_y(data["yearMin"], data["yearMax"])
    S = seg_s(Y)

    errors, warns = [], []

    def err(i, msg):
        errors.append(f"  [{i}] {events[i].get('t', '?')!r} ({events[i].get('y', '?')}): {msg}")

    def warn(i, msg):
        warns.append(f"  [{i}] {events[i].get('t', '?')!r} ({events[i].get('y', '?')}): {msg}")

    # ---- per-note schema and text ----
    for i, e in enumerate(events):
        keys = set(e)
        for k in REQUIRED - keys:
            err(i, f"missing required key {k!r}")
        for k in keys - REQUIRED - OPTIONAL:
            err(i, f"unknown key {k!r} (allowed: {sorted(REQUIRED | OPTIONAL)})")
        if REQUIRED - keys:
            continue
        if not isinstance(e["y"], int):
            err(i, "y must be a whole year (negative = BC; there is no year 0)")
        if e["y"] == 0:
            err(i, "year 0 does not exist -- see spec 3.2b")
        if not (Y[0] <= e["y"] <= Y[4]):
            err(i, f"year outside the map's timeline {Y[0]}..{Y[4]}; it will never be drawn")
        if not (-85 <= e["la"] <= 85) or not (-180 <= e["lo"] <= 180):
            err(i, f"coordinate out of range ({e['la']}, {e['lo']})")
        if e["im"] not in IMPACTS:
            err(i, f"im={e['im']!r} must be one of "
                   + ", ".join(f"{k} ({v})" for k, v in IMPACTS.items()))
        if not (0.5 <= e.get("w", 1) <= 3):
            err(i, f"w={e.get('w')} outside 0.5..3")
        if e.get("pri", 2) not in (1, 2, 3):
            err(i, f"pri={e.get('pri')} must be 1 (headline), 2 (strong) or 3 (flavour)")
        if len(e["t"]) > T_MAX:
            warn(i, f"headline is {len(e['t'])} chars, over {T_MAX}")
        if e["t"].endswith("."):
            warn(i, "headline ends in a full stop; the others do not")
        if "d" in e and len(e["d"]) > D_MAX:
            warn(i, f"detail is {len(e['d'])} chars, over {D_MAX} -- it will run a long way "
                    "across the map")

    # ---- does it point at anything? ----
    for i, e in enumerate(events):
        if REQUIRED - set(e):
            continue
        floor = map_floor(e["y"])
        best = best_any = None
        for c in cities:
            d = haversine(e["la"], e["lo"], c["la"], c["lo"])
            if best_any is None or d < best_any[0]:
                best_any = (d, c, 0.0)
            p = pop_at(c["p"], e["y"])
            if p >= floor and (best is None or d < best[0]):
                best = (d, c, p)
        if best is None:
            warn(i, "no city anywhere is on the map in this year -- check the year")
        elif best[0] > 400:
            near = f"{best[1]['n']} {best[0]:.0f}km"
            if best_any and best_any[0] < best[0]:
                near += f" (nearest city of any year: {best_any[1]['n']} {best_any[0]:.0f}km)"
            warn(i, f"nothing on the map within 400km -- {near}. Fine if that is the point "
                    "(Columbus, Gulf oil), a typo otherwise")

    # ---- does the note's impact match what the bubble under it actually does? ----
    # A headline like "X falls" coloured as growth is the thing to catch: either the word is
    # loaded and the map disagrees, or the colour is wrong. Compares the nearest drawn city at
    # the note's year against the same city fifty years later.
    for i, e in enumerate(events):
        if (REQUIRED - set(e)) or e["im"] == 0:
            continue
        floor = map_floor(e["y"])
        best = None
        for c in cities:
            d = haversine(e["la"], e["lo"], c["la"], c["lo"])
            if d > 60:      # tight: at 120km it starts testing a note against a neighbour
                continue
            p0 = pop_at(c["p"], e["y"])
            if p0 >= floor and (best is None or d < best[0]):
                best = (d, c, p0)
        if best is None:
            continue                       # regional or marker-ish note; nothing to test against
        _, c, p0 = best
        if not p0:
            continue
        # Not the endpoint: im -2 means "heavy loss, RECOVERS LATER", so Warsaw being bigger in
        # 1994 than in 1944 says nothing. The question is whether the curve dips (or rises) at
        # all inside the window, so take the extreme rather than the far end.
        # Straddle the year rather than start at it: the source often books the whole drop at a
        # control point slightly BEFORE the note (Tokyo is already 2.8M in 1945, down from 6.8M in
        # 1940), so a window starting at y sees nothing but the recovery. Baseline is y-15.
        W, BACK = 60, 15
        p0 = pop_at(c["p"], e["y"] - BACK) or p0
        samples = [pop_at(c["p"], e["y"] + k) for k in range(-BACK + 2, W + 1, 2)]
        samples = [p for p in samples if p]
        if not samples:
            continue
        lo_r, hi_r = min(samples) / p0, max(samples) / p0
        if e["im"] < 0 and lo_r > 0.92:
            warn(i, f"says decline but {c['n']} never dips in the next {W} years "
                    f"(low is {fmt_pop(min(samples))} against {fmt_pop(p0)}) -- loaded wording, "
                    "or the wrong im")
        elif e["im"] > 0 and hi_r < 1.08:
            warn(i, f"says growth but {c['n']} never rises in the next {W} years "
                    f"(high is {fmt_pop(max(samples))} against {fmt_pop(p0)})")

    # ---- crowding, measured by actually playing the thing ----
    # Counting neighbours told us 67 notes were "crowded" and nothing about which ones a viewer
    # would really see. So run the viewer's own loop instead: step the scrub fraction at
    # PLAY_SPEED, fire what the step crosses, and apply the same sort and NOTE_MAX cap draw()
    # applies. What comes out is the only question that matters -- which notes never appear.
    def simulate(sel):
        # Mirrors fireNotes/drawNotes exactly, and the exactness matters: since placement was
        # frozen, a note that finds room keeps its slot for its whole life, and one that does not
        # is never shown AT ALL. There is no eviction and no partial appearance -- so the useful
        # question is which notes get blocked, not which get cut short.
        STEP = 0.1                                   # seconds of playback per tick
        dur = lambda k: NOTE_SECONDS * sel[k].get("w", 1)
        fired, draws, busy, ticks = {}, {k: 0 for k in range(len(sel))}, 0, 0
        pos = sorted(((year_to_s(e["y"], Y, S), sel[k].get("pri", 2), k)
                      for k, e in enumerate(sel)), key=lambda r: (r[0], r[1]))
        t, s_prev = 0.0, -1.0
        while True:
            s = t * PLAY_SPEED
            live = [k for k, t0 in fired.items() if t - t0 < dur(k)]
            if s > 1 and not live:
                break        # the run has ended; keep ticking while anything is still fading,
                             # because setPlaying(false) hands the fade to noteFrames() and the
                             # last notes of the timeline do get their full three seconds
            if s <= 1:
                for es, pri, k in pos:               # best pri first within one instant
                    # each tier leaves a slot of headroom for the tier above it -- see fireNotes
                    cap = NOTE_MAX + 1 if pri == 1 else NOTE_MAX + 1 - pri
                    if s_prev < es <= s and len(live) < cap:
                        fired[k] = t
                        live.append(k)
            for k in live:
                draws[k] += 1
            if s <= 1:
                busy += len(live) >= NOTE_MAX
                ticks += 1
            s_prev, t = s, t + STEP
        return draws, busy / ticks

    # Simulate the level the viewer actually ships with (`noteLevel`, the key/more/all selector),
    # not the whole file -- "all" is deliberately more notes than one run has room for.
    level = int(sys.argv[sys.argv.index("--level") + 1]) if "--level" in sys.argv else 2
    sel = [e for e in events if not (REQUIRED - set(e)) and e.get("pri", 2) <= level]
    draws, busy_frac = simulate(sel)
    never = [(sel[k]["y"], sel[k]["t"], sel[k].get("pri", 2))
             for k in range(len(sel)) if draws[k] == 0]
    thin = [(sel[k]["y"], sel[k]["t"], draws[k] * 0.1)
            for k in range(len(sel)) if 0 < draws[k] <= 8]

    # ---- duplicates ----
    for i, e in enumerate(events):
        for j in range(i + 1, len(events)):
            o = events[j]
            if (REQUIRED - set(e)) or (REQUIRED - set(o)):
                continue
            if e["t"].lower() == o["t"].lower():
                err(i, f"headline duplicates [{j}]")
            elif abs(e["y"] - o["y"]) <= 3 and haversine(e["la"], e["lo"], o["la"], o["lo"]) < 100:
                warn(i, f"same place, same decade as [{j}] {o['t']!r} -- is one of them redundant?")

    for label, rows in (("ERRORS", errors), ("WARNINGS", warns)):
        if rows:
            print(f"{label} ({len(rows)})")
            print("\n".join(rows))
            print()
    if not errors and not warns:
        print(f"{len(events)} notes, clean.")
    else:
        print(f"{len(events)} notes: {len(errors)} errors, {len(warns)} warnings.")

    # ---- what a viewer actually sees ----
    tier = {1: "key", 2: "more", 3: "all"}[level]
    print(f"\nPLAYBACK  (at 1x, notes={tier} -- {len(sel)} of {len(events)} notes; "
          f"{NOTE_MAX} fit on screen)")
    print(f"  the screen is full {busy_frac * 100:.0f}% of the run")
    if never:
        print(f"  {len(never)} notes NEVER appear -- they lose every slot they compete for:")
        for y, t, pri in sorted(never):
            print(f"      {fmt_year(y):>7}  pri {pri}  {t}")
    if thin:
        print(f"  {len(thin)} appear for under a second:")
        for y, t, sec in sorted(thin):
            print(f"      {fmt_year(y):>7}  {sec:.1f}s  {t}")
    if not never and not thin:
        print("  every note gets its full time on screen")

    # ---- density: the budget the whole file is spending ----
    # A playthrough at 1x is 1/PLAY_SPEED seconds and each note holds the screen for its own
    # window, so the average number on screen at once is just the total window divided by the
    # track. Past NOTE_MAX the map is a wall of text and the low-priority notes never draw.
    good = [e for e in events if not (REQUIRED - set(e))]
    print(f"\nDENSITY  (a playthrough at 1x is {1 / PLAY_SPEED:.0f}s)")
    for cut in (1, 2, 3):
        sel = [e for e in good if e.get("pri", 2) <= cut]
        if not sel:
            continue
        avg = sum(2 * half_window(e) for e in sel)
        peak = max(sum(1 for o in sel
                       if abs(year_to_s(o["y"], Y, S) - year_to_s(e["y"], Y, S)) < half_window(o))
                   for e in sel)
        label = {1: "pri 1 only", 2: "pri 1-2", 3: "all"}[cut]
        print(f"  {label:12} {len(sel):4} notes   {avg:5.2f} on screen on average, "
              f"{peak} at the worst moment")

    # History is much easier to write about when it goes wrong, so a file left to itself comes
    # back all blue. The spread is worth watching as it grows.
    print("\nIMPACT")
    for v, name in sorted(IMPACTS.items()):
        n = sum(1 for e in good if e["im"] == v)
        print(f"  {v:+d} {name:22} {n:3}  {'#' * n}")

    if coverage:
        buckets = [(-3500, -1000), (-1000, 1), (1, 500), (500, 1000), (1000, 1400),
                   (1400, 1700), (1700, 1850), (1850, 1950), (1950, 2026)]
        regions = ["North America", "South America", "Europe", "Africa", "Middle East",
                   "Central Asia", "South Asia", "Southeast Asia", "East Asia", "Oceania"]
        grid = {r: [0] * len(buckets) for r in regions}
        for e in events:
            if REQUIRED - set(e):
                continue
            r = region_of(e["la"], e["lo"])
            for bi, (a, b) in enumerate(buckets):
                if a <= e["y"] < b:
                    grid[r][bi] += 1
                    break
        head = "".join(f"{a if a > 0 else str(-a) + 'BC':>8}" for a, _ in buckets)
        print(f"\nCOVERAGE  (rows: region, cols: era start)\n{'':16}{head}")
        for r in regions:
            cells = "".join(f"{(v or '.'):>8}" for v in grid[r])
            print(f"{r:16}{cells}")
        totals = [sum(grid[r][bi] for r in regions) for bi in range(len(buckets))]
        print(f"{'TOTAL':16}" + "".join(f"{v:>8}" for v in totals))

    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
