"""analyze_jumps.py -- find the big, fast jumps in the built series, so they can be inspected.

The map has a visible step at the WUP seam for most cities, plus a scatter of other jumps that
are a mix of three different things:

  real history      Lisbon 1755, Berlin 1945, a plague, a partition
  source mixing     populstat's benchmark changes unit, or two fused sources disagree
  our own doing     the seam, the fade ramps, the forward hold, a bad despike

This ranks them so the three can be told apart by eye, and tags each one with whichever of our
own mechanisms it sits on top of, so the artifacts sort themselves out of the way.

TWO THINGS MAKE THE LIST READABLE
---------------------------------
1. **Speed is measured in adjusted years**, the same warp the slider and the fade thresholds
   use (spec 2). A window of 50 adjusted years is ~2 real years after 1900, ~8 across 1400-1900,
   ~25 across AD 1-1400 and 50 before AD 1 -- so "fast" means the same thing to the eye at every
   point on the timeline, and a 40-year Bronze Age doubling does not outrank a 2-year one in
   1995. `--window` changes it; the real-year equivalents are printed every run.

2. **Big for its time**, not big outright. A jump only counts if the city is inside the top
   `--top` (default 500) of every city alive in that year, ranked at the larger end of the jump
   so collapses count as well as spikes. The threshold curve is recomputed from the built data:
   500th place is 2k in AD 1000, 70k in 1900 and 1.06M today. It is printed every run because
   of the one place it does nothing -- before ~AD 600 the dataset holds fewer than 500 cities
   at once, so nothing is filtered out and the ancient counts are not comparable to the modern
   ones.

Usage:  python tools/analyze_jumps.py                  # summary + the worst jumps
        python tools/analyze_jumps.py --detail         # + surrounding control points
        python tools/analyze_jumps.py --class other    # only the ones we cannot already explain
        python tools/analyze_jumps.py --csv out.csv    # everything, for a spreadsheet
        python tools/analyze_jumps.py --window 100 --top 200 --min-ratio 2.0
"""
import json, math, sys, io, os, csv, bisect
from collections import defaultdict, Counter

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CITIES = os.path.join(HERE, "data", "cities.json")

# ---- knobs -------------------------------------------------------------------------------
WINDOW    = 50.0     # adjusted years; the span a jump has to happen within to count as fast
MIN_RATIO = 1.5      # smallest change across that window worth listing
TOPN      = 500      # rank a city must reach, in the year of the jump, to be listed
NGRID     = 700      # grid points (uniform in ADJUSTED time) for the rank-threshold curve
FLAT_TOL  = 0.02     # log10 change under this reads as "the source repeated its benchmark"
FLAT_MIN  = 60.0     # adjusted years of flat either side that make a jump a benchmark STEP

# must match build.py / index.html
ADJ_EDGES  = [-3500, 1, 1400, 1900, 2025]
FADE_FLOOR = 1000    # value build.py plants either side of a very long gap


def _adj_frac():
    a = (ADJ_EDGES[1] - ADJ_EDGES[0]) / 2.0
    b = ADJ_EDGES[2] - ADJ_EDGES[1]
    return [0.0, 0.5 * a / (a + b), 0.5, 0.75, 1.0]


ADJ_FRAC  = _adj_frac()
ADJ_SCALE = (ADJ_EDGES[1] - ADJ_EDGES[0]) / ADJ_FRAC[1]


def adj(y):
    for i in range(4):
        if y <= ADJ_EDGES[i + 1] or i == 3:
            f = (y - ADJ_EDGES[i]) / (ADJ_EDGES[i + 1] - ADJ_EDGES[i])
            return (ADJ_FRAC[i] + f * (ADJ_FRAC[i + 1] - ADJ_FRAC[i])) * ADJ_SCALE


def unadj(a):
    s = a / ADJ_SCALE
    for i in range(4):
        if s <= ADJ_FRAC[i + 1] or i == 3:
            f = (s - ADJ_FRAC[i]) / (ADJ_FRAC[i + 1] - ADJ_FRAC[i])
            return ADJ_EDGES[i] + f * (ADJ_EDGES[i + 1] - ADJ_EDGES[i])


def fmt(v):
    if v >= 1e6:
        return f"{v/1e6:.2f}M"
    if v >= 1e3:
        return f"{v/1e3:.0f}k"
    return f"{v:.0f}"


def yr(y):
    y = int(round(y))
    return f"{-y}BC" if y < 0 else str(y)


def era(y):
    if y < 1:
        return "pre-AD1"
    if y < 1400:
        return "AD1-1400"
    if y < 1900:
        return "1400-1900"
    return "1900+"


# ---- one city's series, as a piecewise-linear function of ADJUSTED time -------------------
# log10(pop) is linear in REAL years between control points (the viewer interpolates that way),
# and adjusted time is linear in real years within an era -- so the function is piecewise
# linear in adjusted time once the era edges are inserted as extra breakpoints. That is what
# lets the sliding window below be exact instead of sampled.

def _log_at(ys, ls, y):
    i = bisect.bisect_right(ys, y) - 1
    if i < 0:
        return ls[0]
    if i >= len(ys) - 1:
        return ls[-1]
    y0, y1 = ys[i], ys[i + 1]
    if y1 == y0:
        return ls[i]
    return ls[i] + (ls[i + 1] - ls[i]) * (y - y0) / (y1 - y0)


def breakpoints(p):
    ys = [int(y) for y, _ in p]
    ls = [math.log10(max(1.0, float(v))) for _, v in p]
    extra = [e for e in ADJ_EDGES if ys[0] < e < ys[-1]]
    allys = sorted(set(ys) | set(extra))
    L = [_log_at(ys, ls, y) for y in allys]
    A = [adj(y) for y in allys]
    return allys, A, L


def _interp_a(A, L, t):
    i = bisect.bisect_right(A, t) - 1
    if i < 0:
        return L[0]
    if i >= len(A) - 1:
        return L[-1]
    a0, a1 = A[i], A[i + 1]
    if a1 == a0:
        return L[i]
    return L[i] + (L[i + 1] - L[i]) * (t - a0) / (a1 - a0)


def find_jumps(A, L, window, min_log):
    """Strongest change across any `window`-wide slice of adjusted time.

    f(t) = L(t+w) - L(t) is piecewise linear in t with breaks only where t or t+w hits a
    breakpoint, so its extrema are all anchored at a breakpoint -- checking each breakpoint
    forwards and backwards finds every one exactly, with no sampling grid to tune."""
    raw = []
    for i in range(len(A)):
        for sgn in (1, -1):
            t = A[i] + sgn * window
            if t < A[0] or t > A[-1]:
                continue
            lt = _interp_a(A, L, t)
            if sgn > 0:
                a0, a1, l0, l1 = A[i], t, L[i], lt
            else:
                a0, a1, l0, l1 = t, A[i], lt, L[i]
            d = l1 - l0
            if abs(d) < min_log:
                continue
            raw.append((abs(d), a0, a1, l0, l1, d))
    # many windows describe the same jump; keep the strongest, drop anything overlapping it
    # in the same direction (a rise and a fall that share an instant are two events, not one)
    out = []
    for mag, a0, a1, l0, l1, d in sorted(raw, key=lambda r: -r[0]):
        if any(a0 < b1 and a1 > b0 and (d > 0) == (e > 0) for b0, b1, _, _, e in out):
            continue
        out.append((a0, a1, l0, l1, d))
    return sorted(out, key=lambda r: r[0])


def flat_span(A, L, i, step):
    """Adjusted years of near-constant value running away from breakpoint i -- the signature of
    a carry-forward benchmark being held before/after it changes."""
    j, span = i, 0.0
    while 0 <= j + step < len(A) and abs(L[j + step] - L[j]) < FLAT_TOL:
        span += abs(A[j + step] - A[j])
        j += step
    return span


# ---- rank threshold: what does it take to be TOPN in a given year ------------------------

def rank_curve(C, topn, ngrid):
    import numpy as np
    lo = min(c["p"][0][0] for c in C)
    hi = max(c["p"][-1][0] for c in C)
    gadj = np.linspace(adj(lo), adj(hi), ngrid)
    gy = np.array([unadj(a) for a in gadj])
    M = np.full((len(C), ngrid), -np.inf, dtype=np.float32)
    for k, c in enumerate(C):
        ys = np.array([y for y, _ in c["p"]], dtype=np.float64)
        ls = np.log10(np.maximum(1.0, np.array([v for _, v in c["p"]], dtype=np.float64)))
        row = np.interp(gy, ys, ls)                       # np.interp clamps outside...
        row[(gy < ys[0]) | (gy > ys[-1])] = -np.inf       # ...so blank the years it is not alive
        M[k] = row
    alive = np.isfinite(M).sum(axis=0)
    thr = np.partition(M, -topn, axis=0)[-topn]           # topn-th largest per column
    thr[alive < topn] = -np.inf                           # fewer than topn cities: everyone counts
    return gadj, thr.astype(np.float64), alive


def thr_at(gadj, thr, a):
    i = min(max(bisect.bisect_right(gadj, a) - 1, 0), len(gadj) - 2)
    a0, a1 = gadj[i], gadj[i + 1]
    t0, t1 = thr[i], thr[i + 1]
    if not (math.isfinite(t0) and math.isfinite(t1)):
        return min(t0, t1)
    return t0 + (t1 - t0) * (a - a0) / (a1 - a0)


# ---- classification ----------------------------------------------------------------------

def ramp_spans(P):
    """Real-year intervals covered by a planted fade ramp -- the segments either side of a
    FADE_FLOOR point. A window landing in the MIDDLE of a ramp has no floor point near its own
    ends, so testing for a nearby floor value misses it (Yuzhou's 200->275 descent read as a
    seam step); the ramp has to be tested as an interval."""
    out = []
    for i, (y, v) in enumerate(P):
        if v != FADE_FLOOR:
            continue
        if i:
            out.append((P[i - 1][0], y))
        if i + 1 < len(P):
            out.append((y, P[i + 1][0]))
    return out


def classify(c, y0, y1, i0, i1, A, L, P, ramps):
    """Which of OUR mechanisms, if any, this jump is sitting on. First match wins."""
    tags = []
    if any(y0 < b and y1 > a for a, b in ramps):
        return "fade", tags
    sw = c.get("sw")
    if sw is not None and y0 <= sw < y1:      # strict: a drop that ENDS at the seam is not it
        return "seam", tags
    hx = c.get("hx")
    if hx is not None and y1 >= hx - 1:
        tags.append(f"frozen{int(round(YEAR_NOW - hx))}y")
        return "terminal", tags
    before = flat_span(A, L, i0, -1)
    after = flat_span(A, L, i1, +1)
    if max(before, after) >= FLAT_MIN:
        tags.append(f"flat{int(before)}/{int(after)}")
        return "step", tags
    return "other", tags


YEAR_NOW = 2025

MEANING = {
    "seam":     "populstat hands over to WUP (spec 3.7) -- ours, and what smoothing would hide",
    "fade":     "a planted fade ramp across a very long gap (spec 3.10) -- ours, deliberate",
    "terminal": "the jump lands ON the last real row, which we then freeze to 2025 (spec 3.11)",
    "step":     "a long flat run either side: populstat changed benchmark, not the city",
    "other":    "not explained by anything we do -- real history, or source mixing",
}
ORDER = ["other", "step", "terminal", "seam", "fade"]


def main():
    argv = sys.argv[1:]

    def opt(name, cast, default):
        return cast(argv[argv.index(name) + 1]) if name in argv else default

    window = opt("--window", float, WINDOW)
    topn = opt("--top", int, TOPN)
    min_ratio = opt("--min-ratio", float, MIN_RATIO)
    only = opt("--class", str, None)
    csv_out = opt("--csv", str, None)
    limit = opt("--limit", int, 60)
    detail = "--detail" in argv
    min_log = math.log10(min_ratio)

    C = json.load(open(CITIES, encoding="utf-8"))["cities"]

    print(f"window {window:g} adjusted years = ", end="")
    print(" / ".join(f"{unadj(adj(y) + window) - y:.3g}y at {yr(y)}"
                     for y in (-2000, 700, 1650, 1990)))
    print(f"ranked against the top {topn} of each year; changes of {min_ratio:g}x and up\n")

    gadj, thr, alive = rank_curve(C, topn, NGRID)

    # Print the bar rather than describing it: before ~AD 600 the dataset holds fewer than
    # `topn` cities at once, so the size filter is INERT there and every ancient jump passes.
    # That is worth seeing on every run -- it is the reason the pre-AD1 buckets are not
    # comparable with the modern ones.
    print(f"{'':10}" + "".join(f"{yr(y):>9}" for y in (-1000, 1, 500, 1000, 1400, 1700, 1900, 2025)))
    cells = []
    for y in (-1000, 1, 500, 1000, 1400, 1700, 1900, 2025):
        t = thr_at(gadj, thr, adj(y))
        cells.append(fmt(10 ** t) if math.isfinite(t) else "(all)")
    print(f"{'top-'+str(topn)+':':<10}" + "".join(f"{c:>9}" for c in cells) + "\n")

    rows = []
    for c in C:
        P = c["p"]
        if len(P) < 2:
            continue
        allys, A, L = breakpoints(P)
        ramps = ramp_spans(P)
        for a0, a1, l0, l1, d in find_jumps(A, L, window, min_log):
            # rank at the BIG end, so a collapse is judged by what the city was
            abig, lbig = (a1, l1) if l1 > l0 else (a0, l0)
            if lbig < thr_at(gadj, thr, abig):
                continue
            y0, y1 = unadj(a0), unadj(a1)
            i0 = min(range(len(A)), key=lambda i: abs(A[i] - a0))
            i1 = min(range(len(A)), key=lambda i: abs(A[i] - a1))
            kind, tags = classify(c, y0, y1, i0, i1, A, L, P, ramps)
            # how many source intervals the change is spread over. 1 = a single straight
            # segment, i.e. a corner the eye reads as a discontinuity and the thing rounding
            # would soften; 4+ = a steep but continuously-anchored climb, which is a real
            # curve and should be left alone.
            nseg = 1 + sum(1 for y, _ in P if y0 < y < y1)
            rows.append(dict(name=c["n"], src=c["t"], kind=kind, tags=tags, nseg=nseg,
                             y0=y0, y1=y1, v0=10 ** l0, v1=10 ** l1,
                             ratio=10 ** abs(d), up=d > 0,
                             span_real=y1 - y0, era=era(y1),
                             peak=max(v for _, v in P), city=c))

    rows.sort(key=lambda r: -r["ratio"])
    print(f"{len(rows):,} jumps in {len({r['name'] for r in rows}):,} cities\n")

    # ---- what kind are they? ----
    by_kind = Counter(r["kind"] for r in rows)
    print(f"{'kind':<10}{'count':>7}   {'share':>6}   what it is")
    for k in ORDER:
        n = by_kind.get(k, 0)
        print(f"  {k:<8}{n:>7}   {n/max(1,len(rows))*100:>5.1f}%   {MEANING[k]}")

    # ---- when? ----
    print(f"\n{'era':<12}{'jumps':>7}  " + "".join(f"{k:>9}" for k in ORDER))
    for e in ("pre-AD1", "AD1-1400", "1400-1900", "1900+"):
        sub = [r for r in rows if r["era"] == e]
        cc = Counter(r["kind"] for r in sub)
        print(f"  {e:<10}{len(sub):>7}  " + "".join(f"{cc[k]:>9}" for k in ORDER))

    # ---- how big? ----
    print(f"\nsize of the change ({window:g} adjusted years):")
    for lo, hi in [(1.5, 2), (2, 3), (3, 5), (5, 10), (10, 1e9)]:
        n = sum(1 for r in rows if lo <= r["ratio"] < hi)
        lab = f"{lo:g}-{hi:g}x" if hi < 1e9 else f"{lo:g}x+"
        print(f"  {lab:<9}{n:>5}  {'#' * round(50 * n / max(1, len(rows)))}")
    ups = sum(1 for r in rows if r["up"])
    print(f"\n  {ups} up, {len(rows)-ups} down")

    # ---- the seam on its own terms -------------------------------------------------------
    # The list above only sees seams big enough AND in big enough cities to clear the filters.
    # For deciding whether to smooth the handover, what matters is the step at EVERY seam, so
    # this measures all of them: the last historical value against the first WUP one.
    seams = []
    for c in C:
        sw = c.get("sw")
        if sw is None:
            continue
        P = c["p"]
        i = max((k for k, (y, _) in enumerate(P) if y <= sw), default=None)
        if i is None or i + 1 >= len(P):
            continue
        v0, v1 = P[i][1], P[i + 1][1]
        if v0 <= 0 or v1 <= 0:
            continue
        seams.append((max(v0, v1) / min(v0, v1), v1 / v0, sw, max(v0, v1), c))
    seams.sort(key=lambda s: -s[0])
    big = [s for s in seams if s[3] >= 10 ** thr_at(gadj, thr, adj(s[2]))]
    print(f"\n{'='*94}\nTHE WUP SEAM, all {len(seams):,} grafted cities "
          f"({len(big):,} of them top-{topn} at the seam)\n{'='*94}")
    print(f"{'step':<12}{'all':>8}{'share':>8}   {'top-'+str(topn):>8}{'share':>8}")
    for lo, hi, lab in [(1.0, 1.1, "<1.1x"), (1.1, 1.25, "1.1-1.25x"), (1.25, 1.5, "1.25-1.5x"),
                        (1.5, 2.0, "1.5-2x"), (2.0, 4.0, "2-4x"), (4.0, 1e9, "4x+")]:
        n = sum(1 for s in seams if lo <= s[0] < hi)
        m = sum(1 for s in big if lo <= s[0] < hi)
        print(f"  {lab:<10}{n:>8}{n/max(1,len(seams))*100:>7.1f}%   {m:>8}{m/max(1,len(big))*100:>7.1f}%")
    dn = sum(1 for s in big if s[1] < 1)
    print(f"\n  of the top-{topn} seams: {len(big)-dn} step UP, {dn} step DOWN")
    print(f"  seam years: " + ", ".join(f"{y}({n})" for y, n in
                                        Counter(int(s[2]) for s in big).most_common(8)))

    # ---- the list ----
    shown = [r for r in rows if only is None or r["kind"] == only]
    head = f"WORST {min(limit, len(shown))}" + (f" OF KIND '{only}'" if only else "")
    print(f"\n{'='*94}\n{head}, by size of change\n{'='*94}")
    print(f"{'city':<22}{'years':<16}{'change':<22}{'x':>7}{'seg':>5}  kind")
    for r in shown[:limit]:
        span = f"{yr(r['y0'])}-{yr(r['y1'])}"
        arrow = "->"
        chg = f"{fmt(r['v0'])} {arrow} {fmt(r['v1'])}"
        tag = r["kind"] + (" " + ",".join(r["tags"]) if r["tags"] else "")
        print(f"{r['name'][:21]:<22}{span:<16}{chg:<22}{r['ratio']:>6.1f}{r['nseg']:>5}  {tag}")
        if detail:
            P = r["city"]["p"]
            lo = max(0, bisect.bisect_left([y for y, _ in P], r["y0"]) - 2)
            hi = min(len(P), bisect.bisect_right([y for y, _ in P], r["y1"]) + 2)
            print("      " + "  ".join(f"{yr(y)}:{fmt(v)}" for y, v in P[lo:hi]))

    # ---- cities with the most jumps: the ones worth fixing at the entry level ----
    per_city = defaultdict(list)
    for r in rows:
        if r["kind"] in ("other", "step"):
            per_city[r["name"]].append(r)
    print(f"\n{'='*94}\nCITIES WITH THE MOST UNEXPLAINED JUMPS (kind other/step)\n{'='*94}")
    for name, rs in sorted(per_city.items(), key=lambda kv: (-len(kv[1]), -max(r["ratio"] for r in kv[1])))[:20]:
        yrs = ", ".join(f"{yr(r['y1'])}({r['ratio']:.1f}x{'^' if r['up'] else 'v'})"
                        for r in sorted(rs, key=lambda r: r["y1"])[:8])
        print(f"  {name[:20]:<21}{len(rs):>3}   {yrs}")

    # ---- shared benchmark blocks -----------------------------------------------------------
    # Tikal, Caracol, Tiahuanaco and Tula all open 200BC:800 / AD0:43,100 / AD100:100,000 --
    # the same three numbers to the digit, for four sites two continents' worth of distance
    # apart. That is one curve copied across several entries, not four independent estimates,
    # and because every copy contributes the same jump it inflates the ancient buckets above.
    # Whole-series equality misses these: the entries diverge after the shared opening, so the
    # unit to look for is a RUN of identical points, not a whole city.
    # Ranked by SIZE, not by how many cities share the block: Buringh's medieval European
    # towns legitimately share values by the hundred (his figures are modelled and land on the
    # same few round numbers), and those blocks are both harmless and unrankable. A shared
    # block at 40,000+ is a different animal -- nobody independently estimates four cities at
    # 43,100.
    RUN, BLOCK_MIN = 3, 25000
    blocks = defaultdict(set)
    for c in C:
        P = c["p"]
        for i in range(len(P) - RUN + 1):
            if max(v for _, v in P[i:i + RUN]) >= BLOCK_MIN:
                blocks[tuple((y, v) for y, v in P[i:i + RUN])].add(c["n"])
    shared = sorted((k for k, v in blocks.items() if len(v) >= 3),
                    key=lambda k: -max(v for _, v in k))
    if shared:
        print(f"\n{'='*94}\nSHARED BENCHMARK BLOCKS ({len(shared)} runs of {RUN} identical points"
              f" >={fmt(BLOCK_MIN)} in 3+ cities)\n{'='*94}")
        for k in shared[:15]:
            pts = " ".join(f"{yr(y)}:{fmt(v)}" for y, v in k)
            print(f"  {len(blocks[k])}x {pts:<38} " + ", ".join(sorted(blocks[k]))[:40])

    if csv_out:
        path = csv_out if os.path.isabs(csv_out) else os.path.join(HERE, csv_out)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["city", "source", "kind", "tags", "y0", "y1", "span_real", "segments",
                        "v0", "v1", "ratio", "direction", "era", "city_peak", "lat", "lon"])
            for r in rows:
                w.writerow([r["name"], r["src"], r["kind"], ";".join(r["tags"]),
                            int(round(r["y0"])), int(round(r["y1"])), round(r["span_real"], 1),
                            r["nseg"], int(r["v0"]), int(r["v1"]), round(r["ratio"], 3),
                            "up" if r["up"] else "down", r["era"], int(r["peak"]),
                            r["city"]["la"], r["city"]["lo"]])
        print(f"\nwrote {len(rows):,} rows to {path}")


if __name__ == "__main__":
    main()
