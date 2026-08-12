"""validate.py -- data-quality report for the cityhistory pipeline.

Run after build.py. Every check below exists because it caught a real, visible bug;
each one prints its worst offenders ranked by how much they distort the map. None of
this mutates data -- it tells you what to add to build.py's DROP_KEYS / RENAME /
COORD_FIX / CLIP_BEFORE tables, or which knob is mistuned.

  A  carry-forward   a pre-modern estimate repeated verbatim until the first census,
                     leaving dead cities frozen at their peak (Vijayanagara 480k to 1890)
  B  whipsaw         order-of-magnitude swings inside a few years -- one entry holding
                     several different geographic units (Wetter (Ruhr): town, Landkreis
                     and the whole Ruhrgebiet in one series)
  C  graft collapse  modern tail far smaller than the historical peak: the WUP centre
                     match went wrong (Guangzhou landing on Huadu)
  D  coord stacking  many distinct cities sharing one coordinate = geocoder fallback to
                     a country centroid (Tashkent in the Kyzylkum desert)
  E  terminal break  series ends with a sudden collapse -- usually a source unit change
                     (Arhus's last point is "219", i.e. thousands)

Usage:  python validate.py            # report on data/cities.json
        python validate.py --raw      # also check the raw source before cleaning
"""
import json, math, sys, io
from collections import Counter, defaultdict

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

CITIES = "data/cities.json"
RAW    = "data/stadester/stadester_cities.json"
WUP    = "data/stadester/wup2025.json"

# --- thresholds. Tuned so each check's output is short enough to read in full. ---
CF_MIN_SPAN   = 150     # A: flat run this long is not a coincidence
CF_MIN_VAL    = 20000   # A: ignore villages
CF_MIN_CLIFF  = 3       # A: drop after the run, in multiples
WHIP_RATIO    = 8       # B: swing size
WHIP_WINDOW   = 30      # B: within this many years
WHIP_MIN_VAL  = 200000  # B: only care when it distorts the top ranks
GRAFT_MIN_PK  = 200000  # C: only check cities that were once this big
GRAFT_RATIO   = 5       # C: tail smaller than peak/this = suspect
STACK_MIN     = 3       # D: distinct names sharing a coordinate
END_RATIO     = 20      # E: final-point collapse
END_MIN_VAL   = 50000   # E: only care when it was a real city
TOPN          = 12      # rank-impact sampling
SHOW          = 15      # rows printed per check


def fmt(v):
    return f"{v/1e6:.1f}M" if v >= 1e6 else f"{v/1e3:.0f}k"


def km(alat, alon, blat, blon):
    return math.hypot((blon - alon) * math.cos(math.radians(alat)) * 111.32,
                      (blat - alat) * 110.57)


def pop_at(c, y):
    """log-linear interpolation -- must match popAt() in index.html."""
    p = c["p"]
    if y < p[0][0] or y > p[-1][0]:
        return 0.0
    lo, hi = 0, len(p) - 1
    while hi - lo > 1:
        m = (lo + hi) // 2
        if p[m][0] <= y:
            lo = m
        else:
            hi = m
    (y0, v0), (y1, v1) = p[lo], p[hi]
    if y1 == y0:
        return v0
    f = (y - y0) / (y1 - y0)
    return 10 ** (math.log10(v0) + (math.log10(v1) - math.log10(v0)) * f)


def head(letter, title, n, note=""):
    print(f"\n{'='*78}\n{letter}. {title}  --  {n} found{('   ' + note) if note else ''}\n{'='*78}")


# ---------------------------------------------------------------- A
def check_carry_forward(C):
    """A flat run of *identical* values, then a cliff. Genuine data essentially never
    repeats a value to the byte across centuries -- interpolation and carry-forward do."""
    out = []
    for c in C:
        p = c["p"]
        for i in range(len(p) - 2):
            (y0, v0), (y1, v1), (y2, v2) = p[i], p[i + 1], p[i + 2]
            if v0 == v1 and y1 - y0 >= CF_MIN_SPAN and v0 >= CF_MIN_VAL \
                    and v2 > 0 and v0 / v2 >= CF_MIN_CLIFF:
                out.append((v0, y0, y1, y2, v0 / v2, c))
    out.sort(key=lambda t: -t[0])
    head("A", "CARRY-FORWARD (dead city frozen at its peak)", len(out),
         f"span>={CF_MIN_SPAN}y, cliff>={CF_MIN_CLIFF}x")
    for v, y0, y1, y2, r, c in out[:SHOW]:
        print(f"  {fmt(v):>7}  frozen {y0:>5} -> {y1:<5} ({y1-y0:>4}y)  then {y2}: /{r:.0f}x   {c['n']}")
    return {id(c) for *_, c in out}, out


# ---------------------------------------------------------------- B
def check_whipsaw(C):
    """Impossible short-interval swings: one entry carrying several geographic units."""
    out = defaultdict(list)
    for c in C:
        p = c["p"]
        for i in range(len(p) - 1):
            (y0, v0), (y1, v1) = p[i], p[i + 1]
            if y1 - y0 > WHIP_WINDOW or min(v0, v1) <= 0:
                continue
            r = max(v0 / v1, v1 / v0)
            if r >= WHIP_RATIO and max(v0, v1) >= WHIP_MIN_VAL:
                out[c["n"]].append((r, y0, v0, y1, v1))
    ranked = sorted(out.items(), key=lambda kv: (-len(kv[1]), -max(x[0] for x in kv[1])))
    head("B", "WHIPSAW (one entry holding several different units)", len(out),
         f">={WHIP_RATIO}x within {WHIP_WINDOW}y at >={fmt(WHIP_MIN_VAL)}")
    for name, sw in ranked[:SHOW]:
        r, y0, v0, y1, v1 = max(sw, key=lambda x: x[0])
        print(f"  {len(sw):>2} swing(s)  worst {y0}:{fmt(v0):>7} -> {y1}:{fmt(v1):<7} ({r:>4.0f}x)   {name}")
    return set(out)


# ---------------------------------------------------------------- C
def check_graft(C):
    """Modern tail far below the historical peak = the city grafted onto the wrong
    (usually much smaller) WUP urban centre, or missed its own centre entirely."""
    out = []
    for c in C:
        pre = [v for y, v in c["p"] if y <= 2000]
        post = [v for y, v in c["p"] if y > 2000]
        if not pre or not post:
            continue
        if max(pre) >= GRAFT_MIN_PK and max(post) < max(pre) / GRAFT_RATIO:
            out.append((max(pre), max(post), c))
    out.sort(key=lambda t: -t[0])
    head("C", "GRAFT COLLAPSE (modern tail lost its agglomeration)", len(out),
         f"peak>={fmt(GRAFT_MIN_PK)}, tail<peak/{GRAFT_RATIO}")
    for a, b, c in out[:SHOW]:
        print(f"  peak {fmt(a):>7} -> tail {fmt(b):<7} ({a/b:>4.0f}x)   {c['n']}  @{c['la']:.2f},{c['lo']:.2f}")
    return out


# ---------------------------------------------------------------- D
def check_stacking(C):
    """A coordinate shared by several DIFFERENT city names is a geocoder fallback --
    the country (or province) centroid. Same name repeated is just a metro variant."""
    names_at = defaultdict(set)
    at = defaultdict(list)
    for c in C:
        pt = (round(c["la"], 2), round(c["lo"], 2))
        names_at[pt].add(c["n"].split(" (")[0].lower())
        at[pt].append(c)
    pts = [(len(v), p) for p, v in names_at.items() if len(v) >= STACK_MIN]
    pts.sort(reverse=True)
    tot = sum(len(at[p]) for _, p in pts)
    big = [c for _, p in pts for c in at[p] if max(v for _, v in c["p"]) >= 100000]
    head("D", "COORD STACKING (geocoder fell back to a centroid)", len(pts),
         f"points, holding {tot} entries ({len(big)} of them >=100k)")
    for n, p in pts[:SHOW]:
        ex = sorted(at[p], key=lambda c: -max(v for _, v in c["p"]))[:4]
        exs = ", ".join(f"{c['n']} {fmt(max(v for _, v in c['p']))}" for c in ex)
        print(f"  {len(at[p]):>3} entries at {p[0]:>8},{p[1]:<9}  {exs}")
    return {id(c) for _, p in pts for c in at[p]}


# ---------------------------------------------------------------- E
def check_terminal(C):
    """Series ends by falling off a cliff -- almost always the source switching units
    (population in thousands) for the final year."""
    out = []
    for c in C:
        p = c["p"]
        if len(p) < 3:
            continue
        (y0, v0), (y1, v1) = p[-2], p[-1]
        if v1 > 0 and v0 / v1 >= END_RATIO and v0 >= END_MIN_VAL:
            out.append((v0 / v1, y0, v0, y1, v1, c))
    out.sort(key=lambda t: -t[2])
    head("E", "TERMINAL BREAK (last point collapses; unit change?)", len(out),
         f">={END_RATIO}x at >={fmt(END_MIN_VAL)}")
    for r, y0, v0, y1, v1, c in out[:SHOW]:
        print(f"  {y0}:{v0:>10,.0f} -> {y1}:{v1:>9,.0f}  ({r:>5.0f}x)   {c['n']}")
    return out


# ---------------------------------------------------------------- impact
def rank_impact(C, flagged_ids, label):
    """How often does a flagged entry occupy a top-N slot? This is what decides whether
    a class of error is cosmetic or is actually rewriting the headline ranking."""
    years = list(range(-1000, 1901, 50)) + [1950, 2000, 2025]
    tot = bad = 0
    hits = defaultdict(list)
    for y in years:
        rows = sorted(((pop_at(c, y), c) for c in C), key=lambda t: -t[0])[:TOPN]
        rows = [(v, c) for v, c in rows if v >= 5000]
        tot += len(rows)
        for v, c in rows:
            if id(c) in flagged_ids:
                bad += 1
                hits[c["n"]].append(y)
    pct = bad / tot * 100 if tot else 0
    print(f"\n  top-{TOPN} slots held by [{label}]: {bad}/{tot} ({pct:.0f}%) over {len(years)} sampled years")
    for n, ys in sorted(hits.items(), key=lambda kv: -len(kv[1]))[:8]:
        print(f"      {n:<28} {len(ys):>2} yrs  ({min(ys)}..{max(ys)})")
    return pct


# ---------------------------------------------------------------- F
OSC_RATIO  = 1.35    # up-then-down (or down-then-up) by at least this much...
OSC_WINDOW = 40      # ...and back again inside this many years
OSC_MIN_VAL = 300000


def check_oscillation(C):
    """A city that jumps up and straight back down (or vice versa) is switching between two
    DEFINITIONS, not growing and shrinking. Real cities do collapse fast -- war, plague,
    a court leaving -- but they do not recover to the old level within a couple of decades
    and then fall again. This is the small-ratio sibling of check B: London alternated
    between the County (4.5M) and Greater London (7.4M) at only 1.6x, far under B's 8x, so
    B never saw it while the graph sawtoothed for fifty years."""
    out = {}
    for c in C:
        p = c["p"]
        hits = []
        for i in range(len(p) - 2):
            (y0, v0), (y1, v1), (y2, v2) = p[i], p[i + 1], p[i + 2]
            if y2 - y0 > OSC_WINDOW or min(v0, v1, v2) <= 0:
                continue
            if max(v0, v1, v2) < OSC_MIN_VAL:
                continue
            up, dn = v1 / v0, v2 / v1
            if (up >= OSC_RATIO and dn <= 1 / OSC_RATIO) or (up <= 1 / OSC_RATIO and dn >= OSC_RATIO):
                hits.append((y0, v0, y1, v1, y2, v2))
        if hits:
            out[c["n"]] = hits
    ranked = sorted(out.items(), key=lambda kv: (-len(kv[1]),
                                                 -max(x[3] for x in kv[1])))
    head("F", "DEFINITION OSCILLATION (series flips between two units)", len(out),
         f">={OSC_RATIO}x and back inside {OSC_WINDOW}y, at >={fmt(OSC_MIN_VAL)}")
    for name, hits in ranked[:SHOW]:
        y0, v0, y1, v1, y2, v2 = hits[0]
        print(f"  {len(hits):>2} flip(s)  {y0}:{fmt(v0):>6} -> {y1}:{fmt(v1):>6} -> {y2}:{fmt(v2):<6}   {name}")
    return set(out)


def main():
    C = json.load(open(CITIES, encoding="utf-8"))["cities"]
    print(f"validating {CITIES}: {len(C):,} cities")

    cf_ids, _ = check_carry_forward(C)
    whip_names = check_whipsaw(C)
    check_graft(C)
    stack_ids = check_stacking(C)
    check_terminal(C)
    osc_names = check_oscillation(C)

    print(f"\n{'='*78}\nRANK IMPACT\n{'='*78}")
    rank_impact(C, cf_ids, "carry-forward")
    rank_impact(C, {id(c) for c in C if c["n"] in whip_names}, "whipsaw")
    rank_impact(C, stack_ids, "bad coords")
    rank_impact(C, {id(c) for c in C if c["n"] in osc_names}, "definition oscillation")

    if "--raw" in sys.argv:
        raw = json.load(open(RAW, encoding="utf-8"))
        n = 0
        for c in raw.values():
            pts = []
            for y, v in (c.get("population") or {}).items():
                try:
                    y, v = int(y), float(v)
                except (ValueError, TypeError):
                    continue
                if v > 0:
                    pts.append((y, v))
            pts.sort()
            i = 0
            while i < len(pts):
                j = i
                while j + 1 < len(pts) and abs(pts[j + 1][1] - pts[i][1]) < 1e-6:
                    j += 1
                if pts[j][0] - pts[i][0] >= CF_MIN_SPAN and pts[i][1] >= CF_MIN_VAL:
                    n += 1
                i = j + 1
        print(f"\n  raw source still contains {n} flat runs >={CF_MIN_SPAN}y at >={fmt(CF_MIN_VAL)}"
              f"  (build.py should be collapsing these)")

    print("\ndone. Fix by adding to DROP_KEYS / RENAME / COORD_FIX / CLIP_BEFORE in build.py.")


if __name__ == "__main__":
    main()
