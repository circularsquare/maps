"""analyze_osc.py -- characterise check F (definition oscillation) before fixing it.

validate.py tells you 88 entries oscillate and that they hold 9% of top-12 slots. Neither
number tells you what SHAPE the defect has, and the shape decides the repair:

  isolated spike/dip   one row in a different unit, neighbours agree  -> delete the row
  alternating run      the source interleaves two units for decades   -> pick a unit
  step                 a permanent definition change, no return       -> not check F's business
  wide gap             the "flip" straddles centuries                 -> probably real history

It also recomputes rank impact HONESTLY. validate.py flags a whole entry and then credits
that flag in every sampled year, so Chengdu's 1936 flip is charged against its year -50
ranking. Here a defect only counts in years its own flips actually span.

Usage:  python tools/analyze_osc.py            # summary + histograms
        python tools/analyze_osc.py --detail   # per-city anchors for the worst offenders
        python tools/analyze_osc.py --raw      # read the source series, not the built one
"""
import json, math, sys, io, os
from collections import defaultdict

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CITIES = os.path.join(HERE, "data", "cities.json")

# same thresholds as validate.py check F, so the population of cases matches
OSC_RATIO, OSC_WINDOW, OSC_MIN_VAL = 1.35, 40, 300000
TOPN = 12

# classification knobs (what this script is for: seeing whether these are the right ones)
AGREE = 1.5      # v0 and v2 within this factor = "the series came back to where it was"
TIGHT = 40       # total span at or under this = no real city could do it
RUN_GAP = 60     # flips this close together are one alternating run, not separate events


def fmt(v):
    return f"{v/1e6:.1f}M" if v >= 1e6 else f"{v/1e3:.0f}k"


def pop_at(c, y):
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


def find_flips(p):
    """Every POINT that is out of line with both of its neighbours.

    Point-centric, not triple-centric. A single bad row trips two consecutive A-B-C
    triples (Xuanwei 70k -> 1.2M -> 78k -> 128k flags both (85,90,91) and (90,91,95)),
    which made one deletable row look like an alternating run. What matters for the
    repair is which POINT is wrong, so that is what this finds."""
    out = []
    for i in range(1, len(p) - 1):
        (y0, v0), (y1, v1), (y2, v2) = p[i - 1], p[i], p[i + 1]
        if y2 - y0 > OSC_WINDOW or min(v0, v1, v2) <= 0:
            continue
        if max(v0, v1, v2) < OSC_MIN_VAL:
            continue
        rp, rn = v1 / v0, v1 / v2
        if (rp >= OSC_RATIO and rn >= OSC_RATIO) or (rp <= 1 / OSC_RATIO and rn <= 1 / OSC_RATIO):
            out.append(dict(i=i, y0=y0, v0=v0, y1=y1, v1=v1, y2=y2, v2=v2,
                            span=y2 - y0, kind="spike" if rp > 1 else "dip",
                            amp=min(max(rp, 1 / rp), max(rn, 1 / rn)),   # the WEAKER leg
                            agree=max(v0, v2) / min(v0, v2)))
    return out


def classify(flips):
    """Group nearby out-of-line points into events; >1 point in an event = alternating."""
    events = []
    cur = []
    for f in sorted(flips, key=lambda f: f["y1"]):
        if cur and f["y1"] - cur[-1]["y1"] > RUN_GAP:
            events.append(cur)
            cur = []
        cur.append(f)
    if cur:
        events.append(cur)
    out = []
    for ev in events:
        span = (ev[0]["y0"], ev[-1]["y2"])
        if len(ev) > 1:
            kind = "alternating"
        elif ev[0]["agree"] > AGREE:
            kind = "step"          # the series did NOT come back: a trend, not a unit switch
        elif ev[0]["span"] <= TIGHT:
            kind = "isolated"
        else:
            kind = "wide"
        out.append((kind, span, ev))
    return out


def main():
    C = json.load(open(CITIES, encoding="utf-8"))["cities"]
    detail = "--detail" in sys.argv

    per_city = {}
    for c in C:
        f = find_flips(c["p"])
        if f:
            per_city[c["n"]] = (c, f, classify(f))
    print(f"{len(per_city)} entries with at least one flip "
          f"({sum(len(v[1]) for v in per_city.values())} flips total)\n")

    # ---- how are the events distributed? ----
    kinds = defaultdict(list)
    for name, (c, flips, events) in per_city.items():
        for kind, span, ev in events:
            kinds[kind].append((name, span, ev))
    print(f"{'event type':<14}{'count':>7}   what it means")
    meaning = {"isolated": "one row in another unit; neighbours agree -> deletable",
               "alternating": "source interleaves two units -> pick one",
               "step": "level changed and stayed -> not an oscillation",
               "wide": "came back, but over a long span -> could be real"}
    for k in ("isolated", "alternating", "step", "wide"):
        print(f"  {k:<12}{len(kinds[k]):>7}   {meaning[k]}")

    # ---- the axis she asked about: how long are these spans? ----
    print(f"\nspan of the flip (y2-y0), all {sum(len(v[1]) for v in per_city.values())} flips:")
    buckets = [(0, 5), (5, 10), (10, 20), (20, 30), (30, 40)]
    allf = [f for _, (c, fl, _) in per_city.items() for f in fl]
    for lo, hi in buckets:
        n = sum(1 for f in allf if lo < f["span"] <= hi or (lo == 0 and f["span"] == 0))
        bar = "#" * round(40 * n / max(1, len(allf)))
        print(f"  {lo:>3}-{hi:<3}y  {n:>4}  {bar}")

    print(f"\nneighbour agreement v0 vs v2 (1.0 = returned exactly):")
    for lo, hi in [(1.0, 1.1), (1.1, 1.25), (1.25, 1.5), (1.5, 2.0), (2.0, 99)]:
        n = sum(1 for f in allf if lo <= f["agree"] < hi)
        bar = "#" * round(40 * n / max(1, len(allf)))
        print(f"  {lo:>4.2f}-{hi:<5.2f} {n:>4}  {bar}")

    # ---- honest rank impact: only charge a defect in the years it spans ----
    years = list(range(-1000, 1901, 50)) + [1950, 2000, 2025]
    tot = naive = honest = 0
    naive_hits, honest_hits = defaultdict(list), defaultdict(list)
    flagged = {n for n in per_city}
    spans = {n: [s for _, s, _ in ev] for n, (c, f, ev) in per_city.items()}
    for y in years:
        rows = sorted(((pop_at(c, y), c) for c in C), key=lambda t: -t[0])[:TOPN]
        rows = [(v, c) for v, c in rows if v >= 5000]
        tot += len(rows)
        for v, c in rows:
            if c["n"] not in flagged:
                continue
            naive += 1
            naive_hits[c["n"]].append(y)
            # does any of this city's flips actually cover the year being ranked?
            if any(a - 25 <= y <= b + 25 for a, b in spans[c["n"]]):
                honest += 1
                honest_hits[c["n"]].append(y)
    print(f"\nrank impact, top-{TOPN} over {len(years)} sampled years:")
    print(f"  validate.py's way (entry flagged in every year):  {naive}/{tot} ({naive/tot*100:.0f}%)")
    print(f"  only in years the flips span:                     {honest}/{tot} ({honest/tot*100:.0f}%)")
    print("\n  cities whose flags are charged in years far from any flip:")
    for n, ys in sorted(naive_hits.items(), key=lambda kv: -len(kv[1]))[:8]:
        h = len(honest_hits.get(n, []))
        sp = ", ".join(f"{a}..{b}" for a, b in spans[n][:3])
        print(f"    {n:<14} {len(ys):>2} yrs charged, {h:>2} deserved   flips at {sp}")

    if detail:
        print(f"\n{'='*78}\nPER-CITY ANCHORS (worst by size)\n{'='*78}")
        rank = sorted(per_city.items(), key=lambda kv: -max(f["v1"] for f in kv[1][1]))
        for name, (c, flips, events) in rank[:25]:
            print(f"\n{name}  ({len(flips)} flip(s))")
            for kind, (a, b), ev in events:
                print(f"   [{kind}] {a}..{b}")
                lo = max(0, ev[0]["i"] - 1)
                hi = min(len(c["p"]), ev[-1]["i"] + 4)
                seg = "  ".join(f"{y}:{fmt(v)}" for y, v in c["p"][lo:hi])
                print(f"      {seg}")


if __name__ == "__main__":
    main()
