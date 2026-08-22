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
  F  definition osc  one entry interleaving two geographic units year by year
  G  strip ate a     a carry-forward run that swallowed real Chandler benchmarks, because
     benchmark       the compiler asserted the same value at several of them (Gao 75,000 at
                     1550/1575/1585/1591) and the fill that follows echoes it to within
                     CF_EPS -- so the strip sees one run and keeps only its first point

Usage:  python validate.py            # report on data/cities.json
        python validate.py --raw      # also check the raw source before cleaning
"""
import csv, json, math, os, sys, io
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
GONE_RATIO    = 2.5     # G: fall this steep...
GONE_WINDOW   = 15      # G: ...within this many years...
GONE_MIN_VAL  = 50000   # G: ...from a level worth seeing...
GONE_BACK     = 0.9     # G: ...and never returns to this fraction of it

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


FADE_FLOOR = 1000       # must match build.py; `s` marks these 'f' so the match is belt-and-braces


def real_points(c):
    """`c["p"]` with the planted floor points removed.

    A fade shoulder is OURS, not a measurement -- build.py says so explicitly by marking it `f`
    in the per-point source string. The step checks (B whipsaw, E terminal break, F oscillation)
    all measure ratios between consecutive points, so a floor next to a real anchor reads as a
    50x collapse that nobody introduced: bracketing 29 lone anchors took E from 3 to 11 and B
    from 74 to 85 without a single series changing shape anywhere a viewer could see.

    Falls back to the value test if `s` is absent or out of step with `p`, so this still does the
    right thing on an older cities.json."""
    p = c["p"]
    s = c.get("s")
    if s and len(s) == len(p):
        return [pt for pt, code in zip(p, s) if code != "f"]
    return [pt for pt in p if pt[1] != FADE_FLOOR]


def head(letter, title, n, note=""):
    print(f"\n{'='*78}\n{letter}. {title}  --  {n} found{('   ' + note) if note else ''}\n{'='*78}")


# ---------------------------------------------------------------- A
def deliberate_plateaus(raw):
    """Display names of every entry whose flat run build.py KEEPS on purpose.

    Check A hunts "flat run then cliff", and that is also the exact shape of a CF_KEEP plateau
    (Constantinople holding 330k across the Macedonian revival) and of a forward CF_END (Sparta
    holding 30k across Chandler's AD 0/100/200). Both are decisions, so without this the check's
    top rows are its own successes and the real finds are buried under them. Marked, not
    dropped -- a kept plateau that later turns out to be wrong should still be visible here."""
    try:
        import build
    except Exception:
        return set()
    keys = {k for k, _ in build.CF_KEEP}
    for k in build.CF_END:
        keys.add(k[0] if isinstance(k, tuple) else k)
    out = set()
    for k in keys:
        # A MERGE_INTO donor is never drawn under its own name, so the row to mark carries the
        # TARGET's label -- Sparta's CF_END is keyed on "Sparti (agglomeration)-Greece" but the
        # plateau appears on "Sparti". Add both and let the name match sort it out.
        for kk in (k, build.MERGE_INTO.get(k)):
            e = raw.get(kk) if kk else None
            if e:
                out.add(build.RENAME.get(kk, e.get("name", "")))
    return out


def check_carry_forward(C, deliberate=frozenset()):
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
    n_kept = sum(1 for t in out if t[5]["n"] in deliberate)
    head("A", "CARRY-FORWARD (dead city frozen at its peak)", len(out),
         f"span>={CF_MIN_SPAN}y, cliff>={CF_MIN_CLIFF}x"
         + (f"  [{n_kept} marked * are deliberate CF_KEEP / CF_END]" if n_kept else ""))
    for v, y0, y1, y2, r, c in out[:SHOW]:
        mark = " *" if c["n"] in deliberate else ""
        print(f"  {fmt(v):>7}  frozen {y0:>5} -> {y1:<5} ({y1-y0:>4}y)  then {y2}: /{r:.0f}x   {c['n']}{mark}")
    spans = defaultdict(list)
    for v, y0, y1, y2, r, c in out:
        spans[id(c)].append((y0, y2))       # the frozen run plus the cliff that ends it
    return spans, out


# ---------------------------------------------------------------- B
def check_whipsaw(C):
    """Impossible short-interval swings: one entry carrying several geographic units."""
    out = defaultdict(list)
    spans = defaultdict(list)
    n_seam = 0
    for c in C:
        p = real_points(c)
        for i in range(len(p) - 1):
            (y0, v0), (y1, v1) = p[i], p[i + 1]
            if y1 - y0 > WHIP_WINDOW or min(v0, v1) <= 0:
                continue
            r = max(v0 / v1, v1 / v0)
            if r >= WHIP_RATIO and max(v0, v1) >= WHIP_MIN_VAL:
                if straddles_seam(c, y0, y1):    # the graft step, not a source unit change
                    n_seam += 1
                    continue
                out[c["n"]].append((r, y0, v0, y1, v1))
                spans[id(c)].append((y0, y1))
    ranked = sorted(out.items(), key=lambda kv: (-len(kv[1]), -max(x[0] for x in kv[1])))
    head("B", "WHIPSAW (one entry holding several different units)", len(out),
         f">={WHIP_RATIO}x within {WHIP_WINDOW}y at >={fmt(WHIP_MIN_VAL)}"
         + (f"  [+{n_seam} skipped at the graft seam]" if n_seam else ""))
    for name, sw in ranked[:SHOW]:
        r, y0, v0, y1, v1 = max(sw, key=lambda x: x[0])
        print(f"  {len(sw):>2} swing(s)  worst {y0}:{fmt(v0):>7} -> {y1}:{fmt(v1):<7} ({r:>4.0f}x)   {name}")
    return spans


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
    # a wrong coordinate is wrong in every year the city is on the map, so the span is
    # the whole series -- unlike the other checks, this one really is entry-wide.
    return {id(c): [(c["p"][0][0], c["p"][-1][0])] for _, p in pts for c in at[p]}


# ---------------------------------------------------------------- E
def check_terminal(C):
    """Series ends by falling off a cliff -- almost always the source switching units
    (population in thousands) for the final year."""
    out = []
    for c in C:
        p = real_points(c)
        if len(p) < 3:
            continue
        # ...and then past the HOLD, or this check cannot see most of the map. build.py §3.11
        # holds a city's last value flat to YEAR_NOW and marks the join `hx`, so for a held city
        # p[-2] and p[-1] are equal BY CONSTRUCTION and the ratio is 1.00 whatever the record
        # did. 13,809 of 22,144 cities end on a flat segment, i.e. this was structurally blind
        # to 62% of the map -- it could only ever see the grafted minority. Walking back to `hx`
        # takes it from 3 hits to 6 and surfaces Codru (653,000 -> 11,500) and Harburg
        # (113,000 -> 5,000, whose series is Harburg near Hamburg while its coordinate is
        # Harburg in Bavaria, 600km away).
        hx = c.get("hx")
        if hx is not None:
            p = [q for q in p if q[0] <= hx]
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


# ---------------------------------------------------------------- H
def check_never_regains(C):
    """A big fall the city never recovers from -- a unit change, not a population change.

    The other step checks all ask "how big?" and that is what lets this family through. B needs
    8x and these are 2.5-6x; F needs the series to come BACK inside 40 years, and the whole
    point here is that it never does; E only looks at the final point; and despike is up-only.

    RECOVERY TIME is the discriminator, and it is a good one because the two hypotheses make
    opposite predictions. A catastrophe is followed by a recovery -- Hiroshima is back inside a
    decade, so are the 1945 war troughs -- because the place is still there. A definition change
    is not, because the smaller unit was never going to reach the bigger one's number. So: fall
    hard, fast, from a real size, and never come back.

    It is precise where it matters. Of the ~86 hits only five are pre-1900 and three of those
    five are real history (Amarapura 1820, Srirangapatna 1870, Matsumae 1870); after 1900 it is
    essentially all district-or-borough figures falling to the town. The graft seam is excluded
    for the usual reason -- past `sw` the neighbour is WUP, a different source, and the step
    there is merge_series' and already measured by SWITCH_STEPS."""
    out = []
    for c in C:
        p = real_points(c)
        sw = c.get("sw")
        for i in range(len(p) - 1):
            (y0, v0), (y1, v1) = p[i], p[i + 1]
            if v1 <= 0 or v0 < GONE_MIN_VAL:
                continue
            if y1 - y0 > GONE_WINDOW or v0 / v1 < GONE_RATIO:
                continue
            if sw is not None and y1 > sw:
                continue
            if any(v >= GONE_BACK * v0 for _, v in p[i + 2:]):
                continue                      # it came back: a trough, not a handover
            out.append((v0 / v1, y0, v0, y1, v1, c))
            break
    out.sort(key=lambda t: -t[2])
    head("H", "NEVER REGAINS (fell hard and stayed down; unit change?)", len(out),
         f">={GONE_RATIO}x within {GONE_WINDOW}y at >={fmt(GONE_MIN_VAL)}, "
         f"never back to {GONE_BACK:.0%}")
    for r, y0, v0, y1, v1, c in out[:SHOW]:
        print(f"  {y0}:{v0:>10,.0f} -> {y1}:{v1:>9,.0f}  ({r:>5.1f}x)   {c['n']}")
    return {id(c): [(y0, y1)] for _, y0, _, y1, _, c in out}


# ---------------------------------------------------------------- impact
IMPACT_PAD = 25   # a defect bleeds this far past its own years, via interpolation


def rank_impact(C, flagged, label):
    """How often does a flagged entry occupy a top-N slot IN THE YEARS ITS DEFECT SPANS?

    `flagged` maps id(city) -> list of (first_year, last_year) the defect covers.

    The span test is the whole point. Charging an entry in every sampled year overstates
    every check that fires on modern data: Chengdu's definition flip is in 1936, but the
    naive count also charged it for its year -50 and year 850 rankings, which the flip
    cannot reach. That inflated check F from 1% to 9% and made it look like the medieval
    ranking was being rewritten when the damage is a sawtooth in 1860-1948. It flattered
    nothing and misdirected the fixing, so impact is now scoped to the defect's own years."""
    years = list(range(-1000, 1901, 50)) + [1950, 2000, 2025]
    tot = bad = loose = 0
    hits = defaultdict(list)
    for y in years:
        rows = sorted(((pop_at(c, y), c) for c in C), key=lambda t: -t[0])[:TOPN]
        rows = [(v, c) for v, c in rows if v >= 5000]
        tot += len(rows)
        for v, c in rows:
            spans = flagged.get(id(c))
            if spans is None:
                continue
            loose += 1
            if any(a - IMPACT_PAD <= y <= b + IMPACT_PAD for a, b in spans):
                bad += 1
                hits[c["n"]].append(y)
    pct = bad / tot * 100 if tot else 0
    extra = f"  (entry-wide, ignoring when: {loose})" if loose != bad else ""
    print(f"\n  top-{TOPN} slots held by [{label}]: {bad}/{tot} ({pct:.0f}%) over {len(years)} sampled years{extra}")
    for n, ys in sorted(hits.items(), key=lambda kv: -len(kv[1]))[:8]:
        print(f"      {n:<28} {len(ys):>2} yrs  ({min(ys)}..{max(ys)})")
    return pct


# ---------------------------------------------------------------- F
OSC_RATIO  = 1.35    # up-then-down (or down-then-up) by at least this much...
OSC_WINDOW = 40      # ...and back again inside this many years
OSC_MIN_VAL = 300000


def straddles_seam(c, y0, y2):
    """Does this flip sit on the graft handover rather than in the source?

    build.py runs the historical series to its last year (`sw`) and then switches to WUP.
    Those are different measurements, so the switch is a step -- populstat's terminal figure
    is often a metro or county one while the WUP urban centre is just the dense core, and the
    centre then ramps back up. Wichita reads 344,000 (2000) -> 56,782 (2010) -> 133,000, which
    is exactly this check's dip-and-recover signature and is not oscillation at all. That step
    has its own metric already (SWITCH_STEPS / `build.py --steps`), and when the blend was
    replaced by a hard switch it became 89% of everything reported here, which buried the
    real source defects this check exists to find."""
    sw = c.get("sw")
    return sw is not None and y0 <= sw <= y2


def check_oscillation(C):
    """A city that jumps up and straight back down (or vice versa) is switching between two
    DEFINITIONS, not growing and shrinking. Real cities do collapse fast -- war, plague,
    a court leaving -- but they do not recover to the old level within a couple of decades
    and then fall again. This is the small-ratio sibling of check B: London alternated
    between the County (4.5M) and Greater London (7.4M) at only 1.6x, far under B's 8x, so
    B never saw it while the graph sawtoothed for fifty years."""
    out = {}
    spans = defaultdict(list)
    n_seam = 0
    for c in C:
        p = real_points(c)
        hits = []
        for i in range(len(p) - 2):
            (y0, v0), (y1, v1), (y2, v2) = p[i], p[i + 1], p[i + 2]
            if y2 - y0 > OSC_WINDOW or min(v0, v1, v2) <= 0:
                continue
            if max(v0, v1, v2) < OSC_MIN_VAL:
                continue
            up, dn = v1 / v0, v2 / v1
            if (up >= OSC_RATIO and dn <= 1 / OSC_RATIO) or (up <= 1 / OSC_RATIO and dn >= OSC_RATIO):
                if straddles_seam(c, y0, y2):
                    n_seam += 1
                    continue
                hits.append((y0, v0, y1, v1, y2, v2))
        if hits:
            out[c["n"]] = hits
            spans[id(c)] = [(h[0], h[4]) for h in hits]
    ranked = sorted(out.items(), key=lambda kv: (-len(kv[1]),
                                                 -max(x[3] for x in kv[1])))
    head("F", "DEFINITION OSCILLATION (series flips between two units)", len(out),
         f">={OSC_RATIO}x and back inside {OSC_WINDOW}y, at >={fmt(OSC_MIN_VAL)}"
         + (f"  [+{n_seam} skipped at the graft seam]" if n_seam else ""))
    for name, hits in ranked[:SHOW]:
        y0, v0, y1, v1, y2, v2 = hits[0]
        print(f"  {len(hits):>2} flip(s)  {y0}:{fmt(v0):>6} -> {y1}:{fmt(v1):>6} -> {y2}:{fmt(v2):<6}   {name}")
    return spans


# ---------------------------------------------------------------- G
GBM_JOIN_KM  = 5        # G: entry <-> Chandler row, by coordinate
GBM_MIN_LOST = 1        # G: report a run that eats at least this many benchmarks
GBM_VALUE_TOL = 2.0     # G: how far the run's value may sit from Chandler's and still count as
                        # the same figure. Above this the two compilers simply disagree and
                        # nothing is being lost -- see the note in the check.


def load_chandler_sites(path=os.path.join("data", "chandlerV2.csv")):
    """[(lat, lon, name, {year: pop})] from chandlerV2.csv.

    provenance.load_chandler() parses the same file but returns it keyed by "{City}-{Country}"
    without coordinates, and the shipped stadester_cities.json has no chandler_modelski_key to
    join on, so the join here is by position instead. That is not a fallback -- it is the more
    reliable key for this purpose, because the entries this check is about are ones stadester
    geocoded FROM the Chandler row, so the two agree to a few hundred metres.
    """
    out = []
    if not os.path.exists(path):
        print(f"note: {path} missing -- check G disabled")
        return out
    with open(path, encoding="cp1252", newline="") as f:
        rdr = csv.DictReader(f)
        ycols = []
        for col in (rdr.fieldnames or []):
            if col and (col.startswith("BC_") or col.startswith("AD_")):
                try:
                    n = int(col[3:])
                except ValueError:
                    continue
                ycols.append((col, n if col.startswith("AD_") else -n))
        for row in rdr:
            try:
                la, lo = float(row["Latitude"]), float(row["Longitude"])
            except (TypeError, ValueError):
                continue
            years = {}
            for col, year in ycols:
                v = (row.get(col) or "").strip().replace(",", "")
                if v:
                    try:
                        f_ = float(v)
                    except ValueError:
                        continue
                    if f_ > 0:
                        years[year] = f_
            if years:
                out.append((la, lo, row.get("City") or "?", years))
    return out


def check_strip_eats_benchmarks(raw):
    """Carry-forward runs that delete real Chandler benchmarks along with the repetition.

    build.py's strip keeps a run's FIRST point and deletes the rest. That is right when the run
    is populstat holding one estimate forward to the first census -- the defect it exists for.
    It is WRONG when the compiler asserted the same value at several successive benchmark years
    and the spline's hold afterwards echoes that value: CF_EPS is relative and stadester's float
    dust is ~1e-16, so the measured half and the repeated half fuse into a single run and the
    measured half goes too.

    Gao is the case that motivated this. Chandler states 75,000 at 1550, 1575, 1585 AND 1591 --
    four separate benchmarks -- and the fill from 1600 to 1930 sits 5e-16 away. The strip sees
    one 380-year run, keeps 1550, and deletes three real benchmarks; the fade then landed at
    1574, so Gao at 75,000 in 1575 -- the largest city in sub-Saharan Africa -- was invisible.
    Timbuktu loses its 1600 benchmark the same way, Sparta its AD 100 and AD 200.

    The fix is CF_END[key] = the run's last REAL benchmark year. `end` is already documented as
    "last year of genuine data" and the strip drops only `y > end`, so setting it FORWARD of the
    run's start keeps the measured half -- the mirror of the existing entries, which set it back
    when the run's own first value is already too late. Note CF_END also forces a fade, so a
    city that did not die wants a DISAPPEARED entry too.
    """
    try:
        import build
    except Exception as e:                                  # pragma: no cover
        print(f"note: cannot import build.py ({e}) -- check G disabled")
        return
    sites = load_chandler_sites()
    if not sites:
        return
    grid = defaultdict(list)
    for rec in sites:
        grid[(round(rec[0]), round(rec[1]))].append(rec)

    out = []
    for key, e in raw.items():
        co = e.get("coords")
        if not co or len(co) != 2:
            continue
        near = []
        for dla in (-1, 0, 1):
            for dlo in (-1, 0, 1):
                near += grid.get((round(co[0]) + dla, round(co[1]) + dlo), [])
        best = None
        for la, lo, nm, years in near:
            d = km(co[0], co[1], la, lo)
            if d <= GBM_JOIN_KM and (best is None or d < best[0]):
                best = (d, nm, years)
        if best is None:
            continue
        _, cname, cyears = best

        S = {}
        for ystr, v in (e.get("population") or {}).items():
            try:
                y = int(ystr)
            except (TypeError, ValueError):
                continue
            if isinstance(v, list):
                v = v[0] if v else None
            if y != 0 and v and float(v) > 0:
                S[y] = float(v)
        if len(S) < 3:
            continue
        pts = sorted(S.items())
        i = 0
        while i < len(pts):
            j = i
            while j + 1 < len(pts) and \
                    abs(pts[j + 1][1] - pts[i][1]) <= build.CF_EPS * max(pts[i][1], 1.0):
                j += 1
            if (key, pts[i][0]) in build.CF_KEEP:
                i = j + 1
                continue
            if pts[j][0] - pts[i][0] >= build.CF_MIN_SPAN and pts[i][1] >= build.CF_MIN_VAL:
                # (key, run start) first, then the bare key -- mirror strip_carry_forward, or the
                # forward CF_END entries are invisible here and Gao/Timbuktu/Sparta keep being
                # reported as unfixed after they have been fixed.
                end = build.CF_END.get((key, pts[i][0]), build.CF_END.get(key, pts[i][0]))
                # The year must be a Chandler benchmark AND Chandler must roughly agree with the
                # run's value, or this reports disagreements BETWEEN the two compilers as deleted
                # data. Guangzhou is the case that showed it: stadester holds 200,000 from 700 to
                # 1000, but Chandler's row STARTS at AD 1000 = 40,000 and carries no 700/800/900
                # benchmark at all -- so the "lost benchmark" at 1000 is populstat contradicting
                # Chandler five-fold, not a figure worth restoring. Deliberately a loose band and
                # not an equality test: stadester rescales, and Timbuktu's genuine case arrives as
                # 21,250 against Chandler's 25,000.
                lost = sorted(y for y, v in pts[i:j + 1]
                              if y > end and cyears.get(y, 0) > 0
                              and 1 / GBM_VALUE_TOL <= v / cyears[y] <= GBM_VALUE_TOL)
                if len(lost) >= GBM_MIN_LOST:
                    # What the deletion costs is not the points but the SHAPE. The strip leaves
                    # one anchor at the run's start, so the viewer draws a straight line from
                    # there to whatever survives next. Where that next value is far below, a
                    # plateau the source actually asserts becomes a centuries-long slide --
                    # which is Gao. Where it is comparable, the redraw is nearly the same line
                    # and the loss is cosmetic. So rank on the cliff, not on the count alone.
                    nxt = pts[j + 1][1] if j + 1 < len(pts) else None
                    cliff = (pts[i][1] / nxt) if nxt else 1.0
                    out.append((len(lost), cliff, pts[i][1], pts[i][0],
                                pts[j][0], lost, key, cname))
            i = j + 1

    out.sort(key=lambda t: -(t[1] * t[0]))
    steep = sum(1 for t in out if t[1] >= 2)
    head("G", "STRIP ATE A BENCHMARK (real Chandler years inside a carry-forward run)",
         len(out), f"join <={GBM_JOIN_KM}km, >={GBM_MIN_LOST} lost; "
                   f"{steep} of them followed by a >=2x cliff")
    for n, cliff, v, y0, y1, lost, key, cname in out[:SHOW]:
        ys = ", ".join(str(y) for y in lost[:5]) + (" ..." if len(lost) > 5 else "")
        # A key already in CF_END is a decision, not a defect: the hand table has declared the
        # city finished before these benchmarks, so losing them is the point. Marked, not
        # suppressed -- if the date is ever revisited this is the evidence against it.
        mark = " *" if key in build.CF_END else ""
        print(f"  {n:>2} lost  {fmt(v):>7}  run {y0:>5}..{y1:<5}  then /{cliff:>5.1f}x  "
              f"[{ys}]   {key}{mark}")
    if out:
        print("\n  * = key is already in CF_END, so the loss is a hand decision, not a defect."
              "\n  -> otherwise fix with CF_END[key] = the LAST year listed, which moves `end`"
              "\n     FORWARD and keeps the measured half of the run. CF_END also forces a fade,"
              "\n     so add a DISAPPEARED entry too if the city did not actually die -- or use"
              "\n     CF_KEEP[(key, run start)] where the whole plateau is real (Baghdad, Istanbul).")
    return out


def main():
    C = json.load(open(CITIES, encoding="utf-8"))["cities"]
    print(f"validating {CITIES}: {len(C):,} cities")

    raw_src = json.load(open(RAW, encoding="utf-8"))
    cf_spans, _ = check_carry_forward(C, deliberate_plateaus(raw_src))
    whip_spans = check_whipsaw(C)
    check_graft(C)
    stack_spans = check_stacking(C)
    check_terminal(C)
    osc_spans = check_oscillation(C)
    check_strip_eats_benchmarks(raw_src)
    gone_spans = check_never_regains(C)

    print(f"\n{'='*78}\nRANK IMPACT  (a defect is charged only in the years it spans)\n{'='*78}")
    rank_impact(C, cf_spans, "carry-forward")
    rank_impact(C, whip_spans, "whipsaw")
    rank_impact(C, stack_spans, "bad coords")
    rank_impact(C, osc_spans, "definition oscillation")
    rank_impact(C, gone_spans, "never regains")

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
