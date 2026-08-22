"""provenance.py -- per-YEAR source attribution for Stadester's merged city series.

stadester_cities.json is a merge of four datasets but each entry carries ONE `type` label
for the whole entry: populstat 21,836 / buringh 2,184 / chandler_modelski 187 / devries 12.
That label is a lie about most of the series. "London-United Kingdom" is typed `devries`
and is in fact Chandler benchmarks to 1690, two Bairoch/Buringh points at 1740 and 1790,
and then populstat's decennial censuses from 1801 to 2000 -- three sources under one word.
This module recovers the missing axis: which source each individual year came from, so the
viewer can colour a city's line by segment instead of by entry.

Three tiers, applied in order, each one narrower and more certain than the next:

  TIER 0  anchor vs fill.  Stadester fills the gaps between its real data points with
          straight lines in LINEAR population space, so a point lying on the chord between
          its neighbours is not a datum at all. Recovering the real points first is what
          makes the other two tiers possible -- and it is by itself the most valuable thing
          here, because "which parts of this line are measurements" is the question the
          chart cannot otherwise answer. 84% of populstat's points are fill.

  TIER 1  Chandler, by value.  data/chandlerV2.csv is vendored beside the source and 1,497
          entries name a row in it; 1,403 resolve. Where a recovered anchor lands on a year
          Chandler has for that city AND carries Chandler's number to within CHANDLER_TOL,
          the attribution is not an inference, it is an identification.

  TIER 2  the year grid.  Bairoch/Buringh publish on benchmark years and populstat does not,
          so the YEAR itself is evidence: an anchor at 1550 is ~400x more likely inside a
          buringh-typed entry than a populstat-typed one, while an anchor at 1911 is barely
          more likely at all. The grid is derived from the data (see year_grid) rather than
          typed in, so a source update that moves it trips REFERENCE_LR instead of quietly
          re-labelling half the map.

Anything the three tiers do not reach falls to the entry's type, at "default" confidence.
de Vries is deliberately NOT given a code of its own: 12 entries is too few to separate
from anything, and its entries (London, Amsterdam, Paris...) are populstat from 1800 on
like everyone else, so they fall through to the same default.

API (all cheap to call in a loop over 24k entries -- the CSV and the grid are derived once):

    init(entries)                -> derive the year grid from an already-loaded source dict
    load_chandler(path)          -> {chandler_key: {year: value}}
    points(pop)                  -> [(year, value), ...]     every year stadester states
    anchors(pop)                 -> [(year, value), ...]     the real ones only
    classify(entry)              -> {year: (source, confidence)}   EVERY year, fill included
    classify_all(entries)        -> {entry key: the above}
    year_grid() / grid_ratios()  -> the tier-2 benchmark years and their likelihood ratios

Wiring it into build.py: `provenance.init(raw)` straight after the source json is loaded,
then `provenance.classify(c)` inside the pass-1 loop. Passing `raw` to init is what keeps
this cheap -- 2.4s for the grid derivation and 0.6s for all 24k classifications, on a 10s
build. Calling classify() without init() works but makes the module load the 41MB source
itself. Note classify() keys on the entry's OWN years, so a year build.py invents later
(a CENSUS figure, a WUP graft point) will not be in the dict; treat a miss as unattributed
rather than assuming a code.

Standalone:
    python provenance.py             summary: source counts, the derived grid, anchor rates
    python provenance.py london      one entry's per-year attribution as a table
"""
import csv, json, os, statistics, sys
from collections import Counter, defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(_HERE, "data", "stadester", "stadester_cities.json")
CHANDLER_CSV = os.path.join(_HERE, "data", "chandlerV2.csv")
# chandlerV2.csv is cp1252, not utf-8: "A Coru\xf1a" at byte 6429 kills a utf-8 read outright.
# Every non-ASCII byte in it is a Latin-1 accented letter in a city or country name.
CHANDLER_ENC = "cp1252"

# --- the vocabulary -----------------------------------------------------------------
# `source` says which of the four merged datasets a year came from; `confidence` says how
# we know. Kept as two fields rather than one fused code because the viewer wants them for
# different jobs -- hue from the source, and opacity (or a dotted stroke) from the
# confidence, so a guessed populstat point does not look like a matched Chandler one.
CHANDLER  = "chandler"      # Chandler & Modelski's benchmark tables
BURINGH   = "buringh"       # Buringh / Bairoch, the European benchmark-year series
POPULSTAT = "populstat"     # populstat's annual census scrape -- the backbone of the merge
UNKNOWN   = "unknown"       # the entry carries a `type` this module has never heard of
FILL      = "fill"          # NOT A MEASUREMENT: stadester's straight-line gap fill

EXACT   = "exact"           # tier 1: identified against Chandler's own published number
GRID    = "grid"            # tier 2: inferred from the benchmark year
DEFAULT = "default"         # neither tier reached it; this is the entry's type talking
CONF_FILL = "fill"          # mirrors FILL so a caller reading only the confidence field
                            # cannot mistake an interpolated point for a datum

# --- tier 0 -------------------------------------------------------------------------
# Douglas-Peucker at an EXACT tolerance instead of build.py's 1%. Same algorithm, different
# question: build.py is compressing a series for the wire and wants the biggest simplification
# that still looks right, while this wants the smallest set of points that regenerates the
# series EXACTLY, i.e. the points stadester actually had. 1e-6 rather than 0.0 because the
# fill is computed in floating point and lands a few ulps off the chord -- the same float dust
# build.py's CF_EPS exists for (200000 vs 200000.00000000006). At 1e-6 a genuine datum would
# have to agree with the interpolation to seven figures to be missed.
ANCHOR_EPS = 1e-6

# --- tier 1 -------------------------------------------------------------------------
# 0.5% relative. Not exact equality, because stadester re-renders the numbers it ingests:
# London's Chandler 40,000 at 1199 survives verbatim but the neighbouring 1200 comes out as
# 40,049.50, and Calcutta's 1970 is 6,916,045.83 against Chandler's 6,900,000 (0.23%). Both
# are plainly the same datum. Widening past ~1% starts matching a DIFFERENT source's estimate
# of the same city-year, which is the one error this tier must not make.
CHANDLER_TOL = 0.005

# --- tier 2 -------------------------------------------------------------------------
# Two gates, and they do genuinely different jobs -- neither alone gives the right grid.
#
# GRID_MIN_LR is the discrimination test: how much more often does an anchor appear at this
# year inside a buringh-typed entry than inside a populstat-typed one? That ratio is huge for
# a Bairoch benchmark year (1550: 397x) and ~1 for an ordinary year.
#
# GRID_MIN_SHARE is the "is this actually a benchmark of theirs" test, and it is the gate that
# earns its keep. 1500 (57x), 1600 (39x) and 1861 (33x) all clear the ratio, and none of them
# is a Buringh benchmark: only 19%, 14% and 16% of buringh entries anchor there, against 82%
# at 1550 and 80% at 1840. Their ratios are high because the POPULSTAT rate is low in those
# years, not because Buringh publishes on them -- 1861 is a UK/Prussian census year carried by
# a few hundred entries, and 1500/1600 are years stadester's buringh series interpolate ACROSS
# (they anchor at 1550 and 1650 instead). Without this gate London's 1861 census reads as
# Buringh, which it is not. A real benchmark year shows up in most of its source's entries.
GRID_MIN_LR = 20.0
GRID_MIN_SHARE = 0.25
# Add-one on the populstat numerator: 700 and 800 have 16 and 17 populstat anchors between
# them across the whole dataset, and a year with zero would otherwise divide by nothing.
GRID_LR_SMOOTH = 1
# Series of 1 or 2 points have no interior, so every point is trivially an "anchor" and the
# rate is 100% by construction. They are excluded from all of the statistics below -- 2,435
# populstat entries and 86 chandler_modelski ones, and leaving them in drags populstat's
# median anchor share from 15.8% up to 17.2% and chandler_modelski's from 60% to 100%.
MIN_SERIES = 3

# What the grid measured on the current source (2026-08, stadester_cities.json as vendored).
# Recorded so a source update is LOUD: these ratios are the whole basis for calling a point
# Buringh rather than populstat, and if the mix underneath them shifts, the labels shift with
# it silently. Checked on first use; see _check_reference.
REFERENCE_LR = {
     700: 363,  800: 331,  900: 330, 1000: 335, 1100: 337, 1200: 334,
    1300: 278, 1400: 261, 1500:  57, 1550: 397, 1600:  39, 1650: 283,
    1740: 171, 1790: 105, 1840:  25, 1861:  33,
}
# Generous, because these are ratios of small counts (the 700 figure rests on 16 populstat
# anchors, so one entry moving is a 6% swing) and the point is to catch a re-merged source,
# not to pin a decimal. A 30% drift on a 400x ratio changes no label; a source update that
# actually moves the grid moves it by an order of magnitude.
REFERENCE_TOL = 0.30

# --- the AD 100 / 100 BC mis-filing --------------------------------------------------
# chandlerV2.csv's `BC_100` column is not 100 BC. It is Chandler's AD 100 benchmark, filed
# 200 years early. Three independent things say so:
#   * of 1,597 rows only Cadiz appears in both BC_100 and AD_100 -- two columns for adjacent
#     benchmarks of the same table should overlap heavily, not once;
#   * Rome is absent from BC_100 and present in AD_100 at 450,000, which is backwards for any
#     table of the ancient world;
#   * BC_100 contains London 30,000, Lyon 50,000, Nimes 44,000, Corinth 50,000 and Carthage
#     100,000 -- cities that in 100 BC did not exist (London, Lyon, Nimes), were a razed field
#     (Carthage, Corinth: both destroyed in 146 BC), and were exactly those sizes in AD 100.
# build.py repairs this by moving the values to year 100, so this module reads the column at
# year 100 to match. Stadester read the SAME csv and inherited the SAME bug, so its own copies
# of those benchmarks currently sit at year -100 and will move to 100 when the repair lands --
# which is why the lookup accepts either (see _chandler_at). Cadiz is the one row with both
# columns populated (BC_100 62,000, AD_100 65,000); the correctly-filed AD_100 wins.
BC100_YEAR = 100

# --- module state ---------------------------------------------------------------------
# All lazy and all derived once. classify() is called 24k times in a build loop, so nothing
# below may re-read a file or re-scan the corpus.
_chandler = None        # {chandler_key: {year: value}}
_grid = None            # {year: likelihood ratio}, the tier-2 benchmark years
_grid_lr = None         # {year: likelihood ratio} for EVERY measurable year (the diagnostic)
_grid_share = None      # {year: fraction of buringh entries anchoring there}
_anchor_cache = None    # {id(population dict): (points, anchors)}
_anchor_owner = None    # the entries dict the cache was built from -- see init()


def points(pop):
    """One series as a clean sorted [(year, value), ...] -- every year stadester states.

    `pop` is stadester's own {"1550": 76941.54, ...} dict: string years, and 17 values across
    the corpus are LISTS (two figures for one year, of which we take the first, as build.py
    does). Zero, null and empty values are absence rather than data and are dropped, so they
    are not reported as fill. Year keys are NOT range-clipped here -- build.py owns YEAR_LO /
    YEAR_HI and this module must describe the entry it is handed, not a different one.
    A plain [(year, value)] sequence is accepted too, for callers holding a cleaned series.
    """
    if not isinstance(pop, dict):
        return sorted(pop)
    pts = []
    for ystr, v in pop.items():
        try:
            y = int(ystr)
        except (TypeError, ValueError):
            continue                           # a year key that is not a number at all
        if isinstance(v, list):
            v = v[0] if v else None
        if not v or v <= 0:
            continue
        pts.append((y, float(v)))
    pts.sort()
    return pts


def anchors(pop):
    """The real data points of one series: [(year, value), ...], stadester's fill removed."""
    return _anchor_split(points(pop))


def _anchor_split(pts):
    if len(pts) < 3:
        return pts[:]                          # no interior -> nothing can be fill
    return _split(pts, ANCHOR_EPS)


def _split(pts, releps):
    """Douglas-Peucker in linear space, deviation relative to local magnitude.

    Deliberately a copy of build.py's dp_simplify rather than an import: build.py imports THIS
    module, so importing it back would be a cycle, and the two want different tolerances for
    different reasons anyway (see ANCHOR_EPS). If dp_simplify's geometry ever changes, this
    must change with it -- they are answering the same question about the same fill.
    """
    if len(pts) <= 2:
        return pts[:]
    x0, y0 = pts[0]
    x1, y1 = pts[-1]
    dmax, idx = 0.0, 0
    dx = x1 - x0
    for i in range(1, len(pts) - 1):
        x, y = pts[i]
        ychord = y0 + (y1 - y0) * ((x - x0) / dx) if dx else y0
        d = abs(y - ychord) / max(abs(ychord), abs(y), 1.0)
        if d > dmax:
            dmax, idx = d, i
    if dmax > releps:
        return _split(pts[:idx + 1], releps)[:-1] + _split(pts[idx:], releps)
    return [pts[0], pts[-1]]


# --- tier 1: the Chandler table ---------------------------------------------------------
def load_chandler(path=CHANDLER_CSV):
    """chandlerV2.csv -> {"{City}-{Country}": {year: population}}.

    The csv is City,OtherName,Country,Latitude,Longitude,Certainty and then 806 year columns
    named BC_nnnn / AD_nnnn, almost all of them blank on any given row. The key format is the
    one stadester's `chandler_modelski_key` field already uses, so entries join straight on.
    1,592 of 1,597 rows carry at least one population; 1,403 of the 1,497 keyed stadester
    entries resolve.
    """
    table = {}
    if not os.path.exists(path):
        print(f"note: {path} missing -- tier 1 (Chandler value match) disabled")
        return table
    with open(path, encoding=CHANDLER_ENC, newline="") as f:
        rdr = csv.DictReader(f)
        ycols = []
        for col in (rdr.fieldnames or []):
            if not col or not (col.startswith("BC_") or col.startswith("AD_")):
                continue
            try:
                n = int(col[3:])
            except ValueError:
                continue
            if col == "BC_100":
                ycols.append((col, BC100_YEAR, True))     # the mis-filed column; see BC100_YEAR
            else:
                ycols.append((col, n if col.startswith("AD_") else -n, False))
        for row in rdr:
            years = {}
            for col, year, is_bc100 in ycols:
                raw = (row.get(col) or "").strip().replace(",", "")
                if not raw:
                    continue
                try:
                    v = float(raw)
                except ValueError:
                    continue
                if v <= 0:
                    continue
                if is_bc100 and year in years:
                    continue                  # Cadiz only: a real AD_100 already won the slot
                years[year] = v
            if years:
                table[f"{row['City']}-{row['Country']}"] = years
    return table


def chandler():
    """The Chandler table, loaded once."""
    global _chandler
    if _chandler is None:
        _chandler = load_chandler()
    return _chandler


def _chandler_at(row, year):
    """Chandler's value for one city-year, or None.

    Year -100 is aliased onto 100 for the AD-100 mis-filing (see BC100_YEAR): stadester holds
    those benchmarks at -100 today and build.py's repair moves them to 100, so accepting both
    keeps this module correct on either side of that change. The alias can only mis-fire if an
    entry holds anchors at BOTH -100 and 100 whose values agree with Chandler's AD 100 figure
    to within CHANDLER_TOL, which no entry in the corpus does.
    """
    if year == -100:
        return row.get(BC100_YEAR)
    return row.get(year)


# --- tier 2: deriving the benchmark-year grid --------------------------------------------
def init(entries=None):
    """Derive the year grid. Call once, with the source dict build.py has already loaded.

    Passing `entries` matters: without it this re-reads and re-parses the 41MB source JSON,
    which costs more than the whole rest of the module. Given the dict, the derivation is one
    anchor pass over 24k series (~2s) and it caches those anchors, so a following loop of
    classify() calls over the SAME entry objects is free. classify() on any other dict -- a
    cleaned or clipped copy -- simply recomputes, so the cache can never answer for a series
    it did not see.
    """
    global _grid, _grid_lr, _grid_share, _anchor_cache, _anchor_owner
    if entries is None:
        with open(SRC, encoding="utf-8") as f:
            entries = json.load(f)
    # Cache keyed by the identity of each population dict, which is safe ONLY because we also
    # hold `entries` alive: every cached id therefore still belongs to the object it was taken
    # from, so no freed dict's id can be recycled underneath us.
    _anchor_cache = {}
    _anchor_owner = entries

    # Per year, over each of the two types with enough entries to measure: how many entries
    # have an ANCHOR there, and how many have any value there at all. The rate we want is
    # anchors / all entries of the type, not anchors / entries covering the year -- the
    # question is "how often does a randomly chosen buringh entry put a real data point on
    # this year", and an entry that does not reach the year is a genuine no.
    anc = {BURINGH: Counter(), POPULSTAT: Counter()}
    total = Counter()
    for c in entries.values():
        pop = c.get("population") or {}
        pts = points(pop)
        anch = _anchor_split(pts)
        _anchor_cache[id(pop)] = (pts, anch)
        t = c.get("type")
        if t not in anc or len(pts) < MIN_SERIES:
            continue                          # no interior; see MIN_SERIES
        total[t] += 1
        for y, _ in anch:
            anc[t][y] += 1
    nb, npo = total[BURINGH], total[POPULSTAT]

    _grid_lr, _grid_share, _grid = {}, {}, {}
    for y in anc[BURINGH]:
        share = anc[BURINGH][y] / nb
        lr = share / ((anc[POPULSTAT][y] + GRID_LR_SMOOTH) / npo)
        _grid_lr[y] = lr
        _grid_share[y] = share
        if lr >= GRID_MIN_LR and share >= GRID_MIN_SHARE:
            _grid[y] = lr
    _check_reference()
    return _grid


def _check_reference():
    """Shout if the derived ratios have moved away from the ones this module was written on."""
    bad = []
    for y, want in sorted(REFERENCE_LR.items()):
        got = _grid_lr.get(y)
        if got is None or abs(got - want) > REFERENCE_TOL * want:
            bad.append(f"{y}: recorded {want}x, derived "
                       + ("not measurable" if got is None else f"{got:.1f}x"))
    if bad:
        raise AssertionError(
            "provenance.py: the year-grid likelihood ratios have moved -- the source's dataset "
            "mix has changed and the tier-2 labels are no longer the ones this module was "
            "validated on. Re-derive with `python provenance.py` and update REFERENCE_LR "
            "(and check GRID_MIN_LR / GRID_MIN_SHARE still cut in the right place):\n  "
            + "\n  ".join(bad))


def year_grid():
    """{year: likelihood ratio} -- the benchmark years tier 2 attributes to Buringh/Bairoch."""
    if _grid is None:
        init()
    return _grid


def grid_ratios():
    """{year: likelihood ratio} for every measurable year, gated or not. Diagnostic only."""
    if _grid_lr is None:
        init()
    return _grid_lr


def _series_of(pop):
    """(points, anchors) for one population dict, from init()'s cache where possible."""
    if _anchor_cache is not None:
        got = _anchor_cache.get(id(pop))
        if got is not None:
            return got
    pts = points(pop)
    return pts, _anchor_split(pts)


# --- the classifier ---------------------------------------------------------------------
def classify(entry, ch=None):
    """{year: (source, confidence)} for EVERY year in one entry's series.

    Fill points are reported, as (FILL, CONF_FILL) -- they are most of the series and the
    single most useful thing here is knowing which parts of a city's line are measurements
    and which are stadester drawing a straight line between two of them. Callers that want
    only the data points can filter on `src != FILL`.
    """
    pop = entry.get("population") or {}
    row = None
    if ch is None:
        ch = chandler()
    key = entry.get("chandler_modelski_key")
    if key:
        row = ch.get(key)
    grid = year_grid()
    t = entry.get("type")
    # The fall-through, once the two evidence tiers have had their say.
    #   chandler_modelski -- the entry is nothing BUT Chandler-Modelski: 187 entries, median
    #     series length 3, and NOT ONE of them carries a chandler_modelski_key, so tier 1 can
    #     never speak for them and the type label is the only evidence there is.
    #   populstat / buringh / devries -- populstat, because it is the annual backbone under all
    #     three. A buringh or devries entry is a handful of benchmark years spliced onto a
    #     populstat census run, and outside the benchmark years (which tier 2 has already
    #     taken) what is left is that run: the anchors at 1861, 1911, 1931, 1947, 1999 inside
    #     stadester's `devries` London are the UK census, not de Vries.
    #   anything else -- UNKNOWN, which is how a fifth dataset appearing in the source
    #     announces itself instead of being quietly labelled populstat.
    if t == "chandler_modelski":
        fallback = (CHANDLER, DEFAULT)
    elif t in ("populstat", "buringh", "devries"):
        fallback = (POPULSTAT, DEFAULT)
    else:
        fallback = (UNKNOWN, DEFAULT)

    pts, anch = _series_of(pop)
    out = {}
    for y, v in anch:
        cv = _chandler_at(row, y) if row else None
        if cv is not None:
            if abs(v - cv) <= CHANDLER_TOL * max(abs(v), abs(cv)):
                out[y] = (CHANDLER, EXACT)      # tier 1: identified, not inferred
                continue
            # Chandler published a benchmark for this city-year and our number is not it, so
            # tier 1 says no. Tier 2 is also barred here, deliberately: Chandler's benchmark
            # years and Buringh's overlap almost completely before 1700 (1550 and 1650 are
            # both), so a year Chandler covers is exactly the year the grid's "buringh, not
            # populstat" prior stops being about anything. London's 1550 and 1650 are this
            # case -- 76,941 against Chandler's 74,000 and 404,969 against 410,000 -- and the
            # honest reading is "a rendering of somebody's benchmark that is not Chandler's",
            # not "Buringh". They fall to the entry's backbone with the rest.
            out[y] = fallback
            continue
        if y in grid:
            out[y] = (BURINGH, GRID)            # tier 2: the year is the evidence
            continue
        out[y] = fallback

    for y, _ in pts:                            # everything not an anchor is stadester's fill
        if y not in out:
            out[y] = (FILL, CONF_FILL)
    return out


def classify_all(entries):
    """{entry key: {year: (source, confidence)}} for a whole source dict."""
    init(entries)
    return {k: classify(c) for k, c in entries.items()}


# --- standalone -------------------------------------------------------------------------
def _load_src():
    with open(SRC, encoding="utf-8") as f:
        return json.load(f)


def _summary():
    entries = _load_src()
    init(entries)
    ch = chandler()
    print(f"{len(entries):,} entries; chandlerV2 {len(ch):,} keyed rows")

    # --- tier 0 -------------------------------------------------------------------
    fracs = defaultdict(list)
    for c in entries.values():
        pts, anch = _series_of(c.get("population") or {})
        if len(pts) >= MIN_SERIES:
            fracs[c.get("type")].append(len(anch) / len(pts))
    print(f"\n-- tier 0: median share of a series that is a real data point"
          f" (series of {MIN_SERIES}+ points)")
    for t, v in sorted(fracs.items(), key=lambda kv: -len(kv[1])):
        print(f"   {t:18s} n={len(v):6,}   {statistics.median(v) * 100:5.1f}%")

    # --- tier 2 -------------------------------------------------------------------
    print(f"\n-- tier 2: derived year grid (LR >= {GRID_MIN_LR:g}x and >= "
          f"{GRID_MIN_SHARE * 100:g}% of buringh entries anchoring there)")
    for y in sorted(_grid):
        ref = REFERENCE_LR.get(y)
        note = "" if ref is None else f"   recorded {ref}x"
        print(f"   {y:5d}  LR {_grid[y]:7.1f}x   buringh share {_grid_share[y] * 100:5.1f}%{note}")
    near = [(y, _grid_lr[y], _grid_share[y]) for y in _grid_lr
            if y not in _grid and (_grid_lr[y] >= GRID_MIN_LR and _grid_share[y] >= 0.05)]
    print("   -- rejected, ratio passes but the year is not a Buringh benchmark:")
    for y, lr, sh in sorted(near, key=lambda r: -r[1])[:8]:
        print(f"   {y:5d}  LR {lr:7.1f}x   buringh share {sh * 100:5.1f}%"
              f"{'   recorded ' + str(REFERENCE_LR[y]) + 'x' if y in REFERENCE_LR else ''}")
    print(f"   reference check ({len(REFERENCE_LR)} recorded ratios, +/-"
          f"{REFERENCE_TOL * 100:g}%): PASS")

    # --- the attribution itself ---------------------------------------------------
    byboth, bysrc, fill = Counter(), Counter(), 0
    tier1_years = tier1_hits = 0
    for c in entries.values():
        row = ch.get(c.get("chandler_modelski_key") or "")
        for y, (s, conf) in classify(c, ch).items():
            if s == FILL:
                fill += 1
                continue
            byboth[(s, conf)] += 1
            bysrc[s] += 1
        if row:
            for y, v in _series_of(c.get("population") or {})[1]:
                cv = _chandler_at(row, y)
                if cv is not None:
                    tier1_years += 1
                    if abs(v - cv) <= CHANDLER_TOL * max(abs(v), abs(cv)):
                        tier1_hits += 1
    n = sum(byboth.values()) or 1
    print(f"\n-- attribution over all {n:,} real data points ({fill:,} fill points excluded,"
          f" {n / (n + fill) * 100:.1f}% of the corpus is real)")
    for (s, conf), k in sorted(byboth.items(), key=lambda kv: -kv[1]):
        print(f"   {s:10s} {conf:8s} {k:8,}  {k / n * 100:5.1f}%")
    print("   " + "-" * 40)
    for s, k in sorted(bysrc.items(), key=lambda kv: -kv[1]):
        print(f"   {s:10s} {'':8s} {k:8,}  {k / n * 100:5.1f}%")
    print(f"\n-- tier 1: anchors landing on a year Chandler has for that city: {tier1_years:,}"
          f"; carrying Chandler's value to {CHANDLER_TOL * 100:g}%: {tier1_hits:,}"
          f" = {tier1_hits / max(tier1_years, 1) * 100:.1f}%")


def _show(fragment):
    entries = _load_src()
    init(entries)
    ch = chandler()
    frag = fragment.lower()
    hits = [k for k in entries if frag in k.lower()]
    if not hits:
        hits = [k for k, c in entries.items() if frag in (c.get("name") or "").lower()]
    if not hits:
        print(f"no entry matching {fragment!r}")
        return
    hits.sort(key=lambda k: (-len(entries[k].get("population") or {}), k))
    if len(hits) > 1:
        print(f"{len(hits)} entries match {fragment!r}; showing the richest."
              f" others: {', '.join(hits[1:9])}{' ...' if len(hits) > 9 else ''}\n")
    key = hits[0]
    c = entries[key]
    values = dict(_series_of(c.get("population") or {})[0])
    row = ch.get(c.get("chandler_modelski_key") or "")
    marks = classify(c, ch)
    real = sum(1 for s, _ in marks.values() if s != FILL)
    print(f"{key}   type={c.get('type')}   chandler_modelski_key="
          f"{c.get('chandler_modelski_key')!r} ({'resolves' if row else 'no row'})")
    print(f"{real} real data points of {len(marks)} values\n")
    print(f"  {'year':>6}  {'value':>14}  {'chandler':>14}  {'source':<10} {'confidence'}")
    for y in sorted(marks):
        s, conf = marks[y]
        cv = _chandler_at(row, y) if row else None
        if s == FILL:
            print(f"  {y:6d}  {values.get(y, 0):14,.0f}  {'':>14}  fill")
        else:
            print(f"  {y:6d}  {values.get(y, 0):14,.0f}  "
                  f"{format(cv, ',.0f') if cv is not None else '':>14}  {s:<10} {conf}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        _show(" ".join(sys.argv[1:]))
    else:
        _summary()
