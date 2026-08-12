"""build.py — turn Stadester's city JSON into a compact cities.json for the viewer.

Steps:
  1. load stadester_cities.json (per-city annual population series + coords + provenance)
  2. clip corrupt year keys, drop empty/no-coord entries
  3. drop parenthetical metro variants ("(agglomeration)", "(Greater ...)", "(inner ...)")
     -> v1 keeps city-proper figures only (see FINDINGS.md / todo.txt)
  4. Douglas-Peucker simplify each series in log-population space -> a handful of
     control points per city (the browser re-interpolates in log space, giving smooth
     year-to-year transitions from a tiny payload)
  5. emit data/cities.json: {yearMin, yearMax, cities:[{n,la,lo,t,p:[[year,pop],...]}]}

Source-defect repairs live in the tables below (DROP_KEYS / CLIP_BEFORE / RENAME / CF_*) and
in data/coord_fixes.json. Run `python validate.py` after building: it re-runs the five checks
that found each class of defect, so a regression -- or a fresh one from a source update --
shows up as a count going the wrong way. Regenerate coord_fixes.json with the two scripts in
tools/ (see tools/make_coordfix.py).
"""
import json, math, os, unicodedata
from collections import defaultdict

SRC = "data/stadester/stadester_cities.json"
GHSL = "data/stadester/wup2025.json"  # UN WUP 2025 agglomerations (annual 1975-2025, GHSL-lineage,
                                      # cleaner+broader than the old ghsl.json; same schema). Run prep_wup.py.
OUT = "data/cities.json"
COORDFIX = "data/coord_fixes.json"   # repaired coords for geocoder-fallback entries

YEAR_LO, YEAR_HI = -4000, 2035      # clip range (kills the one corrupt key 19690310)
DP_EPS_REL = 0.01                   # linear-space relative tolerance for line simplification
DROP_MARKERS = ("(agglomeration)", "(greater", "(inner", "(metropolitan", "(metro")

# --- New World antiquity ramp -------------------------------------------------
# populstat/Chandler-Modelski file several American cities with Classic-era peak
# populations stamped back into deep antiquity: Teotihuacan 150k @ 800 BC, and
# Tikal/Caracol/Tula/Tiwanaku as flat ~100k *constants* from 900 BC onward (the peak
# value entered for every year). These are archaeologically impossible -- no city in
# the Americas approached that size before ~1 CE, and most only urbanized in the Late
# Preclassic. A gap/shape filter can't catch them without also deleting the real Old
# World giants (Pi-Ramesses, Nineveh, Anyang share the same sparse "one ancient
# benchmark, then modern data" shape), so we discriminate on region instead.
#
# Rather than a hard cutoff (which makes American cities pop into existence full-size
# at the boundary), we apply a time-varying CAP that ramps up: nothing American is
# visible before NW_RAMP_START, and real magnitudes are only allowed from NW_RAMP_FULL
# on. Between them the cap rises log-linearly, so a city fades in gradually. The cap
# only ever *reduces* an implausibly-large early value; a genuinely small early city
# is left untouched. Points before NW_RAMP_START are dropped outright.
NEW_WORLD_LON = -30                 # longitudes west of this are the Americas
NW_RAMP_START = -200                # American cities invisible before here (Late Preclassic)
NW_RAMP_FULL  = 400                 # real magnitudes fully allowed from here (Classic era)
NW_CAP_TOP    = 2e7                 # cap ceiling at NW_RAMP_FULL (effectively "no cap")

# Anglo/French North America had no cities before ~1700, so ALL pre-1700 population there is
# populstat placeholder, not data: "Cincinnati" held 10k from 1425, "Saint Louis" 40k from
# 1100, even "Boston" has a fabricated 1000->5k ramp from year 1000 (founded 1630!). Every
# real colonial city (Boston, New York, Philadelphia, Quebec) has its genuine series start
# ~1700, so clipping pre-1700 keeps all real data. Latin America is excluded (lat < 32) --
# it had genuine cities from the 1500s (Potosi, Mexico City, Cusco), and St Augustine (29.9N)
# stays below the cutoff too.
NA_LAT_MIN = 32
NA_LON = (-130, -55)
NA_CLIP_BEFORE = 1700

def nw_cap(y):
    """Max allowed population for an American city at year y (None = uncapped)."""
    if y >= NW_RAMP_FULL:
        return None
    f = (y - NW_RAMP_START) / (NW_RAMP_FULL - NW_RAMP_START)
    return 10 ** (math.log10(PEAK_FLOOR) + f * (math.log10(NW_CAP_TOP) - math.log10(PEAK_FLOOR)))

# --- entries to drop outright (mislabeled duplicates of a city already present) ---
DROP_KEYS = {
    "Kensington and Chelsea-United Kingdom",  # sub-borough carrying Greater London's figure;
                                              # a real "London" entry already exists -> duplicate

    # A garbage-bin entry: populstat's Ruhr page nests several geographic units and they were
    # all merged under one key. Its series swings town -> Regierungsbezirk -> Landkreis ->
    # Ruhrgebiet and back (8k in 1905, 1.8M in 1914, 9.4k in 1933, 5.46M in 1975, 29.9k in
    # 2000) -- only the 5-figure values are Wetter. Essen/Dortmund/Duisburg/Bochum all exist
    # separately and Essen carries the conurbation, so nothing is lost by dropping this.
    "Wetter (Ruhr)-Germany",

    # Hong Kong: populstat's district table was scattered across two country keys and badly
    # geocoded. "Beijiao" (= North Point, a district) holds the whole TERRITORY's series and
    # is renamed to Hong Kong below; everything here is either a district inside that figure
    # or a mis-geocoded duplicate of it, so all of them are double-counts.
    "Xianggang-Hong Kong",    # city-proper Hong Kong, superseded by the territory series
    "Jiulong-Hong Kong",      # Kowloon        - inside the territory total
    "Xinjiulong-Hong Kong",   # New Kowloon    - ditto
    "Quanwan-Hong Kong",      # Tsuen Wan      - ditto
    "Shatian-Hong Kong",      # Sha Tin        - ditto
    "Tunmen-Hong Kong",       # Tuen Mun       - ditto
    # ...and the same districts again, mis-keyed to China and geocoded to its centroid or to
    # random mainland towns of similar name (Jiulong -> a Sichuan county, Quanwan -> Quannan).
    "Xianggang-China",        # 28.8km from the 27.5M Guangzhou centre -> was stealing it
    "Jiulong-China", "Quanwan-China", "Tuen Mun-China",
    "Tia Po-China", "Sha Tin-China", "Yuen Long-China",
}

# --- clip everything before a year (entry mixes two different places' histories) ----
# Danapur is a cantonment 10km from Patna carrying PATALIPUTRA's ancient figures, so both
# entries sat in the top-20 from 400 BC to AD 200 -- one city, counted twice. Patna keeps
# the ancient series; Danapur keeps only its own (real, modern) census record.
CLIP_BEFORE = {
    "Dânâpur-India": 1891,   # its own first census; everything before is Pataliputra's
    # Stadester fuzzy-matched "Shanghai" to Chandler's "Shangqi" (a Shang-dynasty city) and
    # stapled its 100-130k antiquity series onto modern Shanghai's coordinates -- see the
    # entry's own chandler_modelski_key. Shanghai was a fishing and salt village until the
    # Ming; chandlerV2's own Shanghai row correctly starts at 1554, so start there.
    "Shanghai-China": 1554,
}

# --- individual bad data points to delete ------------------------------------------
# For one-off definition switches too small to trip the whipsaw check (<2x) but plainly
# visible on the graph. London's 1875/1880 pair is Greater-London-sized inside an otherwise
# County-of-London run, so the line jumps to 4.24M and falls back to 3.81M in 1881.
DROP_YEARS = {
    "London-United Kingdom": {1875, 1880},
    # Paris has the identical defect in the identical years -- populstat's 1914 and 1925 rows
    # are agglomeration figures (4.0M, 4.8M) dropped into a Ville-de-Paris series that is
    # ~2.9M on either side. Unlike London there is no separate variant entry to switch to,
    # so the spikes just go. (Its 1946->1962 climb is still a definition change rather than
    # growth, but it is at least monotone; check F keeps reporting it.)
    "Paris-France": {1914, 1920, 1925},
}

# --- prefer a metro variant's series over the base entry's -------------------------
# v1's rule is "city-proper history, agglomeration only via the modern graft", and the
# parenthetical variants are dropped. That breaks down when the BASE entry is itself
# internally inconsistent -- populstat's London interleaves two definitions year by year:
#   1911: 4,521,700   (County of London)      1914: 7,419,000   (Greater London)
#   1921: 4,484,500   (County of London)      1925: 7,742,000   (Greater London)
# so the map showed London sawtoothing between 4.5M and 7.7M through the 1910s-20s, then
# sitting at ~4M until 1960 while the real city was over 8M -- and 1961-1968 is a straight
# interpolation across the definition change, not data. The "(Greater London)" variant is a
# clean, consistent series over exactly that span (6.34M in 1901 -> 8.20M in 1931), so for
# the years it covers we take it and let the base entry supply the deep history.
PREFER_VARIANT = {
    "London-United Kingdom": "London (Greater London)-United Kingdom",
    # same story, smaller city: the base flips between Manchester proper (342k in 1881) and
    # the conurbation (590k in 1880, 505k in 1891)
    "Manchester-United Kingdom": "Manchester (agglomeration)-United Kingdom",
}

# --- display renames ----------------------------------------------------------
# Stadester files ancient capitals under the modern town sitting on the ruins, so
# the antiquity charts read "Faqus / Al-Uqsur / Hillah" instead of the names people
# know (and search for). Keyed by the exact source key to avoid renaming a same-named
# entry elsewhere (there are two "Hillah", two "Anyang"). Modern name kept in parens
# where the modern city is itself notable. NOTE: not for sub-district duplicates like
# "Kensington and Chelsea" (a real London already exists) — those need dedup, not rename.
RENAME = {
    "Fâqûs-Egypt":                    "Pi-Ramesses (Faqus)",   # Ramesside capital at Qantir/Avaris
    "Uqsur, Al--Egypt":               "Thebes (Luxor)",        # Al-Uqsur = Luxor = ancient Thebes
    "Badrashayn, Al--Egypt":          "Memphis",               # Al-Badrashayn sits on Memphis
    "Hillah-Iraq":                    "Babylon (Hillah)",      # Al-Hillah beside the Babylon ruins
    "Mosul-Iraq":                     "Nineveh (Mosul)",       # Nineveh across the Tigris from Mosul
    "Anyang-China":                   "Anyang (Yin)",          # Shang capital Yin / Yinxu
    "Drigh Road Cantonment-Pakistan": "Karachi",               # Karachi's number filed under a cantonment
    "Sõul-South Korea":               "Seoul",                 # McCune-Reischauer "Sŏul" mojibake
    "Nunjiang-China":                 "Nanjing",               # garbled name; coords are Nanjing (Ming capital)
    "Hospet-India":                   "Vijayanagara (Hospet)", # modern Hospet on the Vijayanagara/Hampi site

    # Hong Kong's territory series is filed under one of its districts (Beijiao = North Point).
    "Beijiao-Hong Kong":              "Hong Kong",
    # More ancient capitals filed under the modern town on the ruins (see note above).
    "Al Marsâ-Tunisia":               "Carthage (La Marsa)",   # La Marsa on the Carthage headland
    "Al-bu Kamal-Syria":              "Mari (Abu Kamal)",      # Mari, destroyed by Hammurabi c.1761 BC
    "Selçuk-Turkey":                  "Ephesus (Selçuk)",      # Selçuk = the Ephesus site
    "Bergama-Turkey":                 "Pergamon (Bergama)",
    "Sûr-Lebanon":                    "Tyre (Sur)",
    "Al-Quds-Palestine":              "Jerusalem",             # Al-Quds = Jerusalem
    "Al Khums-Libya":                 "Leptis Magna (Al Khums)",
    "Salihli-Turkey":                 "Sardis (Salihli)",
    "Kaspican-Bulgaria":              "Pliska (Kaspichan)",    # First Bulgarian Empire capital
    "Santa Maria Capua Vetere-Italy": "Capua (S.M. Capua Vetere)",
    # Crete: the Bronze Age figures belong to Knossos and Gortyn, not to the modern towns
    # beside them. A real "Iraklio" entry already exists with its own 700 AD -> present series.
    "Nea Alikarnassos-Greece":        "Knossos",               # Minoan palace, 5km from the suburb
    "Mires-Greece":                   "Gortyn",                # Roman capital of Crete, beside Mires
}
PEAK_FLOOR = 2000                   # drop cities that never reach this (runtime cutoff is higher)
GRAFT_RADIUS_KM = 50                # match a city to the largest GHSL urban centre within this.
                                    # Was 30, which cut between a city and its OWN centre when
                                    # the centre's population-weighted centroid sits off the old
                                    # town: Guangzhou is 31.6km from the 27.5M Guangzhou centre
                                    # and so fell through to Huadu (882k); Chiba is 41.5km from
                                    # the Tokyo polygon that contains it and fell through to a
                                    # 62k neighbour. At 50km both reach the right centre -- and
                                    # a city that is NOT its centre's principal now correctly
                                    # gets no graft at all instead of a wrong tiny one.
GRAFT_MIN_FRAC = 0.05               # only graft if the principal is >=5% of the centre's size
                                    # (stops a tiny orphan town inheriting a whole megacity)

# --- populstat carry-forward -------------------------------------------------
# populstat repeats a single pre-modern estimate VERBATIM for every year until the first
# modern census, then cliffs to it. That leaves dead cities frozen at their peak for
# centuries: Vijayanagara was sacked and abandoned in 1565 but sits at 480,000 until 1890;
# Kamakura holds 200,000 for the 690 years after the shogunate fell; Mari holds its Bronze
# Age figure for 3,780 years. On the map they occupy top-10 slots for centuries.
#
# Genuine data essentially never repeats a value to the byte across 150+ years, so an exact
# flat run of that length is the signature. We keep the run's FIRST point (the real estimate)
# and delete the rest, then drop a floor marker so the city fades out after its last real
# datum rather than plateauing. All 136 detected runs were reviewed individually; only the
# four below were judged genuine, so the rule is strip-by-default with an explicit keep-list.
CF_MIN_SPAN = 150                   # exact-flat run this long = carry-forward, not data
CF_MIN_VAL  = 20000                 # ignore villages; their flat runs don't distort anything
CF_EPS      = 1e-9                  # relative tolerance for "the same value". Stadester's
                                    # spline leaves float dust (200000 vs 200000.00000000006),
                                    # and an exact == splits one run into two, leaving a
                                    # surviving anchor at the wrong end of it.
# Keyed by (source key, the run's FIRST year) -- a city can have one genuine plateau and one
# bogus one, so a key-wide exemption is too blunt. Istanbul is exactly that case: its
# 944-1200 plateau is real, but exempting the whole entry also protected a 750,000 held flat
# from 1690 to 1790, which left Istanbul the largest city in the world for that century.
CF_KEEP = {
    ("Istanbul-Turkey", 944),  # Constantinople really did hold ~330k across the Macedonian
                               # and Komnenian periods; the drop is the 1204 Crusader sack
    ("Shîrâz-Iran", 1350),     # continuously one of Iran's largest cities; it submitted to
                               # Timur rather than being sacked, and declined only after 1722
    ("Umma-Sumer", -2500),     # 200yr plateau covers Umma's real peak under Lugalzagesi
    ("Bolgary-Russia", 1200),  # two defensible endpoints: pre-Mongol capital, then the
                               # rebuilt Golden Horde city; the drop is its 1430s destruction
}
# Where the run's own starting value is also too late or too high -- the city was already
# finished by then -- end its real data at this year instead of at the run's start.
CF_END = {
    "Gelibolu-Turkey":      999,    # Gallipoli was never a 300k city; the whole plateau is fake
    "Kaifeng-China":       -225,    # Qin drowned Daliang in 225 BC; Kaifeng's real peak is Song
    "Shanghai-China":      -301,    # a fishing village until the Ming; see the Shangqi note
    "Bûr Sa'îd-Egypt":     1149,    # Port Said was founded in 1859 for the Suez Canal
    "Aksum-Ethiopia":       700,    # a small ceremonial town once Red Sea trade was lost
    "T'bilisi-Georgia":    1226,    # sacked by Jalal al-Din the year before the run starts
    "Siracusa-Italy":      -212,    # Marcellus's sack ended Syracuse; the -200 value is post-collapse
    "Kórinthos-Greece":     400,    # Corinth was literally uninhabited at the 100 BC anchor
    "Yamaguchi-Japan":     1551,    # ended by Sue Harukata's coup, before the 1575 anchor
    "Sagaing-Myanmar":     1364,    # Ava was founded across the river in 1364
    "Agadez-Niger":        1550,    # already in decline after Askia Muhammad's 1500 sack
    "Huancavelica-Peru":   1786,    # the Santa Barbara mine collapsed in 1786
    "Krivodol-Bulgaria":   1396,    # 982 inhabitants in 1880; 40k is implausible at any date
}


def strip_carry_forward(S, key):
    """Collapse populstat carry-forward runs in a {year: pop} dict.

    Returns (S, fade_years) where fade_years are the years a floor point should be planted
    at, so the viewer shows the city fading out after its last real datum instead of holding
    a dead value. Leaves S untouched for CF_KEEP entries."""
    if len(S) < 3:
        return S, []
    pts = sorted(S.items())
    fades, drop = [], set()
    i = 0
    while i < len(pts):
        j = i
        while j + 1 < len(pts) and abs(pts[j + 1][1] - pts[i][1]) <= CF_EPS * max(pts[i][1], 1.0):
            j += 1
        if (key, pts[i][0]) in CF_KEEP:
            i = j + 1
            continue
        if pts[j][0] - pts[i][0] >= CF_MIN_SPAN and pts[i][1] >= CF_MIN_VAL:
            end = CF_END.get(key, pts[i][0])          # last year of genuine data
            for y, _ in pts[i:j + 1]:
                if y > end:
                    drop.add(y)
            fades.append(end)
        i = j + 1
    if not drop:
        return S, []
    return {y: v for y, v in S.items() if y not in drop}, fades


def norm(s):
    return "".join(ch for ch in unicodedata.normalize("NFKD", s)
                   if not unicodedata.combining(ch)).lower().strip()


def load_ghsl():
    """Grid-index GHSL urban centres by 1-deg cell for fast nearest/largest lookup.
    Returns (centres, grid) where centres[i] = (lat, lon, {year:pop})."""
    with open(GHSL, encoding="utf-8") as f:
        g = json.load(f)
    centres, grid = [], defaultdict(list)
    for c in g.values():
        co = c.get("coords")
        if not co or len(co) != 2:
            continue
        pop = {int(y): v for y, v in c.get("population", {}).items() if v and v > 0}
        if not pop:
            continue
        lat, lon = co
        i = len(centres)
        centres.append((lat, lon, pop, c.get("name", "")))
        grid[(round(lat), round(lon))].append(i)
    print(f"ghsl centres indexed: {len(centres):,}")
    return centres, grid


def nearest_centre(lat, lon, centres, grid):
    """Index of the urban centre a city most likely belongs to, else None.

    Scored by peak population over distance among centres within GRAFT_RADIUS_KM. Plain
    'largest within the radius' fails in both directions: it made Guangzhou miss its own
    27.5M centre (31.6km away, just past the old 30km cut) and fall through to Huadu at
    882k, and once the radius was widened it made Hong Kong -- which has its own 4.8M centre
    3.3km away -- get swallowed by Shenzhen's larger one 35km off. Weighting by distance
    picks the big distant centre when a city really is inside a sprawling agglomeration, and
    the city's own centre when it has one.

    The exponent on distance is 1, chosen empirically: over the 2,763 entries that ever pass
    200k, 1/d strands 110 of them on a centre less than a fifth their size, against 116 for
    1/d^1.5 and 120 for 1/d^2 -- and 1/d^2 is steep enough to hand Detroit to Windsor, which
    is closer but across an international border."""
    coslat = math.cos(math.radians(lat))
    best, bestscore = None, 0.0
    for dla in (-1, 0, 1):
        for dlo in (-1, 0, 1):
            for i in grid.get((round(lat) + dla, round(lon) + dlo), []):
                gla, glo, gpop = centres[i][:3]
                dx = (glo - lon) * coslat * 111.32
                dy = (gla - lat) * 110.57
                d2 = dx * dx + dy * dy
                if d2 > GRAFT_RADIUS_KM ** 2:
                    continue
                score = max(gpop.values()) / max(math.sqrt(d2), 1.0)
                if score > bestscore:
                    bestscore, best = score, i
    return best


BLEND_LO, BLEND_HI = 1975, 2000     # morph city-proper history -> GHSL agglomeration here

def interp_log(series_sorted, y):
    """log-linear interpolate a sorted [(year,pop)] list at year y; None if outside."""
    if y < series_sorted[0][0] or y > series_sorted[-1][0]:
        return None
    lo, hi = 0, len(series_sorted) - 1
    while hi - lo > 1:
        m = (lo + hi) // 2
        if series_sorted[m][0] <= y: lo = m
        else: hi = m
    (y0, v0), (y1, v1) = series_sorted[lo], series_sorted[hi]
    if y1 == y0:
        return v0
    f = (y - y0) / (y1 - y0)
    return 10 ** (math.log10(v0) + (math.log10(v1) - math.log10(v0)) * f)


def merge_series(S, G):
    """Blend a city-proper history dict S with a GHSL agglomeration dict G:
      < BLEND_LO      : S as-is (deep history, city-proper)
      BLEND_LO..HI    : geometric blend S->G (definition morphs smoothly)
      > BLEND_HI      : G as-is (modern agglomeration, correct magnitude)
    Keeps the seam continuous while giving right modern sizes. G is annual for
    large centres; sparse (e.g. only 2025) for tiny ones -> then it's ~a hold."""
    if not G:
        return S
    s_sorted = sorted(S.items())
    lo = max(BLEND_LO, min(G))
    hi = BLEND_HI if BLEND_HI > lo else lo
    out = {}
    for y, v in S.items():
        if y < lo:
            out[y] = v
    for y in sorted(G):
        if y < lo or y > hi:
            continue
        sv = interp_log(s_sorted, y)
        if sv is None:
            out[y] = G[y]
        else:
            w = (y - lo) / (hi - lo) if hi > lo else 1.0
            out[y] = 10 ** ((1 - w) * math.log10(sv) + w * math.log10(G[y]))
    for y, v in G.items():
        if y > hi:
            out[y] = v
    return out


def dp_simplify(pts, releps):
    """Douglas-Peucker on a (year, population) series in LINEAR space, with deviation
    measured RELATIVE to the local magnitude.

    Why linear+relative (not log): Stadester fills gaps with straight-line (linear)
    interpolation between its real anchors -- confirmed by e.g. Yuzhou, whose absolute
    increment is a dead-constant +438/yr across 2,300 years. In linear space that fill
    is perfectly collinear, so this collapses it back to the two real anchors; the
    viewer's log-linear interpolation then reconnects them GEOMETRICALLY (a plausible
    growth curve instead of the smear). Genuine data (incl. annual GHSL modern series)
    deviates from the straight chord by more than releps, so it survives -- which also
    stops the modern era from being over-simplified into fake 20-year gaps."""
    if len(pts) <= 2:
        return pts[:]
    x0, y0 = pts[0]
    x1, y1 = pts[-1]
    # find point of max deviation from the chord y0..y1 evaluated at its x, relative to size
    dmax, idx = 0.0, 0
    dx = x1 - x0
    for i in range(1, len(pts) - 1):
        x, y = pts[i]
        ychord = y0 + (y1 - y0) * ((x - x0) / dx) if dx else y0
        d = abs(y - ychord) / max(abs(ychord), abs(y), 1.0)
        if d > dmax:
            dmax, idx = d, i
    if dmax > releps:
        left = dp_simplify(pts[:idx + 1], releps)
        right = dp_simplify(pts[idx:], releps)
        return left[:-1] + right
    return [pts[0], pts[-1]]


FADE_GAP = 700          # a control-point gap longer than this reads as "no data for centuries"
FADE_YEARS = 150        # decay-out / rise-in ramp length at each end of such a gap
FADE_FLOOR = 1000       # floor pop held across the gap middle (below MINPOP -> invisible)

def fade_long_gaps(control):
    """Across gaps longer than FADE_GAP years, insert floor points so a city fades OUT after
    the earlier anchor and back IN before the later anchor, instead of the viewer smearing a
    smooth line across centuries of missing data. This is what kills the interpolation smears:
    Yuzhou / Memphis etc. have a single ancient anchor and a modern one with a ~2,300-yr void
    between -- geometric interp still ramps them to 400-700k across the whole medieval period.
    Absence of data across a millennium means 'not a recorded major city', not 'draw a line'.
    Cities with genuine continuity (Baghdad, Song Kaifeng, Constantinople) have dense anchors
    and short gaps, so they are untouched."""
    if len(control) < 2:
        return control
    out = [control[0]]
    for (y0, v0), (y1, v1) in zip(control, control[1:]):
        if y1 - y0 > FADE_GAP and y0 + FADE_YEARS < y1 - FADE_YEARS and max(v0, v1) > FADE_FLOOR:
            out.append([y0 + FADE_YEARS, FADE_FLOOR])
            out.append([y1 - FADE_YEARS, FADE_FLOOR])
        out.append([y1, v1])
    return out


def plant_fades(control, fades):
    """Fade a city out after each year in `fades` -- the last genuinely-recorded year before
    a carry-forward run we deleted. Without this the viewer just draws a long geometric line
    from the real estimate down to the first census, which reads as a slow decline when what
    actually happened is 'no record for four centuries'. Mirrors fade_long_gaps' shape, but
    is triggered by the strip rather than by raw gap length (Vijayanagara's gap is only 341
    years and Kamakura's 696, both under FADE_GAP)."""
    if not fades:
        return control
    out = [control[0]]
    for (y0, v0), (y1, v1) in zip(control, control[1:]):
        if any(y0 <= f < y1 for f in fades) and v0 > FADE_FLOOR \
                and y0 + FADE_YEARS < y1 - FADE_YEARS:
            out.append([y0 + FADE_YEARS, FADE_FLOOR])
            out.append([y1 - FADE_YEARS, FADE_FLOOR])
        out.append([y1, v1])
    return out


def sig3(x):
    """round to 3 significant figures, integer-valued."""
    if x <= 0:
        return 0
    d = math.floor(math.log10(x))
    p = 10 ** (d - 2)
    return int(round(x / p) * p)


def load_coord_fixes():
    """Repairs for entries the source's geocoder dumped on a country centroid.

    Two kinds, both keyed by stadester key. An entry with lat/lon is relocated to a real WUP
    urban centroid (never a hand-typed coordinate). An entry with "drop": true is a district
    or a since-merged town whose parent city is already in the dataset -- Lyubertsy and
    Mytishchi inside Moscow, Tottenham inside London, Fuse inside Higashiosaka. There is no
    correct standalone location for those, and leaving them stranded paints a phantom city in
    the middle of Siberia, so they go the way of Kensington and Chelsea."""
    if not os.path.exists(COORDFIX):
        print(f"note: {COORDFIX} missing -- no coordinate repairs applied")
        return {}, set()
    with open(COORDFIX, encoding="utf-8") as f:
        fixes = json.load(f)
    moves = {k: (v["lat"], v["lon"]) for k, v in fixes.items() if not v.get("drop")}
    drops = {k for k, v in fixes.items() if v.get("drop")}
    return moves, drops


def main():
    with open(SRC, encoding="utf-8") as f:
        raw = json.load(f)
    print(f"loaded {len(raw):,} entries")
    centres, grid = load_ghsl()
    coord_fix, coord_drop = load_coord_fixes()
    print(f"coordinate repairs available: {len(coord_fix):,} moves, {len(coord_drop):,} drops")

    dropped_variant = dropped_nocoord = dropped_empty = dropped_small = dropped_dup = 0
    clipped_newworld = 0
    n_coordfix = n_clipped = n_carryfwd = n_variant = 0

    # --- pass 1: parse clean city records + their historical population dict ---
    # dedup exact duplicate entries: same name within ~11km (keeps richer series)
    by_id = {}
    for key, c in raw.items():
        if key in DROP_KEYS or key in coord_drop:
            dropped_variant += 1
            continue
        name = c.get("name", "")
        low = name.lower()
        if any(m in low for m in DROP_MARKERS):
            dropped_variant += 1
            continue
        co = c.get("coords")
        if key in coord_fix:                       # geocoder fallback -> repaired location
            co = list(coord_fix[key])
            n_coordfix += 1
        if not co or len(co) != 2 or (co[0] == 0 and co[1] == 0):
            dropped_nocoord += 1
            continue
        new_world = co[1] < NEW_WORLD_LON
        S = {}
        for ystr, v in c.get("population", {}).items():
            try:
                y = int(ystr)
            except ValueError:
                continue
            if y < YEAR_LO or y > YEAR_HI or not v or v <= 0:
                continue
            v = float(v)
            if new_world:
                if y < NW_RAMP_START:                  # deep-antiquity phantom -> gone
                    clipped_newworld += 1
                    continue
                cap = nw_cap(y)
                if cap is not None and v > cap:        # ramp: suppress implausibly-large early value
                    v = cap
                    clipped_newworld += 1
            S[y] = v
        # North-American placeholder data (see NA_* above). No Anglo/French city existed before
        # ~1700, so (a) drop everything pre-1700, then (b) strip a long flat leading run -- the
        # placeholders are either fabricated ramps (Boston-MA-type, gone via (a)) or a constant
        # held for a century+ that spills past 1700 (Cincinnati holds 10k to 1810; St Louis 15k).
        if co[0] > NA_LAT_MIN and NA_LON[0] < co[1] < NA_LON[1]:
            for y in [y for y in S if y < NA_CLIP_BEFORE]:
                del S[y]; clipped_newworld += 1
            pts = sorted(S.items())
            k = 0
            while k < len(pts) and pts[0][1] and abs(pts[k][1] / pts[0][1] - 1) < 0.05:
                k += 1
            if k >= 2 and pts[k - 1][0] - pts[0][0] > 80:      # flat for a century+ = placeholder
                for y, _ in pts[:k]:
                    del S[y]; clipped_newworld += 1
        for y in DROP_YEARS.get(key, ()):        # individual bad points
            S.pop(y, None)
        # base entry mixes two definitions year-by-year -> take a consistent variant's series
        # for the years it covers, keeping the base entry's deeper history outside that range
        vkey = PREFER_VARIANT.get(key)
        if vkey and vkey in raw:
            V = {}
            for ystr, v in (raw[vkey].get("population") or {}).items():
                try:
                    vy = int(ystr)
                except ValueError:
                    continue
                if YEAR_LO <= vy <= YEAR_HI and v and float(v) > 0:
                    V[vy] = float(v)
            if V:
                lo, hi = min(V), max(V)
                S = {y: v for y, v in S.items() if y < lo or y > hi}
                S.update(V)
                n_variant += 1
        # entry mixes two places' histories (Danapur holding Pataliputra's figures)
        clip = CLIP_BEFORE.get(key)
        if clip is not None:
            for y in [y for y in S if y < clip]:
                del S[y]
                n_clipped += 1
        # populstat carry-forward: keep the real estimate, drop its centuries of repetition
        S, cf_fades = strip_carry_forward(S, key)
        if cf_fades:
            n_carryfwd += 1
        if not S:
            dropped_empty += 1
            continue
        # Antiquity-emerging New World cities (Teotihuacan etc.) get a sub-visibility seed so
        # they fade in over ~300 yrs instead of popping in. Modern emergents (colonial cities)
        # don't -- seeding them would paint a spurious pre-founding presence.
        if new_world:
            y0 = min(S)
            if y0 <= NW_RAMP_FULL:
                seed = max(NW_RAMP_START, y0 - 300)
                if seed < y0:
                    S[seed] = PEAK_FLOOR * 0.4   # below peak+display floors: invisible, never resurrects
        disp = RENAME.get(key, name)
        rec = {"n": disp, "la": co[0], "lo": co[1], "t": c.get("type", "?"),
               "S": S, "peak": max(S.values()), "fades": cf_fades}
        idk = (norm(disp), round(co[0], 1), round(co[1], 1))
        old = by_id.get(idk)
        if old is None:
            by_id[idk] = rec
        else:                                   # keep the richer duplicate
            dropped_dup += 1
            if (len(S), rec["peak"]) > (len(old["S"]), old["peak"]):
                by_id[idk] = rec
    recs = list(by_id.values())

    # --- assign each city to its dominant GHSL centre; one principal per centre ---
    # Rank by (name matches the centre, peak). The name test settles the case where a city is
    # genuinely inside a bigger neighbour's polygon and would otherwise win it on size alone:
    # WUP has no Dongguan centre (it falls inside Shenzhen's), and Dongguan's 1.74M history
    # outweighs Shenzhen's own, so without this Dongguan takes Shenzhen's modern tail and
    # Shenzhen is left ending in 2001.
    principal = {}   # centre idx -> rec index that gets the graft
    for ri, r in enumerate(recs):
        ci = nearest_centre(r["la"], r["lo"], centres, grid)
        r["centre"] = ci
        if ci is None:
            continue
        cname = norm(centres[ci][3])
        rname = norm(r["n"])
        rank = (bool(cname) and (rname in cname or cname in rname), r["peak"])
        if ci not in principal or rank > principal[ci][0]:
            principal[ci] = (rank, ri)
    principal = {ci: ri for ci, (rank, ri) in principal.items()}

    # --- pass 2: merge modern tail onto principals, simplify, emit ---
    cities = []
    grafted = 0
    ymin, ymax = 9999, -9999
    total_pts_before = total_pts_after = 0
    for ri, r in enumerate(recs):
        S = r["S"]
        total_pts_before += len(S)
        ci = r["centre"]
        gpop = centres[ci][2] if ci is not None else None
        if (ci is not None and principal.get(ci) == ri
                and r["peak"] >= GRAFT_MIN_FRAC * max(gpop.values())):
            merged = merge_series(S, gpop)
            if max(merged) > max(S):
                grafted += 1
        else:
            merged = S
        pts = sorted(merged.items())

        if max(v for _, v in pts) < PEAK_FLOOR:
            dropped_small += 1
            continue

        simp = dp_simplify(pts, DP_EPS_REL)
        control = plant_fades([[y, sig3(v)] for y, v in simp], r.get("fades"))
        control = fade_long_gaps(control)
        total_pts_after += len(control)

        ymin = min(ymin, control[0][0])
        ymax = max(ymax, control[-1][0])
        cities.append({
            "n": r["n"],
            "la": round(r["la"], 4),
            "lo": round(r["lo"], 4),
            "t": r["t"],
            "p": control,
        })

    cities.sort(key=lambda c: -max(p[1] for p in c["p"]))  # biggest first (draw order)
    out = {"yearMin": ymin, "yearMax": ymax, "cities": cities}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, separators=(",", ":"))

    sz = os.path.getsize(OUT) / 1e6
    print(f"kept {len(cities):,} cities  |  year range {ymin}..{ymax}  |  grafted modern tail onto {grafted:,}")
    print(f"dropped: {dropped_variant:,} metro-variants, {dropped_dup:,} duplicates, "
          f"{dropped_nocoord:,} no-coord, {dropped_empty:,} empty, "
          f"{dropped_small:,} below peak floor {PEAK_FLOOR}")
    print(f"ramped/clipped {clipped_newworld:,} New World antiquity points (cap ramp {NW_RAMP_START}..{NW_RAMP_FULL})")
    print(f"repaired {n_coordfix:,} geocoder-fallback coordinates; "
          f"stripped carry-forward from {n_carryfwd:,} cities; clipped {n_clipped:,} pre-merge points; "
          f"took {n_variant:,} metro-variant series")
    print(f"control points: {total_pts_before:,} -> {total_pts_after:,} "
          f"({total_pts_after/max(total_pts_before,1)*100:.1f}%), avg {total_pts_after/len(cities):.1f}/city")
    print(f"wrote {OUT}  ({sz:.2f} MB)")


if __name__ == "__main__":
    main()
