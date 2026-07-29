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
"""
import json, math, os, unicodedata
from collections import defaultdict

SRC = "data/stadester/stadester_cities.json"
GHSL = "data/stadester/wup2025.json"  # UN WUP 2025 agglomerations (annual 1975-2025, GHSL-lineage,
                                      # cleaner+broader than the old ghsl.json; same schema). Run prep_wup.py.
OUT = "data/cities.json"

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
}
PEAK_FLOOR = 2000                   # drop cities that never reach this (runtime cutoff is higher)
GRAFT_RADIUS_KM = 30                # match a city to the largest GHSL urban centre within this
GRAFT_MIN_FRAC = 0.05               # only graft if the principal is >=5% of the centre's size
                                    # (stops a tiny orphan town inheriting a whole megacity)


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
        centres.append((lat, lon, pop))
        grid[(round(lat), round(lon))].append(i)
    print(f"ghsl centres indexed: {len(centres):,}")
    return centres, grid


def nearest_centre(lat, lon, centres, grid):
    """Index of the largest GHSL urban centre within GRAFT_RADIUS_KM, else None.
    'Largest' (not nearest) so a city maps to its dominant agglomeration even when
    GHSL splits the metro into several polygons (e.g. Delhi/NCR)."""
    coslat = math.cos(math.radians(lat))
    best, bestpop = None, -1
    for dla in (-1, 0, 1):
        for dlo in (-1, 0, 1):
            for i in grid.get((round(lat) + dla, round(lon) + dlo), []):
                gla, glo, gpop = centres[i]
                dx = (glo - lon) * coslat * 111.32
                dy = (gla - lat) * 110.57
                if dx * dx + dy * dy > GRAFT_RADIUS_KM ** 2:
                    continue
                peak = max(gpop.values())
                if peak > bestpop:
                    bestpop, best = peak, i
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


def sig3(x):
    """round to 3 significant figures, integer-valued."""
    if x <= 0:
        return 0
    d = math.floor(math.log10(x))
    p = 10 ** (d - 2)
    return int(round(x / p) * p)


def main():
    with open(SRC, encoding="utf-8") as f:
        raw = json.load(f)
    print(f"loaded {len(raw):,} entries")
    centres, grid = load_ghsl()

    dropped_variant = dropped_nocoord = dropped_empty = dropped_small = dropped_dup = 0
    clipped_newworld = 0

    # --- pass 1: parse clean city records + their historical population dict ---
    # dedup exact duplicate entries: same name within ~11km (keeps richer series)
    by_id = {}
    for key, c in raw.items():
        if key in DROP_KEYS:
            dropped_variant += 1
            continue
        name = c.get("name", "")
        low = name.lower()
        if any(m in low for m in DROP_MARKERS):
            dropped_variant += 1
            continue
        co = c.get("coords")
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
               "S": S, "peak": max(S.values())}
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
    principal = {}   # centre idx -> rec index with the largest history (gets the graft)
    for ri, r in enumerate(recs):
        ci = nearest_centre(r["la"], r["lo"], centres, grid)
        r["centre"] = ci
        if ci is not None and (ci not in principal or r["peak"] > recs[principal[ci]]["peak"]):
            principal[ci] = ri

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
        control = fade_long_gaps([[y, sig3(v)] for y, v in simp])
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
    print(f"control points: {total_pts_before:,} -> {total_pts_after:,} "
          f"({total_pts_after/max(total_pts_before,1)*100:.1f}%), avg {total_pts_after/len(cities):.1f}/city")
    print(f"wrote {OUT}  ({sz:.2f} MB)")


if __name__ == "__main__":
    main()
