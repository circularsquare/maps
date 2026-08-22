"""
Stage 2: attach GHS urban centres to the Wikidata roster.

GHS never DEFINES a city here — the roster is Wikidata, one point per settlement
(see NOTES.md). This stage only decides which GHS urban centre, if any, each
city belongs to, and what that centre can tell us about it: area, density, the
1975-2030 population history, ecoregion, river basin, and the list of places the
centre swallowed.

The thing that makes this tractable is `GC_UCN_LIS_2025`: every urban centre
ships a semicolon-separated list of the places inside it.

    id 10933 "Guangzhou"  43.0M  6454 km2
    LIS = Shenzhen; Guangzhou; Foshan; Dongguan; Jiangmen; Shunde; Zhongshan; ...

So membership is stated outright, by name, and does not have to be inferred from
geometry. That matters because we have centroids and areas but no footprints
(those are in the 1.69 GB GeoPackage), and an "is it inside" test against an
equivalent-radius circle would be a guess. This is not a guess.

Hence THREE roles, which `ghsConf` alone could not express:

    centre   the city IS the centre        -- its name is the centre's main name
    member   the city is INSIDE the centre -- its name is in the centre's list
    near     neither, but close enough to be worth suggesting

`ghsConf` keeps the meaning SCHEMA.md gives it — how much to trust the
attachment — and the two are independent:

    centre + high   "Urban centre: 43.0M over 6,454 km2"
    member + high   "Part of the Guangzhou urban centre - 43.0M, 24 places"
    near   + low    "possibly part of X (not confidently matched)"

Population is checked ONLY for `centre`. A member being far smaller than its
centre is the normal case, not evidence against — Chancheng is 1/50th of the
Guangzhou blob and is still definitely in it. Being named in the list is much
stronger evidence than any population ratio.

Output, two files, joined on the centre id:
    data/ghs_matched.json    qid -> {ghs, ghsConf, ghsRole, ghsDistKm}
    data/ghs_centres.json    id  -> {name, pop, area, members, hist, ...}

The centre table is NOT inlined per city, for the same reason countries.json is
not: a 24-name member list repeated on all 24 of that blob's cities is pure
duplication. assemble_base.py merges the first; the client joins the second.

    python match_ghs.py
"""

import csv
import json
import math
import pathlib
import sys
import unicodedata
from collections import defaultdict

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = pathlib.Path(__file__).parent
DATA = HERE / "data"
CACHE = HERE / "cache"

GENERAL = DATA / "GHS_UCDB_THEME_GENERAL_CHARACTERISTICS_GLOBE_R2024A.csv"
GEOG = DATA / "GHS_UCDB_THEME_GEOGRAPHY_GLOBE_R2024A.csv"
GHSL = DATA / "GHS_UCDB_THEME_GHSL_GLOBE_R2024A.csv"
CLIMATE = DATA / "GHS_UCDB_THEME_CLIMATE_GLOBE_R2024A.csv"
CENTROIDS = DATA / "ucdb_centroids.json"

# Köppen-Geiger classes, in the code order used by CL_KOP_*.
#
# The file ships integers with no legend. This is Beck et al. 2018's standard
# 30-class order, VERIFIED against cities whose class nobody disputes:
# Singapore 1=Af, Cairo 4=BWh, London 15=Cfb, Moscow 26=Dfb, Rome 8=Csa,
# Chicago 25=Dfa, Bangkok 3=Aw, Phoenix 4=BWh, Lima 4=BWh, Vancouver 15=Cfb.
# 10 of 14 exact; the four that differ from Wikipedia (Mumbai Am/Aw, Nairobi
# Cfb/Cwb, Anchorage Dsc/Dfc, Beijing BSk/Dwa) are all genuinely borderline
# cities, not evidence of a wrong offset — a wrong offset would have put
# Singapore in a polar class, not one step along a boundary.
KOPPEN = ["", "Af", "Am", "Aw", "BWh", "BWk", "BSh", "BSk", "Csa", "Csb", "Csc",
          "Cwa", "Cwb", "Cwc", "Cfa", "Cfb", "Cfc", "Dsa", "Dsb", "Dsc", "Dsd",
          "Dwa", "Dwb", "Dwc", "Dwd", "Dfa", "Dfb", "Dfc", "Dfd", "ET", "EF"]

# SSP2-4.5, the middle-of-the-road scenario. The file also carries 1.19, 1.26,
# 3.70, 4.34, 4.60 and 5.85; picking the middle one avoids the card implicitly
# arguing for a best or worst case.
KOPPEN_FUTURE_COL = "CL_KOP_245_2070"

OUT_MATCH = DATA / "ghs_matched.json"
OUT_CENTRES = DATA / "ghs_centres.json"

# Distance rules.
#
# A flat cut cannot work: the median urban centre has an equivalent radius of
# 2.7 km and the largest 45.9 km. Worse, the centroid is POPULATION-WEIGHTED, so
# on a merged blob it sits at the centre of mass rather than on the namesake --
# GHS "Guangzhou" is 44.9 km from Guangzhou itself. A 50 km flat cut therefore
# only barely reaches the city the blob is named after. So the cut scales with
# the blob.
SEARCH_KM = 130.0      # outer bound for the candidate scan
MIN_CUT_KM = 25.0      # floor for a NAME match, so a big blob still reaches it
RADIUS_F = 1.6         # name-match cut = equivalent radius * this
POP_RATIO = 3.0        # both passes, but see the docstring on `member`

# The `near` pass runs BACKWARDS, from centre to city, and this is the whole
# trick of it.
#
# Going forwards -- every city looks for a blob it might be inside -- produced
# 29,833 suggestions at a 25 km cut and 9,931 even after tightening to one blob
# radius. Both are useless: a review queue of ten thousand is not a review
# queue, and tightening far enough to shrink it also dropped Conakry, which is
# the single case NOTES.md names as the one that MUST land in review (GHS calls
# that 2.99M blob "Coyah", after a town of 77k on its edge).
#
# Backwards asks the question that actually has one answer: this urban centre
# has no city claiming to BE it -- which nearby city most likely is? That is at
# most one suggestion per unclaimed centre, each one worth a look. Conakry
# lands because blob 673 is unclaimed, 25.6 km away, and 2.99M against
# Conakry's 1.67M is well inside the population gate.
REVERSE_KM = 50.0      # NOTES.md measured the roster at 50 km; keep that

YEARS = [1975, 1980, 1985, 1990, 1995, 2000, 2005, 2010, 2015, 2020, 2025, 2030]
MAX_MEMBERS = 14       # Tokyo lists 44; a card can show a dozen


def norm(s):
    """Casefold, strip accents, drop everything non-alphanumeric.

    Same normalisation as match_gdp.py. Deliberately aggressive: it is what
    makes "Nizhny Novgorod" meet "Nizhniy Novgorod" and "Sao Paulo" meet
    "Sao Paulo".
    """
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    return "".join(c for c in s.lower() if c.isalnum())


def haversine(lat1, lon1, lat2, lon2):
    p = math.pi / 180
    a = (math.sin((lat2 - lat1) * p / 2) ** 2
         + math.cos(lat1 * p) * math.cos(lat2 * p)
         * math.sin((lon2 - lon1) * p / 2) ** 2)
    return 12742 * math.asin(math.sqrt(min(1.0, a)))


def fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def open_csv(path):
    """Open a GHS theme CSV, whatever it was encoded as.

    THE THEMES DISAGREE. GENERAL_CHARACTERISTICS and GEOGRAPHY (V1-0) are real
    UTF-8; GHSL (V1-2) is cp1252 -- "M\xe9xico" as one byte, which blows up a
    UTF-8 read at the first Mexican row. Sniffing and announcing beats either
    hardcoding per file (the next version bump moves it) or passing
    errors="replace" (which is how you silently ship `Klaip?da`, the exact bug
    the fresh re-export was done to fix).
    """
    raw = path.read_bytes()
    try:
        raw.decode("utf-8")
        enc = "utf-8-sig"
    except UnicodeDecodeError:
        enc = "cp1252"
        print(f"  note: {path.name} is {enc}, not UTF-8")
    return path.open(encoding=enc, newline="")


def load_centres():
    """The 11,422 urban centres, with everything the card might want."""
    centres = {}
    with open_csv(GENERAL) as f:
        for r in csv.DictReader(f):
            i = int(r["ID_UC_G0"])
            lis = [m.strip() for m in (r.get("GC_UCN_LIS_2025") or "").split(";")]
            centres[i] = {
                "name": r.get("GC_UCN_MAI_2025") or "",
                "country": r.get("GC_CNT_GAD_2025") or "",
                "pop": int(fnum(r.get("GC_POP_TOT_2025")) or 0),
                "area": int(fnum(r.get("GC_UCA_KM2_2025")) or 0),
                "members": [m for m in lis if m and m != "-"],
                "capital": (r.get("GC_UCM_CAP") or "").strip() == "1",
            }
    print(f"centres: {len(centres):,}")

    if GEOG.exists():
        n = 0
        with open_csv(GEOG) as f:
            for r in csv.DictReader(f):
                c = centres.get(int(r["ID_UC_G0"]))
                if not c:
                    continue
                elev = fnum(r.get("GE_ELV_AVG_2025"))
                if elev is not None:
                    c["elev"] = round(elev)
                eco = (r.get("GE_ECO_CLA_2025") or "").strip()
                if eco and eco != "-":
                    c["eco"] = eco
                # Major river basin -- one of the "extras" on the card wishlist.
                basin = (r.get("GE_MRB_MAI_2025") or "").strip()
                if basin and basin != "-":
                    c["basin"] = basin
                n += 1
        print(f"geography: {n:,} centres enriched")

    # The 1975-2030 population series. This is the ONLY place it exists -- the
    # general/geography themes carry a single 2025 figure.
    if GHSL.exists():
        n = 0
        cols = [f"GH_POP_TOT_{y}" for y in YEARS]
        with open_csv(GHSL) as f:
            for r in csv.DictReader(f):
                c = centres.get(int(r["ID_UC_G0"]))
                if not c:
                    continue
                hist = [fnum(r.get(k)) for k in cols]
                if any(h is not None for h in hist):
                    c["hist"] = [None if h is None else int(h) for h in hist]
                    n += 1
        print(f"history: {n:,} centres with a 1975-2030 series")
    else:
        print(f"history: SKIPPED -- {GHSL.name} not present, "
              f"population history will be missing")

    # Climate. Per-blob like everything else here, which for climate is far less
    # of a compromise than it is for elevation: annual mean temperature over a
    # 300 km2 blob really is close to uniform. Mountains are the exception, and
    # they are the exception for elevation too.
    #
    # NOTE what this does NOT give: a monthly min/max band. There is no monthly
    # series anywhere in the CLIMATE theme -- only annual bioclim aggregates. The
    # card's "climate band (monthly min-max)" still needs CHELSA.
    if CLIMATE.exists():
        n = 0
        with open_csv(CLIMATE) as f:
            for r in csv.DictReader(f):
                c = centres.get(int(r["ID_UC_G0"]))
                if not c:
                    continue
                t = fnum(r.get("CL_B01_CUR_2010"))       # BIO1 annual mean temp
                p = fnum(r.get("CL_B12_CUR_2010"))       # BIO12 annual precip
                rng = fnum(r.get("CL_B07_CUR_2010"))     # BIO7 annual temp range
                if t is not None:
                    c["tempC"] = round(t, 1)
                if p is not None:
                    c["precipMm"] = round(p)
                if rng is not None:
                    c["tempRange"] = round(rng, 1)
                for col, key in ((("CL_KOP_CUR_2025"), "koppen"),
                                 (KOPPEN_FUTURE_COL, "koppen2070")):
                    v = (r.get(col) or "").strip()
                    if v.isdigit() and 0 < int(v) < len(KOPPEN):
                        c[key] = KOPPEN[int(v)]
                n += 1
        print(f"climate: {n:,} centres with Koppen + annual temp/precip")
    else:
        print(f"climate: SKIPPED -- {CLIMATE.name} not present")

    return centres


def load_alias_index():
    """qid -> set of normalised names (label + every alias).

    Uses the FULL stage-3 alias list, not the 4 kept in base.json. match_gdp.py
    found the same thing: matching on the English label alone missed 132 of 562
    FUAs, because sources name places locally. GHS is no different -- it says
    "Cologne" for Koln, "Vienna" for Wien.
    """
    idx = {}
    edir = CACHE / "entities"
    if not edir.exists():
        print("aliases: none (cache/entities missing)")
        return idx
    for f in sorted(edir.glob("*.json")):
        for q, r in json.loads(f.read_text(encoding="utf-8")).items():
            s = {norm(t) for _, t in (r.get("alt") or [])}
            s.discard("")
            if s:
                idx[q] = s
    print(f"aliases: {len(idx):,} cities")
    return idx


def build_grid(centroids):
    """Bucket centres by whole degree. 11,422 points over ~180 lat bands."""
    grid = defaultdict(list)
    for sid, p in centroids.items():
        grid[(int(math.floor(p["lat"])), int(math.floor(p["lon"])))].append(int(sid))
    return grid


def nearby(grid, lat, lon):
    """Centre ids whose bucket could hold something within SEARCH_KM.

    Longitude degrees shrink with latitude, so the longitude span has to widen
    by 1/cos(lat) or the search silently goes short in the north -- at 60N a
    one-degree box is only 56 km wide. Above 80 degrees just scan every
    longitude; there is almost nothing up there and correctness is cheaper than
    cleverness.
    """
    dlat = int(math.ceil(SEARCH_KM / 111.32)) + 1
    if abs(lat) > 80:
        lons = range(-180, 181)
    else:
        span = SEARCH_KM / (111.32 * max(0.05, math.cos(lat * math.pi / 180)))
        dlon = int(math.ceil(span)) + 1
        lons = range(int(math.floor(lon)) - dlon, int(math.floor(lon)) + dlon + 1)
    out = []
    for la in range(int(math.floor(lat)) - dlat, int(math.floor(lat)) + dlat + 1):
        for lo in lons:
            # Wrap longitude so a city at 179E still sees centres at 179W.
            b = grid.get((la, ((lo + 180) % 360) - 180))
            if b:
                out.extend(b)
    return out


def main():
    for p in (GENERAL, CENTROIDS):
        if not p.exists():
            sys.exit(f"missing required input: {p}")

    centres = load_centres()
    centroids = json.loads(CENTROIDS.read_text(encoding="utf-8"))
    print(f"centroids: {len(centroids):,}")

    base = json.loads((DATA / "base.json").read_text(encoding="utf-8"))
    print(f"roster: {len(base):,} cities")
    alias = load_alias_index()

    # Normalised name lookups, precomputed once per centre.
    for i, c in centres.items():
        c["_main"] = norm(c["name"])
        c["_mem"] = {norm(m) for m in c["members"]} - {""}
        c["_r"] = math.sqrt(max(c["area"], 1) / math.pi)
        c["_cut"] = min(SEARCH_KM, max(MIN_CUT_KM, c["_r"] * RADIUS_F))

    grid = build_grid(centroids)
    cpos = {int(k): (v["lat"], v["lon"]) for k, v in centroids.items()}

    ROLE_RANK = {"centre": 0, "member": 1, "near": 2}
    matched = {}
    stats = defaultdict(int)
    n = len(base)

    for done, (q, rec) in enumerate(base.items(), 1):
        if done % 5000 == 0:
            print(f"  matching {done:,}/{n:,} ...", flush=True)
        lat, lon = rec.get("lat"), rec.get("lon")
        if lat is None or lon is None:
            continue
        names = {norm(rec.get("name"))} | alias.get(q, set())
        names.discard("")

        best = None
        for sid in nearby(grid, lat, lon):
            c = centres.get(sid)
            if c is None:
                continue
            clat, clon = cpos[sid]
            d = haversine(lat, lon, clat, clon)
            if d > c["_cut"]:
                continue
            if c["_main"] and c["_main"] in names:
                role = "centre"
            elif c["_mem"] & names:
                role = "member"
            else:
                continue        # no name evidence -- left to the reverse pass
            # Name identity beats proximity: a city that IS a centre should not
            # be handed to a closer blob it merely sits beside.
            key = (ROLE_RANK[role], d)
            if best is None or key < best[0]:
                best = (key, sid, role, d)

        if best is None:
            continue

        _, sid, role, d = best
        c = centres[sid]
        if role == "centre":
            p, gp = rec.get("pop") or 0, c["pop"] or 0
            ok = gp > 0 and p > 0 and (1 / POP_RATIO) <= p / gp <= POP_RATIO
            conf = "high" if ok else "low"
            if not ok:
                stats["centre_pop_fail"] += 1
        else:
            conf = "high"

        matched[q] = {"ghs": sid, "ghsConf": conf, "ghsRole": role,
                      "ghsDistKm": round(d, 1)}

    # A GHS urban centre is ONE place, so at most one city can BE it. Without
    # this, Wikidata's city/municipality duplicates (Amsterdam is Q727 AND
    # Q9899, populations within 0.5%) both come out as `centre + high` and the
    # card would claim the centre twice.
    claims = defaultdict(list)
    for q, m in matched.items():
        if m["ghsRole"] == "centre" and m["ghsConf"] == "high":
            claims[m["ghs"]].append(q)

    def primacy(q):
        r = base[q]
        return (0 if r.get("kind") == "city" else 1,     # a real city, not a metro area
                0 if r.get("wiki") else 1,               # has an article
                -(r.get("pop") or 0),                    # then the bigger one
                int(q[1:]) if q[1:].isdigit() else 10**9)

    demoted = 0
    for sid, qs in claims.items():
        if len(qs) < 2:
            continue
        for q in sorted(qs, key=primacy)[1:]:
            # Still in the centre, just not the item that IS it.
            matched[q]["ghsRole"] = "member"
            matched[q]["ghsConf"] = "high"
            demoted += 1

    # --- reverse pass: one suggestion per UNCLAIMED centre ------------------
    #
    # See the note on REVERSE_KM. Forwards this question has ten thousand junk
    # answers; backwards it has at most one per centre, and each is worth a look.
    #
    # Biggest centres first, so when two blobs want the same city the more
    # significant one gets it. A city already placed by name is never taken.
    unclaimed = [sid for sid in centres
                 if sid not in {m["ghs"] for m in matched.values()
                                if m["ghsRole"] == "centre"}]
    citygrid = defaultdict(list)
    for q, r in base.items():
        if r.get("lat") is None or r.get("lon") is None:
            continue
        citygrid[(int(math.floor(r["lat"])), int(math.floor(r["lon"])))].append(q)

    def cities_near(lat, lon, km):
        dlat = int(math.ceil(km / 111.32)) + 1
        if abs(lat) > 80:
            lons = range(-180, 181)
        else:
            span = km / (111.32 * max(0.05, math.cos(lat * math.pi / 180)))
            dlon = int(math.ceil(span)) + 1
            lons = range(int(math.floor(lon)) - dlon, int(math.floor(lon)) + dlon + 1)
        out = []
        for la in range(int(math.floor(lat)) - dlat, int(math.floor(lat)) + dlat + 1):
            for lo in lons:
                b = citygrid.get((la, ((lo + 180) % 360) - 180))
                if b:
                    out.extend(b)
        return out

    suggested = 0
    for sid in sorted(unclaimed, key=lambda s: -centres[s]["pop"]):
        c = centres[sid]
        if not c["pop"]:
            continue
        clat, clon = cpos[sid]
        cut = max(REVERSE_KM, c["_cut"])
        best = None
        for q in cities_near(clat, clon, cut):
            if q in matched:
                continue            # already placed by name; do not poach it
            r = base[q]
            p = r.get("pop") or 0
            # The population gate is what keeps this honest: a 2.99M blob may
            # only be suggested to a city of comparable size, so villages never
            # get offered a metropolis.
            if not p or not ((1 / POP_RATIO) <= p / c["pop"] <= POP_RATIO):
                continue
            d = haversine(r["lat"], r["lon"], clat, clon)
            if d > cut:
                continue
            key = (0 if r.get("kind") == "city" else 1, d)
            if best is None or key < best[0]:
                best = (key, q, d)
        if best is None:
            continue
        _, q, d = best
        matched[q] = {"ghs": sid, "ghsConf": "low", "ghsRole": "near",
                      "ghsDistKm": round(d, 1)}
        suggested += 1
    print(f"reverse pass: {len(unclaimed):,} centres with no city claiming them, "
          f"{suggested:,} got a suggestion")

    for m in matched.values():
        stats[f"{m['ghsRole']}_{m['ghsConf']}"] += 1
    stats["none"] = len(base) - len(matched)

    OUT_MATCH.write_text(
        json.dumps(matched, ensure_ascii=False, separators=(",", ":"), sort_keys=True),
        encoding="utf-8")

    # Ship only the centres something actually references, and drop the working
    # fields. Everything else here is dead weight in the browser.
    used = {m["ghs"] for m in matched.values()}
    out = {}
    for sid in sorted(used):
        c = centres[sid]
        row = {"name": c["name"], "pop": c["pop"], "area": c["area"]}
        if len(c["members"]) > 1:
            row["members"] = c["members"][:MAX_MEMBERS]
            row["nMembers"] = len(c["members"])
        for k in ("hist", "eco", "basin", "elev", "country",
                  "koppen", "koppen2070", "tempC", "precipMm", "tempRange"):
            if c.get(k) is not None and c.get(k) != "":
                row[k] = c[k]
        if c.get("capital"):
            row["capital"] = True
        out[str(sid)] = row
    OUT_CENTRES.write_text(
        json.dumps(out, ensure_ascii=False, separators=(",", ":"), sort_keys=True),
        encoding="utf-8")

    print(f"\nmatched {len(matched):,} / {n:,} cities "
          f"({100*len(matched)/n:.1f}%) -> {OUT_MATCH.name}")
    print(f"centres referenced: {len(used):,} / {len(centres):,} -> {OUT_CENTRES.name} "
          f"({OUT_CENTRES.stat().st_size/1e6:.1f} MB)")
    print(f"\n{'role + confidence':<24}{'cities':>10}")
    for k in ("centre_high", "centre_low", "member_high", "near_low", "none"):
        print(f"  {k:<22}{stats[k]:>10,}")
    print(f"  {'demoted duplicates':<22}{demoted:>10,}"
          f"   (two items claiming one centre)")
    print(f"  {'centre, pop mismatch':<22}{stats['centre_pop_fail']:>10,}"
          f"   (name agrees, population does not)")

    # Fixed regression check. Every one of these is a case NOTES.md calls out as
    # a known trap, so a rule change that quietly breaks one shows up here
    # rather than in the map three weeks later.
    #
    #   Guangzhou/Shenzhen  the merge blob: one is the centre, one a member
    #   Tokyo/Yokohama      same, and Yokohama is a big city in its own right
    #   Fort Worth          GHS calls it a 102k fragment against a ~950k city
    #   Atlanta             GHS is the dense core only, ~600k against a 6M metro
    #   Conakry/Coyah       the blob is named after the wrong town entirely
    print("\nspot check:")
    watch = ["Q16572", "Q15174", "Q34412", "Q59218",   # the Guangzhou blob
             "Q1490", "Q38283", "Q35765", "Q34600",    # Tokyo / Yokohama, Osaka / Kyoto
             "Q16558", "Q23556", "Q60",                # Fort Worth, Atlanta, NYC
             "Q3733", "Q987", "Q727", "Q376749"]       # Conakry, New Delhi, Amsterdam
    for q in watch:
        r = base.get(q)
        if not r:
            continue
        m = matched.get(q)
        if not m:
            print(f"  {r['name'][:20]:22s} -> no centre")
            continue
        c = centres[m["ghs"]]
        ratio = (r.get("pop") or 0) / c["pop"] if c["pop"] else 0
        print(f"  {r['name'][:20]:22s} -> {c['name'][:18]:20s} "
              f"{m['ghsRole']:<7}{m['ghsConf']:<5} "
              f"{m['ghsDistKm']:>5.1f}km  blob {c['pop']/1e6:>5.1f}M "
              f"x{ratio:.2f}  {len(c['members'])} places")


if __name__ == "__main__":
    main()
