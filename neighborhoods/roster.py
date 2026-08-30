"""Propose additions to data/cities_seed.csv from Wikidata.

The seed roster is hand-maintained and stays that way (spec 6, data contract): this tool
never writes cities_seed.csv. It writes data/roster_proposal.csv, which is meant to be
read, edited, and pasted in.

    python roster.py --fetch              refresh data/city_candidates.json from WDQS
    python roster.py --propose 250        choose that many additions, write the proposal
    python roster.py --propose 250 --cap 5 --min-pop 600000

Why each filter exists is documented at the constant it belongs to.
"""
import argparse, csv, json, math, pathlib, re, sys, unicodedata
import urllib.parse, urllib.request

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = pathlib.Path(__file__).parent
DATA = HERE / "data"
SEED = DATA / "cities_seed.csv"
CANDIDATES = DATA / "city_candidates.json"
PROPOSAL = DATA / "roster_proposal.csv"
AREAS = DATA / "city_area.json"
UA = "neighborhoods-map/0.1 (anitaxinchen@gmail.com)"

# Wikidata classes that mean "a group of cities", not "a city". They reach the Q515
# subclass tree and would otherwise seed Greater Mexico City NEXT TO Mexico City, with
# two overlapping boxes fighting over the same neighbourhoods.
AGGLOMERATION = {
    "Q1907114",   # metropolitan area
    "Q22865",     # urban agglomeration
    "Q159313",    # urban agglomeration of Argentina
    "Q1768043",   # metropolitan statistical area (US)
    "Q1062177",   # region of the Philippines
}

# The same city is often in Wikidata twice: once as a PLACE and once as the ADMINISTRATIVE
# unit that contains it - Cape Town / City of Cape Town, Belgrade / City of Belgrade,
# Pretoria / City of Tshwane Metropolitan Municipality. Spec 4.4a is explicit that the
# roster wants the place item, because that is the assumption the rest of the pipeline is
# built on. This pattern is only the SUSPICION; see drop_admin_twins for the test.
# Two patterns, because the names differ in how sure they are.
#
# ADMIN_ONLY is never a city's own name - no city is called "... Metropolitan
# Municipality" - so it is dropped on sight. This is the pattern that has to catch
# eThekwini and Ekurhuleni, whose place twins are called Durban and Germiston: no textual
# test relates those, so nothing softer would work.
# `metropol` is the STEM on purpose. Spelling out `metropolitan|metropolis` let "Bordeaux
# Métropole" through, and the next spelling would have been `metropolitana`. The stem plus
# deaccent() covers the family in every language the roster meets.
ADMIN_ONLY = re.compile(
    r"^greater\s|^grand\s|^metro\s|metropol|agglomerat|conurbation"
    r"|municipality|urban area",
    re.I)
# ADMIN_MAYBE is a real prefix on real city items. "City of Valencia" is the ONLY record
# Wikidata offers with a population for Valencia, Spain, so dropping the form outright
# loses the city. It is dropped only when its twin is demonstrably present; see
# drop_admin_twins.
ADMIN_MAYBE = re.compile(r"^city of\s", re.I)

# Checked by hand off the review list below and thrown out. Kept here rather than fixed
# by a cleverer filter because each is wrong for its own reason, and a rule general
# enough to catch all three would catch real cities too.
REJECTED = {
    "Q2621587": "Waqooyi-Bari is a federal member state of Somalia, not a city "
                "(it outranks Mogadishu on population, so it wins Somalia's slot)",
    "Q33271": "Dong Nai is a Vietnamese province; Wikidata's description calls it a "
              "centrally-governed city, and its 4.4M is province-scale",
    "Q188011": "Santo Domingo, Chile is a town of ~10k. Its P1082 of 1,029,110 is a bad "
               "Wikidata value, and MAX() over the statements picks exactly that.",
    # These two are the same failure and worth naming, because no statement-selection
    # rule can fix them: the item IS the city, and the number on it is simply wrong.
    # Veracruz city is ~600k and carries a 2020 P1082 of 8,062,579, which is the STATE;
    # Masvingo is ~90k and carries its province's 1,638,528 as its only statement. Taking
    # the latest-dated or best-ranked value picks the wrong number just as happily.
    "Q173270": "Veracruz's 8,062,579 is the state of Veracruz, not the city (~600k)",
    "Q601142": "Masvingo's 1,638,528 is Masvingo Province, not the city (~90k)",
}

# Classes that assert "this is a settlement". Used only to flag picks for review, never
# to filter: two thirds of perfectly good cities carry none of them directly, so a filter
# would take out more than it saved. What it does catch is the opposite error - Somalia's
# top "city" by population is Waqooyi-Bari, a federal member state.
SETTLEMENT = {"Q515", "Q1549591", "Q51929311", "Q108178728", "Q7930989", "Q200250",
              "Q2264924", "Q748149", "Q3184121", "Q5119", "Q174844"}

QUERY = """
SELECT ?c ?cLabel ?iso (MAX(?p) AS ?pop) (SAMPLE(?la) AS ?lat) (SAMPLE(?lo) AS ?lon)
       (GROUP_CONCAT(DISTINCT ?kid; separator=",") AS ?classes)
WHERE {
  ?c wdt:P31/wdt:P279* wd:Q515 ; wdt:P1082 ?p ; wdt:P17 ?country ; wdt:P31 ?k .
  BIND(STRAFTER(STR(?k), "entity/") AS ?kid)
  ?country wdt:P297 ?iso .
  ?c p:P625/psv:P625 ?co .
  ?co wikibase:geoLatitude ?la ; wikibase:geoLongitude ?lo .
  FILTER(?p >= %d)
  FILTER NOT EXISTS { ?c wdt:P576 ?dissolved }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en" }
}
GROUP BY ?c ?cLabel ?iso
"""


def haversine_km(a, b):
    lon1, lat1, lon2, lat2 = map(math.radians, (a["lon"], a["lat"], b["lon"], b["lat"]))
    h = (math.sin((lat2 - lat1) / 2) ** 2
         + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2)
    return 2 * 6371.0088 * math.asin(math.sqrt(h))


def overlaps(new, accepted):
    """The already-seeded city whose query box swallows this candidate's centre, if any.

    Not an arbitrary spacing rule. fetch_osm assigns every candidate unit to its NEAREST
    seed city, so two overlapping boxes do not give you two cities - they give you one
    city's neighbourhoods split arbitrarily down the middle. Soweto sits 20 km from
    Johannesburg and Kawasaki 18 km from Tokyo; both are inside, and both would split.
    """
    for a in accepted:
        if haversine_km(new, a) < a["radiusKm"]:
            return a
    return None


def deaccent(s):
    """Strip diacritics so the administrative-name patterns are not defeated by them.

    "Bordeaux Métropole" reached the roster because `metropolis` does not match
    `Métropole`, and it is exactly the kind of item §1.6 exists to exclude — an
    agglomeration of 28 communes wearing a city's name. Matching a deaccented copy costs
    nothing and closes the whole class, not just the French spelling.
    """
    return "".join(c for c in unicodedata.normalize("NFKD", s)
                   if not unicodedata.combining(c))


def drop_admin_twins(cand, seed):
    """-> the QIDs of administrative duplicates of a city already in the pool.

    ADMIN_ONLY names go without further argument. For the ambiguous ADMIN_MAYBE prefix,
    what makes a name a *twin* is that the place it duplicates is also here: "City of
    Belgrade" sits on top of "Belgrade", while "City of Valencia" sits 7,000 km from the
    only other "Valencia" in the pool and is therefore not a duplicate of anything.

    So the test is containment plus proximity - another record within TWIN_KM whose name
    appears inside this one.
    """
    TWIN_KM = 25
    others = cand + seed
    out = set()
    for x in cand:
        if ADMIN_ONLY.search(deaccent(x["name"])):
            out.add(x["qid"])
            continue
        if not ADMIN_MAYBE.search(deaccent(x["name"])):
            continue
        low = x["name"].lower()
        for y in others:
            if y["qid"] == x["qid"] or len(y["name"]) < 4:
                continue
            if y["name"].lower() in low and haversine_km(x, y) < TWIN_KM:
                out.add(x["qid"])
                break
    return out


def radius_for(pop):
    """Half-width of the Overpass box. Read off the existing hand-set roster rather than
    invented: London 8.9M->25, Berlin 3.7M->20, Paris 2.1M->15, Tel Aviv 470k->12."""
    for floor, km in ((10_000_000, 30), (6_000_000, 25), (3_000_000, 20),
                      (1_500_000, 15), (700_000, 13)):
        if pop >= floor:
            return km
    return 12


def fetch(floor):
    """MAX(P1082) rather than the value carrying the latest P585 qualifier: the "latest"
    form needs a correlated NOT EXISTS that times WDQS out, and it is the same number for
    almost every city. This is a denominator for a ratio test (spec 3), not a published
    figure.

    `wdt:` is the truthy path, so the MAX is already taken over best-rank statements
    only - where an item marks one population preferred (Jeddah has seven, one preferred)
    that is the single value considered. What neither ranking nor dates can fix is an
    item whose number is just wrong; see REJECTED.
    """
    url = "https://query.wikidata.org/sparql?" + urllib.parse.urlencode(
        {"query": QUERY % floor, "format": "json"})
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    rows = json.loads(urllib.request.urlopen(req, timeout=300).read())["results"]["bindings"]
    out = {}
    for r in rows:
        qid = r["c"]["value"].rsplit("/", 1)[-1]
        out[qid] = {"qid": qid, "name": r["cLabel"]["value"],
                    "country": r["iso"]["value"],
                    "lat": round(float(r["lat"]["value"]), 4),
                    "lon": round(float(r["lon"]["value"]), 4),
                    "pop": int(float(r["pop"]["value"])),
                    "classes": r["classes"]["value"].split(",")}
    CANDIDATES.write_text(
        json.dumps(sorted(out.values(), key=lambda x: -x["pop"]), ensure_ascii=False, indent=1),
        encoding="utf-8")
    print(f"wrote {CANDIDATES} - {len(out)} cities with population >= {floor:,}")


# Left in cities_seed.csv by whatever pastes a proposal in. Its only job here is to warn:
# --propose counts every row in the seed as already-covered, so proposing against a file
# that already holds a previous proposal quietly asks for the leftovers instead of the
# roster, and 245 additions come back as 68.
PASTED_MARK = "--- Added from Wikidata by roster.py"


# P2046 is a QUANTITY, and quantities carry units. Asking for the bare `wdt:P2046` number
# and taking MAX over it is silently wrong: Harare's area is stored in square metres, so
# the raw values are 960,600,000 against another city's 1,572, and MAX picks whichever
# statement happened to use the smallest unit. Every value must be converted BEFORE it can
# be compared, so the unit comes back with it.
AREA_QUERY = """
SELECT ?c ?a ?u WHERE {
  VALUES ?c { %s }
  ?c p:P2046 ?st .
  ?st a wikibase:BestRank .
  ?st psv:P2046 ?v .
  ?v wikibase:quantityAmount ?a ; wikibase:quantityUnit ?u .
}
"""

# Wikidata unit QID -> square kilometres.
AREA_UNITS = {
    "Q712226": 1.0,          # square kilometre
    "Q25343": 1e-6,          # square metre
    "Q35852": 0.01,          # hectare
    "Q232291": 2.589988,     # square mile
    "Q2489629": 0.001,       # dunam
    "Q81292": 0.00404686,    # acre
}


def fetch_areas(qids):
    """QID -> square kilometres, from Wikidata P2046. Written to data/city_area.json.

    This is the cheap stand-in for a city outline. The division estimate's failure without
    a boundary (§3.1) is a SCALE error - the population is the city proper's while the
    unit count covers the whole query box - and fixing scale needs the city's SIZE, not
    its shape. Measured against the real boundary on the 48 cities that have one, a disc
    of this area reproduces the true core-unit count to a median 1.02x, 41 of 48 within
    1.5x, against a status-quo error reaching 5.5x.

    Batched because a VALUES block of 300 QIDs is one fast query, where 300 lookups by
    label is a timeout.
    """
    out, unknown, batch = {}, {}, 60
    ids = list(qids)
    for i in range(0, len(ids), batch):
        chunk = " ".join(f"wd:{q}" for q in ids[i:i + batch])
        url = "https://query.wikidata.org/sparql?" + urllib.parse.urlencode(
            {"query": AREA_QUERY % chunk, "format": "json"})
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        rows = json.loads(urllib.request.urlopen(req, timeout=180).read())
        for r in rows["results"]["bindings"]:
            qid = r["c"]["value"].rsplit("/", 1)[-1]
            unit = r["u"]["value"].rsplit("/", 1)[-1]
            factor = AREA_UNITS.get(unit)
            if factor is None:
                # Never guess at a unit. A wrong factor is worse than a missing area,
                # because it produces a confident number nobody would think to check.
                unknown[unit] = unknown.get(unit, 0) + 1
                continue
            km2 = float(r["a"]["value"]) * factor
            # Largest of the best-rank statements. `a wikibase:BestRank` already drops the
            # strays that make a naive minimum wrong - Munich carries a 0.86 sq mi
            # alongside its preferred 310.71 km2 - and among what survives, the small
            # readings are sub-areas of the city rather than the city (George Town has
            # 305.77 km2 beside a 109 ha). Once units are converted the max is safe, which
            # it was not before.
            out[qid] = max(km2, out.get(qid, 0.0))
    found = len(out)
    out = {k: round(v, 2) for k, v in out.items()}
    # A city that HAS no area is recorded as null rather than left out. Otherwise the
    # cache can never look complete — 20 of 301 cities have no P2046 at all — and every
    # --lint would refetch the whole roster to rediscover that.
    for q in ids:
        out.setdefault(q, None)
    AREAS.write_text(json.dumps(out, indent=1, sort_keys=True), encoding="utf-8")
    print(f"wrote {AREAS} - area for {found} of {len(ids)} cities")
    if unknown:
        print("  unrecognised area units, skipped: "
              + ", ".join(f"{u} x{n}" for u, n in sorted(unknown.items(), key=lambda kv: -kv[1])))
    return out


def equivalent_radius_km(area_km2):
    """Radius of a disc with the city's area. The whole approximation, in one line."""
    return math.sqrt(area_km2 / math.pi)


# How far `radiusKm` and `pop` may disagree before the row is worth a look. The ratio is
# an AREA ratio, so 4 means the query box covers four times the city the population figure
# describes. Deliberately loose: a box is supposed to overshoot the administrative city
# and reach the built-up area around it - Paris at 6.7x is correct, because its `pop` is
# the 20 arrondissements while the box has to reach the Petite Couronne.
LINT_OVERSHOOT = 8.0
# ...and the other direction, where the box is too small to hold the population's own
# city. Rarer and usually harmless, but it means the roster is describing a metro with a
# box that cannot see it.
LINT_UNDERSHOOT = 0.15

# A city that ends up with fewer kept units than this has nothing to browse and nothing to
# quiz — four dots on a map is not "how this city is divided", it is what OSM happened to
# have. The level rule cannot catch these: MIN_UNITS judges one LEVEL, and a 4-unit level
# is legitimate where it is a city's borough scheme sitting beside richer levels (§3.3,
# New York). What is wrong here is the total, which only the city can see.
MIN_BROWSABLE_UNITS = 15


def lint(areas):
    """Flag seed rows whose `pop` and `radiusKm` describe different places.

    Both feed the division estimate: `pop` is the numerator, and the box decides which
    units are counted underneath it. When they disagree the estimate is wrong by exactly
    that factor - and it is invisible, because each column looks reasonable on its own.

    This is a WARNING, not an error. A city with a usable boundary is largely protected:
    `core%` re-restricts the denominator to the city proper and absorbs the mismatch. The
    row matters when the boundary is missing or rejected (§4.4a), which is not known until
    the fetch has run - so where levels.json exists, that verdict is shown alongside.
    """
    seed = read_seed()
    known = {}
    if (DATA / "levels.json").exists():
        known = {q: s["hasBoundary"]
                 for q, s in json.loads((DATA / "levels.json").read_text("utf-8")).items()}

    rows, missing = [], []
    for c in seed:
        a = areas.get(c["qid"])
        if a is None:
            missing.append(c)
            continue
        r_eq = equivalent_radius_km(a)
        ratio = (c["radiusKm"] / r_eq) ** 2 if r_eq else float("inf")
        rows.append((c, a, r_eq, ratio))

    flagged = [r for r in rows if r[3] >= LINT_OVERSHOOT or r[3] <= LINT_UNDERSHOOT]
    flagged.sort(key=lambda r: -r[3])
    print(f"roster lint: {len(rows)} rows checked, {len(flagged)} flagged "
          f"(box/city area outside {LINT_UNDERSHOOT}x - {LINT_OVERSHOOT}x)\n")
    if flagged:
        print(f"  {'city':<24}{'pop':>12}{'areaKm2':>10}{'r_eq':>7}{'r_box':>7}"
              f"{'ratio':>8}   boundary")
        for c, a, r_eq, ratio in flagged:
            b = known.get(c["qid"])
            state = "-" if b is None else ("ok" if b else "REJECTED - estimate is wrong")
            print(f"  {c['name']:<24}{c['pop']:>12,}{a:>10,.0f}{r_eq:>7.1f}"
                  f"{c['radiusKm']:>7}{ratio:>7.1f}x   {state}")
        print("\n  Fix a row by making the two agree: either a metro population to match\n"
              "  the box, or a smaller box to match the city-proper population.")
    if missing:
        print(f"\n  no P2046 for {len(missing)}: "
              + ", ".join(c["name"] for c in missing[:12]))

    # Only answerable once the city has been built, so it is silent before then.
    bjson = DATA / "base.json"
    if bjson.exists():
        built = json.loads(bjson.read_text("utf-8"))["cities"]
        thin = sorted(((c["n"], c["name"]) for c in built.values()
                       if c["n"] < MIN_BROWSABLE_UNITS))
        print(f"\n  {len(thin)} built cities hold fewer than {MIN_BROWSABLE_UNITS} units - "
              f"too thin to browse or quiz:")
        for n, name in thin:
            print(f"    {name:<26}{n:>4} units")
        if thin:
            print("  Either OSM has nothing at neighbourhood scale there, or the kept level\n"
                  "  is the wrong one. Worth dropping from the roster if it stays this thin.")


_WARNED = False


def read_seed():
    global _WARNED
    raw = SEED.read_text(encoding="utf-8")
    if PASTED_MARK in raw and not _WARNED:
        _WARNED = True
        print("NOTE: cities_seed.csv already contains a pasted proposal. Every one of\n"
              "      those cities counts as seeded, so this run proposes only what they\n"
              "      did not cover. Remove that block first to re-propose the whole set.")
    rows = list(csv.DictReader(
        l for l in raw.splitlines() if not l.startswith("#")))
    for r in rows:
        r["lat"], r["lon"] = float(r["lat"]), float(r["lon"])
        r["pop"], r["radiusKm"] = int(r["pop"]), int(r["radiusKm"])
    return rows


def propose(want, cap, min_pop):
    seed = read_seed()
    accepted = list(seed)
    have = {r["qid"] for r in seed}
    per_country = {}
    for r in seed:
        per_country[r["country"]] = per_country.get(r["country"], 0) + 1

    cand = json.loads(CANDIDATES.read_text(encoding="utf-8"))
    twins = drop_admin_twins(cand, seed)
    pool, skipped = [], {"dup": 0, "agglom": 0, "twin": 0, "small": 0, "rejected": 0}
    for x in cand:
        if x["qid"] in have:
            skipped["dup"] += 1
        elif x["qid"] in REJECTED:
            skipped["rejected"] += 1
        elif AGGLOMERATION & set(x["classes"]):
            skipped["agglom"] += 1
        elif x["qid"] in twins:
            skipped["twin"] += 1
        elif x["pop"] < min_pop:
            skipped["small"] += 1
        else:
            pool.append(x)
    pool.sort(key=lambda x: -x["pop"])

    # Round-robin over countries rather than one pass down the population list. A straight
    # sort spends its first 310 picks on Chinese prefecture cities before it reaches
    # Dublin; the roster feeds a "name the city" quiz, so breadth beats size.
    picked, overlapped, review, taken = [], [], [], set()
    for rank in range(cap):
        for x in pool:
            if len(picked) >= want:
                break
            if x["qid"] in taken or per_country.get(x["country"], 0) > rank:
                continue
            hit = overlaps(x, accepted)
            if hit:
                taken.add(x["qid"])
                overlapped.append((x, hit))
                continue
            taken.add(x["qid"])
            row = {"qid": x["qid"], "name": x["name"], "country": x["country"],
                   "lat": x["lat"], "lon": x["lon"], "pop": x["pop"],
                   "radiusKm": radius_for(x["pop"])}
            if not SETTLEMENT & set(x["classes"]):
                review.append(row)
            picked.append(row)
            accepted.append(row)
            per_country[x["country"]] = per_country.get(x["country"], 0) + 1
        if len(picked) >= want:
            break

    cols = ["qid", "name", "country", "lat", "lon", "pop", "radiusKm"]
    with PROPOSAL.open("w", encoding="utf-8", newline="") as f:
        f.write(f"# Proposed additions to cities_seed.csv - {len(picked)} cities.\n"
                "# Generated by roster.py; loaded by nothing. Review, then paste the rows\n"
                "# you want into cities_seed.csv.\n")
        w = csv.DictWriter(f, cols)
        w.writeheader()
        w.writerows(picked)

    print(f"wrote {PROPOSAL} - {len(picked)} proposed additions "
          f"({len(seed)} seeded already -> {len(seed) + len(picked)} total)")
    print(f"  candidates {len(cand)}; skipped {skipped['dup']} already seeded, "
          f"{skipped['agglom']} agglomerations, {skipped['twin']} administrative twins, "
          f"{skipped['rejected']} hand-rejected, {skipped['small']} under {min_pop:,}")
    print(f"  {len(overlapped)} rejected for overlapping a seeded box, largest first:")
    for x, hit in sorted(overlapped, key=lambda t: -t[0]["pop"])[:15]:
        print(f"    {x['name']:<24} {x['pop']:>10,}  inside {hit['name']} "
              f"({haversine_km(x, hit):.0f} km of {hit['radiusKm']} km)")
    counts = {}
    for r in picked:
        counts[r["country"]] = counts.get(r["country"], 0) + 1
    print(f"  {len(counts)} countries; most-picked " + ", ".join(
        f"{k} {v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1])[:12]))
    if review:
        print(f"  {len(review)} picked with no settlement class - CHECK THESE BY HAND:")
        for r in review:
            print(f"    {r['qid']:<12} {r['name']:<28} {r['country']} {r['pop']:>10,}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--floor", type=int, default=400_000,
                    help="population floor for the WDQS fetch")
    ap.add_argument("--propose", type=int, metavar="N")
    ap.add_argument("--cap", type=int, default=6,
                    help="max cities per country, already-seeded ones counted")
    ap.add_argument("--min-pop", type=int, default=500_000)
    ap.add_argument("--areas", action="store_true",
                    help="fetch P2046 for every seeded city -> data/city_area.json")
    ap.add_argument("--lint", action="store_true",
                    help="flag seed rows whose pop and radiusKm disagree")
    a = ap.parse_args()
    if a.fetch:
        fetch(a.floor)
    if a.propose:
        propose(a.propose, a.cap, a.min_pop)
    if a.areas or a.lint:
        # --lint reuses a cached city_area.json when it has every seeded city, so the
        # common case (edit a row, re-lint) makes no network call at all.
        seeded = [c["qid"] for c in read_seed()]
        areas = json.loads(AREAS.read_text("utf-8")) if AREAS.exists() else {}
        if a.areas or not set(seeded) <= set(areas):
            areas = fetch_areas(seeded)
    if a.lint:
        lint(areas)
    if not any((a.fetch, a.propose, a.areas, a.lint)):
        ap.error("nothing to do - pass --fetch, --propose N, --areas or --lint")


if __name__ == "__main__":
    main()
