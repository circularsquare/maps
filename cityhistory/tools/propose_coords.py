"""Propose real coordinates for cities the geocoder dumped on a country centroid.

A genuine fallback point holds >=3 entries with DISTINCT names (Makhachkala, Volzhsky,
Dzerzhinsk all at Russia's centroid). A point holding the same name repeatedly is just
an agglomeration variant of one real city -- not a fallback, leave it alone.

Each stadester key ends in a country name. We learn country -> ISO3 empirically: take
that country's entries that are NOT on a fallback point (i.e. correctly geocoded), find
each one's nearest WUP centre, and take the modal ISO3. Then a broken entry can only be
re-homed to a WUP centre in its own country.
"""
import json, math, re, sys, io, unicodedata, difflib
from collections import Counter, defaultdict

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
MIN_STACK  = 3       # >=N distinct names on one point = geocoder fallback
AUTO_SIM   = 0.88    # accept automatically at/above this
REVIEW_SIM = 0.74    # between REVIEW_SIM and AUTO_SIM -> print for manual review


def norm(s):
    s = "".join(ch for ch in unicodedata.normalize("NFKD", s or "")
                if not unicodedata.combining(ch)).lower().strip()
    if "," in s:                                   # "Iskandariyah, Al-" -> "al iskandariyah"
        head, _, tail = s.partition(",")
        s = (tail.strip().strip("-") + " " + head.strip()).strip()
    for ch in "`'’":                               # mojibake'd carons/acutes -> drop
        s = s.replace(ch, "")
    for ch in "-().":
        s = s.replace(ch, " ")
    return " ".join(s.split())


def base(s):
    """name with any parenthetical variant removed: 'London (Greater London)' -> 'london'"""
    return norm(re.sub(r"\(.*", "", s or ""))


def km(alat, alon, blat, blon):
    return math.hypot((blon - alon) * math.cos(math.radians(alat)) * 111.32,
                      (blat - alat) * 110.57)


def peak_of(c):
    vals = []
    for v in (c.get("population") or {}).values():
        try:
            vals.append(float(v))
        except (ValueError, TypeError):
            pass
    return max(vals) if vals else 0.0


def main():
    wup = json.load(open("data/stadester/wup2025.json", encoding="utf-8"))
    centres = []
    for v in wup.values():
        co, pop = v.get("coords"), v.get("population") or {}
        if co and pop:
            centres.append((co[0], co[1], max(pop.values()), v.get("name", ""), v.get("iso3", "")))
    grid = defaultdict(list)
    for i, c in enumerate(centres):
        grid[(round(c[0]), round(c[1]))].append(i)
    by_iso = defaultdict(list)
    for c in centres:
        by_iso[c[4]].append(c)

    def nearest_iso(lat, lon, rad=60):
        best, bestd = None, rad
        for dla in (-1, 0, 1):
            for dlo in (-1, 0, 1):
                for i in grid.get((round(lat) + dla, round(lon) + dlo), []):
                    c = centres[i]
                    d = km(lat, lon, c[0], c[1])
                    if d < bestd:
                        bestd, best = d, c[4]
        return best

    raw = json.load(open("data/stadester/stadester_cities.json", encoding="utf-8"))

    # --- 1. find genuine fallback points: >=3 DISTINCT names stacked on one coord ---
    names_at = defaultdict(set)
    count_at = Counter()
    for c in raw.values():
        co = c.get("coords")
        if co and len(co) == 2:
            pt = (round(co[0], 2), round(co[1], 2))
            names_at[pt].add(base(c.get("name", "")))
            count_at[pt] += 1
    fallback = {p for p, ns in names_at.items() if len(ns) >= MIN_STACK}

    # --- 2. learn country -> ISO3 from the correctly-geocoded entries of that country ---
    votes = defaultdict(Counter)
    for key, c in raw.items():
        co = c.get("coords")
        if not co or len(co) != 2:
            continue
        if (round(co[0], 2), round(co[1], 2)) in fallback:
            continue
        country = c.get("country") or key.rsplit("-", 1)[-1]
        iso = nearest_iso(co[0], co[1])
        if iso:
            votes[country][iso] += 1
    iso_of_country = {k: v.most_common(1)[0][0] for k, v in votes.items()
                      if v.most_common(1)[0][1] >= 3}

    # --- 3. re-home each broken entry within its own country ---
    rows = []
    for key, c in raw.items():
        co = c.get("coords")
        if not co or len(co) != 2:
            continue
        pt = (round(co[0], 2), round(co[1], 2))
        if pt not in fallback:
            continue
        country = c.get("country") or key.rsplit("-", 1)[-1]
        iso = iso_of_country.get(country)
        cands = by_iso.get(iso, []) if iso else []
        best, bestsim, via = None, 0.0, ""
        for nm in [c.get("name", "")] + list(c.get("other_names") or []):
            n = norm(nm)
            if not n:
                continue
            for cand in cands:
                sim = difflib.SequenceMatcher(None, n, norm(cand[3])).ratio()
                if sim > bestsim:
                    bestsim, best, via = sim, cand, nm
        rows.append(dict(key=key, name=c.get("name", ""), country=country, iso=iso,
                         peak=int(peak_of(c)), at=[round(co[0], 4), round(co[1], 4)],
                         stack=count_at[pt], match=(best[3] if best else None),
                         match_pop=(int(best[2]) if best else 0),
                         to=([round(best[0], 4), round(best[1], 4)] if best else None),
                         sim=round(bestsim, 3), via=via,
                         moved_km=(round(km(co[0], co[1], best[0], best[1])) if best else None)))
    rows.sort(key=lambda r: -r["peak"])
    json.dump(rows, open(sys.argv[1], "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    MIN_MOVE = 25   # a real fallback is far from the city; a 3km 'move' means it was fine
    auto = [r for r in rows if r["sim"] >= AUTO_SIM and (r["moved_km"] or 0) >= MIN_MOVE]
    rev  = [r for r in rows if r not in auto and r["sim"] >= REVIEW_SIM and (r["moved_km"] or 0) >= MIN_MOVE]
    no   = [r for r in rows if r not in auto and r not in rev]
    print(f"{len(fallback)} genuine fallback points ({MIN_STACK}+ distinct names) holding {len(rows)} entries")
    print(f"  learned ISO3 for {len(iso_of_country)} countries")
    print(f"  auto  (sim>={AUTO_SIM}): {len(auto)}   review: {len(rev)}   no match: {len(no)}")
    for label, group in (("AUTO-ACCEPT", auto), ("NEEDS REVIEW", rev), ("NO MATCH", no)):
        big = [r for r in group if r["peak"] >= 100000]
        print(f"\n-- {label}, peak>=100k ({len(big)} of {len(group)}) --")
        for r in big[:40]:
            tgt = f"-> {r['match'][:20]:<20} {r['moved_km']:>5}km wup {r['match_pop']:>9,}" if r["match"] else "-> (none)"
            print(f"  {r['peak']:>9,}  {r['name'][:24]:<24} [{r['iso']}] {tgt} sim{r['sim']:.2f}")


main()
