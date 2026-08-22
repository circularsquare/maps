"""
cache/wikidata_f*/  ->  data/base.json

Stage between fetching and building. Applies the settlement-type filter, dedupes
across country files (historical states re-list the same cities), and drops the
fields the map does not use.

The type filter is the load-bearing step. base.json rows become MAP POINTS, so a
diocese or a province in here is a visible wrong dot, not merely a bad candidate.
Direct P31 is not enough on its own — in the Netherlands only 14 rows carry
Q486972 while 190 carry Q1852859 ("cadastral populated place", real towns) and
501 carry Q2039348 ("municipality", an administrative division). Only the
subclass closure separates them, so that is fetched once and applied here.

Rows whose types are entirely unknown to the closure are DROPPED but counted, so
a closure that is quietly wrong shows up as a big number rather than as silence.

Usage:
    python assemble_base.py
    python assemble_base.py --keep-untyped   # diagnostic: skip the filter
"""

import argparse
import json
import pathlib
import sys
from collections import Counter

from kinds import classify, EXTRA_SETTLEMENT


def _useful_alt(name, alt):
    """Drop disambiguation forms masquerading as alternative names.

    Wikidata aliases are full of "Boston, MA", "Boston, USA", "City of Boston",
    "Boston MA, United States" -- all just the same name with a qualifier. They
    crowd out the ones worth having ("Beantown", "The Hub", "Puritan City") in a
    picker that only shows a dozen.

    Filtering here rather than in fetch_entities.py so it applies to batches
    already cached, and so a bad rule costs a re-assemble instead of a refetch.
    """
    if not name or not alt:
        return True
    n, a = name.strip().lower(), alt.strip().lower()
    if a == n:
        return False
    for sep in (",", "(", " - ", " – "):
        if a.startswith(n + sep):
            return False
    # "City of Boston", "Boston City", "Municipality of Boston"
    for pre in ("city of ", "town of ", "municipality of ", "borough of ",
                "commune of ", "comune di ", "municipio de ", "ciudad de "):
        if a == pre + n:
            return False
    if a in (n + " city", n + " town", n + " municipality"):
        return False
    return True

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = pathlib.Path(__file__).parent
CACHE = HERE / "cache"
DATA = HERE / "data"
OUT = DATA / "base.json"

KEEP = ("name", "lat", "lon", "pop", "elev", "admin", "admin_name")

# Alt candidates are 12 MB of a 20 MB base.json if all 14 per city are inlined,
# and the browser pays that on every load. Only a few are needed IN base -- they
# feed search ("Bombay" must find Mumbai). The full picker list is fetched per
# city from /api/alts/<qid> when the edit panel opens.
ALT_IN_BASE = 4


def best_pool():
    """Pick the pool to build from.

    A TYPED pool always wins, however many files it has. Choosing purely by file
    count picked the old untyped pool (323 country files) over the new typed one
    (322) — one extra file, and the settlement-type filter silently did nothing.
    Type coverage is the thing that matters; file count only breaks ties.
    """
    best = None
    for p in sorted(CACHE.glob("wikidata*")):
        if not p.is_dir():
            continue
        files = list(p.glob("Q*.json"))
        if not files:
            continue
        # Sample a few files rather than loading the whole pool.
        typed = False
        for f in files[:12]:
            rows = json.loads(f.read_text(encoding="utf-8"))
            if any(r.get("types") for r in rows.values()):
                typed = True
                break
        score = (typed, len(files))
        if best is None or score > best[0]:
            best = (score, p, len(files), typed)
    if not best:
        return None, 0, False
    return best[1], best[2], best[3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep-untyped", action="store_true")
    args = ap.parse_args()

    pool, n_files, typed = best_pool()
    if not pool:
        sys.exit("no fetch cache found — run fetch_wikidata.py first")
    print(f"pool: {pool.name} ({n_files} country files, "
          f"{'typed' if typed else 'UNTYPED'})")

    # Which country file a city came from IS its P17 -- that is how stage 1
    # queried. Recording it here avoids a refetch just to carry the country.
    #
    # A city can appear in several files: historical states (USSR,
    # Austria-Hungary) list the same cities as the modern country. Prefer the
    # file that has a real ISO code, since the QID-only entries are exactly
    # those historical and fringe entities.
    iso_of = dict(json.load(open(pool / "_countries.json"))) if (pool / "_countries.json").exists() else {}
    has_iso = {q for q, code in iso_of.items() if not code.startswith("Q")}

    rows, origin = {}, {}
    for p in pool.glob("Q*.json"):
        if p.stem == "_countries":
            continue
        cq = p.stem
        for q, r in json.loads(p.read_text(encoding="utf-8")).items():
            prev = origin.get(q)
            if prev is None or (cq in has_iso and prev not in has_iso):
                origin[q] = cq
            rows[q] = r
    print(f"rows (deduped by QID): {len(rows):,}")

    tp = CACHE / "settlement_types.json"
    closure = set(json.loads(tp.read_text(encoding="utf-8"))) if tp.exists() else None
    if closure:
        closure |= EXTRA_SETTLEMENT
        print(f"settlement type closure: {len(closure):,} types "
              f"(incl. {len(EXTRA_SETTLEMENT)} hand-added city types)")

    typed = sum(1 for r in rows.values() if r.get("types"))
    if closure and typed and not args.keep_untyped:
        dropped = Counter()
        kept = {}
        for q, r in rows.items():
            ts = r.get("types") or []
            if ts and not any(t in closure for t in ts):
                dropped[r.get("name") or q] = int(r.get("pop") or 0)
                continue
            kept[q] = r
        print(f"type filter: {len(rows):,} -> {len(kept):,} "
              f"({len(rows)-len(kept):,} dropped)")
        for nm, pop in sorted(dropped.items(), key=lambda kv: -kv[1])[:8]:
            print(f"    dropped  {nm[:44]:46s} {pop:>12,}")
        rows = kept
    elif not typed:
        print("type filter: SKIPPED — this pool has no stored types "
              "(pre-dates the P31 fetch); rerun fetch_wikidata.py")

    # Type labels, so the card can say what a point actually IS and the settings
    # panel can offer real categories. Optional: absent on a first run.
    lp = CACHE / "type_labels.json"
    labels = json.loads(lp.read_text(encoding="utf-8")) if lp.exists() else {}

    # Stage 3 output: alt-name candidates and the Wikipedia title. Optional --
    # absent until fetch_entities.py has run, and partial while it is running.
    ents = {}
    edir = CACHE / "entities"
    if edir.exists():
        for f in sorted(edir.glob("*.json")):
            ents.update(json.loads(f.read_text(encoding="utf-8")))
        print(f"entities: {len(ents):,} cities with aliases/wiki")

    # Country reference: name for display, languages as a SEED for curation.
    cp = CACHE / "countries.json"
    countries = json.loads(cp.read_text(encoding="utf-8")) if cp.exists() else {}
    if countries:
        print(f"countries: {len(countries)} with reference data")

    # GDP per capita, taken DIRECTLY from OECD FUA (never derived by dividing
    # one source's GDP by another's population — see the hard rule in TODO.md).
    gp = DATA / "gdp" / "oecd_matched.json"
    gdp = json.loads(gp.read_text(encoding="utf-8")) if gp.exists() else {}
    if gdp:
        print(f"gdp: {len(gdp):,} cities with a dedicated figure")

    # GHS urban centres (stage 2, match_ghs.py). Optional -- absent on a first
    # run, in which case every city keeps ghsConf="none" and the card simply
    # shows no urban-centre section.
    mp = DATA / "ghs_matched.json"
    ghs = json.loads(mp.read_text(encoding="utf-8")) if mp.exists() else {}
    if ghs:
        roles = Counter(m["ghsRole"] for m in ghs.values())
        print(f"ghs: {len(ghs):,} cities attached to an urban centre "
              f"({', '.join(f'{k}={v:,}' for k, v in roles.most_common())})")

    base = {}
    kinds = Counter()
    for q, r in rows.items():
        rec = {k: r.get(k) for k in KEEP if r.get(k) is not None}
        rec["adminName"] = rec.pop("admin_name", None)
        # The centre's own fields (name, pop, area, member list, history) are
        # NOT copied in. The client joins data/ghs_centres.json on `ghs`, the
        # same way it joins countries.json on `country` -- a 24-name member
        # list repeated across all 24 of that blob's cities is pure duplication.
        m = ghs.get(q)
        if m:
            rec.update(m)
        else:
            rec["ghs"] = None
            rec["ghsConf"] = "none"
        cq = origin.get(q)
        if cq:
            rec["country"] = cq
            # Country name and language seeds are NOT inlined per city. They
            # are identical for every city in a country, so duplicating them
            # across 61,866 records added ~4 MB to every page load. The client
            # joins data/countries.json (209 rows) on this QID instead.
        ts = r.get("types") or []
        rec["kind"] = classify(ts)
        kinds[rec["kind"]] += 1
        if labels:
            # Up to three human-readable type names for the card. Showing a few
            # is more honest than picking one and pretending it is definitive.
            names = []
            for t in ts:
                nm = labels.get(t, {}).get("label")
                if nm and nm not in names:
                    names.append(nm)
            rec["typeNames"] = names[:3]
        g = gdp.get(q)
        if g:
            rec.update(g)
        e = ents.get(q)
        if e:
            # altCandidates is a POOL to pick from, not the displayed value.
            # The card shows `altNames`, which only curation sets.
            alts = [x for x in (e.get("alt") or [])
                    if _useful_alt(rec.get("name"), x[1])]
            if alts:
                rec["altCandidates"] = alts[:ALT_IN_BASE]
            if e.get("wiki"):
                rec["wiki"] = e["wiki"]
        base[q] = rec
    print("kinds:", ", ".join(f"{k}={v:,}" for k, v in kinds.most_common()))

    DATA.mkdir(exist_ok=True)
    tmp = OUT.with_suffix(".tmp")
    tmp.write_text(json.dumps(base, ensure_ascii=False, separators=(",", ":")),
                   encoding="utf-8")
    tmp.replace(OUT)

    # Ship the country table alongside, trimmed to what the UI uses.
    if countries:
        # Order language seeds by GLOBAL PROMINENCE, not official-first.
        #
        # Official-first looks right and fails badly on the United States: it has
        # no federal official language, so P37 lists only territorial ones and
        # English appears merely as "used". That ordering suggested
        # Spanish/Carolinian/Chamorro/Hawaiian/Samoan before English for Boston.
        #
        # How many countries list a language is a decent prominence proxy and
        # needs no extra data: English and Spanish appear in dozens, Carolinian
        # in one. Official status is only the tiebreak, which keeps Japanese
        # ahead of Ainu where both are rare.
        prom = Counter()
        for r in countries.values():
            for L in set((r.get("official") or []) + (r.get("used") or [])):
                prom[L] += 1
        slim = {}
        for cq, r in countries.items():
            off = set(r.get("official") or [])
            langs = list(dict.fromkeys((r.get("official") or []) + (r.get("used") or [])))
            langs.sort(key=lambda L: (-prom[L], 0 if L in off else 1, L))
            slim[cq] = {"name": r.get("name"), "langs": langs[:10]}
        (DATA / "countries.json").write_text(
            json.dumps(slim, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8")
        print(f"wrote countries.json: {len(slim)} countries")
    big = sorted(base.items(), key=lambda kv: -(kv[1].get("pop") or 0))[:5]
    print(f"\nwrote {OUT.name}: {len(base):,} cities "
          f"({OUT.stat().st_size/1048576:.1f} MB)")
    print("largest:", ", ".join(f"{v['name']}" for _, v in big))

    # Orphan check. Curation is the irreplaceable half of this project, so an
    # override that no longer resolves to a city must be LOUD. Silently vanishing
    # edits is the same failure class as the cities.json staleness bug — the work
    # is still on disk, but nothing shows it.
    ov_path = DATA / "overrides.json"
    if ov_path.exists():
        ov = json.loads(ov_path.read_text(encoding="utf-8"))
        orphans = [k for k, v in ov.items()
                   if k not in base and not v.get("_created")]
        if orphans:
            print(f"\n!! {len(orphans)} ORPHANED OVERRIDES — edited cities that "
                  f"are not in the new base:")
            for k in orphans[:20]:
                fields = [f for f in ov[k] if not f.startswith("_")]
                print(f"     {k:14s} {', '.join(fields) or '(tombstone)'}")
            print("   They are still in overrides.json and nothing is lost, but "
                  "they will not appear on the map.\n   Likely the settlement-type "
                  "filter dropped them. Check with --keep-untyped.")
        else:
            print(f"overrides: {len(ov)} curated cities, all resolve")


if __name__ == "__main__":
    main()
