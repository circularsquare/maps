"""
OECD Functional Urban Area GDP per capita -> cities in base.json.

562 FUAs across 33 countries, GDP per capita in USD PPP given DIRECTLY. That
"directly" matters: the hard rule for this project is never to divide one
source's GDP by another source's population, because the boundaries differ
(GHS "Guangzhou" is four cities; OECD's Paris FUA is 41% larger than the GHS
urban centre). A ratio taken across definitions is wrong by 2-4x. Taken from
one consistent boundary it is off by 5-25%, which is fine.

Matching is name + country, normalised. An FUA is a metro region, so its figure
is attached to the CITY the FUA is named after — Paris FUA's GDP per capita
goes on Paris. Anything ambiguous is left unmatched rather than guessed.

Output: writes `gdpPc` / `gdpYear` / `gdpSrc` into data/gdp/oecd_matched.json,
which assemble_base.py merges.
"""
import csv
import json
import pathlib
import sys
import unicodedata
from collections import defaultdict

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = pathlib.Path(__file__).parent
GDP = HERE / "data" / "gdp"
OUT = GDP / "oecd_matched.json"

# OECD country prefixes -> the Wikidata country QIDs our roster uses.
ISO2_TO_Q = {}


def norm(s):
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    return "".join(c for c in s.lower() if c.isalnum())


def main():
    codes = json.loads((GDP / "fua_codes.json").read_text(encoding="utf-8"))
    names = {}
    for cl in codes["data"]["codelists"]:
        for c in cl.get("codes", []):
            names[c["id"]] = c.get("name") or c.get("names", {}).get("en")
    print(f"FUA codes: {len(names):,}")

    # Latest per-capita observation per FUA.
    best = {}
    with open(GDP / "oecd_fua.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["UNIT_MEASURE"] != "USD_PPP_PS":
                continue
            try:
                yr, val = int(r["TIME_PERIOD"]), float(r["OBS_VALUE"])
            except ValueError:
                continue
            k = r["REF_AREA"]
            if k not in best or yr > best[k][0]:
                best[k] = (yr, val)
    print(f"FUAs with GDP per capita: {len(best):,}")

    base = json.loads((HERE / "data" / "base.json").read_text(encoding="utf-8"))

    # Alt names matter here. OECD labels FUAs in the LOCAL language --
    # "Helsinki/Helsingfors", "Athina", "Bucuresti", "Goteborg", "Kobenhavn",
    # "Bruxelles/Brussel/Leuven" -- so matching on the English name alone missed
    # 132 of 562. The stage-3 alias cache has these; use the full list, not the
    # 4 kept in base.json.
    alts = {}
    edir = HERE / "cache" / "entities"
    if edir.exists():
        for f in sorted(edir.glob("*.json")):
            for q, r in json.loads(f.read_text(encoding="utf-8")).items():
                alts[q] = [t for _, t in (r.get("alt") or [])]

    # TWO indexes, primary and alias, consulted in that order. Merging them into
    # one made matching WORSE (422 -> 359): aliases collide across cities, so
    # Amsterdam and Rotterdam became "ambiguous" and were dropped. A primary
    # name is much stronger evidence than an alias and must win outright.
    prim, alias = defaultdict(list), defaultdict(list)
    for q, r in base.items():
        if r.get("kind") != "city":
            continue
        n = norm(r.get("name"))
        if n:
            prim[n].append(q)
        for a in alts.get(q, []):
            an = norm(a)
            if an and an != n:
                alias[an].append(q)

    # A Wikipedia article is a good proxy for "this is the primary item".
    # Wikidata often has BOTH a city and its municipality with near-identical
    # populations -- Amsterdam is Q727 (921,468) and Q9899 (917,923) -- so the
    # population rule cannot separate them, but only one is the real article.
    # Lower QID breaks the remaining ties: older item, usually the canonical one.
    def primacy(q):
        return (0 if base[q].get("wiki") else 1,
                int(q[1:]) if q[1:].isdigit() else 10**9)

    # The FUA code prefix is the country (FR062F -> FR), which is the reliable
    # disambiguator WHERE IT WORKS -- note the NL ISO code sits on Q29999
    # "Kingdom of the Netherlands" while Dutch cities use Q55, so this filter
    # silently does nothing for some countries. It narrows, never decides.
    iso_pool = HERE / "cache" / "wikidata_f10000" / "_countries.json"
    q_by_iso = {}
    if iso_pool.exists():
        for cq, code in json.load(open(iso_pool)):
            if not code.startswith("Q"):
                q_by_iso.setdefault(code, cq)
    q_by_iso["EL"] = q_by_iso.get("GR")
    q_by_iso["UK"] = q_by_iso.get("GB")

    def resolve(nm, iso):
        want = q_by_iso.get(iso)
        for part in [p for p in nm.split("(")[0].split("/") if p.strip()]:
            k = norm(part)
            for table in (prim, alias):
                hits = table.get(k, [])
                if want:
                    incountry = [q for q in hits if base[q].get("country") == want]
                    if incountry:
                        hits = incountry
                if len(hits) == 1:
                    return hits[0]
                if len(hits) > 1:
                    hits = sorted(hits, key=lambda q: -(base[q].get("pop") or 0))
                    a = base[hits[0]].get("pop") or 0
                    b = base[hits[1]].get("pop") or 0
                    if a > b * 3:
                        return hits[0]
                    # Same place duplicated (city vs municipality): populations
                    # are close, so decide on primacy instead of guessing.
                    close = [q for q in hits if (base[q].get("pop") or 0) > a * 0.6]
                    ranked = sorted(close, key=primacy)
                    if ranked and primacy(ranked[0]) != primacy(ranked[-1]):
                        return ranked[0]
        return None

    # Country for an FUA comes from its code prefix (FR062F -> FR). Map ISO2 to
    # the country QIDs actually present in base.
    iso_guess = defaultdict(set)
    for q, r in base.items():
        if r.get("country"):
            iso_guess[r["country"]].add(r.get("countryName"))

    matched, unmatched = {}, []
    for code, (yr, val) in best.items():
        nm = names.get(code)
        if not nm:
            unmatched.append((code, None))
            continue
        q = resolve(nm, code[:2])
        if q:
            matched[q] = {"gdpPc": round(val), "gdpYear": yr, "gdpSrc": "OECD FUA"}
        else:
            unmatched.append((code, nm))

    OUT.write_text(json.dumps(matched, ensure_ascii=False, indent=0, sort_keys=True),
                   encoding="utf-8")
    print(f"matched {len(matched):,} / {len(best):,} FUAs -> {OUT.name}")
    print(f"unmatched {len(unmatched):,}; sample:")
    for code, nm in unmatched[:12]:
        print(f"    {code:10s} {nm}")
    print("\nsample matches:")
    for q in list(matched)[:6]:
        print(f"    {base[q]['name'][:26]:28s} ${matched[q]['gdpPc']:>8,} "
              f"({matched[q]['gdpYear']})")


if __name__ == "__main__":
    main()
