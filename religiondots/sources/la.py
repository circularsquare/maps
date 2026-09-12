"""Laos — Lao Statistics Bureau, Population and Housing Census 2015, at VILLAGE level.

Reads (or fetches) data/raw/la/ and writes data/normalized/la.csv.

**8,499 villages, 6,481,482 people, six categories.** The finest geography on this map
outside a handful of small-island wards: 763 people per unit, against Nepal's 38,400 and
Cambodia's 622,000. Laos was recorded here as *"blocked -- no subnational religion was ever
published"* (sources.md §9an) on the strength of the 2015 report, whose `Table P2.9` really
is national only. That verdict was wrong about the OFFICE rather than about the report: LSB
published the same census at village level through a different channel, and it has been open
the whole time.

**THE CHANNEL IS `k4d.la`, WHICH IS THE 2008 SOCIO-ECONOMIC ATLAS's DATA PLATFORM STILL
RUNNING.** The atlas (`Socio-Economic Atlas of the Lao PDR`, NCCR North-South and Geographica
Bernensia, 2008) drew a village-level religion dot map from the 2005 census; its `DECIDE
Info` platform became K4D, run by the Ministry of Agriculture and Forestry with Swiss
support, and the ArcGIS server behind it at `gis.cde.unibe.ch` carries 1,365 services of
which about 264 are 2005 and 400 are 2015. Every item says `Data Source: Lao Population and
Housing Census 2015` and names LSB, and every item's licence field says **"With proper
citation of the data sources, the data can be used freely."** No key, no login, `Query`
enabled, geometry included.

**THE SIX CATEGORIES COME OFF FIVE SERVICES AND A RESIDUAL.** Three publish counts
(`distribution_of_buddhists`, `_christians`, `_muslims`), two publish percentages
(`percentage_of_bahais`, `percentage_of_no_religion`), and there is no sixth: the service
called `percentage_of_population_with_other_religion` is misnamed and holds the village's
DOMINANT-religion class as a string (`Buddhist 80-99%`, `Village not dominated by a group of
more than 80%`). So `Others/not stated` is taken as the residual of the other five against
the village's own population, and the arithmetic is what confirms it: it comes to 133,296,
which with Muslim and Baha'i added is 137,020 against the published row's 137,640 -- the
difference being the villages missing from the file entirely, below.

**THE PERCENTAGE SERVICES CARRY EIGHT SIGNIFICANT DIGITS, WHICH TURNS A CONVENIENCE INTO A
CHECK.** `0.70600098%` of 1,983 people is 13.999999, so the integer count is recoverable
exactly rather than apportioned (contrast Cambodia, whose one decimal place costs a rounding
band on every cell). It also means **the percentage and population services can be asserted
to be talking about the same village**: if either were joined to the wrong row the product
would not land within 1e-3 of an integer, and it does on all 8,499 rows of both.

**THE REAL CHECK IS `Table 2.3` OF THE PUBLISHED REPORT, AND IT IS UNUSUALLY SHARP.** That
table gives district count and population for the 18 provinces. Aggregating the villages by
the province prefix of their code reproduces it **to the person in 14 of 18 provinces**, with
148 districts against 148 published, and the four differences are all in the same direction:

    Savannakhet        -7,324        Vientiane Capital  -1,474
    Khammuane          -1,388        Phongsaly            -560

-- 10,746 people, 0.17%, being villages the census enumerated and this file does not carry.
That is a §3.5 gap and it is stated in `countries.py`. Nothing is in excess anywhere, which
is the shape a coverage gap has and not the shape a bad join has.

**TWO VILLAGES SIT IN A DIFFERENT PROVINCE FROM THEIR CODE, AND THEY ARE WHY THE CHECK USES
THE CODE.** `B. Phou pard` (770) and `B. Phou lar` (506) carry Louangnamtha VCODEs and the
file places them in Namor district, Oudomxai; twelve villages in all have moved district
since their code was minted. Table 2.3 counts them where their code says, so aggregating by
the `PCODE` attribute puts Louangnamtha 1,276 short and Oudomxai 1,276 over and aggregating
by the code prefix reconciles both exactly. **The attribute is right about where the village
IS and the code is right about where the report COUNTED it**, so the check uses the code and
the drawn geography uses the polygon, which is `sources/la_geo.py`.

**THE 31.45% IS NOT WHAT ITS ENGLISH LABEL SAYS AND THE TAXONOMY IS WHERE THAT IS ARGUED.**
The service is titled `Percentage of No Religion`; its own Lao subtitle reads *"following
other religions or not following any religion"*; the census report's summary calls it *"no
religion or being animist"*; and the 2008 atlas, on the identical 2005 category, says
outright that *"a more appropriate term for the 'other' category would be Animism"*. See
`taxonomy/la2015.py`, which does not file it on `unaffiliated`.

Usage:
    python sources/la.py --fetch    six services, ~9 MB of JSON, about a minute
    python sources/la.py            normalise from data/raw/la/
"""

import csv
import json
import os
import sys
import time
import urllib.parse
import urllib.request

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "la")
OUT = os.path.join(ROOT, "data", "normalized", "la.csv")

SOURCE_ID = "la_phc_2015"
YEAR = 2015
BASIS = "self_id"

COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count",
           "basis", "year", "source_id", "note"]

SERVER = "https://gis.cde.unibe.ch/gis/rest/services/Decide"
PAGE = 2000                      # the server's own maxRecordCount, asserted in fetch()

# The attribute fields are LSB's census variable codes and are asserted verbatim: a renamed
# field stops the run rather than silently reading a different variable. `kind` is `count`
# where the service publishes people and `pct` where it publishes a percentage of the
# village population.
LAYERS = {
    "pop":       ("laos_2015_total_population", "UNPDeAA01", "count"),
    "buddhist":  ("laos_2015_distribution_of_buddhists", "UNPReAB27", "count"),
    "christian": ("laos_2015_distribution_of_christians", "UNPReAB28", "count"),
    "muslim":    ("laos_2015_distribution_of_muslims", "UNPReAB30", "count"),
    "bahai":     ("laos_2015_percentage_of_bahais", "urpreab36", "pct"),
    "none":      ("laos_2015_percentage_of_no_religion", "URPReAE26", "pct"),
}
GEO_FIELDS = ["VCODE", "PCODE", "PNAME", "DCODE", "DNAME", "VName"]

# The drawn categories, in the order they are written. `Others/not stated` is LSB's own row
# label in Table 3.5 and is the residual here; the other five are published directly.
CATEGORIES = ["Buddhist", "Christian", "Muslim", "Baha'i", "No religion",
              "Others/not stated"]
RESIDUAL = "Others/not stated"
TOTAL_CAT = "Total Population"

EXPECTED_VILLAGES = 8_499
EXPECTED_PROVINCES = 18
EXPECTED_DISTRICTS = 148

# 2015 PHC report, Table 3.5 (p.37), sourced there to Table P2.9 of Appendix 1. National
# only -- this is the entire published religion output of the census and it is what the
# village file is reconciled against. Muslim and Baha'i are inside `Others/not stated`
# here; the village file separates them, so the comparison below adds them back.
NATIONAL_POPULATION = 6_492_228
TABLE_35 = {
    "Buddhist": 4_201_993,
    "Christian": 112_230,
    "No religion": 2_040_365,
    "Others/not stated": 137_640,
}

# 2015 PHC report, Table 2.3 (p.29), sourced there to Table P1.2 of Appendix 1: districts
# and population for the 18 provinces. Keyed by the province code, with the report's own
# romanisation kept -- the GIS file romanises several differently (Luangnamtha/Louangnamtha,
# Xayabury/Xaignabouly) and neither spelling is consulted by anything here.
TABLE_23 = {
    1: ("Vientiane Capital", 9, 820_940), 2: ("Phongsaly", 7, 177_989),
    3: ("Luangnamtha", 5, 175_753), 4: ("Oudomxay", 7, 307_622),
    5: ("Bokeo", 5, 179_243), 6: ("Luangprabang", 12, 431_889),
    7: ("Huaphanh", 10, 289_393), 8: ("Xayabury", 11, 381_376),
    9: ("Xienkhuang", 7, 244_684), 10: ("Vientiane Province", 11, 419_090),
    11: ("Borikhamxay", 7, 273_691), 12: ("Khammuane", 10, 392_052),
    13: ("Savannakhet", 15, 969_697), 14: ("Saravane", 8, 396_942),
    15: ("Sekong", 4, 113_048), 16: ("Champasack", 10, 694_023),
    17: ("Attapeu", 5, 139_628), 18: ("Xaysomboun", 5, 85_168),
}

# The four provinces the file is short in, and by how much. Named rather than tolerated so
# that a fifth one appearing, or one of these changing size, fails the run.
KNOWN_SHORT = {1: 1_474, 2: 560, 12: 1_388, 13: 7_324}

# A percentage times a population must land on an integer. Eight significant digits on a
# village of at most ~30,000 people leaves an error of order 1e-4; 1e-3 is a band around
# that and not a tolerance chosen to pass. See the docstring.
INTEGER_EPS = 1e-3

# **ONE CELL OF 16,998 MISSES, AND IT IS LSB's OWN INCONSISTENCY RATHER THAN THE JOIN.**
# `B. Nongbua` in Salavan carries a Baha'i share of 0.5533597%, which is 7/1265 exactly
# while the population service gives the village 1,244. The recovered count is 7 either
# way, so nothing downstream changes; the cell is named here rather than the band widened,
# because widening it to 0.12 would let a genuinely misjoined row through on any village
# under about 2,000 people. VCODE -> (implied denominator, recovered count).
KNOWN_DENOMINATOR_MISMATCH = {1403002: (1265, 7)}


def _get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read().decode("utf-8"))


def _layer_meta(svc):
    return _get(f"{SERVER}/{svc}/MapServer/0?f=json")


def fetch():
    os.makedirs(RAW, exist_ok=True)
    for key, (svc, field, _) in LAYERS.items():
        dest = os.path.join(RAW, f"{key}.json")
        if os.path.exists(dest) and os.path.getsize(dest) > 100_000:
            print(f"already have {dest}")
            continue

        # Assert the schema before taking a figure off it. A renamed variable code, or a
        # service that has lost `Query`, must stop the run here rather than downstream.
        meta = _layer_meta(svc)
        names = {f["name"] for f in meta.get("fields", [])}
        if field not in names:
            raise SystemExit(f"{svc}: no field {field!r} -- LSB has renamed the variable. "
                             f"Fields are {sorted(names)}")
        if "Query" not in str(meta.get("capabilities", "")):
            raise SystemExit(f"{svc}: capabilities are {meta.get('capabilities')!r}, "
                             "so the attributes can no longer be read")
        if int(meta.get("maxRecordCount", 0)) < PAGE:
            raise SystemExit(f"{svc}: maxRecordCount is {meta.get('maxRecordCount')}, "
                             f"below the {PAGE} this pages at")

        rows, offset = [], 0
        while True:
            q = urllib.parse.urlencode({
                "where": "1=1",
                "outFields": ",".join(GEO_FIELDS + [field]),
                "returnGeometry": "false",
                "resultOffset": offset,
                "resultRecordCount": PAGE,
                "orderByFields": "VCODE",
                "f": "json",
            })
            d = _get(f"{SERVER}/{svc}/MapServer/0/query?{q}")
            if "error" in d:
                raise SystemExit(f"{svc}: {d['error']}")
            feats = d.get("features", [])
            rows += [f["attributes"] for f in feats]
            if len(feats) < PAGE:
                break
            offset += PAGE
            time.sleep(0.2)

        # §5a: a 200 is not a download. A paged query that silently returns nothing is the
        # failure mode here, and it looks exactly like a successful empty answer.
        if len(rows) < 8_000:
            raise SystemExit(f"{svc}: read {len(rows)} rows, expected ~{EXPECTED_VILLAGES}")
        with open(dest, "w", encoding="utf-8") as fh:
            json.dump({"service": svc, "field": field, "rows": rows}, fh)
        print(f"  {svc}: {len(rows):,} villages -> {dest} "
              f"({os.path.getsize(dest):,} bytes)")


def _load(key):
    p = os.path.join(RAW, f"{key}.json")
    if not os.path.exists(p):
        raise SystemExit(f"missing {p} -- run with --fetch first")
    with open(p, encoding="utf-8") as fh:
        d = json.load(fh)
    svc, field, _ = LAYERS[key]
    if d["service"] != svc or d["field"] != field:
        raise SystemExit(f"{p} holds {d['service']}/{d['field']}, expected {svc}/{field}")
    return {r["VCODE"]: r for r in d["rows"]}


def read():
    data = {k: _load(k) for k in LAYERS}

    # --- the village roster must be the same set on all six services. Six separate map
    # --- services could perfectly well be different vintages of the file; this is what
    # --- says they are not, and it has to pass before any figure is combined across them.
    rosters = {k: set(v) for k, v in data.items()}
    base = rosters["pop"]
    for k, s in rosters.items():
        if s != base:
            raise SystemExit(f"the {k!r} service covers {len(s)} villages and the "
                             f"population service {len(base)}; "
                             f"{len(base - s)} missing, {len(s - base)} extra")

    # --- and they must agree on WHERE each village is, not only that it exists ---
    for k, v in data.items():
        if k == "pop":
            continue
        bad = [vc for vc in base
               if any(v[vc][f] != data["pop"][vc][f] for f in ("PCODE", "DCODE", "VName"))]
        if bad:
            raise SystemExit(f"the {k!r} service disagrees with the population service "
                             f"about the province, district or name of {len(bad)} "
                             f"villages, e.g. {bad[:5]}")

    rows, counts, exact = [], {}, []
    for vc in sorted(base):
        a = data["pop"][vc]
        pop = int(a[LAYERS["pop"][1]] or 0)
        rec = {}
        for key, cat in (("buddhist", "Buddhist"), ("christian", "Christian"),
                         ("muslim", "Muslim"), ("bahai", "Baha'i"), ("none", "No religion")):
            _, field, kind = LAYERS[key]
            raw = data[key][vc][field] or 0
            if kind == "count":
                rec[cat] = int(raw)
            else:
                # The recovery IS the join check: see the module docstring.
                v = float(raw) / 100.0 * pop
                rec[cat] = int(round(v))
                exact.append((abs(v - round(v)), vc, cat, float(raw), pop, rec[cat]))
        rec[RESIDUAL] = pop - sum(rec.values())
        counts[vc] = (pop, rec)

        note = (f"level=village; province={a['PNAME']}; district={a['DNAME']}; "
                f"pcode={a['PCODE']}; dcode={a['DCODE']}")
        name = " ".join(str(a["VName"] or "").split())
        for cat in CATEGORIES:
            n = note if cat != RESIDUAL else (
                note + "; residual of the five published categories against the "
                       "village population")
            rows.append({"geo_id": f"LA-{vc}", "geo_level": "village", "geo_name": name,
                         "source_category": cat, "count": rec[cat], "basis": BASIS,
                         "year": YEAR, "source_id": SOURCE_ID, "note": n})
        rows.append({"geo_id": f"LA-{vc}", "geo_level": "village", "geo_name": name,
                     "source_category": TOTAL_CAT, "count": pop, "basis": BASIS,
                     "year": YEAR, "source_id": SOURCE_ID,
                     "note": note + "; universe total, not a religion category"})

    return rows, data, counts, exact


def check(rows, data, counts, exact):
    ok = True
    pop_of = {vc: counts[vc][0] for vc in counts}
    grand = sum(pop_of.values())

    good = len(counts) == EXPECTED_VILLAGES
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} villages        {len(counts):>6,} "
          f"(expected {EXPECTED_VILLAGES:,})")

    prov = {a["PCODE"] for a in data["pop"].values()}
    dist = {a["DCODE"] for a in data["pop"].values()}
    for label, got, want in (("provinces", len(prov), EXPECTED_PROVINCES),
                             ("districts", len(dist), EXPECTED_DISTRICTS)):
        good = got == want
        ok &= good
        print(f"  {'OK ' if good else 'BAD'} {label:<15} {got:>6,} "
              f"(Table 2.3 publishes {want})")

    # --- the integer-recovery identity on the two percentage services ---
    miss = sorted((e for e in exact if e[0] >= INTEGER_EPS), reverse=True)
    clean = [e for e in exact if e[0] < INTEGER_EPS]
    worst = max(e[0] for e in clean) if clean else 0.0
    good = {e[1] for e in miss} == set(KNOWN_DENOMINATOR_MISMATCH)
    ok &= good
    print(f"\n  {'OK ' if good else 'BAD'} percentage x population lands on an integer on "
          f"{len(clean):,} of {len(exact):,} cells\n      of the two percentage services; "
          f"worst residue among them {worst:.2e} (band {INTEGER_EPS:.0e}).")
    print("      This is the join: a percentage read against the wrong village's "
          "population\n      would miss an integer by an arbitrary amount, on almost every "
          "row.")
    for e, vc, cat, pct, pop, got in miss:
        implied = got / (pct / 100.0) if pct else 0.0
        want = KNOWN_DENOMINATOR_MISMATCH.get(vc)
        agree = want is not None and abs(implied - want[0]) < 0.5 and got == want[1]
        ok &= agree
        print(f"      {'OK ' if agree else 'BAD'} LA-{vc} {cat}: {pct:.7f}% of the "
              f"service's own {implied:,.0f} rather than\n          this file's {pop:,}, "
              f"and {got} people either way. LSB's inconsistency, not the join.")

    # --- the residual must never go negative ---
    neg = [(vc, counts[vc][1][RESIDUAL]) for vc in counts if counts[vc][1][RESIDUAL] < 0]
    ok &= not neg
    print(f"\n  {'OK ' if not neg else 'BAD'} the residual is non-negative in all "
          f"{len(counts):,} villages ({len(neg)} negative) {neg[:4]}")
    print("      Five published categories that could sum past a village's own population "
          "would\n      mean the services do not belong to one another; none does.")

    # --- categories sum to the village population, by construction ---
    bad = [vc for vc in counts if sum(counts[vc][1].values()) != counts[vc][0]]
    ok &= not bad
    print(f"  {'OK ' if not bad else 'BAD'} the six categories sum to the village "
          f"population in all {len(counts):,} villages")

    # --- Table 2.3, by the province prefix of the village code (see the docstring) ---
    by_code = {}
    for vc, (p, _) in counts.items():
        by_code[vc // 100_000] = by_code.get(vc // 100_000, 0) + p
    dist_by_code = {}
    for a in data["pop"].values():
        dist_by_code.setdefault(a["VCODE"] // 100_000, set()).add(a["VCODE"] // 1_000)

    print(f"\n  against Table 2.3 of the 2015 report, by the province prefix of the "
          f"village code:")
    print(f"    {'province':<20} {'published':>10} {'villages':>10} {'diff':>8} "
          f"{'districts':>10}")
    n_exact, short = 0, {}
    for pc in sorted(TABLE_23):
        nm, nd, pub = TABLE_23[pc]
        got = by_code.get(pc, 0)
        nd_got = len(dist_by_code.get(pc, ()))
        if got == pub:
            n_exact += 1
        elif got < pub:
            short[pc] = pub - got
        flag = "" if nd_got == nd else f"  BAD want {nd}"
        print(f"    {nm:<20} {pub:>10,} {got:>10,} {got - pub:>8,} {nd_got:>10}{flag}")
        ok &= nd_got == nd
    print(f"    {'TOTAL':<20} {NATIONAL_POPULATION:>10,} {grand:>10,} "
          f"{grand - NATIONAL_POPULATION:>8,} {len(dist):>10}")

    good = n_exact >= 14
    ok &= good
    print(f"\n  {'OK ' if good else 'BAD'} {n_exact} of {EXPECTED_PROVINCES} provinces "
          f"reproduce Table 2.3 TO THE PERSON (14 expected)")
    over = {pc: by_code.get(pc, 0) - TABLE_23[pc][2] for pc in TABLE_23
            if by_code.get(pc, 0) > TABLE_23[pc][2]}
    ok &= not over
    print(f"  {'OK ' if not over else 'BAD'} no province is in EXCESS of the published "
          f"figure ({len(over)} are) {over}")
    good = short == KNOWN_SHORT
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} the four short provinces are the four expected, "
          f"by the expected amounts")
    if not good:
        print(f"      got      {short}")
        print(f"      expected {KNOWN_SHORT}")
    miss = NATIONAL_POPULATION - grand
    print(f"      {miss:,} people ({100.0 * miss / NATIONAL_POPULATION:.2f}%) are in "
          "villages the census\n      enumerated and this file does not carry. §3.5 gap, "
          "stated in countries.py.")

    # --- Table 3.5, the whole published religion output of the census ---
    nat = {c: sum(counts[vc][1][c] for vc in counts) for c in CATEGORIES}
    print(f"\n  against Table 3.5, the entire published religion output of the 2015 "
          f"census:")
    print(f"    {'category':<20} {'published':>10} {'villages':>10} {'ratio':>7}  share")
    for cat in ("Buddhist", "Christian", "No religion"):
        got, pub = nat[cat], TABLE_35[cat]
        r = got / pub
        good = 0.995 <= r <= 1.0
        ok &= good
        print(f"    {'OK ' if good else 'BAD'} {cat:<16} {pub:>10,} {got:>10,} "
              f"{r:>7.4f}  {100.0 * got / grand:5.2f}%")
    # LSB folds Muslim and Baha'i into `Others/not stated`; the village file separates
    # them, so they are added back for the comparison and named separately below.
    got = nat[RESIDUAL] + nat["Muslim"] + nat["Baha'i"]
    pub = TABLE_35[RESIDUAL]
    r = got / pub
    good = 0.99 <= r <= 1.0
    ok &= good
    print(f"    {'OK ' if good else 'BAD'} {'Others/not stated':<16} {pub:>10,} "
          f"{got:>10,} {r:>7.4f}  {100.0 * got / grand:5.2f}%")
    bahai = nat["Baha'i"]
    print(f"      = residual {nat[RESIDUAL]:,} + Muslim {nat['Muslim']:,} + "
          f"Baha'i {bahai:,}, which LSB prints as one row.")
    print("      Every ratio is at or just below 1 because of the missing villages above; "
          "none\n      exceeds it, which a mis-scaled or double-counted category would.")

    print(f"\n  {len(rows):,} rows. The six drawn categories:")
    for cat in CATEGORIES:
        print(f"    {nat[cat]:>11,}  {100.0 * nat[cat] / grand:6.3f}%  {cat}")
    print(f"    {grand:>11,}  100.000%  (village populations)")

    if not ok:
        raise SystemExit("reconciliation FAILED")


def main():
    if "--fetch" in sys.argv:
        fetch()
    rows, data, counts, exact = read()
    check(rows, data, counts, exact)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print("\nwrote", OUT, f"({len(rows):,} rows)")


if __name__ == "__main__":
    main()
