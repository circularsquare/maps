"""Qatar -- 2004 census, population by religion, sex and municipality (Table 6).

Reads (or fetches) six of the 2004 census's population table pages, archived on the Wayback
Machine, into data/normalized/qa.csv. `sources/qa.md` is the write-up; `sources/qa_geo.py`
rebuilds the ten 2004 municipalities from COD-AB's zones and imports `read_zones` from here.

## THE TABLE

Planning Council, Statistics Department (its successor is the National Planning Council), Census
2004, population tables, **Table No (6), "Population By Religion, Gender And Municipality, March
2004"**. Persons, males and females for the ten municipalities of 2004 and the nation, in three
religions: Muslim, Christian, Other. 744,029 people. No other table of the census crosses religion
with anything, as far as the archived tree was opened (sources/qa.md §1).

## THE PAGES SURVIVE ONLY ON WAYBACK

`www.mdps.gov.qa/en/statistics/census/Census2004/Population/Pages/Tables/Pubulation/T0n.aspx`
(the folder is spelled `Pubulation`), captured on 27 June 2017. The Ministry of Development
Planning and Statistics has since become the National Planning Council (`npc.qa`) and the tree is
gone. They are HTML pages, so there are no magic bytes to test: each page's CDX digest is pinned
and its caption asserted.

## WHAT THE NUMBERS ARE: A COUNT FOR QATARIS, A CALIBRATED SAMPLE FOR EVERYONE ELSE

The census's population introduction page (captured 2017-06-25) says buildings, dwellings,
Qataris and establishments were enumerated in full, while the "non-Qatari population residing in
the State, whether in residential units or labor gatherings, were enumerated via a sample",
stratified by municipality, weighted for selection and non-response, and then calibrated "to the
March 2004 census results based on complete counts of the non-Qatari population", by
municipality and sex. So each municipality's total and sex split below is a count, and the
religion split of its non-Qataris is a weighted estimate from that sample. No table crosses
religion with nationality.

## THE CHECKS

    Table 6 parsed off the page equals the transcription below
    persons = males + females in every cell; the municipalities sum to the national row in all
        twelve columns; the three religions sum to the total columns in every row
    Table 1 (population by sex and municipality) equals Table 6's total columns
    Table 2's zones sum to their municipality's printed total and to Table 1, sexes included
    Tables 3, 4 and 5 (age by municipality for persons, males, females): the age rows sum to the
        total row in every column, the totals equal Table 1, and persons = males + females
    UNSD Demographic Yearbook table 28, Qatar 2004, equals Table 6's national row

## THE QUESTIONNAIRE

UNSD's copy (`QAT2004en.pdf`) is the Qatari form, "Questionnaire of Qatari Characteristics"
(form 4 P.C.), with a religion column coded 1 Muslim, 2 Christian, 3 Other, and no code for no
religion or for not stated. The non-Qatari form (4BPC) and the labour gatherings form were not
found.

Usage:
    python sources/qa.py --fetch    the six pages (about 0.8 MB) from Wayback
    python sources/qa.py            normalise from data/raw/qa/
"""

import csv
import html
import os
import re
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "qa")
OUT = os.path.join(ROOT, "data", "normalized", "qa.csv")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from fetch_checks import digest, wayback_raw   # noqa: E402

SOURCE_ID = "qa_census2004_population_t06"
YEAR = 2004
BASIS = "self_id"
COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count",
           "basis", "year", "source_id", "note"]

_TREE = ("http://www.mdps.gov.qa/en/statistics/census/Census2004/Population/Pages/Tables/"
         "Pubulation/")
# CDX `digest` of each capture (SHA-1 of the archived body, base32) and the caption the page must
# carry, exactly as the parser joins its cell text.
PAGES = {
    "T01": dict(ts="20170627120708", digest="QWROGGBTNTDY6NEJO3ALGDQ4OAJ6JFGZ",
                caption="Population By gender And Municipality"),
    "T02": dict(ts="20170627120719", digest="QU4TY64ZCYVSSSW6JTMGD2CWRKJOWQKN",
                caption="Population By Gender ,Municipality And Zone"),
    "T03": dict(ts="20170627120725", digest="EVZLHTN7UBN73H6NJ4Q665TJXYXCWFI6",
                caption="Population By Municipality And Age Groups"),
    "T04": dict(ts="20170627120731", digest="HFDHBPLSKK77BZXQAQMJ6CMXSSVPT2WV",
                caption="Males By Municipality And Age Groups"),
    "T05": dict(ts="20170627120737", digest="GRD6JZ6UJD6PICG3EAU5SHGMUDQWD7MI",
                caption="Females By Municipality And Age Groups"),
    "T06": dict(ts="20170627120742", digest="BSAO3D3IWND3W6IKEG3DSUCZGN37AZWU",
                caption="Population By Religion, Gender And Municipality"),
}

NATIONAL = "Qatar"
# The ten municipalities of 2004, in the tables' own order, under the names used here.
UNITS = ["Doha", "Al Rayyan", "Al Wakra", "Umm Salal", "Al Khor", "Al Shamal", "Al Ghuwairiya",
         "Al Jemailya", "Jeryan Al Batna", "Mesaieed"]
# Each table spells them its own way. Exact labels, every one required exactly once.
T06_LABELS = dict(zip(["Doha", "Al rayyan", "Al wakra", "Umm salal", "Al khor", "Al shamal",
                       "Al ghuwairiya", "Al jemailya", "Jerian al betna", "Mesaied"], UNITS))
T02_HEADINGS = dict(zip(["DOHA", "AL RAYYAN", "AL WAKRA", "UMM SALAL", "AL Khor", "AL SHAMAL",
                         "AL GHUWAIRIYA", "AL JEMAILYA", "JERYAN AL BATN", "MESAIEED"], UNITS))
AGE_LABELS = dict(zip(["Doha", "Al Rayyan", "Al Wakra", "Umm Salal", "Al Khor", "Al Shamal",
                       "Al Ghuwairiya", "Al Jemailya", "Jeryan Al Betna", "Mesaieed"], UNITS))
AGES = ["0", "1 - 4", "5 - 9", "10 - 14", "15 - 19", "20 - 24", "25 - 29", "30 - 34", "35 - 39",
        "40 - 44", "45 - 49", "50 - 54", "55 - 59", "60 - 64", "65 - 69", "70 - 74", "75 - 79",
        "80 +"]
CATS = ["Muslim", "Christian", "Other"]
ZONES = 87          # zone rows in Table 2, counted by hand off the page (2026-09-15)

# Table 6, persons, (Muslim, Christian, Other), transcribed from the page and asserted equal to it.
T6 = {
    "Doha":            (268_915, 34_482, 36_450),
    "Al Rayyan":       (213_675, 16_445, 42_740),
    "Al Wakra":        (26_768, 1_266, 3_407),
    "Umm Salal":       (27_750, 2_016, 1_839),
    "Al Khor":         (17_699, 3_491, 10_357),
    "Al Shamal":       (4_236, 333, 346),
    "Al Ghuwairiya":   (1_666, 52, 441),
    "Al Jemailya":     (6_782, 955, 2_566),
    "Jeryan Al Batna": (4_019, 2_269, 390),
    "Mesaieed":        (4_881, 1_903, 5_890),
}
T6_NATIONAL = (576_391, 63_212, 104_426)
TOTAL = 744_029

UNSD_NAME = "Qatar"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) religiondots/1.0"}


def path(key):
    return os.path.join(RAW, f"census2004_population_{key}.html")


def fetch():
    import urllib.request

    os.makedirs(RAW, exist_ok=True)
    for key, p in PAGES.items():
        dst = path(key)
        if os.path.exists(dst):
            with open(dst, "rb") as fh:
                if digest(fh.read()) == p["digest"]:
                    print("already have", dst)
                    continue
        url = wayback_raw(p["ts"], _TREE + f"{key}.aspx")
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=300) as r:
            body = r.read()
        got = digest(body)
        if got != p["digest"]:
            raise SystemExit(f"{url}: {len(body):,} bytes with digest {got}, not the pinned "
                             f"{p['digest']}; a different capture or a truncated body")
        with open(dst + ".part", "wb") as fh:
            fh.write(body)
        os.replace(dst + ".part", dst)
        print(f"wrote {dst} ({len(body):,} bytes)")


def page_rows(key):
    """Every <tr> on the page as a list of whitespace-normalised cell texts, empty rows dropped.
    Stops unless the pinned digest matches and the caption is on the page."""
    p = path(key)
    if not os.path.exists(p):
        raise SystemExit(f"{p} missing -- run: python sources/qa.py --fetch")
    with open(p, "rb") as fh:
        body = fh.read()
    if digest(body) != PAGES[key]["digest"]:
        raise SystemExit(f"{p}: digest {digest(body)} is not the pinned {PAGES[key]['digest']}")
    text = body.decode("utf-8")
    rows = []
    for tr in re.findall(r"<tr\b.*?</tr>", text, flags=re.S | re.I):
        cells = [" ".join(html.unescape(re.sub(r"<[^>]+>", " ", c)).split())
                 for c in re.findall(r"<t[dh]\b.*?</t[dh]>", tr, flags=re.S | re.I)]
        if any(cells):
            rows.append(cells)
    if not any(PAGES[key]["caption"] in " ".join(r) for r in rows):
        raise SystemExit(f"{key}: caption {PAGES[key]['caption']!r} not on the page")
    return rows


def count(v, where):
    """One count cell. Anything that is not a run of digits stops."""
    if re.fullmatch(r"\d+", v):
        return int(v)
    raise SystemExit(f"{where}: {v!r} is not a count")


def english(cell):
    return " ".join(re.findall(r"[A-Za-z]+", cell))


def read_t06():
    """{unit or NATIONAL: {'Total'|cat: (persons, males, females)}}. The page prints each group
    as Total, Females, Males; the group and sub-header rows are asserted to say so."""
    rows = page_rows("T06")
    heads = [i for i, r in enumerate(rows) if r[0] == "Municipality"]
    if len(heads) != 1:
        raise SystemExit(f"T06: {len(heads)} header rows starting `Municipality`")
    groups = [english(c) for c in rows[heads[0]][1:-1]]
    if groups != ["Total", "Other", "Christian", "Muslim"]:
        raise SystemExit(f"T06: column groups read {groups}")
    sub = [english(c) for c in rows[heads[0] + 1]]
    if sub != ["Total", "Females", "Males"] * 4:
        raise SystemExit(f"T06: sub-header reads {sub}")
    out = {}
    for r in rows[heads[0] + 2:]:
        unit = T06_LABELS.get(r[0]) or (NATIONAL if r[0] == "Total" else None)
        if unit is None:
            raise SystemExit(f"T06: unexpected row {r!r}")
        if unit in out or len(r) != 14:
            raise SystemExit(f"T06: row {r[0]!r} repeated or not 14 cells ({len(r)})")
        v = [count(x, f"T06 {r[0]!r}") for x in r[1:13]]
        out[unit] = {g: (v[3 * i], v[3 * i + 2], v[3 * i + 1]) for i, g in enumerate(groups)}
    if list(out) != UNITS + [NATIONAL]:
        raise SystemExit(f"T06: rows {list(out)}, expected {UNITS + [NATIONAL]}")
    return out


def read_t01():
    """{unit or NATIONAL: (persons, males, females)}."""
    out = {}
    for r in page_rows("T01"):
        unit = T06_LABELS.get(r[0]) or (NATIONAL if r[0] == "Total" else None)
        if unit is None or len(r) != 5 or not r[1].isdigit():
            continue
        if unit in out:
            raise SystemExit(f"T01: row {r[0]!r} twice")
        t, f, m = (count(x, f"T01 {r[0]!r}") for x in r[1:4])
        out[unit] = (t, m, f)
    if list(out) != UNITS + [NATIONAL]:
        raise SystemExit(f"T01: rows {list(out)}, expected {UNITS + [NATIONAL]}")
    return out


def read_zones():
    """Table 2: ({zone number: (unit, zone name, (persons, males, females))},
    {unit: printed total (persons, males, females)}, national (persons, males, females)).

    Zones sit under a heading row per municipality and close on a `Total` row; the page ends on a
    `G.Total` row. Each municipality's zones are asserted to sum to its own total row, and the
    totals to the grand total, persons, males and females."""
    rows = page_rows("T02")
    zones, totals, order = {}, {}, []
    grand, cur = None, None
    for r in rows:
        lab = r[0]
        if lab in T02_HEADINGS and not any(r[1:-1]):
            cur = T02_HEADINGS[lab]
            if cur in order:
                raise SystemExit(f"T02: heading {lab!r} twice")
            order.append(cur)
            continue
        if cur is None:
            continue
        if re.fullmatch(r"\d+", lab):
            if len(r) != 7 or r[6] != lab:
                raise SystemExit(f"T02: zone row {r!r} is not 7 cells ending in its own number")
            z = int(lab)
            if z in zones:
                raise SystemExit(f"T02: zone {z} twice")
            t, f, m = (count(x, f"T02 zone {z}") for x in r[2:5])
            zones[z] = (cur, r[1], (t, m, f))
        elif lab == "Total" and len(r) == 5:
            t, f, m = (count(x, f"T02 {cur} total") for x in r[1:4])
            totals[cur] = (t, m, f)
        elif lab == "G.Total":
            t, f, m = (count(x, "T02 G.Total") for x in r[1:4])
            grand = (t, m, f)
            break
        else:
            raise SystemExit(f"T02: unexpected row {r!r}")
    if order != UNITS or list(totals) != UNITS or grand is None:
        raise SystemExit(f"T02: headings {order}, totals {list(totals)}, grand total {grand}")
    for u in UNITS:
        s = tuple(sum(v[2][i] for v in zones.values() if v[0] == u) for i in range(3))
        if s != totals[u]:
            raise SystemExit(f"T02 {u}: zones sum to {s}, the total row prints {totals[u]}")
    if tuple(sum(totals[u][i] for u in UNITS) for i in range(3)) != grand:
        raise SystemExit(f"T02: municipal totals do not sum to G.Total {grand}")
    bad = [z for z, v in zones.items() if v[2][0] != v[2][1] + v[2][2]]
    if bad:
        raise SystemExit(f"T02: persons != males + females in zones {bad}")
    return zones, totals, grand


def read_age(key):
    """Tables 3-5: ({age: {unit or NATIONAL: n}}, {unit or NATIONAL: printed total})."""
    rows = page_rows(key)
    heads = [r for r in rows if r[0] == "Total" and len(r) == 11 and not r[1].isdigit()]
    if len(heads) != 1:
        raise SystemExit(f"{key}: {len(heads)} English header rows")
    unknown = [c for c in heads[0][1:] if c not in AGE_LABELS]
    if unknown:
        raise SystemExit(f"{key}: column labels {unknown} are not municipalities")
    cols = [NATIONAL] + [AGE_LABELS[c] for c in heads[0][1:]]
    if sorted(cols[1:]) != sorted(UNITS):
        raise SystemExit(f"{key}: columns {cols}")
    ages, total = {}, None
    for r in rows:
        if len(r) != 13:
            continue
        if r[0] in AGES:
            if r[0] in ages:
                raise SystemExit(f"{key}: age row {r[0]!r} twice")
            ages[r[0]] = dict(zip(cols, (count(x, f"{key} {r[0]!r}") for x in r[1:12])))
        elif r[0] == "Total":
            total = dict(zip(cols, (count(x, f"{key} total") for x in r[1:12])))
    if list(ages) != AGES or total is None:
        raise SystemExit(f"{key}: age rows {list(ages)}, total row {'found' if total else 'missing'}")
    return ages, total


def check():
    ok = True

    def say(good, msg):
        nonlocal ok
        ok &= bool(good)
        print(f"  {'OK ' if good else 'BAD'} {msg}")

    print("Qatar -- Census 2004, population Table 6\n")
    for key in PAGES:
        page_rows(key)              # stops on a digest or caption mismatch
    say(True, "all six pages match their pinned CDX digests and carry their captions")
    t6 = read_t06()

    # 1. the page and the transcription are one table
    say({u: tuple(t6[u][c][0] for c in CATS) for u in UNITS} == T6,
        "Table 6's persons by municipality equal the transcription")
    say(tuple(t6[NATIONAL][c][0] for c in CATS) == T6_NATIONAL and t6[NATIONAL]["Total"][0] == TOTAL,
        f"Table 6's national row is {T6_NATIONAL}, total {TOTAL:,}")

    # 2. the table closes on itself
    groups = ["Total"] + CATS
    bad = [(u, g) for u in t6 for g in groups if t6[u][g][0] != t6[u][g][1] + t6[u][g][2]]
    say(not bad, f"persons = males + females in all {len(t6) * 4} cells {bad or ''}")
    bad = [(g, i) for g in groups for i in range(3)
           if t6[NATIONAL][g][i] != sum(t6[u][g][i] for u in UNITS)]
    say(not bad, f"the ten municipalities sum to the national row in all 12 columns {bad or ''}")
    bad = [(u, i) for u in t6 for i in range(3)
           if t6[u]["Total"][i] != sum(t6[u][c][i] for c in CATS)]
    say(not bad, f"the three religions sum to the total in every row, persons, males, females {bad or ''}")

    # 3. Table 1
    t1 = read_t01()
    say(all(t1[u] == t6[u]["Total"] for u in t1),
        "Table 1 (sex by municipality) equals Table 6's total columns, national row included")

    # 4. Table 2: zones
    zones, z_tot, z_grand = read_zones()
    say(len(zones) == ZONES, f"Table 2 lists {len(zones)} zones (pinned {ZONES})")
    say(all(z_tot[u] == t1[u] for u in UNITS) and z_grand == t1[NATIONAL],
        "Table 2's zones sum to their municipality's printed total, and those equal Table 1, "
        "persons, males and females")

    # 5. Tables 3-5: age
    a3, s3 = read_age("T03")
    a4, s4 = read_age("T04")
    a5, s5 = read_age("T05")
    for key, ages, tot, i in (("T03", a3, s3, 0), ("T04", a4, s4, 1), ("T05", a5, s5, 2)):
        say(all(sum(ages[a][col] for a in AGES) == tot[col] for col in tot),
            f"{key}'s 18 age rows sum to its total row in all 11 columns")
        say(all(tot[col] == t1[col][i] for col in tot),
            f"{key}'s totals equal Table 1's {['persons', 'males', 'females'][i]}")
    say(all(a3[a][col] == a4[a][col] + a5[a][col] for a in AGES for col in a3[a]),
        "Table 3 = Table 4 + Table 5 in every age and column")

    # 6. UNSD Demographic Yearbook table 28
    import oracle as unsd
    got = unsd.oracle(UNSD_NAME, YEAR)
    if got is None:
        say(False, f"UNSD has no {UNSD_NAME} {YEAR} row (run python tools/oracle.py --fetch)")
    else:
        cats, total, exact = unsd.partition(got.get(unsd.TOTAL, {}))
        say(exact and total == TOTAL and cats == dict(zip(CATS, T6_NATIONAL)),
            f"UNSD table 28, {UNSD_NAME} {YEAR}: {cats}, total {total:,}, equal to Table 6 to the person")

    if not ok:
        raise SystemExit("reconciliation FAILED")
    return t6, zones, a3


def emit():
    rows = []
    for u in UNITS:
        for c, n in zip(CATS, T6[u]):
            if n <= 0:
                continue
            rows.append({
                "geo_id": u, "geo_level": "municipality_2004", "geo_name": u,
                "source_category": c, "count": n, "basis": BASIS, "year": YEAR,
                "source_id": SOURCE_ID,
                "note": f"Table 6 persons; municipality total {sum(T6[u])}",
            })
    return rows


def main():
    if "--fetch" in sys.argv:
        fetch()
    t6, zones, a3 = check()
    rows = emit()

    total = sum(r["count"] for r in rows)
    print(f"\n  10 municipalities of 2004, {total:,} people, {len(zones)} zones")
    for c, n in zip(CATS, T6_NATIONAL):
        print(f"    {n:>9,}  {100.0 * n / total:6.3f}%  {c}")
    print(f"\n  {'municipality':<16}{'people':>9}  " + "  ".join(f"{c:>9}" for c in CATS)
          + "   male share: " + " / ".join(CATS))
    for u in UNITS + [NATIONAL]:
        t = t6[u]["Total"][0]
        print(f"  {u:<16}{t:>9,}  " + "  ".join(f"{100.0 * t6[u][c][0] / t:8.2f}%" for c in CATS)
              + "   " + " / ".join(f"{100.0 * t6[u][c][1] / t6[u][c][0]:.0f}%" for c in CATS))
    kids = sum(a3[a][NATIONAL] for a in AGES[:4])
    print(f"\n  under 15 (Table 3): {kids:,}, {100.0 * kids / TOTAL:.1f}% of everyone")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT + ".part", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    os.replace(OUT + ".part", OUT)
    print("\nwrote", OUT, f"({len(rows)} rows)")


if __name__ == "__main__":
    main()
