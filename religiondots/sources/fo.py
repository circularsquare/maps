"""Faroe Islands: Census 2011 (Manntal 11. november 2011), Hagstova Føroya, congregation
association by the census's seven districts.

Fetches from Hagstova's open PxWeb API and writes

    data/normalized/fo.csv      six answers by 7 districts, geo_level `district`, persons 15+

`sources/fo_geo.py` builds the placement layer; `taxonomy/fo2011.py` is the mapping;
`sources/fo.md` is the write-up.

## THE TABLES

Statbank `H2/MT` (Census 2011), open PxWeb v1, POST with a JSON query, no key:
  - **MT325** *MT10.2.1 Congregration associations by age, sex and district* (their spelling):
    nine bodies, none, more than one, responses, not stated, not queried. **Drawn.**
  - **MT321** *MT10.1.1 Religion by age, sex and district*: Christian, six named religions,
    other, no religious belief. A witness only (see below).
  - **MT1** *MT1.1.1 Population by age, sex and place of usual residence*: the whole census
    population by single year of age, for the universe check.
The table page's notes (read 2026-09-15): "persons 15 years or older"; "The associations are
enumerated. Some persons have more than one association"; `Not queried` is "Persons 15 years or
older, that did not fill out the query form"; under 3 persons is shown as `...`.

## THE QUESTION

The English census form (UNSD's questionnaire archive, `FRO2011en.pdf`), question **E23**, "This
question is voluntary. With which Christian church, congregation or community are you
associated? Select all that apply.": National Lutheran Church; Christian missionary movements
(*Missiónshús, Meinigheitshús, Salvation Army, KFUM, KFUK, etc.*); Plymouth Brethren;
Charismatic, evangelical congregations (*»Hvítusunnusamkomur«, etc.*); Seventh Day Adventist;
Catholic Church; Orthodox Church; Jehova's Witness; Other; None. E22, also voluntary, asks
religious belief. E16 ends the form for anyone aged 14 or under. The census act (Løgtingsmál
115/2010 §3 stk.3) makes every other question compulsory. Hagstova's Faroese label for the
second box is `Samkomur nærri fólkakirkjuni`, congregations close to the National Church.

## MULTIPLE ASSOCIATIONS: WHAT IS WRITTEN IS PEOPLE, NOT TICKS

MT325 counts a tick, so its bodies and `None` sum to 39,558 against 34,436 people who answered;
4,596 people ticked more than one box, and the table names only one pair (713 ticked both the
National Church and the Brethren). A dot is a person, so each district's five body columns are
scaled by (people who named a body) / (bodies named): 0.80 in Eysturoy to 0.91 in the two
Streymoy districts. `None` is a person count and is written as printed. The rescale assumes the
extra ticks fall on every body in proportion to its size. The obvious alternative, that the
overlaps are National Church plus missionary movement, cannot hold: in six of seven districts
the missionary movements are fewer than the extra ticks left after the named pair (check 6).

## SUPPRESSION

The Adventist, Catholic, Orthodox and Jehovah's Witness cells are `...` in every district and
are folded into that district's `Other congregations`, which sums to 585 over the districts
against 106 nationally; 585 is those four plus the national `Other` exactly (check 4). So a
district's `Other congregations` is written with that note, and no national-only body is drawn
on its own.

## THE CHECKS

Every cell is classified (a count or `...`, anything else stops). The API's values are pinned
against the transcription in `sources.md` §scout-2026-09-15-europe. Within each table: districts
sum to the national cell, ages and sexes to the total, responses + not stated + not queried to
the persons. Across tables: MT321 and MT325 have the same persons and not-queried cells in every
column, and MT1's population aged 15 and over in each district is MT325's persons, so the
universe is everyone aged 15 and over and nobody else. The form is checked for E23's wording.

Usage:
    python sources/fo.py --fetch    fetch the three tables and the form into data/raw/fo/
    python sources/fo.py            check and write data/normalized/fo.csv
"""

import csv
import json
import os
import re
import sys
import unicodedata

os.environ.setdefault("OMP_NUM_THREADS", "6")          # [[feedback_cap_cpu]]
os.environ.setdefault("OPENBLAS_NUM_THREADS", "6")
os.environ.setdefault("MKL_NUM_THREADS", "6")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "fo")
OUT = os.path.join(ROOT, "data", "normalized", "fo.csv")
sys.path.insert(0, HERE)

import micro                                                    # noqa: E402  COLUMNS
from fetch_checks import check_body                             # noqa: E402

SOURCE_ID = "hagstova_census2011_mt325"
YEAR = 2011
BASIS = "self_id"
COLUMNS = micro.COLUMNS

API = "https://statbank.hagstova.fo/api/v1/en/H2/"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/125.0 Safari/537.36")
FORM_URL = "https://unstats.un.org/unsd/demographic/sources/census/quest/FRO2011en.pdf"
FORM = os.path.join(RAW, "FRO2011en.pdf")
FORM_SIZE = 635_037

DISTRICTS = ["Norðoyar", "Eysturoy", "N-streymoy", "S-streymoy", "Vágar", "Sandoy", "Suðuroy"]
AGES = ["15-29 years", "30-49 years", "50-69 years", "70 years or older"]
SEXES = ["Male", "Female"]
HEADER = ["Total"] + AGES + SEXES + DISTRICTS
COLS = ["Total"] + DISTRICTS


def _all(code):
    return {"code": code, "selection": {"filter": "all", "values": ["*"]}}


TABLES = {
    "MT325": ("MT/MT10/MT1002/MT325.px",
              [_all("Congregration associations"), _all("age, sex and district")]),
    "MT321": ("MT/MT10/MT1001/MT321.px", [_all("religion"), _all("age, sex and district")]),
    "MT1": ("MT/MT01/MT0101/MT1.px",
            [_all("place of usual residence"),
             {"code": "sex", "selection": {"filter": "item", "values": ["TOT"]}},
             _all("age")]),
}

TOTAL = "Total number of persons"
BODIES = ["National Lutheran Church", "Christian missionary movements", "Plymouth Brethren",
          "Charismatic, evangelical congregations", "Other congregations"]
NONE = "No congregation association"
MULTI = "More than one congregation association"
NAT_BROTH = "of these both National Lutheran Church and Plymoth Brethren"
RESP = "Responses"
NS = "Not stated"
NQ = "Not queried"

# MT325, transcribed in sources.md §scout-2026-09-15-europe: national, then DISTRICTS order.
T325 = {
    TOTAL: (37965, 4545, 8415, 2961, 14832, 2378, 1063, 3771),
    "National Lutheran Church": (27002, 2547, 6221, 2316, 10302, 1916, 851, 2849),
    "Christian missionary movements": (4085, 625, 1616, 209, 882, 339, 89, 325),
    "Plymouth Brethren": (5381, 1547, 1233, 130, 1829, 129, 38, 475),
    "Charismatic, evangelical congregations": (1262, 154, 340, 65, 505, 80, 82, 36),
    "Other congregations": (106, 75, 126, 48, 262, 27, 17, 30),
    MULTI: (4596, 743, 1778, 227, 1113, 320, 88, 327),
    NAT_BROTH: (713, 145, 170, 26, 280, 20, 7, 65),
    NONE: (1243, 76, 160, 102, 742, 63, 21, 79),
    RESP: (34436, 4184, 7782, 2626, 13234, 2190, 993, 3427),
    NS: (2369, 263, 439, 191, 1073, 140, 47, 216),
    NQ: (1160, 98, 194, 144, 525, 48, 23, 128),
}
T325_NATIONAL_ONLY = {"Seventh Day Adventist": 93, "Catholic Church": 167, "Orthodox Church": 93,
                      "Jehova's Witness": 126}

T321 = {
    TOTAL: T325[TOTAL],
    "Christianity": (33018, 4153, 7670, 2484, 12258, 2137, 964, 3352),
    "Other belief": (149, 16, 43, 15, 150, 13, 6, 30),
    "More than one religous belief": (85, 6, 16, 4, 46, 2, 1, 10),
    "No religous belief": (1397, 71, 151, 127, 896, 60, 18, 74),
    "Responses": (34595, 4234, 7848, 2622, 13250, 2208, 987, 3446),
    "Did not state relgion": (2210, 213, 373, 195, 1057, 122, 53, 197),
    "Not queried": T325[NQ],
}
T321_NATIONAL_ONLY = {"Islam": 23, "Hindu": 7, "Buddism": 66, "Judaism": 12, "Bahá'i": 13,
                      "Sikh": 3}
RESIDENTS = 48_346

# Phrases from the English form, whitespace folded; each must be in its text layer.
FORM_PHRASES = [
    ("With which Christian church, congregation or community are you associated?",
     "E23 asks which Christian church, congregation or community a person is associated with"),
    ("Missiónshús, Meinigheitshús,", "E23's missionary-movements box names mission houses and "
                                    "congregation houses"),
    ("Salvation Army, KFUM, KFUK", "... and the Salvation Army, KFUM and KFUK"),
    ("Hvítusunnusamkomur", "E23's charismatic box names the Pentecostal congregations"),
    ("Those who are 14 years old or younger are finished.",
     "E16 ends the form for anyone aged 14 or under"),
    ("What is your religious belief", "E22 asks religious belief"),
]

SPACES = dict.fromkeys([0x00A0, 0x2007, 0x2008, 0x2009, 0x202F, 0x205F], " ")


def despace(s):
    """Copied from sources/gw.py (not shared yet)."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", str(s)).translate(SPACES)).strip()


def _write(path, body):
    with open(path + ".part", "wb") as fh:
        fh.write(body)
    os.replace(path + ".part", path)                            # [[reference_wb_truncates]]


def post_csv(path, query, name):
    """POST a PxWeb v1 query and return the CSV body, refusing anything that is not one."""
    import requests

    r = requests.post(API + path, json={"query": query, "response": {"format": "csv"}},
                      headers={"User-Agent": UA}, timeout=120)
    r.raise_for_status()
    body = r.content
    if len(body) < 200 or not body.lstrip(b"\xef\xbb\xbf").startswith(b'"'):
        raise SystemExit(f"{name}: not a PxWeb CSV ({len(body)} bytes): {body[:160]!r}")
    return body


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    for name, (path, query) in TABLES.items():
        print("POST", API + path)
        body = post_csv(path, query, name)
        _write(os.path.join(RAW, name + ".csv"), body)
        with open(os.path.join(RAW, name + ".query.json"), "w", encoding="utf-8") as fh:
            json.dump({"url": API + path, "query": query}, fh, ensure_ascii=False, indent=1)
        print(f"  {len(body):,} bytes")
    if os.path.exists(FORM) and os.path.getsize(FORM) == FORM_SIZE:
        print("already have", FORM)
    else:
        print("GET", FORM_URL)
        r = requests.get(FORM_URL, headers={"User-Agent": UA}, timeout=120)
        r.raise_for_status()
        print("  " + check_body(r.content, "pdf", where="FRO2011en.pdf", pin_size=FORM_SIZE))
        _write(FORM, r.content)


def cell(tok, where):
    """A count, or None for Hagstova's `...` (fewer than 3). Anything else stops."""
    if tok == "...":
        return None
    if re.fullmatch(r"\d+", tok):
        return int(tok)
    raise SystemExit(f"{where}: not a count: {tok!r}")


def read_rows(name):
    path = os.path.join(RAW, name + ".csv")
    if not os.path.exists(path):
        raise SystemExit(f"{path} missing; run with --fetch")
    with open(path, encoding="utf-8-sig", newline="") as fh:
        rows = [[despace(c) for c in r] for r in csv.reader(fh)]
    return rows[0], rows[1:]


def read_cross(name):
    """MT325 or MT321 -> {row label: {column label: count or None}}, header asserted."""
    head, body = read_rows(name)
    if head[1:] != HEADER:
        raise SystemExit(f"{name}: unexpected columns {head}")
    out = {}
    for r in body:
        if r[0] in out:
            raise SystemExit(f"{name}: row {r[0]!r} twice")
        out[r[0]] = {c: cell(v, f"{name} {r[0]} / {c}") for c, v in zip(HEADER, r[1:])}
    return out


def read_population():
    """MT1 -> {place: {age label: count}} for the whole census population, sex total."""
    head, body = read_rows("MT1")
    if head[:3] != ["place of usual residence", "sex", "SUM (age)"]:
        raise SystemExit(f"MT1: unexpected columns {head[:4]}")
    out = {}
    for r in body:
        if r[1] != "SUM (sex)":
            raise SystemExit(f"MT1: unexpected sex {r[1]!r}")
        out[r[0]] = {c: cell(v, f"MT1 {r[0]} / {c}") for c, v in zip(head[2:], r[2:])}
    return head[2:], out


UNDER_15 = ["Less than enn 1 year"] + [f"{a} years" for a in range(1, 15)]


def check():
    ok = True

    def say(good, msg):
        nonlocal ok
        ok &= bool(good)
        print(f"  {'OK ' if good else 'BAD'} {msg}")

    import fitz

    print("Faroe Islands: Census 2011, Hagstova MT325 (congregation), MT321 (religion), MT1\n")

    # 0. the form
    with open(FORM, "rb") as fh:
        body = fh.read()
    say(len(body) == FORM_SIZE and body[:5] == b"%PDF-",
        f"the English form is {len(body):,} bytes (expected {FORM_SIZE:,}), a PDF")
    doc = fitz.open(FORM)
    text = despace(" ".join(doc.load_page(i).get_text() for i in range(doc.page_count)))
    for phrase, what in FORM_PHRASES:
        say(despace(phrase) in text, f"form: {what}")
    at = text.find("are you associated?")
    say(at >= 0 and "Select all that apply" in text[at:at + 80],
        "form: E23 says 'Select all that apply', so a person can be counted under two bodies")

    # 1. MT325: pins
    t = read_cross("MT325")
    want_rows = list(T325) + list(T325_NATIONAL_ONLY)
    say(set(t) == set(want_rows), f"MT325 has exactly the {len(want_rows)} expected rows"
        + ("" if set(t) == set(want_rows) else f"; extra {set(t) - set(want_rows)}, "
                                                f"missing {set(want_rows) - set(t)}"))
    bad = [lab for lab, v in T325.items() if tuple(t[lab][c] for c in COLS) != v]
    say(not bad, "MT325's national and district cells equal the transcription"
        + (f"; differ: {bad}" if bad else ""))
    bad = [lab for lab, n in T325_NATIONAL_ONLY.items()
           if t[lab]["Total"] != n or any(t[lab][d] is not None for d in DISTRICTS)]
    say(not bad, "the Adventist, Catholic, Orthodox and Jehovah's Witness rows are national only "
                 "(`...` in every district)" + (f"; not so: {bad}" if bad else ""))

    # 2. arithmetic inside MT325 and MT321
    def identities(tab, fold):
        """{row: ok} for every row with no `...`. The `fold` row's districts hold the suppressed
        bodies too, so only its ages and sexes are summed here (check 4 does its districts)."""
        out = {}
        for lab, v in tab.items():
            if any(v[c] is None for c in HEADER):
                continue
            out[lab] = (sum(v[a] for a in AGES) == v["Total"]
                        and sum(v[s] for s in SEXES) == v["Total"]
                        and (lab == fold or sum(v[d] for d in DISTRICTS) == v["Total"]))
        return out

    r = identities(t, "Other congregations")
    say(all(r.values()) and len(r) == len(T325),
        f"MT325: in all {len(r)} unsuppressed rows the four age bands and the two sexes sum to "
        "the national cell, and so do the districts except in `Other congregations`"
        + (f"; fail: {[k for k, v in r.items() if not v]}" if not all(r.values()) else ""))

    # 3. universe
    say(all(t[RESP][c] + t[NS][c] + t[NQ][c] == t[TOTAL][c] for c in HEADER),
        "MT325: responses + not stated + not queried = persons, in all 14 columns")

    # 4. suppression folds into the district Other
    folded = t["Other congregations"]["Total"] + sum(T325_NATIONAL_ONLY.values())
    dsum = sum(t["Other congregations"][d] for d in DISTRICTS)
    say(dsum == folded,
        f"district `Other congregations` sums to {dsum:,}, which is national Other "
        f"{t['Other congregations']['Total']:,} + the four suppressed bodies "
        f"{sum(T325_NATIONAL_ONLY.values()):,}: the districts fold the suppressed bodies into Other")

    # 5. multiple associations
    national_bodies = [b for b in BODIES] + list(T325_NATIONAL_ONLY)
    excess = {"Total": sum(t[b]["Total"] for b in national_bodies) + t[NONE]["Total"]
              - t[RESP]["Total"]}
    for d in DISTRICTS:
        excess[d] = sum(t[b][d] for b in BODIES) + t[NONE][d] - t[RESP][d]
    say(all(excess[c] >= t[MULTI][c] for c in COLS),
        f"ticks beyond one per person ({excess['Total']:,} nationally) are at least the people "
        f"with more than one association ({t[MULTI]['Total']:,}) in every column; the "
        f"{excess['Total'] - t[MULTI]['Total']:,} left over are third and fourth ticks")
    say(sum(excess[d] for d in DISTRICTS) == excess["Total"],
        "the districts' extra ticks sum to the national figure")
    say(all(t[NAT_BROTH][c] <= min(t["National Lutheran Church"][c], t["Plymouth Brethren"][c],
                                   t[MULTI][c]) for c in COLS),
        "the named pair (National Church and Brethren) fits inside both bodies and inside "
        "`More than one` in every column")

    # 6. the reading 'the other overlaps are National Church + missionary movement' fails
    fails = [d for d in DISTRICTS
             if t["Christian missionary movements"][d] < excess[d] - t[NAT_BROTH][d]]
    say(len(fails) >= 1,
        f"the missionary movements are fewer than the extra ticks left after the named pair in "
        f"{len(fails)} of 7 districts ({', '.join(fails)}), so the overlaps are not all National "
        "Church plus missionary movement, and the rescale is proportional")

    # 7. MT321
    r = read_cross("MT321")
    want_rows = list(T321) + list(T321_NATIONAL_ONLY)
    say(set(r) == set(want_rows), f"MT321 has exactly the {len(want_rows)} expected rows")
    bad = [lab for lab, v in T321.items() if tuple(r[lab][c] for c in COLS) != v]
    say(not bad, "MT321's national and district cells equal the transcription, and its persons "
                 "and not-queried rows are MT325's" + (f"; differ: {bad}" if bad else ""))
    say(all(r[lab]["Total"] == n and all(r[lab][d] is None for d in DISTRICTS)
            for lab, n in T321_NATIONAL_ONLY.items()),
        "MT321's six named non-Christian religions are national only")
    rr = identities(r, "Other belief")
    say(all(rr.values()) and len(rr) == len(T321),
        f"MT321: in all {len(rr)} unsuppressed rows ages and sexes sum to the total, and so do "
        "the districts except in `Other belief`"
        + (f"; fail: {[k for k, v in rr.items() if not v]}" if not all(rr.values()) else ""))
    say(all(r["Responses"][c] + r["Did not state relgion"][c] + r["Not queried"][c]
            == r[TOTAL][c] for c in HEADER),
        "MT321: responses + not stated + not queried = persons, in all 14 columns")
    ob = sum(r["Other belief"][d] for d in DISTRICTS)
    say(ob == r["Other belief"]["Total"] + sum(T321_NATIONAL_ONLY.values()),
        f"MT321's district `Other belief` sums to {ob}, national other belief plus the six named "
        "religions: the same fold as MT325")

    # 8. MT1: the universe is everyone aged 15 and over
    ages, pop = read_population()
    say(pop["SUM (place of usual residence)"]["SUM (age)"] == RESIDENTS,
        f"MT1: {RESIDENTS:,} residents on 11 November 2011")
    say(ages[1:16] == UNDER_15, "MT1's first fifteen single-year columns are ages 0 to 14")
    over15 = {}
    for place, v in pop.items():
        young = [v[a] for a in UNDER_15]
        if any(x is None for x in young):
            raise SystemExit(f"MT1 {place}: a suppressed cell under age 15")
        over15["Total" if place.startswith("SUM") else place] = v["SUM (age)"] - sum(young)
    bad = [(c, over15.get(c), t[TOTAL][c]) for c in COLS if over15.get(c) != t[TOTAL][c]]
    say(not bad, "MT1's population aged 15 and over equals MT325's persons in every district "
                 "and nationally" + (f"; differ: {bad}" if bad else ""))
    under15 = RESIDENTS - t[TOTAL]["Total"]
    print(f"\n  under 15: {under15:,} ({100 * under15 / RESIDENTS:.2f}% of residents); "
          f"15+ who did not state or did not return the form: {t[NS]['Total'] + t[NQ]['Total']:,} "
          f"({100 * (t[NS]['Total'] + t[NQ]['Total']) / t[TOTAL]['Total']:.2f}% of 15+, "
          f"{100 * (t[NS]['Total'] + t[NQ]['Total']) / RESIDENTS:.2f}% of residents); drawn "
          f"{t[RESP]['Total']:,}, {100 * t[RESP]['Total'] / RESIDENTS:.2f}% of residents")
    print(f"  MT321 witness: no religious belief {r['No religous belief']['Total']:,} against "
          f"MT325's no association {t[NONE]['Total']:,}; named non-Christian religions and other "
          f"belief {ob:,}")

    if not ok:
        raise SystemExit("reconciliation FAILED")
    return t, pop


def scale(t):
    """{district: {body: people}} and the factor, per the module docstring."""
    out, factor = {}, {}
    for d in DISTRICTS:
        named = t[RESP][d] - t[NONE][d]
        ticks = sum(t[b][d] for b in BODIES)
        factor[d] = named / ticks
        out[d] = {b: t[b][d] * factor[d] for b in BODIES}
    return out, factor


def emit(t):
    people, factor = scale(t)
    rows = []
    base = dict(geo_level="district", basis=BASIS, year=YEAR, source_id=SOURCE_ID)
    for d in DISTRICTS:
        named = t[RESP][d] - t[NONE][d]
        ticks = sum(t[b][d] for b in BODIES)
        for b in BODIES:
            extra = ("; in a district this cell also holds the Adventist, Catholic, Orthodox and "
                     "Jehovah's Witness answers, which are suppressed there"
                     if b == "Other congregations" else "")
            rows.append(dict(base, geo_id=d, geo_name=d, source_category=b,
                             count=round(people[d][b], 4),
                             note=f"MT325 counts {t[b][d]:,} ticks; x {factor[d]:.5f} = "
                                  f"{named:,} people naming a body / {ticks:,} bodies named"
                                  f"{extra}"))
        rows.append(dict(base, geo_id=d, geo_name=d, source_category=NONE, count=t[NONE][d],
                         note="MT325, people, as printed"))
        for lab, what in ((NS, "did not answer the voluntary question"),
                          (NQ, "did not return the form"),
                          (TOTAL, "the universe, everyone aged 15 and over")):
            rows.append(dict(base, geo_id=d, geo_name=d, source_category=lab, count=t[lab][d],
                             note=f"MT325, not drawn: {what}"))
        drawn = sum(r["count"] for r in rows if r["geo_id"] == d
                    and r["source_category"] in BODIES + [NONE])
        if abs(drawn - t[RESP][d]) > 0.01:
            raise SystemExit(f"{d}: drawn {drawn} != responses {t[RESP][d]}")
    return rows, people, factor


def report(t, people, factor):
    print("\n  per district, people aged 15+ who answered, after the rescale (share of answers):")
    for d in DISTRICTS:
        R = t[RESP][d]
        print(f"   {d:<11} answered {R:>6,}  factor {factor[d]:.3f}  "
              + "  ".join(f"{b.split()[0][:5]} {people[d][b]:>7,.0f} {100 * people[d][b] / R:4.1f}%"
                          for b in BODIES)
              + f"  None {t[NONE][d]:>5,} {100 * t[NONE][d] / R:4.1f}%")
    R = t[RESP]["Total"]
    tot = {b: sum(people[d][b] for d in DISTRICTS) for b in BODIES}
    print(f"   {'nation':<11} answered {R:>6,}  "
          + "  ".join(f"{b.split()[0][:5]} {tot[b]:>7,.0f} {100 * tot[b] / R:4.1f}%" for b in BODIES)
          + f"  None {t[NONE]['Total']:>5,} {100 * t[NONE]['Total'] / R:4.1f}%")
    brethren = {d: people[d]["Plymouth Brethren"] for d in DISTRICTS}
    print(f"  Norðoyar holds {100 * brethren['Norðoyar'] / sum(brethren.values()):.1f}% of the "
          f"drawn Brethren and {100 * t[RESP]['Norðoyar'] / R:.1f}% of the answers")


def main():
    if "--fetch" in sys.argv:
        fetch()
    t, _pop = check()
    rows, people, factor = emit(t)
    report(t, people, factor)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT + ".part", "w", encoding="utf-8", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=COLUMNS)
        wr.writeheader()
        wr.writerows(rows)
    os.replace(OUT + ".part", OUT)                                 # [[reference_wb_truncates]]
    drawn = sum(r["count"] for r in rows if r["source_category"] in BODIES + [NONE])
    print(f"\nwrote {OUT} ({len(rows)} rows, {drawn:,.0f} people drawn)")


if __name__ == "__main__":
    main()
