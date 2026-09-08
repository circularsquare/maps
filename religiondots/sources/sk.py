"""Slovakia — SODB 2021, religion by obec, from the census's own ArcGIS server.

Reads (or fetches) data/raw/sk/ and writes data/normalized/sk.csv.

**FOUR SWEEPS RECORDED SLOVAKIA AS WALLED AND THE ROUTE IS NONE OF THE ONES THEY LOOKED FOR.**
`sources.md` §11 ("three walls, each now named"), §11c, §11k and §11o's *"untouched; four
sweeps stand"* all went at `slovak.statistics.sk` (still a 403) and `datacube.statistics.sk`
(open, **678 cubes, no religion** — SODB 2021 is not in DATAcube at all). The 2021 census
results live on their own GIS portal, `gis.scitanie.sk`, which is a **public ArcGIS Server
with 86 hosted services, no key, `capabilities: Query`**:

    https://gis.scitanie.sk/server/rest/services/Hosted/obyv_ekchar_nabo_vekskup/FeatureServer/4

**THE DATA AND THE BOUNDARIES ARE THE SAME LAYER, WHICH IS A FIRST HERE.** Layer 4 is
`AR4318_obec_t_SK`: 2,927 polygons, each carrying its own religion counts as fields. There is
no join to get wrong, so §12's whole "shapes of failure" 1 and 2 do not apply to this country.
`sources/sk_geo.py` reads the geometry off the same service.

**THE PARTITION IS EXACT AND DOUBLY WITNESSED.** The eleven category columns sum to
`spolu` sums to **5,449,270**, which is the figure Slovakia forwarded to the UN Demographic
Yearbook, to the person. Fetching the **8 kraje** from layer 3 independently reproduces the
same total and the same eleven category totals — a free tier cross-check of the kind §9af had
to construct by hand.

**`ostatné` IS 7.83% AND IT IS MOSTLY `nezistené`, WHICH IS WHY IT IS NOT DRAWN.** This is the
one real limit of the country and it was settled off a second publication rather than guessed.
UNSD table 28 carries Slovakia 2021 with **21 categories**, and it agrees with this service to
the person on the total and on all nine large categories. Its tail is broken out where the GIS
folds it:

    Not Stated        353,797   6.493%      <- the bulk of `ostatné`
    Other Religions    64,990   1.193%
    Christian          10,811 ) the ten small named bodies: Baptists, Fraternity Church,
    Baptists            3,883 ) Adventists, Jewish, Old Catholic, Czechoslovak Hussite,
    ... eight more            ) Latter-day Saints, Bahá'í, New Apostolic

and **`cv_8` + `cv_50` = 445,049 = the sum of those twelve rows exactly.** So the two
publications agree on the mass and partition the tail differently, and `ostatné` demonstrably
contains the 353,797 who did not state a religion.

**IT IS DRAWN ANYWAY, ON `other.sk` — Anita's call, 2026-09-08**, reversing a first build
that excluded it on §3.5. Excluding left 7.83% of Slovakia as a hole, and §6.12 says a hole
on a dot map reads as an absence of *people*. **100% of Slovakia is drawn.** The price is a
node that is not comparable with any other country's `other`, so it is labelled `Other or not
stated (Slovakia)` and `note_public` warns that **its density maps the census's reach rather
than religion**: the cell runs 0.0-58.8% between obce and peaks in Roma settlements and city
centres (Košice-Luník IX 58.8%, Pavlovce nad Uhom 28.7%, Bratislava-Staré Mesto 16.4%)
against under 3% in the Orava and Kysuce villages.

Note this is not Kazakhstan's case (§9aq) even though both end up drawn: there an offered
`Отказываюсь указать` box that people actively ticked was drawn *as a refusal*, on the
questionnaire's evidence. Slovakia's form has no such box, so `nezistené` is a derived
residual and drawing it is a §6.12 judgement about holes rather than a §3.5 reclassification.

Usage:
    python sources/sk.py --fetch    two paged queries, ~1.5 MB of JSON
    python sources/sk.py            normalise from data/raw/sk/
"""

import csv
import json
import os
import sys
import time
import urllib.parse

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "sk")
OUT = os.path.join(ROOT, "data", "normalized", "sk.csv")

SOURCE_ID = "sk_sodb_2021"
YEAR = 2021
BASIS = "self_id"

COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count",
           "basis", "year", "source_id", "note"]

SERVICE = ("https://gis.scitanie.sk/server/rest/services/Hosted/"
           "obyv_ekchar_nabo_vekskup/FeatureServer")
OBEC_LAYER = 4          # AR4318_obec_t_SK   — 2,927 polygons
KRAJ_LAYER = 3          # AR4318_kraj_t_SK   — 8 polygons, the cross-check

OBEC_JSON = os.path.join(RAW, "sk_obec_religion.json")
KRAJ_JSON = os.path.join(RAW, "sk_kraj_religion.json")

# The service's field names, and the church each one is. The aliases are Slovak and are
# asserted against the live layer in fetch() -- if the office renumbers `cv_*` this build must
# fail rather than silently relabel a church. `cv_28` and `cv_50` are not sequential with the
# rest because they are the office's own code numbers, not positions.
FIELDS = [
    ("cv_1",  "Rímskokatolícka cirkev v Slovenskej republike (rímskokatolícke)"),
    ("cv_2",  "Evanjelická cirkev augsburského vyznania na Slovensku (evanjelické)"),
    ("cv_3",  "Gréckokatolícka cirkev na Slovensku (gréckokatolícke)"),
    ("cv_4",  "Reformovaná kresťanská cirkev na Slovensku (kalvínske)"),
    ("cv_5",  "Pravoslávna cirkev na Slovensku (pravoslávne)"),
    ("cv_6",  "Náboženská spoločnosť Jehovovi svedkovia v Slovenskej republike"),
    ("cv_7",  "Evanjelická cirkev metodistická, Slovenská oblasť"),
    ("cv_8",  "Kresťanské zbory na Slovensku"),
    ("cv_9",  "Apoštolská cirkev na Slovensku"),
    ("cv_28", "bez náboženského vyznania"),
    ("cv_50", "ostatné"),
]
CODES = [c for c, _ in FIELDS]
LABEL = dict(FIELDS)
TOTAL_CAT = "spolu"
UNDRAWN_CAT = "ostatné"          # cv_50 — see the docstring

KEY = "uzemie"                   # LAU code, e.g. SK0101528595
NAME = "nazov"
# The kraj layer carries no `cislo`/`kraj`/`okres` — asking for them is a 500, not an empty
# column — so the field list is per layer rather than shared.
OUT_FIELDS = {
    4: [KEY, "cislo", NAME, "kraj", "okres", TOTAL_CAT] + CODES,
    3: [KEY, NAME, TOTAL_CAT] + CODES,
}

PAGE = 2000                      # the service's own maxRecordCount
EXPECTED_OBCE = 2_927
EXPECTED_KRAJE = 8
NATIONAL = 5_449_270

# Slovakia's own 2021 figures as forwarded to UNSD Demographic Yearbook table 28. Used ONLY to
# characterise what `ostatné` contains -- nothing here is scaled to them. See the docstring.
UNSD_2021 = {
    "Roman Catholic": 3_038_511,
    "No Religion": 1_296_142,
    "Not Stated": 353_797,
    "Evangelic of Augsburg Affiliation": 286_907,
    "Byzantine Catholic Church": 218_235,
    "Reformed Christian": 85_271,
    "Other Religions": 64_990,
    "Orthodox": 50_677,
    "Jehovah Witness": 16_416,
    "Christian": 10_811,
    "Apostolic": 9_044,
    "Baptists Fraternity Union": 3_883,
    "Fraternity Church": 3_440,
    "Evangelic Methodist Church": 3_018,
    "Seventh Day Adventist": 3_001,
    "Jewish": 2_007,
    "Old Catholic Church": 1_778,
    "Czechoslovak Hussite Church": 581,
    "Church of Jesus Christ of Latter-day Saints": 377,
    "Baha'i": 311,
    "New Apostolic": 73,
}
# the nine the two publications name identically, keyed by our field code
UNSD_SAME = {
    "cv_1": "Roman Catholic",
    "cv_2": "Evangelic of Augsburg Affiliation",
    "cv_3": "Byzantine Catholic Church",
    "cv_4": "Reformed Christian",
    "cv_5": "Orthodox",
    "cv_6": "Jehovah Witness",
    "cv_7": "Evangelic Methodist Church",
    "cv_9": "Apostolic",
    "cv_28": "No Religion",
}


def _ctx():
    """gis.scitanie.sk needs certifi's bundle, and the Windows store is not enough.

    Spec §9h's test says a TLS failure in ONE client is a local problem. This is the
    in-between case `sources/gr.py` documents: the system default store fails with
    `unable to get local issuer certificate` and **certifi's current bundle verifies it
    fine**, so the server is not omitting its intermediate (that is `gh.py`'s case, where
    curl, urllib and certifi all fail together and `verify=False` is the only option).
    Use certifi rather than turning verification off.
    """
    import ssl

    import certifi
    return ssl.create_default_context(cafile=certifi.where())


def _get(url, timeout=180):
    import urllib.request

    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36",
        "Accept": "*/*"})
    with urllib.request.urlopen(req, timeout=timeout, context=_ctx()) as r:
        body = r.read()
    # §5a: a 200 is not a download. ArcGIS answers a bad layer with an HTML error page.
    if body[:1] != b"{":
        raise SystemExit(f"not JSON from {url}\n  first bytes: {body[:120]!r}")
    d = json.loads(body)
    if "error" in d:
        raise SystemExit(f"ArcGIS error from {url}: {d['error']}")
    return d


def _assert_aliases(layer):
    """The field aliases carry the church names. If they move, stop."""
    meta = _get(f"{SERVICE}/{layer}?f=json")
    alias = {f["name"]: f.get("alias") for f in meta.get("fields", [])}
    for code, want in FIELDS:
        got = " ".join(str(alias.get(code, "")).split())
        if got != want:
            raise SystemExit(
                f"layer {layer} field {code} is now {got!r}, expected {want!r}.\n"
                "The office has renumbered or relabelled its religion columns. Re-read the\n"
                "field list before trusting FIELDS -- a positional read would relabel a church.")
    return meta


def _fetch_layer(layer, expected, path):
    meta = _assert_aliases(layer)
    cap = meta.get("maxRecordCount", PAGE)
    feats, offset = [], 0
    while True:
        q = urllib.parse.urlencode({
            "where": "1=1",
            "outFields": ",".join(OUT_FIELDS[layer]),
            "returnGeometry": "false",
            "orderByFields": "objectid",
            "resultOffset": offset,
            "resultRecordCount": min(PAGE, cap),
            "f": "json",
        })
        d = _get(f"{SERVICE}/{layer}/query?{q}")
        got = d.get("features", [])
        feats += got
        print(f"    offset {offset:>5d}: {len(got):>5d} features"
              f"{'  (exceededTransferLimit)' if d.get('exceededTransferLimit') else ''}")
        if len(got) < min(PAGE, cap):
            break
        offset += len(got)
        time.sleep(0.3)

    # THE TRAP THIS GUARDS. maxRecordCount is 2000 and layer 4 has 2,927 rows, so an unpaged
    # query returns 2,000 of them with `exceededTransferLimit: true` buried in the response
    # and no error. Every per-row check would still pass on a Slovakia missing a third of its
    # municipalities. Assert the count against the layer's own count endpoint, not against the
    # length of one response.
    n = _get(f"{SERVICE}/{layer}/query?where=1%3D1&returnCountOnly=true&f=json")["count"]
    if len(feats) != n or n != expected:
        raise SystemExit(f"layer {layer}: paged {len(feats)}, server says {n}, "
                         f"expected {expected}")
    os.makedirs(RAW, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump([f["attributes"] for f in feats], fh, ensure_ascii=False)
    print(f"  wrote {path} ({os.path.getsize(path):,} bytes, {len(feats):,} rows)")


def fetch():
    print(f"GET {SERVICE}/{OBEC_LAYER} (obce)")
    _fetch_layer(OBEC_LAYER, EXPECTED_OBCE, OBEC_JSON)
    print(f"GET {SERVICE}/{KRAJ_LAYER} (kraje, the cross-check)")
    _fetch_layer(KRAJ_LAYER, EXPECTED_KRAJE, KRAJ_JSON)


def _load(path):
    if not os.path.exists(path):
        raise SystemExit(f"missing {path} -- run with --fetch first")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def read():
    obce = _load(OBEC_JSON)
    kraje = _load(KRAJ_JSON)
    if len(obce) != EXPECTED_OBCE:
        raise SystemExit(f"{len(obce)} obce, expected {EXPECTED_OBCE}")

    rows, stats = [], {}
    seen = set()
    empty = []
    for a in obce:
        code = str(a[KEY]).strip()
        if code in seen:
            raise SystemExit(f"duplicate obec code {code!r} -- the key is not unique")
        seen.add(code)
        name = " ".join(str(a[NAME]).split())
        total = int(a[TOTAL_CAT] or 0)
        parts = sum(int(a[c] or 0) for c in CODES)
        if parts != total:
            raise SystemExit(f"{name} ({code}): categories sum to {parts:,}, "
                             f"its own total is {total:,}")
        if total == 0:
            empty.append(name)
        for cat, n in [(TOTAL_CAT, total)] + [(LABEL[c], int(a[c] or 0)) for c in CODES]:
            note = "level=obec"
            if cat == TOTAL_CAT:
                note += "; universe total, not a religion category"
            elif cat == UNDRAWN_CAT:
                note += ("; drawn on `other.sk`, which is NOT an other-religion cell: the "
                         "office folds `nezistené` (not stated) into it, and UNSD table 28 "
                         "breaks the same census's tail out to put not-stated at 353,797, "
                         "83% of this bucket. Its density measures the census's reach "
                         "rather than religion")
            rows.append({"geo_id": code, "geo_level": "obec", "geo_name": name,
                         "source_category": cat, "count": n, "basis": BASIS,
                         "year": YEAR, "source_id": SOURCE_ID, "note": note})

    nat = {c: sum(int(a[c] or 0) for a in obce) for c in CODES}
    nat[TOTAL_CAT] = sum(int(a[TOTAL_CAT] or 0) for a in obce)
    for cat, n in [(TOTAL_CAT, nat[TOTAL_CAT])] + [(LABEL[c], nat[c]) for c in CODES]:
        rows.append({"geo_id": "SK", "geo_level": "country", "geo_name": "Slovensko",
                     "source_category": cat, "count": n, "basis": BASIS, "year": YEAR,
                     "source_id": SOURCE_ID, "note": "level=country; summed from the obce"})

    stats["obce"] = len(obce)
    stats["empty"] = empty
    stats["national"] = nat
    stats["kraj"] = {c: sum(int(a[c] or 0) for a in kraje) for c in CODES}
    stats["kraj"][TOTAL_CAT] = sum(int(a[TOTAL_CAT] or 0) for a in kraje)
    stats["kraje"] = len(kraje)
    return rows, stats


def check(rows, stats):
    ok = True
    nat, kraj = stats["national"], stats["kraj"]

    good = stats["obce"] == EXPECTED_OBCE
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} {stats['obce']:,} obce "
          f"(expected {EXPECTED_OBCE:,}), all codes distinct")

    good = nat[TOTAL_CAT] == NATIONAL
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} national total {nat[TOTAL_CAT]:,} "
          f"(expected {NATIONAL:,})")

    parts = sum(nat[c] for c in CODES)
    good = parts == NATIONAL
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} the 11 categories partition the country exactly "
          f"({parts:,}, difference {NATIONAL - parts})")

    # the free second witness: a different tier of the same service, fetched separately
    mismatch = [c for c in [TOTAL_CAT] + CODES if kraj[c] != nat[c]]
    good = stats["kraje"] == EXPECTED_KRAJE and not mismatch
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} the {stats['kraje']} kraje reproduce every one of "
          f"the 12 totals independently"
          + ("" if good else f" — differs on {mismatch}"))

    # and a third, from a different publisher of the same census
    bad = [(c, nat[c], UNSD_2021[k]) for c, k in UNSD_SAME.items() if nat[c] != UNSD_2021[k]]
    good = not bad
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} the nine categories UNSD names identically agree to "
          f"the person" + ("" if good else f" — {bad}"))

    tail = nat["cv_8"] + nat["cv_50"]
    unsd_tail = sum(v for k, v in UNSD_2021.items()
                    if k not in set(UNSD_SAME.values()))
    good = tail == unsd_tail
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} `Kresťanské zbory` + `ostatné` = {tail:,} = the sum "
          f"of the twelve UNSD rows\n      this service does not name ({unsd_tail:,}) — which "
          "is what proves `ostatné` holds the\n      353,797 who did not state a religion")

    print(f"\n  DRAWN: {NATIONAL:,} of {NATIONAL:,} = 100.00% — all eleven categories are "
          f"drawn (Anita, 2026-09-08).")
    print(f"  BUT `ostatné` {nat['cv_50']:,} ({100.0 * nat['cv_50'] / NATIONAL:.2f}%) goes "
          f"to `other.sk`, and UNSD says\n      {UNSD_2021['Not Stated']:,} of it "
          f"({100.0 * UNSD_2021['Not Stated'] / nat['cv_50']:.0f}% of the cell, "
          f"{100.0 * UNSD_2021['Not Stated'] / NATIONAL:.2f}% of the country) is "
          f"`nezistené` —\n      people who did not answer, not people of another religion. "
          "That node is labelled accordingly\n      and must never be pooled with another "
          "country's `other`.")

    if stats["empty"]:
        print(f"\n  {len(stats['empty'])} obec with nobody in it: "
              f"{', '.join(stats['empty'])} — a military district. §9p: a source publishing a "
              "unit as\n      empty is not the same as omitting it, so it keeps its polygon.")

    print(f"\n  {len(rows):,} rows. Categories, national:")
    for c in sorted(CODES, key=lambda c: -nat[c]):
        mark = "  <- `other.sk`, and 83% of it is `nezistené`" if c == "cv_50" else ""
        print(f"    {nat[c]:>10,}  {100.0 * nat[c] / NATIONAL:6.2f}%  {LABEL[c][:58]}{mark}")

    if not ok:
        raise SystemExit("reconciliation FAILED")


def main():
    if "--fetch" in sys.argv:
        fetch()
    rows, stats = read()
    check(rows, stats)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print("\nwrote", OUT, f"({len(rows):,} rows)")


if __name__ == "__main__":
    main()
