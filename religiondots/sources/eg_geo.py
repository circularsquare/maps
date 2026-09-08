"""Egypt — boundaries and populations for the 27 governorates.

Writes data/geo/eg/eg_governorates.gpkg and data/geo/eg/eg_lookup.csv.

Two sources, and the second one is the find:

  * **boundaries** — OCHA COD-AB Egypt (`cod-ab-egy`), the shapefile bundle rather than the
    geodatabase, on §12's Chile rule about a .gdb that opens, reports the right CRS and
    returns zero features. Read with `engine="fiona"`; the feature count is asserted anyway.
  * **populations** — **CAPMAS's own governorate population estimates, from CAPMAS's own
    API**, not from COD-PS.

## WHY NOT COD-PS, WHICH IS WHAT EVERY OTHER COUNTRY HERE USES

`cod-ps-egy` on HDX is the **2012 COMPAS estimate**: 81,395,541 people over the same 27
pcodes. Egypt passed 100 million in 2020. Drawing today's Copts on a fourteen-year-old
denominator would put a quarter of the country's people nowhere, and unevenly, because Egypt's
governorates have not grown at the same rate (Minya is 5.82% of the country in 2018 and 6.09%
in 2026 by CAPMAS's own series, while Cairo falls from 10.04% to 9.65%). §9bn's Ecuador made
the same call against a COD-PS error of 3.4%; this one is 34%.

## THE CAPMAS API, WHICH §11d AND §11af BOTH CONCLUDED WAS NOT THERE

Both sections tested `www.capmas.gov.eg`, found a 1,421-byte React shell that returns the
same 1,421 bytes for every path including nonsense ones, and recorded that its whole API
namespace was `/api/Language`. **That was the shell answering, not the API.**
`[[reference_spa_hidden_apis]]`'s instruction is to grep the JS bundle, and
`/static/js/main.81637733.js` carries the constants block:

    API_ENDPOINT_URL: "https://www.capmas.gov.eg:8080"

**The API is on a different PORT, not a different path**, which is why every probe of the
443 host came back identical: that host has no API on it at all. Port 8080 answers with real
JSON and no key, and among its 120 routes are `api/Governorate` (27 governorates with their
Arabic and English names, ids and areas) and `api/GovernoratePopulation`, which takes
`governorateId` and `date` and returns CAPMAS's estimate for that governorate on that day.

**It is a clock, so the date is PINNED.** `api/GovernoratePopulation/MinAndMaxDate` reports
2018-01-01 to now, and the same governorate asked twice for the same date returns the same
integer, so pinning makes the build reproducible. `POP_DATE` below is that pin. The response
is cached to `data/raw/eg/` and re-used, so a later run does not silently re-level the country
by a few months of growth.

**And it is a per-governorate series rather than one national clock apportioned**, which was
checked rather than assumed: the governorate SHARES move between 2018 and 2026 (Cairo
10.039% -> 9.647%, Minya 5.819% -> 6.093%). A national clock split by fixed weights would
hold them constant.

## THE JOIN, WHICH IS ON THE ARABIC AND NOT ON THE ENGLISH

The two sides transliterate differently and neither is wrong: COD says `Suhag`, `Kalyoubia`,
`Kafr El-Shikh`, `Sharkia`, `Behera`; CAPMAS says `Sohag`, `Qalyubia`, `Kafr ElSheikh`,
`ElSharqeya`, `Behaira`. **So the English is not a key here**, and this file uses three
witnesses instead:

  1. an authored `GOVERNORATES` table, pcode -> (CAPMAS id, display name), so a change on
     either side is a failure in this file rather than a quiet re-pairing downstream;
  2. the **Arabic** name on each pcode against the Arabic name CAPMAS returns for that id,
     normalised for alef and ya and ta-marbuta only, which is a signal completely independent
     of the authored table;
  3. density: New Valley must come out the sparsest governorate and Cairo the densest.

`sources/ni_geo.py` is why there are three. Nicaragua's code join looked perfect, matched 145
of 153, and silently sent Waspam's Moravians inland. A permutation preserves every total, so
no arithmetic check can find one; two independent keys can.

Usage:
    python sources/eg_geo.py --fetch    one ~15 MB zip from HDX, then 27 small API calls
    python sources/eg_geo.py            rebuild from data/raw/eg/
"""

import json
import os
import re
import ssl
import sys
import urllib.parse
import urllib.request
import zipfile

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import geopandas as gpd
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "eg")
SHP_DIR = os.path.join(RAW, "shp")
OUT_DIR = os.path.join(ROOT, "data", "geo", "eg")
OUT = os.path.join(OUT_DIR, "eg_governorates.gpkg")
LOOKUP = os.path.join(OUT_DIR, "eg_lookup.csv")

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) religiondots/1.0"}

COD_AB = ("https://data.humdata.org/dataset/b90d81ba-7c7a-4283-9899-827480d80a79/resource/"
          "6115d7e5-4ba4-451d-988e-f791f4716e7a/download/egy_admin_boundaries.shp.zip")

CAPMAS = "https://www.capmas.gov.eg:8080/"
# The pin. Any date in the API's 2018-01-01..today window works and returns the same integer
# every time; this one is chosen because "1 January 2026" is a vintage a reader can hold.
POP_DATE = "2026-01-01"
POP_CACHE = os.path.join(RAW, f"capmas_governorate_population_{POP_DATE}.json")

# pcode -> (CAPMAS governorate id, the name this map prints).
#
# The display names are authored here rather than taken from either source, because the two
# sources disagree with each other about the romanisation of nine of the twenty-seven and
# neither set is the one an English reader expects. §12's Chile rule (take the name from the
# statistical source, not from the polygon) is about not letting a boundary file rename a
# unit; the join below is asserted on the Arabic, so nothing here depends on a spelling.
GOVERNORATES = {
    "EG01": (1,  "Cairo"),
    "EG02": (6,  "Alexandria"),
    "EG03": (2,  "Port Said"),
    "EG04": (22, "Suez"),
    "EG11": (23, "Damietta"),
    "EG12": (24, "Dakahlia"),
    "EG13": (3,  "Sharqia"),
    "EG14": (4,  "Qalyubia"),
    "EG15": (7,  "Kafr El Sheikh"),
    "EG16": (8,  "Gharbia"),
    "EG17": (9,  "Monufia"),
    "EG18": (10, "Beheira"),
    "EG19": (25, "Ismailia"),
    "EG21": (5,  "Giza"),
    "EG22": (12, "Beni Suef"),
    "EG23": (13, "Faiyum"),
    "EG24": (14, "Minya"),
    "EG25": (15, "Asyut"),
    "EG26": (16, "Sohag"),
    "EG27": (17, "Qena"),
    "EG28": (18, "Aswan"),
    "EG29": (19, "Luxor"),
    "EG31": (20, "Red Sea"),
    "EG32": (21, "New Valley"),
    "EG33": (11, "Matrouh"),
    "EG34": (26, "North Sinai"),
    "EG35": (27, "South Sinai"),
}

# Arabic orthography that differs between two spellings of the same name and never between
# two different names: the alef forms, final ya, ta marbuta, the tatweel, and the diacritics.
# `مدينة` ("city of") is COD's prefix on Luxor and CAPMAS does not use it.
_ALEF = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي", "ة": "ه", "ـ": ""})
_MARKS = re.compile(r"[ً-ْٰ]")


def ar(s):
    """Fold an Arabic governorate name to a key that two spellings of it share."""
    s = _MARKS.sub("", str(s))
    s = s.translate(_ALEF)
    s = re.sub(r"\s+", "", s)
    return s.replace("مدينه", "", 1) if s.startswith("مدينه") else s


def _ctx():
    ctx = ssl.create_default_context()
    # capmas.gov.eg:8080 serves the API behind a chain this machine does not complete. The
    # payload is a public population table and is checked structurally below.
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def _capmas(path, params=None):
    url = CAPMAS + path + ("?" + urllib.parse.urlencode(params) if params else "")
    req = urllib.request.Request(url, headers={**UA, "Accept": "application/json",
                                               "Referer": "https://www.capmas.gov.eg/",
                                               "Accept-Language": "en"})
    with urllib.request.urlopen(req, timeout=120, context=_ctx()) as r:
        return json.loads(r.read())


def fetch():
    os.makedirs(RAW, exist_ok=True)
    dst = os.path.join(RAW, "egy_admin_boundaries.shp.zip")
    if os.path.exists(dst) and os.path.getsize(dst) > 1_000_000:
        print(f"  have egy_admin_boundaries.shp.zip ({os.path.getsize(dst):,} bytes)")
    else:
        req = urllib.request.Request(COD_AB, headers=UA)
        with urllib.request.urlopen(req, timeout=900) as r:
            data = r.read()
        if data[:2] != b"PK":
            raise SystemExit(f"COD-AB download is not a zip — starts {data[:16]!r}")
        with open(dst + ".part", "wb") as f:
            f.write(data)
        os.replace(dst + ".part", dst)
        print(f"  got  egy_admin_boundaries.shp.zip ({os.path.getsize(dst):,} bytes)")

    if os.path.exists(POP_CACHE):
        print(f"  have {os.path.basename(POP_CACHE)}")
        return
    govs = _capmas("api/Governorate")["data"]
    if len(govs) != 27:
        raise SystemExit(f"api/Governorate returned {len(govs)} governorates, expected 27")
    span = _capmas("api/GovernoratePopulation/MinAndMaxDate")["data"]
    if POP_DATE < span["minDate"][:10] or POP_DATE > span["maxDate"][:10]:
        raise SystemExit(f"POP_DATE {POP_DATE} is outside the API's {span} window")
    rows = []
    for g in sorted(govs, key=lambda x: x["id"]):
        j = _capmas("api/GovernoratePopulation",
                    {"governorateId": g["id"], "date": POP_DATE})
        pop = j["data"]["population"]
        if not isinstance(pop, (int, float)) or pop <= 0:
            raise SystemExit(f"api/GovernoratePopulation gave {pop!r} for id={g['id']}")
        en = g["governorateTranslations"][0]["name"] if g.get("governorateTranslations") else ""
        rows.append({"id": g["id"], "name_ar": g["name"], "name_en": en,
                     "population": int(round(pop))})
        print(f"    {g['id']:>3}  {en:<16} {int(round(pop)):>12,}")
    with open(POP_CACHE, "w", encoding="utf-8") as f:
        json.dump({"date": POP_DATE, "source": CAPMAS, "governorates": rows}, f,
                  ensure_ascii=False, indent=1)
    print(f"  wrote {POP_CACHE} ({sum(r['population'] for r in rows):,} people)")


def main():
    if "--fetch" in sys.argv:
        fetch()

    zpath = os.path.join(RAW, "egy_admin_boundaries.shp.zip")
    if not os.path.exists(zpath):
        raise SystemExit(f"{zpath} missing — run with --fetch")
    if not os.path.exists(POP_CACHE):
        raise SystemExit(f"{POP_CACHE} missing — run with --fetch")
    os.makedirs(SHP_DIR, exist_ok=True)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(SHP_DIR)

    shp = os.path.join(SHP_DIR, "egy_admin1.shp")
    g = gpd.read_file(shp, engine="fiona")
    if len(g) != 27:
        raise SystemExit(f"{len(g)} ADM1 features, expected 27 — COD has re-cut Egypt")
    if g.crs is None or g.crs.to_epsg() != 4326:
        raise SystemExit(f"unexpected CRS {g.crs}")
    print(f"read {shp}: {len(g)} governorates, {g.crs}")

    g["pcode"] = g["adm1_pcode"].astype(str).str.strip()

    # ---- witness 1: the authored table covers COD's pcodes exactly ----
    off = sorted(set(g["pcode"]) ^ set(GOVERNORATES))
    if off:
        raise SystemExit(f"COD's pcodes and GOVERNORATES do not agree: {off}")
    print(f"  witness 1 — the authored table covers COD's 27 pcodes exactly")

    cap = json.load(open(POP_CACHE, encoding="utf-8"))
    if cap["date"] != POP_DATE:
        raise SystemExit(f"{POP_CACHE} is pinned to {cap['date']}, not {POP_DATE}")
    by_id = {r["id"]: r for r in cap["governorates"]}
    if sorted(by_id) != sorted(v[0] for v in GOVERNORATES.values()):
        raise SystemExit("CAPMAS's governorate ids and GOVERNORATES' do not agree")

    # ---- witness 2: and the ARABIC name on each pcode is the one CAPMAS has on that id ----
    mism = []
    for pcode, (cap_id, _name) in GOVERNORATES.items():
        cod_ar = g.loc[g["pcode"] == pcode, "adm1_name1"].iloc[0]
        if ar(cod_ar) != ar(by_id[cap_id]["name_ar"]):
            mism.append((pcode, cod_ar, by_id[cap_id]["name_ar"], by_id[cap_id]["name_en"]))
    if mism:
        for pcode, a, b, en in mism:
            print(f"    {pcode}: COD says {a!r}, CAPMAS id has {b!r} ({en})")
        raise SystemExit("a pcode is paired with a different governorate's CAPMAS id — the "
                         "table is a permutation and every downstream join is wrong. STOP.")
    print(f"  witness 2 — the Arabic name agrees on all {len(GOVERNORATES)}, independently "
          "of the authored English")

    g["geo_id"] = g["pcode"]
    g["name"] = g["pcode"].map(lambda p: GOVERNORATES[p][1])
    g["pop"] = g["pcode"].map(lambda p: by_id[GOVERNORATES[p][0]]["population"]).astype("int64")
    print(f"  witness 3 pending — CAPMAS {POP_DATE}: {g['pop'].sum():,} people")

    # ---- witness 3: the shape of the population, which a permutation would break ----
    g["density"] = g["pop"] / g["area_sqkm"]
    lo = g.loc[g["density"].idxmin(), "name"]
    hi = g.loc[g["density"].idxmax(), "name"]
    print(f"    sparsest {lo!r} at {g['density'].min():.2f}/km2, densest {hi!r} at "
          f"{g['density'].max():,.0f}/km2")
    if lo != "New Valley" or hi != "Cairo":
        raise SystemExit("New Valley is not the sparsest governorate or Cairo is not the "
                         "densest — the population join is permuted")
    print("  witness 3 — New Valley sparsest, Cairo densest")

    g["unit"] = g["pcode"]
    os.makedirs(OUT_DIR, exist_ok=True)
    out = g[["unit", "name", "pcode", "geo_id", "pop", "geometry"]]
    out.to_file(OUT, layer="governorates", driver="GPKG")
    print(f"\nwrote {OUT} ({len(out)} polygons)")

    lut = pd.DataFrame({"geo_id": sorted(GOVERNORATES),
                        "unit": sorted(GOVERNORATES),
                        "name": [GOVERNORATES[p][1] for p in sorted(GOVERNORATES)],
                        "pop": [by_id[GOVERNORATES[p][0]]["population"]
                                for p in sorted(GOVERNORATES)]})
    lut.to_csv(LOOKUP, index=False, encoding="utf-8")
    print(f"wrote {LOOKUP} ({len(lut)} rows, {lut['pop'].sum():,} people)")
    print(lut.sort_values("pop", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
