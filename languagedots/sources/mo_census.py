"""Macau: 2021 Population Census, usual language, by the 23 statistical districts.

    python sources/mo_census.py --fetch    download into data/raw/mo/, then normalise
    python sources/mo_census.py            normalise what is on disk

Source: DSEC (Statistics and Census Service), the census's "Population Statistics Database"
(https://www.dsec.gov.mo/CensosWebDB/), an AngularJS page over an open JSON API with no key:

    GET  https://www.dsec.gov.mo/InReportApi/Censos2016/AllResidentMeta2021     the dimensions
    POST https://www.dsec.gov.mo/InReportApi/Censos2016/AllResidentData
         {"dimensions": "zona,family_language", "filters": null, "year": 2021}

(The "Censos2016" in the path is the app's name; `year` picks the census.) The cube is the whole
resident population, 682,070. `family_language` is "Usual language" (日常用語, Língua corrente):
Cantonese, Mandarin, Other Chinese dialects, Portuguese, English, Tagalog, Others, and "Not
applicable" (18,288), the children under 3 whom the question does not cover. `zona` is the 23
statistical districts (統計分區) plus id 26, the maritime area (people living on boats), which
has no district polygon.

Writes data/normalized/mo.csv: geo_level=zona, geo_id = DSEC's district id (1-23), one row per
language with a non-zero count. The maritime area and "Not applicable" are not written; their
counts are printed for the record's `gap`.

CHECKS, none a tolerance:
  * the cube's national usual-language table (dimension family_language alone) equals the
    districts summed, language by language, and its total is the census's 682,070;
  * a second cut of the same cube, districts x language x nationality, summed over nationality,
    equals the districts x language table cell by cell;
  * the parish x language table (7 parishes + maritime) sums to the same national totals.
The district totals are checked against the GIS's building-level population in mo_geo.py.
"""
import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "mo"
OUT = ROOT / "data" / "normalized" / "mo.csv"

API = "https://www.dsec.gov.mo/InReportApi/Censos2016/"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"}
QUERIES = {
    "lang": "family_language",
    "zona_lang": "zona,family_language",
    "zona_lang_nat": "zona,family_language,nationality",
    "freg_lang": "freguesias,family_language",
}
CENSUS_TOTAL = 682070          # DSEC, 2021 Population Census detailed results: total population
NOT_APPLICABLE = "9999"        # under 3
MARITIME = "26"


def fetch():
    import requests
    RAW.mkdir(parents=True, exist_ok=True)
    meta = RAW / "AllResidentMeta2021.json"
    if not meta.exists():
        r = requests.get(API + "AllResidentMeta2021", headers=UA, timeout=120)
        r.raise_for_status()
        meta.write_text(r.text, encoding="utf-8")
        print("got AllResidentMeta2021")
    for name, dims in QUERIES.items():
        dest = RAW / f"{name}.json"
        if dest.exists():
            print("already have", dest.name)
            continue
        body = {"dimensions": dims, "filters": None, "year": 2021}
        r = requests.post(API + "AllResidentData", json=body, headers=UA, timeout=300)
        r.raise_for_status()
        d = r.json()
        if d.get("Status") != "OK":
            raise SystemExit(f"!! {dims}: {d.get('Status')} {d.get('Debug')}")
        d["_request"] = {"url": API + "AllResidentData", "body": body}
        tmp = dest.with_suffix(".part")
        tmp.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, dest)
        print(f"got {dims}: {len(d['Value']['data'])} cells")


def cube(name):
    """The API's flat array, row-major over the listed dimensions -> {(id, id, ...): count}."""
    import itertools
    v = json.loads((RAW / f"{name}.json").read_text(encoding="utf-8"))["Value"]
    keys = list(itertools.product(*v["dimension"]))
    if len(keys) != len(v["data"]):
        raise SystemExit(f"!! {name}: {len(keys)} keys, {len(v['data'])} values")
    out = {}
    for k, x in zip(keys, v["data"]):
        if str(x).strip() in ("", "-"):
            raise SystemExit(f"!! {name} {k}: empty cell {x!r}")
        out[k] = int(x)
    return out


def main():
    if "--fetch" in sys.argv:
        fetch()
    meta = json.loads((RAW / "AllResidentMeta2021.json").read_text(encoding="utf-8"))
    dims = {m["dimension"]: m for m in meta["Value"]["metaResponses"]}
    lang_name = {x["id"]: x["ename"].strip() for x in dims["family_language"]["values"]}
    zona_name = {x["id"]: x["pname"].split("##")[0].strip() for x in dims["zona_2021"]["values"]}

    nat = cube("lang")
    zl = cube("zona_lang")
    if sum(nat.values()) != CENSUS_TOTAL:
        raise SystemExit(f"!! national table sums to {sum(nat.values()):,}, not {CENSUS_TOTAL:,}")
    zones = sorted({z for z, _ in zl}, key=int)
    if zones != [str(i) for i in range(1, 24)] + [MARITIME]:
        raise SystemExit(f"!! unexpected district ids {zones}")
    for (lang,), n in nat.items():
        s = sum(v for (z, l), v in zl.items() if l == lang)
        if s != n:
            raise SystemExit(f"!! {lang_name[lang]}: districts sum to {s:,}, national {n:,}")
    print(f"national usual-language table: {CENSUS_TOTAL:,} people; the 23 districts plus the "
          "maritime area sum to it in every language")
    print("  " + ", ".join(f"{lang_name[l]} {n:,}" for (l,), n in sorted(nat.items(), key=lambda kv: -kv[1])))

    zln = cube("zona_lang_nat")
    agg = {}
    for (z, l, _), v in zln.items():
        agg[(z, l)] = agg.get((z, l), 0) + v
    bad = [k for k in zl if agg.get(k) != zl[k]]
    if bad:
        raise SystemExit(f"!! districts x language x nationality disagrees in {len(bad)} cells: {bad[:5]}")
    print(f"districts x language x nationality ({len(zln):,} cells), summed over nationality, "
          f"equals districts x language in all {len(zl)} cells")

    fl = cube("freg_lang")
    for (lang,), n in nat.items():
        s = sum(v for (f, l), v in fl.items() if l == lang)
        if s != n:
            raise SystemExit(f"!! parishes' {lang_name[lang]} {s:,} != national {n:,}")
    print("parish x language sums to the national table in every language")

    mar = sum(v for (z, l), v in zl.items() if z == MARITIME)
    mar3 = sum(v for (z, l), v in zl.items() if z == MARITIME and l != NOT_APPLICABLE)
    na = sum(v for (z, l), v in zl.items() if l == NOT_APPLICABLE and z != MARITIME)
    print(f"not written: maritime area {mar:,} people ({mar3:,} aged 3+); under 3 on land {na:,}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".part")
    n_rows = drawn = 0
    with open(tmp, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["geo_level", "geo_id", "geo_name", "source_category", "count"])
        for (z, l), v in sorted(zl.items(), key=lambda kv: (int(kv[0][0]), int(kv[0][1]))):
            if z == MARITIME or l == NOT_APPLICABLE or v == 0:
                continue
            w.writerow(["zona", z, zona_name[z], lang_name[l], v])
            n_rows += 1
            drawn += v
    os.replace(tmp, OUT)
    print(f"wrote {OUT}: {n_rows} rows, {drawn:,} people aged 3+ in the 23 districts")


if __name__ == "__main__":
    main()
