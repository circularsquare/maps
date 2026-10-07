"""Kosovo: Kosovo Agency of Statistics (ASK), Census 2024, mother tongue by municipality.

    python sources/xk_census.py --fetch    three PxWeb POSTs, seconds
    python sources/xk_census.py            normalise from data/raw/xk/

-> data/normalized/xk.csv   (levels `country` and `municipality`, 2024 only; municipality drawn)
-> data/normalized/xk_north.csv   ASK's own estimate for the four northern municipalities

Tables, in ASK's PxWeb (https://askdata.rks-gov.net/api/v1/en/ASKdata/Census population/
1_Demographic_Characteristics/):

  census2024_22.px  population by mother tongue and sex, country and 38 municipalities, 2011 and
                    2024. Categories: Albanian, Serb (Serbian), Bosnian, Turkish, Romani,
                    Other (specify), Not available. The drawn table (2024, both sexes).
  census2024_05.px  ethnicity by sex and municipality, 2011 and 2024, as enumerated.
  census2024_63.px  the same "(with estimation)": identical except in the four northern
                    municipalities, where ASK restores the population the census did not reach
                    (Serbs who boycotted). religiondots reads the same pair (sources/xk.py).

The root of this PxWeb answers `dbid` where others answer `id` (religiondots' note); the paths
here are fixed, so no walk is needed.

CHECKS: national total 1,585,566 (enumerated, as published); the categories partition every
municipality; the 38 municipalities sum to the national row in every category; every
municipality's mother-tongue total equals its 2024 total in the ethnicity table; the estimated
ethnicity table differs from the enumerated one in the four northern municipalities and the
country row only, and what it adds there is over 90% Serb.
"""

import csv
import json
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "xk")
OUT = os.path.join(ROOT, "data", "normalized", "xk.csv")
NORTH_OUT = os.path.join(ROOT, "data", "normalized", "xk_north.csv")

SOURCE_ID = "xk_census_2024_mt"
YEAR = "2024"
COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count", "tier", "year",
           "source_id", "note"]
NORTH_COLUMNS = ["geo_id", "geo_name", "ethnicity", "enumerated", "estimated", "added",
                 "source_id", "note"]

BASE = ("https://askdata.rks-gov.net/api/v1/en/ASKdata/Census population/"
        "1_Demographic_Characteristics/")
TABLES = {"mt": "census2024_22.px", "eth": "census2024_05.px", "eth_est": "census2024_63.px"}

NATIONAL = 1_602_515          # with ASK's estimate for the north; 1,585,566 enumerated
NATIONAL_ENUMERATED = 1_585_566
COUNTRY = "KOSOVA"
N_UNITS = 38
NORTH = ("Leposaviq", "Zubin Potok", "Zveqan", "Mitrovicë e Veriut")
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}


def fetch():
    import requests
    os.makedirs(RAW, exist_ok=True)
    for table in TABLES.values():
        dest = os.path.join(RAW, table.replace(".px", ".json"))
        if os.path.exists(dest) and os.path.getsize(dest) > 2_000:
            print("already have", dest)
            continue
        url = BASE + table
        print("POST", url)
        r = requests.post(url, headers=UA, timeout=300,
                          json={"query": [], "response": {"format": "json-stat2"}})
        r.raise_for_status()
        doc = r.json()
        for key in ("id", "size", "dimension", "value"):
            if key not in doc:
                raise SystemExit(f"{table}: not a json-stat2 cube, keys {list(doc)}")
        with open(dest, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, ensure_ascii=False)
        print(f"  {dest} {os.path.getsize(dest):,} bytes")


def cube(table, cat_must):
    """{(geo_code, geo_label): {category: count}} for 2024, both sexes, read by dimension LABEL
    (the tables do not agree on dimension order or codes)."""
    path = os.path.join(RAW, TABLES[table].replace(".px", ".json"))
    if not os.path.exists(path):
        raise SystemExit(f"missing {path}; run with --fetch")
    doc = json.load(open(path, encoding="utf-8"))
    ids, size, dim, val = doc["id"], doc["size"], doc["dimension"], doc["value"]
    order = {}
    for k in ids:
        idx = dim[k]["category"]["index"]
        order[k] = sorted(idx, key=lambda c: idx[c]) if isinstance(idx, dict) else list(idx)
    label = {k: {c: str(v).strip() for c, v in dim[k]["category"]["label"].items()} for k in ids}

    def axis(*wanted):
        for k in ids:
            if set(wanted) <= set(label[k].values()):
                return k
        raise SystemExit(f"{table}: no dimension with {wanted}")
    k_geo, k_year, k_cat = axis(COUNTRY), axis(YEAR, "2011"), axis(*cat_must)
    k_sex = next(k for k in ids if k not in (k_geo, k_year, k_cat))
    fixed = {k_year: order[k_year].index(next(c for c in order[k_year] if label[k_year][c] == YEAR)),
             k_sex: order[k_sex].index(next(c for c in order[k_sex] if label[k_sex][c] == "Total"))}
    out = {}
    for gi, gc in enumerate(order[k_geo]):
        row = {}
        for ci, cc in enumerate(order[k_cat]):
            pos = dict(fixed, **{k_geo: gi, k_cat: ci})
            n = 0
            for k, s in zip(ids, size):
                n = n * s + pos[k]
            v = val[n]
            row[label[k_cat][cc]] = 0 if v is None else int(v)
        out[(gc, label[k_geo][gc])] = row
    return out


def main():
    if "--fetch" in sys.argv:
        fetch()
    ok = True

    def say(cond, msg):
        nonlocal ok
        print(("  OK  " if cond else "  BAD ") + msg)
        ok = ok and cond

    mt = cube("mt", ("Albanian", "Turkish", "Romani"))
    eth = cube("eth", ("Albanian", "Serb"))
    est = cube("eth_est", ("Albanian", "Serb"))
    cats = [c for c in next(iter(mt.values())) if c != "Total"]
    print("  categories:", cats)
    nat = next(v for (c, n), v in mt.items() if n == COUNTRY)
    munis = {k: v for k, v in mt.items() if k[1] != COUNTRY}
    say(nat["Total"] == NATIONAL, f"national total {nat['Total']:,} (published {NATIONAL:,})")
    say(len(munis) == N_UNITS, f"{len(munis)} municipalities")
    bad = [n for (c, n), v in munis.items() if sum(v[k] for k in cats) != v["Total"]]
    say(not bad, f"the {len(cats)} categories partition every municipality (off: {bad})")
    bad = [k for k in cats + ["Total"] if sum(v[k] for v in munis.values()) != nat[k]]
    say(not bad, f"the municipalities sum to the national row in every category (off: {bad})")

    # THE MOTHER-TONGUE TABLE ALREADY CARRIES ASK'S ESTIMATE FOR THE NORTH: its totals are the
    # estimated ethnicity table's, not the enumerated one's, in the four northern municipalities
    # and match both everywhere else.
    est_tot = {c: v["Total"] for (c, n), v in est.items()}
    eth_tot = {c: v["Total"] for (c, n), v in eth.items()}
    bad = [n for (c, n), v in munis.items() if est_tot.get(c) != v["Total"]]
    say(not bad, f"every municipality's mother-tongue total equals its total in the ethnicity "
                 f"table with estimation, joined by ASK code (off: {bad})")
    differ = sorted(n for (c, n), v in munis.items() if eth_tot.get(c) != v["Total"])
    say(differ == sorted(NORTH), f"and differs from the enumerated ethnicity total only in {differ}")
    say(eth_tot["0"] == NATIONAL_ENUMERATED, f"enumerated national total {eth_tot['0']:,}")
    en = {n: v for (c, n), v in est.items()}
    print(f"  national, mother tongue against ethnicity: Albanian {nat['Albanian']:,} / "
          f"{en[COUNTRY]['Albanian']:,}; Serbian {nat['Serb']:,} / {en[COUNTRY]['Serb']:,}; "
          f"Bosnian {nat['Bosnian']:,} / {en[COUNTRY]['Bosniak']:,}; Turkish "
          f"{nat['Turkish']:,} / {en[COUNTRY]['Turk']:,}; Romani {nat['Romani']:,} / "
          f"Roma {en[COUNTRY]['Romani']:,} (Ashkali {en[COUNTRY]['Ashkali']:,}, Egyptian "
          f"{en[COUNTRY]['Egyptian']:,}); Gorani {en[COUNTRY]['Gorani']:,}")

    rows = []
    for (code, name), v in mt.items():
        level = "country" if name == COUNTRY else "municipality"
        for k in ["Total"] + cats:
            note = f"level={level}; code={code}"
            if k == "Total":
                note += "; universe total, not a language"
            if name in NORTH:
                note += ("; NORTHERN MUNICIPALITY: the Serb population largely boycotted the 2024 "
                         "census and ASK's published figure includes its own estimate for those "
                         "not enumerated; see xk_north.csv")
            rows.append({"geo_id": code, "geo_level": level, "geo_name": name,
                         "source_category": k, "count": v[k], "tier": "measured", "year": YEAR,
                         "source_id": SOURCE_ID, "note": note})

    # ASK's own estimate for the north
    raw_n = {n: v for (c, n), v in eth.items()}
    est_n = {n: v for (c, n), v in est.items()}
    code_of = {n: c for (c, n) in eth}
    differing = sorted(n for n in raw_n if raw_n[n] != est_n.get(n))
    say(differing == sorted(NORTH + (COUNTRY,)),
        f"the estimated ethnicity table differs from the enumerated one only in {differing}")
    north = []
    for n in NORTH:
        for e in raw_n[n]:
            a, b = raw_n[n][e], est_n[n][e]
            if a == b:
                continue
            north.append({"geo_id": code_of[n], "geo_name": n, "ethnicity": e, "enumerated": a,
                          "estimated": b, "added": b - a, "source_id": SOURCE_ID,
                          "note": "ASK's own estimate (census2024_63) for a municipality the "
                                  "2024 census did not reach"})
    add_tot = sum(r["added"] for r in north if r["ethnicity"] == "Total")
    add_serb = sum(r["added"] for r in north if r["ethnicity"] == "Serb")
    say(add_serb > 0.9 * add_tot, f"the estimate adds {add_tot:,} people in the north, "
                                  f"{add_serb:,} Serbs ({add_serb / add_tot:.1%})")
    for n in NORTH:
        v = munis[(code_of[n], n)]
        print(f"    {n}: published mother-tongue total (with the estimate) {v['Total']:,} (Serbian {v['Serb']:,}, "
              f"Albanian {v['Albanian']:,}, Bosnian {v['Bosnian']:,}); estimate adds "
              f"{next(r['added'] for r in north if r['geo_name'] == n and r['ethnicity'] == 'Total'):,}")
    if not ok:
        raise SystemExit("checks failed; nothing written")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    with open(NORTH_OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=NORTH_COLUMNS)
        w.writeheader()
        w.writerows(north)
    print(f"  wrote {OUT} ({len(rows)} rows) and {NORTH_OUT} ({len(north)} rows)")


if __name__ == "__main__":
    main()
