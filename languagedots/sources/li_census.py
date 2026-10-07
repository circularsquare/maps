"""Liechtenstein: Volkszählung 2020, main language (Hauptsprache) by Gemeinde (Amt für Statistik).

Reads (or fetches) data/raw/li/ and writes data/normalized/li.csv. The record is sources/li.md.

One PxWeb table on the office's eTab server, **213.011d** "Ständige Bevölkerung nach Stichtag,
Hauptsprache, Geschlecht und Gemeinde": 34 language categories plus a total, over the 11
Gemeinden plus the country, for 31.12.2020, 2015 and 2010. No key, no wall. The English tree
(213.011e) stops at 2015; the German one carries 2020, so this reads the German.

Server quirks, learned on the same server by religiondots (sources/li.py there):
- dimension values are positional indices, not codes;
- `filter: "all"` with `["*"]` silently returns one cell, so every selection is an explicit list;
- json-stat2 declares the full shape and returns one value. This reads json-stat (v1), and
  checks the value count against the declared size.

Cell status: `-` is a true zero (no disclosure threshold: single people are published). `..`
(not available) occurs only for "Übrige westeuropäische" and "Übrige osteuropäische Sprachen" in
2020, two categories the 2020 release does not fill; read() asserts that and that the remaining
categories still sum to each Gemeinde's total, so nothing was lost with them.

Checks:
- 34 categories sum to each Gemeinde's total, exactly;
- 11 Gemeinden sum to the national row, category by category, exactly;
- the national total is 39,055.

Usage:
    python sources/li_census.py --fetch    one POST, ~30 KB
    python sources/li_census.py            normalise from data/raw/li/
"""
import json
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "li")
OUT = os.path.join(ROOT, "data", "normalized", "li.csv")
BOOK = os.path.join(RAW, "213.011d.json")

API = ("https://etab.llv.li/PXWEb/api/v1/de/eTab/Bev%C3%B6lkerung/"
       "Bev%C3%B6lkerungsstruktur/213.011d.px")
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

SOURCE_ID = "li_volkszaehlung_2020_213011d"
YEAR = "31.12.2020"
N_DATES, N_LANG, N_GEM = 3, 35, 12
TOTAL = "Hauptsprache - Total"
COUNTRY = "Liechtenstein"
NATIONAL = 39_055
NOT_FILLED_2020 = {"Übrige westeuropäische Sprachen", "Übrige osteuropäische Sprachen"}


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    q = {"query": [
        {"code": "Stichtag", "selection": {"filter": "item",
                                           "values": [str(i) for i in range(N_DATES)]}},
        {"code": "Hauptsprache", "selection": {"filter": "item",
                                               "values": [str(i) for i in range(N_LANG)]}},
        {"code": "Geschlecht", "selection": {"filter": "item", "values": ["0"]}},
        {"code": "Gemeinde", "selection": {"filter": "item",
                                           "values": [str(i) for i in range(N_GEM)]}}],
        "response": {"format": "json-stat"}}
    print("POST", API)
    r = requests.post(API, json=q, timeout=120, headers=UA)
    r.raise_for_status()
    ds = json.loads(r.content.decode("utf-8-sig"))["dataset"]
    want = N_DATES * N_LANG * N_GEM
    if len(ds["value"]) != want:
        raise SystemExit(f"declared {ds['dimension']['size']}, got {len(ds['value'])} values")
    with open(BOOK, "w", encoding="utf-8") as fh:
        json.dump(ds, fh, ensure_ascii=False)
    print(f"  {os.path.getsize(BOOK):,} bytes")


def _labels(ds, dim):
    c = ds["dimension"][dim]["category"]
    return [c["label"][k] for k in sorted(c["index"], key=c["index"].get)]


def read():
    """{date: {gemeinde: {category: count or None}}}, None for `..`."""
    with open(BOOK, encoding="utf-8") as fh:
        ds = json.load(fh)
    dims = ds["dimension"]["id"]
    if dims != ["Stichtag", "Hauptsprache", "Geschlecht", "Gemeinde"]:
        raise SystemExit(f"unexpected dimensions {dims}")
    dates, langs, gems = _labels(ds, "Stichtag"), _labels(ds, "Hauptsprache"), _labels(ds, "Gemeinde")
    status = ds.get("status", {})
    out, i = {}, 0
    for d in dates:
        for lang in langs:
            for g in gems:
                v, s = ds["value"][i], status.get(str(i))
                if s == "..":
                    n = None
                elif v is None:
                    if s != "-":
                        raise SystemExit(f"cell {i} empty with status {s!r}")
                    n = 0
                else:
                    n = int(v)
                out.setdefault(d, {}).setdefault(g, {})[lang] = n
                i += 1
    return out


def check(tab):
    t = tab[YEAR]
    gems = [g for g in t if g != COUNTRY]
    assert len(gems) == 11, gems
    cats = [c for c in t[COUNTRY] if c != TOTAL]
    unfilled = {c for c in cats if t[COUNTRY][c] is None}
    assert unfilled == NOT_FILLED_2020, unfilled
    for g in [COUNTRY] + gems:
        s = sum(t[g][c] or 0 for c in cats)
        assert s == t[g][TOTAL], (g, s, t[g][TOTAL])
        assert all(t[g][c] is None for c in unfilled)
    for c in cats:
        if c in unfilled:
            continue
        s = sum(t[g][c] for g in gems)
        assert s == t[COUNTRY][c], (c, s, t[COUNTRY][c])
    assert t[COUNTRY][TOTAL] == NATIONAL, t[COUNTRY][TOTAL]
    print(f"  ok: 11 Gemeinden, {len(cats) - len(unfilled)} categories, partition exact both "
          f"ways, national {NATIONAL:,}")
    # the earlier rounds, for the record
    for d in tab:
        n = tab[d][COUNTRY]
        de = n["Deutsch"] / n[TOTAL]
        print(f"  {d}: total {n[TOTAL]:,}, Deutsch {de:.1%}, "
              f"top others {sorted(((v or 0, c) for c, v in n.items() if c not in (TOTAL, 'Deutsch')), reverse=True)[:5]}")


def main():
    if "--fetch" in sys.argv:
        fetch()
    tab = read()
    check(tab)
    import csv
    t = tab[YEAR]
    rows = []
    for g, cats in t.items():
        if g == COUNTRY:
            continue
        for c, n in cats.items():
            if c == TOTAL or n is None:
                continue
            rows.append([g, "gemeinde", g, c, n, "measured", 2020, SOURCE_ID])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["geo_id", "geo_level", "geo_name", "source_category", "count", "tier",
                    "year", "source_id"])
        w.writerows(rows)
    print(f"wrote {OUT}: {len(rows)} rows, {sum(r[4] for r in rows):,} people")


if __name__ == "__main__":
    main()
