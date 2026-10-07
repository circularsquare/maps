"""Latvia: Tautas skaitīšana 2011 (CSB), language mostly spoken at home, by municipality.

    python sources/lv_census.py --fetch    four PxWeb POSTs (about 1 MB) if missing
    python sources/lv_census.py            normalise from data/raw/lv/

-> data/normalized/lv.csv (levels `country`, `region`, `municipality`; alternatives, never summed;
   only `municipality` is drawn)

WHY 2011. Latvia's 2021 census was built from registers and asked nothing about language (CSB;
the coverage sweep, 2026-10-03). The 2011 census is the last count of everyone's language at any
geography; the Adult Education Survey 2022 asks mother tongue and home language, but for ages
18-69 and nationally (with Latgale picked out for Latgalian only).

THE TABLE. PxWeb `OSP_OD/tautassk/taut/tsk2011/TSG11-07.px` on data.stat.gov.lv, "Resident
population in statistical regions, cities under state jurisdiction and counties by sex, language
mostly spoken at home and age group; on 1 March 2011". Question: `mājās pārsvarā lietotā valoda`,
the language mostly spoken at home, one answer. Seven answers (Latvian, Russian, Belarusian,
Ukrainian, Polish, Lithuanian, other) and a total.

THE TOTAL IS NOT THE POPULATION. TSG11-07's total is exactly the sum of its seven answers,
1,876,812, against the census population of 2,070,371: it counts only the people whose home
language is known. The rest, 193,559 (9.35%), are taken here per municipality as the population
in TSG11-060 (ethnicity by territory, whose total is the full population) less TSG11-07's total,
and written as `Not stated (population less those with a home language)`; it is not drawn.

Territory: Latvia, the 6 statistical regions and the 119 municipalities of 2011 (9 republican
cities and 110 novadi), coded LV + the seven-digit ATVK code, which is also GISCO's LAU_ID for the
pre-2021 LAUs that religiondots' lv_lau.gpkg holds.

THE CHECKS. The population against 2,070,371 (the published 2011 census population); the
language total against the sum of the seven answers in every unit; level counts (1, 6, 119);
regions and municipalities each sum to the country in every column; and two other tables of the
same census, which cross home language with ethnicity (TSG11-071) and with daily use of Latgalian
(TSG11-08), must equal TSG11-07 cell for cell when summed over their extra dimension, in every
unit. TSG11-08 also gives the Latgalian figures sources/lv.md quotes (they are not drawn;
sources/lv.md says why).
"""

import csv
import json
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "lv")
OUT = os.path.join(ROOT, "data", "normalized", "lv.csv")

SOURCE_ID = "lv_tsk2011_homelang"
YEAR = 2011
API = "https://data.stat.gov.lv/api/v1/en/OSP_OD/tautassk/taut/tsk2011/{}"

GEO = "Teritoriālā vienība"
LANG = "Mājās pārsvarā lietotā valoda"
MEAS = "Skaits, īpatsvars"
SEX = "Dzimums"
AGE = "Vecums (pilni gadi)"
ETH = "Tautība"
LTG = "Latgaliešu valodas lietošana ikdienā"
GEO060 = "Statistiskā reģiona un administratīvās teritorijas nosaukums"   # TSG11-060's own name

# table -> {variable: values to ask for, None meaning all}
TABLES = {
    "TSG11-07.px": {GEO: None, SEX: ["T"], LANG: None, MEAS: ["NUMB"], AGE: ["TOTAL"]},
    "TSG11-071.px": {GEO: None, ETH: None, MEAS: ["NUMB"], LANG: None},
    "TSG11-08.px": {GEO: None, AGE: ["TOTAL"], LTG: None, MEAS: ["NUMB"], LANG: None},
    "TSG11-060.px": {GEO060: None, ETH: ["0"]},
}

COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count", "tier", "year",
           "source_id", "note"]
POPULATION = 2_070_371
EXPECTED = {"country": 1, "region": 6, "municipality": 119}
NOT_STATED = "Not stated (population less those with a home language)"


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    hdr = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    for table, sel in TABLES.items():
        path = os.path.join(RAW, table.replace(".px", ".json"))
        meta_path = os.path.join(RAW, table.replace(".px", "_meta.json"))
        if os.path.exists(path) and os.path.getsize(path) > 1_000 and os.path.exists(meta_path):
            print("already have", path)
            continue
        url = API.format(table)
        meta = requests.get(url, timeout=120, headers=hdr)
        meta.raise_for_status()
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(meta.json(), fh, ensure_ascii=False)
        query = []
        for code, vals in sel.items():
            if vals is None:
                query.append({"code": code, "selection": {"filter": "all", "values": ["*"]}})
            else:
                query.append({"code": code, "selection": {"filter": "item", "values": vals}})
        print("POST", url)
        r = requests.post(url, json={"query": query, "response": {"format": "json"}},
                          timeout=300, headers=hdr)
        r.raise_for_status()
        doc = r.json()
        if "data" not in doc or "columns" not in doc:
            raise SystemExit(f"{table}: unexpected response keys {list(doc)}")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, ensure_ascii=False)
        print(f"  {os.path.getsize(path):,} bytes, {len(doc['data']):,} cells")


def _load(table):
    path = os.path.join(RAW, table.replace(".px", ".json"))
    meta_path = os.path.join(RAW, table.replace(".px", "_meta.json"))
    if not os.path.exists(path) or not os.path.exists(meta_path):
        raise SystemExit(f"missing {path} -- run with --fetch first")
    with open(path, encoding="utf-8") as fh:
        doc = json.load(fh)
    with open(meta_path, encoding="utf-8") as fh:
        meta = json.load(fh)
    return doc, meta


def cells(table, keep):
    """{tuple of the `keep` variables' codes: count} from a PxWeb json response."""
    doc, _ = _load(table)
    codes = [c["code"] for c in doc["columns"] if c.get("type") != "c"]
    idx = [codes.index(k) for k in keep]
    out = {}
    for d in doc["data"]:
        key = tuple(d["key"][i] for i in idx)
        v = d["values"][0]
        if v == "-":
            v = 0           # PxWeb's dash: nothing to count
        if v in ("..", "", None):
            raise SystemExit(f"{table} {d['key']}: value {v!r}, not a count")
        if key in out:
            raise SystemExit(f"{table}: duplicate {key}; a fixed dimension was not fixed")
        out[key] = int(float(v))
    return out


def names(table, var):
    _, meta = _load(table)
    v = next(x for x in meta["variables"] if x["code"] == var)
    return dict(zip(v["values"], v["valueTexts"]))


def level(code):
    if code == "LV":
        return "country"
    if len(code) == 5:
        return "region"
    if len(code) == 9:
        return "municipality"
    raise SystemExit(f"unexpected territorial code {code!r}")


def main():
    if "--fetch" in sys.argv:
        fetch()
    ok = True
    geo_names = names("TSG11-07.px", GEO)
    lang_names = names("TSG11-07.px", LANG)
    langs = [c for c in lang_names if c != "TOTAL"]
    t07 = cells("TSG11-07.px", [GEO, LANG])
    popn = {g: n for (g,), n in cells("TSG11-060.px", [GEO060]).items()}
    if set(popn) != set(geo_names):
        raise SystemExit("TSG11-060 and TSG11-07 list different territorial units")

    rows, by, bad_total = [], {}, []
    for g, gname in geo_names.items():
        lv = level(g)
        unit_rows = []
        named = 0
        for c in langs:
            n = t07[(g, c)]
            named += n
            unit_rows.append(dict(geo_id=g, geo_level=lv, geo_name=gname,
                                  source_category=lang_names[c], count=n, tier="measured",
                                  year=YEAR, source_id=SOURCE_ID, note=f"level={lv}; code={c}"))
        if named != t07[(g, "TOTAL")]:
            bad_total.append(g)
        ns = popn[g] - named
        if ns < 0:
            raise SystemExit(f"{g}: the languages exceed the population by {-ns}")
        unit_rows.insert(0, dict(geo_id=g, geo_level=lv, geo_name=gname,
                                 source_category="Population", count=popn[g], tier="measured",
                                 year=YEAR, source_id=SOURCE_ID,
                                 note=f"level={lv}; TSG11-060 total, not a language"))
        unit_rows.append(dict(geo_id=g, geo_level=lv, geo_name=gname,
                              source_category=NOT_STATED, count=ns, tier="measured", year=YEAR,
                              source_id=SOURCE_ID,
                              note=f"level={lv}; TSG11-060 population less TSG11-07 total"))
        rows += unit_rows
        by.setdefault(lv, {})[g] = {r["source_category"]: r["count"] for r in unit_rows}

    good = not bad_total
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} TSG11-07's total is the sum of its seven answers in "
          f"every unit {bad_total[:5] if bad_total else ''}")
    for lv, want in EXPECTED.items():
        got = len(by.get(lv, {}))
        good = got == want
        ok &= good
        print(f"  {'OK ' if good else 'BAD'} {lv:<13} {got:>3} units (expected {want})")
    nat = by["country"]["LV"]
    good = nat["Population"] == POPULATION
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} population {nat['Population']:,} "
          f"(published {POPULATION:,})")
    for lv in ("region", "municipality"):
        bad = [cat for cat in nat if sum(u[cat] for u in by[lv].values()) != nat[cat]]
        ok &= not bad
        print(f"  {'OK ' if not bad else 'BAD'} {lv:<13} sum to the country in every column "
              f"{bad or ''}")

    # the same census's two other home-language tables, summed over their extra dimension
    for table, extra in (("TSG11-071.px", ETH), ("TSG11-08.px", LTG)):
        t = cells(table, [GEO, extra, LANG])
        ext_vals = sorted({k[1] for k in t} - {"TOTAL"})
        n_cmp, bad_tot, bad_sum = 0, [], []
        for g in geo_names:
            for c in lang_names:
                n_cmp += 1
                if t[(g, "TOTAL", c)] != t07[(g, c)]:
                    bad_tot.append((g, c, t[(g, "TOTAL", c)], t07[(g, c)]))
                s = sum(t[(g, e, c)] for e in ext_vals)
                if s != t07[(g, c)]:
                    bad_sum.append((g, c, s, t07[(g, c)]))
        if table == "TSG11-08.px":
            # TSG11-08 is 5 people short nationally (1,876,807; Latvian -4, Russian -1): Riga
            # city 2 (one Latvian, one Russian at home), and one Latvian-at-home person in each
            # of the Pierīga, Vidzeme and Zemgale REGION rows, whose municipalities match
            # exactly. TSG11-071 agrees with the drawn table exactly, so TSG11-08 is held to 2
            # people per cell and 5 nationally; nothing drawn comes from it.
            muni_short = sum(b[3] - b[2] for b in bad_tot
                             if level(b[0]) == "municipality" and b[1] == "TOTAL")
            print(f"    TSG11-08 short by {muni_short} people over the municipalities, in "
                  f"{sum(1 for b in bad_tot if level(b[0]) == 'municipality')} cells")
            ok &= muni_short <= 5
            for lst in (bad_tot, bad_sum):
                lst[:] = [b for b in lst if not 0 < b[3] - b[2] <= (5 if b[0] == "LV" else 2)]
        good = not bad_tot and not bad_sum
        ok &= good
        print(f"  {'OK ' if good else 'BAD'} {table}: its total and the sum over {len(ext_vals)} "
              f"values of {extra!r} equal TSG11-07 in all {n_cmp} (unit, language) cells"
              f"{' (within the tolerance above)' if table == 'TSG11-08.px' else ''} "
              f"{(bad_tot[:3], bad_sum[:3]) if not good else ''}")

    # Latgalian, for the record only
    t08 = cells("TSG11-08.px", [GEO, LTG, LANG])
    print("\n  Latgalian used daily (TSG11-08), of those with a home language:")
    for g in ("LV", "LV005", "LV0210000", "LV0010000"):
        y = t08[(g, "Y", "TOTAL")]
        print(f"    {geo_names[g]:<16} {y:>8,} of {t07[(g, 'TOTAL')]:>9,} "
              f"({100 * y / t07[(g, 'TOTAL')]:.1f}%); Latvian at home {t08[(g, 'Y', 'LAV')]:,}, "
              f"Russian {t08[(g, 'Y', 'RUS')]:,}, other answers "
              f"{y - t08[(g, 'Y', 'LAV')] - t08[(g, 'Y', 'RUS')]:,}")

    print("\n  national categories (share of the population):")
    for label, n in sorted(nat.items(), key=lambda kv: -kv[1]):
        print(f"    {n:>10,}  {100.0 * n / POPULATION:6.2f}%  {label}")
    known = POPULATION - nat[NOT_STATED]
    print("  of those with a home language: "
          + ", ".join(f"{lang_names[c]} {100 * nat[lang_names[c]] / known:.2f}%" for c in langs))
    for lv in ("region", "municipality"):
        u = by[lv]
        ns = sorted(((v[NOT_STATED] / v["Population"], g) for g, v in u.items()), reverse=True)
        print(f"  not stated by {lv}, highest: "
              + ", ".join(f"{geo_names[g]} {100 * r:.1f}%" for r, g in ns[:5])
              + "; lowest: "
              + ", ".join(f"{geo_names[g]} {100 * r:.1f}%" for r, g in ns[-3:]))
    if not ok:
        raise SystemExit("reconciliation FAILED")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print("\nwrote", OUT, f"({len(rows):,} rows)")


if __name__ == "__main__":
    main()
