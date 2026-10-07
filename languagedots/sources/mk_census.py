"""North Macedonia: State Statistical Office (SSO), Popis 2021, mother tongue by municipality.

    python sources/mk_census.py --fetch    three PxWeb POSTs, seconds
    python sources/mk_census.py            normalise from data/raw/mk/

-> data/normalized/mk.csv (levels `country` and `municipality`; only `municipality` is drawn)

Tables, all in MakStat's PxWeb (https://makstat.stat.gov.mk/pxweb/api/v1/<lang>/MakStat/...):

  Popisi/Popis2021/NaselenieVkupno/NaseleniePopis2021/EtnoKulturniKarakteristiki/T1015P21.px
      mother tongue x sex x 82 geographies (the country, the City of Skopje, 80 municipalities).
      11 categories: 7 named languages, sign language, other, unknown, and code 88, "persons for
      whom data are taken from administrative sources". Fetched in English (category labels,
      the taxonomy keys) and Macedonian (municipality names). The drawn table.
  Popisi/Popis2021/NaselenieSet/T1013P21.px
      mother tongue x sex, national only, 39 categories: the same 7 named languages and 28 more
      that the municipal table folds into `Other languages not mentioned`. The second-table
      check (its 7 shared languages must equal the municipal table's national row, and its extra
      languages plus its own `Other` must equal the municipal `Other`), and the record's
      breakdown of what `other` holds. Not drawn.

The City of Skopje (code 0019) is the sum of its ten municipalities, which are also rows; it is
dropped, as religiondots' sources/mk.py drops it from the religion table of the same census.

CHECKS: national total 1,836,713; the 80 municipalities sum to the national row in every
category exactly; the 11 categories partition every municipality; every municipality's total
equals religiondots' total for the same PxWeb code from the religion table (T1012P21) of the same
census, which is a join check because the geography codes are shared; and the national table.
"""

import csv
import json
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "mk")
OUT = os.path.join(ROOT, "data", "normalized", "mk.csv")
RD_NORM = os.path.join(os.path.dirname(ROOT), "religiondots", "data", "normalized", "mk.csv")

SOURCE_ID = "mk_popis_2021_mt"
YEAR = 2021
COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count", "tier", "year",
           "source_id", "note"]

BASE = "https://makstat.stat.gov.mk/pxweb/api/v1/{lang}/MakStat/Popisi/Popis2021/"
MUNI = "NaselenieVkupno/NaseleniePopis2021/EtnoKulturniKarakteristiki/T1015P21.px"
NAT = "NaselenieSet/T1013P21.px"
FILES = {
    "T1015P21_en.json": ("en", MUNI),
    "T1015P21_mk.json": ("mk", MUNI),
    "T1013P21_en.json": ("en", NAT),
}

NATIONAL = 1_836_713          # resident population, Popis 2021
NATIONAL_CODE = "0000"
SKOPJE_CODE = "0019"
TOTAL_CAT = "00"
OTHER = "Other languages not mentioned"
N_MUNI = 80


def fetch():
    import requests
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    h = {"User-Agent": "Mozilla/5.0"}
    os.makedirs(RAW, exist_ok=True)
    for name, (lang, path) in FILES.items():
        dest = os.path.join(RAW, name)
        if os.path.exists(dest) and os.path.getsize(dest) > 2_000:
            print("already have", dest)
            continue
        url = BASE.format(lang=lang) + path
        meta = requests.get(url, timeout=120, verify=False, headers=h).json()
        query = {"query": [{"code": v["code"], "selection": {"filter": "all", "values": ["*"]}}
                           for v in meta["variables"]],
                 "response": {"format": "json-stat2"}}
        print("POST", url)
        r = requests.post(url, json=query, timeout=300, verify=False, headers=h)
        r.raise_for_status()
        doc = r.json()
        if "value" not in doc or "dimension" not in doc:
            raise SystemExit(f"{name}: not a json-stat2 cube, keys {list(doc)}")
        with open(dest, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, ensure_ascii=False)
        print(f"  {os.path.getsize(dest):,} bytes")


def _load(name):
    p = os.path.join(RAW, name)
    if not os.path.exists(p):
        raise SystemExit(f"missing {p} -- run with --fetch first")
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def _cube(doc):
    """json-stat2: a flat row-major array over doc['id']. Returns (ids, codes, labels, at)."""
    ids, sizes = doc["id"], doc["size"]
    codes, labels = {}, {}
    for d in ids:
        cat = doc["dimension"][d]["category"]
        idx = cat["index"]
        order = sorted(idx, key=lambda k: idx[k]) if isinstance(idx, dict) else list(idx)
        codes[d] = order
        labels[d] = {k: cat.get("label", {}).get(k, k) for k in order}
    for d, n in zip(ids, sizes):
        if len(codes[d]) != n:
            raise SystemExit(f"dimension {d}: {len(codes[d])} codes but size {n}")
    stride, acc = {}, 1
    for d, n in zip(reversed(ids), reversed(sizes)):
        stride[d] = acc
        acc *= n
    pos = {d: {c: i for i, c in enumerate(codes[d])} for d in ids}

    def at(**kw):
        return doc["value"][sum(stride[d] * pos[d][kw[d]] for d in ids)]
    return ids, codes, labels, at


def _dims(ids, codes, want):
    """Identify dimensions by their size, not their (Macedonian) names. `want`: {size: role}."""
    found = {}
    for d in ids:
        role = want.get(len(codes[d]))
        if role:
            found[role] = d
    if set(found) != set(want.values()):
        raise SystemExit(f"could not identify {set(want.values()) - set(found)} among "
                         f"{[(d, len(codes[d])) for d in ids]}")
    return found


def _sex_total(codes, labels, d):
    c = codes[d][0]
    if "total" not in labels[d][c].lower():
        raise SystemExit(f"first sex value is {labels[d][c]!r}, expected the total")
    return c


def read():
    en, mk = _load("T1015P21_en.json"), _load("T1015P21_mk.json")
    ids, codes, lab, at = _cube(en)
    ids_mk, codes_mk, lab_mk, _ = _cube(mk)
    dim = _dims(ids, codes, {82: "geo", 12: "cat", 3: "sex"})
    gmk = _dims(ids_mk, codes_mk, {82: "geo", 12: "cat", 3: "sex"})["geo"]
    if codes[dim["geo"]] != codes_mk[gmk]:
        raise SystemExit("the two language editions order the municipalities differently")
    sex = _sex_total(codes, lab, dim["sex"])
    rows = []
    for g in codes[dim["geo"]]:
        if g == SKOPJE_CODE:
            continue                     # the sum of the ten Skopje municipalities below it
        level = "country" if g == NATIONAL_CODE else "municipality"
        for c in codes[dim["cat"]]:
            n = at(**{dim["geo"]: g, dim["cat"]: c, dim["sex"]: sex})
            if n is None:
                continue
            label = lab[dim["cat"]][c].strip()
            note = f"level={level}; code={c}"
            if c == TOTAL_CAT:
                label = "Total"
                note += "; universe total, not a language"
            rows.append({"geo_id": g, "geo_level": level, "geo_name": lab_mk[gmk][g].strip(),
                         "source_category": label, "count": int(n), "tier": "measured",
                         "year": YEAR, "source_id": SOURCE_ID, "note": note})
    return rows


def national_table():
    doc = _load("T1013P21_en.json")
    ids, codes, lab, at = _cube(doc)
    dim = _dims(ids, codes, {40: "cat", 3: "sex"})
    sex = _sex_total(codes, lab, dim["sex"])
    return {lab[dim["cat"]][c].strip(): int(at(**{dim["cat"]: c, dim["sex"]: sex}) or 0)
            for c in codes[dim["cat"]]}


def check(rows):
    ok = True
    muni = [r for r in rows if r["geo_level"] == "municipality"]
    ids = {r["geo_id"] for r in muni}
    good = len(ids) == N_MUNI
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} {len(ids)} municipalities (expected {N_MUNI})")

    nat = {r["source_category"]: r["count"] for r in rows if r["geo_level"] == "country"}
    good = nat.get("Total") == NATIONAL
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} national total {nat.get('Total'):,} (published {NATIONAL:,})")

    bad = 0
    for label, v in nat.items():
        s = sum(r["count"] for r in muni if r["source_category"] == label)
        if s != v:
            bad += 1
            print(f"  BAD {label}: municipalities {s:,} vs national {v:,}")
    ok &= bad == 0
    print(f"  {'OK ' if not bad else 'BAD'} all {len(nat)} categories sum from the "
          f"{N_MUNI} municipalities to the national row exactly")

    tot, parts = {}, {}
    for r in muni:
        d = tot if r["source_category"] == "Total" else parts
        d[r["geo_id"]] = d.get(r["geo_id"], 0) + r["count"]
    off = [g for g in tot if tot[g] != parts.get(g)]
    ok &= not off
    print(f"  {'OK ' if not off else 'BAD'} the 11 categories partition every municipality"
          + (f" (off: {off[:5]})" if off else ""))

    # join check against the religion table of the same census, read only
    if os.path.exists(RD_NORM):
        rd = {}
        with open(RD_NORM, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if r["geo_level"] == "municipality" and r["note"].endswith("not a religion category"):
                    rd[r["geo_id"]] = int(r["count"])
        diff = sorted(set(rd) ^ set(tot)) + [g for g in tot if g in rd and rd[g] != tot[g]]
        ok &= not diff and len(rd) == N_MUNI
        print(f"  {'OK ' if not diff else 'BAD'} every municipality's total equals religiondots' "
              f"religion table (T1012P21) for the same code ({len(rd)} codes)"
              + (f": {diff[:5]}" if diff else ""))
    else:
        print("  -- religiondots' mk.csv not found; join check skipped")

    # the national 39-category table
    t13 = national_table()
    shared = [k for k in nat if k != "Total" and k != OTHER]
    off = [k for k in shared if t13.get(k) != nat[k]]
    good = not off and t13.get("Mother tongue - TOTAL") == NATIONAL
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} the national table T1013P21 agrees on the total and the "
          f"{len(shared)} shared categories" + (f" (off: {off})" if off else ""))
    extra = {k: v for k, v in t13.items() if (k not in nat or k == OTHER)
             and k != "Mother tongue - TOTAL"}
    s = sum(extra.values())
    good = s == nat[OTHER]
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} its {len(extra) - 1} extra languages and its own `other` "
          f"sum to the municipal table's `{OTHER}` ({s:,} vs {nat[OTHER]:,}). They are:")
    for k, v in sorted(extra.items(), key=lambda kv: -kv[1]):
        print(f"      {v:>7,}  {k}")

    print("\n  Categories, national:")
    for label, n in sorted(nat.items(), key=lambda kv: -kv[1]):
        print(f"    {n:>10,}  {100.0 * n / NATIONAL:5.2f}%  {label}")
    if not ok:
        raise SystemExit("reconciliation FAILED")


def main():
    if "--fetch" in sys.argv:
        fetch()
    rows = read()
    check(rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print("\nwrote", OUT, f"({len(rows):,} rows)")


if __name__ == "__main__":
    main()
