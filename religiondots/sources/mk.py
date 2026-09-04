"""North Macedonia — State Statistical Office, Popis 2021, religion by municipality.

Reads (or fetches) data/raw/mk/ and writes data/normalized/mk.csv.

One PxWeb table, **T1012P21**: 14 religious affiliations x 80 municipalities, 1.84M people.
Found by walking `/pxweb/api/v1/en/MakStat` — §12's "try a PxWeb API before anything else",
and it took minutes where Slovakia has now taken two sessions and is still not found.

**80 units for 1.84M people is coarse**, about the grain of Estonia's 79. It is also the
ceiling: the census publishes ethnicity by SETTLEMENT (T1503P21, 1,700-odd units) but
religion only by municipality, which is §3.9's trade made by the office rather than by us.
Refining religion inside a municipality from the settlement ETHNICITY table would be
exactly the move §14.4 forbids — the correlation here is near-total and the map would stop
being a religion map — so the coarse geography stands.

TWO CATEGORIES ARE NOT RELIGIONS AND ONE OF THEM IS BIG. The 2021 census enumerated part of
the population from administrative registers rather than in person, and those people get
their own category (code 88) with no religion attached. It is a coverage residual in the
shape of a category, it is 9.6% of the country, and mistaking it for a religion — or for
irreligion — would be the single worst error available here. See sources/mk.md §3.

Usage:
    python sources/mk.py --fetch    two PxWeb POSTs, seconds
    python sources/mk.py            normalise from data/raw/mk/
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

SOURCE_ID = "mk_popis_2021"
YEAR = 2021
BASIS = "self_id"

COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count",
           "basis", "year", "source_id", "note"]

API = ("https://makstat.stat.gov.mk/pxweb/api/v1/{lang}/MakStat/Popisi/Popis2021/"
       "NaselenieVkupno/NaseleniePopis2021/EtnoKulturniKarakteristiki/T1012P21.px")
# English for the category labels (the taxonomy keys), Macedonian for the municipality
# names, because GISCO carries MK in CYRILLIC and that is what the join needs.
LANGS = {"en": "T1012P21_en.json", "mk": "T1012P21_mk.json"}

NATIONAL = 1_836_713          # resident population, Popis 2021

NATIONAL_CODE = "0000"        # Republic of North Macedonia
SKOPJE_CODE = "0019"          # City of Skopje: the 10 Skopje municipalities summed
TOTAL_CAT = "00"              # Religious affiliation - TOTAL
ADMIN_CAT = "88"              # persons taken from administrative sources

EXPECTED_MUNICIPALITIES = 80
EXPECTED_CATEGORIES = 14


def fetch():
    import requests
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    os.makedirs(RAW, exist_ok=True)
    for lang, name in LANGS.items():
        dest = os.path.join(RAW, name)
        if os.path.exists(dest) and os.path.getsize(dest) > 10_000:
            print("already have", dest)
            continue
        url = API.format(lang=lang)
        meta = requests.get(url, timeout=120, verify=False,
                            headers={"User-Agent": "Mozilla/5.0"}).json()
        codes = [v["code"] for v in meta["variables"]]
        query = {"query": [{"code": c,
                            "selection": {"filter": "all", "values": ["*"]}}
                           for c in codes],
                 "response": {"format": "json-stat2"}}
        print("POST", url)
        r = requests.post(url, json=query, timeout=300, verify=False,
                          headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        doc = r.json()
        # §5a: HTTP 200 is not a download. Assert it is really the cube we asked for.
        if "value" not in doc or "dimension" not in doc:
            raise SystemExit(f"{lang}: not a json-stat2 cube, keys {list(doc)}")
        with open(dest, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, ensure_ascii=False)
        print(f"  {os.path.getsize(dest):,} bytes")


def _load(name):
    p = os.path.join(RAW, name)
    if not os.path.exists(p):
        raise SystemExit(f"missing {p} -- run with --fetch first")
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def _axes(doc):
    """json-stat2 is a FLAT cube in row-major order over `id`, not a table (§12).

    Returns (ids, sizes, {dim: [codes]}, {dim: {code: label}}).
    """
    ids = doc["id"]
    sizes = doc["size"]
    codes, labels = {}, {}
    for d in ids:
        cat = doc["dimension"][d]["category"]
        idx = cat["index"]
        if isinstance(idx, dict):
            order = sorted(idx, key=lambda k: idx[k])
        else:
            order = list(idx)
        codes[d] = order
        labels[d] = {k: cat.get("label", {}).get(k, k) for k in order}
    if len(sizes) != len(ids):
        raise SystemExit("json-stat2 size/id mismatch")
    for d, n in zip(ids, sizes):
        if len(codes[d]) != n:
            raise SystemExit(f"dimension {d}: {len(codes[d])} codes but size {n}")
    return ids, sizes, codes, labels


def _dims(ids, codes):
    """Identify the three dimensions by their CONTENT, not by their names.

    The dimension ids are Macedonian in both language editions ('Општина'), so matching on
    the name would work here and break the moment SSO translates them. Sizes are unique.
    """
    found = {}
    for d in ids:
        n = len(codes[d])
        if n == 82:
            found["geo"] = d
        elif n == EXPECTED_CATEGORIES:
            found["cat"] = d
        elif n == 3:
            found["sex"] = d
    missing = {"geo", "cat", "sex"} - set(found)
    if missing:
        raise SystemExit(f"could not identify dimensions {missing} among "
                         f"{[(d, len(codes[d])) for d in ids]}")
    return found


def read():
    en, mk = _load(LANGS["en"]), _load(LANGS["mk"])
    ids, sizes, codes, labels_en = _axes(en)
    ids_mk, _, codes_mk, labels_mk = _axes(mk)
    dim = _dims(ids, codes)

    if codes[dim["geo"]] != codes_mk[ids_mk[ids.index(dim["geo"])]]:
        raise SystemExit("the two language editions order the municipalities differently")

    # strides for row-major order
    stride, acc = {}, 1
    for d, n in zip(reversed(ids), reversed(sizes)):
        stride[d] = acc
        acc *= n
    values = en["value"]

    def at(**pos):
        i = sum(stride[d] * codes[d].index(pos[d]) for d in ids)
        return values[i]

    sex_total = codes[dim["sex"]][0]     # 'Sex - TOTAL' is first in the source's order
    if "total" not in labels_en[dim["sex"]][sex_total].lower():
        raise SystemExit(f"first sex value is {labels_en[dim['sex']][sex_total]!r}, "
                         "expected the total — SSO reordered the dimension")

    geo_labels_mk = labels_mk[ids_mk[ids.index(dim["geo"])]]
    rows = []
    for g in codes[dim["geo"]]:
        if g == SKOPJE_CODE:
            continue                     # an aggregate of ten municipalities below it
        level = "country" if g == NATIONAL_CODE else "municipality"
        name = geo_labels_mk[g]
        for c in codes[dim["cat"]]:
            n = at(**{dim["geo"]: g, dim["cat"]: c, dim["sex"]: sex_total})
            if n is None:
                continue
            note = f"level={level}; code={c}; mkcode={g}"
            if c == TOTAL_CAT:
                note += "; universe total, not a religion category"
            rows.append({"geo_id": g, "geo_level": level, "geo_name": name,
                         "source_category": labels_en[dim["cat"]][c].strip(),
                         "count": int(n), "basis": BASIS, "year": YEAR,
                         "source_id": SOURCE_ID, "note": note})
    return rows, labels_en[dim["cat"]]


def check(rows, cat_labels):
    ok = True
    total_label = cat_labels[TOTAL_CAT].strip()

    levels = {}
    for r in rows:
        levels.setdefault(r["geo_level"], set()).add(r["geo_id"])
    for lv, want in (("municipality", EXPECTED_MUNICIPALITIES), ("country", 1)):
        got = len(levels.get(lv, ()))
        good = got == want
        ok &= good
        print(f"  {'OK ' if good else 'BAD'} {lv:<13} {got:>4} units (expected {want})")

    nat = {r["source_category"]: r["count"] for r in rows if r["geo_level"] == "country"}
    good = nat.get(total_label) == NATIONAL
    ok &= good
    print(f"\n  {'OK ' if good else 'BAD'} national total {nat.get(total_label):,} "
          f"(published {NATIONAL:,})")

    # The 80 municipalities must sum to the national row exactly: SSO neither rounds nor
    # suppresses this table, so this is an equality and not a band.
    for label in sorted(nat):
        s = sum(r["count"] for r in rows
                if r["geo_level"] == "municipality" and r["source_category"] == label)
        good = s == nat[label]
        ok &= good
        if not good:
            print(f"  BAD {label}: municipalities {s:,} vs national {nat[label]:,}")
    print(f"  {'OK ' if ok else 'BAD'} all {len(nat)} categories sum from the 80 "
          "municipalities to the national row exactly")

    parts = sum(v for k, v in nat.items() if k != total_label)
    good = parts == NATIONAL
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} the 13 categories partition the population "
          f"({parts:,})")

    print(f"\n  {len(rows):,} rows. Categories, national:")
    for label, n in sorted(nat.items(), key=lambda kv: -kv[1]):
        mark = "  <- universe" if label == total_label else ""
        print(f"    {n:>10,}  {100.0 * n / NATIONAL:5.2f}%  {label}{mark}")

    if not ok:
        raise SystemExit("reconciliation FAILED")


def main():
    if "--fetch" in sys.argv:
        fetch()
    rows, cat_labels = read()
    check(rows, cat_labels)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print("\nwrote", OUT, f"({len(rows):,} rows)")


if __name__ == "__main__":
    main()
