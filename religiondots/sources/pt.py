"""Portugal — INE, Censos 2021, religion by freguesia.

Reads (or fetches) data/raw/pt/ and writes data/normalized/pt.csv.

One indicator, **`0012311`** — *População residente com 15 e mais anos de idade por Local de
residência à data dos Censos [2021] (NUTS - 2024) e Religião* — 11 religions plus a total,
over 3,439 nested territories, 7.4 MB of JSON, no key, no login, no wall.
**3,092 freguesias**, 8,781,900 people who answered.

THE CATALOGUE IS THE FIND, NOT THE INDICATOR. `xml_indic.jsp?opc=3` is the endpoint that
looks like INE's catalogue and it is a trap: 326 *recently updated* indicators, 517 KB, and
**zero** hits for `religi`. The full catalogue is `opc=2` — 13,098 indicators, 21 MB — and it
carries `geo_lastlevel` per indicator, so "which Portuguese census tables go down to the
freguesia" is a string search rather than a crawl. Every other `opc` value (0, 1, 4, 5, 6, 7)
returns the same 80 KB nav page. **When a catalogue endpoint takes a mode parameter, get the
full mode before concluding the data is not there**; the earlier probe of this office
concluded Portugal was silent on religion and it was one digit away from the answer.

`0011644` is the same table on the NUTS-2013 geography and `0006396` is the 2011 census at
the same tier. This drawis 0012311 because GISCO LAU 2021 matches its codes exactly.

**THE UNIVERSE IS NOT THE POPULATION, AND INE DOES NOT SAY SO IN THE TABLE.** The religion
question is *de resposta facultativa* and the 11 categories sum to the published total on
every one of the 3,439 units, to the person — a perfect partition, no `não respondeu` cell
anywhere. That looks like a mandatory question and is not one: the 15+ resident population
is **9,011,878** (10,343,066 total minus 1,331,188 aged 0-14, indicator 0011609) against
this table's **8,781,900**, so **229,978 people — 2.55% of the universe — declined and were
removed from the denominator rather than published as a category.** An office can prorate
its own non-response away and only the questionnaire says so; `sources/gy.md` hit the same
thing in Guyana and found it in a footnote. Every share here is therefore a share of those
who answered. `check()` asserts the gap, so a future vintage that changes it will fail loudly
rather than quietly restate what the map claims.

Usage:
    python sources/pt.py --fetch    one GET, 7.4 MB
    python sources/pt.py            normalise from data/raw/pt/
"""

import csv
import json
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "pt")
OUT = os.path.join(ROOT, "data", "normalized", "pt.csv")

SOURCE_ID = "pt_censos_2021"
YEAR = 2021
BASIS = "self_id"

COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count",
           "basis", "year", "source_id", "note"]

VARCD = "0012311"
API = ("https://www.ine.pt/ine/json_indicador/pindica.jsp"
       f"?op=2&varcd={VARCD}&Dim1=S7A2021&lang=PT")
CUBE = os.path.join(RAW, f"{VARCD}.json")

PERIOD = "2021"
TOTAL_CAT = "T"

# The published universe of this table, and the 15+ population it is drawn from. The
# difference is the non-response INE removed; see the module docstring.
ANSWERED = 8_781_900
POP_15_PLUS = 9_011_878
POP_TOTAL = 10_343_066

# geocod length -> level. Portugal's codes are strictly nested and the length is the level,
# with `PT` the one alphabetic exception at length 2 (Hungary's shape rule, §9h).
EXPECTED = {"country": 1, "nuts1": 3, "nuts2": 9, "nuts3": 26,
            "municipio": 308, "freguesia": 3_092}


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    if os.path.exists(CUBE) and os.path.getsize(CUBE) > 5_000_000:
        print("already have", CUBE)
        return
    # TLS verifies normally on www.ine.pt. Nothing is disabled here (§9h).
    print("GET", API)
    r = requests.get(API, timeout=900,
                     headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    r.raise_for_status()
    doc = r.json()
    # §5a: HTTP 200 is not a download. INE answers errors with a 200 and a `Sucesso`
    # envelope carrying `Falso`, so assert the shape rather than the status.
    if not isinstance(doc, list) or not doc:
        raise SystemExit(f"expected a one-element list, got {type(doc).__name__}")
    if "Dados" not in doc[0] or PERIOD not in doc[0]["Dados"]:
        raise SystemExit(f"no Dados/{PERIOD} in the response, keys {list(doc[0])}")
    with open(CUBE, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, ensure_ascii=False)
    print(f"  {os.path.getsize(CUBE):,} bytes")


def _load():
    if not os.path.exists(CUBE):
        raise SystemExit(f"missing {CUBE} -- run with --fetch first")
    with open(CUBE, encoding="utf-8") as fh:
        return json.load(fh)


def _level(code):
    """One dimension, six nested levels, told apart by the SHAPE of the code."""
    if code == "PT":
        return "country"
    n = len(code)
    if n == 1:
        return "nuts1"          # 1 Continente, 2 Açores, 3 Madeira
    if n == 2:
        return "nuts2"          # 11 Norte … 1A Grande Lisboa … 30 Madeira
    if n == 3:
        return "nuts3"
    if n == 4:
        return "municipio"
    if n == 6:
        return "freguesia"
    raise SystemExit(f"unclassifiable geocod {code!r} -- INE has changed the code shape")


def read():
    doc = _load()
    dat = doc[0]["Dados"][PERIOD]

    rows, cats = [], {}
    for x in dat:
        code = str(x["geocod"])
        cat_id = str(x["dim_3"])
        label = str(x["dim_3_t"]).strip()
        cats[cat_id] = label
        raw = x.get("valor")
        # §5a again, at cell scale: an absent value is not a zero. Portugal publishes none,
        # so this raises rather than filling — a future vintage that suppresses must be seen.
        if raw is None or raw == "":
            raise SystemExit(f"{code}/{cat_id}: empty cell. Portugal has never published "
                             "one; decide explicitly whether it is a zero or a suppression "
                             "before letting this through (spec §3.8).")
        level = _level(code)
        note = f"level={level}; code={cat_id}"
        if cat_id == TOTAL_CAT:
            note += "; universe total (those who ANSWERED), not a religion category"
        rows.append({"geo_id": code, "geo_level": level,
                     "geo_name": str(x["geodsg"]).strip(),
                     "source_category": label, "count": int(raw),
                     "basis": BASIS, "year": YEAR, "source_id": SOURCE_ID, "note": note})
    return rows, cats


def check(rows, cats):
    ok = True
    total_label = cats[TOTAL_CAT]

    levels = {}
    for r in rows:
        levels.setdefault(r["geo_level"], set()).add(r["geo_id"])
    for lv, want in EXPECTED.items():
        got = len(levels.get(lv, ()))
        good = got == want
        ok &= good
        print(f"  {'OK ' if good else 'BAD'} {lv:<11} {got:>5} units (expected {want:,})")

    nat = {r["source_category"]: r["count"] for r in rows if r["geo_level"] == "country"}
    good = nat.get(total_label) == ANSWERED
    ok &= good
    print(f"\n  {'OK ' if good else 'BAD'} national universe {nat.get(total_label):,} "
          f"(expected {ANSWERED:,})")

    parts = sum(v for k, v in nat.items() if k != total_label)
    good = parts == ANSWERED
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} the 11 categories partition the universe exactly "
          f"({parts:,}) — there is no `did not answer` cell")

    # Every level must reproduce the country exactly, for the total AND for each category:
    # nothing is suppressed anywhere, which is what makes the freguesia tier safe to draw.
    for lv in ("nuts1", "nuts2", "nuts3", "municipio", "freguesia"):
        s = sum(r["count"] for r in rows
                if r["geo_level"] == lv and r["source_category"] == total_label)
        good = s == ANSWERED
        ok &= good
        print(f"  {'OK ' if good else 'BAD'} {lv:<11} totals sum to the country ({s:,})")

    print("\n  every category, at every level, must sum to its national figure:")
    worst = 0
    for label in sorted(nat, key=lambda k: -nat[k]):
        if label == total_label:
            continue
        for lv in ("municipio", "freguesia"):
            s = sum(r["count"] for r in rows
                    if r["geo_level"] == lv and r["source_category"] == label)
            gap = abs(nat[label] - s)
            worst = max(worst, gap)
            if gap:
                ok = False
                print(f"    BAD {label} at {lv}: {s:,} vs {nat[label]:,} (gap {gap:,})")
    print(f"    {'OK ' if not worst else 'BAD'} largest discrepancy across 11 categories "
          f"x 2 drawn levels: {worst:,}")

    # ---- the universe gap, which is the thing the table does not say (§3.5) ----
    gap = POP_15_PLUS - ANSWERED
    frac = gap / POP_15_PLUS
    print(f"\n  THE UNIVERSE IS NOT THE POPULATION:")
    print(f"    resident population, all ages   {POP_TOTAL:>10,}")
    print(f"    resident population 15+         {POP_15_PLUS:>10,}   (indicator 0011609)")
    print(f"    published in this table         {ANSWERED:>10,}")
    print(f"    removed as non-response         {gap:>10,}   ({frac:.2%} of the 15+ "
          "universe, drawn nowhere)")
    good = 0 < gap < 0.05 * POP_15_PLUS
    ok &= good
    if not good:
        print("    BAD the non-response gap is not what sources/pt.md records. Re-derive "
              "it before drawing: every share in this table is a share of those who "
              "answered, and how big that qualifier is matters.")

    print(f"\n  {len(rows):,} rows. Categories, national:")
    for label, n in sorted(nat.items(), key=lambda kv: -kv[1]):
        mark = "  <- universe" if label == total_label else ""
        print(f"    {n:>10,}  {100.0 * n / ANSWERED:6.2f}%  {label}{mark}")

    if not ok:
        raise SystemExit("reconciliation FAILED")


def main():
    if "--fetch" in sys.argv:
        fetch()
    rows, cats = read()
    check(rows, cats)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print("\nwrote", OUT, f"({len(rows):,} rows)")


if __name__ == "__main__":
    main()
