"""Slovenia: SURS, Popis 2002, mother tongue (materni jezik) by obcina.

Reads (or fetches) data/raw/si/ and writes data/normalized/si.csv. The record is sources/si.md.

Popis 2002 is the last Slovenian census that asked language; 2011 and 2021 were register-based
and carry none. Three tables of the same census, all on SURS's PxWeb API (no key, no login):

  05W1007S.px  mother tongue x 193 obcina rows (SLOVENIJA + 192), 12 named languages, Drugi,
               Neznano. The drawn table.
  05W1607S.px  mother tongue, national, 1991 and 2002, 33 named languages. A check level: its
               2002 cells must match the drawn table's national row, and its extra languages
               must sum to the drawn table's `Drugi`. Never drawn.
  05W0405S.px  population by settlement; its municipality rows carry the official obcina codes.

THE OBCINA CODES IN 05W1007S ARE NOT SLOVENIA'S MUNICIPALITY CODES (religiondots found the same
in the religion table, 05W1006S): the dimension runs 001-193 alphabetically with 001 = SLOVENIJA,
so joining on it shifts every unit by one and every total still reconciles. The real codes come
from 05W0405S's labels (`001  AJDOVSCINA`); the name join is confirmed by population, all 192
must agree to the person.

Status flags: `z` (withheld) is not a zero and is not reconstructed; `-` is a true zero.

Usage:
    python sources/si_popis2002.py --fetch
    python sources/si_popis2002.py
"""
import csv
import json
import os
import re
import sys
import urllib.request

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "si")
OUT = os.path.join(ROOT, "data", "normalized", "si.csv")

SOURCE_ID = "si_popis_2002_mt"
YEAR = 2002
COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count", "tier", "year",
           "source_id", "note"]

API = "https://pxweb.stat.si/SiStatData/api/v1/sl/Data/"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
MT_TABLE, MT_JSON = "05W1007S.px", os.path.join(RAW, "05W1007S.json")
DET_TABLE, DET_JSON = "05W1607S.px", os.path.join(RAW, "05W1607S.json")
POP_TABLE, POP_JSON = "05W0405S.px", os.path.join(RAW, "05W0405S.json")

NATIONAL = 1_964_036
EXPECTED_MUNICIPALITIES = 192
COUNTRY_LABEL = "SLOVENIJA"
TOTAL_LABEL = "Materni jezik - SKUPAJ"
OTHER_LABEL = "Drugi"
UNKNOWN_LABEL = "Neznano"
STATUS_WITHHELD = "z"
STATUS_TRUE_ZERO = "-"

# The settlement table spells two municipalities in older forms (religiondots/sources/si.py):
# 018 Destrnik is DESTERNIK there; 116 Sveti Jurij ob Scavnici is SVETI JURIJ, its pre-2001 name.
ALIASES = {"DESTRNIK": "DESTERNIK", "SVETI JURIJ OB ŠČAVNICI": "SVETI JURIJ"}


def _post(table, query):
    req = urllib.request.Request(API + table, data=json.dumps(query).encode("utf-8"),
                                 headers={"Content-Type": "application/json", **UA})
    return urllib.request.urlopen(req, timeout=300).read()


def _get(table):
    return json.loads(urllib.request.urlopen(urllib.request.Request(API + table, headers=UA),
                                             timeout=300).read())


def _all(code):
    return {"code": code, "selection": {"filter": "all", "values": ["*"]}}


def fetch():
    os.makedirs(RAW, exist_ok=True)
    if not os.path.exists(MT_JSON):
        print("POST", API + MT_TABLE)
        raw = _post(MT_TABLE, {"query": [_all("OBČINA"), _all("MATERIN JEZIK")],
                               "response": {"format": "json-stat2"}})
        doc = json.loads(raw)
        if doc.get("size") != [193, 14]:
            raise SystemExit(f"unexpected cube shape {doc.get('size')} for {MT_TABLE}")
        open(MT_JSON, "wb").write(raw)
    if not os.path.exists(DET_JSON):
        print("POST", API + DET_TABLE)
        raw = _post(DET_TABLE, {"query": [
            _all("MATERIN JEZIK"),
            {"code": "MERITVE", "selection": {"filter": "item", "values": ["1"]}},
            _all("LETO")], "response": {"format": "json-stat2"}})
        doc = json.loads(raw)
        if doc.get("size") != [35, 1, 2]:
            raise SystemExit(f"unexpected cube shape {doc.get('size')} for {DET_TABLE}")
        open(DET_JSON, "wb").write(raw)
    if not os.path.exists(POP_JSON):
        meta = _get(POP_TABLE)
        var = next(v for v in meta["variables"] if v["code"] == "NASELJA")
        muni = [v for v, t in zip(var["values"], var["valueTexts"]) if re.match(r"^\d{3}\s\s", t)]
        if len(muni) != EXPECTED_MUNICIPALITIES:
            raise SystemExit(f"{len(muni)} municipality rows in {POP_TABLE}")
        print("POST", API + POP_TABLE)
        raw = _post(POP_TABLE, {"query": [
            {"code": "NASELJA", "selection": {"filter": "item", "values": muni}},
            {"code": "SPOL", "selection": {"filter": "item", "values": ["1"]}}],
            "response": {"format": "json-stat2"}})
        open(POP_JSON, "wb").write(raw)


def _load(path):
    if not os.path.exists(path):
        raise SystemExit(f"missing {path}; run with --fetch")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _key(name):
    s = " ".join(str(name).split())
    return re.sub(r"\s*-\s*", "-", s).upper()


def codes():
    doc = _load(POP_JSON)
    cat = doc["dimension"]["NASELJA"]["category"]
    out = {}
    for value, pos in cat["index"].items():
        m = re.match(r"^(\d{3})\s\s(.+)$", cat["label"][value])
        if not m:
            raise SystemExit(f"settlement-table label {cat['label'][value]!r}")
        out[m.group(1)] = (m.group(2).strip(), doc["value"][pos])
    if len(out) != EXPECTED_MUNICIPALITIES:
        raise SystemExit(f"{len(out)} codes")
    return out


def read():
    doc = _load(MT_JSON)
    ocat = doc["dimension"]["OBČINA"]["category"]
    vcat = doc["dimension"]["MATERIN JEZIK"]["category"]
    vcodes = sorted(vcat["index"], key=lambda k: vcat["index"][k])
    nv = len(vcodes)
    value, status = doc["value"], doc.get("status", {}) or {}

    lookup = {_key(n): (c, n, p) for c, (n, p) in codes().items()}
    for census_name, older in ALIASES.items():
        hit = lookup.pop(_key(older), None)
        if hit is None:
            raise SystemExit(f"alias {older!r} gone from the settlement table")
        lookup[_key(census_name)] = hit

    rows, withheld, zeros, seen = [], {}, 0, {}
    for oval, opos in sorted(ocat["index"].items(), key=lambda kv: kv[1]):
        name = ocat["label"][oval]
        if name.upper() == COUNTRY_LABEL:
            geo_id, level, pop = "SI", "country", NATIONAL
        else:
            hit = lookup.get(_key(name))
            if hit is None:
                raise SystemExit(f"no official code for {name!r}")
            geo_id, level, pop = hit[0], "municipality", hit[2]
            if geo_id in seen:
                raise SystemExit(f"code {geo_id} claimed by {seen[geo_id]!r} and {name!r}")
            seen[geo_id] = name
        for j, vval in enumerate(vcodes):
            label = vcat["label"][vval]
            idx = opos * nv + j
            n, flag = value[idx], status.get(str(idx))
            if n is None:
                if flag == STATUS_WITHHELD:
                    withheld.setdefault(label, []).append(geo_id)
                    continue
                if flag == STATUS_TRUE_ZERO:
                    zeros += 1
                    n = 0
                else:
                    raise SystemExit(f"{geo_id}/{label}: null with status {flag!r}")
            note = f"level={level}; code={vval}"
            if label == TOTAL_LABEL:
                note += "; universe total, not a language"
            rows.append({"geo_id": geo_id, "geo_level": level, "geo_name": name,
                         "source_category": label, "count": int(n), "tier": "measured",
                         "year": YEAR, "source_id": SOURCE_ID, "note": note})
        tot = value[opos * nv + vcodes.index(next(v for v in vcodes
                                                   if vcat["label"][v] == TOTAL_LABEL))]
        if tot != pop:
            raise SystemExit(f"{name} ({geo_id}): {tot:,} here, {pop:,} in the settlement "
                             "table; the join is wrong")
    return rows, withheld, zeros


def detail_rows():
    doc = _load(DET_JSON)
    dims, sizes = doc["id"], doc["size"]
    cats = {k: doc["dimension"][k]["category"] for k in dims}
    vdim = "MATERIN JEZIK"
    rows = []
    for year in cats["LETO"]["index"]:
        for v, _ in sorted(cats[vdim]["index"].items(), key=lambda kv: kv[1]):
            idx = 0
            sel = {vdim: v, "MERITVE": next(iter(cats["MERITVE"]["index"])), "LETO": year}
            for name, size in zip(dims, sizes):
                idx = idx * size + cats[name]["index"][sel[name]]
            n = doc["value"][idx]
            if n is None:
                continue
            rows.append({"geo_id": f"SI-{year}", "geo_level": "country_detail",
                         "geo_name": COUNTRY_LABEL, "source_category": cats[vdim]["label"][v],
                         "count": int(n), "tier": "measured", "year": int(year),
                         "source_id": SOURCE_ID,
                         "note": f"level=country_detail; code={v}; 05W1607S, national only, "
                                 "a check level, not drawn"})
    return rows


def check(rows, withheld, zeros):
    ok = True

    def say(good, msg):
        nonlocal ok
        ok &= good
        print(f"  {'OK ' if good else 'BAD'} {msg}")

    muni = [r for r in rows if r["geo_level"] == "municipality"]
    nat = {r["source_category"]: r["count"] for r in rows if r["geo_level"] == "country"}
    det = {r["source_category"]: r["count"] for r in rows
           if r["geo_level"] == "country_detail" and r["year"] == YEAR}
    labels = [k for k in nat if k != TOTAL_LABEL]

    say(len({r["geo_id"] for r in muni}) == EXPECTED_MUNICIPALITIES,
        f"{len({r['geo_id'] for r in muni})} municipalities, each joined by name to its official "
        "code and confirmed on population")
    print(f"  {zeros:,} true zeros; {sum(len(v) for v in withheld.values()):,} cells withheld")
    say(nat[TOTAL_LABEL] == NATIONAL, f"national total {nat[TOTAL_LABEL]:,}")
    say(sum(nat[k] for k in labels) == NATIONAL,
        f"the {len(labels)} categories partition the country ({sum(nat[k] for k in labels):,})")
    s = sum(r["count"] for r in muni if r["source_category"] == TOTAL_LABEL)
    say(s == NATIONAL, f"municipality totals sum to the country ({s:,})")

    print("\n  per category, municipality sum vs national (gap = withheld cells):")
    gap_all = 0
    for k in labels:
        s = sum(r["count"] for r in muni if r["source_category"] == k)
        gap = nat[k] - s
        gap_all += gap
        good = gap >= 0 and (gap > 0) == bool(withheld.get(k))
        ok &= good
        print(f"    {'OK ' if good else 'BAD'} {k:<16} {nat[k]:>10,}  munis {s:>10,}  "
              f"withheld {gap:>5,} in {len(withheld.get(k, ())):>3} cells")
    print(f"    total withheld {gap_all:,} = {gap_all / NATIONAL:.4%}")
    say(gap_all / NATIONAL < 0.005, "withheld under 0.5% of the country")

    print("\n  the 33-language national table (05W1607S, 2002) against the drawn national row:")
    for k in labels:
        if k == OTHER_LABEL:
            continue
        say(det.get(k) == nat[k], f"{k:<16} {nat[k]:>10,} = {det.get(k)}")
    extra = {k: v for k, v in det.items() if k not in nat}
    say(sum(extra.values()) + det[OTHER_LABEL] == nat[OTHER_LABEL],
        f"Drugi {nat[OTHER_LABEL]:,} = the {len(extra)} further labels of 05W1607S "
        f"({sum(extra.values()):,}) + its own Drugi ({det[OTHER_LABEL]:,})")
    for k, v in sorted(extra.items(), key=lambda kv: -kv[1]):
        print(f"        {v:>8,}  {k}")

    print("\n  national, 2002:")
    for k in labels:
        print(f"    {nat[k]:>10,}  {100 * nat[k] / NATIONAL:6.2f}%  {k}")
    if not ok:
        raise SystemExit("reconciliation FAILED")


def main():
    if "--fetch" in sys.argv:
        fetch()
    rows, withheld, zeros = read()
    rows += detail_rows()
    check(rows, withheld, zeros)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print("\nwrote", OUT, f"({len(rows):,} rows)")


if __name__ == "__main__":
    main()
