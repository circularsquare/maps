"""Estonia: Statistics Estonia, Rahvaloendus 2021 (REL2021), mother tongue by place of residence.

    python sources/ee_census.py --fetch    three PxWeb POSTs (about 0.5 MB) if missing
    python sources/ee_census.py            normalise from data/raw/ee/

-> data/normalized/ee.csv, levels:
     country, county, municipality      the administrative tree (alternatives, never summed)
     part                               the finest piece the table publishes; THESE ARE DRAWN
     settlement_region                  RL21431's five places with the 245-language list (check only)

THE TABLE. RL21434 "Population by mother tongue, sex, age group and place of residence
(administrative unit), 31 December 2021", age group total, both sexes: 15 named languages,
"Other mother tongue" and "Mother tongue unknown", for the whole population (the 2021 census was
register-based; the religion question in religiondots was asked from 15, this one covers
everyone). A person could give two mother tongues; RL214312 lists the pairs (nationally), and
RL21434 counts each person once, on the first.

THE PLACES. Each municipality is published whole and, where it holds a town, also cut into the
town as a settlement unit ("Kehra city as a settlement unit", code with the EHAK settlement code
in [8:12]) and the rest ("Anija rural municipality, excl. Kehra ...", a bare serial code such as
"4"). Tallinn is cut into its 8 linnaosad and Kohtla-Järve into its 5. The finest cover, `part`,
is: a municipality's children where it has any, else the municipality. Asserted: each set of
children sums to its parent in every category, and the parts tile the country.

CHECKS: unit counts (1, 15, 79); national total against the published 1,331,824; categories
sum to each place's total, and children to their parent in every category, within the cube's
disclosure noise (±10; see check()); the drawn parts sum to the country less Kohtla-Järve's 24
undistricted people; the national row against RL21431's "Whole country" (a separate tabulation
with 245 languages) after folding its long list into RL21434's 17 categories: exact.
"""

import csv
import json
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "ee")
OUT = os.path.join(ROOT, "data", "normalized", "ee.csv")

SOURCE_ID = "ee_rel2021_mt"
YEAR = 2021
NATIONAL = 1_331_824   # REL2021 population, 31.12.2021 (Statistics Estonia)

API = "https://andmed.stat.ee/api/v1/en/stat/rahvaloendus/rel2021/" \
      "rahvastiku-demograafilised-ja-etno-kultuurilised-naitajad/rahvus-emakeel/"
TABLES = {
    "RL21434": dict(px="RL21434.px", fixed={"Vanuserühm": "1", "Sugu": "1"},
                    all=("Elukoht", "Emakeel")),
    "RL21431": dict(px="RL21431.px", fixed={"Sugu": "1"}, all=("Elukoht", "Emakeel")),
    "RL214312": dict(px="RL214312.px", fixed={}, all=("Emakeel 1", "Emakeel 2", "Näitaja")),
}

COLUMNS = ["geo_id", "geo_level", "geo_name", "parent", "source_category", "count", "tier",
           "year", "source_id", "note"]
TOTAL = "Mother tongue total"
UNKNOWN = "Mother tongue unknown"
EXPECTED = {"country": 1, "county": 15, "municipality": 79}
NOISE = 10             # the cube's disclosure noise, see check()
KJ = "004503210000L4"  # Kohtla-Järve city
KJ_UNDISTRICTED = {TOTAL: 24, "Russian": 19, "Estonian": 5}


def fetch():
    import requests
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    os.makedirs(RAW, exist_ok=True)
    for name, cfg in TABLES.items():
        dest = os.path.join(RAW, name + ".json")
        if os.path.exists(dest) and os.path.getsize(dest) > 1_000:
            print("already have", dest)
            continue
        q = [{"code": k, "selection": {"filter": "item", "values": [v]}}
             for k, v in cfg["fixed"].items()]
        q += [{"code": k, "selection": {"filter": "all", "values": ["*"]}} for k in cfg["all"]]
        print("POST", API + cfg["px"])
        r = requests.post(API + cfg["px"], json={"query": q, "response": {"format": "json-stat2"}},
                          timeout=300, verify=False, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        js = r.json()
        if "value" not in js:
            raise SystemExit(f"{name}: not json-stat2, keys {list(js)}")
        with open(dest, "w", encoding="utf-8") as fh:
            json.dump(js, fh, ensure_ascii=False)
        print(f"  {os.path.getsize(dest):,} bytes")


def _load(name):
    p = os.path.join(RAW, name + ".json")
    if not os.path.exists(p):
        raise SystemExit(f"missing {p} -- run with --fetch first")
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def _cells(js):
    """Yield ({dim: (code, label)}, value) over a json-stat2 cube (row-major over `id`)."""
    ids, sizes = js["id"], js["size"]
    dims = []
    for d in ids:
        cat = js["dimension"][d]["category"]
        order = sorted(cat["index"], key=lambda k: cat["index"][k])
        dims.append([(k, cat["label"][k]) for k in order])
    strides, acc = [0] * len(sizes), 1
    for i in range(len(sizes) - 1, -1, -1):
        strides[i] = acc
        acc *= sizes[i]
    for flat, v in enumerate(js["value"]):
        yield {d: dims[i][(flat // strides[i]) % sizes[i]] for i, d in enumerate(ids)}, v


def depth(text):
    n = 0
    while text[n:n + 2] == "..":
        n += 2
    return n // 2


def places(js):
    """The place tree of RL21434, in published order: [(code, name, level, parent)].

    Skips the analytical aggregates (settlement regions, NUTS3 'EE00x', county settlement
    regions '0037L'). Depth from the leading dots: 0 = county (all caps) or country,
    1 = municipality, 2 = a municipality's part."""
    cat = js["dimension"]["Elukoht"]["category"]
    order = sorted(cat["index"], key=lambda k: cat["index"][k])
    out, county, muni = [], None, None
    for code in order:
        label = cat["label"][code]
        d, name = depth(label), label.lstrip(".").strip()
        if code == "1":
            out.append((code, name, "country", ""))
        elif code in ("L1", "M1", "V1") or code.startswith("EE") or "settlement region" in name:
            continue
        elif d == 0:
            county = code
            out.append((code, name, "county", "1"))
        elif d == 1:
            muni = code
            out.append((code, name, "municipality", county))
        elif d == 2:
            out.append((code, name, "part", muni))
        else:
            raise SystemExit(f"unexpected depth {d}: {code} {label!r}")
    return out


def read():
    js = _load("RL21434")
    tree = places(js)
    vals = {}
    for key, v in _cells(js):
        pc = key["Elukoht"][0]
        lang = key["Emakeel"][1]
        if v is None:
            raise SystemExit(f"null cell {pc} {lang}")
        vals[(pc, lang)] = int(v)
    langs = [lab for _, lab in
             sorted(js["dimension"]["Emakeel"]["category"]["label"].items(),
                    key=lambda kv: js["dimension"]["Emakeel"]["category"]["index"][kv[0]])]
    return tree, vals, langs


def build(tree, vals, langs):
    parents_with_parts = {p for _, _, lv, p in tree if lv == "part"}
    rows = []
    for code, name, level, parent in tree:
        drawn = level == "part" or (level == "municipality" and code not in parents_with_parts)
        levels = [level] + (["part"] if level == "municipality" and drawn else [])
        for lv in levels:
            for lang in langs:
                note = f"RL21434; level={level}"
                if lang == TOTAL:
                    note += "; universe total, not a language"
                rows.append(dict(geo_id=code, geo_level=lv, geo_name=name,
                                 parent=parent if lv == level else code,
                                 source_category=lang, count=vals[(code, lang)], tier="measured",
                                 year=YEAR, source_id=SOURCE_ID, note=note))
    # RL21431: the 245-language list, at the five settlement-region places
    js = _load("RL21431")
    for key, v in _cells(js):
        pc, pname = key["Elukoht"]
        rows.append(dict(geo_id=pc, geo_level="settlement_region", geo_name=pname.lstrip(".").strip(),
                         parent="", source_category=key["Emakeel"][1], count=int(v or 0),
                         tier="measured", year=YEAR, source_id=SOURCE_ID,
                         note="RL21431; 245-language list; check only, not drawn"))
    return rows


def check(tree, vals, langs, rows):
    ok = True
    lv_units = {}
    for code, _, lv, _ in tree:
        lv_units.setdefault(lv, []).append(code)
    for lv, want in EXPECTED.items():
        got = len(lv_units.get(lv, []))
        good = got == want
        ok &= good
        print(f"  {'OK ' if good else 'BAD'} {lv:<13} {got:>3} units (expected {want})")

    nat = vals[("1", TOTAL)]
    good = nat == NATIONAL
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} national total {nat:,} (published {NATIONAL:,})")

    # THE CUBE IS PERTURBED. Every cell below the national row carries a small noise (checked
    # 2026-10-05): the categories miss their place's total by -7..+9, and a parent misses the
    # sum of its children by -5..+6 in 219 of 850 cells. That is disclosure noise, the same
    # in kind as the religion table's base-10 rounding: drawn as published, never reconciled.
    # The one larger gap is real: Kohtla-Järve (0321) is 24 more than its five linnaosad, 19 of
    # them Russian, people the register places in the city but in no district. Drawing the
    # districts loses them (0.002% of the country).
    worst = max(abs(sum(vals[(c, l)] for l in langs if l != TOTAL) - vals[(c, TOTAL)])
                for c, _, _, _ in tree)
    good = worst <= NOISE
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} in all {len(tree)} places the 16 categories sum to the "
          f"total within the noise (worst {worst}, bar {NOISE})")

    kids = {}
    for code, _, lv, parent in tree:
        if parent:
            kids.setdefault(parent, []).append(code)
    n_cmp, n_off, bad = 0, 0, []
    for p, cs in kids.items():
        for l in langs:
            n_cmp += 1
            d = vals[(p, l)] - sum(vals[(c, l)] for c in cs)
            n_off += d != 0
            if abs(d) > NOISE and not (p == KJ and KJ_UNDISTRICTED.get(l) == d):
                bad.append((p, l, vals[(p, l)], d))
    ok &= not bad
    print(f"  {'OK ' if not bad else 'BAD'} every parent equals the sum of its children within "
          f"the noise in all {n_cmp:,} (parent, category) cells ({n_off} not exact; "
          f"Kohtla-Järve's undistricted {KJ_UNDISTRICTED[TOTAL]} aside)")
    for b in bad[:10]:
        print("     ", b)

    parts = [r for r in rows if r["geo_level"] == "part" and r["source_category"] == TOTAL]
    s = sum(r["count"] for r in parts)
    good = s == NATIONAL - KJ_UNDISTRICTED[TOTAL]
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} the {len(parts)} drawn parts sum to the country less "
          f"Kohtla-Järve's {KJ_UNDISTRICTED[TOTAL]} ({s:,})")

    # second table: RL21431 national row, 245 languages folded to RL21434's list
    long = {r["source_category"]: r["count"] for r in rows
            if r["geo_level"] == "settlement_region" and r["geo_id"] == "1"}
    short = {l: vals[("1", l)] for l in langs}
    folded = {}
    for lab, n in long.items():
        tgt = lab if lab in short else ("Other mother tongue" if lab not in (TOTAL,) else TOTAL)
        folded[tgt] = folded.get(tgt, 0) + n
    print("\n  national row against RL21431 (245 languages, folded):")
    for l in langs:
        a, b = short[l], folded.get(l, 0)
        good = a == b
        ok &= good
        print(f"    {'OK ' if good else 'BAD'} {l:<26} {a:>10,} {b:>10,} {b - a:+d}")

    print("\n  the 'Other mother tongue' bucket at national level, largest (RL21431):")
    other = sorted(((n, lab) for lab, n in long.items() if lab not in short), reverse=True)
    for n, lab in other[:25]:
        print(f"    {n:>7,}  {lab}")
    print(f"    ... {len(other)} languages in all, {sum(n for n, _ in other):,} people")

    js = _load("RL214312")
    pairs = []
    for key, v in _cells(js):
        a, b = key["Emakeel 1"][1], key["Emakeel 2"][1]
        if v and a != "Mother tongues total" and b != "Mother tongues total":
            pairs.append((int(v), a, b))
        if a == "Mother tongues total" and b == "Mother tongues total":
            print(f"\n  two mother tongues (RL214312): {int(v or 0):,} people; largest pairs:")
    for n, a, b in sorted(pairs, reverse=True)[:8]:
        print(f"    {n:>6,}  {a} + {b}")

    print("\n  national categories (RL21434):")
    for l in sorted(langs, key=lambda l: -short[l]):
        print(f"    {short[l]:>10,}  {100 * short[l] / NATIONAL:6.2f}%  {l}")
    if not ok:
        raise SystemExit("reconciliation FAILED")


def main():
    if "--fetch" in sys.argv:
        fetch()
    tree, vals, langs = read()
    rows = build(tree, vals, langs)
    check(tree, vals, langs, rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print("\nwrote", OUT, f"({len(rows):,} rows)")


if __name__ == "__main__":
    main()
