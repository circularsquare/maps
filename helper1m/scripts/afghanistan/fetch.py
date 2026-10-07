"""Afghanistan population for helper1m: 34 provinces, 401 COD districts.

Default source, NSIA (now GSIA, the statistics office) "Estimated Population
of Afghanistan" for 1403, 1404 and 1405 (solar years starting March 2024,
2025, 2026; written here as 2024, 2025, 2026). Settled population only: the
1.5 million Kuchi nomads NSIA adds at national level have no place and are
left out. NSIA prints 457 administrative units; scripts/afghanistan/
nsia_to_cod.csv says which of COD-AB v03's 401 districts each belongs to
(see make_mapping.py and the README).

    python fetch.py            # NSIA (default)
    python fetch.py --source codps

--source codps writes UNFPA/OCHA's COD-PS instead: 2026 by district, and 2021
by province only (its 2021 release has no district table), so a district's
2021 figure is its province's 2021 total times its 2026 share of the province.

Inputs, all in helper1m/data/afghanistan/:
  raw/nsia_1403.xlsx, raw/nsia_1404.xlsx    GSIA workbooks (district sheet)
  raw/nsia_1405_v4.pdf                      GSIA PDF, 2026-08-25 upload
  raw/afg_admpop_2021_v2.xlsx               COD-PS 2021 (province)
  kontur_adm2.csv                           from kontur.py (Kaldar split)
and religiondots/data/raw/af/afg_admpop_adm2_2026.csv (COD-PS 2026, read only).

Writes helper1m/data/afghanistan/population.csv (code, level, year, pop).
"""
from __future__ import annotations

import argparse
import csv
import difflib
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import nsia

HERE = Path(__file__).parent
HELPER = HERE.parents[1]
REPO = HELPER.parent
DATA = HELPER / "data" / "afghanistan"
RAW = DATA / "raw"
OUT = DATA / "population.csv"
MAPPING = HERE / "nsia_to_cod.csv"
KONTUR = DATA / "kontur_adm2.csv"
CODPS_2026 = REPO / "religiondots" / "data" / "raw" / "af" / "afg_admpop_adm2_2026.csv"
CODPS_2021 = RAW / "afg_admpop_2021_v2.xlsx"

NSIA_YEARS = {2024: ("xlsx", "nsia_1403.xlsx"), 2025: ("xlsx", "nsia_1404.xlsx"),
              2026: ("pdf", "nsia_1405_v4.pdf")}
BASE_YEAR = 2025   # the 1404 tables nsia_to_cod.csv was written against


def unit_key(name):
    """Letters of a unit's name, using what is in brackets for a provincial
    centre ('Provincial Capital (Kabul )' -> 'kabul'), so a renamed or
    relabelled centre still pairs up across years."""
    m = re.search(r"\((.+)\)", name)
    if m and re.search(r"capital|cent(er|re)", name, re.I):
        name = m.group(1)
    return re.sub(r"[^a-z]", "", name.lower())


def align(base, other):
    """Pair each unit of another year with a 1404 unit: same province (by
    table order), then the same name, then the closest name. Returns
    {(prov_index, base_name): row}."""
    def by_prov(rows):
        out = {}
        for r in rows:
            out.setdefault(r["prov"], []).append(r)
        return list(out.values())
    B, O = by_prov(base), by_prov(other)
    assert len(B) == len(O) == 34
    pairs = {}
    for i, (pb, po) in enumerate(zip(B, O)):
        left = {unit_key(r["name"]): r for r in pb}
        todo = []
        for r in po:
            k = unit_key(r["name"])
            if k in left:
                pairs[(i, left.pop(k)["name"])] = r
            else:
                todo.append(r)
        for r in todo:
            k = unit_key(r["name"])
            best = max(left, key=lambda c: difflib.SequenceMatcher(None, k, c).ratio(), default=None)
            score = difflib.SequenceMatcher(None, k, best).ratio() if best else 0
            if best is None or score < 0.6:
                raise SystemExit(f"no 1404 unit for {pb[0]['prov']} / {r['name']}")
            pairs[(i, left.pop(best)["name"])] = r
    return pairs


def load_mapping():
    rows = list(csv.DictReader(MAPPING.open(encoding="utf-8")))
    return {(r["province"], r["name_1404"]): r["target"] for r in rows}


def kontur():
    return {r["code"]: int(r["kontur"]) for r in csv.DictReader(KONTUR.open(encoding="utf-8"))}


def shares(target, kon):
    """{pcode: share} for one mapping target."""
    if target.startswith("split:"):
        codes = target[6:].split(",")
        tot = sum(kon[c] for c in codes)
        return {c: kon[c] / tot for c in codes}
    if ":" in target:
        parts = [p.split(":") for p in target.split(";")]
        tot = sum(float(w) for _, w in parts)
        return {c: float(w) / tot for c, w in parts}
    return {target: 1.0}


def nsia_rows():
    import geopandas as gpd
    years = {}
    for y, (kind, fn) in NSIA_YEARS.items():
        years[y] = nsia.read_xlsx(RAW / fn) if kind == "xlsx" else nsia.read_pdf(RAW / fn)
        print(f"NSIA {y}: {len(years[y])} units, settled {sum(r['pop'] for r in years[y]):,}")
    base = years[BASE_YEAR]
    mapping, kon = load_mapping(), kontur()
    prov_order = list(dict.fromkeys(r["prov"] for r in base))
    assert len(mapping) == len(base) == 457

    adm2 = gpd.read_file(DATA / "boundaries" / "adm2.gpkg", ignore_geometry=True)
    parent = dict(zip(adm2["code"], adm2["parent"]))

    dist = defaultdict(lambda: defaultdict(float))
    prov_tot = defaultdict(lambda: defaultdict(int))
    for y, rows in years.items():
        if y == BASE_YEAR:
            pairs = {(prov_order.index(r["prov"]), r["name"]): r for r in rows}
        else:
            pairs = align(base, rows)
            missing = [r["name"] for r in base
                       if (prov_order.index(r["prov"]), r["name"]) not in pairs]
            print(f"  {y}: {len(pairs)} units paired with 1404; 1404 units absent: {missing}")
        for (pi, bname), r in pairs.items():
            target = mapping[(prov_order[pi], bname)]
            for code, s in shares(target, kon).items():
                dist[code][y] += r["pop"] * s
            prov_tot[prov_order[pi]][y] += r["pop"]

    # whole people, rounded so each province keeps its exact NSIA total
    out = []
    by_parent = defaultdict(list)
    for code in dist:
        by_parent[parent[code]].append(code)
    missing = sorted(set(parent) - set(dist))
    assert not missing, f"districts with no NSIA unit: {missing}"
    for p, codes in sorted(by_parent.items()):
        for y in NSIA_YEARS:
            vals = {c: dist[c][y] for c in codes}
            floor = {c: int(v) for c, v in vals.items()}
            rest = round(sum(vals.values())) - sum(floor.values())
            for c in sorted(codes, key=lambda c: vals[c] - floor[c], reverse=True)[:rest]:
                floor[c] += 1
            for c in codes:
                out.append((c, 2, y, floor[c]))
            out.append((p, 1, y, sum(floor.values())))
    return out


def codps_rows():
    import openpyxl
    pc = list(csv.DictReader(CODPS_2026.open(encoding="utf-8")))[1:]  # row 2 is HXL tags
    d26 = {r["district_code"]: float(r["population_total"]) for r in pc}
    par = {r["district_code"]: r["province_code"] for r in pc}
    wb = openpyxl.load_workbook(CODPS_2021, read_only=True, data_only=True)
    ws = wb["afg_admpop_adm1_2021_v2"]
    rows = list(ws.iter_rows(values_only=True))
    h = list(rows[0])
    p21 = {r[h.index("Admin1_Code")]: int(r[h.index("T_TL")]) for r in rows[2:] if r[0]}
    p26 = defaultdict(float)
    for c, v in d26.items():
        p26[par[c]] += v
    out = []
    for c, v in d26.items():
        out.append((c, 2, 2026, round(v)))
        out.append((c, 2, 2021, round(p21[par[c]] * v / p26[par[c]])))
    for p in p26:
        for y in (2021, 2026):
            out.append((p, 1, y, sum(r[3] for r in out if r[1] == 2 and r[2] == y and par[r[0]] == p)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["nsia", "codps"], default="nsia")
    args = ap.parse_args()
    rows = nsia_rows() if args.source == "nsia" else codps_rows()
    for y in sorted({r[2] for r in rows}):
        n1 = sum(r[3] for r in rows if r[1] == 1 and r[2] == y)
        n2 = sum(r[3] for r in rows if r[1] == 2 and r[2] == y)
        print(f"  {y}: provinces {n1:,}  districts {n2:,}")
        assert n1 == n2
    rows.sort(key=lambda r: (r[1], r[0], r[2]))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["code", "level", "year", "pop"])
        w.writerows(rows)
    print(f"wrote {len(rows)} rows ({args.source}) -> {OUT}")


if __name__ == "__main__":
    main()
