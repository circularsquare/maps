"""Russia stage 1: what Tariff Guide No. 4 (Тарифное руководство № 4) gives as a line register.

    python probe_ru_tr4.py > data/raw/ru/probe_tr4.txt

Reads the official XLS of the CIS Council for Rail Transport (sovetgt.org/tr4/<y>/<m>/<d>/, one
file per book, refreshed daily; see ru_sources.md) from data/raw/ru/:

- Book 1, tariff sections (участки): every section between two tariff nodes, its points IN
  ORDER with their six-digit ESR code and integer tariff km from each end. One sheet per
  railway; the Russian ones are marked "(Р)".
- Book 2, part 1 (sheet РП): every separation point (station, passing loop, post) with its
  commercial operations; П, Б and О are the passenger ones. Part 2 (sheet ОП): every passenger
  stopping point and platform, О/Б/П, or Х (no operations).

Writes data/raw/ru/tr4_sections.json (all Russian-administration sections, parsed) for the OSM
probes, and prints the counts.
"""
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import xlrd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw" / "ru"
HEAD = re.compile(r'^\s*\d+\)\s*участок\s+(\d+-\d+)\s+"(.*)"\s*(?:\((.*)\))?\s*$')
KM = re.compile(r"^\s*(-?\d+)\s*км")
# Sheets of the Russian administration that are NOT in Russia's internationally recognised
# territory. Crimea is in by Anita's decision (de facto); the 2022 annexations are listed so
# a reader can decide separately.
OCCUPIED_2022 = {"Донец (Р)", "ЛУГАН (Р)", "МЕЛИТ (Р)"}
CRIMEA = {"Крым (Р)"}


def newest(pattern):
    files = sorted(RAW.glob(pattern))
    if not files:
        sys.exit(f"no {pattern} in {RAW}")
    return files[-1]


def book1():
    wb = xlrd.open_workbook(str(newest("tr4_kniga1_*.xls")))
    out = []
    for sh in wb.sheets():
        if "(Р)" not in sh.name:
            continue
        cur = None
        for r in range(sh.nrows):
            row = [str(sh.cell_value(r, c)).strip() for c in range(sh.ncols)]
            m = HEAD.match(row[0])
            if m:
                cur = {"sheet": sh.name, "id": m.group(1), "name": m.group(2),
                       "type": (m.group(3) or "").strip(), "points": []}
                out.append(cur)
                continue
            if cur is None or row[0].startswith("№"):
                continue
            code = row[1].replace(".", "").strip()
            if re.fullmatch(r"\d{6}", code):
                kms = [KM.match(v) for v in row[3:]]
                kms = [int(k.group(1)) if k else None for k in kms]
                cur["points"].append({"esr": code, "name": row[2], "km": kms})
            elif not row[0] and not row[1] and row[2] and cur["points"]:
                cur["points"][-1]["name"] += " " + row[2]      # wrapped name
    return out


def book2():
    wb = xlrd.open_workbook(str(newest("tr4_kniga2_*.xls")))
    rp, op = {}, {}
    sh = wb.sheet_by_name("РП")
    last = None
    for r in range(6, sh.nrows):
        row = [str(sh.cell_value(r, c)).strip() for c in range(sh.ncols)]
        if row[0]:
            last = row[5]
            rp[last] = {"name": row[1], "ops": row[2], "road": row[3]}
        elif last and row[2]:
            rp[last]["ops"] += row[2]                     # wrapped operations
    sh = wb.sheet_by_name("ОП")
    for r in range(6, sh.nrows):
        row = [str(sh.cell_value(r, c)).strip() for c in range(sh.ncols)]
        if row[0] and row[4]:
            op[row[4]] = {"name": row[1], "ops": row[2], "road": row[3]}
    return rp, op


def passenger(code, rp, op):
    """'stop' if a passenger operation is listed, 'none' if listed without one, None if absent."""
    rec = rp.get(code) or op.get(code)
    if rec is None:
        return None
    letters = re.split(r"[\s,]", rec["ops"])
    return "stop" if any(x in ("П", "Б", "О") for x in letters) else "none"


def main():
    secs = book1()
    rp, op = book2()
    (RAW / "tr4_sections.json").write_text(
        json.dumps(secs, ensure_ascii=False, indent=0), "utf-8")

    ru_rp = {k: v for k, v in rp.items() if "(Р)" in v["road"]}
    ru_op = {k: v for k, v in op.items() if "(Р)" in v["road"]}
    print(f"Book 2: {len(rp)} separation points ({len(ru_rp)} Russian administration), "
          f"{len(op)} stopping points/platforms ({len(ru_op)} Russian)")
    pr = Counter(passenger(k, rp, op) for k in ru_rp)
    po = Counter(passenger(k, rp, op) for k in ru_op)
    print(f"  Russian separation points with a passenger operation: {pr['stop']} of {len(ru_rp)}")
    print(f"  Russian stopping points with a passenger operation:   {po['stop']} of {len(ru_op)}"
          f" (the rest are marked Х)")

    print(f"\nBook 1: {len(secs)} tariff sections on Russian-administration sheets")
    by_sheet = defaultdict(list)
    for s in secs:
        by_sheet[s["sheet"]].append(s)
    types = Counter(s["type"] for s in secs)
    print("  section types:", types.most_common())

    def length(s):
        k = [p["km"][0] for p in s["points"] if p["km"] and p["km"][0] is not None]
        return max(k) if k else 0

    print(f"\n  {'sheet':<14} {'sections':>8} {'main':>6} {'km main':>8} {'km all':>8} "
          f"{'points':>7} {'pax pts':>7}")
    tot = Counter()
    pts_all = {}
    for sheet, ss in sorted(by_sheet.items()):
        main = [s for s in ss if s["type"].startswith("Основной")]
        codes = {p["esr"] for s in ss for p in s["points"]}
        pax = sum(1 for c in codes if passenger(c, rp, op) == "stop")
        km_m = sum(length(s) for s in main)
        km_a = sum(length(s) for s in ss)
        flag = " *2022" if sheet in OCCUPIED_2022 else (" *Crimea" if sheet in CRIMEA else "")
        print(f"  {sheet:<14} {len(ss):>8} {len(main):>6} {km_m:>8,} {km_a:>8,} "
              f"{len(codes):>7} {pax:>7}{flag}")
        if sheet not in OCCUPIED_2022:
            tot["sections"] += len(ss)
            tot["main"] += len(main)
            tot["km_main"] += km_m
            tot["km_all"] += km_a
            for c in codes:
                pts_all[c] = passenger(c, rp, op)
    print(f"  Russia + Crimea (2022 annexations left out): {tot['sections']} sections, "
          f"{tot['main']} main, {tot['km_main']:,} km in main sections, "
          f"{tot['km_all']:,} km all sections")
    pc = Counter(pts_all.values())
    print(f"  distinct points {len(pts_all)}: passenger {pc['stop']}, no passenger op "
          f"{pc['none']}, not in Book 2 {pc[None]}")

    # Points with no passenger op that are not junctions: freight-only stretches show up as
    # sections whose inner points are all 'none'.
    freight = []
    for s in secs:
        if s["sheet"] in OCCUPIED_2022:
            continue
        inner = [p for p in s["points"][1:-1]]
        kinds = Counter(passenger(p["esr"], rp, op) for p in s["points"])
        if s["type"].startswith("Основной") and kinds["stop"] == 0:
            freight.append((length(s), s["id"], s["name"]))
    freight.sort(reverse=True)
    print(f"\n  main sections with NO passenger point at all: {len(freight)}, "
          f"{sum(f[0] for f in freight):,} km; longest:")
    for f in freight[:25]:
        print(f"    {f[0]:5d} km  {f[1]}  {f[2]}")

    # Sample: the Moscow - St Petersburg main line pieces, to show what a section looks like
    print("\n  sample sections:")
    for s in secs:
        if s["id"] in ("01-011", "17-002", "85-003", "96-001"):
            print(f"   {s['id']} {s['name']} ({s['type']}), {len(s['points'])} points, "
                  f"{length(s)} km")
            for p in s["points"][:8]:
                print(f"      {p['esr']} {p['name']:<32} {p['km']} {passenger(p['esr'], rp, op)}")

    # How many sections share points (a section's end is a node of others): sections chain
    # into longer routes. Count node degrees.
    deg = Counter()
    for s in secs:
        if s["points"]:
            deg[s["points"][0]["esr"]] += 1
            deg[s["points"][-1]["esr"]] += 1
    print(f"\n  section end nodes: {len(deg)}; degree 1 {sum(1 for v in deg.values() if v == 1)}, "
          f"2 {sum(1 for v in deg.values() if v == 2)}, 3+ {sum(1 for v in deg.values() if v >= 3)}")
    # repeated points: the same ESR on several sections (overlapping 'via' sections)
    occ = Counter(p["esr"] for s in secs for p in s["points"][1:-1])
    print(f"  inner points listed on 2+ sections: {sum(1 for v in occ.values() if v > 1)}")


if __name__ == "__main__":
    main()
