"""Checks for the Afghanistan build (reads only; prints).

1. National and province totals in population.csv against NSIA's own
   three-year province table in the 1405 PDF (Table 4: 1403, 1404, 1405).
2. NSIA and COD-PS against UN WPP 2024 nationally, and against Kontur
   (2023 hexes summed into the districts by kontur.py) for how each spreads
   people: Kontur is scaled to each source's total (nationally for provinces,
   within the province for districts) and the share of units within 10% is
   reported, with the worst districts.

Run fetch.py (NSIA) first; COD-PS is read from its own files here.
"""
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import nsia  # noqa: E402

HELPER = HERE.parents[1]
REPO = HELPER.parent
DATA = HELPER / "data" / "afghanistan"
WPP = {2025: 43_844_111, 2026: 45_047_069}  # WPP 2024, medium, mid-year


def table4():
    """{province: {2024: p, 2025: p, 2026: p}} and the national settled line,
    from the 1405 PDF's 'by province and sex in the last three years' table."""
    import fitz
    doc = fitz.open(DATA / "raw" / "nsia_1405_v4.pdf")
    for pno, page in enumerate(doc):
        t = page.get_text()
        if "Population of Afghanistan by" in t and "last three" in t.lower() and "Kabul" in t:
            break
    else:
        raise SystemExit("three-year province table not found")
    # the table runs onto the next page
    t += "\n" + doc[pno + 1].get_text()
    lines = [l.strip() for l in t.splitlines() if l.strip()]
    num = re.compile(r"^(\d{1,3}(?:,\d{3})+|\d+)")
    out, i = {}, 0
    nat = None
    while i < len(lines):
        if re.fullmatch(r"\d{1,2}", lines[i]) and re.search(r"[A-Za-z]", lines[i + 1]):
            name = lines[i + 1].strip()
            vals = []
            j = i + 2
            while len(vals) < 9 and j < len(lines) and num.match(lines[j]):
                vals.append(int(num.match(lines[j]).group(1).replace(",", "")))
                j += 1
            if len(vals) == 9:
                out[name] = {2024: vals[2], 2025: vals[5], 2026: vals[8]}
            i = j
            continue
        if lines[i] == "Total" and nat is None:
            vals = [int(num.match(l).group(1).replace(",", "")) for l in lines[i + 1:i + 10]]
            nat = {2024: vals[2], 2025: vals[5], 2026: vals[8]}
        i += 1
    return out, nat


def read_pop():
    pop = defaultdict(dict)
    for r in csv.DictReader((DATA / "population.csv").open(encoding="utf-8")):
        pop[(int(r["level"]), r["code"])][int(r["year"])] = int(r["pop"])
    return pop


def within(a, b, tol=0.10):
    return abs(a / b - 1) <= tol if b else False


def main():
    import geopandas as gpd
    adm2 = gpd.read_file(DATA / "boundaries" / "adm2.gpkg", ignore_geometry=True)
    parent = dict(zip(adm2["code"], adm2["parent"]))
    name = dict(zip(adm2["code"], adm2["name"]))
    pname = dict(zip(adm2["parent"], adm2["parent_name"]))
    kon = {r["code"]: int(r["kontur"]) for r in csv.DictReader((DATA / "kontur_adm2.csv").open(encoding="utf-8"))}
    pop = read_pop()
    a1codes = set(adm2["parent"])
    for lv, codes in ((1, a1codes), (2, set(adm2["code"]))):
        have = {c for (l, c) in pop if l == lv}
        print(f"level {lv}: {len(codes)} boundary units, {len(have)} with population; "
              f"no population: {sorted(codes - have)}; no boundary: {sorted(have - codes)}")

    # 1. totals against NSIA's own table
    t4, nat = table4()
    print("== national settled population, ours vs NSIA 1405 Table 4")
    for y in (2024, 2025, 2026):
        ours = sum(v[y] for (lv, c), v in pop.items() if lv == 1)
        print(f"  {y}: {ours:,} vs {nat[y]:,} ({ours - nat[y]:+,})")
    # NSIA province names in table order == our province order in the 1404 tables
    order = list(dict.fromkeys(r["prov"] for r in nsia.read_xlsx(DATA / "raw" / "nsia_1404.xlsx")))
    mp = {}
    for row in csv.DictReader((HERE / "nsia_to_cod.csv").open(encoding="utf-8")):
        tgt = row["target"].replace("split:", "").split(",")[0].split(";")[0].split(":")[0]
        mp.setdefault(row["province"], parent[tgt])
    worst = 0
    for i, (tname, vals) in enumerate(t4.items()):
        pc = mp[order[i]]
        for y in (2024, 2025, 2026):
            worst = max(worst, abs(pop[(1, pc)][y] - vals[y]))
    print(f"  34 provinces x 3 years: largest gap to Table 4 is {worst} people ({len(t4)} rows read)")

    # 2. sources against WPP and Kontur
    codps = {}
    rows = list(csv.DictReader((REPO / "religiondots/data/raw/af/afg_admpop_adm2_2026.csv").open(encoding="utf-8")))[1:]
    for r in rows:
        codps[r["district_code"]] = float(r["population_total"])
    nsia26 = {c: v[2026] for (lv, c), v in pop.items() if lv == 2}
    print("\n== national, against UN WPP 2024 (medium, mid-year)")
    print(f"  NSIA 1405 settled {sum(nsia26.values()):,} + 1.5M Kuchi = {sum(nsia26.values()) + 1_500_000:,}: "
          f"{(sum(nsia26.values()) + 1_500_000) / WPP[2026]:.3f} of WPP 2026 ({WPP[2026]:,})")
    print(f"  COD-PS 2026 {sum(codps.values()):,.0f}: {sum(codps.values()) / WPP[2026]:.3f} of WPP 2026")
    print(f"  Kontur 2023 {sum(kon.values()):,}: {sum(kon.values()) / WPP[2025]:.3f} of WPP 2025")

    for label, src in (("NSIA 2026", nsia26), ("COD-PS 2026", codps)):
        tot, ktot = sum(src.values()), sum(kon.values())
        ps, pk = defaultdict(float), defaultdict(float)
        for c in src:
            ps[parent[c]] += src[c]
            pk[parent[c]] += kon[c]
        prov_ok = sum(within(ps[p], pk[p] * tot / ktot) for p in ps)
        ratios = sorted((ps[p] / (pk[p] * tot / ktot), pname[p]) for p in ps)
        print(f"\n== {label}: provinces within 10% of Kontur (scaled to national): {prov_ok}/34")
        print("  lowest " + ", ".join(f"{n} {r:.2f}" for r, n in ratios[:4]))
        print("  highest " + ", ".join(f"{n} {r:.2f}" for r, n in ratios[-4:]))
        dr = []
        for c in src:
            k = kon[c] * ps[parent[c]] / pk[parent[c]]
            dr.append((src[c] / k if k else float("inf"), c, src[c], k))
        ok = sum(within(s, k) for _, _, s, k in dr)
        ok25 = sum(within(s, k, 0.25) for _, _, s, k in dr)
        big = [d for d in dr if d[2] >= 100_000]
        okbig = sum(within(s, k) for _, _, s, k in big)
        print(f"  districts within 10% of Kontur (scaled within province): {ok}/401; within 25%: {ok25}/401; "
              f"of districts over 100k: {okbig}/{len(big)}")
        dr.sort()
        print("  lowest: " + "; ".join(f"{name[c]} ({pname[parent[c]]}) {r:.2f} [{s:,.0f} vs {k:,.0f}]" for r, c, s, k in dr[:6]))
        print("  highest: " + "; ".join(f"{name[c]} ({pname[parent[c]]}) {r:.2f} [{s:,.0f} vs {k:,.0f}]" for r, c, s, k in dr[-6:]))

    # NSIA vs COD-PS per district, after scaling COD-PS to NSIA's province totals
    ps_n, ps_c = defaultdict(float), defaultdict(float)
    for c in nsia26:
        ps_n[parent[c]] += nsia26[c]
        ps_c[parent[c]] += codps[c]
    agree = sum(within(nsia26[c], codps[c] * ps_n[parent[c]] / ps_c[parent[c]]) for c in nsia26)
    print(f"\n== NSIA vs COD-PS district shapes (COD-PS scaled to NSIA's province totals): {agree}/401 within 10%")


if __name__ == "__main__":
    main()
