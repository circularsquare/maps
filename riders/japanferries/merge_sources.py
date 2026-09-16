"""Combine sources/*.csv (one file per research region) into data/route_figures.csv.

Every source file shares one column layout (see README). This adds a `region`
column (the file name), parses the numbers, and prints a per-region summary
plus anything that looks wrong: a bad header, an unparseable number, a year
outside 2000-2026, or an implausible passenger count.
"""
import csv
import glob
import os
import re
import sys
from collections import defaultdict

sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCES = os.path.join(HERE, "sources")
OUT = os.path.join(HERE, "data", "route_figures.csv")
FIELDS = ["prefecture", "operator", "route", "ports", "vessel", "fiscal_year", "year_basis",
          "annual_passengers", "passenger_km", "route_km", "needs_download", "source_title",
          "source_url", "source_page", "notes"]


def parse_num(s):
    s = (s or "").strip().replace(",", "")
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return "bad"


def parse_year(s):
    m = re.search(r"(19|20)\d\d", s or "")
    return int(m.group(0)) if m else None


def main():
    rows, problems = [], []
    for path in sorted(glob.glob(os.path.join(SOURCES, "*.csv"))):
        region = os.path.splitext(os.path.basename(path))[0]
        with open(path, encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames != FIELDS:
                problems.append(f"{region}: header differs: {reader.fieldnames}")
            for n, r in enumerate(reader, start=2):
                pax, pkm, rkm = (parse_num(r.get(k)) for k in ("annual_passengers", "passenger_km", "route_km"))
                year = parse_year(r.get("fiscal_year"))
                for label, v in (("annual_passengers", pax), ("passenger_km", pkm), ("route_km", rkm)):
                    if v == "bad":
                        problems.append(f"{region}:{n} unparseable {label}: {r.get(label)!r}")
                if isinstance(pax, float) and not 0 < pax < 30_000_000:
                    problems.append(f"{region}:{n} odd passenger count {pax:,.0f} ({r.get('route')})")
                if year is not None and not 2000 <= year <= 2026:
                    problems.append(f"{region}:{n} odd year {year} ({r.get('route')})")
                out = {"region": region, **{k: r.get(k, "") for k in FIELDS}}
                out["year"] = year if year is not None else ""
                out["passengers"] = int(pax) if isinstance(pax, float) else ""
                out["passenger_km_num"] = pkm if isinstance(pkm, float) else ""
                out["route_km_num"] = rkm if isinstance(rkm, float) else ""
                rows.append(out)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT}: {len(rows)} rows")

    by_region = defaultdict(list)
    for r in rows:
        by_region[r["region"]].append(r)
    print(f"\n{'region':<22} {'rows':>5} {'w/ pax':>7} {'routes':>7} {'pkm':>5} {'to dl':>6}  years")
    for region, rs in by_region.items():
        with_pax = [r for r in rs if r["passengers"] != ""]
        routes = {(r["prefecture"], r["route"]) for r in with_pax}
        years = sorted({r["year"] for r in with_pax if r["year"] != ""})
        span = f"{years[0]}-{years[-1]}" if years else "-"
        pkm = sum(1 for r in rs if r["passenger_km_num"] != "")
        dl = sum(1 for r in rs if (r["needs_download"] or "").strip().lower() == "yes")
        print(f"{region:<22} {len(rs):>5} {len(with_pax):>7} {len(routes):>7} {pkm:>5} {dl:>6}  {span}")

    print("\nlargest latest-year figure per route, per region (top 6):")
    for region, rs in by_region.items():
        latest = {}
        for r in rs:
            if r["passengers"] == "" or r["year"] == "":
                continue
            k = (r["prefecture"], r["route"])
            if k not in latest or r["year"] > latest[k]["year"]:
                latest[k] = r
        top = sorted(latest.values(), key=lambda r: -r["passengers"])[:6]
        print(f"  {region}:")
        for r in top:
            print(f"    {r['passengers']:>10,}  {r['year']} {r['year_basis']:<3} {r['prefecture']} {r['route'][:40]}")

    if problems:
        print(f"\n{len(problems)} problems:")
        for p in problems[:60]:
            print("  " + p)


if __name__ == "__main__":
    main()
