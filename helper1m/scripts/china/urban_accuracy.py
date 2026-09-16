"""What the township counts are worth where a 1M region covers little ground.

Two different things get confused when asking whether a 100 m grid is "accurate
enough", so they are measured apart.

**Resolution** is how many cells land in the unit. It is not the binding
constraint for regions: even in the densest band a million people cover
thousands of cells. It does bind for a handful of individual subdistricts —
the very densest are under a hundred cells, and a few are under twenty.

**Agreement** is whether our township is the polygon ASPECT actually spread that
township's census count over. Ours are 2018 and theirs are 2019 updated to 2020,
43,655 against 40,718, so often it is not, and then the figure is the grid
model's guess at a partial overlap rather than a count. This is the error that
matters, and the only place it can be measured against published numbers is the
county level, so that is what the second table does.

The reassuring part is that these errors cancel on aggregation. They come in
pairs — an urban core short, the county beside it long by nearly the same,
because that core annexed populated fringe after 2018. A region assembled from
dozens of townships is far better than any single one of them.
"""
import io
import json
import math
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "data/china"
ADM4 = HERE.parents[1] / "countries/china/adm4"

PIX_DEG = 0.00089832
M_PER_DEG_LAT = 111132.0
M_PER_DEG_LON = 111320.0

DENSITY_BANDS = [(0, 500), (500, 2000), (2000, 5000), (5000, 15000), (15000, None)]


def cell_area_km2(lat):
    return (PIX_DEG * M_PER_DEG_LAT) * (PIX_DEG * M_PER_DEG_LON *
                                        math.cos(math.radians(lat))) / 1e6


def band_label(lo, hi):
    return f"{lo:,}-{hi:,}" if hi else f"{lo:,}+"


def bands(frame):
    for lo, hi in DENSITY_BANDS:
        b = frame[(frame["density"] >= lo) & (frame["density"] < (hi or 1e12))]
        if len(b):
            yield band_label(lo, hi), b


def load_townships():
    rows = []
    for path in sorted(ADM4.glob("*.geojson")):
        for f in json.loads(path.read_text(encoding="utf-8"))["features"]:
            p = f["properties"]
            rows.append((p["code"], p["name"], p["area_km2"],
                         p["populations"].get("2020", 0)))
    t = pd.DataFrame(rows, columns=["code", "name", "area_km2", "pop"])
    t = t[(t["pop"] > 0) & (t["area_km2"] > 0)].copy()
    t["density"] = t["pop"] / t["area_km2"]
    t["cells"] = t["area_km2"] / cell_area_km2(34)
    return t


def main():
    print("cell size on the ground — the grid is square in degrees, not metres,")
    print("so cells narrow towards the north:")
    for lat, where in ((20, "Hainan"), (31, "Shanghai"), (40, "Beijing"),
                       (50, "Harbin")):
        ns = PIX_DEG * M_PER_DEG_LAT
        ew = PIX_DEG * M_PER_DEG_LON * math.cos(math.radians(lat))
        print(f"   {lat}N  {where:<10} {ns:5.1f} x {ew:5.1f} m")
    print()

    t = load_townships()
    print("resolution, by township density:")
    print(f"{'people /km2':<16}{'townships':>11}{'median cells':>14}"
          f"{'p5 cells':>10}{'km2 per 1M':>12}{'cells per 1M':>14}")
    for label, b in bands(t):
        km2 = 1e6 / b["density"].median()
        print(f"{label:<16}{len(b):>11,}{int(b['cells'].median()):>14,}"
              f"{int(b['cells'].quantile(.05)):>10,}"
              f"{km2:>12,.1f}{int(km2 / cell_area_km2(34)):>14,}")

    # Anything this small is fewer numbers than a city block, and the implied
    # density runs past the densest places on earth, so the polygon is drawn too
    # small rather than the people being real. Worth knowing, not worth fixing:
    # they are a rounding error on the country.
    tiny = t[t["cells"] < 50]
    print()
    print(f"townships under 50 cells: {len(tiny)} of {len(t)} "
          f"({len(tiny) / len(t):.2%}), holding {tiny['pop'].sum() / 1e6:.2f} M "
          f"people, {tiny['pop'].sum() / t['pop'].sum():.2%} of the country.")
    print("Their figures are the least trustworthy in the set.")

    print()
    print("the densest townships, where resolution really is thin:")
    for r in t.sort_values("density", ascending=False).head(6).itertuples():
        print(f"   {r.name:<30} {int(r.pop):>8,} in {r.area_km2:>5.2f} km2 "
              f"= {int(r.density):>7,}/km2, {int(r.cells):>5,} cells")

    # Agreement, against the published county census.
    cnty = pd.read_csv(DATA / "county_ratios.csv", dtype={"code": str})
    pop = pd.read_csv(DATA / "population.csv", dtype={"code": str})
    cnty["built"] = cnty["code"].map(
        pop[(pop.level == 3) & (pop.year == 2020)].set_index("code")["pop"])
    cnty["area_km2"] = cnty["code"].map(t.groupby(t["code"].str[:6])["area_km2"].sum())
    m = cnty[cnty["panel_2020"].notna() & cnty["area_km2"].notna()].copy()
    m["density"] = m["panel_2020"] / m["area_km2"]
    m["err"] = ((m["built"] - m["panel_2020"]) / m["panel_2020"]).abs()

    print()
    print("agreement — error against the published county census, by county density:")
    print(f"{'people /km2':<16}{'counties':>11}{'median':>9}{'p90':>8}"
          f"{'within 10%':>12}")
    for label, b in bands(m):
        print(f"{label:<16}{len(b):>11,}{b['err'].median():>8.1%}"
              f"{b['err'].quantile(.9):>8.1%}{(b['err'] <= .10).mean():>11.0%}")
    print(f"{'ALL':<16}{len(m):>11,}{m['err'].median():>8.1%}"
          f"{m['err'].quantile(.9):>8.1%}{(m['err'] <= .10).mean():>11.0%}")


if __name__ == "__main__":
    main()
