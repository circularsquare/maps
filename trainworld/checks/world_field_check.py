"""Check the world field (T-375): Kontur totals by country against the World Bank, fixed links
joined and open straits left open, and the reader counting every person once.

    ..\\venv\\Scripts\\python.exe checks/world_field_check.py      (from trainworld/)"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "3")
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
import world_field

JOINED = [  # pairs a fixed link joins: (name, lon, lat), (name, lon, lat), link
    (("Aomori", 140.74, 40.82), ("Hakodate", 140.73, 41.77), "Seikan Tunnel"),
    (("Shimonoseki", 130.94, 33.96), ("Kitakyushu", 130.88, 33.88), "Kanmon"),
    (("Kobe", 135.19, 34.69), ("Tokushima", 134.55, 34.07), "Akashi and Onaruto bridges"),
    (("Okayama", 133.93, 34.66), ("Takamatsu", 134.05, 34.34), "Great Seto Bridge"),
    (("Onomichi", 133.20, 34.41), ("Imabari", 133.00, 34.07), "Shimanami Kaido"),
    (("London", -0.12, 51.51), ("Paris", 2.35, 48.86), "Channel Tunnel"),
    (("Copenhagen", 12.57, 55.68), ("Malmo", 13.00, 55.60), "Oresund"),
    (("Odense", 10.39, 55.40), ("Copenhagen", 12.57, 55.68), "Great Belt"),
    (("Tung Chung", 113.94, 22.29), ("Zhuhai", 113.57, 22.27), "Hong Kong-Zhuhai-Macau Bridge"),
    (("Hong Kong Island", 114.16, 22.28), ("Shenzhen", 114.06, 22.54), "Hong Kong's harbour tunnels"),
    (("Charlottetown", -63.13, 46.24), ("Moncton", -64.78, 46.09), "Confederation Bridge"),
    (("Manama", 50.58, 26.22), ("Dammam", 50.10, 26.43), "King Fahd Causeway"),
    (("Singapore", 103.82, 1.35), ("Johor Bahru", 103.76, 1.49), "Johor-Singapore Causeway"),
]
APART = [  # no fixed link
    (("London", -0.12, 51.51), ("Dublin", -6.26, 53.35)),
    (("Tokyo", 139.69, 35.69), ("Seoul", 126.98, 37.57)),
    (("Palermo", 13.36, 38.12), ("Rome", 12.50, 41.90)),
    (("Taipei", 121.56, 25.04), ("Shanghai", 121.47, 31.23)),
    (("Palma", 2.65, 39.57), ("Barcelona", 2.17, 41.39)),
    (("Yuzhno-Sakhalinsk", 142.74, 46.96), ("Sapporo", 141.35, 43.06)),
    (("Yuzhno-Sakhalinsk", 142.74, 46.96), ("Khabarovsk", 135.07, 48.48)),  # Sakhalin Tunnel unbuilt
    (("Medan", 98.67, 3.59), ("Kuala Lumpur", 101.69, 3.14)),  # Malacca Strait Bridge unbuilt
    (("Jakarta", 106.85, -6.21), ("Palembang", 104.76, -2.98)),
    (("Colombo", 79.86, 6.93), ("Chennai", 80.27, 13.08)),
    (("Hobart", 147.33, -42.88), ("Melbourne", 144.96, -37.81)),
    (("Honolulu", -157.86, 21.31), ("Los Angeles", -118.24, 34.05)),
    (("Auckland", 174.76, -36.85), ("Wellington", 174.78, -41.29)),  # same island: control below
]


def main():
    wf = world_field.load()
    r5, C = wf.r5, wf.countries
    ok = True
    print(f"cells r5 {len(r5.pop):,} r4 {len(wf.r4.pop):,} r3 {len(wf.r3.pop):,}; people {r5.pop.sum() / 1e9:.3f} bn")

    # countries
    tot = np.bincount(r5.country[r5.country >= 0], weights=r5.pop[r5.country >= 0], minlength=len(C))
    print(f"people in no country: {r5.pop[r5.country < 0].sum():,.0f}")
    rows = [(C[i]["iso2"] or C[i]["iso3"], C[i]["name"], tot[i], C[i]["wb_pop"], C[i]["gdp"], C[i]["gdp_src"])
            for i in range(len(C)) if C[i]["wb_pop"] and C[i]["wb_pop"] >= 1e6]
    rows.sort(key=lambda r: -r[3])
    within5 = within10 = 0
    w_all = sum(r[3] for r in rows)
    print(f"\n{'':4}{'country':28}{'Kontur M':>10}{'WB M':>9}{'ratio':>7}{'GDP/head':>10}  GDP source")
    for iso, name, k, wb, g, src in rows:
        rr = k / wb
        within5 += wb * (abs(rr - 1) <= 0.05); within10 += wb * (abs(rr - 1) <= 0.10)
        flag = "" if abs(rr - 1) <= 0.10 else "  <- off by over 10%"
        if wb >= 2e7 or flag:
            print(f"{iso:4}{name[:27]:28}{k / 1e6:10.2f}{wb / 1e6:9.2f}{rr:7.3f}{g:10,.0f}  {src}{flag}")
    print(f"\npeople in countries within 5% of the World Bank: {within5 / w_all:.1%}; within 10%: {within10 / w_all:.1%}")
    world = sum(c["wb_pop"] or 0 for c in C)
    print(f"world: Kontur {r5.pop.sum() / 1e9:.3f} bn, World Bank (countries summed) {world / 1e9:.3f} bn")
    if within10 / w_all < 0.9 or abs(r5.pop.sum() / world - 1) > 0.05:
        ok = False; print("FAIL country totals")
    else:
        print("PASS country totals")
    fallback = [c["name"] for c in C if not c["gdp_src"].startswith("World Bank") and c["wb_pop"] and c["wb_pop"] > 1e6]
    print(f"GDP not from the World Bank (over 1M people): {', '.join(fallback)}")

    # fixed links
    print()
    for a, b, link in JOINED:
        la, lb = wf.locate(a[1], a[2])[1], wf.locate(b[1], b[2])[1]
        good = la == lb
        ok &= good
        print(f"{'PASS' if good else 'FAIL'} joined {a[0]} - {b[0]} ({link}): landmass {la} / {lb}")
    for a, b in APART:
        la, lb = wf.locate(a[1], a[2])[1], wf.locate(b[1], b[2])[1]
        good = la != lb if a[0] != "Auckland" else la == lb
        ok &= good
        print(f"{'PASS' if good else 'FAIL'} {'apart' if a[0] != 'Auckland' else 'same island'} {a[0]} - {b[0]}: landmass {la} / {lb}")

    # reader: every person once, at each place
    for name, lon, lat in [("New York", -73.98, 40.75), ("Tokyo", 139.69, 35.69), ("Fiji", 178.44, -18.14)]:
        f = wf.around(lon, lat)
        n5, n4, n3 = (f["res"] == 5).sum(), (f["res"] == 4).sum(), (f["res"] == 3).sum()
        good = abs(f["pop"].sum() / r5.pop.sum() - 1) < 1e-9
        ok &= good
        print(f"{'PASS' if good else 'FAIL'} around {name}: {len(f['pop']):,} cells (r5 {n5:,}, r4 {n4:,}, r3 {n3:,})")
    print("\nALL PASS" if ok else "\nSOME FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
