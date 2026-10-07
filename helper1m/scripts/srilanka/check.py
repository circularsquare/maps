"""Checks for helper1m Sri Lanka, run after fetch.py and prep_boundaries.py. Read-only.

1. Districts and the national total, 2012 and 2024, against Table 3.3 of the CPH 2024
   final report (which prints both censuses by district); provinces against Table 3.2.
2. 2024 DS figures against the population DCS stores on its own DS polygons.
3. Boundary <-> population coverage at every level.
4. Every 2012 -> 2024 name pairing where the names differ, so a wrong pair is visible.
5. Growth 2012 -> 2024 by DS division: spread and extremes.
6. Kontur Population 2023-11-01 (400 m hexes) summed by DS polygon, as an independent
   witness for the 2024 DS figures and the boundaries.
7. Spot-check of a few big DS divisions.
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "6")

import json  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import geopandas as gpd  # noqa: E402
import pandas as pd  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent))
import a1_2012  # noqa: E402
import download  # noqa: E402
import fetch  # noqa: E402

HELPER = Path(__file__).resolve().parents[2]
RAW = HELPER / "data" / "srilanka" / "raw"
POP = HELPER / "data" / "srilanka" / "population.csv"
BND = HELPER / "data" / "srilanka" / "boundaries"
KONTUR = RAW / "kontur_population_LK_20231101.gpkg"


def report_tables():
    """Table 3.3 (districts, 1981-2024) and Table 3.2 (provinces, 2024) from the report."""
    import fitz
    doc = fitz.open(RAW / "CPH2024_Final_Eng.pdf")
    t33 = t32 = None
    for page in doc:
        t = " ".join(page.get_text().split())
        if "Table 3.3 : Population and Average Annual Growth Rate by District" in t:
            t33 = t
        if "Table 3.2 : Distribution of Population by Province and District" in t:
            t32 = t
    names = ["Sri Lanka", "Colombo", "Gampaha", "Kalutara", "Kandy", "Matale",
             "Nuwara Eliya", "Galle", "Matara", "Hambantota", "Jaffna", "Mannar", "Vavuniya",
             "Mullaitivu", "Kilinochchi", "Batticaloa", "Ampara", "Trincomalee",
             "Kurunegala", "Puttalam", "Anuradhapura", "Polonnaruwa", "Badulla",
             "Moneragala", "Ratnapura", "Kegalle"]
    num = r"(\d{1,3}(?:,\d{3})+|-)"
    dist = {}
    for nm in names:
        m = re.search(re.escape(nm) + r" " + num + r" " + num + r" " + num + r" " + num, t33)
        if not m:
            raise SystemExit(f"Table 3.3: no row for {nm}")
        dist[nm] = (int(m.group(3).replace(",", "")), int(m.group(4).replace(",", "")))
    prov = {}
    for nm in fetch_provinces():
        m = re.search(re.escape(nm) + r" Province " + num, t32)
        prov[nm] = int(m.group(1).replace(",", ""))
    return dist, prov


def fetch_provinces():
    import prep_boundaries
    return list(prep_boundaries.PROVINCES.values())


def main():
    pop = pd.read_csv(POP, dtype={"code": str})
    wide = pop.pivot_table(index=["level", "code"], columns="year", values="pop")
    names = {}
    for lv in (1, 2, 3):
        b = gpd.read_file(BND / f"adm{lv}.gpkg")
        names.update({(lv, c): n for c, n in zip(b["code"], b["name"])})

    print("== 1. totals against the CPH 2024 final report")
    dist, prov = report_tables()
    nat = {y: int(wide.loc[1][y].sum()) for y in (2012, 2024)}
    print(f"  national 2012 {nat[2012]:,} (report {dist['Sri Lanka'][0]:,}), "
          f"2024 {nat[2024]:,} (report {dist['Sri Lanka'][1]:,})")
    bad = 0
    for folder, dc in fetch.DISTRICT_CODE.items():
        nm = names[(2, dc)]
        want = dist[nm]
        got = (int(wide.loc[(2, dc)][2012]), int(wide.loc[(2, dc)][2024]))
        if got != want:
            bad += 1
            print(f"  BAD district {nm}: {got} vs report {want}")
    print(f"  districts: {25 - bad}/25 equal Table 3.3 in both years")
    pb = 0
    for code, nm in fetch_prov_codes().items():
        got = int(wide.loc[(1, code)][2024])
        if got != prov[nm]:
            pb += 1
            print(f"  BAD province {nm}: {got:,} vs report {prov[nm]:,}")
    print(f"  provinces: {9 - pb}/9 equal Table 3.2 (2024)")

    print("\n== 2. 2024 DS figures against DCS's DS polygons")
    fc = json.loads((RAW / "arcgis" / "dsd.geojson").read_text(encoding="utf-8"))
    agol = {str(int(f["properties"]["ds_uid"])): f["properties"]["SUM_Popula"]
            for f in fc["features"]}
    diff = [c for c in agol if agol[c] != wide.loc[(3, c)][2024]]
    print(f"  {340 - len(diff)}/340 identical")

    print("\n== 3. coverage")
    for lv in (1, 2, 3):
        b = set(gpd.read_file(BND / f"adm{lv}.gpkg")["code"])
        have = set(wide.loc[lv].index)
        full = wide.loc[lv].dropna()
        print(f"  adm{lv}: {len(b)} polygons, {len(have)} coded rows, no population "
              f"{len(b - have)}, no polygon {len(have - b)}, with both years {len(full)}")

    print("\n== 4. 2012 -> 2024 pairings where the name changed")
    ds24 = {c: (d, nm) for c, (d, nm, _) in fetch.read_2024().items()}
    for folder in download.DISTRICTS_2012:
        d = fetch.DISTRICT_CODE[folder]
        _, rows12 = a1_2012.read(folder)
        split12 = {p for ps, _ in fetch.SPLITS.get(d, []) for p in ps}
        n24 = {nm: c for c, (dd, nm) in ds24.items() if dd == d}
        f24 = {fetch.fold(nm): nm for nm in n24}
        for nm, _ in rows12:
            if nm in split12:
                continue
            t = fetch.ALIAS.get((d, nm)) or f24[fetch.fold(nm)]
            if t != nm:
                how = "alias" if (d, nm) in fetch.ALIAS else "fold "
                print(f"  {how} {d} {nm!r:<40} -> {t!r}")

    print("\n== 5. growth 2012 -> 2024 by DS division")
    ds = wide.loc[3].copy()
    ds["g"] = ds[2024] / ds[2012] - 1
    ds["name"] = [names[(3, c)] for c in ds.index]
    print(f"  median {ds['g'].median():+.1%}, p5 {ds['g'].quantile(.05):+.1%}, "
          f"p95 {ds['g'].quantile(.95):+.1%}")
    for lab, part in (("fastest", ds.nlargest(8, "g")), ("slowest", ds.nsmallest(8, "g"))):
        print(f"  {lab}: " + "; ".join(f"{r['name']} {r['g']:+.0%}"
                                       for _, r in part.iterrows()))

    print("\n== 6. Kontur 2023-11 (hex centroids in DS polygons)")
    if KONTUR.exists():
        k = gpd.read_file(KONTUR)
        k["geometry"] = k.geometry.centroid          # in the file's own metric CRS
        k = k.to_crs("EPSG:4326")
        b3 = gpd.read_file(BND / "adm3.gpkg")
        j = gpd.sjoin(k, b3[["code", "geometry"]], predicate="within", how="left")
        kk = j.groupby("code")["population"].sum()
        miss = j["code"].isna()
        print(f"  Kontur total {k['population'].sum():,.0f}; outside every DS polygon "
              f"{j.loc[miss, 'population'].sum():,.0f}")
        # census 2023, interpolated on the line, against Kontur 2023
        c23 = ds[2012] + (ds[2024] - ds[2012]) * (2023 - 2012) / 12
        df = pd.DataFrame({"census": c23, "kontur": kk}).fillna(0)
        f = df["kontur"].sum() / df["census"].sum()
        df["rel"] = df["kontur"] / df["census"] / f
        for lab, lv in (("DS", 3), ("district", 2)):
            if lv == 2:
                g = df.groupby(df.index.str[:2])[["census", "kontur"]].sum()
                r = g["kontur"] / g["census"] / f
            else:
                r = df["rel"]
            print(f"  {lab}: national factor {f:.3f}; after it, within 10% "
                  f"{((r - 1).abs() <= .10).mean():.0%}, within 25% "
                  f"{((r - 1).abs() <= .25).mean():.0%}, quartiles "
                  f"{r.quantile(.25):.2f}-{r.quantile(.75):.2f}")
        df["name"] = [names[(3, c)] for c in df.index]
        df["dev"] = df["rel"].apply(lambda v: max(v, 1 / v) if v > 0 else 99)
        print("  furthest out (Kontur/census after the national factor):")
        for c, r in df.sort_values("dev", ascending=False).head(12).iterrows():
            print(f"    {c} {r['name']:<34} census {int(r['census']):>8,}  Kontur "
                  f"{int(r['kontur']):>8,}  x{r['rel']:.2f}")
    else:
        print("  (no Kontur file; run download.py)")

    print("\n== 7. spot checks (2024 census, DS division)")
    for c in ("1103", "1127", "1112", "2130", "3139", "4136", "5118", "6154", "1203"):
        print(f"  {c} {names[(3, c)]:<38} 2012 {int(ds.loc[c, 2012]):>8,}  "
              f"2024 {int(ds.loc[c, 2024]):>8,}")


def fetch_prov_codes():
    import prep_boundaries
    return prep_boundaries.PROVINCES


if __name__ == "__main__":
    main()
