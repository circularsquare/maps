"""Cyprus — the drawn units: the 396 municipalities and communities the 2021 census enumerated.

Writes data/geo/cy/cy_lau.gpkg (`unit`, `name`, `district`) and data/geo/cy/cy_lookup.csv.

**THE JOIN IS AN INTEGER, AND CYPRUS HANDS IT OVER FOR FREE.** CYSTAT's PxWeb table 1891213E
labels its community dimension in English (`Lefkosia`, `Agios Dometios`) and *codes* it with
Cyprus's own LAU code (`1000`, `1010`). Eurostat's GISCO LAU 2021 layer carries the same code
in `LAU_ID`, and the bundle's `EU-27-LAU-2021-NUTS-2021.xlsx` carries a `LAU NAME LATIN`
column that is the same romanisation, string for string. So the join is made on the code and
the 396 names are then asserted equal as a free second check. That order matters:
[[reference_name_join_wrong_neighbour]] is invisible to every totals test, and Cyprus is full
of the shape it needs — `Agios Theodoros` exists in Larnaka and in Lemesos, `Kellaki`,
`Kalo Chorio` and `Pyrgos` all repeat across districts, and CYSTAT and GISCO disambiguate them
the same way (`Agios Theodoros Lemesou`) only because they are the same list.

**615 POLYGONS, 396 OF THEM ENUMERATED, AND THE 219 LEFT OVER ARE TWO DIFFERENT THINGS.**
GISCO covers the whole island. Eurostat's own workbook prints `n.a.` for the population of
**182** of them, which is the area under Turkish Cypriot administration, and **0** for 37 more.
The 37 are not a data problem: they are the Turkish Cypriot villages of Pafos and Larnaka
emptied in 1974 (Vretsia, Fasli, Melandra, Sarama, Evretou, Trimithousa, Kios, Zacharia,
Lapithiou, Foinikas, Maronas, Livadi and the rest), plus the uninhabited Troodos summit, and
nobody was enumerated in any of them. The split is asserted rather than assumed.

**KERYNEIA HAS NO CODE AT ALL.** Cyprus's six districts are numbered 1 to 6 and the census's
district list runs **1, 3, 4, 5, 6**: district 2 is Keryneia, all 47 of its communities are in
GISCO, and none is in the census. Ammochostos loses 89 of its 98 and Lefkosia 65 of its 174.
5,846 km² are drawn of the island's 9,249, which is 63.2%.

Usage:
    python sources/cy_geo.py     no download; the GISCO LAU 2021 bundle is a shared asset
"""

import csv
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "6")          # [[feedback_cap_cpu]]
os.environ.setdefault("OPENBLAS_NUM_THREADS", "6")
os.environ.setdefault("MKL_NUM_THREADS", "6")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

LAU_SHP = os.path.join(ROOT, "data", "geo", "lau2021", "shp4326",
                       "LAU_RG_01M_2021_4326.shp")
LAU_XLSX = os.path.join(ROOT, "data", "geo", "lau2021",
                        "EU-27-LAU-2021-NUTS-2021.xlsx")
CENSUS_JSON = os.path.join(ROOT, "data", "raw", "cy", "cit_comm.json")

OUT_DIR = os.path.join(ROOT, "data", "geo", "cy")
OUT = os.path.join(OUT_DIR, "cy_lau.gpkg")
LOOKUP = os.path.join(OUT_DIR, "cy_lookup.csv")

EXPECTED_COMMUNITIES = 396
EXPECTED_GISCO = 615
# Cyprus in whole: 9,251 km² by every published figure; GISCO's polygons sum to 9,249.
ISLAND_KM2 = 9249.0
# The government-controlled area is quoted at about 5,896 km². GISCO's 396 enumerated
# communities come to 5,846, 0.8% under, because the UN buffer zone and the two Sovereign
# Base Areas are carved differently. Asserted as a band, not a number.
DRAWN_KM2 = (5700.0, 6000.0)

# The five districts that have communities in the census. 2 is Keryneia and is absent.
DISTRICTS = {"1": "Lefkosia", "3": "Ammochostos", "4": "Larnaka",
             "5": "Lemesos", "6": "Pafos"}


def census_communities():
    """(code -> English name) for the 396 communities CYSTAT enumerated in 2021."""
    if not os.path.exists(CENSUS_JSON):
        sys.exit(f"missing {CENSUS_JSON} -- run `python sources/cy.py --fetch` first")
    with open(CENSUS_JSON, encoding="utf-8") as fh:
        js = json.load(fh)
    cat = js["dimension"]["DISTRICT, MUNICIPALITY/COMMUNITY"]["category"]
    idx = cat["index"]
    keys = sorted(idx, key=lambda k: idx[k]) if isinstance(idx, dict) else list(idx)
    out = {k: cat["label"][k] for k in keys if k.isdigit() and len(k) == 4}
    districts = [k for k in keys if k.isdigit() and len(k) == 1]
    if sorted(districts) != sorted(DISTRICTS):
        sys.exit(f"!! census district codes are {sorted(districts)}, "
                 f"expected {sorted(DISTRICTS)} (2 = Keryneia, not enumerated)")
    return out


def main():
    import geopandas as gpd
    import pandas as pd

    for p in (LAU_SHP, LAU_XLSX):
        if not os.path.exists(p):
            sys.exit(f"missing {p} -- the GISCO LAU 2021 bundle is a shared asset")

    census = census_communities()
    print(f"census communities: {len(census)}")
    if len(census) != EXPECTED_COMMUNITIES:
        sys.exit(f"!! expected {EXPECTED_COMMUNITIES}")

    g = gpd.read_file(LAU_SHP, where="CNTR_CODE='CY'")
    g["unit"] = g["LAU_ID"].astype(str).str.strip()
    print(f"GISCO CY polygons: {len(g)}, crs={g.crs}")
    if len(g) != EXPECTED_GISCO:
        sys.exit(f"!! expected {EXPECTED_GISCO} GISCO polygons for CY")
    if g["unit"].duplicated().any():
        sys.exit("!! duplicate LAU codes in the GISCO layer")

    xl = pd.read_excel(LAU_XLSX, sheet_name="CY")
    xl["unit"] = xl["LAU CODE"].astype(str).str.strip()
    latin = xl.set_index("unit")["LAU NAME LATIN"].to_dict()
    pop = xl.set_index("unit")["POPULATION"].astype(str).str.strip().to_dict()

    missing = sorted(set(census) - set(g["unit"]))
    if missing:
        sys.exit(f"!! {len(missing)} census communities have no GISCO polygon: {missing[:10]}")

    # the free check: the code join and the name join must agree on all 396
    bad = [(c, n, latin.get(c)) for c, n in census.items()
           if str(latin.get(c, "")).strip() != n.strip()]
    if bad:
        print(f"!! {len(bad)} names disagree between CYSTAT and GISCO on the code join:")
        for b in bad[:15]:
            print("   ", b)
        sys.exit("!! a code/name disagreement means one of the two lists has moved")
    print(f"  all {len(census)} Latin names agree with CYSTAT's English labels")

    # what is being left out, and why each one is out
    left = g[~g["unit"].isin(census)]
    na = sum(1 for c in left["unit"] if pop.get(c) == "n.a.")
    zero = sum(1 for c in left["unit"] if pop.get(c) in ("0", "0.0"))
    print(f"  not enumerated: {len(left)} polygons -- {na} with Eurostat population `n.a.` "
          f"(under Turkish Cypriot administration), {zero} with population 0")
    if na + zero != len(left):
        sys.exit(f"!! {len(left) - na - zero} excluded polygons are neither `n.a.` nor 0, "
                 "which would mean real people are being dropped")

    keep = g[g["unit"].isin(census)].copy()
    keep["name"] = keep["unit"].map(census)
    keep["district"] = keep["unit"].str[0].map(DISTRICTS)
    if keep["district"].isna().any():
        sys.exit("!! a community code does not begin with a known district digit")

    area = float(keep["AREA_KM2"].sum())
    total = float(g["AREA_KM2"].sum())
    print(f"  drawn area {area:,.0f} km² of {total:,.0f} km² ({100 * area / total:.1f}%)")
    if not DRAWN_KM2[0] <= area <= DRAWN_KM2[1]:
        sys.exit(f"!! drawn area {area:,.0f} km² is outside {DRAWN_KM2}")
    if abs(total - ISLAND_KM2) > 50:
        sys.exit(f"!! the island sums to {total:,.0f} km², expected about {ISLAND_KM2:,.0f}")

    w, s, e, n = keep.total_bounds
    print(f"  bbox {w:.3f} {s:.3f} {e:.3f} {n:.3f}")
    if not (31.5 < w < 33.0 and 34.4 < s < 35.0 and 33.8 < e < 34.8 and 35.0 < n < 35.5):
        sys.exit("!! the drawn bbox is not the government-controlled area of Cyprus")

    by_d = keep.groupby("district").size().sort_values(ascending=False)
    print("  " + ", ".join(f"{k} {v}" for k, v in by_d.items()))

    os.makedirs(OUT_DIR, exist_ok=True)
    out = keep[["unit", "name", "district", "geometry"]].reset_index(drop=True)
    out.to_file(OUT, driver="GPKG", layer="cy_lau")
    print(f"\nwrote {OUT}  ({len(out)} units)")

    with open(LOOKUP, "w", encoding="utf-8", newline="") as fh:
        wtr = csv.writer(fh)
        wtr.writerow(["unit", "name", "district"])
        for _, r in out.iterrows():
            wtr.writerow([r["unit"], r["name"], r["district"]])
    print(f"wrote {LOOKUP}")


if __name__ == "__main__":
    main()
