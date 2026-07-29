"""prep_wup.py -- turn the UN WUP 2025 DEGURBA-Cities bulk CSV into a compact
wup2025.json in the SAME schema as stadester/ghsl.json, so build.py can graft the
modern tail from it unchanged (just repoint GHSL).

WUP 2025 is annual 1975-2050, global urban agglomerations >=50k, GHSL-lineage (same
family as the old graft) but cleaner: official, population-weighted centroids, to 50k.
We keep 1975..2025 (drop the 2026-2050 projection tail), Pop is in thousands.

Source: https://population.un.org/wup/  (CC BY 3.0 IGO)
Input csv.gz path is passed as argv[1] (default: the scratchpad download).
"""
import gzip, csv, json, os, sys
from collections import defaultdict

SRC = sys.argv[1] if len(sys.argv) > 1 else "data/stadester/wup2025_source.csv.gz"
OUT = "data/stadester/wup2025.json"
YEAR_LO, YEAR_HI = 1975, 2025

def main():
    pops = defaultdict(dict)          # city_code -> {year: pop}
    coords = {}                       # city_code -> (lat, lon) from the row nearest 2020
    coord_year = {}                   # city_code -> year of the stored coord
    meta = {}                         # city_code -> (name, iso3, capital)
    n_rows = 0
    with gzip.open(SRC, "rt", encoding="utf-8") as f:
        r = csv.DictReader(f)
        # DictReader keeps the BOM on the first key; normalize
        for row in r:
            n_rows += 1
            try:
                y = int(float(row["Year"]))
            except (ValueError, KeyError):
                continue
            if y < YEAR_LO or y > YEAR_HI:
                continue
            code = row["City_Code"]
            try:
                pop = float(row["Pop"]) * 1000.0      # WUP Pop is in thousands
                lat = float(row["PWCent_Latitude"])
                lon = float(row["PWCent_Longitude"])
            except (ValueError, KeyError):
                continue
            if pop <= 0:
                continue
            pops[code][y] = pop
            # keep the centroid from the row closest to 2020 (centroids drift a little)
            if code not in coords or abs(y - 2020) < abs(coord_year[code] - 2020):
                coords[code] = (lat, lon); coord_year[code] = y
            meta.setdefault(code, (row.get("City_Name", ""),
                                   row.get("ISO3_Code", ""), row.get("Capital", "0")))

    out = {}
    for code, series in pops.items():
        if code not in coords:
            continue
        lat, lon = coords[code]
        name, iso3, cap = meta[code]
        out[code] = {
            "coords": [lat, lon],
            "population": {str(y): round(v) for y, v in sorted(series.items())},
            "name": name, "iso3": iso3,
        }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, separators=(",", ":"))
    sz = os.path.getsize(OUT) / 1e6
    print(f"read {n_rows:,} rows -> {len(out):,} WUP cities ({YEAR_LO}..{YEAR_HI})")
    print(f"wrote {OUT}  ({sz:.1f} MB)")

    # quick peek at some megacities (match by name, informational)
    def peek(sub):
        for code, c in out.items():
            if sub.lower() in c["name"].lower():
                s = c["population"]
                ys = sorted(int(y) for y in s)
                print(f"  {c['name']:22} {c['iso3']} @({c['coords'][0]:.2f},{c['coords'][1]:.2f})  "
                      f"{ys[0]}:{s[str(ys[0])]:,}  2000:{s.get('2000','-')}  2025:{s.get('2025','-')}")
                return
        print(f"  {sub}: not found by that name")
    print("megacity spot-check:")
    for s in ["Tokyo", "Delhi", "Jakarta", "Lagos", "Osaka"]:
        peek(s)

if __name__ == "__main__":
    main()
