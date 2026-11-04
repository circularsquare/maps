"""T-2213: the land parts of the packs named in notes/T-2207.md 3 against its table: the islands no
road reaches are parts of their own, the places joined by road are in the pack's first part.

    python checks/t2213_islands.py [--px]     (--px: from parts_run.py's pixels in data/work/parts/px,
                                               before they are written; else from the packs' `part`)

One line per place, PASS or FAIL, then a PASS or FAIL line. Writes the islands found (people 300 and
over off each pack's first part) to ../artifacts/trainworld/t2213/islands.json.
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "pipeline"))
import routes  # noqa: E402

# (pack, name, lon, lat, island): an island has no road to the pack's first part
PLACES = [
    ("nyc", "Manhattan", -73.985, 40.755, False), ("nyc", "Staten Island", -74.15, 40.58, False),
    ("nyc", "Roosevelt Island", -73.951, 40.761, False), ("nyc", "Long Island (Hempstead)", -73.62, 40.71, False),
    ("nyc", "Rikers Island (T-2257: bridge access=no, bus=yes)", -73.887, 40.792, False),
    ("nyc", "Shelter Island", -72.34, 41.069, True), ("nyc", "Fishers Island", -72.004, 41.267, True),
    ("nyc", "Liberty Island (T-2421: an islet holding cells)", -74.0453, 40.6880, True),
    ("ven", "San Servolo (T-2421)", 12.3555, 45.4205, True), ("sto", "Fjaderholmarna (T-2421)", 18.1766, 59.3287, True),
    ("ams", "Forteiland, IJmuiden (T-2421)", 4.5769, 52.4656, True), ("ams", "Amsterdam", 4.90, 52.37, False),
    ("hon", "Hong Kong Island", 114.17, 22.28, False), ("hon", "Lantau (Tung Chung)", 113.94, 22.29, False),
    ("hon", "Ma Wan", 114.059, 22.351, False), ("hon", "Cheung Chau", 114.029, 22.207, True),
    ("hon", "Lamma", 114.11, 22.217, True), ("hon", "Peng Chau", 114.039, 22.285, True),
    ("tyo", "Haneda", 139.781, 35.549, False), ("tyo", "Odaiba", 139.776, 35.627, False),
    ("sea", "Bainbridge", -122.52, 47.63, False), ("sea", "Mercer Island", -122.225, 47.57, False),
    ("sea", "Vashon", -122.465, 47.423, True), ("sea", "Anderson Island", -122.695, 47.16, True),
    ("ath", "Salamis", 23.501, 37.953, True), ("ath", "Aegina", 23.463, 37.747, True),
    ("ath", "Spetses", 23.155, 37.264, True), ("ath", "Poros", 23.46, 37.506, True),
    ("ath", "Hydra", 23.466, 37.345, True), ("ath", "Angistri", 23.354, 37.707, True),
    ("ath", "Piraeus", 23.645, 37.94, False),
    ("nig", "Sado (Ryotsu)", 138.362, 38.004, True), ("nig", "Niigata", 139.04, 37.91, False),
    ("kgw", "Shodoshima", 134.244, 34.488, True), ("kgw", "Naoshima", 133.982, 34.46, True),
    ("kgw", "Teshima", 134.074, 34.487, True), ("kgw", "Takamatsu", 134.05, 34.34, False),
    ("hij", "Osakikamijima", 132.896, 34.245, True), ("hij", "Miyajima", 132.323, 34.298, True),
    ("hij", "Ninoshima", 132.437, 34.311, True), ("hij", "Hiroshima", 132.46, 34.39, False),
    ("nap", "Ischia", 13.912, 40.732, True), ("nap", "Capri", 14.23, 40.552, True),
    ("nap", "Procida", 14.02, 40.76, True), ("nap", "Naples", 14.25, 40.85, False),
    ("ven", "Lido", 12.361, 45.401, True), ("ven", "Murano", 12.353, 45.457, True),
    ("ven", "Pellestrina", 12.307, 45.29, True), ("ven", "Sant'Erasmo", 12.407, 45.457, True),
    ("ven", "Venice centre", 12.335, 45.437, False), ("ven", "Mestre", 12.24, 45.49, False),
    ("cop", "Bornholm (Ronne)", 14.71, 55.10, True), ("cop", "Copenhagen", 12.57, 55.68, False),
    ("cop", "Amager", 12.60, 55.64, False),
    ("van", "Bowen Island", -123.357, 49.377, True), ("van", "Vancouver", -123.12, 49.28, False),
    ("sto", "Stockholm", 18.07, 59.33, False), ("sto", "Lidingo", 18.15, 59.36, False),
    ("syd", "Sydney", 151.21, -33.87, False),
]


def pack_cells_parts(cid, use_px):
    """(header, res-9 arrays, part per res-9 cell, parts block)."""
    h, a = routes.res9_pack(cid)
    if use_px:
        import parts_run
        exs = parts_run.plan([cid])[cid]
        bar, origin = routes.water50(h, a)
        opened = np.unique(np.concatenate([np.load(parts_run.px_path(cid, e)) for e in exs]))
        lab, edge, info = routes.land_labels(bar, opened, (a["x_m"].astype(np.float64), a["y_m"].astype(np.float64), origin))
        part, block, _ = routes.land_parts(h, a, lab, edge, origin, info, exs)
        return h, a, part, block
    return h, a, a["part"], h["parts"]


def main(args):
    use_px = "--px" in args
    out, fails = {}, 0
    for cid in dict.fromkeys(p[0] for p in PLACES):
        h, a, part, block = pack_cells_parts(cid, use_px)
        lon0, lat0 = h["origin"]["lon"], h["origin"]["lat"]
        x, y = a["x_m"].astype(np.float64), a["y_m"].astype(np.float64)
        isl = [{"part": i, "people": block["people"][i], "jobs": block["jobs"][i], "open": block["open"][i],
                "lon": block["lon"][i], "lat": block["lat"][i]} for i in range(1, block["parts"]) if block["people"][i] >= 300]
        out[cid] = {"parts": block["parts"], "people_off_first": round(sum(block["people"][1:])), "islands": isl}
        print(f"== {cid}: {block['parts']} parts, {sum(block['people'][1:]):,.0f} people off the first part; "
              f"{len(isl)} with 300+ people: " + ", ".join(f"{p['people']:,.0f} at {p['lon']:.3f},{p['lat']:.3f}" for p in isl[:14]))
        for pk, name, lon, lat, island in PLACES:
            if pk != cid:
                continue
            k = np.cos(np.radians(lat0))
            px, py = routes.R_EARTH * np.radians(lon - lon0) * k, routes.R_EARTH * np.radians(lat - lat0)
            c = int(np.argmin(np.hypot(x - px, y - py)))
            p = int(part[c])
            ok = (p != 0) if island else (p == 0)
            fails += not ok
            what = f"part {p} ({block['people'][p]:,.0f} people, {'open' if block['open'][p] else 'interior'})"
            print(f"  {'PASS' if ok else 'FAIL'} {name}: {what}, want {'an island' if island else 'the first part'} "
                  f"(nearest cell {np.hypot(x[c] - px, y[c] - py):.0f} m)")
    os.makedirs(os.path.join(ROOT, "..", "artifacts", "trainworld", "t2213"), exist_ok=True)
    with open(os.path.join(ROOT, "..", "artifacts", "trainworld", "t2213", "islands.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(f"{'FAIL' if fails else 'PASS'} t2213 islands: {len(PLACES) - fails} of {len(PLACES)} places")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main(sys.argv[1:])
