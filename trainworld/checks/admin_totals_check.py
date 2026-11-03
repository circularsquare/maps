"""T-2084: tier 0's admin totals (tier0.scale_to_units) and the generic reader (pipeline/admin_totals.py),
without writing anything live.

    ..\\venv\\Scripts\\python.exe checks/admin_totals_check.py      (from trainworld; about a minute)

1. A unit split by the pack's outline: half of its GHSL people inside, so the part inside takes half
   its census total (the old rule put all of it there); a unit wholly inside takes its whole total.
2. tier0.ghsl_people on the real GHSL tiles: two halves of a box hold exactly the box's people, and
   (whole, part) with the box's west half as `inside` gives the west half's people as the part.
3. The old-census mode: each unit takes GHSL's parent total times its census share of the parent.
4. A config's join on made-up files (a temp GeoPackage and CSV): a good one passes, a repeated code
   and a unit missing from its parent's sum fail; the shares mode hands the hook held totals.
5. The real configs' joins (ID, PH, KH a partial config, NZ's SA2s against the published total, IN): codes
   unique, every unit a total, parents' sums exact.
One PASS or FAIL line each.
"""
import contextlib
import io
import os
import sys
import tempfile
from pathlib import Path

TW = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TW / "pipeline"))
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import shapely  # noqa: E402
from shapely.geometry import box  # noqa: E402

fails = 0


def report(name, ok, detail=""):
    global fails
    fails += not ok
    print(f"{'PASS' if ok else 'FAIL'} {name}{': ' + detail if detail else ''}", flush=True)


def grid(w, s, e, n, k):
    """k x k points over a box, as lon and lat arrays."""
    xs, ys = np.meshgrid(np.linspace(w, e, k + 2)[1:-1], np.linspace(s, n, k + 2)[1:-1])
    return xs.ravel(), ys.ravel()


def old_rule(out, units):
    """The hook as it was before T-2084: each unit's whole total into its cells inside the pack."""
    pts = shapely.points(out["lon"], out["lat"])
    pop = out["pop"].copy()
    for _, g, t in units:
        sel = shapely.contains(g, pts)
        if pop[sel].sum() > 0:
            pop[sel] *= t / pop[sel].sum()
    return pop


def split_unit():
    import tier0
    pack = box(0.0, 0.0, 1.0, 1.0)
    a, bb = box(0.5, 0.0, 1.5, 1.0), box(0.0, 0.0, 0.5, 1.0)  # a: half inside the pack; bb wholly inside
    lon, lat = grid(0.0, 0.0, 1.0, 1.0, 40)  # the pack's cells, GHSL's people spread evenly
    out = {"lon": lon, "lat": lat, "pop": np.full(len(lon), 25.0)}  # 40,000 people: 20,000 in a's half, 20,000 in bb
    # GHSL over the whole of each unit: a holds 20,000 more outside the pack
    people = lambda geoms, inside: (np.array([40_000.0, 20_000.0]), np.array([20_000.0, 20_000.0]))  # noqa: E731
    units = [("a", a, 70_000.0), ("bb", bb, 30_000.0)]
    pts = shapely.points(lon, lat)
    old = old_rule(out, units)
    with contextlib.redirect_stdout(io.StringIO()):
        rows = tier0.scale_to_units(out, pack, units, people=people)
    in_a, in_b = shapely.contains(a, pts), shapely.contains(bb, pts)
    report("a unit cut by the outline brings only its part",
           abs(out["pop"][in_a].sum() - 35_000) < 1 and abs(out["pop"][in_b].sum() - 30_000) < 1
           and abs(old[in_a].sum() - 70_000) < 1 and rows[0]["share"] == 0.5 and not rows[0]["flag"],
           f"a (census 70,000, GHSL 40,000 over the unit, half inside): old rule {old[in_a].sum():,.0f}, now "
           f"{out['pop'][in_a].sum():,.0f}; bb wholly inside {out['pop'][in_b].sum():,.0f} of 30,000")
    out2 = {"lon": lon, "lat": lat, "pop": np.full(len(lon), 25.0)}
    with contextlib.redirect_stdout(io.StringIO()):
        r2 = tier0.scale_to_units(out2, pack, [("c", a, 100_000.0)], people=lambda g, i: (np.array([10_000.0]), np.array([5_000.0])))
    report("a unit at 10x GHSL is flagged", r2[0]["flag"], f"factor {r2[0]['census'] / r2[0]['ghsl']:.0f}x")


def real_ghsl():
    import tier0
    tile = tier0._tile_path("pop", 2020, 8, 31)  # Metro Manila's tile, cached by the Philippine run
    if tile is None or not os.path.exists(tile):
        report("ghsl_people on real tiles", False, "GHSL tile R8_C31 not cached")
        return
    w, s, e, n = 120.95, 14.50, 121.10, 14.65
    whole_box = box(w, s, e, n)
    west, east = box(w, s, (w + e) / 2, n), box((w + e) / 2, s, e, n)
    g_box = tier0.ghsl_people([whole_box])[0]
    halves = tier0.ghsl_people([west, east])
    wh, part = tier0.ghsl_people([whole_box], inside=west)
    report("ghsl_people: two halves hold the box's people, and the part inside is the half's",
           g_box > 1e5 and abs(halves.sum() - g_box) < 1e-3 * g_box and abs(part[0] - halves[0]) < 1e-3 * g_box
           and abs(wh[0] - g_box) < 1e-6 * g_box,
           f"box {g_box:,.0f}, halves {halves[0]:,.0f} + {halves[1]:,.0f}, part {part[0]:,.0f}")


def shares_mode():
    import admin_totals as A
    census = [60, 40, 50, 0]
    ghsl = [70, 80, 40, 5]
    parent = ["p1", "p1", "p2", "p3"]
    held = A.held_shares(census, ghsl, parent)
    report("old-census mode holds GHSL's parent total, census shares within it",
           np.allclose(held, [90, 60, 40, 0]), f"census 60/40 of p1 (GHSL 150) -> {held[0]:g}/{held[1]:g}; p2 {held[2]:g} of GHSL 40")


def temp_config(d, rows, totals, mode="factor"):
    import pandas as pd
    import pyogrio  # noqa: F401
    import geopandas as gpd
    gpkg, csv = os.path.join(d, f"u{len(os.listdir(d))}.gpkg"), os.path.join(d, f"t{len(os.listdir(d))}.csv")
    gpd.GeoDataFrame({"code": [r[0] for r in rows], "nm": [r[0] for r in rows]},
                     geometry=[r[1] for r in rows], crs=4326).to_file(gpkg, layer="units", driver="GPKG")
    pd.DataFrame({"code": list(totals), "pop": list(totals.values())}).to_csv(csv, index=False)
    return {"source": "test", "polygons": {"file": gpkg, "layer": "units", "code": "code", "name": "nm"},
            "totals": {"file": csv, "code": "code", "pop": "pop"}, "parent": r"^(\d\d)\d\d$", "national": "00", "mode": mode}


def join_configs():
    import admin_totals as A
    rows = [("0101", box(0, 0, 1, 1)), ("0102", box(1, 0, 2, 1)), ("0201", box(0, 1, 1, 2))]
    good = {"00": 600, "01": 300, "02": 300, "0101": 100, "0102": 200, "0201": 300}
    with tempfile.TemporaryDirectory() as d:
        try:
            A.CONFIGS["ZZ"] = temp_config(d, rows, good)
            ok, lines = A.join_check("ZZ")
            report("a good config's join passes", ok, "; ".join(lines))
            A._UNITS.pop("ZZ", None)
            A.CONFIGS["ZZ"] = temp_config(d, rows + [("0102", box(2, 0, 3, 1))], good)
            ok, lines = A.join_check("ZZ")
            report("a repeated code fails the join", not ok and any("BAD" in x and "repeated" in x for x in lines))
            A._UNITS.pop("ZZ", None)
            A.CONFIGS["ZZ"] = temp_config(d, rows, {**good, "02": 350})
            ok, lines = A.join_check("ZZ")
            report("a parent whose units do not sum to it fails the join", not ok and any(x.startswith("BAD") and "parents" in x for x in lines))
            A._UNITS.pop("ZZ", None)
            A.CONFIGS["ZZ"] = temp_config(d, rows, good, mode="shares")
            real = A.ghsl_table
            A.ghsl_table = lambda iso, epoch=2020: {"0101": 50.0, "0102": 250.0, "0201": 420.0}
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    got = {name.split()[-1]: float(t) for name, _, t in A.hook("ZZ")(box(0.5, 0.5, 1.5, 1.5))}
            finally:
                A.ghsl_table = real
            report("the shares mode hands the hook held totals",
                   got == {"0101": 100.0, "0102": 200.0, "0201": 420.0},
                   f"province 01: GHSL 300 split 1:2 as the census; 02 alone: its GHSL 420 ({got})")
        finally:
            A.CONFIGS.pop("ZZ", None)
            A._UNITS.pop("ZZ", None)


def real_joins():
    import admin_totals as A
    for iso in ("ID", "PH", "KH", "NZ", "IN"):
        try:
            ok, lines = A.join_check(iso)
        except (SystemExit, Exception) as e:  # a missing source file
            ok, lines = False, [str(e)]
        report(f"{iso} config's join", ok, "; ".join(x[5:] for x in lines))


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", message="organizePolygons")
    split_unit()
    real_ghsl()
    shares_mode()
    join_configs()
    real_joins()
    sys.exit(1 if fails else 0)
