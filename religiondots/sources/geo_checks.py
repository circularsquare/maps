"""
Shared geography checks: the traps in `playbooks/geography.md` that every country meets, as one
set of functions instead of a copy per country. Written 2026-09-14 (WORKFLOW_PLAN.md item 4).

A CHECK NEVER CHANGES OUTPUT. Each one measures, then returns, prints a warning, or raises
SystemExit with a message that names the fix. None of them edits a count, a mapping, a polygon or
a weight, so wiring one into a build moves no dot.

    check        asserts                                                     called from
    unplaced     every unit with people in the counts has a placement row    scatter.py::main
    torn         no placement polygon part spans over 180 degrees of lon     scatter.py::main
    grid_floor   a population grid has enough cells per unit to weight by    scatter.py::main
    shadowing    no module name repeats across the top level, sources/,      scatter.py::main
                 taxonomy/ and tools/, or shadows the standard library
    read_layer   a layer read returns at least one feature                   any new *_grid.py
    ratio_band   a second population source agrees with the base per unit    any new *_geo.py
    file_neighbour_outliers
                 a name join puts each unit in a parent one of its file      any name-joined *_geo.py
                 neighbours is in (the wrong same-named twin)

THE REGISTRY is `geo_checks.csv` beside this file, the `kontur_cap.csv` pattern: one row per case
a check has met on a drawn country, with the reason.

    check,cc,key,status,why,reviewed
    accepted   recorded and known to be right; one summary line per run
    warn       met on a drawn country and not resolved; a loud warning every run, never a stop

For `unplaced`, `key` is the unit code. A case no row names STOPS the scatter (unplaced, torn,
shadowing) or warns (grid_floor, which is a judgement about noise rather than a lost dot).

Usage, standalone:
    python sources/geo_checks.py shadowing     the module-name check, over the whole tree
    python sources/geo_checks.py registry      print the registry
"""
import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REGISTRY = HERE / "geo_checks.csv"
FIELDS = ["check", "cc", "key", "status", "why", "reviewed"]
STATUSES = ("accepted", "warn")
CHECKS = ("unplaced", "torn", "grid_floor", "shadowing")

# A placement polygon PART wider than this is torn across the antimeridian. No administrative
# unit and no grid cell is one ring half the globe wide; a torn one is ~358 degrees wide, because
# its vertices sit at both edges of the plane (spec §12 "The shapes of failure", item 4; Fiji's
# provinces and nine of its Kontur hexes). Parts, not features: the US Aleutians tract is a
# correct multipolygon with a piece either side of 180, and only its parts are narrow.
TORN_DEG = 180.0

# spec §8.2e: "divide the median unit area by 0.74 km². If the answer is single digits, or if a
# large share of units would come back with no hex, do not build the grid". Counted here as
# cells per unit in the layer itself, which is what the weighting actually splits over, and the
# share of units whose cells sum to no people (they fall back to equal shares, silently).
GRID_MIN_MEDIAN_CELLS = 10
GRID_MAX_ZERO_SHARE = 0.10


def read_registry(check=None, cc=None):
    """Registry rows as a list of dicts, filtered to one check and country when given."""
    if not REGISTRY.exists():
        return []
    with open(REGISTRY, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    bad = sorted({r["status"] for r in rows} - set(STATUSES))
    if bad:
        raise SystemExit(f"{REGISTRY.name}: unknown status {bad}; allowed: {', '.join(STATUSES)}")
    badc = sorted({r["check"] for r in rows} - set(CHECKS))
    if badc:
        raise SystemExit(f"{REGISTRY.name}: unknown check {badc}; allowed: {', '.join(CHECKS)}")
    return [r for r in rows
            if (check is None or r["check"] == check) and (cc is None or r["cc"] == cc)]


def _row_hint(check, cc, key):
    return f"      {check},{cc},{key},<accepted|warn>,<why>,<date and session>"


# ---------------------------------------------------------------------------------------------
# unplaced: a counted unit with no placement row draws nothing, silently.
#
# scatter.py allocates dots to every (unit, node) row, including units with no placement polygon
# (they are walked last on each node's Hilbert curve), and then skips them when placing, because
# there is nowhere to put the points. So their dots are allocated and never drawn, and the
# fraction carried along the curve lands in them instead of in a real unit. Nothing downstream
# notices: every national total still adds up. China's six dense city counties with no r6 hex
# (sources/cn_geo.md, 27,487 people) and Cyprus's Akrotiri (sources/cy_grid.py) were both found
# by reading the log line by eye.
# ---------------------------------------------------------------------------------------------

def unplaced_units(df, place):
    """Units with people (count > 0) in the counts and no row in `place`, largest first.

    A DataFrame of `unit` and `people`. Units whose counts are all zero or missing draw no dot and
    are left out, since losing them loses nothing.
    """
    import pandas as pd

    c = df[df["count"].notna() & (df["count"] > 0)]
    miss = c[~c["unit"].isin(set(place["unit"]))]
    if miss.empty:
        return pd.DataFrame({"unit": pd.Series(dtype=object), "people": pd.Series(dtype=float)})
    out = miss.groupby("unit")["count"].sum().sort_values(ascending=False, kind="mergesort")
    return out.rename("people").reset_index()


def check_unplaced(cc, missing, verbose=True):
    """Stop on a unit with people and no placement row that the registry does not name.

    `missing` is `unplaced_units(...)`. Registered units print one line (accepted) or a loud
    warning (warn); their dots are still not drawn, and the line says how many people that is.
    Returns the registered units, so the caller can account for their dots.
    """
    if missing is None or len(missing) == 0:
        return []
    rows = {r["key"]: r for r in read_registry("unplaced", cc)}
    units = missing["unit"].astype(str)
    listed = missing[units.isin(rows)]
    unlisted = missing[~units.isin(rows)]

    if verbose and len(listed):
        for status in STATUSES:
            part = listed[[rows[str(u)]["status"] == status for u in listed["unit"]]]
            if not len(part):
                continue
            tag = "!! UNRESOLVED" if status == "warn" else "registered"
            print(f"  {tag}: {len(part):,} unit(s) with no placement polygon draw nothing, "
                  f"{part['people'].sum():,.0f} people ({REGISTRY.name}, `{status}`): "
                  f"{list(part['unit'].astype(str))[:6]}")

    if len(unlisted):
        lines = [f"!! geo check STOPPED the scatter: {len(unlisted):,} unit(s) have people in the "
                 f"counts and no placement polygon, {unlisted['people'].sum():,.0f} people. "
                 "scatter.py would allocate their dots and never draw them, carrying fractions "
                 "into units that cannot place them (playbooks/geography.md, \"A unit missing "
                 "from the place layer\").",
                 "  Fix the place layer, not this check: give each unit its own polygon at census "
                 "population (sources/cy_grid.py, Akrotiri; sources/cn_geo.md), or fix the join "
                 "that lost it. If the loss is recorded and right, add a row per unit to "
                 f"sources/{REGISTRY.name}:"]
        for u, p in zip(unlisted["unit"].head(40), unlisted["people"].head(40)):
            lines.append(_row_hint("unplaced", cc, u) + f"      ({p:,.0f} people)")
        if len(unlisted) > 40:
            lines.append(f"      ... and {len(unlisted) - 40:,} more")
        raise SystemExit("\n".join(lines))
    return list(listed["unit"])


# ---------------------------------------------------------------------------------------------
# torn: a polygon torn at the antimeridian samples dots across the whole globe band.
# ---------------------------------------------------------------------------------------------

def torn_parts(place, max_deg=TORN_DEG):
    """Placement polygon parts wider than `max_deg` degrees of longitude.

    `place` must already be in EPSG:4326 (scatter.py reprojects first). A DataFrame of the row,
    its unit and the part's width, widest first; empty when nothing is torn.
    """
    import numpy as np
    import pandas as pd
    import shapely

    geoms = place.geometry.to_numpy()
    parts, idx = shapely.get_parts(geoms, return_index=True)
    b = shapely.bounds(parts)
    w = b[:, 2] - b[:, 0]
    bad = np.nonzero(w > max_deg)[0]
    out = pd.DataFrame({"row": idx[bad], "width": w[bad]})
    out["unit"] = place["unit"].to_numpy()[out["row"].to_numpy()] if len(out) else []
    return out.sort_values("width", ascending=False).reset_index(drop=True)


def check_torn(cc, torn):
    """Stop on any torn placement polygon. There is no registry exception for this one."""
    if torn is None or len(torn) == 0:
        return
    units = sorted(set(map(str, torn["unit"])))
    raise SystemExit(
        f"!! geo check STOPPED the scatter: {len(torn):,} placement polygon part(s) in {cc} span "
        f"more than {TORN_DEG:g} degrees of longitude (widest {torn['width'].max():.1f}), so they "
        "are torn across the antimeridian and dots would be sampled anywhere in the band they "
        f"cover. Units: {units[:10]}.\n"
        "  Fix the layer where it is built: shift negative longitudes +360 in EPSG:4326 before "
        "any join or centroid, and assert the span there (sources/fj_grid.py, "
        "sources/ki_geo.py; spec §12 \"The shapes of failure\", item 4). A Pacific CRS does not "
        "fix a tear, it moves it. If the widths are metres, the layer has no CRS and is not in "
        "degrees at all.")


# ---------------------------------------------------------------------------------------------
# grid_floor: a population grid no finer than the counting units is noise, not a weight.
# ---------------------------------------------------------------------------------------------

def grid_floor(place):
    """Cells per unit and zero-weight units in a grid place layer. A dict, cheap to store."""
    cells = place.groupby("unit").size()
    out = dict(units=int(len(cells)), median_cells=float(cells.median()) if len(cells) else 0.0,
               p10_cells=float(cells.quantile(0.1)) if len(cells) else 0.0,
               one_cell=int((cells == 1).sum()), zero_weight=0, zero_share=0.0)
    if "pop" in place.columns and len(cells):
        w = place.groupby("unit")["pop"].sum()
        out["zero_weight"] = int((w <= 0).sum())
        out["zero_share"] = out["zero_weight"] / len(w)
    return out


def check_grid_floor(cc, stats, verbose=True):
    """Warn when a grid place layer is at or below spec §8.2e's resolution floor. Never stops.

    Returns True when the layer passes or is registered, False when it warned.
    """
    if not stats or stats["units"] <= 1:
        return True                     # the one-unit tier: every cell is the country's
    low = stats["median_cells"] < GRID_MIN_MEDIAN_CELLS
    hollow = stats["zero_share"] > GRID_MAX_ZERO_SHARE
    if not (low or hollow):
        return True
    rows = read_registry("grid_floor", cc)
    what = (f"median {stats['median_cells']:g} cells per unit (p10 {stats['p10_cells']:g}, "
            f"{stats['one_cell']:,} of {stats['units']:,} units in one cell), "
            f"{stats['zero_weight']:,} units ({100 * stats['zero_share']:.1f}%) whose cells hold "
            "no people and fall back to equal shares")
    if rows and rows[0]["status"] == "accepted":
        if verbose:
            print(f"  grid floor: {what}; registered as measured ({REGISTRY.name})")
        return True
    if verbose:
        print(f"  !! grid floor: {what}. spec §8.2e: a grid needs well over "
              f"{GRID_MIN_MEDIAN_CELLS} cells per unit to weight anything, and below that "
              "uniform placement is the better answer. Measure it and say so in "
              f"sources/{cc}_geo.md; then add `grid_floor,{cc},*,accepted,<why>,<date>` to "
              f"sources/{REGISTRY.name}"
              + (" (registered as `warn`)" if rows else ""))
    return False


# ---------------------------------------------------------------------------------------------
# shadowing: a sources/ script named like a mapping module replaces it on import.
#
# coverage.py puts sources/ ahead of taxonomy/ on sys.path, so inside tiles.py `import pk2023`
# loaded the parser `sources/pk2023.py` instead of `taxonomy/pk2023.py`, and the build tail died
# after writing the archive (spec §12, "WHEN ONE PART OF A COUNTRY NEEDS A SECOND BOUNDARY
# SOURCE"). check_mapping.py and a bare coverage.py both passed, because they import in a
# different order, which is why this is a name check and not an import test.
# ---------------------------------------------------------------------------------------------

def module_shadows(root=ROOT):
    """{module name: [paths]} for every name that repeats across the four code folders, or that
    a standard-library module already has."""
    import sysconfig

    root = Path(root)
    seen = {}
    for folder in ("", "sources", "taxonomy", "tools"):
        d = root / folder if folder else root
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*.py")):
            if p.stem.startswith("__"):
                continue
            seen.setdefault(p.stem, []).append(f"{folder + '/' if folder else ''}{p.name}")
    out = {n: w for n, w in seen.items() if len(w) > 1}
    stdlib = Path(sysconfig.get_paths()["stdlib"])
    for n, w in seen.items():
        if (n in sys.builtin_module_names or (stdlib / f"{n}.py").exists()
                or (stdlib / n / "__init__.py").exists()):
            out.setdefault(n, list(w)).append("(standard library)")
    return out


def check_module_shadowing(root=ROOT, verbose=True):
    """Stop on a module name that another folder or the standard library already uses."""
    clashes = module_shadows(root)
    ok = {r["key"] for r in read_registry("shadowing") if r["status"] == "accepted"}
    bad = {n: w for n, w in clashes.items() if n not in ok}
    if verbose:
        for n in sorted(set(clashes) - set(bad)):
            print(f"  module name `{n}` repeats ({', '.join(clashes[n])}); registered as accepted")
    if bad:
        lines = [f"!! geo check STOPPED: {len(bad)} module name(s) are used twice, so which file "
                 "an `import` loads depends on sys.path order, and tiles.py and coverage.py do "
                 "not share one (spec §12, sources/pk2023.py shadowing taxonomy/pk2023.py):"]
        for n, w in sorted(bad.items()):
            lines.append(f"    {n}: {', '.join(w)}")
        lines.append("  Rename the newer file. A vintage source script takes an underscore "
                     "(sources/pk_2023.py beside taxonomy/pk2023.py).")
        raise SystemExit("\n".join(lines))


# ---------------------------------------------------------------------------------------------
# Helpers for the per-country builders. Not wired into scatter.py: they guard inputs that only a
# builder reads. Import with `sys.path.insert(0, HERE)` from a sources/ script.
# ---------------------------------------------------------------------------------------------

def read_layer(path, what=None, **kwargs):
    """`geopandas.read_file` that stops on zero features.

    A Kontur extract can be a valid file with nothing in it (the `BQ` extract; spec §12 "A KONTUR
    EXTRACT CAN BE A VALID FILE WITH NOTHING IN IT"), and every step after an empty read passes.
    """
    import geopandas as gpd

    g = gpd.read_file(path, **kwargs)
    if len(g) == 0:
        raise SystemExit(f"{what or Path(str(path)).name}: read returned ZERO features "
                         f"({path}); a valid file with nothing in it, so re-fetch it or use the "
                         "per-country extract")
    return g


def ratio_band(base, other, lo, hi, what="unit", min_base=0):
    """Assert `other / base` per unit sits in [lo, hi]. Both are {unit: number} or Series.

    For a second population source read beside the base, such as COD-PS beside a census: a
    national match hides provincial error (Dominican COD-PS is 0.56% off nationally, -23.7% to
    +10.3% by province; spec §12 "A COD-PS PROJECTION THAT AGREES NATIONALLY CAN BE WILDLY WRONG
    PER UNIT"). Units in only one of the two, and units whose base is at or under `min_base`,
    are reported and not banded. Returns a DataFrame of every unit's ratio; raises SystemExit
    listing the units outside the band.
    """
    import pandas as pd

    b = pd.Series(base, dtype=float)
    o = pd.Series(other, dtype=float)
    one_side = sorted(set(b.index) ^ set(o.index), key=str)
    if one_side:
        print(f"  ratio band: {len(one_side)} {what}(s) in only one source: {one_side[:8]}")
    common = b.index.intersection(o.index)
    t = pd.DataFrame({"base": b[common], "other": o[common]})
    t = t[t["base"] > min_base]
    t["ratio"] = t["other"] / t["base"]
    out = t[(t["ratio"] < lo) | (t["ratio"] > hi)].sort_values("ratio")
    if len(out):
        body = "\n".join(f"    {u}: base {r.base:,.0f}, other {r.other:,.0f}, ratio {r.ratio:.3f}"
                         for u, r in out.head(20).iterrows())
        raise SystemExit(f"{len(out)} {what}(s) outside the ratio band [{lo}, {hi}]:\n{body}")
    return t


def file_neighbour_outliers(parents):
    """Positions whose parent is in neither file neighbour's parent: the wrong-twin test.

    `parents` is the parent key of each matched unit, in the order the SOURCE lists its units
    (a census table in code order, say), with None where a row matched nothing. Where both lists
    run in code order, a same-named unit matched to the wrong twin lands in a parent that
    neither neighbour is in (spec §12 "Joining on NAMES, where there is no code"; §14.19;
    tools/check_cn_prefecture.py is this test for China). Returns a list of
    (position, parent, previous parent, next parent). The first and last rows compare with one
    neighbour only, and a real administrative move also shows up here, so the caller keeps its
    own list of explained positions.
    """
    out = []
    keyed = [i for i, p in enumerate(parents) if p is not None]
    for k, i in enumerate(keyed):
        prev = parents[keyed[k - 1]] if k > 0 else None
        nxt = parents[keyed[k + 1]] if k + 1 < len(keyed) else None
        near = {p for p in (prev, nxt) if p is not None}
        if near and parents[i] not in near:
            out.append((i, parents[i], prev, nxt))
    return out


def main(argv):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    what = argv[1] if len(argv) > 1 else ""
    if what == "shadowing":
        check_module_shadowing()
        print("no module name is used twice")
    elif what == "registry":
        for r in read_registry():
            print(",".join(r[f] for f in FIELDS))
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
