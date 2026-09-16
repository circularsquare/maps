"""
Kontur caps density at 46,200 people/km2, and a block of hexes at that cap is sometimes a city
core and sometimes a false concentration. This finds the blocks, stops on any nobody has looked
at, and lowers the ones judged false. spec §12, "KONTUR'S DENSITY CAP", 2026-09-14.

THE PROBLEM. Kontur's population grid is modelled, and its pipeline limits every hex to 46,200
people/km2 (the scan measured every top hex in 24 countries at 46,199-46,200). Where the model's
input put far too many people in one place, the output is a flat top at the cap with a ramp
around it. In Tashkent that block sits 15.6 km south of the centre and holds 58% of the city's
placement weight while Kontur has 2,432/km2 at the city's own centre, so most of the city's dots
were drawn in a patch of southern suburbs. Luxor's block holds 2,185,844 people, 1.5x the whole
governorate. Real cores hit the same cap (Dhaka, Cairo, Karachi, Hong Kong, Seoul), so the cap
alone cannot tell the two apart, and nothing in the grid can.

SO THE DECISION IS WRITTEN DOWN PER BLOCK, in `kontur_cap.csv` beside this file: one row per
block, a point inside it, and a status.

    real         a genuine dense core. Drawn as Kontur has it; one summary line.
    capped       a false concentration. Every hex in the block is lowered to the median density
                 of the populated hexes within 3 km around it (below).
    isolated     a false concentration with no ring to set a ceiling: no populated hex within
                 3 km lies outside every dense block (Sudan's Red Sea hills, 2026-09-15). Every
                 hex in the block is lowered to the median density of the populated hexes of
                 its own unit that are in no dense block, which spreads the block's excess over
                 the rest of the unit in proportion to Kontur. Refused, and the scatter stops,
                 on a block that does have such a ring: that block is `capped`.
    unreviewed   seen, written down, not yet judged. Drawn as Kontur has it, with a loud warning
                 on every scatter. It WARNS rather than stops because the stop exists to force
                 someone to look at a block, and an unreviewed row is proof someone did; stopping
                 on it again would halt every rebuild of bd, et, ng and np on blocks nobody is
                 assigned to review. Promote it to real or capped when somebody does.

A block that reaches the cap and matches NO row stops the scatter, and prints the row to add.

WHAT A BLOCK IS. The contiguous run of hexes at 15,000/km2 or more around the capped ones, with
contiguity by centroid distance (1.3x the layer's median hex spacing). The ramp is part of the
artefact: Tashkent has 11 hexes at the cap and 54 in the block, and capping only the 11 would
leave most of the false weight in place.

THE FIX IS A CAP, NOT A MOVE. Lowering the block and letting the unit's other hexes absorb its
share spreads the unit's dots over wherever Kontur puts the rest of the unit's people. The
alternative, pouring the block's excess into the hexes around it, keeps the false people a
kilometre from where Kontur put them, which in Tashkent is still the southern suburbs. The
ceiling is the ring's MEDIAN because the ring around a false block includes its own ramp.

NO COUNT CAN MOVE. The allocation of dots to (unit, node) is computed in scatter.py before and
without the weights; this changes `pop` only, only downward, and asserts that no unit's weight
falls to zero, so no unit silently drops to equal shares. scatter.py asserts that every
(unit, node) places exactly the dots it was allocated.

It runs only on Kontur place layers (`*_hexes.gpkg`, `*_grid_<n>m.gpkg`, `*_grid_<n>km.gpkg`),
and skips a layer whose density goes above the cap, because that layer was rescaled or comes
from another grid (cn, kr, bg) and a hex at 46,200 there means nothing.

Usage, standalone (scatter.py calls `apply`):
    python kontur_cap.py              check every Kontur layer on disk against the registry
    python kontur_cap.py uz eg        the same for named countries, listing every block
"""

import csv
import glob
import os
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
REGISTRY = HERE / "kontur_cap.csv"

CAP = 46_200.0
AT_CAP = 0.995 * CAP        # a hex at or above this is AT the cap
OVER_CAP = 1.005 * CAP      # a layer with any hex above this is not raw Kontur
HI = 15_000.0               # per km2: block membership
RING_KM = 3.0
MATCH_KM = 1.0
STATUSES = ("real", "capped", "isolated", "unreviewed")
FIELDS = ["cc", "lon", "lat", "status", "place", "unit", "unit_share", "why", "reviewed"]

_LAYER = re.compile(r"(_hexes|_grid_\d+k?m)\.gpkg$")


def is_kontur_layer(src):
    return bool(_LAYER.search(Path(str(src)).name))


def read_registry(cc=None):
    if not REGISTRY.exists():
        return []
    with open(REGISTRY, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    bad = sorted({r["status"] for r in rows} - set(STATUSES))
    if bad:
        raise SystemExit(f"{REGISTRY.name}: unknown status {bad}; allowed: {', '.join(STATUSES)}")
    return [r for r in rows if cc is None or r["cc"] == cc]


def _metric(lon, lat):
    lon, lat = np.asarray(lon, dtype=float), np.asarray(lat, dtype=float)
    return np.column_stack([lon * 111320.0 * np.cos(np.radians(lat)), lat * 110574.0])


def find_blocks(place):
    """Density per hex and the dense blocks. `place` must be in EPSG:4326 with a `pop` column."""
    import shapely
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    pop = place["pop"].to_numpy(dtype=float)
    area = place.geometry.to_crs(6933).area.to_numpy() / 1e6
    dens = np.divide(pop, area, out=np.zeros_like(pop), where=area > 0)
    c = shapely.centroid(place.geometry.values)
    lon, lat = shapely.get_x(c), shapely.get_y(c)
    xy = _metric(lon, lat)
    tree = cKDTree(xy)
    spacing = float(np.median(tree.query(xy, k=2)[0][:, 1])) if len(xy) > 1 else 0.0

    hi = np.nonzero(dens >= HI)[0]
    groups = []
    if len(hi):
        pr = cKDTree(xy[hi]).query_pairs(1.3 * spacing, output_type="ndarray")
        m = coo_matrix((np.ones(len(pr), dtype=np.int8), (pr[:, 0], pr[:, 1])),
                       shape=(len(hi), len(hi)))
        _, lab = connected_components(m, directed=False)
        order = np.argsort(lab, kind="stable")
        _, starts = np.unique(lab[order], return_index=True)
        groups = np.split(hi[order], starts[1:])
    label = np.full(len(pop), -1)
    for k, idx in enumerate(groups):
        label[idx] = k
    return dict(pop=pop, area=area, dens=dens, lon=lon, lat=lat, xy=xy, tree=tree,
                groups=groups, label=label)


def _describe(b, idx, unit, utot):
    pk = idx[np.argmax(b["dens"][idx])]
    us, uc = np.unique(unit[idx], return_counts=True)
    u = us[np.argmax(uc)]
    share = float(b["pop"][idx][unit[idx] == u].sum()) / max(utot.get(u, 0.0), 1.0)
    return dict(lon=float(b["lon"][pk]), lat=float(b["lat"][pk]), unit=u, share=share,
                n=len(idx), n_cap=int((b["dens"][idx] >= AT_CAP).sum()),
                people=float(b["pop"][idx].sum()))


def apply(place, cc, src, verbose=True):
    """`place` with every `capped` block lowered. Same rows and order; only `pop` changes.

    Raises SystemExit on a block at the cap that no registry row names, on a `capped` row that
    no longer lands on a block, and on a block two rows disagree about.
    """
    if "pop" not in place.columns or not is_kontur_layer(src) or len(place) == 0:
        return place
    if place.crs is not None and place.crs.to_epsg() != 4326:
        raise SystemExit("kontur_cap.apply expects EPSG:4326")

    b = find_blocks(place)
    dens, pop, area = b["dens"], b["pop"], b["area"]
    if dens.max() > OVER_CAP:
        if verbose:
            print(f"  kontur cap: densities reach {dens.max():,.0f}/km2, above Kontur's "
                  f"{CAP:,.0f}, so this layer is not raw Kontur and is not checked")
        return place

    unit = place["unit"].astype(str).to_numpy()
    uniq, inv = np.unique(unit, return_inverse=True)
    sums = np.bincount(inv, weights=pop, minlength=len(uniq))
    utot = dict(zip(uniq, sums))

    rows_for = {}
    stale = []
    for r in read_registry(cc):
        p = _metric([float(r["lon"])], [float(r["lat"])])[0]
        d, j = b["tree"].query(p)
        k = int(b["label"][j]) if d <= MATCH_KM * 1000.0 else -1
        if k < 0:
            stale.append(r)
        else:
            rows_for.setdefault(k, []).append(r)

    problems = []
    for r in stale:
        if r["status"] == "capped":
            problems.append(f"registry row ({r['lon']}, {r['lat']}) '{r['place']}' is `capped` "
                            "but no longer lands on a dense block, so the fix would silently "
                            "not apply")
        elif verbose:
            print(f"  kontur cap: registry row '{r['place']}' ({r['status']}) matches no dense "
                  "block in this layer")

    unknown = []
    for k, idx in enumerate(b["groups"]):
        statuses = {r["status"] for r in rows_for.get(k, [])}
        if len(statuses) > 1:
            d = _describe(b, idx, unit, utot)
            problems.append(f"block at ({d['lon']:.5f}, {d['lat']:.5f}) has rows that disagree: "
                            f"{sorted(statuses)}")
        if not statuses and (dens[idx] >= AT_CAP).any():
            unknown.append(_describe(b, idx, unit, utot))

    if unknown:
        lines = [f"  {cc}: {len(unknown)} block(s) reach Kontur's cap of {CAP:,.0f}/km2 and "
                 f"are not in {REGISTRY.name}. A block at the cap is either a real dense core "
                 "or a false concentration (spec §12, KONTUR'S DENSITY CAP); decide which, "
                 "then add its row:"]
        for d in sorted(unknown, key=lambda d: -d["share"]):
            lines.append(f"    {d['n']} hexes, {d['n_cap']} at the cap, {d['people']:,.0f} people, "
                         f"{100 * d['share']:.1f}% of unit {d['unit']}")
            lines.append(f"      {cc},{d['lon']:.5f},{d['lat']:.5f},<real|capped|unreviewed>,"
                         f"<place>,{d['unit']},{d['share']:.3f},<why>,<date and session>")
        problems.append("\n".join(lines))

    if problems:
        raise SystemExit("!! kontur cap STOPPED the scatter:\n" + "\n".join(problems))

    new = pop.copy()
    n_real = 0
    base_ceiling = {}           # unit -> median density outside the blocks, for `isolated`
    for k, idx in enumerate(b["groups"]):
        rows = rows_for.get(k, [])
        if not rows:
            continue
        status, name = rows[0]["status"], rows[0]["place"]
        d = _describe(b, idx, unit, utot)
        if status == "real":
            n_real += 1
        elif status == "unreviewed":
            if verbose:
                print(f"  !! kontur cap: UNREVIEWED block '{name}', {d['n']} hexes, "
                      f"{d['n_cap']} at the cap, {100 * d['share']:.1f}% of unit {d['unit']}; "
                      f"drawn as Kontur has it ({REGISTRY.name})")
        elif status == "capped":
            inblock = np.zeros(len(pop), dtype=bool)
            inblock[idx] = True
            near = b["tree"].query_ball_point(b["xy"][idx], RING_KM * 1000.0)
            ring = np.unique(np.concatenate([np.asarray(n, dtype=np.int64) for n in near]))
            ring = ring[~inblock[ring] & (pop[ring] > 0)]
            if len(ring) == 0:
                raise SystemExit(f"!! kontur cap STOPPED the scatter: capped block '{name}' "
                                 f"has no populated hex within {RING_KM:g} km to set its ceiling")
            ceiling = float(np.median(dens[ring]))
            new[idx] = np.minimum(pop[idx], ceiling * area[idx])
            if verbose:
                u = d["unit"]
                after = (utot[u] - pop[idx][unit[idx] == u].sum()
                         + new[idx][unit[idx] == u].sum())
                share_after = float(new[idx][unit[idx] == u].sum()) / max(after, 1.0)
                print(f"  kontur cap: CAPPED '{name}', {d['n']} hexes holding "
                      f"{d['people']:,.0f} ({100 * d['share']:.1f}% of unit {u}) lowered to the "
                      f"{RING_KM:g} km ring's median of {ceiling:,.0f}/km2: now "
                      f"{new[idx].sum():,.0f} ({100 * share_after:.1f}%)")
        elif status == "isolated":
            near = b["tree"].query_ball_point(b["xy"][idx], RING_KM * 1000.0)
            ring = np.unique(np.concatenate([np.asarray(n, dtype=np.int64) for n in near]))
            ring = ring[(b["label"][ring] < 0) & (pop[ring] > 0)]
            if len(ring):
                raise SystemExit(f"!! kontur cap STOPPED the scatter: block '{name}' is `isolated` "
                                 f"but {len(ring)} populated hex(es) outside every dense block lie "
                                 f"within {RING_KM:g} km; it has a ring, so it is `capped`")
            for u in np.unique(unit[idx]):
                if u not in base_ceiling:
                    base = (unit == u) & (b["label"] < 0) & (pop > 0)
                    if not base.any():
                        raise SystemExit(f"!! kontur cap STOPPED the scatter: unit {u} has no "
                                         f"populated hex outside a dense block to set "
                                         f"'{name}''s ceiling")
                    base_ceiling[u] = float(np.median(dens[base]))
                h = idx[unit[idx] == u]
                new[h] = np.minimum(pop[h], base_ceiling[u] * area[h])
            if verbose:
                u = d["unit"]
                print(f"  kontur cap: ISOLATED '{name}', {d['n']} hexes holding "
                      f"{d['people']:,.0f} ({100 * d['share']:.1f}% of unit {u}), no populated "
                      f"ring outside the blocks; lowered to unit {u}'s median of "
                      f"{base_ceiling[u]:,.1f}/km2 outside the blocks: now {new[idx].sum():,.0f}")
    if verbose and n_real:
        print(f"  kontur cap: {n_real} block(s) at the cap registered as real cores, drawn as is")

    # The invariants that make this a placement change and not a count change.
    assert len(new) == len(pop)
    assert np.all(new <= pop + 1e-9), "kontur cap raised a hex"
    after = dict(zip(uniq, np.bincount(inv, weights=new, minlength=len(uniq))))
    lost = [u for u in uniq if utot[u] > 0 and after[u] <= 0]
    assert not lost, f"kontur cap left units with zero weight: {lost[:5]}"
    if np.array_equal(new, pop):
        return place
    out = place.copy()
    out["pop"] = new
    return out


def _layers():
    root = HERE / "data" / "geo"
    out = {}
    for p in sorted(glob.glob(str(root / "[a-z][a-z]" / "*_hexes.gpkg"))
                    + glob.glob(str(root / "[a-z][a-z]" / "*_grid_*.gpkg"))):
        if is_kontur_layer(p):
            out.setdefault(os.path.basename(os.path.dirname(p)), p)
    return out


def main():
    import geopandas as gpd

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    lay = _layers()
    ccs = sys.argv[1:] or sorted(lay)
    failed = []
    for cc in ccs:
        if cc not in lay:
            print(f"{cc}: no Kontur layer on disk")
            continue
        g = gpd.read_file(lay[cc], engine="pyogrio")
        if "pop" not in g.columns or "unit" not in g.columns:
            print(f"{cc}: {os.path.basename(lay[cc])} has no pop/unit column, skipped")
            continue
        if g.crs is not None and g.crs.to_epsg() != 4326:
            g = g.to_crs(4326)
        print(f"{cc}: {os.path.basename(lay[cc])}, {len(g):,} cells")
        try:
            apply(g, cc, lay[cc], verbose=bool(sys.argv[1:]))
        except SystemExit as e:
            print(e)
            failed.append(cc)
        sys.stdout.flush()
    print(f"\n{len(ccs)} layer(s) checked; {'STOPS in: ' + ', '.join(failed) if failed else 'no stops'}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
