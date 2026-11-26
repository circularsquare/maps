"""T-4364 dry runs and before/after sheets for plan_edit specs (read-only: never applies a spec).

    venv\\Scripts\\python.exe -I trainworld\\notes\\T-4364\\kit\\dry.py <spec.json> --out <folder> [--png <sheet.png>]

Uses the T-3688 kit's plan_edit (a copy beside this file, --go refused here) for the change itself, then adds, per
row the spec touches:
- people: the plan's (unit sums) and the game's estimate (each unit's plan people x its old row's pack/plan ratio,
  so Israel's census and line cut show as the game has them);
- floor: the row's country floor (ZONE_RULES, else 0.5M; 100,000 over 100,000 km2), zones only;
- pop_vs_fua (metro rows with source.fua): estimated pack people over the row's FUA people (pack_gate.fua_people),
  pass 0.7-1.5, warn outside 0.8-1.3;
- hull: area over convex hull of the row's units (the merge rule's 75%);
- ring: any row or registered pack B with half its area in the row's hull and half of its land border on the row
  (world_plan's wrap test, WRAP_HULL and WRAP_BORDER); "ring none" when nothing is wrapped.
Writes <out>/<spec name>.dry.txt and <spec name>.waves.json; with --png, the sheet (before | after).
"""
import collections
import json
import math
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import plan_edit as pe  # noqa: E402
import shared_lock  # noqa: E402
import shapely  # noqa: E402
import shapely.ops  # noqa: E402
import world_plan as wp  # noqa: E402

TW = pe.TW
sys.path.insert(0, os.path.join(TW, "checks"))
PACKS = pe.PACKS
_units = {}


def units(iso):
    if iso not in _units:
        _units[iso] = pe.load_units(iso)
    return _units[iso]


def pack_pop(cid):
    try:
        return (json.load(open(os.path.join(PACKS, f"{cid}.json"), encoding="utf-8")).get("totals") or {}).get("pop")
    except OSError:
        return None


def floor_of(iso, km2):
    r = wp.ZONE_RULES.get(iso, {})
    f = r.get("floor", wp.ZONE_FLOOR)
    return wp.ZONE_MIN_POP if km2 >= wp.BIG_ZONE_KM2 else f


def geom(ids, by):
    return shapely.union_all([by[u].geom for u in ids if u in by])


def checks(spec, doc):
    iso = spec["iso"]
    by = units(iso)
    before = {r["id"]: r for r in doc["metros"] if r.get("units", {}).get(iso)}
    new, gone, waves, others, lines = pe.plan_change(spec, json.loads(json.dumps(doc)), by)
    after = {k: v for k, v in before.items() if k not in gone}
    after.update({k: v for k, v in new.items() if k not in gone})
    touched = sorted(k for k in new if k not in gone)
    # the game's people per unit: plan people x its old row's pack/plan ratio
    ratio = {}
    for k, r in before.items():
        us = r["units"][iso]
        plan = sum(by[u].pop for u in us if u in by)
        pp = pack_pop(k)
        ratio[k] = (pp / plan) if pp and plan else 1.0
    owner = {u: k for k, r in before.items() for u in r["units"][iso]}
    game = {u: by[u].pop * ratio.get(owner.get(u), 1.0) for u in by}

    def people(r):
        us = [u for u in r["units"][iso] if u in by]
        return sum(by[u].pop for u in us), sum(game[u] for u in us)
    out = list(lines) + ["", "checks (T-4364 dry.py):"]
    # every other area a touched row could wrap: this country's rows after, other countries' rows near, hand packs
    shapes = {k: geom(r["units"][iso], by) for k, r in after.items()}
    ctry = {k: iso for k in shapes}
    reg = json.load(open(os.path.join(TW, "app", "src", "game", "registry.json"), encoding="utf-8"))["cities"]
    retired = wp.retired_ids() if spec.get("new") else set()
    box = shapely.union_all([shapes[k] for k in touched]).buffer(50_000)
    bll = wp.to_ll(box).bounds
    for c in reg:
        if c["id"] in shapes or not c.get("box"):
            continue
        w, s, e, n = c["box"]
        if e < bll[0] or w > bll[2] or n < bll[1] or s > bll[3]:
            continue
        r = next((r for r in doc["metros"] if r["id"] == c["id"] and r.get("units")), None)
        if r:
            iso2 = next(iter(r["units"]))
            if iso2 == iso:
                continue
            g = geom(r["units"][iso2], units(iso2))
            ctry[c["id"]] = iso2
        else:
            try:
                gj = json.load(open(os.path.join(PACKS, f"{c['id']}.boundary.geojson"), encoding="utf-8"))
                g = wp.to_moll(wp.polygonal(shapely.union_all([shapely.make_valid(shapely.geometry.shape(f["geometry"])) for f in gj["features"]])))
            except OSError:
                continue
            ctry[c["id"]] = c.get("country")
        if not g.is_empty and g.intersects(box):
            shapes[c["id"]] = g
    near_all = {k: g.buffer(1000) for k, g in shapes.items()}
    land = {}
    for q, g in shapes.items():
        L = 0.0
        for o, nb in near_all.items():
            if o != q and ctry.get(o) == ctry.get(q) and g.intersects(nb):  # other countries do not count
                L += g.boundary.intersection(nb).length
        land[q] = L
    rows_out = {}
    for k in touched:
        r = after[k]
        g = shapes[k]
        km2 = g.area / 1e6
        plan_p, game_p = people(r)
        h = g.area / g.convex_hull.area
        g0 = geom(before[k]["units"][iso], by) if k in before else None
        h0 = g0.area / g0.convex_hull.area if g0 is not None and not g0.is_empty else None
        main = max(getattr(g, "geoms", [g]), key=lambda q: q.area)
        hm = main.area / main.convex_hull.area
        verdict = ("pass" if h >= wp.COMPACT_HULL else
                   "pass (no worse than before less 0.05)" if h0 is not None and h >= h0 - wp.COMPACT_SLACK else "BELOW 0.75")
        res = [f"{k} {r['name']}: plan people {plan_p:,.0f}, game about {game_p:,.0f}, {km2:,.0f} km2, "
               f"hull {h:.2f} (before {h0:.2f}, main piece {hm:.2f}) {verdict}" if h0 is not None else
               f"{k} {r['name']}: plan people {plan_p:,.0f}, game about {game_p:,.0f}, {km2:,.0f} km2, hull {h:.2f} "
               f"(new row, main piece {hm:.2f}) {verdict}"]
        if r.get("kind") == "zone":
            fl = floor_of(iso, km2)
            cap = wp.ZONE_RULES.get(iso, {}).get("cap", wp.ZONE_CAP)
            res.append(f"floor {fl:,} {'pass' if plan_p >= fl else 'UNDER'}; cap {cap:,} "
                       + ("within" if plan_p <= cap else f"OVER by {plan_p - cap:,.0f} (a hand edit may; a zones rerun would not make it)"))
        if k in spec.get("new", {}):
            taken = retired | set(os.path.basename(p).split(".")[0] for p in os.listdir(PACKS)) | {c["id"] for c in reg} \
                | {x["id"] for x in doc["metros"]}
            res.append(f"new id {k} {'TAKEN or retired' if k in taken else 'free'}")
        fua = (r.get("source") or {}).get("fua")
        if fua and r.get("kind") != "zone":
            import pack_gate
            fp, names = pack_gate.fua_people(g, r)
            if fp:
                v = game_p / fp
                st = "FAIL" if not 0.7 <= v <= 1.5 else ("warn" if not 0.8 <= v <= 1.3 else "pass")
                res.append(f"pop_vs_fua about {v:.2f} ({game_p:,.0f} over {fp:,.0f} in {', '.join(names[:3])}) {st}")
        wr = []
        for q, gq in shapes.items():
            # the other way: q (untouched or touched) wrapping this row as it is after
            if q != k and ctry.get(q) == iso and land.get(k) and g.intersects(near_all[q]):
                a = g.intersection(gq.convex_hull).area / g.area
                b = g.boundary.intersection(near_all[q]).length / land[k]
                if a >= wp.WRAP_HULL and b >= wp.WRAP_BORDER:
                    wr.append(f"wrapped by {q} (hull {a:.2f}, border {b:.2f})")
            if q == k or ctry.get(q) != iso or not land.get(q):
                continue
            def wraps(gg):
                if gg is None or gg.is_empty:
                    return None
                nb = gg.buffer(1000)
                if not gq.intersects(nb):
                    return None
                a = gq.intersection(gg.convex_hull).area / gq.area
                b = gq.boundary.intersection(nb).length / land[q]
                return (a, b) if a >= wp.WRAP_HULL and b >= wp.WRAP_BORDER else None
            w1 = wraps(g)
            if w1:
                wr.append(f"{q} (hull {w1[0]:.2f}, border {w1[1]:.2f})" + (" (already before)" if wraps(g0) else ""))
        res.append("ring: " + ", ".join(x if x.startswith("wrapped") else "wraps " + x for x in wr) if wr else "ring none")
        rows_out[k] = {"plan": plan_p, "game": game_p, "hull": h, "km2": km2}
        out.append("; ".join(res))
    for k in gone:
        out.append(f"{k} {before[k]['name']}: retired into {spec['retire'][k]}; zone_successors {k} -> {spec['retire'][k]}")
    return before, after, touched, gone, waves, out, game, rows_out


# ---------------------------------------------------------------- the sheet

PAL = ["#8dd3c7", "#ffffb3", "#bebada", "#fb8072", "#80b1d3", "#fdb462", "#b3de69", "#fccde5", "#d9d9d9", "#bc80bd",
       "#ccebc5", "#ffed6f", "#a6cee3", "#b2df8a", "#fdbf6f", "#cab2d6"]


def density(view, nx=360):
    """GHS-POP 2025 (world_plan.Pop, 1 km) binned on a lon/lat grid over the view: log10 people a km2 scaled 0-1
    (10 to 10,000), masked where empty; None if the raster cannot be read."""
    import numpy as np
    try:
        pop = wp.Pop()
        w, s, e, n = view
        x, y, p = pop.pixels(wp.to_moll(shapely.box(w, s, e, n)).bounds)
    except Exception as x_:  # the sheet is still useful without it
        print("density layer skipped:", x_)
        return None
    if not len(p):
        return None
    lon, lat = wp.INV.transform(x, y)
    asp = 1 / max(0.2, math.cos(math.radians((s + n) / 2)))
    # bins of at least 2 km: finer than the 1 km pixels they show stripes
    nx = max(40, min(nx, int((e - w) * 111.32 * math.cos(math.radians((s + n) / 2)) / 2.0)))
    ny = max(10, int(nx * (n - s) * asp / (e - w)))
    h, _, _ = np.histogram2d(lat, lon, bins=[ny, nx], range=[[s, n], [w, e]], weights=p)
    km2 = ((e - w) / nx * 111.32 * math.cos(math.radians((s + n) / 2))) * ((n - s) / ny * 110.57)
    d = h / km2
    # scaled to this view (a Ganges-plain sheet is dense everywhere): shown from the 60th percentile of populated
    # cells (at least 30 a km2) to the 99.5th, log
    nz = d[d > 0]
    lo = max(30.0, float(np.percentile(nz, 60))) if len(nz) else 30.0
    hi = max(lo * 10, float(np.percentile(nz, 99.5))) if len(nz) else 3000.0
    v = np.clip(np.log10(np.maximum(d, 1e-9) / lo) / np.log10(hi / lo), 0, 1)
    return np.ma.masked_where(d < lo, v)


def sheet(png, spec, before, after, touched, gone, game):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    iso = spec["iso"]
    by = units(iso)
    hot = set(touched) | set(gone)
    if "view" in spec:
        view = spec["view"]
    else:
        g = wp.to_ll(shapely.union_all([geom(before[k]["units"][iso], by) for k in hot if k in before]))
        w, s, e, n = g.bounds
        m = 0.15 * max(e - w, n - s)
        view = (w - m, s - m, e + m, n + m)
    w, s, e, n = view
    bx = shapely.box(w, s, e, n)
    ids = sorted(set(before) | set(after))
    order = sorted(hot) + [k for k in ids if k not in hot]  # the changed rows take distinct colours first
    col = {k: PAL[i % len(PAL)] for i, k in enumerate(order)}
    asp = 1 / max(0.2, math.cos(math.radians((s + n) / 2)))
    ratio = (n - s) * asp / (e - w)
    fig, axes = plt.subplots(1, 2, figsize=(15, max(5.0, 7.2 * ratio) + 0.8))
    # other countries' packs in view, grey, for context
    reg = json.load(open(os.path.join(TW, "app", "src", "game", "registry.json"), encoding="utf-8"))["cities"]
    ctx = []
    for c in reg:
        if c["id"] in before or c["id"] in after or c["id"] in gone or not c.get("box"):
            continue
        bw, bs, be, bn = c["box"]
        if be < w or bw > e or bn < s or bs > n:
            continue
        try:
            gj = json.load(open(os.path.join(PACKS, f"{c['id']}.boundary.geojson"), encoding="utf-8"))
        except OSError:
            continue
        g = wp.polygonal(shapely.union_all([shapely.make_valid(shapely.geometry.shape(f["geometry"])) for f in gj["features"]]))
        if g.intersects(bx):
            ctx.append((c, g))
    dens = density(view)
    for ax, rows, title in ((axes[0], before, "now"), (axes[1], after, "proposed")):
        if dens is not None:
            ax.imshow(dens, extent=(w, e, s, n), origin="lower", cmap="Greys", vmin=0, vmax=1, alpha=0.55, zorder=2,
                      interpolation="nearest", aspect="auto")
        for c, g in ctx:
            for part in getattr(g, "geoms", [g]):
                ax.fill(*part.exterior.xy, color="#eeeeee", ec="#aaa", lw=0.4, zorder=0)
            lp = g.intersection(bx)
            if not lp.is_empty and lp.geom_type in ("Polygon", "MultiPolygon"):
                big = max(getattr(lp, "geoms", [lp]), key=lambda q: q.area)
                if big.area >= 0.002 * (e - w) * (n - s):
                    pt = shapely.ops.polylabel(big, 0.002)
                    ax.text(pt.x, pt.y, c["name"], ha="center", va="center", fontsize=5.5, color="#777", zorder=0)
        for k in ids:
            if k not in rows:
                continue
            us = [u for u in rows[k]["units"][iso] if u in by]
            if not us:
                continue
            gm = geom(us, by)
            g = wp.to_ll(gm)
            if not g.intersects(bx):
                continue
            t = k in hot
            for part in getattr(g, "geoms", [g]):
                if part.geom_type != "Polygon":
                    continue
                ax.fill(*part.exterior.xy, color=col[k], lw=0, zorder=1)
                ax.plot(*part.exterior.xy, color="#222" if t else "#888", lw=1.6 if t else 0.4, zorder=4 if t else 3)
            lp = g.intersection(bx)
            if lp.is_empty or lp.geom_type not in ("Polygon", "MultiPolygon"):
                continue
            big = max(getattr(lp, "geoms", [lp]), key=lambda q: q.area)
            if big.area < 0.0004 * (e - w) * (n - s):
                continue
            pt = shapely.ops.polylabel(big, 0.002)
            ppl = sum(game[u] for u in us)
            name = (spec.get("show_names") or {}).get(k) or rows[k]["name"]
            hh = gm.area / gm.convex_hull.area
            mp = max(getattr(gm, "geoms", [gm]), key=lambda q: q.area)
            hp = mp.area / mp.convex_hull.area
            txt = f"{name} ({k})\n{ppl / 1e6:.2f}M" + ((f"\nhull {hh:.2f}" + (f"\n(main piece {hp:.2f})" if hp - hh > 0.05 else ""))
                                                       if t else "")
            ax.text(pt.x, pt.y, txt, ha="center", va="center", fontsize=8 if t else 6, weight="bold" if t else "normal",
                    zorder=5)
        ax.set_xlim(w, e)
        ax.set_ylim(s, n)
        ax.set_aspect(asp)
        ax.set_title(f"{spec.get('label', spec['task'])}: {title}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    if spec.get("note"):
        fig.text(0.5, 0.035, spec["note"], ha="center", fontsize=9, color="#222")
    fig.text(0.5, 0.01, "People as the game counts them (after: estimated from each unit's old zone). Bold outlines "
             "change; hull = area over convex hull. Grey: GHS-POP 2025 people a km2, log scale over the denser part of this view.", ha="center", fontsize=8, color="#555")
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.savefig(png, dpi=100)
    plt.close(fig)


def main():
    if "--go" in sys.argv or "retire" in sys.argv[2:]:
        raise SystemExit("dry.py never applies a spec")
    path = sys.argv[1]
    out = sys.argv[sys.argv.index("--out") + 1]
    os.makedirs(out, exist_ok=True)
    spec = json.load(open(path, encoding="utf-8"))
    doc = shared_lock.read_json(pe.PLAN)
    before, after, touched, gone, waves, lines, game, rows = checks(spec, doc)
    lines.append("waves: " + " | ".join(",".join(w) for w in waves))
    # build cost: each rebuilt row's recorded step seconds (a new row its template's), one core a row
    rows_by = {r["id"]: r for r in doc["metros"]}
    tot = 0.0
    parts = []
    for k in touched:
        src = rows_by.get(k) or rows_by.get(spec.get("new", {}).get(k, {}).get("like"))
        sec = sum(((src or {}).get("run") or {}).get("steps", {}).values())
        tot += sec
        parts.append(f"{k} {sec / 60:.1f}")
    lines.append(f"build: {len(touched)} rows, recorded steps {tot / 60:.0f} core-min ({', '.join(parts)}); retired {gone or 'none'}; "
                 f"then the {spec['iso']} join (see notes/T-4364/joins.txt)")
    stem = os.path.splitext(os.path.basename(path))[0]
    open(os.path.join(out, stem + ".dry.txt"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    json.dump({"waves": waves, "retire": gone, "rows": touched}, open(os.path.join(out, stem + ".waves.json"), "w"), indent=1)
    print("\n".join(lines))
    if "--png" in sys.argv:
        png = sys.argv[sys.argv.index("--png") + 1]
        sheet(png, spec, before, after, touched, gone, game)
        print("->", png)


if __name__ == "__main__":
    main()
