"""T-463 / T-465: Japan's packs with at home by municipality of work, then one joint commute gravity.

    python notes/t463_jp.py fit <id>        the zone fit grid (japan_zones.fit's rule) with the pack's
                                            old and new commute margins: does the choice move?
    python notes/t463_jp.py commute <id>    the res-9 pack in data/packs: new commute arrays, header,
                                            then its own gravity re-solved (pack_gravity)
    python notes/t463_jp.py flows           cross-pack commuters against the census (after the joint)

Run from the maps root with the venv; `commute` expects the res-9 pack in data/packs
(merge_cells.py --undo first). Results to ../artifacts/trainworld/T-463-*.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
import json
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "pipeline"))
import build_city as b  # noqa: E402
import japan  # noqa: E402
import tier0  # noqa: E402
import england  # noqa: E402
from read_pack import load_pack, PACKS  # noqa: E402

ART = os.path.join(HERE, "..", "..", "artifacts", "trainworld")
GRID = [(pw, km) for pw in (1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4) for km in (16.0, 32.0, 50.0, 80.0, 160.0)]
WEST = {"Musashino": "13203", "Mitaka": "13204", "Suginami": "13115", "Nerima": "13120", "Chofu": "13208",
        "Fuchu": "13206", "Tachikawa": "13202", "Hachioji": "13201"}
C3, C5 = ["13101", "13102", "13103"], ["13101", "13102", "13103", "13104", "13113"]


def lonlat(h, a):
    o = h["origin"]
    lat = o["lat"] + np.degrees(a["y_m"].astype(np.float64) / b.R_EARTH)
    lon = o["lon"] + np.degrees(a["x_m"].astype(np.float64) / (b.R_EARTH * np.cos(np.radians(o["lat"]))))
    return lon, lat


def new_commute(cid, h, a):
    cfg = b.CITIES[cid]
    lon, lat = lonlat(h, a)
    at = cfg["at_home"](lat, lon)
    jobs, pop = a["jobs"].astype(np.float64), a["pop"].astype(np.float64)
    work = jobs * (1 - at) * (1 - cfg["telework"])
    home = pop * work.sum() / pop.sum()
    return home, work, at


def miss(m, c):
    return 0.5 * float(np.abs(m / m.sum() - c / c.sum()).sum())


def fit(cid):
    cfg = b.CITIES[cid]
    units = cfg["prefectures"]
    h, a = load_pack(cid, os.path.join(b.ROOT, "data", "work", "res9"))
    home_new, work_new, _ = new_commute(cid, h, a)
    x, y = a["x_m"].astype(np.float64), a["y_m"].astype(np.float64)
    lon, lat = lonlat(h, a)
    ms = japan.municipalities(units)
    nl = len(ms)
    ci = {m[0]: i for i, m in enumerate(ms)}
    li, near = england.cell_units(lat, lon, [(m[0], m[3]) for m in ms])
    li = np.where(li < 0, 0, li)
    for u, p in near.items():
        li[p] = u
    fl = japan.flows(units)
    fl = fl[fl.h.isin(ci) & fl.w.isin(ci)]
    Cl = np.zeros((nl, nl))
    np.add.at(Cl, (fl.h.map(ci).to_numpy(), fl.w.map(ci).to_numpy()), fl.n.to_numpy(float))
    own_census = float(np.trace(Cl) / Cl.sum())
    zone, gx, gy = tier0.zones_of(x, y)
    nz = len(gx)
    d = tier0.zone_distances(gx, gy)
    out = {"id": cid, "zones": nz, "own_census": own_census, "shipped": [float(h["gravity"]["decay_pow"]), float(h["gravity"]["decay_km"])]}
    shipped = (round(out["shipped"][0], 2), round(out["shipped"][1], 2))
    for tag, hc, wc in (("old", a["commute_home"].astype(np.float64), a["commute_work"].astype(np.float64)), ("new", home_new, work_new)):
        H = np.bincount(zone, weights=hc, minlength=nz)
        J = np.bincount(zone, weights=wc, minlength=nz)
        P = np.zeros((nz, nl)); Q = np.zeros((nz, nl))
        np.add.at(P, (zone, li), hc); np.add.at(Q, (zone, li), wc)
        P /= np.maximum(P.sum(1, keepdims=True), 1e-12)
        Q /= np.maximum(Q.sum(1, keepdims=True), 1e-12)
        rows = []
        for pw, km in GRID + ([shipped] if shipped not in GRID else []):
            T = tier0.gravity(H, J, d, pw, km)
            C = P.T @ T.astype(np.float64) @ Q
            rows.append({"pow": pw, "km": km, "misplaced": miss(C, Cl), "own": float(np.trace(C) / C.sum())})
            if (pw, km) == shipped and cid == "tyo":
                out[tag + "_tyo"] = tyo_table(C, Cl, ms)
        grid = [r for r in rows if (r["pow"], r["km"]) in GRID]
        lo = min(r["misplaced"] for r in grid)
        best = min((r for r in grid if r["misplaced"] <= lo + 0.003), key=lambda r: abs(r["own"] - own_census))
        out[tag] = {"best": best, "shipped": next(r for r in rows if (r["pow"], r["km"]) == shipped),
                    "commuters": float(wc.sum())}
        print(f"{cid} {tag}: commuters {wc.sum():,.0f}; grid choice d^-{best['pow']} exp(-d/{best['km']:g}) misplaced "
              f"{best['misplaced']:.1%} own {best['own']:.1%} (census {own_census:.1%}); shipped {shipped} misplaced "
              f"{out[tag]['shipped']['misplaced']:.1%} own {out[tag]['shipped']['own']:.1%}", flush=True)
    os.makedirs(ART, exist_ok=True)
    with open(os.path.join(ART, f"T-463-fit-{cid}.json"), "w") as f:
        json.dump(out, f, indent=1)


def tyo_table(C, Cl, ms):
    """notes/T-465.md's west-to-centre shares, model and census."""
    ci = {m[0]: i for i, m in enumerate(ms)}
    r = {}
    for tag, M in (("model", C), ("census", Cl)):
        o = [ci[c] for c in WEST.values()]
        r[tag] = {"west_c3": float(M[o][:, [ci[c] for c in C3]].sum() / M[o].sum()),
                  "west_c5": float(M[o][:, [ci[c] for c in C5]].sum() / M[o].sum()),
                  "musashino_c5": float(M[ci["13203"], [ci[c] for c in C5]].sum() / M[ci["13203"]].sum()),
                  "all_c3": float(M[:, [ci[c] for c in C3]].sum() / M.sum())}
        for n, c in WEST.items():
            r[tag][n] = [float(M[ci[c], [ci[k] for k in C3]].sum() / M[ci[c]].sum()), float(M[ci[c], [ci[k] for k in C5]].sum() / M[ci[c]].sum())]
    return r


def write_res9(cid, h, a, arrays):
    """The res-9 pack in data/packs with some per-cell arrays replaced (same layout)."""
    stem = os.path.join(PACKS, cid)
    buf = bytearray()
    for spec in h["arrays"]:
        v = arrays.get(spec["name"], a.get(spec["name"]))
        if v is None:
            sys.exit(f"{cid}: array {spec['name']} cannot be carried over")
        v = np.asarray(v).astype(a[spec["name"]].dtype) if spec["name"] in arrays else v
        buf += b"\0" * (-len(buf) % 8)
        spec["offset"], spec["count"] = len(buf), int(len(v))
        buf += np.ascontiguousarray(v).tobytes()
    with open(stem + ".bin.part", "wb") as f:
        f.write(buf)
    with open(stem + ".json.part", "w") as f:
        json.dump(h, f, indent=2)
    os.replace(stem + ".bin.part", stem + ".bin")
    os.replace(stem + ".json.part", stem + ".json")


def commute(cid):
    cfg = b.CITIES[cid]
    h, a = load_pack(cid)
    assert "merged" not in h, f"{cid}: data/packs holds the merged pack; merge_cells.py {cid} --undo first"
    home, work, at = new_commute(cid, h, a)
    wfh = 1 - work.sum() / a["jobs"].astype(np.float64).sum()
    before = float(a["commute_work"].astype(np.float64).sum())
    h = json.loads(json.dumps(h))
    h["country"] = "JP"
    h["totals"]["commute_home"] = int(round(home.sum()))
    h["totals"]["commute_work"] = int(round(work.sum()))
    h["commute"]["work"] = tier0.work_rule(cfg, wfh)
    h["commute"]["region_share"] = round(float(wfh), 4)
    write_res9(cid, h, a, {"commute_home": home, "commute_work": work})
    print(f"{cid}: commuters {before:,.0f} -> {work.sum():,.0f}; work from home {wfh:.3f} (cells {at.min():.3f}-{at.max():.3f} at home)", flush=True)
    b.CITIES[cid].setdefault("border_factor", 1.0)
    b.add_gravity(os.path.join(PACKS, f"{cid}.json"), cid)


def flows(path=None):
    """Commuters between packs (data/work/jp.joint_report.json) against the census's workers
    between the same prefectures (table 4 by prefecture of work; residents of the home pack's
    prefectures working in the work pack's). Hokkaido's packs are left out (subprefecture units)."""
    rep = json.load(open(path or os.path.join(b.WORK, "jp.joint_report.json")))
    ids, bp = rep["packs"], np.array(rep["by_pack"]).reshape(len(rep["packs"]), -1)
    pref = {c: b.CITIES[c]["prefectures"] for c in ids if all(len(p) == 2 for p in b.CITIES[c]["prefectures"])}
    tabs = {}
    for c, ps in pref.items():
        for p in ps:
            t = japan.flows_table(p)
            t = t[~t.h.isin(["all", "home"])]
            tabs[p] = t.assign(hp=t.h.str[:2])
    rows = []
    for i, a in enumerate(ids):
        for j, w in enumerate(ids):
            if i == j or a not in pref or w not in pref:
                continue
            census = sum(float(tabs[p][tabs[p].hp.isin(pref[a])].n.sum()) for p in pref[w])
            rows.append((a, w, float(bp[i, j]), census))
    rows.sort(key=lambda r: -max(r[2], r[3]))
    lines = ["| From | To | Model | Census | Ratio |", "|---|---|---:|---:|---:|"]
    for a, w, m, c in rows[:30]:
        lines.append(f"| {b.CITIES[a].get('region_name')} | {b.CITIES[w].get('region_name')} | {m:,.0f} | {c:,.0f} | {m / max(c, 1):.2f} |")
    tm, tc = sum(r[2] for r in rows), sum(r[3] for r in rows)
    mis = 0.5 * sum(abs(r[2] / tm - r[3] / tc) for r in rows)
    lines.append(f"\nall pairs: model {tm:,.0f}, census {tc:,.0f} ({tm / tc:.2f}); pair shares misplaced {mis:.1%}; "
                 f"commuters {rep['commuters']:,.0f}, across {rep['cross']:,.0f}")
    print("\n".join(lines))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "fit":
        for c in args:
            fit(c)
    elif cmd == "commute":
        for c in args:
            commute(c)
    elif cmd == "flows":
        flows(args[0] if args else None)
