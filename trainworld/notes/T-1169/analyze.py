"""T-1169: summarise a t1169-measure.mjs JSON: per worker, live heap at rest and at the solve's peak
by category and by method, and per zone."""
import json, sys, re, collections

MB = 1 << 20
d = json.load(open(sys.argv[1]))
print("rail", d.get("rail"), "computeN", d.get("computeN"), "final s", d.get("finalSecs"), "stations", d.get("stations"), "lines", d.get("lines"), "opened", len(d.get("opened", [])), "errors", len(d.get("errors", [])))
for m in d.get("memory", []):
    print("  client memory", m)
diag = d.get("diag", {})

def cat(name):
    base = name.split("@")[0]
    if base.startswith("mkt."):
        return base
    if base == "skims.cache":
        return "skims cache"
    cls, _, meth = base.partition(".")
    if cls == "DemandApi" and meth == "new":
        return "pack: commutes (DemandApi.new)"
    if cls == "EverydayApi" and meth.startswith("beside"):
        return "pack: everyday (EverydayApi.beside_with)"
    if cls in ("DemandApi", "EverydayApi"):
        if "network" in meth or meth in ("set_complexes", "set_upgrades", "set_passages", "set_span", "set_bus", "set_fares", "set_fare_curve"):
            return f"{cls}: network setup ({meth})" if False else f"city: network ({cls})"
        if meth in ("solve", "crowd_with_external", "estimate", "crowd", "solve_round"):
            return f"city: solve state ({cls})"
        return f"city: other ({cls}.{meth})"
    if cls == "MarketPacks":
        return f"market: composition ({meth})"
    if cls == "IntercityApi":
        if meth in ("beside", "on"):
            return "market part: made (IntercityApi.beside)"
        if "network" in meth or meth.startswith("set_"):
            return "market part: network"
        if meth in ("solve", "crowd_with_external", "estimate"):
            return "market part: solve state"
        return f"market part: other ({meth})"
    return base

def show(rep, label):
    tags = rep["tags"]
    print(f"\n== {label}: live {rep['live']/MB:.0f} MB, peak {rep['peak']/MB:.0f} MB, wasm {rep['mem']/MB:.0f} MB, blocks {rep['blocks']}")
    c_live, c_peak = collections.Counter(), collections.Counter()
    m_live, m_peak = collections.Counter(), collections.Counter()
    z_live = collections.Counter()
    for name, live, mx, ap in tags:
        c_live[cat(name)] += live
        c_peak[cat(name)] += ap
        m_live[name.split("@")[0]] += live
        m_peak[name.split("@")[0]] += ap
        if "@" in name:
            z_live[name.split("@")[1]] += live
    print(f"  {'category':52} {'at rest':>9} {'at peak':>9}")
    for k in sorted(set(c_live) | set(c_peak), key=lambda k: -max(c_live[k], c_peak[k])):
        if max(c_live[k], c_peak[k]) < 1 * MB:
            continue
        print(f"  {k:52} {c_live[k]/MB:9.1f} {c_peak[k]/MB:9.1f}")
    print(f"  {'(sum)':52} {sum(c_live.values())/MB:9.1f} {sum(c_peak.values())/MB:9.1f}")
    print("  by method (>5 MB):")
    for k in sorted(set(m_live) | set(m_peak), key=lambda k: -max(m_live[k], m_peak[k])):
        if max(m_live[k], m_peak[k]) < 5 * MB:
            continue
        print(f"    {k:50} {m_live[k]/MB:9.1f} {m_peak[k]/MB:9.1f}")
    if z_live:
        v = sorted(z_live.values())
        n = len(v)
        print(f"  zones with tagged memory: {n}; live per zone MB: median {v[n//2]/MB:.1f}, p90 {v[int(n*0.9)]/MB:.1f}, max {v[-1]/MB:.1f}, sum {sum(v)/MB:.0f}")
        print("  biggest zones:", ", ".join(f"{z} {b/MB:.0f}" for z, b in z_live.most_common(8)))
    if rep.get("counts"):
        print("  counts:", rep["counts"])
    return c_live, c_peak, z_live

full = diag.get("full", {})
for key, rep in sorted(full.items()):
    if rep["kind"] in ("done",):
        show(rep, f"worker {rep['w']} aux={rep['aux']} mine={rep['mine']} {rep['kind']} v{rep.get('version')} cities {len(rep['cities'])} bare {len(rep['bare'])}")
# per-zone breakdown by kind for one solving worker
for key, rep in sorted(full.items()):
    if rep["kind"] == "done" and not rep["aux"]:
        per = collections.defaultdict(collections.Counter)
        for name, live, mx, ap in rep["tags"]:
            if "@" in name:
                z = name.split("@")[1]
                per[z][cat(name)] += live
        if len(sys.argv) > 2:
            out = {z: dict(c) for z, c in per.items()}
            json.dump(out, open(sys.argv[2], "w"))
        break
# opening timeline (light): live against cities open, per worker
light = diag.get("light", [])
byw = collections.defaultdict(list)
for x in light:
    byw[(x["w"], x["aux"])].append(x)
for (w, aux), xs in byw.items():
    opens = [x for x in xs if x["kind"] in ("opened", "ready")]
    if opens:
        last = opens[-1]
        print(f"\nworker {w} aux={aux}: {len(opens)} opens; last open: live {last['live']/MB:.0f} MB with {last['ncities']} cities, {last['nbare']} bare, wasm {last['mem']/MB:.0f}")
    for x in xs:
        if x["kind"] == "done":
            print(f"   done at {x['t']/1000:.0f} s: live {x['live']/MB:.0f}, peak {x['peak']/MB:.0f}, wasm {x['mem']/MB:.0f}")
