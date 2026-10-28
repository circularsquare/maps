"""T-1611: summarise measure.mjs JSON from the diag build: per solving-worker report, live heap and
at-peak by category, by method, coarse vs fine zones."""
import json, sys, collections

MB = 1 << 20
d = json.load(open(sys.argv[1]))
which = sys.argv[2] if len(sys.argv) > 2 else None


def cat(name):
    base = name.split("@")[0]
    if base.startswith("mkt.") or base.startswith("comp."):
        return base
    cls, _, meth = base.partition(".")
    if cls == "DemandApi" and meth in ("new", "at_level"):
        return "zone pack: commutes"
    if cls == "EverydayApi" and meth.startswith("beside"):
        return "zone pack: everyday"
    if cls in ("DemandApi", "EverydayApi"):
        if "network" in meth or meth.startswith("set_"):
            return f"zone network ({cls}.{meth})"
        if meth in ("solve", "crowd_with_external", "estimate", "crowd", "solve_round", "solve_part"):
            return f"zone solved ({cls}.{meth})"
        return f"zone other ({cls}.{meth})"
    if cls == "MarketPacks":
        return f"market packs ({meth})"
    if cls == "IntercityApi":
        return f"market part ({meth})"
    return base


def show(rep, fine, label):
    tags = rep["tags"]
    print(f"\n== {label}: live {rep['live']/MB:.0f} MB, peak {rep['peak']/MB:.0f}, wasm {rep['mem']/MB:.0f}, blocks {rep['blocks']}")
    c_live, c_peak = collections.Counter(), collections.Counter()
    z = collections.defaultdict(collections.Counter)
    zp = collections.defaultdict(collections.Counter)
    for name, live, mx, ap in tags:
        k = cat(name)
        c_live[k] += live
        c_peak[k] += ap
        if "@" in name:
            zid = name.split("@")[1]
            lvl = "fine" if zid in fine else "coarse"
            z[lvl][k] += live
            zp[lvl][zid] += live
    for k in sorted(set(c_live) | set(c_peak), key=lambda k: -max(c_live[k], c_peak[k])):
        if max(c_live[k], c_peak[k]) < 2 * MB:
            continue
        print(f"  {k:60} {c_live[k]/MB:9.1f} {c_peak[k]/MB:9.1f}")
    print(f"  {'(sum)':60} {sum(c_live.values())/MB:9.1f} {sum(c_peak.values())/MB:9.1f}")
    for lvl in z:
        n = len(zp[lvl])
        v = sorted(zp[lvl].values())
        print(f"  {lvl} zones {n}: total {sum(v)/MB:.0f} MB, median {v[n//2]/MB:.2f}, max {v[-1]/MB:.1f}")
        for k, b in z[lvl].most_common():
            if b > MB:
                print(f"      {k:56} {b/MB:9.1f}")


for key in ("final", "refined", "atError"):
    r = d.get(key)
    if not r or not r.get("diag"):
        continue
    fine = set((r.get("lod") or {}).get("fineOpen") or [])
    print(f"\n######## {key}: rail {r['rail']:.0f}, memory {r['memory']}, fine {sorted(fine)}")
    for rep in r["diag"]["full"]:
        if rep["aux"]:
            continue
        if which and rep["kind"] != which:
            continue
        show(rep, fine, f"{key} {rep['kind']} v{rep.get('version')} cities {len(rep['cities'])}")
    lt = [x for x in r["diag"]["light"] if not x["aux"]]
    if lt:
        print("  light (solving):", " ".join(f"{x['kind']}:{x['live']/MB:.0f}/{x['peak']/MB:.0f}/{x['mem']/MB:.0f}" for x in lt[-30:]))
