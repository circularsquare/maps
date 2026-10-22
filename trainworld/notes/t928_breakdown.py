"""Where a batch's gate and join minutes go, from the world_run logs' '### <time> step' headers and
the gate files' mtimes. One line per batch, then totals.

    python breakdown.py 20261020 [20261023]     batches whose name sorts in [from, to)
"""
import datetime
import json
import re
import sys
from pathlib import Path

BASE = Path(r"C:\Users\anita\projects\maps\artifacts\trainworld\world_run")


def heads(p):
    """[(time, step text)] of a log's '###' headers, and the log's mtime."""
    out = []
    for ln in p.read_text(encoding="utf-8", errors="replace").splitlines():
        m = re.match(r"### (\S+) (.*)", ln)
        if m:
            out.append((datetime.datetime.fromisoformat(m.group(1)), m.group(2)))
    return out, datetime.datetime.fromtimestamp(p.stat().st_mtime)


def mt(p):
    return datetime.datetime.fromtimestamp(p.stat().st_mtime)


def batch(d):
    r = {"register": 0, "gate_pre": 0, "browser": 0, "gate_between": 0, "zones_gated": 0, "regions": 0,
         "groups": 0, "solve": 0, "merge": 0, "regate": 0, "market": 0, "tail": 0, "join_packs": 0}
    for reg in sorted(d.glob("region*.register.log")):
        k = reg.name.split(".")[0]
        h, end = heads(reg)
        if not h:
            continue
        r["register"] += (end - h[0][0]).total_seconds()
        g = d / f"{k}.gate.log"
        if not g.exists():
            continue
        gh, gend = heads(g)
        if not gh:
            continue
        r["regions"] += 1
        ids = [a for a in gh[0][1].split()[1:] if not a.startswith("--") and "\\" not in a and "=" not in a]
        rows = sorted((mt(d / "gate" / f"{i}.row.json"), i) for i in ids if (d / "gate" / f"{i}.row.json").exists()
                      and mt(d / "gate" / f"{i}.row.json") >= gh[0][0])
        if not rows:
            r["gate_pre"] += (gend - gh[0][0]).total_seconds()
            continue
        r["gate_pre"] += (rows[0][0] - gh[0][0]).total_seconds()
        prev = None
        for t, i in rows:
            b = d / "gate" / f"{i}.browser.json"
            if b.exists() and mt(b) >= t:
                r["browser"] += (mt(b) - t).total_seconds()
                r["zones_gated"] += 1
                if prev is not None:
                    r["gate_between"] += (t - prev).total_seconds()
                prev = mt(b)
        if prev is not None:
            r["gate_between"] += (gend - prev).total_seconds()
    for gl in sorted(d.glob("join.group-*.log")):
        if gl.name.endswith(".gate.log"):
            continue
        h, end = heads(gl)
        h.sort(key=lambda x: (x[0], 0 if "snapshot" in x[1] else 1 if "--undo" in x[1] else 2 if x[1].startswith("intercity") else 3))
        for k, (t, what) in enumerate(h):
            nxt = h[k + 1][0] if k + 1 < len(h) else end
            s = (nxt - t).total_seconds()
            if what.startswith("intercity"):
                r["solve"] += s
                m = re.search(r"--solve (\S+)", what)
                r["join_packs"] += len(m.group(1).split(",")) if m else 0
            elif what.startswith("merge_cells"):
                r["merge"] += s
        ga = gl.with_name(gl.name[:-4] + ".gate.log")
        if ga.exists():
            gh, gend = heads(ga)
            if gh:
                r["regate"] += (gend - gh[0][0]).total_seconds()
    for name, key in (("join.groups.log", "groups"), ("join.market.log", "market"), ("join.outlines.log", "tail"), ("join.aliases.log", "tail")):
        p = d / name
        if p.exists():
            h, end = heads(p)
            if h:
                r[key] += (end - h[0][0]).total_seconds()
    return r


if __name__ == "__main__":
    lo = sys.argv[1] if len(sys.argv) > 1 else "20261020"
    hi = sys.argv[2] if len(sys.argv) > 2 else "99999999"
    keys = ["register", "gate_pre", "browser", "gate_between", "groups", "solve", "merge", "regate", "market", "tail"]
    tot = {k: 0 for k in keys + ["zones_gated", "regions", "join_packs"]}
    print(f"{'batch':22} {'wall':>5} {'reg':>3} {'zones':>5} " + " ".join(f"{k[:8]:>8}" for k in keys) + "  (minutes)")
    for rep in sorted(BASE.glob("*/report.json")):
        d = rep.parent
        if not (lo <= d.name < hi):
            continue
        r = batch(d)
        wall = json.loads(rep.read_text(encoding="utf-8")).get("seconds", 0)
        for k in tot:
            tot[k] += r[k]
        print(f"{d.name:22} {wall / 60:5.0f} {r['regions']:3} {r['zones_gated']:5} " + " ".join(f"{r[k] / 60:8.1f}" for k in keys))
    print(f"{'total':22} {'':5} {tot['regions']:3} {tot['zones_gated']:5} " + " ".join(f"{tot[k] / 60:8.1f}" for k in keys))
    if tot["zones_gated"]:
        print(f"per gated zone: browser {tot['browser'] / tot['zones_gated']:.1f} s, figures and other checks between "
              f"{tot['gate_between'] / tot['zones_gated']:.1f} s; per region: register {tot['register'] / max(tot['regions'], 1) / 60:.1f} min, "
              f"before the first zone (rates) {tot['gate_pre'] / max(tot['regions'], 1) / 60:.1f} min")
    if tot["join_packs"]:
        print(f"joins: {tot['join_packs']} pack solves, {tot['solve'] / tot['join_packs']:.1f} s of solve a pack solved")
