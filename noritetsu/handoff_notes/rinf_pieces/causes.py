"""Why each register line ships in pieces, from a harness run's stages.json and lines.json.

    python causes.py <base dir> [cc ...] [--lines N] [--detail NAME]

Per line, pieces at each stage: RINF's own section graph (raw), the reader's output, before
and after drop_unridden_sections, and the final lines.json. A cause is counted where a stage
adds pieces:
  rinf-hole    RINF's own sections of the line do not connect
  reader       the reader left sections out (rejected/untraceable trace, unplaced point, redundant)
  drop-gap     drop_unridden_sections dropped a junction-ended section between two pieces
  other        merge / border / split hooks
Also lists each dropped junction-ended section that sat between two pieces (both ends still on
the line after the drop, in different pieces), with its route share."""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent))
from shipped import pieces, RINF  # noqa: E402


def comps(secs):
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for s in secs:
        parent[find(s[0])] = find(s[1])
    return find


def analyse(base, cc, rows, gaps):
    d = base / cc
    if not (d / "stages.json").exists():
        return None
    st = json.loads((d / "stages.json").read_text(encoding="utf-8"))
    fin = {l["id"]: l for l in json.loads((d / "lines.json").read_text(encoding="utf-8"))["lines"]}
    stn = json.loads((d / "stations.json").read_text(encoding="utf-8"))["stations"]
    name = lambda s: (stn.get(s) or {}).get("n", s)
    groups, reader = st.get("groups", {}), st.get("reader", {})
    pre, post = st.get("predrop", {}), st.get("postdrop", {})
    share = st.get("share", {})
    junction = set(st.get("junction", ()))
    cnt = Counter()
    n_lines = 0
    logf = base / f"{cc}.log"
    loglines = logf.read_text(encoding="utf-8", errors="replace").splitlines() if logf.exists() else []
    rej = [x.split("rejected:", 1)[1].strip() for x in loglines if "  rejected: " in x]
    untr = [x.split("untraceable:", 1)[1].strip() for x in loglines if "  untraceable: " in x]

    def reader_why(l):
        ids = l.get("rinf_ids") or []
        key = "/".join(ids)[:24]
        r = sum(1 for x in rej if x.startswith(key))
        u = sum(1 for x in untr if x.split(" ", 1)[0] in ids)
        unp = sum(1 for x in untr if x.split(" ", 1)[0] in ids and x.endswith("unplaced"))
        return f"(rejected {r}, untraceable {u - unp}, unplaced {unp})"
    for lid, l in fin.items():
        if l.get("src", "osm") == "osm" or l.get("service"):
            continue
        n_lines += 1
        kf = pieces(l["sections"])
        if len(kf) < 2:
            continue
        raw = len((groups.get(lid) or {}).get("raw_pieces") or [1])
        kr = len(pieces(reader[lid]["sections"])) if lid in reader else 1
        kp = len(pieces(pre[lid]["sections"])) if lid in pre else kr
        kd = len(pieces(post[lid]["sections"])) if lid in post else kp
        why = []
        if raw > 1:
            why.append(f"rinf-hole+{raw - 1}")
        if kr > raw:
            why.append(f"reader+{kr - raw}{reader_why(l)}")
        if kd > kp:
            why.append(f"drop-gap+{kd - kp}")
        other = (kp - kr) + (len(kf) - kd)
        if other > 0:
            why.append(f"other+{other}")
        if not why:
            why.append("?")
        for w in why:
            cnt[w.split("+")[0]] += 1
        if any(w.startswith("reader") for w in why):
            r = reader_why(l)
            cnt["reader:rejected"] += " rejected 0," not in r
            cnt["reader:untraceable"] += ", untraceable 0," not in r
            cnt["reader:unplaced"] += ", unplaced 0)" not in r
        rows.append((sum(kf[1:]), cc, l["name"], l.get("ref", ""), [round(x, 1) for x in kf],
                     " ".join(why)))
        # dropped junction-ended sections between two pieces after the drop
        if lid in pre and lid in post:
            find = comps(post[lid]["sections"])
            have = {s for sec in post[lid]["sections"] for s in sec[:2]}
            kept = {frozenset(s[:2]) for s in post[lid]["sections"]}
            for a, b, km in pre[lid]["sections"]:
                if frozenset((a, b)) in kept:
                    continue
                if a in have and b in have and find(a) != find(b):
                    gaps.append((km, cc, l["name"], name(a), name(b),
                                 share.get(f"{lid}|{a}|{b}", share.get(f"{lid}|{b}|{a}"))))
    return n_lines, cnt


def main():
    args = sys.argv[1:]
    base = Path(args.pop(0))
    nshow = 15
    if "--lines" in args:
        i = args.index("--lines")
        nshow = int(args[i + 1])
        del args[i:i + 2]
    ccs = args or RINF
    rows, gaps, tot = [], [], Counter()
    for cc in ccs:
        r0 = len(rows)
        got = analyse(base, cc, rows, gaps)
        if got is None:
            continue
        n_lines, cnt = got
        mine = rows[r0:]
        tot.update(cnt)
        print(f"{cc}: {n_lines} register lines, {len(mine)} in pieces, "
              f"{sum(sum(r[4]) for r in mine):,.0f} km on them, "
              f"{sum(r[0] for r in mine):,.0f} km outside the biggest piece; causes {dict(cnt)}")
    print(f"ALL: {len(rows)} in pieces, {sum(r[0] for r in rows):,.0f} km outside; {dict(tot)}")
    print(f"worst {nshow}:")
    for r in sorted(rows, reverse=True)[:nshow]:
        print(f"  {r[0]:7.1f}  {r[1]} {r[2]} [{r[3]}] {r[4]}  {r[5]}")
    print(f"dropped junction-ended sections between two pieces: {len(gaps)}, "
          f"{sum(g[0] for g in gaps):,.0f} km; route share histogram "
          f"{Counter(round((g[5] or 0) * 10) / 10 for g in gaps).most_common()}")
    for g in sorted(gaps, reverse=True)[:nshow]:
        print(f"  {g[0]:6.1f} km  {g[1]} {g[2]}: {g[3]} - {g[4]}  share {g[5]}")


if __name__ == "__main__":
    main()
