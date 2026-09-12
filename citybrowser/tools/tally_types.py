"""
Tally every P31 type in use, with labels and example members.

This is the report the `kinds.py` sets were picked from, and the one to rerun
before editing them -- a type that has grown from 4 rows to 400 since the last
pass is exactly what a hand-curated list gets wrong.

    python tools/tally_types.py                 # top 150 by row count
    python tools/tally_types.py --all           # every type, to a file
    python tools/tally_types.py --kind rural    # what a set currently catches
    python tools/tally_types.py --grep county   # types whose label matches

The tally is over the rows that survive INTO base.json, not the raw fetch pool:
a type carried only by rows the settlement filter drops is not a decision
anybody has to make.
"""
import argparse
import json
import pathlib
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import kinds

HERE = pathlib.Path(__file__).resolve().parent.parent
CACHE = HERE / "cache"
DATA = HERE / "data"

SETS = {
    "aggregate": kinds.AGGREGATE, "rural": kinds.RURAL,
    "district": kinds.DISTRICT, "admin": kinds.ADMIN,
    "neutral": kinds.NEUTRAL, "extra": kinds.EXTRA_SETTLEMENT,
}


def pool():
    """The typed fetch pool -- same choice rule as assemble_base.best_pool()."""
    best = None
    for p in sorted(CACHE.glob("wikidata*")):
        if p.is_dir() and list(p.glob("Q*.json")):
            best = p if best is None or len(list(p.glob("Q*.json"))) >= 1 else best
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="write every type to a file")
    ap.add_argument("--kind", choices=sorted(SETS), help="only types in this set")
    ap.add_argument("--grep", help="only types whose label matches this regex")
    ap.add_argument("--top", type=int, default=150)
    args = ap.parse_args()

    base = json.loads((DATA / "base.json").read_text(encoding="utf-8"))
    lp = CACHE / "type_labels.json"
    lab = json.loads(lp.read_text(encoding="utf-8")) if lp.exists() else {}

    rows = {}
    for p in pool().glob("Q*.json"):
        if p.stem != "_countries":
            rows.update(json.loads(p.read_text(encoding="utf-8")))
    rows = {q: r for q, r in rows.items() if q in base}

    cnt = Counter()
    ex = defaultdict(list)
    for q, r in rows.items():
        for t in (r.get("types") or []):
            cnt[t] += 1
            if len(ex[t]) < 4:
                ex[t].append((r.get("name") or q, int(r.get("pop") or 0)))

    sel = set(cnt)
    if args.kind:
        sel &= SETS[args.kind]
    if args.grep:
        rx = re.compile(args.grep, re.I)
        sel = {t for t in sel if rx.search(lab.get(t, {}).get("label", t))}

    where = {t: k for k, s in SETS.items() for t in s}
    lines = []
    for t in sorted(sel, key=lambda t: -cnt[t]):
        m = lab.get(t, {})
        tag = where.get(t, "")
        names = ", ".join(f"{n[:18]} {p:,}" for n, p in ex[t])
        lines.append(f"{cnt[t]:7,}  {t:13} {tag:10} {m.get('label', '?')[:38]:40} "
                     f"| {m.get('desc', '')[:44]:46} | {names}")

    print(f"{len(rows):,} rows in base.json, {len(cnt):,} distinct types, "
          f"{len(sel):,} shown")
    if args.all:
        out = DATA / "type_tally.txt"
        out.write_text("\n".join(lines), encoding="utf-8")
        print(f"wrote {out}")
    else:
        print("\n".join(lines[:args.top]))

    if not (args.kind or args.grep):
        print("\nclassified:", ", ".join(
            f"{k}={v:,}" for k, v in
            Counter(kinds.classify(r.get("types") or [])
                    for r in rows.values()).most_common()))


if __name__ == "__main__":
    main()
