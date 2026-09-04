"""Measure the OVERVIEW palette — the one the all-religions view actually draws (spec §6.9).

`check_palette.py` measures the thirty authored family hues against each other. That is the
depth-1 question. This is the depth-2 one, and it is where the collisions now are: a family's
band is subdivided among its branches, so what a reader compares is Catholic against Baptist
inside 68° of yellow, not Christianity against Islam.

What it checks, per country and for the all-countries view:

  1. every drawn category's contrast against the near-black map background;
  2. every PAIR of drawn categories in Lab, so a pair 3° apart in hue but two tiers apart in
     lightness passes and a pair 3° apart in both does not.

The drawn set is computed the way the viewer's `drawnSet` computes it — presence-pruned,
split branches pushed ahead of their children (§6.6), small rows folded (§6.10) — so what is
measured is the legend as it ships. The colours come out of index.html itself via
`palette_dump.js`, so this cannot drift from the palette.

Run: python tools/check_overview.py       (needs node on PATH)

  --near N     Lab dE below which two 2px dots read as one colour (default 25)
  --big N      ignore a pair unless both sides have at least N dots there (default 20)
  --depth N    the depth cut to measure (default 2, the viewer's default)
  --focus ID   measure a focus view instead: the full-wheel palette under that scope
  --all        list every drawn colour, not only the problems
"""
import argparse
import collections
import itertools
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from check_palette import BG, as_hex, delta_e, luminance, rgb   # noqa: E402

HERE = Path(__file__).parent.parent
sys.path.insert(0, str(HERE / "taxonomy"))
from branches import LINEAGE                                    # noqa: E402

FOLD_SHARE = 1e-4          # keep in step with index.html §6.10
FOLD_MIN = 2
HIDDEN_ROOTS = ("unaffiliated",)


def load_palette():
    """index.html's own tables, via node. Not a reimplementation -- see the docstring."""
    with tempfile.TemporaryDirectory() as td:
        lin = Path(td) / "lin.json"
        out = Path(td) / "pal.json"
        lin.write_text(json.dumps(
            {p: [{"label": lab, "ids": ids} for lab, ids in gs]
             for p, gs in LINEAGE.items()}), encoding="utf-8")
        r = subprocess.run(
            ["node", str(HERE / "tools" / "palette_dump.js"), str(lin), str(out)],
            capture_output=True, text=True)
        if r.returncode:
            sys.exit("palette_dump.js failed:\n" + r.stderr)
        return json.loads(out.read_text(encoding="utf-8"))


def hsl_of(css):
    """'hsl(50.0,92%,64%)' -> (50.0, 92, 64), which is what the Lab helpers take."""
    body = css[css.index("(") + 1:css.rindex(")")]
    h, s, lightness = (p.strip().rstrip("%") for p in body.split(","))
    return float(h), float(s), float(lightness)


class Tree:
    """The viewer's `rebuild` / `buildTree` / `drawnSet`, in enough detail to measure."""

    def __init__(self):
        raw = json.loads((HERE / "taxonomy" / "religions.json").read_text(encoding="utf-8"))
        self.nodes = [n["id"] for n in raw["nodes"]]
        self.label = {n["id"]: n["label"] for n in raw["nodes"]}
        self.rank = {}
        for parent, groups in LINEAGE.items():
            i = 0
            for _, ids in groups:
                for nid in ids:
                    self.rank[nid] = i
                    i += 1

    def build(self, dots):
        self.own = collections.Counter(dots)
        self.counts = collections.Counter(dots)
        for nid in sorted(self.nodes, key=len, reverse=True):
            p = nid.rpartition(".")[0]
            if p:
                self.counts[p] += self.counts[nid]

        present = set()
        for nid, v in dots.items():
            if not v:
                continue
            while nid:
                present.add(nid)
                nid = nid.rpartition(".")[0]

        self.kids = collections.defaultdict(list)
        self.roots = []
        for nid in self.nodes:
            if nid not in present:
                continue
            p = nid.rpartition(".")[0]
            (self.kids[p] if p in present else self.roots).append(nid)
        for p, ks in self.kids.items():
            if p in LINEAGE:
                ks.sort(key=lambda k: (self.rank.get(k, 10 ** 9), -self.counts[k], k))
            else:
                ks.sort(key=lambda k: (-self.counts[k], k))
        self.roots.sort(key=lambda k: (-self.counts[k], k))

        self.folded = {}
        for p, ks in self.kids.items():
            if len(ks) < FOLD_MIN or not self.counts[p]:
                continue
            small = [k for k in ks if self.counts[k] < self.counts[p] * FOLD_SHARE]
            if len(small) >= FOLD_MIN:
                self.folded[p] = set(small)

    def vis(self, nid):
        return [k for k in self.kids[nid] if k not in self.folded.get(nid, ())]

    def drawn(self, scope, depth):
        out = []

        def walk(nid, d):
            ks = self.vis(nid)
            if d >= depth or not ks:
                out.append(nid)
                return
            if self.own[nid]:
                out.append(nid)
            for k in ks:
                walk(k, d + 1)

        if scope and self.own[scope] and self.kids[scope]:
            out.append(scope)
        for t in (self.vis(scope) if scope else self.roots):
            walk(t, 1)
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--near", type=float, default=25.0)
    ap.add_argument("--within", type=float, default=12.0,
                    help="the bar for two categories in the SAME family, which "
                         "the overview deliberately draws close (default 12)")
    ap.add_argument("--big", type=int, default=20)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--focus", default=None)
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    pal = load_palette()
    table = pal["NODE_COLOR"] if args.focus else pal["OVERVIEW_COLOR"]
    which = f"focus on {args.focus}" if args.focus else "all religions"
    print(f"{len(table)} node colours read out of index.html — {which}, "
          f"depth {args.depth}")

    counts = json.loads(
        (HERE / "data" / "processed" / "counts.json").read_text(encoding="utf-8"))
    tree = Tree()
    bgl = luminance((BG, BG, BG))
    problems = 0

    for cc in list(counts["countries"]) + ["all countries"]:
        dots = (counts["dots"] if cc == "all countries"
                else counts["countries"][cc]["dots"])
        dots = {k: v for k, v in dots.items()
                if not any(k == h or k.startswith(h + ".") for h in HIDDEN_ROOTS)}
        tree.build(dots)
        if args.focus and not tree.counts.get(args.focus):
            continue
        drawn = tree.drawn(args.focus, args.depth)

        # a drawn node's own area of colour is its subtree MINUS what its drawn
        # descendants painted over (§6.6) -- that is the size the pair test wants
        size = {}
        for nid in drawn:
            inner = sum(tree.counts[o] for o in drawn
                        if o != nid and o.startswith(nid + "."))
            size[nid] = tree.counts[nid] - inner
        cols = {nid: hsl_of(table[nid]) for nid in drawn if nid in table}

        print(f"\n{'=' * 70}\n{cc}: {len(drawn)} drawn categories, "
              f"{sum(size.values()):,} dots")
        if args.all:
            for nid in sorted(drawn, key=lambda x: -size[x]):
                shown = tuple(round(c) for c in cols[nid])
                print(f"   {size[nid]:>8,}  {as_hex(cols[nid])}  hsl{shown}  "
                      f"{tree.label[nid]}")
        dim = [f"{tree.label[n]} ({r:.1f})" for n, r in
               (((n, (luminance(rgb(cols[n])) + 0.05) / (bgl + 0.05))
                 for n in cols if size[n] >= args.big)) if r < 2.9]
        if dim:
            print("   dim against the background: " + ", ".join(dim))
        # two different questions, and one bar cannot ask both -- see `show`
        pairs = sorted((delta_e(cols[a], cols[b]), a, b)
                       for a, b in itertools.combinations(cols, 2)
                       if min(size[a], size[b]) >= args.big)
        cross = [p for p in pairs if p[1].split(".")[0] != p[2].split(".")[0]
                 and p[0] < args.near]
        within = [p for p in pairs if p[1].split(".")[0] == p[2].split(".")[0]
                  and p[0] < args.within]

        def show(rows, head):
            print(f"   {head}")
            for d, a, b in rows:
                print(f"     dE {d:5.1f}   {tree.label[a][:26]:<26}{size[a]:>8,}"
                      f"  /  {tree.label[b][:26]:<26}{size[b]:>8,}")

        if cross:
            show(cross, f"ACROSS FAMILIES, under dE {args.near:.0f} "
                        "-- these are failures: the first thing a dot has to say is "
                        "which family it is")
        else:
            print(f"   across families: every pair over {args.big} dots is at least "
                  f"dE {args.near:.0f} apart")
        if within:
            show(within, f"inside one family, under dE {args.within:.0f} -- the "
                         "overview means these to be close (§6); only a pair that is "
                         "BIG on both sides is worth a PIN")
        problems += len(cross)

    print(f"\n{problems} close pair(s) over the size threshold.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
