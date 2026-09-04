"""Measure the authored family palette (spec §6.3).

Two things a table of hues cannot tell you by eye, and both of which have already caught a
mistake in this palette:

  1. is every colour bright enough to read as a 2px dot on the near-black map;
  2. is every PAIR of families that actually co-occurs in a built country far enough apart
     in CIE Lab that two adjacent dots can be told apart.

The second is the one that matters. 21 of the 30 roots share one indigo→magenta wedge, so
separation is not a property of the table — it is a property of the table *against a
country's tallies*, and it changes every time a country lands. A pair 4° apart in hue is
fine if the two groups never appear in the same country and useless if they are Brazil's
two largest non-Christian traditions.

ROOT_HSL is read out of index.html rather than copied, so this can never drift from the
palette that ships.  Run: python tools/check_palette.py

  --near N   Lab dE below which two 2px dots read as one colour (default 25)
  --big N    ignore pairs where either side has fewer than N dots in that country
             (default 20) -- spec §6 accepts that distant tiny leaves share a shade,
             because they never appear in the same view
"""
import argparse
import colorsys
import itertools
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent.parent
VIEWER = HERE / "index.html"
COUNTS = HERE / "data" / "processed" / "counts.json"
TREE = HERE / "taxonomy" / "religions.json"
BG = 0x11 / 255.0          # body background, and openfreemap's dark style is no lighter

# Roots whose colour is MEANT to sit under the contrast floor. spec §6.3a: on a dark map
# lightness is prominence, and `unaffiliated` is the largest node anywhere — being quiet is
# the whole point of its colour, so reporting it as too faint would invite undoing the
# decision. Nothing else belongs here without the same kind of reason.
# `unrecorded` joins it for the same reason and a stronger one: it is 51.8% of Germany,
# and a mass that size drawn bright would bury the only signal the German map carries.
DIM_ON_PURPOSE = {"unaffiliated", "unrecorded"}


def read_palette():
    """ROOT_HSL out of index.html. Deliberately not a copy -- see the module docstring."""
    src = VIEWER.read_text(encoding="utf-8")
    m = re.search(r"const ROOT_HSL = \{(.*?)\n\};", src, re.S)
    if not m:
        sys.exit("could not find `const ROOT_HSL = {` in index.html")
    pal = {}
    for k, h, s, l in re.findall(
            r"^\s*([A-Za-z]\w*)\s*:\s*\[\s*(-?\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]",
            m.group(1), re.M):
        pal[k] = (int(h), int(s), int(l))
    return pal


def rgb(hsl):
    h, s, l = hsl
    return colorsys.hls_to_rgb(h / 360.0, l / 100.0, s / 100.0)


def _lin(c):
    return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4


def luminance(t):
    r, g, b = (_lin(c) for c in t)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def lab(t):
    r, g, b = (_lin(c) for c in t)
    x = (0.4124 * r + 0.3576 * g + 0.1805 * b) / 0.95047
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    z = (0.0193 * r + 0.1192 * g + 0.9505 * b) / 1.08883

    def f(v):
        return v ** (1 / 3.0) if v > 0.008856 else 7.787 * v + 16 / 116.0

    fx, fy, fz = f(x), f(y), f(z)
    return 116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)


def delta_e(a, b):
    """CIE76. Crude next to CIEDE2000 and quite good enough to catch a 4-degree collision."""
    return math.sqrt(sum((p - q) ** 2 for p, q in zip(lab(rgb(a)), lab(rgb(b)))))


def as_hex(hsl):
    return "#%02x%02x%02x" % tuple(int(round(c * 255)) for c in rgb(hsl))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--near", type=float, default=25.0)
    ap.add_argument("--big", type=int, default=20)
    ap.add_argument("--all", action="store_true",
                    help="list every colour, not just the problems")
    args = ap.parse_args()

    pal = read_palette()
    print(f"{len(pal)} authored root colours from index.html\n")

    roots = [n["id"] for n in json.loads(TREE.read_text(encoding="utf-8"))["nodes"]
             if "." not in n["id"]]
    missing = [r for r in roots if r not in pal]
    extra = [k for k in pal if k not in roots]
    if missing:
        print("ROOTS WITH NO AUTHORED COLOUR (they fall back to a wheel position that will "
              "look fine and mean nothing):")
        for r in missing:
            print("   ", r)
    if extra:
        print("authored but not a root in religions.json:", ", ".join(extra))

    bgl = luminance((BG, BG, BG))
    rows = sorted(pal.items(), key=lambda kv: luminance(rgb(kv[1])))
    dim = [(k, (luminance(rgb(v)) + 0.05) / (bgl + 0.05)) for k, v in rows]
    if args.all:
        print("\n-- contrast against the map background --")
        for k, v in rows:
            r = (luminance(rgb(v)) + 0.05) / (bgl + 0.05)
            print(f"  {r:5.2f}  {k:<24} {as_hex(v)}  hsl{v}")
    else:
        faint = [f"{k} ({r:.1f})" for k, r in dim if r < 2.9 and k not in DIM_ON_PURPOSE]
        print("\ndim against the map background:", ", ".join(faint) or "none")
        held = [f"{k} ({r:.1f})" for k, r in dim if k in DIM_ON_PURPOSE]
        if held:
            print("dim on purpose, not a finding (spec §6.3a):", ", ".join(held))

    counts = json.loads(COUNTS.read_text(encoding="utf-8"))
    per_country = counts["countries"]
    problems = 0
    for cc in list(per_country) + ["all countries"]:
        dots = counts["dots"] if cc == "all countries" else per_country[cc]["dots"]
        size = Counter()
        for nid, v in dots.items():
            size[nid.split(".")[0]] += v
        here = [r for r in size if r in pal and size[r]]
        close = sorted((delta_e(pal[a], pal[b]), a, b)
                       for a, b in itertools.combinations(here, 2)
                       if delta_e(pal[a], pal[b]) < args.near
                       and min(size[a], size[b]) >= args.big)
        print(f"\n{cc}: {len(here)} families with dots")
        for d, a, b in close:
            print(f"   dE {d:5.1f}   {a:<22}{size[a]:>8,}  /  {b:<22}{size[b]:>8,}")
        if not close:
            print(f"   every pair over {args.big} dots is at least dE {args.near:.0f} apart")
        problems += len(close)

    print(f"\n{problems} close pair(s) over the size threshold."
          " spec 6.3 lists which of these are accepted and why.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
