"""
Render the legend band that runs along the bottom of the ancestrydots poster.

Names the largest ancestries in the dataset, grouped under the 14 top-level
categories, alongside a title block.

The grid is auto-fitted: column count and type size are chosen together so the
rows fill the band's height and each entry is only as wide as its longest label
actually needs (measured from the font, not guessed). Whatever width is left
over goes to the title block rather than being spread as dead gutter, which is
what the earlier fixed 5-column layout did.

    python legend.py                 # names the 100 largest
    python legend.py --named 150

Writes build/legend_band.png at the sheet width from layout.json.
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from fontTools.ttLib import TTFont

from typeface import FAMILY, REGULAR, BOLD

HERE = Path(__file__).parent
BUILD = HERE / "build"

INK, DIM, FAINT = "#e8e6e1", "#9aa2ac", "#5d646d"
BG = "#0d0d0f"

GROUP_NAMES = {
    "western": "Western European", "eastern": "Eastern European",
    "american": "North American", "native": "Indigenous",
    "african": "African", "afro_carib": "West Indian",
    "latino": "Latin American", "mena": "Middle Eastern & North African",
    "s_c_asian": "South & Central Asian", "se_asian": "Southeast Asian",
    "east_asian": "East Asian", "pacific": "Pacific Islander",
    "other": "Other", "no_ancestry": "No ancestry reported",
}
GROUP_ORDER = ["western", "eastern", "american", "native", "african",
               "afro_carib", "latino", "mena", "s_c_asian", "se_asian",
               "east_asian", "pacific", "other", "no_ancestry"]

# The two census sources are public domain / an open licence that permits
# sale, but the insets' coastal water is the OSM water-polygons extract, which
# is ODbL. A printed sheet is an ODbL "Produced Work", so share-alike does not
# reach it — attribution does, and it is a condition rather than a courtesy.
# (The main map's ocean and all the lakes are Natural Earth, public domain, so
# they need no line.)
CITATION = ("US Census Bureau, American Community Survey 2020-2024, "
            "table B04006  ·  Statistics Canada, 2021 Census of Population"
            "  ·  Inset water © OpenStreetMap contributors, ODbL")


def human(n):
    return f"{n / 1e6:.1f}M" if n >= 1_000_000 else f"{n / 1e3:.0f}k"


class Metrics:
    """Real advance widths, so column sizing is measured rather than assumed."""

    def __init__(self, path):
        f = TTFont(path)
        self.upm = f["head"].unitsPerEm
        self.hmtx = f["hmtx"]
        self.cmap = f.getBestCmap()

    def width(self, s, fs):
        total = sum(self.hmtx[self.cmap[ord(c)]][0]
                    for c in s if ord(c) in self.cmap)
        return total / self.upm * fs / 72


def flow(lines, rows):
    """Break the line list into columns of at most `rows`, never leaving a
    group header stranded at the foot of a column with fewer than two of its
    entries under it."""
    cols, cur = [], []
    for ln in lines:
        if len(cur) >= rows or (ln[0] == "header" and len(cur) >= rows - 2):
            cols.append(cur)
            cur = []
        cur.append(ln)
    if cur:
        cols.append(cur)
    return cols


def fit(lines, entries, grid_w, usable_h, met, max_fs):
    """Pick rows-per-column and type size to fill the band without overflowing
    the width. Fewer rows means taller lines and bigger type but more columns,
    so there is a sweet spot. Column count comes from actually flowing the
    lines, not from dividing — the no-stranded-header rule can add a column."""
    widest = max(met.width(e["label"], 100) for e in entries) / 100
    widest_n = max(met.width(human(e["population"]), 100) for e in entries) / 100
    best = None
    for rows in range(8, 31):
        cols = flow(lines, rows)
        line_h = usable_h / rows
        fs = min(line_h * 72 * 0.60, max_fs)
        entry_w = 0.10 + widest * fs + 0.14 + widest_n * fs
        gutter = 0.26
        total = len(cols) * entry_w + (len(cols) - 1) * gutter
        if total <= grid_w and (best is None or fs > best[2]):
            best = (cols, rows, fs, entry_w, gutter, total)
    if best is None:
        raise SystemExit("legend does not fit — widen the band or lower --named")
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--named", type=int, default=100,
                    help="how many of the largest ancestries to name")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--max-fs", type=float, default=10.0,
                    help="cap on legend type. Too low a cap forces the fitter "
                         "into many narrow columns and leaves the band short")
    ap.add_argument("--title-min", type=float, default=10.5,
                    help="width reserved for the title block. The fitter "
                         "maximises legend type, so without a floor here it "
                         "widens the grid until the title has nowhere to go")
    ap.add_argument("--body-fs", type=float, default=10.0,
                    help="title-block body type")
    ap.add_argument("--safe-bottom", type=float, default=0.45,
                    help="keep the lowest text this far off the sheet edge. "
                         "0.45 leaves 0.20 in of margin after a 0.25 in trim")
    # 9.5 in rather than 8: at 8 the body ran one line longer and collided with
    # the citation once that was lifted clear of the trim edge.
    ap.add_argument("--measure-in", type=float, default=9.5,
                    help="line length for the body paragraphs. The title block "
                         "is much wider than a readable measure, so wrap to "
                         "this rather than to the block")
    ap.add_argument("--opaque", dest="transparent", action="store_false")
    args = ap.parse_args()

    lay = json.loads((BUILD / "layout.json").read_text(encoding="utf-8"))
    W, H = lay["sheet_in"][0], lay["band_in"]
    km_in = lay["km_per_inch"]

    entries = json.loads((BUILD / "palette.json").read_text(encoding="utf-8"))
    named = sorted(entries, key=lambda e: -e["population"])[:args.named]
    floor = named[-1]["population"]
    keep = {e["label"] for e in named}
    by_group = {g: sorted((e for e in entries
                           if e["group"] == g and e["label"] in keep),
                          key=lambda e: -e["population"])
                for g in GROUP_ORDER}

    lines = []
    for g in GROUP_ORDER:
        if by_group[g]:
            lines.append(("header", GROUP_NAMES[g], None))
            lines += [("entry", e["label"], e) for e in by_group[g]]

    met = Metrics(str(REGULAR))
    top, bottom, margin = 0.30, 0.16, 0.34
    usable_h = H - top - bottom
    cols, rows, fs, entry_w, gutter, grid_total = fit(
        lines, named, W - 2 * margin - args.title_min, usable_h, met,
        args.max_fs)

    # the grid sits right; everything left of it belongs to the title block
    grid_x = W - margin - grid_total
    title_w = grid_x - margin - 0.5
    print(f"{len(named)} of {len(entries)} named (floor {floor:,}); "
          f"{len(cols)} cols x {rows} rows at {fs:.1f} pt, "
          f"grid {grid_total:.1f} in, title block {title_w:.1f} in")

    fig = plt.figure(figsize=(W, H), dpi=args.dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_axis_off()
    fig.patch.set_alpha(0)
    if not args.transparent:
        ax.add_patch(Rectangle((0, 0), W, H, color=BG, zorder=0))

    # --- title block ------------------------------------------------------
    x = margin
    y = H - 0.42
    ax.text(x, y, "Ancestry Dots in North America", color=INK, fontsize=26,
            fontweight="bold", va="center", family=FAMILY)
    y -= 0.52
    ax.text(x, y, "1 dot = 100 reported ancestries", color=INK, fontsize=16,
            va="center", family=FAMILY)

    body = [
        # NB: the ACS release used is the 2020-2024 five-year, not 2021 — the
        # citation below says so, and the two must not contradict each other.
        "Data is self-reported ancestries from the 2020-2024 American Community "
        "Survey and the 2021 Census of Canada. Surveyed people can report "
        "multiple or zero ancestries, so dot count does not directly correspond "
        "to number of people.",
        # Checked against scatter_dots.py: the residuals are ACS B02009 and
        # B02008 minus BLACK_SUBTRACT_GROUPS {african, afro_carib} and
        # WHITE_SUBTRACT_GROUPS {western, eastern, american} respectively.
        "White (n.a.) and Black (n.a.) are estimates of people who choose not "
        "to report an ancestry. Black, no reported ancestry is calculated per "
        "census tract as the Black (race) population minus all the ancestries "
        "in the African and West Indian groups. White, no reported ancestry is "
        "analogously the White (race) population minus the Western European, "
        "Eastern European and North American groups.",
        f"The legend shows the largest {len(named)} ancestries in the dataset, "
        "but smaller ones are also mapped.",
        "Interactive version is at anita.garden/ancestrydotsna",
    ]
    y -= 0.26
    measure = min(args.measure_in, title_w)
    wrap_at = max(int(measure / met.width("n", args.body_fs)), 40)
    for para in body:
        for line in textwrap.wrap(para, wrap_at):
            y -= 0.195
            ax.text(x, y, line, color=DIM, fontsize=args.body_fs, va="center",
                    family=FAMILY)
        y -= 0.075

    # The citation used to be pinned at 0.30 and 0.14 in from the sheet edge,
    # which a 0.25 in trim would cut straight through. Sit it --safe-bottom
    # clear of the edge instead, and shout if the body has grown down into it.
    cite_y = args.safe_bottom + 0.16
    if y < cite_y + 0.10:
        print(f"  !! body text reaches {y:.2f} in, citation starts at "
              f"{cite_y:.2f} in — widen --measure-in or shrink --body-fs")
    # The citation is one unwrapped line, so it can only grow sideways. Measure
    # it rather than trusting that a source added later still fits.
    cite_w = met.width(CITATION, 7.5)
    if cite_w > title_w:
        print(f"  !! citation is {cite_w:.2f} in against a {title_w:.2f} in "
              f"title block — shorten CITATION or widen the block")
    ax.text(x, cite_y, CITATION, color=FAINT, fontsize=7.5, va="center",
            family=FAMILY)
    ax.text(x, args.safe_bottom,
            f"Albers equal-area conic, {km_in:.0f} km per inch",
            color=FAINT, fontsize=7.5, va="center", family=FAMILY)

    # --- legend grid ------------------------------------------------------
    line_h = usable_h / rows
    for ci, col in enumerate(cols):
        cx = grid_x + ci * (entry_w + gutter)
        y = H - top
        for kind, label, e in col:
            draw_line(ax, kind, label, e, cx, y, entry_w, fs, line_h)
            y -= line_h

    out = BUILD / "legend_band.png"
    fig.savefig(out, dpi=args.dpi,
                **({"transparent": True} if args.transparent
                   else {"facecolor": BG}))
    plt.close(fig)
    print(f"wrote {out}  ({int(W * args.dpi)}x{int(H * args.dpi)} px)")


def draw_line(ax, kind, label, e, cx, y, entry_w, fs, line_h):
    if kind == "header":
        ax.text(cx, y, label, color=INK, fontsize=fs + 0.8,
                fontweight="bold", va="center", family=FAMILY)
        ax.plot([cx, cx + entry_w], [y - line_h * 0.38, y - line_h * 0.38],
                color=FAINT, lw=0.5)
    else:
        ax.add_patch(Circle((cx + 0.045, y), 0.036, color=e["color"], ec="none"))
        ax.text(cx + 0.13, y, label, color=DIM, fontsize=fs, va="center",
                family=FAMILY)
        ax.text(cx + entry_w, y, human(e["population"]), color=FAINT,
                fontsize=fs, va="center", ha="right", family=FAMILY)


if __name__ == "__main__":
    main()
