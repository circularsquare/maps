"""
The parameters that decide what the sheet looks like, in one place.

render.py and nudge/prepare.py both need them and they must agree: the editor
bakes control points from these numbers and the sheet draws from those control
points, so a stripe that is one width in the editor and another on the sheet is
an editor that lies. Keeping the argparse defaults here rather than duplicated
in two files removes that as a possibility.
"""

SHEET = "24x36"          # inches
PAD_KM = 0.6
DPI = 300

# Stripe width is linear in riders, from MIN_WIDTH at zero to MAX_WIDTH at
# WIDTH_REF. The maximum is double the first pass (60->120 mil) because the
# outer branches were disappearing at whole-sheet scale. The minimum was 12 mil
# (10:1, the web map's ratio) until 2026-09-30, when it went down to one printed
# pixel so that width reads as closer to proportional: at 12 mil a 10k line
# drew at a fifth of the busiest, not a fourteenth.
#
# It costs something: a 120-mil stripe is 125 m wide on the GROUND at a
# kilometre to the inch, so the four-track bundles are now 300-400 m across and
# the tightest curves have less room than ever to turn a bundle through. See
# smooth_nodes in ribbons.py, and the hand-edit tool for the rest.
MIN_WIDTH = 1 / 300      # inches; one pixel at 300 dpi
MAX_WIDTH = 0.120        # ...drawn at WIDTH_REF riders per day
# Width is pinned to a fixed rider count rather than to whatever the busiest
# stripe happens to be. It used to be the busiest, and when a fold bug that had
# inflated the 6/7/F was fixed (2026-09-30) the busiest fell from 140k to 111k,
# which would have widened every other stripe by 25% and moved every bundle
# under the hand nudges. 140k is the old maximum, so nothing else moved.
WIDTH_REF = 140_000
WIDTH_GAMMA = 1.0        # <1 lifts the quiet branches, at the cost of honesty
GAP = 0.12               # between stripes, as a fraction of MAX_WIDTH

CENTRE_REG = 0.06        # bundle-centre solve: lower holds one offset for longer
SMOOTH_PASSES = 160      # corner rounding on the shared centrelines
SMOOTH_CAP = 0.75        # ...capped per node at this fraction of its bundle

BUBBLE_MIN = 0.00936     # station bubble radius, inches (1.8x then 1.3x the first pass)
BUBBLE_MAX = 0.2691
