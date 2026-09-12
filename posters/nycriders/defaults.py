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

# Stripe width is linear in riders / busiest segment. 10:1 from thinnest to
# thickest, which is the ratio the interactive map uses; the absolute size is
# double the first pass (6->12 and 60->120 mil) because the outer branches were
# disappearing at whole-sheet scale.
#
# It costs something: a 120-mil stripe is 125 m wide on the GROUND at a
# kilometre to the inch, so the four-track bundles are now 300-400 m across and
# the tightest curves have less room than ever to turn a bundle through. See
# smooth_nodes in ribbons.py, and the hand-edit tool for the rest.
MIN_WIDTH = 0.012        # inches
MAX_WIDTH = 0.120
WIDTH_GAMMA = 1.0        # <1 lifts the quiet branches, at the cost of honesty
GAP = 0.12               # between stripes, as a fraction of MAX_WIDTH

CENTRE_REG = 0.06        # bundle-centre solve: lower holds one offset for longer
SMOOTH_PASSES = 160      # corner rounding on the shared centrelines
SMOOTH_CAP = 0.75        # ...capped per node at this fraction of its bundle

BUBBLE_MIN = 0.004       # station bubble radius, inches
BUBBLE_MAX = 0.115
