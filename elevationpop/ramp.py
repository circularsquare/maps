"""The population colour ramp. Hand-edited: change the stops, re-run tiles.py.

Values are people per 100 m cell (a density, see prep.py), read on a log
scale.  Anything at or below LO is drawn fully transparent so the empty
countryside shows the dark base and the hillshade above it, and the bottom
FADE of the ramp fades in rather than starting at full opacity.
"""

import numpy as np

LO = 0.25  # people per cell at the bottom of the ramp
HI = 1000.0  # and at the top
FADE = 0.22  # fraction of the ramp over which alpha climbs 0 -> 1

# position along the log ramp -> colour
STOPS = [
    (0.00, "#1b0c30"),
    (0.18, "#4a1361"),
    (0.36, "#8a1f6b"),
    (0.54, "#c53a60"),
    (0.70, "#ec6f3f"),
    (0.84, "#f9a72b"),
    (0.93, "#fdd453"),
    (1.00, "#fff8d4"),
]

_LOG_LO = np.log10(LO)
_LOG_HI = np.log10(HI)


def _table(n=1024):
    """Expand STOPS into an n-entry lookup table of uint8 RGB."""
    pos = np.array([p for p, _ in STOPS], dtype="float64")
    cols = np.array(
        [[int(c[i : i + 2], 16) for i in (1, 3, 5)] for _, c in STOPS], dtype="float64"
    )
    t = np.linspace(0.0, 1.0, n)
    return np.stack([np.interp(t, pos, cols[:, k]) for k in range(3)], axis=1).round().astype("uint8")


TABLE = _table()


def colorize(values):
    """float array of people-per-cell -> (h, w, 4) uint8 RGBA."""
    v = np.asarray(values, dtype="float32")
    t = (np.log10(np.clip(v, LO, HI)) - _LOG_LO) / (_LOG_HI - _LOG_LO)

    idx = np.clip((t * (len(TABLE) - 1)).astype("int32"), 0, len(TABLE) - 1)
    rgba = np.empty(v.shape + (4,), dtype="uint8")
    rgba[..., :3] = TABLE[idx]

    alpha = np.clip(t / FADE, 0.0, 1.0)
    alpha[v <= LO] = 0.0
    rgba[..., 3] = (alpha * 255.0).round().astype("uint8")
    return rgba


def legend_css():
    """CSS gradient stops and tick positions for the legend in index.html."""
    stops = []
    for p, c in STOPS:
        r, g, b = (int(c[i : i + 2], 16) for i in (1, 3, 5))
        stops.append(f"rgba({r},{g},{b},{min(p / FADE, 1.0):.2f}) {p * 100:.0f}%")
    ticks = [
        (v, (np.log10(v) - _LOG_LO) / (_LOG_HI - _LOG_LO)) for v in (1, 10, 100, 1000)
    ]
    return ", ".join(stops), ticks


if __name__ == "__main__":
    grad, ticks = legend_css()
    print("linear-gradient(to right, " + grad + ")")
    print()
    for v, pos in ticks:
        print(f"  {v:>5}  left: {pos * 100:.1f}%")
