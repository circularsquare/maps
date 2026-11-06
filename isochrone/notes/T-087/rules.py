"""T-087: 1 km speed grids under named road rules, for compare.mjs.

    python isochrone/notes/T-087/rules.py <tiles dir> <rule> [<rule> ...]

Writes <tiles dir>/<tile>/l1_<rule>.bin (uint8 km/h x 40, 0 = water) for every tile with a feat.npz.
Rules (build_tiles.py's blend: ratio = 1 - e + e x min(1, off-road / road speed), bridges at road speed):
  none            no roads
  share_A         e = min(1, A s)                      (the page: share_3)
  sat_K           e = 1 - (1 - s)^K
  pow_A_P         e = min(1, A s^P)
  span_A_B        e = min(1, A s + B span/10)          span: longest run of road cells across the cell
  thru_A_B        e = min(1, A s + B through)          through: a road touches opposite cell edges
  cross_A_B       e = min(1, A s + B cross/4)
  kind_A_B_C      e = min(1, A s_path + B s_local + C s_main)   (a cell with two kinds counts in both)
  spthru_A_B_C    e = min(1, A s + B through + C span/10)
  max_A_B         e = min(1, max(A s, B through))
  str_A_B_C       e = min(1, (A + B max(0, q - C)) s), q = span / (100 s), the cell's road straightness
  strs_A_B_C      the same with q from span and count averaged over the 3 x 3 cells round it
  det, detd, pay  road walked slower by a detour D / dearer-road weighting (poor, see notes/T-087.md)
"""
import os
import sys

import numpy as np

q = None


def quant(kmh):
    return np.where(kmh > 0, np.clip(np.rint(kmh * 40), 1, 255), 0).astype(np.uint8)


def effective(rule, f):
    p = rule.split("_")
    k, a = p[0], [float(x) for x in p[1:]]
    s = f["s"]
    if k == "none":
        return np.zeros_like(s)
    if k == "share":
        return np.minimum(1.0, a[0] * s)
    if k == "sat":
        return 1 - (1 - s) ** a[0]
    if k == "pow":
        return np.minimum(1.0, a[0] * s ** a[1])
    if k == "span":
        return np.minimum(1.0, a[0] * s + a[1] * f["span"] / 10)
    if k == "thru":
        return np.minimum(1.0, a[0] * s + a[1] * f["through"])
    if k == "cross":
        return np.minimum(1.0, a[0] * s + a[1] * f["cross"] / 4)
    if k == "kind":
        return np.minimum(1.0, a[0] * f["s1"] + a[1] * f["s2"] + a[2] * f["s4"])
    if k == "spthru":
        return np.minimum(1.0, a[0] * s + a[1] * f["through"] + a[2] * f["span"] / 10)
    if k == "max":
        return np.minimum(1.0, np.maximum(a[0] * s, a[1] * f["through"]))
    if k in ("str", "strs"):            # e = min(1, (A + B max(0, straight - C)) s); strs: straightness over 3 x 3 cells
        from scipy.ndimage import uniform_filter
        span, cnt = f["span"], 100 * s
        if k == "strs":
            span, cnt = uniform_filter(span, 3, mode="nearest"), uniform_filter(cnt, 3, mode="nearest")
        st = np.where(cnt > 0, span / np.maximum(cnt, 1), 0)
        return np.minimum(1.0, (a[0] + a[1] * np.maximum(0, st - a[2])) * s)
    if k == "det":                      # e = min(1, A s), the road walked at road speed / D (detour)
        return np.minimum(1.0, a[0] * s)
    if k == "pay":                      # A0 + A1 (1 - off-road / road speed), the more the road gains the more it is used
        pay = 1 - np.minimum(1.0, f["walk"] / np.maximum(f["road_kmh"], 1e-3))
        return np.minimum(1.0, (a[0] + a[1] * pay) * s)
    if k == "detd":                     # detour falls with density: D = 1 + C max(0, 0.25 - s)/0.25
        return np.minimum(1.0, a[0] * s)
    raise SystemExit("unknown rule " + rule)


def detour(rule, f):
    p = rule.split("_")
    if p[0] == "det":
        return float(p[2])
    if p[0] == "detd":
        return 1 + float(p[2]) * np.maximum(0, 0.25 - f["s"]) / 0.25
    return 1.0


def speed(rule, f):
    walk, rk = f["walk"], f["road_kmh"]
    land = walk > 0
    e = effective(rule, f).astype(np.float32)
    ratio = 1 - e + e * np.minimum(1.0, detour(rule, f) * walk / np.maximum(rk, 1e-3))
    v = np.where(land, walk / np.where(land, ratio, 1), 0)
    if rule != "none":
        v = np.where(~land & (f["s"] > 0) & f["bridge"], np.maximum(rk, 0.05), v)
    return v


def main(d, rules):
    for name in sorted(os.listdir(d)):
        p = os.path.join(d, name, "feat.npz")
        if not os.path.exists(p):
            continue
        f = dict(np.load(p))
        for r in rules:
            quant(speed(r, f)).tofile(os.path.join(d, name, "l1_%s.bin" % r))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
