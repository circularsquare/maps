"""Write a starting colors.csv: one colour per group, spread so groups that live together differ.

colors.csv is the palette and is meant to be edited by hand, so this REFUSES to overwrite it.
Pass --out somewhere else to generate a comparison.

How the starting colours are chosen. There are 58 columns and nowhere near 58 colours a
reader can tell apart, so what matters is which pairs must differ: two groups that share
counties. For every pair, `overlap` is the share of the smaller group living in rows where
the other group also lives (sum over rows of the smaller of the two counts, divided by the
smaller group's total). Groups are then coloured largest first, each taking the candidate
colour that is furthest, in OKLab, from the groups it overlaps most. Groups that never meet
can end up similar, which is the trade.

Han is pinned to a muted grey-blue. It is 90% of the people drawn and would otherwise decide
the whole map's colour; the two residual columns are pinned to neutrals.

Usage:
    python palette.py
    python palette.py --out colors_try.csv
"""
import argparse
import csv
import math
import os
import sys

import numpy as np
import pandas as pd

from common import GROUPS, HERE, KEYS, WORK

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

LEAVES = os.path.join(WORK, "leaves_2020.csv")
PINNED = {"han": "#7d8796", "undetermined": "#a39d8c", "naturalised": "#b3b3b3"}


def oklch_to_srgb(L, C, H):
    a, b = C * math.cos(math.radians(H)), C * math.sin(math.radians(H))
    l_ = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3
    m_ = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3
    s_ = (L - 0.0894841775 * a - 1.2914855480 * b) ** 3
    r = 4.0767416621 * l_ - 3.3077115913 * m_ + 0.2309699292 * s_
    g = -1.2684380046 * l_ + 2.6097574011 * m_ - 0.3413193965 * s_
    bl = -0.0041960863 * l_ - 0.7034186147 * m_ + 1.7076147010 * s_

    def enc(x):
        return 12.92 * x if x <= 0.0031308 else 1.055 * x ** (1 / 2.4) - 0.055
    rgb = [enc(v) for v in (r, g, bl)]
    if any(v < -0.001 or v > 1.001 for v in rgb):
        return None
    return tuple(min(1, max(0, v)) for v in rgb)


def hex_to_oklab(hx):
    r, g, b = (int(hx[i:i + 2], 16) / 255 for i in (1, 3, 5))

    def dec(x):
        return x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4
    r, g, b = dec(r), dec(g), dec(b)
    l_ = (0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b) ** (1 / 3)
    m_ = (0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b) ** (1 / 3)
    s_ = (0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b) ** (1 / 3)
    return np.array([0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
                     1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
                     0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_])


def to_hex(rgb):
    return "#" + "".join(f"{round(v * 255):02x}" for v in rgb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "colors.csv"))
    args = ap.parse_args()
    if os.path.exists(args.out) and os.path.abspath(args.out) == os.path.join(HERE, "colors.csv"):
        raise SystemExit(f"{args.out} exists and is the hand-edited palette; "
                         f"use --out to write a comparison instead")

    df = pd.read_csv(LEAVES)
    X = df[KEYS].to_numpy(dtype=np.float64)
    tot = X.sum(axis=0)
    ng = len(KEYS)
    overlap = np.zeros((ng, ng))
    for i in range(ng):
        for j in range(i + 1, ng):
            small = min(tot[i], tot[j])
            if small > 0:
                overlap[i, j] = overlap[j, i] = np.minimum(X[:, i], X[:, j]).sum() / small

    cands = []
    for L in (0.66, 0.74, 0.82):
        for C in (0.11, 0.16):
            for H in range(0, 360, 8):
                rgb = oklch_to_srgb(L, C, H)
                if rgb:
                    cands.append(to_hex(rgb))
    cand_lab = np.array([hex_to_oklab(c) for c in cands])
    print(f"{len(cands)} candidate colours in gamut")

    assigned = dict(PINNED)
    lab = {k: hex_to_oklab(v) for k, v in assigned.items()}
    taken = set()
    sigma = 0.07
    for i in np.argsort(-tot):
        key = KEYS[i]
        if key in assigned:
            continue
        score = np.zeros(len(cands))
        for other, olab in lab.items():
            j = KEYS.index(other)
            dist = np.linalg.norm(cand_lab - olab, axis=1)
            weight = overlap[i, j] + 0.03          # a little pressure from every group
            score += weight * np.exp(-dist / sigma)
        for t in taken:
            score[t] = np.inf
        best = int(np.argmin(score))
        taken.add(best)
        assigned[key] = cands[best]
        lab[key] = cand_lab[best]

    with open(args.out, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["key", "en", "cn", "color", "people_drawn"])
        for (k, en, cn), t in sorted(zip(GROUPS, tot), key=lambda r: -r[1]):
            w.writerow([k, en, cn, assigned[k], int(t)])
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
