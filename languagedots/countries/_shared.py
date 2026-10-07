"""Helpers every countries/<cc>.py may use. `from _shared import *` at the top of an entry file.

Do not put a country's own logic here: a change to this file changes every country, and several
agents work on countries at once.
"""
import sys
from pathlib import Path

import pandas as pd  # noqa: F401  (re-exported for entry files)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "taxonomy"))
from rdlink import RD, RD_GEO  # noqa: E402,F401

NORM = ROOT / "data" / "normalized"
GEO = ROOT / "data" / "geo"          # languagedots' own geography, for units religiondots lacks


class PopWeighter:
    """Dots inside a unit go where the placement layer's own `pop` says people live.

    Every language's dots in a unit are spread the same way: nothing published says where
    inside a unit its speakers of one language live, so this is a population weight only.
    """

    def __init__(self, place):
        self.pop = place["pop"].to_numpy(dtype=float)
        self.n_pop = self.n_uniform = 0

    def weights(self, node, idx, count, plain=False):
        p = self.pop[idx]
        if p.sum() > 0:
            self.n_pop += 1
            return p
        self.n_uniform += 1
        return None

    def summary(self):
        return (f"{self.n_pop:,} (unit, language) rows placed on the layer's population, "
                f"{self.n_uniform:,} on equal shares where a unit's population sums to zero")


def pop_weight(place):
    """countries entry hook: weight by the placement layer's `pop` column, if it has one."""
    if "pop" not in place.columns:
        print("  !! placement layer has no `pop` column; equal shares")
        return None
    return PopWeighter(place)


def by_unit(df):
    """The shape counts() must return: one row per (unit, node), summed."""
    return df.groupby(["unit", "node"], as_index=False)["count"].sum()
