"""CLEAR Global's sub-unit language shares as a placement weight inside a census unit.

Shared by sources/sn_place.py and sources/ml_place.py (2026-10-07, fix-place), on the pattern
of countries/gn.py: the census gives counts per unit (région); CLEAR Global's HDX files
(`<country>-languages`, CC BY-SA, proportions of the main household language from an IPUMS
census sample) give each language's share in smaller areas (départements, cercles). Inside a
unit, a language's dots go to each hex in proportion to

    the hex's population  x  CLEAR's share of that language in the hex's zone

so a language goes where CLEAR finds it, and every unit's counts stay the census's. A placement
weight only (AGENT_BRIEF §4.4).

Each hex of a country's place layer carries a `zone`: the CLEAR location code whose shares it
takes, or "" for a hex in an area CLEAR does not cover. A "" hex takes the population-weighted
mean share of its unit's zoned hexes (it neither attracts nor repels any language), and a unit
with no zoned hex, or one zone only, comes out exactly on population.

    read_clear(path)                 a CLEAR CSV, checked (shares sum to 1 per location)
    nearest_join(hexes, polys, col)  each hex's polygon code by centroid, nearest for misses
    ClearWeighter(place, shares, node_codes)   the scatter's weighter
"""
import numpy as np
import pandas as pd


def read_clear(path):
    d = pd.read_csv(path)
    need = {"location_code", "location_name", "language_code", "language_name",
            "proportion_value"}
    if not need <= set(d.columns):
        raise SystemExit(f"{path}: not a CLEAR language-use CSV ({list(d.columns)})")
    tot = d.groupby("location_code")["proportion_value"].sum()
    if (tot - 1).abs().max() > 1e-3:
        raise SystemExit(f"{path}: shares do not sum to 1: {tot[(tot - 1).abs() > 1e-3]}")
    return d


def nearest_join(hexes, polys, col, allowed=None):
    """Code `col` of the polygon each hex's centroid falls in. A centroid in none (offshore, a
    sliver), or in one not in allowed[hex unit], goes to the nearest allowed polygon. Returns
    (Series of codes, number moved)."""
    import geopandas as gpd
    polys = polys[[col, "geometry"]].to_crs(hexes.crs)
    pts = hexes[["unit"]].copy()
    pts = gpd.GeoDataFrame(pts, geometry=hexes.to_crs(3857).centroid.to_crs(hexes.crs))
    j = gpd.sjoin(pts, polys, how="left", predicate="within")
    j = j[~j.index.duplicated(keep="first")]
    code = j[col].reindex(hexes.index)
    unit = hexes["unit"].astype(str)
    bad = [i for i in hexes.index if not isinstance(code[i], str)
           or (allowed is not None and code[i] not in allowed[unit[i]])]
    if bad:
        p = pts.loc[bad].to_crs(3857)
        a = polys.to_crs(3857)
        for i, g in p.geometry.items():
            cand = a if allowed is None else a[a[col].isin(allowed[unit[i]])]
            code[i] = cand.loc[cand.distance(g).idxmin(), col]
    return code.astype(str), len(bad)


class ClearWeighter:
    """See the module docstring. `shares`: DataFrame zone, clear_code, share. `node_codes`:
    {node: [CLEAR language codes]}; a node not in it (remainders, Bayot...) goes on population."""

    def __init__(self, place, shares, node_codes, label="CLEAR"):
        self.label = label
        self.pop = place["pop"].to_numpy(dtype=float)
        unit = place["unit"].astype(str).to_numpy()
        zone = place["zone"].fillna("").astype(str).to_numpy()
        S = shares.pivot_table(index="zone", columns="clear_code", values="share",
                               aggfunc="sum").fillna(0.0)
        unknown = sorted(set(zone) - set(S.index) - {""})
        if unknown:
            raise SystemExit(f"{label}: place layer zones with no shares: {unknown[:8]}")
        missing = sorted({c for cs in node_codes.values() for c in cs} - set(S.columns))
        if missing:
            raise SystemExit(f"{label}: codes not in the shares: {missing}")
        zoned = zone != ""
        self.w = {}
        for node, codes in node_codes.items():
            s = pd.Series(zone).map(S[list(codes)].sum(axis=1)).to_numpy(dtype=float)
            # a hex outside CLEAR's zones takes its unit's population-weighted mean share
            df = pd.DataFrame({"u": unit, "ps": np.where(zoned, self.pop * np.nan_to_num(s), 0.0),
                               "p": np.where(zoned, self.pop, 0.0)})
            g = df.groupby("u")[["ps", "p"]].sum()
            mean = (g["ps"] / g["p"].where(g["p"] > 0)).reindex(unit).to_numpy()
            s = np.where(zoned, s, mean)
            s = np.where(np.isnan(s), 1.0, s)  # a unit with no zoned hex: population
            self.w[node] = self.pop * s
        self.n = {"clear": 0, "pop": 0, "none": 0}

    def weights(self, node, idx, count, plain=False):
        w = self.w.get(node)
        if w is not None and w[idx].sum() > 0:
            self.n["clear"] += 1
            return w[idx]
        p = self.pop[idx]
        if p.sum() > 0:
            self.n["pop"] += 1
            return p
        self.n["none"] += 1
        return None

    def summary(self):
        return (f"{self.n['clear']:,} (unit, language) rows placed by {self.label}'s shares, "
                f"{self.n['pop']:,} on population, {self.n['none']:,} on equal shares")
