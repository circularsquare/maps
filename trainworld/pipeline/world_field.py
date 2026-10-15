"""Reader for the world field (T-375; built by world_field_build.py; design notes/T-362.md).

    wf = world_field.load()
    cells = wf.around(lon, lat)       # r5 within 300 km, r4 to 1,000 km, r3 beyond: every person once
    wf.locate(lon, lat)               # (country index, landmass) of the place
    wf.countries[i]                   # {"iso2", "iso3", "name", "gdp", "gdp_src", "wb_pop", ...}

Cells are dicts of arrays: lon, lat, pop, gdp (people-weighted GDP per head at PPP, constant 2021
international $), country (index of the most populous country in the cell), landmass (same: two
places may be driven between only on one landmass), res. Landmass 0 is Afro-Eurasia."""
import json
from pathlib import Path
import numpy as np
import h3
import h3.api.basic_int as h3i

WORK = Path(__file__).resolve().parents[1] / "data/work"
R_KM = 6371.0088
NEAR_KM, MID_KM = 300.0, 1000.0


def _unit(lon, lat):
    lo, la = np.radians(lon), np.radians(lat)
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], axis=-1)


def km(lon0, lat0, lon, lat):
    """Great-circle km from one place to arrays of places."""
    d = np.linalg.norm(_unit(lon, lat) - _unit(np.float64(lon0), np.float64(lat0)), axis=-1)
    return 2 * R_KM * np.arcsin(np.clip(d / 2, 0, 1))


def _dominant(group, lab, w, n):
    """Per group, the label carrying the most weight."""
    off = lab.min()
    span = int(lab.max() - off) + 1
    key = group.astype(np.int64) * span + (lab - off)
    uk, ki = np.unique(key, return_inverse=True)
    tot = np.bincount(ki, weights=w)
    g, v = uk // span, uk % span + off
    o = np.lexsort((-tot, g))
    first = np.ones(len(o), bool)
    first[1:] = g[o][1:] != g[o][:-1]
    out = np.empty(n, lab.dtype)
    out[g[o][first]] = v[o][first]
    return out


class Level:
    """One resolution's cells, sorted so each coarser cell's children are contiguous."""
    def __init__(self, res, h, lon, lat, pop, gdp, country, landmass):
        self.res, self.h, self.lon, self.lat, self.pop = res, h, lon, lat, pop
        self.gdp, self.country, self.landmass = gdp, country, landmass
        self.xyz = _unit(lon, lat)
        self.start = self.end = None  # child ranges in the next finer level

    def take(self, idx):
        return {"lon": self.lon[idx], "lat": self.lat[idx], "pop": self.pop[idx], "gdp": self.gdp[idx],
                "country": self.country[idx], "landmass": self.landmass[idx],
                "res": np.full(len(idx), self.res, np.int8)}


def _coarsen(fine, res):
    par = np.array([h3i.cell_to_parent(int(c), res) for c in fine.h], np.uint64)
    order = np.argsort(par, kind="stable")
    for a in ("h", "lon", "lat", "pop", "gdp", "country", "landmass"):
        setattr(fine, a, getattr(fine, a)[order])
    fine.xyz = fine.xyz[order]
    par = par[order]
    u, start, inv = np.unique(par, return_index=True, return_inverse=True)
    w = fine.pop.astype(np.float64)
    pop = np.bincount(inv, weights=w)
    xyz = np.stack([np.bincount(inv, weights=w * fine.xyz[:, k]) for k in range(3)], axis=1)
    lon = np.degrees(np.arctan2(xyz[:, 1], xyz[:, 0]))
    lat = np.degrees(np.arctan2(xyz[:, 2], np.hypot(xyz[:, 0], xyz[:, 1])))
    gdp = np.bincount(inv, weights=w * fine.gdp) / pop
    lv = Level(res, u, lon, lat, pop, gdp, _dominant(inv, fine.country, w, len(u)),
               _dominant(inv, fine.landmass, w, len(u)))
    lv.start, lv.end = start, np.append(start[1:], len(par))
    return lv


class WorldField:
    def __init__(self, path=WORK / "world_field.npz"):
        z = np.load(path)
        self.countries = json.load(open(WORK / "world_field.countries.json", encoding="utf-8"))["countries"]
        gdp_c = np.array([c["gdp"] for c in self.countries] + [np.nan])
        country = z["country"].astype(np.int64)
        self.r5 = Level(5, z["h3"], z["lon"].astype(np.float64), z["lat"].astype(np.float64),
                        z["pop"].astype(np.float64), gdp_c[country], country, z["landmass"].astype(np.int64))
        gmed = np.nanmedian(gdp_c)
        self.r5.gdp = np.where(np.isnan(self.r5.gdp), gmed, self.r5.gdp)  # the few cells with no country
        self.r4 = _coarsen(self.r5, 4)
        self.r3 = _coarsen(self.r4, 3)
        self._index = {int(c): i for i, c in enumerate(self.r5.h)}

    def around(self, lon, lat, near_km=NEAR_KM, mid_km=MID_KM):
        """The field as one site sees it; every person counted once, at r5, r4 or r3 by distance."""
        d3 = km(lon, lat, self.r3.lon, self.r3.lat)
        far = np.nonzero(d3 > mid_km)[0]
        i4 = _children(self.r3, np.nonzero(d3 <= mid_km)[0])
        d4 = km(lon, lat, self.r4.lon[i4], self.r4.lat[i4])
        mid = i4[d4 > near_km]
        i5 = _children(self.r4, i4[d4 <= near_km])
        parts = [self.r5.take(i5), self.r4.take(mid), self.r3.take(far)]
        return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}

    def locate(self, lon, lat):
        """(country index, landmass) at a place: its r5 cell, else the nearest populated one."""
        i = self._index.get(h3.str_to_int(h3.latlng_to_cell(lat, lon, 5)))
        if i is None:
            i = int(np.argmin(km(lon, lat, self.r5.lon, self.r5.lat)))
        return int(self.r5.country[i]), int(self.r5.landmass[i])


def _children(level, idx):
    if len(idx) == 0:
        return np.zeros(0, np.int64)
    s, e = level.start[idx], level.end[idx]
    n = e - s
    return np.repeat(s - np.cumsum(n) + n, n) + np.arange(n.sum())


_cache = None


def load():
    global _cache
    if _cache is None:
        _cache = WorldField()
    return _cache
