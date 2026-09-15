"""Helpers that two or more countries/<cc>.py files use, and the imports all of them had.

countries.py loads this before any country file, and every country file starts with
`from countries._shared import *`, so each name here is visible there as it was when all of
this was one file. A helper that only one country uses belongs in that country's file.
"""
import json
from pathlib import Path
import re
import sys

import pandas as pd

# The religiondots folder, the same path countries.py calls HERE; not this file's own folder.
HERE = Path(__file__).parent.parent


def _column_of(note, key="parent_column"):
    """The source COLUMN a derived row was split out of, off the normalised file's note.

    spec §7a-i-1. `allocate.py` writes `parent_column=<the fine table's own column>` as the
    LAST field of the note and `br_rescale.py` writes `group22=<the 2022 category>`; both
    name a category the source measured AT THE DRAWN UNIT, which is what the roll-up needs
    and what an ancestor walk up the religion tree cannot give it.

    THE `home` STRIP IS A SCAR, not a feature. Between 2026-09-03 and 2026-09-07 allocate.py
    emitted a pandas Series repr here instead of the column name, so `in` and `hu` carry
    `parent_column=home    Other religions and persuasions\\nhome    ...\\nName: 6, dtype:
    object`. The bug is fixed; the two files still have to be regenerated, and until they
    are this reads them correctly rather than silently dropping two countries.
    """
    if not isinstance(note, str):
        return None
    i = note.find(key + "=")
    if i < 0:
        return None
    # Stop at a `;` as well as a newline: allocate.py puts `parent_column` LAST so nothing
    # follows it, but br_rescale.py writes `group22=...; structure_share=...` and taking the
    # rest of the line there silently matched no category at all.
    v = note[i + len(key) + 1:].split("\n")[0].split(";")[0].strip()
    v = re.sub(r"^home\s+", "", v)
    return v or None


def _add_roll(df, columns, key="parent_column"):
    """Attach the `roll` column: the node of each derived row's own source column."""
    col = df["note"].map(lambda n: _column_of(n, key))
    df["roll"] = col.map(lambda c: columns.get(c) if c is not None else None)
    # A measured row is already at a level somebody counted and must never be moved.
    df.loc[df["tier"] != "derived", "roll"] = None
    return df


def _allocated_counts(cc, fine, module):
    """The shared shape for a source that arrived through allocate.py.

    au / ie / mx all split fine categories from fine geography (spec §3.9), so the file
    countries.py reads is `<cc>_<fine>_allocated.csv` rather than `<cc>.csv`: it carries
    every category at the fine geography, tagged `measured` where a fine column had a
    single child and `derived` where a coarse total was spread out.

    §3.10: a derived count may never become a ring. Allocation spreads a total and cannot
    establish that anyone is present, so `may_ring` is exactly `tier == "measured"`.
    """
    df = pd.read_csv(HERE / "data" / "normalized" / f"{cc}_{fine}_allocated.csv",
                     dtype={"geo_id": str}, low_memory=False)
    df["node"] = df["source_category"].map(module.resolve)
    df = df[df["node"].notna()]
    # An allocated zero is an absence, and an absence must not become a presence ring
    # (§4.3) — the lesson Czechia's 417,083 explicit zeros taught.
    df = df[df["count"] > 0]
    df["congregations"] = 0
    df["may_ring"] = df["tier"] == "measured"
    # §7a-i-1: where the module names its fine columns, a derived row remembers which one it
    # came out of. Without it the roll-up has only the tree to walk, and the tree does not
    # know that Ireland's Anglicans were counted in a cell called `Other religion`.
    _add_roll(df, getattr(module, "COLUMNS", {}))
    return df.rename(columns={"geo_id": "unit"})[
        ["unit", "node", "count", "congregations", "may_ring", "tier", "roll"]]


class _KeHexWeighter:
    """Split a county's dots across Kontur 400 m hexagons by hex POPULATION.

    Kenya is the case that makes §8.2 load-bearing rather than tidy. The counts are 47
    counties for 47.2M people — the coarsest counting geography on this map — and the
    counties are wildly uneven in habitability: Turkana is 68,680 km² with 926,976 people,
    Marsabit 70,961 km² with 459,785. An equal share per polygon would wash the northern
    half of the country in evenly spaced dots over empty desert, and since Wajir, Mandera
    and Garissa are each 97-99% Muslim that wash would be one colour and the loudest thing
    on the map.

    So the weight is the hexagon's own modelled population (sources/ke_grid.py). It is a
    POPULATION weight, not a religion one: nothing measures where Kenya's Catholics sit
    inside a county, so a Catholic dot and a Muslim dot are spread identically. Read the
    map as "religion by county, drawn where Kenyans live", never as county-internal detail.
    """

    def __init__(self, place):
        self.pop = place["pop"].to_numpy(dtype=float)
        self.n_pop = 0
        self.n_uniform = 0

    def weights(self, node, idx, count, plain=False):
        pop = self.pop[idx]
        if pop.sum() > 0:
            self.n_pop += 1
            return pop
        self.n_uniform += 1
        return None

    def summary(self):
        return (f"{self.n_pop:,} (unit, node) rows placed on Kontur hex population, "
                f"{self.n_uniform:,} on equal shares where a county's hexes sum to zero "
                f"(sources/ke_grid.py)")


class _KonturHexWeighter:
    """Split a unit's dots across its Kontur hexes by hex POPULATION.

    Serbia and Lithuania are Kenya's problem at a tenth and a twentieth of the scale, and
    the same answer applies to all three: the counting units are historical districts —
    Serbian opštine average 461 km², Lithuanian savivaldybės 1,088 km² — rather than units
    an agency engineered to a population target, so §8.2's equal share over a fine layer
    has nothing here to be equal over. The weight is a measured population surface
    (Kontur H3 r8; sources/rs_geo.py, sources/lt_geo.py).

    It is a POPULATION weight and not a religion one: neither country measures where a
    given church's members live inside a municipality, so a Catholic dot and an Orthodox
    dot in Subotica are spread the same way. Read a cluster as "this municipality, drawn
    where people actually live", never as a neighbourhood reading.

    One class rather than one per country, because there is nothing per-country in it. The
    Russian and Kenyan weighters above predate it and are the same code; folding them in is
    a tidy-up nobody has needed yet.
    """

    def __init__(self, place, built_by):
        self.pop = place["pop"].to_numpy(dtype=float)
        self.built_by = built_by
        self.n_pop = 0
        self.n_uniform = 0

    def weights(self, node, idx, count, plain=False):
        pop = self.pop[idx]
        if pop.sum() > 0:
            self.n_pop += 1
            return pop
        self.n_uniform += 1
        return None

    def summary(self):
        return (f"{self.n_pop:,} (unit, node) rows placed on Kontur hex population, "
                f"{self.n_uniform:,} on equal shares where a unit's hexes sum to zero "
                f"({self.built_by})")


def _kontur_place_weight(place, grid, builder):
    if "pop" not in place.columns:
        print(f"  !! {grid} has no `pop` column — run {builder}; "
              "placing on equal shares (§8.2)")
        return None
    return _KonturHexWeighter(place, builder)


def _micro_place_weight(cc, builder="sources/micro.py"):
    """countries.py hook factory for the microstate tier. `place` is the Kontur hex layer.

    `builder` names the module that writes the hex layer, for the error message. It is
    `sources/micro.py` for the nine the tier started with and the country's own module for
    anyone who joins the tier from a different instrument, which so far is Nauru.
    """
    def hook(place):
        return _kontur_place_weight(place, f"{cc}_hexes.gpkg", builder)
    return hook


def _micro_counts(cc, module):
    """A national religion partition on Kontur hexes: the microstate tier.

    NINE OF THE TEN ARE UNSD DYB TABLE 28 (`sources/micro.py`). The tenth is Nauru, which is
    not in the Yearbook at all and comes from the Nauru Bureau of Statistics' own census
    workbook (`sources/nr.py`) — same shape, same one unit, nineteen categories instead of
    the Yearbook's four to twenty-three. Everything below applies to both.

    ONE UNIT, WHICH IS THE WHOLE COUNTRY, and that is the point rather than a shortcoming.
    Anita, 2026-09-08: *"for the really small island countries we might not even need any
    divisions. like for instance if we do palau, it has 17000 people so itll just be 17
    dots."* At 1 dot = 1,000 people the placement inside a country of 20,000 asserts nothing,
    so a national-only table is a complete source and not a coarse one. That is what retires
    spec §3.9b's unit-count floor and §3.9c's variety floor for these nine.

    THERE IS NO JOIN. `unit` is the country code, the Kontur layer carries the same code on
    every hex, and §12's first two shapes of failure cannot arise anywhere in this tier.

    EVERY ROW IS `measured` AND MAY RING. Nothing is allocated, spread or modelled: the
    Yearbook publishes these counts at this geography and the map draws exactly them. Rings
    matter more here than usual, because at 1:1,000 most of these categories are under one
    dot and a ring is the only thing that says a religion is present at all (§4.3).
    """
    import importlib

    resolve = importlib.import_module(module).resolve
    # keep_default_na=False: UNSD prints no religion as `None`, which pandas' defaults read as
    # NaN, so Bermuda's 11,466, Niue's 59 and Tuvalu's 26 were dropped until 2026-09-14
    # (tools/check_na_readers.py).
    df = pd.read_csv(HERE / "data" / "normalized" / f"{cc}.csv",
                     dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    df = df[df["geo_level"] == "country"].copy()
    if df["geo_id"].nunique() != 1:
        raise SystemExit(f"{cc}: {df['geo_id'].nunique()} units, expected exactly 1")
    df["unit"] = df["geo_id"].astype(str)
    df["node"] = df["source_category"].map(resolve)
    df = df[df["node"].notna() & (df["count"] > 0)]
    if df.empty:
        raise SystemExit(f"{cc}: nothing resolved — check taxonomy/{module}.py")
    df["congregations"] = 0
    # Several categories share a node (Bermuda's two residuals, Palau's two Protestant rows).
    return (df.groupby(["unit", "node"], as_index=False)
              .agg(count=("count", "sum"), congregations=("congregations", "sum")))


def _terr_counts(cc, module, n_units, fold=None, tier=None):
    """The small-territory batch of 2026-09-14 (sources/terr.py, sources/bq.py, terr.md).

    `_micro_counts` for territories whose table has more than one unit, or whose labels
    pandas would misread. Units come off `geo_id`, with `fold` merging a census unit into
    the one it is drawn with (the three British Virgin Islands too small to carry a Kontur
    hex). `keep_default_na=False` because St Kitts and Nevis prints a category called `None`.
    `tier` is `modelled` for the Caribbean Netherlands, whose counts are survey shares.
    """
    import importlib

    resolve = importlib.import_module(module).resolve
    df = pd.read_csv(HERE / "data" / "normalized" / f"{cc}.csv", dtype={"geo_id": str},
                     low_memory=False, keep_default_na=False, na_values=[""])
    df["unit"] = df["geo_id"].astype(str).replace(fold or {})
    if df["unit"].nunique() != n_units:
        raise SystemExit(f"{cc}: {df['unit'].nunique()} units, expected {n_units}")
    df["node"] = df["source_category"].map(resolve)
    df = df[df["node"].notna() & (df["count"] > 0)].copy()
    if df.empty:
        raise SystemExit(f"{cc}: nothing resolved; check taxonomy/{module}.py")
    df["congregations"] = 0
    out = (df.groupby(["unit", "node"], as_index=False)
             .agg(count=("count", "sum"), congregations=("congregations", "sum")))
    if tier:
        out["tier"] = tier
    return out


class _GrLauWeighter:
    """Split a NUTS 2 region's dots across its LAUs, by LAU population.

    Greece's counting geography is NUTS 2 — thirteen regions and Mount Athos — and the
    placement layer is 6,137 LAUs, so this is doing most of the work of making the map look
    like a country rather than like thirteen blobs. It is a population weight and not a
    religion one: nothing in Greece measures religion below the region, so a Muslim dot in
    Attiki sits where Attiki's people are rather than where its mosques are.

    **The one place that costs something visible is Thrace.** The minority is concentrated in
    Rodopi and Xanthi, which are two of Anatoliki Makedonia-Thraki's five regional units, and
    this spreads it evenly across all five — so Kavala and Drama draw Muslim dots they should
    not, and Rodopi draws fewer than it should. The foreign half is available at NUTS 3 and
    could be placed that way; the minority is not, and mixing the two would put the sharper
    geography on the half that has the weaker claim to it.
    """

    def __init__(self, place):
        self.pop = place["pop"].to_numpy(dtype=float)
        self.n_pop = 0
        self.n_uniform = 0

    def weights(self, node, idx, count, plain=False):
        pop = self.pop[idx]
        if pop.sum() > 0:
            self.n_pop += 1
            return pop
        self.n_uniform += 1
        return None

    def summary(self):
        return (f"{self.n_pop:,} (unit, node) rows placed on LAU population, "
                f"{self.n_uniform:,} on equal shares")


class _ItWeighter:
    """Place each half of Italy by ITS OWN population, not by the province's total.

    Anita, on the finished map: *"it does look a bit weird to see hinduism spread like
    population-proportionally throughout the italian countryside when in reality i imagine
    it's much more urbanized."* Correct, and the cause is not a missing urban/rural religion
    statistic — Italy's Hindus are 166,000 people and **essentially all of them are in the
    foreign half**, which was being scattered across each province in proportion to where
    *Italians* live. ESS's `domicil` was checked as the obvious fix and is useless for it:
    among citizens it gives Islam no urban gradient at all (1.10% village against 1.10% big
    city) and "Eastern religions" a 4.4x one on about twenty respondents.

    So `sources/it_geo.py` puts ISTAT's comune-level citizenship counts on the placement
    layer and this splits the two:

        citizen-half people  ->  `ital`     53.6M Italian citizens per comune
        foreign-half people  ->  `foreign`   5.4M foreign citizens per comune

    **A node is usually both**, which is why this cannot be a column swap. `islam.sunni`
    draws 491,000 citizens and 1.65M foreign residents, so its weight in a province is the
    two vectors blended in that province's own proportion — computed per (unit, node) from
    the same two CSVs `_it_counts` reads, so the blend cannot drift from the counts.

    **Nothing about a magnitude changes.** Every province's totals are still Eurostat's;
    only the within-province scatter moves, which is what §8.2 asks for.

    Its limit, stated because the map still invites the opposite reading: this places
    *foreigners* well, not *Hindus* specifically. RCS is comune x individual citizenship and
    would put Indians in the Agro Pontino and Chinese in Prato by name, but ISTAT's own
    country-code list is a dead link (see `it_geo.py`), and at 107 province the composition
    already carries most of that — Prato and Latina are each their own provincia.
    """

    def __init__(self, place, fshare, citizen_col="ital", foreign_col="foreign",
                 citizen_label="Italian", place_label="comune"):
        self.pop = place["pop"].to_numpy(dtype=float)
        self.ital = place[citizen_col].to_numpy(dtype=float)
        self.foreign = place[foreign_col].to_numpy(dtype=float)
        self.unit = place["unit"].to_numpy()
        self.fshare = fshare
        self.citizen_label = citizen_label
        self.place_label = place_label
        self.n_split = 0
        self.n_one = 0
        self.n_pop = 0
        self.n_uniform = 0

    def weights(self, node, idx, count, plain=False):
        import numpy as np

        fs = self.fshare.get((self.unit[idx[0]], node))
        if fs is None:
            # A node with no row in either CSV for this unit should not reach here, but a
            # future edition could add one; fall back to the old behaviour rather than
            # dropping it.
            pop = self.pop[idx]
            if pop.sum() > 0:
                self.n_pop += 1
                return pop
            self.n_uniform += 1
            return None

        i, f, p = self.ital[idx], self.foreign[idx], self.pop[idx]
        w = np.zeros(len(idx), dtype=float)
        for share, vec in ((1.0 - fs, i), (fs, f)):
            if share <= 0:
                continue
            s = vec.sum()
            # 16 comuni have no RCS row (they merged after the 2021 boundary vintage) and a
            # handful of provinces could in principle have no foreign residents at all;
            # either way that half falls back to total population rather than vanishing.
            base = vec / s if s > 0 else (p / p.sum() if p.sum() > 0 else None)
            if base is None:
                continue
            w += share * base
        if w.sum() <= 0:
            if p.sum() > 0:
                self.n_pop += 1
                return p
            self.n_uniform += 1
            return None
        if 0.0 < fs < 1.0:
            self.n_split += 1
        else:
            self.n_one += 1
        return w

    def summary(self):
        return (f"{self.n_split:,} (unit, node) rows placed on a blend of "
                f"{self.citizen_label} and foreign {self.place_label} population, "
                f"{self.n_one:,} on one of the two alone, {self.n_pop:,} on total "
                f"population, {self.n_uniform:,} on equal shares")


def _foreign_share(cc, resolve, level):
    """{(unit, node): foreign people / all people} — the blend `_ItWeighter` applies.

    Read from the same two files the country's `_counts` reads and summed the same way, so a
    change to either half moves the counts and the placement together. Italy, Spain and
    France all have exactly this shape: a citizen half whose categories need resolving
    through a taxonomy module, and a foreign half already carrying `node`.
    """
    cit = pd.read_csv(HERE / "data" / "normalized" / f"{cc}.csv", dtype={"geo_id": str},
                      keep_default_na=False, na_values=[""])
    cit = cit[cit["geo_level"] == level].copy()
    cit["node"] = cit["source_category"].map(resolve)
    ext = pd.read_csv(HERE / "data" / "normalized" / f"{cc}_foreign.csv",
                      dtype={"geo_id": str})
    ext = ext[ext["geo_level"] == level]

    c = cit.groupby(["geo_id", "node"])["count"].sum()
    f = ext.groupby(["geo_id", "node"])["count"].sum()
    both = pd.concat([c.rename("cit"), f.rename("for")], axis=1).fillna(0.0)
    tot = both["cit"] + both["for"]
    share = (both["for"] / tot.where(tot > 0)).fillna(0.0)
    return {k: float(v) for k, v in share.items()}


# Every name above, underscored ones included, for the country files' star import.
__all__ = [_name for _name in globals() if not _name.startswith("__")]
