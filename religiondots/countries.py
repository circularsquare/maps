"""
Per-country wiring for the scatter step: where the counts are, where the polygons are, and
how a source category becomes a religiondots node.

Everything downstream of this file is country-agnostic. Everything above it is per-country by
necessity — the sources do not agree about anything, and spec §3.9, §8.1 and §2.3 are three
different ways of saying so.

Each entry supplies:
  counts()      -> DataFrame [unit, node, count]   unit = the geography the counts are ON
  units         path to the polygons for `unit`, and the column holding its id
  place         path to a finer polygon layer used to place dots inside a unit, and the
                column linking it back to `unit`. spec §8.2: these units are designed to a
                population target, so an equal share of dots per unit is already a population
                weighting and no population data is read.
"""
from pathlib import Path
import sys

import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "taxonomy"))


def _us_counts():
    """ASARB 2020: county x 372 bodies, already mapped to the taxonomy by hand."""
    paths = pd.read_csv(HERE / "taxonomy" / "usrc_groups.csv", dtype={"Group Code": str})
    path_of = dict(zip(paths["Group Code"], paths["path"]))
    df = pd.read_excel(HERE / "data" / "raw" / "2020_USRC_Group_Detail.xlsx",
                       sheet_name="2020 Group by County",
                       dtype={"FIPS": str, "Group Code": str})
    df["unit"] = df["FIPS"].str.strip().str.zfill(5)
    df["node"] = df["Group Code"].str.strip().str.zfill(3).map(path_of)
    df = df[df["node"].notna() & (df["node"] != "UNMAPPED")]
    return df.rename(columns={"Adherents": "count", "Congregations": "congregations"})[
        ["unit", "node", "count", "congregations"]]


def _ca_counts():
    """StatCan 2021 at CSD, allocated to 147 categories, mapped at branch level (§2.4)."""
    from ca2021 import resolve

    src = pd.read_csv(HERE / "data" / "normalized" / "ca.csv",
                      dtype={"geo_id": str}, low_memory=False)
    src["parent"] = src["note"].str.extract(r"parent=([^;]*)")
    prov = src[src.geo_level == "province"]
    parent_of = (prov.dropna(subset=["parent"]).drop_duplicates("source_category")
                 .set_index("source_category")["parent"].to_dict())
    parent_of = {k: (v if isinstance(v, str) and v else None) for k, v in parent_of.items()}

    df = pd.read_csv(HERE / "data" / "normalized" / "ca_csd_allocated.csv",
                     dtype={"geo_id": str}, low_memory=False)
    df["node"] = df["source_category"].map(lambda c: resolve(c, parent_of))
    df = df[df["node"].notna()]
    df["congregations"] = 0
    # spec §3.10: an allocated count may never become a ring, because a ring asserts presence
    # and allocation only spreads a total. `tier` is `measured` where a fine column had a
    # single child (nothing was allocated) and `derived` otherwise.
    df["may_ring"] = df["tier"] == "measured"
    return df.rename(columns={"geo_id": "unit"})[
        ["unit", "node", "count", "congregations", "may_ring"]]


COUNTRIES = {
    "us": dict(
        counts=_us_counts,
        units=None,              # counts are on counties; tracts carry both ids
        unit_key=None,
        place=HERE / "data" / "geo" / "tracts2020" / "cb_2020_us_tract_500k.shp",
        place_unit=lambda g: g["STATEFP"] + g["COUNTYFP"],
        note="ASARB is a roll (spec §3.1); adherents are attributed to the congregation's "
             "county, not the member's (§3.6).",
    ),
    "ca": dict(
        counts=_ca_counts,
        # StatCan's DA boundary file carries only DAUID / PRUID — no CSD link, and a DAUID
        # (province + census division + DA) does not contain one. Rather than fetch the
        # Geographic Attribute File for the lookup, derive it spatially: dissemination areas
        # nest exactly inside census subdivisions by construction, so a representative-point
        # join is not an approximation. `sjoin` also generalises to any country whose fine
        # layer omits the id of the unit the counts are on.
        units=HERE / "data" / "geo" / "ca" / "csd" / "lcsd000b21a_e.shp",
        unit_key="DGUID",
        place=HERE / "data" / "geo" / "ca" / "da" / "lda_000b21a_e.shp",
        place_unit="sjoin",
        note="StatCan is self_id from a 25% long-form sample; not comparable with the US "
             "roll across the border (spec §3.1).",
    ),
}
