# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _ca_counts():
    """StatCan 2021 at CSD, allocated to 147 categories, mapped at branch level (§2.4)."""
    import ca2021
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
    _add_roll(df, ca2021.COLUMNS)
    return df.rename(columns={"geo_id": "unit"})[
        ["unit", "node", "count", "congregations", "may_ring", "tier", "roll"]]


ENTRY = {
    "ca": dict(
        name="Canada",
        source="Census of Population 2021 (Statistics Canada)",
        basis="self-identification, 25% long-form sample",
        view=[-128.0, 42.0, -55.0, 58.0],
        note_public=(
            "The census asks the person, so this is what people say they are rather than who "
            "is on a roll — and self-description is always the larger number. Categories "
            "below the province level are derived: StatCan publishes 168 religions by "
            "province and 25 by subdivision, never both, so the fine ones are split out "
            "proportionally and can show composition but not presence. 241 subdivisions "
            "publish religion built on ≥50% long-form non-response."),
        how="census, 2021, 25% sample",
        fill="from the same census at province level",
        grain="census subdivisions, 7,000 people on average",
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
