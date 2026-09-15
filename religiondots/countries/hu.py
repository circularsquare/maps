# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _hu_counts():
    """Népszámlálás 2022 at settlement: 28 categories on 3,177 units.

    ONE level, and only the allocated file is read. hu.csv carries settlement, county and
    country — the same 9.6 million people counted three times — and
    `hu_settlement_allocated.csv` already holds every settlement column untouched plus the
    three that WBS008 refines, so reading hu.csv as well would double the country.

    98.1% of it is MEASURED. The allocation only touches three of the eleven settlement
    columns — Orthodox Christian, Other Christian denomination, and the non-Christian
    bucket, 184,147 people between them — and 160 of the (vármegye, column) pairs have a
    single category and so come out exact rather than derived. The other eight columns,
    including all 2.6M Roman Catholics and all 944k Calvinists, are published at the
    settlement itself.

    ALLOCATED WITHIN EACH VÁRMEGYE, not pooled. Hungary's minority churches are as
    regional as India's: the Romanian Orthodox are along the Romanian border, the Serbian
    Orthodox around Szentendre and Lórév, the Greek Catholics overwhelmingly in
    Szabolcs-Szatmár-Bereg. A pooled national composition would smear each of them evenly
    across the country, which is the failure --within exists to prevent.

    BUDAPEST IS 23 UNITS, NOT ONE. The capital is 17.9% of Hungary and GISCO stops at the
    city boundary; sources/hu_geo.py takes the 23 kerület from geoBoundaries ADM2 and clips
    them to GISCO's Budapest. This is the fix Croatia could not make for Zagreb.
    """
    import hu2022
    from hu2022 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "hu_settlement_allocated.csv",
                     dtype={"geo_id": str}, low_memory=False)

    lut = pd.read_csv(HERE / "data" / "geo" / "hu" / "hu_lookup.csv",
                      dtype={"geo_id": str, "kod": str})
    df["unit"] = df["geo_id"].map(dict(zip(lut["geo_id"], lut["kod"])))
    missing = df["unit"].isna().sum()
    if missing:
        raise SystemExit(f"{missing} hu rows have no settlement code -- re-run "
                         "sources/hu_geo.py, the lookup is stale")

    df["node"] = df["source_category"].map(resolve)
    df = df[df["node"].notna() & (df["count"] > 0)]
    df["congregations"] = 0
    # spec §3.10: an allocated count may never ring, because a ring asserts presence and
    # allocation only spreads a total. The Anglicans of Hungary are 372 people in a
    # country of 3,177 settlements, and a ring in every one of them would be a claim the
    # source does not make.
    df["may_ring"] = df["tier"] == "measured"
    _add_roll(df, hu2022.COLUMNS)
    # Several geo_ids can share one settlement code, so the rows are summed. `tier` and
    # `roll` KEY the group rather than being aggregated over it — spec §7's "a qualifier on
    # a row must not be aggregated over the rows it qualifies", which taking `min(tier)`
    # here did: a settlement with one large measured row and one small allocated one had
    # the whole of it relabelled `derived` by the small one.
    return (df.groupby(["unit", "node", "tier", "roll"], as_index=False, dropna=False)
              .agg(count=("count", "sum"), congregations=("congregations", "max"),
                   may_ring=("may_ring", "max")))


ENTRY = {
    "hu": dict(
        name="Hungary",
        source="Népszámlálás 2022, tables WBS003 and WBS008 (KSH)",
        basis="self-identification",
        view=[16.0, 45.6, 23.0, 48.7],
        gap="40.1% of the country, who gave no answer to the religion question",
        gap_share=0.4013,
        note_public=(
            "Two out of every five Hungarians did not answer the religion question in "
            "2022 — 3.85 million people, the largest non-response on this map by a wide "
            "margin, and up from 27% in 2011. Answering was voluntary and the share who "
            "declined has risen at every census since the question came back in 2001, so "
            "the blank is a fact about the question rather than about belief: nothing "
            "here says what those people are, and this map does not guess. What is left "
            "is 60% of the country, and within it the historic pattern is still sharp. "
            "Catholic Hungary is the west and the north — 55% of the answers west of the "
            "Danube. East of the Tisza it is 20%, and a third of the answers there are "
            "Calvinist instead: the Reformation took hold on the plain in the 16th "
            "century and the Counter-Reformation never fully undid it. The Greek "
            "Catholics, 165,000 of them, are almost all in the north-east, and their "
            "historic seat at Hajdúdorog is still four-fifths Greek Catholic. Budapest "
            "is drawn as its 23 districts rather than as one shape, and they are not "
            "alike: 27% report no religion in the Castle district and 40% in Csepel."),
        how="census, 2022",
        fill="from the same census at county level",
        grain="settlements, 1,800 people on average",
        counts=_hu_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "hu" / "hu_settlements.gpkg",
        place_unit=lambda g: g["kod"].astype(str),
        note="Settlement counts are measured for 98.1% of the population; the Orthodox, "
             "other-Christian and non-Christian columns are split from vármegye-level "
             "structure WITHIN each vármegye (allocate.py --within, spec §3.10). "
             "`Catholic, rite not stated` is derived as Catholic minus its two named "
             "rites — KSH publishes the parent and the children but never the remainder "
             "(sources/hu.md §4).",
    ),
}
