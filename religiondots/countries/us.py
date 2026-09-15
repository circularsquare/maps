# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


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


def _us_counts_rebased():
    """spec §3.5a: ASARB's rolls, plus the self-identification residual on top.

    ASARB's numbers are untouched — every one of its 372 bodies keeps the county figure it
    always had, tagged `measured`. What is added is one row per (county, root) for the people
    the survey finds and no roll holds, tagged `derived` — recorded per §7, drawn identically
    to everything else since the desaturation was removed on 2026-09-04.

    `_us_counts` stays ASARB-only ON PURPOSE and is not merely an implementation detail:
    us_weights.py fits its §8.4 demographic model against it, and fitting a model of where
    ASARB's adherents live against rows that are a survey residual would be training on the
    output. The two functions must not be merged.
    """
    from us_rebase import residual_counts

    roll = _us_counts()
    res = residual_counts(roll)      # measured against the roll as drawn, not ASARB's state sheet
    roll["may_ring"] = True
    roll["tier"] = "measured"
    return pd.concat([roll, res], ignore_index=True)


def _us_place_weight(place):
    """spec §8.4 — imported lazily so a country that does not use it never pays for it."""
    from us_weights import load_weighter

    return load_weighter(place)


ENTRY = {
    "us": dict(
        name="United States",
        name_in="the United States",
        source="U.S. Religion Census 2020 (ASARB) and Pew Religious Landscape Study 2023-24",
        basis="self-identification, with membership rolls inside it",
        # §3.5a's "the declaration stays quiet": one sentence here, the numbers in the build
        # log and in counts.json for anyone who looks, and nothing on the map itself. It
        # assumed §7's desaturation would carry "this is modelled" on screen; that was
        # removed 2026-09-04, so THIS NOTE IS NOW THE ONLY PLACE A READER LEARNS IT, and the
        # sentence about the paler dots below has to keep doing that work alone.
        note_public=(
            "The United States asks no religion question, so the totals here are Pew's "
            "survey and the detail inside them is the U.S. Religion Census — 372 bodies "
            "reporting who is on their books, which is 48.4% of the country. A little over "
            "half of the dots here are the difference between the two: people the survey "
            "finds and no membership roll holds, placed among each county's residents who "
            "are on nobody's roll. Those are an estimate and are drawn the same as the "
            "counted ones — nothing on the map marks which is which. It is also the only "
            "reason an American non-religious population can be drawn at all: a roll's "
            "residual means “on no roll”, which is not “no religion”. "
            "Two things it rests on: the survey counts adults and this applies their answers "
            "to children too, and 1.4% of people answered nothing and are not drawn at all. "
            "Adherents are attributed to "
            "the congregation's county rather than the member's home. Counties are the "
            "finest thing anyone counts — the study does not record congregation addresses "
            "— so where a dot sits INSIDE a county is an estimate from the neighbourhood's "
            "ancestry and birthplace, not a measurement, and only for bodies where that "
            "could be checked against the county figures. The rest are spread across the "
            "county's population. Judaism is the known bad case: nothing in the census marks "
            "it, so Jewish neighbourhoods are not drawn as Jewish and the bodies that can be "
            "placed take the space instead."),
        view=[-125.0, 24.0, -66.5, 49.8],
        how="church membership rolls, topped up by a national survey",
        grain="counties, 104,000 people on average",
        counts=_us_counts_rebased,
        units=None,              # counts are on counties; tracts carry both ids
        unit_key=None,
        place=HERE / "data" / "geo" / "tracts2020" / "cb_2020_us_tract_500k.shp",
        place_unit=lambda g: g["STATEFP"] + g["COUNTYFP"],
        # spec §8.4. Real ACS tract populations for every node instead of §8.2's equal-share
        # approximation, plus a demographic redistribution inside the county for the 26 nodes
        # whose held-out-metro correlation earned one. Falls back to §8.2 if unbuilt.
        place_weight=_us_place_weight,
        note="re-based on self-identification (spec §3.5a): Pew supplies the root totals, "
             "ASARB's rolls are the structure inside them, and the residual is drawn "
             "`modelled` — half the American dots, and nobody counted them at any level "
             "(us_rebase.py, 2026-09-05). Adherents are attributed to the congregation's county, not the "
             "member's (§3.6). Within a county, placement is a demographic estimate for "
             "some bodies and population-weighted for the rest (§8.4).",
    ),
}
