# Azerbaijan. 2019 census mother tongue, printed for the whole country only (sources/az_census.py),
# placed by rayon through a nationality model (sources/az_model.py), on religiondots' Kontur hexes
# for the 66 rayons and cities the census found populated. sources/az.md is the record.
# Karabakh as it is now (the Azerbaijanis resettled since 2022, 40,000 by September 2026) is added
# as settlement units KB-* from sources/az_karabakh.py (record: sources/az.md, "Karabakh").
from _shared import *  # noqa: F401,F403

_KB = 40_000

# ask 006, ruled 2026-10-05 (Anita: "okay yes"): True draws sources/az_model.py's nationality model
# by rayon (every row `modelled`); run that script first, then re-scatter. False draws the census's
# national figures as one unit, and then `how`, `grain` and `note_public` need the national wording
# back (sources/az.md has it).
MODEL = True

_LUT = pd.read_csv(RD_GEO / "az" / "az_lookup.csv", dtype=str)
# The 8 units with no EXISTING population in 2019 (the districts outside government control,
# which the census did not enumerate) keep their own ids and get no rows, so nothing is drawn there.
_POPULATED = set(_LUT.loc[_LUT["pop"].astype(int) > 0, "geo_id"])


def _counts():
    import az2019
    if MODEL:
        df = pd.read_csv(NORM / "az_model.csv", dtype={"geo_id": str})
        df["unit"] = df["geo_id"]
    else:
        df = pd.read_csv(NORM / "az.csv")
        df = df[df["area"] == "total"].copy()
        df["unit"] = "AZ"
    df["node"] = df["source_category"].map(az2019.resolve)
    import az2026_karabakh
    kb = pd.read_csv(NORM / "az_karabakh.csv")
    if abs(kb["count"].sum() - _KB) > 1:
        raise SystemExit(f"az_karabakh.csv sums to {kb['count'].sum():,.0f}; re-run sources/az_karabakh.py")
    kb["unit"] = kb["geo_id"]
    kb["node"] = kb["source_category"].map(az2026_karabakh.resolve)
    df = pd.concat([df[["unit", "node", "count", "tier"]], kb[["unit", "node", "count", "tier"]]])
    return df.groupby(["unit", "node", "tier"], as_index=False)["count"].sum()


def _place_unit(g):
    u = g["unit"].astype(str)
    return u if MODEL else u.where(~u.isin(_POPULATED), "AZ")


ENTRY = dict(
    name="Azerbaijan",
    source=("Population Census 2019, Volume B, Table 30, national (ethnic) composition and mother "
            "tongue (State Statistical Committee of Azerbaijan); for Karabakh, government "
            "statements of the number resettled, 2025-26"),
    how=("model: the 2019 census's native language split by nationality, printed for the whole "
         "country, placed by each nationality's 2009 census count by rayon; Karabakh's resettled "
         "population drawn as Azerbaijani"),
    parts=[
        dict(covers="Karabakh, resettled since 2022",
             source="President's office, September 2026: 40,000 returned, drawn as Azerbaijani "
                    "on the resettled towns and villages",
             people=_KB),
        dict(covers="The rest of Azerbaijan",
             source="2019 census, native language by nationality, placed by the 2009 census",
             rest=True),
    ],
    grain=("66 rayons and cities, 151,000 people on average; the languages within each are "
           "modelled, since the census prints native language for the whole country only; "
           "Karabakh on 15 resettled towns and villages"),
    gap=("nobody is left out of the national table; in Karabakh, people working there without "
         "having resettled"),
    view=[44.7, 38.3, 50.7, 42.0],
    # Natural Earth breakaway areas this entry now draws: not_drawn.py stops hatching them whole
    drawn_named=("Artsakh",),
    counts=_counts,
    mappings=["az2019", "az2026_karabakh"],
    place=GEO / "az" / "az_plus_hexes.gpkg",
    place_unit=_place_unit,
    place_weight=pop_weight,
    note_public=(
        "The 2019 census asked each person's native language, which in the countries of the "
        "former Soviet Union leans towards identity rather than everyday use. It prints the "
        "answers only for the whole country, split by nationality. Here each nationality's 2019 "
        "total is placed where the 2009 census counted that nationality, rayon by rayon, and its "
        "people are split over languages the same way everywhere. So the places are modelled, "
        "not counted: rural Lezgins in Gusar probably name Lezgian more often than Lezgins in "
        "Baku, and the map cannot show that. Many people of the smaller nationalities named "
        "Azerbaijani: half of the Talysh, a quarter of the Lezgins and two thirds of the Tats. "
        "The census did not count Nagorno-Karabakh and the districts around it. Nearly all of "
        "Karabakh's Armenians, more than 100,000 people, left for Armenia in September 2023, "
        "and they are not drawn here. The map draws Karabakh as it is now: the displaced "
        "Azerbaijanis the government has resettled there since 2022, spread over the resettled "
        "towns and villages using partial official and press figures. They are also in the "
        "2019 census figures where they lived before, so they are drawn twice; that is 0.4% "
        "of the country."),
)
