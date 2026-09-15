# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _mm_place_weight(place):
    """countries.py hook. `place` is the 400m hex layer scatter.py has read.

    15 states for 51.5M is ~3.4M each, the second coarsest counting geography on this map,
    and they are wildly uneven: Yangon is 7.36M in 9,867 km2 against Kachin's 1.64M in
    88,978 and Chin's 479k in 36,018 of mountain. An equal share would wash the empty north
    and squash a seventh of the country into one speck.

    AND THE GRID IS 2023 AGAINST A 2014 CENSUS, WHICH MATTERS IN ONE STATE. Rakhine reads
    0.58x against the population it is drawn on and 0.89x against the enumerated count
    alone; the difference is the ~750,000 Rohingya who left in 2017 and are not in a 2023
    surface. So Rakhine's `unenumerated` dots are weighted towards where people live now,
    which is south of where those people were. Nothing published can fix it — see
    sources/mm_grid.py.
    """
    return _kontur_place_weight(place, "mm_hexes.gpkg", "sources/mm_grid.py")


def _mm_counts():
    """DOP 2014 Census Vol 2-C Table 1 at State/Region: 8 drawn categories on 15 units.

    ONE level and no allocation — but NOT all `measured`, which is the point of the country.

    THE EIGHTH CATEGORY IS NOT A RELIGION. `Estimated Non-enumerated population` is DOP's
    own estimate of the people the census did not reach: 1,206,353 nationally, of which
    Rakhine is 1,090,000 and Kayin and Kachin the rest. It maps to `unenumerated` and every
    row of it is `modelled` (§7), because the tiers are about whether anybody was counted
    and here nobody was — 1,090,000 is a round number in the source because it is an
    estimate. `inferred dots: hidden` therefore empties exactly that node and shows the
    census as the state published it.

    THE `Total` ROW IS THE ENUMERATED TOTAL AND IS NOT THIS COUNTRY'S UNIVERSE. Enumerated
    50,279,900 plus non-enumerated 1,206,353 = 51,486,253, which is the report's own overall
    figure and what Myanmar draws.
    """
    from mm2014 import MODELLED, resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "mm.csv",
                     dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    df = df[df["geo_level"] == "state_region"].copy()

    lut = pd.read_csv(HERE / "data" / "geo" / "mm" / "mm_lookup.csv", dtype=str)
    df["unit"] = df["geo_id"].map(dict(zip(lut["geo_id"], lut["unit"])))
    missing = sorted(df.loc[df["unit"].isna(), "geo_id"].unique())
    if missing:
        raise SystemExit(f"mm.csv states with no polygon: {missing} -- re-run "
                         "sources/mm_geo.py, the lookup is stale")
    if df["unit"].nunique() != 15:
        raise SystemExit(f"{df['unit'].nunique()} states, expected 15")

    df["node"] = df["source_category"].map(resolve)
    df = df[df["node"].notna() & (df["count"] > 0)]
    if "unenumerated" not in set(df["node"]):
        raise SystemExit("mm.csv resolved no `unenumerated` rows -- the non-enumerated "
                         "column has been lost, and Rakhine is about to draw 96% Buddhist "
                         "(spec §14.2)")
    df["tier"] = df["node"].map(lambda n: "modelled" if n in MODELLED else "measured")
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations", "tier"]]


ENTRY = {
    "mm": dict(
        name="Myanmar",
        source="2014 Census Report Volume 2-C: Religion, Table 1 (Department of Population)",
        basis="self-identification, enumerated census population, plus the state's own "
              "estimate of who it did not enumerate",
        view=[92.0, 9.4, 101.4, 28.7],
        note_public=(
            "**The most important number on this map of Myanmar is the one that is not a "
            "religion.** The 2014 census did not enumerate an estimated 1,206,353 people, "
            "and 1,090,000 of them are in Rakhine State — **34% of everybody there.** The "
            "census report says why, in its own words: *\"In Rakhine, an estimated 1.09 "
            "million people were not enumerated in the Census because they were not allowed "
            "to self-identify using a name not recognized by the Government.\"* That is the "
            "Rohingya. The remainder is Kayin (69,753) and Kachin (46,600), areas that were "
            "not under government control when the census was taken. "
            "**They are drawn as *not enumerated* rather than as Muslim, and the report "
            "itself would allow the stronger claim.** It states that *\"it is assumed that "
            "the non-enumerated population in Rakhine is mainly affiliated with the Islamic "
            "faith\"*, and publishes a second national figure on that basis: Islam 4.3% "
            "instead of the 2.3% the enumerated count gives. But it applies that assumption "
            "only to the country as a whole, and this map draws states — so these dots say "
            "that these people were not counted, and nothing about what they believe. "
            "**Without them Rakhine reads 96.2% Buddhist**, which is the census's own "
            "exclusion turned into a finding. "
            "**Christianity is an upland religion here and the boundary is sharp.** Chin is "
            "**85.4% Christian**, Kayah 45.8% and Kachin 33.8% — the American Baptist "
            "mission field from 1813 onward, plus Catholics and Anglicans, none of which the "
            "census separates — against 1.1% in Magway and Nay Pyi Taw. Chin and Kachin are "
            "the only states where something other than Buddhism holds a plurality or comes "
            "close. "
            "**Almost all of the country's traditional religion is in one state.** Shan "
            "holds 383,072 of Myanmar's 408,045 Animists, 94% of the national figure. Read "
            "that as a floor: nat propitiation is close to universal in Myanmar and normally "
            "accompanies Buddhism rather than replacing it, and a census that allows one "
            "religion per person counts those people as Buddhist. "
            "**The census is from 2014 and the country has since had a coup and a civil "
            "war.** Roughly three quarters of a million Rohingya left Rakhine for Bangladesh "
            "in 2017, three years after this count, and millions more people have been "
            "displaced since 2021. This is a picture of 2014 and is not a current one. "
            "**Fifteen units and seven categories is everything the census published.** "
            "There is no district or township religion table anywhere, no Buddhist school, "
            "no branch of Islam, and no Christian body named."),
        how="census, 2014, plus the state's count of who it missed",
        grain="states and regions, 3.4 million people on average",
        counts=_mm_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "mm" / "mm_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_mm_place_weight,
        note="THE ENTIRE PUBLISHED RELIGION OUTPUT OF THE 2014 CENSUS IS TWO TABLES IN A "
             "17-PAGE REPORT — Union plus 15 States/Regions, and a 1973/1983/2014 time "
             "series. There is no district or township religion table anywhere: not "
             "withheld at a finer tier, not hard to find, absent. So 15 units is not §14's "
             "resolution limit being applied, it is all there is, and §3.9b is why it is "
             "drawn. READ IT AS COMPOSITION, NEVER AS LOCATION — a state here averages 3.4 "
             "million people. "
             "THE NON-ENUMERATED ARE AN EIGHTH CATEGORY AND EVERY ROW OF IT IS `modelled`. "
             "They are not a religion and not an answer: they are DOP's own estimate of "
             "people the census did not reach, and 1,090,000 is a round number in the source "
             "because it is an estimate. Mapping them to `islam` — which DOP's own Union-"
             "level assumption would support — would invent a magnitude at a resolution the "
             "source does not publish, which is §14 rule 1. `inferred dots: hidden` empties "
             "this node exactly, which is the honest test of the country. "
             "THE `Total` ROW IS THE ENUMERATED TOTAL AND IS NOT THE UNIVERSE. 50,279,900 "
             "enumerated plus 1,206,353 non-enumerated is 51,486,253, the report's own "
             "overall figure and what is drawn. "
             "THE COUNTING GEOGRAPHY IS NOT THE ADMINISTRATIVE ONE AND THE FEATURE COUNT "
             "SAYS SO: COD's ADM1 has 18 features because the standard p-codes split Bago in "
             "two and Shan in three, against the census's 15 rows. They are dissolved by "
             "rule — strip a parenthesised qualifier and group — and MIMU's own religion "
             "sheet confirms the intent by coding them MMR111 and MMR222. Every "
             "multi-member group is asserted CONTIGUOUS, and the 15 are asserted to tile "
             "with no overlap. "
             "THE PARSE HAS TWO CHECKS A COLUMN PERMUTATION CANNOT SURVIVE. DOP prints a "
             "percentage under every count, so count/total must reproduce it on all 110 "
             "cells; and MIMU published an independent p-coded transcription of the same "
             "table, matched here BY ITS FIGURES rather than by name or position, 1:1 both "
             "ways. One cell of the report is simply wrong — Kachin's Hindu share is 0.349% "
             "and prints as 0.4 — and it is listed so a second one fails the build. "
             "THE PLACEMENT GRID IS 2023 AND THE CENSUS IS 2014, AND IN ONE STATE THAT IS "
             "VISIBLE. Kontur reads 0.58× on Rakhine against the population drawn there and "
             "0.89× against the enumerated count alone, in line with the other fourteen "
             "states; the gap is the ~750,000 Rohingya who left in 2017. So Rakhine's "
             "not-enumerated dots sit south of where those people actually lived, in the "
             "northern townships. No source publishes them below state level, a uniform "
             "spread would put them in the Arakan mountains, and inventing a northern "
             "concentration would be §14.4.",
    ),
}
