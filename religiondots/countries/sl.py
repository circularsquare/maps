# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _sl_place_weight(place):
    """countries.py hook. `place` is the Kontur 400m hex layer scatter.py has read.

    Fourteen districts over about 72,000 km2 is 5,200 km2 a unit, and Western Area Urban is
    Freetown, some 80 km2 holding a million people, while Koinadugu is 12,000 km2 with its
    people along a few roads (sources/sl_geo.py).
    """
    return _kontur_place_weight(place, "sl_hexes.gpkg", "sources/sl_geo.py")


def _sl_counts():
    """Sierra Leone 2015 PHC at district: 6 nodes on 14 units, every row `measured`.

    Table 5.3's one-decimal shares on Table 2.2's district populations, scaled to the household
    population of 7,076,119 (sources/sl.py). The 15,994 enumerated in institutions are in no
    religion table.
    """
    from sl2015 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "sl.csv",
                     low_memory=False, keep_default_na=False, na_values=[""])
    df["count"] = df["count"].astype(int)
    if df["geo_id"].nunique() != 14:
        raise SystemExit(f"{df['geo_id'].nunique()} districts in sl.csv, expected 14 -- re-run "
                         "sources/sl.py")
    df["node"] = df["source_category"].map(resolve)
    unmapped = sorted(set(df.loc[df["node"].isna(), "source_category"]))
    if unmapped:
        raise SystemExit(f"sl.csv categories with no node: {unmapped}")
    df = df[df["count"] > 0].rename(columns={"geo_id": "unit"})
    df["congregations"] = 0
    df["may_ring"] = True
    df["tier"] = "measured"
    return df[["unit", "node", "count", "congregations", "may_ring", "tier"]]


ENTRY = {
    "sl": dict(
        name="Sierra Leone",
        source="2015 Population and Housing Census, Thematic Report on Population Structure "
               "and Population Distribution (Statistics Sierra Leone), Table 5.3",
        basis="self-identification, household population",
        note_public=(
            "**Sierra Leone's 2015 census publishes religion for its 14 districts.** Islam is "
            "**77.0%** of people living in households and Christianity 21.9%, and every district "
            "has a Muslim majority. Christianity is most common in the east and in Freetown: Kono "
            "is **43.5%** Christian, Kailahun 34.0% and Western Area Urban 31.3%, while Pujehun, "
            "Kambia and Port Loko are each under 6%. "
            "**The districts are the ones the census was taken on.** Karene, created in 2017, is "
            "drawn inside Bombali and Port Loko, and Falaba inside Koinadugu. "
            "**Christianity is one colour because its churches were published only for the "
            "whole country.** The form had six Christian answers, but no table splits them by "
            "district. "
            "**Each person was recorded with one religion.** The 0.1% who answered traditional "
            "religion does not include Muslims or Christians who also follow traditional "
            "practice, and most of the 0.3% recorded with no religion are children: **87%** of "
            "them are under 15."),
        how="census, 2015",
        grain="districts, 505,000 people on average",
        gap="0.23% enumerated in institutions rather than households, who are in no religion "
            "table",
        gap_share=0.0023,
        counts=_sl_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "sl" / "sl_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_sl_place_weight,
        note="TABLE 5.3 OF THE POPULATION STRUCTURE THEMATIC REPORT is district x religion in "
             "percentages to one decimal over the household population; nothing finer is "
             "published. Shares normalised within each row (99.9 to 100.1 as printed) and applied "
             "to Table 2.2's district totals times 7,076,119 / 7,092,113 (household population, "
             "National Analytical Report Table 4.1); no national count by religion exists in "
             "persons, so there is no per-religion rescale. THE NATIONAL ROW MISPRINTS BAHAI AS "
             "0.5; districts weight to 0.04 and Table 3.25 prints 0.0. THE 14 DISTRICTS ARE "
             "REBUILT FROM COD-AB v02 CHIEFDOMS (Karene and Falaba returned to Bombali, Port Loko "
             "and Koinadugu), each district's chiefdom count asserted against Table 3.3, and "
             "checked against geoBoundaries' own 14 by IoU. The form's 11 codes (code list "
             "`Religion (P07)`) have no non-response code. sources/sl.md has the record.",
    ),
}
