# Jordan. No census asks language. Jordanians from three Arab Barometer rounds with a first-language
# question (2011, 2013, 2016) plus 2021-22's ethnic group for Circassian; everyone else by the 2015
# census nationality, each on its home language or mix; 2015 shares per governorate applied to
# the Department of Statistics' end-2025 populations (sources/jo_build.py). Placed on
# religiondots' Kontur 400 m hexes. Record: sources/jo.md.
from _shared import *  # noqa: F401,F403

POP_2025 = 11_937_000


def _counts():
    import jo2015
    df = pd.read_csv(NORM / "jo.csv")
    if df["geo_id"].nunique() != 12:
        raise SystemExit(f"jo.csv: {df['geo_id'].nunique()} governorates, expected 12")
    if int(df["count"].sum()) != POP_2025:
        raise SystemExit(f"jo.csv sums to {df['count'].sum():,}, expected {POP_2025:,}")
    df["node"] = df["source_category"].map(jo2015.resolve)
    df["unit"] = df["geo_id"]
    return df.groupby(["unit", "node", "tier"], as_index=False)["count"].sum()


ENTRY = dict(
    name="Jordan",
    source=("Arab Barometer waves II (2011), III (2013), IV (2016) and VII (2021-22); Department "
            "of Statistics, Population and Housing Census 2015, Tables 3.1 and 8.1 (nationality "
            "by governorate) and end-2025 governorate population estimates; Rannut, 'Circassian "
            "language maintenance in Jordan', Journal of Multilingual and Multicultural "
            "Development 30:4 (2009); home languages of other countries from their censuses on "
            "this map"),
    how=("Jordanians: a survey, three rounds 2011 to 2016 pooled, first language of adults; "
         "everyone else: census, 2015, nationality, each drawn on its country's language; "
         "both as shares per governorate applied to 2025 populations"),
    parts=[
        dict(covers="Jordanian citizens",
             source="Arab Barometer 2011-2016 pooled, first language of adults; Circassian "
                    "from the 2021-22 round's ethnic group",
             people=8_286_365),
        dict(covers="Residents who are not Jordanian", source="2015 census, nationality, drawn on that "
             "country's language", rest=True),
    ],
    grain="12 governorates, 1 million people on average",
    gap="survey answers with no language given, left out before the shares",
    view=[34.9, 29.1, 39.4, 33.4],
    counts=_counts,
    mappings=["jo2015"],
    place=RD_GEO / "jo" / "jo_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "Jordan's census asks nationality but not language. Jordanian citizens are drawn from "
        "surveys of about 4,500 adults, almost all answering Arabic, drawn as Levantine "
        "Arabic. The 2015 census counted 2.9 million residents who were not Jordanian, among "
        "them 1.27 million Syrians; each nationality is drawn on its home country's language. "
        "Many Syrians have gone home since the end of 2024, so Syrian dots are likely too many. "
        "Circassian rests on seven survey answers and is likely too few; Chechen is not drawn "
        "apart."),
)
