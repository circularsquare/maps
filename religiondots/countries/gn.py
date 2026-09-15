# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _gn_place_weight(place):
    """countries.py hook. `place` is the Kontur 400m hex layer scatter.py has read.

    Eight régions over about 245,000 km2 is 30,000 km2 a unit, and Conakry is 436 km2 holding
    1.66 million people while Kankan is 71,700 km2 at 27 a km2. Forest Guinea, where nearly
    every animist and most Christians in the country are, is mountain and forest with its
    people in a few towns (sources/gn_geo.py).
    """
    return _kontur_place_weight(place, "gn_hexes.gpkg", "sources/gn_geo.py")


def _gn_counts():
    """Guinea RGPH 2014 at région: 5 nodes on 8 units, every row `measured` and may ring.

    Tableau 5.10's one-decimal shares on Tableau 2.07's région populations, each religion
    rescaled to UNSD table 28's national count (sources/gn.py). The universe is ordinary
    households, 10,503,132; the 20,129 in collective households are in no religion table.
    """
    from gn2014 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "gn.csv",
                     low_memory=False, keep_default_na=False, na_values=[""])
    df["count"] = df["count"].astype(int)
    if df["geo_id"].nunique() != 8:
        raise SystemExit(f"{df['geo_id'].nunique()} régions in gn.csv, expected 8 -- re-run "
                         "sources/gn.py")
    df["node"] = df["source_category"].map(resolve)
    unmapped = sorted(set(df.loc[df["node"].isna(), "source_category"]))
    if unmapped:
        raise SystemExit(f"gn.csv categories with no node: {unmapped}")
    df = df[df["count"] > 0].rename(columns={"geo_id": "unit"})
    df["congregations"] = 0
    df["may_ring"] = True
    df["tier"] = "measured"
    return df[["unit", "node", "count", "congregations", "may_ring", "tier"]]


ENTRY = {
    "gn": dict(
        name="Guinea",
        source="RGPH 2014, État et structure de la population (INS Guinée), Tableau 5.10",
        basis="self-identification, ordinary households",
        note_public=(
            "**Guinea's 2014 census publishes religion for its eight régions, and seven of "
            "them look alike.** Each of the seven is at least 89% Muslim, and Labé and Mamou "
            "in the Fouta Djallon are **99.4%**. "
            "**The exception is N'Zérékoré, in Forest Guinea.** It is **46.7%** Muslim, 28.1% "
            "Christian, 10.4% animist and 14.2% no religion. It holds 15% of Guinea's people "
            "and **62.5%** of its Christians, **98.8%** of its animists and 88.1% of those who "
            "gave no religion. "
            "**Christianity is one colour because the census never asked a denomination.** "
            "The form had five boxes: no religion, Muslim, Christian, animist and other. The "
            "animist box was an alternative to the others, so it does not count Muslims or "
            "Christians who also follow a traditional religion."),
        how="census, 2014",
        grain="régions, 1.3 million people on average",
        gap="0.19% living in collective households such as barracks and boarding schools, "
            "who are in no religion table",
        gap_share=0.0019,
        counts=_gn_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "gn" / "gn_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_gn_place_weight,
        note="§11w SAID NO RELIGION VOLUME AND WAS RIGHT ABOUT THE TITLE ONLY. The RGPH 2014 "
             "État et structure de la population volume prints religion in chapter 5: "
             "Tableau 5.10 is région x religion in percentages to one decimal, drawn here. "
             "Every other RGPH 2014 report and the scanned 1996 series were checked and "
             "nothing is finer than the région. "
             "THREE SOURCES, CHECKED AGAINST EACH OTHER: shares from Tableau 5.10, région "
             "populations from Tableau 2.07, national counts from UNSD table 28, each "
             "religion rescaled to its UNSD count (factors 0.989 to 1.008). "
             "UNSD'S `Unknown` IS THE COLLECTIVE-HOUSEHOLD POPULATION, 20,129 national, 5,750 "
             "urban, 14,379 rural, to the person against Tableaux 2.04/2.05; the other five "
             "UNSD counts reproduce all fifteen cells of Tableau 5.09 over the "
             "ordinary-household denominators. The 2014 questionnaire's P11 codes match the "
             "table's column order. sources/gn.md has the record.",
    ),
}
