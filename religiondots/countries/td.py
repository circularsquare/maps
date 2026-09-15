# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _td_place_weight(place):
    """countries.py hook. `place` is the Kontur 400m hex layer scatter.py has read.

    N'Djaména is 436 km2 holding 951,418 people in 2009 and Tibesti 213,590 km2 holding
    21,303; the Sahelian régions have their people along wadis and the lake
    (sources/td_geo.py).
    """
    return _kontur_place_weight(place, "td_hexes.gpkg", "sources/td_geo.py")


def _td_counts():
    """Chad RGPH2 2009 at région: 6 nodes on 22 units, every row `measured` and may ring.

    Tableau 5.07's shares raked to its région populations and Tableau 5.06's national counts
    (sources/td.py). The universe is the censused population, 10,941,682, collective
    households and refugee camps included; the 98,191 estimated are in no table.
    """
    from td2009 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "td.csv",
                     low_memory=False, keep_default_na=False, na_values=[""])
    df["count"] = df["count"].astype(int)
    if df["geo_id"].nunique() != 22:
        raise SystemExit(f"{df['geo_id'].nunique()} régions in td.csv, expected 22 -- re-run "
                         "sources/td.py")
    df["node"] = df["source_category"].map(resolve)
    unmapped = sorted(set(df.loc[df["node"].isna(), "source_category"]))
    if unmapped:
        raise SystemExit(f"td.csv categories with no node: {unmapped}")
    df = df[df["count"] > 0].rename(columns={"geo_id": "unit"})
    df["congregations"] = 0
    df["may_ring"] = True
    df["tier"] = "measured"
    return df[["unit", "node", "count", "congregations", "may_ring", "tier"]]


ENTRY = {
    "td": dict(
        name="Chad",
        source="RGPH2 2009, État et structures de la population (INSEED), Tableau 5.07",
        basis="self-identification, censused population",
        note_public=(
            "**Chad's 2009 census publishes religion for its 22 régions, and the country "
            "divides north and south.** Outside the capital, twelve of the fourteen northern "
            "and central régions are 98% to 99.5% Muslim. The seven southern régions hold 90% "
            "of Chad's Catholics, 89% of its Protestants and 94% of its animists, and Mayo "
            "Kebbi Est is **32.0%** animist. N'Djaména is 71% Muslim and 28% Christian."),
        how="census, 2009",
        grain="régions, 500,000 people on average",
        gap="0.89%: 98,191 people in parts of Sila and Tibesti that enumerators could not "
            "reach, whose number was estimated",
        gap_share=0.008894,
        counts=_td_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "td" / "td_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_td_place_weight,
        note="RGPH2 2009 structure volume, Tableau 5.07 (région x religion, % to one decimal, "
             "with populations), only on INSEED's retired jdownloads store via Wayback. Raked "
             "to 5.07's région populations and 5.06's national counts. The form's B12 codes "
             "match the columns and animists had their own box. Régions rebuilt from COD-AB "
             "départements: Ennedi Est + Ouest, and Djourf Al Ahmar and Abdi moved back to "
             "Sila and Ouaddaï. sources/td.md has the record.",
    ),
}
