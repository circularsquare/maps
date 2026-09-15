# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _ar_place_weight(place):
    """countries.py hook. `place` is the 400m hex layer scatter.py has read.

    SIX UNITS FOR 46 MILLION PEOPLE, and most of their area is empty: Patagonia is a third of
    the country's land and 5.6% of its people, and NOA's people are in a few Andean valleys.
    An equal share per polygon would put most of Argentina's dots on steppe and puna
    (sources/ar_grid.py). A population weight, never a religious one.
    """
    return _kontur_place_weight(place, "ar_hexes.gpkg", "sources/ar_grid.py")


def _ar_counts():
    """CEIL-CONICET 2019 at its six survey regions: 6 drawn nodes, every row `modelled`.

    A 2,421-person survey applied to INDEC's 2022 census population, region by region, so
    §7b's test (was anybody COUNTED) fails everywhere, as in Türkiye and Guatemala. The survey
    decides the column and the census decides how many people it applies to.

    SIX REGIONS IS THE SOURCE'S OWN CEILING: the report publishes nothing finer and the
    microdata is embargoed until 2026-12-31 (sources/ar.py). Three answers carry their own
    regional shares and three are at national proportions inside each region's residual;
    that is sources/ar.py's stability test, not a taxonomy decision.
    """
    from ar2019 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "ar.csv",
                     dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    df = df[df["geo_level"] == "region"].copy()
    df["unit"] = df["geo_id"]
    if df["unit"].nunique() != 6:
        raise SystemExit(f"{df['unit'].nunique()} Argentine survey regions, expected 6")
    df["node"] = df["source_category"].map(resolve)
    unmapped = sorted(df.loc[df["node"].isna(), "source_category"].unique())
    if unmapped:
        raise SystemExit(f"ar.csv categories with no node: {unmapped}")
    df = df[df["count"] > 0]
    df = df.groupby(["unit", "node"], as_index=False)["count"].sum()
    df["congregations"] = 0
    df["tier"] = "modelled"
    return df[["unit", "node", "count", "congregations", "tier"]]


ENTRY = {
    "ar": dict(
        name="Argentina",
        source="Segunda Encuesta Nacional sobre Creencias y Actitudes Religiosas 2019 "
               "(CEIL-CONICET), against INDEC's 2022 census population",
        basis="self-identification, adults 18 and over in towns of 5,000 or more",
        note_public=(
            "**Argentina's census no longer asks about religion, so this map is drawn from a "
            "survey.** CEIL-CONICET interviewed 2,421 adults in towns of 5,000 or more in 2019 "
            "and published the results for six regions and nothing finer. Each region's "
            "shares are applied to its population in the 2022 census. "
            "**The country is 62.9% Catholic, down from 76.5% when the same team asked in "
            "2008.** The northwest (NOA) is the most Catholic region at **76.0%** and "
            "Patagonia the least at **51.0%**. No religion is highest in greater Buenos Aires "
            "at **26.2%** and lowest in the northwest at 5.0%. Evangelicals are **24.4%** of "
            "Patagonia and 23.1% of the northeast, and in the northwest they went from 3.7% "
            "to 16.7% in eleven years. "
            "**Some detail is only published for the whole country.** About 13 of the 15.3 "
            "points of evangelicals are Pentecostal, and the 18.9% with no religion are 6.0% "
            "atheist, 3.2% agnostic and 9.7% none. The regional figures for Jehovah's "
            "Witnesses and Mormons (one answer), other religions and don't know are too small "
            "to use."),
        how="survey, 2,421 adults, 2019; the census no longer asks",
        grain="six survey regions, 7.6m people each; the source's own limit",
        gap="any breakdown of the 1.2% who named another religion, which includes "
            "Argentina's Jewish and Muslim communities",
        counts=_ar_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "ar" / "ar_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_ar_place_weight,
        note="A SURVEY ON A CENSUS, AND EVERY ROW IS `modelled` (§7b). CEIL-CONICET's 2019 "
             "regional shares (Tabla 5 of Mallimaci, Esquivel and Giménez Béliveau 2020, "
             "checked against the CEIL report's p18) applied to INDEC 2022 population, with "
             "Buenos Aires province split on the census's printed 24-partido row. "
             "THE REGIONS ARE WRITTEN DOWN NOWHERE: they were read off the vector fills of "
             "CONICET's own infographic map, which puts Entre Ríos in NEA and La Pampa in "
             "Centro; AMBA is the capital plus the 24 partidos, on the 2008 report's `Capital "
             "y GBA` label. "
             "WHICH ANSWERS CARRY A GEOGRAPHY: the microdata is deposited and embargoed until "
             "2026-12-31, so the split-half is replaced by the 2008 wave's regional ranking "
             "against the exact six-unit bar (sources/spearman_null.py). Católica and Sin "
             "filiación pass; Evangélica fails at +0.600 and is drawn under OVERRIDE, because "
             "the failure is NOA's documented rise from 3.7% to 16.7% and the regions differ "
             "at chi-square p < 1e-5 under both sample allocations; the three small answers go "
             "at national proportions inside each region's residual. "
             "TABLA 7 OF THE SAME ARTICLE SWAPS CUYO'S ROWS IN BOTH YEARS; do not use it. "
             "COD-AB's ADM1 is Web Mercator and its ADM2 is lon/lat. sources/ar.md is the "
             "record.",
    ),
}
