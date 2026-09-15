# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _co_place_weight(place):
    """countries.py hook. `place` is the 400m hex layer scatter.py has read.

    Colombia needs it for both of §8.2's reasons. Bogotá is 15% of the people on a sliver of
    Cundinamarca's plateau, the Andean departments put their people in the valleys between
    three cordilleras, and Meta, Casanare and Caquetá are llanos and forest with their towns
    on a few roads (sources/co_grid.py).
    """
    return _kontur_place_weight(place, "co_hexes.gpkg", "sources/co_grid.py")


def _co_counts():
    """LAPOP AmericasBarometer, waves 2010-2023 pooled, at departamento: 11 categories, 26 of
    33 units drawn, and EVERY ROW IS `modelled` IN §7.

    sources/co.py builds it and sources/co.md is the record. The department labels were
    checked three ways before use, because Honduras's merge prints one wave's labels on every
    wave: `municipio` (2012-2023) and `upm` (2010, which carries DANE municipality codes) name
    the respondent's department independently of `prov`, and each wave's sample share tracks
    COD-PS. 24 Florida interviews printed under Nariño in 2012 are Valle del Cauca's and were
    moved. Seven departments with no LAPOP code (Chocó, Arauca, San Andrés, Amazonas, Guainía,
    Guaviare, Vichada) are NOT in co.csv and are in `gap=`. Of the four measured in one round,
    La Guajira and Casanare take their LAPOP design region's shares for the three placed
    answers and Quindío and Vaupés the national rate (spec §12, 2026-09-14,
    co.py::region_fallback).
    """
    from co2023 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "co.csv",
                     dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    lut = pd.read_csv(HERE / "data" / "geo" / "co" / "co_lookup.csv", dtype=str)
    df["unit"] = df["geo_id"].map(dict(zip(lut["geo_id"], lut["unit"])))
    missing = sorted(df.loc[df["unit"].isna(), "geo_id"].unique())
    if missing:
        raise SystemExit(f"co.csv departments with no polygon: {missing} -- re-run "
                         "sources/co_geo.py, the lookup is stale")
    if df["unit"].nunique() != 26:
        raise SystemExit(f"{df['unit'].nunique()} departments, expected 26")
    blank = {"CO27", "CO81", "CO88", "CO91", "CO94", "CO95", "CO99"}
    if blank & set(df["unit"]):
        raise SystemExit(f"{sorted(blank & set(df['unit']))} is in co.csv and must not be")

    df["node"] = df["source_category"].map(resolve)
    unmapped = sorted(df.loc[df["node"].isna(), "source_category"].unique())
    if unmapped:
        raise SystemExit(f"co.csv categories with no node: {unmapped}")
    df = df[df["count"] > 0]
    df["tier"] = "modelled"
    df["congregations"] = 0
    # three LAPOP answers share other.co
    return (df.groupby(["unit", "node"], as_index=False)
              .agg(count=("count", "sum"), congregations=("congregations", "sum"),
                   tier=("tier", "first")))


ENTRY = {
    "co": dict(
        name="Colombia",
        source="AmericasBarometer, five rounds 2010 to 2023 (LAPOP Lab, Vanderbilt "
               "University), against OCHA COD-PS 2025 department populations (DANE projections)",
        basis="self-identification, adults 18 and over",
        view=[-79.5, -4.4, -66.7, 12.6],
        note_public=(
            "**Colombia's census does not ask about religion, so this is a survey standing "
            "where a census would be.** The map is drawn from the LAPOP AmericasBarometer, "
            "**7,532 people** across five rounds between 2010 and 2023, pooled and applied to "
            "DANE's 2025 population projections. The dots are drawn desaturated to say so. "
            "**The Caribbean departments are the least Catholic part of the country.** "
            "Atlántico, Cesar, Magdalena, Sucre and Córdoba are 61% to 65% Catholic, against "
            "81% or more in Boyacá, Caldas and Cauca. Evangelicals run the other way, "
            "**17.5% of Magdalena** and 15.2% of Atlántico against 2.2% of Boyacá. "
            "**Three answers are drawn where the survey found them.** Catholic, evangelical, "
            "and believing in God without belonging to a religion. For the other eight the "
            "survey could not show a difference between departments that held up when its "
            "rounds were split in half, so they are spread at the national rate; that "
            "includes traditional Protestants, who are 7.0% of the country. "
            "**Seven departments are blank because the survey never went there.** Chocó, "
            "Arauca, Vichada, Guaviare, Amazonas, Guainía and San Andrés hold **2.45%** of "
            "Colombians. La Guajira, Quindío, Casanare and Vaupés were visited in one round "
            "only. La Guajira and Casanare are drawn at the rates of the survey's Caribbean and "
            "eastern regions, and Quindío and Vaupés at the national rate. "
            "**The level is a fourteen-year average.** Catholic identification runs 75.8% in "
            "2010 to **67.0% in 2023** across the pooled rounds, and evangelicals from 6.2% to "
            "9.7%, so this map is more Catholic than the last round alone would draw. After "
            "2016 the survey stopped offering Jehovah's Witnesses and Mormons as answers of "
            "their own, so both are undercounts."),
        how="survey, five rounds 2010 to 2023 pooled",
        grain="departments, 2.0 million people on average",
        gap=("Chocó, Arauca, Vichada, Guaviare, Amazonas, Guainía and San Andrés, 1,303,929 "
             "people, 2.45% of the country, which the survey never sampled"),
        gap_share=0.0245,
        counts=_co_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "co" / "co_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_co_place_weight,
        note="A SURVEY ON A PROJECTION AND EVERY ROW IS `modelled` (§7b): LAPOP's `q3c`/`q3cn` "
             "gives a department share and COD-PS 2025, which is DANE's post-2018-census "
             "projection, the people. sources/co.py builds it, sources/co.md is the record, "
             "sources.md §11ap scouted it and §9dk writes it up. "
             "THE LABELS WERE CHECKED BEFORE THEY WERE TRUSTED, because Honduras's merge prints "
             "one wave's department labels on every wave. `prov` is 800 plus the DANE code and "
             "the code and name joins agree on all 26. `municipio` (2012-2023, 51 municipalities, "
             "every label COD's name at its DANE code with three short-form aliases) and `upm` "
             "(2010, DANE municipality codes) each name the department without the label, and "
             "each wave's sample share tracks COD-PS at r=+0.89 to +0.97, none of 20,000 random "
             "pairings reaching any of them. One error was found and fixed: 24 Florida (Valle "
             "del Cauca) interviews printed under Nariño in 2012. "
             "WHICH ANSWERS ARE PLACED: median split-half over all ten 2-against-3 halvings "
             "against a per-wave permutation null, plus Sweden's chi-square veto, at the 22 "
             "departments in every wave. Catholic +0.62, evangelical +0.69, believer without a "
             "church +0.47 pass; Witnesses pass the rank test (p=0.048) and fail the chi-square "
             "(p=0.29), refused; traditional Protestant +0.03. At LAPOP's 6 design regions "
             "nothing passes that failed at the department, so the mixed-level construction "
             "places nothing at the coarse level. "
             "FOUR ASSUMED, SEVEN BLANK, on Ecuador's line: La Guajira, Quindío, Casanare and "
             "Vaupés (one wave each, 4.02%) assumed; the seven departments with no LAPOP code "
             "(2.45%) not drawn. REDRAWN 2026-09-14 on spec §12's one-round rule "
             "(co.py::region_fallback): La Guajira and Casanare take their design region's "
             "shares for the three placed answers, because Atlántica (6 of 6, mean error 6.1 "
             "against 15.2 points) and Oriental (4 of 5, 9.7 against 12.6) predict their "
             "every-round departments better than the country; Quindío (Central, 2 of 5) and "
             "Vaupés (1 of 2) stay at the national rate. `Religiones Tradicionales` is on other.co, not "
             "indigenous: 18 of its 37 respondents are in Bogotá (taxonomy/co2023.py).",
    ),
}
