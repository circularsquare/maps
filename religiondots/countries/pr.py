# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _pr_place_weight(place):
    """countries.py hook. `place` is the 400m hex layer scatter.py has read.

    Six regions of about 1,500 km2 each, and every one of them is a dense coastal strip beside
    empty uplands: Este holds El Yunque, Centro is the Cordillera Central, and Oeste's Mayagüez
    polygon takes in Mona Island (sources/pr_grid.py).
    """
    return _kontur_place_weight(place, "pr_hexes.gpkg", "sources/pr_grid.py")


def _pr_counts():
    """World Values Survey wave 7, Puerto Rico 2018, at the survey's six regions: 6 categories,
    EVERY ROW `modelled`.

    1,127 adults in 18 municipios, three per region; the regions are built from the survey
    team's own map of which municipios make up each (sources/pr_geo.py), and the shares are laid
    on the Census Bureau's Vintage 2024 municipio estimates. sources/pr.md is the record.
    """
    from pr2018 import resolve
    import pr2018

    df = pd.read_csv(HERE / "data" / "normalized" / "pr.csv",
                     dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    lut = pd.read_csv(HERE / "data" / "geo" / "pr" / "pr_lookup.csv", dtype=str)
    df["unit"] = df["geo_id"].map(dict(zip(lut["geo_id"], lut["unit"])))
    missing = sorted(df.loc[df["unit"].isna(), "geo_id"].unique())
    if missing:
        raise SystemExit(f"pr.csv regions with no polygon: {missing} -- re-run sources/pr_geo.py")
    if df["unit"].nunique() != 6:
        raise SystemExit(f"{df['unit'].nunique()} regions, expected 6")
    df["node"] = df["source_category"].map(resolve)
    unmapped = sorted(set(df.loc[df["node"].isna(), "source_category"]) - set(pr2018.EXCLUDED))
    if unmapped:
        raise SystemExit(f"pr.csv categories with no node: {unmapped}")
    df = df[df["node"].notna() & (df["count"] > 0)].copy()
    df["tier"] = "modelled"
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations", "tier"]]


ENTRY = {
    "pr": dict(
        name="Puerto Rico",
        source="World Values Survey wave 7, 2018 (Universidad del Sagrado Corazón with the "
               "Instituto de Estadísticas de Puerto Rico), against the U.S. Census Bureau's 2024 "
               "municipio population estimates",
        basis="self-identification, adults 18 and over",
        view=[-67.35, 17.85, -65.2, 18.6],
        note_public=(
            "The United States census does not ask about religion, and that includes Puerto "
            "Rico. This map is drawn from the 2018 World Values Survey, which interviewed "
            "**1,127 adults** in 18 municipios, three in each of six regions, and each region "
            "is drawn with the answers from its three municipios. "
            "**Catholics are about half of Puerto Rico** and **31%** of the east, where a third "
            "of people gave some other Christian answer. That answer was a write-in on the "
            "Puerto Rican questionnaire, and the survey's file records it only as other "
            "Christian. "
            "Protestants and people with no religion did not differ reliably between regions in "
            "this sample, so each region's remainder is shared between them at the island-wide "
            "ratio."),
        how="survey, one round of 1,127 interviews in 2018",
        grain="regions, 530,000 people on average",
        gap="the 0.9% who did not answer",
        gap_share=0.0093,
        counts=_pr_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "pr" / "pr_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_pr_place_weight,
        note="WVS-7 PUERTO RICO, SIX REGIONS BUILT FROM THE SURVEY TEAM'S OWN MAP. The file ends "
             "every row with ';', which shifts every column by one under pandas' default; read "
             "with index_col=False. N_REGION_WVS is the region and N_REGION_ISO the municipio "
             "(3 per region); the report (Instituto de Estadísticas, 2019) p.16 maps all 78 "
             "municipios to regions, and Vieques and Culebra, not on it, are put in Este. Join "
             "pinned by the report's region and municipio interview totals; sample share against "
             "adults does not pin it (Centro over-sampled 1.54x). Split-half on municipios, one "
             "against two, median of 23,328 halvings against a null of random groupings: "
             "Católico and Otros carry region shares; Protestante, No pertenece, Budista "
             "(3 of 5 in San Juan) and Hindú share each region's remainder at national "
             "proportions. Otros is a write-in the archive coded Other Christian. sources/pr.md.",
    ),
}
