# Dominican Republic. No census language question: ONE's ENHOGAR-MICS6 2019 household survey asked
# each household head's mother tongue (HC1B), applied to everyone in the household and to the
# 2022 census province populations (sources/do_enhogar.py). Every row modelled. Placed on
# religiondots' Kontur hexes for 32 provinces (read-only). Record: sources/do.md.
from _shared import *  # noqa: F401,F403


def _counts():
    import do2019
    df = pd.read_csv(NORM / "do.csv", dtype={"geo_id": str})
    if df["geo_id"].nunique() != 32:
        raise SystemExit(f"do.csv: {df['geo_id'].nunique()} provinces, expected 32 -- "
                         "run sources/do_enhogar.py")
    df["node"] = df["source_category"].map(do2019.resolve)
    df = df[df["count"] > 0]
    df["unit"] = df["geo_id"]   # religiondots' do_lookup.csv: geo_id == unit
    out = by_unit(df)
    out["tier"] = "modelled"
    return out


ENTRY = dict(
    name="Dominican Republic",
    source=("Encuesta Nacional de Hogares de Propósitos Múltiples ENHOGAR-MICS6 2019 (Oficina "
            "Nacional de Estadística), item HC1B, on the X Censo Nacional de Población y "
            "Vivienda 2022 province populations"),
    how=("a household survey, 2019, the mother tongue of the household head, applied to "
         "everyone in the household; shares per province applied to the 2022 census population"),
    parts=[dict(covers="Everyone",
                source="ENHOGAR-MICS6 2019 survey, household head's mother tongue, provincial "
                       "shares on the 2022 census", rest=True)],
    grain="32 provinces, 337,000 people on average",
    view=[-72.2, 17.3, -68.2, 20.1],
    counts=_counts,
    mappings=["do2019"],
    place=RD_GEO / "do" / "do_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "The Dominican census does not ask about language, so this map comes from the "
        "statistics office's 2019 household survey (ENHOGAR, run with UNICEF as MICS6), which "
        "asked the mother tongue of the head of each of 31,488 households in all 32 provinces. "
        "Everyone in a household is drawn on the head's language, and the provincial shares "
        "are applied to the 2022 census population. About 6% of people live in a household "
        "headed by a Haitian Creole speaker, up to a quarter along the Haitian border. Children "
        "born in the country to Haitian parents often grow up speaking Spanish, so the head's "
        "language overstates Creole among them. English, French and other languages are each "
        "under 0.1% and rest on a handful of households in any one province."),
)
