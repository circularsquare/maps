# Guatemala. Censo 2018, the language each person learned to speak in, per municipio
# (sources/gt_censo.py, INE's REDATAM server). Placed on Kontur hexes per municipio
# (sources/gt_geo.py).
from _shared import *  # noqa: F401,F403


def _counts():
    import gt2018
    df = pd.read_csv(NORM / "gt.csv", dtype={"geo_id": str})
    df["node"] = df["source_category"].map(gt2018.resolve)
    df = df[df["node"].notna()].copy()        # "No habla": not drawn, in `gap`
    df["unit"] = df["geo_id"]                 # INE's code; sources/gt_geo.py keys the hexes on it
    return by_unit(df)


ENTRY = dict(
    name="Guatemala",
    source="XII Censo Nacional de Población 2018 (Instituto Nacional de Estadística), "
           "tabulated on INE's REDATAM server",
    how="census, 2018, mother tongue (the language the person learned to speak in)",
    parts=[dict(covers="People aged 4 and over", source="2018 census, mother tongue",
                rest=True)],
    grain="340 municipios, 44,000 people on average",
    gap="children under 4, 1.33 million (9.0%), whom the census does not ask",
    view=[-92.4, 13.6, -88.1, 17.9],
    counts=_counts,
    mappings=["gt2018"],
    place=GEO / "gt" / "gt_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "The 2018 census asked everyone aged 4 and over which language they learned to speak "
        "in. 29.6% answered a Mayan language, while 41.2% said they belong to the Maya people, "
        "so many Maya learned Spanish first. Inside each municipio the dots follow where people "
        "live, so a Spanish-speaking town and Mayan-speaking villages show mixed together."),
)
