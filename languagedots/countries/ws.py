# Samoa. No language question in the 2021 census: Samoan citizens drawn as Samoan per traditional
# district, non-citizens (nationality unpublished) not drawn (sources/ws_census.py). Placed on
# religiondots' Kontur hexes for the 25 traditional districts (read-only). Record: sources/ws.md.
from _shared import *  # noqa: F401,F403


def _counts():
    import ws2021
    df = pd.read_csv(NORM / "ws.csv")
    if df["geo_id"].nunique() != 25:
        raise SystemExit(f"ws.csv: {df['geo_id'].nunique()} districts, expected 25 -- "
                         "run sources/ws_census.py")
    df["node"] = df["source_category"].map(ws2021.resolve)
    df["unit"] = df["geo_id"]   # religiondots' ws_lookup.csv `unit`
    out = by_unit(df)
    out["tier"] = "derived"
    return out


ENTRY = dict(
    name="Samoa",
    source="Population and Housing Census 2021 (Samoa Bureau of Statistics)",
    how=("no language question: every Samoan citizen drawn as Samoan, the first language of "
         "nearly all Samoans; census 2021"),
    parts=[dict(covers="Samoan citizens", source="2021 census, drawn as Samoan", rest=True)],
    grain="25 traditional districts, 8,200 people on average",
    gap="1,218 non-citizens (0.6%), whose nationality the census does not publish",
    view=[-172.9, -14.15, -171.35, -13.4],
    counts=_counts,
    mappings=["ws2021"],
    place=RD_GEO / "ws" / "ws_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "Samoa's census does not ask about language, so every Samoan citizen is drawn as a "
        "Samoan speaker. Samoan is the language nearly all Samoans grow up with and the main "
        "language of government, church and village life; English is learned at school. "
        "Families who speak English at home, most of them in Apia, have no count and are "
        "drawn as Samoan speakers. The 1,218 residents who are not Samoan citizens are left "
        "off the map because the census does not say where they are from."),
)
