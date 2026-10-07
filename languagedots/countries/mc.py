# Monaco. No language statistics: residents by nationality from IMSEE's 2025 census, Monegasques
# on French, foreign nationalities on their country's languages (sources/mc_census.py). Placed
# on Kontur hexes for the whole principality (data/geo/mc/mc_hexes.gpkg). Record: sources/mc.md.
from _shared import *  # noqa: F401,F403


def _counts():
    import mc2025
    df = pd.read_csv(NORM / "mc.csv")
    if int(df["count"].sum()) != 38_857:
        raise SystemExit("mc.csv: expected 38,857 people -- run sources/mc_census.py")
    df["node"] = df["source_category"].map(mc2025.resolve)
    df["unit"] = "MC"
    out = by_unit(df)
    out["tier"] = "derived"
    return out


ENTRY = dict(
    name="Monaco",
    source=("IMSEE, Recensement de la population 2025, Tableau 4 (residents by nationality); "
            "home languages of other countries from their sources on this map"),
    how=("census, 2025, nationality; no language question, so Monegasques are drawn as French "
         "and foreign residents on their nationality's languages"),
    parts=[
        dict(covers="Monegasques", source="2025 census, drawn as French", people=9_333),
        dict(covers="Foreign residents",
             source="2025 census, nationality, drawn on that country's languages",
             rest=True),
    ],
    grain="the principality as one unit, 38,857 people",
    view=[7.40, 43.72, 7.44, 43.76],
    counts=_counts,
    mappings=["mc2025"],
    place=ROOT / "data" / "geo" / "mc" / "mc_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "Monaco's census counts residents by nationality but does not ask about language. "
        "Monegasques, 24% of residents, are drawn as French speakers: Monegasque, the local "
        "Ligurian language, is taught in schools but almost nobody grows up speaking it. "
        "Foreign residents are drawn on the languages of their nationality. People holding two "
        "nationalities are split between them, and the 114 smallest nationalities, about 2,200 "
        "people, are drawn as language not named."),
)
