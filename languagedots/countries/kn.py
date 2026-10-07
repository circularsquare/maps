# St Kitts and Nevis. No census language question: the 2011 census's country of birth by island
# (sources/kn_census.py): the native-born on the Leeward creole, the foreign-born on their birth
# country's languages. Every row derived. Placed on religiondots' Kontur hexes re-keyed by island
# (data/geo/kn/kn_hexes.gpkg). Record: sources/kn.md.
from _shared import *  # noqa: F401,F403


def _counts():
    import kn2011
    df = pd.read_csv(NORM / "kn.csv")
    if sorted(df["geo_id"].unique()) != ["KN-K", "KN-N"]:
        raise SystemExit("kn.csv: expected the two islands KN-K, KN-N -- run sources/kn_census.py")
    df["node"] = df["source_category"].map(kn2011.resolve)
    df["unit"] = df["geo_id"]
    out = by_unit(df)
    out["tier"] = "derived"
    return out


ENTRY = dict(
    name="Saint Kitts and Nevis",
    source=("2011 Population and Housing Census (Department of Statistics, St Kitts and Nevis): "
            "foreign-born population by country of birth, and population by island"),
    how=("no language question: people born in St Kitts and Nevis drawn as the Leeward creole "
         "(on the map's Antiguan and Barbudan Creole), people born abroad on their birth "
         "country's languages; census 2011"),
    parts=[
        dict(covers="People born abroad",
             source="2011 census, country of birth, drawn on that country's languages",
             people=8_167),
        dict(covers="Everyone else", source="born in the federation, drawn as the Leeward creole",
             rest=True),
    ],
    grain="two islands, 23,500 people on average",
    gap="212 people whose birthplace was not stated",
    view=[-62.88, 17.08, -62.52, 17.42],
    counts=_counts,
    mappings=["kn2011"],
    place=GEO / "kn" / "kn_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "The census of St Kitts and Nevis does not ask about language. People born in the "
        "federation, 82% of the 2011 count, are drawn as speakers of the English-based creole "
        "of the Leeward Islands, which linguists treat as one language with Antiguan Creole; "
        "standard English is learned at school. People born abroad are drawn on the language "
        "of their birth country, such as Guyanese Creole for the 1,672 born in Guyana. Those "
        "born in the United States, Canada and Britain, many of them emigrants' children, are "
        "drawn as English speakers."),
)
