# British Virgin Islands. No census language question: the 2010 census's grouped country of birth
# by island (sources/vg_census.py): the BVI-born on Virgin Islands Creole, the foreign-born on
# their birth country's language. Every row derived. Placed on religiondots' Kontur hexes for the
# four islands (read-only). Record: sources/vg.md.
from _shared import *  # noqa: F401,F403


def _counts():
    import vg2010
    df = pd.read_csv(NORM / "vg.csv")
    want = ["VG-ANEGADA", "VG-JOST-VAN-DYKE", "VG-TORTOLA", "VG-VIRGIN-GORDA"]
    if sorted(df["geo_id"].unique()) != want:
        raise SystemExit(f"vg.csv: expected units {want} -- run sources/vg_census.py")
    df["node"] = df["source_category"].map(vg2010.resolve)
    df["unit"] = df["geo_id"]   # religiondots' vg_hexes.gpkg units
    out = by_unit(df)
    out["tier"] = "derived"
    return out


ENTRY = dict(
    name="British Virgin Islands",
    source=("2010 Population and Housing Census Report, Table 84, grouped country of birth by "
            "island (Government of the Virgin Islands)"),
    how=("no language question: people born in the territory drawn as Virgin Islands Creole, "
         "people born abroad on their birth country's language; census 2010"),
    parts=[
        dict(covers="People born in the British or US Virgin Islands",
             source="2010 census, drawn as Virgin Islands Creole",
             nodes=["creole.english_based.virgin_islands"]),
        dict(covers="People born abroad",
             source="2010 census, country of birth, drawn on that country's language",
             rest=True),
    ],
    grain="4 islands, 7,000 people on average",
    gap="62 people whose birthplace was not stated",
    view=[-64.85, 18.30, -64.25, 18.78],
    counts=_counts,
    mappings=["vg2010"],
    place=RD_GEO / "vg" / "vg_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "The census of the British Virgin Islands does not ask about language. Only 39% of "
        "residents in 2010 were born in the territory; they are drawn as speakers of Virgin "
        "Islands Creole, the English-based creole shared with the US Virgin Islands. The other "
        "61% come from more than a hundred countries and are drawn on the language of where "
        "they were born; people born in the United States and Britain, many of them children "
        "of islanders, are drawn as English speakers. The census groups some birthplaces into "
        "regions, and those people are drawn as language not known."),
)
