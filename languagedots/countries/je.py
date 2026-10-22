# Jersey. No census language question: the 2021 census's ethnic and cultural identity
# (sources/je_census.py): Jersey, British and Irish identities on English, the Portuguese /
# Madeiran, Polish, Romanian, French and other groups on their country's languages through
# origin_mix. Every row derived. Placed on religiondots' Kontur hexes for the island as one unit
# (read-only). Record: sources/je.md.
from _shared import *  # noqa: F401,F403


def _counts():
    import je2021
    df = pd.read_csv(NORM / "je.csv")
    if df["geo_id"].nunique() != 1 or abs(df["count"].sum() - 103_267) > 1:
        raise SystemExit("je.csv: expected 103,267 people in one island unit -- run sources/je_census.py")
    df["node"] = df["source_category"].map(je2021.resolve)
    df["unit"] = "JE"   # religiondots' je_hexes.gpkg: one unit, JE
    out = by_unit(df)
    out["tier"] = "derived"
    return out


ENTRY = dict(
    name="Jersey",
    source=("2021 Census, ethnic and cultural identity and place of birth (Statistics Jersey, "
            "opendata.gov.je)"),
    how=("no language question: people of Jersey, British and Irish identity drawn as English "
         "speakers, other identities on their country's languages; census 2021"),
    parts=[
        dict(covers="People of French, Polish, Portuguese / Madeiran, Romanian and other identities",
             source="2021 census, ethnic and cultural identity, drawn on that country's languages",
             people=20_362),
        dict(covers="Everyone else", source="Jersey, British, Irish, South African and mixed "
                                            "identities, drawn as English", rest=True),
    ],
    grain="the island as one unit, 103,300 people",
    gap="none: every resident is in the identity table; Jerriais speakers (about 100 use it daily) "
        "are drawn as English",
    view=[-2.30, 49.15, -1.98, 49.28],
    counts=_counts,
    mappings=["je2021"],
    place=RD_GEO / "je" / "je_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "Jersey's 2021 census did not ask about language. It asked about ethnic and cultural "
        "identity, and the languages here follow from the answers. The 79,600 people who gave "
        "their identity as Jersey, British or Irish are drawn as English speakers. The 9,739 "
        "who chose Portuguese or Madeiran, the island's largest minority, are drawn as "
        "Portuguese speakers, and the Polish, Romanian and French groups likewise. The other "
        "groups are drawn on the languages of the countries their members were born in, "
        "weighted by how many Jersey residents were born in each. Identity includes the "
        "Jersey-born children of immigrants, some of whom speak English at home, so the minority "
        "languages may be overstated. Jerriais, the island's own language, was last counted in "
        "2001, when 113 people used it every day, and is not drawn."),
)
