# Denmark. No language statistics: Danish, German in Sonderjylland from a cited estimate, the
# Greenland- and Faroe-born on their home languages, and immigrant languages by country of origin
# (immigrants and descendants), per kommune, Statistics Denmark 1 January 2026
# (sources/dk_dst.py). Placed on Kontur hexes keyed to the 99 LAUs (98 kommuner and Christianso;
# religiondots' dk_lau.gpkg, read-only). Record: sources/dk.md.
from _shared import *  # noqa: F401,F403

DST_2026 = 6_025_603


def _counts():
    import dk2026
    df = pd.read_csv(NORM / "dk.csv", dtype={"geo_id": str})
    if df["geo_id"].nunique() != 99 or int(df["count"].sum()) != DST_2026:
        raise SystemExit(f"dk.csv: {df['geo_id'].nunique()} kommuner, {df['count'].sum():,} "
                         "people -- run sources/dk_dst.py")
    df["node"] = df["source_category"].map(dk2026.resolve)
    df["unit"] = df["geo_id"]
    return df.groupby(["unit", "node", "tier"], as_index=False)["count"].sum()


ENTRY = dict(
    name="Denmark",
    source=("Statistics Denmark, FOLK1C (population by ancestry and country of origin per "
            "kommune, 1 January 2026), BEF5G and BEF5F (people born in Greenland and the Faroe "
            "Islands, 2026); Graenseforeningen and Den Store Danske for the German minority; "
            "home languages of other countries from their sources on this map"),
    how=("no language question: immigrants and their descendants drawn on their country of "
         "origin's languages, German from a published estimate, everyone else Danish"),
    parts=[
        dict(covers="Immigrants and their children",
             source="Statistics Denmark 2026, country of origin, drawn on that country's "
                    "languages; 22% of the children as Danish", people=961_246),
        dict(covers="People from Greenland and the Faroe Islands",
             source="Statistics Denmark 2026, place of birth, drawn as Greenlandic and Faroese",
             nodes=["eskimoaleut.greenlandic", "indoeuropean.germanic.north.faroese"]),
        dict(covers="German minority", source="published estimate, 5,000 speakers in South "
             "Jutland", people=5_000),
        dict(covers="Everyone else", source="drawn as Danish", rest=True),
    ],
    grain="98 kommuner and Christianso, 61,000 people on average",
    gap="1,538 people of stateless or unstated origin drawn as other",
    view=[8.0, 54.5, 15.3, 57.8],
    counts=_counts,
    mappings=["dk2026"],
    place=ROOT / "data" / "geo" / "dk" / "dk_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "Denmark does not ask about language in any census or register. Statistics Denmark "
        "counts, in every kommune, immigrants and their children born in Denmark by country "
        "of origin. Immigrants are drawn on the languages of the country they came from, and "
        "their children too, except for 22% of them, who are drawn as Danish speakers (a "
        "Swedish study's share, as no Danish figure exists). People born in Greenland or the "
        "Faroe Islands with no parent born in Denmark are drawn on Greenlandic and Faroese, "
        "spread by population since no kommune figure exists. About a third of the German "
        "minority of South Jutland speak German at home, so 5,000 are drawn as German speakers "
        "there. Everyone else is drawn as Danish."),
)
