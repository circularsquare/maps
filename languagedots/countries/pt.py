# Portugal. No language question in the census: Portuguese, plus Mirandese from a published
# speaker estimate, plus foreign nationals by nationality (Censos 2021) drawn on their country's
# main language less a retention share, every row derived (sources/pt_censos.py). Placed on
# religiondots' 3,092 freguesia polygons (read-only), one polygon per counted unit. The record is
# sources/pt.md.
from _shared import *  # noqa: F401,F403


def _counts():
    import pt2021
    df = pd.read_csv(NORM / "pt.csv", dtype={"geo_id": str})
    df = df[df["geo_level"] == "freguesia"]
    if df["geo_id"].nunique() != 3_092:
        raise SystemExit(f"pt.csv: {df['geo_id'].nunique()} freguesias, expected 3,092 -- "
                         "run sources/pt_censos.py")
    df["node"] = df["source_category"].map(pt2021.resolve)
    df = df[df["count"] > 0]
    df["unit"] = df["geo_id"]
    return df.groupby(["unit", "node", "tier"], as_index=False)["count"].sum()


def _weight(place):
    # one polygon per freguesia: nothing to weight inside a unit
    return None


ENTRY = dict(
    name="Portugal",
    source="INE, Censos 2021 (residents by nationality, every freguesia); Eurostat census 2021 "
           "citizenship by NUTS 3; INED/INSEE Trajectoires et Origines 2 (2019-20, home "
           "language retention by origin); Universidade de Vigo Mirandese survey (2020)",
    how="no language question; foreign nationals (census 2021) drawn on their country's "
        "languages less a share speaking only Portuguese, Mirandese from a 2020 survey, "
        "everyone else Portuguese",
    parts=[
        dict(covers="Foreign nationals, languages other than Portuguese",
             source="2021 census, nationality, drawn on that country's languages; a share moved "
                    "to Portuguese by France's TeO2 survey (2019-20)",
             people=205_634),
        dict(covers="Mirandese",
             source="Universidade de Vigo survey (2020), about 1,500 regular users in Miranda "
                    "do Douro",
             nodes=["indoeuropean.romance.mirandese"]),
        dict(covers="Everyone else", source="2021 census, drawn as Portuguese", rest=True),
    ],
    grain="3,092 freguesias, 3,300 people on average",
    view=[-9.6, 36.9, -6.1, 42.2],
    counts=_counts,
    mappings=["pt2021"],
    place=RD_GEO / "pt" / "pt_freguesias.gpkg",
    place_unit=lambda g: g["kod"].astype(str),
    place_weight=_weight,
    note_public=(
        "Portugal's census does not ask about language, so every figure here is an estimate. "
        "Portuguese and Brazilian citizens are drawn as Portuguese speakers. Other foreign "
        "residents, counted by nationality in every parish in 2021, are drawn on their "
        "country's languages, less the share France's Trajectoires et Origines survey finds "
        "speaking only the host language (about 48% for Africa, 33% for Europe). Naturalised "
        "immigrants and their children are drawn as Portuguese. Residence permits more than "
        "doubled between 2021 and 2024, so immigrant languages are undercounted for today. "
        "Mirandese is drawn from a 2020 University of Vigo survey, about 1,500 regular users in "
        "Miranda do Douro's rural parishes and two parishes of Vimioso. Barranquenho and "
        "Minderico have no count and are drawn as Portuguese."),
)
