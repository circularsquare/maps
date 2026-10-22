# Falkland Islands. 2016 census Table 5a, language other than English spoken at home, by location
# (sources/fk_census.py); everyone else a first-language English speaker. People who named two
# languages are shared (496 people, 561 mentions). Every row derived. Placed on religiondots'
# Kontur hexes for Stanley, Camp and the Mount Pleasant Complex (read-only). Record: sources/fk.md.
from _shared import *  # noqa: F401,F403


def _counts():
    import fk2016
    df = pd.read_csv(NORM / "fk.csv")
    if df["geo_id"].nunique() != 3 or abs(df["count"].sum() - 3_198) > 1e-6:
        raise SystemExit("fk.csv: expected 3,198 people in three locations -- run sources/fk_census.py")
    df["node"] = df["source_category"].map(fk2016.resolve)
    df["unit"] = df["geo_id"]      # religiondots' fk_hexes.gpkg: FK-STANLEY, FK-CAMP, FK-MOUNT-PLEASANT-COMPLEX
    out = by_unit(df)
    out["tier"] = "derived"
    return out


ENTRY = dict(
    name="Falkland Islands",
    source="Falkland Islands Census 2016, Table 5a (Falkland Islands Government)",
    how=("census, 2016, first language and languages spoken at home other than English; people who "
         "named two languages shared between them"),
    parts=[dict(covers="People who speak a language other than English at home",
                source="2016 census, Table 5a, 561 mentions by 496 people, scaled to the people",
                people=496),
           dict(covers="Everyone else", source="2016 census, English the first language", rest=True)],
    grain="3 locations, 1,100 people on average",
    gap="none: the census counted 3,198 usual residents, all in the three locations",
    view=[-61.5, -52.5, -57.5, -50.9],
    counts=_counts,
    mappings=["fk2016"],
    place=RD_GEO / "fk" / "fk_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "The 2016 census asked everyone their first language and, for those whose first language "
        "is not English, what they speak at home. English is the first language of 85% of the "
        "3,198 usual residents. The other 496 people speak 31 languages: Spanish (325 mentions), "
        "Shona (73), the Philippine languages Filipino, Tagalog "
        "and Cebuano (64), and French (26). About 65 people named two languages, so their "
        "answers are shared between them. The Mount Pleasant Complex is the military base. "
        "The 2021 census counted 3,662 residents; its language figures are "
        "not used here."),
)
