# Tuvalu. No language question: 2022 census ethnicity read as language, with Nui (Gilbertese-
# speaking) apart (sources/tv_census.py). Placed on religiondots' Kontur hexes re-keyed into Nui
# and the rest (data/geo/tv/tv_hexes.gpkg). Record: sources/tv.md.
from _shared import *  # noqa: F401,F403


def _counts():
    import tv2022
    df = pd.read_csv(NORM / "tv.csv")
    if int(df["count"].sum()) != 10_632 or sorted(df["geo_id"].unique()) != ["TV-NUI", "TV-REST"]:
        raise SystemExit("tv.csv: expected 10,632 people on TV-NUI, TV-REST -- run sources/tv_census.py")
    df["node"] = df["source_category"].map(tv2022.resolve)
    df["unit"] = df["geo_id"]
    out = by_unit(df)
    out["tier"] = "derived"
    return out


ENTRY = dict(
    name="Tuvalu",
    source="2022 Population and Housing Census, analytical report (Tuvalu CSD and SPC, 2025)",
    how=("no language question: census 2022 ethnicity read as language; everyone on Nui drawn "
         "as Gilbertese (Nuian)"),
    parts=[
        dict(covers="Nui", source="2022 census population of Nui, drawn as Gilbertese (Nuian)",
             nodes=["austronesian.oceanic.gilbertese"]),
        dict(covers="Everyone else", source="2022 census, ethnicity read as language",
             rest=True),
    ],
    grain="Nui and the rest of the country, 5,300 people on average",
    gap="about 100 people whose ethnicity was not stated",
    view=[176.0, -10.9, 179.9, -5.6],
    counts=_counts,
    mappings=["tv2022"],
    place=ROOT / "data" / "geo" / "tv" / "tv_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "Tuvalu's census asks which languages people can read, not which they speak, so "
        "ethnicity is drawn as language. 99% of residents gave a Tuvaluan answer (including 4% "
        "Tuvaluan and I-Kiribati) and are drawn as Tuvaluan speakers. The people of Nui speak "
        "Nuian, a dialect of Gilbertese, the language of Kiribati, so everyone counted on Nui "
        "is drawn as Gilbertese; Nui people living on Funafuti are drawn as Tuvaluan, as no "
        "count of them exists. The 1% of other ethnicity are drawn as language not named."),
)
