# Yemen. No census asks language. Arab Barometer III (2013) first language per governorate, all
# Arabic, applied to the Population Task Force's 2025 estimates; Socotra, which no survey reached,
# on Soqotri (sources/ye_surveys.py). Placed on religiondots' Kontur 400 m hexes. Record:
# sources/ye.md.
from _shared import *  # noqa: F401,F403

POP_2025 = 34_879_018


def _counts():
    import ye2013
    df = pd.read_csv(NORM / "ye.csv")
    if df["geo_id"].nunique() != 22:
        raise SystemExit(f"ye.csv: {df['geo_id'].nunique()} governorates, expected 22")
    if int(df["count"].sum()) != POP_2025:
        raise SystemExit(f"ye.csv sums to {df['count'].sum():,}, expected {POP_2025:,}")
    df["node"] = df["source_category"].map(ye2013.resolve)
    df["unit"] = df["geo_id"]
    return df.groupby(["unit", "node", "tier"], as_index=False)["count"].sum()


ENTRY = dict(
    name="Yemen",
    source=("Arab Barometer wave III (2013); Population Task Force 2025 governorate estimates "
            "(Central Statistical Organization, UNFPA, IOM, OCHA)"),
    how=("a survey, 2013, first language of adults, applied to each governorate's 2025 "
         "population; Socotra drawn as Soqotri"),
    parts=[
        dict(covers="The mainland",
             source="Arab Barometer 2013, first language, 1,200 adults, on 2025 governorate "
                    "populations",
             rest=True),
        dict(covers="Socotra", source="2025 population estimate, drawn as Soqotri",
             nodes=["afroasiatic.soqotri"]),
    ],
    grain="22 governorates, 1.6 million people on average",
    gap="none drawn apart: Mehri speakers and about 63,000 refugees are inside the Arabic counts",
    view=[41.8, 12.0, 54.6, 19.0],
    counts=_counts,
    mappings=["ye2013"],
    place=RD_GEO / "ye" / "ye_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "Yemen has had no census since 2004 and none asked language. The Arab Barometer asked "
        "1,200 adults in 20 governorates their first language in 2013 and every one answered "
        "Arabic, so the mainland is drawn as Yemeni Arabic, applied to the 2025 population "
        "estimates of the Population Task Force. Yemeni Arabic is several dialects, but no "
        "source counts them apart. Socotra, which no survey reached, is drawn as Soqotri, the "
        "island's own language. Mehri, spoken in al-Mahrah, is not drawn: published figures "
        "cover Yemen and Oman together. About 63,000 refugees, most of them Somali, are not "
        "drawn apart because UNHCR publishes them for the whole country only."),
)
