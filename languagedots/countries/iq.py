# Iraq. No census asks language (the 2024 census left it out on purpose). Five survey rounds with
# a first or home language question (World Values Survey 2004, 2013, 2018; Arab Barometer 2011,
# 2013), pooled per governorate, applied to the 2024 census governorate populations; Duhok, which
# no language question reached, from Arab Barometer 2020-22's ethnic group
# (sources/iq_surveys.py). Placed on religiondots' Kontur 400 m hexes. Record: sources/iq.md.
from _shared import *  # noqa: F401,F403

POP_2024 = 46_118_793


def _counts():
    import iq2018
    df = pd.read_csv(NORM / "iq.csv")
    if df["geo_id"].nunique() != 18:
        raise SystemExit(f"iq.csv: {df['geo_id'].nunique()} governorates, expected 18")
    if int(df["count"].sum()) != POP_2024:
        raise SystemExit(f"iq.csv sums to {df['count'].sum():,}, expected {POP_2024:,}")
    df["node"] = df["source_category"].map(iq2018.resolve)
    df["unit"] = df["geo_id"]
    out = by_unit(df)
    out["tier"] = "modelled"
    return out


ENTRY = dict(
    name="Iraq",
    source=("World Values Survey, Iraq waves 4 (2004), 6 (2013) and 7 (2018), read through the "
            "WVS online analysis tool; Arab Barometer waves II (2011), III (2013), VI-3 (2020-21) "
            "and VII (2022); 2024 census governorate populations (Central Statistical "
            "Organisation)"),
    how=("a survey, five rounds 2004 to 2018 pooled, language spoken at home or first language "
         "of adults; shares per governorate applied to each governorate's whole 2024 population; "
         "Duhok from the ethnic group asked in 2020-22"),
    parts=[
        dict(covers="Every governorate but Duhok",
             source="World Values Survey and Arab Barometer, 2004-2018 pooled, home or first "
                    "language of adults",
             people=44_518_922),
        dict(covers="Duhok", source="Arab Barometer 2020-22, ethnic group", rest=True),
    ],
    grain="18 governorates, 2.6 million people on average",
    gap="survey answers with no language given, left out before the shares",
    view=[38.7, 29.0, 48.8, 37.4],
    counts=_counts,
    mappings=["iq2018"],
    place=RD_GEO / "iq" / "iq_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "Iraq's 2024 census asked no language question, so these are survey shares of about "
        "7,100 adults, applied to each governorate's 2024 population, children included. "
        "Duhok is drawn from the ethnic group asked in 2020 to 2022. Kurdish was one answer, "
        "so Sorani and Badini are not told apart. Smaller groups are undercounted: the samples "
        "found almost no Syriac speakers, few Shabak or Yazidis in Nineveh, and no Kurds in "
        "Diyala, and Kirkuk comes out 71% Arabic, more than most estimates."),
)
