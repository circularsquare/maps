# Lebanon. No census since 1932. Lebanese residents per caza (OCHA's 2026 population package, the
# base religiondots uses) at their mohafaza's pooled first / home language from Arab Barometer
# 2011-2016 and the World Values Survey 2018; Armenian placed inside each mohafaza by the
# Armenian sects' register; Syrians and Palestinians by OCHA's caza counts on Levantine Arabic
# (sources/lb_build.py). Placed on religiondots' Kontur 400 m hexes. Record: sources/lb.md.
from _shared import *  # noqa: F401,F403

DRAWN = 5_209_087      # OCHA 2026: 3,864,296 Lebanese + 1,120,000 Syrians + 224,791 Palestinians


def _counts():
    import lb2026
    df = pd.read_csv(NORM / "lb.csv")
    if df["geo_id"].nunique() != 26:
        raise SystemExit(f"lb.csv: {df['geo_id'].nunique()} cazas, expected 26")
    if int(df["count"].sum()) != DRAWN:
        raise SystemExit(f"lb.csv sums to {df['count'].sum():,}, expected {DRAWN:,}")
    df["node"] = df["source_category"].map(lb2026.resolve)
    df["unit"] = df["geo_id"]
    return df.groupby(["unit", "node", "tier"], as_index=False)["count"].sum()


ENTRY = dict(
    name="Lebanon",
    source=("Arab Barometer waves II (2011), III (2013), IV (2016) and VII (2021-22); World Values "
            "Survey wave 7, Lebanon (2018); OCHA Lebanon, 2026 LRP population package (residents "
            "by caza, from the Central Administration of Statistics' 2018-19 labour force "
            "survey); the 2022 electoral register by sect (Ministry of Interior and "
            "Municipalities, carried to cazas by religiondots) for where Armenians live"),
    how=("no census since 1932; survey language shares per mohafaza for the Lebanese, refugees "
         "drawn as Levantine Arabic"),
    parts=[
        dict(covers="Lebanese",
             source="Arab Barometer 2011-16 and World Values Survey 2018, first language or "
                    "language at home, about 5,100 adults",
             people=3_864_296),
        dict(covers="Syrians and Palestinians",
             source="OCHA 2026 counts by caza, drawn as Levantine Arabic",
             rest=True),
    ],
    grain="26 cazas for population, 6 mohafazat for the language shares",
    gap="164,097 migrant workers (3.1%), whose nationalities nothing gives by caza.",
    view=[35.0, 33.0, 36.7, 34.75],
    counts=_counts,
    mappings=["lb2026"],
    place=RD_GEO / "lb" / "lb_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "Lebanon has held no census since 1932. The Lebanese are drawn where OCHA's 2026 "
        "figures say they live, at four surveys' language shares pooled for each of the six "
        "older governorates. Armenian dots are placed by where the Armenian churches' members "
        "are registered. The surveys were held in Arabic and probably find too few Armenian "
        "speakers: the 2022 electoral register lists 104,000 Armenian Orthodox and Catholic "
        "voters aged 21 and over, against 42,000 Armenian speakers of all ages drawn here. "
        "Syrian Kurds are not drawn apart because no source counts them."),
)
