# Bhutan. No census asks language. Bhutanese from the 2015 Gross National Happiness survey's mother
# tongue by dzongkhag (Table A1.5, weighted), non-Bhutanese by UN DESA origin on each origin's home
# mix, both on the 2017 census's dzongkhag counts (sources/bt_gnh.py). Placed on religiondots'
# Kontur hexes, already scaled to each dzongkhag's census count (read-only). Record: sources/bt.md.
from _shared import *  # noqa: F401,F403

POP_2017 = 727_145


def _counts():
    import bt2015
    df = pd.read_csv(NORM / "bt.csv", dtype={"geo_id": str})
    if df["geo_id"].nunique() != 20 or int(df["count"].sum()) != POP_2017:
        raise SystemExit(f"bt.csv: {df['geo_id'].nunique()} dzongkhags, {df['count'].sum():,} "
                         "people -- run sources/bt_gnh.py")
    df["node"] = df["source_category"].map(bt2015.resolve)
    df["unit"] = df["geo_id"]   # religiondots' bt_lookup.csv: geo_id == unit
    out = df.groupby(["unit", "node", "tier"], as_index=False)["count"].sum()
    return out


ENTRY = dict(
    name="Bhutan",
    source=("Centre for Bhutan & GNH Studies, 2015 Gross National Happiness survey, Table A1.5 "
            "(mother tongue by dzongkhag); National Statistics Bureau, Population and Housing "
            "Census of Bhutan 2017 (Bhutanese and non-Bhutanese by dzongkhag); UN DESA "
            "International Migrant Stock 2020"),
    how=("survey, 2015, mother tongue of Bhutanese aged 15 and over, as shares per dzongkhag "
         "applied to the 2017 census; foreign residents by country of origin, each drawn on "
         "its country's languages"),
    parts=[
        dict(covers="Foreign residents",
             source="2017 census count, drawn by UN DESA 2020 migrants' countries of birth and "
                    "their languages", people=45_425),
        dict(covers="Bhutanese",
             source="Gross National Happiness survey 2015, mother tongue, aged 15 and over, "
                    "dzongkhag shares on the 2017 census", rest=True),
    ],
    grain="20 dzongkhags, 36,000 people on average",
    view=[88.7, 26.7, 92.2, 28.3],
    counts=_counts,
    mappings=["bt2015"],
    place=RD_GEO / "bt" / "bt_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "Bhutan's census does not ask about language. The Gross National Happiness survey of "
        "2015 asked about 7,000 Bhutanese aged 15 and over their mother tongue and published "
        "the weighted shares for each dzongkhag; those shares are applied here to the 2017 "
        "census count of Bhutanese in each dzongkhag, children included. About 4% answered "
        "\"other\", a quarter of Tsirang and a fifth of Dagana; these are probably Tamang, "
        "Gurung, Rai and Limbu speakers among the Lhotshampa, but the survey does not say. The "
        "census counted 45,425 foreign residents but not their nationality; they are drawn by "
        "the United Nations' count of migrants in Bhutan by country of birth, 88% Indian, and "
        "Indians on the languages of the Indian states most of them come from. Day workers who "
        "cross from India each morning are not on the map."),
)
