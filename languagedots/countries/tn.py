# Tunisia. No census asks language (RGPH 2024's form has no language item). Three Arab Barometer
# rounds with a first-language question (2011, 2013, 2016) pooled per governorate and applied to
# the RGPH 2024 governorate populations, plus Tunisian Berber from Gabsi (2011)'s 45,000-50,000
# speakers (sources/tn_surveys.py). Placed on religiondots' Kontur 400 m hexes. Record:
# sources/tn.md.
from _shared import *  # noqa: F401,F403

POP_2024 = 11_972_169


def _counts():
    import tn2016
    df = pd.read_csv(NORM / "tn.csv")
    if df["geo_id"].nunique() != 24:
        raise SystemExit(f"tn.csv: {df['geo_id'].nunique()} governorates, expected 24")
    if int(df["count"].sum()) != POP_2024:
        raise SystemExit(f"tn.csv sums to {df['count'].sum():,}, expected {POP_2024:,}")
    df["node"] = df["source_category"].map(tn2016.resolve)
    df["unit"] = df["geo_id"]
    out = by_unit(df)
    out["tier"] = "modelled"
    return out


ENTRY = dict(
    name="Tunisia",
    source=("Arab Barometer waves II (2011), III (2013) and IV (2016); Gabsi, 'Attrition and "
            "maintenance of the Berber language in Tunisia', International Journal of the "
            "Sociology of Language 211 (2011); 2024 census governorate populations (Institut "
            "National de la Statistique)"),
    how=("a survey, three rounds 2011 to 2016 pooled, first language of adults, applied to "
         "each governorate's 2024 population; Berber from a published estimate"),
    parts=[
        dict(covers="Berber",
             source="Gabsi (2011), 45,000 to 50,000 speakers, shared over Gabes, Medenine and "
                    "Tataouine",
             nodes=["afroasiatic.berber.tunisian_berber"]),
        dict(covers="Everyone else",
             source="Arab Barometer 2011-2016, first language, about 3,600 adults, on 2024 "
                    "census governorate populations",
             rest=True),
    ],
    grain="24 governorates, 500,000 people on average",
    gap="the survey answers with no language given, left out before the shares",
    view=[7.5, 30.2, 11.6, 37.6],
    counts=_counts,
    mappings=["tn2016"],
    place=RD_GEO / "tn" / "tn_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "Tunisia's census asks no language question, so these are survey shares, not a count. "
        "Three Arab Barometer rounds between 2011 and 2016 asked about 3,600 adults their first "
        "language, and each governorate's answers are applied to its 2024 census population. "
        "All but ten answered Arabic. The survey found only one Berber speaker, so Berber is "
        "drawn from a linguist's estimate of 45,000 to 50,000 speakers (Gabsi, 2011), who live "
        "in a few villages on Djerba and around Tataouine and Matmata; its dots spread over "
        "those three governorates instead of gathering in the villages. The few French, "
        "English and Italian dots are single survey answers scaled to a governorate."),
)
