# Iraq. No census asks language (the 2024 census left it out on purpose). MICS6 2018 (UNICEF and
# the Central Statistical Organization), language of the household head (Kurdish split by the
# respondent's native language), read as every member's, weighted shares per governorate applied to the 2024 census governorate
# populations (sources/iq_mics6.py; the 2004-2018 survey pool it replaced is sources/iq_surveys.py).
# Placed on religiondots' Kontur 400 m hexes. Record: sources/iq.md.
from _shared import *  # noqa: F401,F403

POP_2024 = 46_118_793


def _counts():
    import iq2018
    df = pd.read_csv(NORM / "iq.csv")
    if df["geo_id"].nunique() != 18:
        raise SystemExit(f"iq.csv: {df['geo_id'].nunique()} governorates, expected 18")
    if int(df["count"].sum()) != POP_2024:
        raise SystemExit(f"iq.csv sums to {df['count'].sum():,}, expected {POP_2024:,}")
    if set(df["source_id"]) != {"mics6_2018_hc1b"}:
        raise SystemExit(f"iq.csv sources {sorted(set(df['source_id']))}: rerun "
                         "sources/iq_mics6.py")
    df["node"] = df["source_category"].map(iq2018.resolve)
    df["unit"] = df["geo_id"]
    out = by_unit(df)
    out["tier"] = "modelled"
    return out


ENTRY = dict(
    name="Iraq",
    source=("Iraq Multiple Indicator Cluster Survey 2018 (MICS6; Central Statistical "
            "Organization, Kurdistan Region Statistics Office, UNICEF), microdata; Arab "
            "Barometer VI-3 (2020-21) and VII (2022), ethnic group, Baghdad's Turkmen only; 2024 "
            "census governorate populations (Central Statistical Organization)"),
    how=("a household survey, 2018, language of the household head, read as every member's; "
         "weighted shares per governorate applied to each governorate's 2024 population"),
    parts=[
        dict(covers="Everyone",
             source="UNICEF MICS 2018, about 20,000 households, language of the household head",
             rest=True),
    ],
    grain="18 governorates, 2.6 million people on average",
    gap="households that were not interviewed, left out before the shares",
    view=[38.7, 29.0, 48.8, 37.4],
    counts=_counts,
    mappings=["iq2018"],
    place=RD_GEO / "iq" / "iq_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "Iraq's 2024 census asked no language question, so these are shares from UNICEF's 2018 "
        "household survey of about 20,000 households, applied to each governorate's 2024 "
        "population. Each household is drawn on the language of its head. Kurdish is split into "
        "Sorani and Badini where the person answering named one, and is plain Kurdish where they "
        "did not, as for most Kurds in Baghdad and Diyala; in the neighbouring countries it is "
        "one group. Inside a governorate dots follow population, so Tal Afar's Turkmen are "
        "spread across Nineveh and Tuz Khurmatu's across Salah al-Din. The survey found no "
        "Turkmen households in Baghdad, so Baghdad's Turkmen are the share who gave Turkmen as "
        "their ethnic group in the Arab Barometer of 2020 to 2022. It found few Yazidis or "
        "Syriac speakers in Nineveh, and 7% of Nineveh spoke a language it did not name."),
)
