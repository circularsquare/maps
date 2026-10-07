# Vanuatu. 2020 census Table 6.16, first language learnt to speak, by the 64 rural area councils
# plus Port Vila and Luganville (sources/vu_census.py), on religiondots' Kontur hexes for the same
# 66 units, joined by name through its vu_lookup.csv (read-only). The record is sources/vu.md.
from _shared import *  # noqa: F401,F403

# The 29,448 people aged 3+ who speak no indigenous language were not asked their first language
# (questionnaire E10 skips them). On (Anita, ask 008, 2026-10-05): they are drawn on Bislama as
# `derived`, the spec 3.5 pattern applied to this skip. Off: they are not drawn and sit in `gap`
# (then `how`, `gap` and `note_public` need the earlier wording back; sources/vu.md has it).
DRAW_NOT_ASKED = True
BISLAMA = "creole.english_based.bislama"
NOT_ASKED = "Not asked (speaks no indigenous language)"


def _counts():
    import vu2020
    df = pd.read_csv(NORM / "vu.csv")
    if df["geo_id"].nunique() != 66:
        raise SystemExit(f"vu.csv: {df['geo_id'].nunique()} units, expected 66")
    unresolved = sorted(set(df["source_category"]) - set(vu2020.NAMES))
    if unresolved:
        raise SystemExit(f"vu.csv categories with no mapping: {unresolved}")
    lut = pd.read_csv(RD_GEO / "vu" / "vu_lookup.csv", dtype=str)
    df["unit"] = df["geo_id"].map(dict(zip(lut["geo_id"], lut["unit"])))
    if df["unit"].isna().any():
        raise SystemExit(f"vu: {sorted(df.loc[df['unit'].isna(), 'geo_id'].unique())} missing "
                         "from religiondots' vu_lookup.csv")
    na = df[df["source_category"] == NOT_ASKED].copy()
    df["node"] = df["source_category"].map(vu2020.resolve)
    df = df[df["node"].notna()].copy()
    df["tier"] = "measured"
    if DRAW_NOT_ASKED:
        na["node"], na["tier"] = BISLAMA, "derived"
        df = pd.concat([df, na])
    df = df[df["count"] > 0]
    return df.groupby(["unit", "node", "tier"], as_index=False)["count"].sum()


ENTRY = dict(
    name="Vanuatu",
    source="2020 National Population and Housing Census, Basic Tables Volume 1, Table 6.16 "
           "(Vanuatu National Statistics Office)",
    how="census, 2020, first language learnt to speak, asked only of people who can speak an "
        "indigenous language; everyone else aged three and over drawn as Bislama",
    parts=[
        dict(covers="People who speak an indigenous language",
             source="2020 census, first language learnt to speak", people=239_829),
        dict(covers="Everyone else aged 3 and over",
             source="2020 census, not asked the question, drawn as Bislama", rest=True),
    ],
    grain="66 area councils, 4,100 people aged three and over on average",
    gap="children under three; people outside private households; 3 not stated",
    view=[166.0, -20.6, 170.8, -12.7],
    counts=_counts,
    mappings=["vu2020"],
    place=RD_GEO / "vu" / "vu_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "Vanuatu's census counts its hundred or so indigenous languages as one answer, so they "
        "are drawn together as Vanuatu languages, with no language named. What the map can show "
        "is where people grew up speaking one of them and where they grew up speaking Bislama, "
        "English or French. The census asked this question only of people who can speak an "
        "indigenous language. The 29,448 aged three and over who cannot were not asked, and they "
        "are drawn here as Bislama speakers, a third of Luganville and a fifth of Port Vila. "
        "Some of them grew up with English or French, so in the two towns Bislama is somewhat "
        "overcounted and English and French undercounted."),
)
