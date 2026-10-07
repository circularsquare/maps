# Sudan. No census or open survey counts its languages fairly (the surveys interviewed in
# Arabic). Published speaker estimates placed on their home states, the rest of each state on
# Sudanese Arabic (sources/sd_estimate.py; ask 019's ruling); every row `modelled`. COD-PS 2022
# state populations, on religiondots' Kontur hexes for the 18 states. Record: sources/sd.md.
from _shared import *  # noqa: F401,F403

STATES = 18
CODPS_2022 = 46_934_433


def _counts():
    import sd2026
    df = pd.read_csv(NORM / "sd.csv", dtype={"geo_id": str})
    if df["geo_id"].nunique() != STATES:
        raise SystemExit(f"sd.csv: {df['geo_id'].nunique()} states, expected {STATES} -- "
                         "re-run sources/sd_estimate.py")
    if int(df["count"].sum()) != CODPS_2022:
        raise SystemExit(f"sd.csv sums to {int(df['count'].sum()):,}, not {CODPS_2022:,}")
    df["node"] = df["source_category"].map(sd2026.resolve)
    lut = pd.read_csv(RD_GEO / "sd" / "sd_lookup.csv", dtype=str)
    df["unit"] = df["geo_id"].map(dict(zip(lut["geo_id"], lut["unit"])))
    if df["unit"].isna().any():
        raise SystemExit("sd.csv states missing from religiondots' sd_lookup.csv")
    df = df[df["count"] > 0]
    out = df.groupby(["unit", "node", "tier"], as_index=False)["count"].sum()
    return out[["unit", "node", "count", "tier"]]


ENTRY = dict(
    name="Sudan",
    source=("Ethnologue's speaker figures for Sudan as Wikipedia carries them (Beja, Fur, "
            "Masalit, Nyimang, Gaam, Midob, Moro, Dongolawi); Joshua Project's people groups "
            "in Sudan by primary language for the other languages; state populations from "
            "COD-PS 2022 (Central Bureau of Statistics projection, published by OCHA)"),
    how=("no census since 1956 and no fairly sampled survey counts languages; published speaker "
         "estimates placed on each language's home states, everyone else Sudanese Arabic; no "
         "state drawn more than 80% non-Arabic, the excess in Khartoum"),
    parts=[
        dict(covers="Eight larger languages",
             source="Ethnologue's speaker estimates (Beja, Masalit, Fur, Nyimang, Gaam, Midob, "
                    "Moro, Dongolawi), placed on their home states",
             nodes=["afroasiatic.cushitic.beja", "nilosaharan.maban.masalit", "nilosaharan.fur",
                    "nilosaharan.nyimang", "nilosaharan.gaam", "nilosaharan.midob",
                    "nigercongo.moro", "nilosaharan.dongolawi"]),
        dict(covers="Other minority languages",
             source="Joshua Project's people groups by primary language, placed on their home "
                    "states",
             rest=True),
        dict(covers="Everyone else",
             source="2022 state population projection, drawn as Sudanese Arabic",
             nodes=["afroasiatic.sudanese_arabic"]),
    ],
    grain="18 states, 2.6 million people on average",
    gap=("speakers living outside their home states are drawn there only for Nobiin, Hausa and "
         "the excess over the 80% limit; foreign residents and refugees, including the Eritrean "
         "and Ethiopian communities in the east, are drawn on Sudanese Arabic; Abyei, which has "
         "no population figure, is not drawn"),
    view=[21.8, 8.6, 38.6, 22.3],
    counts=_counts,
    mappings=["sd2026"],
    place=RD_GEO / "sd" / "sd_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "Sudan's censuses have not asked about language since 1956, and the surveys that do ask "
        "held almost every interview in Arabic and found 98% Arabic speakers, far too many. So "
        "nothing on this map of Sudan is a count of a language. Each language is drawn at a "
        "published estimate of its speakers, placed in the states where they live, and everyone "
        "else in a state is drawn as Sudanese Arabic. The estimates are Ethnologue's for the "
        "larger languages (Beja 2.55 million, Masalit 980,000, Fur 790,000, Nyimang, Gaam, "
        "Midob, Moro, Dongolawi) and Joshua Project's people-group figures for the rest, which "
        "already count Arabic-speaking Nuba, Zaghawa and others as Arabic speakers. The two "
        "sources differ a lot for some languages: Joshua Project has 1.3 million Fur and half a "
        "million Masalit. Beja is drawn in Red Sea and Kassala, Nubian along the Nile in "
        "Northern State, at New Halfa and in Khartoum, Fur, Masalit, Zaghawa and their "
        "neighbours in Darfur, about 40 Nuba languages in South and West Kordofan, and Berta, "
        "Gaam and the other Blue Nile languages there. South Kordofan's projected population "
        "is smaller than its languages' speaker estimates, so no state is drawn more than 80% "
        "non-Arabic and the rest are drawn in Khartoum, where many displaced Nuba and Darfuris "
        "live. About 18% of Sudan is drawn on a language other than Arabic; the 1955-56 census "
        "put it near 30% for what is now Sudan. The map shows Sudan before the war that began "
        "in April 2023: the population projection is from before it, and more than 12 million "
        "people have since fled their homes."),
)
