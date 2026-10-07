# Hong Kong. 2021 Population Census, usual spoken language (sources/hk_census.py): five groups
# for 1,746 Large Subunit Groups, the two remainders shared out by the district's 15-group mix;
# placed on Kontur hexes cut to the LSUGs (sources/hk_geo.py).
from _shared import *  # noqa: F401,F403

SPLIT = ("Other Chinese dialects", "Other languages")


def _counts():
    import hk2021
    df = pd.read_csv(NORM / "hk.csv", dtype={"geo_id": str})
    lsg = df[df["geo_level"] == "lsg"]
    dc = df[df["geo_level"] == "dc"]

    # Cantonese, Putonghua and English are the small-area table as published.
    kept = lsg[~lsg["group"].isin(SPLIT)].assign(tier="measured")

    # The two remainders: each LSUG's subtotal shared by its district's mix inside that group
    # (hk_lookup.csv weights the districts for the 80 LSUGs that cross a district line).
    mix = dc[dc["group"].isin(SPLIT)].copy()
    mix["share"] = mix["count"] / mix.groupby(["geo_id", "group"])["count"].transform("sum")
    wts = pd.read_csv(GEO / "hk" / "hk_lookup.csv", dtype={"unit": str})
    rem = lsg[lsg["group"].isin(SPLIT)][["geo_id", "group", "count"]]
    rem = rem.merge(wts, left_on="geo_id", right_on="unit", how="left", validate="many_to_many")
    if rem["dc"].isna().any():
        raise SystemExit(f"hk: {rem['dc'].isna().sum()} LSUG rows with no district weight")
    rem = rem.merge(mix[["geo_id", "group", "source_category", "share"]].rename(columns={"geo_id": "dc"}),
                    on=["dc", "group"], how="left", validate="many_to_many")
    if rem["share"].isna().any():
        raise SystemExit("hk: a district has no mix for a group")
    rem["count"] = rem["count"] * rem["w"] * rem["share"]
    rem = rem[["geo_id", "source_category", "count"]].assign(tier="derived")

    out = pd.concat([kept[["geo_id", "source_category", "count", "tier"]], rem], ignore_index=True)
    out["node"] = out["source_category"].map(hk2021.resolve)
    out = out.rename(columns={"geo_id": "unit"})
    return out.groupby(["unit", "node", "tier"], as_index=False)["count"].sum()


ENTRY = dict(
    name="Hong Kong",
    source="2021 Population Census: Large Subunit Group statistics (LSUG_21C) and the "
           "Interactive Data Dissemination Service's district table of usual spoken language "
           "(Census and Statistics Department)",
    how="census, 2021, usual spoken language at home",
    parts=[
        dict(covers="Cantonese, Putonghua, English",
             source="2021 census, usual spoken language, by small area",
             nodes=["sinotibetan.sinitic.cantonese", "sinotibetan.sinitic.mandarin",
                    "indoeuropean.germanic.english"]),
        dict(covers="Other Chinese varieties and other languages",
             source="2021 census, small-area totals split by the district's mix", rest=True),
    ],
    grain="1,746 Large Subunit Groups, 4,200 people on average; smaller languages split by "
          "18 districts",
    gap="children under 5 and mute persons, 233,903 (3.2%), and 1,125 people living on boats",
    view=[113.80, 22.13, 114.52, 22.58],
    counts=_counts,
    mappings=["hk2021"],
    place=GEO / "hk" / "hk_cut.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "The census asks everyone aged 5 and over which language they use in daily "
        "conversation at home, so this map shows home language, not mother tongue. Small "
        "areas have Cantonese, Putonghua and English; the other Chinese varieties and other "
        "languages are published only by district, so in each small area they are split in "
        "its district's proportions. \"Others\" (83,546) holds Urdu, Nepali, Hindi, Korean "
        "and every language not listed separately. Most of the 201,291 Filipinos and 142,065 "
        "Indonesians are domestic workers, and far fewer give Filipino or Indonesian as their "
        "home language."),
)
