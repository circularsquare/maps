# Guyana. Census 2012 ethnic background by region (sources/gy_census.py, Compendium 2 Table
# 2.3), each group read as a language, with the share of each region's Amerindians (and the few
# Spanish, Portuguese and other speakers) taken from MICS6 2019-20's language of the household
# head (sources/gy_mics.py, taxonomy/gy2019.py). Placed on religiondots' 400 m grid for the same
# ten regions (read-only). Record: sources/gy.md.
from _shared import *  # noqa: F401,F403

POP_2012 = 746_955


def _counts():
    import gy2019
    df = pd.read_csv(NORM / "gy_mics.csv", dtype={"geo_id": str})
    if df["geo_id"].nunique() != 10:
        raise SystemExit("gy: expected 10 regions")
    if int(df["count"].sum()) != POP_2012:
        raise SystemExit(f"gy_mics.csv sums to {df['count'].sum():,}, expected {POP_2012:,}")
    if set(df["source_id"]) != {"mics6_2019_hc1b"}:
        raise SystemExit("gy_mics.csv: rerun sources/gy_mics.py")
    df["node"] = df["source_category"].map(gy2019.resolve)
    df["unit"] = df["geo_id"]
    return df.groupby(["unit", "node", "tier"], as_index=False)["count"].sum()


ENTRY = dict(
    name="Guyana",
    source=("Population and Housing Census 2012 (Bureau of Statistics), Compendium 2, Table 2.3; "
            "Guyana Multiple Indicator Cluster Survey 2019-20 (MICS6; Bureau of Statistics, "
            "Ministry of Health, UNICEF), microdata"),
    how=("census, 2012, ethnic background (no language question), each group drawn as its "
         "language: Guyanese Creole for most, English for white Guyanese. The share of each "
         "region's Amerindians whose household head's first language is indigenous, and the "
         "few Spanish and Portuguese speakers, are from a 2019-20 household survey"),
    parts=[
        dict(covers="Amerindian languages",
             source="2012 census Amerindians x the share per region in UNICEF MICS 2019-20 "
                    "(about 1,300 Amerindian households) whose head's language is indigenous",
             nodes=["americas_other"]),
        dict(covers="Spanish, Portuguese and other languages",
             source="UNICEF MICS 2019-20, language of the household head, shares per region",
             nodes=["indoeuropean.romance.spanish", "indoeuropean.romance.portuguese", "other"]),
        dict(covers="English", source="2012 census, white Guyanese",
             nodes=["indoeuropean.germanic.english"]),
        dict(covers="Everyone else", source="2012 census, ethnic background, drawn as Guyanese "
             "Creole", rest=True),
    ],
    grain="10 regions, 75,000 people on average",
    gap="households the survey did not interview, left out before the shares",
    view=[-61.5, 1.1, -56.4, 8.6],
    counts=_counts,
    mappings=["gy2019"],
    place=RD_GEO / "gy" / "gy_grid_400m.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "Guyana's census does not ask about language. It asks ethnic background, and this map "
        "draws Guyanese Creole for every group except white Guyanese (English) and part of the "
        "Amerindians. How many Amerindians speak their own language comes from UNICEF's "
        "2019-20 household survey, which asked each household head's language: about 37% of "
        "Amerindians in the interior (Cuyuni-Mazaruni, Potaro-Siparuni and the Rupununi) and "
        "under 3% in Barima-Waini and Pomeroon-Supenaam. The survey has a single answer for "
        "all indigenous languages, so Wapishana, Makushi, Patamona, Akawaio, Arawak, Carib, "
        "Warao and the others are drawn together as one unnamed Amerindian language. Some "
        "interviewers recorded no indigenous language at all in villages where their "
        "colleagues found many, and their households were left out of the shares. The survey "
        "also found a few Spanish and Portuguese speaking households; Venezuelans who arrived "
        "after 2018 are probably undercounted. Indo-Guyanese are drawn as Creole speakers: "
        "Guyanese Bhojpuri is remembered by some elderly people, but no count exists."),
)
