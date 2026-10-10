# Pakistan. Census 2023 Table 11 (sources/pk_t11.py) for the four provinces and Islamabad;
# Gilgit-Baltistan and Azad Kashmir, which Table 11 leaves out, modelled by sources/pk_north.py.
# MICS microdata (sources/pk_mics.py, 2026-10-09): KP 2019 splits Khyber Pakhtunkhwa's OTHERS;
# GB 2016-17 gives pk_north.py its weighted district shares.
# Placed on religiondots' 2023 district hexes with GB's ten districts added
# (sources/pk_north_geo.py), re-keyed to tehsils (sources/pk_tehsil_geo.py, 2026-10-07).
# The record is sources/pk.md.
from _shared import *  # noqa: F401,F403


def _counts():
    import pk2023
    df = pd.read_csv(NORM / "pk.csv")
    # tehsil -> drawn unit (a tehsil, a group of tehsils, or a whole district), from
    # sources/pk_tehsil_geo.py
    lut = pd.read_csv(GEO / "pk" / "pk_tehsil_units.csv")
    unit = dict(zip(lut["geo_id"], lut["unit"]))
    if set(df["geo_id"]) != set(unit):
        raise SystemExit("pk.csv tehsils and pk_tehsil_units.csv differ; rerun "
                         "sources/pk_t11.py and sources/pk_tehsil_geo.py")
    df["tier"] = "measured"
    # Khyber Pakhtunkhwa: each tehsil's OTHERS replaced by its split from MICS 2019
    # (sources/pk_mics.py): Khowar, Gujari, "Kohistani or Gujari" (modelled) and what is left
    kp = pd.read_csv(NORM / "pk_kp_mics.csv")
    was = df[df["source_category"].eq("OTHERS") & df["geo_id"].isin(kp["geo_id"])]
    if (was.set_index("geo_id")["count"].sort_index()
            != kp.groupby("geo_id")["count"].sum().sort_index()).any() or \
            len(was) != kp["geo_id"].nunique():
        raise SystemExit("pk_kp_mics.csv no longer adds up to pk.csv's OTHERS; rerun "
                         "sources/pk_mics.py")
    df = pd.concat([df.drop(was.index), kp[df.columns]], ignore_index=True)
    df["geo_id"] = df["geo_id"].map(unit)
    n = pd.read_csv(NORM / "pk_north.csv")
    prov = n.drop_duplicates("geo_id")["geo_id"].str.split("/").str[0].value_counts().to_dict()
    if prov != {"PK23-gilgit-baltistan": 10, "PK23-azad-jammu-and-kashmir": 10}:
        raise SystemExit(f"pk_north.csv: districts per area {prov}, expected 10 and 10")
    n["tier"] = "modelled"
    df = pd.concat([df, n[df.columns]], ignore_index=True)
    df["node"] = df["source_category"].map(pk2023.resolve)
    df = df[df["count"] > 0].rename(columns={"geo_id": "unit"})
    return df.groupby(["unit", "node", "tier"], as_index=False)["count"].sum()


ENTRY = dict(
    name="Pakistan",
    source=("Digital Census 2023, Table 11 (Pakistan Bureau of Statistics), via the PakPC2023 R "
            "package; Khyber Pakhtunkhwa MICS 2019 microdata (Bureau of Statistics KP, UNICEF), "
            "for the census's Others there; Gilgit-Baltistan from GB at a Glance 2025, the GB "
            "MICS 2016-17 microdata and the GB MICS 2024-25 report; Azad Kashmir from the AJK "
            "Statistical Year Book 2025, Table 15.31"),
    how=("census, 2023, mother tongue; in Khyber Pakhtunkhwa its Others split by the language "
         "of the household head in a 2019 household survey; Gilgit-Baltistan and Azad Kashmir, "
         "which the district table leaves out, modelled from regional figures"),
    grain=("463 units, 530,000 people on average: 372 tehsils, 71 groups of tehsils or whole "
           "districts where no boundary file draws the 2023 tehsils, and 20 districts in "
           "Gilgit-Baltistan and Azad Kashmir"),
    gap="people in restricted areas counted by head only (1,041,342)",
    view=[60.8, 23.6, 77.9, 37.1],
    counts=_counts,
    mappings=["pk2023"],
    place=GEO / "pk" / "pk_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    parts=[
        dict(covers="Four provinces and Islamabad",
             source="2023 census, mother tongue, Table 11", rest=True),
        dict(covers="Khyber Pakhtunkhwa's Others in the census",
             source="UNICEF MICS 2019, about 23,500 households, language of the household head",
             people=959151),
        dict(covers="Gilgit-Baltistan",
             source="2023 census GB totals, split by district with MICS 2016-17", people=1709030),
        dict(covers="Azad Kashmir",
             source="AJK Statistical Year Book 2025, district estimates", people=4333467),
    ],
    note_public=(
        "The 2023 census asked everyone's mother tongue, and the statistics bureau publishes it "
        "by tehsil for the four provinces and Islamabad, with 14 languages named. Where no "
        "boundary file draws a district's 2023 tehsils, neighbouring tehsils are drawn "
        "together. Its "
        "*Others* (3.3 million people, among them Khowar, Burushaski, Wakhi and Gujari) is "
        "drawn as other languages, except in Khyber Pakhtunkhwa, where a UNICEF "
        "household survey of 2019 names Khowar in Chitral and Gujari in Hazara. The survey "
        "counts Kohistani and Gujari as one answer, so outside Hazara that answer is named by "
        "where each tehsil lies: Torwali and Gawri in equal parts in upper Swat, Gawri in Dir Kohistan, Kohistani in Bisham, and Gujari "
        "elsewhere. Gilgit-Baltistan and Azad Kashmir are not in that "
        "table and are modelled. Gilgit-Baltistan's census shares, published for the region "
        "as a whole, are split by district using the region's MICS household surveys, which "
        "recorded the household head's language. Azad Kashmir's figures are the AJK "
        "government's own district estimates in whole percents, applied to the 2023 census "
        "population; its local Pahari varieties are all drawn as Pahari-Pothwari. Kashmir is "
        "disputed between India and Pakistan, and this map draws each side of the Line of "
        "Control with the country that administers it."),
)
