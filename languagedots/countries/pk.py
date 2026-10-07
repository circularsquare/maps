# Pakistan. Census 2023 Table 11 (sources/pk_t11.py) for the four provinces and Islamabad;
# Gilgit-Baltistan and Azad Kashmir, which Table 11 leaves out, modelled by sources/pk_north.py.
# Placed on religiondots' 2023 district hexes with GB's ten districts added
# (sources/pk_north_geo.py). The record is sources/pk.md.
from _shared import *  # noqa: F401,F403


def _counts():
    import pk2023
    df = pd.read_csv(NORM / "pk.csv")
    df["tier"] = "measured"
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
            "package; Gilgit-Baltistan from GB at a Glance 2025 and the GB MICS 2016-17 and "
            "2024-25; Azad Kashmir from the AJK Statistical Year Book 2025, Table 15.31"),
    how=("census, 2023, mother tongue; Gilgit-Baltistan and Azad Kashmir, which the district "
         "table leaves out, modelled from regional figures"),
    grain="156 districts, 1.6m people on average",
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
        dict(covers="Gilgit-Baltistan",
             source="2023 census GB totals, split by district with MICS 2016-17", people=1709030),
        dict(covers="Azad Kashmir",
             source="AJK Statistical Year Book 2025, district estimates", people=4333467),
    ],
    note_public=(
        "The 2023 census asked everyone's mother tongue, and the statistics bureau publishes it "
        "by district for the four provinces and Islamabad, with 14 languages named. Its "
        "*Others* (3.3 million people, among them Khowar, Burushaski, Wakhi and Gujari) is "
        "drawn as other languages there. Gilgit-Baltistan and Azad Kashmir are not in that "
        "table and are modelled. Gilgit-Baltistan's census shares, published for the region "
        "as a whole, are split by district using the region's MICS household surveys, which "
        "recorded the household head's language. Azad Kashmir's figures are the AJK "
        "government's own district estimates in whole percents, applied to the 2023 census "
        "population; its local Pahari varieties are all drawn as Pahari-Pothwari. Kashmir is "
        "disputed between India and Pakistan, and this map draws each side of the Line of "
        "Control with the country that administers it."),
)
