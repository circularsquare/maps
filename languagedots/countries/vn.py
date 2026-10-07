# Vietnam. 2019 census ethnic group by province and urban/rural (sources/vn_census.py), read as
# languages through taxonomy/vn2019.py: a proxy Anita allowed on 2026-10-05, every row `derived`.
# Each minority is split by the 2024 survey of the 53 minorities' household home language shares
# (own language / Vietnamese / another minority language). Placed on religiondots' 400m Kontur
# hexes for the 63 provinces (read-only), each province split into an urban and a rural part by
# density (sources/vn_geo.py). The record is sources/vn.md.
from _shared import *  # noqa: F401,F403


def _counts():
    import vn2019
    df = pd.read_csv(NORM / "vn.csv", dtype={"geo_id": str})
    if df["geo_id"].str[:-2].nunique() != 63:
        raise SystemExit("vn.csv: expected 63 provinces; re-run sources/vn_census.py")
    df = df[~df["source_category"].isin(vn2019.EXCLUDED)].copy()
    df["node"] = df["source_category"].map(vn2019.resolve)
    missing = sorted(set(df.loc[df["node"].isna(), "source_category"]))
    if missing:
        raise SystemExit(f"vn.csv categories that resolve to nothing: {missing}")
    if set(df["tier"]) != {"derived"}:
        raise SystemExit(f"vn.csv: tiers {sorted(set(df['tier']))}")
    df["unit"] = df["geo_id"]
    df = df[df["count"] > 0]
    return df.groupby(["unit", "node", "tier"], as_index=False)["count"].sum()


ENTRY = dict(
    name="Vietnam",
    source=("Population and Housing Census 2019, Table 2, population by ethnic group, urban/rural "
            "and province (General Statistics Office); home language shares from the 2024 survey "
            "of the 53 ethnic minorities, Table 3.9 (National Statistics Office and the Committee "
            "for Ethnic Minority Affairs)"),
    how=("census, 2019, ethnic group drawn as language, corrected by a 2024 survey of which "
         "language each minority's households speak at home"),
    parts=[
        dict(covers="Kinh", source="2019 census, ethnic group, drawn as Vietnamese",
             people=82_085_826),
        dict(covers="The 53 ethnic minorities",
             source="2019 census, ethnic group, each drawn as its language less the share the "
                    "2024 minorities survey finds speaking Vietnamese or another language at home",
             rest=True),
    ],
    grain="63 provinces, 1.5 million people on average, each split into urban and rural",
    gap="349 people whose ethnic group was not stated; the census asked no language question",
    view=[101.5, 8.0, 110.5, 23.6],
    counts=_counts,
    mappings=["vn2019"],
    place=GEO / "vn" / "vn_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "Vietnam's census asks ethnic group, not language. This map draws each of the 54 "
        "recognised groups as its language (Tay as Tay, Hmong as Hmong, Hoa as Chinese), then "
        "corrects for people who no longer speak it at home. A 2024 government survey of the 53 "
        "minorities asked which language each household mainly speaks within the family, and "
        "found that 23% of minority households mainly speak Vietnamese and 2% another minority "
        "language, which is drawn unnamed. The survey counts households, not people, and gives "
        "each group one national share, so a group that keeps its language in the hills and "
        "has given it up in town is drawn with one average. The Hoa are drawn as Chinese and "
        "the Dao as one language, since no source splits their varieties. Inside each "
        "province, people the census counted as urban are placed in its densest areas and "
        "the rest in the countryside."),
)
