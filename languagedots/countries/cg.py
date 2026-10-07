# Republic of the Congo. Afrobarometer R9 (2022-23) home language, lingua francas read through the
# respondent's ethnic group (ask 018's first-language reading), by département, on the RGPH-5 2023
# preliminary populations (sources/cg_afro.py); religiondots' 400m Kontur hexes (read-only).
# The record is sources/cg.md.
from _shared import *  # noqa: F401,F403


def _counts():
    import cg2023
    df = pd.read_csv(NORM / "cg.csv")
    if df["geo_id"].nunique() != 12 or df["count"].sum() != 6_142_180:
        raise SystemExit("cg.csv: expected 12 départements, 6,142,180; re-run sources/cg_afro.py")
    df["node"] = df["source_category"].map(cg2023.resolve)
    missing = sorted(set(df.loc[df["node"].isna(), "source_category"]))
    if missing:
        raise SystemExit(f"cg.csv categories that resolve to nothing: {missing}")
    df["unit"] = df["geo_id"]
    df = df[df["count"] > 0]
    return df.groupby(["unit", "node", "tier"], as_index=False)["count"].sum()


ENTRY = dict(
    name="Republic of the Congo",
    source=("Afrobarometer round 9 (2022-23), language spoken at home and ethnic group; on the "
            "RGPH-5 2023 preliminary populations by département (Institut National de la "
            "Statistique)"),
    how=("survey, 2022-23, language spoken at home, 1,200 adults; French, Kituba and Lingala "
         "answers drawn as the respondent's ethnic language where one is named"),
    parts=[dict(covers="Everyone",
                source="Afrobarometer 2022-23, language at home, 1,200 adults, on the 2023 "
                       "census", rest=True)],
    grain="12 départements, 512,000 people on average",
    gap="nothing left out: answers with no ethnic language keep the language given",
    view=[11.0, -5.2, 18.8, 3.8],
    counts=_counts,
    mappings=["cg2023"],
    place=RD_GEO / "cg" / "cg_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "No Congolese census has asked about language. This map uses the one survey that "
        "did, Afrobarometer's 2022-23 round: 1,200 adults asked which language they speak at "
        "home, applied to each département's people in the 2023 census. Half named French, and "
        "most of the rest Kituba or Lingala, the two national languages of trade. Those answers "
        "are drawn as the language of the person's own ethnic group where they named one, and "
        "stay as given only for people who named none, or one with no language of its own. "
        "Many people in Brazzaville and Pointe-Noire, especially the young, grew up speaking "
        "Lingala, Kituba or French, so these three are drawn below their real share as first "
        "languages. The Kongo dots cover many related varieties that the survey does not tell "
        "apart. Only 24 to 88 people were asked in most départements outside the two cities."),
)
