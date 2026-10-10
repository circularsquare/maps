# DR Congo. No census since 1984. MICS-Palu 2017-18 (INS, UNICEF) microdata: the household head's
# mother tongue (Lingala, Swahili, Tshiluba, Kikongo, French or another language) by province and
# urban stratum, laid on the territoires; "another language" split among the local languages by
# the Enquete 1-2-3's household heads by ethnic group (2005 and 2012 rounds, as USCB tabulated
# them), each group read as its language (sources/cd_mics.py on top of sources/cd_e123.py). OCHA's
# COD-PS 2024 territoire populations and religiondots' calibrated Kontur hexes, keyed by
# territoire (read-only). Every row `modelled`. sources/cd.md is the record.
#
# Until 2026-10-09 the ethnic model drew everyone, with six cities partly moved onto Lingala or
# Swahili from two local studies (CITY_SHIFT); MICS measures the cities directly, so that is gone
# (sources/cd.md §0 and §4).
from _shared import *  # noqa: F401,F403

N_TERR = 164
POP_2024 = 117_808_872
KG = "nigercongo.bantu.kongo_dialects"
NAMED = ["nigercongo.bantu.lingala", "nigercongo.bantu.swahili_congo",
         "nigercongo.bantu.luba_kasai", "nigercongo.bantu.kituba", "indoeuropean.romance.french"] + [
    f"{KG}.{v}" for v in ("yombe", "ndibu", "manyanga", "ntandu", "mbata", "lemfu", "besingombe",
                          "kongo_se", "mboma")]


def _counts():
    import cd2017
    df = pd.read_csv(NORM / "cd_mics.csv", dtype={"geo_id": str, "province": str})
    ter = pd.read_csv(RD_GEO / "cd" / "cd_territoires.csv", dtype={"territoire": str})
    if df["geo_id"].nunique() != N_TERR or set(df["geo_id"]) != set(ter["territoire"]):
        raise SystemExit("cd: cd_mics.csv territoires differ from religiondots' "
                         "cd_territoires.csv; re-run sources/cd_e123.py and sources/cd_mics.py")
    if int(df["count"].sum()) != POP_2024 or set(df["source_id"]) != {"mics_2017_hc1b"}:
        raise SystemExit(f"cd: cd_mics.csv sums to {df['count'].sum():,} from "
                         f"{sorted(set(df['source_id']))}; re-run sources/cd_mics.py")
    df["node"] = [cd2017.resolve(l, p) for l, p in zip(df["source_category"], df["province"])]
    if df["node"].isna().any():
        raise SystemExit(f"cd: unmapped {sorted(df.loc[df.node.isna(), 'source_category'].unique())}")
    df["unit"] = df["geo_id"]
    out = by_unit(df)
    out["tier"] = "modelled"
    return out


ENTRY = dict(
    name="DR Congo",
    source=("MICS-Palu 2017-18 (Institut National de la Statistique, UNICEF), microdata, mother "
            "tongue of the household head; Enquête 1-2-3, 2005 and 2012 (Institut National de la "
            "Statistique), ethnic group of the household head as tabulated by the U.S. Census "
            "Bureau; OCHA's COD-PS 2024 territory populations"),
    how=("household survey, 2017-18, mother tongue of the household head, five languages named; "
         "the other languages shared out by ethnic group from a 2005 and 2012 survey"),
    parts=[
        dict(covers="Lingala, Swahili, Tshiluba, Kikongo and French",
             source="UNICEF MICS 2017-18, about 20,800 households, mother tongue of the "
                    "household head, by province, town and country", nodes=NAMED),
        dict(covers="Other languages",
             source="Enquête 1-2-3 2005 and 2012, household head's ethnic group read as "
                    "language, sharing out the survey's \"another language\"", rest=True),
    ],
    grain=("164 territories and cities, 718,000 people on average; the five named languages "
           "measured by province, town and country"),
    gap="16 surveyed households with no answer for the head's mother tongue",
    view=[12.2, -13.5, 31.3, 5.4],
    counts=_counts,
    mappings=["cd2012", "cd2017"],
    place=RD_GEO / "cd" / "cd_hexes.gpkg",
    place_unit=lambda g: g["territoire"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "No census has been taken in the DR Congo since 1984. This map is drawn from UNICEF's "
        "2017-18 household survey of about 20,800 households, which asked the mother tongue of "
        "each household head; everyone in a household is drawn on the head's answer. The survey "
        "offered Lingala, Swahili, Tshiluba, Kikongo, French or another language, and is "
        "reported by province, town and country: the 18 cities that are units of their own take "
        "their province's town figures, and the rest of the province takes the remainder. The "
        "41% of heads who gave another language are shared among the local languages by the "
        "ethnic group of 31,755 household heads in the statistics institute's Enquête 1-2-3 of "
        "2005 and 2012, as the U.S. Census Bureau counted them by territory, each group read as "
        "its own language. Kikongo is drawn as the Kongo varieties in Kinshasa and Kongo Central "
        "and as Kikongo ya leta (Kituba) in Kwilu and Kwango. The shares are laid on the UN "
        "humanitarian office's 2024 population projection, so the dots are survey shares and not "
        "a count. Younger people name Lingala or Swahili more often than household heads do: in "
        "Kinshasa 46% of heads gave Lingala, against 62% of men aged 15 to 49 asked in their own "
        "interview. Heads in the older survey who named a group it did not list, and the Twa, "
        "are drawn as other African languages."),
)
