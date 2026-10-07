# DR Congo. No census since 1984: the Enquete 1-2-3's household heads by ethnic group (2005 and
# 2012 rounds, as USCB tabulated them), each group read as its language, by territoire
# (sources/cd_e123.py), on OCHA's COD-PS 2024 territoire populations and religiondots' calibrated
# Kontur hexes, keyed by territoire (read-only). Every row `modelled`. sources/cd.md is the record.
from _shared import *  # noqa: F401,F403

# Share of Kinshasa drawn on Lingala, whatever the head's ethnic group: 58% of 500 secondary
# pupils in Ngaliema commune declared Lingala their mother tongue "regardless of their ancestral
# language" (Mavita Tseki et al., IJSSMR 9(2)). The only figure found; sources/cd.md §4.
# Set to 0 to draw Kinshasa purely by ethnic group.
KIN_LINGALA = 0.58
KINSHASA = "CD1000"
LINGALA = "nigercongo.bantu.lingala"
N_TERR = 164


def _counts():
    import cd2012
    df = pd.read_csv(NORM / "cd.csv", dtype={"geo_id": str, "province": str})
    ter = pd.read_csv(RD_GEO / "cd" / "cd_territoires.csv", dtype={"territoire": str})
    pop = ter.set_index("territoire")["codps_2024"]
    if df["geo_id"].nunique() != N_TERR or set(df["geo_id"]) != set(pop.index):
        raise SystemExit("cd: cd.csv territoires differ from religiondots' cd_territoires.csv; "
                         "re-run sources/cd_e123.py")
    df["node"] = [cd2012.resolve(l, p) for l, p in zip(df["source_category"], df["province"])]
    if df["node"].isna().any():
        raise SystemExit(f"cd: unmapped {sorted(df.loc[df.node.isna(), 'source_category'].unique())}")
    # heads with no ethnic group recorded are not drawn (`gap`)
    df["count"] = df["share"] * (1 - df["nodata_share"]) * df["geo_id"].map(pop)
    kin = df["geo_id"] == KINSHASA
    drawn_kin = df.loc[kin, "count"].sum()
    df.loc[kin, "count"] *= (1 - KIN_LINGALA)
    if KIN_LINGALA > 0:
        df = pd.concat([df, pd.DataFrame([dict(geo_id=KINSHASA, node=LINGALA,
                                               count=KIN_LINGALA * drawn_kin)])])
    df = df.rename(columns={"geo_id": "unit"})
    out = df.groupby(["unit", "node"], as_index=False)["count"].sum()
    out["count"] = out["count"].round().astype("int64")
    out = out[out["count"] > 0]
    out["tier"] = "modelled"
    return out


ENTRY = dict(
    name="DR Congo",
    source=("Enquête 1-2-3, 2005 and 2012 (Institut National de la Statistique), ethnic group of "
            "the household head as tabulated by the U.S. Census Bureau, on OCHA's COD-PS 2024 "
            "territory populations"),
    how="household survey, 2005 and 2012, ethnic group read as language",
    parts=[
        dict(covers="Kinshasa, Lingala",
             source="58% of the city, the share of Kinshasa secondary pupils in one study who "
                    "called Lingala their mother tongue", people=8_232_242),
        dict(covers="Everyone else",
             source="Enquête 1-2-3 2005 and 2012, household head's ethnic group read as "
                    "language, on the 2024 population projection", rest=True),
    ],
    grain="164 territories and cities, 718,000 people on average",
    gap="2.3% of household heads, whose ethnic group was not recorded",
    view=[12.2, -13.5, 31.3, 5.4],
    counts=_counts,
    mappings=["cd2012"],
    place=RD_GEO / "cd" / "cd_hexes.gpkg",
    place_unit=lambda g: g["territoire"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "No census has been taken in the DR Congo since 1984, and no survey publishes a "
        "language table below the whole country. This map is drawn from the ethnic group of "
        "31,755 household heads in the statistics institute's Enquête 1-2-3 of 2005 and 2012, "
        "as the U.S. Census Bureau counted them by territory, with each group read as its own "
        "language and everyone in a household drawn in the head's group. The shares are laid on "
        "the UN humanitarian office's 2024 population projection, so the dots are survey shares "
        "and not a count; 16 territories the survey did not visit take their province's shares. "
        "Reading ethnic group as language leaves out the four national languages wherever "
        "people speak them at home instead of their group's own. Kinshasa is the exception: "
        "58% of the city is drawn as Lingala, the share of secondary pupils in one study who "
        "called it their mother tongue. Lubumbashi, Kisangani, Goma and Bukavu, where Swahili "
        "or Lingala is the first language of many, are drawn by ethnic group alone. A 2010 "
        "UNICEF survey found Swahili 25%, Lingala 18%, Tshiluba 12% and Kikongo 9% as the main "
        "household language nationally, which is how far daily use runs ahead of this map. Ten "
        "percent of heads named a group the survey did not list; they and the Twa are drawn as "
        "other African languages."),
)
