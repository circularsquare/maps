# Zimbabwe. 2022 census mother tongue by province (ZIMSTAT main report, Table 2.17; persons aged
# 3+), every row `measured` (sources/zw_census.py). On religiondots' Kontur hexes for its 10
# provinces, with each hex's district added (sources/zw_place.py). Inside a province, each
# language's dots lean towards the districts where Afrobarometer respondents (R4, R6, R7, R9)
# named it (sources/zw_afro.py); the counts stay the census's. The record is sources/zw.md.
from _shared import *  # noqa: F401,F403

UNITS = 10
TOTAL = 13_913_253
CENSUS_UNIT = {"ZW01": "ZW10", "ZW02": "ZW11", "ZW03": "ZW12", "ZW04": "ZW13", "ZW05": "ZW14",
               "ZW06": "ZW15", "ZW07": "ZW16", "ZW08": "ZW17", "ZW09": "ZW18", "ZW10": "ZW19"}


def _counts():
    import zw2022
    df = pd.read_csv(NORM / "zw.csv", dtype={"geo_id": str})
    if df["geo_id"].nunique() != UNITS or int(df["count"].sum()) != TOTAL:
        raise SystemExit(f"zw.csv: {df['geo_id'].nunique()} provinces, "
                         f"{int(df['count'].sum()):,} people -- re-run sources/zw_census.py")
    # the census geo_id (religiondots' zw_lookup.csv) -> the hex unit, checked against it
    lut = pd.read_csv(RD_GEO / "zw" / "zw_lookup.csv", dtype=str)
    if dict(zip(lut["geo_id"], lut["unit"])) != CENSUS_UNIT:
        raise SystemExit("religiondots' zw_lookup.csv no longer matches CENSUS_UNIT")
    df["unit"] = df["geo_id"].map(CENSUS_UNIT)
    df["node"] = df["source_category"].map(zw2022.resolve)
    missing = sorted(set(df.loc[df["node"].isna(), "source_category"]))
    if missing:
        raise SystemExit(f"zw.csv answers with no node: {missing}")
    df = df[df["count"] > 0]
    out = by_unit(df)
    out["tier"] = "measured"
    return out


class _ZwDistrictWeighter:
    """Inside a province, a language's dots go to each hex in proportion to its Kontur
    population times that language's share in the hex's district (data/normalized/
    zw_district.csv: Afrobarometer respondents per district, shrunk towards the census's own
    province share with 8 respondents' weight). A placement weight only."""

    def __init__(self, place):
        import zw2022
        self.dist = place["district"].astype(str).to_numpy()
        self.unit = place["unit"].astype(str).to_numpy()
        self.pop = place["pop"].to_numpy(dtype=float)
        t = pd.read_csv(NORM / "zw_district.csv", dtype=str)
        t["share"] = t["share"].astype(float)
        t["node"] = t["source_category"].map(zw2022.resolve)
        s = t.groupby(["unit", "node", "district"])["share"].sum()
        self.share = {}
        for (u, n), x in s.groupby(level=[0, 1]):
            self.share[(u, n)] = x.droplevel([0, 1]).to_dict()
        self.n = {"district": 0, "pop": 0, "none": 0}

    def weights(self, node, idx, count, plain=False):
        s = self.share.get((self.unit[idx[0]], node))
        if s is not None:
            w = pd.Series(self.dist[idx]).map(s).fillna(0.0).to_numpy() * self.pop[idx]
            if w.sum() > 0:
                self.n["district"] += 1
                return w
        p = self.pop[idx]
        if p.sum() > 0:
            self.n["pop"] += 1
            return p
        self.n["none"] += 1
        return None

    def summary(self):
        return (f"{self.n['district']:,} (province, language) rows placed by the survey's "
                f"district shares, {self.n['pop']:,} on population, {self.n['none']:,} on "
                "equal shares")


def _weight(place):
    if "district" not in place.columns:
        raise SystemExit("zw_hexes.gpkg has no `district` column: run sources/zw_place.py")
    return _ZwDistrictWeighter(place)


ENTRY = dict(
    name="Zimbabwe",
    source=("2022 Population and Housing Census Report, Table 2.17, mother tongue by province "
            "(ZIMSTAT); placement inside provinces from Afrobarometer rounds 4, 6, 7 and 9"),
    how="census, 2022, mother tongue, aged 3 and over, by province",
    parts=[dict(covers="Everyone aged 3 and over",
                source="2022 census, mother tongue by province; placed by district from about "
                       "6,000 Afrobarometer respondents (2009-2022)",
                rest=True)],
    grain=("10 provinces, 1.4 million people on average; inside a province, placed by district "
           "(65) from survey respondents"),
    gap="children under 3, 1,265,704 people, who are not in the table",
    view=[24.8, -22.6, 33.4, -15.4],
    counts=_counts,
    mappings=["zw2022"],
    place=GEO / "zw" / "zw_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=_weight,
    note_public=(
        "Zimbabwe's 2022 census asked each person's mother tongue, the language spoken at home "
        "in early childhood, and ZIMSTAT published it by province only. Shona is the mother "
        "tongue of 81% and Ndebele of 11%; the census counts Shona as one language, so its "
        "dialects (Zezuru, Karanga, Manyika, Korekore) are not drawn apart. Ndau, 18% of "
        "Manicaland, and Kalanga and Nambya are counted separately, though linguists usually "
        "group them with Shona. Inside each province the dots are placed by district from the "
        "Afrobarometer survey, which rests on a few hundred interviews per province, so it "
        "shows roughly where a language is, not exact shares."),
)
