# Curaçao. Census 2023, language spoken most often at home, published as island-wide shares only
# (sources/cw_census.py), on religiondots' Kontur hexes given their geozone and calibrated to the
# 2023 population of each of 60 geozones (sources/cw_geo.py). Spanish, English and other dots are
# placed by where people with a nationality other than Dutch live (Census 2023 Table G-3);
# Papiamentu and Dutch by population. Record: sources/cw.md.
from _shared import *  # noqa: F401,F403

BY_FOREIGN = {"indoeuropean.romance.spanish", "indoeuropean.germanic.english", "other"}


def _counts():
    import cw2023
    df = pd.read_csv(NORM / "cw.csv", dtype={"geo_id": str})
    df = df[df["geo_level"] == "national"]
    if len(df) != 5 or abs(df["count"].sum() - 147_498) > 1:
        raise SystemExit("cw.csv: expected five 2023 rows summing to 147,498")
    df["node"] = df["source_category"].map(cw2023.resolve)
    df["unit"] = "CW"
    out = by_unit(df)
    out["tier"] = "derived"      # published shares (one decimal) times the 147,498 who answered
    return out


class _CwWeighter:
    """Spanish, English and other dots go on each hex's people times its geozone's share with a
    nationality other than Dutch (Census 2023 Table G-3); Papiamentu and Dutch on people. A
    placement weight only: the island's counts are the census's either way."""

    def __init__(self, place):
        self.pop = place["pop"].to_numpy(dtype=float)
        self.foreign = place["foreign"].to_numpy(dtype=float)
        self.n = {"foreign": 0, "pop": 0}

    def weights(self, node, idx, count, plain=False):
        p = self.pop[idx]
        if node in BY_FOREIGN:
            w = p * self.foreign[idx]
            if w.sum() > 0:
                self.n["foreign"] += 1
                return w
        self.n["pop"] += 1
        return p

    def summary(self):
        return (f"{self.n['foreign']} languages placed on people x the geozone's other-nationality "
                f"share, {self.n['pop']} on people")


def _weight(place):
    if "foreign" not in place.columns:
        raise SystemExit("cw_hexes.gpkg has no `foreign` column: run sources/cw_geo.py")
    return _CwWeighter(place)


ENTRY = dict(
    name="Curaçao",
    source="Census 2023, Eerste Resultaten (CBS Curaçao), language spoken most often at home; "
           "Table G-3 (population by geozone and nationality) for placement",
    how=("census, 2023, language spoken most often at home; published for the whole island as "
         "five shares; Spanish, English and other placed where people of other nationalities "
         "live, by geozone"),
    parts=[dict(covers="Everyone",
                source="2023 census, language spoken most often at home, island shares",
                rest=True)],
    grain="the island as one unit, 147,500 people; inside it, placed per geozone (60)",
    gap="8,328 people, 5.3%, who did not answer the question",
    view=[-69.20, 12.02, -68.72, 12.41],
    counts=_counts,
    mappings=["cw2023"],
    place=GEO / "cw" / "cw_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=_weight,
    note_public=(
        "The 2023 census asked everyone which language they speak most often at home. CBS "
        "Curaçao has published the answers only for the whole island, as five shares: "
        "Papiamentu 78.0%, Spanish 8.4%, Dutch 7.9%, English 3.8% and other languages 2.0%. "
        "So the dots say nothing measured about where on the island each language is spoken. "
        "Spanish, English and other dots are placed where people without Dutch nationality "
        "live, geozone by geozone, since most of these speakers were born abroad; Papiamentu "
        "and Dutch dots follow where people live. In the 2011 census the other languages were "
        "mostly Haitian Creole, Chinese, Portuguese, Hindi and Arabic."),
)
