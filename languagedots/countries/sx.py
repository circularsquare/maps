# Sint Maarten (the Dutch side). Census 2011, Table F-07, persons in private households by the
# language most spoken in the household, for the whole country (sources/sx_census.py), on
# religiondots' Kontur hexes cut along the eight census regions and calibrated to their 2011
# populations (sources/sx_geo.py). Inside the country each immigrant language's dots go where
# people born in the matching countries live (Census 2011 Table B-18, region by country of birth);
# English by population. Record: sources/sx.md.
from _shared import *  # noqa: F401,F403

# node -> the birthplace column of sx_hexes.gpkg that places it (B-18 groups, sources/sx_geo.py)
BY_BIRTH = {
    "indoeuropean.romance.spanish": "b_hispanic",          # Dominican Republic, Colombia, Venezuela
    "creole.french_based.haitian": "b_haiti",
    "indoeuropean.indoaryan.central.hindi": "b_india",
    "indoeuropean.indoaryan.gujarati.gujarati": "b_india",
    "dravidian.southcentral.telugu": "b_india",
    "sinotibetan.sinitic": "b_china",
    "austronesian.philippine.filipino": "b_philippines",
    "indoeuropean.germanic.continental.dutch": "b_netherlands",
    "creole.portuguese_based.papiamento": "b_abc",         # Aruba, Bonaire, Curacao
    "creole.english_based.sranan": "b_suriname",
    "indoeuropean.romance.french": "b_french",             # France, Guadeloupe
}
BY_POP = {"indoeuropean.germanic.english"}
# everything else (Arabic, Italian, Turkish, German, Portuguese, Swedish, Vietnamese, Hebrew,
# "nigerian", "creole", "other"): all foreign-born


def _counts():
    import sx2011
    df = pd.read_csv(NORM / "sx.csv", dtype={"geo_id": str})
    if len(df) != 26 or df["count"].sum() != 33_160:
        raise SystemExit("sx.csv: expected Table F-07's 26 rows summing to 33,160")
    df["node"] = df["source_category"].map(sx2011.resolve)
    df = df[df["node"].notna()]
    df["unit"] = "SX"
    out = by_unit(df)
    out["tier"] = "measured"
    return out


class _SxWeighter:
    """Each hex piece's people (calibrated to its region's 2011 population) times its region's
    share of people born where the language comes from. A placement weight only: the counts are
    the census's national figures either way."""

    def __init__(self, place):
        self.place = place
        self.pop = place["pop"].to_numpy(dtype=float)
        self.n = {"birth": 0, "foreign": 0, "pop": 0}

    def weights(self, node, idx, count, plain=False):
        p = self.pop[idx]
        if node in BY_POP:
            self.n["pop"] += 1
            return p
        col = BY_BIRTH.get(node, "b_foreign")
        w = p * self.place[col].to_numpy(dtype=float)[idx]
        if w.sum() > 0:
            self.n["birth" if col != "b_foreign" else "foreign"] += 1
            return w
        self.n["pop"] += 1
        return p

    def summary(self):
        return (f"{self.n['birth']} languages placed by birthplace in the matching countries, "
                f"{self.n['foreign']} by all foreign-born, {self.n['pop']} by population")


def _weight(place):
    need = {"pop", "b_foreign", *BY_BIRTH.values()}
    if not need <= set(place.columns):
        raise SystemExit("sx_hexes.gpkg lacks birthplace columns: run sources/sx_geo.py")
    return _SxWeighter(place)


ENTRY = dict(
    name="Sint Maarten",
    source=("Population and Housing Census 2011, Table F-07 (Department of Statistics, STAT); "
            "Table B-18 (population by region and country of birth) for placement; regions from "
            "COD-AB"),
    how=("census, 2011, language most spoken in the household, published for the whole "
         "country only"),
    parts=[dict(covers="Everyone",
                source="2011 census, household's main language; immigrant languages placed "
                       "where people born in matching countries live",
                rest=True)],
    grain="the country as one unit, 32,800 people; inside it, placed per census region (8)",
    gap=("794 people, 2.4%, in households whose language was not reported or in institutions, "
         "who were not asked"),
    view=[-63.16, 17.99, -62.99, 18.08],
    counts=_counts,
    mappings=["sx2011"],
    place=GEO / "sx" / "sx_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=_weight,
    note_public=(
        "The 2011 census asked each household which language it spoke most, and gave that "
        "answer to everyone in it. The answers were published for the whole country only, so "
        "the dots say nothing measured about where each language is spoken. Spanish, Haitian "
        "Creole, Hindi, Dutch and the other immigrant languages are placed where people born "
        "in the matching countries lived in 2011; English follows where people live. The 2022 "
        "census published households' main language only as national shares, close to "
        "2011's."),
)
