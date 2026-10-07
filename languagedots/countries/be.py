# Belgium. No language question since 1947: Dutch, French and German by language area, BRIO's
# Taalbarometer surveys for Brussels and the Vlaamse Rand, immigrant languages by country of birth
# (Eurostat census 2021), every row derived (sources/be_census.py). Counted by arrondissement,
# placed by commune on each commune's own figures (sources/be_geo.py). The record is sources/be.md.
from _shared import *  # noqa: F401,F403


def _counts():
    import be2021
    df = pd.read_csv(NORM / "be.csv", dtype={"geo_id": str})
    df = df[df["geo_level"] == "nuts3"]
    if df["geo_id"].nunique() != 44:
        raise SystemExit(f"be.csv: {df['geo_id'].nunique()} arrondissements, expected 44")
    df["node"] = df["source_category"].map(be2021.resolve)
    df["unit"] = df["geo_id"]
    return df.groupby(["unit", "node", "tier"], as_index=False)["count"].sum()


class _BeWeighter:
    """Inside an arrondissement, each language's dots go to communes in proportion to that
    commune's own figure for it (be_communes.csv): Dutch, French and German by language area
    and the minority shares of Voeren, Comines-Warneton and Mouscron; the BRIO survey shares in
    Brussels and the Rand; immigrant languages by commune population, Eurostat publishing
    country of birth by arrondissement only. A placement weight: the counts are be.csv's."""

    def __init__(self, place):
        import be2021
        c = pd.read_csv(NORM / "be_communes.csv", dtype={"lau": str})
        c["node"] = c["label"].map(be2021.resolve)
        self.w = c.groupby(["node", "lau"])["count"].sum()
        self.lau = place["lau"].astype(str).to_numpy()
        self.pop = place["pop"].to_numpy(dtype=float)
        self.n = {"commune": 0, "pop": 0, "none": 0}

    def weights(self, node, idx, count, plain=False):
        if node in self.w.index.get_level_values(0):
            s = self.w.loc[node]
            w = s.reindex(self.lau[idx]).fillna(0.0).to_numpy()
            if w.sum() > 0:
                self.n["commune"] += 1
                return w
        p = self.pop[idx]
        self.n["pop" if p.sum() > 0 else "none"] += 1
        return p if p.sum() > 0 else None

    def summary(self):
        n = self.n
        return (f"{n['commune']:,} (arrondissement, language) rows placed on the communes' own "
                f"figures, {n['pop']:,} on population, {n['none']:,} on equal shares")


def _weight(place):
    if not {"lau", "pop"} <= set(place.columns):
        raise SystemExit("be_place.gpkg lacks lau/pop: run sources/be_geo.py")
    return _BeWeighter(place)


ENTRY = dict(
    name="Belgium",
    source="Eurostat census 2021 (population by country of birth, cens_21cob_r3); BRIO "
           "Taalbarometer 5 (Brussels, 2024) and Taalbarometer Rand 2 (2018); INED/INSEE "
           "Trajectoires et Origines 2 (2019-20, home language retention by origin); the "
           "language areas of the 1962-63 language laws",
    how="no language question: Dutch, French or German by the commune's language area; "
        "Brussels and the communes round it from BRIO's home-language surveys; immigrants on "
        "their birth country's languages, less a share moved to the area's language",
    parts=[
        dict(covers="Brussels",
             source="BRIO Taalbarometer 5 (2024), language at home; those with neither French "
                    "nor Dutch drawn on the 2021 census's countries of birth",
             people=1_226_329),
        dict(covers="The 19 communes round Brussels",
             source="BRIO Taalbarometer Rand 2 (2018), language at home",
             people=445_437),
        dict(covers="People born abroad, elsewhere",
             source="2021 census, country of birth, drawn on that country's languages; 28% "
                    "moved to the area's language by France's TeO2 survey",
             people=1_421_748),
        dict(covers="People born in Belgium, elsewhere",
             source="the commune's official language area (language laws of 1962-63)",
             rest=True),
    ],
    grain="44 arrondissements, 263,000 people on average; inside an arrondissement, placed by "
          "commune",
    gap="the 4,958 people whose country of birth the census did not record are drawn on their "
        "area's language",
    view=[2.5, 49.45, 6.45, 51.55],
    counts=_counts,
    mappings=["be2021"],
    place=GEO / "be" / "be_place.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=_weight,
    note_public=(
        "Belgium has not asked about language since the census of 1947, whose results fixed "
        "the language border in 1962-63, so every figure here is an estimate. People born in "
        "Belgium are drawn on the official language of their commune: Dutch in Flanders, "
        "French in Wallonia, German in the nine communes of the German-speaking Community. "
        "Regional languages such as West Flemish, Limburgish and Walloon are drawn as Dutch, "
        "French or German, since no survey counts their speakers. Brussels comes from BRIO's "
        "2024 survey of 1,627 adults, which asked which languages they grew up speaking at "
        "home; those speaking neither French nor Dutch are drawn on the languages of the "
        "capital's foreign-born residents. The 19 communes round Brussels come from BRIO's 2018 "
        "survey of the Flemish periphery. Voeren, Comines-Warneton and Mouscron, on the "
        "language border, are drawn with a share of the other language from older figures. "
        "Immigrants are drawn on their birth country's languages, with people born in Morocco "
        "40% Tarifit, since many of Belgium's Moroccans come from the Rif. Those who speak only "
        "the national language with their children are moved onto their commune's language, "
        "by the share France's Trajectoires et Origines survey found for their region of "
        "origin. Belgian-born children of immigrants are drawn on their commune's language, "
        "though many still speak their parents' language at home. Immigrants are placed by "
        "arrondissement and spread over its communes by population."),
)
