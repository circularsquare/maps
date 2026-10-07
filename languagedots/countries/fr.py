# France. No language question in the census: French, plus regional languages from regional
# surveys and immigrant languages from INSEE's immigrants by country of birth (RP 2023), every
# row derived (sources/fr_insee.py, sources/fr_build.py, sources/fr_regional.py). Placed on
# religiondots' communes plus Corsica (sources/fr_geo.py). The record is sources/fr.md.
from _shared import *  # noqa: F401,F403
import numpy as np

REGIONAL = {"Breton", "Gallo", "Basque", "Alsatian", "Lorraine Franconian", "Occitan",
            "Corsican", "Catalan", "Antillean Creole", "Guianese Creole", "Reunion Creole",
            "Shimaore", "Kibushi"}


def _counts():
    import fr2023
    df = pd.read_csv(NORM / "fr.csv", dtype={"geo_id": str})
    df = df[df["geo_level"] == "dep"]
    if df["geo_id"].nunique() != 101:
        raise SystemExit(f"fr.csv: {df['geo_id'].nunique()} départements, expected 101")
    df["node"] = df["source_category"].map(fr2023.resolve)
    df["unit"] = df["geo_id"]
    return df.groupby(["unit", "node", "tier"], as_index=False)["count"].sum()


class _FrWeighter:
    """Inside a département: French on French nationals per commune (INSEE RP 2021 TD_NAT1),
    immigrant languages on foreign nationals, regional and overseas languages on population.
    Basque goes on the Pays Basque's communes weighted by its survey zone's first-language
    share; Occitan in Pyrenees-Atlantiques on the communes outside the Pays Basque (Bearn).
    Overseas the layer is Kontur hexes with no nationality split, so everything falls back to
    population there. A placement weight only: the counts are fr.csv's either way."""

    def __init__(self, place):
        import fr2023
        self.pop = place["pop"].to_numpy(dtype=float)
        self.french = place["french"].to_numpy(dtype=float)
        self.foreign = place["foreign"].to_numpy(dtype=float)
        self.basque = place["w_basque"].to_numpy(dtype=float)
        self.outside_pb = (place["zone"].astype(str) == "").to_numpy()
        self.unit = place["unit"].astype(str).to_numpy()
        self.regional = {fr2023.resolve(l) for l in REGIONAL}
        self.n = {"french": 0, "foreign": 0, "pop": 0, "basque": 0, "none": 0}

    def weights(self, node, idx, count, plain=False):
        unit = self.unit[idx[0]]
        if node == "isolate.basque" and self.basque[idx].sum() > 0:
            self.n["basque"] += 1
            return self.basque[idx]
        if node == "indoeuropean.romance.occitan" and unit == "64":
            w = self.pop[idx] * self.outside_pb[idx]
            if w.sum() > 0:
                self.n["pop"] += 1
                return w
        if node == "indoeuropean.romance.french":
            w, key = self.french[idx], "french"
        elif node in self.regional:
            w, key = self.pop[idx], "pop"
        else:
            w, key = self.foreign[idx], "foreign"
        if w.sum() > 0:
            self.n[key] += 1
            return w
        p = self.pop[idx]
        self.n["pop" if p.sum() > 0 else "none"] += 1
        return p if p.sum() > 0 else None

    def summary(self):
        n = self.n
        return (f"{n['french']:,} (département, language) rows placed on French nationals, "
                f"{n['foreign']:,} on foreign nationals, {n['basque']:,} on the Basque survey "
                f"zones, {n['pop']:,} on population, {n['none']:,} on equal shares")


def _weight(place):
    need = {"pop", "french", "foreign", "w_basque", "zone"}
    if not need <= set(place.columns):
        raise SystemExit("fr_place.gpkg lacks columns: run sources/fr_geo.py")
    return _FrWeighter(place)


ENTRY = dict(
    name="France",
    source="INSEE census 2023 (immigrants by country of birth, population); INED/INSEE "
           "Trajectoires et Origines 2 (2019-20, home language retention by origin); "
           "regional language "
           "surveys (Brittany 2024, Pays Basque 2021, Alsace 2012 and 2022, Moselle 2024, "
           "Occitan 2020, Corsica 2021, Northern Catalonia 2015); overseas, INED/INSEE MFV "
           "2009-10 and INSEE surveys of Guyane and Mayotte",
    how="no language question: immigrants drawn on their birth country's languages, less the "
        "share speaking only French at home; regional and overseas languages from surveys; "
        "everyone else drawn as French",
    parts=[
        dict(covers="People born abroad, languages other than French",
             source="2023 census, country of birth, drawn on that country's languages; about a "
                    "third moved to French by the Trajectoires et Origines 2 survey (2019-20)",
             people=4_272_964),
        dict(covers="Regional languages in mainland France",
             source="regional surveys, 2012-2024 (Brittany, Pays Basque, Alsace, Moselle, "
                    "Occitan, Corsica, Northern Catalonia), applied to adults",
             people=1_415_311),
        dict(covers="Overseas departments: creoles, Shimaore, Kibushi",
             source="INED/INSEE family survey 2009-10; INSEE surveys of Guyane (2019-20) and "
                    "Mayotte (2019), Mayotte census 2017",
             people=1_409_128),
        dict(covers="Everyone else", source="drawn as French", rest=True),
    ],
    grain="101 departments, 680,000 people on average; inside a department, placed by "
          "commune",
    view=[-5.4, 41.3, 9.7, 51.2],
    counts=_counts,
    mappings=["fr2023"],
    place=GEO / "fr" / "fr_place.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=_weight,
    note_public=(
        "France's census has never asked about language, so every figure here is an estimate. "
        "Immigrants are drawn on the languages of their birth country (people born in Algeria "
        "30% Kabyle, in Turkey 6% Kurdish), then the share of their region of origin that "
        "speaks only French with its children in the Trajectoires et Origines survey is moved "
        "to French. That survey covers mainland France only, so immigrants overseas are not "
        "adjusted. French-born children of immigrants are drawn as French. Regional surveys "
        "measure different things: Basque, Catalan and Corsican by first or childhood "
        "language, Breton, Gallo, Alsatian, Moselle Platt and Occitan by speakers who learned "
        "the language from their parents. Occitan is surveyed only in Nouvelle-Aquitaine and "
        "Occitanie, and no survey counts West Flemish, Franco-Provençal or the langues d'oïl "
        "other than Gallo, so all of these are drawn as French elsewhere."),
)
