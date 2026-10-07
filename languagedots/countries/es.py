# Spain. INE's ECEPOV 2021, first language ("lengua inicial") by province
# (sources/es_ecepov.py), on religiondots' 8,131 GISCO municipio polygons. Inside a province
# each language's dots follow sources/es_place.py's municipal weights: immigrant languages by
# the Padron's residents of the nationalities that speak them, regional languages by Eustat,
# EULP and the language laws' zones. The record is sources/es.md.
from _shared import *  # noqa: F401,F403


def _counts():
    import es2021
    df = pd.read_csv(NORM / "es.csv", dtype={"geo_id": str})
    df = df[df["geo_level"] == "province"]
    if df["geo_id"].nunique() != 52:
        raise SystemExit(f"es.csv: {df['geo_id'].nunique()} provinces, expected 52")
    df["node"] = df["source_category"].map(es2021.resolve)
    df = df[df["node"].notna() & (df["count"] > 0)]
    df["unit"] = df["geo_id"]
    # modelled: ECEPOV's single-language cells (a sample survey). derived: combinations shared
    # across their languages, and the splits of "Otra" (es2021's docstring).
    return df.groupby(["unit", "node", "tier"], as_index=False)["count"].sum()


class _EsWeighter:
    """A province's dots for one language go to its municipios by sources/es_place.py's weight
    for that language; with no weight anywhere in the province, by its Spanish-language weight
    (locals), then by GISCO population. Placement only: the counts are ECEPOV's."""

    def __init__(self, place):
        w = pd.read_csv(GEO / "es" / "es_weights.csv", dtype={"muni": str, "unit": str})
        self.muni = place["muni"].astype(str).str.zfill(5).to_numpy()
        self.pop = place["pop"].to_numpy(dtype=float)
        self.w = {n: g.set_index("muni")["weight"] for n, g in w.groupby("node")}
        self.loc = self.w["indoeuropean.romance.spanish"]
        self.n = {"own": 0, "locals": 0, "pop": 0, "none": 0}

    def weights(self, node, idx, count, plain=False):
        m = self.muni[idx]
        for key, s in (("own", self.w.get(node)), ("locals", self.loc)):
            if s is None:
                continue
            v = s.reindex(m).fillna(0.0).to_numpy()
            if v.sum() > 0:
                self.n[key] += 1
                return v
        p = self.pop[idx]
        self.n["pop" if p.sum() > 0 else "none"] += 1
        return p if p.sum() > 0 else None

    def summary(self):
        return (f"{self.n['own']:,} (province, language) rows placed on their own municipal "
                f"weight, {self.n['locals']:,} on locals, {self.n['pop']:,} on population, "
                f"{self.n['none']:,} on equal shares")


def _weight(place):
    if not (GEO / "es" / "es_weights.csv").exists():
        raise SystemExit("data/geo/es/es_weights.csv missing: run sources/es_place.py")
    return _EsWeighter(place)


ENTRY = dict(
    name="Spain",
    source="INE, Encuesta de Características Esenciales de la Población y las Viviendas "
           "(ECEPOV) 2021, first language by province; Padrón 2022 nationality by municipality",
    how="household survey, 2021, first language, two first languages shared between them; "
        "'other' split by nationality for foreign residents",
    parts=[
        dict(covers="Everyone aged 2 and over in private homes",
             source="ECEPOV 2021 survey (about 309,000 dwellings), first language",
             people=44_619_046),
        dict(covers="Foreign residents who answered 'other'",
             source="2022 population register, nationality, drawn on that country's languages",
             people=1_458_495),
        dict(covers="Spanish citizens who answered 'other' in the Balearics, Asturias and "
                    "Melilla",
             source="the excess over the rest of Spain, drawn as Catalan, Asturian and Tarifit",
             rest=True),
    ],
    grain="52 provinces, 888,000 people on average; inside a province, placed by municipality",
    gap="children under 2 and people in communal housing: 1.3 million, 2.7% of the 2022 "
        "population register",
    view=[-18.3, 27.5, 4.5, 43.9],
    counts=_counts,
    mappings=["es2021"],
    place=RD_GEO / "es" / "es_municipios.gpkg",
    place_unit=lambda g: g["unit"].astype(str).str.zfill(2),
    place_weight=_weight,
    note_public=(
        "Spain's census asks nothing about language. These figures come from a survey INE ran "
        "alongside the 2021 census, which asked everyone aged 2 and over which language they "
        "spoke first. INE publishes the answers by province. People who named two first "
        "languages, such as Spanish and Catalan, are drawn half on each. Valencian is drawn "
        "apart from Catalan because the survey names it separately. For foreign residents the "
        "survey's 'other' is divided by nationality, so Chinese, Punjabi or Wolof appear where "
        "the survey only says 'other'. Inside a province, regional languages follow regional "
        "statistics and language-law zones, and immigrant languages follow where each "
        "nationality is registered. Aragonese, Leonese and the Galician of western Asturias are "
        "not drawn: none stands out of 'other' in its province."),
)
