# Colombia. CNPV 2018, speakers of their own people's language per municipio and zone
# (sources/co_cnpv.py); everyone else drawn as Spanish (spec §3.5). Placed on Kontur hexes split
# per municipio into resguardo, cabecera and rural zones (sources/co_geo.py).
from _shared import *  # noqa: F401,F403
import numpy as np

SPANISH = "indoeuropean.romance.spanish"


def _counts():
    import co2018
    df = pd.read_csv(NORM / "co.csv", dtype={"geo_id": str})
    lut = pd.read_csv(GEO / "co" / "co_zones.csv", dtype=str)
    df = df.merge(lut, on=["geo_id", "zone"], how="left")
    if df["unit"].isna().any():
        raise SystemExit(f"co: {df['unit'].isna().sum()} rows with no unit in co_zones.csv; "
                         "re-run sources/co_geo.py")
    df["node"] = df["code"].map(co2018.resolve)
    df = df[df["node"].notna()].copy()       # 1010, 1011: not drawn, in `gap`
    df["tier"] = np.where(df["node"] == SPANISH, "derived", "measured")
    df = df.groupby(["unit", "node", "tier"], as_index=False)["count"].sum()
    return _immigrants(df, lut)


# 2026-10-05 (session edd42a8c-lats, sources/co.md "Immigrant languages"): the foreign-born by
# country of birth per municipio (sources/co_immig.py) on their origin's languages (origin_mix),
# retained at France's TeO2 rate (sources/latam_immig.py), taken out of the Spanish remainder.
# The census counts them per municipio, not per zone, so each municipio's are shared over its
# resguardo, town and countryside zones in proportion to the zones' Spanish remainder.
def _immigrants(df, lut):
    sys.path.insert(0, str(ROOT / "sources"))
    import latam_immig
    if latam_immig.active():          # an origin's home mix for another country's build
        return df
    imm = pd.read_csv(NORM / "co_immig.csv", keep_default_na=False)     # "NA" is Namibia
    sp = df[df["node"] == SPANISH][["unit", "count"]].copy()
    sp["geo_id"] = sp["unit"].map(dict(zip(lut["unit"], lut["geo_id"])))
    sp["share"] = sp["count"] / sp.groupby("geo_id")["count"].transform("sum")
    imm = imm.rename(columns={"unit": "geo_id"}).merge(sp[["geo_id", "unit", "share"]],
                                                      on="geo_id")
    imm["count"] = imm["count"] * imm["share"]
    rows = latam_immig.immigrant_languages(imm[["unit", "iso", "count"]], SPANISH, "co")
    out, _ = latam_immig.fold_into(df, rows, SPANISH)
    return out


class _CoWeighter:
    """Inside a zone, dots go where Kontur puts people, except in a resguardo zone, where half
    the weight is Kontur's and half is spread evenly over the zone's hexes. Kontur holds far
    fewer people inside ANT's resguardo polygons than the census counts there (Cauca 2.7% of
    the department against 19.1%; sources/co_geo.py prints it per department), so Kontur alone
    would pile a resguardo's people onto its one village hex. Every zone's total is the
    census's either way."""

    def __init__(self, place):
        self.pop = place["pop"].to_numpy(dtype=float)
        self.r = (place["zone"].astype(str) == "r").to_numpy()
        self.n = {"kontur": 0, "resguardo": 0, "uniform": 0}

    def weights(self, node, idx, count, plain=False):
        p = self.pop[idx]
        if self.r[idx].all() and len(idx):
            self.n["resguardo"] += 1
            s = p.sum()
            return (0.5 * p / s if s > 0 else 0) + 0.5 / len(idx)
        if p.sum() > 0:
            self.n["kontur"] += 1
            return p
        self.n["uniform"] += 1
        return None

    def summary(self):
        return (f"{self.n['kontur']:,} rows on Kontur, {self.n['resguardo']:,} resguardo rows "
                f"half Kontur half even, {self.n['uniform']:,} on equal shares")


def _weight(place):
    return _CoWeighter(place)


ENTRY = dict(
    name="Colombia",
    source="Censo Nacional de Población y Vivienda 2018 (DANE), tabulated on DANE's REDATAM "
           "server; resguardo polygons from the Agencia Nacional de Tierras",
    how="census, 2018, speaks the native language of their own people; people born abroad "
        "drawn by their birth country's languages, less the share France's TeO2 survey finds "
        "speaking only the host language at home; everyone else drawn as Spanish",
    parts=[
        dict(covers="Indigenous, Rrom, Raizal and Palenquero languages",
             source="2018 census, people who speak their own people's language",
             people=862_420),
        dict(covers="People born abroad, languages other than Spanish",
             source="2018 census, country of birth, drawn on that country's languages; about "
                    "a quarter moved to Spanish by France's TeO2 survey", people=37_729),
        dict(covers="Everyone else", source="drawn as Spanish", rest=True),
    ],
    grain="1,122 municipios, 39,000 people on average, each split into resguardo, town and "
          "countryside",
    gap="33,342 people of an ethnic group who speak another native language the census does "
        "not name (17,988) or did not answer (15,354)",
    view=[-79.1, -4.3, -66.8, 12.6],
    counts=_counts,
    mappings=["co2018"],
    place=GEO / "co" / "co_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=_weight,
    note_public=(
        "The 2018 census asked about language only of people who identify with an indigenous "
        "people, the Rrom, the Raizales of San Andrés or the Palenqueros of San Basilio, and "
        "asked whether they speak their own people's language, without naming it. So each "
        "speaker is drawn on the language of the people they named, and everyone else as a "
        "Spanish speaker, except people born abroad, who are drawn by the languages of "
        "their birth country. Nearly nine in ten of them were born in Venezuela and speak "
        "Spanish; of the rest, about a quarter are drawn as Spanish speakers, the share of "
        "immigrants in France who speak only French with their children. It is a question "
        "about speaking, not about which language came first. Some peoples whose ancestral "
        "language is no longer in everyday use also answered yes, among them the Zenú, "
        "Pastos, Pijao, Yanacona and Mokaná; they are drawn as the census recorded them, but "
        "are probably people who know some of a heritage or revived language. Inside each "
        "municipio, the census counts people in resguardos, in the town and in the "
        "countryside separately, and the dots follow that split."),
)
