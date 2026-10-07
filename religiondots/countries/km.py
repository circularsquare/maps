# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _km_place_weight(place):
    """countries.py hook. `place` is the Kontur 400 m hex layer scatter.py has read.

    Kontur KM has Anjouan's and Mohéli's people swapped (raw 0.18 and 5.84 of their census
    shares), so sources/km_geo.py scales each island's hexes to its 2017 census count; inside each
    island the préfectures then read 0.79 to 1.16 of the census's shares.
    """
    return _kontur_place_weight(place, "km_hexes.gpkg", "sources/km_geo.py")


def _km_counts():
    """Afrobarometer R10's national shares on the 2017 census: 3 islands, one mix.

    EVERY ROW IS `modelled` (§7b). The census asked no religion, and the survey's summary prints
    religion nationally only, so each island's people take the national mix. sources/km.py and
    sources/km.md.
    """
    from km2025 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "km.csv", keep_default_na=False,
                     na_values=[""])
    lut = pd.read_csv(HERE / "data" / "geo" / "km" / "km_lookup.csv", dtype=str)
    df["unit"] = df["geo_id"].map(dict(zip(lut["geo_id"], lut["unit"])))
    missing = sorted(df.loc[df["unit"].isna(), "geo_id"].unique())
    if missing:
        raise SystemExit(f"km.csv islands with no polygon: {missing}; re-run sources/km_geo.py")
    if df["unit"].nunique() != 3:
        raise SystemExit(f"{df['unit'].nunique()} islands, expected 3")
    df["node"] = df["source_category"].map(resolve)
    unmapped = sorted(df.loc[df["node"].isna(), "source_category"].unique())
    if unmapped:
        raise SystemExit(f"km.csv categories with no node: {unmapped}")
    df = df[df["count"] > 0]
    df["tier"] = "modelled"
    df["congregations"] = 0
    # spec §3.10, as Bahrain: a national share spread over the islands cannot put a ring on one of
    # them. No religion (757 people, one respondent) draws no dot and so no ring.
    df["may_ring"] = False
    return df[["unit", "node", "count", "congregations", "tier", "may_ring"]]


ENTRY = {
    "km": dict(
        name="Comoros",
        name_in="the Comoros",
        source="Afrobarometer round 10 (2025), Comores summary of results, on the islands' "
               "populations in the 2017 census (INSEED)",
        basis="self-identification, citizens 18 and over",
        note_public=(
            "**No count of religion in the Comoros has been published below the whole country.** "
            "The 2017 census did not ask the question, and the surveys that do print national "
            "figures only. This map uses Afrobarometer's first survey of the Comoros, which asked "
            "**1,200** adult citizens their religion in May and June 2025: **99.6%** answered "
            "Islam, 0.3% Christianity and 0.1% no religion. "
            "**Every island is drawn at the same mix.** Grande Comore, Anjouan and Mohéli each "
            "take those shares on their population in the 2017 census, 758,316 people in all, so "
            "the map cannot show where any religion is more common. Nobody counted these dots, so "
            "they disappear when inferred dots are turned off. Children and foreign residents are "
            "drawn at the adult citizens' shares. "
            "**The survey does not split Muslims into branches.** Most answered Muslim without "
            "naming one; "
            "1.0% said Sunni and 2.4% Ismaili, and nothing says where either lives. "
            "**Other figures are close.** The 2012 Demographic and Health Survey found 0.3% of "
            "women and men aged 15 to 49 Catholic or Protestant. Pew Research Center's estimate "
            "for 2020 is 98.3% Muslim and 0.5% Christian, with 1.1% in other religions. "
            "Inside each island the dots follow Kontur's population map, scaled to the island's "
            "census count. Mayotte, which the Comoros also claims, is drawn with France."),
        how="survey, 2025, one national mix",
        grain="islands, 253,000 people on average",
        counts=_km_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "km" / "km_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_km_place_weight,
        note="BUILT ON ANITA'S RULINGS (ask/RULINGS.md 2026-09-15 priority holes, 2026-09-16 "
             "Mauritania: a country with no published religion table is drawn on the best survey "
             "or compiler figure, with the method said). sources/km.md is the record. NO CENSUS "
             "RELIGION: RGPH 2017 (NADA DDI-COM-RPGH-2017, 351 variables) and 2003 (556). SURVEY: "
             "Afrobarometer R10 Comoros (1,200 citizens 18+, May-June 2025), summary of results "
             "rev. 1 Nov 2025 p.5 weighted religion line (Christian 0.3, Muslim 99.6, Autre 0.1 = "
             "Q97 Aucune, refused 0.0 dropped); data set unreleased on 2026-10-03, so no island "
             "split. WITNESSES: DHS 2012 Tableau 3.1 (0.3% Catholic/Protestant, women and men "
             "15-49); Pew 2020 98.3 Muslim, 0.51 Christian, 1.06 other. Not drawn: a foreigner "
             "layer (UN DESA's 12,449 migrants, 9,569 Madagascar-born, are a flat back-projection "
             "and likely mostly Comorians born in Madagascar). POPULATION: RGPH 2017 island "
             "counts, INSEED annex Tableau 1 (COVID-19 vulnerability report, 2020). GEOGRAPHY: "
             "COD-AB adm1 by p-code. PLACEMENT: Kontur KM calibrated per island (raw has "
             "Anjouan and Mohéli swapped); préfecture witness 0.79-1.16.",
    ),
}
