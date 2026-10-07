# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _er_place_weight(place):
    """countries.py hook. `place` is the cell layer scatter.py has read.

    Meta's 2020 high-resolution population grid (Data for Good at Meta and CIESIN) binned into
    cells of about 0.75 km2 and scaled to each zoba's population (sources/er_grid.py). Kontur was
    rejected: its blocks at the 46,200/km2 cap put Massawa near 270,000 and Karora near 95,000.
    """
    return _kontur_place_weight(place, "er_cells.gpkg", "sources/er_grid.py")


def _er_counts():
    """EPHS 2010's national shares on the UN's 2020 estimate: 6 zobas, one mix.

    EVERY ROW IS `modelled` (§7b). Eritrea has never held a census, and the survey report prints
    religion nationally only, so each zoba's people take the national mix of women's and men's
    answers. sources/er.py and sources/er.md.
    """
    from er2010 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "er.csv",
                     dtype={"geo_id": str}, keep_default_na=False, na_values=[""])
    lut = pd.read_csv(HERE / "data" / "geo" / "er" / "er_lookup.csv", dtype=str)
    df["unit"] = df["geo_id"].map(dict(zip(lut["geo_id"], lut["unit"])))
    missing = sorted(df.loc[df["unit"].isna(), "geo_id"].unique())
    if missing:
        raise SystemExit(f"er.csv zobas with no polygon: {missing}; re-run sources/er_geo.py")
    if df["unit"].nunique() != 6:
        raise SystemExit(f"{df['unit'].nunique()} zobas, expected 6")
    df["node"] = df["source_category"].map(resolve)
    unmapped = sorted(df.loc[df["node"].isna(), "source_category"].unique())
    if unmapped:
        raise SystemExit(f"er.csv categories with no node: {unmapped}")
    df = df[df["count"] > 0]
    df["tier"] = "modelled"
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations", "tier"]]


ENTRY = {
    "er": dict(
        name="Eritrea",
        source="Eritrea Population and Health Survey 2010 (National Statistics Office and Fafo, "
               "final report Table 3-1), against the UN's World Population Prospects 2024 estimate "
               "for 2020, divided between the zobas in the survey's own proportions",
        basis="self-identification, women and men 15 to 49",
        note_public=(
            "**Eritrea has never held a census.** Three national health surveys have asked people "
            "their religion, in 1995, 2002 and 2010, and each printed the answers for the whole "
            "country only. This map uses the latest, the Eritrea Population and Health Survey 2010, "
            "which asked **30,224 women and 4,299 men** aged 15 to 49. Their answers are combined "
            "in the proportion the surveyed households held, 36.7% men; many men were away in "
            "national service or abroad. Nobody counted these dots, so they disappear when "
            "inferred dots are turned off. Children and older people are drawn at the shares of "
            "those aged 15 to 49. "
            "**Every zoba is drawn at the same mix.** Nothing published says where any religion "
            "is more common than another, so the map cannot show it. Each zoba is drawn **57.39%** "
            "Eritrean Orthodox, **37.18%** Muslim, 4.36% Catholic, 0.87% Protestant and 0.20% "
            "traditional religion. The survey had no box for no religion. "
            "**Other estimates put Muslims near half.** Pew Research Center's estimate for 2020 "
            "is 51.7% Muslim and 46.7% Christian. Pew takes it from the World Religion Database, "
            "which works from the size of each ethnic group rather than from asking people, and "
            "the US State Department's 2023 report on religious freedom says there are no "
            "reliable figures on religious affiliation. The 2002 survey found 36.5% of women "
            "Muslim, and the 2010 survey 39.0% of women and 33.9% of men. Children, nearly half "
            "the population, were not asked. "
            "**Only four religious bodies are registered.** They are the Eritrean Orthodox Church, "
            "Sunni Islam, the Catholic Church and the Evangelical Lutheran Church. The State "
            "Department reports "
            "that members of other churches risk arrest for worshipping, so some of them may not "
            "have given their religion, and the Protestant share may be low. "
            "**The population is an estimate.** The dots add up to the UN's figure for 2020, "
            "**3,291,271**, divided between the zobas in the proportions the 2010 survey found; "
            "the US government's estimate for 2023 is 6.3 million. Inside each zoba the dots "
            "follow Meta's 2020 population map."),
        how="survey, 2010, one national mix",
        grain="zobas, 549,000 people on average",
        counts=_er_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "er" / "er_cells.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_er_place_weight,
        note="BUILT ON ANITA'S RULINGS (ask/RULINGS.md 2026-09-15 priority holes, 2026-09-16 "
             "Mauritania: a country with no published religion table is drawn on the best survey "
             "or compiler figure, with the method said). sources/er.md is the record; ask 053 "
             "(spec 14). NO CENSUS EVER. SURVEY: EPHS 2010 final report Table 3-1 (national; "
             "Women ALL 30,224 and Men 15-49 4,299, weighted), combined at Table 2-1's household "
             "15-49 sex split (36.7% men; even split would give Muslim 36.51 not 37.18). No table "
             "crosses religion with zoba or ethnicity in EDHS 1995, 2002 or EPHS 2010; DHS lists "
             "no Eritrean microdata. WITNESSES: EDHS 2002 women Muslim 36.5 (asserted within 3.5 "
             "of 2010's 39.0); Pew 2020 51.7 Muslim = World Religion Database (Pew 2025 Appendix "
             "A), ethnic ascription, not drawn. POPULATION: UN WPP 2024 for 2020 (3,291,271, via "
             "Pew's file), zoba shares from EPHS 2010 Table 2-16 (asserted within 1 point of the "
             "2010 frame, Table A-3). GEOGRAPHY: COD-AB v01, 6 zobas by p-code. PLACEMENT: Meta "
             "HRSL 2020 binned to 0.008 deg cells, scaled per zoba; Kontur ER rejected (cap blocks "
             "at Massawa, Ghinda, Karora; Maekel 0.45, Semenawi Keih Bahri 2.68 of the survey share).",
    ),
}
