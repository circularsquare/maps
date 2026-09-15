# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _mz_counts():
    """INE Moçambique, IV RGPH 2017, Quadro 11 of the provincial tables: 7 drawn answers.

    ONE level, no allocation, nothing modelled: every row is `measured` and may ring. INE
    published religion by province and no finer (sources/mz.py lists what was searched), and
    the dots are spread over Kontur hexagons by population inside each province
    (sources/mz_grid.py). `Total` and `Desconhecida` (674,680, 2.5%) resolve to nothing.
    """
    from mz2017 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "mz.csv",
                     dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    df = df[df["geo_level"] == "province"].copy()

    lut = pd.read_csv(HERE / "data" / "geo" / "mz" / "mz_lookup.csv", dtype=str)
    df["unit"] = df["geo_id"].map(dict(zip(lut["geo_id"], lut["unit"])))
    missing = sorted(df.loc[df["unit"].isna(), "geo_id"].unique())
    if missing:
        raise SystemExit(f"mz.csv provinces with no polygon: {missing} -- re-run "
                         "sources/mz_geo.py, the lookup is stale")
    if df["unit"].nunique() != 11:
        raise SystemExit(f"{df['unit'].nunique()} provinces, expected 11")

    df["node"] = df["source_category"].map(resolve)
    df = df[df["node"].notna() & (df["count"] > 0)]
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations"]]


def _mz_place_weight(place):
    """countries.py hook. `place` is the hex layer scatter.py has read.

    Kenya's weighter unchanged. Mozambique's need for it is Niassa: 129,000 km2 whose people
    are on the lakeshore and a few roads, with the Niassa Special Reserve covering much of
    the rest.
    """
    if "pop" not in place.columns:
        print("  !! mz_hexes.gpkg has no `pop` column — run sources/mz_grid.py; "
              "placing on equal shares (§8.2)")
        return None
    return _KeHexWeighter(place)


ENTRY = {
    "mz": dict(
        name="Mozambique",
        source="IV Recenseamento Geral da População e Habitação 2017, Quadro 11 of the "
               "provincial tables (INE Moçambique)",
        basis="self-identification, whole census population",
        note_public=(
            "**Mozambique's census counts the Zionist churches as an answer of their own.** "
            "The *Zione/Sião* box holds **4.2 million** people, 15.6% of the country, and "
            "nearly three quarters of them live in the six provinces from Manica and Sofala "
            "south to Maputo. Inhambane and Gaza are both **39.6%** Zionist; Cabo Delgado is "
            "0.4%. "
            "**Islam is the north.** Niassa is **59.0%** Muslim and Cabo Delgado 52.6%, and "
            "those two with Nampula hold 86% of the country's Muslims. "
            "**The census's no-religion answer is drawn as no religion.** Its questionnaire "
            "words that answer *Sem religião (ateu, animista, agnóstico,...)*, so it could hold "
            "followers of traditional religion too. Surveys that offer both answers find few: "
            "in Afrobarometer's rounds from 2008 to 2023, 0.17% of adults name traditional "
            "religion and 7.6% name none. Those surveys record what people call themselves, not "
            "what they practise. The answer is **32.8%** of Tete and a quarter of Manica and "
            "Sofala, against under 1% of Niassa."),
        how="census, 2017",
        grain="provinces, 2.45m people on average",
        gap="2.5% whose religion was recorded as unknown",
        gap_share=0.0251,
        counts=_mz_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "mz" / "mz_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_mz_place_weight,
        note="INE PUBLISHED RELIGION BY PROVINCE AND NO FINER. The 2019 definitive results "
             "were one set of xlsx tables per province on INE's old Plone site, and Quadro 11 "
             "(religion by residence, age and sex) is in all eleven; the census's own table "
             "list has no religion table by district (sources.md §11w). That site was "
             "replaced in 2022 and every path on it is now a 404, so the twelve files are "
             "read from the Wayback Machine's 2019-11-14 captures (sources/mz.py). The "
             "eleven provinces sum to the national Quadro 11's total to the person and to "
             "each category within 150, and the national table is figure for figure the one "
             "in INE's printed Brochura. UNSD's table 28 has the same total and a different "
             "split, with about half the unknowns; it is not published by province and is "
             "not used. The finer route is INE itself, whose mozdata abstract offers custom "
             "tabulations to any administrative level; the microdata is licensed. "
             "Dots are spread over Kontur 400m hexagons by population inside each province "
             "(sources/mz_grid.py). `Sem religião` includes animists by the questionnaire's "
             "own wording and is drawn as `unaffiliated`, Anita's ruling of the night of "
             "2026-09-14, because Afrobarometer and Pew put it at 93-98% no religion as a "
             "self-description (it was `unknown` from that evening); taxonomy/mz2017.py REVIEW "
             "has why, and sources/mz.md §6 the national sources on the split.",
    ),
}
