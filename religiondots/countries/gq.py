# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _gq_place_weight(place):
    """countries.py hook. `place` is the 400 m hex layer scatter.py has read.

    The mainland interior is forest, and the districts of Malabo and Bata hold 47% of the country. Kontur's hexes are scaled to each province's 2015 census count; one block at Kontur's
    cap in Malabo is left as Kontur has it (sources/gq_grid.py).
    """
    return _kontur_place_weight(place, "gq_hexes.gpkg", "sources/gq_grid.py")


def _gq_counts():
    """DHS 2011's national shares on the 2015 census count: 7 provinces, one mix.

    EVERY ROW IS `modelled` (§7b). The 2015 census asked religion and published no table of it,
    and the DHS report prints religion nationally only, so each province's people take the
    national mix of women's and men's answers, with Christians split at the government's 2015
    estimate. sources/gq.py and sources/gq.md.
    """
    from gq2011 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "gq.csv",
                     dtype={"geo_id": str}, keep_default_na=False, na_values=[""])
    lut = pd.read_csv(HERE / "data" / "geo" / "gq" / "gq_lookup.csv", dtype=str)
    df["unit"] = df["geo_id"].map(dict(zip(lut["geo_id"], lut["unit"])))
    missing = sorted(df.loc[df["unit"].isna(), "geo_id"].unique())
    if missing:
        raise SystemExit(f"gq.csv provinces with no polygon: {missing}; re-run sources/gq_geo.py")
    if df["unit"].nunique() != 7:
        raise SystemExit(f"{df['unit'].nunique()} provinces, expected 7")
    df["node"] = df["source_category"].map(resolve)
    unmapped = sorted(df.loc[df["node"].isna(), "source_category"].unique())
    if unmapped:
        raise SystemExit(f"gq.csv categories with no node: {unmapped}")
    df = df[df["count"] > 0]
    df["tier"] = "modelled"
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations", "tier"]]


ENTRY = {
    "gq": dict(
        name="Equatorial Guinea",
        source="Demographic and Health Survey 2011 (EDSGE-I, final report, Cuadro 3.1), with "
               "Christians divided at the government's 2015 estimate as quoted by the US State "
               "Department, against each province's people in the 2015 census (Instituto Nacional "
               "de Estadística de Guinea Ecuatorial, Resultados definitivos)",
        basis="self-identification, women and men 15 to 49",
        note_public=(
            "**The 2015 census asked about religion, but published no table of it.** None of its "
            "three volumes of results, preliminary or final, has one. This map uses the 2011 "
            "Demographic and Health Survey instead, which asked **3,575 women and 1,612 men** aged "
            "15 to 49 their religion and printed the answers for the whole country only. Women's "
            "and men's answers are combined in the proportion the census found, 53.3% men, and "
            "applied to each province's people in the 2015 census, **1,225,377**. Nobody counted "
            "these dots, so they disappear when inferred dots are turned off. "
            "**Every province is drawn at the same mix.** Nothing published says where any "
            "religion is more common than another. Children and older people are drawn at the "
            "shares of those aged 15 to 49. "
            "**The survey had one box for Christians.** They are divided here at a government "
            "estimate for 2015, quoted by the US State Department's 2023 report on religious "
            "freedom: 88% Catholic, 5% Protestant, 2% Muslim and 5% other. The estimate does not "
            "say how it was made. So the survey's **94.87%** Christians are drawn as 89.77% "
            "Catholic and 5.10% Protestant. The rest: Muslim **3.81%**, traditional religion "
            "(the survey's word is animist) 0.79%, another religion 0.30%, and no religion 0.23%. "
            "**Most Muslims are probably foreign residents.** In the survey 5.4% of men were Muslim "
            "against 1.9% of women, and the State Department says most Muslims in the country are "
            "migrants from West Africa. The preliminary census results counted 209,611 foreign "
            "residents, 17.1% of the people, three quarters of them men, and most in Litoral, "
            "Bioko Norte and Wele-Nzas, but did not say where they came from. Foreigners were "
            "4.1% of the women and 8.2% of the men the survey interviewed in 2011, so if most "
            "Muslims are foreign, the Muslim share drawn here is probably too low, and Muslims "
            "are more concentrated in those three provinces than this map shows. Pew Research "
            "Center's estimate for 2020 is 88.7% Christian, 4.0% Muslim and 5.0% religiously "
            "unaffiliated. "
            "**The map shows the country as the 2015 census counted it.** The US government "
            "estimated 1.7 million people in 2023."),
        how="survey, 2011, one national share and mix",
        grain="provinces, 175,000 people on average",
        counts=_gq_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "gq" / "gq_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_gq_place_weight,
        note="BUILT ON ANITA'S RULINGS (ask/RULINGS.md 2026-09-15 priority holes, 2026-09-16 "
             "Mauritania: a country with no published religion table is drawn on the best survey "
             "or compiler figure, with the method said). sources/gq.md is the record. NOTHING "
             "PUBLISHED: 2015 form Q9 asks religion; Sintesis (3 pp), Resultados preliminares (16 "
             "pp) and Resultados definitivos (72 pp) all read page by page, no table. SURVEY: DHS "
             "EDSGE-I 2011 report FR271 Cuadro 3.1 (national, women and men 15-49, weighted), "
             "Sin informacion out of the base, women and men combined at census 2015's 53.3% men. "
             "Cristiano split 88:5 at the 2015 government estimate quoted by State Department IRF "
             "2023 (asserted within 4 points of the survey's Christian total; fallback "
             "christianity). Microdata (4 domains) behind a DHS registration: ask. WITNESSES: "
             "Pew 2020 88.7/4.0/5.0 (not drawn; compiler), Factbook = the 2015 estimate. "
             "POPULATION: census 2015 definitive Tabla 2.1, 1,225,377, transcribed from the scan "
             "and checked against Tabla 1.1, Tabla 3.1's 18 districts and the preliminary volume. "
             "GEOGRAPHY: COD-AB v01 7 provinces (no Djibloho; it was Wele-Nzas in 2015); area "
             "witness 0.885-1.067 against preliminary density. PLACEMENT: Kontur GQ 1.397x, every "
             "province 0.90-1.03 of its share, scaled per province; Malabo's cap block left "
             "(133,284 after scaling, under Bioko Norte's urban 272,249). NOT DRAWN: Muslims "
             "placed with foreigners (no nationality by province, no religion by nationality).",
    ),
}
