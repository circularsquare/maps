# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _lu_place_weight(place):
    """Kontur's hexes cut by the 102 communes, each piece scaled so its commune holds the census
    count of 8 November 2021 (sources/lu_geo.py). A population weight, not a religion one."""
    return _kontur_place_weight(place, "lu_grid_400m.gpkg", "sources/lu_geo.py")


def _lu_counts():
    """One national mix on the 2021 census's 102 communes.

    EVERY ROW IS `modelled` (§7b). The mix is the mean of two national surveys of all residents,
    EVS 2020/21 (STATEC Regards 03/23) and TNS Ilres 2022 for AHA, and no survey publishes
    religion below the country, so each commune's people take the national shares; the communes
    only place the dots. sources/lu.py and sources/lu.md.
    """
    from lu2021 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "lu.csv", dtype={"geo_id": str},
                     keep_default_na=False, na_values=[""])
    if df["geo_id"].nunique() != 102:
        raise SystemExit(f"lu.csv has {df['geo_id'].nunique()} communes, expected 102")
    df["node"] = df["source_category"].map(resolve)
    unmapped = sorted(set(df.loc[df["node"].isna(), "source_category"]))
    if unmapped:
        raise SystemExit(f"lu.csv has unmapped source categories: {unmapped}")
    df = df[df["count"] > 0].copy()
    df["congregations"] = 0
    df["tier"] = "modelled"
    return df.rename(columns={"geo_id": "unit"})[
        ["unit", "node", "count", "congregations", "tier"]]


ENTRY = {
    "lu": dict(
        name="Luxembourg",
        name_in="Luxembourg",
        source="European Values Study 2020/21 (STATEC) and TNS Ilres 2022 for AHA, on the 2021 "
               "census (STATEC)",
        basis="self-identification, sample survey of residents of every nationality",
        note_public=(
            "**The census has not asked about religion in Luxembourg since 1970**, and a 1979 law "
            "keeps the question off it. This map uses the two most recent national surveys that "
            "asked residents of every nationality: the European Values Study, carried out for the "
            "University of Luxembourg in late 2020 and early 2021 and published by STATEC, and a "
            "poll of **515** residents by TNS Ilres for the humanist association AHA in March 2022. "
            "They disagree: 48% belonged to a religion in the first and 59% in the second. The map "
            "draws the mean of the two, **46.1%** Catholic, **46.5%** no religion, 2.6% Protestant, "
            "2.1% Muslim and 2.7% another religion. Children are drawn at the adults' shares. "
            "Nobody counted these dots, so they disappear when inferred dots are turned off. "
            "**Every commune is drawn at the same mix.** Neither survey published religion for any "
            "part of the country, so the 102 communes of the 2021 census only decide where the "
            "dots go. 47.2% of the people the census counted were foreign citizens. Drawing them "
            "by nationality, as Belgium and Sweden are drawn, does not work here: on the national "
            "make-up of their home countries, the foreigners alone would hold more Muslims "
            "(**4.4%** of Luxembourg) than either survey found in the whole population. "
            "**Older surveys found far more belonging.** The European Social Survey found 72% in "
            "2002-04 and the European Values Study 75% in 2008. They are not used because so much "
            "has changed since."),
        how="two surveys, 2020 to 2022, one national mix",
        grain="the whole country, 644,000 people",
        counts=_lu_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "lu" / "lu_grid_400m.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_lu_place_weight,
        note="BUILT ON ANITA'S ONE-NATIONAL-MIX RULINGS (cu, kp, er, km, sy: drawn on the best "
             "national figure with the method said). sources/lu.md is the record, sources.md "
             "§lu-2026-10-03. NO CENSUS RELIGION SINCE 1970. LEVEL: mean of EVS 2020/21 (STATEC "
             "Regards 03/23: 48% belong; Catholic 85.3%, Christian 92%, Muslim 2.7% of those who "
             "belong; not in the EVS 2017 integrated or joint v5.0 release) and TNS Ilres 2022 for "
             "AHA (515 residents 16+, belong 59: Catholic 53, Muslim 3, Protestant 2, other 3; "
             "parts scaled 61 to 59), each normalised to 100. FOREIGN HALF TESTED AND REJECTED: "
             "Pew origin compositions by nationality (census DF_B1625 x geoportail.lu shares x "
             "cens_21ctz_r3) put 9.4% of foreigners Muslim, 4.43% of the country, against 1.30 "
             "and 2.90 measured and Pew's own 1.83; the citizen residual goes negative for "
             "Protestant, Muslim and other (pinned in sources/lu.py). ESS 1-2 (2002-04) a witness "
             "only: regionlu one value, no split-half; 30.5% non-citizens sampled against 36.9% "
             "in the 2001 census; Pew matches its Portuguese, French, Italian, Belgian and German "
             "residents on Christian and none but not on Muslims (France 2.5 vs 9.1); its "
             "`Other Christian` is 16% of every Catholic nationality, a card artefact. "
             "PLACEMENT: GISCO LAU 2021 communes, Kontur LU cut by commune (Malta's area share) "
             "and scaled to census totals; neighbour-side border hexes dropped by Natural Earth.",
    ),
}
