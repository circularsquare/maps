# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _mv_place_weight(place):
    """countries.py hook. `place` is Kontur's hexes cut to COD-AB's islands, every island the 2014
    census counts scaled to its count (sources/mv_geo.py)."""
    return _kontur_place_weight(place, "mv_hexes.gpkg", "sources/mv_geo.py")


def _mv_counts():
    """2014 census: Maldivians on Islam (not asked), foreign residents on their own answers.

    Malé's foreign rows are the census's count (`measured`); the atolls' are the census's count for
    all the atolls together, spread by sex, kind of island and citizenship (`derived`); Maldivians
    are `modelled`. The tier is in data/normalized/mv.csv. sources/mv.py and sources/mv.md.
    """
    from mv2014 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "mv.csv", dtype={"geo_id": str},
                     keep_default_na=False, na_values=[""])
    df["node"] = df["source_category"].map(resolve)
    known = {"Not Stated"}
    unmapped = sorted(set(df.loc[df["node"].isna(), "source_category"]) - known)
    if unmapped:
        raise SystemExit(f"mv.csv categories with no node: {unmapped}")
    df = df[df["node"].notna() & (df["count"] > 0)].copy()
    df["unit"] = df["geo_id"]
    if df["unit"].nunique() != 21:
        raise SystemExit(f"{df['unit'].nunique()} units, expected 21")
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations", "tier"]]


ENTRY = {
    "mv": dict(
        name="Maldives",
        source="Maldives Bureau of Statistics, 2014 Population and Housing Census: foreign "
               "residents' religion for Malé and the atolls (UN Demographic Yearbook), and residents "
               "by citizenship, sex, atoll and island; Pew Research Center's 2020 estimates to share "
               "the atolls' answers between citizenships",
        basis="foreign residents' self-identification; Maldivians drawn as Muslim, which nobody "
              "asked",
        note_public=(
            "**Maldivians are not asked their religion.** The 2014 census form skips the question "
            "for Maldivian citizens, and the constitution says a non-Muslim may not become a "
            "citizen, so the **338,434** Maldivians the census counted are all drawn as Muslim. "
            "Nobody counted these dots, so they disappear when inferred dots are turned off. "
            "Maldivians who "
            "are not Muslim are not on this map, because no source counts them. The 2022 census "
            "published no table on religion, so the map shows 2014. "
            "**The census did ask the 63,637 foreign residents.** Their answers are published for "
            "Malé and for all the atolls together, by sex: 39,217 Muslim, 10,163 Hindu, 6,403 "
            "Christian, 5,337 Buddhist, 1,289 another religion and 1,228 who did not say. Most "
            "were born in Bangladesh (37,003), India (13,076) or Sri Lanka (6,722), and 23,110 "
            "were counted on resort and industrial islands. In Malé the dots are the census's own "
            "count: **8,524** of its 24,523 foreign residents are not Muslim. "
            "**Each atoll's foreign residents are shared out from the atolls' total.** The census "
            "counts each atoll's foreign residents by sex and country of birth, for its inhabited "
            "islands and its resort and industrial islands, but their religion only for the atolls "
            "together. So that total is shared out by citizenship and sex, with each citizenship's "
            "religion fitted to the census's totals starting from Pew Research Center's 2020 "
            "figures for its home country. Starting every citizenship from the same mix instead "
            "would put about 800 of the atolls' 14,668 foreign residents who are not Muslim on a "
            "different atoll."),
        how="census, 2014, which asked foreign residents only; Maldivians drawn as Muslim",
        grain="atolls and Malé, 19,000 people on average",
        fill="from the census's count for all the atolls together",
        gap="the 1,228 foreign residents who did not state a religion, 0.31% of residents",
        gap_share=0.003054,
        counts=_mv_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "mv" / "mv_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_mv_place_weight,
        note="REOPENED 2026-10-03 (fafd1067-mv) on the Mauritania construction (ask/RULINGS.md "
             "2026-09-15/16). sources/mv.md is the record. NATIONALS: the 2014 form skips M7 for "
             "Maldivians; constitution Art. 9(d); all 338,434 on islam, modelled. FOREIGNERS: UNSD "
             "table 28 2014 Urban (= Malé, 153,904) and Rural by sex, less PP3's Maldivians: Malé "
             "measured; the atolls' 39,114 spread over 20 atolls by an IPF of sex x island kind x "
             "citizenship x answer to MG15 and the census margins (Pew 2020 seed, others from UN "
             "DESA 2015), applied to PP3 (inhabited islands) and MG14 scaled to MG15 (resort and "
             "industrial islands), derived. Flat seed moves 795 of 14,668 non-Muslims. GEOGRAPHY: "
             "COD-AB MDV 21 units joined on the census's atoll letters, witnessed by PP5 island "
             "names. PLACEMENT: Kontur MV cut to COD's 1,556 islands; 187 inhabited islands and "
             "Malé's three scaled to PP5's 2014 counts, the rest of each atoll on Kontur. NOT DRAWN: "
             "not stated (gap_share), non-Muslim Maldivians, foreigners the census missed (no "
             "opened source sizes them; sources/mv.md).",
    ),
}
