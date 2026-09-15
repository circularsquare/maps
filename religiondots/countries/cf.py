# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


class _CfHexWeighter(_KeHexWeighter):
    """Kenya's hex weighter, on the Central African Republic's communes, and CAR is a
    stronger case for it than either Kenya or Ethiopia: the 50 largest communes are 73.0%
    of the country's land and 28.9% of its people. Yalinga is 42,260 km² with 4,768 people
    and Djémah 37,065 km² with 1,845 — 0.11 and 0.05 people per km². An equal share per
    polygon would fill the whole east with an even wash over country that is very nearly
    empty. sources/cf_geo.py has the numbers.

    A POPULATION weight, not a religion one — nothing measures where a commune's Muslims
    sit inside it, so every node's dots are spread identically.
    """

    def summary(self):
        return (f"{self.n_pop:,} (unit, node) rows placed on Kontur hex population, "
                f"{self.n_uniform:,} on equal shares where a commune's hexes sum to zero "
                f"(sources/cf_geo.py)")


def _cf_place_weight(place):
    """countries.py hook. `place` is the hex layer scatter.py has read."""
    if "pop" not in place.columns:
        print("  !! cf_hexes.gpkg has no `pop` column — run sources/cf_geo.py; "
              "placing on equal shares (§8.2)")
        return None
    return _CfHexWeighter(place)


def _cf_counts():
    """CAR RGPH03 2003 at commune: 5 nodes on 177 units.

    ONE level, no allocation, nothing modelled — every row is `measured` and may ring.

    **98.50% of the census is drawn, and the missing 1.50% is not a category.** The five
    cells partition the religion universe of 3,836,736 to within ±2 per row (independent
    rounding, §9at's shape), but that universe is itself 58,403 short of the RGPH03
    population of 3,895,139, which the Ethnicity sheet of the same workbook carries. Those
    people were counted and not asked, or asked and not tabulated; there is no cell for
    them. See note_public — that is a fact about the tabulation rather than about CAR.

    cf.csv also carries the country, prefecture and sous-préfecture rows, which are the
    same people three more times; only `commune` is read.
    """
    from cf2003 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "cf.csv",
                     dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    df = df[df["geo_level"] == "commune"].copy()
    df["count"] = df["count"].astype(int)
    if df["geo_id"].nunique() != 177:
        raise SystemExit(f"{df['geo_id'].nunique()} communes, expected 177 -- re-run "
                         "sources/cf.py")

    df["node"] = df["source_category"].map(resolve)
    df = df[df["node"].notna() & (df["count"] > 0)]
    df = df.rename(columns={"geo_id": "unit"})
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations"]]


ENTRY = {
    "cf": dict(
        name="Central African Republic",
        source="RGPH03 2003 census (ICASEES), U.S. Census Bureau tabulation",
        basis="self-identification",
        view=[14.2, 2.0, 27.6, 11.2],
        note_public=(
            "**The Central African Republic is the most Protestant country on this map.** "
            "52.3% against 29.3% Catholic — a ratio no other country here reaches, and it "
            "is the map of four mission societies that divided the country between them in "
            "the 1920s and never overlapped much afterwards. The census names none of them, "
            "so this is one undivided colour where Malawi next door would show three. "
            "**The Muslim north-east is the sharpest line in the country.** Vakaga, on the "
            "Chad and Sudan borders, is 87.4% Muslim and Bamingui-Bangoran 44.5%, against "
            "2.1% in Nana-Grébizi and 2.3% in Ouham. Ouandja commune is 96.1%. Nationally "
            "Islam is 10.5%, and the top twenty communes out of 177 hold 57% of it. "
            "**This is where the country was in 2003, and that matters more here than "
            "almost anywhere else on this map.** The war that began in 2013 displaced a "
            "large part of the Muslim population of the west and centre — Bangui's PK5 "
            "quarter, Bossangoa, Bouar, Carnot — and much of it never came back. There has "
            "been no census since RGPH03, so nothing newer exists to draw. Read this as the "
            "religious geography of the CAR immediately before that, not as it stands. "
            "**The census offers no traditional religion box at all**, where Ghana, Kenya, "
            "Ethiopia, Malawi and Benin each offer one. The form has five answers: "
            "Catholic, Protestant, Muslim, other religion, no religion. So CAR's "
            "traditional religions are not on this map — not undercounted, absent — and "
            "the two cells that would hold them tell you where they are: **`other "
            "religion` reaches 23.7% in Topia and 21% in Moboma and Baleloko, and `no "
            "religion` peaks in exactly the same communes**, all of them in Lobaye and "
            "Mambéré-Kadéï, the south-western forest, the Aka homeland and Gbaya and "
            "Ngbaka country. Two residuals with one geography, and it is the geography of "
            "the question that was not asked."),
        how="census, 2003",
        grain="communes, 21,700 people on average",
        # `gap` (§6.12): 1.50% of the census is not in the religion table and has no cell.
        gap_share=0.015,
        gap="1.5% of the census, which was not asked, or not tabulated, and has no category",
        counts=_cf_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "cf" / "cf_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_cf_place_weight,
        note="98.50% OF THE CENSUS IS DRAWN AND THE MISSING 1.50% IS NOT A CATEGORY. The "
             "five cells partition a religion universe of 3,836,736; the RGPH03 population "
             "is 3,895,139, which the Ethnicity sheet of the same USCB workbook carries. "
             "So 58,403 people were counted and are not in this table, with no cell "
             "saying so — the data dictionary's wording, `Total population reporting a "
             "religion or belief system`, is the only place it is stated. Per commune the "
             "coverage runs 92.6%–99.9%, median 98.8%, two communes below 95%, which is "
             "evenly-spread non-response rather than a structural hole. Reported, not "
             "filled (§3.5). "
             "THE `Age-Sex` SHEET IS A 2016 ESTIMATE AND IS NOT A DENOMINATOR. Its "
             "national total is 5,052,901, 31% above the census; using it would turn a "
             "98.5%-covered country into a 76%-covered one. Only the Ethnicity sheet is "
             "the right vintage. "
             "THE COUNTS AND THE BOUNDARIES ARE THE SAME VINTAGE, which Ethiopia's were "
             "not. The religion layer keys to `CF_GEOG1_ADM3_2003`, the 2003 set, so there "
             "is no re-cutting to trust and only four communes carry a USCB note at all. "
             "The workbook also ships a 2021 boundary set with 181 communes for its "
             "Population and Displacement tables; sources/cf_geo.py asserts the layer name "
             "so the wrong one cannot be read as a join failure. "
             "THE JOIN IS AN IDENTITY AND IT IS MEASURED: 177 polygons, 177 count rows, "
             "zero keys on either side alone, zero duplicates — §11h's free join, checked "
             "on the sixth country of the series. "
             "THE PARTITION IS EXACT TO ROUNDING AND TWO-SIDED: categories minus total "
             "runs −2..+2 over all 267 rows, 131 of them exact, and the national row is "
             "−1 on 3.8 million. Nothing is one-sided, which is what independently rounded "
             "published figures look like (§9at) rather than a dropped category. "
             "Placement is Kontur's H3 grid, 34,651 hexes weighted by hex population, and "
             "CAR needs it more than Ethiopia did: the 50 largest communes are 73.0% of "
             "the land and 28.9% of the people. "
             "**AND KONTUR IS NOT INDEPENDENT OF THIS CENSUS HERE, WHICH IS NEW.** 78.5% "
             "of communes sit within ±5% of the median Kontur/census ratio, against 34.0% "
             "of Ethiopia's woredas — too tight to be independent modelling. CAR has had "
             "no census since 2003, so Kontur had nothing newer to build on and its "
             "extract is close to a constant rescale of the table being drawn. The ratio "
             "agreeing therefore proves nothing about either source. The grid is still "
             "used, because the thing it is used FOR — where inside a commune the people "
             "are — comes from settlement footprints and is its own measurement; the level "
             "is never read. sources/cf_geo.py measures this on every run.",
    ),
}
