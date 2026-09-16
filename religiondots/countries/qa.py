# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _qa_place_weight(place):
    """countries.py hook. `place` is Kontur's 400m hexes cut to COD-AB's zones.

    Its `pop` is not Kontur's own: each zone's pieces are scaled to that zone's population in the
    2004 census (Table 2), so a municipality's dots follow 2004's zones and Kontur 2023 only says
    where inside a zone. How many dots each municipality gets is Table 6's either way; what this
    changes is the zone inside it. Plain Kontur 2023 would put 16.6% of the dots in a different
    zone, and 30.5% in Mesaieed, whose industrial-area zone held 9 people in 2004 and a fifth of
    the municipality's Kontur people now (sources/qa_geo.py, sources/qa.md §6).
    """
    return _kontur_place_weight(place, "qa_hexes.gpkg", "sources/qa_geo.py")


def _qa_counts():
    """Qatar Census 2004 at the ten municipalities of 2004: 3 nodes on 10 units, all `measured`.

    Table 6 in persons, everyone present on 16 March 2004, labour gatherings included. Qataris
    were enumerated in full; the religion of non-Qataris is the census sample's weighted estimate,
    calibrated to the counted population by municipality and sex. The municipality is the
    sample's stratum and the table is the office's own at that geography, so the rows stay
    `measured` (sources/qa.md §3).
    """
    from qa2004 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "qa.csv",
                     low_memory=False, keep_default_na=False, na_values=[""])
    df["count"] = df["count"].astype(int)
    if df["geo_id"].nunique() != 10:
        raise SystemExit(f"{df['geo_id'].nunique()} municipalities in qa.csv, expected 10 -- "
                         "re-run sources/qa.py")
    df["node"] = df["source_category"].map(resolve)
    unmapped = sorted(set(df.loc[df["node"].isna(), "source_category"]))
    if unmapped:
        raise SystemExit(f"qa.csv categories with no node: {unmapped}")
    df = df[df["count"] > 0].rename(columns={"geo_id": "unit"})
    df["congregations"] = 0
    df["may_ring"] = True
    df["tier"] = "measured"
    return df[["unit", "node", "count", "congregations", "may_ring", "tier"]]


ENTRY = {
    "qa": dict(
        name="Qatar",
        source="Census 2004 (Planning Council, Statistics Department), population Table 6",
        basis="self-identification, total population",
        note_public=(
            "**Qatar's 2004 census published religion for each of its ten municipalities.** "
            "Muslims were **77.5%** of the 744,029 people counted, Christians 8.5% and Other "
            "14.0%. The census published only those three answers, so Other holds Qatar's Hindus "
            "and Buddhists, and anyone with no religion. Other was **46.5%** of Mesaieed and "
            "32.8% of Al Khor, and 89% of the people in it were men. Jeryan Al Batna was 34.0% "
            "Christian, almost all of them men. "
            "**These are 2004 figures.** The 2020 census counted 2,846,118 people, nearly four "
            "times as many, and its published tables have no religion. The map is drawn on the "
            "municipalities of 2004, four of which no longer exist. Inside each one, dots follow "
            "the population of each zone in 2004, and inside a zone, where people live now. "
            "**Qataris were all counted, and everyone else was sampled.** The religion of "
            "non-Qataris comes from a sample of households and labour camps, which the census "
            "weighted up to the number of people counted in each municipality."),
        how="census, 2004; non-Qataris from its sample",
        grain="municipalities of 2004, 74,000 people on average",
        counts=_qa_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "qa" / "qa_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_qa_place_weight,
        note="TABLE 6 OF THE 2004 CENSUS'S POPULATION TABLES is municipality x religion x sex in "
             "persons, read from the Wayback capture of mdps.gov.qa's Pubulation/T06.aspx "
             "(20170627120742, digest pinned) and checked against Tables 1-5 and UNSD table 28 "
             "to the person. Qataris fully enumerated, non-Qataris' characteristics from a "
             "sample calibrated to the counted population by municipality and sex (the census's "
             "introduction page). The Qatari form's religion column codes 1 Muslim, 2 Christian, "
             "3 Other, with no no-religion or not-stated code; Other drawn on other.qa. No "
             "non-response row, so no gap. The ten 2004 municipalities are rebuilt from COD-AB "
             "v02's 91 zones by zone number against Table 2's 87 zones, witnessed by zone names "
             "and by Kontur's rank of the zones (Spearman 0.876); placement weights are 2004 "
             "zone populations spread by Kontur inside each zone. sources/qa.md has the record.",
    ),
}
