# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _pg_place_weight(place):
    """countries.py hook. `place` is the 400 m hex layer scatter.py has read.

    Western, Gulf and East Sepik are 176,000 km2 of swamp and forest with 11% of the 2024 count;
    with Kontur an empty hex takes no dots (sources/pg_grid.py).
    """
    return _kontur_place_weight(place, "pg_hexes.gpkg", "sources/pg_grid.py")


def _pg_counts():
    """2011 census national shares and each province's largest church, the rest fitted: 22 provinces.

    EVERY ROW IS `modelled` (§7b). PNG printed religion for the nation and one church per
    province; sources/pg.py fits the other churches to the national totals, seeded by the Catholic
    dioceses' 2004 figures and the 2000 census's largest church, on the 2024 count. sources/pg.md.
    """
    from pg2011 import resolve, EXCLUDED

    df = pd.read_csv(HERE / "data" / "normalized" / "pg.csv", dtype={"geo_id": str},
                     keep_default_na=False, na_values=[""])
    df = df[~df["source_category"].isin(EXCLUDED)].copy()
    lut = pd.read_csv(HERE / "data" / "geo" / "pg" / "pg_lookup.csv", dtype=str)
    df["unit"] = df["geo_id"].map(dict(zip(lut["geo_id"], lut["unit"])))
    missing = sorted(df.loc[df["unit"].isna(), "geo_id"].unique())
    if missing:
        raise SystemExit(f"pg.csv provinces with no polygon: {missing}; re-run sources/pg_geo.py")
    if df["unit"].nunique() != 22:
        raise SystemExit(f"{df['unit'].nunique()} provinces, expected 22")
    df["node"] = df["source_category"].map(resolve)
    unmapped = sorted(df.loc[df["node"].isna(), "source_category"].unique())
    if unmapped:
        raise SystemExit(f"pg.csv categories with no node: {unmapped}")
    df = df[df["count"] > 0]
    df["tier"] = "modelled"
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations", "tier"]]


ENTRY = {
    "pg": dict(
        name="Papua New Guinea",
        source="2011 census, national religion table and each province's largest church "
               "(National Statistical Office, 2011 National Report), with the 2000 census's "
               "largest church by province and the Catholic Church's 2004 diocese figures "
               "(Annuario Pontificio, via catholic-hierarchy.org); 2024 census Final Figures for "
               "each province's people",
        basis="self-identification, citizens in private dwellings, 2011",
        note_public=(
            "**Papua New Guinea has never published religion by province.** The 2011 census "
            "asked everyone their church, and the National Statistical Office printed the "
            "answers for the whole country: 95.6% of citizens Christian, 1.4% another religion "
            "and 3.1% not stated, with eleven churches as shares of the Christians, from the "
            "Catholic Church at **26.0%** and the Evangelical Lutheran Church at **18.4%** down "
            "to the Kwato Church at 0.2%. For each province it printed one more figure, the "
            "largest church there and its share. This map draws those numbers and estimates "
            "the rest, so nobody counted these dots province by province, and they "
            "disappear when inferred dots are turned off. "
            "**The largest church in each province is the census's own figure.** Of the "
            "Christians, Morobe's are **67.0%** Lutheran, Bougainville's **68.4%** Catholic, "
            "Northern (Oro)'s **60.6%** Anglican, Milne Bay's **54.9%** United Church, Eastern "
            "Highlands' **39.6%** Seventh-day Adventist and Western's **37.1%** Evangelical "
            "Alliance. The 2000 census "
            "printed the same church as the largest in 17 of its 20 provinces. "
            "**The other churches in each province are an estimate.** Each province's other "
            "Christians are shared among the remaining ten churches so that every church adds "
            "up to its national total, and no church passes the province's largest. Catholics "
            "follow the Catholic Church's own count of its members in each diocese in 2004, "
            "which puts them at 3.5% of Eastern Highlands and 24% of Enga. Where the largest "
            "church in 2000 was a different one, it keeps a larger share: the United Church in "
            "New Ireland, and the other Christian churches in Southern Highlands and Hela. "
            "The remaining churches are split in the same proportions in every province, "
            "because nothing printed says otherwise. So a church that is strong in a "
            "few provinces without being the largest in any of them is drawn too thinly there "
            "and too thickly everywhere else; Lutherans, for instance, are drawn at 5% to 10% "
            "of the people in the Sepik and the island provinces. "
            "**The shares are from 2011 and the people from 2024.** The 2024 census counted "
            "**10,185,363** people and has not published religion yet. The 2011 figures are "
            "for citizens; the few non-citizens are drawn at the same shares."),
        how="census, national shares and each province's largest church; other churches estimated",
        grain="provinces, 463,000 people on average",
        gap="the 3.1% of citizens whose religion was not stated in 2011",
        gap_share=0.03097,           # the Not Stated rows, 315,431 of 10,185,363 (gap_share.py, B rows)
        counts=_pg_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "pg" / "pg_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_pg_place_weight,
        note="REOPENED 2026-10-03 UNDER THE 2026-09-15/16 RULINGS (draw on the best available figure, "
             "method disclosed); was blocked on IPUMS (sources.md 11ab). sources/pg.md is the "
             "record. NATIONAL: 2011 National Report Table 2.4 (Christian 95.6, Non-Christian 1.4, "
             "No religion 0.0, Not stated 3.1) and Figure 2.1 (11 churches, % of Christians, sum "
             "100.2), checked against p33's text layer. PER PROVINCE: Summary Indicators pp28-29, "
             "'Main religion', one church and its share (read as % of Christians, like the "
             "national row it equals), checked against the text layer. FIT: the other ten "
             "churches by IPF to Figure 2.1's totals on 2011 Table 2 populations, a church held "
             "0.1 points under the province's largest (4 cells capped); seeds: Catholic by "
             "diocese 2004 (catholic-hierarchy scpg1; Oro inside Port Moresby archdiocese on its "
             "population; Central split Bereina/Port Moresby), and the 2000 census's largest where "
             "it differs (New Ireland United x3.36, SHP and Hela Other Christian x2.72); all else "
             "uniform. ROWS: shares onto 2024 Final Figures Table 2, all modelled. GEOGRAPHY: COD-AB "
             "ADM1 22, p-codes witnessed by COD-PS 2011 = Table 2's 2011 column. PLACEMENT: Kontur "
             "PG, ratio 1.014, rank witness +0.877; NCD 0.57 and Hela 1.79 of share (within-province "
             "only). OPEN: DHS 2016-18 by province (registration) and the 2000 census Basic Tables "
             "(print, religion by province) would replace the fit; see the ask.",
    ),
}
