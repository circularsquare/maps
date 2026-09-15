# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


class _BdHexWeighter(_KeHexWeighter):
    """Kenya's hex weighter, on Bangladesh's upazilas — and here for the OPPOSITE reason.

    Ethiopia and Pakistan need it because a handful of enormous desert units would wash half
    the map in one colour. Bangladesh is the most uniform counting geography on this map —
    544 units averaging 258 km² over a country at 1,027 people/km² — and it needs the weight
    anyway, because the few units that are NOT uniform are precisely the ones the country is
    worth drawing for:

      * the **Chittagong Hill Tracts** — Thanchi at 22 people/km², Belai Chhari 27,
        Baghaichhari 60, against a national 1,027 — which hold every Buddhist and
        tribal-Christian dot in Bangladesh (Juraichhari is 94.6% Buddhist, Ruma 38.2%
        Christian). Placed uniformly, the most distinctive geography on the map smears
        evenly across empty forested ridge.
      * the **Sundarbans** — Shyamnagar, Koyra, Mongla and Dacope, the largest units outside
        the Hill Tracts and mostly uninhabited mangrove. Dacope is also the most Hindu
        upazila in the country at 56.5%.

    A POPULATION weight, not a religion one. Nothing measures where Dacope's Hindus sit
    inside Dacope, so every node's dots are spread identically. sources/bd_geo.py has the
    numbers, and the limit at the other end — central Dhaka thanas smaller than a few hexes.
    """

    def summary(self):
        return (f"{self.n_pop:,} (unit, node) rows placed on Kontur hex population, "
                f"{self.n_uniform:,} on equal shares where an upazila's hexes sum to zero "
                f"(sources/bd_geo.py)")


def _bd_place_weight(place):
    """countries.py hook. `place` is the hex layer scatter.py has read."""
    if "pop" not in place.columns:
        print("  !! bd_hexes.gpkg has no `pop` column — run sources/bd_geo.py; "
              "placing on equal shares (§8.2)")
        return None
    return _BdHexWeighter(place)


def _bd_counts():
    """Bangladesh 2011 census at upazila: 5 nodes on 544 units.

    ONE level, no allocation, nothing modelled — every row is `measured` and may ring.

    **100% OF THE PUBLISHED TABULATION IS DRAWN, and the source proves it twice over.** The
    five categories sum to each unit's own published `RLG_TPOP` on all 617 rows of the file,
    and the 544 upazilas sum to 144,043,696 — BBS's census population — category by
    category. There is no non-response cell to leave out; see note_public, because that is a
    fact about the tabulation rather than about Bangladesh.

    bd.csv also carries the country, division and zila rows, which are the same people three
    more times; only `upazila` is read.
    """
    from bd2011 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "bd.csv",
                     dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    df = df[df["geo_level"] == "upazila"].copy()
    df["count"] = df["count"].astype(int)
    if df["geo_id"].nunique() != 544:
        raise SystemExit(f"{df['geo_id'].nunique()} upazilas, expected 544 -- re-run "
                         "sources/bd.py")

    df["node"] = df["source_category"].map(resolve)
    df = df[df["node"].notna() & (df["count"] > 0)]
    df = df.rename(columns={"geo_id": "unit"})
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations"]]


ENTRY = {
    "bd": dict(
        name="Bangladesh",
        source="2011 Population and Housing Census (BBS), U.S. Census Bureau tabulation",
        basis="self-identification",
        view=[88.0, 20.6, 92.7, 26.7],
        note_public=(
            "**Bangladesh is 90.4% Muslim, and the reason to look at it is the other "
            "9.6%.** It is the fourth-largest Muslim population in the world — 130.2 "
            "million, more than every Arab country combined — and on 544 upazilas of about "
            "265,000 people each the minorities resolve into three quite separate "
            "geographies rather than a thin national scatter. "
            "**The Hindus are the largest Hindu population anywhere outside India: 12.3 "
            "million, 8.5%.** Larger than Nepal's, and about the size of Ohio. They are "
            "concentrated and the concentration is old — **Dacope upazila is 56.5% Hindu**, "
            "Kotalipara 49.9%, Sulla 47.0% — running in three belts: the southwest of "
            "Khulna division around the Sundarbans, the tea districts of Sylhet, and the "
            "northwest around Dinajpur and Thakurgaon. Read the figure as a moment in a "
            "long decline rather than a steady state: the Hindu share of this territory "
            "was about 22% at partition and roughly 13.5% in 1974. "
            "**The Chittagong Hill Tracts are a Buddhist and tribal-Christian country "
            "inside a Muslim one, and nothing else on this map looks like them.** "
            "Juraichhari is **94.6% Buddhist**, Naniarchar 83.4%, Lakshmichhari 79.4%, and "
            "six Hill Tracts upazilas are majority Buddhist. The Buddhism is Theravada — "
            "the Chakma, Marma and Rakhine, and the Barua of Chattogram plain, who are "
            "among the oldest continuously Buddhist communities in South Asia. Beside it, "
            "in the same hills, is the only place in Bangladesh where Christianity is "
            "visible at all: **Ruma is 38.2% Christian and Thanchi 36.4%**, against 0.31% "
            "nationally — twentieth-century mission ground among the Bawm, Mru and Khumi. "
            "**And the `Other` cell is where the indigenous religions went.** 0.14% "
            "nationally but **15.3% in Ruma** and 7.8% in Thanchi, sitting beside the "
            "Christian and Buddhist peaks rather than instead of them. A five-box question "
            "has nowhere to put Mru, Khyang or Bawm traditional practice, so it lands here. "
            "Treat it as a floor, and as the shape of a category the census does not have. "
            "**Two things the census cannot show.** It offers no cell for Ahmadi Muslims, "
            "who are perhaps 100,000 people and have had mosques sealed and communities "
            "attacked — Pakistan's census counts them separately and this one does not, so "
            "they are inside the 130 million. And it names no Christian denomination, "
            "although roughly two-thirds of Bangladeshi Christians are Catholic. "
            "**The counts are 2011 and there has been a census since.** The 2022 census "
            "counts about 165 million and asked religion again; its upazila tables are not "
            "reachable from here (sources.md §11j). Read the shares as current and the "
            "magnitudes as a decade old."),
        how="census, 2011",
        grain="upazilas, 265,000 people on average",
        counts=_bd_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "bd" / "bd_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_bd_place_weight,
        note="THE COUNTS AND THE BOUNDARIES COME OUT OF THE SAME FILE, the third country "
             "here to do so after Ethiopia (§9u) and Pakistan (§9t). The source is not the "
             "Bangladesh Bureau of Statistics directly: the **U.S. Census Bureau** publishes "
             "BBS's 2011 census tabulations on HDX as a geodatabase with the upazila "
             "boundaries beside them, both keyed on `GEO_MATCH`, so the join is an identity "
             "— **544 polygons, 544 counted units, zero unmatched**. sources.md §11h is the "
             "general finding and §11j is the audit that picked this country out of the "
             "four left in that series. "
             "**AND THE VINTAGES MATCH, WHICH ETHIOPIA'S DID NOT.** Ethiopia is a 2007 "
             "census re-cut onto 2021 woredas, with 418 units carrying a lineage note "
             "saying which census-era unit they came from. Bangladesh's layers are "
             "`BD_GEOG_ADM3_2011` against `..._2011census`, so there is no re-cutting at "
             "all and spec §8.1 is satisfied outright. The tell is that `USCBCMNT` is empty "
             "on every one of the 617 rows — an empty lineage column is what a matched "
             "vintage looks like, and `sources/bd.py` asserts it rather than skipping it "
             "for being blank. "
             "**A perfect partition, checked two ways.** The five categories sum to each "
             "unit's own published `RLG_TPOP` on all 617 rows, and the 544 upazilas sum to "
             "144,043,696 — the published census population — category by category. So "
             "100% of the tabulation is drawn. There is **no non-response cell at all**, as "
             "in Ethiopia and Pakistan; that is a fact about the tabulation and does not "
             "mean nobody refused. "
             "**Neither null convention appears here.** Ethiopia's file uses `-999` as a "
             "sentinel that parses as a number; Pakistan's uses real nulls; this one has "
             "zero of each, so the convention is per FILE rather than per publisher. "
             "`sources/bd.py` asserts both counts at zero, which is a stronger check than "
             "masking defensively would have been. "
             "**Five categories is the shallowest question on this map attached to its "
             "fourth-largest population**, and every one of them maps to a parent or a root: "
             "Muslim to `islam` with no school (Hanafi Sunni, but the census does not say), "
             "Buddhist to `buddhism` and not `.theravada` — the same call lk2024.py makes "
             "for Sri Lanka, and taking it differently here would draw a Theravada boundary "
             "at the Bengal border that is an artefact of two ingest decisions. "
             "Placement is Kontur's 400 m H3 grid, 145,658 hexes weighted by hex "
             "population. Bangladesh is the most uniform counting geography here and needs "
             "the weight anyway, because the units that are not uniform are the Hill Tracts "
             "and the Sundarbans — the two places the minorities live. Kontur 2023 against "
             "a 2011 census is 1.199x, a band derived from twelve years of growth rather "
             "than copied from Ethiopia's or Pakistan's. "
             "**Inland water is NOT free here, unlike Ethiopia.** That file carries lakes "
             "and parks as separate polygons cut out of the woredas; this one has 544 "
             "polygons and 544 counted units, so the delta's rivers are inside them. "
             "`water.py`'s tidal clip reaches far up the Meghna estuary and Kontur is empty "
             "over open channel, which between them handle it. "
             "**And one limit at the small end.** A Kontur r8 hex is about 0.80 km² and "
             "central Dhaka's thanas are 0.8-3 km², so four of them — Adabor, Sutrapur, "
             "Kalabagan, Kotwali, 0.41% of the country — are under 60% covered by hex "
             "centroids and their dots crowd into the covered part. Left alone because the "
             "alternative is an equal share over the same 2 km², which is not better; "
             "reported in the bd_geo.py build log rather than silently accepted.",
    ),
}
