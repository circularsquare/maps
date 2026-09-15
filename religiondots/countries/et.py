# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


class _EtHexWeighter(_KeHexWeighter):
    """Kenya's hex weighter, on Ethiopia's woredas. Same logic, and it is needed for the
    same reason more sharply: the 50 largest woredas are 38.4% of the land, 5.3% of the
    people and 75% Muslim on average, so an equal share per polygon would wash two fifths
    of the country in one colour over the Ogaden. sources/et_geo.py has the numbers.

    A POPULATION weight, not a religion one — nothing measures where a woreda's Orthodox
    sit inside it, so every node's dots are spread identically.
    """

    def summary(self):
        return (f"{self.n_pop:,} (unit, node) rows placed on Kontur hex population, "
                f"{self.n_uniform:,} on equal shares where a woreda's hexes sum to zero "
                f"(sources/et_geo.py)")


def _et_place_weight(place):
    """countries.py hook. `place` is the hex layer scatter.py has read."""
    if "pop" not in place.columns:
        print("  !! et_hexes.gpkg has no `pop` column — run sources/et_geo.py; "
              "placing on equal shares (§8.2)")
        return None
    return _EtHexWeighter(place)


def _et_counts():
    """Ethiopia 2007 census at woreda: 6 nodes on 738 units.

    ONE level, no allocation, nothing modelled — every row is `measured` and may ring.

    **100% OF THE PUBLISHED TABULATION IS DRAWN, and that is not a rounding claim.** The
    six categories partition the census population exactly: the 738 woredas with data sum
    to 73,750,932 category by category, which is the national figure. There is no
    non-response cell to leave out — see note_public, because that is a fact about the
    tabulation rather than about Ethiopia.

    et.csv also carries the country, region and zone rows, which are the same people two
    and three more times; only `woreda` is read.
    """
    from et2007 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "et.csv",
                     dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    df = df[df["geo_level"] == "woreda"].copy()
    df["count"] = df["count"].astype(int)
    if df["geo_id"].nunique() != 738:
        raise SystemExit(f"{df['geo_id'].nunique()} woredas, expected 738 -- re-run "
                         "sources/et.py")

    df["node"] = df["source_category"].map(resolve)
    df = df[df["node"].notna() & (df["count"] > 0)]
    df = df.rename(columns={"geo_id": "unit"})
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations"]]


ENTRY = {
    "et": dict(
        name="Ethiopia",
        source="2007 Population and Housing Census (CSA), U.S. Census Bureau tabulation",
        basis="self-identification",
        view=[32.9, 3.3, 48.1, 15.1],
        note_public=(
            "**Ethiopia is the sharpest religious boundary on this map, and it is a line "
            "of altitude.** The Orthodox highland and the Muslim lowland meet along the "
            "edge of the escarpment and barely mix across it: **Tigray is 95.6% Orthodox "
            "and Amhara 82.5%**, while **the Somali region is 98.4% Muslim and Āfar "
            "95.3%**. That is not a gradient. **104 of the 738 woredas are over 99% one "
            "religion** — 50 Orthodox, 54 Muslim — and there is no other country here "
            "where so much of the map is effectively single-coloured. "
            "**The Ethiopian Orthodox Tewahedo Church is 32.1 million people and it is not "
            "Eastern Orthodox.** It is *Oriental* Orthodox — out of communion with "
            "Constantinople since the Council of Chalcedon in 451, alongside the Copts, "
            "the Armenians and the Syriacs — and it is by a wide margin the largest body "
            "in that communion anywhere. Most maps colour it the same as Greek and Russian "
            "Orthodoxy; this one does not. "
            "**The Protestant south is the fastest-changing thing in the country.** "
            "*Protestant* in Ethiopia means **P'ent'ay** — the evangelical and Pentecostal "
            "churches together, the Mekane Yesus and Kale Heywet above all — and in the "
            "south it is not a minority: **Sidama is 84.4% Protestant, Gambella 70.1%, and "
            "the old SNNPR 48.4%**, against 18.5% nationally. Bensa woreda alone is 92.8%. "
            "This is a twentieth-century mission geography laid over ground the Orthodox "
            "Church never held. "
            "**Oromia is where all three meet.** The largest region, 27.0 million people, "
            "and the only one with no majority at all: **47.6% Muslim, 30.4% Orthodox, "
            "17.7% Protestant**. The most religiously mixed woredas in Ethiopia are all "
            "here or just south of it — Ale is 37% Muslim, 33% Orthodox, 29% Protestant. "
            "**Traditional religion survives in the south-west and the Borana, and nowhere "
            "else.** 2.65% nationally, but **Surima is 96.3%, Hamer 91.3%, Dasenech 81.9% "
            "and Bena Tsemay 74.5%** — the South Omo peoples — and Dire in the Borana "
            "lowlands is 75.6%. Half of everyone counted as Traditional in Ethiopia lives "
            "in 25 of the 738 woredas. **Treat the number as a floor**: the box is "
            "exclusive of the Christian and Muslim ones, and Ethiopian traditional "
            "practice commonly accompanies one of them rather than replacing it. "
            "**Catholics are 0.72% and they have one homeland and one enclave.** Erob, in "
            "the Tigrayan mountains on the Eritrean border, is **40.6% Catholic** and "
            "nothing else in the country is close; the second cluster is in Wolayta "
            "(Damot Pulasa 17.1%, Damot Gale 11.1%). Most of them belong to the Ethiopian "
            "Catholic Church, which is Eastern Catholic of the Ge'ez rite, not Latin. "
            "**Addis Ababa has an internal gradient worth zooming into.** The city is "
            "74.7% Orthodox, but Addis Ketema is **30.6% Muslim** and Kolfe Keraniyo "
            "27.7%, against Yeka's 6.8% — the old merkato quarters against the newer "
            "eastern ones. "
            "**One caution about the residual.** `Other` is 0.64% nationally but reaches "
            "**21.8% in Bore woreda** and 19.1% in Girja, both in Guji, Oromia — a "
            "concentration a six-cell question cannot explain, and most likely "
            "Waaqeffanna, the Oromo traditional religion, being recorded here rather than "
            "under *Traditional*. The census does not say, so it is drawn as it was "
            "published."),
        how="census, 2007",
        grain="woredas, 100,000 people on average",
        counts=_et_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "et" / "et_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_et_place_weight,
        note="THE COUNTS AND THE BOUNDARIES COME OUT OF THE SAME FILE, which has not "
             "happened before here. The source is not the Ethiopian statistical agency: "
             "the **U.S. Census Bureau** publishes Ethiopia's 2007 census tabulations on "
             "HDX as a geodatabase with the boundaries beside them, both keyed on "
             "`GEO_MATCH`, so the join is an identity and there is nothing to verify about "
             "it. sources.md §11b had ranked Ethiopia 2007 the largest untried African "
             "source and priced it at eleven regional PDF volumes on a moved website; none "
             "of that was needed. sources.md §11h is the general finding. "
             "**A perfect partition.** 738 woredas sum to 73,750,932 — the published 2007 "
             "census population — category by category, all six, so 100% of the tabulation "
             "is drawn. **There is no non-response cell at all**, which is unusual (§3.5) "
             "and does not mean nobody refused: it means the 2007 tabulation distributed or "
             "never published one, and nothing here can undo that. "
             "**The sentinel is `-999` and it parses as a number.** Four woredas carry it "
             "in every category; summed naively they remove 23,976 people, 0.03%, small "
             "enough to look like rounding. Masked before summing, which is how the exact "
             "partition appears. Five woredas have no data at all — three in Āfar marked "
             "'Population data not available' by USCB, which is the tell that the 2007 "
             "census itself did not fully enumerate parts of Āfar and Somali. They are "
             "dropped and cost nothing, because the national total excludes them too. "
             "**The counts are 2007 and the boundaries are 2021.** Sidama is a separate "
             "region here and was inside SNNPR in 2007, so USCB's re-cutting is real and "
             "reaches ADM1; at woreda level 418 units carry a note saying which census-era "
             "unit they came out of, and 70 census-era woredas are split across two to "
             "four modern ones. Every count is an integer and the partition is exact, so "
             "nothing is duplicated — but the per-unit split is USCB's work and is not "
             "independently checked here. The note is carried into every row of et.csv. "
             "**The census is 2007 and there has been no successor.** The 2017 census was "
             "postponed four times and abandoned; Ethiopia has not counted itself in "
             "eighteen years, and the population has since roughly doubled. Read this as "
             "the shape of Ethiopian religion, not its current size. "
             "Placement is Kontur's 400 m H3 grid, 422,726 hexes weighted by hex "
             "population, and Ethiopia needs it more than Kenya did: the 50 largest "
             "woredas are 38.4% of the land, 5.3% of the people and 75% Muslim on average. "
             "Kontur is 2023 against a 2007 census, so its total is 1.71x — expected, and "
             "used only as a within-woreda weight. Inland water needed no clipping: the "
             "eleven great lakes, Mago and Nech Sar and the Gambella reserve are separate "
             "polygons in the geodatabase and are cut out of the woredas already.",
    ),
}
