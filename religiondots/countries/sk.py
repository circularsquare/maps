# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


class _SkGridWeighter(_KeHexWeighter):
    """Split an obec's dots across the census's OWN 1 km cells by cell population.

    Slovakia is the second country here after Germany whose placement layer is measured
    rather than modelled: `obyv_grid_1km` is SODB 2021 redistributed to 49,969 cells and
    sums to 5,449,270, the census total to the person. So this is a Kontur-shaped weighter
    with none of Kontur's modelling caveat.

    It is still a POPULATION weight and not a religion one — nothing measures where an
    obec's Lutherans sit inside it — so a Catholic dot and a Lutheran dot spread
    identically. Read the map as religion by obec, drawn where Slovaks live.

    2,927 obce over 49,000 km² is 17 km² each, so this matters less than it does in Kenya
    or Kazakhstan; it earns its place on the big rural obce of the north, where the built-up
    part is one valley floor inside a polygon that runs up into the Tatras.
    """

    def summary(self):
        return (f"{self.n_pop:,} (unit, node) rows placed on the census's own 1 km cell "
                f"population, {self.n_uniform:,} on equal shares where an obec's cells sum "
                f"to zero (sources/sk_geo.py)")


def _sk_place_weight(place):
    """countries.py hook. `place` is the 1 km grid GeoDataFrame scatter.py has read."""
    if "pop" not in place.columns:
        print("  !! sk_grid_1km.gpkg has no `pop` column — run sources/sk_geo.py; "
              "placing on equal shares (§8.2)")
        return None
    return _SkGridWeighter(place)


def _sk_counts():
    """SODB 2021 at obec: 10 nodes on 2,927 units.

    ONE level, no allocation, nothing modelled — every drawn row is `measured` and may ring.
    sk.csv also carries the national row, which is the same people again, and the `ostatné`
    residual this drops.

    THERE IS NO JOIN. The religion counts and the municipal polygons are fields and geometry
    on the SAME ArcGIS feature layer (`gis.scitanie.sk`, layer 4 of
    `obyv_ekchar_nabo_vekskup`), so `geo_id` is the polygon's own `uzemie` and §12's first
    two shapes of failure — a silent drop and a confident wrong pairing — cannot arise here.
    Four sweeps had recorded the country as walled while this sat open; see sources.md §11aa.

    `ostatné` IS 7.83% AND IS DRAWN ON `other.sk` — Anita's call, 2026-09-08, reversing a
    first build that excluded it on §3.5. UNSD table 28 publishes the same census with 21
    categories, agrees with this service to the person on the total and on all nine named
    churches, and shows the residual contains `nezistené` — 353,797 people, 83% of the cell.
    Excluding it left 7.83% of the country as a hole, and §6.12 says a hole reads as an
    absence of PEOPLE; drawing it keeps them, at the cost of a node that is not comparable
    with any other country's `other`. The node is labelled `Other or not stated (Slovakia)`
    for that reason, and its density is a map of the census's reach rather than of religion:
    0.0-58.8% between obce, peaking in Roma settlements and city centres. Every drawn row is
    still `measured`. See taxonomy/sk2021.py and sources/sk.md.
    """
    from sk2021 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "sk.csv",
                     dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    df = df[df["geo_level"] == "obec"].copy()
    df["unit"] = df["geo_id"].astype(str)
    if df["unit"].nunique() != 2927:
        raise SystemExit(f"{df['unit'].nunique()} obce, expected 2,927")

    df["node"] = df["source_category"].map(resolve)
    df = df[df["node"].notna() & (df["count"] > 0)]
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations"]]


ENTRY = {
    "sk": dict(
        name="Slovakia",
        source="SODB 2021 (Štatistický úrad SR)",
        basis="self-identification",
        view=[16.83, 47.73, 22.57, 49.61],
        gap=("none, every counted person is drawn; but 7.8% of them are one other or not "
             "stated cell, and 83% of that is people who did not answer rather than people "
             "of another religion"),
        note_public=(
            "**Two halves of one state, and one of the sharpest religious contrasts on "
            "this map runs along the border between them.** Slovakia is **55.8% Roman "
            "Catholic** and Czechia, which it was part of until 1993, is 7.0%. Nothing "
            "else about the two countries diverges like this, and the line is visible on "
            "the map as an edge rather than a gradient. "
            "**The Lutherans are the national revival and they are in the middle of the "
            "country.** The Evangelical Church of the Augsburg Confession is 5.3% "
            "nationally but the historic church of the Slovak literary language, and its "
            "people are in the central uplands — Turiec, Liptov, Gemer and the Zvolen "
            "basin — rather than in the Catholic west or the Greek Catholic east. "
            "**The Greek Catholics are the east, and this is the largest Byzantine-rite "
            "Catholic population on this map after Romania's.** 218,235 people, 4.0%, "
            "concentrated in Prešov and the Rusyn villages along the Polish and Ukrainian "
            "borders. The church was suppressed outright in 1950 and restored in 1968, "
            "which is a shorter and less complete rupture than Romania's. "
            "**The Reformed are Hungarian.** The Reformed Christian Church's 85,271 people "
            "sit in a narrow strip along the southern border, in the same districts that "
            "report a Hungarian mother tongue — the same church and the same minority as "
            "the Reformed across the frontier in Hungary and in Romania's Székely Land. "
            "**Nearly a quarter of the country reports no religion, and the question it "
            "answers is not Czechia's.** Slovakia's form offers one box, *bez "
            "náboženského vyznania*, and no atheist, agnostic or believing-without-"
            "belonging option; Czechia offers all of those. So 23.8% here and 47.8% there "
            "are answers to differently-shaped questions and the gap between them is "
            "partly the form. "
            "**The grey *other or not stated* dots are mostly people who did not answer, "
            "and where they cluster they are measuring the census rather than religion.** "
            "The published municipal table folds *not ascertained* — 6.5% of the country — "
            "into one cell together with the Baptists, Adventists, Jews, Old Catholics, "
            "Hussites and Bahá'ís, and nothing at this geography separates them. It is "
            "7.8% of Slovakia nationally but runs from nothing to **58.8% in Košice's "
            "Luník IX**, with Pavlovce nad Uhom at 28.7%, Jasov at 26.9% and Bratislava's "
            "old town at 16.4%, against under 3% across the Orava and Kysuce villages. "
            "Those peaks are Roma settlements and city centres — places where a census "
            "form comes back unanswered — so read a dense patch of this colour as the "
            "limit of the count, not as an unusual faith."),
        how="a census question, 2021",
        grain="municipalities, 1,900 people on average",
        counts=_sk_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "sk" / "sk_grid_1km.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_sk_place_weight,
        note="**FOUR SWEEPS RECORDED SLOVAKIA AS WALLED AND THE ROUTE WAS NONE OF THE ONES "
             "THEY LOOKED FOR** (sources.md §11aa). §11, §11c, §11k and §11o all went at "
             "`slovak.statistics.sk`, which is still a 403, and at "
             "`datacube.statistics.sk`, which is wide open and whose **678 cubes contain no "
             "religion at all** — SODB 2021 is simply not in DATAcube. The census results "
             "live on their own GIS portal: `gis.scitanie.sk` is a public ArcGIS Server "
             "with 86 hosted services, no key, and layer 4 of `obyv_ekchar_nabo_vekskup` "
             "carries **religion counts and municipal polygons as fields and geometry on "
             "the same features**. That makes Slovakia the first country here with no "
             "join between counts and boundaries, so §12's two commonest failures cannot "
             "arise. "
             "**THE PARTITION IS EXACT AND WITNESSED THREE TIMES.** The eleven category "
             "columns sum to `spolu` sums to 5,449,270 across all 2,927 obce with a "
             "difference of zero; fetching the 8 kraje from a different layer reproduces "
             "every one of the twelve totals; and UNSD Demographic Yearbook table 28, a "
             "different publisher of the same census, agrees to the person on the total "
             "and on all nine named churches. "
             "**IT IS ALSO THE SECOND COUNTRY AFTER GERMANY WHOSE PLACEMENT IS MEASURED.** "
             "`obyv_grid_1km` is SODB 2021 on 49,969 1 km cells and sums to the census "
             "total exactly, so no Kontur extract is used. **But it is not an independent "
             "check** (§9av): the grid is the same enumeration as the counts, so the "
             "ratio band says nothing about the census and only validates the cell-to-obec "
             "assignment. "
             "**THREE ARCGIS TRAPS, ALL LIVE.** `maxRecordCount` is 2000 against 2,927 "
             "obce, so an unpaged query returns two-thirds of the country with "
             "`exceededTransferLimit` buried in the response and no error — §5a in a new "
             "disguise, and every per-row check would still pass. `supportsPagination` is "
             "not advertised and `resultOffset` works anyway. `supportedQueryFormats` says "
             "JSON only and `f=geojson` works. "
             "**AND THE PLACEMENT RULE ME_GEO.PY USES IS WRONG HERE.** Assigning a 1 km "
             "cell to the obec containing its centre credits a village smaller than the "
             "cell with the whole cell's people: Záborie, 170 people, came out weighted at "
             "1,409. Cells are split by area of intersection instead, and renormalised "
             "against the area inside Slovakia so that a border cell's people are not lost "
             "to the part of its square lying over Austria."),
}
