# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _li_place_weight(place):
    """countries.py hook. `place` is the 400m hex layer scatter.py has read."""
    return _kontur_place_weight(place, "li_grid_400m.gpkg", "sources/li_geo.py")


def _li_counts():
    """Amt für Statistik Volkszählung 2015 at commune: 10 nodes on 11 units.

    ONE level, no allocation, nothing modelled — every drawn row is `measured` and may ring.
    li.csv also carries the Liechtenstein national row, which is the same people again.

    **AN EXACT PARTITION IN BOTH DIRECTIONS**, which a country of 37,622 people can afford:
    the 11 categories sum to each commune's own total, and the 11 communes sum to the
    national row category by category, both with a gap of zero. Nothing is suppressed,
    rounded or prorated — the census publishes single people, and Planken's one member of
    `Other Christian communities` is on the map.

    **96.7% of the country is drawn** — 36,393 of 37,622. What is not is `Not stated`, 1,229
    people, which taxonomy/li2015.py excludes per §3.5 and which is NOT `No religious
    affiliation`, a separate answer taken by 2,623.
    """
    from li2015 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "li.csv", low_memory=False)
    df = df[df["geo_level"] == "municipality"].copy()
    if df["geo_id"].nunique() != 11:
        raise SystemExit(f"{df['geo_id'].nunique()} communes, expected 11 -- re-run "
                         "sources/li.py")

    df["node"] = df["source_category"].map(resolve)
    df = df[df["node"].notna() & (df["count"] > 0)]
    df = df.rename(columns={"geo_id": "unit"})
    df["congregations"] = 0
    df["tier"] = "measured"
    return df[["unit", "node", "count", "congregations", "tier"]]


ENTRY = {
    "li": dict(
        name="Liechtenstein",
        source="Volkszählung 2015 (Amt für Statistik)",
        basis="self-identification",
        view=[9.41, 47.03, 9.68, 47.29],
        gap="3.3% who did not state a religion, spread evenly across the country",
        gap_share=0.03267,
        note_public=(
            "**The smallest country on this map and one of the most finely counted** — "
            "37,622 people over 11 communes, about 3,400 each, which is finer per head than "
            "most of Europe here. The census publishes single people: Planken's one member "
            "of an 'other Christian' church is a dot. "
            "**73.4% Roman Catholic, and it is the state church** — Article 37 of the "
            "constitution names it, which makes Liechtenstein one of the last places in "
            "Europe where that is literally so. Disestablishment has been debated since 2012 "
            "and has not happened. "
            "**Only 7.0% report no religion, about a fifth of Switzerland's share twenty "
            "kilometres away** — the lowest in Western Europe on this map. That gap between "
            "two neighbouring Alpine countries is the most striking thing here, and it is "
            "not a measurement artefact: both asked the same kind of question. "
            "**Islam is 5.9% and it is the guest-worker migration**, Turkish and Bosnian, "
            "settled around the industrial communes: **Eschen 11.4% and Gamprin 8.0% "
            "against Planken 0.2% and Schellenberg 1.2%**. The citizenship split is the "
            "sharpest in the table — **13.1% of foreign residents against 2.2% of "
            "Liechtenstein citizens** — and the same split runs the other way for "
            "Catholicism, 84.0% of citizens against 52.6% of foreigners. A third of the "
            "country holds a foreign passport. "
            "**The form separates Reformed from Lutheran**, which almost nothing else on "
            "this map does: two state-recognised Protestant churches, one Swiss-facing and "
            "one Austrian, at 2,365 and 447 people. "
            "**There is no Jewish box on the form at all**, so Liechtenstein's Jews are "
            "inside 'other religious communities' and the country stays unlit when Judaism "
            "is selected — the question was not put. **3.3% did not state a religion** and "
            "are not drawn."),
        how="census, 2015",
        grain="communes, 3,400 people on average",
        counts=_li_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "li" / "li_grid_400m.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_li_place_weight,
        note="**FOUND IN SWITZERLAND'S CATALOGUE, WHICH IS THE REUSABLE PART.** "
             "`ckan.opendata.swiss` carries the Liechtenstein statistics office alongside "
             "BFS, so the single search that solved Switzerland (§9ad) returned this too. "
             "Nobody had looked: §11c and §11k both swept Europe and neither mentions "
             "Liechtenstein. **A national open-data portal may index a neighbour**, and a "
             "microstate is exactly the kind of country a sweep walks past. "
             "**THE BOUNDARIES WERE ALREADY ON DISK AND THE JOIN IS ELEVEN EXACT STRINGS.** "
             "GISCO LAU 2021 — §9e's file, downloaded for Poland — carries all 11 communes "
             "with the names the census uses, so there is no folding, no stemming and no "
             "alias table. Liechtenstein is absent from the EU-27 correspondence *workbook* "
             "and present in the boundary *shapefile*, which is the same distinction "
             "Switzerland turns on. "
             "**ITS json-stat2 EMITTER IS BROKEN AND NOTHING SAYS SO.** Asking for "
             "`json-stat2` returns HTTP 200, a well-formed document declaring "
             "`size: [1,12,1,1,12]` — 144 cells — and a `value` array holding **one "
             "element**. `json-stat` and `csv` both answer correctly, so this is one "
             "serialiser rather than a wall. §5a again: *a 200 with a valid-looking envelope "
             "is not a download; check the payload against the shape the same response "
             "declares.* `filter: \"all\"` with `values: [\"*\"]` is also ignored here and "
             "has to be an explicit item list — the second PxWeb server in two days with "
             "non-portable selection semantics, after BFS's empty query dropping the "
             "geography dimension. "
             "**THE COMMUNES ARE NOT CONTIGUOUS, WHICH IS WHY THE GRID IS NOT OPTIONAL.** "
             "Liechtenstein divides its high alpine pasture among the valley communes as "
             "exclaves: **Vaduz is six separate polygons, Schaan four, Balzers and Planken "
             "three**, and seven of the eleven are fragmented. Nobody lives in the detached "
             "pieces — summer grazing above 1,500 m — so §8.2's equal share would scatter a "
             "third of Vaduz's dots onto an empty mountainside. Kontur's 171 hexes fix it. "
             "**The `-` cells are true zeros and the partition proves it**, which is "
             "Kosovo's argument rather than Lithuania's: there is no disclosure threshold "
             "here at all, so a blank cannot hide a small number. "
             "**Vintage: 2015 is the last one.** The table offers 2010 and 2015 and nothing "
             "since; Liechtenstein's later population statistics are register-based and "
             "carry no religion.",
    ),
}
