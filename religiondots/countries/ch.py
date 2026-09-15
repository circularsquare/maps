# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _ch_place_weight(place):
    """countries.py hook. `place` is the 400m hex layer scatter.py has read."""
    return _kontur_place_weight(place, "ch_grid_400m.gpkg", "sources/ch_geo.py")


def _ch_counts():
    """Switzerland at commune: 17 nodes on 2,196 units, and every row is `derived`.

    **THE COUNTS COME FROM `ch_rescale.py`, NOT FROM `ch.csv`**, and the difference is the
    whole country. `sources/ch.py` normalises the 2000 census — the last time Switzerland
    asked everybody — onto 2021 commune boundaries. `ch_rescale.py` then fits current
    Strukturerhebung canton totals onto that commune-by-category structure (spec §3.4), which
    is Brazil's move (§9-br) with one extra step.

    **NOTHING HERE IS `measured` AND THAT IS NOT AN OVERSIGHT.** Brazil's rescale changes the
    categories and keeps the geography, so a 2022 município total is a measured number for
    that município and 103.6M Brazilians pass through untouched. Switzerland's changes the
    geography too: the measured quantity is a *canton* total being spread over that canton's
    communes. Every drawn row is therefore `derived`, and §3.10 forbids all of it from
    becoming a presence ring.

    **TWO MARGINS ARE PRESERVED, NOT ONE.** A plain Brazil-style rescale would hold each
    commune's share of its canton's Reformed population fixed since 2000, which puts dots in
    alpine communes that have emptied and starves the suburbs people moved to — a real
    distortion on a map whose subject is where people are. So the fit is an IPF on current
    commune populations (GISCO's `POP_2021`) *and* the survey's canton category totals, with
    the 2000 census supplying only the association between commune and religion. Both margins
    come out exact.

    **99.1% of the survey's universe is drawn** — 7,438,908 of 7,506,664. The remainder is
    `Ohne Angabe`, which taxonomy/ch2000.py excludes per §3.5 and which must not be read as
    `Keine Zugehörigkeit`, a separate answer taken by 36.8%.
    """
    from ch2000 import resolve

    path = HERE / "data" / "normalized" / "ch_commune_rescaled.csv"
    if not path.exists():
        raise SystemExit(f"missing {path} -- run `python ch_rescale.py`. countries.py "
                         "deliberately does NOT read data/normalized/ch.csv: that file is "
                         "the 2000 census as counted, and drawing it would put a "
                         "twenty-six-year-old Switzerland on the map.")
    df = pd.read_csv(path, dtype={"geo_id": str}, low_memory=False)
    if df["geo_id"].nunique() != 2196:
        raise SystemExit(f"{df['geo_id'].nunique()} communes, expected 2,196 -- re-run "
                         "ch_rescale.py")

    df["node"] = df["node_category"].map(resolve)
    df = df[df["node"].notna() & (df["count"] > 0)]
    df = df.rename(columns={"geo_id": "unit"})
    df["congregations"] = 0
    df["tier"] = "derived"
    return df[["unit", "node", "count", "congregations", "tier"]]


ENTRY = {
    "ch": dict(
        name="Switzerland",
        source="Volkszählung 2000 structure on Strukturerhebung 2024 canton totals (BFS)",
        basis="self-identification; current magnitudes, 2000 composition",
        view=[5.8, 45.75, 10.6, 47.85],
        note_public=(
            "**This map is 2024 in size and 2000 in shape, and the difference matters.** "
            "Switzerland last asked everybody about religion in the 2000 census. Since 2010 "
            "the question lives in a sample survey that publishes eight categories and stops "
            "at the canton, so the choice is between the right detail on the right places a "
            "quarter-century out of date, and the right year on twenty-six units. Both are "
            "used: the survey says how many, the census says where and which church. "
            "**Nothing here is a count.** "
            "**What has actually happened is the fastest religious change on this map.** In "
            "2000 Switzerland was 41.8% Catholic and 33.0% Reformed, with 11.1% reporting no "
            "religion. It is now 30.0%, 18.7% and **36.8%** — no religion is the largest "
            "single answer in the country, and it grew more than threefold inside one "
            "generation. "
            "**The confessional map underneath it is five hundred years old and still "
            "legible.** The Reformation split Switzerland canton by canton and the line has "
            "barely moved: **Uri and Appenzell Innerrhoden are 68% Catholic, Valais 62%**, "
            "against **Bern at 14% and Basel-Stadt at 13%**. At commune level it is sharper "
            "still — Muotathal 84% Catholic, Poschiavo 82%, against the Emmental villages of "
            "Sumiswald and Lützelflüh at 63% Reformed. Appenzell was partitioned into two "
            "half-cantons over religion in 1597 and the two halves are still 68% Catholic and "
            "overwhelmingly Reformed respectively. "
            "**Irreligion is urban, and it is Protestant cantons that went furthest.** "
            "Basel-Stadt is 60%, Neuchâtel 57%, Geneva 50%, against 22% in Uri and 18% in "
            "Appenzell Innerrhoden. "
            "**The 6.0% Muslim population is industrial rather than metropolitan** — "
            "Böttstein 24%, Gerlafingen 23%, St. Margrethen 21%, small towns along the Aare "
            "and the Rhine rather than the big cities — and it is Balkan and Turkish. **The "
            "Hindus are Tamil**, 46,000 of them and more numerous than Switzerland's Jews and "
            "Buddhists combined, from the Sri Lankan asylum migration of the 1980s; they show "
            "up in Solothurn and Emmental factory towns. And **Möhlin and Magden in the Fricktal "
            "are 15% Christ Catholic** — the Old Catholic church that broke with Rome in 1871 "
            "over papal infallibility, which survives as a public-law church in a handful of "
            "Swiss communes and almost nowhere else. "
            "**A Swiss dot is an adult.** The survey asks people aged 15 and over living in "
            "private households, so children, and people in institutions, are outside the map "
            "rather than inside it — as in Brazil, Chile and Portugal."),
        how="census, 2000, resized to 2024 survey totals",
        fill="from the 2000 census",
        grain="communes, 3,400 people on average",
        counts=_ch_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "ch" / "ch_grid_400m.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_ch_place_weight,
        note="**§11c PRICED THIS COUNTRY AT AN HOUR OF CRAWLING AND THE CRAWL DOES NOT "
             "WORK.** It recorded that BFS PxWeb lists 650 databases with opaque ids and the "
             "human title only inside each one, so finding the religion table means "
             "enumerating all of them. That was tried: **BFS rate-limits it into total "
             "failure, 55 of 55 requests returning nothing at one per second.** "
             "`ckan.opendata.swiss` — the national portal that mirrors BFS's own catalogue — "
             "answers the same question in one call and hands over `px-x-4003000000_122` "
             "directly. **When an office rate-limits its own API, ask the national open-data "
             "portal that mirrors it**; the same call also turned up Liechtenstein's "
             "statistics office, which nobody had looked at. "
             "**THE CATEGORY LIST IS THE REASON TO DRAW A SOURCE THIS OLD.** Nineteen cells, "
             "seven of them Protestant — Reformed, Methodist, neo-pietist evangelical, "
             "Pentecostal, New Apostolic, Jehovah's Witnesses and a named Protestant "
             "residual — plus Christ Catholic separately from Roman Catholic, and Orthodox, "
             "Jewish, Islamic, Buddhist and Hindu. No other European census here can express "
             "half of that; the survey that replaced it has **eight**. "
             "**THE COMMUNE VINTAGE WAS THE REAL WORK.** The census counts 2,896 communes "
             "and GISCO LAU 2021 carries 2,242, because Swiss communes have been merging "
             "continuously; a naive join on the BFS number loses 820 of them and 606,086 "
             "people. BFS publishes the correspondence itself as an open keyless API "
             "(`agvchapp.bfs.admin.ch/api/communes/correspondances`), and three things about "
             "using it are worth keeping: `startPeriod` has to be the census date and not the "
             "following 1 January, or the 2001 Fribourg mergers vanish; territory exchanges "
             "must be excluded or the map stops being a function; and **GISCO's 'LAU 2021' "
             "for Switzerland is actually the 1 January 2020 commune state** — asking for "
             "2021 leaves fifteen communes pointing at codes the shapefile does not have. "
             "Both states are requested and whichever target exists wins, which resolves all "
             "2,896. *A boundary file named for a year is not necessarily that year's state.* "
             "**45 OF THE 2,242 LAU FEATURES ARE NOT COMMUNES** — the lake surfaces, which "
             "BFS numbers in the 9xxx block and apportions to no municipality, and the Ticino "
             "and Graubünden *comunanze*, common land held jointly. They are dropped against "
             "BFS's own register rather than by a code range, so a renumbering fails loudly. "
             "**AND ONE SILENT TRAP IN THE PxWeb CALL ITSELF**: `{\"query\": []}` returns a "
             "1.6 KB cube with the geography dimension **absent** — a 200, valid json-stat2, "
             "and no geography at all. An empty query does not mean everything on this "
             "server; `sources/ch.py` asserts the dimension survived.",
    ),
}
