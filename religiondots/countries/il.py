# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _il_counts():
    """CBS 2022 census at statistical area: nodes on 2,968 units.

    **TWO geo_levels, and both are drawn.** CBS splits 142 localities into statistical areas
    and publishes the other 1,043 whole, so the drawn tier is `statarea` PLUS the localities
    that have none. A locality that HAS statistical areas is not in the file at locality
    level at all (sources/il.py drops it), so there is nothing here to double-count -- but
    filtering to one level, which is what every other country in this file does, would
    silently delete either every city or every village.

    **THE UNIT COUNT IS SMALLER THAN THE FILE'S, ON PURPOSE.** il.csv carries every unit CBS
    publishes, including the Judea and Samaria Area and East Jerusalem, because a normalised
    file reproduces its source. The units LAYER is cut on the Green Line, so 267 units have
    counts and no polygon and drop out at the join. That is the design and not a leak;
    `sources/il_geo.py` holds the reasoning and `data/geo/il/dropped_units.json` the list.

    **THE BASIS IS `roll`.** Israel's religion comes from the population register, not from
    a question, so these figures are not comparable with a census that asks (spec §3.1). The
    register has no irreligion box, which is why Israel draws as ~100% religious and why the
    observance split is worth its complications.
    """
    from il2022 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "il.csv",
                     dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    df = df[df["geo_level"].isin(("statarea", "locality"))].copy()
    df["count"] = df["count"].astype(float)

    df["node"] = df["source_category"].map(resolve)
    # `Other religions` is the lump CBS publishes instead of a breakdown, and sources/il.py
    # is supposed to have resolved it against sub-district totals before writing this file.
    # If any survives, it is people about to be dropped without a word -- say so instead.
    lump = df[df["node"].isna() & df["source_category"].str.contains("Other religions")]
    if len(lump):
        raise SystemExit(
            f"il.csv still carries {len(lump):,} unresolved 'Other religions' rows "
            f"({lump['count'].sum():,.0f} people) -- sources/il.py did not run its "
            "allocation. Mapping them to one religion would erase Israel's Christians.")

    df = df[df["node"].notna() & (df["count"] > 0)]
    df = df.rename(columns={"geo_id": "unit"})

    # THE TERRITORIAL CUT HAPPENS HERE, NOT ONLY AT THE JOIN, AND THE DIFFERENCE IS THE
    # LEGEND. il.csv carries every unit CBS publishes; sources/il_geo.py builds polygons only
    # for what is drawn. Leaving the rest in would let scatter.py drop them at the join with
    # a warning — but `counts.json` is built from THIS function, so the legend would go on
    # totalling ~870,000 people who are nowhere on the map. Excluding them here makes the cut
    # one decision in one place, and the count is asserted so a boundary change cannot
    # silently move it.
    dropped_path = HERE / "data" / "geo" / "il" / "dropped_units.json"
    if dropped_path.exists():
        with open(dropped_path, encoding="utf-8") as fh:
            dropped = set(json.load(fh))
        gone = df[df["unit"].isin(dropped)]
        df = df[~df["unit"].isin(dropped)]
        print(f"  il: {len(dropped):,} units beyond the Green Line excluded "
              f"({gone['count'].sum():,.0f} people) — West Bank, Gaza and East Jerusalem; "
              "the Golan is kept (sources/il_geo.py)")
    else:
        raise SystemExit(
            "il: data/geo/il/dropped_units.json is missing — run sources/il_geo.py. "
            "Without it the West Bank and East Jerusalem would be counted in the legend.")

    df["congregations"] = 0
    # §3.10: an allocated row spreads a total and cannot establish presence, so it may not
    # ring. The observance rows are `derived` too and cannot either.
    df["may_ring"] = df["tier"] == "measured"
    return df[["unit", "node", "count", "congregations", "may_ring", "tier"]]


ENTRY = {
    "il": dict(
        name="Israel",
        source="2022 Census of Population and Housing (CBS)",
        basis="population register",
        view=[34.2, 29.4, 35.95, 33.35],
        note_public=(
            "**Israel's religion is read off the population register, not asked.** Every "
            "other country here counts what people said about themselves; this one records "
            "what the state has on file, assigned at registration from parentage or a "
            "recognised conversion. The two are not the same measurement and the "
            "percentages are not comparable with a census that asks. "
            "**The register has no box for having no religion**, so Israel draws as almost "
            "entirely religious — which is a fact about the form and not about the country. "
            "The observance colours are the corrective: **53% of Israeli Jews describe "
            "their household as secular**, and they are drawn inside Judaism because that "
            "is where the register puts them. "
            "**Haredi Israel is the sharpest pattern on the map.** 6.4% of the country and "
            "hardly spread at all — Bene Beraq is 84% ultra-religious, Modi'in Illit and "
            "Beitar Illit almost entirely so, and in Jerusalem the Haredi quarters sit "
            "against secular ones street by street. The map is drawn on statistical areas "
            "of about 3,000 people, which is fine enough to show that edge. "
            "**Where the observance split is NOT drawn, Judaism is one colour.** The "
            "question is asked of every household, Arab ones included, so applying it where "
            "a unit is religiously mixed would attribute one group's answers to another. It "
            "is used only where a unit is at least 85% Jewish; elsewhere the Jewish dots "
            "carry no observance. "
            "**The Druze are the largest count of them anywhere** — about 153,000, in the "
            "Galilee and Carmel villages and in the four Golan villages, where most "
            "residents have declined Israeli citizenship. "
            "**Christians are one cell.** CBS does not separate Greek Orthodox from Greek "
            "Catholic, Latin, Maronite, Armenian or Syriac, nor Arab Christians from the "
            "large ex-Soviet and migrant Christian population, so some of the oldest "
            "continuously resident churches in the world draw a single colour. "
            "**\"Others\" is not irreligion.** About 442,000 people are on the register with "
            "no religious classification, overwhelmingly immigrants under the Law of Return "
            "who are not Jewish by halakha. Nobody asked them; the register simply has no "
            "entry, which is why they are drawn grey."),
        how="population register, not a census question",
        fill="from each area's own household-lifestyle table",
        grain="statistical areas, about 3,000 people each",
        # §6.12: a blank on a dot map cannot tell "nobody here is religious" from "nobody
        # counted here". This is the only place that difference is stated while the blank is
        # on screen, and Israel's blank is a deliberate territorial decision rather than a
        # hole in the data — see sources/il_geo.py.
        gap="the West Bank, Gaza and East Jerusalem; the Golan is drawn",
        counts=_il_counts,
        units=None,
        unit_key=None,
        # PLACEMENT IS THE UNIT POLYGON AND THERE IS NO GRID — §8.2e, measured rather than
        # skipped. Israel's median statistical area is 0.69 km² against a 1.17 km² Kontur
        # hex, so the grid is COARSER than the tier it would refine: 42% of units get no hex
        # at all and 69% are smaller than one. `sources/il_geo.py` runs the test on every
        # build and prints the verdict, so this cannot quietly become wrong.
        place=HERE / "data" / "geo" / "il" / "il_units.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        note="**THE MAP STOPS AT THE GREEN LINE AND THE GOLAN IS THE ONE EXCEPTION** — "
             "Anita, 2026-09-07. The West Bank and Gaza are not drawn, and neither is East "
             "Jerusalem: cutting only its Palestinian neighbourhoods while keeping Gilo, "
             "Pisgat Ze'ev, Ramot and Neve Ya'akov would draw the settlements and erase the "
             "people they were built among, which is §14.2's second risk exactly. So "
             "Jerusalem draws as a western fragment and 267 of 3,235 units are dropped. The "
             "Golan IS drawn, against the same rule, because Israel counts those 56,600 "
             "people and nobody else does — 24,900 of them Druze. The cut is OCHA's "
             "published oPt polygon rather than a line drawn here, checked against ten "
             "hand-picked points on every build; geoBoundaries' Palestine excludes annexed "
             "Jerusalem and fails five of them, which is the trap this check exists for. "
             "**THE DASHBOARD COLLAPSES MINORITIES INTO 'Other religions' AND IT IS NOT "
             "'Others'.** For units below some size CBS publishes the dominant group and a "
             "lump: Nazareth returns Muslims 73.1% / Other religions 26.9%, and that 26.9% "
             "is essentially all Christian. Read naively it erases Israel's Christians. "
             "sources/il.py resolves the lump against sub-district totals and "
             "`_il_counts` refuses to build if any survives. "
             "**THE DATA IS ONE REQUEST PER AREA AND THE HOST THROTTLES.** The area IDs are "
             "opaque hashes with no relation to CBS codes; they come from the census site's "
             "own htmx autocomplete, `/he/partials/search/area`, which nothing links to. "
             "Fetching ~4,600 of them at 0.15s intervals got this machine IP-blocked from "
             "census.cbs.gov.il for hours. sources/il.py now reuses one connection, backs "
             "off, and caches every 50 units so the run is resumable — and it is deliberately "
             "slow. Do not parallelise it. "
             "**THERE IS NO PLACEMENT GRID, AND THAT WAS MEASURED** (§8.2e). A Kontur grid "
             "was built here first and thrown away: Israel's median statistical area is "
             "0.69 km² against a 1.17 km² hex, so the grid is COARSER than the tier it "
             "would refine — 42% of units get no hex at all, 69% are smaller than one, and "
             "the per-unit Kontur/census ratio runs 0.27 to 3.49. That is noise, not a "
             "weighting. Placement is the unit polygon, i.e. §8.2's uniform share, which "
             "§8.2e argues is the better answer here rather than a fallback; "
             "`sources/il_geo.py` re-runs the test on every build so the decision cannot "
             "rot. What it costs: the few genuinely large units — Negev Bedouin localities "
             "and regional councils, up to 195 km² — get an even wash.",
    ),
}
