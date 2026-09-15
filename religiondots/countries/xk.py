# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _xk_place_weight(place):
    """countries.py hook. `place` is the 400m hex layer scatter.py has read."""
    return _kontur_place_weight(place, "xk_grid_400m.gpkg", "sources/xk_geo.py")


def _xk_counts():
    """ASK Census 2024 at municipality: 5 nodes on 38 units.

    ONE level, no allocation, nothing modelled. xk.csv also carries the KOSOVA row, which
    is the same people again, and the 2011 census, which sources/xk.py drops.

    **A PERFECT PARTITION, AND THE BLANKS ARE PROVEN ZEROS RATHER THAN ASSUMED ONES.** The
    six categories sum to each unit's total and the 38 municipalities sum to the national
    row category by category, both exactly; 21 blank cells read as zero is the only reading
    that leaves those sums intact (sources/xk.py). Lithuania's §3.8 trap, with the opposite
    answer and a proof rather than a guess.

    **THE FOUR NORTHERN MUNICIPALITIES ARE CORRECTED FROM ASK'S OWN ESTIMATE — Anita,
    2026-09-06.** They were drawn as published for one build and that was wrong in a way the
    note could not fix: their Serb population refused enumeration, and what WAS counted there
    is not a thin sample but a different population, so three of the four came out MAJORITY
    MUSLIM in a table whose own author knows better.

    ASK publishes the correction itself. `census2024_63.px` is the ethnicity table *with
    estimation* and it is identical to the enumerated one everywhere except these four, where
    it restores 16,949 people — **16,369 of them Serbs, 96.6%**. There is no
    religion-with-estimation table, so the step is Serb -> Orthodox: spec §14.5, argued in
    `taxonomy/xk2024.py`, and Kosovo passes its three tests more cleanly than China does.

    THREE THINGS THIS DELIBERATELY DOES NOT DO. It does not touch the enumerated rows — the
    derivation ADDS the missing Orthodox rather than restating what was counted. It does not
    place the 580 non-Serb people in the estimate, because nothing says what they are and the
    enumerated composition of these four is the thing that is not representative (§3.5). And
    it does not launder itself: every added row is `tier="derived"`, so §7a's control removes
    all of it at once and the raw table is one click away.
    """
    from xk2024 import ETHNIC_DERIVATION, resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "xk.csv",
                     dtype={"geo_id": str}, low_memory=False)
    df = df[df["geo_level"] == "municipality"].copy()
    if df["geo_id"].nunique() != 38:
        raise SystemExit(f"{df['geo_id'].nunique()} municipalities, expected 38 -- re-run "
                         "sources/xk.py")

    df["node"] = df["source_category"].map(resolve)
    df = df[df["node"].notna() & (df["count"] > 0)]
    df = df.rename(columns={"geo_id": "unit"})
    df["congregations"] = 0
    df["tier"] = "measured"
    out = df[["unit", "node", "count", "congregations", "tier"]]

    # ---- the northern derivation (§14.5) --------------------------------------------
    north_path = HERE / "data" / "normalized" / "xk_north.csv"
    if not north_path.exists():
        raise SystemExit(f"missing {north_path} -- re-run sources/xk.py, which writes it "
                         "alongside xk.csv. Without it the four northern municipalities "
                         "draw as published, which reads them as majority Muslim.")
    nn = pd.read_csv(north_path, dtype={"geo_id": str}, low_memory=False)
    nn = nn[nn["ethnicity"].isin(ETHNIC_DERIVATION) & (nn["added"] > 0)].copy()
    if nn.empty:
        raise SystemExit("xk_north.csv carries no positive derived rows -- the estimate is "
                         "the whole point of reading it")

    # xk_north.csv keys on the municipality NAME; xk.csv keys on ASK's numeric code. Join
    # them through the name rather than assuming the codes line up across two tables.
    key = (pd.read_csv(HERE / "data" / "normalized" / "xk.csv",
                       dtype={"geo_id": str}, low_memory=False)
           .query("geo_level == 'municipality'")[["geo_id", "geo_name"]]
           .drop_duplicates("geo_id"))
    key = dict(zip(key["geo_name"], key["geo_id"]))
    nn["unit"] = nn["geo_name"].map(key)
    if nn["unit"].isna().any():
        raise SystemExit(f"northern municipalities not found in xk.csv: "
                         f"{sorted(nn.loc[nn['unit'].isna(), 'geo_name'])}")

    nn["node"] = nn["ethnicity"].map(ETHNIC_DERIVATION)
    nn = nn.rename(columns={"added": "count"})
    nn["congregations"] = 0
    nn["tier"] = "derived"
    print(f"  xk: +{nn['count'].sum():,} derived Orthodox across "
          f"{nn['unit'].nunique()} northern municipalities (spec §14.5, ASK's own estimate)")

    return pd.concat([out, nn[["unit", "node", "count", "congregations", "tier"]]],
                     ignore_index=True)


ENTRY = {
    "xk": dict(
        name="Kosovo",
        source="Census 2024 (Kosovo Agency of Statistics)",
        basis="self-identification",
        view=[20.0, 41.85, 21.8, 43.25],
        gap="1.5% who preferred not to answer, and 4.2% of Prishtina",
        gap_share=0.01496,
        note_public=(
            "**The four municipalities in the north are estimated, not counted.** Kosovo's "
            "Serbs largely refused the 2024 census, and there the result was not a thin "
            "count but a misleading one: Zveçan returned 434 people and Zubin Potok 763, in "
            "Serb-majority municipalities of several thousand, and because whoever did "
            "answer was disproportionately not Serb, three of the four came out as majority "
            "Muslim. That is not what those places are. So the missing people are put back "
            "from **the statistics agency's own estimate** — its ethnicity table published "
            "*with estimation* restores 16,949 people across the four, 96.6% of them Serbs — "
            "and they are drawn as Orthodox. **Those dots are an inference from ethnicity, "
            "not a count of anybody's answer**, and the confidence control removes them. "
            "The 2011 census did not enumerate these four at all. "
            "**Everywhere else the map is straightforward and the country is 93.5% "
            "Muslim** — the highest Muslim share in Europe, Hanafi Sunni, with a Sufi tekke "
            "tradition in Gjakovë and Prizren that no census category separates. "
            "**The Catholics are the interesting minority, and they are Albanian rather "
            "than foreign.** 1.75% nationally, but **16.8% of Klinë and 14.6% of "
            "Gjakovë** — the Catholic Albanians of the Dukagjin plain in the west, who "
            "never converted under Ottoman rule — against 0.07% in Gjilan in the east. "
            "Mother Teresa came from this community. "
            "**The Orthodox that were counted are almost entirely in the enclaves.** "
            "Partesh is 99.5% Orthodox, Ranillug 94.5%, Shtërpcë 75.1%, Graçanicë 46.5% — "
            "Serb municipalities created after 2008, each a few thousand people, and each "
            "sitting inside an otherwise Muslim country. Those are real measurements, "
            "unlike the north's. "
            "**Only 0.50% report no religion — the lowest share of any country on this "
            "map**, and a third the number who declined to answer at all. Those 23,718 "
            "refusals are not drawn; 40% of them are in Prishtinë."),
        how="census, 2024",
        fill="from the agency's own estimate for the four northern municipalities",
        grain="municipalities, 41,000 people on average",
        counts=_xk_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "xk" / "xk_grid_400m.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_xk_place_weight,
        note="**THE COUNTS ARE CLEAN AND THE COUNTRY IS NOT, AND THOSE ARE SEPARATE "
             "FACTS.** ASK's `census2024_10.px` is an open PxWeb table — 10 KB, no key, no "
             "wall — and it reconciles exactly: six categories partition every unit, and "
             "the 38 municipalities sum to the national row category by category with a "
             "gap of zero. 21 cells come back blank and the exact partition is what proves "
             "they are true zeros rather than Lithuania's disclosure control (§9q); "
             "sources/xk.py asserts that rather than assuming it. "
             "**The catalogue walk nearly missed it for a dull reason.** askdata's PxWeb "
             "root returns `dbid` where every other PxWeb here returns `id`, so a walker "
             "keyed on `id` reads the root as nameless nodes and descends into none. "
             "Moldova's does the same and looked equally empty. "
             "**KOSOVO IS THE HOLE IN THE GISCO LAU FILE.** That file covers the EU27 plus "
             "the candidates — AL, BG, CH, IS, MK, NO, RS are all in it — and Kosovo is the "
             "one Balkan country it does not carry, because the EU has no agreed status "
             "for it. geoBoundaries XKX ADM2 supplies the 38 instead, joined by name: "
             "Albanian nouns have definite and indefinite forms and ASK writes one while "
             "geoBoundaries writes the other, so ten differ by a single final vowel. "
             "Stemming that away plus five explicit aliases gives 38/38. "
             "**The independent check measures the boycott instead of assuming it.** §9p "
             "verifies a name-join by requiring every unit's population ratio to sit in a "
             "tight band. Kosovo cannot pass that and should not: Kontur 2023 knows about "
             "people the 2024 census did not reach, so the 34 enumerated municipalities sit "
             "at a median 1.06x while Leposaviq, Zubin Potok and Zveçan come out at 3.2x, "
             "6.6x and 12.4x. The band is asserted on the 34 and reported on the four — the "
             "boycott confirmed by a second, unrelated source rather than by a news story. "
             "**North Mitrovica is the one it cannot see**, at 1.11x, because "
             "geoBoundaries splits Mitrovica along the Ibar through the middle of one "
             "continuous city and the north bank's hexes fall to the southern municipality. "
             "**THE FOUR NORTHERN MUNICIPALITIES ARE DERIVED FROM ASK'S OWN ESTIMATE — "
             "Anita, 2026-09-06, after one build that drew them as published.** Drawing "
             "them raw was wrong in a way no note could repair: the map itself said Zubin "
             "Potok was 89% Muslim. `census2024_63.px` is ASK's ethnicity table *with "
             "estimation*, identical to the enumerated one everywhere except these four, "
             "where it restores 16,949 people of whom **16,369 — 96.6% — are Serbs**. There "
             "is no religion-with-estimation table, so the step is Serb -> Orthodox: spec "
             "§14.5, and Kosovo passes its three tests more cleanly than China does. The "
             "Serb/Croat/Bosniak distinction *is* a religious boundary drawn over a common "
             "language, which is condition one; Kosovo's Serbs are not a religiously mixed "
             "group, which is condition two, and `sources/xk.py` asserts the addition stays "
             "above 90% one ethnicity so it cannot quietly stop being true; and both tables "
             "are per municipality, so nothing is spread finer than it was published. "
             "**Every derived row is `tier=\"derived\"`** and §7a's control strips them in "
             "one click. The enumerated rows are untouched — the derivation adds the missing "
             "Orthodox rather than restating what was counted — and the 580 non-Serb people "
             "in the estimate are left undrawn, because nothing says what they are and the "
             "enumerated composition of these four is precisely what is not representative. "
             "**No earlier census helps**: 2011 leaves the same four null and counts fewer "
             "Orthodox nationally (25,837), 1991 was boycotted from the Albanian side, and "
             "1981 — the last with full participation — asked nationality rather than "
             "religion, on pre-2008 boundaries. "
             "**Both censuses are in the table**, 2011 and 2024; only 2024 is drawn. "
             "Placement is Kontur's 400 m H3 grid, 9,258 hexes, because 38 municipalities "
             "over 10,900 km² average 287 km² and Kosovo's people are in the "
             "Prishtinë-Ferizaj-Prizren corridor rather than on the Sharr.",
    ),
}
