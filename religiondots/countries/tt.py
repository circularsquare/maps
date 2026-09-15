# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _tt_place_weight(place):
    """countries.py hook. `place` is the 400m hex layer scatter.py has read."""
    return _kontur_place_weight(place, "tt_hexes.gpkg", "sources/tt_grid.py")


def _tt_counts():
    """CSO 2011 census at municipality: 16 nodes on 15 units.

    ONE level, no allocation, nothing modelled — every drawn row is `measured` and may ring.
    tt.csv carries only the 15 drawn units; the table's `TRINIDAD AND TOBAGO` and `TRINIDAD`
    rows are nested universes used as checks in sources/tt.py and are not written.

    **`keep_default_na=False`, for the same reason as Belize.** `None` is a category name
    here too — 28,842 people — and a bare `pd.read_csv` silently turns it into NaN.

    **88.90% of the country is drawn, and the 11.10% that is not is the largest non-answer
    on this map outside the United States.** `Not Stated` is 146,798 people; §3.5 marks it
    rather than filling it, and `note_public` says so, because every share drawn here is a
    share of the whole non-institutional population rather than of the people who answered.

    **The universe is the NON-INSTITUTIONAL population**, 1,322,546 of the census's
    1,328,019. The missing 5,473 are not scaled in (§14.4).
    """
    from tt2011 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "tt.csv",
                     dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[])
    df = df[df["geo_level"] == "municipality"].copy()
    if df["geo_id"].nunique() != 15:
        raise SystemExit(f"{df['geo_id'].nunique()} municipalities, expected 15 -- re-run "
                         "sources/tt.py")
    if "None" not in set(df["source_category"]):
        raise SystemExit("tt.csv has no `None` category -- it has been read as NaN. "
                         "pd.read_csv needs keep_default_na=False here.")

    df["count"] = pd.to_numeric(df["count"])
    df["node"] = df["source_category"].map(resolve)
    unmapped = sorted(set(df.loc[df["node"].isna(), "source_category"])
                      - {"Not Stated", "Total"})
    if unmapped:
        raise SystemExit(f"tt.csv has unmapped source categories: {unmapped}")
    df = df[df["node"].notna() & (df["count"] > 0)]
    df = df.rename(columns={"geo_id": "unit"})
    df["congregations"] = 0
    df["tier"] = "measured"
    return df[["unit", "node", "count", "congregations", "tier"]]


ENTRY = {
    "tt": dict(
        name="Trinidad and Tobago",
        source="Census 2011 (Central Statistical Office)",
        basis="self-identification",
        view=[-62.10, 9.95, -60.40, 11.40],
        gap=("11.1% who did not state a religion; and the institutional population, in "
             "prisons, hospitals and homes"),
        gap_share=0.111,
        note_public=(
            "**The only census on this map that counts Orisha and Spiritual Baptists.** "
            "Between them they are **86,920 people, 6.6% of the country — more than its "
            "Muslims**, and neither tradition has ever been counted under its own name "
            "anywhere else this map draws. "
            "**Spiritual Baptists are 5.67%, more numerous than Trinidad's Anglicans** and "
            "nearly five times its ordinary Baptists, which the census counts as a separate "
            "answer. Baptist Protestantism and West African practice fused in this one "
            "rather than one absorbing the other — scripture and hymnody alongside spirit "
            "possession, the mourning ground, bell-ringing and water rites. The religion "
            "was **banned outright from 1917 to 1951**, and 30 March is a public holiday, "
            "Spiritual Baptist Liberation Day. They are 13.0% of Point Fortin and 10.6% of "
            "Tobago. "
            "**Orisha — historically Shango — is 11,918 people**, Yoruba orisha worship "
            "carried over in the nineteenth century and the direct sibling of Candomblé and "
            "Santería. Read it as a floor: in Trinidad, Orisha and Spiritual Baptist "
            "practice overlap heavily and many people take part in both, while a census "
            "offers one box. "
            "**Trinidad is the second Hindu geography in the Americas.** 240,100 people, "
            "18.2%, and it is concentrated exactly where indenture put it — **43.0% of "
            "Penal/Debe**, 31.3% of Couva/Tabaquite/Talparo, 30.0% of Chaguanas — against "
            "0.7% of Tobago. Muslims, at 5.0%, follow the same belt. "
            "**And the Presbyterians are Indo-Trinidadian, which the map shows without "
            "being told.** They peak in San Fernando (5.5%) and Penal/Debe (5.3%), the same "
            "units that are most Hindu, because Trinidad's Presbyterian church grew out of "
            "the Canadian Mission to the Indians from 1868 and drew its converts from the "
            "indentured population. Tobago is 0.2%. "
            "**Tobago is a different country religiously.** Roman Catholic 6.6% there "
            "against 44.8% in Diego Martin, with the country's highest Anglican (12.8%), "
            "Adventist (16.3%) and Pentecostal (14.7%) shares. Trinidad is Catholic and "
            "Hindu; Tobago is Protestant. "
            "**11.1% did not state a religion — the largest non-answer on this map outside "
            "the United States.** Every share here is a share of everybody, not of the "
            "people who answered, so a religion's share among answerers is about a tenth "
            "higher than what is drawn. Nothing redistributes it."),
        how="census, 2011",
        grain="municipalities, 88,000 people on average",
        counts=_tt_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "tt" / "tt_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_tt_place_weight,
        note="**THE SOURCE PDF THE PUBLISHER LINKS IS TRUNCATED, AND THE FINDING "
             "GENERALISES.** CSO serves the 2011 Demographic Report at two paths. The one "
             "the site links and a media search finds first — `/2019/03/TRINIDAD-AND-"
             "TOBAGO-2011-Demographic-Report.pdf` — is **281,190 bytes, starts `%PDF-1.4`, "
             "ends mid-stream with no `%%EOF`, and the server's `Content-Length` matches "
             "the delivered bytes exactly**. PyMuPDF opens it, sets `is_repaired=True` and "
             "reports `page_count = 0` without raising. The intact 442-page copy is live at "
             "`/2020/01/2011-Demographic-Report.pdf`. **A complete download is not an "
             "intact file — check the trailer, not the byte count** — and `sources/tt.py` "
             "asserts `%%EOF` and the page count in `fetch()`. Wayback also held two good "
             "captures, but the publisher's own second path was found first and a live copy "
             "is the citable one. `sources.md` §11t. "
             "**NO IDENTITY IN TABLE 8 IS EXACT AND THAT IS THE SOURCE, NOT THE PARSE.** "
             "CSO's 2011 figures are weighted — its own per-municipality workbooks publish "
             "fractional people — so every printed integer is independently rounded. Across "
             "the 359 identity checks `sources/tt.py` runs, the spread is **-2:4, -1:55, "
             "0:249, +1:47, +2:4**: symmetric, bounded at two people, 69% exact. The bound "
             "is asserted at 2 and the distribution is printed, because that is what would "
             "reveal a re-typeset page; a parse error is one-sided and large. "
             "**THE PARSE IS DRIVEN BY EXPECTATION AND WORKS ON TOKENS, NOT LINES.** Page "
             "168 packs three figures onto one line and puts the national row's first "
             "figure on its label line, while every other page is one figure per line. And "
             "**`-` is the nil marker**, not a missing value — a parser that skipped "
             "non-numeric tokens would shift every later figure in the row left by one and "
             "still produce nine plausible numbers. "
             "**ONE LABEL CHANGES CASE BETWEEN PANELS**: the island is `Tobago` in BOTH "
             "SEXES and `TOBAGO` in MALE and FEMALE. Matching is case-folded, which is safe "
             "only because the walk is expectation-driven — `TRINIDAD` and `TRINIDAD AND "
             "TOBAGO` are told apart by what comes next rather than by matching. "
             "**TWO NESTED UNIVERSES GIVE A TWO-LEVEL RECONCILIATION**: the 14 Trinidad "
             "municipalities sum to the printed `TRINIDAD` row, and TRINIDAD + Tobago sums "
             "to `TRINIDAD AND TOBAGO`, on all 18 columns. The table then repeats in full "
             "for MALE (p186-203) and FEMALE (p204-221), and Male + Female == Both Sexes on "
             "all 306 cells is the only check that would catch a token landing in the wrong "
             "column, since every other identity reconciles inside one panel. "
             "**THE JOIN IS 15/15 BOTH WAYS.** COD's ADM1 is the census's municipality tier "
             "exactly, independently confirmed by CSO publishing those same 15 as separate "
             "workbooks. Ten names differ — Table 8 writes `City of` and `Borough of` and "
             "uses `/` where COD uses `-` — and `fold()` handles all ten with no alias "
             "table. CSO publishes no code, so `tt.py` carries COD's pcode by name and "
             "`tt_geo.py` asserts the pairing from the other side. "
             "**PLACEMENT IS KONTUR'S 400 m GRID**, 4,610 hexes of 0.670 km². The problem "
             "here is not empty land but the unit size range — **Arima 13.1 km² to Sangre "
             "Grande 931.0 km², 71-fold** — and the small units are the dense ones. The "
             "per-municipality Kontur/census ratio runs **0.95x to 1.31x** around a national "
             "1.15x, which is a level shift across a twelve-year vintage gap rather than a "
             "shape problem, and only the shape is used. Arima and Port of Spain hold just "
             "18 hexes each; that is thin, and it is also where it matters least, which is "
             "the opposite of Saint Vincent's case where the grid was coarser than the "
             "counting units and was removed.",
    ),
}
