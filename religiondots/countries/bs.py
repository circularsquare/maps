# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _bs_place_weight(place):
    """countries.py hook. `place` is the 400m hex layer scatter.py has read."""
    return _kontur_place_weight(place, "bs_hexes.gpkg", "sources/bs_grid.py")


def _bs_counts():
    """BNSI 2022 census at island: 22 nodes on 18 units.

    ONE level, no allocation, nothing modelled — every drawn row is `measured` and may ring.

    **`keep_default_na=False` IS LOAD-BEARING**, for the third time in the Caribbean after
    Belize and Trinidad. `NONE` is a category name here too — 24,668 people, 6.20% of the
    country, the sixth largest answer — and a bare `pd.read_csv` turns those sixteen rows
    into NaN before `resolve()` ever sees them. Guarded below by asserting the category
    survives the parse.

    **95.21% of the Bahamas is drawn.** 379,091 of 398,165; what is not is `NOT STATED`,
    19,074 people, which taxonomy/bs2022.py excludes per §3.5.

    **ONE UNIT HOLDS 74.5% OF THE COUNTRY.** New Providence is 296,732 people, so most of
    what a reader sees on this map is one polygon's composition spread over Nassau by
    Kontur's grid. The seventeen Family Islands are the part with real geography in it, and
    they are also where the census's own suppression bites — see `bs2022.py` on `OTHER
    RELIGION`.
    """
    from bs2022 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "bs.csv",
                     dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[])
    df = df[df["geo_level"] == "island"].copy()
    if df["geo_id"].nunique() != 18:
        raise SystemExit(f"{df['geo_id'].nunique()} islands, expected 18 -- re-run "
                         "sources/bs.py")
    if "NONE" not in set(df["source_category"]):
        raise SystemExit("bs.csv has no `NONE` category -- it has been read as NaN. "
                         "pd.read_csv needs keep_default_na=False here; without it 6.2% of "
                         "the Bahamas disappears and every check in sources/bs.py still "
                         "passes, because that file checks the PDF and not this read.")

    df["count"] = pd.to_numeric(df["count"])
    df["node"] = df["source_category"].map(resolve)
    unmapped = sorted(set(df.loc[df["node"].isna(), "source_category"])
                      - {"NOT STATED", "TOTAL"})
    if unmapped:
        raise SystemExit(f"bs.csv has unmapped source categories: {unmapped}")
    df = df[df["node"].notna() & (df["count"] > 0)]
    df = df.rename(columns={"geo_id": "unit"})
    df["congregations"] = 0
    df["tier"] = "measured"
    return df[["unit", "node", "count", "congregations", "tier"]]


ENTRY = {
    "bs": dict(
        name="The Bahamas",
        name_in="the Bahamas",
        source="2022 Census of Population and Housing (Bahamas National Statistical "
               "Institute)",
        basis="self-identification",
        view=[-79.60, 20.75, -72.40, 27.45],
        gap=("the 4.8% who did not state a religion; and the smallest religions on 17 of the "
             "18 islands, pooled into one Other religion cell"),
        gap_share=0.0479,
        note_public=(
            "**The most Baptist country on this map, and it is not close.** One Bahamian in "
            "three — 135,875 people, **35.8% of everyone drawn here**. The next three are "
            "Saint Vincent at 9.3%, the United States at 7.3% and Jamaica at 6.9%, so the "
            "Bahamas is nearly four times the runner-up. Nothing about being Caribbean or "
            "Anglophone predicts it. "
            "**Two islands three kilometres apart are nothing like each other or the "
            "country.** Off the north end of Eleuthera, **Harbour Island is 32.0% Roman "
            "Catholic and 1.2% Baptist** — the only place in the Bahamas where the national "
            "religion is a rounding error — while **Spanish Wells is 23.2% Brethren**, "
            "against 1.5% nationally, a sixteenfold concentration of a church that barely "
            "registers anywhere else, alongside Methodists at 25.6% and almost no Catholics "
            "or Anglicans at all. Spanish Wells is a Loyalist fishing settlement and "
            "Harbour Island's Dunmore Town is the old colonial capital; the two histories "
            "are still legible in the answers. "
            "**Anglicanism is a Family Island religion here, not a Nassau one.** 43.2% of "
            "Long Island, 34.9% of Inagua, 34.7% of the Berry Islands, against 11.2% on "
            "New Providence. Seventh Day Adventists are **29.7% of Crooked Island**. "
            "Mayaguana is **73.9% Baptist**. "
            "**And Rastafari runs the opposite way from every stereotype about it** — "
            "highest on Cat Island (1.02%) and Andros (1.01%), lowest in Nassau (0.24%). "
            "It is the fourth census count of Rastafari on this map, after Jamaica, Saint "
            "Vincent and Trinidad. "
            "**This is the first census outside the United States to count African "
            "Methodists under their own name** — 1,028 people, and 282 of them on "
            "Eleuthera, which is 3.1% of that island against 0.02% of Grand Bahama. "
            "**It is also one of the very few that separates `no religion` from `atheist`**, "
            "and offers both boxes: 24,668 people take the first and 281 the second. Most "
            "censuses collapse the two and this map can then only show the first. "
            "**Three quarters of the country is one unit.** New Providence holds 296,732 of "
            "398,165 people, so for most Bahamians this map shows the composition of a "
            "single island spread across Nassau by where people live, not by what they "
            "answer street by street. The seventeen Family Islands are where the geography "
            "is real. "
            "**On seventeen of the eighteen islands the census pools its smallest answers.** "
            "Only New Providence prints all 24 religions; everywhere else the rarest go "
            "into one `Other religion` cell, with a footnote naming which. It is 372 people "
            "nationally — 0.09% — but 30.4% of Ragged Island and 11.8% of Mayaguana, and on "
            "those two the pooled cell swallowed the `no religion` answers too, so neither "
            "island reports any. "
            "**4.8% stated no religion at all** and are not drawn; that is 23.2% on "
            "Acklins, which the census does not explain."),
        how="census, 2022",
        grain="islands, 22,000 people on average",
        counts=_bs_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "bs" / "bs_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_bs_place_weight,
        note="**THE COUNTRY WAS 'OPEN' BECAUSE THE PUBLISHER'S OWN LISTING HIDES THE "
             "SOURCE.** §11t left the Bahamas as a lead: BNSI's first release — the one "
             "`bnsistats.gov.bs/publications` lists and the one a search finds — carries "
             "religion for **All Bahamas only**, and says so in its preface. The **579-page "
             "All-Island Report**, published June 2026 with a religion table per island, is "
             "absent from that listing and was found on the **CARICOM regional mirror** "
             "(§11v), which turned out to hold the census reports of the whole region. It "
             "IS on BNSI's own CDN, and that is what `sources/bs.py` fetches; the mirror is "
             "how the file becomes findable, not where it has to be cited from. "
             "**THE TABLE NUMBER IS THE CENSUS'S OWN ISLAND CODE.** The 2022 questionnaire "
             "pre-fills `Name of Island` from a numbered list of 18 and Tables 12.1-12.18 "
             "come in exactly that order, so `geo_id` is BNSI's and not invented here. The "
             "printed caption is checked against it on every run. "
             "**THE CATEGORY LIST IS NOT FIXED ACROSS ISLANDS**, which is the structural "
             "surprise and why the parse validates labels against a set instead of walking "
             "a fixed sequence the way `sources/tt.py` does. See `gap` and `taxonomy/"
             "bs2022.py`; the residual is 372 people and every island's footnote names its "
             "contents, which `bs.csv` carries in the `note` column. "
             "**FOUR RECONCILIATIONS, AND THE OUTER ONES ARE CROSS-DOCUMENT.** The seven "
             "age bands sum to each cell's own TOTAL column on all 753 cells — a check on "
             "every figure read rather than on the margins — Male + Female == TOTAL on 251, "
             "the categories sum to each island's own total on 18, and **the 18 islands sum "
             "to the FIRST RELEASE's national Table 6.0 category by category**, with the "
             "shortfall on each category exactly equal to what the island footnotes say was "
             "pooled. Two separately published tabulations agreeing to the person. "
             "**TWO CROSS-DOCUMENT ALIASES, ONE OF THEM BNSI'S TYPO.** The first release "
             "writes `Church of God (including Church of God of Prophecy)` where the "
             "All-Island Report writes `... AND Church of God of Prophecy`, and it spells "
             "the atheist row **`Athiest`**. Both are held in a two-entry table rather than "
             "solved by fuzzy matching, which would happily pair `Church of God` with "
             "`Church of God of Prophecy` if BNSI ever split them. "
             "**NOBODY PUBLISHES THE CENSUS'S OWN TIER, SO IT IS DISSOLVED.** Every "
             "boundary set for the Bahamas — COD-AB and geoBoundaries are the same geometry "
             "— gives the **32 local-government districts**, which nest inside the 18 "
             "islands exactly. Twenty-six carry their island in the name; the six cays that "
             "do not were each checked against BNSI's own publications rather than a map — "
             "**Black Point is enumeration district 420201 in `EXUMA AND CAYS POPULATION BY "
             "SETTLEMENT: 2010`**, and Mangrove Cay is named in the 2010 ANDROS report. "
             "**Harbour Island and Spanish Wells are census islands in their own right**, "
             "not part of Eleuthera, which is where the census tier and the geographic "
             "intuition disagree. The partition is asserted both ways. "
             "**PLACEMENT IS KONTUR'S 400 m GRID, AND 6% OF IT MISSES THE COUNTRY.** A "
             "plain `within` join leaves 836 hexes and 25,095 modelled people outside every "
             "island, because COD-AB's coastline is generalised GDAMS 2009 and a hex is "
             "400 m across. Measured before deciding: **all but three of those people are "
             "within 500 m of an island**, and nine of the twelve heaviest are Nassau's own "
             "waterfront. In a country where the population IS the coastline, dropping them "
             "tilts every island's dots inland — so they are snapped to the nearest island "
             "within **1 km**, a threshold that sits in an empty gap in the measured "
             "distribution rather than through the middle of it. The histogram prints on "
             "every run. "
             "**THE PER-ISLAND KONTUR/CENSUS RATIO RUNS 0.80x TO 1.92x** around a national "
             "1.036x. Exuma at 1.92x and Abaco at 1.32x are second homes and resorts read "
             "as population by a building-footprint model, not a grouping error; only the "
             "within-island shape is used, so no island gets the wrong number of dots "
             "(§9t).",
    ),
}
