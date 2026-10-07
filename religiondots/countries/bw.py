# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _bw_place_weight(place):
    """countries.py hook. `place` is the 400 m Kontur hex layer scatter.py has read.

    Botswana needs it for the reason Namibia would: the localities are ADM3 catchments and
    they are enormous in the west and small in the east. Ghanzi, Kgalagadi and the CKGR run
    to tens of thousands of km² each with a few hundred people in them, while the Kgatleng
    and South East units around Gaborone are a few hundred km² holding tens of thousands.
    An equal share per polygon would paint the empty Kalahari and compress the corridor
    where almost everybody actually lives.
    """
    return _kontur_place_weight(place, "bw_hexes.gpkg", "sources/bw_geo.py")


def _bw_counts():
    """Botswana — eight categories at 519 ADM3 localities, 2011 census, with Catholics and
    Adventists split out of `Christian` from the Afrobarometer (the block below).

    THE UNIVERSE IS AGE 12 AND OVER, which is Peru's shape. 1,384,276 people answered a
    religion question in the eighteen district booklets that exist; the under-twelves were
    never asked and are in `gap=` rather than drawn as a §3.5 undercount. `Not stated` is
    5,146 people and is EXCLUDED, so what is drawn is an exact partition of the answers.

    TWO OF THE TWENTY-EIGHT DISTRICTS ARE NOT DRAWN AT ALL. Central Boteti and Central
    Bobonong had no booklet published; they are 129,312 people in 2011 and are named in
    `gap=`. That is why this file draws 93.7% of UNSD's national 2011 total rather than all
    of it, and sources/bw.py asserts exactly that band.

    THE TIER COMES OUT OF THE LOOKUP AND IS NOT RE-DERIVED HERE. `bw_geo.py` marks a row
    `measured` when the census locality matched its own ADM3 polygon by name, and `derived`
    when it is a district's `Other` residual or one of six Delta villages spread across the
    district instead. Deriving it from the weight would call a residual that happened to
    land on one polygon `measured`, which is the failure mode and not the shortcut.
    """
    from bw2011 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "bw.csv",
                     dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    df = df[df["geo_level"] == "locality"].copy()
    lut = pd.read_csv(HERE / "data" / "geo" / "bw" / "bw_lookup.csv",
                      dtype={"geo_id": str, "unit": str})
    orphan = sorted(set(df["geo_id"]) - set(lut["geo_id"]))
    if orphan:
        raise SystemExit(f"bw.csv places with no lookup row: {orphan[:6]} -- re-run "
                         "sources/bw_geo.py, the lookup is stale")
    # THE TWO CHURCHES (2026-10-03, sources/bw_churches.py, sources/bw.md §10) come out of each
    # locality's census `Christian` cell: subtracted here, added back as their own rows, always
    # `derived`, rolling back to `christianity` (bw2011.COLUMNS) when inferred dots are hidden.
    sp = pd.read_csv(HERE / "data" / "normalized" / "bw_split.csv",
                     dtype={"geo_id": str}, keep_default_na=False, na_values=[""])
    chr_rows = df["source_category"] == "Christian"
    taken = sp.groupby("geo_id")["count"].sum()
    have = df.loc[chr_rows].set_index("geo_id")["count"].astype(int)
    stray = sorted(set(taken.index) - set(have.index))
    if stray or (taken.reindex(have.index, fill_value=0) > have).any():
        raise SystemExit("bw_split.csv does not fit inside bw.csv's Christians -- re-run "
                         "sources/bw_churches.py")
    df.loc[chr_rows, "count"] = (df.loc[chr_rows, "count"].astype(int)
                                 - df.loc[chr_rows, "geo_id"].map(taken).fillna(0).astype(int))
    df["split"] = False
    sp["split"] = True
    df = pd.concat([df, sp], ignore_index=True)
    m = df.merge(lut, on="geo_id", how="left")
    m.loc[m["split"], "tier"] = "derived"
    m["node"] = m["source_category"].map(resolve)
    m = m[m["node"].notna()].copy()
    m["count"] = m["count"].astype(float) * m["weight"].astype(float)
    m = m[m["count"] > 0]
    if m["unit"].nunique() > 519:
        raise SystemExit(f"{m['unit'].nunique()} ADM3 units, expected at most 519")
    # One row per (unit, node). A polygon can receive both a named locality's own count and
    # a slice of its district's residual; where it does, the pair is part measurement and
    # part spreading, and the weaker tier is the honest label for it.
    g = m.groupby(["unit", "node"], as_index=False).agg(
        count=("count", "sum"),
        tier=("tier", lambda s: "derived" if (s == "derived").any() else "measured"),
        split=("split", "any"))
    # Only the church rows roll; a district residual is `derived` in the spatial sense and has
    # no coarser cell to fall back to (sources/bw.md §9).
    g["roll"] = g["split"].map(lambda s: "christianity" if s else None)
    g["congregations"] = 0
    return g[["unit", "node", "count", "tier", "congregations", "roll"]]


ENTRY = {
    "bw": dict(
        name="Botswana",
        source="Population and Housing Census 2011 Selected Indicators, the eighteen "
               "district booklets (Statistics Botswana); Catholics and Adventists from "
               "Afrobarometer rounds 4 to 8",
        basis="self-identification, population aged 12 and over",
        note_public=(
            "**Botswana recorded 14.8% of the people it asked as having no religion, and "
            "that is the highest figure on this map anywhere in sub-Saharan Africa.** "
            "Zimbabwe's is 8.3% and Zambia's census puts Christianity at 98%. "
            "**It is also not an urban figure, which is the surprising part.** The seven "
            "cities and towns come in at 9.8%, below the national rate; Kweneng West, the "
            "Kalahari district behind Molepolole, is **29.8%**, and Central Mahalapye and "
            "Central Serowe/Palapye are both above 21%. Irreligion in Botswana rises as you "
            "leave town, which is the reverse of the pattern almost everywhere else here. "
            "Individual villages go a long way further: Monwane, Tsetseng and Leologane are "
            "each above half. Kgalagadi South, also deep in the desert, is the most "
            "Christian district in the country at 90.8%, so this is a pattern about village "
            "size and distance rather than about any one part of Botswana. "
            "**Badimo is 3.9%, and read that as a floor.** It is the census's own Setswana "
            "word, the plural for the ancestors, and the box is exclusive of the Christian "
            "one, so it counts people who gave the ancestors instead of a church rather "
            "than the much larger number who keep both. The same caution applies across the "
            "continent and it has a second edge here: where a census offers this box, some "
            "people whose practice is ancestral rather than congregational answer no "
            "religion instead, so the two cells are not independent of each other. "
            "**The map is the 2011 census, not the 2022 one, and the reason is that 2022 "
            "published no geography at all.** Statistics Botswana asked the religion "
            "question again and printed it crossed with sex, marital status, employment "
            "and a three-way town/village/countryside split, and never once by district. "
            "The 2011 census was published the other way round: its national volumes have "
            "no geography either, but the office issued a separate booklet for each census "
            "district and every one of them carries a religion table by named village. "
            "**One thing the newer census does say should be held against this one.** It "
            "puts no religion at 6.9%, against 15.3% in 2011, a fall of more than half in "
            "eleven years that no plausible amount of conversion accounts for. Something "
            "about how the question was asked or coded changed between the two, and nobody "
            "has said what. The 14.8% drawn here is what enumerators recorded in 2011. "
            "**Two of the twenty-eight districts are missing.** Central Boteti and Central "
            "Bobonong are the gaps in the booklet series and were never put online, so "
            "129,312 people in the Central District are not drawn. Their absence is "
            "not neutral: the eight categories drawn here come to 93.7% of the national "
            "total the office reported to the UN, but Badimo comes to only 88.2% of its "
            "national figure, so the two missing districts are more traditional than the "
            "country as a whole and this map understates Badimo slightly. "
            "**Two churches come from a survey, and both are floors.** The census has one "
            "Christian box, 79.9%. The Afrobarometer names churches, but the share of its "
            "Christians who named none rose from 7% in 2008 to two thirds by 2019, and most "
            "churches' answers fell with it. Catholics and Seventh-day Adventists kept their "
            "share while it rose, so each village's census Christians are split at their "
            "district's survey share: Catholics are **4.9%** of Christians, highest in South "
            "East, Lobatse and Kweneng East, and Adventists 2.5%, highest in Ngamiland and "
            "Chobe. Some of their members will have answered just Christian, so both are "
            "low. The Zion Christian Church, the Congregational church of the London "
            "Missionary Society, the Anglicans, Methodists, Lutherans and Pentecostals stay "
            "in the one Christian colour."),
        how="census, 2011",
        grain="census localities, 2,900 people on average",
        fill="district totals where no village was named; two churches from a survey",
        gap="under-twelves, who were not asked, and the Central Boteti and Central "
            "Bobonong districts, whose census booklets were never published",
        counts=_bw_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "bw" / "bw_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_bw_place_weight,
        note="THE SOURCE IS EIGHTEEN SEPARATE BOOKLETS AND THEY DISAGREE WITH EACH OTHER "
             "ABOUT LAYOUT. Six put the row total in the last column, ten put it first, and "
             "Ghanzi has no total column at all; Chobe and Ngamiland East print no `Not "
             "stated` column. sources/bw.py detects all of that arithmetically, because the "
             "captions cannot be trusted: Central Serowe/Palapye captions both halves of "
             "its pair `(%)` while the first holds the counts, and the Cities and Towns "
             "booklet captions its religion table `Number of people by marital status`. "
             "TWO BOOKLETS DO NOT ADD UP AND BOTH ARE THE OFFICE'S ARITHMETIC. Ghanzi's "
             "printed total leaves out the CKGR row, which is a census district in its own "
             "right and is re-homed here; Central Tutume's total row is 16 out on Christian "
             "and 8 each the other way on Badimo and No religion, netting to zero. Both are "
             "asserted with a tight bound rather than tolerated silently. "
             "THE JOIN IS ON NAME AND KONTUR IS WHAT TESTS IT. There is no code on the "
             "census side, and eight ADM3 names occur twice in Botswana, so names are "
             "matched only within their own district. The log-log correlation between "
             "census locality population and Kontur's modelled population is 0.744 against "
             "0.174 for the best of 500 random pairings, which is the quantity the join "
             "does not determine. "
             "CATHOLICS AND ADVENTISTS ARE SPLIT OUT OF EACH LOCALITY'S CENSUS CHRISTIANS "
             "(2026-10-03, sources/bw_churches.py, sources/bw.md §10) at their survey unit's "
             "Afrobarometer share, rounds 4-8, `derived`, rolling back to christianity; the "
             "other churches fall with the rising `Christian only` and are not drawn.",
        view=[19.6, -27.2, 29.6, -17.6]),
}
