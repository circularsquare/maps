# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _nl_place_weight(place):
    return _kontur_place_weight(place, "nl_hexes.gpkg", "sources/nl_grid.py")


def _nl_counts():
    """CBS's own religion table for 403 gemeenten: 10 categories, 394 units drawn.

    THE SOURCE IS A CBS MAATWERK TABLE AND NOT A STATLINE ONE, which is why §11k closed the
    Netherlands on `82904NED` and was right about that table and wrong about the country.
    `Religie en kerkbezoek naar gemeente 2010-2014` pools five years of the Enquete
    Beroepsbevolking, 460,000 adults, and prints nine denominations plus the religious total
    as percentages of the 18-and-overs for every gemeente in the 2014 classification. It is
    the finest religion geography anywhere in western Europe on this map: 42,700 people to a
    unit, against Germany's Gemeinden and nothing else close.

    EVERY ROW IS `modelled`. This is a sample survey, not a register and not a census, and
    §7's tier is about how the number came to exist rather than about who published it.
    Greece, France, Italy and Finland are the same call.

    THE THREE REFORMED ANSWERS ARE THE POINT AND THEY NEEDED THREE NEW NODES. CBS offers
    `Nederlands hervormd`, `Gereformeerd` and `PKN` as separate boxes and Dutch respondents
    treat them as separate things: Staphorst is 47.5% hervormd, Urk 52.2% gereformeerd and
    Dongeradeel 32.4% PKN, and no two of those three maps look alike. taxonomy/nl2014.py has
    the reasoning and ask/009-nl-three-dutch-reformed-nodes-hervormd-gereform.md puts the
    legend cost to Anita.

    THE SHARES ARE OF ADULTS AND ARE APPLIED TO EVERYBODY, which scales 3.5 million Dutch
    children up into their gemeente's adult composition ([[feedback_leave_children_out]]).
    note_public says so. NINE GEMEENTEN ARE BLANK, the four inhabited Wadden islands and five
    small mainland ones, because CBS suppresses a gemeente with fewer than 150 respondents in
    the pool; they are 0.21% of the country and are the whole of `gap`.
    """
    from nl2014 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "nl.csv", dtype={"geo_id": str},
                     keep_default_na=False, na_values=[""])
    df = df[df["geo_level"] == "gemeente"].copy()
    if df["geo_id"].nunique() != 394:
        raise SystemExit(f"nl.csv has {df['geo_id'].nunique()} gemeenten, expected 394")

    df["node"] = df["source_category"].map(resolve)
    unmapped = sorted(set(df.loc[df["node"].isna(), "source_category"]))
    if unmapped:
        raise SystemExit(f"nl.csv has unmapped source categories: {unmapped}")
    df = df[df["count"] > 0]
    df["congregations"] = 0
    df["tier"] = "modelled"
    return df.rename(columns={"geo_id": "unit"})[
        ["unit", "node", "count", "congregations", "tier"]]


ENTRY = {
    "nl": dict(
        name="Netherlands",
        name_in="the Netherlands",
        source="CBS, Religie en kerkbezoek naar gemeente 2010-2014 "
               "(Enquete Beroepsbevolking)",
        basis="self-identification, sample survey",
        note_public=(
            "**The last Dutch census to ask about religion was in 1971, and then Statistics "
            "Netherlands asked 460,000 people anyway.** From 2010 to 2014 the religion "
            "question rode along on the labour force survey, which is large enough that CBS "
            "could publish the answers for every one of the country's 403 gemeenten. It did "
            "so once, in 2015, as a custom table, and the question left the survey the "
            "following year. Nothing published since comes close: the office's live series "
            "reaches forty regions with four categories, where this has **394 gemeenten** "
            "and nine denominations. So this is a good picture of the Netherlands as it was "
            "a decade ago rather than a poor picture of it now, and the last paragraph but "
            "one says how much has moved. "
            "**The three Protestant answers are the reason to draw this country at this "
            "depth.** CBS offers Nederlands hervormd, gereformeerd and the Protestantse "
            "Kerk in Nederland as three separate boxes, and Dutch people answer them as "
            "three separate things. Together they are **16.7%** of the country, and they "
            "are nowhere the same 16.7%: Staphorst is **47.5%** hervormd, Urk **52.2%** "
            "gereformeerd and Dongeradeel in Friesland **32.4%** PKN. The strip of "
            "gemeenten running from Zeeland through the Alblasserwaard and the Veluwe to "
            "north-west Overijssel is the bevindelijk gereformeerde Bible Belt, and here it "
            "is a row of units rather than a claim about a region. Taking the three "
            "together, Urk is **84.0%**, then Bunschoten **71.5%**, Staphorst **70.4%**, "
            "Putten **65.9%**, Aalburg **65.4%**, Zwartewaterland **62.8%**, Oldebroek "
            "**62.5%**, Dongeradeel **62.4%** and Rijssen-Holten **61.8%**. Urk is also the "
            "most religious "
            "gemeente in the country outright, at **98.1%**. "
            "**The bigger pattern is the Catholic south.** Catholicism is the largest "
            "single denomination at **25.9%** and it is concentrated in a way no Protestant "
            "answer is: Simpelveld in southern Limburg is **88.2%** Catholic and so is most "
            "of the province around it, while the whole of Limburg was **77.2%** religious "
            "over these five years. Between the two, no denomination at all is the largest "
            "answer in the country at **47.2%**, and it peaks above **78%** in the old "
            "peat colonies of east Groningen and in the villages north of Amsterdam. "
            "**Islam and Hinduism are city religions here, and the Hindu one is "
            "Surinamese.** Islam is **4.6%** of the country and reaches **17.1%** in "
            "Leerdam, **14.8%** in The Hague and **13.4%** in Rotterdam. Hinduism is "
            "**0.62%**, which is high for Europe, and nearly all of it is the Hindustani "
            "Surinamese who came around independence in 1975: The Hague is **4.6%** Hindu "
            "and Rotterdam **3.1%**, and outside the Randstad the answer barely appears. "
            "**Nine years is a long time in Dutch religion.** The same office's current "
            "survey, asking a differently worded question of people aged 15 and over, puts "
            "**42.9%** of the country in some denomination for 2021 to 2025, where this "
            "pool puts **52.8%**. The fall is not even: Limburg went from **77.2%** to "
            "**57.9%** and Groningen from **36.3%** to **33.7%**, so the Catholic south has "
            "lost most and the already secular north-east least. Nothing here is rescaled "
            "onto those newer figures, because they come from a different survey with a "
            "different age base and CBS treats the two as different series. "
            "**What this cannot do.** The question was asked of adults, and the shares of "
            "the 18-and-overs are applied to everybody, so the roughly one Dutch person in "
            "five who was a child in 2014 is drawn in their gemeente's adult composition. "
            "The second largest religious answer after Catholicism is `andere gezindte` at "
            "**4.4%**, and CBS never splits it: the evangelical and pentecostal churches, "
            "the Baptists, the Orthodox parishes, the Old Catholics, the Remonstrants, the "
            "Doopsgezinden and the Jehovah's Witnesses are all in there together. Its map "
            "is the Bible Belt, so most of it is probably free-church Protestant, but that "
            "is a reading of a residual and not a measurement. And the card has no atheist "
            "or agnostic box, so everyone who says they belong to no denomination lands in "
            "one category and the Netherlands puts nothing on the secular node."),
        how="survey, 460,000 adults, pooled 2010 to 2014",
        grain="gemeenten, 42,600 people on average",
        gap_share=0.002090,
        gap="nine gemeenten with too few respondents for CBS to publish, 0.21% of the "
            "country: Ameland, Schiermonnikoog, Terschelling, Vlieland, Rozendaal, "
            "Renswoude, Graft-De Rijp, Schermer and Zeevang",
        counts=_nl_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "nl" / "nl_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_nl_place_weight,
        note="THE TABLE IS ON THE MAATWERK SHELF AND THAT IS THE WHOLE FINDING. §11k closed "
             "the Netherlands on StatLine `82904NED`, which is the same question asked of a "
             "much smaller survey and published nationally, and the closure was right about "
             "that table. CBS also publishes `maatwerk`, custom tabulations made for "
             "somebody else and left up afterwards, and `Religie en kerkbezoek naar "
             "gemeente 2010-2014` (2015/20) lives there with nine denominations for all 403 "
             "gemeenten of the 2014 classification. It is not in the OData catalogue, it is "
             "not linked from the StatLine religion tables, and the CBS paper that draws "
             "maps from the same data (`De religieuze kaart van Nederland, 2010-2015`) "
             "publishes only the maps. **Ask a European statistical office's custom-table "
             "shelf before concluding it publishes religion nationally only** — this is the "
             "same shape as §12's rule that an SPA hides real APIs, one shelf up. "
             "THE BOUNDARY VINTAGE IS SOLVED BY PDOK'S PER-YEAR WFS. "
             "`service.pdok.nl/cbs/gebiedsindelingen/<year>/wfs/v1_0` takes the year as a "
             "PATH component, so the 2014 classification comes back without any "
             "back-dating; 2014 answers, and so do 2015, 2016, 2021 and 2025. Only "
             "`_gegeneraliseerd` exists for the older years, where 2023 also offers "
             "`_niet_gegeneraliseerd`. That matters here because the Netherlands merged 403 "
             "gemeenten down to 352 by 2021, and several of the units this country is "
             "interesting for (Molenwaard, Ferwerderadiel, Dongeradeel, Graft-De Rijp) are "
             "gone by then. "
             "KONTUR'S NL EXTRACT CARRIES A SLAB OF BELGIUM AND IT IS NOT A ROUNDING "
             "ERROR. 371,100 people whose nearest gemeente is Sluis, which is Brugge, "
             "Knokke and Zeebrugge sitting 5 to 25 km outside the country. "
             "sources/nl_grid.py separates them from the genuine coastal rim with a 300 m "
             "distance cap, snapping the rim (948 hexes, 94,337 people) per "
             "[[reference_archipelago_grid_snap]] and dropping the rest; without the cap a "
             "nearest-join would have hung Bruges on Zeeuws-Vlaanderen. The nearest-join "
             "has to be done in EPSG:28992, because a distance in degrees is silently "
             "meaningless and geopandas only warns. "
             "THE THREE REFORMED NODES ARE THE ONE THING FLAGGED TO ANITA "
             "(ask/009-nl-three-dutch-reformed-nodes-hervormd-gereform.md). They are three "
             "legend rows no other country uses, "
             "which AGENT_BRIEF §3 says is hers; the country ships with them because "
             "collapsing hervormd, gereformeerd and PKN onto "
             "`christianity.reformed.continental` would leave one Reformed colour over the "
             "country that invented the distinction, and reversing it is two lines of "
             "taxonomy/nl2014.py.",
    ),
}
