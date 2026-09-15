# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _ec_place_weight(place):
    """countries.py hook. `place` is the 400m hex layer scatter.py has read.

    Ecuador needs this for BOTH of §8.2's reasons at once and harder than either country
    before it. EMPTINESS: the six Amazonian provinces are 44% of the land and 5.4% of the
    people, and Pastaza alone is 29,000 km2 -- bigger than Belgium -- with 111,915 people
    almost all on the Puyo road. CONCENTRATION: Guayas and Pichincha are 44% of the people
    on 5% of the land, and inside them Quito and Guayaquil are each a couple of hundred km2
    holding two to three million. An equal share per polygon puts Quito's Catholics on the
    slopes of a volcano and paints a religion across a roadless basin (sources/ec_grid.py).
    """
    return _kontur_place_weight(place, "ec_hexes.gpkg", "sources/ec_grid.py")


def _ec_counts():
    """LAPOP AmericasBarometer, waves 2010-2023 pooled, at provincia: 11 categories, 24
    units, and EVERY ROW IS `modelled` IN §7.

    The third country drawn from the AmericasBarometer, after Guatemala and El Salvador.
    sources.md §11x closed Ecuador for having no census religion question; this build
    CORROBORATES that from INEC's own metadata rather than by reputation, the way §11ac
    closed Venezuela and Colombia. The 2022 census person file publishes 88 variables in
    INEC's ANDA catalogue and none is religion -- Ecuador asks what people are (P11R
    self-identification, P12 indigenous nationality), what they speak (P1001I) and whom they
    descend from, and never what they believe.

    THE POPULATION IS A CENSUS COUNT AND NOT COD-PS, WHICH IS NEW IN THIS SET. Guatemala and
    El Salvador are drawn on OCHA projections because neither has counted recently. Ecuador
    counted 16,938,986 people in November 2022. COD-PS's 2020 projection says 17,510,643 --
    3.4% high, and UNEVENLY: Loja -6.9%, Galapagos -13.5%, Pichincha -4.3%, against Manabi
    +2.0%. A 6-point swing between the second and third provinces of the country is a
    difference in shape and not only in level, and §14.4 rule 1's promise -- every person
    drawn is a person somebody counted in that unit -- is only true if the somebody counted
    them. It also fixes a unit with no polygon: COD-PS 2020 carries a 25th row holding
    41,907 people in a `zona no delimitada` that COD-AB 2024 has no boundary for, because
    Ecuador resolved those disputed zones by referendum in 2015-16. The census has 24
    provinces summing exactly to the national total, one per polygon.

    THE THINNEST-SAMPLED OF THE SET. 7,387 respondents over 23 provinces is 321 apiece
    against El Salvador's 647 and Guatemala's 405, and Ecuador is in four rounds rather than
    six -- 2010, 2012, 2016 and 2023, not 2014 or 2018.

    THREE PROVINCES ARE ASSUMED AND ONE IS LEFT BLANK, AND THE LINE BETWEEN THEM IS WHETHER
    ANYTHING MEASURED THE PLACE AT ALL. Carchi (n=20), Pastaza (n=32) and Orellana (n=55) are
    in the 2010 wave and no other, so the split-half cannot rank them twice and drops them
    from the test that licenses the other twenty -- §14.16's rule about categories applied to
    units. But one wave DID measure them, which is enough to anchor an assumption, so they are
    drawn at the national rate on their own census population: 466,909 people, 2.76% of
    Ecuador. Anita, 2026-09-08: "i feel like carchi is fine to assume and we can just do it."
    GALAPAGOS IS NOT DRAWN AT ALL. LAPOP has no code 920 -- not unsampled, not OFFERED -- so
    no respondent could ever have been placed there and there is no reading of any kind to
    anchor an assumption on. Its 28,583 people are in `gap=` and are not drawn as a §3.5
    undercount; the province keeps its polygon and its hexes and draws no religion. Anita,
    2026-09-08: "galapagos maybe we just leave empty for now. no data." It is also the case
    that makes the distinction worth having: Galapagos is globally famous, so a
    confident-looking national average there is a claim a reader will check and this map
    cannot support.
    DRAWING THE THREE ON THEIR OWN SHARES IS WORSE THREE WAYS: their sampling half-widths are
    +/-19, +/-15 and +/-11 points so none is distinguishable from the national rate anyway;
    they would carry a 2010 LEVEL against every other province's four-wave average, and 2010
    is 4.16 points more Catholic, which is a bias rather than noise; and Carchi's twenty
    interviews contain three of the eleven answers -- eighteen Catholics, one `Ninguna`, one
    agnostic -- so drawing them would give a province of 172,828 people ZERO Evangelicals in
    an 11%-Evangelical country, and hard zeros on eight of the eleven categories.
    A NEIGHBOUR-AVERAGE FALLBACK WAS TESTED AND LOST. Leave-one-out over the twenty measured
    provinces, predicting each from its bordering provinces rather than from the country:
    Catholic 8.08 pp mean error against 7.60, Evangelical 6.14 against 5.29, `Ninguna` 3.39
    against 3.41. Ecuador's religion jumps at province lines rather than varying smoothly,
    because sierra, coast and Amazon interleave. Galapagos has no land neighbours anyway.
    Guatemala drew Zacapa on n=40 and said so, which is the opposite call -- the difference
    is that Zacapa is in every wave and therefore inside the test.

    THE UNIVERSE IS ADULTS AND THE DOTS ARE EVERYBODY, as in both earlier countries. LAPOP
    interviews people aged 18 and over; the shares are applied to the whole population, which
    assumes Ecuador's children are distributed like its adults. Drawing only the adults would
    leave 30% of the country blank, and §6.12 is about how badly a blank reads on a dot map.
    So there is no `gap=` here: nobody is left undrawn, and what is assumed is stated instead.

    THREE CATEGORIES CARRY THEIR OWN PROVINCE GEOGRAPHY on a bar of +0.45 (20 units):
    Catholic +0.71, Evangelical +0.48 and `Ninguna (creyente)` +0.63. So Ecuador is the
    second country here whose no-religion geography is a measurement rather than a national
    rate spread flat. Evangelical clears by 0.03, which is the mirror of El Salvador's
    `Protestante Tradicional` missing by 0.02; both were left where the arithmetic put them.

    AND THE ANSWER CARD CHANGED AFTER 2016, WHICH IS A FINDING ABOUT THE INSTRUMENT AND NOT
    ABOUT ECUADOR. Code 77 `Otro` is EXACTLY ZERO in 2010, 2012 and 2014 across all 28
    countries, on 29,374 / 27,254 / 36,725 valid answers, and appears from 2016. Codes 6, 10
    and 12 -- Mormons, Jewish, Jehovah's Witnesses -- are EXACTLY ZERO in 2018 and 2023 on
    15,107 / 25,649 answers. Zero Witnesses among 25,649 Latin Americans is a missing box,
    not a measurement: LAPOP withdrew the named small denominations and they fall into
    `Otro`. Nothing leaves the partition and the total is right; what is wrong is the
    attribution of about half a percent of Ecuador between `christianity.witnesses` (a
    floor) and `other.ec` (inflated at the late end). It is stated rather than corrected
    because correcting it means deciding how the 2023 `Otro` decomposes and nothing
    published says. THIS APPLIES TO GUATEMALA AND EL SALVADOR TOO and neither file mentions
    it -- it was found while building Ecuador. taxonomy/ec2023.py carries the measurement.
    """
    from ec2023 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "ec.csv",
                     dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])

    lut = pd.read_csv(HERE / "data" / "geo" / "ec" / "ec_lookup.csv", dtype=str)
    df["unit"] = df["geo_id"].map(dict(zip(lut["geo_id"], lut["unit"])))
    missing = sorted(df.loc[df["unit"].isna(), "geo_id"].unique())
    if missing:
        raise SystemExit(f"ec.csv provinces with no polygon: {missing} -- re-run "
                         "sources/ec_geo.py, the lookup is stale")
    # 23 of Ecuador's 24 provinces. Galapagos is deliberately absent -- see the docstring
    # and `gap=`; it keeps its polygon and its hexes and draws no religion.
    if df["unit"].nunique() != 23:
        raise SystemExit(f"{df['unit'].nunique()} provinces, expected 23")
    if "EC20" in set(df["unit"]):
        raise SystemExit("Galapagos is in ec.csv and must not be -- re-run sources/ec.py")

    df["node"] = df["source_category"].map(resolve)
    unmapped = sorted(df.loc[df["node"].isna(), "source_category"].unique())
    if unmapped:
        raise SystemExit(f"ec.csv categories with no node: {unmapped}")
    df = df[df["count"] > 0]
    # EVERY row, without exception -- there is no measured tier in this country (§7).
    df["tier"] = "modelled"
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations", "tier"]]


ENTRY = {
    "ec": dict(
        name="Ecuador",
        source="AmericasBarometer, four rounds 2010 to 2023 (LAPOP Lab, Vanderbilt "
               "University), against the VIII Censo de Población y VII de Vivienda 2022 "
               "province populations (INEC)",
        basis="self-identification, adults 18 and over",
        view=[-81.3, -5.2, -75.0, 1.7],
        note_public=(
            "**Ecuador's census does not ask about religion, so this is a survey standing "
            "where a census would be.** The 2022 census asked 88 questions of every person "
            "in the country and not one of them was about belief: it asks what people call "
            "themselves, what languages they speak and which indigenous nationality they "
            "belong to, and then stops. The map is drawn from the LAPOP AmericasBarometer "
            "instead, **7,387 people** across four rounds between 2010 and 2023, pooled. "
            "The dots are drawn desaturated to say so. "
            "**What the survey is laid on is a real count, which is unusual here.** "
            "Guatemala and El Salvador are drawn on population projections because neither "
            "has counted recently. Ecuador counted 16,938,986 people in November 2022, and "
            "the projections it replaced were 3.4% too high and wrong by different amounts "
            "in different places: 6.9% too high in Loja and 13.5% too high in Galápagos, "
            "while Manabí was 2.0% too low. "
            "**Three quarters of Ecuador is Catholic, which is thirty points more than "
            "either Central American country on this survey, and the interesting thing is "
            "how unevenly it thins.** It is **94.0% of Loja** in the far south against "
            "**61.7% of Sucumbíos** on the Colombian oil frontier, and the evangelical map "
            "is almost the negative of it: 29.5% of Sucumbíos and 20.5% of Santa Elena "
            "against 0.8% of Loja. The two big cities pull apart too, with Guayaquil's "
            "province at 67.1% Catholic and 18.7% Evangelical while Quito's is 74.8% and "
            "9.1%. "
            "**The people who believe in a higher being but belong to no religion are drawn "
            "where the survey found them**, which only El Salvador has managed before. They "
            "are **12.7% of Esmeraldas**, the Afro-Ecuadorian coastal province, and 10.3% "
            "of El Oro, against 0.3% of Zamora Chinchipe in the southern Amazon. That is a "
            "spread of forty to one, the widest this answer has shown anywhere on the map. "
            "**Galápagos is blank, because the survey never went there.** It is not that the "
            "islands were missed in some rounds: they were never on the answer card at all, "
            "so no Ecuadorian could have been recorded there. Rather than paint 28,583 "
            "people the same colours as the mainland and let the map look as confident about "
            "the archipelago as about Quito, it is left empty. "
            "**Three more provinces are drawn at the national rate rather than their own "
            "numbers.** Carchi, Pastaza and Orellana were visited in the 2010 round and no "
            "other, on twenty, thirty-two and fifty-five interviews, so there is no second "
            "reading to check the first against. They hold 2.76% of the country. Carchi is "
            "the reason it matters: its twenty interviews contain no evangelicals at all, and "
            "drawing that would have put a hard zero across a province of 172,828 people in a "
            "country that is 11% evangelical. Using neighbouring provinces instead of the "
            "national figure was tried and measured, and it predicted the rest of the country "
            "slightly worse, because religion here changes sharply at province lines rather "
            "than blending across them. "
            "**0.06% for traditional religion is a floor, in a country that is 7.69% "
            "indigenous by its own 2022 census.** The card offers one worldwide "
            "`Religiones Tradicionales` box, and the same instrument was checked against a "
            "census in Suriname where that box comes out at **0.21x** the census figure. "
            "The Kichwa of the sierra and the Shuar and Waorani of the Oriente answer this "
            "question Catholic or Evangelical, which is true as far as it goes. "
            "**Two more answers are undercounts, and for a duller reason: the question "
            "changed.** After 2016 the survey stopped offering Jehovah's Witnesses, Mormons "
            "and Jews as answers of their own and folded them into `other`. So the 1.43% "
            "here for Witnesses is what three of the four rounds found, diluted by a fourth "
            "that could not record them at all. "
            "**The level is a fourteen-year average and Ecuador moved a long way inside "
            "it.** Catholic identification runs 79.7% in 2010 to **67.7% in 2023** across "
            "the pooled rounds, while the believers without a religion go from 2.6% to "
            "10.0%. So this map is several points more Catholic than the last round alone "
            "would draw. Pooling is what buys the province detail: a single round is 1,500 "
            "people over twenty-three provinces."),
        how="survey, four rounds 2010 to 2023 pooled",
        grain="provinces, 735,000 people on average",
        gap=("Galapagos, 28,583 people, 0.2% of the country, which the survey never offered as "
             "an answer"),
        gap_share=0.0017,
        counts=_ec_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "ec" / "ec_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_ec_place_weight,
        note="THE COUNTRY IS A SURVEY ON A CENSUS COUNT, AND THE COUNT IS WHAT IS NEW. "
             "Guatemala and El Salvador put LAPOP's `q3c` on OCHA COD-PS projections; "
             "Ecuador has counted since, so the magnitude is INEC's own 2022 census. That "
             "is not a preference for the local source: COD-PS's 2020 projection is 3.4% "
             "high overall and UNEVENLY (Loja -6.9%, Galapagos -13.5%, Pichincha -4.3%, "
             "Manabi +2.0%), so it would have distorted the shares as well as the total. It "
             "also carries a 25th row -- 41,907 people in a `zona no delimitada` -- that "
             "COD-AB 2024 has no polygon for, because Ecuador assigned those disputed zones "
             "by referendum in 2015-16. The census has 24 provinces, one per polygon, "
             "summing exactly. Every row is still `modelled` (§7b): nobody counted religion. "
             "THE CENSUS NEGATIVE IS FROM INEC'S OWN METADATA, not from reputation. The 2022 "
             "person file in the ANDA catalogue publishes 88 variables and none is religion, "
             "which is how §11ac closed Venezuela and Colombia and is a stronger negative "
             "than §11x's. "
             "THE JOIN IS GUATEMALA'S AND BOTH KEYS ARE REQUIRED. LAPOP's `prov` is 900 plus "
             "INEC's official province number and COD's pcode is `EC` plus the same number, "
             "so sources/ec_geo.py refuses to run unless the NAMES on those codes agree too, "
             "one at a time, on all 23. One alias is needed: LAPOP abbreviates 923 to `S.D. "
             "De los Tsachilas`. sources/sv_geo.py is why both are checked -- there the two "
             "keys disagree on twelve of fourteen rows while every total reconciles. "
             "THE HELD-OUT CHECK IS THE STRONGEST IN THE SET. LAPOP's weighted province "
             "distribution tracks the census population distribution at r=+0.994 over 23 "
             "provinces, and NONE of 20,000 random pairings reaches it (best random +0.954). "
             "The age comparison is printed beside it and decides nothing, as in both earlier "
             "countries: F=0.35, so it has no power. "
             "THREE PROVINCES ARE ASSUMED AND ONE IS BLANK, AND THE LINE IS WHETHER ANYTHING "
             "MEASURED THE PLACE. Carchi, Pastaza and Orellana are in the 2010 wave only, so "
             "the split-half cannot rank them twice -- §14.16 applied to units instead of "
             "categories -- but one wave did measure them, so they are assumed at the "
             "national rate (2.76%, Anita's call). GALAPAGOS IS NOT DRAWN: LAPOP has no code "
             "920, so nothing measured it at all and there is nothing to assume from. Its "
             "28,583 people are in `gap=`. sources/ec.py has the three reasons drawing the "
             "three on their own shares would have been worse, of which the sharpest is "
             "Carchi's 0.0% Evangelical on twenty interviews, and the neighbour-average "
             "fallback that was tested against the national rate and lost. "
             "AND THE ANSWER CARD CHANGED AFTER 2016, WHICH IS ABOUT THE INSTRUMENT AND NOT "
             "ABOUT ECUADOR. `Otro` is exactly zero in 2010, 2012 and 2014 across all 28 "
             "countries and appears from 2016; Mormons, Jews and Jehovah's Witnesses are "
             "exactly zero in 2018 and 2023 on 15,107 and 25,649 answers. That is a withdrawn "
             "box, not a collapse. GUATEMALA AND EL SALVADOR ARE AFFECTED THE SAME WAY and "
             "neither file mentions it; it was found while building this country. "
             "taxonomy/ec2023.py carries the measurement and sources/ec.md the write-up.",
    ),
}
