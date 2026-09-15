# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _cr_place_weight(place):
    """countries.py hook. `place` is the 400m hex layer scatter.py has read.

    SEVEN UNITS OVER 51,169 km2 IS 7,310 km2 APIECE, the loosest ratio of any country drawn
    from this survey, and 8.2 binds in both directions at once. Guanacaste, Puntarenas and
    Limon are 30,668 km2 between them -- the whole Pacific coast, the Osa peninsula and
    Corcovado, Talamanca and the Caribbean lowland -- and hold 27% of the country, so an
    equal share per polygon paints a religion across rainforest. At the other end the Valle
    Central is most of Costa Rica's people on a plateau about 60 km across that crosses four
    provincial lines rather than sitting inside one, and Heredia is a narrow strip running
    from that plateau to the Nicaraguan border with all its people at the southern end
    (sources/cr_grid.py).
    """
    return _kontur_place_weight(place, "cr_hexes.gpkg", "sources/cr_grid.py")


def _cr_counts():
    """LAPOP AmericasBarometer, waves 2010-2023 pooled, at provincia: 11 categories, 7 units,
    and EVERY ROW IS `modelled` IN 7.

    The fifth country drawn from the AmericasBarometer, after Guatemala, El Salvador, Ecuador
    and Panama, and the one whose office was tested hardest before the survey was touched.
    INEC's own microdata catalogue at sistemas.inec.cr/pad5 serves 183 studies with 45,364
    variables between them and EXACTLY ONE is a religious affiliation question: `HC1A`, the
    religion of the household head, in the Encuesta de Mujeres, Ninez y Adolescencia 2018,
    Costa Rica's MICS round 6. The Censo 2011's own dictionary is 116 variables that include
    indigenous identity, indigenous language and Afrodescendant self-identification and no
    religion. ENAHO runs to 594 variables in 2023 and 596 in 2025 with none. The EMNA's
    341-page report tabulates its own religion question nowhere.

    BUT THE CATALOGUE PUBLISHED THE NATIONAL LEVEL WITHOUT MEANING TO. `HC1A`'s variable-level
    metadata carries its weighted frequency distribution -- 8,490 households, 65.16% Catolica
    -- which is the only independent reading of Costa Rica's level that exists, and the second
    such reading any country in this module has. sources/cr.md 4 compares the two cards box
    by box, because they do not line up.

    AND THE POPULATION IS INEC'S OWN 2022 ESTIMATE, NOT COD-PS, which is Ecuador's call
    (9bn): COD-PS ships a 2021 UNFPA projection built on INEC's superseded 2013 revision and
    it runs +11.24% on Heredia and -3.25% on Guanacaste against INEC's own count.
    """
    from cr2023 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "cr.csv",
                     dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])

    lut = pd.read_csv(HERE / "data" / "geo" / "cr" / "cr_lookup.csv", dtype=str)
    df["unit"] = df["geo_id"].map(dict(zip(lut["geo_id"], lut["unit"])))
    missing = sorted(df.loc[df["unit"].isna(), "geo_id"].unique())
    if missing:
        raise SystemExit(f"cr.csv provinces with no polygon: {missing} -- re-run "
                         "sources/cr_geo.py, the lookup is stale")
    # All seven, unlike Panama and Ecuador: LAPOP names every Costa Rican province and
    # samples every one of them in every wave, so this country has no `gap=`.
    if df["unit"].nunique() != 7:
        raise SystemExit(f"{df['unit'].nunique()} provinces, expected 7")

    df["node"] = df["source_category"].map(resolve)
    unmapped = sorted(df.loc[df["node"].isna(), "source_category"].unique())
    if unmapped:
        raise SystemExit(f"cr.csv categories with no node: {unmapped}")
    df = df[df["count"] > 0]
    # EVERY row, without exception -- there is no measured tier in this country (7).
    df["tier"] = "modelled"
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations", "tier"]]


ENTRY = {
    "cr": dict(
        name="Costa Rica",
        source="AmericasBarometer, four rounds 2010 to 2023 (LAPOP Lab, Vanderbilt "
               "University), against INEC's Estimación de Población y Vivienda 2022",
        basis="self-identification, adults 18 and over",
        view=[-86.2, 7.9, -82.3, 11.4],
        note_public=(
            "**Costa Rica's census does not ask about religion, and neither does its "
            "household survey.** INEC's own microdata catalogue is the evidence for that "
            "rather than a search: it serves 183 studies with **45,364** variables between "
            "them, and exactly one of those variables is a question about which religion "
            "somebody belongs to. The 2011 census asks every person whether they consider "
            "themselves indigenous, which indigenous people they belong to, and how they "
            "identify by race, and asks nothing about belief. The Encuesta Nacional de "
            "Hogares runs to 594 variables in 2023 and 596 in 2025 with none. So the map is "
            "drawn from the LAPOP AmericasBarometer instead, **5,903** people across four "
            "rounds between 2010 and 2023, pooled and applied to INEC's 2022 population "
            "estimate. The dots are drawn desaturated to say that. "
            "**The one survey INEC does ask it on published the answer without publishing a "
            "table.** The Encuesta de Mujeres, Niñez y Adolescencia of 2018, Costa Rica's "
            "round of the UNICEF household survey programme, asked **8,490** households the "
            "religion of the head of the household, and its 341-page report tabulates none "
            "of it; the distribution survives in the variable metadata of INEC's own "
            "catalogue. It reads **65.16%** Catholic, 25.50% Christian of some other kind "
            "and 7.40% with no religion. That is the only independent reading of Costa "
            "Rica's level that exists, and this map lands close to it on two of the three. "
            "Catholic is **63.4%** here. INEC's second box names Mormons inside itself and "
            "has no Protestant, Adventist or Witness box, so what it should be compared "
            "against is this map's evangelical, Protestant, Latter-day Saint and Jehovah's "
            "Witness cells added together, which come to **24.5%**. The third does not "
            "agree: 10.6% here against 7.4%, and the two are not asking the same question, "
            "because INEC asks about the household head and the barometer asks a randomly "
            "chosen adult. "
            "**Cartago and Puntarenas are the two ends of the country.** Cartago is "
            "**82.1%** Catholic, 6.2% Evangelical and 3.6% believing without a religion. "
            "Puntarenas is 50.9%, **23.7%** and 12.3%. "
            "**The largest answer on the map is also the least steady one.** "
            "Catholicism is 63.4% of Costa Rica and it has the weakest provincial ranking of "
            "the five answers drawn where the survey found them: ranked in the early rounds "
            "and again in the later ones it comes back at **+0.79**, against a bar of +0.71 "
            "that a random ordering of seven provinces beats one time in twenty. Almost all "
            "of the wobble is Guanacaste, which falls from the fourth most Catholic province "
            "to the seventh as its Catholic share drops from 63.4% to 48.2%. Until September "
            "2026 this answer was spread flat instead, because the bar had been set with a "
            "formula that made it a one in fifty test rather than the one in twenty it was "
            "meant to be; correcting that moved this answer and one in El Salvador, and "
            "nothing else in the five countries drawn from this survey. "
            "**This survey never offers a plain no religion box.** It offers a believer who "
            "belongs to no religion and a person who does not believe in God as two separate "
            "answers, and here the first is **9.2%** of the country and the second 1.4%, six "
            "to one. Guanacaste is 15.3% believers without a religion against 3.6% in "
            "Cartago. "
            "**Three answers are floors, because the answer card changed between the "
            "rounds.** Jehovah's Witnesses read 1.9%, 1.1% and 1.8% in 2010, 2012 and 2014 "
            "and then exactly zero in 2023; Latter-day Saints and Jews do the same. `Otro` "
            "is the mirror image, absent until 2023 and 2.4% of that round alone. Nobody "
            "stopped being a Jehovah's Witness in Costa Rica between 2014 and 2023. Costa "
            "Rica is still the only country drawn from this survey whose Witnesses are "
            "placed on their own provincial shares rather than spread at the national rate, "
            "**1.2%** of the country and 2.3% of Puntarenas against 0.35% of Cartago. "
            "**Traditional indigenous religion is a floor as well.** It comes back at "
            "**0.31%**, eighteen respondents in thirteen years, from a card that offers one "
            "worldwide box for it, in a country whose census asks every person which of its "
            "indigenous peoples they belong to. The same card read a fifth of a census on "
            "that cell in Suriname."),
        how="survey, four rounds 2010 to 2023 pooled",
        grain="provinces, 721,000 people on average",
        counts=_cr_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "cr" / "cr_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_cr_place_weight,
        note="THE COUNTRY IS A SURVEY ON AN OFFICIAL POPULATION ESTIMATE AND EVERY ROW IS "
             "`modelled` (7b), the same construction as Guatemala, El Salvador, Ecuador and "
             "Panama: LAPOP's `q3c` gives a province share, the magnitude comes from INEC's "
             "own table, and no magnitude is invented. "
             "THE OFFICE WAS TESTED BEFORE THE SURVEY WAS TOUCHED, and harder than for any "
             "other country in this set. sistemas.inec.cr/pad5 is INEC's own NADA microdata "
             "catalogue: 183 studies, 45,364 variables read, and ONE affiliation question in "
             "all of them, `HC1A` in the EMNA 2018. Censo 2011's dictionary is 116 variables "
             "with no religion; ENAHO has none in any year; the EMNA's own 341-page report "
             "tabulates its own question nowhere. But the catalogue's variable metadata "
             "carries `HC1A`'s weighted frequencies, which is the level check note_public "
             "quotes, and the microdata itself is behind a login at INEC, at UNICEF and at "
             "the World Bank alike. sources/cr.md has the sweep. "
             "THE POPULATION IS INEC'S OWN 2022 ESTIMATE AND NOT COD-PS, which is Ecuador's "
             "call (9bn). COD-PS ships a 2021 UNFPA projection built on INEC's superseded "
             "2013 projection revision; against INEC's own count it runs +11.24% on Heredia "
             "and -3.25% on Guanacaste, a 14.5-point spread in both directions across seven "
             "units, and it is the older vintage of the two. sources/cr_geo.py prints the "
             "comparison every run and checks the typed table against the growth-rate column "
             "printed beside it. "
             "SEVEN UNITS AND ALL SEVEN MEASURED, so there is no `gap=`. LAPOP's `prov` "
             "value labels name every Costa Rican province in Spanish and every one is "
             "sampled in every wave. The join is on the NAME with no alias needed. Costa "
             "Rica is also the one country in this set where `prov - 600` -> `CRn` would "
             "have been right, which is exactly why it is not used: the same arithmetic "
             "mispairs twelve of fourteen in El Salvador and eight of ten in Panama, and "
             "sources/cr_geo.py asserts the code join AGREES so that OCHA re-cutting the "
             "pcodes fails loudly. "
             "NEITHER PRE-REGISTERED HELD-OUT TEST PASSED, AND NEITHER BAR HAS BEEN MOVED. "
             "Two were written to a file before the data was loaded, and that file's mtime "
             "precedes every Costa Rican artefact on disk (sources/cr.md 7). The primary, a "
             "population permutation check, gives an exact p of 6.95e-3 over all 5,040 "
             "orderings against a bar of 1e-3. "
             "It is reported and never asserted on, because it is measured to have "
             "no power here: Costa Rica is San Jose, Alajuela and then five provinces "
             "between 412,808 and 545,092, all 35 beating orderings only shuffle those five, "
             "a correct decode fails that bar 82% of the time at this country's own sampling "
             "noise, and the observed correlation of +0.9686 is El Salvador's +0.9685 to "
             "four decimals on a country with twice the units. What is asserted instead is a "
             "joint test that adds urbanisation as a second held-out variable, at 5.95e-4, "
             "and it is POST HOC and labelled as such in sources/cr.py. What the decode "
             "actually rests on is the value labels. "
             "AND THE SECOND PRE-REGISTERED TEST WAS REPORTED NOWHERE UNTIL 2026-09-09, "
             "which is Panama's 6.1 failure one level in: the provenance file is real and the "
             "record did not reconcile with it. It is an unweighted Spearman of the same two "
             "vectors over the same 5,040 orderings at a bar of 1e-2, it returns rho +0.5714 "
             "and an exact p of 9.98e-2, and it fails by a factor of ten. The power argument "
             "above covers it a fortiori, because Spearman throws the magnitudes away and all "
             "five ranks that disagree are the five shelf provinces the levels test could not "
             "order either; sources/cr.md 3 has it in full. Pre-registering two tests and "
             "reporting the one that did better is worse than not pre-registering at all. "
             "AND THE LARGEST CATEGORY DOES NOT CARRY ITS OWN GEOGRAPHY. Catolico returns "
             "+0.79 on a split-half bar of +0.80, the only country in this module where that "
             "happens, and it is applied as written rather than overridden. Nine of the "
             "twelve rank-difference points are Guanacaste alone, falling from 4th to 7th "
             "between the wave halves. Evangelica +1.00, Protestante +0.86, Ninguna +0.86 "
             "and Testigos de Jehova +0.81 do carry theirs; the last of those is the "
             "smallest cell any country in this module places, and the late half of its "
             "split-half is carried by 2014 alone because the box was off the 2023 card. "
             "LEAVE-ONE-OUT SURVIVES ONLY PARTLY, which is the honest cost of seven "
             "units. Dropping each province in turn and re-running over six: Evangelica "
             "holds at +1.00 throughout; Protestante and Ninguna fall to +0.77 at worst, "
             "exactly the exact-null six-unit bar; Testigos de Jehova falls to +0.71 "
             "against +0.77 and does NOT survive. It stays drawn because the stated test "
             "is the split-half on all seven and it passed, and withdrawing it after the "
             "fact would be moving a bar to fit an answer. sources/cr.md 5 has the table. "
             "AND THE 3.5 LEAN IS NOT ESTABLISHED. Costa Rica excludes nothing, so the "
             "hole is the survey's own 2.62% non-response, which the build ABSORBS at the "
             "answerers' rate. Correlated against every drawn share over seven provinces "
             "the largest value is -0.63 against a 5% critical value of about 0.75, and "
             "leave-one-out shows Cartago driving all of them, so no direction is claimed.",
    ),
}
