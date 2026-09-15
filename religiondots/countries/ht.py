# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _ht_place_weight(place):
    """countries.py hook. `place` is the 400m hex layer scatter.py has read.

    Ten departments over 27,750 km2 is 2,775 km2 a unit, nearly twice the Dominican
    Republic's ratio on the same island. **Ouest is the reason**: 3.98 million people, a
    third of Haiti, in a polygon running from Port-au-Prince across the Chaine des Matheux
    and up onto the Plateau, with the agglomeration itself a strip along the bay. Grand'Anse
    and Nippes have the reverse problem, a mountain spine with the people on the coastal
    shelf either side. And there is a third reason specific to Haiti: the population base is
    a 2003 census carried forward by projection, whose commune figures cannot see anything
    settled since, so Kontur's buildings are the only thing in this build that has seen the
    country since 2010 (sources/ht_grid.py).

    TWO COMMUNES ARE SCALED TO THEIR COD-PS SHARE OF THEIR DEPARTMENT: Anse-a-Galets, added
    2026-09-14 (session `f95259a4-clht`, sources/ht.md §12), and Petit-Goave the same evening
    (session `f95259a4-house`, §12.6). Placement only: it moves weight between communes of one
    department and no count changes. See _HT_COMMUNE_LEVEL for which and why.
    """
    if not {"pop", "commune", "commune_pop"} <= set(place.columns):
        print("  !! ht_hexes.gpkg has no commune columns; run sources/ht_grid.py. La Gonave "
              "and Petit-Goave are drawn at Kontur's level")
        return _kontur_place_weight(place, "ht_hexes.gpkg", "sources/ht_grid.py")
    place = place.copy()
    pop = place["pop"].to_numpy(dtype=float).copy()
    unit = place["unit"].astype(str).to_numpy()
    com = place["commune"].astype(str).to_numpy()
    cod = place[["commune", "unit", "commune_pop"]].drop_duplicates("commune")
    for code in _HT_COMMUNE_LEVEL:
        inc = com == code
        if not inc.any():
            raise SystemExit(f"ht: commune {code} has no hex; re-run sources/ht_grid.py")
        dept = unit[inc][0]
        ind = unit == dept
        cod_c = float(cod.loc[cod["commune"] == code, "commune_pop"].iloc[0])
        cod_d = float(cod.loc[cod["unit"] == dept, "commune_pop"].sum())
        before = pop[inc].sum() / pop[ind].sum()
        pop[inc] *= (pop[ind & ~inc].sum() * cod_c / (cod_d - cod_c)) / pop[inc].sum()
        print(f"  ht: commune {code} scaled to its COD-PS share, {100 * before:.2f}% -> "
              f"{100 * pop[inc].sum() / pop[ind].sum():.2f}% of {dept}'s placement weight")
    place["pop"] = pop
    return _kontur_place_weight(place, "ht_hexes.gpkg", "sources/ht_grid.py")


# Communes whose Kontur weight is replaced by their COD-PS share of the department, keeping
# Kontur's shape inside the commune. The bar: every grid tested is over by more than 2x AND
# nothing since the 2003 census explains it. Croix-des-Bouquets (Kontur 2.2x) and Cabaret
# (1.8x) fail the second half, because that is Canaan, settled after 2010 and invisible to a
# projection; Gressier fails it the other way (COD-PS prints 3,987, all three grids 42,000 to
# 46,000). sources/ht.md §12 has the table.
# A SECOND ROUTE IN, added 2026-09-14 (session `f95259a4-house`, sources/ht.md §12.6): a commune
# under 2x whose town block kontur_cap.csv already calls false, where every grid has the same
# excess and the cap draws the town at farmland density. Petit-Goave is the only one.
_HT_COMMUNE_LEVEL = {
    "HT0151": "Anse-a-Galets, La Gonave. Kontur 273,795 against COD-PS 70,156; WorldPop 2020 "
              "constrained 244,402 and unconstrained 254,858, so a different grid carries the "
              "same error. Pointe-a-Raquette, the island's other commune, agrees across all "
              "four (22,474 to 26,818). As drawn it put 6.7% of Ouest's dots on an island COD-PS "
              "gives 2.4%, piled into a few hexes at up to 46,200/km2 where the island's median "
              "populated hex is 27/km2. Capping the town block instead would leave the island "
              "at 4.6% and draw its main town at farmland density.",
    "HT0122": "Petit-Goave, Ouest. Kontur 304,821 against COD-PS 194,867, which is 6.62% of "
              "Ouest's weight against 4.90%, 1.35x and under the 2x bar; it is here on the second "
              "route. WorldPop 2020 has the same excess (constrained 292,091, unconstrained "
              "292,951). Its town block was registered capped the same morning, which left the "
              "commune at 0.68x its share and 8 dots within 1 km of the town centre. Scaled "
              "instead, the block keeps Kontur's shape and its kontur_cap.csv row is `real`.",
}


def _ht_counts():
    """IHSI ECVMAS 2012, at departement: 12 categories, 10 units, and EVERY ROW IS
    `modelled` IN §7.

    THE QUEUE REACHED HAITI THROUGH LAPOP AND THE OFFICE HAS BETTER. §11ad offered 7,252
    AmericasBarometer respondents; IHSI's own ECVMAS 2012 has 17,977 on the same ten
    departments with a twelve-answer Haitian card, and IHSI published the microdata openly.
    sources/ht.py has the record, including that `www.ihsi.ht` is now a squatted domain and
    the file comes from the Wayback Machine.

    THE UNIVERSE IS PEOPLE AGED TEN AND OVER, applied to the whole department population.
    The 2003 census tabulated religion by age group, so the cost is measured rather than
    assumed: no category moves a full point and the largest movement is `Aucune religion`,
    10.22% of everyone against 9.40% of the 10-and-overs.

    THE POPULATION IS A PROJECTION AND THERE IS NO ALTERNATIVE. Haiti last counted in
    January 2003 and the fifth census has not been held, so COD-PS 2024 is the base and
    sources/ht_geo.py prints how far it has moved the departmental shares since the census.

    SEVEN OF THE TWELVE CATEGORIES CARRY THEIR OWN GEOGRAPHY. On ten units the split-half
    bar is +0.65 and the test has almost no power, so four of the seven are OVERRIDE entries
    resting on ECVH 2001, a different IHSI survey eleven years earlier, ranking the same
    departments the same way. `Aucune`, `Autre`, `Episcopale`, `Temoin de Jehovah` and
    `Musulman` are drawn at the national rate inside each department's residual.
    """
    from ht2012 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "ht.csv",
                     dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])

    lut = pd.read_csv(HERE / "data" / "geo" / "ht" / "ht_lookup.csv", dtype=str)
    df["unit"] = df["geo_id"].map(dict(zip(lut["geo_id"], lut["unit"])))
    missing = sorted(df.loc[df["unit"].isna(), "geo_id"].unique())
    if missing:
        raise SystemExit(f"ht.csv departments with no polygon: {missing} -- re-run "
                         "sources/ht_geo.py, the lookup is stale")
    if df["unit"].nunique() != 10:
        raise SystemExit(f"{df['unit'].nunique()} departments, expected 10")

    df["node"] = df["source_category"].map(resolve)
    unmapped = sorted(df.loc[df["node"].isna(), "source_category"].unique())
    if unmapped:
        raise SystemExit(f"ht.csv categories with no node: {unmapped}")
    df = df[df["count"] > 0]
    # EVERY row, without exception -- there is no measured tier in this country (§7).
    df["tier"] = "modelled"
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations", "tier"]]


ENTRY = {
    "ht": dict(
        name="Haiti",
        source="Enquête sur les Conditions de Vie des Ménages Après le Séisme ECVMAS 2012 "
               "(Institut Haïtien de Statistique et d'Informatique), against COD-PS 2024 "
               "department populations",
        basis="self-identification, own religion, people aged ten and over",
        note_public=(
            "**Haiti's census asks about religion and has never published the answer below "
            "the national line.** IHSI's 2003 tables give twelve religions for the whole "
            "country and stop, so this map comes from the office's other instrument: "
            "ECVMAS 2012 asked **17,977** Haitians aged ten and over what their religion "
            "was, across all ten departments, and IHSI published the microdata openly. The "
            "dots are drawn desaturated because these are survey respondents rather than a "
            "count, and the population underneath them is a projection, since Haiti has not "
            "counted anybody since January 2003. "
            "**Haiti is not Catholic the way the other half of the island is.** Catholicism "
            "is **47.6%** here against 52.7% in the Dominican Republic, and the churches "
            "that are not Catholic come to **43.0%** against 25.4% there. Catholics are "
            "under half in four of the ten departments, and those four hold **60.1%** of "
            "the population, so most Haitians live somewhere Catholicism is the largest "
            "answer rather than the majority one. In Artibonite, Centre and Ouest the "
            "Protestant churches together outnumber it outright. "
            "**That Protestant half is not one thing, and its pieces have separate maps.** "
            "The Baptists are the largest of them at **17.5%** and they are a northern "
            "church: **25.0% of Nord** and 24.3% of Nord-Ouest against **7.8% of "
            "Grand'Anse**. The Adventists are strongest in the same corner, **7.2% of "
            "Nord-Ouest** against 0.7% of Grand'Anse. Pentecostals run the other way and "
            "are a capital answer, **16.2% of Ouest** against 3.7% of Nord-Ouest. The "
            "Methodists are 1.3% of the country and 4.5% of Artibonite. A further **9.2%** "
            "answered `other Protestant`, which the survey does not break down; IHSI's 2001 "
            "round, which did offer a Church of God box, found 9.4% of Haitians in it, so "
            "most of this is likely to be that church. "
            "**The Vodou figure is a floor, and it is the number on this map to be most "
            "careful with.** ECVMAS finds **1.5%**, the 2003 census found 2.1% and IHSI's "
            "2001 survey found 2.1%. All three ask for one religion, and Vodou in Haiti is "
            "overwhelmingly served alongside Catholicism rather than instead of it, so what "
            "these numbers count is the people who named it first. That group is "
            "concentrated in one department: Artibonite is **5.9%**, holding **105,171** of "
            "the country's **178,476**, about five times the share of anywhere else. Both "
            "IHSI surveys agree on Artibonite and on Nord-Est at the bottom and on nothing "
            "in between, so read the two ends of the Vodou colour and not the ordering. "
            "**One large answer is drawn flat.** **6.8%** of Haitians report no religion, "
            "and this map puts that share in every department rather than where the survey "
            "found it. Splitting ECVMAS's 500 neighbourhoods into two half-samples and "
            "re-ranking the departments returns +0.56 against a bar of +0.65, and the 2001 "
            "survey offered no no-religion box at all, so there is no second instrument to "
            "check it against. Nobody is deleted; only the claim to know where they are is "
            "withdrawn. The same is done with the Episcopalians, the Jehovah's Witnesses "
            "and the small `other` cell. "
            "**The level is 2012 and the population is 2024.** Both the 2010 earthquake and "
            "the displacement out of Port-au-Prince since 2021 fall in that gap, and the "
            "projection has moved a fair amount of Haiti with them: Ouest was 37.0% of the "
            "country at the 2003 census and is 33.4% of the 2024 estimate. What this map "
            "says is where each department stood in 2012, drawn on the number of people the "
            "UN and IHSI think live there now."),
        how="household survey, one round in 2012, people aged ten and over",
        grain="departments, 1.19 million people on average",
        gap="the six people who did not know or would not say, 0.03% of those asked",
        gap_share=0.000334,
        counts=_ht_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "ht" / "ht_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_ht_place_weight,
        note="THE QUEUE REACHED HAITI THROUGH LAPOP AND THE OFFICE HAS BETTER. §11ad "
             "offered 7,252 AmericasBarometer respondents over ten departments, with "
             "Anita's note that the 4.4% Vodou figure is a self-identification floor "
             "wanting a second source. Asking what IHSI publishes instead turns up three "
             "of its own instruments nobody here had opened: **ECVMAS 2012** (17,977 "
             "respondents aged 10+, ten departments, twelve answers), **ECVH 2001** "
             "(32,840 people, nine departments, published as a table in IHSI's own volume) "
             "and the **4eme RGPH 2003** census (8,373,750 people, twelve categories, "
             "national only). The build is ECVMAS and the other two are cross-checks. "
             "THE HOST IS GONE AND THE FILE IS NOT. IHSI published the whole ECVMAS "
             "database openly with no account at `www.ihsi.ht`; that domain now answers "
             "301 and lands on an unrelated commercial page, and `rgph-haiti.ht`, the fifth "
             "census's own site, answers 200 with a betting site. The office moved to "
             "`ihsi.gouv.ht` and did not bring the microdata, so the zip comes from the "
             "Wayback Machine and the 2003 census tables from the live host. "
             "THE DEPARTMENT JOIN IS ZERO FOR TEN IF DONE ON THE CODE. ECVMAS numbers the "
             "departments 1-10 in French alphabetical order and COD's pcodes run HT01-HT10 "
             "in Haiti's traditional order; both are dense 1..10 sequences over the same "
             "ten units, so the obvious pairing runs clean and mispairs every one of them "
             "([[reference_name_join_wrong_neighbour]]). sources/ht_geo.py asserts the "
             "count is still zero and joins on the French name, with one alias, Grand'Anse "
             "for COD's Grande'Anse. The decode is pinned by the held-out population check "
             "at r = +0.996 over ten, which no random pairing of 20,000 reaches. "
             "THE SPLIT-HALF HAS ALMOST NO POWER ON TEN UNITS, so four of the seven placed "
             "categories are OVERRIDE entries. The bar is +0.65 and `Catholique`, with a "
             "24-point spread and a chi-square of 1e-117, lands within 0.003 of it. What "
             "the overrides rest on is ECVH 2001 ranking the same departments with a "
             "different questionnaire eleven years earlier: Catholic +0.85, Baptist +0.78, "
             "Adventist +0.95, and the Pentecostal bloc +0.90 once the two cards' different "
             "cuts are pooled. sources/ht.py prints all of it on every run. "
             "THE UNIVERSE IS 10 AND OVER and the cost is measured rather than assumed, "
             "because the 2003 census tabulated religion by age group: recomputed over the "
             "10-and-overs no category moves a full point, the largest being `Aucune "
             "religion` from 10.22% to 9.40%. "
             "LAPOP IS NOT WIRED AS A CROSS-CHECK AND THE REASON IS WORTH KEEPING. §11ad's "
             "Haiti row is `pais=22`, and its `prov` cannot be decoded from the merged "
             "file: the official department numbering gives r = +0.23 against COD-PS and "
             "puts 40.7% of Haiti in Centre. It is Honduras's problem in §11ad's own table. "
             "Separately, `pais=41` is CANADA and has ten first-order units with prov codes "
             "4101-4110, so an agent guessing that 41 is Haiti gets a clean-looking ten-unit "
             "join to the wrong country; it only fails visibly because Canada's rows carry "
             "no religion answer.",
    ),
}
