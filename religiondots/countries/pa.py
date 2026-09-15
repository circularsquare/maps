# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _pa_place_weight(place):
    """countries.py hook. `place` is the 400m hex layer scatter.py has read.

    Panama needs this for §8.2's EMPTINESS reason more than any LAPOP country so far.
    Darien is 12,030 km2 of forest holding 54,235 people, and the unit that carries half the
    map -- Panama with Panama Oeste, 2.09 million people over 11,690 km2 -- is a strip along
    the canal and the Pacific coast with the Chagres watershed and the Darien approaches
    behind it. An equal share per polygon paints a religion across roadless country and takes
    it off the city (sources/pa_grid.py).
    """
    return _kontur_place_weight(place, "pa_hexes.gpkg", "sources/pa_grid.py")


def _pa_counts():
    """LAPOP AmericasBarometer, waves 2010-2023 pooled, at provincia: 11 categories, 10
    units, and EVERY ROW IS `modelled` IN §7.

    The fourth country drawn from the AmericasBarometer, after Guatemala, El Salvador and
    Ecuador. sources.md §11x closed Panama's censuses the hardest way anything here has been
    closed -- the person-variable dictionaries of all five censuses INEC serves on its own
    REDATAM, 1980 through 2023, 252 variables, not one of them religion -- and this build
    re-confirmed it and then went further: INEC has asked the question TWICE on household
    surveys and published neither answer at the geography it holds. The MICS 2013 household
    listing asks it of every member of 11,100 households; the Encuesta de Propositos
    Multiples of April 2022 asked it of 11,776 household informants and its own slide deck
    says results can be had by province and comarca. What is published is one national bar
    chart. sources/pa.md has the search.

    TEN UNITS OF THIRTEEN. LAPOP's `prov` value labels name ten Panamanian units and there is
    no code for Comarca Guna Yala or Comarca Embera-Wounaan in any wave, so neither is drawn:
    44,374 people, 1.09% of the country, in `gap=`. That is Ecuador's Galapagos rule (§9bn),
    and it costs more here, because those are two of the three indigenous comarcas. They keep
    their polygons and their hexes and draw no religion.

    PANAMA OESTE IS NOT A GAP, IT IS A MERGE. It became a province in 2014 out of the western
    districts of Panama and LAPOP never adopted it; the 2023 round still codes every one of
    its respondents to `708`. sources/pa_geo.py dissolves the two polygons, so one unit of
    2,092,955 people is 51.5% of the country.

    THE CODE JOIN IS A TRAP AND IT IS EL SALVADOR'S AGAIN. LAPOP's `prov` is 700 plus the
    official province number; COD's `PA` pcodes are alphabetical. Two of ten coincide, and
    the worst of the other eight pairs `708` Panama, 2.09 million people, onto `PA08` Kuna
    Yala, which has 32,016. sources/pa_geo.py joins on the NAME, taken from LAPOP's own value
    labels rather than inferred, and asserts that the code join still mispairs.

    AND THE HELD-OUT PERMUTATION TEST IS RUN EXHAUSTIVELY RATHER THAN SAMPLED. lapop.py's
    version fails if any of 20,000 sampled orderings reaches the observed r, and here 3 did.
    Enumerating all 3,628,800 says why: 468 orderings reach r=+0.996 and ALL 468 keep the
    51%-of-the-country unit in place, so it is leverage and not unit count, and "zero of
    20,000" is a bar no correct decode could clear. The exact p is 1.29e-4, and 1.29e-3 with
    the dominant unit dropped. sources/pa.py::held_out_exact has the argument.
    """
    from pa2023 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "pa.csv",
                     dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])

    lut = pd.read_csv(HERE / "data" / "geo" / "pa" / "pa_lookup.csv", dtype=str)
    df["unit"] = df["geo_id"].map(dict(zip(lut["geo_id"], lut["unit"])))
    missing = sorted(df.loc[df["unit"].isna(), "geo_id"].unique())
    if missing:
        raise SystemExit(f"pa.csv provinces with no polygon: {missing} -- re-run "
                         "sources/pa_geo.py, the lookup is stale")
    # 10 of the 12 polygons. Guna Yala and Embera-Wounaan are deliberately absent -- see the
    # docstring and `gap=`; they keep their polygons and their hexes and draw no religion.
    if df["unit"].nunique() != 10:
        raise SystemExit(f"{df['unit'].nunique()} provinces, expected 10")
    drawn_comarcas = {"PA06", "PA08"} & set(df["unit"])
    if drawn_comarcas:
        raise SystemExit(f"{sorted(drawn_comarcas)} are in pa.csv and must not be -- LAPOP "
                         "has no code for either; re-run sources/pa.py")

    df["node"] = df["source_category"].map(resolve)
    unmapped = sorted(df.loc[df["node"].isna(), "source_category"].unique())
    if unmapped:
        raise SystemExit(f"pa.csv categories with no node: {unmapped}")
    df = df[df["count"] > 0]
    # EVERY row, without exception -- there is no measured tier in this country (§7).
    df["tier"] = "modelled"
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations", "tier"]]


ENTRY = {
    "pa": dict(
        name="Panama",
        source="AmericasBarometer, four rounds 2010 to 2023 (LAPOP Lab, Vanderbilt "
               "University), against OCHA COD-PS 2023 province and comarca populations",
        basis="self-identification, adults 18 and over",
        view=[-83.2, 6.9, -76.9, 9.9],
        note_public=(
            "**Panama's censuses do not ask about religion, and its statistics office has "
            "asked the question twice on household surveys without publishing either "
            "answer.** INEC runs a public microdata server carrying all five of its "
            "censuses, 1980 to 2023, and the 252 variables it records per person across them "
            "include "
            "indigenous group and Afrodescendant group and nothing about belief. The 2013 "
            "MICS asked every member of 11,100 households what religion they profess and its "
            "230-page report tabulates none of it. The April 2022 Encuesta de Propósitos "
            "Múltiples asked 11,776 household informants, and INEC's own slide deck says the "
            "survey can be read by province and comarca; what it published was one national "
            "bar chart. So the map is drawn from the LAPOP AmericasBarometer instead, "
            "**6,105 people** across four rounds between 2010 and 2023, pooled and applied "
            "to the 2023 census population. The dots are drawn desaturated to say that. "
            "**That one bar chart is the only independent check on the level, and this "
            "survey passes it.** INEC read 65% Catholic and 8% with no religion in April "
            "2022, against **63.1%** and 7.4% here. Its evangelical figure needs one "
            "adjustment before it can be compared, because INEC's card has no Protestant box: "
            "its 22% evangelical and 2% Adventist together, **24%**, are the comparator for "
            "this map's evangelical and Protestant cells together, which come to **25.5%**. "
            "All three land within two points. No other country drawn from this survey has a "
            "state reading to be checked against. "
            "Those national shares are an average of provinces that are a long way apart. "
            "Veraguas is **93.6%** Catholic and 4.7% "
            "Evangelical. The Comarca Ngäbe-Buglé is 29.5% Catholic and **58.8%** "
            "Evangelical, the only unit on the map where Evangelicals outnumber Catholics, "
            "and one in seven of Panama's Evangelicals lives there among one in twenty of "
            "its people. "
            "**Bocas del Toro is where people leave a church without leaving belief.** "
            "**21.7%** of it says it believes in a higher being and belongs to no religion, "
            "against 1.4% in Veraguas and in Herrera. This survey never offers a plain "
            "no-religion box: it offers the believer without an affiliation and the "
            "non-believer as separate answers, and the second is 0.5% of Panama. "
            "**Half the map is a single unit.** Panamá Oeste became a province in 2014 and "
            "the survey never adopted it, so its respondents are still coded to Panamá; the "
            "two are drawn together as one unit of **2,092,955** people, 51.5% of the "
            "country, and nothing here distinguishes the capital from the province west of "
            "the canal. "
            "**Traditional indigenous religion is a floor and the map cannot show where it "
            "would be.** It comes back at 0.4% of Panama, against the 9.4% of INEC's own "
            "2022 informants who identify as indigenous, and the two comarcas that would "
            "carry it, Guna Yala and Emberá-Wounaan, are the two the survey never sampled. "
            "**The level is a thirteen-year average of rounds that disagree with each "
            "other.** Catholic identification reads 68.6% in 2010, 61.1% in 2012, 73.6% in "
            "2014 and **54.3%** in 2023. Pooling is what buys the province detail, since a "
            "single round is about 1,500 people over ten provinces, and it is also why this "
            "map sits closer to INEC's 2022 reading than the last round alone would. Two "
            "answers exist in only part of the pool: `Otro` was not on the card before 2023, "
            "and Jehovah's Witnesses and Latter-day Saints were taken off it for 2023, so "
            "both of those are floors."),
        how="survey, four rounds 2010 to 2023 pooled",
        grain="provinces and one comarca, 402,000 people on average",
        gap=("Comarca Guna Yala and Comarca Emberá-Wounaan, 44,374 people, 1.09% of Panama, "
             "which the survey has no code for in any round"),
        gap_share=0.0109,
        counts=_pa_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "pa" / "pa_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_pa_place_weight,
        note="THE COUNTRY IS A SURVEY ON A CENSUS COUNT AND EVERY ROW IS `modelled` (§7b), "
             "the same construction as Guatemala, El Salvador and Ecuador: LAPOP's `q3c` "
             "gives a province share, the population it applies to comes from OCHA's COD-PS "
             "2023 table, and no magnitude is invented. COD-PS is used here rather than "
             "INEC's own census, unlike Ecuador, because for Panama they are the same count: "
             "the 2023 table sums to 4,064,445 against INEC's published 4,064,780, a "
             "difference of 335 people. "
             "TEN UNITS OF THIRTEEN. LAPOP's `prov` value labels name ten Panamanian units "
             "and there is no code for Guna Yala or Embera-Wounaan in any wave, so 44,374 "
             "people are in `gap=` rather than estimated -- Ecuador's Galapagos rule (§9bn), "
             "and it costs more here because those are two of the three indigenous comarcas. "
             "They keep their polygons and their hexes and draw no religion. Panama Oeste is "
             "a different case and not a gap: LAPOP never adopted the 2014 split, so "
             "sources/pa_geo.py dissolves it into Panama. "
             "THE CODE JOIN IS EL SALVADOR'S TRAP AGAIN. `prov` is 700 plus the official "
             "province number and COD's `PA` pcodes are alphabetical; two of ten coincide, "
             "and the worst of the other eight would draw Panama's 2.09 million people in "
             "Kuna Yala, which has 32,016. The join is on the NAME, read out of LAPOP's own "
             "value-label set rather than inferred from a numbering, and sources/pa_geo.py "
             "asserts that the code join still mispairs so nobody restores it. "
             "AND THE HELD-OUT PERMUTATION TEST IS EXHAUSTIVE HERE RATHER THAN SAMPLED. "
             "sources/lapop.py fails if any of 20,000 sampled orderings reaches the observed "
             "r, and 3 did. All 3,628,800 orderings say why: 468 reach r=+0.996 and every "
             "one of them keeps the unit that is half the country in place, so it is leverage "
             "rather than unit count, and a correct decode clears that bar only on the 7.6% "
             "of seeds where the sampler draws none of the 468. The exact p is 1.29e-4, on a "
             "bar of 1e-3 that is deliberately looser than the sampled rule it replaces. The "
             "same test with the dominant unit dropped returns 1.29e-3, which is a consistency "
             "check rather than a second witness: it is the same 468 orderings restricted to "
             "nine units, so it cannot fail while the first passes. "
             "THE THREE CATEGORIES DRAWN ON THEIR OWN PROVINCE SHARES SURVIVE LEAVE-ONE-OUT. "
             "Catholic +0.85, Evangelical +0.87 and the believers-without-a-religion +0.82 "
             "against a bar of +0.65 on ten units; dropping any single unit leaves all three "
             "between +0.75 and +0.93, above the nine-unit bar of +0.69. No one unit, "
             "Ngabe-Bugle included, manufactures any of them.",
    ),
}
