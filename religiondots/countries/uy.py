# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _uy_place_weight(place):
    """countries.py hook. `place` is the 400m hex layer scatter.py has read.

    Uruguay is the most lopsided ADM1 country in the Americas here. Montevideo is 0.3% of the
    land and 37% of the people; the seventeen interior departments run 20 to 40 people per km2
    and their population sits almost entirely in one departmental capital each, with grazing
    country between. The second reason is the coast: Maldonado and Rocha are long thin
    departments whose people are on a 200 km Atlantic strip and whose interiors are nearly
    empty, and those two also carry the highest unaffiliated shares in the country, so getting
    their dots onto the shoreline is getting the map's most striking claim into the right
    place (sources/uy_grid.py).
    """
    return _kontur_place_weight(place, "uy_hexes.gpkg", "sources/uy_grid.py")


def _uy_counts():
    """INE's Encuesta Nacional de Hogares Ampliada 2006 at departamento: 7 categories, 19
    units, and EVERY ROW IS `modelled` IN §7.

    SINCE 2026-09-15 MONTEVIDEO IS ITS 62 BARRIOS (sources/uy.md §12), so the units are 18
    departments plus `UY10-B01`..`UY10-B62`, and there is no `UY10` row. Six of the seven
    answers carry their own barrio shares on a census-segment split-half; `Otra` takes the
    city's. Barrio populations come from the 2023 census persons file, weighted, under INE's
    own Montevideo total, and the place layer follows INE's Montevideo line, not COD's
    (sources/uy_grid.py). What follows describes the department build and is still true of it,
    except the 7+ figures, corrected on 2026-09-15 (sources/uy.md §12.7).

    THE QUEUE PRICED THIS COUNTRY FROM LAPOP AND THE STATISTICS OFFICE HAS SOMETHING FIFTY
    TIMES BIGGER. INE put a religion question on the 2006 ENHA -- the year the continuous
    household survey was widened to cover small localities and rural areas -- and asked it of
    every household member over six. 230,898 people answered, against the AmericasBarometer's
    4,318 pooled over three waves, on the same nineteen departments. Every department has
    between 3,252 and 80,196 respondents, so the thinnest is +/-1.7 points on a share near
    46% and nothing anywhere is drawn at a national rate.

    THE CENSUS NEGATIVE IS REAL AND OLD. Uruguay is absent from the UNSD oracle and the last
    census to ask about religion was 1908, nine years before the 1917 constitution
    separated church and state; 2011 and 2023 do not carry the question. So this is a survey
    standing where no census has stood for over a century, and every row is `modelled`.

    THE UNIVERSE IS AGES 7 AND OVER, AND THE METADATA GETS IT WRONG. ANDA's data dictionary
    says "personas mayores de 6 anos" and then glosses it as "6 anos y mas", which are two
    different universes. The microdata settles it: the not-applicable code covers ages 0 to 6
    exactly and completely, all 4,225 six-year-olds included, and nobody 7 or over carries it.
    7.67% of Uruguay is therefore children and is in `gap=`. There is no non-response cell at
    all -- every person in the universe has one of the seven answers, which is rare here.

    THE CODE JOIN IS EL SALVADOR'S TRAP, HARDER TO SEE. INE numbers departments with
    Montevideo first and the other eighteen alphabetically; COD-AB's UY01..UY19 are
    alphabetical with no exception, so `UY{dpto:02d}` shifts the first ten by one place and
    leaves the last nine alone. NINE OF NINETEEN COINCIDE, and they are the last nine, so a
    spot check on Salto or Treinta y Tres passes while Montevideo's 1.3 million people are
    drawn in Artigas. sources/uy_geo.py joins on the NAME -- all 19 matching with no aliases
    -- and asserts that the code join still mispairs exactly ten.

    THE POPULATION IS INE'S OWN POST-CENSUS ESTIMATE, NOT COD-PS. Uruguay counted in 2023 and
    COD-PS's Uruguay file is a projection off the 2011 census, so §9bn's rule applies. INE's
    Estimaciones y proyecciones revision 2025 gives 3,496,400 at 30 June 2023 by department
    and five-year age band, of whom 3,228,150 are 7 or over. The 7+ cut subtracts two fifths
    of the 5-9 band (ages 5 and 6) because nothing INE publishes for Uruguay is by single year
    of age; sources/uy_geo.py says so.

    TWO CHECKS, AND ONE IS A SECOND SURVEY. The ENHA's own weighted department distribution
    tracks INE's 2006 population distribution at r=+1.0000 over 19, which none of 20,000
    random pairings comes near (best random +0.996) -- compared against 2006 and not 2023,
    because seventeen years of internal migration is not a test of a join. AND THE
    AMERICASBAROMETER MEASURED THE SAME COUNTRY FOUR TO EIGHT YEARS LATER WITH A DIFFERENT
    INSTRUMENT, which is a witness no LAPOP-only country can have: non-Catholic Christian
    r=+0.86 with 0 of 20,000 random pairings reaching it, atheist/agnostic +0.73 with 3, both
    no-religion cells together +0.56 with 118. Catholic is +0.34 and NOT significant, which is
    a fact about the pair of cards rather than about either survey -- LAPOP finds 37.0%
    Catholic and 32.4% believing-without-a-religion where INE finds 46.0% and 26.9%, and that
    boundary is exactly where wording moves people. LAPOP also has 45 respondents in Rivera
    against INE's 9,322.

    FIVE OF THE SEVEN CATEGORIES CLEAR THE SPLIT-HALF, run across the two halves of 2006
    rather than across waves: Catholic +0.92, believer-without-religion +0.88,
    atheist/agnostic +0.88, non-Catholic Christian +0.85, Umbanda +0.78, on a bar of +0.46.
    `Judio` (+0.26) and `Otra` (+0.26) are drawn anyway, named in sources/uy.py's `UNDER_BAR`
    with the reasons and their chi-squares. The Jewish one matters: the rank test is ranking
    eighteen departments whose true share is near zero, while the fact the map draws is that
    Montevideo is 0.92% Jewish against 0.06% elsewhere and holds nine tenths of the cell, on
    80,196 Montevideo respondents. Spreading that at the national rate would be a stronger
    claim than the test declines to license, and a false one.
    """
    from uy2006 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "uy.csv",
                     dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])

    lut = pd.read_csv(HERE / "data" / "geo" / "uy" / "uy_lookup.csv", dtype=str)
    blut = pd.read_csv(HERE / "data" / "geo" / "uy" / "uy_barrios_lookup.csv", dtype=str)
    unit_of = dict(zip(lut["geo_id"], lut["unit"]))
    unit_of.update(zip(blut["geo_id"], blut["unit"]))
    df["unit"] = df["geo_id"].map(unit_of)
    missing = sorted(df.loc[df["unit"].isna(), "geo_id"].unique())
    if missing:
        raise SystemExit(f"uy.csv units with no polygon: {missing} -- re-run "
                         "sources/uy_geo.py, the lookup is stale")
    # Since 2026-09-15: 18 departments, and Montevideo's 62 barrios in place of Montevideo.
    if df["unit"].nunique() != 80 or "UY10" in set(df["unit"]):
        raise SystemExit(f"{df['unit'].nunique()} units, expected 18 departments and 62 "
                         "Montevideo barrios, with no Montevideo department row")

    df["node"] = df["source_category"].map(resolve)
    unmapped = sorted(df.loc[df["node"].isna(), "source_category"].unique())
    if unmapped:
        raise SystemExit(f"uy.csv categories with no node: {unmapped}")
    df = df[df["count"] > 0]
    # EVERY row, without exception -- there is no measured tier in this country (§7).
    df["tier"] = "modelled"
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations", "tier"]]


ENTRY = {
    "uy": dict(
        name="Uruguay",
        source="Encuesta Nacional de Hogares Ampliada 2006 (INE), against INE's own "
               "departmental population estimates for 2023 and, for Montevideo's barrios, "
               "the 2023 census",
        basis="self-identification, people aged 7 or over",
        note_public=(
            "**Uruguay's census has not asked about religion since 1908, so this map is a "
            "household survey standing where a census would be.** The question is the "
            "statistics office's own, put to every household member over six in the Encuesta "
            "Nacional de Hogares Ampliada of 2006, and **230,898 people** answered it. These "
            "are survey respondents rather than a count of the country, and the dots are "
            "drawn desaturated to say so; but it is not a thin survey. The smallest "
            "department, Flores, rests on 3,252 answers and Montevideo on 80,196, so every "
            "department here carries its own measured composition rather than a national "
            "average. Children under seven were not asked and are not drawn. "
            "**42.5% of Uruguayans over six claim no religious affiliation, against 46.1% "
            "Catholic.** That is far more than any other Latin American country on this map, "
            "and the survey splits it where it matters: **27.0%** say they believe in God but "
            "belong to no religion, and **15.5%** call themselves atheist or agnostic. Most "
            "Uruguayans who have left the church have not left belief. The separation goes "
            "back a long way, to an 1861 law taking the cemeteries out of church hands and "
            "the 1917 constitution that finished the job. "
            "**The two halves of the no-religion answer have opposite maps.** Atheists and "
            "agnostics are a Montevideo and Atlantic-coast phenomenon, **21.6%** of the "
            "capital and 16.6% of Maldonado against 3.9% of Artigas on the Brazilian border. "
            "Believers without a religion are the interior and the north, 41.9% of Tacuarembó "
            "and 41.6% of Rocha against 13.6% of Colonia. Put together they run from 20.5% of "
            "Paysandú to **58.1% of Rocha**. "
            "**Catholicism is the other side of that and its range is nearly as wide**, 67.7% "
            "of Paysandú against 30.3% of Rivera. Non-Catholic Christians are 10.1% of the "
            "country and are one box: the question folds evangelicals, Pentecostals, "
            "Baptists, Protestants, Adventists and the Armenian Apostolic Church into a "
            "single answer, so nothing separates them here. They are 27.9% of Rivera and 5.6% "
            "of Maldonado. The Waldensians who founded Colonia Valdense in 1858, the oldest "
            "Protestant body in the country, are inside that box and cannot be taken out of "
            "it. "
            "**Two small answers get their own box here that most surveys in the region do "
            "not offer.** Judaism is **0.36%** of the country and Montevideo holds nine "
            "tenths of it, 0.87% of the city against 0.06% of everywhere else. Umbanda and "
            "Afro-American religion together are 0.63%, highest in Montevideo at 0.95% and in "
            "Rivera at 0.78%, which is the capital and the Brazilian border town. That second "
            "figure is people who define themselves that way first, and Uruguayan practice of "
            "those religions is wider than the number. "
            "**Montevideo is drawn at its 62 barrios rather than as one department.** The "
            "survey reached people in every one of them, from 358 in La Blanqueada to 4,131 "
            "in La Paloma and Tomkinson, and the split was tested before it was drawn: when "
            "each barrio's sampled census segments were divided into two halves, six of the "
            "seven answers ordered the barrios the same way in both. The seventh, another "
            "religion, is drawn at the city-wide share. The barrios' populations are from the "
            "2023 census. The same contrasts appear inside the city. Atheists and agnostics "
            "are **29.8%** of Palermo and 29.7% of Barrio Sur, against 14.7% of Lezica and "
            "Melilla on the rural edge; Catholics are 63.8% of Carrasco Norte and 31.8% of La "
            "Paloma and Tomkinson; non-Catholic Christians are 14.9% of Villa García and Manga "
            "Rural and 3.1% of Barrio Sur. Punta Carretas and Pocitos are each about 8% "
            "Jewish, and with Punta Gorda they hold seven in ten of the city's Jewish "
            "residents. A barrio rests on a few hundred to a few thousand answers, so a gap of "
            "several points between two barrios can be sampling noise. "
            "**The level is 2006 and the country has kept moving.** The AmericasBarometer "
            "asked a similar question between 2010 and 2014 and found 37% Catholic, so what "
            "is drawn here is several points more Catholic than Uruguay is now. What the two "
            "surveys agree on is the shape: their department orderings for non-Catholic "
            "Christians and for atheists match closely, and the ordering is what a map is "
            "for."),
        how="national household survey, 2006, ages 7 and over",
        grain="departamentos, 112,000 people on average; Montevideo's 62 barrios, 19,000",
        gap="children under 7, 7.7% of Uruguay, who were not asked the question",
        gap_share=0.0767,
        counts=_uy_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "uy" / "uy_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_uy_place_weight,
        note="THE QUEUE PRICED THIS COUNTRY FROM LAPOP AND INE HAS SOMETHING FIFTY TIMES "
             "BIGGER. queue.md's §11ad row is 4,318 AmericasBarometer respondents pooled over "
             "three waves; the 2006 ENHA carries a religion question answered by 230,898 "
             "people on the same nineteen departments, 3,252 to 80,196 apiece. So Uruguay is "
             "the first country in this Latin American set drawn from the national statistics "
             "office rather than from the survey, and the only one where nothing is drawn at "
             "a national rate. THE CENSUS NEGATIVE IS REAL: absent from the UNSD oracle, and "
             "the last census to ask was 1908. Every row is still `modelled` (§7b). "
             "THE UNIVERSE IS AGES 7 AND OVER AND ANDA'S METADATA GETS IT WRONG -- the data "
             "dictionary says `personas mayores de 6 anos` and then glosses it `6 anos y "
             "mas`. The microdata settles it: the not-applicable code covers ages 0 to 6 "
             "completely, all 4,225 six-year-olds included. 7.67% of Uruguay is in `gap=` and "
             "there is no non-response cell at all. "
             "THE CODE JOIN IS EL SALVADOR'S TRAP AND HARDER TO SEE. INE numbers departments "
             "Montevideo first then alphabetically; COD-AB's UY01..UY19 are alphabetical with "
             "no exception, so `UY{dpto:02d}` gets the LAST NINE right and shifts the first "
             "ten by one place. A spot check on Salto or Treinta y Tres passes while "
             "Montevideo's 1.3 million are drawn in Artigas. sources/uy_geo.py joins on the "
             "name, all 19 with no aliases, and asserts that the code join still mispairs ten. "
             "THE POPULATION IS INE'S OWN 2023 ESTIMATE, not COD-PS, because Uruguay counted "
             "in 2023 and COD-PS projects off 2011 (§9bn). 3,496,400 people of whom 3,228,150 "
             "are 7 or over; the 7+ cut subtracts two fifths of the 5-9 band (ages 5 and 6) "
             "because nothing INE publishes for Uruguay is by single year of age. "
             "THE SECOND CHECK IS A SECOND SURVEY, which no LAPOP-only country can have. "
             "LAPOP measured the same nineteen departments four to eight years later with a "
             "different instrument: non-Catholic Christian r=+0.86 with 0 of 20,000 random "
             "pairings reaching it, atheist/agnostic +0.73 with 3, both no-religion cells "
             "+0.56 with 118. Catholic is +0.34 and not significant, which is about the pair "
             "of cards -- LAPOP finds 37.0% Catholic and 32.4% believing-without-a-religion "
             "where INE finds 46.0% and 26.9% -- and about LAPOP having 45 respondents in "
             "Rivera against INE's 9,322. The population check is r=+1.0000 over 19 against "
             "INE's 2006 estimate, best random pairing +0.996. "
             "FIVE OF SEVEN CATEGORIES CLEAR THE SPLIT-HALF, run across the two halves of "
             "2006 rather than across waves. `Judio` and `Otra` come in at +0.26 and are drawn "
             "anyway under sources/uy.py's `UNDER_BAR`: for a category near zero in eighteen "
             "of nineteen units the rank test is ranking noise, while the thing the map draws "
             "is Montevideo at 0.92% Jewish against 0.06% elsewhere on 80,196 respondents. "
             "Both exceptions carry a chi-square as their precondition and the bar was not "
             "moved. SINCE 2026-09-15 MONTEVIDEO IS ITS 62 BARRIOS: six answers on their own "
             "barrio shares by a census-segment split-half, `Otra` at the city's, barrio "
             "populations from the 2023 census persons file, placement on INE's Montevideo "
             "line (sources/uy.md §12). Until 2026-09-15 the 7+ cut subtracted three fifths of "
             "the band, ages 7 to 9 instead of 5 and 6, leaving every department short in "
             "proportion (46,154 people in all); fixed and redrawn that day (§12.7). "
             "sources/uy.md is the write-up and taxonomy/uy2006.py the mapping.",
    ),
}
