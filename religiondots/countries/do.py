# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _do_place_weight(place):
    """countries.py hook. `place` is the 400m hex layer scatter.py has read.

    32 provinces over 48,671 km2 is 1,521 km2 a unit, so the Dominican Republic is not drawn
    for §8.2's emptiness reason. It is drawn for the concentration one. **Santo Domingo
    province is 2.77 million people ringing a capital it does not contain**, and its polygon
    runs from that ring north-east into the empty Sierra de Yamasa; the Distrito Nacional is
    another 1.03 million in 91 km2, so a third of the country sits inside a 40 km circle. The
    other end is the Haitian border, where Pedernales, Independencia and Elias Pina string
    their people along a few roads with the Sierra de Bahoruco and Lago Enriquillo in between,
    and those three are also where the map's no-religion share peaks (sources/do_grid.py).
    """
    return _kontur_place_weight(place, "do_hexes.gpkg", "sources/do_grid.py")


def _do_counts():
    """ONE ENHOGAR-MICS6 2019, at provincia: 6 categories, 32 units, and EVERY ROW IS
    `modelled` IN §7.

    THE QUEUE PRICED THIS COUNTRY FROM LAPOP AND §9bn REFUSED IT -- three provinces never
    sampled, seven more in one wave-half only, so a third of the country would have been
    drawn at the national rate. The office's own survey answers instead. ENHOGAR is the
    Dominican household series, run since 2005, and its 2019 round was fielded as MICS6 with
    UNICEF; the household questionnaire carries `HC1A`, "a cual religion pertenece el jefe o
    la jefa del hogar", and ONE published the microdata openly. 31,488 households and 96,968
    people in them, against LAPOP's 8,904, and all 32 provinces against 29. sources/do.md is
    this country's record and sources.md §9cf is the write-up.

    THE CENSUS HALF IS CLOSED AND WAS CHECKED TWICE. CELADE hosts ONE's own REDATAM instance
    at prod.redatam.org/bindom/, and the published CPV2010 dictionary has no religion
    variable; the 2022 census person microdata's codebook runs P25 to P67 with no religion
    question either (P64 is self-identification by skin colour and culture, which is not the
    same thing). The UNSD oracle has no Dominican row, which on its own proves only that
    nothing was forwarded.

    IT IS THE HOUSEHOLD HEAD'S RELIGION, applied to everyone in the household, which is what
    MICS's own tabulations do and is a modelling step rather than a measurement. note_public
    says so to the reader.

    THE POPULATION IS THE 2022 CENSUS AND NOT COD-PS, and the reason is stronger than
    Ecuador's. COD-PS 2023 is a projection and is only 0.56% under ONE's census nationally,
    which is better than the 3.4% that made sources/ec_geo.py reject Ecuador's; per province
    it runs -23.7% in San Jose de Ocoa to +10.3% in Santo Domingo. A national agreement says
    nothing about the split.

    FIVE OF THE SIX CATEGORIES CARRY THEIR OWN GEOGRAPHY, decided by a split-half on cluster
    parity across ENHOGAR's 1,747 sampling units (§14.16). `TESTIGO DE JEHOVA` returns -0.02
    against a bar of +0.35 and is drawn at the national 0.99% in every province.
    """
    from do2019 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "do.csv",
                     dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])

    lut = pd.read_csv(HERE / "data" / "geo" / "do" / "do_lookup.csv", dtype=str)
    df["unit"] = df["geo_id"].map(dict(zip(lut["geo_id"], lut["unit"])))
    missing = sorted(df.loc[df["unit"].isna(), "geo_id"].unique())
    if missing:
        raise SystemExit(f"do.csv provinces with no polygon: {missing} -- re-run "
                         "sources/do_geo.py, the lookup is stale")
    if df["unit"].nunique() != 32:
        raise SystemExit(f"{df['unit'].nunique()} provinces, expected 32")

    df["node"] = df["source_category"].map(resolve)
    unmapped = sorted(df.loc[df["node"].isna(), "source_category"].unique())
    if unmapped:
        raise SystemExit(f"do.csv categories with no node: {unmapped}")
    df = df[df["count"] > 0]
    # EVERY row, without exception -- there is no measured tier in this country (§7).
    df["tier"] = "modelled"
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations", "tier"]]


ENTRY = {
    "do": dict(
        name="Dominican Republic",
        name_in="the Dominican Republic",
        source="Encuesta Nacional de Hogares de Propósitos Múltiples ENHOGAR-MICS6 2019 "
               "(Oficina Nacional de Estadística), against the X Censo Nacional de Población "
               "y Vivienda 2022 province populations",
        basis="self-identification, religion of the household head",
        view=[-72.2, 17.3, -68.2, 20.1],
        note_public=(
            "**No Dominican census has ever asked about religion, so this is a survey "
            "standing where a census would be.** The 2010 census dictionary carries no "
            "religion variable, the 2022 census microdata carries none either, and the UN's "
            "register of census religion tabulations has no Dominican row. What the "
            "statistics office does have is its own household survey: the 2019 round of "
            "ENHOGAR was fielded as MICS6 with UNICEF and asked which religion the head of "
            "each household belonged to. That is **96,968 people in 31,488 households**, "
            "across all 32 provinces, and the dots are drawn desaturated to say the numbers "
            "are survey respondents rather than a count. "
            "**The question was asked once per household, about its head.** Everyone living "
            "in that household is drawn in the head's column, which is what the survey's own "
            "tables do and what a map of people needs, but it is the honest limit of this "
            "map: 52.7% Catholic means 52.7% of Dominicans living in a Catholic-headed "
            "household, not 52.7% of Dominicans asked one by one. A Catholic mother with a "
            "son who reports no religion is drawn here as two Catholics, and the same survey "
            "series says how much that costs: ENHOGAR 2018 asked women aged 15 to 19 for "
            "their own religion, and they answered **37.1%** Catholic and **34.8%** no "
            "religion where this map draws that group at 49.8% and 21.6%. The evangelical "
            "share is the one the household question gets right, to within a tenth of a "
            "point, and the ranking of the provinces is the same either way, so what the "
            "head's answer moves here is the level and not the shape. "
            "**Catholic identification runs from 25.3% of La Romana to 82.8% of Hermanas "
            "Mirabal**, which is fifty-seven points across a country you can drive over in a "
            "day. It is under half in **thirteen of the thirty-two provinces**, and those "
            "thirteen hold 50.9% of the population, so a Dominican is now marginally more "
            "likely than not to live in a province where Catholics are a minority. The "
            "Catholic end is the Cibao, the northern farming interior, and six of its "
            "provinces are above 76%: Hermanas Mirabal, La Vega, Sánchez Ramírez, Espaillat, "
            "Monseñor Nouel and Duarte. "
            "**The two ways of not being Catholic are 200 kilometres apart and do not look "
            "alike.** In the eastern sugar belt the answer is evangelical: La Romana is "
            "**49.3%**, the highest in the country and more than twice the national rate, "
            "then Samaná at 43.3%, La Altagracia at 33.1% and San Pedro de Macorís at 32.4%. "
            "Those are the provinces of the old mills and the Anglophone Afro-Caribbean "
            "*cocolo* migration. On the "
            "Haitian border in the south-west the answer is no religion at all: **42.9% of "
            "Pedernales** and 42.7% of Baoruco, against 4.9% of La Vega. Nationally the two "
            "are 22.4% and 20.6%. "
            "**Evangelicals outnumber Catholics in five provinces**: La Romana, Samaná, La "
            "Altagracia and San Pedro de Macorís in the east, and Pedernales, which is the "
            "far south-western corner and gets there by having so few of either. "
            "**Adventists are 2.0% of the country**, three times their share in Mexico and a "
            "little above Peru's 1.5%, and they get their own geography here: 5.1% of Hato "
            "Mayor and 4.1% of San Cristóbal against 0.15% of San José de Ocoa. "
            "**Jehovah's Witnesses are the one answer not drawn where the survey found "
            "them.** Splitting the survey's 1,747 neighbourhoods into two half-samples and "
            "re-ranking the provinces returns +0.94 for Catholic and +0.91 for no religion, "
            "and **-0.02** for the Witnesses, so their 107,147 people go at the national "
            "0.99% in every province instead. Nobody is deleted; only the claim to know "
            "where they are is withdrawn. "
            "**The level is 2019 and the population is 2022.** Dominican religious "
            "affiliation has been moving fast, so a round fielded today would very likely "
            "find fewer Catholics; this map says where they were seven years ago, on the "
            "population the census counted three years later."),
        how="household survey, one round in 2019",
        grain="provinces, 337,000 people on average",
        counts=_do_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "do" / "do_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_do_place_weight,
        note="THE QUEUE PRICED THIS COUNTRY FROM LAPOP AND §9bn REFUSED IT. §11ad offered "
             "8,904 AmericasBarometer respondents over 29 of 32 provinces, with three never "
             "sampled and seven more in one wave-half only, several on n under 20; Anita's "
             "call on 2026-09-08 was that a third of the units at the national rate is "
             "inference wearing the same colours as measurement. **The office's own survey "
             "answers instead**: ENHOGAR-MICS6 2019 carries `HC1A`, the religion of the "
             "household head, with 31,488 households and 96,968 people in them across all "
             "32 provinces. That is §9ce's Uruguay lesson applied to the country where it "
             "mattered most. "
             "THE CENSUS HALF IS CLOSED AND WAS CHECKED TWICE RATHER THAN ASSUMED. CELADE "
             "hosts ONE's own REDATAM instance at prod.redatam.org/bindom/ and the published "
             "CPV2010 dictionary has no religion variable; the 2022 census person microdata's "
             "codebook runs P25_ORDEN to P67_ANO with no religion question, P64 being "
             "self-identification by skin colour and culture. The UNSD oracle's silence "
             "proves only that nothing was forwarded (§12). "
             "THE POPULATION IS ONE'S 2022 CENSUS AND NOT COD-PS, and the argument is "
             "stronger than Ecuador's. COD-PS 2023 is a projection and lands 0.56% under the "
             "census nationally, better than the 3.4% error that made sources/ec_geo.py "
             "reject Ecuador's; per province it runs -23.7% in San Jose de Ocoa to +10.3% in "
             "Santo Domingo, a 34-point spread under a national agreement of half a point. "
             "It would have put 285,000 people in Santo Domingo who are not there. "
             "THE PROVINCES ARE COD'S ADM2 AND NOT ITS ADM1, which is the trap for whoever "
             "copies this country. Every other Latin American country here reads ADM1; the "
             "Dominican ADM1 is the ten planning regions and reading it gives ten polygons "
             "and no error. The join is on the NAME, with one alias (ENHOGAR writes "
             "Bahoruco where COD and the census write Baoruco), and sources/do_geo.py "
             "asserts that the positional code join still gets only 3 of 32 right. "
             "THE HELD-OUT CHECK IS A PERMUTATION TEST. The survey's weighted province "
             "distribution tracks the 2022 census at r=+0.9988 and **none of 20,000 random "
             "pairings of the same 32 provinces reaches that** (best random +0.933). "
             "AND THE CROSS-CHECK IS THE ONE URUGUAY INVENTED. LAPOP's `prov` for this "
             "country is 2100 plus ONE's official province number, the same numbering "
             "ENHOGAR uses, so the two instruments can be laid side by side on 29 provinces: "
             "Catholic r=+0.84, non-Catholic Christian +0.76, no religion +0.76, none of "
             "20,000 random pairings reaching any of them, and national levels of 52.8/53.2, "
             "25.3/25.9 and 20.6/18.0. Two unrelated surveys eight years apart agree about "
             "this country. It also names §9bn's three unsampled provinces for the first "
             "time: 2110, 2116 and 2131 are Independencia, Pedernales and San Jose de Ocoa.",
    ),
}
