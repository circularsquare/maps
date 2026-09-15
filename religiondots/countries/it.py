# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _it_place_weight(place):
    """Italy's 7,903 comuni, weighted by WHICH population the dots are of. See `_ItWeighter`."""
    missing = [c for c in ("pop", "ital", "foreign", "unit") if c not in place.columns]
    if missing:
        print(f"  !! it_lau.gpkg has no {missing} — run sources/it_geo.py --fetch "
              f"then sources/it_geo.py")
        return None
    return _ItWeighter(place, _it_foreign_share())


def _it_foreign_share():
    from it2024 import resolve
    return _foreign_share("it", resolve, "nuts3")


def _it_counts():
    """Italy at NUTS 3: 107 province, from two halves and THREE resolutions.

    Italy has never asked about religion and **ISTAT does not collect it at all** — it is
    treated as sensitive data and is absent from the census, the permanent census and every
    multiscopo. So the halves are Greece's (§9z) and France's (§9ab):

      * **Italian citizens, 54.0M.** ESS pooled and restricted to `ctzcntr = Yes`.
      * **Foreign residents, 5.03M.** Eurostat's 2021 census table `cens_21ctz_r3`, 200
        named citizenships at NUTS 3 covering **100.00%** of the foreign population,
        crossed with Pew's composition for each origin country.

    **THIS IS THE FIRST COUNTRY DRAWN AT MORE THAN ONE RESOLUTION AT ONCE, AND IT IS
    ANITA'S CALL (§14).** ESS gave Italy NUTS 2 in rounds 6 and 8 and then took it back:
    rounds 9, 10 and 11 hold two thirds of the sample and all of the recent vintage, and
    carry only the five ripartizioni. Greece and France both met an asymmetry like this and
    resolved it by throwing the finer level away. Italy does the opposite:

      | foreign residents        | NUTS 3, 107 province   | measured citizenship counts |
      | Catholic / unaffiliated  | NUTS 2, 20 regioni     | rounds 6+8, 3,368 respondents |
      | every other category     | NUTS 1, 5 ripartizioni | rounds 9+10+11, 7,663 respondents |

    **The reason the mix is worth its explanation is where Italy's minorities are.** The
    citizen minorities are 1.38M people, 2.55% of citizens; the foreign residents are 5.03M
    and are drawn at 107 units. Four fifths of everyone this map exists to show is in the
    half with the finest geography in Europe, and the coarse level lands on the fifth that
    no Italian instrument can locate anyway. Drawing the whole country at NUTS 1 to keep one
    number per unit would have made Italy the coarsest thing on the map to protect a
    precision the foreign half already has.

    **Ten of twenty-one regioni take their Catholic/unaffiliated ratio from the
    ripartizione instead, and that is 14.3% of the population.** Molise is in no ESS round
    at all; the other nine are under `sources/it.py`'s `N_FLOOR` of 100 pooled respondents.
    The floor was added after disbelieving the output — at 23 respondents Trento drove South
    Tyrol to 48% unaffiliated and made it Italy's least Catholic region, which is the
    opposite of everything else known about it.

    **HOW THREE LEVELS BECOME ONE COLUMN.** scatter.py takes a single `unit`, so counting
    happens at NUTS 3 and the coarse rows are spread down proportional to each province's
    own **citizen** population. Nothing is invented: the dots would be placed by population
    inside the coarse unit regardless (§8.2), so this changes nothing spatially — only which
    column the pipeline reads. It does buy one real thing, which is that the citizen half is
    spread by citizen rather than total population, and Italy's foreign share runs from 1.6%
    in Carbonia to 20.6% in Prato. Every citizen row's `note` names the level its
    composition came from.
    """
    from it2024 import resolve

    cit = pd.read_csv(HERE / "data" / "normalized" / "it.csv", dtype={"geo_id": str},
                      keep_default_na=False, na_values=[""])
    cit = cit[cit["geo_level"] == "nuts3"].copy()
    cit["node"] = cit["source_category"].map(resolve)
    unmapped = sorted(set(cit.loc[cit["node"].isna(), "source_category"]))
    if unmapped:
        raise SystemExit(f"it.csv has unmapped source categories: {unmapped}")

    ext = pd.read_csv(HERE / "data" / "normalized" / "it_foreign.csv",
                      dtype={"geo_id": str})
    ext = ext[ext["geo_level"] == "nuts3"]

    df = pd.concat([cit[["geo_id", "node", "count"]], ext[["geo_id", "node", "count"]]],
                   ignore_index=True)
    df["congregations"] = 0
    # A survey is not a count of anybody and a nationality model is not either, so nothing
    # here is `measured` and §7's inferred-dots mode empties the country completely.
    df["tier"] = "modelled"
    return df.rename(columns={"geo_id": "unit"})[
        ["unit", "node", "count", "congregations", "tier"]]


ENTRY = {
    "it": dict(
        name="Italy",
        name_in="Italy",
        source="ESS rounds 6–11 (citizens) + Eurostat census 2021 × Pew 2020 (residents)",
        basis="self-identification, sample survey (citizens); nationality-derived (residents)",
        view=[6.2, 35.3, 18.8, 47.3],
        note_public=(
            "**Italy has never asked, and its statistical office does not collect religion "
            "at all** — it is treated as sensitive data and is absent from the census, the "
            "permanent census and every household survey ISTAT runs. So the country is "
            "drawn the way France and Greece are: a survey that does ask, for the 54.0 "
            "million people with Italian passports, and the 2021 census's own record of who "
            "lives where and where they are from, for the 5.0 million without. "
            "**Two thirds of Italy is Catholic and a quarter belongs to nothing** — 66.6% "
            "against 24.5%, with Islam at 3.8% and Orthodoxy at 2.7%. "
            "**The least religious places are Tuscany and Emilia-Romagna, and that is the "
            "oldest political map in the country.** Tuscany reports **43.2% no religion** "
            "and Emilia-Romagna **37.4%**, against **11.1% in Sicily** and 15.7% in Puglia. "
            "Those two regions are the core of the *zone rosse*, the anticlerical and "
            "communist heartland of the post-war republic, and the survey finds them "
            "unaided sixty years later. Sicily is 83.4% Catholic; Tuscany is 47.0%. "
            "**Islam is a map of where the work is.** Emilia-Romagna and Lombardy are both "
            "**6.0%** Muslim and Liguria 5.4%, against **1.3% in Sardinia** and 1.6% in "
            "Puglia — and at province level it is sharper still: Piacenza 7.1%, Imperia "
            "7.1%, Brescia 7.0%, Bergamo 6.7% and Modena 6.7%, the engineering and "
            "food-processing belt along the Via Emilia and the Lombard valleys, against "
            "0.75% in Oristano. This is the one thing on the map that runs north-south the "
            "opposite way to Catholicism. "
            "**Italy's largest single non-Catholic group is Romanian Orthodox** — 1.05 "
            "million people, more than every Protestant, Jewish, Buddhist and Hindu "
            "community on this map of Italy combined. Orthodoxy peaks in Lazio at 4.5% and "
            "in Viterbo province at 5.0%, and it is almost entirely a story of the last "
            "twenty-five years. "
            "**Prato is 20.6% foreign, the highest of any Italian province**, and Parma, "
            "Piacenza and Milan follow at about 15%. "
            "**What this map cannot do, and the first one is the big one.** The survey gave "
            "Italy twenty regions in 2012 and 2016 and then stopped: every round since has "
            "published only five macro-regions. So the Catholic and no-religion shares are "
            "drawn at the region — for 86% of the population; ten smaller regions had too "
            "few respondents and take their ratio from the macro-region instead — while "
            "**every smaller religion among Italian citizens is drawn at five units of 11.8 "
            "million people.** Foreign residents are drawn at 107 provinces throughout. "
            "That mixture is deliberate: four fifths of Italy's religious minorities are in "
            "the foreign half, so the coarse level lands on the part no Italian source can "
            "locate anyway. "
            "**Within a unit the dots are not scattered blindly.** People counted as "
            "foreign nationals are placed where foreign nationals live, comune by comune, "
            "and Italian citizens where Italian citizens live — 42.5% of Italy's foreign "
            "residents are in cities against 34.4% of its citizens. **And the religion that "
            "prompted this barely moved, which is worth knowing**: Italy's Hindus are in "
            "Rome and Milan and then the Po valley dairy belt and the Agro Pontino — "
            "Brescia, Bergamo, Mantova, Cremona, Latina — because they are largely Punjabi "
            "agricultural labour. They really are a substantially rural population. "
            "**The Jewish figure is roughly two and a half times too high** and is drawn "
            "anyway rather than dropped. Nine survey respondents scale to 64,000; the Union "
            "of Italian Jewish Communities has about 24,000 registered members. It is here "
            "so that a real community is visible, with the error stated rather than hidden. "
            "**Protestants are undercounted for a nameable reason.** The survey finds "
            "71,000, and Italy's Pentecostals alone number over 300,000 — but the "
            "\"other Christian\" answer holds 295,000, and the two added together match the "
            "independent count almost exactly. Italian Pentecostals and Jehovah's Witnesses "
            "do not tick *Protestant* on a form. Read the two together. "
            "**There is no atheist or agnostic box**, so everyone who reports belonging to "
            "nothing lands in a single category and Italy has nothing on the `secular` node "
            "— the same gap as France and Greece. The immigrant half counts people as their "
            "origin country's religion, an upper bound blind to conversion and lapse. And "
            "the **Arbëreshë**, the 60,000 Italo-Albanians of Calabria and Sicily who are "
            "Catholic of the Byzantine rite, are folded into Latin Catholicism, because the "
            "survey offers Italians exactly one Catholic box."),
        how="survey, 11,000 people; foreign residents by nationality",
        grain="provinces for foreign residents; regions and macro-regions for Italian citizens",
        counts=_it_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "it" / "it_lau.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_it_place_weight,
        note="**The first country here drawn at three resolutions at once, and the reason "
             "is that ESS gave Italy a geography and then took it back.** Rounds 6 (2012) "
             "and 8 (2016) carry `region` at NUTS 2 — 19 and 20 regioni. Rounds 9, 10 and "
             "11 carry it at NUTS 1, five ripartizioni, and they hold two thirds of the "
             "sample and all of the recent vintage. `regunit` says which level each round "
             "used and `it.py` asserts it rather than trusting it, so a future release that "
             "moves Italy again cannot be pooled as though nothing had changed. "
             "**Greece and France both met this asymmetry and threw the finer level away; "
             "Italy keeps it, on Anita's call.** Their reasoning was that mixing puts the "
             "sharper geography on the half with the weaker claim to it. Italy inverts the "
             "premise: its citizen minorities are 1.38M people and its foreign residents "
             "5.03M, so **four fifths of what this map exists to show is in the half that "
             "has 107 province**, and flattening everything to five units would have cost "
             "the country its best data to protect its worst. "
             "**THE GEO FILTER THAT LOOKED FINE AND WAS NOT.** The first build drew 97 "
             "province and 52.4M people, reported 99.83% coverage, and balanced perfectly — "
             "because the regex for a NUTS 3 code assumed the last character was a digit. "
             "Lombardia has twelve province and NUTS ran out of numerals, so Mantova is "
             "`ITC4A`, Lodi `ITC4B`, **Milano `ITC4C`** and Monza `ITC4D`; Sardegna and "
             "Sicilia do the same. Ten province and 6.6 million people vanished, including "
             "the largest one in the country, and **every internal check still passed** "
             "because the coverage percentage was taken against the truncated total. "
             "`it.py` now asserts the province count and the national population against "
             "the placement layer, which is built from a different file by a different "
             "script (§8.1). A coverage figure computed from the same filter that lost the "
             "rows cannot detect that the rows were lost. "
             "**THE SAMPLE FLOOR, ADDED BY DISBELIEVING THE OUTPUT.** The first honest build "
             "made **South Tyrol the least Catholic region in Italy** — 48% no religion — "
             "on 29 pooled respondents, with Trento's 23 close behind. Every other account "
             "of South Tyrol has it among the most observant places in the country. spec "
             "§3.9b withdrew the floor on how many UNITS a country may have and says "
             "nothing about how few PEOPLE may stand behind one, so `N_FLOOR = 100` was "
             "added: below it a regione's Catholic/unaffiliated ratio comes from its "
             "ripartizione. Ten of twenty-one fall back, which is **14.3% of the "
             "population** rather than half the map, because the ten are the small ones. "
             "The number is a knob and moving it is a §14 decision, not a tuning. "
             "**The cross-checks are against CESNUR, which nothing in the build reads.** "
             "Muslim citizens land at 491,000 against CESNUR's 417,900 (1.18×) and the "
             "Protestant-plus-other-Christian total at 366,000 against 378,000 (0.97×), "
             "which is the better of the two because it also diagnoses why the Protestant "
             "cell alone is wrong. Orthodox citizens come out at 0.53× — ESS finds about "
             "half of them — though the Orthodox total across BOTH halves, 1.6M, matches "
             "CESNUR's ~1.5M residents, so the shortfall is in the citizen/foreigner split "
             "rather than the magnitude. **Jews are 2.66× and that one is simply wrong**, "
             "on nine respondents. "
             "**And the divergence worth flagging is `unaffiliated`: 24.4% here against "
             "Pew's 13.3% for Italy.** That is an instrument difference rather than an "
             "error — ESS asks whether you belong to a particular religion or denomination, "
             "which collects far more \"no\" than a question asking what your religion is — "
             "and it is the same gap that puts France at 48.8%. Nothing is adjusted to "
             "close it; the basis is stated and the reader is told which question was "
             "asked. "
             "**The otto per mille is NOT used, and the reasoning is in `sources.md` "
             "§11l-ii.** Italy publishes a denominational roll of 42 million taxpayers "
             "across fourteen named confessions, by region, and it looked for one day like "
             "the best source in Western Europe. It is a spending vote and not an "
             "affiliation count: it gives the Waldensians 497,013 choices against a "
             "membership near 25,000, and the Orthodox Archdiocese 39,359 against 1.5 "
             "million residents. **The same column overstates one confession twentyfold and "
             "understates another by forty**, which is the proof it is measuring something "
             "else. Comune-level counts exist and are held by the Agenzia delle Entrate; "
             "they would make a broken measure more precise and are not worth requesting.",
    ),
}
