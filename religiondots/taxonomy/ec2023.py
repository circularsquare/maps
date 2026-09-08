"""
LAPOP AmericasBarometer `q3c`/`q3cn` (Ecuador, waves 2010-2023) -> religiondots taxonomy.

**The same card as `gt2023.py` and `sv2023.py`, so the same eleven answers and the same eleven
calls.** Read `gt2023.py` for the arguments that are shared; only what is DIFFERENT about
Ecuador is written out here. Named for the last wave in the pool, per the registry convention.

    75.54%  Católico                                    -> christianity.catholic.latin
    10.95%  Evangélica y Pentecostal                    -> christianity.evangelical
     5.97%  Ninguna (Cree en un Ser Superior…)          -> unchurched
     2.60%  Protestante, Protestante Tradicional o…     -> christianity.protestant
     2.05%  Otro                                        -> other.ec
     1.43%  Testigos de Jehová                          -> christianity.witnesses
     0.67%  Agnóstico o ateo (no cree en Dios)          -> secular
     0.36%  Iglesia de los Santos de los Últimos Días   -> christianity.latterday
     0.35%  Religiones Orientales no Cristianas         -> other.ec
     0.06%  Religiones Tradicionales                    -> indigenous
     0.02%  Judío (Ortodoxo, Conservador o Reformado)   -> judaism

Every row is `modelled` in §7's sense: a survey share against a population count. **The count
is Ecuador's own 2022 census rather than a COD-PS projection**, which is new in this set and
is `sources/ec_geo.py`'s argument, not this file's.

## ECUADOR IS THE MOST CATHOLIC COUNTRY THIS SURVEY HAS DRAWN

75.54% against Guatemala's 45.5% and El Salvador's 46.65%. That is a different country from
the two Central American ones and it changes what the map is about: in Guatemala the story is
the Evangelical belt, and here it is **how unevenly a large Catholic majority thins out** —
94% of Loja against 62% of Sucumbíos, which is a Catholic share running one and a half times
higher at one end of the country than the other.

## THE ANSWER CARD CHANGED AFTER 2016, AND IT IS NOT A TREND

**This is the finding that a reader of this file most needs, and it is not Ecuador-specific.**
Two of the eleven answers are not available in every wave, and pooling hides it:

    code 77  `Otro`                        EXACTLY ZERO in 2010, 2012 and 2014, in every
                                           one of the 28 countries, on 29,374 / 27,254 /
                                           36,725 valid answers. It appears from 2016.
    codes 6, 10, 12  Mormons, Jewish,      EXACTLY ZERO in 2018 and 2023, in every country,
                     Jehovah's Witnesses   on 15,107 / 25,649 valid answers. They stop
                                           after 2016.

Zero Jehovah's Witnesses among 25,649 Latin Americans is not a measurement, it is a missing
box. LAPOP restructured the card between 2016 and 2018: **the named small denominations lost
their own answers and fall into `Otro`.** So two of the cells above are not what they look
like, and both errors run the same way:

  * **`Testigos de Jehová` at 1.43% is a FLOOR.** It is the Witness share of the three waves
    that offered the box, diluted by a fourth that did not. Ecuador's own wave figures are
    1.80%, 1.85%, 2.08% and then a structural zero.
  * **`Otro` at 2.05% is INFLATED at the late end and absent at the early end** — 0%, 0%,
    3.22%, 5.02% — and the 2023 value contains the Witnesses and Mormons who no longer have
    anywhere else to go. Ecuador's 2016->2023 rise in `Otro` is 1.80 points against the 1.81%
    those three cells hold, which is suggestive rather than proof on n=1,545.

**Nothing is lost from the partition** — every respondent is drawn, and the total is right.
What is wrong is the attribution of roughly half a percent of the country between two small
nodes, `christianity.witnesses` and `other.ec`. It is stated rather than corrected because
correcting it means deciding how the 2023 `Otro` decomposes, and nothing published says.

**This applies to Guatemala and El Salvador exactly as it does here** and neither file
mentions it; it was found while building Ecuador. `sources/ec.md` §3 carries the measurement.

## `Religiones Tradicionales` IS A FLOOR — GUATEMALA'S CASE, NOT EL SALVADOR'S

**0.06% of Ecuador, in a country that is 7.69% indigenous by its own 2022 census** (1,302,057
people self-identifying as *indígena*, plus 1,305,000 *montubios*). Nineteen respondents in
7,387 across fourteen years.

El Salvador's near-zero on this cell was left alone because its 2007 census counted 0.2%
indigenous, so the survey and the census agreed. **Ecuador's do not agree, and by two orders
of magnitude.** §11ad measured this same instrument reading **0.21x** a census on this same
cell in Suriname, and the mechanism is the card: one worldwide `Religiones Tradicionales` box
against a country where the question a local questionnaire would ask is about Andean and
Amazonian practice alongside Catholicism rather than instead of it. The Kichwa of the sierra
and the Shuar, Achuar and Waorani of the Oriente are, on this card, overwhelmingly Catholic
or Evangelical — which is true as far as it goes and is not the whole of what is there.

Drawn as given, at 10,000 people. `note_public` says it is a floor, in Guatemala's words.

## `unchurched` IS DRAWN WHERE THE SURVEY FOUND IT, WHICH IS THE SECOND TIME

5.97% nationally and it passes the split-half at **+0.63** against a bar of +0.45, so Ecuador
joins El Salvador as a country whose no-religion geography is a measurement rather than a
national rate spread flat. It runs **12.7% of Esmeraldas and 9.6% of El Oro against 0.3% of
Zamora Chinchipe** — a fortyfold spread, the widest this cell has shown anywhere in the set.

**And it is `unchurched` rather than `unaffiliated`, for gt2023.py's reason**: LAPOP prints
*cree en un Ser Superior pero no pertenece a ninguna religión* on the card, so folding it into
irreligion would move a million believing Ecuadorians into a category they explicitly declined.
`unaffiliated` draws nothing in this country. `secular` holds the 0.67% who told LAPOP they do
not believe in God, and the two cells are nine times apart.
"""

EXCLUDED = {}

REVIEW = {
    "Evangélica y Pentecostal":
        "-> christianity.evangelical. gt2023.py has the argument in full: the node exists "
        "**for sources that name Evangelical BESIDE Protestant**, which is its own note in "
        "branches.py, and LAPOP's card does exactly that. Ecuador is the third Latin "
        "American country to use it. 10.95%, a third of Guatemala's share and the lowest of "
        "the three. **It clears the split-half by 0.03** (+0.48 against a bar of +0.45 on 20 "
        "provinces) and is drawn on its own provincial shares — the mirror image of El "
        "Salvador's `Protestante Tradicional`, which missed by 0.02 and was not. Both were "
        "left where the arithmetic put them; a bar that moves when a value lands near it is "
        "not a bar.",
    "Protestante, Protestante Tradicional o Protestante no Evangélico":
        "-> christianity.protestant, the 'unspecified' node. 2.60%, a THIRD of El Salvador's "
        "7.97%, and its split-half is +0.16 against a bar of +0.45, so it is drawn at the "
        "national rate inside each province's residual. Note the wave series — 4.66%, 1.92%, "
        "1.34%, 2.49% — which is not a trend but is not stable either, and is the shape of a "
        "box respondents move in and out of depending on how the interviewer reads a "
        "three-clause answer aloud.",
    "Testigos de Jehová":
        "-> christianity.witnesses, and **the number is a floor because the box was "
        "withdrawn**. See the module docstring: codes 6, 10 and 12 are exactly zero in the "
        "2018 and 2023 waves across every country in the merge, which is a card change and "
        "not a collapse of the Witnesses in Latin America. Ecuador's own three measured "
        "waves are 1.80%, 1.85% and 2.08%; the pooled 1.43% is those diluted by a fourth "
        "wave that could not record them. Drawn at 242,000 people, which is fewer than there "
        "are.",
    "Otro":
        "-> other.ec. **The composition of this cell is not constant across the pool.** It "
        "does not exist before 2016 (a structural zero, not a measurement) and from 2018 it "
        "absorbs the Jehovah's Witnesses, Mormons and Jews whose own boxes were withdrawn. "
        "Kept separate from `Religiones Orientales` in `source_category` even though both "
        "resolve here, so a later source can move either one (spec §2.4).",
    "Religiones Orientales no Cristianas":
        "-> other.ec rather than a node of its own, following mx2020.py's `Origen oriental`, "
        "gt2023.py and sv2023.py. 0.35%, and one bucket for everything from Buddhism to "
        "Bahá'í, so it cannot be resolved to a tradition without inventing the split. Its "
        "wave series climbs 0.08% -> 0.99%, which is real growth, a changing card, or both.",
    "Ninguna (Cree en un Ser Superior pero no pertenece a ninguna religión)":
        "-> unchurched, and here it is a real geography rather than a national average — the "
        "second country in this set where that is true. 5.97% nationally, **12.7% of "
        "Esmeraldas and 9.6% of El Oro against 0.3% of Zamora Chinchipe**, split-half +0.63. "
        "Esmeraldas is the Afro-Ecuadorian coastal province and Zamora Chinchipe is the "
        "southern Amazon frontier; that is a reading the map suggests and this file does not "
        "assert. Its wave series is the steepest movement in the country — 2.58% in 2010 to "
        "**10.04% in 2023** — so the pooled level understates where Ecuador now is.",
    "Religiones Tradicionales":
        "-> indigenous, the bare family node. **0.06% against a country that is 7.69% "
        "indigenous by its own 2022 census, and it is a floor** — Guatemala's case rather "
        "than El Salvador's, where a near-zero was plausible because the census agreed. §11ad "
        "measured this instrument reading 0.21x a census on this cell in Suriname. Drawn as "
        "given because nothing published says what the right number is, and `note_public` "
        "says it is a floor.",
    "Agnóstico o ateo (no cree en Dios)":
        "-> secular, following mx2020.py. 0.67%, against 5.97% on `unchurched` beside it. "
        "The Latin American pattern this set keeps finding: leaving a church is common and "
        "leaving belief is not, and the two cells are nine times apart here.",
    "Judío (Ortodoxo, Conservador o Reformado)":
        "-> judaism. **One respondent in 7,387**, and that one is in the 2012 wave. Drawn "
        "because the partition is closed and dropping it would move those people somewhere "
        "else, not because the survey can see Ecuadorian Jewry — and see the docstring: the "
        "box itself was withdrawn after 2016. It draws as a §4.3 presence ring at every dot "
        "value this map offers.",
    "Iglesia de los Santos de los Últimos Días (Mormones)":
        "-> christianity.latterday. 0.36%, and a floor for the same reason as the Witnesses: "
        "the box is gone from 2018. Ecuador has one of the larger Latter-day Saint "
        "populations in South America by the church's own reporting, and 61,000 people is "
        "well under any figure it publishes — but a church's own roll counts differently "
        "from a survey's self-identification (§11q, Japan), so the gap is not evidence of "
        "the size of the undercount.",
}

MAP = {
    # ---------------------------------------------------------------- Catholic
    "Católico": "christianity.catholic.latin",

    # ---------------------------------------------------------------- the two Protestant boxes
    "Evangélica y Pentecostal": "christianity.evangelical",
    "Protestante, Protestante Tradicional o Protestante no Evangélico": "christianity.protestant",
    "Testigos de Jehová": "christianity.witnesses",
    "Iglesia de los Santos de los Últimos Días (Mormones)": "christianity.latterday",

    # ---------------------------------------------------------------- no religion
    "Ninguna (Cree en un Ser Superior pero no pertenece a ninguna religión)": "unchurched",
    "Agnóstico o ateo (no cree en Dios)": "secular",

    # ---------------------------------------------------------------- other religions
    "Judío (Ortodoxo, Conservador o Reformado)": "judaism",
    "Religiones Tradicionales": "indigenous",
    "Religiones Orientales no Cristianas": "other.ec",
    "Otro": "other.ec",
}

# No COLUMNS dict (spec §7a-i-1). Every row here is `modelled` rather than `derived` — §7b's
# test, applied in countries.py::_ec_counts — and the roll-up is about where a DERIVED row was
# actually counted. Nobody counted religion in Ecuador at any level.


def resolve(category):
    """religiondots branch for a LAPOP answer, or None if deliberately off the tree."""
    if category in EXCLUDED:
        return None
    return MAP.get(category)
