"""
LAPOP AmericasBarometer `q3c` (Costa Rica, waves 2010-2023) -> religiondots taxonomy.

**The same card as `gt2023.py`, `sv2023.py`, `ec2023.py` and `pa2023.py`, so the same eleven
answers and the same eleven calls.** Read `gt2023.py` for the arguments; only what is
DIFFERENT about Costa Rica is written out here. Named for the last wave in the pool, per the
registry convention.

    63.06%  Católico                                    -> christianity.catholic.latin
    13.77%  Evangélica y Pentecostal                    -> christianity.evangelical
     9.49%  Protestante, Protestante Tradicional o…     -> christianity.protestant
     9.10%  Ninguna (Cree en un Ser Superior…)          -> unchurched
     1.49%  Agnóstico o ateo (no cree en Dios)          -> secular
     1.20%  Testigos de Jehová                          -> christianity.witnesses
     0.67%  Religiones Orientales no Cristianas         -> other.cr
     0.61%  Otro                                        -> other.cr
     0.31%  Religiones Tradicionales                    -> indigenous
     0.24%  Iglesia de los Santos de los Últimos Días   -> christianity.latterday
     0.07%  Judío (Ortodoxo, Conservador o Reformado)   -> judaism

Every row is `modelled` in §7's sense: a survey share against INEC's own 2022 provincial
population. Four of the eleven carry their own province geography, which is a `sources/`
decision about placement and changes what none of these answers MEANS.

## COSTA RICA IS THE SECOND COUNTRY IN THIS SET WITH A STATE READING TO CHECK AGAINST

INEC asks the religion question on one survey and has published no table of it, but the
variable-level metadata of its own microdata catalogue carries the answer's weighted frequency
distribution. **Encuesta de Mujeres, Niñez y Adolescencia 2018**, Costa Rica's MICS round 6,
variable `HC1A`, 8,490 households answering:

    Católica 65.16%    Religión cristiana (evangélica, pentecostal, mormona, otra) 25.50%
    No tiene religión 7.40%    Otra religión 1.01%
    Religión no cristiana (animista, judía, islámica, otra) 0.45%    NS/NR 0.48%

**The cards do not line up box for box, and the mapping is what has to carry that.** INEC's
`Religión cristiana` explicitly names Mormons inside it and has no separate Protestant,
Adventist or Witness box, so its 25.50% is the comparator for four of this module's cells
together — evangelical, Protestant, Latter-day Saints and Witnesses — which come to **24.69%** of
the survey's own answers, and **24.46%** as drawn once `sources/cr.py`'s residual step has
run; `note_public` quotes the drawn figure because every number there has to reproduce from
`data/normalized/cr.csv`. Catholic is like-for-like at 63.06% against 65.16%. The no-religion pair is NOT
like-for-like in the other direction: INEC offers one `No tiene religión` box at 7.40% where
this card offers a believer-without-a-religion and a non-believer separately, 10.59% of the
answers and 10.66% as drawn,
and it asks about the household head rather than a random adult. `sources/cr.md` §4 works all
three through.

## THE WITHDRAWN BOXES, EXACTLY AS IN PANAMA

`other.ec` documented that LAPOP's code 77 `Otro` is absent before 2016 and that codes 6, 10
and 12 — Mormons, Jews, Witnesses — are absent from 2018. Costa Rica's pool is three early
rounds and one late one, so both halves of that land at once:

    Otro                    2010  0.00%   2012  0.00%   2014  0.00%   2023  2.40%
    Testigos de Jehová            1.90%         1.10%         1.83%         0.00%
    Mormones                      0.14%         0.07%         0.72%         0.00%
    Judío                         0.00%         0.07%         0.20%         0.00%

Nobody stopped being a Jehovah's Witness in Costa Rica between 2014 and 2023. **So
`christianity.witnesses`, `christianity.latterday` and `judaism` are floors and `other.cr` is
one round's answer divided by four.**

## AND COSTA RICA DRAWS ITS WITNESSES ON THEIR OWN PROVINCE SHARES, WHICH NO OTHER LAPOP
## COUNTRY DOES

`Testigos de Jehová` is 1.20% here against 0.60% in Panama, clears §11ad's 1% eligibility
floor, and passes the split-half at +0.81 on a bar of +0.80. It is the smallest category any
country in this module places on its own geography. **The caveat is in the wave table above**:
the late half of that split-half is 2014 and 2023, and 2023 contributes exactly nothing, so
the test compared 2010-2012 against 2014 alone. It passed a real two-sample comparison with
less data behind it than the column widths suggest, and `sources/cr.md` §5 says so.
"""

EXCLUDED = {}

REVIEW = {
    "Católico":
        "-> christianity.catholic.latin, as in all four sibling modules. **What is different "
        "here is that it does NOT carry its own province geography**: the split-half returns "
        "+0.79 against a bar of +0.80 on seven units, the only country in this set where the "
        "largest category fails. That is a `sources/` placement decision and not a mapping "
        "one, and it does not change what the answer means; `sources/cr.py`'s CARRIES block "
        "has the reasoning, which is that Guanacaste alone moves from 4th most Catholic "
        "province to 7th between the wave halves, a fifteen-point fall.",
    "Evangélica y Pentecostal":
        "-> christianity.evangelical. gt2023.py has the argument in full: the node exists "
        "**for sources that name Evangelical BESIDE Protestant**, which is its own note in "
        "branches.py, and LAPOP's card does exactly that. 13.77%, drawn on its own province "
        "shares with a split-half of **+1.00**, a perfect rank agreement across the wave "
        "halves and the strongest reading of this cell in any country in the module.",
    "Protestante, Protestante Tradicional o Protestante no Evangélico":
        "-> christianity.protestant, the 'unspecified' node. **9.49%, and it is the largest "
        "this cell gets anywhere in this module** — three times Panama's 3.30% — and unlike "
        "Panama's it clears the split-half, at +0.86. So Costa Rica is the country where the "
        "Evangelical/Protestant distinction this card draws is doing the most work and is "
        "best supported, and also the one where it matters most that nothing here says which "
        "churches a Costa Rican respondent had in mind on either side of it.",
    "Ninguna (Cree en un Ser Superior pero no pertenece a ninguna religión)":
        "-> unchurched, and it is drawn where the survey found it: 9.10% nationally, "
        "**15.3% of Guanacaste against 3.6% of Cartago**, split-half +0.86. As in Guatemala, "
        "El Salvador and Panama this is the believer without an affiliation and not "
        "irreligion, and folding it into `unaffiliated` would move 462,322 believing Costa "
        "Ricans into `secular`'s branch. LAPOP's card never offers a plain 'none'. "
        "**INEC's own EMNA 2018 does offer one**, at 7.40%, against 10.59% for this cell and "
        "`secular` together; the two instruments are cutting the same population differently "
        "and `sources/cr.md` §4 works it through.",
    "Otro":
        "-> other.cr. **Exactly zero respondents in 2010, 2012 and 2014, and 36 of them in "
        "2023**, which is 2.40% of that round. The box was added to this instrument between "
        "the pools, so the 0.61% drawn here is one round's answer divided by four rounds. "
        "Kept separate from `Religiones Orientales` in `source_category` even though both "
        "resolve here, so a later source can move either one (spec §2.4).",
    "Religiones Orientales no Cristianas":
        "-> other.cr rather than a node of its own, following mx2020.py's `Origen oriental` "
        "and the four LAPOP countries before this one. 0.67% on 40 respondents is one bucket "
        "for everything from Buddhism to Bahá'í, and it cannot be resolved to a tradition "
        "without inventing the split. The wave pattern is 0.49%, 0.14%, 0.72%, 1.33% on 7, "
        "2, 11 and 20 respondents, which is a cell too thin to read a trend in.",
    "Testigos de Jehová":
        "-> christianity.witnesses, and it is a FLOOR even though it is the one cell of its "
        "size this module places on its own geography. 1.90%, 1.10% and 1.83% in the three "
        "early rounds and **exactly zero of the 1,527 answers in 2023**, because the box was "
        "withdrawn from the instrument and not because the Witnesses left. It clears the "
        "split-half at +0.81 on a bar of +0.80, but the late half of that comparison is "
        "carried by 2014 alone. Drawn, with both facts stated in `note_public`.",
    "Iglesia de los Santos de los Últimos Días (Mormones)":
        "-> christianity.latterday, and a floor for the same reason: 0.14%, 0.07%, 0.72%, "
        "then zero in 2023, on 14 respondents in all. **And INEC's card cannot corroborate "
        "it either**, because EMNA 2018 names Mormons inside its `Religión cristiana` box "
        "rather than giving them one of their own.",
    "Judío (Ortodoxo, Conservador o Reformado)":
        "-> judaism. **Four respondents in 5,903**, one in 2012 and three in 2014 and none "
        "in either of the other two rounds, which is 3,401 people once the share is applied. "
        "A floor on the withdrawn-box argument like the two above, and drawn because the "
        "partition is closed rather than because the survey can see Costa Rican Jewry. It "
        "draws three dots at 1:1,000 and falls under one dot at 1:10,000, where it "
        "becomes a §4.3 presence ring, the only one this country has at either value.",
    "Religiones Tradicionales":
        "-> indigenous, the bare family node, and **a floor**. 0.31% on 18 respondents "
        "against a census that asks every person `P07 ¿Se considera indígena?` and `P08 ¿A "
        "qué pueblo indígena pertenece?` and has eight indigenous peoples to record. That is "
        "§11ad's Suriname finding again: a worldwide card with a single `Religiones "
        "Tradicionales` box reads far under a census that asks properly. Milder than "
        "Guatemala's version of the same failure and real. Note also 0.42%, 0.34%, 0.39% and "
        "then **0.07% in 2023**, a single respondent, so the pooled figure is carried by the "
        "three early rounds.",
    "Agnóstico o ateo (no cree en Dios)":
        "-> secular, following mx2020.py. 1.49%, and it fails the split-half at +0.27 "
        "against a bar of +0.80, so it is drawn at the national rate inside each province's "
        "residual. Beside it `unchurched` is 9.10%: the Central American pattern is that "
        "leaving a church does not mean leaving belief, and these two cells are that pattern "
        "in one country. **Costa Rica's version of it is the least lopsided in the module** "
        "at six to one, against Panama's fourteen to one.",
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
    "Religiones Orientales no Cristianas": "other.cr",
    "Otro": "other.cr",
}

# No COLUMNS dict (spec §7a-i-1). Every row here is `modelled` rather than `derived` — §7b's
# test, applied in countries.py::_cr_counts — and the roll-up is about where a DERIVED row was
# actually counted. Nothing here was counted anywhere.


def resolve(category):
    """religiondots branch for a LAPOP answer, or None if deliberately off the tree."""
    if category in EXCLUDED:
        return None
    return MAP.get(category)
