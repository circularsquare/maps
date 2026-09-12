"""
LAPOP AmericasBarometer `q3c` (Panama, waves 2010-2023) -> religiondots taxonomy.

**The same card as `gt2023.py`, `sv2023.py` and `ec2023.py`, so the same eleven answers and
the same eleven calls.** Read `gt2023.py` for the arguments; only what is DIFFERENT about
Panama is written out here. Named for the last wave in the pool, per the registry convention.

    64.44%  Católico                                    -> christianity.catholic.latin
    21.06%  Evangélica y Pentecostal                    -> christianity.evangelical
     6.71%  Ninguna (Cree en un Ser Superior…)          -> unchurched
     3.30%  Protestante, Protestante Tradicional o…     -> christianity.protestant
     1.56%  Otro                                        -> other.pa
     1.20%  Religiones Orientales no Cristianas         -> other.pa
     0.60%  Testigos de Jehová                          -> christianity.witnesses
     0.49%  Agnóstico o ateo (no cree en Dios)          -> secular
     0.43%  Religiones Tradicionales                    -> indigenous
     0.18%  Iglesia de los Santos de los Últimos Días   -> christianity.latterday
     0.02%  Judío (Ortodoxo, Conservador o Reformado)   -> judaism

Every row is `modelled` in §7's sense: a survey share against the COD-PS 2023 population.
Three of the eleven carry their own province geography, the same three as El Salvador, but
that is a `sources/` decision about placement and changes what none of these answers MEANS.

## PANAMA IS THE ONE COUNTRY IN THIS SET WITH A STATE READING TO CHECK AGAINST

INEC's own perception module, run on the Encuesta de Propósitos Múltiples of April 2022 and
answered by 11,776 household informants, publishes a national religion split: **65% Catholic,
22% Evangelical, 2% Adventist, 1% Jehovah's Witnesses, 2% other religions and 8% none.**
Against this pool's 64.44%, 21.06% and 7.20% on the two no-religion cells added, that is
agreement to within a point on all three of the answers both cards carry. Nothing else in this
module has an independent reading of its level.

**Where the two cards differ is what the mapping has to carry.** INEC prints an Adventist box
and LAPOP does not, so Panama's Adventists, 2% of INEC's informants, are inside `Evangélica y
Pentecostal` or `Protestante Tradicional` here and cannot be pulled out. And INEC reads Jehovah's Witnesses at 1% against this pool's 0.60%, which is the
withdrawn-box problem below rather than a disagreement.

## THE WITHDRAWN BOXES, WHICH BITE HARDER HERE THAN IN ECUADOR

`other.ec` documented that LAPOP's code 77 `Otro` is absent before 2016 and that codes 6, 10
and 12 — Mormons, Jews, Witnesses — are absent from 2018. Panama's pool is three early rounds
and one late one, so both halves of that land at once, and the wave table is unambiguous:

    Otro                    2010  0.00%   2012  0.00%   2014  0.00%   2023  6.25%
    Testigos de Jehová            1.12%         0.77%         0.53%         0.00%
    Mormones                      0.20%         0.19%         0.33%         0.00%

Nobody stopped being a Jehovah's Witness in Panama between 2014 and 2023. **So
`christianity.witnesses`, `christianity.latterday` and `judaism` are floors, `other.pa` is a
quarter of what its box reads in the one round that offered it, and the split-half's verdict
of `undefined` on `Otro` is the arithmetic of a box that did not exist rather than a finding
about geography.** `sources/pa.py` and `note_public` both say so.

## `Religiones Tradicionales` IS A FLOOR AND THE MAP CANNOT SHOW WHERE IT WOULD BE

0.43% — twenty-six respondents across thirteen years — in a country where 9.4% of INEC's own
2022 informants self-identify as indigenous and the 2023 census asks the question of everyone.
That is §11ad's Suriname finding again: a worldwide card with one `Religiones Tradicionales`
box, offered in a country with seven indigenous peoples, reads far under a census.

**And Panama makes it worse in a way Guatemala did not.** The two comarcas where Guna and
Emberá-Wounaan practice is concentrated, Guna Yala and Emberá-Wounaan, have no `prov` code in
any LAPOP wave and are not drawn at all; the third, Ngäbe-Buglé, is drawn on n=138 and comes
back 58.8% Evangelical. So the cell is a floor **and** the geography that would have carried
it is the part of the country the survey never reached. Drawn as given, claiming nothing, and
`note_public` names it as a floor the way Guatemala's does.
"""

EXCLUDED = {}

REVIEW = {
    "Evangélica y Pentecostal":
        "-> christianity.evangelical. gt2023.py has the argument in full: the node exists "
        "**for sources that name Evangelical BESIDE Protestant**, which is its own note in "
        "branches.py, and LAPOP's card does exactly that. 21.06%, drawn on its own province "
        "shares with a split-half of +0.87, the highest of the three that pass. "
        "**It also carries Panama's Adventists**, who have a box on INEC's own 2022 card at "
        "2% of informants and none on LAPOP's; they are either here or in `Protestante "
        "Tradicional` and no source in this build can say which.",
    "Protestante, Protestante Tradicional o Protestante no Evangélico":
        "-> christianity.protestant, the 'unspecified' node. 3.30%, the smallest of the four "
        "LAPOP countries, and it fails the split-half at +0.46 against a bar of +0.65 on ten "
        "units, so it is drawn at the national rate inside each province's residual. Panama "
        "is the country where this cell is least of a puzzle: at a tenth of El Salvador's "
        "share there is little in it to place.",
    "Ninguna (Cree en un Ser Superior pero no pertenece a ninguna religión)":
        "-> unchurched, and it is drawn where the survey found it: 6.71% nationally, "
        "**21.7% of Bocas del Toro against 1.4% of Veraguas and Herrera**, split-half +0.82. "
        "As in Guatemala and El Salvador this is the believer without an affiliation and not "
        "irreligion, and folding it into `unaffiliated` would move 278,000 believing "
        "Panamanians into `secular`'s branch. LAPOP's card never offers a plain 'none'.",
    "Otro":
        "-> other.pa. **Exactly zero respondents in 2010, 2012 and 2014, and 94 of them in "
        "2023**, which is 6.25% of that round. The box was added to this instrument between "
        "the pools, so the 1.56% drawn here is one round's answer divided by four rounds. "
        "The split-half reports it as `undefined` for the same reason. Kept separate from "
        "`Religiones Orientales` in `source_category` even though both resolve here, so a "
        "later source can move either one (spec §2.4).",
    "Religiones Orientales no Cristianas":
        "-> other.pa rather than a node of its own, following mx2020.py's `Origen oriental` "
        "and the three LAPOP countries before this one. 1.20% is one bucket for everything "
        "from Buddhism to Bahá'í in a country that has one of the eight Bahá'í continental "
        "Houses of Worship, so it cannot be resolved to a tradition without inventing the "
        "split. Note the wave pattern: 0.59%, 0.13%, 2.93%, 1.13%, on 9, 2, 44 and 17 "
        "respondents, which is a cell too thin to read a trend in.",
    "Testigos de Jehová":
        "-> christianity.witnesses, and it is a FLOOR. 1.12%, 0.77% and 0.53% in the three "
        "early rounds and **exactly zero of the 1,505 answers in 2023**, because the box was "
        "withdrawn from the instrument and not because the Witnesses left. INEC's own April "
        "2022 module reads 1% on a card that still has the box, against the 0.60% pooled "
        "here. `other.ec` documents the same withdrawal.",
    "Iglesia de los Santos de los Últimos Días (Mormones)":
        "-> christianity.latterday, and a floor for the same reason: 0.20%, 0.19%, 0.33%, "
        "then zero in 2023.",
    "Judío (Ortodoxo, Conservador o Reformado)":
        "-> judaism. **One respondent in 6,105**, in the 2014 round, about 800 people once "
        "the share is applied. Panama has one of the larger Jewish communities in Central "
        "America and every published estimate of it is an order of magnitude above this, so "
        "the figure is drawn because the partition is closed and not because the survey can "
        "see Panamanian Jewry. It draws as a §4.3 presence ring at every dot value this map "
        "offers.",
    "Religiones Tradicionales":
        "-> indigenous, the bare family node, and **a floor**. 0.43% against 9.4% indigenous "
        "self-identification among INEC's own 2022 informants. Worse than Guatemala's "
        "version of the same failure, because the two comarcas that would carry the cell — "
        "Guna Yala and Emberá-Wounaan — have no LAPOP code and are not drawn at all. See the "
        "module docstring; `note_public` names it as a floor.",
    "Agnóstico o ateo (no cree en Dios)":
        "-> secular, following mx2020.py. 0.49%, and it is 0.00% in the 2014 round on 1,500 "
        "answers, so it is thin rather than measured. Beside it `unchurched` is 6.71%: the "
        "Central American pattern is that leaving a church does not mean leaving belief, and "
        "these two cells are that pattern in one country.",
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
    "Religiones Orientales no Cristianas": "other.pa",
    "Otro": "other.pa",
}

# No COLUMNS dict (spec §7a-i-1). Every row here is `modelled` rather than `derived` — §7b's
# test, applied in countries.py::_pa_counts — and the roll-up is about where a DERIVED row was
# actually counted. Nothing here was counted anywhere.


def resolve(category):
    """religiondots branch for a LAPOP answer, or None if deliberately off the tree."""
    if category in EXCLUDED:
        return None
    return MAP.get(category)
