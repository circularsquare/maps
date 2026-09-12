"""
INE Encuesta Nacional de Hogares Ampliada 2006, `e29_1` (Uruguay) -> religiondots taxonomy.

Seven answers, and Uruguay's card is unlike any other in the Americas here: it offers Judaism
and Afro-American religion their own boxes while collapsing every non-Catholic Christian body
into one, and it splits no-religion into a believer and a non-believer half. That is a card
written for the country it was asked in.

    46.19%  Catolico                       -> christianity.catholic.latin
    26.90%  Creyente sin confesion         -> unchurched
    15.57%  Ateo/agnostico                 -> secular
    10.02%  Cristiano no catolico          -> christianity.protestant
     0.63%  Umbandista/afroamericano       -> afrodiasporic
     0.38%  Judio                          -> judaism
     0.32%  Otra                           -> other.uy

Every row is `modelled` in §7's sense: a survey share against a population count. Nobody has
counted religion in Uruguay since the census of 1908.

## THE NO-RELIGION SPLIT IS THE POINT OF THE COUNTRY AND THE CARD MAKES IT FOR US

**42.5% of Uruguayans over six claim no religious affiliation**, which is not approached
anywhere else this map has drawn in the Americas, and INE's card splits it where it matters:
*Creyente sin confesión* against *Ateo/agnóstico*, 26.90% against 15.57%. Most Uruguayans who
have left the church have not left belief, and folding the two together would be the specific
error `gt2023.py` argues about at length for LAPOP's equivalent pair. Here it does not have to
be argued, because the two answers are separately printed and separately counted.

The two are also geographically different, which is the thing a single `no religion` cell
would have hidden. `secular` is a Montevideo and coastal-resort phenomenon: 21.7% of
Montevideo and 16.6% of Maldonado against 3.9% of Artigas. `unchurched` is the
opposite, an interior and border pattern: 41.6% of Rocha, 41.9% of Tacuarembó and 41.3% of
Treinta y Tres against 13.6% of Colonia and 15.5% of Paysandú. Uruguay's least religious
department by the combined measure is Rocha at 58.1%; its most religious is Paysandú, where
the two together are 20.5%.

## `Cristiano no catolico` IS ONE BOX FOR EVERYTHING PROTESTANT, AND INE SAYS WHAT IS IN IT

Unusually, the metadata defines the cell rather than leaving it to be guessed: *"incluye las
personas que pertenecen a la iglesia evangélica (incluyendo pentecostal y bautista), los
protestantes, adventistas y armenios"*. So it is Protestantism of every kind, plus the
Armenian Apostolic Church, which is not Protestant at all and which no other cell on this card
could have held. Uruguay's Armenian community is one of the larger ones in South America and
is concentrated in Montevideo; it is a small part of a cell holding 319,000 people, and there
is no way to take it out.

`christianity.protestant` is the answer-not-category node, and mx2020.py's `Cristiana` is the
precedent: an answer meaning non-Catholic Christian that names no body.

## WHAT THIS CARD DOES THAT NO LAPOP CARD CAN

The three AmericasBarometer countries on this map all carry an `other.<cc>` node holding
their non-Christian tail, because the LAPOP card offers `Religiones Orientales no Cristianas`
and `Otro` and nothing else below Christianity. **Uruguay's `other.uy` is 0.32% rather than
1% to 2.4%**, because Judaism and Afro-American religion — which in every LAPOP country fall
into that tail — have boxes of their own here and are drawn where they are.
"""

EXCLUDED = {}

REVIEW = {
    "Cristiano no catolico":
        "-> christianity.protestant, the answer-not-category node, following mx2020.py's "
        "`Cristiana` and `Evangélica`. **INE's own definition of the cell is in the "
        "docstring and it includes the Armenians**, who are Oriental Orthodox rather than "
        "Protestant. That is a real impurity in a node named Protestant; it is accepted "
        "because the alternative is bare `christianity`, which in a country whose other "
        "Christian cell is Catholic would read as 'Christian, unspecified' and could be "
        "taken to include Catholics. 10.02% of the country and no body is named inside it, "
        "so nothing finer is available: Uruguay's Waldensians, who founded Colonia "
        "Valdense in 1858 and are the oldest Protestant body in the country, are invisible "
        "here even though Colonia is the third most Protestant department at 18.1%.",
    "Umbandista/afroamericano":
        "-> afrodiasporic, the bare family node, following mx2020.py's `Raíces afro`. The "
        "cell names Umbanda and then widens to *afroamericano*, so it covers Umbanda, "
        "Quimbanda and the Batuque brought from Rio Grande do Sul, and this file will not "
        "split what INE did not. 0.63% and 20,000 people, concentrated in Montevideo "
        "(0.93%) and Rivera (0.78%) — the capital and the Brazilian border town, which is "
        "where an observer of Uruguayan Umbanda would look. **Almost certainly a floor.** "
        "Uruguayan practice of these religions runs well ahead of identification with them, "
        "as it does in Brazil, and a household survey asking a single self-definition "
        "question catches only the people who put it first.",
    "Creyente sin confesion":
        "-> unchurched, not `unaffiliated`. INE's wording is a believer with no "
        "denomination, which is the node Czechia's `věřící - nehlásící se k žádné církvi` "
        "created and which Mexico's `Sin adscripción religiosa (creyente)` uses. 26.90% of "
        "Uruguay, the largest single answer after Catholic, and putting it in irreligion "
        "would move 856,000 believing Uruguayans into a category they declined.",
    "Ateo/agnostico":
        "-> secular. 15.57%, against 26.90% on `unchurched` beside it. Uruguay is the one "
        "country in this set where the two are the same order of magnitude rather than nine "
        "times apart — Ecuador's pair is 5.97% and 0.67% — and that is the single fact that "
        "most distinguishes Uruguayan irreligion from the rest of Latin America's.",
    "Judio":
        "-> judaism. 0.38% and 12,000 people, of which **Montevideo holds 0.92% against "
        "0.06% across the other eighteen departments**, nine tenths of the whole cell in "
        "one city. Drawn on its own department shares "
        "even though its split-half RANK correlation is under the bar; `sources/uy.py`'s "
        "`UNDER_BAR` carries the argument, which is that the ranking being tested is the "
        "ranking of eighteen near-zeros and the thing being drawn is Montevideo. The two "
        "halves of 2006 agree on the shares at +0.93.",
    "Otra":
        "-> other.uy. **The one `other` cell on this map whose contents are readable**: "
        "`e29_2` is a free-text follow-up and it is transcribed. A third of the cell is "
        "blank and another fifth is `NO SABE`, `SIN DEFINICION`, `NO DEFINIDO` or `NO SE "
        "DEFINE`, so a large part of this node is people who chose `Otra` and then could "
        "not or would not name it. What is named is led by BUDISTA (6.7% of the cell), "
        "ESPIRITISTA (2.5%), METAFISICA (1.8%), PANTEISTA (1.6%), MORMON (1.5%), MUSULMAN "
        "(1.5%), TESTIGO DE JEHOVA (1.4%) and BAHAI, with SEICHO-NO-IE and ORTODOXO RUSO "
        "below them. **None of it is split off**: the largest named group is 0.02% of "
        "Uruguay, seven hundred people at a dot value of a thousand, and §3.11 says a bucket "
        "nothing can resolve stays whole. Drawn on its own department shares under "
        "`sources/uy.py`'s `UNDER_BAR`, with the caveat recorded there that how often an "
        "interviewer reaches for this box is partly a property of the fieldwork team.",
}

MAP = {
    # ---------------------------------------------------------------- Christianity
    "Catolico": "christianity.catholic.latin",
    "Cristiano no catolico": "christianity.protestant",

    # ---------------------------------------------------------------- no religion
    "Creyente sin confesion": "unchurched",
    "Ateo/agnostico": "secular",

    # ---------------------------------------------------------------- other religions
    "Judio": "judaism",
    "Umbandista/afroamericano": "afrodiasporic",
    "Otra": "other.uy",
}

# No COLUMNS dict (spec §7a-i-1). Every row here is `modelled` rather than `derived` — §7b's
# test, applied in countries.py::_uy_counts — and the roll-up is about where a DERIVED row was
# actually counted. Nobody has counted religion in Uruguay since 1908.


def resolve(category):
    """religiondots branch for an ENHA answer, or None if deliberately off the tree."""
    if category in EXCLUDED:
        return None
    return MAP.get(category)
