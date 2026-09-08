"""
LAPOP AmericasBarometer `q3c` (Guatemala, waves 2010-2023) -> religiondots taxonomy.

**Branch-level mapping, like pe2017.py / ni2005.py / mx2020.py.** No leaves are created; the
answer's own wording travels with the row in `source_category` (spec §2.4).

The file is named for the last wave in the pool, which is the registry convention
(`taxonomy/<cc><YYYY>.py`). The source is all six waves from 2010; `sources/gt.py` and
`sources.md` §11ad have the construction.

    51.97%  Católico                                    -> christianity.catholic.latin
    34.52%  Evangélica y Pentecostal                    -> christianity.evangelical
     5.43%  Protestante, Protestante Tradicional o…     -> christianity.protestant
     4.85%  Ninguna (Cree en un Ser Superior…)          -> unchurched
     1.14%  Otro                                        -> other.gt
     0.64%  Agnóstico o ateo (no cree en Dios)          -> secular
     0.49%  Testigos de Jehová                          -> christianity.witnesses
     0.42%  Iglesia de los Santos de los Últimos Días   -> christianity.latterday
     0.31%  Religiones Orientales no Cristianas         -> other.gt
     0.22%  Religiones Tradicionales                    -> indigenous
     0.01%  Judío (Ortodoxo, Conservador o Reformado)   -> judaism

Shares are the pooled weighted national figures; the counts drawn are those shares against
the COD-PS 2024 population, and every row is `modelled` in §7's sense. Two categories carry
their own department geography and nine do not — that is `sources/gt.py`'s split-half test,
not a taxonomy decision, and it does not change what any of them MEANS.

**THE CARD OFFERS TWO NON-CATHOLIC CHRISTIAN BOXES AND THAT IS THE WHOLE REASON THIS COUNTRY
IS INTERESTING.** Every other Latin American source on this map has one. See REVIEW.

**NOTHING LANDS ON `unaffiliated`, WHICH IS UNUSUAL AND IS THE CARD'S DOING**, not an
oversight. See REVIEW.
"""

EXCLUDED = {}

REVIEW = {
    "Evangélica y Pentecostal":
        "-> christianity.evangelical, and this is the call in this file worth arguing. "
        "`christianity.evangelical` exists **for sources that name Evangelical BESIDE "
        "Protestant** — that is its own note in branches.py, written for Kenya, where KNBS "
        "publishes both cells. LAPOP publishes both cells. So Guatemala is the first Latin "
        "American country here to use it, and br2010.py's opposite call is not a "
        "contradiction: IBGE's `Evangélicas` is Brazil's ONLY non-Catholic Christian box and "
        "holds the mission churches and the Pentecostals together, which is what "
        "`christianity.protestant` is for. Guatemala's card splits them, so the split is "
        "drawn. "
        "It does NOT go to `christianity.pentecostal` even though the box says *y "
        "Pentecostal*: the answer merges evangelicals with Pentecostals and sending all "
        "34.5% to the Pentecostal node would assert a division LAPOP did not make (§14.4). "
        "The node holds an ANSWER rather than a church, and is deliberately not a parent of "
        "anything.",
    "Protestante, Protestante Tradicional o Protestante no Evangélico":
        "-> christianity.protestant, the 'unspecified' node, which is what the answer is: "
        "the historic mission churches named by exclusion from the evangelical box. "
        "**It is drawn WITHOUT its own geography**, at 5.43% of every department's residual, "
        "because its split-half correlation across the six waves is **-0.04** — 480 "
        "respondents over 22 departments, an ordering that does not survive being asked "
        "twice. The people are drawn; the claim to know where they are is not made. "
        "`sources/gt.py`'s `stability()` has the argument and the numbers.",
    "Ninguna (Cree en un Ser Superior pero no pertenece a ninguna religión)":
        "-> unchurched, not `unaffiliated`, and the answer's own wording decides it: LAPOP "
        "prints the gloss *cree en un Ser Superior pero no pertenece a ninguna religión* on "
        "the card. That is exactly INEGI's `Sin adscripción religiosa (creyente)`, which is "
        "why `unchurched` exists (mx2020.py, spec §6.3a). Folding it into `unaffiliated` "
        "would move 865,000 believing Guatemalans into irreligion. "
        "Its split-half is **+0.21**, under the +0.43 bar, so like the Protestant cell it is "
        "drawn without its own geography.",
    "Agnóstico o ateo (no cree en Dios)":
        "-> secular, following mx2020.py's `Ateos/Agnósticos` exactly. 0.64%, which is the "
        "smallest irreligion figure of any LAPOP country in the region and about a "
        "twenty-fifth of Uruguay's.",
    "Religiones Tradicionales":
        "-> indigenous, the bare family node, and **the most under-drawn cell on this map's "
        "Guatemala**. 0.22% in a country the 2018 census found 43.6% indigenous. No node is "
        "invented for Maya spirituality (`costumbre`) because LAPOP does not name it: the "
        "card has one worldwide `Religiones Tradicionales` box and the Guatemalan "
        "questionnaire adds no local wording. spec §2.6 forbids assigning a school from "
        "outside the source, so it sits on the family. "
        "**And the figure is a floor rather than a measurement.** sources.md §11ad measured "
        "what this instrument does to folk practice in the one place a census could check "
        "it: in Suriname LAPOP's traditional-religion cell is **0.21x the census's** and the "
        "missing people come back as Christians. Nothing published says what Guatemala's "
        "right number is, so §14.4 forbids correcting it and `note_public` states it "
        "instead.",
    "Religiones Orientales no Cristianas":
        "-> other.gt rather than a node of its own. The box is one bucket for everything "
        "from Buddhism to Hinduism to Baha'i, 0.31%, 27 respondents in fourteen years. "
        "mx2020.py sends INEGI's `Origen oriental` to `other.mx` for the same reason: a "
        "bucket named by what it is not cannot be resolved to a tradition without inventing "
        "the split.",
    "Otro":
        "-> other.gt. 1.14%, the residual box on a card that names ten things. Kept separate "
        "from `Religiones Orientales` in `source_category` even though both resolve here, so "
        "a later source can move either one (spec §2.4).",
    "Judío (Ortodoxo, Conservador o Reformado)":
        "-> judaism. **One respondent in 8,919**, which rounds to about 2,000 Guatemalans "
        "and is the thinnest cell in this country by two orders of magnitude. Drawn because "
        "the partition is closed and dropping it would move those people somewhere else, "
        "not because the survey can see Guatemalan Jewry. It draws as a §4.3 presence ring "
        "at every dot value this map offers.",
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
    "Religiones Orientales no Cristianas": "other.gt",
    "Otro": "other.gt",
}

# No COLUMNS dict (spec §7a-i-1). Every row in this country is `modelled` rather than
# `derived` — §7b's test, applied in countries.py::_gt_counts — and the roll-up is about
# where a DERIVED row was actually counted. Nothing here was counted anywhere.
#
# `unaffiliated` is deliberately empty. LAPOP's card has no plain "none" box: it offers the
# believer without an affiliation and the non-believer as two separate answers, and both have
# their own node. A country with no `unaffiliated` row is what that card looks like drawn
# honestly, and the alternative would be to merge two answers that were chosen instead of
# each other.


def resolve(category):
    """religiondots branch for a LAPOP answer, or None if deliberately off the tree."""
    if category in EXCLUDED:
        return None
    return MAP.get(category)
