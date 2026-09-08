"""
LAPOP AmericasBarometer `q3c` (El Salvador, waves 2010-2023) -> religiondots taxonomy.

**The same card as `gt2023.py`, so the same eleven answers and the same eleven calls.** Read
that file for the arguments; only what is DIFFERENT about El Salvador is written out here.
Named for the last wave in the pool, per the registry convention.

    46.65%  Católico                                    -> christianity.catholic.latin
    29.05%  Evangélica y Pentecostal                    -> christianity.evangelical
    12.44%  Ninguna (Cree en un Ser Superior…)          -> unchurched
     7.97%  Protestante, Protestante Tradicional o…     -> christianity.protestant
     1.42%  Religiones Orientales no Cristianas         -> other.sv
     0.93%  Otro                                        -> other.sv
     0.74%  Testigos de Jehová                          -> christianity.witnesses
     0.41%  Iglesia de los Santos de los Últimos Días   -> christianity.latterday
     0.35%  Agnóstico o ateo (no cree en Dios)          -> secular
     0.03%  Religiones Tradicionales                    -> indigenous
     0.01%  Judío (Ortodoxo, Conservador o Reformado)   -> judaism

Every row is `modelled` in §7's sense: a survey share against the COD-PS 2024 population.
THREE of the eleven carry their own department geography here against Guatemala's two — see
`sources/sv.py` — but that is a `sources/` decision about placement and changes what none of
these answers MEANS.

## What is different from Guatemala, and it is the grey ramp

**`unchurched` is 12.44% here against Guatemala's 4.85%, and it is the third-largest answer in
the country.** It also passes the split-half at **+0.74**, so unlike Guatemala's it is drawn
where the survey found it: 18.3% of Usulután and 17.2% of Morazán against 6.2% of La Paz.
That makes El Salvador the first country in this set whose no-religion geography is a
measurement rather than a national rate spread flat.

**And it is `unchurched` rather than `unaffiliated` for the same reason as Guatemala**: LAPOP
prints the gloss *cree en un Ser Superior pero no pertenece a ninguna religión* on the card,
which is INEGI's `Sin adscripción religiosa (creyente)` in different words. Folding it into
`unaffiliated` would move 790,000 believing Salvadorans into irreligion. `unaffiliated` draws
nothing in this country, which is what LAPOP's card looks like mapped honestly: it offers the
believer without an affiliation and the non-believer as two separate answers, and never a
plain "none". See gt2023.py.

## Two things worth noticing about the small end

**`Religiones Tradicionales` is 0.03%** — twenty-two respondents' worth across fourteen years,
against Guatemala's 0.22%. El Salvador's indigenous population was largely destroyed as a
public identity after the 1932 *matanza* and the 2007 census counted 0.2% indigenous, so a
near-zero here is not the same kind of instrument failure §11ad found in Suriname and
Guatemala. It is drawn as given and claims nothing.

**`Agnóstico o ateo` is 0.35%, the smallest of the nine LAPOP countries**, and a fortieth of
Uruguay's 16.0%. The Central American pattern is that leaving a church does not mean leaving
belief, and these two cells are that pattern in one country: 12.44% believing without a
church, 0.35% not believing.
"""

EXCLUDED = {}

REVIEW = {
    "Evangélica y Pentecostal":
        "-> christianity.evangelical. gt2023.py has the argument in full: the node exists "
        "**for sources that name Evangelical BESIDE Protestant**, which is its own note in "
        "branches.py, and LAPOP's card does exactly that. El Salvador is the second Latin "
        "American country to use it. 29.05%, and unlike Guatemala's it is drawn on its own "
        "department shares with a split-half of +0.82.",
    "Protestante, Protestante Tradicional o Protestante no Evangélico":
        "-> christianity.protestant, the 'unspecified' node. 7.97% and **the largest such "
        "cell in Central America** — half again Guatemala's share. "
        "**It misses the split-half bar by 0.02** (+0.52 against a bar of +0.54 on 14 units) "
        "and is therefore drawn at the national rate inside each department's residual "
        "rather than where the survey found it. That is the closest call in either LAPOP "
        "country so far and it was left alone on purpose: the bar is 1.96/sqrt(n-1), and "
        "moving it because a value landed just underneath is fitting the test to the answer.",
    "Ninguna (Cree en un Ser Superior pero no pertenece a ninguna religión)":
        "-> unchurched, and here it is a real geography rather than a national average. "
        "12.44% nationally, **18.3% of Usulután and 17.2% of Morazán against 6.2% of La "
        "Paz**, split-half +0.74. Those two are the eastern departments that sent the most "
        "people abroad during and after the civil war, which is a reading the map suggests "
        "and this file does not assert.",
    "Religiones Tradicionales":
        "-> indigenous, the bare family node. **0.03%, and El Salvador is the one country in "
        "this set where that is probably close to right** rather than the instrument failing: "
        "the 2007 census counted 0.2% indigenous, after the 1932 killings made Nahua-Pipil "
        "identity dangerous to state. §11ad's Suriname finding — LAPOP reading 0.21x a census "
        "on this cell — is about a card with no local option in a country with a large "
        "indigenous population, and the second half of that does not hold here. Drawn as "
        "given; `note_public` does not claim it is a floor, unlike Guatemala's.",
    "Religiones Orientales no Cristianas":
        "-> other.sv rather than a node of its own, following mx2020.py's `Origen oriental` "
        "and gt2023.py. 1.42% is the largest this cell gets in Central America and it is "
        "still one bucket for everything from Buddhism to Baha'i, so it cannot be resolved "
        "to a tradition without inventing the split.",
    "Otro":
        "-> other.sv. Kept separate from `Religiones Orientales` in `source_category` even "
        "though both resolve here, so a later source can move either one (spec §2.4).",
    "Agnóstico o ateo (no cree en Dios)":
        "-> secular, following mx2020.py. 0.35%, the smallest irreligion figure of the nine "
        "LAPOP countries, against 12.44% on `unchurched` beside it.",
    "Judío (Ortodoxo, Conservador o Reformado)":
        "-> judaism. One respondent in 9,063, about 700 people. Drawn because the partition "
        "is closed and dropping it would move those people somewhere else, not because the "
        "survey can see Salvadoran Jewry. It draws as a §4.3 presence ring at every dot "
        "value this map offers.",
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
    "Religiones Orientales no Cristianas": "other.sv",
    "Otro": "other.sv",
}

# No COLUMNS dict (spec §7a-i-1). Every row here is `modelled` rather than `derived` — §7b's
# test, applied in countries.py::_sv_counts — and the roll-up is about where a DERIVED row was
# actually counted. Nothing here was counted anywhere.


def resolve(category):
    """religiondots branch for a LAPOP answer, or None if deliberately off the tree."""
    if category in EXCLUDED:
        return None
    return MAP.get(category)
