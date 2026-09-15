"""
World Values Survey wave 7, Puerto Rico 2018, `Q289` -> religiondots taxonomy.

**Six answers and a non-response, on the Puerto Rican card.** The questionnaire printed in the
survey team's report (p.121) offers `No pertenece`, `Católico`, `Protestante`, `Ortodoxo`,
`Judío`, `Musulmanes`, `Hindú`, `Budista` and `Otros (escribir)`; nobody answered Orthodox,
Jewish or Muslim. Labels are the card's, verbatim. Shares of the 1,117 who answered, which are
the report's Tabla 80:

    49.60%  Católico        -> christianity.catholic.latin
    20.32%  Otros           -> christianity (the root; christianity.other until 2026-09-14)
    20.32%  No pertenece    -> unaffiliated
     9.22%  Protestante     -> christianity.protestant
     0.45%  Budista         -> buddhism
     0.09%  Hindú           -> hinduism
            No answer/refused   excluded, 10 respondents

Every row is `modelled` (§7): a survey share laid on the Census Bureau's municipio estimates.
Named for the survey year, per the registry convention.
"""

EXCLUDED = {
    "No answer/refused":
        "WVS code -2, 10 of 1,127 respondents (0.89%). Not a religion; in `gap=`.",
}

REVIEW = {
    "Otros":
        "-> christianity, the ROOT (spec §6.6's branch that carries dots, drawn as a Christianity "
        "`unspecified` row). NOT christianity.other and NOT christianity.evangelical. First "
        "built on christianity.other; moved to the root 2026-09-14 after review (sources/pr.md "
        "§10-§11). The Puerto Rican card's code 8 is `Otros (escribir)`, a write-in, and the WVS "
        "archive harmonised all 227 write-ins to `Q289CS9` 80000000 `Other Christian; nfd` "
        "without splitting any of them; the text is not in the public file, so the answer as "
        "held names no church. "
        "WHY NOT christianity.other: its node note is 'bodies with no branch to belong to, not a "
        "residual'. The cells mapped there (gh2021, ke2019, bb2010, gy2012, mu2022, hu2022) are "
        "census rows adding up named small bodies, which fits that note; a write-in coded only "
        "`Other Christian` does not. "
        "WHY THE ROOT: it is the map's precedent for a Christian answer that names nothing. "
        "cl2024.py sends INE's coded Christian write-ins (`Otros cristianos y tradiciones "
        "relacionadas con Cristo`) there, pe2017.py sends `Cristiano` there, and Canada's "
        "`Christian, n.o.s.` is spec §6.6's example of a branch drawn with its own dots. "
        "WHY NOT christianity.evangelical: its own node note keeps it for a source that collects "
        "`Evangelical` as an answer, and this card has no such box. Most of the 227 are very "
        "likely evangelical and Pentecostal Christians who do not call themselves `Protestante` "
        "(Pew's 2014 survey found 33% of Puerto Ricans Protestant, and `Protestante` plus "
        "`Otros` here is 29.5%): the card is the WVS template with its instruction to amend the "
        "list not acted on, so they had no box. But the file cannot show how many wrote in "
        "something else (Witnesses, Adventists, espiritismo), and naming the family would be "
        "putting a word in their mouths. 20.32% of those answering.",
    "Protestante":
        "-> christianity.protestant, the answer node, not a family: the card names no "
        "denomination. 9.22%.",
    "No pertenece":
        "-> unaffiliated. The card has no belief qualifier and no atheist box; the archive codes "
        "it 100000020 `Non-religious`. 20.32%, the same count as `Otros` by coincidence (227 "
        "each, which the report's Tabla 80 prints too).",
    "Budista":
        "-> buddhism. Five respondents, all in Metropolitana (three in San Juan). Too few to "
        "place; sources/pr.py draws it without a regional share.",
    "Hindú":
        "-> hinduism. One respondent, in Yauco. Drawn without a regional share.",
}

MAP = {
    "Católico": "christianity.catholic.latin",
    "Protestante": "christianity.protestant",
    "Otros": "christianity",
    "No pertenece": "unaffiliated",
    "Budista": "buddhism",
    "Hindú": "hinduism",
}

# No COLUMNS dict (spec §7a-i-1): nothing here is `derived`, every row is `modelled`.


def resolve(category):
    """religiondots branch for a WVS-7 Puerto Rico answer, or None if deliberately off the tree."""
    if category in EXCLUDED:
        return None
    return MAP.get(category)
