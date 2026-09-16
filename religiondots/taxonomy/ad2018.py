"""
World Values Survey wave 7, Andorra 2018, `Q289` -> religiondots taxonomy.

**Eight answers, a write-in and a non-response, on the WVS template card.** The Catalan
questionnaire (F00006597) offers `No, no pertanyo a cap religió`, `Sí, catòlic`,
`Sí, protestant`, `Sí, ortodox (rus/grec/etc.)`, `Sí, jueu`, `Sí, musulmà`, `Sí, hindú`,
`Sí, budista` and `Altra, quina?`; the English version (F00006598) prints `Roman Catholic` for
code 1 and `Other (write in)` for code 8, which the file holds as code 9. The card was used in
Catalan, Spanish, French and English. Labels here are the file's `Q289` value labels as the
IHSN catalogue prints them, with the `{...}` short forms dropped (sources/ad.py). Respondents,
of 1,004, unweighted (W_WEIGHT is one value for everyone):

    641  63.84%  Catholic (Roman/Greek/etc)       -> christianity.catholic.latin
    302  30.08%  Do not belong to a denomination  -> unaffiliated
     17   1.69%  Orthodox (Russian/Greek/etc.)    -> christianity.orthodox
     11   1.10%  Muslim                           -> islam
     10   1.00%  Protestant                       -> christianity.protestant
      9   0.90%  Hindu                            -> hinduism
      6   0.60%  Buddhist                         -> buddhism
      5   0.50%  Other                            -> other.ad
      1   0.10%  Jew                              -> judaism
      2   0.20%  No answer/refused                excluded

Every row is `modelled` (§7): a survey share laid on the Department of Statistics' 2018
population estimate. Named for the survey year, per the registry convention.
"""

EXCLUDED = {
    "No answer/refused":
        "WVS code -2, 2 of 1,004 respondents (0.20%). Not a religion; in `gap=`.",
}

REVIEW = {
    "Catholic (Roman/Greek/etc)":
        "-> christianity.catholic.latin. The Catalan card says only `catòlic` and the English "
        "one `Roman Catholic`; the archive's Q289CS9 codes all 641 as 10100000 `Roman Catholic; "
        "Latin Church`. No Eastern Catholic answer is separable, and pr2018's `Católico` went the "
        "same way. 63.84%.",
    "Orthodox (Russian/Greek/etc.)":
        "-> christianity.orthodox, the parent, NOT christianity.orthodox.canonical. The card's "
        "`ortodox (rus/grec/etc.)` names no jurisdiction and Q289CS9 is 30100000 `Eastern "
        "Orthodox; nfd`, so the answer as held cannot tell a church in communion with the "
        "patriarchates from any other a respondent calls Orthodox. The same call as at2001's and "
        "as2015's `Orthodox`; be2024 put ESS's `Eastern Orthodox` on `.canonical`. The catalogue "
        "does not cross religion with nationality, so who the 17 are is not known. 1.69%.",
    "Protestant":
        "-> christianity.protestant, the answer node, not a family: the card names no "
        "denomination and has no evangelical box. 10 respondents, 1.00%.",
    "Other":
        "-> other.ad, a per-source residual (spec §3.11). The card's code 8 is a write-in "
        "(`Altra, quina?`), held as code 9, and the archive harmonised all 5 to Q289CS9 90000000 "
        "`Other; nfd` with the text withheld. NOT the christianity root: Puerto Rico's write-ins "
        "were coded `Other Christian; nfd` (pr2018.py), Andorra's were not, so nothing says they "
        "are Christian. 0.50%.",
    "Do not belong to a denomination":
        "-> unaffiliated. The card has no belief qualifier and no atheist box; the archive codes "
        "it 100000020 `Non-religious`. 30.08%.",
    "Jew":
        "-> judaism. One respondent, 76 people at the 2018 base: under one dot, a ring only.",
}

MAP = {
    "Catholic (Roman/Greek/etc)": "christianity.catholic.latin",
    "Do not belong to a denomination": "unaffiliated",
    "Orthodox (Russian/Greek/etc.)": "christianity.orthodox",
    "Muslim": "islam",
    "Protestant": "christianity.protestant",
    "Hindu": "hinduism",
    "Buddhist": "buddhism",
    "Other": "other.ad",
    "Jew": "judaism",
}

# No COLUMNS dict (spec §7a-i-1): nothing here is `derived`, every row is `modelled`.


def resolve(category):
    """religiondots branch for a WVS-7 Andorra answer, or None if deliberately off the tree."""
    if category in EXCLUDED:
        return None
    return MAP.get(category)
