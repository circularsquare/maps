"""Guinea-Bissau RGPH 2009 religion (INE, Características socioculturais, Anexo Quadro 3) ->
religiondots taxonomy.

Five answers at região, 1,213,509 Guinean nationals who answered, nine units of about 135,000
each. sources/gw.md is the write-up. The census asked P.14 *"Qual é a sua Religião?"* as a
write-in with a two-digit code (Anexo 2 of the report), so the five are the coded answers the
office tabulated, not boxes on a card: there was no printed list whose wording could fold
one answer into another, and no denomination was tabulated.

EXCLUDED holds categories that are deliberately not on the tree.
REVIEW holds calls that are defensible but arguable, with the reason.
"""

EXCLUDED = {
    "Total":
        "the região's own total of Guinean nationals, not a category.",
    "ND":
        "228,718 people, 15.86% of Guinean nationals, who did not answer: the report "
        "(PDF p28) says *não responderam à esta pergunta* and puts it down to data quality "
        "or to religion being personal. A non-answer and not a religion, so it is off the "
        "tree per spec §3.5; it is read and carried in gw.csv so that each região's rows "
        "close on its own total. **It is not flat**: 25.7% of Bolama/Bijagós and 21.5% of "
        "Quinara against 10.5% of Gabú, and by etnia (Quadro 5) 25.5% of Balanta, 24.0% of "
        "Bijagó and 19.8% of Papel against 10.7% of Fula and 10.8% of Mandinga, which are "
        "the peoples with the largest and smallest animist shares. Some part of it is very "
        "probably traditional practice not named as a religion; the census cannot say how "
        "much, so it is not drawn as anything.",
}

REVIEW = {
    "Animista":
        "-> indigenous.african, the node Ghana added. 215,130 people, 14.92% of nationals "
        "and 17.73% of those who answered. Biombo is 40.1% animist and Cacheu 34.0%; Gabú is "
        "0.3%. **Read it as a floor**, on two counts: the report says (PDF p28) that a "
        "significant share of the population practises two religions and the census took "
        "one answer, so a Christian or Muslim who also keeps the balobas is counted once; "
        "and ND is highest among the same peoples (see EXCLUDED).",
    "Muçulmana":
        "-> islam, no branch. 650,402 people, 45.10%. Overwhelmingly Sunni and Maliki, with "
        "the Tijaniyya and Qadiriyya orders among the Fula and Mandinga, but the census "
        "codes neither, so a branch would be an inference.",
    "Cristão":
        "-> christianity, the bare branch. 318,021 people, 22.05%, and 45.8% of them in "
        "Bissau (SAB is 40.2% Christian). The 1991 census tabulated Catholic and other "
        "Christian separately (Resultados, Tabela 6.6, national); the 2009 reports print one "
        "row, so nothing finer exists at região. Left at the parent rather than split on "
        "1991's national ratio.",
    "Sem religião":
        "-> unaffiliated. 29,542 people, 2.05%. A coded write-in, not a box: the form prints "
        "no answer list, so the Mozambique ruling (a no-religion box that names animism, "
        "spec §6.3a) does not apply, and `Animista` is a separate code beside it. Quinara "
        "is 7.1% and Bolama/Bijagós 4.2%. Nothing goes to `secular`, which needs a "
        "separately counted atheist answer.",
    "Outra religião":
        "-> other.gw. 414 people, 0.03%. The report does not say what it holds; 279 of them, "
        "67.4%, are in Cacheu.",
}

MAP = {
    "Animista": "indigenous.african",
    "Muçulmana": "islam",
    "Cristão": "christianity",
    "Sem religião": "unaffiliated",
    "Outra religião": "other.gw",
}

# spec §7a-i-1: every row is measured at the node it is drawn on.
COLUMNS = {v: v for v in MAP.values()}


def _key(cat):
    return " ".join(str(cat).split())


_FOLDED = {_key(k): v for k, v in MAP.items()}
_FOLDED_EXCLUDED = {_key(k) for k in EXCLUDED}


def resolve(cat):
    """Source category -> taxonomy node id, or None if deliberately not on the tree."""
    c = str(cat)
    if c in EXCLUDED or _key(c) in _FOLDED_EXCLUDED:
        return None
    if c in MAP:
        return MAP[c]
    return _FOLDED.get(_key(c))
