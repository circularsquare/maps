"""Niger RGP/H 2012 religion (Institut National de la Statistique, État et structure de la
population du Niger en 2012, Tableau A 11) -> religiondots taxonomy.

Six cells on the 8 régions, 17,138,707 residents, about 2.1 million a région. sources/ne.md is the
write-up. The five religion cells are the household form's five codes (column C07: 0 Sans
religion, 1 Musulmane, 2 Chrétienne, 3 Animiste, 9 Autre à préciser); `ND` is the processing's
own cell, since the form has no code for no answer. Young children were given the code of their
father's or mother's religion.

EXCLUDED holds categories that are deliberately not on the tree.
REVIEW holds calls that are defensible but arguable, with the reason.
"""

EXCLUDED = {
    "ND":
        "42,608 people, 0.249%, whose religion was not recorded; the form has no code for a "
        "non-answer, so this is the processing's own cell. A non-answer, off the tree per spec "
        "§3.5, and carried in ne.csv so each région's rows close on Tableau A 11's total. It is "
        "highest in Niamey (0.61%) and Agadez (0.43%), the two most urban régions (Tableau 8), and "
        "lowest in Tillabéri (0.17%); Maradi holds the largest count, 8,000.",
}

REVIEW = {
    "Musulman":
        "-> islam, no branch. 16,978,889 people, 99.07%. The form has one Muslim code, so no "
        "school, order or movement can be read from this census; a branch would be an inference. "
        "Niamey is the lowest région at 97.45% and Tillabéri next at 98.65%; Tahoua, Diffa and "
        "Maradi are each over 99.3%.",
    "Chrétien":
        "-> christianity, the bare branch. 56,856 people, 0.33%. The form has one Christian code, "
        "so Catholics and Protestants cannot be told apart. Niamey is 1.40% Christian (14,353) and "
        "Tillabéri 0.78% (21,292); those two hold 62.7% of the country's Christians, and every "
        "other région is 0.10-0.24%. The report (p.58) puts Tillabéri's figure down to refugee "
        "camps for people who fled the war in Mali and to early missions, and says Christianity "
        "is mostly practised by foreigners (p.44); no table crosses religion with nationality, so "
        "neither is checked.",
    "Animiste":
        "-> indigenous.african. 34,786 people, 0.20%. Read it as a floor, per §11b's continental "
        "rule: one code per person, so a Muslim who also keeps a traditional practice is counted "
        "as Muslim. Highest in Niamey (0.34%) and Dosso (0.34%), then Zinder (0.26%), which holds "
        "the largest count, 9,053.",
    "Sans religion":
        "-> unaffiliated. 23,048 people, 0.13%. The form offers `Animiste` as its own code beside "
        "it (C07, code 3 against code 0), so this is the `separate` case and not the Mozambique "
        "or Laos box that also took traditional religion (spec §6.3a-ii). Highest in Dosso and "
        "Tillabéri (0.19% and 0.18%), lowest in Agadez and Diffa (0.02%). Nothing goes to "
        "`secular`.",
    "Autre à préciser":
        "-> other.ne, a NEW per-country residual node, following other.bf and other.gm. 2,520 "
        "people, 0.015%. Code 9, with the answer written in; no table prints the write-ins. "
        "Niamey holds 1,044 of them (41%, 0.10% of the city). Kept on a religion node rather than "
        "`unknown` because the form has no code for no answer, so this is a religion someone "
        "named, not a blank.",
}

MAP = {
    "Musulman": "islam",
    "Chrétien": "christianity",
    "Animiste": "indigenous.african",
    "Sans religion": "unaffiliated",
    "Autre à préciser": "other.ne",
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
