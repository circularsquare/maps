"""The Gambia 2013 Population and Housing Census religion (Gambia Bureau of Statistics, Spatial
Distribution Report, Annex H) -> religiondots taxonomy.

Five cells on the 8 Local Government Areas, 1,857,181 people, about 232,000 an LGA. sources/gm.md
is the write-up. The four religion cells are the form's four codes (Form A Part 2, column 7):
1 Islam, 2 Christianity, 3 Traditional, 4 Other. `Not stated` is a blank field; there is no code
for it and none for no religion.

EXCLUDED holds categories that are deliberately not on the tree.
REVIEW holds calls that are defensible but arguable, with the reason.
"""

EXCLUDED = {
    "Not stated":
        "970 people, 0.052%, whose religion field was left blank; the form has no code for a "
        "non-answer, so this is the processing's own cell. A non-answer, off the tree per spec "
        "§3.5, and carried in gm.csv so each LGA's rows close on Table B.1's population. It "
        "leans nowhere visible: 191 of Brikama's 447 also have no age recorded, and the rest are "
        "spread over the age rows like everyone else. 0.24% of Janjanbureh's age-not-stated row "
        "is the only other concentration (13 people).",
}

REVIEW = {
    "Islam":
        "-> islam, no branch. 1,782,859 people, 96.00%. The form has one Islam code, so the "
        "Ahmadiyya and any Sufi order are inside it; a branch would be an inference. At least "
        "99.3% of every LGA outside the Kombos (Banjul 94.7%, Kanifing 91.6%, Brikama 95.0%).",
    "Christianity":
        "-> christianity, the bare branch. 69,638 people, 3.75%. The form has one Christian code, "
        "so no church can be read from this census. Kanifing is 7.70% Christian, Banjul 4.84% and "
        "Brikama 4.82%, and those three hold 91.5% of the country's Christians; every LGA up "
        "river is under 1.3% (Basse 0.48%).",
    "Traditional":
        "-> indigenous.african. 1,028 people, 0.055%. The manual defines it as \"the traditional "
        "African religion\" (para 8.38). **Read it as a floor**, per §11b's continental rule: "
        "one code per person, so a Muslim or Christian who also takes part in traditional practice "
        "is counted under Islam or Christianity. 602 are in Brikama and 309 in Kanifing; Kuntaur "
        "has 3, all rural women.",
    "Other":
        "-> other.gm. 2,686 people, 0.145%. Code 4, which the manual says is for \"other "
        "religions\" with the name written in (\"e.g. Hindu\"); no table prints the write-ins. "
        "The form has no code for no religion, so anyone who gave none was put here or left "
        "blank, and nothing measures which. **1,106 of Kanifing's 1,996 are in the age-not-stated "
        "row**, beside 37 Muslims and 53 Christians, so they look like one block of records with "
        "neither age nor a named religion (an institution or a batch of forms), not a "
        "congregation; drawn as printed, since nothing says what they are. Without them Kanifing "
        "would be 0.24% Other rather than 0.53%. Banjul 0.21%, Basse 0.08%, elsewhere 0.05% or "
        "less.",
}

MAP = {
    "Islam": "islam",
    "Christianity": "christianity",
    "Traditional": "indigenous.african",
    "Other": "other.gm",
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
