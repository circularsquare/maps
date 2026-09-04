"""
DZS Popis 2021 religion classification -> religiondots taxonomy.

**Branch-level mapping**, and the shallowest one in the project: 12 categories, of which
only six are religions and one of those is `Ostali kršćani`. Croatia is 79% Catholic, so
drawn from this table alone it is close to a two-colour map — Catholic everywhere, Serbian
Orthodox along the Bosnian and Serbian borders, and an irreligious wedge in Istria.

**This is not the granularity Croatia actually publishes.** DZS's sheet 5 names **54
individual churches** at the same geography — four Orthodox jurisdictions kept apart
(Serbian, Macedonian, Montenegrin, Bulgarian), eleven separate Jewish communities, the
Croatian Old Catholic Church, a dozen Pentecostal and Baptist bodies. It refines two of
this table's residual categories rather than replacing the partition, so joining the two
needs care that has not been taken yet. `sources/hr.md` §4 records the shape of it. Until
then Croatia is drawn shallow on purpose rather than by necessity, which is a different
thing from every other country here and should not be mistaken for the source's ceiling.

EXCLUDED holds categories that are deliberately not on the tree.
REVIEW holds calls that are defensible but arguable, with the reason.
"""

EXCLUDED = {
    "Ukupno":
        "the unit's own population total, not a category.",
    "Ne izjašnjavaju se":
        "66,581 people, 1.72%, who declined to declare. Not a religion and not 'no "
        "religion', which is its own category.",
    "Nepoznato":
        "83,045 people, 2.14%, for whom the variable is unknown — a coverage residual "
        "rather than a refusal, and DZS keeps the two apart, so this file does too.",
}

REVIEW = {
    "Katolici":
        "-> christianity.catholic, the PARENT, not christianity.catholic.latin. The "
        "category is 'Catholics' with no rite, and it contains the Križevci eparchy's "
        "Greek Catholics as well as the Latin rite. cz2021.py files its equivalent "
        "'katolická víra (katolík)' the same way. 3,057,735 people — 79% of the country, "
        "so this one node is most of the Croatian map.",
    "Istočne religije":
        "-> other.hr, and this is the least satisfying call in the file. 'Oriental "
        "religions' plainly means Buddhism, Hinduism, Sikhism and their neighbours in one "
        "cell, but the tree has no 'some Eastern religion, unspecified' node and picking "
        "any one of them would invent a fact about 3,392 people. Sheet 5 names the "
        "Buddhist, Krishna and Hindu communities separately and would resolve it.",
    "Nisu vjernici i ateisti":
        "-> unaffiliated. DZS merges 'not believers' with 'atheists' into one answer, "
        "where Czechia and Romania ask them apart. `unaffiliated` is a report of no "
        "religion and `secular` is a position; this category is both at once, and the "
        "no-religion reading is the larger part of it. 182,188 people.",
    "Agnostici i skeptici":
        "-> secular, which is the node for positions about religion rather than "
        "affiliations. Kept apart from the above because DZS keeps them apart.",
    "Ostali kršćani1)":
        "-> christianity.other. The footnote marker `1)` is part of the header string and "
        "is retained verbatim, per §2.4 — the key here must match what sources/hr.py "
        "writes, not a tidied version of it.",
    "Protestanti":
        "-> christianity.protestant, the 'named no body' node, which is exactly right "
        "here: the category IS 'Protestants' with no denomination. 9,956 people, and "
        "sheet 5 splits them across a dozen named churches.",
}

MAP = {
    "Katolici": "christianity.catholic",
    "Pravoslavci": "christianity.orthodox.canonical",
    "Protestanti": "christianity.protestant",
    "Ostali kršćani1)": "christianity.other",
    "Muslimani": "islam",
    "Židovi": "judaism",
    "Istočne religije": "other.hr",
    "Ostale religije, pokreti i svjetonazori": "other.hr",
    "Agnostici i skeptici": "secular",
    "Nisu vjernici i ateisti": "unaffiliated",
}


def _key(cat):
    return " ".join(str(cat).split())


def resolve(cat):
    """Source category -> taxonomy node id, or None if deliberately not on the tree."""
    c = _key(cat)
    if c in EXCLUDED:
        return None
    return MAP.get(c)
