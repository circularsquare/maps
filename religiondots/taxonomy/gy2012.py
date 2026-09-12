"""Guyana 2012 Population and Housing Census religious affiliation -> religiondots taxonomy.

Thirteen categories plus the universe total, at administrative region. See sources/gy.md.

**What makes this list worth having is not its length but its shape.** Guyana is a
Christian-majority country in the Americas in which a quarter of the population is Hindu and
a fifteenth Muslim — the descendants of indentured labourers brought from India after
emancipation — and the Bureau of Statistics splits the Christian side seven ways instead of
printing one `Christian` cell. Nothing else on this map has that combination: Trinidad's
neighbours are unavailable, Brazil and Chile have no Hindu population to speak of, and the
Indian censuses that do have one have no Caribbean diaspora in them.

EVERY CATEGORY RESOLVES TO A NODE THAT ALREADY EXISTED except the residual, so this country
cost the tree exactly one node. That is the `other.<cc>` pattern §12 recommends and not a new
root: Guyana's residual is 6,324 people and a root costs the whole palette a degree (§6.3).

EXCLUDED holds categories that are deliberately not on the tree.
REVIEW holds calls that are defensible but arguable, with the reason.
"""

EXCLUDED = {
    "Total":
        "the region's own population total, not a category. Table 2.19 prints it as a row "
        "and sources/gy.py recomputes it from the ten regions because the PDF's own Total "
        "COLUMN is clipped.",
}

REVIEW = {
    "Pentecostal":
        "-> christianity.pentecostal. 170,289 people, 22.8%, the largest single Christian "
        "answer in Guyana and larger than the Roman Catholics and Anglicans put together. "
        "The node is the undivided Pentecostal branch rather than any of its three children "
        "(trinitarian, oneness, charismatic), because the census offers one box and there "
        "is nothing in the table to divide it with. Worth knowing what is inside it: "
        "Guyanese Pentecostalism is overwhelmingly the Assemblies of God and the Church of "
        "God lines, so `trinitarian` would probably be right for most of it — but 'probably "
        "right for most of it' is exactly what §2.4 says to defer rather than guess, and "
        "`source_category` rides on every row so a later source can deepen it.",
    "Other Christians":
        "-> christianity.other. 155,050 people, 20.8% — the second largest Christian answer "
        "and a fifth of the country, which is a lot to leave unnamed. The Bureau publishes "
        "no breakdown of it at any geography. In Guyana this bucket is mainly the Baptists, "
        "the Congregationalists, the Lutherans, the Moravians and the Church of the "
        "Nazarene, all of which the tree could hold apart; a source that names them would be "
        "the single biggest upgrade available for this country. §12's deferred-matching "
        "note: the node wanted is a set of real Protestant branches, and this is where "
        "those people sit meanwhile.",
    "Jehovah Witness":
        "-> christianity.witnesses. Spelled without the possessive in the source and kept "
        "verbatim (§2.4). 9,602 people.",
    "Rastafarian":
        "-> rastafari. 3,496 people, 0.47%. Guyana is one of very few censuses anywhere "
        "that counts Rastafari as its own category rather than folding it into a residual, "
        "and the only one on this map that does. It is a real count of a movement that is "
        "usually invisible in official statistics.",
    "None":
        "-> unaffiliated. 23,419 people, 3.14%. This is an ANSWER — the form offers it — "
        "and is not the same thing as the non-response the Bureau prorated away (see "
        "below). Kept distinct from `Other` accordingly.",
    "Other":
        "-> other.gy. 6,324 people, 0.85%. The residual after thirteen named categories, "
        "and unusually small because the list above it is generous. In Guyana it will "
        "contain the Chinese community's Buddhists, a few hundred Bahá'ís' neighbours in "
        "the small-religion tail, and the Amerindian traditional practice that the form "
        "gives no box to — which is the §12 Ghana point in a South American costume: a "
        "census that offers no indigenous-religion category does not thereby show there is "
        "none, and Guyana's nine Amerindian nations are in Regions 1, 7, 8 and 9. Nothing "
        "here can size that and this file does not try to (§14.4).",
    "Bahai":
        "-> bahai. 421 people. Spelled without diacritics in the source; the tree's label "
        "carries them.",
}

# NON-RESPONSE IS NOT IN THIS MAPPING BECAUSE IT IS NOT IN THE SOURCE. Table 2.19's note
# says 363 `Religious Affiliation Not Stated`, 16,331 `No-Contact Persons` and 7,443
# `Institution Population` — 24,137 people, 3.23% of Guyana — were added together and
# **prorated across the thirteen categories**. So there is no non-response column to
# exclude, report or leave undrawn: the Bureau distributed it before publishing and the
# distribution is not recoverable. spec §3.5's rule is that non-response is reported and
# never filled; here it was filled upstream, which is a different situation and is recorded
# in sources/gy.md and in the country's note_public rather than undone (§14.4 — reversing it
# would mean inventing a distribution nobody published).

MAP = {
    "Anglican": "christianity.anglican",
    "Methodist": "christianity.methodist",
    "Pentecostal": "christianity.pentecostal",
    "Roman Catholic": "christianity.catholic",
    "Jehovah Witness": "christianity.witnesses",
    "Seventh Day Adventist": "christianity.adventist",
    "Bahai": "bahai",
    "Muslim": "islam",
    "Hindu": "hinduism",
    "Rastafarian": "rastafari",
    "Other Christians": "christianity.other",
    "None": "unaffiliated",
    "Other": "other.gy",
}


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
