"""
StatCan Census 2021 religion classification -> religiondots taxonomy.

**Branch-level mapping, deliberately.** StatCan publishes 147 leaf categories; mapping every
one to a leaf of our tree is the "meticulous cross-source matching" spec §2.4 defers. What
this file does instead is map StatCan's 21 INTERNAL NODES plus its 6 root-level leaves — 27
decisions — and let every leaf inherit from its nearest mapped ancestor. The leaf's own name
travels with the row, so deepening later costs nothing and redoes nothing.

That means the map can draw Canada at branch depth today without anyone having decided
whether StatCan's "Mennonite" is the same body as ASARB's "Mennonite Church USA".

`LEAF` holds the exceptions: leaves whose nearest mapped ancestor is the wrong answer, because
StatCan's tree and ours disagree about where something sits. Those disagreements are the
interesting part and each one carries its reason.
"""

# StatCan internal node (and root-level leaf) -> religiondots branch id.
NODE = {
    # --- the root; its children are handled individually
    "Total - Religion": None,

    "Christian": "christianity",
    "Catholic": "christianity.catholic",
    "Eastern Catholic": "christianity.catholic.eastern",
    "Other Catholic denominations": "christianity.catholic.independent",

    # StatCan's "Christian Orthodox" is the parent of BOTH Eastern and Oriental Orthodoxy.
    # Our tree has no such parent — they are separate communions and have been since 451
    # (branches.py) — so the two children map to their own families and the n.o.s. remainder
    # goes to Eastern, which is where the overwhelming majority of an unspecified "Orthodox"
    # answer in Canada belongs.
    "Christian Orthodox": "christianity.orthodox",
    "Eastern Orthodox": "christianity.orthodox",
    "Oriental Orthodox": "christianity.oriental",

    "Anabaptist": "christianity.anabaptist",
    "Reformed": "christianity.reformed.continental",
    "Latter Day Saints": "christianity.latterday",
    "Pentecostal and other Charismatic": "christianity.pentecostal",
    "Other Charismatic": "christianity.pentecostal.charismatic",

    # StatCan files the holiness bodies under Methodist; our tree gives holiness its own
    # branch (it is a movement that left Methodism). The node maps to Methodist and the
    # holiness bodies are pulled out by name in LEAF below.
    "Methodist and Wesleyan (Holiness)": "christianity.methodist",

    # A genuine grab-bag: 33 children spanning Adventists, non-denominational churches,
    # Restorationists and Brethren. The node maps to our own catch-all and the ones with a
    # real home are pulled out in LEAF.
    "Other Christian and Christian-related traditions": "christianity.other",

    "No religion and secular perspectives": "unaffiliated",
    "Secular perspectives": "secular",

    "Other religions and spiritual traditions": "other.ca",   # per-source residual, §3.11
    "Pagan beliefs and spiritual traditions": "paganism",
    "Chinese religions and spiritual traditions": "chinesefolk",
    "Japanese religions and spiritual traditions": "shinto",

    # --- leaves sitting directly under the root
    "Muslim": "islam",
    "Hindu": "hinduism",
    "Sikh": "sikhism",
    "Buddhist": "buddhism",
    "Jewish": "judaism",
    "Traditional (North American Indigenous) spirituality": "indigenous.northamerican",
}

# Leaves whose nearest mapped ancestor is wrong, because the two classifications disagree.
LEAF = {
    # StatCan puts these under Methodist and Wesleyan (Holiness); we give holiness its own
    # branch, and these are its defining bodies.
    "Salvation Army": "christianity.holiness.salvation-army",
    "Church of the Nazarene": "christianity.holiness.nazarene",
    "Free Methodist Church": "christianity.methodist.free",
    "Wesleyan Church": "christianity.holiness.wesleyan",

    # Under StatCan's "Other Christian and Christian-related traditions", but each has a home.
    "Seventh-day Adventist": "christianity.adventist.sda",
    "Non-denominational Christian": "christianity.nondenominational.independent",
    "Churches of Christ": "christianity.restorationist.churches-of-christ",
    "Christian and Missionary Alliance": "christianity.holiness.cma",
    "Jehovah's Witness": "christianity.witnesses",

    # Under "Other religions and spiritual traditions", but these are families in their own
    # right rather than residual.
    "Baha'i": "bahai",
    "Jain": "jainism",
    "Zoroastrian": "zoroastrianism",
    "Druze": "druze",
    "Taoist": "daoism",
    "Unitarian/Unitarian Universalist": "unitarianuniversalist",
    "Spiritualist": "spiritualism",

    # Under "Catholic" but Roman Catholic is specifically the Latin church.
    "Roman Catholic": "christianity.catholic.latin",
}


def resolve(category, parent_of):
    """religiondots branch for a StatCan category, via LEAF then the ancestor chain."""
    if category in LEAF:
        return LEAF[category]
    c, seen = category, 0
    while c is not None and seen < 25:
        if c in NODE:
            return NODE[c]
        c = parent_of.get(c)
        seen += 1
    return None
