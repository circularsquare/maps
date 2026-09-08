"""
StatCan Census 2021 religion classification -> religiondots taxonomy.

**Branch-level mapping, deliberately.** StatCan publishes 147 leaf categories; mapping every
one to a leaf of our tree is the "meticulous cross-source matching" spec §2.4 defers. What
this file does instead is map StatCan's 21 INTERNAL NODES plus its 6 root-level leaves — 27
decisions — and let every leaf inherit from its nearest mapped ancestor. The leaf's own name
travels with the row, so deepening later costs nothing and redoes nothing.

That means the map can draw Canada at branch depth today without anyone having decided
whether StatCan's "Mennonite" is the same body as ASARB's "Mennonite Church USA". **That
particular deferral was taken up on 2026-09-07** — see the Anabaptist block in `LEAF`, and
note that it did not need the US question answered: `christianity.anabaptist.mennonite` is
the branch both sides' bodies hang under, so Canada can name its 130,600 Mennonites without
committing to which American denomination any of them resembles.

`LEAF` holds the exceptions: leaves whose nearest mapped ancestor is the wrong answer, because
StatCan's tree and ours disagree about where something sits. Those disagreements are the
interesting part and each one carries its reason.

**Inheritance is a safe default for DEPTH and not for PLACEMENT**, and the Brethren are the
worked example. `Christian or Plymouth Brethren` sits in StatCan's 33-child
`Other Christian and Christian-related traditions`, so it inherited `christianity.other` —
a residual — when it has a home of its own. Its neighbour `Brethren, n.o.s.` sits in the
same grab-bag and correctly stays there, because its name refuses to say which Brethren it
is. **Read a group's leaf NAMES before deciding the group's own mapping is enough**: the
question is never only "is this ancestor too coarse", it is also "does this leaf belong to
a different family entirely".
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

    # --- StatCan's `Anabaptist` group and its Brethren, added 2026-09-07. 140,540 people
    #     were drawing at the undifferentiated parent while StatCan names four of the five
    #     bodies inside it, and every node was already in the tree.
    #
    # **CANADA IS THE SECOND-LARGEST MENNONITE POPULATION THIS MAP CAN DRAW**, after the
    # United States, and it was one grey cell. `Mennonite` is 127,431 of the group's
    # 140,540 and `Amish` 3,405. Neither needed the §2.4 question in this module's docstring
    # answered: `.mennonite` is the branch both StatCan's cell and ASARB's twenty bodies
    # hang under, so naming Canada's costs no claim about which American denomination any
    # of them resembles.
    "Mennonite": "christianity.anabaptist.mennonite",
    "Amish": "christianity.anabaptist.amish",
    #
    # `Be in Christ Church of Canada` is the **Brethren in Christ**, renamed in 2018 —
    # River Brethren, not Schwarzenau — and is USRC code 075's body under a Canadian name.
    # 3,591 people. `Apostolic Christian Church (Nazarean)` is the Fröhlich line, 898
    # people; the parent rather than `.nazarean-east`/`-west`, because StatCan's cell does
    # not say which conference.
    "Be in Christ Church of Canada": "christianity.anabaptist.brethren.bic",
    "Apostolic Christian Church (Nazarean)": "christianity.anabaptist.apostolic",
    #
    # **`Christian or Plymouth Brethren` IS NOT IN THAT GROUP AND IS FIXED HERE ANYWAY.**
    # 3,495 people. StatCan files it under `Other Christian and Christian-related
    # traditions` — the 33-child grab-bag above — so it was inheriting `christianity.other`
    # rather than being misfiled as Anabaptist, which is what a first look at the name
    # suggests and is worth stating so nobody re-derives it. It is a named body with a
    # home: the **Plymouth / Christian Brethren**, out of 1820s Dublin, which is the §12
    # collision `jm2011.py`, `bs2022.py`, `bb2010.py`, `gd2021.py` and `lc2022.py` all
    # document from the other side.
    "Christian or Plymouth Brethren": "christianity.plymouth",
    #
    # NOT listed, deliberately. `Anabaptist, n.o.s.` (4,962) and `Anabaptist, n.i.e.` (254)
    # say *unspecified* in as many words and keep the parent. `Brethren, n.o.s.` (5,628)
    # sits in the grab-bag beside the Plymouth cell and its own name refuses to say which
    # Brethren it is — Plymouth, Schwarzenau or River — so it keeps `christianity.other`;
    # guessing it would be mapping a category on its string alone, which is the thing §12
    # forbids.
}

# The CSD/CT COLUMN each allocated category was split out of -> the node that column names
# (spec §7a-i-1). StatCan publishes these at the census subdivision and the 147 narrow
# categories only at province, so every narrow row is derived and rolls back to its group.
#
# `No religion and secular perspectives` is 12.57M — a third of the country, and until
# §7a-i-1 all of it left the map whenever a reader asked what was counted. It was counted:
# StatCan published the column at the CSD itself.
#
# TWO ALLOCATED COLUMNS ARE DELIBERATELY ABSENT:
#
#   `Christian Orthodox` — 619,585, of which 64,873 resolve to `christianity.oriental`. The
#   Oriental churches have been a separate communion since 451 and that node's own comment
#   calls conflating the two "the commonest error in religion taxonomies". Rolling StatCan's
#   column to `christianity.orthodox` would commit it for 65,000 people; the tree walk
#   already gives both halves a correct ancestor, so nothing is lost by leaving it out.
#
#   `Methodist and Wesleyan (Holiness)` — 98,140 split between `christianity.methodist` and
#   `christianity.holiness` (the Salvation Army alone is 50,670). StatCan's group is two
#   families and this tree keeps them apart.
COLUMNS = {
    "Anabaptist": "christianity.anabaptist",
    "Catholic": "christianity.catholic",
    "Latter Day Saints": "christianity.latterday",
    "No religion and secular perspectives": "unaffiliated",
    # `christianity`, not `christianity.other` — see hu2022.COLUMNS. StatCan's column is a
    # residual; that node is explicitly not one.
    "Other Christian and Christian-related traditions": "christianity",
    "Other religions and spiritual traditions": "other.ca",
    "Pentecostal and other Charismatic": "christianity.pentecostal",
    "Reformed": "christianity.reformed",
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
