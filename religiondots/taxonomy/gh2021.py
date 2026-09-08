"""GSS 2021 PHC religious affiliation -> religiondots taxonomy.

Nine categories plus the universe total, at district. Seven of the nine are drawn: `Total`
is the universe and `Christian` is a PARENT published beside all four of its own children.

**THE SHAPE OF THIS QUESTION IS THE THING TO UNDERSTAND ABOUT IT.** Ghana's form offers
four Christian boxes and one box each for Islam, Traditionalist, Other and No Religion. So
the census resolves 71% of the country into four Christian streams and leaves the whole
non-Christian world in three cells — the Philippines' lopsidedness (§9m) in a much smaller
list. Nothing here can separate Sunni from Ahmadi in a country with a large and distinct
Ahmadiyya community; nothing can name a single one of the African Independent Churches,
which are most of `Other Christian`; and Traditionalist is one box for every tradition
between the Akan and the Dagomba. The depth this file can reach is fixed by that, not by
effort.

EXCLUDED holds categories that are deliberately not on the tree.
REVIEW holds calls that are defensible but arguable, with the reason.
"""

EXCLUDED = {
    "Total":
        "the unit's own population total, not a category.",
    "Christian":
        "21,932,708 people, and a DUPLICATE rather than a category. GSS publishes it beside "
        "Protestant, Catholic, Pentecostal/Charismatic and Other Christian, and those four "
        "sum to it exactly — on all 295 rows of the cube, checked in sources/gh.py. Drawing "
        "it as well would count 71% of Ghana twice. This is Hungary's `Katolikus` case "
        "(§12) with the easy ending: there Roman and Greek Catholic left a 77,629-person "
        "remainder that had to be emitted, here the children are complete and there is "
        "nothing left over, so the parent is simply dropped.",
}

REVIEW = {
    "Pentecostal/ Charismatic":
        "-> christianity.pentecostal, the PARENT, and deliberately not "
        "christianity.pentecostal.charismatic. 9,703,351 people — 31.6%, the largest single "
        "answer in Ghana and larger than every Christian category in any other country on "
        "this map except US Catholicism. The cell is one box for two things the tree keeps "
        "apart: the classical Pentecostal denominations (the Church of Pentecost, which is "
        "the largest Protestant body in the country, Assemblies of God, Christ Apostolic) "
        "and the neo-charismatic ministries founded from the 1980s (Lighthouse Chapel, "
        "Action Chapel, Perez Chapel). Filing all of it under `.charismatic` would assert "
        "the second about the first; the parent says what the source says. The two-space "
        "typo and the space before `Charismatic` are GSS's and are kept verbatim per §2.4 "
        "— the key must match what sources/gh.py writes, not a tidied version.",
    "Protestant (Anglican, Lutheran, Presbyterian,  Methodist, etc.)":
        "-> christianity.protestant, the 'named no body' node. 5,364,320 people. The label "
        "names four bodies as examples and the census does not ask which, so the four are "
        "an illustration of the box and not a decomposition of it — mapping to "
        "christianity.methodist or .anglican on the strength of the label would be "
        "inventing a split the source does not make. The double space after "
        "`Presbyterian,` is GSS's, and the key is verbatim (§2.4).",
    "Other Christian":
        "-> christianity.other. 3,793,193 people, 12.3%, and the least satisfying call in "
        "this file. It is the residual inside Christianity, chosen INSTEAD of Catholic, "
        "Protestant and Pentecostal, so it says more than a bare `christianity` would — "
        "which is why it goes to `.other` with hu2022.py, au2021.py and ca2021.py rather "
        "than to the root with lk2024.py. What is actually inside it is mostly the AFRICAN "
        "INDEPENDENT CHURCHES: the Musama Disco Christo Church, the Twelve Apostles "
        "Church, the African Faith Tabernacle and the Aladura-type bodies, which are a "
        "distinct stream with a distinct history and no node on this tree. They get none "
        "because no source on the map counts them — Ghana's census does not name them and "
        "neither does anyone else's. If one ever does, this is where the node goes, and "
        "this bucket is where its people are today. Also inside: Jehovah's Witnesses, "
        "Seventh-day Adventists, Orthodox, and the Latter-day Saints, all of which the "
        "tree could hold and none of which Ghana separates.",
    "Islam":
        "-> islam, with no school or branch, because the census gives none. 6,108,530 "
        "people, 19.9%, concentrated in the five northern regions and in the zongo "
        "quarters of every southern city. Ghana has a large Ahmadiyya community — one of "
        "the oldest in West Africa, with its own hospitals and schools — and the tree can "
        "hold it apart from Sunni Islam, but one box means one node.",
    "Traditionalist":
        "-> indigenous.african, a node added for Ghana. 999,319 people, 3.25%, heaviest in "
        "the Northern, Savannah, North East and Oti regions. Read it as a floor rather "
        "than a measurement: the form makes it exclusive of the Christian and Muslim "
        "boxes, and in Ghana traditional practice frequently accompanies one of those "
        "rather than replacing it, so anyone who would answer both is counted elsewhere. "
        "Filed under `indigenous` and not `paganism`, which is for the Western neo-pagan "
        "revival — the same distinction cz2021.py and in2011.py make (§12: never map a "
        "category on its string alone).",
    "No Religion":
        "-> unaffiliated. 1,384,049 people, 4.50%. Nothing goes to `secular`, which needs "
        "an atheist, agnostic or explicitly positional category, and Ghana has none.",
    "Catholic":
        "-> christianity.catholic, the parent and not christianity.catholic.latin. Ghana's "
        "Catholics are overwhelmingly Latin rite and the parent is still the honest node: "
        "the census says `Catholic` and the distinction is not one it makes.",
    "Other Religion":
        "-> other.gh. 328,721 people. Per source, per spec §3.11.",
}

MAP = {
    "Protestant (Anglican, Lutheran, Presbyterian,  Methodist, etc.)":
        "christianity.protestant",
    "Catholic": "christianity.catholic",
    "Pentecostal/ Charismatic": "christianity.pentecostal",
    "Other Christian": "christianity.other",
    "Islam": "islam",
    "Traditionalist": "indigenous.african",
    "No Religion": "unaffiliated",
    "Other Religion": "other.gh",
}


def _key(cat):
    return " ".join(str(cat).split())


# The two GSS labels carrying double spaces would be destroyed by _key(), and §2.4 requires
# the key to match what the normaliser writes. So the lookup is tried verbatim first and
# whitespace-folded only as a fallback.
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
