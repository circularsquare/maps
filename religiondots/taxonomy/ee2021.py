"""
Statistics Estonia Rahvaloendus 2021 religion classification -> religiondots taxonomy.

**Branch-level mapping**, like cz2021.py, pl2021.py and ro2021.py.

Two category lists, and both are mapped here because `sources/ee.py` carries both:

  * **21 categories** in RL21452, which is the only 2021 table published below the county
    and therefore the one that gets drawn.
  * **44 categories** in RL21451, which has only four places (the country and three
    settlement types) but names Anglicans, Quakers, Mormons, Hare Krishna, Wiccans,
    Satanists, Theosophists and Anthroposophists. They are not drawn anywhere — spec §3.9
    in its purest form, since the fine list has effectively no geography at all — but they
    are mapped so that the national tallies are right and so an allocation could be run
    later without revisiting this file.

WHAT ESTONIA CONTRIBUTES that nothing else does:

  * **Maausk and Taarausk**, as `Earth Believer` (3,860) and `Taara Believer` (1,770).
    The Estonian native-faith movement, reconstructionist with a real 1920s lineage, and
    no other census on earth enumerates it. Both go to `paganism`, which is exactly what
    that node is for — reconstruction of a pre-Christian religion, as against `esoteric`.
  * **Old Believers** (2,290), the Russian communities on the west shore of Lake Peipus.
    Third source in a row to need the node, after Poland and Romania.
  * The **least religious population on the map**: 650,900 of 1,114,030 people aged 15+,
    58.4%, feel no affiliation to any religion.

A TYPO IS PART OF THE DATA. Statistics Estonia spells the same category `Taara Beliver`
in RL21452 and `Taara Believer` in RL21451. Both spellings are mapped. Correcting it in
`sources/ee.py` was the alternative and would have been worse: the normalised CSV is
supposed to carry the source's own strings verbatim (spec §2.4), so the fix belongs here.

EXCLUDED holds categories that are deliberately not on the tree.
REVIEW holds calls that are defensible but arguable, with the reason.
"""

EXCLUDED = {
    "Religion total":
        "the unit's own 15+ population, not a category.",
    "Feels an affiliation to a religion":
        "universe subtotal — 321,340, the sum of the named religions below it, so drawing "
        "it would double every one of them.",
    "Refused to answer":
        "126,500 people, 11.4% of those aged 15+, who declined the question. Not a "
        "religion and not 'no religion', which is its own category (650,900). Excluded, "
        "so the Estonian map draws 972,000-odd of 1.11M answered.",
    "Religious affiliation unknown":
        "15,280 for whom the variable could not be established at all — a coverage "
        "residual rather than a refusal. Distinct from `Religion unknown`, which is a "
        "person who DOES report an affiliation without naming it and IS on the tree.",
}

REVIEW = {
    "Earth Believer / Taara Believer":
        "-> paganism, both. Maausk and taarausk are the two wings of the Estonian native-"
        "faith movement (Maavalla Koda holds both), reconstructing pre-Christian Estonian "
        "religion. `paganism` is defined for exactly that and is right in preference to "
        "`indigenous`, which branches.py reserves for the traditional religion of an "
        "indigenous people rather than a modern revival, and to `esoteric`.",
    "Christian Free Congregations":
        "-> christianity.nondenominational. Estonian `kristlikud vabakogudused` are the "
        "independent evangelical congregations that belong to no union. 6,070 people. "
        "`christianity.pietist` would be defensible; the deciding point is that the "
        "category is defined by the ABSENCE of an affiliation to a body.",
    "Religion unknown":
        "-> other.ee. A person who reports feeling an affiliation to a religion but whose "
        "religion is not recorded — the same answer as Czechia's 'věřící - hlásící se k "
        "církvi - název neuveden', and filed the same way. 1,530 people. NOT `unchurched`, "
        "which is belief with an explicit absence of affiliation, and this is the reverse.",
    "Animist":
        "-> paganism, following cz2021.py. An Estonian write-in of 'animist' is a Western "
        "neo-animist self-description rather than the religion of an indigenous people.",
    "Anthroposophist / Theosophist / New Ager / Pantheist / Satanist":
        "-> esoteric, all five. The Western esoteric current, which is what that node "
        "holds; cz2021.py sends satanismus there for the same reason.",
    "Charismatic Episcopal Church":
        "-> christianity.pentecostal.charismatic. A convergence body — charismatic "
        "worship with an episcopal polity — so `christianity.anglican` is arguable, but "
        "it is not in the Anglican Communion.",
    "Hare Krishna":
        "-> hinduism, with `Hindu`. ISKCON is a Gaudiya Vaishnava lineage; the project has "
        "no `hinduism.vaishnava` node and pl2021.py already files the Polish ISKCON here.",
}

MAP = {
    # ---------------------------------------------------------------- Christianity
    "Lutheran": "christianity.lutheran",
    "Orthodox": "christianity.orthodox.canonical",
    "Old Believer": "christianity.orthodox.oldbeliever",
    "Armenian Apostolic Church": "christianity.oriental",
    "Roman Catholic": "christianity.catholic.latin",
    "Calvinist": "christianity.reformed",
    "Anglican": "christianity.anglican",
    "Baptist": "christianity.baptist",
    "Methodist": "christianity.methodist",
    "Pentecostal": "christianity.pentecostal.trinitarian",
    "Charismatic Episcopal Church": "christianity.pentecostal.charismatic",
    "Adventist": "christianity.adventist",
    "Jehovah's Witness": "christianity.witnesses",
    "Mormon": "christianity.latterday",
    "New Apostolic Church": "christianity.restorationist",
    "Quaker": "christianity.friends",
    "Evangelical Christian": "christianity.pietist",
    "Christian Free Congregations": "christianity.nondenominational",
    "Christian (other)": "christianity.other",

    # ---------------------------------------------------------------- other traditions
    "Muslim": "islam",
    "Judaist": "judaism",
    "Buddhist": "buddhism",
    "Hindu": "hinduism",
    "Hare Krishna": "hinduism",
    "Sikh": "sikhism",
    "Shintoist": "shinto",
    "Baha'i": "bahai",

    # ---------------------------------------------------------------- native faith
    # The reason Estonia is worth having at all — see the header.
    "Earth Believer": "paganism",
    "Taara Believer": "paganism",
    "Taara Beliver": "paganism",          # Statistics Estonia's own typo, in RL21452
    "Pagan": "paganism",
    "Wiccan": "paganism",
    "Animist": "paganism",

    # ---------------------------------------------------------------- esoteric
    "Anthroposophist": "esoteric",
    "Theosophist": "esoteric",
    "New Ager": "esoteric",
    "Pantheist": "esoteric",
    "Satanist": "esoteric",

    # ---------------------------------------------------------------- no religion
    "Does not feel an affiliation to any religion": "unaffiliated",

    # ---------------------------------------------------------------- residual
    "Other religion": "other.ee",
    "Religion unknown": "other.ee",
}


def _key(cat):
    return " ".join(str(cat).split())


def resolve(cat):
    """Source category -> taxonomy node id, or None if deliberately not on the tree."""
    c = _key(cat)
    if c in EXCLUDED:
        return None
    return MAP.get(c)
