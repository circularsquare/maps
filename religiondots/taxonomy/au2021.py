"""
ABS Census 2021 religious affiliation (ASCRG) -> religiondots taxonomy.

**Branch-level mapping, like the others.** No leaves created (spec §2.4). 148 categories,
the deepest list in the project after ASARB's 372, and the one that pushed hardest on the
tree: `christianity.churchofeast`, `mandaeism`, `yazidism`, `caodaism`,
`indigenous.australian`, `christianity.maori` and `christianity.united` all exist because
Australia counts them and nothing before it did.

ABS publishes 34 categories at SA2 and 150 nationally, so most of this arrives already
allocated (§3.9) with `tier=derived`; countries.py turns that into `may_ring=False` (§3.10).

THE ONE ABS GETS RIGHT THAT ALMOST NOBODY DOES. Its groups 221, 222 and 223 are Oriental
Orthodox, Assyrian Apostolic and Eastern Orthodox — three separate communions, kept apart.
The Assyrian Church of the East split in 431, twenty years before Chalcedon divided the
other two, and it is neither. Ireland's CSO, by contrast, publishes one row reading
"Orthodox (Greek, Coptic, Russian)", which welds Eastern and Oriental together and cannot
be undone (see ie2022.py).

The Eastern Orthodox bodies map to `christianity.orthodox` rather than to
`christianity.orthodox.canonical`, unlike Czechia's. ABS's group includes the Macedonian
Orthodox Church, whose autocephaly was not recognised until 2022 — after this census — so
asserting canonicity across the whole group would be wrong for at least one member, and the
parent node says only what is certain.
"""

EXCLUDED = {}   # ABS's "not stated" is not in the ingested table; see sources/au.md

REVIEW = {
    "Uniting Church":
        "-> christianity.united. 673,383, Australia's third largest Christian body, formed "
        "1977 from Methodists, Presbyterians and Congregationalists. Methodism was the "
        "largest strand and the conventional filing, but calling it Methodist erases the "
        "other two, so it gets the union node instead.",
    "Brethren":
        "-> christianity.plymouth. In Australian usage 'Brethren' at this size is the "
        "Plymouth/Exclusive line, not the Schwarzenau (German Baptist) Brethren, which "
        "branches.py warns are unrelated. 17,946.",
    "Christian Community Churches of Australia":
        "-> christianity.plymouth. The Open Brethren body in Australia.",
    "Revival Fellowship":
        "-> pentecostal.oneness. Baptises in Jesus' name and holds the Oneness position; "
        "same for Revival Centres, from which it split.",
    "Revival Centres": "-> pentecostal.oneness. See Revival Fellowship.",
    "Own Spiritual Beliefs":
        "-> unchurched. 27,328. ABS files it with the secular and no-religion answers, but "
        "the answer asserts a spiritual belief and no affiliation, which is what the node "
        "is for. `Theism` (5,411) is the same call and clearer.",
    "Multi Faith":
        "-> other.au. spec §3.3 gives a syncretism its own node, but this is the "
        "declaration of multiple belonging rather than one tradition, so it stays residual "
        "— the same call as Brazil's 'Declaracao de multipla religiosidade'.",
    "Ancestor Veneration":
        "-> chinesefolk. 409. In Australia this is overwhelmingly Chinese and Vietnamese "
        "practice, which is what the node holds; it would be wrong for a source where the "
        "same words meant an African or Melanesian tradition.",
    "Ratana (Maori)":
        "-> christianity.maori. Christian in theology, founded by a Maori prophet, and "
        "belonging to none of the missionary denominations.",
    "Christadelphians":
        "-> christianity.other. Nontrinitarian and restorationist in the broad sense, but "
        "not Stone-Campbell, so `christianity.restorationist` would be wrong.",
    "Gnostic Christians":
        "-> christianity.other rather than `esoteric`. The self-description is Christian; "
        "the Mandaeans, who are actually Gnostic, are counted separately and get their own "
        "family.",
    "Grace Communion International (Worldwide Church of God)":
        "-> christianity.protestant. Left its nontrinitarian past in the 1990s and is now "
        "conventionally evangelical; filed where it is now, not where it came from.",
    "Temple Society":
        "-> christianity.pietist. The Templers, a 19th-century German pietist body with a "
        "substantial Australian community.",
}

_ORTHODOX = "christianity.orthodox"
_ORIENTAL = "christianity.oriental"
_EAST = "christianity.churchofeast"
_ECATH = "christianity.catholic.eastern"
_PENT_T = "christianity.pentecostal.trinitarian"
_PENT_O = "christianity.pentecostal.oneness"
_PROT = "christianity.protestant"
_PAGAN = "paganism"

MAP = {
    # ---------------------------------------------------------------- Christian, unspecified
    "Christianity, nfd": "christianity",

    # ---------------------------------------------------------------- Anglican (201)
    "Anglican Church of Australia": "christianity.anglican",
    "Anglican, nec": "christianity.anglican",
    "Anglican Catholic Church": "christianity.anglican.continuing",

    # ---------------------------------------------------------------- Catholic (207)
    "Western Catholic": "christianity.catholic.latin",
    "Maronite Catholic": _ECATH,
    "Chaldean Catholic": _ECATH,
    "Syro Malabar Catholic": _ECATH,
    "Melkite Catholic": _ECATH,
    "Ukrainian Catholic": _ECATH,
    "Catholic, nec": "christianity.catholic",
    "Catholic, nfd": "christianity.catholic",
    "Liberal Catholic Church": "christianity.catholic.independent",

    # ---------------------------------------------------------------- Orthodox (221/222/223)
    "Greek Orthodox": _ORTHODOX,
    "Macedonian Orthodox": _ORTHODOX,
    "Serbian Orthodox": _ORTHODOX,
    "Russian Orthodox": _ORTHODOX,
    "Antiochian Orthodox": _ORTHODOX,
    "Ukrainian Orthodox": _ORTHODOX,
    "Romanian Orthodox": _ORTHODOX,
    "Albanian Orthodox": _ORTHODOX,
    "Eastern Orthodox, nfd": _ORTHODOX,
    "Eastern Orthodox, nec": _ORTHODOX,
    "Coptic Orthodox Church": _ORIENTAL,
    "Syrian Orthodox Church": _ORIENTAL,
    "Armenian Apostolic": _ORIENTAL,
    "Ethiopian Orthodox Church": _ORIENTAL,
    "Oriental Orthodox, nec": _ORIENTAL,
    "Oriental Orthodox, nfd": _ORIENTAL,
    "Assyrian Church of the East": _EAST,
    "Ancient Church of the East": _EAST,
    "Assyrian Apostolic, nfd": _EAST,
    "Assyrian Apostolic, nec": _EAST,

    # ---------------------------------------------------------------- Reformed (225)
    "Presbyterian": "christianity.reformed.presbyterian",
    "Reformed": "christianity.reformed.continental",
    "Free Reformed": "christianity.reformed.continental",
    "Presbyterian and Reformed, nfd": "christianity.reformed",
    "Congregational": "christianity.reformed.congregational",

    # ---------------------------------------------------------------- other Christian families
    "Lutheran": "christianity.lutheran",
    "Baptist": "christianity.baptist",
    "Brethren": "christianity.plymouth",
    "Christian Community Churches of Australia": "christianity.plymouth",
    "Uniting Church": "christianity.united",
    "Methodist, so described": "christianity.methodist",
    "United Methodist Church": "christianity.methodist",
    "Wesleyan Methodist Church": "christianity.holiness",
    "Church of the Nazarene": "christianity.holiness",
    "Christian and Missionary Alliance": "christianity.holiness",
    "Salvation Army": "christianity.holiness",
    "Seventh-day Adventist": "christianity.adventist",
    "Jehovah's Witnesses": "christianity.witnesses",
    "Religious Society of Friends (Quakers)": "christianity.friends",
    "Christian Science": "christianity.christianscience",
    "New Churches (Swedenborgian)": "christianity.swedenborgian",
    "Religious Science": "newthought",
    "Temple Society": "christianity.pietist",

    # ---------------------------------------------------------------- Latter Day Saints (215)
    "The Church of Jesus Christ of Latter-day Saints": "christianity.latterday",
    "Community of Christ": "christianity.latterday",
    "Latter-day Saints, nfd": "christianity.latterday",

    # ---------------------------------------------------------------- Stone-Campbell (211)
    "Churches of Christ (Conference)": "christianity.restorationist",
    "Churches of Christ, nfd": "christianity.restorationist",
    "Church of Christ (Non-denominational)": "christianity.restorationist",
    "International Church of Christ": "christianity.restorationist",

    # ---------------------------------------------------------------- Pentecostal (24)
    "Pentecostal, nfd": "christianity.pentecostal",
    "Pentecostal, nec": "christianity.pentecostal",
    "Australian Christian Churches (Assemblies of God)": _PENT_T,
    "International Network of Churches (Christian Outreach Centres)": _PENT_T,
    "CRC International (Christian Revival Crusade)": _PENT_T,
    "Apostolic Church (Australia)": _PENT_T,
    "C3 Church Global (Christian City Church)": _PENT_T,
    "Foursquare Gospel Church": _PENT_T,
    "Full Gospel Church of Australia (Full Gospel Church)": _PENT_T,
    "Worship Centre Network": _PENT_T,
    "Acts 2 Alliance": _PENT_T,
    "Rhema Family Church": _PENT_T,
    "Bethesda Ministries International (Bethesda Churches)": _PENT_T,
    "Victory Life Centre": _PENT_T,
    "Christian Church in Australia": _PENT_T,
    "Pentecostal City Life Church": _PENT_T,
    "Victory Worship Centre": _PENT_T,
    "United Pentecostal": _PENT_O,
    "Revival Fellowship": _PENT_O,
    "Revival Centres": _PENT_O,

    # ---------------------------------------------------------------- other Protestant (28)
    "Other Protestant, nfd": _PROT,
    "Other Protestant, nec": _PROT,
    "Born Again Christian": _PROT,
    "Ethnic Evangelical Churches": _PROT,
    "Aboriginal Evangelical Missions": _PROT,
    "Grace Communion International (Worldwide Church of God)": _PROT,
    "Independent Evangelical Churches": "christianity.nondenominational",

    # ---------------------------------------------------------------- other Christian (29)
    "Christadelphians": "christianity.other",
    "Other Christian, nec": "christianity.other",
    "Other Christian, nfd": "christianity.other",
    "Apostolic Church of Queensland": "christianity.other",
    "New Apostolic Church": "christianity.other",
    "Gnostic Christians": "christianity.other",
    "Ratana (Maori)": "christianity.maori",

    # ---------------------------------------------------------------- world religions
    "Buddhism": "buddhism",
    "Hinduism": "hinduism",
    "Islam": "islam",
    "Judaism": "judaism",
    "Sikhism": "sikhism",
    "Jainism": "jainism",
    "Zoroastrianism": "zoroastrianism",
    "Baha'i": "bahai",
    "Druse": "druze",
    "Mandaean": "mandaeism",
    "Yezidi": "yazidism",
    "Caodaism": "caodaism",
    "Rastafari": "rastafari",
    "Shinto": "shinto",
    "Taoism": "daoism",
    "Church of Scientology": "scientology",
    "Unitarian Universalism": "unitarianuniversalist",
    "Spiritualism": "spiritualism",

    # ---------------------------------------------------------------- Chinese / Japanese
    "Confucianism": "chinesefolk",
    "Ancestor Veneration": "chinesefolk",
    "Chinese Religions, nec": "chinesefolk",
    "Chinese Religions, nfd": "chinesefolk",
    "Sukyo Mahikari": "japanesenew",
    "Tenrikyo": "japanesenew",
    "Japanese Religions, nec": "japanesenew",
    "Japanese Religions, nfd": "japanesenew",

    # ---------------------------------------------------------------- indigenous
    "Australian Aboriginal Traditional Religions": "indigenous.australian",

    # ---------------------------------------------------------------- pagan / esoteric
    "Paganism": _PAGAN,
    "Wiccan (Witchcraft)": _PAGAN,
    "Druidism": _PAGAN,
    "Animism": _PAGAN,
    "Nature Religions, nec": _PAGAN,
    "Nature Religions, nfd": _PAGAN,
    "Satanism": "esoteric",
    "Theosophy": "esoteric",
    "Eckankar": "esoteric",
    "New Age": "esoteric",

    # ---------------------------------------------------------------- no religion (71/72)
    "No Religion, so described": "unaffiliated",
    "Atheism": "secular",
    "Agnosticism": "secular",
    "Humanism": "secular",
    "Rationalism": "secular",
    "Secular Beliefs, nec": "secular",
    "Secular Beliefs, nfd": "secular",

    # ---------------------------------------------------------------- spiritual, unaffiliated (73)
    "Own Spiritual Beliefs": "unchurched",
    "Theism": "unchurched",
    "Other Spiritual Beliefs, nec": "other.au",
    "Other Spiritual Beliefs, nfd": "other.au",
    "Multi Faith": "other.au",
    "Religious Groups, nec": "other.au",
    "Secular Beliefs and Other Spiritual Beliefs and No Religious Affiliation, nfd": "other.au",
}


def resolve(category):
    """religiondots branch for an ABS category, or None if deliberately off the tree."""
    if category in EXCLUDED:
        return None
    return MAP.get(category)
