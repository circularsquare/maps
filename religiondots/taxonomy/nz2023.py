"""
Stats NZ religious affiliation (2018 level-3 structure on 2023 SA2 totals) -> taxonomy.

**Branch-level mapping, like the others.** No leaves created (spec §2.4).

TWO THINGS ABOUT THIS SOURCE THAT NOTHING ELSE ON THE MAP HAS.

1. **The structure and the totals are five years apart.** Stats NZ published 166 level-3
   categories nationally in 2018 and 13 level-1 columns by SA2 in 2023, so `allocate.py`
   splits the 2023 columns by the 2018 shares — spec §3.4, the same trade as Brazil, but
   inside one country's own classification rather than across two censuses of different
   depth. The consequence shows up in the reconciliation: New Zealand's Hindu column is
   -14.7% and its Muslim column -18.2%, because both grew that much between 2018 and 2023.
   That is immigration, not a mis-assignment, and it is why `allocate.py` grew a
   `--tolerance` flag.

2. **The counts are RESPONSES, not people — up to four per person.** Summing every category
   gives 5,003,112 against an SA2 population of 4,993,920, so about 9,000 people answered
   more than once and are drawn more than once. 0.19%, but it means a New Zealand dot is a
   response and every other country's dot is a person.

Stats NZ also tabulates **political ideologies** as religious affiliations — Socialism,
Marxism, Maoism, Libertarianism — because people wrote them in. They are not religions and
not secular positions either; they go to the per-source residual, which is what its label
("Other Religions, Beliefs and Philosophies") already admits.

And Jedi is 22,605 here, second only to Sikhism among the "other religions" and larger than
Baha'i, Jainism, Taoism and Zoroastrianism combined. Church of the Flying Spaghetti Monster
is 4,705. Both go to `parody`, the node Czechia created.
"""

EXCLUDED = {
    "Object to answering":
        "342,723 people, 6.9% of New Zealand, who declined the question. Not a religion and "
        "not 'no religion', which is separately 2,575,989. Excluded from the dots, as "
        "Czechia's 'Neuvedeno', Ireland's 'Not stated' and Brazil's 'Nao sabe' are. New "
        "Zealand's is the second highest refusal rate on the map after Czechia's 30%.",
}

REVIEW = {
    "Assyrian Orthodox":
        "-> christianity.churchofeast. The name is ambiguous — it could mean the Assyrian "
        "Church of the East or the Syriac Orthodox Church, which are different communions. "
        "Read as the Church of the East because that is what 'Assyrian' names in every "
        "other classification, including the ABS's group 222. 304 people; low confidence.",
    "Church of God":
        "-> pentecostal.trinitarian. A bare name several unrelated bodies use; usrc2020 "
        "makes the same call for ASARB's F53 with the same caveat, and consistency across "
        "sources matters more here than being right about which one New Zealand means.",
    "Chinese Christian":
        "-> christianity, not christianity.protestant. An ethnic congregation label that "
        "names no tradition. Stats NZ lists 'Chinese Presbyterian' and 'Korean "
        "Presbyterian' separately, so this is the remainder after those, and it is probably "
        "mostly Protestant — but probably is not good enough to assert.",
    "Korean Christian": "-> christianity. Same reasoning as Chinese Christian.",
    "Theism":
        "-> unchurched. Belief in God with no affiliation named, the same answer as "
        "Australia's `Theism` and Czechia's 'věřící - nehlásící se k žádné církvi'.",
    "Pantheist":
        "-> paganism. Stats NZ files it under Spiritualism and New Age; Ireland's CSO pairs "
        "it with Pagan in a single category, which is the precedent followed here.",
    "Falun Gong":
        "-> chinesefolk. A qigong movement out of the Buddhist and Daoist traditions, and "
        "not comfortably any of the three. 116 people.",
    "Socialism":
        "-> other.nz, with Marxism, Maoism and Libertarianism. Political ideologies written "
        "in as religion. Not `secular`, which branches.py defines as organised non-theistic "
        "bodies and stated secular POSITIONS — a political programme is neither. 47 people "
        "across the four.",
    "Māori Religions, Beliefs and Philosophies nfd":
        "-> indigenous.maori, while Ratana, Ringatu and Paimarire go to christianity.maori. "
        "The named churches are Christian in theology; the nfd/nec remainder is not "
        "necessarily, and traditional Maori religion is not a Christian church.",
    "Worldwide Church of God":
        "-> christianity.protestant. Same body and same call as Australia's Grace Communion "
        "International: nontrinitarian until the 1990s, conventionally evangelical since.",
}

_CATH = "christianity.catholic"
_ECATH = "christianity.catholic.eastern"
_ORTH = "christianity.orthodox"
_PENT = "christianity.pentecostal"
_PENT_T = "christianity.pentecostal.trinitarian"
_PENT_O = "christianity.pentecostal.oneness"
_PENT_C = "christianity.pentecostal.charismatic"
_PROT = "christianity.protestant"
_BAPT = "christianity.baptist"
_BRETH = "christianity.plymouth"
_METH = "christianity.methodist"
_CONG = "christianity.reformed.congregational"
_PRES = "christianity.reformed.presbyterian"
_HOLI = "christianity.holiness"
_REST = "christianity.restorationist"
_ADV = "christianity.adventist"
_OTHERC = "christianity.other"
_PAGAN = "paganism"

MAP = {
    # ---------------------------------------------------------------- Christian, unspecified
    "Christian nfd": "christianity",
    "Christian nec": "christianity",
    "Jesus Follower": "christianity",
    "Ecumenical": "christianity",
    "Chinese Christian": "christianity",
    "Korean Christian": "christianity",

    # ---------------------------------------------------------------- Catholic
    "Roman Catholic": "christianity.catholic.latin",
    "Catholicism nfd": _CATH,
    "Catholicism nec": _CATH,
    "Liberal Catholic": "christianity.catholic.independent",
    "Maronite Catholic": _ECATH,
    "Chaldean Catholic": _ECATH,
    "Syro-Malabar Catholic": _ECATH,
    "Ukrainian Catholic": _ECATH,
    "Melkite Catholic": _ECATH,

    # ---------------------------------------------------------------- Orthodox
    "Orthodox nfd": _ORTH,
    "Orthodox nec": _ORTH,
    "Greek Orthodox": _ORTH,
    "Russian Orthodox": _ORTH,
    "Serbian Orthodox": _ORTH,
    "Coptic Orthodox": "christianity.oriental",
    "Assyrian Orthodox": "christianity.churchofeast",

    # ---------------------------------------------------------------- Anglican / Reformed
    "Anglican": "christianity.anglican",
    "Presbyterian": _PRES,
    "Korean Presbyterian": _PRES,
    "Chinese Presbyterian": _PRES,
    "Reformed": "christianity.reformed.continental",
    "Congregational": _CONG,
    "Samoan Congregational": _CONG,
    "Cook Island Congregational": _CONG,
    "Uniting/Union Church": "christianity.united",
    "Lutheran": "christianity.lutheran",

    # ---------------------------------------------------------------- Methodist / holiness
    "Methodist nfd": _METH,
    "Methodist nec": _METH,
    "Tongan Methodist": _METH,
    "Wesleyan Methodist": _HOLI,
    "Nazarene": _HOLI,
    "Salvation Army": _HOLI,
    "Christian and Missionary Alliance": _HOLI,

    # ---------------------------------------------------------------- Baptist / Brethren
    "Baptist nfd": _BAPT,
    "Baptist nec": _BAPT,
    "Independent Baptist": _BAPT,
    "Reformed Baptist": _BAPT,
    "Bible Baptist": _BAPT,
    "Plymouth or Exclusive Brethren": _BRETH,
    "Open Brethren": _BRETH,
    "Brethren nfd": _BRETH,
    "Brethren nec": _BRETH,

    # ---------------------------------------------------------------- Stone-Campbell
    "Church of Christ nfd": _REST,
    "Associated Churches of Christ": _REST,
    "Other Church of Christ and Churches of Christ nec": _REST,

    # ---------------------------------------------------------------- Pentecostal
    "Pentecostal nfd": _PENT,
    "Pentecostal nec": _PENT,
    "Independent Pentecostal": _PENT,
    "Assemblies of God": _PENT_T,
    "ACTS Churches": _PENT_T,
    "Elim": _PENT_T,
    "Full Gospel": _PENT_T,
    "Christian Revival Crusade": _PENT_T,
    "Church of God": _PENT_T,
    "United Pentecostal": _PENT_O,
    "Revival Centres": _PENT_O,
    "New Life": _PENT_C,
    "Vineyard Christian Fellowship": _PENT_C,
    "Destiny Church": _PENT_C,
    "Arise Church": _PENT_C,
    "City Impact Church": _PENT_C,
    "Equippers Church": _PENT_C,
    "Christian Outreach": _PENT_C,

    # ---------------------------------------------------------------- other Protestant
    "Protestant nfd": _PROT,
    "Born Again": _PROT,
    "Evangelical": _PROT,
    "Fundamentalist": _PROT,
    "Worldwide Church of God": _PROT,
    "Christian Fellowship": "christianity.nondenominational",
    "Independent Evangelical Churches": "christianity.nondenominational",

    # ---------------------------------------------------------------- other Christian
    "Latter-day Saints": "christianity.latterday",
    "Jehovah's Witnesses": "christianity.witnesses",
    "Seventh Day Adventist": _ADV,
    "Adventist nfd": _ADV,
    "Adventist nec": _ADV,
    "Christian Science": "christianity.christianscience",
    "Religious Society of Friends (Quaker)": "christianity.friends",
    "Unitarian": "unitarianuniversalist",
    "Christadelphian": _OTHERC,
    "Metropolitan Community Church": _OTHERC,
    "Commonwealth Covenant Church": _OTHERC,

    # ---------------------------------------------------------------- Maori
    "Ratana": "christianity.maori",
    "Ringatū": "christianity.maori",
    "Paimarire": "christianity.maori",
    "Māori Religions, Beliefs and Philosophies nfd": "indigenous.maori",
    "Māori Religions, Beliefs and Philosophies nec": "indigenous.maori",

    # ---------------------------------------------------------------- Buddhism
    "Buddhism nfd": "buddhism",
    "Buddhism nec": "buddhism",
    "Theravada Buddhism": "buddhism",
    "Mahayana Buddhism": "buddhism",
    "Zen Buddhism": "buddhism",
    "Nichiren Buddhism": "buddhism",
    "Vajrayana Buddhism": "buddhism",

    # ---------------------------------------------------------------- Hinduism
    "Hinduism nfd": "hinduism",
    "Hinduism nec": "hinduism",
    "Hare Krishna": "hinduism",
    "Yoga": "hinduism",
    "Arya Samaj": "hinduism",

    # ---------------------------------------------------------------- Islam
    "Islam nfd": "islam",
    "Islam nec": "islam",
    "Sunni": "islam",
    "Shi'a": "islam",
    "Ahmadiyya Muslim": "islam",
    "Sufi": "islam",

    # ---------------------------------------------------------------- Judaism
    "Judaism nfd": "judaism",
    "Reformed Judaism": "judaism",
    "Orthodox Judaism": "judaism",
    "Conservative Judaism": "judaism",

    # ---------------------------------------------------------------- other world religions
    "Sikhism": "sikhism",
    "Baha'i": "bahai",
    "Jainism": "jainism",
    "Zoroastrian": "zoroastrianism",
    "Taoism": "daoism",
    "Shinto": "shinto",
    "Cao Dai": "caodaism",
    "Rastafarianism": "rastafari",
    "Unification Church (Moonist)": "unification",
    "Church of Scientology": "scientology",

    # ---------------------------------------------------------------- Chinese / Japanese
    "Confucianism": "chinesefolk",
    "Chinese Religions nfd": "chinesefolk",
    "Chinese Religions nec": "chinesefolk",
    "Falun Gong": "chinesefolk",
    "Mahikari": "japanesenew",
    "Tenrikyo": "japanesenew",
    "Japanese Religion nfd": "japanesenew",
    "Japanese Religion nec": "japanesenew",

    # ---------------------------------------------------------------- pagan / esoteric
    "Pagan": _PAGAN,
    "Wiccan": _PAGAN,
    "Druid": _PAGAN,
    "Animist": _PAGAN,
    "Pantheist": _PAGAN,
    "Nature and Earth Based Religions nfd": _PAGAN,
    "Nature and Earth Based Religions nec": _PAGAN,
    "Spiritualist": "spiritualism",
    "Satanism": "esoteric",
    "New Age nfd": "esoteric",
    "Other New Age Religions nec": "esoteric",

    # ---------------------------------------------------------------- no religion / secular
    "No Religion": "unaffiliated",
    "Atheism": "secular",
    "Agnosticism": "secular",
    "Humanism": "secular",
    "Rationalism": "secular",
    "Deism": "secular",
    "Theism": "unchurched",

    # ---------------------------------------------------------------- parody
    "Jedi": "parody",
    "Church of the Flying Spaghetti Monster": "parody",

    # ---------------------------------------------------------------- residual
    "Other Religions, Beliefs and Philosophies nfd": "other.nz",
    "Other Religions, Beliefs and Philosophies nec": "other.nz",
    "Socialism": "other.nz",
    "Marxism": "other.nz",
    "Maoism": "other.nz",
    "Libertarianism": "other.nz",
}


def resolve(category):
    """religiondots branch for a Stats NZ category, or None if deliberately off the tree."""
    if category in EXCLUDED:
        return None
    return MAP.get(category)
