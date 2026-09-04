"""
The United Kingdom's THREE religion classifications -> religiondots taxonomy.

**Branch-level mapping, like the others.** No leaves created (spec §2.4).

The UK has no census; it has four, from three agencies, with different dates, wording and
category lists (sources/uk.md). One mapping file covers the three that are drawn, because
they land on one tree and the category names do not collide:

    ONS, England and Wales, 21 Mar 2021   56 categories at Output Area (after allocation)
    NRS, Scotland,          20 Mar 2022   13 categories at Output Area, no allocation
    NISRA, Northern Ireland, 21 Mar 2021  32 categories at Data Zone (after allocation)

NISRA's second question — religion brought up in, MS-B23/B24 — is a different variable and
is not mapped or drawn at all.

THE THING TO UNDERSTAND ABOUT THIS SOURCE. **England and Wales publish no Christian
denomination, at any geography, for 27.5 million people.** ONS asks one tick-box question
and the write-in box is only reached by people who tick "Any other religion", so the 50
write-in categories below are all *outside* Christianity — Pagan, Alevi, Jain, Ravidassia,
Yazidi, Vodun, Thelemite — while `Christian` stays one undifferentiated 27.5M node. This is
spec §3.11's irreducible floor, and it is the largest single unresolved block on the map.

The other two do better and differently. Scotland names the Church of Scotland and Roman
Catholics and stops. Northern Ireland, where the denomination is the political fact, names
twenty-two Christian bodies including four kinds of Presbyterian — and NISRA is the only
agency on this map that publishes `Mixed Catholic / Protestant` as a category, which is a
statement about Northern Ireland rather than about religion.
"""

EXCLUDED = {
    "All people":
        "Scotland's population total, not a category.",
    "Religion not stated":
        "Scotland 334,740 and Northern Ireland 30,504. Not a religion and not 'no "
        "religion', which both publish separately. Excluded, as everywhere else. England "
        "and Wales' equivalent ('Not answered', 3,596,019) never reaches this file — TS030 "
        "and TS031 give it different names, so allocate.py could not match the two and "
        "dropped it, which is the outcome this would have chosen anyway.",
}

REVIEW = {
    "Other religion: Alevi":
        "-> alevism, not islam. 25,657 in England and Wales, mostly Kurdish and Turkish. "
        "Usually filed as a branch of Shi'a Islam and many Alevis reject that placement; "
        "given its own family for the same reason as Druze.",
    "Other religion: Ravidassia":
        "-> ravidassia, not sikhism. Declared itself a separate religion in 2010 after the "
        "murder of Sant Ramanand; before that it was counted Sikh, and many still are. "
        "9,583 people, almost all in the West Midlands and Punjab-origin.",
    "Other religion: Valmiki":
        "-> hinduism. A Punjabi Dalit tradition that is variously counted Hindu, Sikh or "
        "its own; filed Hindu because that is where most UK Valmiki temples affiliate. "
        "1,031 people, low confidence.",
    "Other religion: Shamanism":
        "-> paganism. A Western neo-shamanic self-description rather than an indigenous "
        "practice, the same call as Czechia's 'animismus'. 7,912 people.",
    "Other religion: Traditional African Religion":
        "-> indigenous, the family rather than a child, because ONS names no tradition. "
        "Distinct from `afrodiasporic`, which is where Vodun goes: one is the religion in "
        "Africa, the other is what the diaspora made of it.",
    "Other religion: Spiritual":
        "-> unchurched. 31,680 people who ticked 'any other religion' and wrote "
        "'spiritual' — asserting a belief and no affiliation, which is the node. `Believe "
        "in God` (2,406) and `Theism` (848) are the same answer.",
    "Other religion: Universalist":
        "-> unitarianuniversalist. The bare word could mean Christian universalism instead; "
        "761 people, low confidence.",
    "Christian: Mixed Catholic / Protestant":
        "-> other.uk. NISRA publishes this and nobody else does. It is not a denomination "
        "and not a syncretism — it is a statement about mixed families in Northern Ireland "
        "— so it goes to the residual rather than being forced into either side. 1,689.",
    "Church of Scotland":
        "-> reformed.presbyterian. The national church of Scotland is Presbyterian in "
        "polity and doctrine; 1,107,708 people, the largest single denomination NRS names.",
    "Christian: Free Presbyterian":
        "-> reformed.presbyterian, with Presbyterian Church in Ireland, Reformed "
        "Presbyterian and Non-Subscribing Presbyterian. Four Presbyterian bodies in one "
        "small country, and NISRA counts them separately because the differences matter "
        "there; at branch level they land together.",
}

_PAGAN = "paganism"
_ESO = "esoteric"
_PRES = "christianity.reformed.presbyterian"
_PROT = "christianity.protestant"

MAP = {
    # ================================================================ England and Wales
    "Christian": "christianity",
    "Muslim": "islam",
    "Hindu": "hinduism",
    "Sikh": "sikhism",
    "Buddhist": "buddhism",
    "Jewish": "judaism",

    # --- the write-in tail, all of it outside Christianity
    "No religion: No religion": "unaffiliated",
    "No religion: Agnostic": "secular",
    "No religion: Atheist": "secular",
    "No religion: Humanist": "secular",
    "No religion: Free Thinker": "secular",
    "No religion: Realist": "secular",
    "Other religion: Deist": "secular",

    "Other religion: Spiritual": "unchurched",
    "Other religion: Believe in God": "unchurched",
    "Other religion: Theism": "unchurched",

    "Other religion: Pagan": _PAGAN,
    "Other religion: Wicca": _PAGAN,
    "Other religion: Witchcraft": _PAGAN,
    "Other religion: Druid": _PAGAN,
    "Other religion: Heathen": _PAGAN,
    "Other religion: Shamanism": _PAGAN,
    "Other religion: Animism": _PAGAN,
    "Other religion: Pantheism": _PAGAN,
    "Other religion: Reconstructionist": _PAGAN,

    "Other religion: Spiritualist": "spiritualism",
    "Other religion: Satanism": _ESO,
    "Other religion: Occult": _ESO,
    "Other religion: Mysticism": _ESO,
    "Other religion: New Age": _ESO,
    "Other religion: Thelemite": _ESO,
    "Other religion: Eckankar": _ESO,

    "Other religion: Alevi": "alevism",
    "Other religion: Ravidassia": "ravidassia",
    "Other religion: Valmiki": "hinduism",
    "Other religion: Brahma Kumari": "hinduism",
    "Other religion: Jain": "jainism",
    "Other religion: Baha'i": "bahai",
    "Other religion: Zoroastrian": "zoroastrianism",
    "Other religion: Druze": "druze",
    "Other religion: Yazidi": "yazidism",
    "Other religion: Rastafarian": "rastafari",
    "Other religion: Taoist": "daoism",
    "Other religion: Shintoism": "shinto",
    "Other religion: Confucianist": "chinesefolk",
    "Other religion: Chinese Religion": "chinesefolk",
    "Other religion: Scientology": "scientology",
    "Other religion: Unification Church": "unification",
    "Other religion: Universalist": "unitarianuniversalist",
    "Other religion: Vodun": "afrodiasporic",
    "Other religion: Traditional African Religion": "indigenous",
    "Other religion: Native American Church": "indigenous.northamerican",

    "Other religion: Other religions": "other.uk",
    "Other religion: Mixed Religion": "other.uk",
    "Other religion: Own Belief System": "other.uk",
    "Other religion: Church of All Religion": "other.uk",

    # ================================================================ Scotland
    "No religion": "unaffiliated",
    "Church of Scotland": _PRES,
    "Roman Catholic": "christianity.catholic.latin",
    "Other Christian": "christianity",
    "Pagan": _PAGAN,
    "Other religion": "other.uk",

    # ================================================================ Northern Ireland
    "Christian: Catholic": "christianity.catholic.latin",
    "Christian: Church of Ireland": "christianity.anglican",
    "Christian: Church of England": "christianity.anglican",
    "Christian: Presbyterian Church in Ireland": _PRES,
    "Christian: Free Presbyterian": _PRES,
    "Christian: Presbyterian": _PRES,
    "Christian: Reformed Presbyterian": _PRES,
    "Christian: Non-Subscribing Presbyterian": _PRES,
    "Christian: Methodist Church in Ireland": "christianity.methodist",
    "Christian: Independent Methodist": "christianity.methodist",
    "Christian: Congregational Church": "christianity.reformed.congregational",
    "Christian: Baptist": "christianity.baptist",
    "Christian: Brethren": "christianity.plymouth",
    "Christian: Pentecostal": "christianity.pentecostal",
    "Christian: Orthodox Church": "christianity.orthodox",
    "Christian: Romanian Orthodox Church": "christianity.orthodox",
    "Christian: Jehovah’s Witness": "christianity.witnesses",
    "Christian: Church of Jesus Christ of Latter Day Saints (Mormons)":
        "christianity.latterday",
    "Christian: Non Denominational": "christianity.nondenominational",
    "Christian: Christian Fellowship Church": "christianity.nondenominational",
    "Christian: Protestant": _PROT,
    "Christian: Protestant (Mixed)": _PROT,
    "Christian: Evangelical": _PROT,
    "Christian: Christian": "christianity",
    "Christian: Other Christian denominations": "christianity.other",
    "Christian: Mixed Catholic / Protestant": "other.uk",
    "Other Religions: Muslim": "islam",
    "Other Religions: Hindu": "hinduism",
    "Other Religions: Buddhist": "buddhism",
    "Other Religions: Other Religions": "other.uk",
}


def resolve(category):
    """religiondots branch for a UK category, or None if deliberately off the tree."""
    if category in EXCLUDED:
        return None
    return MAP.get(category)
