"""NSO 2010 census religion (Table 4) -> religiondots taxonomy.

Nine categories plus the universe total. **The category list is a world-religions list and
its one interesting feature is what Thailand's census does NOT ask**: there is no cell for a
Buddhist school, none for a Christian denomination, and none for the animist practice of the
hill peoples, who therefore arrive as `อื่น ๆ` or as `ไม่มีศาสนา` depending on how they
answered.

**THE CATEGORIES ARE MEASURED AT REGION AND THE GEOGRAPHY AT PROVINCE, AND THEY MEET IN
`allocate.py`** — spec §3.10. Only `พุทธ` and `อิสลาม` are published per province; the other
seven are a province residual split by its own region's composition. So a Buddhist or Muslim
row is `measured` and everything else is `derived`, which `countries.py::_th_counts` sets from
the presence of allocate.py's `parent_column=` note rather than from a list here.

**Confucianism is the row worth noticing**, and it is the reason this file is not just
Cambodia's with more rows: Thailand is the only census on this map that offers `ขงจื้อ` as its
own cell, so the tree's `confucianism` node finally gets a country that counted it.

EXCLUDED holds categories deliberately not on the tree.
REVIEW holds calls that are defensible but arguable, with the reason.
"""

EXCLUDED = {
    "ยอดรวม":
        "the unit's own population total, not a category.",
    "ไม่ทราบ":
        "`Unknown` — non-response, and EXCLUDED per spec §3.5, which marks undercounting "
        "rather than filling it. 8,255 people nationally, 0.013%, and its geography is an "
        "artefact of enumeration rather than of belief: Bangkok alone holds 1,053 of them "
        "and the North 75. This is the convention every source here follows — Georgia's "
        "`Not stated`, Barbados's `Not Stated`, Belize's `Don't Know/Not Stated` are all "
        "excluded the same way. **Not `unknown`**, the §6.3a-ii node: that one is for a "
        "source that counted people and could not establish what they practise AT ANY "
        "GEOGRAPHY, which is Vietnam's 81.8% and China's 97.5%. A 0.013% non-response cell "
        "is the opposite thing and folding it in would cheapen the node.",
    # The three province-level categories sources/th.py writes. They are inputs to
    # allocate.py and never survive it -- the allocated file carries the nine region
    # categories at province geography -- but check_mapping.py reads th.csv itself and
    # would otherwise report them unmapped.
    "อื่น ๆ ไม่มีศาสนา และไม่ทราบ":
        "the province residual (population minus Buddhist minus Muslim) that allocate.py "
        "splits into the seven small categories. Never drawn as itself.",
}

REVIEW = {
    "พุทธ": (
        "-> buddhism, the PARENT, and deliberately not buddhism.theravada. 61.7M people, "
        "93.6% of the country. Thai Buddhism is Theravada in overwhelming proportion and "
        "the Maha Nikaya and Dhammayuttika orders are both Theravada, so filing it there "
        "would almost certainly be true. It is still not what the source says: Table 4 "
        "offers one cell labelled `พุทธ` and asks for a religion rather than a school. "
        "**This is lk2024.py's call and kh2019.py's, made a third time on the same "
        "tradition in the third country running**, and in2011.py makes it a fourth on "
        "India's Buddhists. At some point the tree will have a source that separates the "
        "vehicles in mainland Southeast Asia; none of these four is it."),
    "อิสลาม": (
        "-> islam, with no branch, because the census gives none. 3.26M people, 4.9%, and "
        "**it is the most concentrated religious geography on this map after Israel's**: "
        "the four southern border provinces are the old Sultanate of Patani, Malay-speaking "
        "and Shafi'i Sunni, and Narathiwat is 85.9% Muslim, Pattani 84.4%, Yala 79% and "
        "Satun 68% against a national 4.9%. Nothing here splits Sunni from Shia — Thailand "
        "has a small Shia community of Persian descent in Bangkok, the *Chao Sen*, which "
        "no cell in this census reaches."),
    "คริสต์": (
        "-> christianity, the parent, with no denomination anywhere in the source. 789,376 "
        "people, 1.2%. **Its geography is the hill north rather than Bangkok**: the "
        "Northern region is 3.05% Christian against 0.35% in the Northeast, because the "
        "Karen, Lahu, Lisu and Akha of the highlands were reached by Protestant and "
        "Catholic missions from the 1880s and the lowland Thai were not. That is the same "
        "population China's `cn2000.py` draws on the other side of the border, and this is "
        "the census that counts them rather than deriving them."),
    "ขงจื้อ": (
        "-> confucianism, and Thailand is the FIRST COUNTRY ON THIS MAP TO COUNT IT. "
        "20,151 people. The node has existed since the first tree and has been carried by "
        "nobody, because the censuses that would find Confucians — China's, Taiwan's, "
        "Singapore's — either do not ask or are not drawable. What Thailand's cell holds "
        "is the Thai Chinese who named Confucianism rather than Buddhism, which is a "
        "reporting choice more than a boundary in practice: the same household keeps a "
        "shrine, visits a wat and observes Qingming. Filed as published (§2.3)."),
    "อื่น ๆ": (
        "-> other.th. 55,545 people, 0.08%, and **the hill peoples' animism is the thing "
        "most likely to be in it** — but Thailand's census, unlike Cambodia's, says "
        "nothing about what the cell contains, so this residual gets no argument beyond "
        "its size. kh2019.py could file `Other` as highland indigenous religion because "
        "NIS wrote the sentence; nothing equivalent exists here, and §3.11's floor is what "
        "stops the guess."),
    "ไม่มีศาสนา": (
        "-> unaffiliated. 46,122 people, 0.07% — **the smallest irreligious share of any "
        "country on this map**, and it should be read as a fact about the question rather "
        "than about Thailand. A census that asks which religion you are, in a country "
        "where Buddhist identity is close to civic default, will not find the people who "
        "never go to a wat; they answer `พุทธ`. Compare the same instrument in Vietnam, "
        "where the answer set was narrow and 81.8% fell out of it."),
}

MAP = {
    "พุทธ": "buddhism",
    "อิสลาม": "islam",
    "คริสต์": "christianity",
    "ฮินดู": "hinduism",
    "ขงจื้อ": "confucianism",
    "ซิกข์": "sikhism",
    "อื่น ๆ": "other.th",
    "ไม่มีศาสนา": "unaffiliated",
}


def _key(cat):
    return " ".join(str(cat).split())


def resolve(cat):
    """Source category -> taxonomy node id, or None if deliberately not on the tree."""
    c = _key(cat)
    if c in EXCLUDED:
        return None
    return MAP.get(c)
