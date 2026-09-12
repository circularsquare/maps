"""Vietnam census religion -> religiondots taxonomy. See sources/vn.md.

**Two censuses, one mapping.** The drawn data is the 2009 census's Biểu 7 — thirteen religions
by province — and the 2019 census's Biểu 3 is carried in the same normalised file at country
level, national only, because that is the only geography it has. The 2019 labels are different
strings for mostly the same bodies, so both sets are mapped here; nothing 2019 is drawn.

WHAT MAKES VIETNAM WORTH DRAWING IS THE MEKONG DELTA, AND IT IS A LINEAGE RATHER THAN A LIST.
Bửu Sơn Kỳ Hương (1849), Tứ Ân Hiếu Nghĩa (1867), Hòa Hảo (1939) and Hiếu Nghĩa Tà Lơn are one
documented descent through An Giang and the Bảy Núi hills, and **Vietnam's census counts all
four separately** — the first source anywhere on this map to hand `branches.py` a whole
lineage instead of one more member of an existing group. Caodaism, which the tree has held
since Australia at 677 adherents, arrives here at 807,915. Cham Balamon is the last living
Hindu tradition of the Indianised kingdoms and has a cell of its own.

EIGHT NEW NODES, AND NO `other.vn`. There is no residual category to contain: Vietnam
recognises a fixed list of religious organisations and the census counts exactly that list,
so the table has no `other religion` row at all. What it has instead is
`Không xác định tôn giáo` — not stated, 30 people nationally — which is non-response and is
EXCLUDED under spec §3.5, not a residual.

**THE LARGEST FACT ABOUT VIETNAM IS THE 81.8%, AND SINCE 2026-09-06 IT IS DRAWN.** The census
asks which of the recognised organisations a person belongs to and 70.2 million people answered
none. That is not irreligion: it is ancestor veneration, the village đình, đạo Mẫu, and the
great majority of Vietnamese Buddhist practice, none of which has a box. Biểu 7 has no row for
any of them — its universe is people WITH a religion — so `sources/vn.py` computes the row as
each province's population minus its religious total and it lands on **`unknown`**, a node whose
whole content is that we do not know what these people practise. Anita's call; spec §14.7's
decision for China, taken first for the country that got there first. See REVIEW below.

EXCLUDED holds categories that are deliberately not on the tree.
REVIEW holds calls that are defensible but arguable, with the reason.
"""

import unicodedata

EXCLUDED = {
    "Tổng số - Total":
        "Biểu 7's own total row, and it is NOT the population — it is the number of people "
        "with a religion, 15,651,467 against a census population of 85,846,997. sources/vn.py "
        "emits it so coverage is checkable in the data.",
    "Dân số - Population":
        "the province's census population, read from Biểu 1 of the same volume and emitted "
        "here as the denominator. Not a religion category and not part of Biểu 7.",
    "Không xác định tôn giáo - Not stated":
        "religion not stated — 30 people nationally in 2009, the smallest non-response on "
        "this map by three orders of magnitude. Reported, not filled, not drawn (spec §3.5).",
}

REVIEW = {
    "Không theo tôn giáo - No religion":
        "-> unknown. **70,195,530 people, 81.8% of Vietnam, and the largest single call in "
        "this file.** It is 4.5 times everything else here put together. "
        "**Where the number comes from.** Biểu 7's universe is people WITH a religion and it "
        "has no row for anyone else at any geography. sources/vn.py computes the row as each "
        "province's Biểu 1 population minus its Biểu 7 total — the complement of a published "
        "partition, checked to the person in all 63 provinces, so it is `measured` rather "
        "than modelled. The 2019 census publishes the same category directly, nationally, at "
        "83,046,105, which is where the label comes from. "
        "**Why `unknown` and not `unaffiliated`.** The Vietnamese question asks which of the "
        "state-recognised religious organisations a person belongs to. Answering none of them "
        "is not the same claim as reporting no religion, and reading it as irreligion would "
        "be the largest error available on this map: ancestor veneration is near-universal, "
        "the village đình and đạo Mẫu are everywhere, and most of the people who would call "
        "themselves Buddhist are in this bucket rather than in the 4.6M the census counts, in "
        "a country usually described as around 45% Buddhist by practice. "
        "**Why not `unrecorded` either** — that node is for a source that never asked, and "
        "these people were asked. See the `unknown` node's own note for the distinction and "
        "for the test that admits anything else to it.",
    "Phật Giáo - Buddish":
        "-> buddhism. 6,802,318 in 2009, the largest cell. Kept on the undivided parent, "
        "which is lk2024.py's call for Sri Lanka and for the same reason: the census offers "
        "one word and there is nothing in the table to split it with. What is inside it is "
        "known and not stated — Vietnamese Buddhism is overwhelmingly Mahayana, except for "
        "the Khmer Krom of Trà Vinh and Sóc Trăng, who are Theravada and are perhaps a "
        "million of this number. Mapping the whole cell to `buddhism.mahayana` would be "
        "right about most of it and would erase them, which §2.4 says to defer rather than "
        "guess; `source_category` rides on every row so a later source can deepen it.",
    "Phật Giáo - Buddish  Hòa Hảo":
        "-> buddhism.hoahao. 1,433,252. The double space is in the source and is kept "
        "verbatim (§2.4). Note the label: the census's own name for it is Hòa Hảo BUDDHISM, "
        "which is why the node is under `buddhism` rather than beside Caodaism.",
    "Hồi Giáo - Muslim":
        "-> islam. 75,268, and **the cell merges two communities that are not the same "
        "religion to the people in them.** Vietnam's Muslims are almost all Cham, and they "
        "are split between the Sunni of An Giang, Tây Ninh and Ho Chi Minh City, who pray "
        "five times daily and read Arabic, and the **Bani** of Ninh Thuận and Bình Thuận, "
        "whose practice is a thousand-year-old localisation with a hereditary priesthood, a "
        "three-day Ramadan and Cham-language liturgy. Bani sit beside Cham Balamon in the "
        "same villages and the two are often described as one Cham religious system in two "
        "halves. The census gives Balamon its own row and folds Bani into Islam, so the map "
        "shows one and not the other; nothing here can separate them and this file does not "
        "try (§14.4). A finer source would give Bani a node.",
    "Tin Lành - Protestantism":
        "-> christianity.protestant. 734,168 in 2009 and 960,558 in 2019 — the fastest-"
        "growing category in the country, up 31% while the census's religious total fell "
        "16%. The node is the unspecified-answer one, because the census names no body. "
        "**Two things it holds are worth knowing.** Institutionally it is mostly the "
        "Evangelical Church of Vietnam, North and South, which are separate bodies. "
        "Geographically it is not lowland Kinh Christianity at all: the concentrations are "
        "Hmong in the northern uplands and Montagnard peoples in the Central Highlands, "
        "which is why Điện Biên, Lai Châu, Gia Lai, Đắk Lắk and Kon Tum have Protestant "
        "shares an order of magnitude above the national one. The census counts registered "
        "congregations; unregistered house churches are the ones under pressure and are not "
        "separable here.",
    "Bà La Môn":
        "-> hinduism.chambalamon. 56,427 in 2009, labelled `Chăm Bà la môn` in 2019 — the "
        "census's own word is Brahmanism and its own qualifier is Cham. Almost all of it is "
        "Ninh Thuận (40,695) and Bình Thuận. See the node's own note for why it is a child "
        "rather than plain `hinduism`.",
    "Tịnh độ cư sĩ Phật hội Việt Nam":
        "-> buddhism.tinhdo. 11,093 in 2009 and 2,306 in 2019, an 79% fall that is "
        "definitional rather than real — see sources/vn.md on the 2009/2019 discontinuity. "
        "Given its own node rather than `buddhism.mahayana`; the node's note says why.",
    "Minh Sư Đạo":
        "-> eastasiannew.vietnamese.minhsu. 709 people. Under the East Asian new religions "
        "and not under Daoism or `chinesefolk`: it is the Vietnamese Xiantiandao line, "
        "organised in Vietnam from the 1860s, and its own self-description is a three-"
        "teachings synthesis rather than any one of them.",
    "Minh Lý Đạo":
        "-> eastasiannew.vietnamese.minhly. 366 people, and the smallest thing drawn in "
        "Vietnam. At 1:1,000 people per dot it reaches no dot in any province and is a "
        "presence ring under spec §4.3, which is exactly what that grammar is for.",
    "Cao Đài":
        "-> caodaism. 807,915 in 2009, against the 677 in Australia that the node was "
        "created for. Tây Ninh, its holy see, is 32.5% Caodaist. The node stays a ROOT "
        "rather than moving under `eastasiannew.vietnamese`, which is where it belongs — "
        "Anita's call 2026-09-05, restated in that node's note with the id it would take.",
    "Ba Ha'i":
        "-> bahai. 731 in 2009 and 2,153 in 2019, one of only three categories that grew. "
        "Spelled without diacritics in the source; the tree's label carries them.",
    "Giá o hội Cơ đố c Phục lâm Việt Nam":
        "-> christianity.adventist. 11,830, and **2019 only** — the Seventh-day Adventists "
        "were recognised in 2008 and so are not in the 2009 table. The broken spacing in "
        "the key is the PDF's own glyph splitting and is kept verbatim, because that is what "
        "sources/vn.py writes into the normalised file (§2.4).",
    "Giáo hội Các thành hữu Ngày sau của Chúa Giê su Ky tô Việt Nam (Mormon)":
        "-> christianity.latterday. 4,281, 2019 only; recognised 2016. The label is "
        "reassembled by sources/vn.py from three lines of the PDF, because the census wraps "
        "it around its own figures.",
    "Phật giáo Hiếu Nghĩa Tà Lơn":
        "-> buddhism.talon. 401, 2019 only; recognised 2010. The fourth and last body of the "
        "Bửu Sơn Kỳ Hương lineage to be registered, and the only one with a national figure "
        "and no geography at all.",
}

# The 2009 categories — the drawn ones. Keys are exactly the strings sources/vn.py writes.
MAP = {
    "Phật Giáo - Buddish": "buddhism",
    "Công Giáo - Catholics": "christianity.catholic",
    "Phật Giáo - Buddish  Hòa Hảo": "buddhism.hoahao",
    "Hồi Giáo - Muslim": "islam",
    "Cao Đài": "caodaism",
    "Minh Sư Đạo": "eastasiannew.vietnamese.minhsu",
    "Minh Lý Đạo": "eastasiannew.vietnamese.minhly",
    "Tin Lành - Protestantism": "christianity.protestant",
    "Tịnh độ cư sĩ Phật hội Việt Nam": "buddhism.tinhdo",
    "Đạo Tứ ấn hiếu nghĩa": "buddhism.tuan",
    "Bửu sơn Kỳ hương": "buddhism.buuson",
    "Ba Ha'i": "bahai",
    "Bà La Môn": "hinduism.chambalamon",
    # Not a Biểu 7 row — computed by sources/vn.py as population minus religious total. See
    # REVIEW; this one line is 81.8% of the country.
    "Không theo tôn giáo - No religion": "unknown",
}

# The 2019 categories, national only and not drawn. Same bodies, different strings — GSO
# re-typeset the table and renamed most rows, including its own typo `Bửi Sơn` for Bửu Sơn.
# Mapped so tools/check_mapping.py has no unresolved category and so the comparison in
# sources/vn.py can be repeated against nodes rather than against strings.
MAP_2019 = {
    "Phật Giáo - Buddish": "buddhism",
    "Công giáo - Catholics": "christianity.catholic",
    "Phật giáo Hòa Hảo - Buddish Hòa Hảo": "buddhism.hoahao",
    "Hồi giáo - Muslim": "islam",
    "Cao Đài": "caodaism",
    "Giáo hội Phật đường Nam Tông Minh Sư đạo": "eastasiannew.vietnamese.minhsu",
    "Hội thánh Minh lý đạo - Tam Tông Miếu": "eastasiannew.vietnamese.minhly",
    "Tin lành - Protestantism": "christianity.protestant",
    "Tịnh độ Cư sỹ Phật hội Việt Nam": "buddhism.tinhdo",
    "Đạo Tứ Ân Hiếu nghĩa": "buddhism.tuan",
    "Bửi Sơn Kỳ hương": "buddhism.buuson",
    "Tôn giáo Baha'i": "bahai",
    "Chăm Bà la môn": "hinduism.chambalamon",
    "Không theo tôn giáo - No religion": "unknown",
    "Phật giáo Hiếu Nghĩa Tà Lơn": "buddhism.talon",
    "Giáo hội Các thành hữu Ngày sau của Chúa Giê su Ky tô Việt Nam (Mormon)":
        "christianity.latterday",
    "Giá o hội Cơ đố c Phục lâm Việt Nam": "christianity.adventist",
}

MAP.update(MAP_2019)


def _key(cat):
    # NFC as well as whitespace: the 2019 PDF returns the Adventist label with `á` and `ố`
    # DECOMPOSED while every other label on the page is precomposed, so a byte comparison
    # against a visually identical literal here fails and the category silently resolves to
    # nothing. sources/vn.py already normalises on write; this is the second guard, because
    # the failure is invisible in a terminal, a diff and a code review alike.
    return unicodedata.normalize("NFC", " ".join(str(cat).split()))


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
