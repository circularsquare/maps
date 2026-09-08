"""ZIMSTAT 2022 PHC religion -> religiondots taxonomy.

Eleven categories plus the universe total, at province. Short, and one of the eleven is the
reason the country is drawn: **`Apostolic Sect` is 6,112,503 people, 40.3% of Zimbabwe, the
largest single religious answer in the country and nearly twice the size of everything
`christianity.africaninstituted` held before.**

The list is otherwise conventional — Catholic, Protestant, Pentecost, Other Christian, and
one traditional cell — with the unusual feature that Islam, Judaism and Hinduism each get a
box despite being 0.58%, 0.05% and 0.02%. That is why `other.zw` is a genuinely small tail
rather than the crowded residual most censuses hand over.

EXCLUDED holds categories that are deliberately not on the tree.
REVIEW holds calls that are defensible but arguable, with the reason.
"""

EXCLUDED = {
    "Total":
        "the province's own population total, not a category.",
}

REVIEW = {
    "African Tradition":
        "-> indigenous.african, the node Ghana added. 762,660 people, 5.02%. Its geography "
        "is northern and rural — Mashonaland Central 8.8%, Matabeleland North 7.1%, "
        "Matabeleland South 6.3% — against 3.3% in Harare and **2.4% in Bulawayo, the "
        "lowest in the country**. "
        "**Read it as a floor**, for the reason sources.md §11b gives for the whole "
        "continent: the box is exclusive of the Christian ones, and Shona and Ndebele "
        "practice — the *midzimu* ancestral cult, consultation of a *n'anga*, the Mwari "
        "shrines of the Matobo hills — very commonly accompanies church membership rather "
        "than replacing it. **Zimbabwe is a sharper case than most**, because the Vapostori "
        "churches themselves grew out of exactly that overlap and are counted separately "
        "at eight times the size. No child node: no source names a Zimbabwean tradition "
        "individually (§2.4).",
    "Roman Catholic":
        "-> christianity.catholic, the parent rather than .latin. 975,488 people, 6.4%. "
        "Latin rite throughout and the census does not say so, which is the call ke2019.py, "
        "mw2018.py and bj2013.py all make. Urban and Midlands — Harare 8.9%, Bulawayo 8.5%, "
        "Midlands 8.3% — and lowest in Mashonaland Central at 3.2%.",
    "Protestant":
        "-> christianity.protestant, which holds an ANSWER and not a church. 2,089,735 "
        "people, 13.8%. The mission inheritance: Anglicans, Methodists (both the British "
        "and the American United Methodist mission at Old Mutare), the London Missionary "
        "Society's Ndebele congregations, the Dutch Reformed in Masvingo, the Salvation "
        "Army and the Brethren in Christ. **No branch is inferred** — the census gives none "
        "and those bodies sit in five different places on the tree. Bulawayo 20.2% and "
        "Harare 17.9% against 7.4% in Mashonaland Central, which is the mission-station map "
        "rather than a modern one.",
    "Apostolic Sect":
        "-> christianity.africaninstituted, and **this is the category the country is worth "
        "drawing for**. 6,112,503 people, 40.3% — the largest single religious answer in "
        "Zimbabwe, larger than every other Christian cell put together, and nearly twice "
        "the size of Kenya's African Instituted Churches figure which until now was the "
        "whole of that node. "
        "The Vapostori: the *masowe* (open-air) churches founded in the 1930s by **Johane "
        "Masowe** (Shoniwa Masedza) and **Johane Marange** (Muchabaya Momberume), worshipping "
        "in white robes in the open rather than in buildings, with prophecy and faith "
        "healing at the centre and a deliberate break from both the mission churches and "
        "the ancestral cult. They are dozens of distinct bodies — the African Apostolic "
        "Church, the Johane Masowe eChishanu, the Gospel of God Church and many more — and "
        "**ZIMSTAT gives them one cell**, so nothing here can separate them. "
        "**The mapping is not in doubt but the LABEL is worth stating**: `Sect` is "
        "ZIMSTAT's word, not this project's, and it is kept in `source_category` per §2.4 "
        "while the node is named for what these churches are. "
        "Its geography is the Shona north and east — Mashonaland Central 53.3%, Manicaland "
        "50.2% (Marange's own country), Mashonaland East 45.9% — against **21.2% in "
        "Bulawayo and 27.2% in Harare**. It is a rural majority religion and a large urban "
        "minority, which is the opposite of how new churches usually distribute.",
    "Pentecost":
        "-> christianity.pentecostal. 2,582,565 people, 17.0%. ZAOGA (Ezekiel Guti's "
        "Zimbabwe Assemblies of God Africa), the Apostolic Faith Mission, the United Family "
        "International Church and the newer prophetic ministries. **The one clearly URBAN "
        "religion in the country**: Harare 28.5% and Bulawayo 24.9% against 10.5% in "
        "Mashonaland Central — the exact inverse of the Vapostori, and the two together are "
        "most of Zimbabwe's Christianity. No branch given and none inferred.",
    "Other Christian":
        "-> christianity.other. 1,177,513 people, 7.8%. Smaller than the identically-named "
        "residual in Ghana (12.3%) or Malawi (26.6%), because ZIMSTAT has already lifted "
        "out the two things that would otherwise dominate it — the Vapostori and the "
        "Pentecostals. Holds the Adventists (large in Zimbabwe), the Jehovah's Witnesses, "
        "the Latter-day Saints, the Orthodox and the independent evangelical churches. "
        "**Its geography is Matabeleland and Masvingo** — 16.1%, 15.5%, 13.2% — which is "
        "the Brethren in Christ and Seventh-day Adventist mission field, against 3.3% in "
        "Mashonaland Central.",
    "Islam":
        "-> islam, with no branch, because the census gives none. 88,628 people, 0.58%. "
        "Two unrelated communities the census cannot separate: the Shona and Ndebele "
        "Muslims of the Zambezi valley and the northern farming districts, and the Indian "
        "and Malawian-descended Muslims of Harare and the towns. Highest in Mashonaland "
        "Central and West at 1.1%.",
    "Judaism":
        "-> judaism, with no branch. 6,845 people, 0.05%. **Almost certainly not what a "
        "reader will assume, and worth the note.** Zimbabwe's historic Ashkenazi community "
        "— Harare and Bulawayo synagogues, a few thousand at its 1960s peak — has largely "
        "emigrated and numbers in the hundreds. The census figure is an order of magnitude "
        "larger and its geography is wrong for that community: it peaks in **Midlands, "
        "Masvingo and Manicaland**, rural provinces, not in the two cities. The likeliest "
        "reading is that it counts the **Lemba**, who claim Judaic descent and observe "
        "dietary and circumcision laws, and members of Judaising churches. The census does "
        "not say, and nothing here resolves it — recorded per §2.4 so a source that does "
        "is a lookup.",
    "Hinduism":
        "-> hinduism, with no branch. 3,425 people, 0.02% — the smallest cell in the "
        "country. The remnant of Zimbabwe's Indian community, concentrated in Harare and "
        "Bulawayo at 0.1% each. Under one dot at the national dot value in several "
        "provinces, so it will draw sparsely and may ring (§4.3).",
    "None":
        "-> unaffiliated. 1,255,578 people, 8.3%. One cell, so nothing goes to `secular`, "
        "which needs a separately-counted atheist or humanist answer — the call ke2019.py, "
        "mw2018.py, bj2013.py, hr2021.py and mk2021.py all make. "
        "**Its geography is Matabeleland and the Mashonaland west/centre**, at 13.5%, 12.1% "
        "and 12.1%, against **4.5% in Manicaland and 4.9% in Masvingo** — and it is LOW in "
        "both cities (Harare 7.1%, Bulawayo 7.8%). So this is not an urban secularising "
        "figure. Benin's `Aucune` warning applies with less force here, because Zimbabwe's "
        "`African Tradition` does not track it the same way (Mashonaland Central is high on "
        "both, Matabeleland South high on `None` and middling on tradition), but the "
        "possibility that some of it is unaffiliated traditional practice cannot be ruled "
        "out from this table. See sources/zw.md §5.",
    "Other":
        "-> other.zw. 124,017 people, 0.82%. See the node's own note: a genuinely small "
        "tail, because Islam, Judaism and Hinduism all have their own boxes and the "
        "Vapostori are counted by name. Per §3.11.",
}

MAP = {
    "African Tradition": "indigenous.african",
    "Roman Catholic": "christianity.catholic",
    "Protestant": "christianity.protestant",
    "Apostolic Sect": "christianity.africaninstituted",
    "Pentecost": "christianity.pentecostal",
    "Other Christian": "christianity.other",
    "Islam": "islam",
    "Judaism": "judaism",
    "Hinduism": "hinduism",
    "None": "unaffiliated",
    "Other": "other.zw",
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
