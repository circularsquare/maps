"""
INSTAT Census 2023 table 1.13 `Besimi fetar` -> religiondots taxonomy.

Ten categories, of which eight are drawn. It is a shallow table by the standards of a
European census and an unusually well-shaped one: the form keeps Bektashis apart from other
Muslims, keeps believers of no denomination apart from atheists, and keeps a refusal apart
from a missing value. Only two other censuses here do the middle one and none does the first.

**`Mysliman - Bektashi` IS THE POINT OF THE COUNTRY.** 115,644 people, 4.81%. It gets a new
node, `islam.bektashi`; `taxonomy/branches.py` carries the argument for putting it under
`islam` rather than under `alevism` or `islam.shia`.

**`Mysliman` IS NOT MAPPED TO `islam.sunni`, AND THAT IS DELIBERATE.** The form offers
`Mysliman` and `Mysliman - Bektashi` and asks about nothing else, so the plain row is
"Muslim, and not Bektashi" rather than "Sunni": it holds the Sunni majority, and also any
Halveti, Rifa'i or Sa'di respondent, and any Bektashi who wrote `Mysliman`. Croatia's
`Muslimani` and twenty other censuses land on the parent for the same reason. Sunni Islam is
a real node here (Russia and Türkiye ask), which is exactly why an unasked question must not
be answered on its behalf.

EXCLUDED holds categories that are deliberately not on the tree.
REVIEW holds calls that are defensible but arguable, with the reason.
"""

EXCLUDED = {
    "Prefer not to answer":
        "244,331 people, 10.17% of Albania, choosing a box the form offers by name, "
        "`Preferoj të mos përgjigjem`. Not a religion and not irreligion; §3.5. "
        "sources/al.md §5 has the history behind a refusal cell this large.",
    "Not available":
        "134,451 people, 5.60%, INSTAT's `Nuk disponohet`. A coverage residual and not a "
        "refusal: the office keeps the two apart in its own table, so this file does too. "
        "Croatia's `Nepoznato` and Rwanda's `Not stated` are the same shape.",
}

REVIEW = {
    "Muslim":
        "-> islam, the PARENT, and not islam.sunni. See the module docstring: the census "
        "splits Bektashis out of Muslims and asks nothing else, so the residual row is "
        "'Muslim, not Bektashi' and reads Sunni only by inference. 1,101,718 people, "
        "45.86%, which makes this one node most of the Albanian map.",
    "Muslim - Bektashism":
        "-> islam.bektashi, a NEW NODE, added with this country because no other source on "
        "this map counts the order at all. The category is INSTAT's own and its English is "
        "the office's own. Placed under `islam` because that is where the census puts it "
        "and where Albanian Bektashis put themselves; branches.py has the full argument, "
        "including why it is not filed under `alevism`.",
    "Christian - Orthodoxy":
        "-> christianity.orthodox.canonical. The Orthodox Autocephalous Church of Albania "
        "is in communion with Constantinople, so this is the canonical node and not "
        "`christianity.orthodox`. hr2021.py and me2023.py file their equivalents the same "
        "way. 173,645 people; the census does not name the church, but Albania has only "
        "the one.",
    "Christian - Catholicism":
        "-> christianity.catholic, the PARENT, with no rite. Albania's Catholics are "
        "overwhelmingly Latin, but the country also has the Greek Catholic Apostolic "
        "Administration of Southern Albania, and the census offers one Catholic box, so "
        "naming the rite would invent a fact about 201,530 people.",
    "Christian - Evangelists (Protestant)":
        "-> christianity.evangelical, the 'named no body' node. INSTAT's own English gloss "
        "is `Evangelists (Protestant)` and the Albanian is `Ungjillore`, which is what the "
        "Albanian Evangelical Alliance calls itself; the census names no denomination. "
        "9,658 people, 0.40%, the smallest religion drawn here.",
    "Believers without denomination":
        "-> unchurched, the node Czechia's `věřící nehlásící se k žádné církvi` created. "
        "The Albanian `Besimtarë të pacilësuar` is the same measurement and it is large: "
        "332,155 people, 13.83%, Albania's second largest answer after Islam and half again "
        "the size of the Catholic and Orthodox cells put together. NOT `unaffiliated`, "
        "which is a report of no religion, and not `secular`, which is a position; these "
        "people reported belief and no denomination, and the census offered atheism "
        "separately, so a respondent choosing this one has declined both of those.",
    "Atheists":
        "-> secular and NOT unaffiliated. `Ateist` is an offered box and a stated position, "
        "which is branches.py's own distinction, and Ireland's and India's `Atheist` rows "
        "resolve the same way. Albania's form has no 'no religion' answer at all, so "
        "`unaffiliated` would be reporting an option nobody was given. 85,311 people, "
        "3.55%, in the country that declared itself the world's first atheist state in "
        "1967 and closed every mosque and church in it.",
    "Other religion or faith":
        "-> other.al. 3,670 people, 0.15%, and no breakdown published. Albania's Jewish, "
        "Baha'i and Protestant-outside-the-Alliance communities are all in here and none "
        "of them can be separated out.",
}

MAP = {
    "Muslim": "islam",
    "Muslim - Bektashism": "islam.bektashi",
    "Christian - Catholicism": "christianity.catholic",
    "Christian - Orthodoxy": "christianity.orthodox.canonical",
    "Christian - Evangelists (Protestant)": "christianity.evangelical",
    "Other religion or faith": "other.al",
    "Believers without denomination": "unchurched",
    "Atheists": "secular",
}


def _key(cat):
    return " ".join(str(cat).split())


def resolve(cat):
    """Source category -> taxonomy node id, or None if deliberately not on the tree."""
    c = _key(cat)
    if c in EXCLUDED:
        return None
    return MAP.get(c)
