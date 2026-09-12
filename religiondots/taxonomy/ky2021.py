"""ESO 2021 census religion classification -> religiondots taxonomy.

Seventeen named categories plus a non-answer, on 6 districts. Shares are of ESO's census
survey tabular population count, 68,811, which is the universe every table in the report
uses (`sources/ky.py`).

    19.51%  Church of God          -> christianity.holiness        REVIEW
    16.72%  None                   -> unaffiliated
    13.58%  Roman Catholic         -> christianity.catholic
     8.71%  Seventh-day Adventist  -> christianity.adventist
     8.34%  Non-denominational     -> christianity.nondenominational
     6.92%  Baptist                -> christianity.baptist
     6.81%  Pentecostal            -> christianity.pentecostal
     5.69%  Presbyterian/United    -> christianity.united           REVIEW
     3.89%  Other                  -> other.ky   (a NEW node)
     2.83%  Anglican               -> christianity.anglican
     1.73%  Hindu                  -> hinduism
     1.50%  Wesleyan Holiness      -> christianity.holiness
     1.41%  DK/NS                  -> EXCLUDED
     0.92%  Jehovah Witness        -> christianity.witnesses
     0.50%  Methodist              -> christianity.methodist
     0.37%  Muslim                 -> islam
     0.31%  Rastafarian            -> rastafari
     0.24%  Judaism                -> judaism

**THE CAYMAN ISLANDS ARE THE MOST EVENLY RELIGIOUS PLACE ON THIS MAP, AND THE MOST FOREIGN.**
Over half the resident population was born abroad, and the religion table reads like it: the
largest answer reaches only 19.5%, and the top five are a Holiness church, no religion,
Roman Catholicism, Adventism and non-denominational Christianity — five different things,
none dominant.

**THE CHURCH OF GOD IS THE NATIONAL CHURCH AND IS ALSO THE FLATTEST CATEGORY**: 27.2% in
North Side, 25.3% in Bodden Town, 23.5% on the Sister Islands, 20.9% in East End, 18.3% in
West Bay, 16.8% in George Town. Nothing else here is that even. Everything that *does* vary
varies with foreign birth — Roman Catholicism is 18.4% in George Town against 3.6% in North
Side, which is the Filipino and Latin American workforce in the capital, and the Hindu share
peaks at 6.6% in East End.

**AND CAYMAN BRAC IS A DIFFERENT COUNTRY.** The Sister Islands are **30.5% Baptist** against
2.8-9.5% on every Grand Cayman district — the sharpest single contrast in the country, and it
is the old Brac Baptist settlement showing through a population that has otherwise been
remade by immigration.

EXCLUDED holds categories that are deliberately not on the tree.
REVIEW holds calls that are defensible but arguable, with the reason.
"""

EXCLUDED = {
    "Total":
        "the district's own population total, not a category. Carried in ky.csv because "
        "sources/ky.py checks the categories against it.",
    "DK/NS":
        "967 people, **1.41%** — `don't know / not stated`, and one of the smallest "
        "non-answers on this map. Spec §3.5: marked, not filled, never redistributed. "
        "Its geography is flat (0.86% to 2.03%), so it costs the map nothing in "
        "particular. "
        "**It is not, however, what the Cayman Islands is missing.** The universe of every "
        "table in this report is ESO's *census survey tabular population count*, 68,811, "
        "and the census counted 71,432 — the difference is 327 people in institutions plus "
        "a **2,294-person weighted non-response estimate** that exists only as a national "
        "figure. So 3.67% of the country is outside this map before `DK/NS` is reached, "
        "and `sources/ky.py` prints the whole ladder.",
}

REVIEW = {
    "Church of God":
        "-> christianity.holiness, NOT christianity.pentecostal, and this is the largest "
        "single call in the file — **13,423 people, 19.51%, the biggest religion in the "
        "country**. "
        "**The name alone cannot decide it and §12 says so**: `Church of God (Cleveland, "
        "Tennessee)` is Pentecostal and `Church of God (Anderson, Indiana)` is Holiness and "
        "explicitly rejected the tongues doctrine when the movement divided. ESO offers one "
        "unqualified cell. "
        "**What decides it is that the Cayman body is the Anderson one.** The Church of God "
        "congregations on Grand Cayman are the *Church of God Chapel* network, which is the "
        "**Cayman Islands Regional Mission Council of the Church of God (Anderson, "
        "Indiana)**. That is the historic and still the largest Church of God presence in "
        "the territory. "
        "**And it matches `jm2011.py` on the neighbour that shares the church history.** "
        "STATIN splits the family four ways: `Church of God in Jamaica` — the Anderson line "
        "— goes to `christianity.holiness`, while `New Testament Church of God` and `Church "
        "of God of Prophecy` (both Cleveland) go to `christianity.pentecostal`. Cayman's "
        "single cell is filed with the Anderson sibling. "
        "**The counter-argument, kept.** Cleveland-lineage congregations exist in Cayman "
        "too — Church of God of Prophecy has a presence — and they are inside this cell "
        "with no way to separate them. If ESO ever splits the family, the Pentecostal share "
        "of it should come out. Flagged per §2.4 so the fix is a lookup rather than an "
        "investigation.",
    "Presbyterian/United":
        "-> christianity.united, and NOT christianity.reformed.presbyterian. 3,913 people, "
        "5.69%. "
        "**The cell names one body, not two traditions.** The Presbyterian church in the "
        "Cayman Islands *is* the **United Church in Jamaica and the Cayman Islands**, formed "
        "in 1965 from the Presbyterian and Congregational churches and joined by the "
        "Disciples of Christ in 1992 — which is why ESO prints the two words together. "
        "`christianity.united` exists for exactly this: a union across Protestant families "
        "whose result is a branch of none of them, and filing it under Presbyterian would "
        "lose the Congregationalists and Disciples who are equally in it. "
        "**`jm2011.py` maps the same body the same way** from the other side of the same "
        "union, which is the strongest argument available — two censuses, one church. "
        "Its geography is the old Caymanian districts: 13.8% in North Side and 10.0% in "
        "East End against 4.1% in George Town and **0.66% on the Sister Islands**.",
    "Wesleyan Holiness":
        "-> christianity.holiness, the same node as `Church of God` above, which means the "
        "map cannot show them apart. That is a real loss of 1,031 people's distinctness and "
        "it is accepted rather than papered over: both are Holiness bodies in the strict "
        "sense, the tree has no children under `holiness`, and inventing two for one "
        "country would be §2.4's mistake in the other direction. The Wesleyan Holiness "
        "Church is strongest in North Side (5.1%) and West Bay (4.2%).",
    "Other":
        "-> other.ky, a per-source residual (§3.11). 2,679 people, 3.89%, and a NARROW one: "
        "the form already names Hindu, Muslim, Jewish and Rastafari, so the non-Christian "
        "population is not hiding in here. "
        "**Its geography is flat — 2.26% to 6.03% across the six districts — which by §9r's "
        "rule makes it a mixture rather than a missing category**, and nothing is assigned "
        "to it. Likely contents are named in branches.py.",
    "Non-denominational":
        "-> christianity.nondenominational, which is a real answer people give rather than "
        "a failure to classify, and at 8.34% it is the fifth largest here. Note the "
        "contrast with the Bahamas (`bs2022.py`), where the equivalent cell is worded "
        "`Other Christian Denomination (including non-denominational groups)` and therefore "
        "has to go to the branch root: ESO asks the cleaner question and gets a cleaner "
        "answer.",
}

MAP = {
    "Anglican": "christianity.anglican",
    "Methodist": "christianity.methodist",
    "Hindu": "hinduism",
    "Muslim": "islam",
    "Judaism": "judaism",
    "Rastafarian": "rastafari",
    "Non-denominational": "christianity.nondenominational",
    "None": "unaffiliated",
    "Other": "other.ky",
    "Baptist": "christianity.baptist",
    "Church of God": "christianity.holiness",
    "Jehovah Witness": "christianity.witnesses",
    "Pentecostal": "christianity.pentecostal",
    "Presbyterian/United": "christianity.united",
    "Roman Catholic": "christianity.catholic",
    "Seventh-day Adventist": "christianity.adventist",
    "Wesleyan Holiness": "christianity.holiness",
}


def _key(cat):
    return " ".join(str(cat).split())


EXCLUDED = {_key(k): v for k, v in EXCLUDED.items()}
MAP = {_key(k): v for k, v in MAP.items()}
REVIEW = {_key(k): v for k, v in REVIEW.items()}


def resolve(cat):
    """Source category -> taxonomy node id, or None if deliberately not on the tree.

    **`None` IS A CATEGORY NAME HERE, NOT A MISSING VALUE** — 11,502 people, 16.72%, the
    second largest answer in the country. Any caller reading ky.csv with a bare
    `pandas.read_csv` will have turned those six rows into NaN before this function sees
    them; see `_ky_counts` in countries.py, which passes `keep_default_na=False`.
    """
    c = _key(cat)
    if c in EXCLUDED:
        return None
    return MAP.get(c)
