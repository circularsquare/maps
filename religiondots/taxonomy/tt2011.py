"""CSO 2011 census religion classification -> religiondots taxonomy.

Seventeen categories on 15 municipalities — fourteen named bodies, a residual, a
no-religion answer and a non-answer. After Jamaica's 19 and Saint Vincent's 18 this is the
third-best religion question in the Americas outside the United States, and it is the most
*plural* of the three: no single answer reaches 22%.

    21.60%  Roman Catholic               -> christianity.catholic
    18.15%  Hinduism                     -> hinduism
    12.02%  Pentecostal/ Evangelical/    -> christianity.pentecostal
            Full Gospel
    11.10%  Not Stated                   -> EXCLUDED
     7.27%  Other                        -> other.tt   (a NEW node)
     5.67%  Anglican                     -> christianity.anglican
     5.67%  Baptist-Spiritual Shouter    -> afrodiasporic.spiritualbaptist  (NEW)  REVIEW
     4.97%  Islam                        -> islam
     4.09%  Seventh Day Adventist        -> christianity.adventist
     2.49%  Presbyterian/ Congregational -> christianity.reformed           REVIEW
     2.18%  None                         -> unaffiliated
     1.47%  Jehovah's Witness            -> christianity.witnesses
     1.21%  Baptist-Other                -> christianity.baptist
     0.90%  Orisha                       -> afrodiasporic.orisha   (a NEW node)
     0.65%  Methodist                    -> christianity.methodist
     0.27%  Moravian                     -> christianity.moravian
     0.27%  Rastafarian                  -> rastafari

**TWO NEW NODES, AND BOTH ARE THE REASON THE COUNTRY IS DRAWN.** `afrodiasporic.orisha` and
`afrodiasporic.spiritualbaptist` are the first census counts of either tradition anywhere on
this map. Between them they are **86,920 people, 6.6% of Trinidad and Tobago** — more than
its Muslims.

**THE HINDU GEOGRAPHY IS THE OTHER REASON.** 240,100 people, and it is not spread: **43.0%
of Penal/Debe**, 31.3% of Couva/Tabaquite/Talparo, 30.0% of Chaguanas — the Indo-Trinidadian
sugar belt of the central and southern plain — against **0.7% of Tobago**. With Guyana
(§9r) and, if it is ever built, Suriname, this is the Indo-Caribbean geography the map was
missing.

**AND TOBAGO IS A DIFFERENT COUNTRY RELIGIOUSLY, WHICH THE 15-UNIT TIER IS JUST FINE ENOUGH
TO SHOW.** Roman Catholic 6.6% there against 44.8% in Diego Martin; Anglican 12.8%, Seventh
Day Adventist 16.3% and Pentecostal 14.7%, all the highest in the country. Trinidad is
Catholic and Hindu; Tobago is Protestant.

EXCLUDED holds categories that are deliberately not on the tree.
REVIEW holds calls that are defensible but arguable, with the reason.
"""

EXCLUDED = {
    "Total":
        "the unit's own population total, not a category. Carried in tt.csv because "
        "sources/tt.py checks the seventeen categories against it.",
    "Not Stated":
        "146,798 people, **11.10% — the largest non-answer on this map outside the "
        "United States**, and large enough that it changes how every share here should be "
        "read: the drawn percentages are of the whole non-institutional population, so a "
        "religion's share among people who answered is about 12% higher than the number "
        "drawn. Spec §3.5: marked, not filled, and never redistributed. "
        "**Its geography is uneven and unexplained** — 19.3% in Tunapuna/Piarco and 15.0% "
        "in Point Fortin against 6.9% in Sangre Grande, a near threefold spread. CSO "
        "offers no comment on it. That pattern does not look like a refusal (which would "
        "track the most religiously mixed places) or like an enumeration failure (which "
        "would track the remotest); Tunapuna/Piarco is the largest and most suburban "
        "corporation in the country. Named as an open question rather than guessed at.",
}

REVIEW = {
    "Baptist-Spiritual Shouter":
        "-> afrodiasporic.spiritualbaptist, a NEW node, and **the most consequential call "
        "in this file**. 75,002 people, 5.67% — more numerous than Trinidad's Anglicans "
        "and nearly five times its ordinary Baptists. "
        "**The census itself makes the distinction**: CSO offers `Baptist-Spiritual "
        "Shouter` and `Baptist-Other` as two separate answers, so filing them together "
        "would discard a split the source went out of its way to publish. "
        "**The placement follows Jamaica's `Revivalist` exactly (§9ab), and the parallel is "
        "structural rather than loose.** The Spiritual or Shouter Baptists are a Trinidadian "
        "religion in which Baptist Protestantism and West African — mainly Yoruba and Kongo "
        "— practice fused rather than one absorbing the other: scripture and hymnody "
        "alongside spirit possession, the *mourning* ground, bell-ringing, candles, water "
        "rites and head-tying. Spec §3.3 says the syncretism gets a node instead of being "
        "split across its ingredients, and `afrodiasporic` is where Umbanda, Candomblé and "
        "Revival Zion already sit for the same reason. "
        "**The counter-argument is real and is recorded rather than dismissed.** Spiritual "
        "Baptists overwhelmingly describe themselves as Christians and as Baptists, and "
        "many would reject being filed outside Christianity. The same is true of Revival "
        "Zion, and the project made the same call there; what decides it is that the tree "
        "is a genealogy of traditions rather than a register of self-description, and this "
        "tradition's descent is genuinely double. "
        "**Its geography supports the placement**: 13.0% in Point Fortin, 10.6% in Tobago "
        "and 9.6% in San Juan/Laventille — Afro-Trinidadian areas — against 2.6% in "
        "Penal/Debe. **And Trinidad marks it as a matter of history**: the Shouters "
        "Prohibition Ordinance banned the religion outright from 1917 to 1951, and 30 March "
        "is a public holiday, Spiritual Baptist / Shouter Liberation Day.",
    "Orisha":
        "-> afrodiasporic.orisha, a NEW node. 11,918 people, 0.90%. Trinidad Orisha — "
        "historically called Shango — is Yoruba orisha worship carried across in the "
        "nineteenth century, and it is the direct sibling of Candomblé and Santería, which "
        "the tree already holds. This is the **first census count of an orisha religion "
        "under its own name** anywhere on this map. "
        "**One thing this cell cannot show, and it matters for reading the number.** In "
        "Trinidad, Orisha practice and Spiritual Baptist practice overlap heavily — many "
        "people participate in both, and a substantial literature treats them as one "
        "religious complex rather than two memberships. A census offers one box, so 11,918 "
        "is a count of people who chose Orisha *over* the alternatives, not a count of "
        "people who practise it. Read as a floor, like Jamaica's Rastafari. Nothing here "
        "corrects it (§14.4). Its geography is Mayaro/Rio Claro 2.1%, Point Fortin 1.8% "
        "and Port of Spain 1.6%.",
    "Presbyterian/ Congregational":
        "-> christianity.reformed, the PARENT, and deliberately not "
        "`christianity.reformed.presbyterian`. CSO publishes the two traditions in one "
        "cell, and the tree holds both as children of `reformed`; picking the larger child "
        "would silently drop the other, which is the mistake `christianity.united` exists "
        "to prevent elsewhere. The parent is the honest grain for a combined cell. "
        "**And this is the most interesting 2.49% in the country.** Its geography is not "
        "the Scottish-settler pattern the name suggests — it is **San Fernando 5.5%, "
        "Penal/Debe 5.3%, Princes Town 4.0%, Couva/Tabaquite/Talparo 3.7%**, which is the "
        "*Indo-Trinidadian* belt, the same units that are 43%, 27% and 31% Hindu. Tobago is "
        "0.2%. That is the Canadian Presbyterian Mission to the Indians, which from 1868 "
        "built schools among the indentured Indian population and drew its converts from "
        "it; Trinidad's Presbyterians are largely Indo-Trinidadian, and the map shows it "
        "without being told.",
    "Pentecostal/ Evangelical/ Full Gospel":
        "-> christianity.pentecostal. Three labels in one cell, and the tree has no node "
        "for 'evangelical, unspecified' that would hold the middle one separately. "
        "Pentecostal is the right parent because the Trinidadian bodies in this cell are "
        "overwhelmingly Pentecostal or neo-Pentecostal in descent, and because the cell's "
        "own name leads with it. 159,033 people, 12.02%, and the second largest Christian "
        "answer after Roman Catholic.",
    "Other":
        "-> other.tt, a per-source residual (§3.11). 96,166 people, **7.27%**, which is "
        "wide for a question that already names fourteen bodies. Its geography is sharp — "
        "14.4% in Point Fortin against 4.5% in Port of Spain — which by §9r's rule points "
        "at a missing category, and the likeliest content is named in branches.py rather "
        "than assumed here.",
}

MAP = {
    "Anglican": "christianity.anglican",
    "Baptist-Spiritual Shouter": "afrodiasporic.spiritualbaptist",
    "Baptist-Other": "christianity.baptist",
    "Hinduism": "hinduism",
    "Islam": "islam",
    "Jehovah’s Witness": "christianity.witnesses",
    "Methodist": "christianity.methodist",
    "Moravian": "christianity.moravian",
    "Orisha": "afrodiasporic.orisha",
    "Pentecostal/ Evangelical/ Full Gospel": "christianity.pentecostal",
    "Presbyterian/ Congregational": "christianity.reformed",
    "Rastafarian": "rastafari",
    "Roman Catholic": "christianity.catholic",
    "Seventh Day Adventist": "christianity.adventist",
    "Other": "other.tt",
    "None": "unaffiliated",
}


def _key(cat):
    # CSO prints a CURLY apostrophe in `Jehovah’s Witness`. Normalise it, so a future
    # vintage that uses a straight one still resolves (§12: never map on the string alone).
    return " ".join(str(cat).replace("’", "'").split())


EXCLUDED = {_key(k): v for k, v in EXCLUDED.items()}
MAP = {_key(k): v for k, v in MAP.items()}
REVIEW = {_key(k): v for k, v in REVIEW.items()}


def resolve(cat):
    """Source category -> taxonomy node id, or None if deliberately not on the tree.

    **`None` IS A CATEGORY NAME HERE, NOT A MISSING VALUE.** Any caller that reads tt.csv
    with a bare `pandas.read_csv` will have turned those rows into NaN before this function
    sees them — see `_tt_counts` in countries.py, which passes `keep_default_na=False`.
    """
    c = _key(cat)
    if c in EXCLUDED:
        return None
    return MAP.get(c)
