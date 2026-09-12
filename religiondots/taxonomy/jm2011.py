"""STATIN 2011 census religion classification -> religiondots taxonomy.

Nineteen categories on 14 parishes. **The best denominational detail in the Americas outside
ASARB**, and the reason Jamaica is drawn at all — `sources.md` §11j put it as *"Kenya was
drawn on 47 units for its categories, and Jamaica's are better while its units are three
times fewer"*, and spec §3.9b removed the rule that was holding it back.

    21.35%  Non-religious                -> unaffiliated
    12.03%  Seventh Day Adventist        -> christianity.adventist
    11.02%  Pentecostal                  -> christianity.pentecostal
     9.21%  Other Church of God          -> christianity.holiness      REVIEW
     7.17%  New Testament Church of God  -> christianity.pentecostal
     6.74%  Baptist                      -> christianity.baptist
     6.31%  Other religion               -> other.jm   (a NEW node)
     4.83%  Church of God in Jamaica     -> christianity.holiness      REVIEW
     4.53%  Church of God of Prophecy    -> christianity.pentecostal
     2.79%  Anglican                     -> christianity.anglican
     2.25%  No Data                      -> EXCLUDED
     2.16%  Roman Catholic               -> christianity.catholic
     2.07%  United Church                -> christianity.united
     1.90%  Jehovah's Witness            -> christianity.witnesses
     1.62%  Methodist                    -> christianity.methodist
     1.35%  Revivalist                   -> afrodiasporic.revival  (a NEW node)
     1.08%  Rastafarian                  -> rastafari
     0.88%  Brethren                     -> christianity.plymouth
     0.68%  Moravian                     -> christianity.moravian

**THREE OF THESE EXIST NOWHERE ELSE ON THIS MAP AND THEY ARE THE POINT OF THE COUNTRY.**
`rastafari` has been a root since Czechia counted **190** of them; Jamaica counts **29,026**
and is where the religion began. `afrodiasporic.revival` is new. And the four Church of God
bodies kept apart are **689,868 people, 25.7% of Jamaica** — the largest religious bloc in
the country, and no other source here splits that family at all.

**WHAT THIS FILE CANNOT MAP, BECAUSE IT IS NOT IN THE TABLE.** Jamaica's Bahá'ís, Hindus,
Muslims and Jews — 4,124 people — were excluded from the parish tables by STATIN and are
absent from the source entirely (`sources/jm.py`). There is no category here to map them to
and nothing is invented to hold them.

EXCLUDED holds categories that are deliberately not on the tree.
REVIEW holds calls that are defensible but arguable, with the reason.
"""

EXCLUDED = {
    "No Data":
        "60,326 people, 2.25%, for whom the variable was not established. A coverage "
        "residual rather than a refusal — STATIN offers no 'prefer not to say' cell, so "
        "unlike Croatia and Bosnia there is no pair to keep apart here, and this is the "
        "only non-answer. Spec §3.5: marked, not filled. **Its geography is mild** — 0.8% "
        "in Portland to 3.5% in Kingston — which is the ordinary urban pattern and not the "
        "sign of an enumeration failure anywhere in particular.",
}

REVIEW = {
    "Other Church of God":
        "-> christianity.holiness, and this is the least certain call in the file. STATIN "
        "splits the Church of God family into four cells and this is its residual — "
        "246,838 people, 9.21%, **the largest single Christian cell in Jamaica**. The "
        "family spans the Holiness/Pentecostal continuum and the named siblings sit on "
        "both sides of it, so the residual necessarily contains both. `christianity."
        "holiness` is chosen over `christianity.pentecostal` because the parent of the "
        "whole family is the Holiness movement and the Pentecostal members are the ones "
        "that could be named; a residual is more likely to hold the older and smaller "
        "bodies. **What would settle it is a body-level table, which STATIN does not "
        "publish** — noted per §2.4 so the fix is a lookup rather than an investigation.",
    "Church of God in Jamaica":
        "-> christianity.holiness. The Jamaican body in the Church of God (Anderson, "
        "Indiana) line, which is Holiness and explicitly not Pentecostal — it rejected the "
        "tongues doctrine at the point the movement divided. 129,544 people. Filed apart "
        "from its two Cleveland-lineage siblings for that reason, and flagged because the "
        "name alone does not carry the distinction and §12 warns against mapping a "
        "category on its string.",
    "New Testament Church of God":
        "-> christianity.pentecostal. The Caribbean and British name of the **Church of "
        "God (Cleveland, Tennessee)**, which is Pentecostal-Holiness and documented as "
        "such. 192,086 people, and the clearest of the four.",
    "Church of God of Prophecy":
        "-> christianity.pentecostal. Also Cleveland, Tennessee — it separated from the "
        "body above in the 1923 split and is Pentecostal on both sides of that. 121,400 "
        "people.",
    "Brethren":
        "-> christianity.plymouth, NOT christianity.anabaptist.brethren. Jamaica's Brethren "
        "assemblies are the **Plymouth/Christian Brethren**, an evangelical movement out of "
        "1820s Dublin, and they arrived through 19th-century British missions. The "
        "Schwarzenau (German Baptist) Brethren are a different and unrelated body with no "
        "Jamaican presence, and the tree holds both — this is exactly the collision §12 "
        "means by *never map a category on its string alone*. 23,647 people.",
    "United Church":
        "-> christianity.united. The **United Church in Jamaica and the Cayman Islands**, "
        "formed 1965/1992 from the Presbyterian, Congregational and Disciples of Christ "
        "traditions. `christianity.united` is the node for uniting churches, and it is "
        "right here rather than `christianity.reformed`, which would pick one of the three "
        "ancestors and drop the other two. 55,360 people.",
    "Revivalist":
        "-> afrodiasporic.revival, a NEW node. Revival Zion and Pukkumina, out of the Great "
        "Revival of 1860-61. Filed with Umbanda and Candomblé rather than under "
        "Christianity, per spec §3.3 — see branches.py for the argument. **Its geography "
        "supports the placement**: it peaks at **3.58% in Saint Thomas** against 1.35% "
        "nationally and 0.68% in Saint Ann, and eastern Jamaica is where the Kongo-derived "
        "practice concentrated. A purely Christian revival would not have that shape.",
    "Rastafarian":
        "-> rastafari, which is a ROOT in branches.py and not a branch of Christianity. "
        "29,026 people. **This is the node's home**: it was created for Czechia's 190 and "
        "has been waiting for this source. Read the number as a floor rather than a count — "
        "Rastafari is a way of life more than a membership, census enumeration of it in "
        "Jamaica is widely held to undercount, and the 1.08% here is far below any "
        "cultural estimate. Not corrected, per §14.4; said in `note_public`.",
    "Non-religious":
        "-> unaffiliated, and NOT `secular`. STATIN offers one no-religion answer with no "
        "atheist or agnostic split, so the coarser node is the honest one. **572,008 "
        "people, 21.35% — the highest irreligious share of any country on this map in the "
        "Americas**, and it reaches **34.06% in Kingston** against 11.84% in Manchester.",
    "Other religion":
        "-> other.jm, a per-source residual (§3.11). 6.30%, and its geography is sharp "
        "rather than flat — 14.18% in Westmoreland against 2.92% in Kingston. See "
        "branches.py.",
}

MAP = {
    "Anglican": "christianity.anglican",
    "Baptist": "christianity.baptist",
    "Brethren": "christianity.plymouth",
    "Church of God in Jamaica": "christianity.holiness",
    "Church of God of Prophecy": "christianity.pentecostal",
    "New Testament Church of God": "christianity.pentecostal",
    "Other Church of God": "christianity.holiness",
    "Jehovah's Witness": "christianity.witnesses",
    "Methodist": "christianity.methodist",
    "Moravian": "christianity.moravian",
    "Pentecostal": "christianity.pentecostal",
    "Rastafarian": "rastafari",
    "Revivalist": "afrodiasporic.revival",
    "Roman Catholic": "christianity.catholic",
    "Seventh Day Adventist": "christianity.adventist",
    "United Church": "christianity.united",
    "Other religion": "other.jm",
    "Non-religious": "unaffiliated",
}


def _key(cat):
    return " ".join(str(cat).split())


EXCLUDED = {_key(k): v for k, v in EXCLUDED.items()}
MAP = {_key(k): v for k, v in MAP.items()}
REVIEW = {_key(k): v for k, v in REVIEW.items()}


def resolve(cat):
    """Source category -> taxonomy node id, or None if deliberately not on the tree."""
    c = _key(cat)
    if c in EXCLUDED:
        return None
    return MAP.get(c)
