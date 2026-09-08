"""SIB 2022 census religion classification -> religiondots taxonomy.

Twelve categories on 6 districts. Nine named Christian bodies, one residual, one
no-religion answer and one non-answer — a short list by Caribbean standards (Jamaica names
19, Trinidad 14, Saint Vincent 18), and the shortness is the thing to keep in view when
reading the map.

    31.85%  Roman Catholic          -> christianity.catholic
    31.04%  None                    -> unaffiliated
     9.17%  Pentecostal             -> christianity.pentecostal
     6.32%  Other                   -> other.bz   (a NEW node)
     4.69%  Seventh Day Adventist   -> christianity.adventist
     4.01%  Anglican                -> christianity.anglican
     3.88%  Mennonite               -> christianity.anabaptist.mennonite
     3.55%  Baptist                 -> christianity.baptist
     1.67%  Methodist               -> christianity.methodist
     1.65%  Nazarene                -> christianity.holiness      REVIEW
     1.13%  Jehovah's Witness       -> christianity.witnesses
     1.04%  Don't Know/Not Stated   -> EXCLUDED

**`Mennonite` IS WHY BELIZE IS DRAWN.** 15,440 people, and **no other census on this map
names Mennonites at all** — `christianity.anabaptist.mennonite` has existed since the United
States arrived and only ASARB has ever put anyone in it. Belize's are the Kleine Gemeinde,
Old Colony and Noah Martin settlements that came from Mexico, Canada and the United States
from 1958, and they are concentrated exactly where the history says: **9.9% of Orange Walk
and 8.9% of Corozal**, against 0.5% of Belize District and 0.5% of Stann Creek.

**`None` AT 31.04% IS THE HIGHEST IRRELIGIOUS SHARE THIS MAP DRAWS IN THE AMERICAS**, above
Jamaica's 21.35%, and it is not flat: **46.6% in Stann Creek** against 21.7% in Toledo. See
`sources/bz.md` §4 — it is a large rise on 2010 and the reasons offered for it are not in
this source.

EXCLUDED holds categories that are deliberately not on the tree.
REVIEW holds calls that are defensible but arguable, with the reason.
"""

EXCLUDED = {
    "Total":
        "the district's own population total, not a category. Carried in bz.csv because "
        "sources/bz.py checks the twelve categories against it, and because it is the "
        "denominator every share in this file is quoted against.",
    "Don't Know/Not Stated":
        "4,135 people, 1.04%. SIB pools 'don't know' with 'not stated' in one cell, so — "
        "unlike Croatia and Bosnia, which publish a refusal separately from a blank — there "
        "is no pair to keep apart and nothing here can distinguish the two. Spec §3.5: "
        "marked, not filled. **Its geography is mild and urban** — 1.7% in Belize District "
        "against 0.3% in Toledo — which is the ordinary pattern and not the signature of an "
        "enumeration failure anywhere in particular.",
}

REVIEW = {
    "Nazarene":
        "-> christianity.holiness. The **Church of the Nazarene**, which is the flagship "
        "body of the Holiness movement and the one the node was made for — it left "
        "Methodism over entire sanctification and explicitly rejected the Pentecostal "
        "tongues doctrine in 1919, dropping 'Pentecostal' from its own name to say so. "
        "6,568 people. Filed at `christianity.holiness` rather than given a child of its "
        "own because no other source here splits the Holiness family and a one-country "
        "child would be a node with a single occupant. Flagged because the name is a "
        "denomination and the node is a movement, which is the shape of call §12 warns "
        "about.",
    "Other":
        "-> other.bz, a per-source residual (§3.11). 25,117 people, 6.32%, and **its "
        "geography is sharp rather than flat** — 11.9% in Orange Walk against 2.7% in "
        "Stann Creek, a 4.4x spread — which by §9r's rule makes it a missing category "
        "rather than a mixture of small ones. What is inside it is **not published**, and "
        "it is left as an open question rather than guessed at. What is known to exist in "
        "Belize and is not named anywhere in this table: Hindus and Muslims (largely Belize "
        "City and the north), a Bahá'í community, Rastafari, and Maya traditional practice "
        "in Toledo. Any of those could be in `Other` or, for the syncretic ones, inside a "
        "named Christian cell; the census does not say and nothing here assumes.",
    "Pentecostal":
        "-> christianity.pentecostal. One undivided cell of 36,460, and Belize's "
        "Pentecostalism is mostly Assemblies of God and Church of God (Cleveland) — but "
        "SIB names neither, so the parent is the honest grain. Its geography is the "
        "opposite of the Catholic one: **14.9% in Cayo and 13.3% in Toledo** against 5.3% "
        "in Corozal.",
    "Baptist":
        "-> christianity.baptist. 14,109 people and **12.0% of Toledo** against 0.9% of "
        "Orange Walk — by far the most concentrated of the named Christian bodies. Toledo "
        "is the Maya south, and the Baptist presence there is missionary in origin. One "
        "cell, no sub-division published.",
}

MAP = {
    "Roman Catholic": "christianity.catholic",
    "Pentecostal": "christianity.pentecostal",
    "Seventh Day Adventist": "christianity.adventist",
    "Anglican": "christianity.anglican",
    "Mennonite": "christianity.anabaptist.mennonite",
    "Baptist": "christianity.baptist",
    "Methodist": "christianity.methodist",
    "Nazarene": "christianity.holiness",
    "Jehovah's Witness": "christianity.witnesses",
    "Other": "other.bz",
    "None": "unaffiliated",
}


def _key(cat):
    return " ".join(str(cat).split())


EXCLUDED = {_key(k): v for k, v in EXCLUDED.items()}
MAP = {_key(k): v for k, v in MAP.items()}
REVIEW = {_key(k): v for k, v in REVIEW.items()}


def resolve(cat):
    """Source category -> taxonomy node id, or None if deliberately not on the tree.

    **`None` IS A CATEGORY NAME HERE, NOT A MISSING VALUE**, and it is the second largest
    one. Any caller that reads bz.csv with a bare `pandas.read_csv` will have turned those
    six rows into NaN before this function ever sees them — see `_bz_counts` in
    countries.py, which passes `keep_default_na=False` for exactly this reason.
    """
    c = _key(cat)
    if c in EXCLUDED:
        return None
    return MAP.get(c)
