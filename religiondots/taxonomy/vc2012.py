"""SVG Statistical Office 2012 census religion classification -> religiondots taxonomy.

Eighteen categories on 221 enumeration districts — **the finest geography in the project per
head**, a median of 415 people and 0.66 km² per unit. Saint Vincent is the companion §11j
asked for beside Jamaica (§9ab), and between them the two carry a Caribbean Protestant
picture that exists nowhere else on this map.

    27.57%  Pentecostal            -> christianity.pentecostal
    13.90%  Anglican               -> christianity.anglican
    11.64%  Seventh Day Adventist  -> christianity.adventist
     8.86%  Baptist                -> christianity.baptist
     8.66%  Methodist              -> christianity.methodist
     7.46%  Without religion       -> unaffiliated
     6.30%  Roman Catholic         -> christianity.catholic
     4.67%  Not stated             -> EXCLUDED
     4.28%  Other religion         -> other.vc   (a NEW node)
     3.77%  Evangelical Christian  -> christianity.evangelical
     1.08%  Rastafarian            -> rastafari
     0.83%  Jehovah's Witness      -> christianity.witnesses
     0.27%  Presbyterian           -> christianity.reformed.presbyterian
     0.26%  Salvation Army         -> christianity.methodist.holiness
     0.19%  Mormon                 -> christianity.latterday
     0.10%  Muslim                 -> islam
     0.08%  Hindu                  -> hinduism
     0.07%  Traditional            -> other.vc                          REVIEW

**SIX CATEGORIES ARE UNDER 400 PEOPLE AND THAT IS THE POINT OF THE COUNTRY.** Presbyterian
294, Salvation Army 287, Mormon 207, Muslim 111, Hindu 89, Traditional 74. At 1:1,000 none of
them draws a dot; §4.3's presence rings are what puts them on the map, and this is the source
that makes rings do real work. A census that publishes single people on units of 415 is why
they are visible at all.

EXCLUDED holds categories that are deliberately not on the tree.
REVIEW holds calls that are defensible but arguable, with the reason.
"""

EXCLUDED = {
    "Not stated":
        "5,095 people, 4.67%, for whom no religion was recorded. The only non-answer cell — "
        "SVG offers no separate refusal category, unlike Bosnia's `Nisu se izjasnili` / "
        "`Bez odgovora` pair — so there is nothing here to keep apart. Spec §3.5: marked, "
        "not filled. It is not concentrated anywhere that suggests an enumeration failure.",
}

REVIEW = {
    "Traditional":
        "-> other.vc, and the interesting part is what it is NOT. 74 people, 0.07%, in 25 of "
        "221 enumeration districts. **The obvious reading is the Kalinago (Carib) "
        "population's traditional practice, and the data rules that out.** The same sheet "
        "carries ethnicity, so §12's Philippines technique applies directly: across the 219 "
        "populated EDs, Traditional's share correlates with the Indigenous share at "
        "**r = -0.03**, and — decisively — **every one of the eight most-indigenous EDs has "
        "ZERO Traditional**, including the three Sandy Bay districts at 79-86% Indigenous. "
        "Traditional's largest cells are in Georgetown, Northern Grenadines and Kingstown, "
        "which are 75-83% Black and 0-1% Indigenous. So it is not `indigenous`. "
        "**What it IS cannot be established from this source.** The plausible readings are "
        "African-derived practice (Obeah) and the Spiritual Baptist / 'Converted' tradition "
        "for which Saint Vincent is the home — the latter was criminalised here from 1912 to "
        "1965 — but the census offers a `Baptist` cell that may already absorb the Shakers, "
        "and 74 people across 25 districts carries no signal either way. Filed in the "
        "residual rather than guessed at (§14.4). **The node that would be wanted if it were "
        "ever resolved is `afrodiasporic`**, beside Jamaica's Revival Zion — recorded per "
        "§2.4 so a future source makes this a lookup rather than an investigation.",
    "Evangelical Christian":
        "-> christianity.evangelical, the 'named an answer, not a body' node, which is "
        "exactly what this cell is. 4,119 people. Kept apart from `Pentecostal`, which the "
        "census counts separately and which is seven times larger.",
    "Salvation Army":
        "-> christianity.methodist.holiness. The Salvation Army is a Holiness body of "
        "Methodist descent — Booth was a Methodist New Connexion minister and its doctrine "
        "is Wesleyan — so it sits under Methodism rather than in the general `holiness` "
        "node, which branches.py keeps for bodies without that lineage. 287 people.",
    "Presbyterian":
        "-> christianity.reformed.presbyterian. 294 people, and one of the smallest "
        "categories any source here names. Saint Vincent's Presbyterians are largely the "
        "descendants of the Scottish planter and missionary presence.",
    "Mormon":
        "-> christianity.latterday. The census's own word; the tree's node covers the "
        "Latter Day Saint movement. 207 people.",
    "Muslim":
        "-> islam, the parent, with no branch. 111 people and nothing says which tradition.",
    "Hindu":
        "-> hinduism, the parent. 89 people. Saint Vincent's East Indian population is "
        "1.10% of the country and descends from post-emancipation indenture; **almost all "
        "of it is now Christian**, which the numbers show plainly — 1,199 East Indians "
        "against 89 Hindus. The comparison is available because ethnicity and religion sit "
        "in the same sheet.",
    "Rastafarian":
        "-> rastafari, a ROOT in branches.py rather than a branch of Christianity. 1,181 "
        "people, 1.08% — **almost exactly Jamaica's share (1.08%)**, which is a striking "
        "agreement between two independently designed censuses. And it is *dispersed* where "
        "the small religions here are concentrated: present in **186 of 221 enumeration "
        "districts**, against Hindu in 26 and Traditional in 25. Read as a floor, for the "
        "reasons jm2011.py gives.",
    "Without religion":
        "-> unaffiliated, and NOT `secular`. One no-religion answer with no atheist or "
        "agnostic split, so the coarser node is the honest one. 8,147 people, 7.46% — a "
        "third of Jamaica's 21.35%, in a country 100 km away, which is the sharpest "
        "irreligion contrast between neighbours anywhere on this map.",
    "Other religion":
        "-> other.vc, a per-source residual (§3.11). 4,672 people, 4.28%. See branches.py.",
}

MAP = {
    "Anglican": "christianity.anglican",
    "Evangelical Christian": "christianity.evangelical",
    "Methodist": "christianity.methodist",
    "Pentecostal": "christianity.pentecostal",
    "Presbyterian": "christianity.reformed.presbyterian",
    "Roman Catholic": "christianity.catholic",
    "Salvation Army": "christianity.methodist.holiness",
    "Seventh Day Adventist": "christianity.adventist",
    "Jehovah's Witness": "christianity.witnesses",
    "Baptist": "christianity.baptist",
    "Hindu": "hinduism",
    "Mormon": "christianity.latterday",
    "Muslim": "islam",
    "Rastafarian": "rastafari",
    "Traditional": "other.vc",
    "Without religion": "unaffiliated",
    "Other religion": "other.vc",
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
