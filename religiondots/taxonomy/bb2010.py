"""BSS 2010 census religion classification -> religiondots taxonomy.

Twenty-two named categories plus a non-answer, on 11 parishes. Shares are of the tabulable
population, 226,193, which is 81.4% of the estimated resident population — see
`sources/bb.py` on the undercount, and `bb.md` §1 on why 2010 is drawn and not 2021.

    23.87%  Anglican                  -> christianity.anglican
    20.59%  No Religious Affiliation  -> unaffiliated
    19.49%  Other Pentecostal         -> christianity.pentecostal
     5.94%  Adventist                 -> christianity.adventist
     4.18%  Methodist                 -> christianity.methodist
     3.84%  Roman Catholic            -> christianity.catholic
     3.40%  Wesleyan                  -> christianity.holiness.wesleyan
     3.35%  Other Christian           -> christianity.other
     3.23%  Nazarene                  -> christianity.holiness.nazarene   REVIEW
     2.37%  Church of God             -> christianity.holiness            REVIEW
     2.00%  Jehovah Witness           -> christianity.witnesses
     1.81%  Baptist                   -> christianity.baptist
     1.23%  Not Stated                -> EXCLUDED
     1.19%  Moravian                  -> christianity.moravian
     1.03%  Rastafarian               -> rastafari
     0.71%  Muslim                    -> islam
     0.47%  Brethren                  -> christianity.plymouth            REVIEW
     0.47%  Hindu                     -> hinduism
     0.39%  Salvation Army            -> christianity.holiness.salvation-army
     0.27%  Other Non-Christian       -> other.bb   (a NEW node)
     0.10%  Mormon                    -> christianity.latterday
     0.04%  Baha'i                    -> bahai
     0.04%  Jewish                    -> judaism

**BARBADOS IS THE FIRST CENSUS OUTSIDE THE UNITED STATES TO REACH THREE OF THESE NODES.**
`christianity.holiness.nazarene`, `christianity.holiness.wesleyan` and
`christianity.holiness.salvation-army` have existed since the U.S. Religion Census arrived
and nothing else here had a cell for any of them. Barbados counts all three separately —
**7,299 Nazarenes, 7,694 Wesleyans and 878 Salvationists** — and a fourth Holiness answer,
`Church of God`, on top. Together that is **9.4% of the country in one family, split four
ways**, which no other source on this map does at all.

**AND IT IS THE MOST ANGLICAN COUNTRY HERE.** 23.87%, against 13.9% in Saint Vincent, 11.9%
in the Bahamas, 5.7% in Trinidad and 2.8% in Jamaica. "Little England" is still legible:
**St. John is 37.5% Anglican** and the parish churches are the oldest institutions on the
island.

**THE TWO BIG ANSWERS RUN OPPOSITE TO EACH OTHER ACROSS THE ISLAND.** Anglicanism peaks in
St. John (37.5%) and bottoms in St. Andrew (15.3%); `Other Pentecostal` does the reverse —
**29.4% of St. Andrew** against the island's 19.5%. The old established church holds the
south and east; the Pentecostal churches hold the rugged north-centre, which is also the
poorest part of Barbados.

EXCLUDED holds categories that are deliberately not on the tree.
REVIEW holds calls that are defensible but arguable, with the reason.
"""

EXCLUDED = {
    "Total":
        "the parish's own population total, not a category. Carried in bb.csv because "
        "sources/bb.py checks the categories against it.",
    "Not Stated":
        "2,774 people, **1.23%** — one of the smallest non-answers on this map, and far "
        "smaller than what Barbados is actually missing. Spec §3.5: marked, not filled. "
        "**The real gap is the census's own 18% undercount**, which is a different thing "
        "and is not a refusal: 49,115 people were never enumerated at all, and BSS "
        "publishes an estimated resident population by parish showing where. Coverage runs "
        "from 74.6% of St. James to 96.1% of St. John, so an under-covered parish draws "
        "proportionally fewer dots than its true population warrants. Nothing here scales "
        "it (§14.4) and `sources/bb.py` prints the whole ladder.",
}

REVIEW = {
    "Nazarene":
        "-> christianity.holiness.nazarene, and **not** anything under "
        "`christianity.anabaptist`. 7,299 people, 3.23%. "
        "**This is a §12 string collision with a real trap in it.** The U.S. Religion "
        "Census names FOUR bodies containing the word Nazarene: the **Church of the "
        "Nazarene** (905,690 adherents, Holiness, the Wesleyan-holiness denomination out of "
        "1908 Pilot Point) and three Apostolic Christian bodies — `Apostolic Christian "
        "Church (Nazarene) - East Conference`, `- West Conference` and `Nazarene Christian "
        "Congregation` — which are **Anabaptist** and unrelated. Barbados's is the Church "
        "of the Nazarene: it has had a Barbadian district since the 1920s, and the "
        "Anabaptist bodies have no Caribbean presence at all. "
        "**The census counts more Nazarenes in Barbados than Anglicans in St. Andrew**, "
        "which is the kind of thing this cell being separate makes visible. It peaks in "
        "St. George (5.1%) and St. Philip (4.8%).",
    "Church of God":
        "-> christianity.holiness, the PARENT, and deliberately not one of its children. "
        "5,356 people, 2.37%. BSS offers one unqualified cell, and unlike the Cayman "
        "Islands (`ky2021.py`) there is no external evidence naming which Church of God "
        "body Barbados means — the Cleveland (Pentecostal) and Anderson (Holiness) lines "
        "both have Caribbean presences and the census does not say. "
        "**The parent is the honest grain for an unidentified cell in this family**, which "
        "is `jm2011.py`'s call for STATIN's residual `Other Church of God` and the same "
        "reasoning: the family's parent is the Holiness movement, and the Pentecostal "
        "members are the ones a form usually names. **This does mean it sits at the same "
        "node as nothing else in Barbados**, since Nazarene, Wesleyan and the Salvation "
        "Army all have children of their own here — so the map shows it as an "
        "undifferentiated Holiness cell beside three named ones, which is exactly what the "
        "source supports. Its geography is the north and centre: St. Andrew 6.0%, "
        "St. Peter 5.6%, St. Thomas 4.4%, against 0.8% in St. John.",
    "Brethren":
        "-> christianity.plymouth, NOT christianity.anabaptist.brethren. 1,074 people. The "
        "same Caribbean collision `jm2011.py` and `bs2022.py` document: the Brethren "
        "assemblies of the Anglophone Caribbean are the **Plymouth / Christian Brethren**, "
        "out of 1820s Dublin and arriving through nineteenth-century British missions, and "
        "the Schwarzenau (German Baptist) Brethren have no Barbadian presence. §12: never "
        "map a category on its string alone.",
    "Other Pentecostal":
        "-> christianity.pentecostal. **44,084 people, 19.49% — the second largest answer "
        "in the country**, and the word `Other` in its label is misleading: there is no "
        "un-othered Pentecostal cell above it. The list names Church of God, Nazarene, "
        "Wesleyan and the Salvation Army separately, so `Other Pentecostal` is what is left "
        "of the Pentecostal and holiness-Pentecostal field once those four are taken out — "
        "the Assemblies of God, the People's Cathedral, the Abundant Life and independent "
        "charismatic congregations that grew fastest in Barbados after the 1970s. It is "
        "filed at the branch parent rather than a child because the cell names no body.",
    "Other Christian":
        "-> christianity.other. 7,567 people, 3.35%, and a genuine tail rather than a "
        "bucket: fifteen Christian bodies are named above it, including four in one family. "
        "Its geography is flat — 3.9% in Christ Church down to about 2% — which by §9r's "
        "rule makes it a mixture rather than a missing category.",
    "Other Non-Christian":
        "-> other.bb, a per-source residual (§3.11). 622 people, **0.27% — the narrowest "
        "residual on this map**, because the form already names Bahá'í, Hindu, Jewish, "
        "Muslim and Rastafari. What is left is a few hundred people and it is not guessed "
        "at; likely contents are in branches.py.",
}

MAP = {
    "Adventist": "christianity.adventist",
    "Anglican": "christianity.anglican",
    "Baptist": "christianity.baptist",
    "Brethren": "christianity.plymouth",
    "Church of God": "christianity.holiness",
    "Jehovah Witness": "christianity.witnesses",
    "Methodist": "christianity.methodist",
    "Moravian": "christianity.moravian",
    "Mormon": "christianity.latterday",
    "Nazarene": "christianity.holiness.nazarene",
    "Other Pentecostal": "christianity.pentecostal",
    "Roman Catholic": "christianity.catholic",
    "Salvation Army": "christianity.holiness.salvation-army",
    "Wesleyan": "christianity.holiness.wesleyan",
    "Other Christian": "christianity.other",
    "Baha'i": "bahai",
    "Hindu": "hinduism",
    "Jewish": "judaism",
    "Muslim": "islam",
    "Rastafarian": "rastafari",
    "Other Non-Christian": "other.bb",
    "No Religious Affiliation": "unaffiliated",
}


def _key(cat):
    """Normalise the apostrophe and whitespace.

    BSS spells `Baha'i` with a straight apostrophe in the 2010 sheet and a curly one in the
    2021 sheet, and `Jehovah Witness` gains a possessive in 2021. `sources/bb.py` reads the
    2010 spellings; folding here means a future switch to the 2021 vintage does not silently
    unmap two categories (§12).
    """
    return " ".join(str(cat).replace("’", "'").split())


EXCLUDED = {_key(k): v for k, v in EXCLUDED.items()}
MAP = {_key(k): v for k, v in MAP.items()}
REVIEW = {_key(k): v for k, v in REVIEW.items()}


def resolve(cat):
    """Source category -> taxonomy node id, or None if deliberately not on the tree.

    Barbados's no-religion cell is spelled `No Religious Affiliation`, so unlike Belize,
    Trinidad, the Bahamas and Cayman it is NOT the literal string `None` and survives a
    bare `pandas.read_csv`. `_bb_counts` in countries.py still passes
    `keep_default_na=False`, for consistency and because `Not Stated` would otherwise be
    the next thing to go wrong.
    """
    c = _key(cat)
    if c in EXCLUDED:
        return None
    return MAP.get(c)
