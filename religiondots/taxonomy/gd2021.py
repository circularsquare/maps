"""CSO 2021 census religion classification -> religiondots taxonomy.

Twenty-five named categories plus a non-answer, on 8 census units drawn as 7. Shares are of
the non-institutional population in private dwellings, 108,279 — 99.3% of the census's own
total, which is the cleanest universe of any Caribbean source here.

    31.53%  ROMAN CATHOLIC            -> christianity.catholic
    19.93%  PENTECOSTAL               -> christianity.pentecostal
    12.32%  SEVENTH DAY ADVENTIST     -> christianity.adventist
     7.31%  ANGLICAN                  -> christianity.anglican
     7.11%  NOT STATED                -> EXCLUDED
     5.95%  NO RELIGIOUS AFFILIATION  -> unaffiliated
     3.60%  CHURCH OF GOD             -> christianity.holiness              REVIEW
     2.36%  EVANGELICAL               -> christianity.evangelical
     1.70%  SPIRITUAL BAPTIST         -> afrodiasporic.spiritualbaptist     REVIEW
     1.58%  OTHER (SPECIFY)           -> other.gd   (a NEW node)
     1.50%  INDEPENDENT BAPTISTE      -> christianity.baptist               REVIEW
     1.26%  METHODIST                 -> christianity.methodist
     1.07%  RASTAFARIAN               -> rastafari
     1.01%  JEHOVAH WITNESSES         -> christianity.witnesses
     0.43%  PRESBYTERIAN              -> christianity.reformed.presbyterian REVIEW
     0.37%  MUSLIM                    -> islam
     0.29%  BRETHREN                  -> christianity.plymouth              REVIEW
     0.26%  MENNONITE                 -> christianity.anabaptist.mennonite  REVIEW
     0.14%  HINDU                     -> hinduism
     0.09%  MORMOM                    -> christianity.latterday             REVIEW
     0.05%  ATHEIST                   -> secular                            REVIEW
     0.05%  SALVATION ARMY            -> christianity.holiness.salvation-army
     0.04%  LUTHERAN                  -> christianity.lutheran
     0.02%  BUDDHIST                  -> buddhism
     0.02%  MORAVIAN                  -> christianity.moravian
     0.01%  BAHAI                     -> bahai

**GRENADA IS THE MOST EVENLY DIVIDED CARIBBEAN COUNTRY ON THIS MAP.** Its largest religion
is 31.5% — against Saint Lucia's 50.6% Catholic 150 km away, Jamaica's 24.9%, the Bahamas'
34.9% Baptist. **Four bodies hold more than 7% each** and no two of them are the same
tradition: Roman Catholic 31.5%, Pentecostal 19.9%, Seventh Day Adventist 12.3%, Anglican
7.3%. The French and British both held the island and both left a church behind, and the
twentieth-century missions landed on top of the pair.

**THE ADVENTISTS ARE A NORTHERN COUNTRY AND THE PENTECOSTALS A SOUTHERN ONE.** Seventh Day
Adventists are **24.1% of St. Andrew and 22.6% of St. Patrick**, the north and north-east,
against **7.3% of St. George** and 7.3% of Carriacou. Pentecostals do the reverse: **25.9%
of St. David and 23.0% of St. Andrew** against **7.2% of Carriacou** and 8.0% of St. John.
The two largest Protestant streams in the country barely overlap.

**AND CARRIACOU IS A DIFFERENT ISLAND IN THE RELIGIOUS SENSE TOO.** **22.1% Anglican** —
seven times St. David's 3.0% and three times the national 7.3% — and 40.6% Catholic, with
Pentecostals at a third of their national share. That is the Scottish and English settlement
of the Grenadines still legible three centuries on, and it is the strongest single-unit
signal in the country.

**THE FORM SPLITS DISBELIEF FROM NON-AFFILIATION**, as Saint Lucia's does (`lc2022.py`):
`NO RELIGIOUS AFFILIATION` 5.95% against `ATHEIST` 0.05%, **a 130-fold gap**, the widest
either country produces. Two censuses that asked both questions and got the same answer.

EXCLUDED holds categories that are deliberately not on the tree.
REVIEW holds calls that are defensible but arguable, with the reason.
"""

EXCLUDED = {
    "TOTAL":
        "the unit's own population total, not a category. Carried in gd.csv because "
        "sources/gd.py checks the categories against it.",
    "NOT STATED":
        "7,698 people, **7.11%**, and one of the largest non-answers on this map. Spec "
        "§3.5: marked, not filled. "
        "**Its geography is the sharpest thing in the country and the map cannot show all "
        "of it.** 10.2% of St. George against **0.91% of St. Mark** — an elevenfold spread. "
        "Inside St. George it is sharper still: the census reports the **Town of St. George "
        "at 15.8%** against 9.9% for the rest of the parish, and the town has no boundary "
        "anywhere so the two are folded together before drawing (`sources/gd.md` §3). The "
        "capital's refusal rate is the one finding this country's geography loses. "
        "**It is not the same thing as an undercount**: Grenada enumerated 109,021 people "
        "and tabulates 108,279 of them, so the missing 0.7% is institutional and homeless "
        "population rather than a coverage gap. These 7,698 were counted and declined the "
        "question.",
}

REVIEW = {
    "CHURCH OF GOD":
        "-> christianity.holiness, the PARENT, and deliberately not one of its children. "
        "3,900 people, 3.60%. "
        "**The name alone cannot decide it and §12 says so**: `Church of God (Cleveland, "
        "Tennessee)` is Pentecostal and `Church of God (Anderson, Indiana)` is Holiness and "
        "rejected the tongues doctrine when the movement divided. Both lines are present in "
        "the eastern Caribbean and CSO offers one unqualified cell. "
        "**Unlike Cayman (`ky2021.py`), no external evidence names which body Grenada "
        "means**, so the call falls back to `bb2010.py`'s: the parent is the honest grain "
        "for an unidentified cell in this family, and the Pentecostal members are the ones "
        "a form usually names — as this one does, separately, at 19.93%. "
        "**Its geography is one parish.** 8.48% of St. Andrew against 0.66% of St. Mark, "
        "1.41% of St. David and 1.50% of Carriacou — a thirteenfold spread, which for a "
        "cell this size means a specific set of congregations in Grenville and the "
        "north-east rather than a national body.",
    "SPIRITUAL BAPTIST":
        "-> afrodiasporic.spiritualbaptist, **not** `christianity.baptist`, and the node "
        "exists because Trinidad reached it first (`tt2011.py`). 1,843 people, 1.70%. "
        "**The structural argument is the same one and it is stronger here**: CSO offers "
        "`SPIRITUAL BAPTIST` and `INDEPENDENT BAPTISTE` as two separate answers, so filing "
        "them together would merge two cells the source deliberately kept apart. The "
        "Spiritual or Shouter Baptists are an Afro-Caribbean tradition of Baptist descent "
        "whose practice — bell, candles, adoption of the mourning ground, spirit "
        "possession — is not Baptist worship, and Trinidad marks their history as a matter "
        "of state: the churches were banned by the Shouters Prohibition Ordinance from 1917 "
        "to 1951. "
        "**The objection is recorded rather than dismissed**, as in `tt2011.py`: many "
        "Spiritual Baptists would describe themselves as Christians and nothing else, and "
        "the node's placement outside Christianity is a claim about the tradition rather "
        "than about them. "
        "**Grenada's geography for it is St. Mark, 4.77%**, nearly three times the national "
        "rate, with St. David at 3.01%; lowest in Carriacou at 0.80%.",
    "INDEPENDENT BAPTISTE":
        "-> christianity.baptist. 1,625 people, 1.50%. **Grenada has no plain `BAPTIST` "
        "cell** — the form offers only this and `SPIRITUAL BAPTIST` — so this is the "
        "ordinary Baptist answer under an unusual name, and `Independent Baptist` is a real "
        "and specific thing in the Anglophone Caribbean: unaffiliated fundamentalist "
        "congregations, largely out of twentieth-century North American missions, as "
        "against the Baptist Union congregations of the older British missionary line. "
        "**Filed at the branch parent rather than a child** because the tree has no "
        "independent-Baptist node and one cell in one country does not earn one (spec "
        "§2.4). The trailing `E` in `BAPTISTE` is CSO's spelling and is left alone in "
        "`gd.csv` (§12); this module maps the string the source prints.",
    "PRESBYTERIAN":
        "-> christianity.reformed.presbyterian. 467 people, 0.43%, and **it is one parish**: "
        "**3.17% of St. Mark**, seven times the national rate, against 0.02% in Carriacou "
        "and 0.18% in St. John. That is the Scottish planter and missionary presence on the "
        "west coast around Victoria, and it is the same reading `vc2012.py` gives Saint "
        "Vincent's 294 Presbyterians. A cell this small with a geography this sharp is a "
        "congregation, not a denomination.",
    "MENNONITE":
        "-> christianity.anabaptist.mennonite, and **here the label is believed**, which is "
        "worth stating because 150 km away it is not. 280 people, 0.26%. "
        "**Saint Lucia's `Mennonite` row is the same census round's `Evangelical` option "
        "mislabelled** — 3,760 people, 2.19%, and `lc2022.py` sets out the three "
        "independent proofs. Grenada is the control that makes that reading safe: **its "
        "form has BOTH cells**, `MENNONITE` at 0.26% and `EVANGELICAL` at 2.36%, and the "
        "two shares are exactly what a real Mennonite count beside a real Evangelical count "
        "looks like in the eastern Caribbean. Saint Lucia has one cell, at Evangelical's "
        "magnitude, under Mennonite's name. "
        "**Grenada's Mennonites are where a mission would be**: 0.50% of St. David and "
        "0.46% of St. George, none at all in St. Mark, St. Patrick or Carriacou.",
    "MORMOM":
        "-> christianity.latterday. 95 people. **`MORMOM` is CSO's spelling and it is left "
        "alone in `gd.csv`** — the normalised file records what the source printed (§12) "
        "and this module maps that string. Named here only so that nobody 'fixes' it in the "
        "data and silently unmaps the category.",
    "BRETHREN":
        "-> christianity.plymouth, NOT christianity.anabaptist.brethren. 312 people. The "
        "same Caribbean collision `jm2011.py`, `bs2022.py`, `bb2010.py` and `lc2022.py` "
        "document: the Brethren assemblies of the Anglophone Caribbean are the **Plymouth / "
        "Christian Brethren**, out of 1820s Dublin and arriving through nineteenth-century "
        "British missions, and the Schwarzenau (German Baptist) Brethren have no Grenadian "
        "presence. §12: never map a category on its string alone. "
        "**Grenada is the one place in this group where the two could be confused from the "
        "data**, because the form also has a `MENNONITE` cell and the Schwarzenau Brethren "
        "are Anabaptist — but the two cells' geographies are the same two parishes "
        "(St. David 0.63% and 0.50%, St. George 0.46% both), which is what a small "
        "twentieth-century mission field looks like and not what a shared lineage would "
        "require.",
    "ATHEIST":
        "-> secular, and NOT `unaffiliated`, which is the distinction branches.py exists "
        "for: `unaffiliated` is the absence of a religion and `secular` is a stated "
        "non-theistic position. CSO put both on the form. 49 people, **0.05%**, against "
        "`NO RELIGIOUS AFFILIATION` at 5.95% — **a 130-fold gap, the widest this map has "
        "from a source that offered both boxes**, and wider than Saint Lucia's 47-fold. "
        "Forty-nine people is small enough that nothing about its geography is read.",
    "OTHER (SPECIFY)":
        "-> other.gd, a per-source residual (§3.11). 1,716 people, 1.58%, on a form that "
        "names twenty-five answers. **It is written in and Grenada does not publish what "
        "was written**, which is unusual: the contents exist in CSO's microdata. Its "
        "geography has a peak — St. John 4.19% and St. Mark 4.01% against St. Andrew's "
        "0.48% — and branches.py explains why that is not read as a missing category.",
}

MAP = {
    "ANGLICAN": "christianity.anglican",
    "BUDDHIST": "buddhism",
    "BAHAI": "bahai",
    "BRETHREN": "christianity.plymouth",
    "CHURCH OF GOD": "christianity.holiness",
    "EVANGELICAL": "christianity.evangelical",
    "HINDU": "hinduism",
    "INDEPENDENT BAPTISTE": "christianity.baptist",
    "JEHOVAH WITNESSES": "christianity.witnesses",
    "METHODIST": "christianity.methodist",
    "MENNONITE": "christianity.anabaptist.mennonite",
    "MORAVIAN": "christianity.moravian",
    "MORMOM": "christianity.latterday",
    "MUSLIM": "islam",
    "PENTECOSTAL": "christianity.pentecostal",
    "PRESBYTERIAN": "christianity.reformed.presbyterian",
    "RASTAFARIAN": "rastafari",
    "ROMAN CATHOLIC": "christianity.catholic",
    "SALVATION ARMY": "christianity.holiness.salvation-army",
    "SEVENTH DAY ADVENTIST": "christianity.adventist",
    "SPIRITUAL BAPTIST": "afrodiasporic.spiritualbaptist",
    "LUTHERAN": "christianity.lutheran",
    "ATHEIST": "secular",
    "NO RELIGIOUS AFFILIATION": "unaffiliated",
    "OTHER (SPECIFY)": "other.gd",
}


def _key(cat):
    """Normalise case, whitespace and the PDF's ogonek.

    The report's text layer emits `Ǫ` (U+01EA) where the page shows a Q — see
    `sources/gd.py` — and the headings wrap mid-word, so a category can reach here with
    stray spacing. Folding means neither can silently unmap a category (§12).
    """
    s = str(cat).replace("Ǫ", "Q").replace("ǫ", "q").replace("’", "'")
    return " ".join(s.split()).upper()


EXCLUDED = {_key(k): v for k, v in EXCLUDED.items()}
MAP = {_key(k): v for k, v in MAP.items()}
REVIEW = {_key(k): v for k, v in REVIEW.items()}


def resolve(cat):
    """Source category -> taxonomy node id, or None if deliberately not on the tree.

    Grenada's no-religion cell is spelled `NO RELIGIOUS AFFILIATION`, so unlike Belize,
    Trinidad, the Bahamas and Cayman it is NOT the literal string `None` and survives a
    bare `pandas.read_csv`. `_gd_counts` in countries.py still passes
    `keep_default_na=False`, for consistency and because `NOT STATED` would otherwise be
    the next thing to go wrong.
    """
    c = _key(cat)
    if c in EXCLUDED:
        return None
    return MAP.get(c)
