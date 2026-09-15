"""Afrobarometer Madagascar religion -> religiondots taxonomy.

Twelve categories at the 22 regions of the 2018 census (RGPH-3), from rounds 5, 6, 7 and 9 pooled.
`sources/mg.py`'s docstring has the construction: the three large churches kept apart because
almost nobody answers just `Christian` in Madagascar, and `None` and `Traditional/ethnic religion`
placed as one box and split back at one national ratio. `sources/mg.md` is the record.

EXCLUDED holds categories that are deliberately not on the tree.
REVIEW holds calls that are defensible but arguable, with the reason.
"""

EXCLUDED = {}

REVIEW = {
    "Roman Catholic":
        "-> christianity.catholic, the bare branch, as `zm2022.py`, `rw2022.py` and `ke2019.py` "
        "file a card's `Catholic`. 37.85% as drawn. The Afrobarometer names no rite, and Madagascar's "
        "Catholics are Latin-rite almost without exception, so `christianity.catholic.latin` would "
        "be right in substance and is not what the source says (ask 004).",
    "Calvinist (FJKM)":
        "-> christianity.reformed, the bare branch. Round 4's card calls this box `Calviniste "
        "(FJKM)`, the Fiangonan'i Jesoa Kristy eto Madagasikara (Church of Jesus Christ in "
        "Madagascar), and rounds 5-9's `Calvinist` is the same code position on the Madagascar card "
        "at the same level (23.5, 19.7, 23.8, 20.7, 23.8% by round). The FJKM was formed in 1968 "
        "from the London Missionary Society's Congregational churches, the Paris mission's Reformed "
        "churches and the Friends' mission, so neither `.congregational` nor `.presbyterian` is "
        "right, and a Madagascar-only node would add a legend row nobody else uses (§3 of the "
        "brief), so it stays at the family. 21.37% as drawn; the split-half is +0.826.",
    "Lutheran":
        "-> christianity.lutheran, bare. In Madagascar this is almost entirely the Malagasy "
        "Lutheran Church (FLM), strongest in the south (Menabe 37.3%, Atsimo Atsinanana 35.2%). "
        "13.91% as drawn; the split-half is +0.797. **Nothing outside the Afrobarometer splits it "
        "from the FJKM**: both DHS surveys print `FJKM/FLM/Anglikana` or `Protestante/FLM` as one "
        "box, which witnesses the sum of the three churches against Catholics and not each "
        "church's own size.",
    "Seventh Day Adventist":
        "-> christianity.adventist.sda, as `zm2022.py`. 1.45% as drawn and it carries its own "
        "geography (+0.548): Diana 6.9% and Analanjirofo 5.7%, and the survey found none in seven "
        "regions.",
    "Other Christian":
        "-> christianity, bare. A grouping made in `sources/mg.py`, not a box on the card: "
        "`Christian only` with every Christian answer too small to level on 4,788 respondents "
        "(Independent, Evangelical, Baptist, Orthodox, Coptic, Methodist, Presbyterian, Dutch "
        "Reformed, Apostolic, Church of Christ, Mennonite, Zionist Christian Church) and four "
        "Malagasy boxes: `Rhema`, `Vahao ny Oloko` and `Toby Betela` (round 5 only, 16 people) and "
        "`Fifohazana` (round 9 only, 10 people), the revival movement inside the FJKM, Lutheran, "
        "Anglican and Catholic churches. Not `christianity.other`, whose note says it holds bodies "
        "with no branch rather than a residual. 3.66% as drawn, on its own shares (+0.385).",
    "None":
        "-> unaffiliated. **Step 2 of the draft \"no religion\" procedure** (WORKFLOW_PLAN.md; "
        "Anita's Madagascar ruling, 2026-09-14 night): every drawn round's card offers "
        "`Traditional/ethnic religion` beside `None`, `Atheist` and `Agnostic` (`report_card()`). "
        "12.69% as drawn. It is not drawn on its own shares: nationally traditional religion runs "
        "8.5, 4.5, 1.5, 1.3% by round while `None` runs 8.2, 4.0, 13.0, 12.7%, and in Melaky, "
        "Atsimo Atsinanana, Atsinanana and Betsiboka the same regions move from the first answer "
        "to the second, so the two are placed together (split-half +0.682) and split at rounds 7 "
        "and 9's national ratio, 90.5% `None`. Both DHS surveys offer the two separately too and "
        "put `None` at 89.6-95.8% of the pair, but at 20-25% of respondents aged 15-49 against "
        "this map's 12.7%; the level is the soft part. What these people practise is not "
        "measured by either survey.",
    "Traditional/ethnic religion":
        "-> indigenous.african, as `tz2022.py` files the same box. Madagascar's ancestral religion "
        "is its own tradition, and no Madagascar node exists; one would add a legend row nobody "
        "else uses. 1.33% as drawn: 9.5% of the none-or-traditional pair in every region (see "
        "`None`), so its geography is that pair's, and a region with none of it is one where no "
        "pooled respondent gave either answer (the five central highland regions).",
    "Anglican":
        "-> christianity.anglican. 1.15% as drawn; fails the split-half (+0.217), so it takes its "
        "national proportion within what each region's placed categories leave.",
    "Pentecostal":
        "-> christianity.pentecostal. 1.43% as drawn; fails the split-half (-0.070) and is spread "
        "in the residual. Jesosy Mamonjy, Madagascar's largest Pentecostal church, had a box on "
        "round 4's card only, which nobody chose; it is presumably inside this box and `Other "
        "Christian` in the drawn rounds.",
    "Jehovah's Witness":
        "-> christianity.witnesses, as `zm2022.py` and `rw2022.py`. 0.53% as drawn; it clears the "
        "rank test (+0.333) but is under the 1% eligibility floor, so it is spread in the residual.",
    "Muslim":
        "-> islam, with no branch. `Muslim only`, `Sunni only` and `Ismaeli` together; the survey "
        "finds almost no one naming a branch, so none is drawn. 1.30% as drawn, on its own shares "
        "(+0.574): Diana 14.7%, Melaky 7.8%, Boeny 4.9%, and none found in ten regions.",
    "Other":
        "-> other.mg. 3.34% as drawn, 153 respondents over four rounds, no specify text; fails the "
        "split-half (+0.242) and is spread in the residual. `Jewish` (1 respondent) is folded in.",
}

MAP = {
    "Roman Catholic": "christianity.catholic",
    "Calvinist (FJKM)": "christianity.reformed",
    "Lutheran": "christianity.lutheran",
    "Anglican": "christianity.anglican",
    "Seventh Day Adventist": "christianity.adventist.sda",
    "Pentecostal": "christianity.pentecostal",
    "Jehovah's Witness": "christianity.witnesses",
    "Other Christian": "christianity",
    "Muslim": "islam",
    "Traditional/ethnic religion": "indigenous.african",
    "None": "unaffiliated",
    "Other": "other.mg",
}

# spec §7a-i-1: every row is drawn at the node its own category names; nothing is inferred downward.
COLUMNS = {v: v for v in MAP.values()}


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
