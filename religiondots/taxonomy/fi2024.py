"""ESS `rlgblg` x `rlgdnafi` -> religiondots taxonomy. Finland's Finnish-citizen half.

The European Social Survey asks `rlgblg` — do you consider yourself as belonging to any
particular religion or denomination — and only then which one. For Finland the second
question is `rlgdnafi`, a country-specific list, and **not `rlgdnfi`, which does not exist
in any round**. That matters more here than the naming accident suggests: the harmonised
`rlgdnm` offers seven families and would put 43% of Finland on `Protestant`, which is true
and useless. `rlgdnafi` names the bodies.

    1   Evangelical Lutheran            -> christianity.lutheran
    2   Eastern Orthodox                -> christianity.orthodox.canonical
    3   Roman Catholic                  -> christianity.catholic.latin
    4   Pentecostalism                  -> christianity.pentecostal
    5   Free church                     -> christianity.evangelical
    6   Advent church                   -> christianity.adventist
    7   Jehovah's Witness               -> christianity.witnesses
    8   Mormon                          -> christianity.latterday
    9   Jewish                          -> judaism
    10  Islam                           -> islam.sunni
    11  Other Protestant denomination   -> christianity.protestant
    12  Other Christian denomination    -> christianity
    13  Eastern religions               -> other.fi
    14  Other Non-Christian religions   -> other.fi
    (rlgblg = No)                       -> unaffiliated
    (refusal / don't know / no answer)  -> excluded, spec §3.5

THE CATEGORY LIST SHRINKS AS THE SAMPLE DOES, which is Greece's lesson (gr2024.py) and is a
property of surveys rather than of Finland. Rounds 5 and 6 offer all fourteen codes; round 8
offers twelve and round 10 eleven, and `Mormon` and `Jewish` drop out of the later lists not
because Finland stopped having them but because 1,600 respondents reached none. Pooling seven
rounds is what keeps the small categories alive at all. Read the Lutheran and unaffiliated
shares; distrust everything below one percent, and the note_public says so.

AND THE THING THIS FILE CANNOT DO, which is the country's whole subject. Finland's population
register records membership of a registered religious community for every resident, and
Statistics Finland publishes it nationally: 62.9% Evangelical Lutheran at the end of 2024.
This file's instrument asks a different question and gets a different number, about twenty
points lower. **Neither is wrong and they are not reconcilable**, because a person who has
never resigned from the church and does not consider themselves as belonging to it answers
one way to the registrar and the other way to the interviewer. sources/fi.md §3 has the
comparison; `basis` on every row says which quantity this is.
"""

EXCLUDED = {
    "__refused__":
        "Everyone whose `rlgblg` is Refusal, Don't know or No answer, and everyone who "
        "answered Yes to `rlgblg` and then declined to name a denomination. spec §3.5 marks "
        "a refusal rather than filling it, so these people are not drawn and the share they "
        "represent is stated in `gap` instead. Small in Finland; sources/fi.py prints it.",
}

REVIEW = {
    "Evangelical Lutheran":
        "-> christianity.lutheran, the branch, and NOT a new node for the Evangelical "
        "Lutheran Church of Finland. Almost every Finnish Lutheran is in that one church, so "
        "a named node would be accurate — but it would also be a single-country legend row "
        "for a body no other source on this map counts, which AGENT_BRIEF §3 sends to Anita "
        "rather than deciding alone. The branch says the true thing at no legend cost, and "
        "note_public names the church in words. **Laestadianism is the real loss here**: the "
        "Conservative Laestadian revival is perhaps 90,000 people concentrated in "
        "Pohjois-Pohjanmaa and Lappi, it is the reason those two maakunnat differ from the "
        "rest of Finland, and `christianity.lutheran.laestadian` already exists on the tree. "
        "ESS does not ask it — Laestadians are members of the national church and answer "
        "code 1 — so it cannot be drawn from this instrument at all, and drawing it from a "
        "movement's own estimate would be inventing a geography.",
    "Eastern Orthodox":
        "-> christianity.orthodox.canonical, the branch, not a named national church. The "
        "Orthodox Church of Finland is autonomous under the Ecumenical Patriarchate and is "
        "most of this cell; the rest is the Moscow-Patriarchate parishes and Russian- and "
        "Estonian-origin Orthodox who are Finnish citizens. ESS does not distinguish them "
        "and no node for the Finnish church exists, so the parent is the honest place. "
        "Greece could use `.greek` because that church has a node; Finland cannot.",
    "Free church":
        "-> christianity.evangelical, `Evangelical, unspecified`. `Vapaakirkko` in Finnish "
        "is the Finnish Free Church, a congregational evangelical body of about 15,000 in "
        "the Evangelical Alliance tradition, and the ESS label is a translation of that "
        "name rather than a generic 'some free church'. `christianity.protestant` would lose "
        "the evangelical character and `christianity.pentecostal` would assert one it does "
        "not have. Note that ESS lists it BESIDE Pentecostalism, so the two are separate "
        "answers here and are not double-counting each other.",
    "Mormon":
        "-> christianity.latterday, the branch, not `.lds`. In Finland the Latter Day Saint "
        "population is essentially all the LDS Church and `.lds` would very probably be "
        "right, but ESS's label is `Mormon` and the branch is what that word names; picking "
        "the leaf would be reading a body into a word. It is also a category only rounds 5 "
        "and 6 offer, and the pooled cell is a handful of respondents.",
    "Islam":
        "-> islam.sunni. Finland's oldest Muslim community is the Tatars of Helsinki, who "
        "are Sunni and have been Finnish citizens since the 1920s, and the large "
        "Somali-origin population is Sunni too. **The Iraqi- and Iranian-origin Shia are "
        "the exception and are not separable in this half**: ESS offers one Islam code and "
        "the citizen sample reaches a few dozen Muslims in total. The foreign half does "
        "better, because origin_religion.py splits Iraq and Iran by their own Pew "
        "compositions, so Finland's Shia dots come from the foreign residents only and the "
        "citizen Shia are inside `islam.sunni`. Said here rather than silently rolled in.",
    "Other Christian denomination":
        "-> christianity, the root, following gr2024.py, ru2012.py and mk2021.py. The answer "
        "names no body and the list it sits at the end of has already ruled out Lutherans, "
        "Orthodox, Catholics, Pentecostals, the Free Church, Adventists, Jehovah's Witnesses "
        "and Mormons, so there is nothing left to name and the parent is honest.",
    "Eastern religions":
        "-> other.fi with `Other Non-Christian religions`, for gr2024.py's and hr2021.py's "
        "reason: the tree has no node for 'some Eastern religion, unspecified' and choosing "
        "Buddhism or Hinduism would invent a fact. ESS does not offer them separately.",
    "No religion":
        "-> unaffiliated, and it is the largest cell in the file. It is not a missing value: "
        "it is everyone who answered NO to `rlgblg`, which is most of the Finnish sample. "
        "branches.py's line is whether a POSITION is stated — `unaffiliated` is a report of "
        "not belonging, `secular` is a stated non-theistic stance — and 'I do not belong to "
        "a religion' is plainly the first. **Nothing in Finland reaches `secular` at all**, "
        "because ESS never offers atheist or agnostic as a denomination; gr2024.py and "
        "ge2014.py make the same call for the same reason.",
}

MAP = {
    "Evangelical Lutheran": "christianity.lutheran",
    "Eastern Orthodox": "christianity.orthodox.canonical",
    "Roman Catholic": "christianity.catholic.latin",
    "Pentecostalism": "christianity.pentecostal",
    "Free church": "christianity.evangelical",
    "Advent church": "christianity.adventist",
    "Jehovah's Witness": "christianity.witnesses",
    "Mormon": "christianity.latterday",
    "Jewish": "judaism",
    "Islam": "islam.sunni",
    "Other Protestant denomination": "christianity.protestant",
    "Other Christian denomination": "christianity",
    "Eastern religions": "other.fi",
    "Other Non-Christian religions": "other.fi",
    "No religion": "unaffiliated",
}


def _key(cat):
    return " ".join(str(cat).split())


def resolve(cat):
    """Source category -> taxonomy node id, or None if deliberately not on the tree."""
    c = _key(cat)
    if c in EXCLUDED:
        return None
    return MAP.get(c)
