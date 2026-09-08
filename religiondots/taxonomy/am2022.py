"""Armstat 2022 census religion classification -> religiondots taxonomy.

Sixteen named religions plus a residual, a no-religion answer and a refusal, at marz. The
list is long and the country is not plural: one church is 95.2% of it, and everything worth
looking at is in the other 5%.

    95.24%  Armenian apostolic        -> christianity.oriental
     1.68%  Refused to answer         -> EXCLUDED (non-response)
     0.61%  Catholic                  -> christianity.catholic
     0.60%  No religion               -> unaffiliated
     0.54%  Evangelical               -> christianity.evangelical
     0.49%  Shar-fadinian             -> yazidism
     0.26%  Other religious groups    -> other.am   (a NEW node)
     0.22%  Orthodox                  -> christianity.orthodox.canonical
     0.18%  Jehovah's witness         -> christianity.witnesses
     0.07%  Pagan                     -> paganism
     0.07%  Molokai                   -> christianity.other
     0.02%  Islam                     -> islam
     0.02%  Nestorian                 -> christianity.churchofeast
     0.01%  Krishna consciousness ...  -> hinduism
     0.01%  Protestant                -> christianity.protestant
     0.00%  Judaism                   -> judaism
     0.00%  TM (Transcendental medit.) -> other.am

Percentages are of the census's own 2,932,731 and are the marz sums, which is what is drawn;
sources/am.py explains why those differ from the national table by up to 45 people.

**THE CATEGORY LIST IS NOT THE SAME IN EVERY MARZ**, because Armstat prints only the columns
a marz has people in. Syunik's table names five religions and Yerevan's fourteen, and an
answer with no column in a marz is inside that marz's `Other religious groups`. So this file
maps the UNION of what the eleven tables print, and `other.am` is a slightly different thing
in each marz. sources/am.py has the arithmetic.

EXCLUDED holds categories that are deliberately not on the tree.
REVIEW holds calls that are defensible but arguable, with the reason.
"""

EXCLUDED = {
    "Total":
        "the unit's own population total, not a category.",
    "Refused to answer":
        "49,359 people, 1.68% (spec §3.5). **Four fifths of it is in one city**: Yerevan "
        "holds 38,931 of the 49,359, which is 3.58% of Yerevan against 0.26% in Tavush and "
        "0.29% in Vayots Dzor. The census offered no separate `not stated` box, so this "
        "cell is doing both jobs and the gradient is the whole of what can be said about "
        "it. Armenia's refusal rate is low next to a voluntary European religion question "
        "(Czechia 30%, Hungary 40%), and the question here was not voluntary.",
}

REVIEW = {
    "Armenian apostolic":
        "-> christianity.oriental, the Oriental Orthodox parent, and **this is the call in "
        "the file most worth a second look.** 2,793,041 people, 95.24% of Armenia and by an "
        "enormous margin the largest count `christianity.oriental` has ever had: Georgia's "
        "109,041 was the previous largest and ge2014.py already recorded that a "
        "`christianity.oriental.armenian` child was becoming arguable on it. Armenia settles "
        "that argument on the numbers and does not act on it, for the reason ge2014.py gave "
        "and which has not changed: **au2021.py, ee2021.py, cy2021.py, pl2021.py and "
        "ro2021.py all file Coptic, Syriac, Ethiopian and Armenian bodies at the parent**, "
        "so adding the child for Armenia alone would put one country below a node the "
        "others sit on and assert a distinction those sources do not make. Doing it "
        "properly means re-pointing six mapping files and re-scattering six drawn "
        "countries, which is spec §3's *changes an already-drawn country's numbers* and is "
        "Anita's. Filed as `ask/` so the country ships either way. **The cost of the parent "
        "is real and is worth naming**: at 95% of one country the node stops reading as a "
        "branch and starts reading as `Armenia`, and the Copts and the Ethiopians are now a "
        "rounding error inside a colour that is mostly Armenian.",
    "Catholic":
        "-> christianity.catholic, the parent, and NOT `christianity.catholic.eastern`. "
        "17,855 people, and **the geography says exactly which church it is**: Lori 7,019 "
        "(3.15% of the marz) and Shirak 6,813 (2.89%) hold 13,832 of the 17,855, which is "
        "77% of Armenia's Catholics in the two northern marzes, against 0.03% in Syunik. "
        "That is the Armenian Catholic heartland, the villages around Artik, Gyumri and "
        "Tashir whose people are still called *Frank* locally, and the Armenian Catholic "
        "Church is an Eastern Catholic church in communion with Rome. Filing them on "
        "`.eastern` would be right about the body and wrong about the source, which prints "
        "one undivided `Կաթոլիկ` and never names a rite; Yerevan's 3,116 will include the "
        "Latin-rite parish. ge2014.py's identical cell sits at the parent and this follows "
        "it.",
    "Evangelical":
        "-> christianity.evangelical, the unspecified-answer node, and not `.protestant`. "
        "15,837 people, 0.54%, and the census ALSO prints a separate `Protestant` column in "
        "four marzes (154 people), so Armstat treats the two as different answers and this "
        "file has to as well. The body behind almost all of it is the **Armenian Evangelical "
        "Church**, founded at Constantinople in 1846 out of the Apostolic Church's own "
        "reform movement and Congregational-Reformed in polity; that is a specific "
        "denomination and `christianity.reformed.congregational` would be defensible for "
        "it. Not taken, because the cell is a one-word census answer that will also be "
        "carrying the Pentecostal and independent evangelical congregations of Yerevan and "
        "Lori, and `christianity.evangelical` is the node this map uses for exactly that "
        "ambiguity. Its geography is northern and rural rather than the capital, at 1.24% "
        "in Lori and 0.97% in Ararat against 0.30% in Yerevan and 0.15% in Shirak.",
    "Shar-fadinian":
        "-> yazidism. 14,344 people, 0.49%, and **the name is the finding**: `Շարֆադինական` "
        "is Sharfadin, from Şerfedîn, which is what Yazidis call their own religion, and no "
        "other census on this map uses it in place of the ethnonym. Armenia's Yazidis are "
        "Kurmanji-speaking, descended largely from refugees of the Ottoman persecutions of "
        "the 1910s and 1920s, and the temple at Aknalich (2019) is one of the largest built "
        "anywhere. The geography is the Aragats slopes and the Ararat plain: Aragatsotn "
        "3.70% and Armavir 2.21% hold 10,376 of the 14,344, against 0.03% in Shirak. "
        "**It is not all of Armenia's Yazidis and the census says so.** The national table "
        "cuts religion by ethnicity: of 31,079 people who gave `Yezidi` as their ethnicity, "
        "13,256 answered Shar-fadinian, 9,939 answered Armenian apostolic, 3,246 are in "
        "`Other religious groups` and 1,672 in `Pagan`. Those rows are in am.csv at "
        "`geo_level=country_by_ethnicity` and are not drawn.",
    "Pagan":
        "-> paganism, per source, and the node is holding two unrelated things. 2,123 "
        "people. The obvious reading is **Armenian Hetanism**, the revived pre-Christian "
        "religion organised as the Order of the Children of Ari since 1991, and 237 of the "
        "national cell's 2,132 are ethnic Armenians. **The other 1,887 are not**: 1,672 are "
        "Yezidi and 215 Kurd, which is why the cell peaks in Armavir (0.36%) and Ararat "
        "(0.19%) rather than in Yerevan (0.03%) where a neopagan revival would be. So most "
        "of `paganism` in Armenia is a Yazidi answer given to a different box than "
        "`Shar-fadinian`. Moving those 1,672 to `yazidism` was considered and NOT done: the "
        "census offered both boxes at the same question and these people chose this one, "
        "and reassigning an answer from the ethnicity of the person who gave it is exactly "
        "what spec §14.5 refuses. Recorded instead, because a reader looking at the Ararat "
        "plain should know the two colours there are one community.",
    "Molokai":
        "-> christianity.other, and Armstat's spelling is kept as printed. 1,982 people: "
        "the **Molokans**, Russian Spiritual Christians who broke with the Orthodox Church "
        "in the 18th century over icons, priesthood and the sacraments, were exiled to the "
        "Caucasus under Nicholas I, and have farmed the high country of Lori since the "
        "1840s. **1,578 of the 1,982 are in Lori**, 0.71% of that marz against 0.00% in "
        "Armavir, and the villages are Fioletovo and Lermontovo. A node of their own is "
        "arguable and is not taken: it would be a legend row no other country uses (spec "
        "§3) for two dots at 1:1,000, and no other source on this map counts Molokans, "
        "Doukhobors or any other Spiritual Christian body separately. If one ever does, "
        "these rows still carry `source_category=Molokai` and can be moved (spec §2.4). "
        "**They are not Orthodox and must not be filed there**, which is the mistake this "
        "cell invites: rejecting the Orthodox Church is the whole of what defines them.",
    "Nestorian":
        "-> christianity.churchofeast. 479 people, the **Assyrian Church of the East**, and "
        "the geography is one place: Ararat holds 346 of the 479 at 0.14% of the marz, "
        "against 0.00% in Armavir. That is Verin Dvin and the Assyrian villages of the "
        "Ararat plain, settled after 1828. `Nestorian` is the census's word and is a "
        "confessional label the church itself rejects; the node is named for what the "
        "church calls itself.",
    "Orthodox":
        "-> christianity.orthodox.canonical. 6,316 people, 0.22%, chiefly the Russian "
        "Orthodox parishes; the cell will also carry the Greek and Georgian Orthodox "
        "minorities, which the census does not separate. Highest in Shirak at 0.47% and "
        "Yerevan at 0.31%, which is Gyumri's garrison history and the capital.",
    "Krishna consciousness or Hare Krishna":
        "-> hinduism, the parent, and not a new ISKCON node. 200 people, **all of them in "
        "Yerevan**; three of the seventeen cells sit entirely in one marz, this one and "
        "Judaism (96, also Yerevan) and TM (9, Ararat). ISKCON is a Gaudiya Vaishnava "
        "lineage and belongs "
        "under Hinduism rather than beside it; `hinduism.vedanta` shows the tree already "
        "carries a Western-transmission Hindu body as a child, so a `hinduism.iskcon` is "
        "possible, but 200 people do not earn a legend row (spec §3).",
    "TM (Transcendental meditation)":
        "-> other.am, folded into Armenia's residual. **Nine people, in Ararat, and it is a "
        "named census category** rather than something the tree lost: Armstat prints a TM "
        "column in exactly one marz's table. The tree has no node for the Transcendental "
        "Meditation movement and nine people are not a reason to make one, so this is the "
        "honest place for it, and the row still carries its own `source_category` if that "
        "ever changes.",
    "No religion":
        "-> unaffiliated and NOT `secular`. One no-religion answer with no atheist or "
        "agnostic split, 17,497 people, 0.60%, which is a striking figure for a country "
        "that spent seventy years in the Soviet Union. Its range across the eleven marzes "
        "is 0.13% in Shirak to 1.25% in Lori.",
    "Protestant":
        "-> christianity.protestant, the unspecified node. 154 people in four marzes "
        "(Tavush 54, Lori 46, Kotayk 45, Ararat 9) and absent from the other seven, where "
        "whatever it counts is inside `Other religious groups`. It exists as a separate "
        "column from `Evangelical` and nothing in the source says how Armstat drew the "
        "line.",
}

MAP = {
    "Armenian apostolic": "christianity.oriental",
    "Catholic": "christianity.catholic",
    "Orthodox": "christianity.orthodox.canonical",
    "Nestorian": "christianity.churchofeast",
    "Evangelical": "christianity.evangelical",
    "Jehovah's witness": "christianity.witnesses",
    "Protestant": "christianity.protestant",
    "Molokai": "christianity.other",
    "Shar-fadinian": "yazidism",
    "Pagan": "paganism",
    "Islam": "islam",
    "Judaism": "judaism",
    "Krishna consciousness or Hare Krishna": "hinduism",
    "TM (Transcendental meditation)": "other.am",
    "Other religious groups": "other.am",
    "No religion": "unaffiliated",
}


def _key(cat):
    # Armstat pads some labels and hyphenates others for the printed column width; am.py has
    # already folded the Armenian, so this only has to survive whitespace and a curly quote.
    return " ".join(str(cat).replace("’", "'").split())


EXCLUDED = {_key(k): v for k, v in EXCLUDED.items()}
MAP = {_key(k): v for k, v in MAP.items()}
REVIEW = {_key(k): v for k, v in REVIEW.items()}


def resolve(cat):
    """Source category -> taxonomy node id, or None if deliberately not on the tree."""
    c = _key(cat)
    if c in EXCLUDED:
        return None
    return MAP.get(c)
