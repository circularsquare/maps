"""NBS Seychelles, Population and Housing Census 2010, district Table 3 -> religiondots taxonomy.

57 labels across the 26 districts, for the whole enumerated population of every age. One,
`Not stated` (4,328 people, 4.76%), is off the tree; the other 56 are drawn.

**THE FORM HAD SEVEN BOXES AND A WRITE-IN LINE, AND NBS KEPT THE WRITE-INS.** The 2010
enumeration form lists Roman Catholic, Anglican, 7th Day Adventist, Moslem, Baha'i, Hindu and
No Religion, then "Others: write full name". The national report (Table 2.9) folds the rarest
write-ins into `Other Christian` and `Other non-Christian`; the district supplement does not,
which is why 1-person rows like `Presbyterian` and `Agnostic` appear here. They are mapped to
their own families rather than folded, because the source names them.

**A POST-CODED CATEGORY IS A NAME SOMEONE WROTE, NOT A DENOMINATION NBS VERIFIED.** Several
labels are local bodies that cannot be placed with confidence (`Christian Community
Fellowship`, `Nazarite Christian`, the three `Tabernacle`/`End Time` labels). They go on
`christianity.other`, which is for bodies with no branch to belong to, and each is argued in
REVIEW.

EXCLUDED holds categories that are deliberately not on the tree.
REVIEW holds calls that are defensible but arguable, with the reason.
"""

EXCLUDED = {
    "Not stated":
        "4,328 people, 4.76% of the census. Not a religion, so off the tree per spec §3.5. "
        "It is 37% of Other Islands (387 of 1,042), where most people were hotel and "
        "construction staff enumerated away from home, and 1-3% of most Mahé districts. "
        "Leaving it out leans every share in Other Islands upward; nowhere else does it "
        "move a drawn share by more than a few points.",
}

REVIEW = {
    "Roman Catholic":
        "-> christianity.catholic.latin. 69,277 people, 76.2%. The Diocese of Port Victoria "
        "is Latin-rite and there is no Eastern Catholic body in the country; NBS's label "
        "says Roman, which is `br2010.py`'s and `st2012.py`'s reading. The Anglophone "
        "Caribbean tables sit on the `christianity.catholic` parent instead; either is "
        "defensible and this one follows the label.",
    "Latin Catholic":
        "-> christianity.catholic.latin. 20 people, all men and all in Beau Vallon. It is a "
        "coding of what someone wrote and it is the same church as `Roman Catholic`; kept "
        "on the same node rather than guessed at.",
    "Christian Church of England":
        "-> christianity.anglican. 29 people. Read as a respondent's name for the Church of "
        "England, i.e. the Anglican Church of the Indian Ocean, not as a separate body.",
    "7th Day Adventist":
        "-> christianity.adventist. 1,128 people, a pre-printed box on the form. On the "
        "family node, as every Anglophone census table here maps `Seventh Day Adventist`. "
        "The label carries a stray leading apostrophe in the PDF, stripped in sc.py.",
    "Pentecostal Assembly":
        "-> christianity.pentecostal.trinitarian. **1,333 people, 1.5%, the largest "
        "Protestant body after the Anglicans.** The Pentecostal Assemblies of Seychelles, a "
        "classical Trinitarian Pentecostal church. On the family's Trinitarian node rather "
        "than a named child, since nothing NBS prints names a parent body.",
    "Assembly of God":
        "-> christianity.pentecostal.trinitarian. 831 people. Not on "
        "`...trinitarian.assemblies-of-god`, which is the USRC's American denomination; "
        "`cv2021.py`, `ao2024.py` and `st2012.py` make the same call for a national "
        "Assemblies of God outside the US.",
    "Born Again Christian":
        "-> christianity.evangelical. 605 people. A self-description, not a church. "
        "`ie2022.py` put Ireland's `Born Again Christian` on `christianity.protestant`; here "
        "it goes on `Evangelical, unspecified` because the phrase is the evangelical "
        "conversion claim and Seychelles has no Protestant tradition outside the Anglican "
        "church for it to be a synonym of. `New Born Christian` (5) goes with it.",
    "New Born Christian":
        "-> christianity.evangelical. 5 people; see `Born Again Christian`.",
    "Redeemed Christian Church":
        "-> christianity.pentecostal. 394 people. The Redeemed Christian Church of God, the "
        "Nigerian Pentecostal church with parishes across Africa. On the family node: its "
        "holiness roots and its size would argue for a child, and nothing here counts it "
        "separately enough to need one.",
    "Deeper Life":
        "-> christianity.pentecostal. 50 people. The Deeper Life Bible Church, the Nigerian "
        "holiness-Pentecostal church founded by W. F. Kumuyi. `christianity.holiness` would "
        "also be defensible; the family node is the safer of the two.",
    "New Testament Church":
        "-> christianity.pentecostal. 27 people. Read as the New Testament Church of God, "
        "which `jm2011.py` and `vg2010.py` map here; NBS prints the name without `of God`.",
    "Full Gospel Assembly":
        "-> christianity.pentecostal. 1 person. A Full Gospel name is a Pentecostal name "
        "(`au2021.py`, `nz2023.py`); the family node because the body is not identified.",
    "United Pentecost":
        "-> christianity.pentecostal.oneness. 15 people. The United Pentecostal Church, "
        "which `fj2007.py` and `ph2020.py` map to Oneness.",
    "Grace and Peace":
        "-> christianity.baptist. 52 people. The Baptist Union of Southern Africa lists "
        "`Grace and Peace Baptist Church (Seychelles)` among its member churches "
        "(baptistunion.org.za, read 2026-09-14; the listing gives a Seychelles phone number "
        "and nothing else), and the SeyFind directory has it in Mahé. No other body of that "
        "name was found in Seychelles. On the family node, since BUSA is a union of "
        "independent congregations and the tree has no child for it.",
    "Christian Community Fellowship":
        "-> christianity.other. **241 people nationally with `Christian Life Fellowship`, "
        "229 on their own.** Not identified: no body of this name was found in a search on "
        "2026-09-14. It is a named congregation rather than a residual, which is what "
        "`christianity.other` is for, and it is not guessed onto a Pentecostal node.",
    "Christian Life Fellowship":
        "-> christianity.other. 12 people; Table 2.9 folds them into `Christian Community "
        "Fellowship`, which is NBS's own statement that the two are the same or kindred. "
        "Same node.",
    "Nazarite Christian":
        "-> christianity.other. 109 people. A Nazarite church is named among Seychelles' "
        "smaller Christian groups in US State Department religious freedom reports (read "
        "via a search summary; the report pages refused the fetch), with nothing on its "
        "origin. Not `christianity.holiness.nazarene`: `Nazarite` is not `Nazarene`, and "
        "the Church of the Nazarene would have been coded under its own name.",
    "End-Time-Bride Tabernacle":
        "-> christianity.other. 19 people. `End Time Message` and `Bride` are the vocabulary "
        "of William Branham's followers, who reject the Trinity and are organised as "
        "independent tabernacles; they belong to no branch on this tree. Not "
        "`christianity.pentecostal.oneness`, which is a set of Pentecostal denominations "
        "the Message churches are not part of.",
    "End Time Message":
        "-> christianity.other. 9 people; see `End-Time-Bride Tabernacle`.",
    "Peniel Tabernacle":
        "-> christianity.other. 12 people. Not identified; `Tabernacle` suggests the same "
        "Message movement as the two labels above, and it goes on the same node either way.",
    "Seychelles Believers International":
        "-> christianity.other. 6 people. Not identified.",
    "United Christian Church":
        "-> christianity.other. 2 people. Not identified.",
    "Grace Assembly":
        "-> christianity.other. 1 person. Not identified; `Assembly` alone is not enough to "
        "call it Pentecostal.",
    "Other Christian":
        "-> christianity.other. 27 people in the districts, 112 in Table 2.9 after its folds. "
        "NBS's own residual for a Christian answer it did not code; `mu2022.py`, the other "
        "Indian Ocean census here, maps its `Other Christian` to the same node.",
    "Christians Unspecified Denomination":
        "-> christianity. 322 people who answered `Christian` and nothing more (report "
        "§2.7). On the root, which is the node for a Christian with no denomination.",
    "Dutch Reform Church":
        "-> christianity.reformed.continental. 3 people.",
    "Tamil":
        "-> hinduism.tamil. 12 people who gave `Tamil` as their religion. Tamil is a "
        "language, and in the Indian Ocean it is also how Tamil Hindus name their religion "
        "as distinct from North Indian Hinduism; `mu2022.py` has the same node for "
        "Mauritius's `Tamil/Tamil Hindu`. The Seychelles Hindu Kovil is Tamil.",
    "Padayachi":
        "-> hinduism.tamil. 2 people. Padayachi is a Tamil caste name, common among Tamil "
        "families in Mauritius and Seychelles; as a religious answer it can only mean Tamil "
        "Hindu. Same node as `Tamil`.",
    "Sai Baba devotee":
        "-> hinduism. 3 people. A devotee of Sathya Sai Baba; the movement is Hindu in "
        "practice and has no node of its own.",
    "Bobo Chanty":
        "-> rastafari. 2 people. The Bobo Shanti, one of the three Rastafari mansions, "
        "spelled as heard.",
    "Pantheist":
        "-> esoteric. 6 people, as `ee2021.py` maps Estonia's.",
    "Meditate":
        "-> other.sc. 4 people. A practice rather than a religion; nothing on the tree "
        "fits it better than the country's own other.",
    "Atheist":
        "-> secular. 3 people, a write-in beside the pre-printed `No Religion` box, as "
        "`ie2022.py` and `gd2021.py` map it.",
    "Agnostic":
        "-> secular. 1 person; see `Atheist`.",
    "No Religion":
        "-> unaffiliated. 840 people, a pre-printed box (`NONE`). The form offers no "
        "traditional-religion box for it to be lumped with, so it is a plain absence.",
    "Other":
        "-> other.sc. 30 people, NBS's residual for a non-Christian answer it did not code.",
}

MAP = {
    "Roman Catholic": "christianity.catholic.latin",
    "Latin Catholic": "christianity.catholic.latin",
    "Anglican": "christianity.anglican",
    "Christian Church of England": "christianity.anglican",
    "7th Day Adventist": "christianity.adventist",
    "Jehovah Witness": "christianity.witnesses",
    "Pentecostal Assembly": "christianity.pentecostal.trinitarian",
    "Assembly of God": "christianity.pentecostal.trinitarian",
    "Born Again Christian": "christianity.evangelical",
    "New Born Christian": "christianity.evangelical",
    "Redeemed Christian Church": "christianity.pentecostal",
    "Deeper Life": "christianity.pentecostal",
    "New Testament Church": "christianity.pentecostal",
    "Full Gospel Assembly": "christianity.pentecostal",
    "United Pentecost": "christianity.pentecostal.oneness",
    "Christians Unspecified Denomination": "christianity",
    "Christian Community Fellowship": "christianity.other",
    "Christian Life Fellowship": "christianity.other",
    "Nazarite Christian": "christianity.other",
    "End-Time-Bride Tabernacle": "christianity.other",
    "End Time Message": "christianity.other",
    "Peniel Tabernacle": "christianity.other",
    "Seychelles Believers International": "christianity.other",
    "United Christian Church": "christianity.other",
    "Grace Assembly": "christianity.other",
    "Other Christian": "christianity.other",
    "Orthodox": "christianity.orthodox",
    "Neo Apostolic": "christianity.newapostolic",
    "Grace and Peace": "christianity.baptist",
    "Baptist": "christianity.baptist",
    "Church of Christ": "christianity.restorationist",
    "Christ Holiness Church": "christianity.holiness",
    "Methodist": "christianity.methodist",
    "Lutheran": "christianity.lutheran",
    "Presbyterian": "christianity.reformed.presbyterian",
    "Dutch Reform Church": "christianity.reformed.continental",
    "Protestant Christian": "christianity.protestant",
    "Hindu": "hinduism",
    "Tamil": "hinduism.tamil",
    "Padayachi": "hinduism.tamil",
    "Sai Baba devotee": "hinduism",
    "Islam": "islam",
    "Baha'i": "bahai",
    "Buddhist": "buddhism",
    "Taoism": "daoism",
    "Jain": "jainism",
    "Zoroastrian": "zoroastrianism",
    "Jew": "judaism",
    "Rastafarian": "rastafari",
    "Bobo Chanty": "rastafari",
    "Pantheist": "esoteric",
    "Meditate": "other.sc",
    "Atheist": "secular",
    "Agnostic": "secular",
    "No Religion": "unaffiliated",
    "Other": "other.sc",
}

# spec 7a-i-1: the level this source COUNTED each node at. NBS counts every label at the
# district it is drawn on and nothing is filled in from a coarser tier, so nothing rolls.
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
