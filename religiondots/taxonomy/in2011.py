"""
Census of India 2011, table C-01 and its Appendix -> religiondots taxonomy.

**Branch-level mapping, like cz2021.py.** Nothing here creates a leaf; the census's own
category name travels on every row in `source_category`, so deepening later costs nothing
(spec §2.4).

Two tables, and they are different in kind:

  C-01           eight columns, of which six are religions, one is the residual bucket and
                 one is the universe. The six are the only categories India publishes at a
                 fine geography, so they carry 99.1% of the map.
  C-01 Appendix  83 named religions inside that residual bucket, at state level, 7,788,066
                 people. This is the entire reason India is worth drawing at more than
                 Pew depth, and almost all of it is Adivasi.

**The Annexure is deliberately absent from this file.** It is arithmetically a partition —
`Religion:X` = the unspecified remainder + the named sects, per state, to within a few
hundred people nationally — so it *looks* usable for splitting Hindus into Lingayats or
Christians into Catholics. It is not, and the reason is in the numbers rather than in the
structure: it records 8,399 Catholics among 27.8M Christians, 573 Shia and 267 Sunni among
172.2M Muslims, 3,269 Digambar among 4.45M Jains. Nobody believes India has 573 Shia
Muslims. What the Annexure counts is people who wrote a SECT where the form asked for a
religion, which is a measure of insistence, not of membership — and it undercounts every
group it names, including the plausible-looking ones (Lingayat 2,663,229 against a Karnataka
community usually put near 10M). Mapping any of it would put figures on the map that are
wrong by one to three orders of magnitude in a direction the map cannot show. It is
normalised into in.csv with a note, and drawn nowhere. See sources/in.md §4.

EXCLUDED holds categories that are deliberately not on the tree.
REVIEW holds calls that are defensible but arguable, with the reason.
"""

# Categories that are not religious affiliations and are kept off the tree entirely.
EXCLUDED = {
    "Total":
        "the unit's own population total, not a category.",
    "Religion not stated":
        "2,867,303 people, 0.24% of India — the smallest such residual of any country on "
        "the map, and the reason is worth knowing rather than celebrating. The religion "
        "question was answered by the head of household FOR the household, so it is not a "
        "personal self-identification and nobody had the opportunity to decline on their "
        "own behalf. Compare Poland's 20.5% refusal of a voluntary personal question. "
        "Excluded from the dots, so India draws 1.208bn of 1.211bn people.",
    "Other religions and persuasions":
        "the parent bucket, not a category. It is replaced by its 83 Appendix children "
        "wherever they can be allocated (spec §3.10) and by `other.in` for the 1.9% "
        "remainder the Appendix does not name. Drawing it as well would double every "
        "Adivasi religion in the country.",
}

# Defensible but arguable, recorded so the reasoning is not lost and can be overturned.
REVIEW = {
    "Sari Dharma":
        "-> indigenous.indian.sarna, NOT a node of its own, and this is the largest "
        "judgement call in the file: 506,369 people, 100% of them in West Bengal. `Sari "
        "Dharma` (also Sari/Sarna Dharam) is the Santal and Oraon name for the same "
        "sacred-grove religion that Jharkhand's census respondents wrote as `Sarna`; the "
        "two are one religion under two regional spellings, and the perfect "
        "state-complementarity — Sarna 83% Jharkhand, Sari Dharma 100% West Bengal — is "
        "itself evidence of that rather than of two religions. Overturnable: if they are "
        "genuinely distinct, this merges 506k people into the wrong node. `Sarnam` "
        "(1,494) and `Saranath` (837) go the same way for the same reason.",
    "Pagan":
        "-> indigenous.indian, NOT `paganism`. 2,088 people, 62% in Meghalaya. `Pagan` in "
        "the Khasi hills is the colonial-era label for the traditional religion and is "
        "still used that way locally; it is not the Western neo-pagan revival that "
        "branches.py's `paganism` node describes. Mapping it there would file Khasi "
        "traditionalists with Wiccans.",
    "Animist":
        "-> indigenous.indian. 4,130 people, 80% in Sikkim. The mirror of cz2021.py's "
        "`animismus`, which goes to `paganism` — there the write-in is a Western "
        "self-description, here it is an outsider's word for a tribal religion in a state "
        "full of them. Same string, opposite meaning, decided by where it was written.",
    "Non Christians":
        "-> indigenous.indian. 1,538 people, 96% in Meghalaya, a state that is 75% "
        "Christian. A negative self-description given in a place where the traditional "
        "religion is defined locally by not being the missionary one. `unaffiliated` "
        "would be a clear misreading; it says nothing about belief.",
    "Nirankari":
        "-> other.in. The Sant Nirankari Mission, 1,781 people, 64% Punjab. A distinct "
        "Sikh-derived movement that would deserve its own node if any source counted it "
        "properly; these are only the Nirankaris who declined all six census religions, "
        "and the great majority are recorded as Sikh or Hindu. Not mapped to `sikhism`, "
        "because answering `other` is exactly the datum.",
    "Dera Sarsa":
        "-> other.in. Dera Sacha Sauda, 139 people. Same reasoning as Nirankari at a "
        "twentieth of the size.",
    "ADI DHARM":
        "-> indigenous.indian. 82,255 people, but 65% in ODISHA rather than Punjab, which "
        "is the giveaway: this is `Adi Dharam`, the generic 'original religion' write-in "
        "used across the Adivasi belt, and not the Ad Dharm movement of Punjab's "
        "Ravidassia Dalits. `ravidassia` would have been the obvious wrong answer.",
    "Tadvi":
        "-> indigenous.indian. 1,786 people, 99% Maharashtra. The Tadvi Bhils are a Bhil "
        "group with substantial Muslim practice; those who wrote `Tadvi` under `Other "
        "religions` rather than answering Muslim are being taken at their word.",
    "A.C.":
        "-> other.in. 1,317 people, Maharashtra and Gujarat. The abbreviation is not "
        "expanded anywhere in the census documentation and no confident reading is "
        "available, so it goes to the residual rather than to a guess.",
    "Jews / Judaism":
        "-> judaism. 4,429 people, and the state split is the interesting part: 46% "
        "Manipur, which is the Bnei Menashe of Manipur and Mizoram rather than the older "
        "Cochin, Bene Israel and Baghdadi communities. Those are largely counted here too, "
        "but the Bnei Menashe are why the number is as large as it is.",
    "Atheist":
        "-> secular, not `unaffiliated`. 33,304 people who wrote the word. India's census "
        "offers no `no religion` box at all, so there is no `unaffiliated` figure for "
        "India anywhere — an absence worth stating, because on this map India will show as "
        "a country with no irreligion, and that is a property of the question.",
}

# 83 Appendix names + the six C-01 religions. Grouped by where they land.
MAP = {
    # ------------------------------------------------------------ C-01, the six religions
    # These carry 99.1% of India and are the only categories published at sub-district
    # level. Each is a whole family here: the census asks for the religion and clubs every
    # sect into it (see the Annexure note above), so `Muslim` really is all of Islam in
    # India and nothing finer is knowable from this source.
    "Hindu": "hinduism",
    "Muslim": "islam",
    "Christian": "christianity",
    "Sikh": "sikhism",
    "Buddhist": "buddhism",
    "Jain": "jainism",

    # ------------------------------------------------------------ Appendix: Sarna
    "Sarna": "indigenous.indian.sarna",
    "Sari Dharma": "indigenous.indian.sarna",
    "Sarnam": "indigenous.indian.sarna",
    "Saranath": "indigenous.indian.sarna",

    # ------------------------------------------------------------ Appendix: Gondi
    "Gond / Gondi": "indigenous.indian.gondi",
    "Koyatur": "indigenous.indian.gondi",
    "Budhadeo": "indigenous.indian.gondi",

    # ------------------------------------------------------------ Appendix: Donyi-Polo
    # The Tani religion of Arunachal Pradesh, plus the smaller Arunachal names that are
    # local forms of the same organised revival. Nocte, Rangfra and the Mishmi names are
    # separate peoples; they sit here rather than on the parent because Arunachal's
    # traditional religions were codified together and the census reports them together.
    "Doni Polo / Sidonyi Polo": "indigenous.indian.donyipolo",
    "Nani Intiya": "indigenous.indian.donyipolo",
    "Intaya": "indigenous.indian.donyipolo",
    "Nyarino": "indigenous.indian.donyipolo",
    "Rangfra": "indigenous.indian.donyipolo",
    "Dongi": "indigenous.indian.donyipolo",
    "Rangkho thak": "indigenous.indian.donyipolo",
    "Apo Rangang": "indigenous.indian.donyipolo",
    "Nocte": "indigenous.indian.donyipolo",
    "Idu / Idu Mishmi": "indigenous.indian.donyipolo",
    "Kaman  / Miju Mishmi / Kaman Mishmi / Miju": "indigenous.indian.donyipolo",
    "Hill Miri": "indigenous.indian.donyipolo",
    "Aka": "indigenous.indian.donyipolo",

    # ------------------------------------------------------------ Appendix: Sanamahi
    "Sanamahi": "indigenous.indian.sanamahi",
    "Heraka": "indigenous.indian.sanamahi",
    "Tikao Ragong": "indigenous.indian.sanamahi",
    "Chang Naga": "indigenous.indian.sanamahi",

    # ------------------------------------------------------------ Appendix: Meghalaya
    "Khasi": "indigenous.indian.khasi",
    "Niamtre": "indigenous.indian.khasi",
    "Niam Shnong": "indigenous.indian.khasi",
    "Songsarek": "indigenous.indian.khasi",
    "Garo": "indigenous.indian.khasi",
    "Traditional Religion": "indigenous.indian.khasi",
    "Non Christians": "indigenous.indian.khasi",
    "Pagan": "indigenous.indian.khasi",

    # ------------------------------------------------------------ Appendix: the rest of
    # the Adivasi tail. Sixty names, mostly one people in one district, none of them large
    # enough to earn a node of its own (§2.4). The people are real and the names are the
    # census's own, so they travel in `source_category` and can be split out later.
    "Addi Bassi": "indigenous.indian",
    "ADI DHARM": "indigenous.indian",
    "Adim dhamm": "indigenous.indian",
    "Adi": "indigenous.indian",
    "ADI KURUM": "indigenous.indian",
    "Bidin": "indigenous.indian",
    "Yumasam": "indigenous.indian",
    "Tribal Religion": "indigenous.indian",
    "Nature Religion": "indigenous.indian",
    "Animist": "indigenous.indian",
    "Santal": "indigenous.indian",
    "Ho": "indigenous.indian",
    "Munda": "indigenous.indian",
    "Oraon": "indigenous.indian",
    "Kharwar": "indigenous.indian",
    "Paharia": "indigenous.indian",
    "Birsa": "indigenous.indian",
    "Tana Bhagat": "indigenous.indian",
    "Sadri": "indigenous.indian",
    "Kisan": "indigenous.indian",
    "Marangboro": "indigenous.indian",
    "Swarna": "indigenous.indian",
    "Krupa": "indigenous.indian",
    "Dupub": "indigenous.indian",
    "Fralung": "indigenous.indian",
    "Bamanya": "indigenous.indian",
    "Bori": "indigenous.indian",
    "Baiga": "indigenous.indian",
    "Baigani Dharam": "indigenous.indian",
    "Bhumia": "indigenous.indian",
    "Korku": "indigenous.indian",
    "Bhil": "indigenous.indian",
    "Tadvi": "indigenous.indian",
    "Halba": "indigenous.indian",
    "Katkari": "indigenous.indian",
    "Pardhi": "indigenous.indian",
    "Bhoi": "indigenous.indian",
    "Bodo / Boro": "indigenous.indian",
    "Karbi / Mikir": "indigenous.indian",
    "Hajong": "indigenous.indian",
    "subba": "indigenous.indian",
    "Mannan": "indigenous.indian",
    "paniyar": "indigenous.indian",
    "Hidmaraj": "indigenous.indian",

    # ------------------------------------------------------------ Appendix: not indigenous
    "Parsi/Zorastrian": "zoroastrianism",
    "Bahai / Bahais": "bahai",
    "Jews / Judaism": "judaism",
    "Atheist": "secular",

    # ------------------------------------------------------------ residual
    # spec §3.11: residual buckets are per source and never merged across countries.
    #
    # The first entry is the Appendix's own floor: it names a religion only at 100+
    # adherents nationally, so 149,668 people (1.9% of the bucket) are in religions the
    # census recorded and did not publish. sources/in.py emits them as a category rather
    # than letting the allocation absorb them into the named ones; this is where they land.
    # tools/check_mapping.py caught them being dropped — they were exactly the gap between
    # the 1,207,838,006 that reached countries.py and the 1,207,987,674 that should have.
    "Other religions and persuasions, not separately named": "other.in",
    "Nirankari": "other.in",
    "Dera Sarsa": "other.in",
    "A.C.": "other.in",
}


# ------------------------------------------------------------------------ the Annexure
#
# All 47 write-in sects, excluded by name rather than by a `startswith` rule, so that a
# sect appearing in a reissue shows up as unmapped and fails the check instead of being
# swallowed. The reasoning is in the module docstring and in sources/in.md §4; the short
# version is that these count insistence rather than membership.
#
# They are listed in the census's own order of religion — Hindu, Muslim, Christian, Sikh,
# Buddhist, Jain — because that grouping is what makes the problem visible: the Christian
# block names nine denominations totalling 13,391 people out of 27.8M Christians.
_ANNEXURE_SECTS = [
    # under Hindu (966.3M)
    "Hindu", "Lingayat / Veer Shaiva", "Bathau / Bathew / Bathou",
    "Ghasidas / Satnam / Satnami", "Sanatan Dharma", "Ravidasi", "Meitei",
    "Balmiki / Walmiki / Valmiki", "Baishnav / Vaishnav", "Vishwa Karma", "Kabir Panthi",
    "Alakh/Mahima", "Nath Panthi", "Anukul Thakur", "Swami Narayan", "Bairagi",
    "Parnami/Pranami", "Sai", "Brahm Kumar / Brahm Kumari", "Anand Margi",
    # under Muslim (172.2M)
    "Islam/Muslim", "Bohra", "Agakhani", "Shia", "Sunni", "Ahmadia",
    # under Christian (27.8M)
    "Christian", "Catholic", "Unitarian", "Protestant", "Anglo Indian",
    "Seventh Day Adventists", "Orthodox", "Jacobite", "Marthomite", "Jehova Witness",
    # under Sikh (20.8M)
    "Sikh", "Nirmala",
    # under Buddhist (8.4M)
    "Buddhist", "Nav Buddhist / Neo Buddhist / Nav Baudha / Nav Boudha", "Hinayana", "Bon",
    # under Jain (4.45M)
    "Jain", "Digambar", "Samanar", "Swetamber / Shwetambar", "Moksha Margi",
]

_ANNEXURE_REASON = (
    "C-01 Annexure write-in sect. Recorded in in.csv, drawn nowhere. The Annexure is a "
    "true partition of its parent religion but every named sect is an undercount of the "
    "real community by one to three orders of magnitude, because the sect was recorded "
    "only where the respondent volunteered it instead of the religion. See the module "
    "docstring and sources/in.md §4.")

EXCLUDED.update({f"Sect: {s}": _ANNEXURE_REASON for s in _ANNEXURE_SECTS})


def resolve(category):
    """religiondots branch for a Census of India category, or None if off the tree."""
    if category in EXCLUDED:
        return None
    return MAP.get(category)
