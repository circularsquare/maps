"""Ethiopia 2007 census religion (USCB tabulation) -> religiondots taxonomy.

Six categories at woreda, 73.75 million people. Shallow next to Kenya's thirteen (§9o) and
about as deep as Ghana's, and it buys the second most populous country in Africa at 99,900
people per unit instead of Kenya's 1.01 million. sources/et.md is the source write-up.

EXCLUDED holds categories that are deliberately not on the tree.
REVIEW holds calls that are defensible but arguable, with the reason.

**There is no non-response category at all.** Every one of the six cells is a religion, and
they sum to the census population exactly. That is unusual on this map — §3.5 normally has
something to say about every country — and it does not mean nobody refused: it means the
2007 tabulation distributed or never published a refusal cell. Nothing here can undo that,
and et.py's `note_public` says so rather than implying the map has full coverage of an
answered question.
"""

EXCLUDED = {}

REVIEW = {
    "Orthodox":
        "-> christianity.oriental.ethiopian, and this is the most consequential line in "
        "the file. 32,092,182 people, 43.5% — the largest single body anywhere on this "
        "map outside the US Catholic Church. The Ethiopian census's `Orthodox` is the "
        "**Ethiopian Orthodox Tewahedo Church**, which is ORIENTAL Orthodox: "
        "non-Chalcedonian, out of communion with Constantinople since 451. Filing it under "
        "`christianity.orthodox` — Eastern Orthodoxy — is the commonest error in religion "
        "taxonomies and the note on `christianity.oriental` says so. "
        "ke2019.py deliberately sent Kenya's bare `Orthodox` to the PARENT node instead, "
        "because Kenya's cell genuinely mixes the Greek Patriarchate of Alexandria's "
        "Kenyan Orthodox (Eastern) with the Ethiopian and Eritrean communities of Nairobi "
        "(Oriental), and one cell over two communions goes to the node asserting less. "
        "**Ethiopia is the opposite case and needs the opposite call**: there is no "
        "ambiguity to preserve. In Ethiopia `Orthodox` means the Tewahedo Church and "
        "nothing else, and refusing to say so would throw away the fact. "
        "The leaf already existed: usrc2020.py maps ASARB group 204 to it for a US "
        "diaspora of about 66,000. Ethiopia arrives on the same node with four hundred "
        "times as many people, which is spec §2.4's 'a node earns its place by being "
        "countable somewhere' paying off in the direction nobody expected.",
    "Protestant":
        "-> christianity.protestant, the 'named no body' answer-node. 13,661,588 people, "
        "18.5%, and the fastest-growing category in Ethiopia. In Ethiopian usage this is "
        "**P'ent'ay** — the evangelical and Pentecostal stream taken together: the "
        "Ethiopian Evangelical Church Mekane Yesus (Lutheran-rooted and one of the largest "
        "Lutheran bodies in the world), the Kale Heywet Church (SIM-rooted), the Mulu "
        "Wongel and the Pentecostal churches. The tree could hold every one of those apart "
        "and the census names none of them, so the answer-node is what the source "
        "supports. Note that this is a WIDER category than Kenya's identically-spelled "
        "one, which excludes the evangelicals into a cell of their own — an Ethiopian "
        "Protestant here would mostly be a Kenyan `Evangelical Church` there. Two censuses "
        "using the same word for different sets is exactly why `source_category` is kept "
        "verbatim (§2.4).",
    "Catholic":
        "-> christianity.catholic, the parent, and deliberately neither .latin nor "
        ".eastern. 532,187 people, 0.72%. Ethiopia's Catholics are mostly the **Ethiopian "
        "Catholic Church**, a sui iuris Eastern Catholic church of the Alexandrian/Ge'ez "
        "rite — so `.latin` would be wrong for most of them — but there are Latin-rite "
        "vicariates in the south and west too, so `.eastern` would be wrong for the rest. "
        "The census says only `Catholic`. The parent asserts what is known.",
    "Islam":
        "-> islam, with no branch, because the census gives none. 25,037,646 people, "
        "33.9%, and the second largest Muslim population in sub-Saharan Africa after "
        "Nigeria's. Overwhelmingly Sunni, largely Shafi'i, with a strong Sufi presence "
        "(the Qadiriyya and Tijaniyya) and the Harari and Argobba communities; `islam.sunni` "
        "would be an inference rather than a reading. The geography is the point — the "
        "Somali and Āfar regions run above 95%, and the contrast with the Orthodox highland "
        "is the sharpest religious boundary on the African map.",
    "Traditional":
        "-> indigenous.african, the node Ghana added and Kenya reused. 1,956,647 people, "
        "2.65% — higher than Kenya's 0.68% and lower than Ghana's 3.25%. **Read it as a "
        "floor**, for the reason sources.md §11b gives for the whole continent: the cell is "
        "exclusive of the Christian and Muslim boxes, and Ethiopian traditional practice — "
        "the Oromo Waaqeffanna above all, plus the Gamo, Konso and South Omo systems — "
        "commonly accompanies one of them rather than replacing it. Waaqeffanna has been "
        "the subject of an organised revival and a census-recognition campaign since, so "
        "the 2007 figure is also old. Where it does show it is concentrated hard: South "
        "Omo, Bench Maji and the Oromo west.",
    "Other":
        "-> other.et. 470,682 people, 0.64%. Per source, per spec §3.11.",
}

MAP = {
    "Orthodox": "christianity.oriental.ethiopian",
    "Protestant": "christianity.protestant",
    "Catholic": "christianity.catholic",
    "Islam": "islam",
    "Traditional": "indigenous.african",
    "Other": "other.et",
}


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
