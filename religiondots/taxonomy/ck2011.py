"""Cook Islands 2011 census religion (UNSD DYB table 28) -> religiondots taxonomy.

Nine categories, 14,974 people, an exact partition. **One new node**, the national church.

    49.1%  Cook Islands Christian Church -> christianity.reformed.congregational.cicc  <- new
    17.0%  Roman Catholic                -> christianity.catholic
     8.0%  Other Religions               -> other.ck
     7.9%  Seventh Day Adventist         -> christianity.adventist
     5.6%  No Religion                   -> unaffiliated
     4.4%  Latter Day Saints             -> christianity.latterday
     3.7%  Assembly of God               -> christianity.pentecostal.trinitarian
     2.2%  Unknown                       -> not on the tree
     2.1%  Apostolic                     -> christianity.pentecostal

**THE COOK ISLANDS CHRISTIAN CHURCH IS THE LONDON MISSIONARY SOCIETY'S**, planted in 1821 and
still half the country. Its siblings on this map are Tuvalu's Ekalesia Kelisiano Tuvalu and
Niue's Ekalesia Niue, both LMS descendants and both filed under `congregational` beside it.

EXCLUDED holds categories that are deliberately not on the tree.
REVIEW holds calls that are defensible but arguable, with the reason.
"""

EXCLUDED = {
    "Unknown":
        "323 people, 2.2%, whose religion the census did not record. A coverage residual "
        "rather than an answer, and the same table keeps it apart from `Other Religions`, "
        "so this file does too.",
}

REVIEW = {
    "Apostolic":
        "-> christianity.pentecostal, the parent. The Apostolic Church in the Cook Islands "
        "descends from the Welsh Apostolic Church and is trinitarian, so "
        "`christianity.pentecostal.trinitarian` is arguable; the census says only "
        "`Apostolic`, which in other sources can equally mean a Oneness body, and the "
        "parent is the node that asserts neither. 310 people.",
    "Assembly of God":
        "-> christianity.pentecostal.trinitarian, following bs2022.py's "
        "`ASSEMBLIESOFGOD`. Named unambiguously, unlike `Apostolic` above.",
    "Other Religions":
        "-> other.ck. Kept apart from `Unknown`, which is excluded: the census asks the "
        "question and gets an answer here, and gets none there.",
}

MAP = {
    "Cook Islands Christian Church": "christianity.reformed.congregational.cicc",
    "Roman Catholic": "christianity.catholic",
    "Seventh Day Adventist": "christianity.adventist",
    "Latter Day Saints": "christianity.latterday",
    "Assembly of God": "christianity.pentecostal.trinitarian",
    "Apostolic": "christianity.pentecostal",
    "No Religion": "unaffiliated",
    "Other Religions": "other.ck",
}


def _key(cat):
    return " ".join(str(cat).split())


def resolve(cat):
    """Source category -> taxonomy node id, or None if deliberately not on the tree."""
    c = _key(cat)
    if c in EXCLUDED:
        return None
    return MAP.get(c)
