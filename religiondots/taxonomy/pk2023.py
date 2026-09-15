"""Pakistan 2023 census religion (PBS Table 9) -> religiondots taxonomy.

Eight categories at district, 240.5 million people. `sources/pk.md` §9 is the write-up of
what 2023 changed and `taxonomy/pk2017.py` is the vintage this replaces; every decision that
module took is kept here, and the reasoning is not repeated where it did not change.

EXCLUDED holds categories that are deliberately not on the tree.
REVIEW holds calls that are defensible but arguable, with the reason.

**What is new against 2017 is two cells, and both land on nodes that already exist.** The
2017 form's `Other` held Pakistan's Sikhs and Parsis with everyone else; 2023 prints them
separately, so `Sikh` -> `sikhism` and `Parsi` -> `zoroastrianism`, both on the tree since
the US build. No node is added.

**There is still no non-response category.** The eight sum to Table 9's total exactly. The
people who are not in the table at all are the 1,041,342 in restricted areas that were
counted by head only (NCR 2023 p.124), which is a gap, not a category.
"""

EXCLUDED = {}

REVIEW = {
    "Qadiani/Ahmadi":
        "-> islam.ahmadiyya, as in 2017, and the placement UNDER islam is still the decision. "
        "162,684 people in 2023, down from 191,737 in 2017, which is a fall in the count and "
        "not evidence of a fall in the community: registering as Ahmadi carries a separate "
        "electoral roll and a passport declaration, and the community has boycotted the "
        "census on that ground since 1974. The count is a floor by a large and unknown "
        "margin. The full reasoning for keeping them, and for filing them under Islam rather "
        "than beside it as the state's form does, is taxonomy/pk2017.py and sources/pk.md §4a. "
        "**The drawn tier is still district.** The 2023 state prints Ahmadis by tehsil, so "
        "§14.4's ceiling is no longer what stops a finer map; Anita chose district on "
        "2026-09-14 (sources/pk.md §9). Chiniot district, not Lalian tehsil.",
    "Scheduled Castes":
        "-> hinduism, MERGED with `Hindu Jati`, reusing pk2017.py's decision. 1,349,487 "
        "people. It is a caste category and not a religion: Pakistan's Dalit communities, "
        "Meghwar, Bheel, Kolhi, Bagri and Oad, who are Hindu. **One of the two 2017 reasons is "
        "weaker in 2023 and it is worth saying so.** pk2017.py merged partly because PBS "
        "called the 2017 split unreliable and fixed it in 2023; the 2023 split is the fixed "
        "one, so a separate drawing would now rest on a boundary the publisher stands behind. "
        "The merge stands on the other reason, which did not change: a caste is not a "
        "religion, and a `Scheduled Castes` legend row beside `Hinduism` would print the "
        "state's caste line as if it were a line between faiths. Splitting them would also "
        "need a node no other country uses (AGENT_BRIEF §3). `source_category` is kept "
        "verbatim, so pk.csv can still separate them.",
    "Hindu Jati":
        "-> hinduism. 3,867,729 before the merge with Scheduled Castes, 5,217,216 after. "
        "`Jati` is the form's own word for caste Hindus, the counterpart of the Scheduled "
        "Castes cell, which is where the reasoning is.",
    "Sikh":
        "-> sikhism. NEW IN 2023: the 2017 form had no Sikh cell and its Sikhs were inside "
        "`Other`. The node already exists (US build). Nankana Sahib, Guru Nanak's birthplace, "
        "and Peshawar are where the 2017 `Other` spiked for them (sources/pk.md §7b).",
    "Parsi":
        "-> zoroastrianism. NEW IN 2023, and the form's word for Pakistan's Zoroastrians, who "
        "are Parsis by community. Karachi is where the 2017 `Other` spiked for them. A very "
        "small cell; drawn because it is a real census count of a community with a real "
        "geography, the same reason this map draws Guyana's Rastafarians.",
    "Muslim":
        "-> islam, with no branch, because the census gives none; as pk2017.py. Sunni with a "
        "Shia minority usually put at 10-15%, and Gilgit-Baltistan, where the split would be "
        "clearest, is still outside the census.",
    "Christian":
        "-> christianity, the parent, because the census names no denomination; as pk2017.py.",
    "Others":
        "-> other.pk. In 2023 this no longer holds the Sikhs or the Parsis. What is left is "
        "the Kalasha of Chitral, the Bahá'ís, and everyone else the form has no box for. Its "
        "node note says what it holds.",
}

MAP = {
    "Muslim": "islam",
    "Christian": "christianity",
    "Hindu Jati": "hinduism",
    "Scheduled Castes": "hinduism",
    "Qadiani/Ahmadi": "islam.ahmadiyya",
    "Sikh": "sikhism",
    "Parsi": "zoroastrianism",
    "Others": "other.pk",
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
