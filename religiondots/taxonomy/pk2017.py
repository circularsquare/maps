"""Pakistan 2017 census religion (USCB tabulation) -> religiondots taxonomy.

Six categories at district, 207.7 million people. `sources/pk.md` is the source write-up
and `spec.md` §14 is why this country was discussed before it was built rather than after.

EXCLUDED holds categories that are deliberately not on the tree.
REVIEW holds calls that are defensible but arguable, with the reason.

**Two of the six needed a decision that is not a mapping decision, and both are recorded in
full below**: `Qadiani/Ahmadi`, which the Pakistani state prints as a peer of `Muslim`
because its constitution says so, and `Scheduled Castes`, which is a caste and not a
religion at all.

**There is no non-response category.** The six sum to the census population exactly. As in
Ethiopia (§9u) that is a fact about the tabulation, not about the country, and it does not
mean nobody refused.
"""

EXCLUDED = {}

REVIEW = {
    "Qadiani/Ahmadi":
        "-> islam.ahmadiyya, a node added for Pakistan, and the placement UNDER islam is "
        "the decision. 191,737 people, 0.09%. Ahmadis identify as Muslim. Pakistan's "
        "constitution declares them non-Muslim, its penal code makes it a criminal offence "
        "for them to call themselves Muslim, and its census therefore prints "
        "`Qadiani/Ahmadi` as a PEER of `Muslim` rather than beneath it. This file does not "
        "follow the census's structure there, because spec §2 says the tree is containment "
        "as it holds for people now — filing them outside Islam would make Pakistani law "
        "the map's taxonomy, which is a substantive claim and not a neutral one. "
        "**The number is a floor by a large and unknown margin.** Registering as Ahmadi "
        "carries legal consequences — a separate electoral roll, and a declaration "
        "disavowing the movement's founder required for a passport — and the community has "
        "organised census boycotts on that ground since 1974. Every independent estimate is "
        "several times the census figure. Said in `note_public`; not corrected, because "
        "correcting it would mean inventing a magnitude (§14.4). "
        "**And this is why the map draws districts and not tehsils.** PBS publishes "
        "religion by district; its tehsil release (`sindh_tehsil.pdf` and its siblings) "
        "carries Table 4 only — area, population, sex ratio, density — and no religion at "
        "all. The USCB file offers 585 tehsil-level units and drawing them would be finer "
        "than the state's own publication of this variable, which §14.4 forbids. At "
        "district the map shows Chiniot at 4.4%; at tehsil it would show Lalian at 13.6%. "
        "sources/pk.md §3.",
    "Scheduled Castes":
        "-> hinduism, MERGED with the `Hinduism` cell, and the merge is the argument. "
        "849,614 people. Scheduled Castes is a CASTE category, not a religion: these are "
        "Pakistan's Dalit communities — Meghwar, Bheel, Kolhi, Bagri, Oad — who are Hindu "
        "by religion and are printed by PBS as a peer of `Hinduism` rather than inside it. "
        "Filing a caste as a religion would be a category error; leaving it in a residual "
        "would erase the largest Dalit population outside India. So both cells go to "
        "`hinduism`, and Pakistan's Hindus come to **4,444,870** — which is the figure "
        "usually quoted for the country and which neither cell gives on its own. "
        "**What the merge costs, and it is real.** The two cells have different "
        "geographies: their shares correlate only 0.43 across tehsils, because the caste "
        "Hindus of the irrigated Sindh belt record as `Hinduism` (Samaro 51.0% Hindu, 0.4% "
        "SC) while the Dalit communities of the Thar desert record as `Scheduled Castes` "
        "(Islamkot 43.8% SC, 15.4% Hindu). Merging flattens a real line. "
        "**What decided it is that the 2017 split is unreliable and PBS says so.** The 2023 "
        "census report states the only change from 2017 is 'improvement in reporting of "
        "scheduled caste by clear differentiation between Hindu and scheduled caste', and "
        "Sindh's Scheduled Caste count went from 831,562 to 1,325,559 — up 59% — while its "
        "Hindu count barely moved. A boundary the publisher describes as newly fixed is not "
        "one to draw on the 2017 side of. `source_category` is kept verbatim, so a future "
        "2023 ingest can separate them (§2.4).",
    "Islam":
        "-> islam, with no branch, because the census gives none. 200,362,718 people, "
        "96.47%, the largest single count on this map. Overwhelmingly Sunni — Hanafi, with "
        "the Barelvi and Deobandi movements inside that — and a Shia minority usually put "
        "at 10-15% and concentrated in Gilgit-Baltistan, Parachinar and urban Punjab. "
        "`islam.sunni` would be an inference rather than a reading, and Gilgit-Baltistan, "
        "the one place where the split would be most visible, has no data at all.",
    "Christianity":
        "-> christianity, the parent, because the census names no denomination. 2,642,048 "
        "people, 1.27%. Pakistan's Christians are roughly half Catholic and half Protestant "
        "(the Church of Pakistan, a 1970 union of Anglicans, Methodists, Lutherans and "
        "Presbyterians, is the largest of the latter), and the census separates none of it. "
        "The same call in2011.py makes for India's identical cell.",
    "Hinduism":
        "-> hinduism. 3,595,256 before the Scheduled Caste merge, 4,444,870 after. See the "
        "`Scheduled Castes` note, which is where the reasoning is.",
    "Other":
        "-> other.pk. 43,253 people, 0.021% — the smallest residual on this map. Its "
        "contents and why the 2023 census matters for it are in the node's own note.",
}

MAP = {
    "Islam": "islam",
    "Christianity": "christianity",
    "Hinduism": "hinduism",
    "Scheduled Castes": "hinduism",
    "Qadiani/Ahmadi": "islam.ahmadiyya",
    "Other": "other.pk",
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
