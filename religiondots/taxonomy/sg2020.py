"""SingStat Census of Population 2020 religion classification -> religiondots taxonomy.

Nine categories plus the universe total, on 31 planning-area units. The question is asked of
residents aged 15 and over and the glossary defines the answer as "the religious faith or
spiritual belief of a person ... It is as declared by the person", so this is `self_id`
throughout and nothing here is a roll.

**NINE IS NOT A DEEP LIST BY THIS MAP'S STANDARDS AND THE INTERESTING PART IS WHICH NINE.**
The Philippines names 129 bodies and Vietnam 28, so Singapore is nowhere near the deepest
census in Southeast Asia; it is ahead of Indonesia's seven, Thailand's eight and Cambodia's
four. What is unusual is the shape rather than the length. It splits Christianity into
Catholic and everything else, and it prints **Sikhism as its own cell at 12,051 people,
0.35%** rather than folding it into a residual. Naming a group of twelve thousand in a country
of three and a half million, while leaving 411,674 non-Catholic Christians in one undivided
cell, is a statement about what the office thinks the salient divisions are.

Only ONE new node was needed, `other.sg`, which is the per-country residual every source gets.

EXCLUDED holds categories that are deliberately not on the tree.
REVIEW holds calls that are defensible but arguable, with the reason.
"""

EXCLUDED = {
    "Total":
        "the unit's own population total, not a category. The nine below partition it "
        "exactly, so drawing this as well would double the country.",
}

REVIEW = {
    "Christianity_OtherChristians":
        "-> christianity, the ROOT, which renders as a Christianity `unspecified` row "
        "beside the Catholics. **411,674 people, 11.9% of the counted population, and the "
        "most consequential call in this file.** SingStat splits Christianity two ways and "
        "two ways only, so this cell is defined entirely by what it is not: the Methodist "
        "Church in Singapore, the Anglican diocese, the Presbyterians, the Brethren, the "
        "Bible-Presbyterians, the Baptists, the Assemblies of God and the large independent "
        "charismatic congregations, plus a small Orthodox presence. "
        "**`christianity.protestant` is the tempting move and is wrong twice over.** That "
        "node holds the ANSWER and not the category, per its own note in branches.py, and "
        "nobody in Singapore answered `Protestant` — the form offered `Other Christians`. "
        "And it would assert Protestantism of the Orthodox and of anyone in the cell who "
        "would reject the word. `christianity.other` is wrong for the opposite reason: its "
        "note says it is for bodies with no branch to belong to and explicitly NOT for a "
        "residual, and this is nothing but a residual. "
        "**The precedent is exact and is Sri Lanka.** `lk2024.py` faces the same "
        "Catholic / not-Catholic binary from DCS, sends `Other Christian` to the root, and "
        "rejects `christianity.protestant` on these two grounds in those words. Australia's "
        "`Other Christian`, Chile, Fiji, France, Germany and Switzerland all sit at the root "
        "too. The cost is real and is stated in `note_public`: the map cannot show that "
        "Singapore's Protestants outnumber its Catholics by nearly two to one, because the "
        "census did not use the word.",
    "Taoism1":
        "-> chinesefolk, and NOT daoism, on the strength of the source's own footnote: "
        "**\"'Taoism' includes Chinese Traditional Beliefs\"**, printed under every religion "
        "table in Statistical Release 1. 303,960 people, 8.8%. That footnote makes the cell "
        "a declared combination the source will not separate, which is §3.3's test for a "
        "syncretic node, and `chinesefolk` is the node for exactly that: 'Chinese folk "
        "religion, Confucianism and the syncretic practice that census categories usually "
        "cannot separate'. Ancestor veneration, the deity temples, the Hungry Ghost "
        "festival and lineage practice are all inside this number, and most of the people "
        "in it would not describe themselves as followers of the Daoist canon. "
        "**Hong Kong sends its Taoism answer to `daoism` and the two will therefore draw "
        "different colours across one strait, which is worth knowing about.** The "
        "divergence is in the sources rather than in the mapping: `hk2021.py`'s Taoism is a "
        "SURVEY option offered beside Buddhism with no gloss at all, so a respondent picked "
        "the word unaided, while Singapore's is a census cell the office has told you is "
        "broader than the word. Following the label instead of the footnote would be the "
        "error [[reference_census_questionnaire]] exists to catch. Mauritius (`mu2022.py`, "
        "`Buddhist/Chinese` -> chinesefolk) is the closest existing call.",
    "Christianity_Catholic":
        "-> christianity.catholic.latin. 242,681 people, 7.0%. The source says `Catholic` "
        "without a rite, and Singapore's Catholics are the Roman Catholic Archdiocese of "
        "Singapore, a Latin-rite see; the Eastern Catholic presence is a handful of "
        "chaplaincies and is not separately counted anywhere. Same call as `lk2024.py`'s "
        "`Roman Catholic` and `hk2021.py`'s Catholic share, and the alternative "
        "(`christianity.catholic`, the parent) would draw an `unspecified` rite row for a "
        "country whose Catholics are not in fact of unspecified rite.",
    "OtherReligions":
        "-> other.sg. 9,827 people, 0.28%, and a genuinely small tail because the eight "
        "boxes above it are unusually many — Sikhism, which most censuses bury here, has "
        "its own cell. Per source, per spec §3.11.",
    "NoReligion":
        "-> unaffiliated. 692,528 people, 20.0%, and the fastest-moving number in the "
        "census: 17.0% in 2010, and 24.2% among 15-24 year olds against 15.2% of the "
        "over-55s. `unaffiliated` and not `secular`: the question asks what faith a person "
        "declares and this is the absence of one, which says nothing about whether they "
        "hold a positive irreligious position. It is also not `unknown` — there is no "
        "not-stated cell in this table at all, so nobody is hiding in here.",
}

MAP = {
    "NoReligion": "unaffiliated",
    "Buddhism": "buddhism",
    "Taoism1": "chinesefolk",
    "Islam": "islam",
    "Hinduism": "hinduism",
    "Sikhism": "sikhism",
    "Christianity_Catholic": "christianity.catholic.latin",
    "Christianity_OtherChristians": "christianity",
    "OtherReligions": "other.sg",
}

# spec §7a-i-1: for a derived row, the node its SOURCE COLUMN names. Singapore has no derived
# rows -- every cell is published at the unit it is drawn on -- so there is nothing to roll up
# and no COLUMNS dict is needed. tools/check_rollup.py should never name `sg`.


def _key(cat):
    return " ".join(str(cat).split())


def resolve(cat):
    """Source category -> taxonomy node id, or None if deliberately not on the tree."""
    c = _key(cat)
    if c in EXCLUDED:
        return None
    return MAP.get(c)
