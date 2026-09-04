"""SSO Popis 2021 religion classification -> religiondots taxonomy.

13 categories plus the universe total, at municipality. Shallow, and unusually so on the
Orthodox side: the census asks for a religion, not a church, so 847,390 Orthodox arrive
with no jurisdiction attached at all.

**The category that matters most here is not a religion.** `Лица за кои податоците се
превземени од административни извори` — persons whose data were taken from administrative
sources — is 132,260 people, 7.20% of the country, and it is a COVERAGE residual: the 2021
census enumerated part of the population from registers rather than in person, and those
records carry no religion. It is not irreligion and it is not a refusal. Filing it as
either would be the worst error available in this file, so it is EXCLUDED and named.

EXCLUDED holds categories that are deliberately not on the tree.
REVIEW holds calls that are defensible but arguable, with the reason.
"""

EXCLUDED = {
    "Religious affiliation - TOTAL":
        "the unit's own population total, not a category.",
    "Persons for whom data are taken from administrative sources":
        "132,260 people, 7.20%. The 2021 census took part of the resident population from "
        "administrative registers instead of enumerating them, and those records carry no "
        "religion question at all. A coverage residual wearing a category's clothes: not a "
        "religion, not 'no religion', and not a refusal — those last two are categories 15 "
        "and 16 and are much smaller. North Macedonia is therefore drawn at 92.5% "
        "coverage, and note_public says so.",
    "Not declare":
        "1,964 people who declined the question. A refusal, not an answer (spec §3.5).",
    "Unknown":
        "894 people for whom the variable is not known — a residual rather than a refusal, "
        "and SSO keeps the two apart, so this file does too.",
}

REVIEW = {
    "Orthodox":
        "-> christianity.orthodox, the PARENT, and deliberately not "
        "christianity.orthodox.canonical, which is where hr2021.py files Croatia's bare "
        "`Pravoslavci`. Two reasons to differ. The category names no jurisdiction — it is "
        "'Orthodox', covering the Macedonian, Serbian and Vlach communities at once. And "
        "canonicity was live at the census date: the Macedonian Orthodox Church – Ohrid "
        "Archbishopric had been in schism since 1967 and was recognised by the Serbian "
        "Patriarchate and Constantinople only in May 2022, MONTHS AFTER the September 2021 "
        "count. Filing 847,390 people as canonical would be asserting something that was "
        "not true when they were asked, and filing them as `.other` would assert the "
        "reverse about a church that is canonical now. The parent says what the source "
        "says.",
    "Christians":
        "-> christianity, the root, which is the node for an answer that names "
        "Christianity and no church. 242,579 people — 13.2%, the third largest category "
        "in the country — and the same call au2021.py makes for 'Christianity, nfd', "
        "ca2021.py for 'Christian' and cz2021.py for 'křesťanství'. Worth noticing how "
        "big it is: Orthodox plus Christians is 59.3%, which is close to the combined "
        "Macedonian, Serb and Vlach share of the population, so a large part of what "
        "would elsewhere be written 'Orthodox' is written 'Christian' here.",
    "Muslims (Islam)":
        "-> islam, with no school or branch, because the census gives none. 590,878 "
        "people, 32.2%, overwhelmingly Albanian, Turkish, Roma, Bosniak and Torbeš. The "
        "tree can hold Sunni and Bektashi apart and North Macedonia has both — the Bektashi "
        "tekke at Tetovo is one of the most important in the Balkans — but no table here "
        "separates them.",
    "atheist":
        "-> unaffiliated. SSO's `Не е верник (атеист)` is 'not a believer (atheist)', one "
        "cell for both. hr2021.py files Croatia's identical `Nisu vjernici i ateisti` the "
        "same way, and for the same reason: the no-religion reading is the larger part of "
        "it. Nothing goes to `secular`, which would need an agnostic or explicitly "
        "positional category, and North Macedonia has none. 8,764 people, 0.48% — one of "
        "the lowest irreligion figures on this map.",
    "Evangelists":
        "-> christianity.protestant, the 'named no body' node, which is not quite right "
        "and is the least satisfying call here. SSO keeps `Евангелисти` (678) apart from "
        "`Протестанти` (1,313), so it means something more specific than 'a Protestant' — "
        "the evangelical free-church stream — and branches.py has no evangelical node. "
        "Adding one for 678 people would cost the palette more than it is worth (§12). "
        "The distinction survives in `source_category` if anyone wants it back.",
    "Evangelists- methodist":
        "-> christianity.methodist. This is the Евангелско-методистичка црква, the United "
        "Methodist Church in North Macedonia, which is a named body and the oldest "
        "Protestant church in the country. 889 people. The leading space in SSO's English "
        "label is theirs and is kept verbatim per §2.4 — the key must match what "
        "sources/mk.py writes, not a tidied version.",
    "Other":
        "-> other.mk. 1,221 people. Per source, per spec §3.11.",
}

MAP = {
    "Orthodox": "christianity.orthodox",
    "Muslims (Islam)": "islam",
    "Catholics": "christianity.catholic",
    "Christians": "christianity",
    "Protestants": "christianity.protestant",
    "Evangelists": "christianity.protestant",
    "Evangelists- methodist": "christianity.methodist",
    "Jehovah's Witnesses": "christianity.witnesses",
    "Other": "other.mk",
    "atheist": "unaffiliated",
}


def _key(cat):
    return " ".join(str(cat).split())


def resolve(cat):
    """Source category -> taxonomy node id, or None if deliberately not on the tree."""
    c = _key(cat)
    if c in EXCLUDED:
        return None
    return MAP.get(c)
