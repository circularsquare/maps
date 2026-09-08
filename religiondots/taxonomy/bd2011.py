"""Bangladesh 2011 census religion (USCB tabulation) -> religiondots taxonomy.

Five categories at upazila, 144.04 million people. **The shallowest question on this map
attached to the fourth-largest population on it** — shallower than Ethiopia's six and Sri
Lanka's six, level with Indonesia's official list, and deeper only than Germany's three.
sources/bd.md is the source write-up.

EXCLUDED holds categories that are deliberately not on the tree.
REVIEW holds calls that are defensible but arguable, with the reason.

**NOTHING HERE NEEDED A NEW NODE EXCEPT THE RESIDUAL**, which is the same sentence lk2024.py
opens with and means the same thing: a census that offers Muslim, Hindu, Christian and
Buddhist is asking at the level of world religions, and the tree has had those since its
first source. Bangladesh's value to this map is not taxonomic. It is that the largest Hindu
population outside India, and a Theravada Buddhist and tribal-Christian geography that
appears nowhere else here, arrive sorted into 544 units of 265,000 people each.

**THERE IS NO NON-RESPONSE CATEGORY AT ALL.** All five cells are religions and they sum to
the census population exactly. Ethiopia has the same shape and et2007.py says what has to be
said about it: this does not mean nobody refused, it means BBS distributed or never published
a refusal cell. §3.5's machinery has nothing to mark here, and `note_public` says so rather
than letting the map imply full coverage of an answered question.

**EVERY MAPPING BELOW GOES TO A PARENT OR A ROOT, AND THAT IS THE WHOLE FILE.** Four of the
five are world-religion cells with no branch named, so four of the five resolve to the node
that asserts exactly what the source says and nothing more. Bangladesh is the country where
the temptation to add the branch is strongest — the sub-composition of all four is well known
to anyone who knows the country — and §2 forbids inventing at ingest what the source does not
distinguish. The REVIEW notes below are mostly a record of the four inferences NOT made.
"""

EXCLUDED = {}

REVIEW = {
    "Muslim":
        "-> islam, with no branch, because the census gives none. 130,204,817 people, "
        "90.39%, and **the fourth-largest Muslim population in the world** — behind "
        "Indonesia, Pakistan and India, and larger than every Arab country combined. "
        "Bangladeshi Islam is overwhelmingly Sunni of the **Hanafi** school, with a deep "
        "Sufi layer (the Chishti and Qadiri orders, the Maijbhandari tariqa of Chattogram) "
        "and small Shia and Ahmadiyya communities that have both been the target of "
        "violence. `islam.sunni` would be an inference rather than a reading — the same "
        "call et2007.py makes for Ethiopia and lk2024.py for Sri Lanka. **The Ahmadiyya "
        "loss is the one worth naming**: pk2017.py has an `islam.ahmadiyya` node because "
        "Pakistan's census counts Ahmadis in a cell of their own, and Bangladesh's does "
        "not, so a community of perhaps 100,000 that has been attacked and had its mosques "
        "sealed is inside this 130 million and cannot be brought out.",
    "Hindu":
        "-> hinduism. 12,299,981 people, 8.54% — and **the largest Hindu population "
        "anywhere on this map outside India itself**, larger than Nepal's would be and "
        "roughly the size of Ohio. This is the single strongest reason to draw the country. "
        "No branch: the census names none, and Bangladeshi Hinduism is in any case mostly "
        "Shakta and Vaishnava practice that the tree does not separate for India either "
        "(in2011.py). **The geography is the point and it is sharp** — Khulna division's "
        "southwest runs three to five times the national rate, Dacope upazila reaching "
        "56.5%, with a second belt through Sylhet's tea districts and a third in the "
        "northwest around Dinajpur. Read the 2011 figure as a moment in a long decline: "
        "the Hindu share was about 22% at partition and roughly 13.5% in 1974.",
    "Christian":
        "-> christianity, the ROOT, which is spec §6.6's 'branch that carries dots' and "
        "renders as a Christianity `unspecified` row. 447,010 people, 0.31%. The category "
        "is one undivided cell and the population behind it is **roughly two-thirds "
        "Catholic** — the Portuguese-descended communities of Dhaka and Chattogram, the "
        "Holy Cross missions, and the Garo and Santal converts of Mymensingh — with "
        "Baptists (the oldest Protestant presence in the subcontinent, Carey's Serampore "
        "mission worked this ground) and a growing Pentecostal share making up the rest. "
        "Filing it on `christianity.catholic` would be true of most of them and asserted "
        "by nothing in the source; the root is the node for 'Christian, body not named', "
        "which is exactly what this is. Same call as lk2024.py's `Other Christian`.",
    "Buddhist":
        "-> buddhism, the PARENT, and deliberately not buddhism.theravada. 889,721 people, "
        "0.62%. Bangladeshi Buddhism is **Theravada in overwhelming proportion** — the "
        "Barua of Chattogram, who are among the oldest continuously Buddhist communities in "
        "South Asia, and the Chakma, Marma and Rakhine of the Hill Tracts, all in the "
        "Theravada lineage and organised under the Sangharaj and Mahasthabir Nikayas — so "
        "filing it as Theravada would almost certainly be true. It is still not what the "
        "source says, and lk2024.py made the identical call on 15.2 million Sri Lankan "
        "Buddhists, in2011.py on India's 8.4 million. Consistency here is not pedantry: if "
        "Bangladesh were filed as Theravada while Sri Lanka is not, the map would show a "
        "Theravada boundary at the Bengal border that is an artefact of two ingest "
        "decisions rather than a fact. **The geography is extraordinary and survives the "
        "coarse node**: Juraichhari upazila is 94.6% Buddhist, Naniarchar 83.4%, and six "
        "Hill Tracts upazilas are majority Buddhist in a country that is 90% Muslim.",
    "Other religion":
        "-> other.bd. 202,167 people, 0.14%. Per source, per spec §3.11, and see the node "
        "note in branches.py — the concentration in the Hill Tracts is the interesting "
        "thing about it and is almost certainly indigenous religion with no box to go in.",
}

MAP = {
    "Muslim": "islam",
    "Hindu": "hinduism",
    "Christian": "christianity",
    "Buddhist": "buddhism",
    "Other religion": "other.bd",
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
