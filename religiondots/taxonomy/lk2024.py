"""DCS Census of Population and Housing 2024 religion classification -> religiondots taxonomy.

Six categories plus the universe total, on 14,003 GN divisions. This is the shallowest
category list of any country on the map and the finest geography, which is §3.9's trade
made about as hard as it goes in one direction.

**Nothing here needed a new node except the residual**, which is itself worth noticing: a
census that distinguishes Buddhist, Hindu, Muslim, Roman Catholic and other Christian is
asking the question at the level of world religions, and the tree has had those since the
first source. Sri Lanka's value to this map is not taxonomic depth. It is that 21.8M people
arrive already sorted into 14,003 neighbourhoods, and that four of the world's large
traditions are present in numbers, in sharply separated places, at that grain.

EXCLUDED holds categories that are deliberately not on the tree.
REVIEW holds calls that are defensible but arguable, with the reason.
"""

EXCLUDED = {
    "Total":
        "the unit's own population total, not a category.",
}

REVIEW = {
    "Buddhist":
        "-> buddhism, the PARENT, and deliberately not buddhism.theravada. 15,196,960 "
        "people, 69.8% of the country, and by a wide margin the largest single mapping "
        "decision on this map. Sri Lankan Buddhism is Theravada in overwhelming "
        "proportion — the island is one of the tradition's historic centres and its "
        "Mahayana presence is negligible — so filing it as Theravada would almost "
        "certainly be true. It is still not what the source says. DCS offers one cell "
        "labelled `Buddhist` and asks for a religion rather than a school, so the "
        "sub-school distribution here is inferred from general knowledge and not measured, "
        "and §2 forbids inventing the distinction at ingest time. in2011.py makes the same "
        "call on India's 8.4M Buddhists for the same reason. The tree can hold the three "
        "vehicles apart the moment a source separates them; no census on this map does.",
    "Islam":
        "-> islam, with no school or branch. 2,327,605 people, 10.7%. Sri Lanka's Muslims "
        "are predominantly Sunni of the Shafi'i school, with small Memon, Bohra and "
        "Ahmadiyya communities that are of real interest and that no table here separates. "
        "Same shape as the Buddhist call above and as mk2021.py's.",
    "Other Christian":
        "-> christianity, the ROOT, which is spec §6.6's 'branch that carries dots' and "
        "renders as a Christianity `unspecified` row beside the Roman Catholics. 266,515 "
        "people, 1.2%. The category is defined by what it is not: DCS names Roman "
        "Catholics and puts every other Christian in one cell — the Church of Ceylon "
        "(Anglican), the Methodists, the Baptists, the Dutch Reformed, the Assemblies of "
        "God and the fast-growing independent Pentecostal congregations, plus a small "
        "Orthodox presence. Filing it on `christianity.protestant` would be the tempting "
        "move and is wrong twice over: it would assert Protestantism of the Orthodox "
        "minority, and it would place Anglicans by a claim the source does not make. The "
        "root is the node for 'Christian, body not named', which is exactly what this is.",
    "Other":
        "-> other.lk. 63,494 people, 0.29%, and NOT a clean residual — DCS moves any "
        "religion with fewer than 10 people in a GN division into this cell, so it holds "
        "an unknown mixture of genuine other-religion answers and the suppressed tail of "
        "the five named categories. See sources/lk.py and spec §3.8. Per source, per "
        "spec §3.11.",
}

MAP = {
    "Buddhist": "buddhism",
    "Hindu": "hinduism",
    "Islam": "islam",
    "Roman Catholic": "christianity.catholic.latin",
    "Other Christian": "christianity",
    "Other": "other.lk",
}


def _key(cat):
    return " ".join(str(cat).split())


def resolve(cat):
    """Source category -> taxonomy node id, or None if deliberately not on the tree."""
    c = _key(cat)
    if c in EXCLUDED:
        return None
    return MAP.get(c)
