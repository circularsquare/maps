"""
Lebanon, 2022 electoral register -> religiondots taxonomy.

The register counts every voter's sect of record (the family register's, `madhhab al-sijill`),
which is one of Lebanon's eighteen recognised sects or `Others`. `sources/lb.py` carries it from the
fifteen electoral districts to the 26 cazas and scales it to OCHA's resident Lebanese;
`sources/lb.md` §10 is the record.

Categories, as `data/normalized/lb.csv` writes them:

    Lebanese, <sect>           -> the sect's node     measured (a caza that is a whole district)
                                                      or derived (split from its district by the
                                                      2014 register)
    Syrian, <node>             -> <node>              modelled: OCHA's 2026 count by caza, Pew 2020
    Palestinian, <node>        -> <node>              modelled, as Syrian
    Migrants (not drawn)       EXCLUDED, the gap
"""

EXCLUDED = {
    "Migrants (not drawn)":
        "OCHA's 2026 planning figure for migrant workers by caza (164,097, from IOM's Migrant "
        "Presence Monitoring). Nothing gives their nationalities by caza, so no religion can be "
        "put on them; they are the `gap`. Leans Christian and Muslim from Ethiopia, Bangladesh, "
        "the Philippines and Sri Lanka, with Buddhists and Hindus among the Asian workers.",
}

REVIEW = {
    "Lebanese, Greek Orthodox":
        "-> christianity.orthodox.canonical.antiochian, the Greek Orthodox Patriarchate of "
        "Antioch, whose seat is in Damascus and whose largest flock is Lebanon's. The node's label "
        "is `Antiochian Orthodox` (branches.py LEAF_LABEL); origin_religion's LB and SY rows "
        "already file here.",
    "Lebanese, Maronite":
        "-> christianity.catholic.eastern.maronite, added 2026-10-03 with Lebanon. A new node at "
        "depth 4, not a legend row at the default depth; Cyprus's `Maronite church` stays on the "
        "parent, as an already-drawn country (moving it is Anita's).",
    "Lebanese, Greek Catholic":
        "-> christianity.catholic.eastern.melkite, added with the Maronite node, for the same "
        "reason: the register names it, and its map (Zahle, the Bekaa, east Saida) is not the "
        "Maronites'.",
    "Lebanese, Armenian Catholic":
        "-> christianity.catholic.eastern, the parent. No node per small Eastern Catholic church; "
        "Syriac Catholic and Chaldean share it.",
    "Lebanese, Armenian Orthodox":
        "-> christianity.oriental.armenian, not .cilicia. Almost all of Lebanon's Armenian "
        "Orthodox are under the Catholicosate of Cilicia at Antelias, but the register names the "
        "church and not the catholicosate, and branches.py's rule (ask 004) files a cell that does "
        "not name the catholicosate one level up.",
    "Lebanese, Alawite":
        "-> islam, the bare family node. The tree has no Alawite node, and origin_religion.py "
        "files Syria's Alawites on bare `islam` (spec §6.6: neither Sunni nor Twelver Shia). "
        "38,537 voters, almost all in Tripoli (Jabal Mohsen) and Akkar; an `islam.alawite` node "
        "would be the right home if a second country ever counts them.",
    "Lebanese, Shia":
        "-> islam.shia, not .jaafari: the register says Shia. Lebanon's Shia are Twelvers.",
    "Lebanese, Evangelical":
        "-> christianity.protestant. The Evangelical community is Lebanon's recognised Protestant "
        "sect (Presbyterian, Armenian Evangelical, Baptist, Anglican and others under one council), "
        "so no single tradition fits; the Armenian Evangelicals are counted in it.",
    "Lebanese, Assyrian Orthodox":
        "-> christianity.churchofeast. The Monthly's `Assyrian Orthodox` is the minorities seat's "
        "Assyrian sect, the Church of the East (lub-anan's `اشوري` and `نسطوري`).",
    "Lebanese, Israeli":
        "-> judaism. The register's name for Lebanese Jews. 4,309 voters in 2022, almost all in "
        "Beirut II; most of the families left after 1967 and in the war years, and they stay on "
        "the register. Drawn where registered, as every Lebanese dot is.",
    "Lebanese, Others":
        "-> other.lb. In 2014 (lub-anan) the cell was mostly women with no sect recorded, then "
        "Copts and Christians of no named church, and a few hundred who struck their sect.",
    "Syrian, islam":
        "-> islam, bare: Pew 2020's Syria row, Muslims not split, as Syria itself is drawn "
        "(sources/sy.py; asks 040, 043). Nothing measures the sects of Syrians in Lebanon.",
    "Syrian, christianity":
        "-> christianity, bare: nothing measures which churches the Syrians in Lebanon belong to.",
    "Palestinian, islam.sunni":
        "-> islam.sunni, origin_religion.py's default for the Palestinian territories.",
}

MAP = {
    "Lebanese, Sunni": "islam.sunni",
    "Lebanese, Shia": "islam.shia",
    "Lebanese, Alawite": "islam",
    "Lebanese, Druze": "druze",
    "Lebanese, Maronite": "christianity.catholic.eastern.maronite",
    "Lebanese, Greek Catholic": "christianity.catholic.eastern.melkite",
    "Lebanese, Armenian Catholic": "christianity.catholic.eastern",
    "Lebanese, Syriac Catholic": "christianity.catholic.eastern",
    "Lebanese, Chaldean": "christianity.catholic.eastern",
    "Lebanese, Latin": "christianity.catholic.latin",
    "Lebanese, Greek Orthodox": "christianity.orthodox.canonical.antiochian",
    "Lebanese, Armenian Orthodox": "christianity.oriental.armenian",
    "Lebanese, Syriac Orthodox": "christianity.oriental.syriac",
    "Lebanese, Assyrian Orthodox": "christianity.churchofeast",
    "Lebanese, Evangelical": "christianity.protestant",
    "Lebanese, Israeli": "judaism",
    "Lebanese, Others": "other.lb",
    "Syrian, islam": "islam",
    "Syrian, christianity": "christianity",
    "Syrian, unaffiliated": "unaffiliated",
    "Syrian, hinduism": "hinduism",
    "Syrian, judaism": "judaism",
    "Syrian, buddhism": "buddhism",
    "Syrian, other.lb": "other.lb",
    "Palestinian, islam.sunni": "islam.sunni",
    "Palestinian, christianity": "christianity",
    "Palestinian, unaffiliated": "unaffiliated",
    "Palestinian, judaism": "judaism",
    "Palestinian, hinduism": "hinduism",
    "Palestinian, buddhism": "buddhism",
    "Palestinian, other.lb": "other.lb",
}

# No COLUMNS, deliberately (spec §7a-i-1, rollup.py's same-unit rule). A derived caza cell's sect
# was counted for its electoral district, not for the caza, which is Switzerland's canton-over-
# communes shape; so countries/lb.py writes `roll = NOWHERE` and those dots go under `inferred
# dots: not shown`, leaving the six cazas that are whole districts (Beirut is two). The refugees'
# religions were never measured in Lebanon at all.


def _key(category):
    return category


def resolve(category):
    """religiondots branch for a category, or None if deliberately off the tree."""
    if category in EXCLUDED:
        return None
    return MAP.get(category)
