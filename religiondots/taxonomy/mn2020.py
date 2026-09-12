"""NSO 2020 Population and Housing Census religion -> religiondots taxonomy.

Six answers, at aimag, for the population aged 15 and over. The census's own English wording
is on the questionnaire NSO publishes with the microdata study
(`web.nso.mn/nada/index.php/catalog/175/download/1227`, question 29):

    29. DO YOU HAVE A RELIGION?
        No religion 1 / Buddhism 2 / Christianity 3 / Islam 4 / Shamanism 5 / Other 6

The keys below are the Mongolian labels as the aimag volumes print them, because those
volumes are what `sources/mn.py` reads and §2.4 keeps a source's own spelling. NSO's English
is given against each.

EXCLUDED holds categories that are deliberately not on the tree.
REVIEW holds calls that are defensible but arguable, with the reason.
"""

EXCLUDED = {
    "Total":
        "the unit's own universe total (population aged 15 and over), not a category.",
}

REVIEW = {
    "Будда":
        "-> buddhism.vajrayana, and NOT the undivided `buddhism` node. 1.07 million people, "
        "51.8% of Mongolian adults and the largest answer in the country. The census says "
        "only `Buddhism`, so the ordinary rule here would send it to the parent, which is "
        "what ke2019.py does with `Orthodox` and `Catholic`. It is not applied for two "
        "reasons. First, the tree's own node already says so: `buddhism.vajrayana` is "
        "labelled *Tibetan Buddhism and its Himalayan and MONGOLIAN extensions*, so filing "
        "Mongolia's Buddhists on the parent would leave the node describing a country it "
        "does not contain. Second, the underlying fact is not in doubt in the way Kenya's "
        "`Orthodox` cell was: Mongolian Buddhism is Gelug, transmitted from Tibet in the "
        "16th century, and its institutions (Gandantegchinlen, the Bogd Gegeen lineage, the "
        "Jebtsundamba Khutuktu) are the same tradition as the Tibetan one rather than a "
        "cousin of it. There is no Theravada or Han Mahayana population of any size to be "
        "buried by the choice. "
        "**This is also the first VAJRAYANA anywhere on the map that was measured rather "
        "than inferred.** The node's other 6.3M are China's, and they are derived from "
        "Tibetan, Yugur and Monba NATIONALITY in cn2000.py because China's census asks no "
        "religion question at all. Mongolia's are people who were asked and answered. "
        "cn2000.py's REVIEW deliberately keeps China's MONGOLS out of the node; that "
        "exclusion is about the unavailability of a religio-ethnic derivation for them "
        "(spec §14.5) and says nothing against a direct census answer, which is what this "
        "is.",
    "Бөө":
        "-> indigenous.northeurasian. 50,408 people, 2.4% of adults, NSO's `Shamanism`. "
        "Mongolian shamanism (böö mörgöl) is the same North Eurasian complex the node "
        "already holds for Russia, and its nearest relatives are on the other side of the "
        "border: Tuvan, Buryat and Sakha practice, which Arena 2012 counts under the answer "
        "this node was built for. It gets no child of its own on purpose. A single-country "
        "node would add a legend row nobody else uses, which AGENT_BRIEF §3 says is Anita's "
        "call and not an agent's, and there is nothing here that the regional node fails to "
        "carry. "
        "Worth knowing that this cell is NOT the fringe it looks like nationally. It is "
        "10.9% of the religious in Dornod and 7.1% in Khövsgöl, against 0.4% in "
        "Övörkhangai, so it has a real and readable geography: the north and east, the "
        "Buryat and Darkhad country, rather than the Buddhist heartland.",
    "Христ":
        "-> christianity, with no branch, because the census gives none. 27,041 people, "
        "1.3%. Mongolia's Christians are overwhelmingly post-1990 evangelical and "
        "Pentecostal missions plus a Korean-sent Protestant presence, with a Catholic "
        "prefecture of a few hundred; the tree could hold all of those apart and NSO "
        "separates none of them, so anything below the root would be inference. It is "
        "strongly urban: 4.9% of Ulaanbaatar's religious against 0.9% in Khövsgöl.",
    "Ислам":
        "-> islam, with no branch. 68,880 people, 3.3%, and the most concentrated "
        "distribution in the country by a wide margin: 92.5% of Bayan-Ölgii's religious "
        "population and 13.6% of Khovd's, against effectively nil everywhere else. These "
        "are the Kazakhs, who are Sunni Hanafi, but the census says only `Islam` and "
        "`islam.sunni.hanafi` would be a derivation from ethnicity rather than a reading of "
        "the table. Same call as ke2019.py.",
    "Шүтдэггүй":
        "-> unaffiliated. 836,251 people, 40.4% of adults, the second largest answer. "
        "**The question is `Та шашин шүтдэг үү` -- do you have a religion -- so this is a "
        "single cell covering the irreligious, the indifferent and the atheist alike**, and "
        "nothing goes to `secular`, which needs a separately counted atheist or humanist "
        "answer. Mongolia's own volumes call these people `шашингүйчүүд`. "
        "Read it knowing the state suppressed religious practice until 1990 and destroyed "
        "most of the monasteries in the 1930s; the share is not a steady-state figure and "
        "it fell from 38.6% in 2010 only in some aimags while rising in others.",
    "Бусад":
        "-> other.mn. 13,699 people, 0.66%. Per source, per spec §3.11.",
}

MAP = {
    "Будда": "buddhism.vajrayana",        # NSO English: Buddhism
    "Христ": "christianity",              # NSO English: Christianity
    "Ислам": "islam",                     # NSO English: Islam
    "Бөө": "indigenous.northeurasian",    # NSO English: Shamanism
    "Бусад": "other.mn",                  # NSO English: Other
    "Шүтдэггүй": "unaffiliated",          # NSO English: No religion
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
