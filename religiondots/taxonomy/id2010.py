"""BPS Sensus Penduduk 2010 `Agama` -> religiondots taxonomy.

Nine categories plus the universe total, at kabupaten/kota. 237.6M people — the second
largest country on this map — through the shallowest question on it, and the shallowness is
a fact about Indonesian law rather than about Indonesian religion.

**THE SIX RECOGNISED RELIGIONS ARE A LEGAL LIST, NOT A CLASSIFICATION.** Indonesia
recognises Islam, Protestantism, Catholicism, Hinduism, Buddhism and Confucianism, and the
census asks which of them you belong to. That is Germany's §3.9a shape — a category list
produced by an instrument other than a question about belief — and it has three consequences
this file has to carry rather than smooth over:

  * **`Kristen` means PROTESTANT, not Christian.** In Indonesian official usage *Agama
    Kristen* and *Agama Katolik* are two of the six, side by side, and `Kristen` excludes
    Catholics. Reading it as "Christian" would make Catholics a subset of it and double
    count 6.8M people. This is Hungary's `RE_CA` (§12) in a different language: the
    plausible reading of the label is the wrong one.
  * **There is no `no religion` cell at all**, and no `other Christian`, no Sikh, no Jewish,
    no Bahá'í. Indonesia will therefore draw with an EMPTY unaffiliated family — which is
    §6.12's case exactly, an empty map meaning "not asked" rather than "nobody". It is not
    that Indonesian atheists were counted at zero; they were not offered the box, and the
    ones who exist are inside one of the six.
  * **`Lainnya` is doing far more work than an ordinary residual**, and is systematically
    too small. In 2010 *Aliran Kepercayaan* — the indigenous belief systems, Kejawen,
    Sunda Wiwitan, Parmalim, Kaharingan and the rest — had no legal standing on the form;
    the Constitutional Court only granted registration in 2017. Adherents in 2010 commonly
    recorded one of the six instead, usually Islam or Hinduism (Kaharingan was
    administratively folded into Hinduism outright). So `Lainnya`'s 292,889 is a FLOOR on
    Indonesia's indigenous religion by an unknown and large margin, in the same way §11b
    says every African `Traditionalist` cell is a floor. Said in note_public.

EXCLUDED holds categories that are deliberately not on the tree.
REVIEW holds calls that are defensible but arguable, with the reason.
"""

EXCLUDED = {
    "Total":
        "the unit's own population total, not a category.",
    "Tidak Terjawab":
        "'not answered' — 138,735 people, 0.06%. The question was put and no answer was "
        "recorded. A non-answer, not an answer (spec §3.5).",
    "Tidak Ditanyakan":
        "'not asked' — 749,657 people, 0.32%, five times larger than `Tidak Terjawab` and "
        "a different thing: the question never reached these people at all. BPS keeps the "
        "two apart and so does this file. Serbia's pair in §9p is the precedent, and the "
        "geography is worth a look before anyone merges them — 'not asked' concentrates "
        "where enumeration was hardest, which is not where refusals concentrate.",
}

REVIEW = {
    "Kristen":
        "-> christianity.protestant, the 'named no body' node, NOT christianity. 16.18M "
        "people, 6.9%. See the module docstring: in Indonesian official usage this is "
        "Protestantism specifically, a peer of `Katolik` rather than its parent. The node "
        "is right for a second reason as well — BPS names no denomination anywhere, and "
        "Indonesian Protestantism is overwhelmingly the ethnic-territorial churches (HKBP "
        "among the Batak, GMIM in Minahasa, GPM in Maluku, GKI in Papua) which are "
        "Lutheran, Reformed and Pentecostal in origin and cannot be separated from this "
        "cell. The answer-node asserts only what the form asked.",
    "Khong Hu Chu":
        "-> chinesefolk. 106,568 people, 0.045%. Confucianism as a state-recognised "
        "RELIGION is a specifically Indonesian construction, and the number is a political "
        "artefact as much as a count: recognition was withdrawn in 1979 under the New "
        "Order, so Chinese Indonesians registered as Buddhist or Christian for two decades, "
        "and it was restored in 2000. 2010 is the first census that counts it and the "
        "figure is certainly a floor — the community is generally put at millions. It goes "
        "to `chinesefolk` rather than to a node of its own because that node already exists "
        "for exactly this: Confucianism and Chinese folk practice as one thing census "
        "categories cannot separate.",
    "Lainnya":
        "-> other.id, a per-source residual (spec §3.11). 292,889 people. It is NOT sent to "
        "indigenous.austronesian or anything like it, even though most of what is really "
        "inside it is Aliran Kepercayaan, because the cell also holds Judaism, Sikhism, "
        "Bahá'í and Shinto — everything outside the six — and mapping a mixed residual to "
        "one tradition would assert a composition the source does not give. The docstring "
        "explains why the number is a floor.",
    "Budha":
        "-> buddhism, the parent. BPS's spelling of Buddha/Buddhis; no vehicle is named. "
        "Indonesian Buddhism is mostly Mahayana with a Theravada minority and the "
        "Buddhayana synthesis on top, and the census separates none of it.",
}

MAP = {
    "Islam": "islam",
    "Kristen": "christianity.protestant",
    "Katolik": "christianity.catholic",
    "Hindu": "hinduism",
    "Budha": "buddhism",
    "Khong Hu Chu": "chinesefolk",
    "Lainnya": "other.id",
}


def _key(cat):
    return " ".join(str(cat).split())


def resolve(cat):
    k = _key(cat)
    if k in EXCLUDED:
        return None
    return MAP.get(k)
