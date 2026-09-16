"""ESS `rlgblg` x `rlgdnis`/`rlgdnais` -> religiondots taxonomy. Iceland's citizen half.

ESS asks `rlgblg` (do you consider yourself as belonging to any particular religion or
denomination) and then Iceland's own card. Rounds 6 and 8 carry `rlgdnis`, rounds 9-11
`rlgdnais`; sources/is.py harmonises the two answers whose labels changed (the Russian Orthodox
Church became the Orthodox Church, the Muslim Association of Iceland became Islam) and proves
every answer's nesting in the harmonised `rlgdnm` from each round's cross-tab.

    Þjóðkirkjunni                              Church of Iceland     -> christianity.lutheran
    Fríkirkjunni                               the Free Church       -> christianity.lutheran
    Öðru kristnu trúfélagi innan lúthersku     other Lutheran body   -> christianity.lutheran
    Kaþólsku kirkjunni                         Catholic Church       -> christianity.catholic.latin
    Rétttrúnaðarkirkju                         Orthodox Church       -> christianity.orthodox
    Öðru trúfélagi utan lúthersku              other body, not Lutheran -> christianity
    Íslamstrú                                  Islam                 -> islam.sunni
    Austrænum trúarbrögðum                     Eastern religions     -> other.is
    Ásatrúarfélaginu                           the Ásatrú Association -> paganism
    Öðrum trúarbrögðum utan kristni            other non-Christian   -> other.is
    Öðrum                                      other (rounds 6, 8)   -> other.is
    (rlgblg = No)                              no religion           -> unaffiliated
    (refusal / don't know / no answer)         excluded, spec §3.5

The labels are the card's own, in the dative the question's wording puts them in ("do you belong
to ..."), and stay in Icelandic because they are data values.
"""

NATIONAL = "Þjóðkirkjunni"
FREE = "Fríkirkjunni"
OTHER_LUTHERAN = "Öðru kristnu trúfélagi innan lúthersku"
CATHOLIC = "Kaþólsku kirkjunni"
ORTHODOX = "Rétttrúnaðarkirkju"
OTHER_CHRISTIAN = "Öðru trúfélagi utan lúthersku"
ISLAM = "Íslamstrú"
EASTERN = "Austrænum trúarbrögðum"
ASATRU = "Ásatrúarfélaginu"
OTHER_NONCHRISTIAN = "Öðrum trúarbrögðum utan kristni"
OTHER = "Öðrum"
NO_RELIGION = "No religion"
REFUSAL = "__refused__"

# The answers the pooled rounds (6, 8, 10, 11) produce among citizens. sources/is.py asserts the
# pool against this set in both directions.
SOURCE = {NATIONAL, FREE, OTHER_LUTHERAN, CATHOLIC, ORTHODOX, OTHER_CHRISTIAN, ISLAM, EASTERN,
          ASATRU, OTHER_NONCHRISTIAN, OTHER, NO_RELIGION}

EXCLUDED = {
    REFUSAL:
        "Everyone whose `rlgblg` is Refusal, Don't know or No answer, and everyone who answered "
        "Yes and then declined to name a body. spec §3.5 marks a refusal rather than filling it; "
        "sources/is.py prints the share and `gap` states it.",
}

REVIEW = {
    NATIONAL:
        "-> christianity.lutheran. The Evangelical Lutheran Church of Iceland; 56.3% of the "
        "population on Hagstofa's register at 1 January 2026 (MAN10001: 222,012 of 394,324), "
        "printed by sources/is.py beside the survey. **Self-identification, not registration**: "
        "the survey's share among citizens is well under the register's.",
    FREE:
        "-> christianity.lutheran. The card's `Fríkirkjunni` is the Lutheran free congregations "
        "outside the national church, which Hagstofa's register lists as Fríkirkjan í Reykjavík "
        "(9,938 in 2026), Fríkirkjan í Hafnarfirði (7,919) and Óháði söfnuðurinn (2,932). "
        "Lutheran in confession, not in the state church, so they share the Church of Iceland's "
        "node, as Sweden's and Norway's Lutheran church answers do.",
    OTHER_LUTHERAN:
        "-> christianity.lutheran, on the card's own words, 'another Christian body within "
        "Lutheranism', and on `rlgdnm`, which nests it in `Protestant`. A respondent in a "
        "Pentecostal or evangelical church who read the box loosely would be misfiled here; "
        "those churches are small on the register (Hvítasunnukirkjan 2,028 in 2026).",
    CATHOLIC:
        "-> christianity.catholic.latin. One Latin diocese, Reykjavík; 15,558 on the register in "
        "2026, largely Polish, Lithuanian and Filipino in origin. Eastern-rite Catholics are not "
        "separable.",
    ORTHODOX:
        "-> christianity.orthodox, unsplit. The register's Orthodox bodies in 2026: Russian "
        "Orthodox 780, Serbian Orthodox 473, Ethiopian Tewahedo 26, so 98.0% Eastern. Rounds 6 and "
        "8 offered only `Rússnesku rétttrúnaðarkirkjunni` (the Russian Orthodox Church), "
        "harmonised to the later card's `Rétttrúnaðarkirkju` by sources/is.py.",
    OTHER_CHRISTIAN:
        "-> christianity, the root, following dk2024.py and no2024.py: 'another body outside "
        "Lutheranism' names no church, and on the register it would hold Pentecostals, "
        "Adventists, Jehovah's Witnesses and Latter-day Saints, which are different branches.",
    ISLAM:
        "-> islam.sunni, dk2024.py's and no2024.py's call. The register's Muslim bodies in 2026 "
        "are Stofnun múslima á Íslandi 870, the Islamic Cultural Centre of Iceland 690, Félag "
        "múslima á Íslandi 569 and the Ahmadiyya community 10; none is a Shia body. Rounds 6 and "
        "8 offered only `Félags múslima á Íslandi` (the Muslim Association of Iceland), "
        "harmonised to the later card's `Íslamstrú`.",
    EASTERN:
        "-> other.is with the next two answers, for dk2024.py's and no2024.py's reason: the box "
        "counts several traditions as one, and choosing Buddhism over Hinduism would invent a "
        "fact about a respondent.",
    ASATRU:
        "-> paganism. Ásatrúarfélagið reconstructs the pre-Christian Norse religion, which is what "
        "the `paganism` node holds (Wicca, Druidry, Neopaganism). A real body with its own box on "
        "the card, 6,158 on the register in 2026 (1.6% of Iceland), so it is drawn as itself "
        "rather than inside `other.is`, and no Iceland-only node is added.",
    OTHER_NONCHRISTIAN:
        "-> other.is, with whoever the card did not name: the Bahá'í, the Jewish cultural "
        "association, and any religion neither Christian nor Eastern.",
    OTHER:
        "-> other.is. Rounds 6 and 8 only; `rlgdnm` nests it in `Other Non-Christian religions`.",
    NO_RELIGION:
        "-> unaffiliated. Everyone who answered No to `rlgblg`, which includes many registered "
        "members of the Church of Iceland. ESS offers no humanist answer, so Siðmennt's registered "
        "members (6,387 in 2026) are inside this answer or the national church's, and nothing "
        "reaches `secular`.",
}

MAP = {
    NATIONAL: "christianity.lutheran",
    FREE: "christianity.lutheran",
    OTHER_LUTHERAN: "christianity.lutheran",
    CATHOLIC: "christianity.catholic.latin",
    ORTHODOX: "christianity.orthodox",
    OTHER_CHRISTIAN: "christianity",
    ISLAM: "islam.sunni",
    EASTERN: "other.is",
    ASATRU: "paganism",
    OTHER_NONCHRISTIAN: "other.is",
    OTHER: "other.is",
    NO_RELIGION: "unaffiliated",
}


def _key(cat):
    import unicodedata
    return unicodedata.normalize("NFC", " ".join(str(cat).split()))


def resolve(cat):
    """Source category -> taxonomy node id, or None if deliberately not on the tree."""
    c = _key(cat)
    if c in EXCLUDED:
        return None
    return MAP.get(c)
