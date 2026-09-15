"""
Taiwan Social Change Survey, seven rounds 1994-2018, harmonised answers -> religiondots taxonomy.

**Twelve answers.** `sources/tw.py` folds each round's card into these (the short cards of 1994
and 2015 and the long card of the other five), so the labels are the harmonised ones and not
any single card's. The long card asks *what is your religious belief* and the interviewer codes
the answer; *worships the gods* (拜神) is coded as folk religion and *Buddha worship* (拜佛) as
Buddhism, which is why the Buddhist and folk shares move between cards and why the level is read
from 2014 and 2018 only.

Every row is `modelled` (§7). Named for the last round, per the registry convention.
"""

EXCLUDED = {}

REVIEW = {
    "Folk religion":
        "-> chinesefolk. The card's four folk codes (self-identified; worships the gods; not "
        "clearly specified; other folk religion) are one answer. Its node note is Chinese folk "
        "religion and the syncretic practice censuses cannot separate, which is this exactly.",
    "Buddhism and Taoism, or the three teachings":
        "-> chinesefolk, not split between buddhism and daoism. Long-card codes 101-103: both "
        "Buddhism and Taoism, the three teachings in one, other polytheism. Splitting a person "
        "who named two traditions would invent a proportion nobody gave; `chinesefolk` is the "
        "tree's node for the combination (§3.3). About 2% of answers.",
    "Buddhism":
        "-> buddhism.mahayana. Every Buddhist code (Buddha worship, Pure Land, Chan, Mi Zong, "
        "Chan and Pure Land, exoteric and esoteric, other, don't know which). Taiwan's Buddhism "
        "is the Chinese Mahayana transmission. Mi Zong, the esoteric school, is under 1% of "
        "answers and includes Tibetan lineages that belong under Vajrayana; the file does not "
        "separate them, so they stay here.",
    "Yiguan Dao":
        "-> eastasiannew, the family's own node, drawn as unspecified. Yiguan Dao (I-Kuan Tao) "
        "is a 20th-century Chinese founding out of the Xiantiandao sects, legal in Taiwan since "
        "1987. The tree has Japanese, Korean and Vietnamese children and no Chinese one; a "
        "Chinese child would be a legend row only Taiwan uses (AGENT_BRIEF §3), so it is not "
        "added here. About 2% of answers, the largest single new religion on the card.",
    "Other Chinese religions":
        "-> eastasiannew, with Yiguan Dao. Cihui Tang (1949, Hualien), Tiandi jiao (1980), "
        "Tiande jiao and the card's `other local religion`. Nineteen respondents in seven "
        "rounds.",
    "Japanese religions":
        "-> eastasiannew.japanese. Soka Gakkai, Nichiren Shoshu and `other religions in Japan`. "
        "Soka Gakkai sits there in jp2024 too; Nichiren Shoshu is a Buddhist school, not a new "
        "religion, and is two of the respondents.",
    "Other":
        "-> other.tw. The long card's `cannot be categorized`, `other foreign religion` and "
        "the Unification Church (one respondent), and the short cards' `other`.",
    "Catholicism":
        "-> christianity.catholic.latin. Taiwan's dioceses are Latin rite.",
}

MAP = {
    "No religious belief": "unaffiliated",
    "Folk religion": "chinesefolk",
    "Buddhism": "buddhism.mahayana",
    "Taoism": "daoism",
    "Buddhism and Taoism, or the three teachings": "chinesefolk",
    "Yiguan Dao": "eastasiannew",
    "Protestant Christianity": "christianity.protestant",
    "Catholicism": "christianity.catholic.latin",
    "Japanese religions": "eastasiannew.japanese",
    "Other Chinese religions": "eastasiannew",
    "Islam": "islam",
    "Other": "other.tw",
}

# No COLUMNS dict (spec §7a-i-1): nothing here is `derived`, every row is `modelled`.


def resolve(category):
    """religiondots branch for a harmonised TSCS answer, or None if deliberately off the tree."""
    if category in EXCLUDED:
        return None
    return MAP.get(category)
