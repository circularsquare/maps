"""
Taiwan Social Change Survey, seven rounds 1994-2018, harmonised answers -> religiondots taxonomy.

**Twelve answers.** `sources/tw.py` folds each round's card into these (the short cards of 1994
and 2015 and the long card of the other five), so the labels are the harmonised ones and not
any single card's. The long card asks *what is your religious belief* and the interviewer codes
the answer; *worships the gods* (拜神) is coded as folk religion and *Buddha worship* (拜佛) as
Buddhism, which is why the Buddhist and folk shares move between cards and why the level is read
from 2014 and 2018 only.

**Fifteen in `tw.csv` since 2026-09-15 (spec §3.13).** `sources/tw_altar.py` splits the drawn folk
answer into self-identified (code 021) and the interviewer-coded rest with and without a religious
home altar, and the no-religion answer by the altar, so `Folk religion` itself no longer reaches
the file.

Every row is `modelled` (§7). Named for the last round, per the registry convention.
"""

EXCLUDED = {}

REVIEW = {
    "Folk religion, self-identified":
        "-> chinesefolk. Long-card code 021, the respondent volunteers 'folk religion': 4.0% of "
        "the folk answer in 2014 and 2018, applied nationally. Until 2026-09-15 all four folk "
        "codes were one answer on this node; spec §3.13 split them (sources/tw_altar.py).",
    "Folk religion, worships the gods, religious altar at home":
        "-> chinesefolk, under spec §3.13 (Anita, 2026-09-15): folk religion named, or a religious "
        "altar kept by someone who names no religion. Codes 022-024 are interviewer-coded, and "
        "the 2018 report's rules put 'no religion but worships along with my family' on 022, so "
        "these people did not name folk religion; the ISSP altar item (2009 havshrin, 2014 v27, "
        "2018 v55) is what puts them here. County rates, shrunk.",
    "Folk religion, worships the gods, no religious altar":
        "-> unknown, the node China and Hong Kong use for people who named no religion and whose "
        "practice this map does not establish. Not `unaffiliated`: the interviewer coded worship "
        "of the gods, and calling 13% of Taiwan irreligious on that coding would say the "
        "opposite of it.",
    "No religious belief, no religious altar":
        "-> unknown, Anita 2026-09-15 (\"ah yeah we can switch taiwan to unknown\"), so that "
        "people who name nothing and keep no altar are the same grey in China, Taiwan and Hong "
        "Kong. Code 010 without a religious altar, 6.35% of Taiwan. The first build that day "
        "left it on `unaffiliated` as `No religious belief`.",
    "No religious belief, religious altar at home":
        "-> chinesefolk, spec §3.13. Code 010 with a religious altar: 47.4% of those answers "
        "(2009-2018 weighted), at the national rate because the county test missed (p 0.052).",
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
    "No religious belief, no religious altar": "unknown",
    "No religious belief, religious altar at home": "chinesefolk",
    "Folk religion, self-identified": "chinesefolk",
    "Folk religion, worships the gods, religious altar at home": "chinesefolk",
    "Folk religion, worships the gods, no religious altar": "unknown",
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
