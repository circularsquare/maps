"""Japan: JGSS XXRL (four-digit religion codes), pooled 2021H-2024N -> religiondots nodes.

`sources/jp.py` writes `data/normalized/jp.csv` with `source_category` = "<code> <label>", so
this module resolves on the leading four digits and every code on the page is placed, including
the 55 nobody in the current pool chose, so a re-pool cannot arrive unmapped.

Pooled shares, 10,612 respondents (sources/jp.md §1 has the full table):

    8888 非該当 (no religion, no family religion)   71.68%  -> unaffiliated
    Buddhism, no school given (2000)                7.01%  -> buddhism.mahayana
    Pure Land: Jodo Shinshu, Jodo-shu, Ji, Yuzu     6.80%  -> buddhism.mahayana.pureland
    Zen: Soto, Rinzai, Obaku                        1.84%  -> buddhism.mahayana.zen
    Shingon                                         1.48%  -> buddhism.mahayana.shingon
    Nichiren schools                                1.43%  -> buddhism.mahayana.nichiren
    Tendai                                          0.24%  -> buddhism.mahayana.tendai
    Japanese new religions (Soka Gakkai 1.80%)      2.97%  -> eastasiannew.japanese
    Christianity, all codes                         1.14%  -> the Christian branches
    Shinto and the sect Shinto bodies               0.83%  -> shinto
    combinations, ancestor veneration, unclassed    0.84%  -> other.jp
    9999 no answer, 8700 don't know                 3.60%  -> not drawn
"""

import re

EXCLUDED = {
    "9999 無回答":
        "No answer: DORL left blank, or DORL answered yes / family religion and XXRL left "
        "blank. spec §3.5 marks it rather than filling it; stated in `gap`.",
    "8700 回答者が「わからない」と回答":
        "The respondent wrote that they did not know. Same treatment as a blank.",
}

SHINTO = {1000, 1001, 1002, 1003, 1004, 1006, 1007, 1008, 1009}
NEW_FROM_SHINTO = {1005, 1010, 1011, 1012, 1013, 1014, 1015, 1016, 1017, 1018}
NEW_FROM_BUDDHISM = {2209, 2210, 2809, 2811, 2812, 2813, 2814, 2815, 2816, 2817, 2818,
                     2902, 2903, 2904, 2905, 2906, 2907, 2908, 2909, 2910}
# The traditional schools, from the page's own code list (2026-09-14). Shugendo (2103, 2104)
# stays on buddhism.mahayana: it is mountain asceticism with Tendai and Shingon branches both,
# and nobody in the pool chose it. 2901 正法事門法華宗 is a Hokke school the page files under 29xx.
TENDAI = {2101, 2102}
SHINGON = set(range(2200, 2209))
PURELAND = {2301, 2302, 2303, 2400, 2401, 2402, 2403, 2404, 2501, 2601}
ZEN = set(range(2700, 2800))
NICHIREN = (set(range(2801, 2820)) | {2901}) - NEW_FROM_BUDDHISM
SCHOOLS = [(TENDAI, "buddhism.mahayana.tendai"), (SHINGON, "buddhism.mahayana.shingon"),
           (PURELAND, "buddhism.mahayana.pureland"), (ZEN, "buddhism.mahayana.zen"),
           (NICHIREN, "buddhism.mahayana.nichiren")]
CHRISTIAN = {
    3000: "christianity",
    3100: "christianity.catholic.latin",
    3200: "christianity.orthodox",
    3300: "christianity.protestant",
    3301: "christianity.anglican",
    3302: "christianity.united",
    3303: "christianity.lutheran",
    3304: "christianity.reformed",
    3305: "christianity.reformed",
    3306: "christianity.baptist",
    3307: "christianity.baptist",
    3308: "christianity.latterday",
    3309: "christianity.adventist",
    3310: "christianity.protestant",
    3401: "unification",
    3402: "christianity.witnesses",
}
OTHER = {8000, 8500} | set(range(5001, 5027))

REVIEW = {
    "Buddhist schools -> five children of buddhism.mahayana":
        "Anita said yes to the schools as Japan-only legend rows in ask 014 (2026-09-14), which "
        "reverses the first build's single node. The split follows what the prefecture source "
        "can place, not only what JGSS names: NHK 1996 asked Jodo-shu and Jodo Shinshu as one "
        "answer, so they are one node, `pureland`, rather than two that would draw Jodo-shu "
        "(1.2%) with Jodo Shinshu's (5.5%) Hokuriku map. It also asked Tendai and Shingon as "
        "one answer; those stay two nodes because nothing names the pair as a tradition, and "
        "they share a map, which Shingon, six times larger, mostly sets. 2000 仏教（宗派不明）, "
        "7.01% and the largest Buddhist code, stays on the parent. Shingon is kept under "
        "Mahayana, the usual placement for East Asian esoteric Buddhism; `buddhism.vajrayana` "
        "is the Tibetan tradition. Soka Gakkai and the other lay movements stay on "
        "`eastasiannew.japanese` (below).",
    "2905 創価学会 -> eastasiannew.japanese":
        "Soka Gakkai is a Nichiren Buddhist lay movement founded in 1930 and the node's own "
        "note names it. 1.80% of the pool, the largest single body after the schools. The "
        "other lay and new movements with a Buddhist lineage go the same way: Rissho "
        "Kosei-kai, Reiyukai and its offshoots, Shinnyo-en, Agonshu, Kenshokai, Happy "
        "Science, Jodo Shinshu Shinrankai.",
    "1xxx split between shinto and eastasiannew.japanese":
        "JGSS files everything of Shinto descent under 1xxx. 1000 神道 and the prewar sect "
        "Shinto bodies it names (Izumo-kyo, Kurozumikyo, Izumo Oyashirokyo, Maruyamakyo, "
        "Ontakekyo, Shinrikyo, Misogikyo, Konkokyo) -> shinto. Oomoto and the twentieth-"
        "century foundings (Sekai Shindokyo, Shoroku Shinto Yamatoyama, World Mate, "
        "Shinji Shumeikai and the Okada Mokichi groups) -> eastasiannew.japanese. Konkokyo "
        "is the arguable one: it was one of the thirteen recognised sect Shinto bodies and is "
        "also routinely described as a new religion. 13 respondents.",
    "5001 複数回答（神道、仏教） -> other.jp":
        "38 respondents, 0.36%, named Shinto and Buddhism together. spec §3.3's test for a "
        "combination node is met exactly (the source reports the combination), and the node "
        "it would open is §3.3's own example, `japan.shinbutsu`. Not opened here: it would be "
        "a single-country legend row, which is Anita's, and §11q-i records that the roll's "
        "150% total is NOT evidence for it. The other 25 combination codes (15 respondents) "
        "go the same way.",
    "8000 先祖供養 / 8500 unclassifiable -> other.jp":
        "Ancestor veneration named as the religion (17), and answers the coders could not "
        "place (18). Neither is a body; both are real answers, so they are drawn in the "
        "country's own residual rather than as unknown.",
    "3302 日本基督教団 -> christianity.united":
        "The United Church of Christ in Japan, formed in 1941 when the government merged the "
        "Protestant denominations; the largest Protestant body in Japan.",
    "3401 世界平和統一家庭連合 -> unification":
        "The tree carries the Unification Church as its own root, from ASARB. The Agency "
        "for Cultural Affairs files it under キリスト教系; JGSS files it under 3xxx too.",
    "family-religion answers are drawn as the religion named":
        "DORL's middle answer, 特に信仰していないが、家の宗教はある, leads to XXRL like a yes. "
        "About two thirds of the people drawn under a religion here gave that answer in "
        "JGSS-2015 (21.2% of 30.4%). Drawing only the yes answers would put Japan at about 9% "
        "religious, against 28% for ISM's 'personal religious faith' and 31% for NHK's 1996 "
        "'which religion do you believe in'; drawing both lands with them.",
}


def _code(cat):
    m = re.match(r"\s*(\d{4})\b", str(cat))
    return int(m.group(1)) if m else None


def resolve(cat):
    """Source category -> taxonomy node id, or None if deliberately not on the tree."""
    if str(cat).strip() in EXCLUDED:
        return None
    c = _code(cat)
    if c is None or c in (8700, 9999, 6000):
        return None
    if c == 8888:
        return "unaffiliated"
    if c in SHINTO:
        return "shinto"
    if c in NEW_FROM_SHINTO or c in NEW_FROM_BUDDHISM or 4001 <= c <= 4504:
        return "eastasiannew.japanese"
    for codes, node in SCHOOLS:
        if c in codes:
            return node
    if 2000 <= c <= 2999:
        return "buddhism.mahayana"
    if c in CHRISTIAN:
        return CHRISTIAN[c]
    if c == 4601:
        return "islam"
    if c == 4602:
        return "hinduism"
    if c == 4603:
        return "confucianism"
    if c in OTHER:
        return "other.jp"
    return None


# coverage.py reads MAP's VALUES as the nodes this classification can express. This module
# resolves on code ranges, so MAP is generated from resolve() over the whole four-digit code
# space the page uses, keyed by code. It includes nodes nobody in the current pool chose
# (Hinduism, Confucianism): the card has codes for them, so Japan lights as "asked" for them.
MAP = {str(c): n for c in list(range(1000, 5027)) + [8000, 8500, 8888]
       if (n := resolve(str(c))) is not None}
