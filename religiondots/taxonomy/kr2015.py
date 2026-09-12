"""KOSIS 인구총조사 2015 religious affiliation -> religiondots taxonomy.

Twelve categories, two of them universes, at si/gun/gu. See sources/kr.md.

**This is the country that paid for its own nodes.** Eight of the ten drawn categories were
already on the tree or needed only a child; the other four are religions no other source on
this map counts at all, and they are the entire argument for drawing South Korea:

    원불교      Won Buddhism        84,141   -> buddhism.won                    (new)
    유교        Confucianism        75,703   -> confucianism                    (new root)
    천도교      Cheondogyo          65,964   -> eastasiannew.korean.cheondogyo  (new)
    대순진리회  Daesun Jinrihoe     41,176   -> eastasiannew.korean.daesun      (new)
    대종교      Daejonggyo           3,101   -> eastasiannew.korean.daejong     (new)

Korea is also why `japanesenew` became `eastasiannew` with Japanese and Korean children: a
Japanese-only node could not hold these, and a second root would have cost the palette a
degree it does not have (branches.py, §12).

**AND IT IS THE LAST TIME ANY OF THEM WILL BE COUNTED.** The religion question was dropped
after 2015.

EXCLUDED holds categories that are deliberately not on the tree.
REVIEW holds calls that are defensible but arguable, with the reason.
"""

EXCLUDED = {
    "계":
        "the unit's own population total, not a category. 49,052,389 nationally.",
    "종교있음-계":
        "'has a religion, total' — the sum of the nine named religions, and an intermediate "
        "universe. 21,553,674 nationally, 43.94%. Drawing it would double every religion "
        "beneath it (spec §12). Its complement, 종교없음-계, IS drawn, because 'no religion' "
        "is an answer rather than a universe.",
}

REVIEW = {
    "불교":
        "-> buddhism, the undivided parent, NOT buddhism.mahayana. 7,619,332 people, 15.5%. "
        "Korean Buddhism is overwhelmingly the Jogye Order and is Mahayana by any reading, "
        "so the school here is knowable — but it is not STATED, and lk2024.py set the "
        "precedent by keeping Sri Lanka's 15.2M undivided `Buddhist` on the parent for "
        "exactly that reason. Anita confirmed the same treatment on 2026-09-05. The cost is "
        "that `buddhism.mahayana` stays empty, which its own node note already laments; the "
        "gain is that the map never asserts a school a census did not ask about. If that "
        "trade is ever reversed, Korea is the country to reverse it with.",
    "기독교(개신교)":
        "-> christianity.protestant, the 'named no body' node. 9,675,761 people, 19.7%, the "
        "largest religion in South Korea. The census offers one Protestant box and no "
        "denomination, which is a real loss here: Korean Protestantism is majority "
        "Presbyterian — the Hapdong and Tonghap assemblies alone are larger than most "
        "European national churches — with substantial Methodist, Baptist and Holiness "
        "bodies, all of which this tree could hold apart. Nothing in KOSIS reaches them.",
    "기독교(천주교)":
        "-> christianity.catholic. 3,890,311 people, 7.9%.",
    "원불교":
        "-> buddhism.won, a node added for Korea. 84,141 people. Filed under Buddhism "
        "because that is the movement's own self-description and spec §2.1's relation is "
        "containment. Scholars commonly class it with the Korean new religions instead, "
        "which would make it `eastasiannew.korean.won`; the branches.py note records that "
        "reading so it can be taken later without re-investigation.",
    "유교":
        "-> confucianism, a NEW ROOT, added on Anita's call 2026-09-05. 75,703 people. "
        "**The only cell on this map anywhere that counts Confucianism as a religion.** "
        "What it is not: a count of Confucian practice, which in Korea is ancestral rites "
        "and lineage observance and is near-universal. These are people naming it as their "
        "religion — the institutional core around Seonggyungwan and the local hyanggyo — "
        "which is a much smaller and more deliberate claim. Do not read the map as 'only "
        "0.15% of Koreans are Confucian'.",
    "천도교":
        "-> eastasiannew.korean.cheondogyo. 65,964 people. The continuation of Donghak, "
        "which is a nationalist and anti-colonial lineage as much as a religious one — its "
        "leaders wrote the 1919 Declaration of Independence. Filed as a new religion rather "
        "than as `indigenous` because it has a founder, a date and a doctrine.",
    "대순진리회":
        "-> eastasiannew.korean.daesun. 41,176 people. The largest Jeungsanist body. Its own "
        "membership claims are far higher than the census figure; spec §3.1 says the source "
        "counts what it counts and this file does not adjudicate between a census and a "
        "movement's self-report.",
    "대종교":
        "-> eastasiannew.korean.daejong. 3,101 people, the smallest cell in the table and "
        "the weakest fit in its family. The Dangun religion is ethnic-national rather than "
        "syncretic-new, and `indigenous` was considered; it lost because Daejonggyo is a "
        "1909 REVIVAL with a hierarchy and a scripture rather than a continuous folk "
        "tradition, which is the distinction that node is for.",
    "기타":
        "-> other.kr. 98,185 people, 0.20%. Small, because the named list above it is "
        "generous. It carries Islam — Korea's Muslim population is mostly foreign residents "
        "and outside this table's universe anyway — the Unification Church, which the census "
        "does not name despite being Korean and internationally the best known of these "
        "movements, and the rest of the Jeungsanist and Donghak tail.",
    "종교없음-계":
        "-> unaffiliated. 27,498,715 people, **56.06% — the largest single answer in the "
        "country and a majority of it.** South Korea is the most irreligious country drawn "
        "on this map by this measure. That is a real finding and not a non-response: "
        "refusals are not in this table at all (see sources/kr.md §5 on the universe gap).",
}

MAP = {
    "불교": "buddhism",
    "기독교(개신교)": "christianity.protestant",
    "기독교(천주교)": "christianity.catholic",
    "원불교": "buddhism.won",
    "유교": "confucianism",
    "천도교": "eastasiannew.korean.cheondogyo",
    "대순진리회": "eastasiannew.korean.daesun",
    "대종교": "eastasiannew.korean.daejong",
    "기타": "other.kr",
    "종교없음-계": "unaffiliated",
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
