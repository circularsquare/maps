"""ASK Census 2024 religion classification -> religiondots taxonomy.

Six categories plus the universe total, at municipality. A short list on a coarse
geography — the least detailed source in Europe on this map, and worth drawing anyway,
because the thing it shows is a boundary rather than a composition.

    93.49%  Islam                     -> islam
     2.31%  Orthodox                  -> christianity.orthodox.canonical
     1.75%  Catholic                  -> christianity.catholic
     1.50%  Prefers not to answer     -> EXCLUDED (explicit non-response)
     0.50%  No religious affiliation  -> unaffiliated
     0.45%  Others                    -> other.xk   (a NEW node)

**Read the Orthodox figure with sources/xk.py's warning in hand.** It is not an estimate of
Kosovo's Serbs; it is a count of the ones the 2024 census reached, and in the north it
reached almost none of them.

EXCLUDED holds categories that are deliberately not on the tree.
REVIEW holds calls that are defensible but arguable, with the reason.
"""

EXCLUDED = {
    "Total":
        "the unit's own population total, not a category.",
    "Prefers not to answer":
        "23,718 people, 1.50%, and it is not irreligion — `No religious affiliation` is a "
        "separate answer taken by 7,899. An explicit refusal cell, so spec §3.5 applies "
        "and these people are not drawn. **Its geography is Prishtinë**: 9,536 of the "
        "23,718 are in the capital, 4.2% of it, against 1.5% nationally — the same urban, "
        "educated pattern the refusal takes in Czechia and Hungary, at a twentieth of "
        "their magnitude. The one municipality above it is **Zveçan at 8.5%**, which is a "
        "different thing entirely: the north's census boycott leaving a trace inside the "
        "small population that did answer.",
}

REVIEW = {
    "Islam":
        "-> islam, the parent, with no branch. Kosovo's Muslims are overwhelmingly Hanafi "
        "Sunni and the census does not say so. What is lost by not splitting is the **Sufi "
        "tekkes** — Bektashi, Halveti, Rufai and Sa'di, with a real institutional presence "
        "in Gjakovë, Prizren and Rahovec and a history going back to the Ottoman period. "
        "They are inside this cell either way: a Kosovar dervish answers `Islam` on a "
        "census form, so no split of this category would have found them, and "
        "`alevism` — which the tree does hold — is a Turkish and Albanian category that "
        "the source never offers.",
    "Orthodox":
        "-> christianity.orthodox.canonical. The Serbian Orthodox Church's Eparchy of "
        "Raška and Prizren, canonical throughout, and the custodian of Gračanica, Dečani "
        "and the Patriarchate of Peć. **The 36,683 counted is far below any estimate of "
        "the Serb population of Kosovo**, because the north largely refused enumeration; "
        "sources/xk.py measures the gap against Kontur and countries.py states it on the "
        "map. Nothing here scales it up (§14.4).",
    "Catholic":
        "-> christianity.catholic, the parent. Kosovo's Catholics are **Albanian, not a "
        "foreign community**, and they are old: the Catholic Albanians of the Dukagjin "
        "plain never converted under Ottoman rule. The geography is sharp and is the "
        "reason a 1.75% category is worth drawing — **Klinë is 16.8% Catholic and Gjakovë "
        "14.6%**, both in the Dukagjin west, against 0.07% in Gjilan and 0.01% in Dragash "
        "on the other side of the country. Mother Teresa was of this community, from "
        "Skopje one province south.",
    "Others":
        "-> other.xk, a per-source residual (§3.11). See branches.py.",
    "No religious affiliation":
        "-> unaffiliated and NOT `secular`. One no-religion answer, no atheist/agnostic "
        "split, so the coarser node is the honest one. 7,899 people, 0.50% — **the "
        "smallest irreligious share of any country on this map**, and less than a third of "
        "the number who declined to answer.",
}

MAP = {
    "Islam": "islam",
    "Orthodox": "christianity.orthodox.canonical",
    "Catholic": "christianity.catholic",
    "Others": "other.xk",
    "No religious affiliation": "unaffiliated",
}


# --------------------------------------------------------------------------------------
# THE NORTHERN DERIVATION — spec §14.5, and the only ethnicity -> religion step outside China
# --------------------------------------------------------------------------------------
#
# The 2024 census did not reach the Serb population of the four northern municipalities, and
# what it did enumerate there is not a thin sample of them but a different population: three
# of the four come out MAJORITY MUSLIM in a table whose own author knows better. Drawing that
# asserts something false about contested ground; drawing nothing asserts something too.
#
# ASK publishes the correction itself, in `census2024_63.px` — *Population by ethnicity …
# (with estimation)* — which is identical to the enumerated table everywhere except those
# four, where it restores 16,949 people, 16,369 of them Serbs. **There is no
# religion-with-estimation table**, so using it at all means one ethnic category implying one
# religion. §14.5 permits that on three conditions, and Kosovo meets them more cleanly than
# China does:
#
#   1. **The category must itself have been constituted religiously.** The Serb/Croat/Bosniak
#      distinction is the textbook case in the literature — a religious boundary (Orthodox /
#      Catholic / Muslim) drawn over a common language, which is why it is a boundary at all.
#      China's nationalities meet this test unevenly; this one meets it squarely.
#   2. **The group must not be religiously mixed.** Kosovo's Serbs are Serbian Orthodox to a
#      degree that no serious source disputes. sources/xk.py asserts the addition is >90%
#      Serb, so the estimate cannot quietly become a mixed population without failing.
#   3. **No finer than the ethnicity is published at.** Both tables are per municipality, and
#      the derived rows go on the same four municipalities. Nothing is spread.
#
# What is NOT done: the 580 non-Serb people in the estimate are left undrawn (§3.5) — nothing
# says what they are, and the enumerated composition of these four is exactly the thing that
# is not representative. And the enumerated rows for the four are kept as they are: the
# derivation ADDS the missing Orthodox, it does not restate what was counted.
#
# Every derived row is `tier="derived"`, so §7a's control removes all of it in one click and
# a reader who does not want anybody's estimate can have the raw table back.
ETHNIC_DERIVATION = {
    "Serb": "christianity.orthodox.canonical",
}


def _key(cat):
    return " ".join(str(cat).split())


EXCLUDED = {_key(k): v for k, v in EXCLUDED.items()}
MAP = {_key(k): v for k, v in MAP.items()}
REVIEW = {_key(k): v for k, v in REVIEW.items()}


def resolve(cat):
    """Source category -> taxonomy node id, or None if deliberately not on the tree."""
    c = _key(cat)
    if c in EXCLUDED:
        return None
    return MAP.get(c)
