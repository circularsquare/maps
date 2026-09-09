"""
Arab Barometer `Q1012` (Jordan, waves II to VIII) -> religiondots taxonomy.

**Branch-level mapping, and the same shallow shape as Egypt's.** No leaves are created; the
answer's own wording travels with the row in `source_category` (spec §2.4).

The file is named for the last wave in the pool, which is the registry convention
(`taxonomy/<cc><YYYY>.py`); wave VIII's fieldwork ran September 2023 to July 2024 and wave
II's was 2010-2011. `sources/jo.py` and `sources/jo.md` have the construction.

    98.60%  Muslim      -> islam
     1.40%  Christian   -> christianity

Shares are as drawn, which is the pooled weighted governorate shares against the Department of
Statistics' own end-2025 governorate populations; every row is `modelled` in §7's sense,
because nobody has published a count of religion in Jordan.

**TWO BOXES ARE THE WHOLE CARD IN EVERY ONE OF THE NINE WAVES, AND NEITHER IS DEEPENED.**
Jordan's Muslims are overwhelmingly Sunni and its Christians are mostly Greek Orthodox, with
sizeable Greek Catholic (Melkite), Latin, Syriac and Protestant communities; both of those are
true and neither is in the data. Spec §2.6 forbids assigning a school from outside the source
and §14.4 rule 1 forbids inventing a magnitude, so the two answers sit on the bare family
nodes. See REVIEW, which states the case for going deeper properly before declining it,
because a reader who knows Jordan will raise it.
"""

EXCLUDED = {}

REVIEW = {
    "Muslim":
        "-> islam, the bare family node, NOT `islam.sunni`. Jordan is about as Sunni as a "
        "country gets — the Shia and Ismaili presence is tiny and the state's own religious "
        "establishment is Sunni — so sending this to the Sunni node would be right about "
        "essentially everybody and would still be **an assertion the source does not make**. "
        "The card has one Muslim box in all nine waves. `branches.py`'s own note on "
        "`islam.sunni` says the thing: a census that asks about religion at all normally "
        "stops at `Muslim`, and this survey does too. "
        "The survey does carry `Q1012A_MUSLIM`, a denomination follow-up, and it is not a way "
        "in: §11af measured it across five Arab countries and 45-82% of respondents answer "
        "*just a Muslim*, so it records which label a person volunteers rather than which "
        "school they follow. Egypt (taxonomy/eg2022.py) declined it for the same reason and "
        "this file follows that, deliberately rather than by habit.",
    "Christian":
        "-> christianity, the bare family node, NOT `christianity.eastern`. This is the call "
        "worth arguing here. Jordan's Christians really are mostly Eastern Orthodox: the "
        "Greek Orthodox Patriarchate of Jerusalem is the largest church by a wide margin, and "
        "sending 166,950 people to `christianity.eastern` would be closer to the truth than "
        "leaving them on the family node. "
        "It stays on the family for three reasons and the third is the decisive one. **The "
        "source does not say it** (spec §2.6). **The size of the Orthodox majority is not a "
        "published Jordanian figure** — the shares that circulate come from church claims and "
        "from journalism, and applying one would invent the magnitude §14.4 rule 1 exists to "
        "forbid. And **the Melkite Greek Catholic community is large enough that the split is "
        "not a rounding error**: it is routinely put at a fifth to a third of Jordanian "
        "Christians and it belongs on `christianity.catholic`, not on `christianity.eastern`, "
        "so a wrong guess here would not be a shade off, it would be tens of thousands of "
        "people on the wrong branch. "
        "The practical cost is the same one Egypt pays and is worth stating: `christianity` "
        "and `christianity.eastern` draw as different colours, so Jordan's Christians do not "
        "share a colour with Greece's or Cyprus's on this map even though most of them share "
        "a communion. That is the source's silence being drawn, which is what this project "
        "wants a blank to mean.",
}

MAP = {
    "Muslim": "islam",
    "Christian": "christianity",
}

# No COLUMNS dict (spec §7a-i-1). Every row in this country is `modelled` rather than
# `derived` — §7b's test, applied in countries.py::_jo_counts — and the roll-up is about where
# a DERIVED row was actually counted. Nothing here was counted anywhere.


def resolve(category):
    """religiondots branch for an Arab Barometer answer, or None if deliberately off the tree."""
    if category in EXCLUDED:
        return None
    return MAP.get(category)
