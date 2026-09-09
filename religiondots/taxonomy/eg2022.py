"""
Arab Barometer `Q1012` (Egypt, waves III, IV, V and VII) -> religiondots taxonomy.

**Branch-level mapping, and the shallowest one in the project.** No leaves are created; the
answer's own wording travels with the row in `source_category` (spec §2.4).

The file is named for the last wave in the pool, which is the registry convention
(`taxonomy/<cc><YYYY>.py`); wave VII's Egyptian fieldwork is January 2022 and wave III's is
April 2013. `sources/eg.py` and `sources/eg.md` have the construction.

    93.98%  Muslim      -> islam
     6.02%  Christian   -> christianity

Shares are as drawn, which is the pooled weighted governorate shares against CAPMAS's own
2026-01-01 governorate populations; every row is `modelled` in §7's sense.

**A TWO-BOX CARD IS THE WHOLE OF WHAT THIS SOURCE MEASURES, AND NEITHER BOX IS DEEPENED.**
Egypt is about 90% Sunni and its Christians are overwhelmingly Coptic Orthodox, and both of
those are true and neither is in the data. Spec §2.6 forbids assigning a school from outside
the source and §14.4 rule 1 forbids inventing a magnitude, so the two answers sit on the bare
family nodes. See REVIEW, where the case for going deeper is stated properly before being
declined, because it is a case a reader will raise.

**`Atheist` never reaches this file.** Two wave V respondents chose it, wave VII's card offers
`No religion` instead and waves III and IV offer neither, so the pooled share would measure the
questionnaire. `sources/eg.py`'s `DROPPED` carries the argument and drops them from the universe
before the CSV is written; there is nothing here to exclude.
"""

EXCLUDED = {}

REVIEW = {
    "Muslim":
        "-> islam, the bare family node, NOT `islam.sunni`. Egypt is somewhere around 90% "
        "Sunni and the Shia minority is small and unmeasured, so sending this to the Sunni "
        "node would be right about most of the people and would still be **an assertion the "
        "source does not make**. The card has one Muslim box. `branches.py`'s own note on "
        "`islam.sunni` says the thing: *a census that asks about religion at all normally "
        "stops at 'Muslim'*, and this survey does too. "
        "The survey does carry `Q1012A_MUSLIM`, a denomination follow-up, and it is not a "
        "way in: §11af measured it across five Arab countries and 45-82% of respondents "
        "answer *just a Muslim*, so it records which label a person volunteers rather than "
        "which school they follow.",
    "Christian":
        "-> christianity, the bare family node, NOT `christianity.oriental`. This is the call "
        "in this file worth arguing, because the Copts ARE the Oriental Orthodox communion's "
        "largest church and roughly nine in ten Egyptian Christians belong to it; the "
        "remainder are Coptic Catholics, Greek Orthodox, the Evangelical Church of Egypt and "
        "several smaller bodies. "
        "It stays on the family for two reasons. **The source does not say it** (spec §2.6), "
        "and the one wave that asks Christians a follow-up question, wave VII's "
        "`Q1012A_CHRISTIAN`, has 66 Egyptian Christians in it, which cannot carry a national "
        "split let alone a governorate one. And **the nine-in-ten is not a published Egyptian "
        "figure** but an estimate assembled from church claims, so applying it would be "
        "inventing the magnitude that §14.4 rule 1 exists to forbid. "
        "The practical cost is small and worth stating: `christianity` and "
        "`christianity.oriental` draw as different colours, so Egypt's Christians do not "
        "share a colour with Ethiopia's or Armenia's on this map even though most of them "
        "share a communion. That is the source's silence being drawn, which is what this "
        "project wants a blank to mean.",
}

MAP = {
    "Muslim": "islam",
    "Christian": "christianity",
}

# No COLUMNS dict (spec §7a-i-1). Every row in this country is `modelled` rather than
# `derived` — §7b's test, applied in countries.py::_eg_counts — and the roll-up is about where
# a DERIVED row was actually counted. Nothing here was counted anywhere.


def resolve(category):
    """religiondots branch for an Arab Barometer answer, or None if deliberately off the tree."""
    if category in EXCLUDED:
        return None
    return MAP.get(category)
