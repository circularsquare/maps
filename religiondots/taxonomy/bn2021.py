"""Brunei 2021 Population and Housing Census religion (DEPS, BPP 2021 Annex A, Table A4) ->
religiondots taxonomy.

Four cells on the four districts, 440,715 people (everyone enumerated, temporary residents
included), about 110,000 a district. sources/bn.md is the write-up.

The form's item E10 offers five answers: 1 Islam, 2 Christianity, 3 Buddhism, 4 Hindu, 5 Others
(please specify). There is no box for no religion or for not stated, and every published table
adds code 4 to code 5, so the table's four rows are Islam, Christianity, Buddhism and Others.

EXCLUDED holds categories that are deliberately not on the tree.
REVIEW holds calls that are defensible but arguable, with the reason.
"""

EXCLUDED = {}

REVIEW = {
    "Islam":
        "-> islam, no branch. 362,035 people, 82.15%. Islam is the state religion and the state "
        "follows the Shafi'i school of Sunni Islam, but the form asks only `Islam`, so "
        "`islam.sunni` would be an inference (the call my2020.py and bd2011.py make). 43,284 of "
        "these people are temporary residents (A11). Every district has a Muslim majority: "
        "Brunei Muara 84.47%, Tutong 84.23%, Temburong 75.46%, Belait 70.31%.",
    "Christianity":
        "-> christianity, the bare branch; DEPS prints no church. 29,462 people, 6.69%. **63.3% "
        "of them are temporary residents** (18,653, A11), so most of the layer is foreign "
        "workers and their families, and in Brunei Muara 14,893 of 20,076 are. Temburong is the "
        "exception: 955 of its 1,207 Christians are citizens (A12 (d)), 12.78% of the district. "
        "No table crosses religion with race, so the ethnic make-up is not published.",
    "Buddhism":
        "-> buddhism, no branch; the form names none. 27,745 people, 6.30%, split almost evenly "
        "between citizens (9,357), permanent residents (9,126) and temporary residents (9,262) "
        "(A11). Belait is the highest at 11.04%, and 4,418 of its 7,235 Buddhists are permanent "
        "residents. Chinese folk practice has no box of its own and may be recorded here or "
        "under Others; nothing published says which.",
    "Others":
        "-> other.bn, a new node. 21,473 people, 4.87%. **It is the form's code 4 Hindu and code "
        "5 Others (please specify) added together**, and with no box for no religion, anyone "
        "with none could only be recorded here, as could anyone following an indigenous "
        "religion. No table prints a non-response row and A4's total is the whole population, "
        "so what happened to a blank answer is not published. Its geography is two different "
        "things: in Tutong it is 11.00% of the district and 4,718 of 5,192 are citizens, 15.3% "
        "under 15, which is a settled population and not a migrant one; in Brunei Muara it is "
        "3.16%, and 7,615 of 10,074 are temporary residents, 71.5% of them men (A12 (a)), which "
        "reads as foreign workers. Drawn on one node because the source gives one number; "
        "splitting Hindus out or reading Tutong's as indigenous religion would be ours, not "
        "DEPS's. A lumped box, but not a no-religion box: its label is Others, and "
        "`unknown` (§6.3a-ii) would hide the Hindus, whom the form did ask about by name.",
}

MAP = {
    "Islam": "islam",
    "Christianity": "christianity",
    "Buddhism": "buddhism",
    "Others": "other.bn",
}

# spec §7a-i-1: every row is measured at the node it is drawn on.
COLUMNS = {v: v for v in MAP.values()}


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
