"""
NSO Malta, Census of Population and Housing 2021, Final Report Volume 1, Table 5.3 ->
religiondots taxonomy.

**Ten answers, a flat partition of the population aged 15 and over, by the 68 localities.**
The ten sum to every row's total and there is no not-stated column; `sources/mt.py` shows the
totals equal each locality's population aged 15 and over in Table 1.5, so the only people not
drawn are the under-15s, who were not asked (`countries/mt.py` `gap`).

    Roman Catholicism           373,304  82.64%  -> christianity.catholic.latin   REVIEW
    No religious affiliation     23,243   5.15%  -> unaffiliated
    Islam                        17,454   3.86%  -> islam
    Orthodoxy                    16,457   3.64%  -> christianity.orthodox         REVIEW
    Hinduism                      6,411   1.42%  -> hinduism
    Church of England             5,706   1.26%  -> christianity.anglican
    Protestantism                 4,516   1.00%  -> christianity.protestant       REVIEW
    Buddhism                      2,495   0.55%  -> buddhism
    Judaism                       1,249   0.28%  -> judaism
    Other religious groups          911   0.20%  -> other.mt   (a NEW node)       REVIEW

The questionnaire is not in Volume 1 and was not read (`census2021.gov.mt` no longer resolves,
`nso.gov.mt` answers 403). So whether these ten are the form's boxes or the office's coding of
written answers is not known; the labels are the tables' own. Buddhism names no school and stays
on the plain node (ask 025's ruling: a source that names none stays on `buddhism`).

EXCLUDED holds categories that are deliberately not on the tree (none).
REVIEW holds calls that are defensible but arguable, with the reason.
"""

EXCLUDED = {
    # Nothing. sources/mt.py writes only the ten answers, never a total row.
}

REVIEW = {
    "Roman Catholicism":
        "-> christianity.catholic.latin, as every other census that prints `Roman Catholic` "
        "(be2024, cy2021, ie2022, it2024, gi2022). **373,304 people.** An Eastern Catholic who "
        "answered here is filed Latin, and there is one nameable group that could hold some: "
        "Table 5.5 counts 2,505 Indian citizens as Roman Catholic, and Kerala's Syro-Malabar "
        "church is Eastern-rite. Nothing splits them, and they are 0.7% of the category.",

    "Orthodoxy":
        "-> christianity.orthodox (Eastern Orthodox, the parent), NOT `.canonical` and NOT the "
        "`christianity` parent. **16,457 people, 3.64%.** One word covers every Orthodox "
        "answer, and Malta's Orthodox are mostly Eastern: by citizenship (Table 5.5) Serbian "
        "4,202, other European countries 3,543, other EU 3,195, Bulgarian 2,391, Maltese "
        "1,346. Some are Oriental Orthodox, which is a separate communion "
        "(`christianity.oriental`): by racial origin (Table 5.7) 587 Asian, 498 African and "
        "197 Arab, and by citizenship 224 Indian, which is where Malankara, Eritrean, Ethiopian "
        "and Coptic answers would sit. Those three racial-origin groups together, 1,282 people "
        "(7.8%), are a ceiling on the Oriental share, not an estimate of it. Not `.canonical`, "
        "because the census names no jurisdiction; `at2001` and `hu2022` put a bare `Orthodox` "
        "on the same parent.",

    "Protestantism":
        "-> christianity.protestant, the node for 'Protestant' given as an answer with no body "
        "named. **4,516 people, 1.00%.** Evangelical and Pentecostal answers presumably sit "
        "inside it, since no category names them: other EU citizens are 1,829 of the 4,516, "
        "and 706 are of Asian and 283 of African racial origin (Tables 5.5, 5.7). The Church "
        "of England has its own answer, so British Protestants here, 452, are the ones who did "
        "not give that one.",

    "Other religious groups":
        "-> other.mt, a per-source residual (§3.11). **911 people, 0.20%.** Nothing in the "
        "volume says what it holds. The biggest citizenship in it is Indian, 202 (161 of them "
        "men, Table 5.5), which fits Sikhs, since the census has no Sikh answer; then Maltese "
        "190, British 90 and Italian 79. Where Jehovah's Witnesses and Latter-day Saints were "
        "filed is not stated either.",
}

COLUMNS = {
    # Every category is counted at the locality, which is the unit drawn, so no row is
    # derived and nothing rolls up. Recorded per COMMANDS.txt's check_rollup note.
}

MAP = {
    "Roman Catholicism":        "christianity.catholic.latin",
    "Islam":                    "islam",
    "Orthodoxy":                "christianity.orthodox",
    "Hinduism":                 "hinduism",
    "Church of England":        "christianity.anglican",
    "Protestantism":            "christianity.protestant",
    "Buddhism":                 "buddhism",
    "Judaism":                  "judaism",
    "Other religious groups":   "other.mt",
    "No religious affiliation": "unaffiliated",
}


def resolve(category):
    """Source category -> node, or None for a category deliberately not on the tree."""
    if category in EXCLUDED:
        return None
    return MAP.get(category)
