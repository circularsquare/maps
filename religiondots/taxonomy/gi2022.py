"""
Census of Gibraltar 2022 (HM Government of Gibraltar), Table 42, total row -> religiondots
taxonomy.

**Eight answers, a flat partition of the usually-resident population, summing to the census's
own 37,936 with a difference of zero.** One geography, which is the territory; see
`sources/gi.py` and `sources/gi.md` §5 for why the seven residential areas the same table prints
are not drawn.

    Roman Catholic       24,098  63.52%  -> christianity.catholic.latin
    No Religion           5,343  14.08%  -> unaffiliated
    Church of England     2,541   6.70%  -> christianity.anglican
    Muslim                1,909   5.03%  -> islam
    Other Christian       1,503   3.96%  -> christianity                  REVIEW
    Jewish                1,070   2.82%  -> judaism
    Other/Not stated        779   2.05%  -> other.gi   (a NEW node)       REVIEW
    Hindu                   693   1.83%  -> hinduism

**The form's question 11 (report p.480) offers exactly eight boxes**: Roman Catholic, Church of
England, Other Christian, Muslim, Jewish, Hindu, Other, No religion. No write-in line and no box
for not stated, so no category here is an office's back-coding and the only non-answer is a
blank, which the report prints inside `Other/Not stated`.

Roman Catholic is the Latin-rite Diocese of Gibraltar, and Church of England is the Diocese in
Europe, whose cathedral is in Gibraltar; neither call is arguable and neither is in REVIEW.

EXCLUDED holds categories that are deliberately not on the tree (none).
REVIEW holds calls that are defensible but arguable, with the reason.
"""

EXCLUDED = {
    # Nothing. `sources/gi.py` writes only the eight answers, never a total row, and the table
    # has no non-response column of its own: blanks are inside `Other/Not stated` (REVIEW).
}

REVIEW = {
    "Other Christian":
        "-> christianity, the parent, and NOT christianity.other. **1,503 people, 3.96%.** The "
        "form names two churches and then offers one box for every other Christian, so this is "
        "a residual holding whole branches, not a set of bodies with no branch; "
        "`christianity.other`'s own note says it is the second and not the first. Belgium's "
        "`Other Christian denomination` is on the parent for the same reason (`be2024.py`). "
        "Barbados put its `Other Christian` on `christianity.other` because fifteen bodies were "
        "named above it (`bb2010.py`), which is not the case here. It is flat across the "
        "areas, from 2.6% of Eastside to 4.4% of North District.",

    "Other/Not stated":
        "-> other.gi, a per-source residual (§3.11), and NOT `unknown` or `gap`. **779 people, "
        "2.05%.** The report prints other religion and not stated as one column in 2022. Its "
        "1970-2022 series (Figure 8, p.52) keeps that merged label for every year, and the 2012 "
        "report printed the two apart: 365 other and 44 not stated, so in 2012 the pair was 89% "
        "other. The form has no not-stated box, so a not-stated answer is a blank.\n\n"
        "**Anita's ruling on ask 023 settles the shape**: Iran's merged `other and not stated` "
        "column stays on a coloured node rather than in `gap`, and a small religion with no "
        "answer of its own sitting inside it does not hold a build. It is a ruling for Iran; "
        "the case is the same shape, a merged column that cannot be split.\n\n"
        "**Where it leans**: 50 of the 779 are in Institutions (hospitals, the prison, care "
        "homes, hostels, religious houses and marinas, Appendix 8), 9.1% of their 552 residents "
        "against 1.95% of everyone else and the highest share of any row. The report says the "
        "institutions were enumerated with their managers providing the information (p.37); "
        "that this is where the blanks come from is this file's inference, not the report's. "
        "Elsewhere it is 2.7% of North District and 0.2% of Eastside.",
}

COLUMNS = {
    # Every category is measured at the territory, which is the unit drawn, so no row is
    # derived and nothing rolls up. Recorded per COMMANDS.txt's check_rollup note.
}

MAP = {
    "Roman Catholic":    "christianity.catholic.latin",
    "Church of England": "christianity.anglican",
    "Other Christian":   "christianity",
    "Muslim":            "islam",
    "Jewish":            "judaism",
    "Hindu":             "hinduism",
    "No Religion":       "unaffiliated",
    "Other/Not stated":  "other.gi",
}


def resolve(category):
    """Source category -> node, or None for a category deliberately not on the tree."""
    if category in EXCLUDED:
        return None
    return MAP.get(category)
