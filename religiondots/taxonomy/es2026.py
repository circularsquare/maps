"""CIS barómetro `RELIGION` -> religiondots taxonomy. Spain's Spanish-citizen half.

Six offered answers to one question, asked in every monthly barómetro since the 1970s, at
province. The codes are stable across waves, the labels drift, and — the thing that
matters — CIS occasionally reuses the variable NAME for a different question entirely.
`accepts()` at the bottom of this file is what stops that mixing, and the comment above it
is the most important paragraph here.

    1  Católico/a practicante      -> christianity.catholic.latin
    2  Católico/a no practicante   -> christianity.catholic.latin
    3  Creyente de otra religión   -> other.es, and see below: most of it is Islam
    4  Agnóstico/a                 -> secular
    5  Indiferente, no creyente    -> unaffiliated
    6  Ateo/a                      -> secular
    9  N.C.                        -> excluded, spec §3.5

THE UNIVERSE IS THE WHOLE POINT AND IT IS EASY TO MISS. `NACIONALIDAD` in a CIS barómetro
takes exactly two values — "la nacionalidad española" and "la nacionalidad española y otra" —
so **the barómetro does not sample foreign nationals at all.** Spain has 7.4 million of them,
15% of the population, and they are not undersampled here, they are outside the frame. Every
share this file maps is a share of Spanish citizens; sources/es.py covers the other 15%
separately and the two partition the country exactly. Reading CIS's 3.2% "other religion" as
Spain's non-Catholic minority is the single mistake this source invites, and it is what makes
the CIS figure look irreconcilable with the confessional counts. It is not: 3.2% of 42.4M
citizens is ~1.36M, and UCIDE's estimate of Spanish-CITIZEN Muslims is 1.09M, which fits
inside it.

THE PRACTISING SPLIT IS THROWN AWAY, DELIBERATELY. Codes 1 and 2 both go to
`christianity.catholic.latin`. This is a map of which religion, not of how much of it — no
other source here separates practice, and a `catholic.practising` node would be a category
only Spain could ever fill (§6.6). The distinction is not lost, it is reported: nationally
17.4% practising against 37.4% non-practising in June 2026, and the ratio has a real
geography that belongs in `note_public`.

`christianity.catholic.LATIN` rather than the parent, because Spain's Catholics are Latin
rite to within rounding — the Spanish Eastern Catholic communities are a few thousand
Ukrainian and Romanian Greek Catholics, who arrive through sources/es.py's foreign half and
are counted there.
"""

CANONICAL = {
    1: "Católico/a practicante",
    2: "Católico/a no practicante",
    3: "Creyente de otra religión",
    4: "Agnóstico/a",
    5: "Indiferente, no creyente",
    6: "Ateo/a",
}

EXCLUDED = {
    "N.C.": "N.C. — 1.4% of respondents, about 590,000 Spanish citizens at the current "
       "population. A refusal, not an answer, so spec §3.5 marks it rather than filling it "
       "and Spain is drawn at ~98.6% of its citizen population. Not `unaffiliated` (CIS "
       "asks that separately, as code 5) and not `unrecorded` (that is for a register that "
       "never asked; this one asked).",
}

REVIEW = {
    "Creyente de otra religión": "-> other.es. **The most consequential call in this file, "
       "and the least satisfying.** One cell for every religion that is not Catholicism, "
       "3.2% of Spanish citizens, and CIS never asks which — there is no follow-up item in "
       "any barómetro. `other.es` is spec §3.11's per-source residual and the same call "
       "ru2012.py makes for Arena's Eastern-religions cell. **But it is not left whole**: "
       "sources/es.py splits UCIDE's province-level count of Spanish-citizen Muslims out of "
       "it first, because otherwise Ceuta and Melilla — majority Muslim and majority "
       "Spanish-citizen — would draw grey, which is a false statement about the two most "
       "Muslim places in Spain rather than an honest silence. What stays in `other.es` after "
       "that split is Spain's evangelical Protestants, its Orthodox citizens, roughly "
       "110,000 Jehovah's Witnesses, ~45,000 Jews and the Buddhist tail.",
    "Agnóstico/a": "-> secular, with Ateo/a. The house convention, set by ca2021.py and "
       "followed by cz2021.py, ro2021.py, mx2020.py, hr2021.py, ie2022.py, au2021.py and "
       "nz2023.py: atheism and agnosticism are stated positions and both live on `secular`. "
       "Spain is unusual only in the size — 13.0% agnostic and 15.1% atheist in June 2026, "
       "28.1% between them, which is a larger `secular` share than any other country drawn "
       "here except Czechia.",
    "Indiferente, no creyente": "-> unaffiliated, NOT `secular`, and this is the one place "
       "Spain departs from the paragraph above. branches.py draws the line at whether a "
       "POSITION is stated: `secular` is a stated non-theistic stance, `unaffiliated` is a "
       "report of not having a religion. 'Indiferente' is indifference — the absence of the "
       "stance, not a version of it — and CIS offers it alongside atheist and agnostic "
       "rather than instead of them, so the respondent who picks it has declined both of "
       "those. Merging all three would throw away a distinction the source paid three boxes "
       "for, which is ro2021.py's argument used the other way round. 12.0% of citizens.",
    "Católico/a practicante": "With `Católico/a no practicante`, -> christianity.catholic.latin. See the docstring: this is a map of which "
            "religion. The practising share is carried in note_public instead.",
}

MAP = {
    "Católico/a practicante": "christianity.catholic.latin",
    "Católico/a no practicante": "christianity.catholic.latin",
    "Creyente de otra religión": "other.es",
    "Agnóstico/a": "secular",
    "Indiferente, no creyente": "unaffiliated",
    "Ateo/a": "secular",
    # sources/es.py's UCIDE split of code 3 — spec §3.1's permitted SPLIT of a survey
    # category by an outside source. Written as its own category so the operation is
    # visible in es.csv and checkable by tools/check_mapping.py.
    "Creyente de otra religión — Muslim (UCIDE split)": "islam.sunni",
}

# ---------------------------------------------------------------------------------------
# THE CHECK THAT SAVED THIS COUNTRY, and the reason it is a whole-signature match rather
# than a per-code one.
#
# CIS reuses the variable name `RELIGION` for TWO DIFFERENT QUESTIONS. 122 of the 170
# studies downloaded ask the six-way one this file maps. Two of them — estudios 3462 and
# 3506 — ask a three-way one instead:
#
#       1 'Sin religión/no profesa ninguna religión'
#       2 'Católico/a'
#       3 'Creyente de otra religión'
#
# Same variable name, same low code numbers, completely different meanings: code 1 is
# PRACTISING CATHOLIC in one and NO RELIGION in the other. Pooling them would have moved
# several hundred thousand irreligious Spaniards into the Catholic column and nothing would
# have looked wrong — the totals would still have summed, the provinces would still have
# been complete, and the national share would have drifted by about a point.
#
# So the accept test is the WHOLE signature: codes 1-6 must all be present and every label
# must be one this file recognises. A study whose RELIGION means something else fails on
# code 1 and is dropped with a reason. LABELS carries the wording variants actually seen
# across the pool, which are cosmetic — CIS drops and restores the parenthetical glosses on
# codes 4 and 6, and shortened code 3 for exactly one study.
# ---------------------------------------------------------------------------------------
LABELS = {
    1: {"Católico/a practicante"},
    2: {"Católico/a no practicante"},
    3: {"Creyente de otra religión", "De otra religión"},
    4: {"Agnóstico/a",
        "Agnóstico/a (no niegan la existencia de Dios pero tampoco la descartan)"},
    5: {"Indiferente, no creyente"},
    6: {"Ateo/a", "Ateo/a (niegan la existencia de Dios)"},
}

# N.C. is code 9 in 121 studies and code 7 in one (estudio 3434). Either is excluded.
NC_CODES = {7, 9}


def accepts(labels):
    """`labels` is {code: label} read from a study's SPSS syntax.

    Returns None if the study asks this question, or a one-line reason if it does not.
    """
    for code, ok in LABELS.items():
        if code not in labels:
            return f"code {code} missing"
        if labels[code] not in ok:
            return f"code {code} is {labels[code]!r}, not this question"
    return None


def _key(cat):
    return " ".join(str(cat).split())


def resolve(cat):
    """Source category -> taxonomy node id, or None if deliberately not on the tree.

    sources/es.py writes the CANONICAL label for each code, so the wording variants CIS
    drifts between never reach this function — `accepts()` above is what checks them.
    """
    c = _key(cat)
    if c in EXCLUDED:
        return None
    return MAP.get(c)
