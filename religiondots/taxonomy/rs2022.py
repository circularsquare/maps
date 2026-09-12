"""RZS Popis 2022 religion classification -> religiondots taxonomy.

Eleven drawn categories plus the universe total, a duplicate parent and two residuals that
are not answers, at municipality. Croatia's file one country over, and it maps almost the
same way — which is the argument for keeping the calls identical where the categories are
identical, rather than re-deriving them and drifting.

    81.05%  Christian - Orthodox           -> christianity.orthodox.canonical
     5.35%  Unknown                        -> EXCLUDED
     4.19%  Islam                          -> islam
     3.87%  Christian - Catholic           -> christianity.catholic
     2.55%  Did not declare                -> EXCLUDED
     1.12%  Not believers (atheists)       -> unaffiliated
     0.89%  Christian - Other Christian    -> christianity.other
     0.82%  Christian - Protestant         -> christianity.protestant
     0.13%  Agnostics                      -> secular
     0.02%  Eastern religions              -> other.rs
     0.01%  Judaism                        -> judaism
     0.01%  Otherreligions                 -> other.rs

**THE TWO EXCLUSIONS ARE 7.9% BETWEEN THEM AND THEY ARE DIFFERENT THINGS**, which is why
RZS keeps them apart and this file does too. See EXCLUDED — the geography of each says
what it is, and neither is irreligion.

EXCLUDED holds categories that are deliberately not on the tree.
REVIEW holds calls that are defensible but arguable, with the reason.
"""

EXCLUDED = {
    "Total":
        "the unit's own population total, not a category.",
    "Christian - All":
        "RZS publishes `Хришћанска / свега` beside its four children and it equals their "
        "sum in all 204 rows of the sheet, checked per row. Ghana's case and not "
        "Hungary's (spec §12): there is no unpublished remainder, so this is a duplicate "
        "to drop. Drawing it would double 5.76 million people.",
    "Did not declare":
        "169,486 people, 2.55%. Article 47 of the Serbian constitution says nobody is "
        "obliged to declare a religion and the census question is voluntary, so this is a "
        "refusal and not an answer (spec §3.5). Its geography says the same: it peaks in "
        "the multi-ethnic municipalities of Vojvodina — Dimitrovgrad 10.2%, Subotica "
        "10.1%, Sombor 9.2%, Bački Petrovac 7.2%, Kovačica 6.5% — which is where "
        "declaring anything has the most consequences and where the ethnicity question "
        "gets the same treatment.",
    "Unknown":
        "355,484 people, 5.35%, and the largest thing this map does not draw in Serbia. "
        "RZS distinguishes it from a refusal, and the two behave differently across the "
        "168 municipalities: `Did not declare` is a Vojvodina minority pattern, while "
        "`Unknown` is a CENTRAL BELGRADE pattern — Savski venac 17.3%, Stari grad 14.4%, "
        "Vračar 11.3%, against 0.97% in Preševo and under 1.7% across rural central "
        "Serbia. It correlates +0.60 with the declared-atheist share and only +0.28 with "
        "the refusal share, so it is missingness concentrated among the young, urban and "
        "secular rather than a group hiding inside a residual. **That is also the reason "
        "excluding it is not neutral**: dropping it removes proportionally more people "
        "from the least religious parts of the country than from the most, so every share "
        "drawn here is very slightly more religious than the country is. note_public "
        "says so.",
}

REVIEW = {
    "Christian - Orthodox":
        "-> christianity.orthodox.canonical, which is hr2021.py's call for Croatia's bare "
        "`Pravoslavci` and NOT mk2021.py's for North Macedonia's bare `Orthodox`. The two "
        "differ on whether canonicity was live at the census date, and in Serbia it was "
        "not: the Serbian Orthodox Church is the overwhelming majority of these 5.39 "
        "million people and its canonical standing has never been in question. The "
        "category still holds the Romanian Orthodox of Banat and the Macedonian and "
        "Montenegrin communities, all small and all now canonical — the Macedonian "
        "Orthodox Church was recognised in May 2022, five months BEFORE this census, "
        "which is the same fact that pushed North Macedonia's own 2021 figures to the "
        "parent node and pulls Serbia's to the child.",
    "Christian - Other Christian":
        "-> christianity.other, following hr2021.py's `Ostali kršćani`, gh2021.py's "
        "`Other Christian` and ee2021.py's `Christian (other)`. 59,346 people. branches.py "
        "warns that this node is for bodies with no branch rather than for a computed "
        "residual, and this is a published residual rather than a computed one — the "
        "distinction the note draws is against `…, other or unspecified` rows this "
        "project generates itself, and a source's own catch-all cell is what every other "
        "country puts here.",
    "Christian - Protestant":
        "-> christianity.protestant, the 'named an answer, not a body' node. 54,678 "
        "people, and in Serbia this is a much more specific population than the label "
        "suggests: it is overwhelmingly the SLOVAK LUTHERANS of Vojvodina, who are 57.3% "
        "of Bački Petrovac and 41.4% of Kovačica, plus the Reformed Hungarians of "
        "Bačka. `christianity.lutheran` and `christianity.reformed` both exist and Serbia "
        "is filed at neither, because the census offers one cell and says nothing about "
        "which. The concentration is visible on the map without the tree asserting it.",
    "Agnostics":
        "-> secular, and `Not believers (atheists)` -> unaffiliated, which is exactly "
        "hr2021.py's split for Croatia's identical pair. branches.py draws the line at "
        "whether a POSITION is stated: `Агностици` is one, and `Нису верници (атеисти)` "
        "leads with 'are not believers' and glosses it as atheism, so it is a report of "
        "no religion first. 8,654 and 74,139 people.",
    "Eastern religions":
        "-> other.rs, together with `Otherreligions`. `Источњачке вероисповести` is one "
        "cell for Buddhists, Hindus and everything else east of Islam — 1,207 people, "
        "0.018% — and nothing separates them at any geography. hr2021.py files Croatia's "
        "`Istočne religije` this way and ru2012.py files Arena's Eastern-religions answer "
        "the same, for the same reason: a bucket is not evidence that its contents belong "
        "together, and `buddhism` would assert something the source does not say.",
    "Otherreligions":
        "-> other.rs. 500 people, and RZS's own header cell really is written unspaced. "
        "The key here must match what sources/rs.py writes rather than a tidied version "
        "(spec §2.4).",
}

MAP = {
    "Christian - Orthodox": "christianity.orthodox.canonical",
    "Christian - Catholic": "christianity.catholic",
    "Christian - Protestant": "christianity.protestant",
    "Christian - Other Christian": "christianity.other",
    "Islam": "islam",
    "Judaism": "judaism",
    "Eastern religions": "other.rs",
    "Otherreligions": "other.rs",
    "Agnostics": "secular",
    "Not believers (atheists)": "unaffiliated",
}


def _key(cat):
    return " ".join(str(cat).split())


def resolve(cat):
    """Source category -> taxonomy node id, or None if deliberately not on the tree."""
    c = _key(cat)
    if c in EXCLUDED:
        return None
    return MAP.get(c)
