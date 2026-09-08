"""CBS 2022 Census of Population and Housing (Israel) -> religiondots taxonomy.

Two axes from one dashboard, and they are not the same kind of thing.

**Religion**, from the population register, five categories nationwide:

    72.7%  Jews             -> judaism  (or one of its four observance children, below)
    17.8%  Muslims          -> islam
     6.1%  Others           -> unrecorded
     1.9%  Christians       -> christianity
     1.5%  Druze            -> druze

**Observance**, "main lifestyle in the household", applied to the Jewish count only where a
unit is at least 85% Jewish (`sources/il.py` decides that; see its docstring for why, and
for the coverage table behind the number):

    53.2%  Secular                      -> judaism.hiloni
    24.4%  Traditional                  -> judaism.masorti
    12.2%  Religious / Very religious   -> judaism.dati
     6.4%  Ultra-religious              -> judaism.haredi
     2.4%  Mixed                        -> judaism        (the parent; §6.15's MERGE_OWN)
     1.4%  Other                        -> judaism        (the parent)

**THE ONE CATEGORY THAT IS NOT A CATEGORY.** CBS's dashboard publishes the full five-way
breakdown for large units and, for everything else, the dominant group plus a lump called
`Other religions` — Nazareth returns `Muslims 73.1% / Other religions 26.9%` where that
26.9% is essentially all Christian, and Shefar'am's 37.1% is Christians and Druze together.
It is deliberately absent from MAP: `sources/il.py` must resolve it against its
sub-district's published totals BEFORE the normalised file is written, and leaving it
unmapped means `tools/check_mapping.py` fails loudly if that step is ever skipped. Mapping
it to `other.il` would have been the quiet wrong answer, and it would have erased most of
Israel's Christians.

EXCLUDED holds categories deliberately not on the tree.
REVIEW holds calls that are defensible but arguable, with the reason.
"""

EXCLUDED = {
    "Total":
        "the unit's own population total, not a category.",
}

REVIEW = {
    "Jews":
        "-> judaism. Carries every Jew in a unit that is under 85% Jewish, and every Jew in "
        "a `Mixed` or `Other` observance household anywhere. §6.15's MERGE_OWN case: the "
        "parent's own row is the 'observance not established' child of itself, and it is a "
        "real answer rather than a residual.",
    "Muslims":
        "-> islam, the parent, with no branch. Israel's Muslims are overwhelmingly Sunni "
        "and the register does not record the distinction, so nothing here permits a "
        "`islam.sunni` mapping — Russia is still the only source on this map that asks "
        "(branches.py, `islam.sunni`).",
    "Christians":
        "-> christianity, the parent. **The biggest loss in this file.** Israel's Christians "
        "are mostly Arab and mostly Greek Orthodox and Greek Catholic (Melkite), with Latin, "
        "Maronite, Armenian and Syriac communities beside them, plus a large non-Arab "
        "population of ex-Soviet and migrant Christians who are a different group again. "
        "CBS's own footnote to ST02-11x says the cell holds 'Arab Christians and non-Arab "
        "Christians' and stops there. One cell of ~182,000 for the oldest continuously "
        "resident Christian communities anywhere, and nothing in the source splits it.",
    "Druze":
        "-> druze, its own family and not a branch of Islam, which is how the node was "
        "already written for Canada. **~153,000 people, and this is the largest Druze count "
        "on the map by a wide margin.** Two separate populations that the register does not "
        "distinguish and the geography does: the Galilee and Carmel villages, who are "
        "Israeli citizens (Yirka 98.1%, Hurfeish 96.9%), and the Golan villages — Majdal "
        "Shams 99.9%, Buq'ata, Mas'ade, Ein Qiniyye — most of whom have declined Israeli "
        "citizenship and hold permanent-resident status. See sources/il_geo.py for why the "
        "Golan is drawn at all; it is an exception to the Green Line rule and a deliberate "
        "one.",
    "Others":
        "-> unrecorded, NOT `unaffiliated` and NOT `secular`. ~442,000 people whom the "
        "population register holds with no religious classification: overwhelmingly "
        "immigrants under the Law of Return who are not Jewish by halakha, largely from the "
        "former Soviet Union. Nobody asked them anything, so this is Germany's case exactly "
        "and `unrecorded` was written to cover it — 'any register-basis source with the "
        "same shape belongs here' (branches.py). Reading it as irreligion would invent a "
        "belief the instrument never enquired about.",
    "Jews [Secular]":
        "-> judaism.hiloni. **53.2% of the country's Jews and the single largest answer in "
        "Israel.** It is filed inside Judaism rather than in §6.3a's grey family because "
        "the register has no irreligion box at all: a secular Israeli Jew is registered as "
        "a Jew, and drawing them grey would assert something nobody measured. What the row "
        "carries is household observance, which is the only thing that was.",
    "Jews [Traditional]":
        "-> judaism.masorti, and see branches.py's warning on that node: Israeli *masorti* "
        "is traditional-but-not-strictly-observant and is NOT the Conservative movement, "
        "which everywhere else in the Jewish world is called Masorti and which is "
        "`judaism.conservative`. Mapping this row there would be §12's `animismus` trap.",
    "Jews [Ultra-religious]":
        "-> judaism.haredi. CBS's English for Haredi. The best-measured category on this "
        "axis and the one with the sharpest geography: Bene Beraq returns 83.8%.",
    "Jews [Religious / Very religious]":
        "-> judaism.dati. Two of CBS's answers collapsed into one published cell, so the "
        "map cannot separate 'religious' from 'very religious'. Largely though not only "
        "the Religious Zionist stream.",
    "Jews [Mixed]":
        "-> judaism, the parent. A household whose members answer differently. Kept on the "
        "parent rather than given a node: it is a property of the household and not a "
        "position anybody holds.",
    "Jews [Other]":
        "-> judaism, the parent. The observance question's own residual. **CBS spells this "
        "two ways in one field** — `Other` in some units and `Other main lifestyle` in "
        "others — so both are mapped; see the note in MAP.",
    "Jews [Other main lifestyle]":
        "-> judaism, the parent. The same residual as `Jews [Other]` under CBS's other "
        "spelling of it, and the larger of the two at 106,411 people.",
}

MAP = {
    # ---- religion, the partition
    "Jews": "judaism",
    "Muslims": "islam",
    "Christians": "christianity",
    "Druze": "druze",
    "Others": "unrecorded",

    # ---- the Jewish observance split, applied only in units >= 85% Jewish
    "Jews [Ultra-religious]": "judaism.haredi",
    "Jews [Religious / Very religious]": "judaism.dati",
    "Jews [Traditional]": "judaism.masorti",
    "Jews [Secular]": "judaism.hiloni",
    "Jews [Mixed]": "judaism",
    # CBS EMITS TWO SPELLINGS OF THE OBSERVANCE RESIDUAL and both are in the data: 24,090
    # people arrive as `Other` and 106,411 as `Other main lifestyle`, from the same field of
    # the same dashboard, differing by unit. Mapping only the short one dropped 106,411
    # people; `tools/check_mapping.py` is what said so, and nothing upstream of it noticed —
    # every reconciliation in sources/il.py passed, because the rows were present and
    # correct right up to the taxonomy.
    "Jews [Other]": "judaism",
    "Jews [Other main lifestyle]": "judaism",
}

# `Other religions` is deliberately NOT here -- see the module docstring. If it reaches the
# taxonomy, sources/il.py failed to resolve the lump and check_mapping.py should say so.


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
