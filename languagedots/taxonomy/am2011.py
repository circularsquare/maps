"""Armenia 2011 census, mother tongue by marz (sources/am_census.py) -> node.

Keyed by data/normalized/am.csv's `source_category`, Armstat's English heads for table 5.2-1.

CALLS.
  Yezidian: Armstat's English spelling of `Եզդիերեն`, the Yezidi language. Its own node, a sibling
    of Kurdish (tree.d/am.txt says why): the census prints the two apart and every label it prints
    gets a node. Linguists class both as Northern Kurdish (Kurmanji); the map draws the answer.
  Assyrian: `Ասորերեն`. Armenia's Assyrians speak Assyrian Neo-Aramaic (the Urmia dialects their
    families brought from Persia in the 1820s), so the existing `afroasiatic.assyrian` leaf.
  Greek: `Հունարեն`. Some of Armenia's Greeks descend from Pontic Greek miners settled in Lori in
    the 18th century; the census prints one Greek answer and it is drawn on Greek.
  Other: bare `other`. Each marz prints only the languages it has people for, so a marz's Other
    holds both the languages no table names and the named ones that marz gives no column (1,270 of
    the 2,183 people the marz tables print as Other, by the national table). Nothing narrower holds
    both.
  Refused to answer: not drawn; ENTRY's `gap`. 29 people.
"""
NAMES = {
    "Armenian": "indoeuropean.armenian.armenian",
    "Yezidian": "indoeuropean.iranian.yezidi",
    "Russian": "indoeuropean.slavic.east.russian",
    "Assyrian": "afroasiatic.assyrian",
    "Kurdish": "indoeuropean.iranian.kurdish",
    "Ukrainian": "indoeuropean.slavic.east.ukrainian",
    "English": "indoeuropean.germanic.english",
    "Georgian": "kartvelian.georgian",
    "Persian": "indoeuropean.iranian.persian",
    "Greek": "indoeuropean.hellenic.greek",
    "Other": "other",
}
SKIP = {"Total", "Refused to answer"}


def resolve(label):
    if label in SKIP:
        return None
    if label not in NAMES:
        raise KeyError(f"am2011: unmapped label {label!r}")
    return NAMES[label]
