"""Bulgaria, Census 2021, mother tongue (майчин език) -> node. sources/bg_census.py.

Keyed by NSI's Bulgarian column headers verbatim, footnote digit included, exactly as
data/normalized/bg.csv carries them. NSI prints three named languages and one "other" at every
level it publishes (obshtina, oblast, country); nothing finer exists in the 2021 release.

CALLS (sources/bg.md says more):
  "Български", "Турски": Bulgarian and Turkish, nodes already in the tree (cz.txt, us.txt).
  "Ромски" (Romani, 227,974): the leaf `romani.romani`, variety not stated (as Serbia's,
    Hungary's and Romania's).
  "Друг" (other, 62,906): `other`. NSI gives no breakdown at any level and the census's own
    "other" mixes long-settled minorities (Armenian, Aromanian, Vlach Romanian, Russian, Greek)
    with recent migrants (its largest count is Sofia's, 17,131), so there is no indigenous
    remainder that can be told apart from a foreign one.
  NOT_STATED: "Не мога да определя" (cannot determine, 10,633), "Не желая да отговоря" (do not
    wish to answer, 49,602) and "Непоказан1" (616,681, people added from administrative
    registers and never asked): not drawn, the entry's gap.
"""
IE = "indoeuropean"

NAMES = {
    "Български": f"{IE}.slavic.south.bulgarian",
    "Турски": "turkic.turkish",
    "Ромски": f"{IE}.indoaryan.romani.romani",
    "Друг": "other",
}
NOT_STATED = {"Общо", "Не мога да определя", "Не желая да отговоря", "Непоказан1"}


def resolve(label):
    if label in NOT_STATED:
        return None
    if label not in NAMES:
        raise KeyError(f"bg2021: unmapped label {label!r}")
    return NAMES[label]
