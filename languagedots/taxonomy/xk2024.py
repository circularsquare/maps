"""Kosovo, Census 2024, mother tongue (gjuha amtare) -> node. sources/xk_census.py.

Keyed by the English labels of ASK's PxWeb table census2024_22, exactly as data/normalized/xk.csv
carries them. Five named languages, "Other (specify)" and "Not available" (zero everywhere).
ASK defines mother tongue as "the first language a person has learned from birth and is still
able to speak"; one answer.

CALLS (sources/xk.md says more):
  Serb (55,168): `serbian`. The label is the PxWeb English edition's; the national table
    census2024_56 and the results PDF (Tab. 3.7) print "Serbian". It includes most of Dragash's
    Gorani-area answers that were not Bosnian or other (1,676 there against 17 Serbs).
  Bosnian (28,854): the existing `bosnian` leaf, as printed.
  Romani (5,597): `romani.romani`, variety not stated, as mk2021 and rs2022.
  Albanian, Turkish: as printed.
  "Other (specify)" (8,978): `other`. ASK prints no breakdown. Two thirds (5,980) is in Dragash,
    where 7,828 people declared Gorani ethnicity: it is mostly the Gorani's own Slavic speech
    (nashinski), but nothing published says how much, so it stays on the bare remainder rather
    than being guessed into a node. No indigenous remainder to keep apart.
  NOT_STATED: "Not available" (0 everywhere in 2024) and "Total", the universe row.
"""
IE = "indoeuropean"
SL = f"{IE}.slavic.south"

NAMES = {
    "Albanian": f"{IE}.albanian.albanian",
    "Serb": f"{SL}.serbian",
    "Bosnian": f"{SL}.bosnian",
    "Turkish": "turkic.turkish",
    "Romani": f"{IE}.indoaryan.romani.romani",
    "Other (specify)": "other",
}
NOT_STATED = {"Not available", "Total"}


def resolve(label):
    if label in NOT_STATED:
        return None
    if label not in NAMES:
        raise KeyError(f"xk2024: unmapped label {label!r}")
    return NAMES[label]
