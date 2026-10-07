"""Liechtenstein, Volkszählung 2020, main language (Hauptsprache), eTab table 213.011d
(sources/li_census.py) -> node. Keyed by the office's German labels exactly as
data/normalized/li.csv carries them.

One language per person; the table has no "not stated" row and its 32 filled categories
partition all 39,055 residents. The categories are Switzerland's BFS list (taxonomy/ch2000.py),
and the calls follow ch2000's, so the two countries read the same across the Rhine:

  "Deutsch": German. It includes the Liechtenstein Alemannic dialects: the question asks for a
    language, and the dialect is not an answer of its own in this table. The tree's
    `swiss_german` node is not used, because the census does not name it.
  "Rätoromanisch" (3 people): `rhaetoromance`, as ch2000.
  "Serbisch und Kroatisch": Serbo-Croatian, one category (Bosnian and Montenegrin included).
  REGIONAL REMAINDERS whose members cross families go on `other`, as ch2000: "Übrige
    nordeuropäische" (Finnish is listed among the northern ones, so it can hold Estonian or
    Sami), "Westasiatische" (Kurdish, Persian, Armenian, Hebrew...), "Indoarische und
    drawidische" (two families), "Ostasiatische" (Chinese, Thai, Vietnamese, Tagalog...),
    "Übrige Sprachen".
  "Afrikanische Sprachen": `africa_other`, as ch2000 (Arabic is printed apart).
  "Übrige westeuropäische" and "Übrige osteuropäische Sprachen" are `..` in 2020 and are not in
    li.csv; they are listed so a later release that fills them maps without an edit.
"""
GE = "indoeuropean.germanic"
RO = "indoeuropean.romance"
SL = "indoeuropean.slavic"

NAMES = {
    "Deutsch": f"{GE}.continental.german",
    "Französisch": f"{RO}.french",
    "Italienisch": f"{RO}.italian",
    "Rätoromanisch": f"{RO}.rhaetoromance",
    "Englisch": f"{GE}.english",
    "Niederländisch": f"{GE}.continental.dutch",
    "Spanisch": f"{RO}.spanish",
    "Portugiesisch": f"{RO}.portuguese",
    "Dänisch": f"{GE}.north.danish",
    "Norwegisch": f"{GE}.north.norwegian",
    "Schwedisch": f"{GE}.north.swedish",
    "Finnisch": "uralic.finnish",
    "Übrige nordeuropäische Sprachen": "other",
    "Serbisch und Kroatisch": f"{SL}.south.serbocroatian",
    "Russisch": f"{SL}.east.russian",
    "Polnisch": f"{SL}.west.polish",
    "Tschechisch": f"{SL}.west.czech",
    "Slowakisch": f"{SL}.west.slovak",
    "Mazedonisch": f"{SL}.south.macedonian",
    "Slowenisch": f"{SL}.south.slovenian",
    "Bulgarisch": f"{SL}.south.bulgarian",
    "Albanisch": "indoeuropean.albanian.albanian",
    "Türkisch": "turkic.turkish",
    "Ungarisch": "uralic.hungarian",
    "Rumänisch": f"{RO}.romanian",
    "Griechisch": "indoeuropean.hellenic.greek",
    "Afrikanische Sprachen": "africa_other",
    "Arabisch": "afroasiatic.arabic",
    "Westasiatische Sprachen": "other",
    "Indoarische und drawidische Sprachen": "other",
    "Ostasiatische Sprachen": "other",
    "Übrige Sprachen": "other",
    "Übrige westeuropäische Sprachen": "other",
    "Übrige osteuropäische Sprachen": "other",
}
NOT_STATED = {"Hauptsprache - Total"}


def resolve(label):
    if label in NOT_STATED:
        return None
    if label not in NAMES:
        raise KeyError(f"li2020: unmapped label {label!r}")
    return NAMES[label]
