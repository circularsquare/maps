"""Abkhazia 2011 and South Ossetia 2015 (sources/ge_breakaway.py) -> node.

Keyed by data/normalized/ge_breakaway.csv's `source_category`: an English language name that
sources/ge_breakaway.py writes. In South Ossetia it is the census's own native-language column
(tables 4.2.1-4.2.5, measured). In Abkhazia it is a nationality read as a language through a
retention share (the census asked nationality only); the docstring there has every share.

CALLS.
  Mingrelian and Svan: their own nodes, for the 3,207 Abkhazian census respondents who gave
    Mingrelian and the 43 who gave Svan as their nationality. The 43,248 who gave Georgian are on
    Georgian, though most of them, in Gal, speak Mingrelian at home: that matches how Georgia's
    2024 native-language census draws Samegrelo across the Enguri (sources/ge_breakaway.py).
  Romani: the leaf `romani.romani`, variety not stated (as Moldova and Ukraine).
  Other: bare `other`, Abkhazia's "other nationalities" and South Ossetia's "other languages".
"""
NAMES = {
    "Abkhaz": "abkhazadyghe.abkhaz",
    "Georgian": "kartvelian.georgian",
    "Mingrelian": "kartvelian.mingrelian",
    "Svan": "kartvelian.svan",
    "Armenian": "indoeuropean.armenian.armenian",
    "Russian": "indoeuropean.slavic.east.russian",
    "Ukrainian": "indoeuropean.slavic.east.ukrainian",
    "Belarusian": "indoeuropean.slavic.east.belarusian",
    "Greek": "indoeuropean.hellenic.greek",
    "Ossetian": "indoeuropean.iranian.ossetian",
    "Turkish": "turkic.turkish",
    "Tatar": "turkic.tatar",
    "Azerbaijani": "turkic.azerbaijani",
    "Estonian": "uralic.estonian",
    "Romani": "indoeuropean.indoaryan.romani.romani",
    "Other": "other",
}


def resolve(label):
    if label not in NAMES:
        raise KeyError(f"ge2015_breakaway: unmapped label {label!r}")
    return NAMES[label]
