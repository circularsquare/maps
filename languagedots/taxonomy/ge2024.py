"""Georgia 2024 census, native language by self-governed unit (sources/ge_census.py) -> node.

Keyed by data/normalized/ge.csv's `source_category`, the table's own column heads.

CALLS.
  Georgian: `kartvelian.georgian`. The census has one answer for the Kartvelian languages, so
    Mingrelian, Svan and Laz speakers are inside it wherever they answered Georgian (Samegrelo and
    Svaneti); nothing published separates them, and they are not split out by place.
  Azerbaijanian: the census's spelling of Azerbaijani.
  Other: bare `other`. Every language the form did not list: Kurmanji (11,324 Yazidis), Chechen
    (the Kists of Pankisi, Akhmeta), Avar (Kvareli), Greek, Ukrainian, Assyrian, and the languages
    of the 2024 census's new foreign residents (23,996 Indians and 12,533 Arabs, mostly in
    Tbilisi). Georgia's own minority languages and immigrants' languages share the one answer, so
    it cannot sit on any narrower node. 73,373 people, 1.9%.
  Not stated: not drawn; ENTRY's `gap`. 48,580, 1.2%.
"""
NAMES = {
    "Georgian": "kartvelian.georgian",
    "Abkhaz": "abkhazadyghe.abkhaz",
    "Ossetian": "indoeuropean.iranian.ossetian",
    "Azerbaijanian": "turkic.azerbaijani",
    "Russian": "indoeuropean.slavic.east.russian",
    "Armenian": "indoeuropean.armenian.armenian",
    "Other": "other",
}
SKIP = {"Total", "Not stated"}


def resolve(label):
    if label in SKIP:
        return None
    if label not in NAMES:
        raise KeyError(f"ge2024: unmapped label {label!r}")
    return NAMES[label]
