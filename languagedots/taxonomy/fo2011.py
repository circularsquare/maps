"""Faroe Islands, Census 2011, first language (Hagstova MT16) -> node.
sources/fo_census.py; keyed by the table's English labels as data/normalized/fo.csv carries them.

Ten answers. CALLS (sources/fo.md §3 says more):
  Faroese, Danish: their North Germanic leaves.
  "Other Nordic languages" (411): `other`. It is not only North Germanic: 56 of its speakers were
    born in Greenland, where the only Nordic first language besides Danish and Faroese (both
    separate rows) is Greenlandic (Eskimo-Aleut), and Finnish and Sami would also file here.
  "Other European languages" (607), "Asian languages" (290), "Middle East/North African
    languages" (40): `other`. Each names a part of the world and crosses families (bw2011,
    na2011 and at2001 file the same labels the same way).
  "Other African languages" (31): `africa_other`, as at2001's "sonstige afrikanische Sprachen".
  "South American languages" (1): `americas_other`. Spanish and Portuguese are filed as European
    here: 42 people born in South America have an "Other European" first language, and at most 2
    a "South American" one, so this row is an indigenous language of South America.
  "Sign language" (18): `signlanguage`.
  "No language" (41; 22 of them under 5): not drawn, in `gap`.
"""
GN = "indoeuropean.germanic.north"

NAMES = {
    "Faroese": f"{GN}.faroese",
    "Danish": f"{GN}.danish",
    "Other Nordic languages": "other",
    "Other European languages": "other",
    "Asian languages": "other",
    "Middle East/North African languages": "other",
    "Other African languages": "africa_other",
    "South American languages": "americas_other",
    "Sign language": "signlanguage",
}
SKIP = {"No language"}


def resolve(label):
    if label in SKIP:
        return None
    if label not in NAMES:
        raise KeyError(f"fo2011: unmapped label {label!r}")
    return NAMES[label]
