"""Palau, 2015 Census of Population, Housing and Agriculture, Table 16 -> node.

Rows of Table 16 (sources/pw_census.py). Each person is either "Palauan only" at home or filed
under ONE other language spoken at home (C23). countries/pw.py moves the people who use that
other language less often than Palauan (C24) back onto Palauan; see sources/pw.md.

Calls:
  - "Carolinian" (12) on Sonsorolese and Tobian: in Palau's census the word means the
    southwest islanders (tree.d/pw.txt), not Saipan's Carolinian.
  - "Other micronesian" (357, 329 of them in Koror) on `pacific_other`: Palau's "Micronesian"
    is the region, so it can hold Chamorro (not Oceanic) as well as Chuukese, Yapese,
    Pohnpeian and Marshallese. The narrowest node holding all of them is the regional
    remainder, kept off `other`.
  - "Philippine languages" (2,069) on the Philippine group: the table names no language
    (Tagalog, Ilocano, Cebuano and others), so it draws as "language not named".
  - "Chinese languages" (247) on the Sinitic group; "Taiwanese" (52), printed as its own row
    beside it, on Min Nan, the language the word names.
  - "Other language" (402) on `other`: it spans families.
"""

NAMES = {
    "Yes, Palauan only": "austronesian.palauan",
    "English": "indoeuropean.germanic.english",
    "Carolinian": "austronesian.oceanic.sonsorol_tobi",
    "Other micronesian": "pacific_other",
    "Philippine languages": "austronesian.philippine",
    "Japanese": "japonic.japanese",
    "Korean": "koreanic.korean",
    "Chinese languages": "sinotibetan.sinitic",
    "Taiwanese": "sinotibetan.sinitic.min_nan",
    "Other language": "other",
}


def resolve(label):
    return NAMES.get(label)
