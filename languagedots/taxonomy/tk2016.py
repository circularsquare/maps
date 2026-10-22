"""Tokelau, 2016 census Table 5.5, language spoken most often at home (sources/tk_census.py,
sources/tk.md). One answer per person aged 5 and over present on census night. "Other language"
(16) is the unnamed remainder, on `other`. "Kiribati" is the printed label for Gilbertese."""
NAMES = {
    "Tokelauan": "austronesian.oceanic.tokelauan",
    "Samoan": "austronesian.oceanic.samoan",
    "English": "indoeuropean.germanic.english",
    "Tuvaluan": "austronesian.oceanic.tuvaluan",
    "Kiribati": "austronesian.oceanic.gilbertese",
    "Other language": "other",
}


def resolve(label):
    return NAMES[label]
