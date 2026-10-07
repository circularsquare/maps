"""Malta, Census of Population and Housing 2021, main language spoken from early childhood -> node.
sources/mt_census.py; keyed by the census's own column heads (Vol. 3, Tables 3.3 and 3.6), as
data/normalized/mt.csv carries them.

Seven answers. CALLS (sources/mt.md says more):
  Maltese, English, Italian, German, French, Arabic: one leaf each, the nodes every other
    country uses. Maltese is `afroasiatic.maltese` (Glottolog malt1254, Afroasiatic), as au2021,
    ca2021, fi2025, pl2021 and uk2021 already file it.
  "Other" (57,818, 11.6%; 55,541 of them non-Maltese citizens): `other`. It holds every language
    but the six, and Volume 1 Table 2.4 says what it mostly is (Serbian, Bulgarian, Indian,
    Filipino, Nepalese and Albanian citizens are 30,130 of the non-Maltese), but nothing names
    the languages, and no Maltese indigenous language other than Maltese exists to be kept apart.
  No not-stated column: every row sums to its total.
"""
NAMES = {
    "Maltese": "afroasiatic.maltese",
    "English": "indoeuropean.germanic.english",
    "Italian": "indoeuropean.romance.italian",
    "German": "indoeuropean.germanic.continental.german",
    "French": "indoeuropean.romance.french",
    "Arabic": "afroasiatic.arabic",
    "Other": "other",
}
SKIP = {"Total", "Non-Maltese population, all ages"}


def resolve(label):
    if label in SKIP:
        return None
    if label not in NAMES:
        raise KeyError(f"mt2021: unmapped label {label!r}")
    return NAMES[label]
