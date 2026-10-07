"""Aruba, Census 2010, Table P-D.2, language most spoken in the household -> node.
sources/aw_census.py; keyed by the table's English column heads, as data/normalized/aw.csv carries
them.

CALLS (sources/aw.md says more):
  Papiamento: `creole.portuguese_based.papiamento`, the node pl2021 and cw2023 use.
  Spanish, Dutch, English: the nodes every other country uses.
  Chinese (1,456): `sinotibetan.sinitic`, as cw, cy, kg, kh: "Chinese" names no variety, and
    build.py draws that node unwashed as "Chinese".
  Others (1,725): `other`. The census printed no breakdown; the island has no indigenous
    language, so nothing here belongs on an indigenous remainder.
  Does not speak (yet) (1,563, nine in ten of them under five) and Not reported (432): not drawn;
    countries/aw.py counts both in `gap`.
"""
NAMES = {
    "Papiamento": "creole.portuguese_based.papiamento",
    "Spanish": "indoeuropean.romance.spanish",
    "Dutch": "indoeuropean.germanic.continental.dutch",
    "English": "indoeuropean.germanic.english",
    "Chinese": "sinotibetan.sinitic",
    "Others": "other",
}
SKIP = {"Does not speak (yet)", "Not reported"}


def resolve(label):
    if label in SKIP:
        return None
    if label in NAMES:
        return NAMES[label]
    raise KeyError(f"aw2010: unmapped label {label!r}")
