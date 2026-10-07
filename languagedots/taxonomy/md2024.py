"""Moldova, RPL 2024, mother tongue (limba maternă), table 5.15 (sources/md_census.py) -> node.

Keyed by BNS's column headers as data/normalized/md.csv carries them (ş/ţ normalised to the
cedilla forms by sources/md_census.py). Every node already exists (tree.d/ua.txt, cz.txt, us.txt).

CALLS (sources/md.md says more):
  "Moldovenească" (1,159,857) and "Română" (765,838): two answers, two nodes, as in Ukraine,
    Czechia, Finland and Russia here. Glottolog has one language (Moldavian, mold1248, is a
    dialect of Romanian), and BNS itself prints a "Moldovenească sau Română" sum column; but the
    census asked and printed them apart, and the split is the point of the question in Moldova.
    The sum column is not read.
  "Romani (Ţigănească)" (7,640): the leaf `romani.romani`, variety not stated (as Ukraine's).
  "Altă limbă" (6,116): `other`. By UAT nothing is named; sheet 5.37 (ages 3+, national) shows it
    is a mix of Turkish, Arabic, English, Belarusian, Armenian, Azerbaijani, Italian, German and
    others, so no narrower node holds it all.
  NOT_STATED: "Nu au declarat limba maternă" (1,582, 0.07%): not drawn, the entry's gap.
"""
RO = "indoeuropean.romance"
SL = "indoeuropean.slavic"

NAMES = {
    "Moldovenească": f"{RO}.moldovan",
    "Română": f"{RO}.romanian",
    "Ucraineană": f"{SL}.east.ukrainian",
    "Rusă": f"{SL}.east.russian",
    "Găgăuză": "turkic.gagauz",
    "Bulgară": f"{SL}.south.bulgarian",
    "Romani (Ţigănească)": "indoeuropean.indoaryan.romani.romani",
    "Altă limbă": "other",
}
NOT_STATED = {"Nu au declarat limba maternă"}


def resolve(label):
    if label in NOT_STATED:
        return None
    if label not in NAMES:
        raise KeyError(f"md2024: unmapped label {label!r}")
    return NAMES[label]
