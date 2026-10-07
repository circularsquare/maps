"""Transnistria and Bender, 2015 census of the Transnistrian authorities (sources/md_pmr.py) -> node.

Keyed by data/normalized/md_pmr.csv's `source_category`, an English language name that
sources/md_pmr.py writes after spreading each nationality over languages in Moldova's 2024
nationality-by-mother-tongue shares (BNS table 5.33). The census was read as nationality: no
native-language table by raion was found.

CALLS.
  Moldovan: BNS's "Moldovenească sau Română" sum, all drawn as Moldovan. The Transnistrian
    authorities' official language is Moldovan (Cyrillic); the right bank's split between the two
    names is not carried across the river.
  Romani: the leaf `romani.romani`, as md2024.
  Other: bare `other` (the census's "other" nationalities, and BNS's minor languages).
"""
RO = "indoeuropean.romance"
SL = "indoeuropean.slavic"

NAMES = {
    "Moldovan": f"{RO}.moldovan",
    "Russian": f"{SL}.east.russian",
    "Ukrainian": f"{SL}.east.ukrainian",
    "Belarusian": f"{SL}.east.belarusian",
    "Bulgarian": f"{SL}.south.bulgarian",
    "Polish": f"{SL}.west.polish",
    "Gagauz": "turkic.gagauz",
    "German": "indoeuropean.germanic.continental.german",
    "Romani": "indoeuropean.indoaryan.romani.romani",
    "Other": "other",
}


def resolve(label):
    if label not in NAMES:
        raise KeyError(f"md2015_pmr: unmapped label {label!r}")
    return NAMES[label]
