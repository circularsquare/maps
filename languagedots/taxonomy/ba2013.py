"""Bosnia and Herzegovina, Popis 2013, mother tongue (maternji jezik) -> node. sources/ba_census.py.

Keyed by the English half of BHAS's bilingual header in Knjiga 2 table 6.1, exactly as
data/normalized/ba.csv carries them. BHAS prints 16 named answers, "Other" and "Unknown" at every
level; the national tables of the same book (6, and 4 by ethnicity) split nothing further.

FOLDED 2026-10-08 (Anita): Croato-Serbian, the three compound answers and Bosnian-Herzegovinian
now all go on `serbocroatian`, reversing the per-label leaves described below. Each names the
common language rather than one standard, they were 0.6-2k people apiece, and drawn apart they
cluttered the legend; the BCMS group (regroup.txt) holds the standards and this together.

CALLS (sources/ba.md says more):
  Bosnian, Serbian, Croatian: three leaves, as printed (rs2022, hr2021). Glottolog files all
    three standards as dialects of Serbian-Croatian-Bosnian (sout1528). The labels follow
    ethnicity almost one to one (the census's own table 4 crosses them), and the map says so.
  Serbo-Croatian (27,299) and Croato-Serbian (1,195): the existing `serbocroatian` (cz, fi, us,
    hr) and `croatoserbian` (hr) leaves; the two word orders of the Yugoslav-era name, printed
    apart, kept apart as in Croatia.
  Bosnian-Croatian-Serbian (1,897), Bosnian-Serbian-Croatian (890), Bosnian-Croatian (714):
    three new leaves. Compound answers naming the standards together; the census prints each in a
    column of its own, so each keeps a node (brief §3: every printed label). The two three-part
    orders are not merged, for the same reason Serbo-Croatian and Croato-Serbian are not: the
    order is what people chose to write, and BHAS counted them apart.
  Bosniak (Bošnjački, 1,167): a new leaf `bosniak`, apart from Bosnian (Bosanski). It is the name
    for the language that Serb and Croat usage prefers over "Bosnian"; the census prints it apart.
    Thinly spread: no municipality has more than 0.3% (Kalinovik, 6 people).
  Bosnian-Herzegovinian (Bosanskohercegovački, 636): a new leaf `bosnianherzegovinian`; a
    civic, non-ethnic name for the common language, printed apart.
  Romani (5,766): `romani.romani`, variety not stated, as rs2022 and hr2021.
  Turkish, Albanian, Ukrainian (Prnjavor's Galician Ukrainians, settled under Austria-Hungary),
    German: as printed, on the existing leaves.
  "Other" (10,649): `other`. BHAS gives no breakdown at any level, and its other tables print
    Slovenian, Macedonian and Montenegrin only as ethnicities, never as languages, so those
    languages are inside this cell along with migrants' languages; nothing lets an indigenous
    remainder be told apart.
  NOT_STATED: "Unknown" (7,487, 0.21%): not drawn, the entry's gap.
"""
IE = "indoeuropean"
SL = f"{IE}.slavic.south"

NAMES = {
    "Bosnian": f"{SL}.bosnian",
    "Serbian": f"{SL}.serbian",
    "Croatian": f"{SL}.croatian",
    "Serbo-Croatian": f"{SL}.serbocroatian",
    "Croato-Serbian": f"{SL}.serbocroatian",
    "Bosnian-Croatian-Serbian": f"{SL}.serbocroatian",
    "Bosnian-Serbian-Croatian": f"{SL}.serbocroatian",
    "Bosnian-Croatian": f"{SL}.serbocroatian",
    "Bosniak": f"{SL}.bosniak",
    "Bosnian-Herzegovinian": f"{SL}.serbocroatian",
    "Romani": f"{IE}.indoaryan.romani.romani",
    "Albanian": f"{IE}.albanian.albanian",
    "Turkish": "turkic.turkish",
    "Ukrainian": f"{IE}.slavic.east.ukrainian",
    "German": f"{IE}.germanic.continental.german",
    "Other": "other",
}
NOT_STATED = {"Unknown", "Total"}


def resolve(label):
    if label in NOT_STATED:
        return None
    if label not in NAMES:
        raise KeyError(f"ba2013: unmapped label {label!r}")
    return NAMES[label]
