"""Lithuania, Gyventojų surašymas 2021, mother tongue (gimtoji kalba) -> node.
sources/lt_census.py; keyed by Statistics Lithuania's own labels, exactly as data/normalized/lt.csv
carries them (the SDMX cube's value names; "Dvi  gimtosios kalbos" has two spaces in the source).

Ten categories at municipality level. CALLS (sources/lt.md says more):
  Lithuanian, Russian, Polish, Belarusian, Ukrainian, Latvian, German: one leaf each, the nodes
    every other country uses.
  "Romų" (Romani, 1,973; Vilnius city, Kaunas, Panevėžys, Šiauliai): `romani.romani`, variety
    not stated, as cz2021 and hr2021. Lithuania's Roma mostly speak Baltic Romani (Glottolog
    balt1257, under Northeastern Romani), but the census answer is just "Romani".
  "Dvi gimtosios kalbos" (two mother tongues, 49,066, 1.75%): the census let a person give two
    and publishes only that they did, never which two. Not a language and not a remainder of any
    one group, so it is a leaf of its own under `other`, as az2019 puts the census's "Jewish" and
    et2007 its unclassifiable names: drawn grey, labelled for what it is. The national cube
    (S3R778_GBS010302_1) shows who they are by ethnicity: 17,822 Lithuanians, 12,417 Poles, 8,905
    Russians, 3,219 Belarusians, 1,728 Ukrainians, 894 others and 4,081 of unstated ethnicity,
    so the pairs are almost all among Lithuanian, Polish, Russian and Belarusian, but nothing
    publishes them, and sharing each person across a guessed pair would change the counts.
  "Kitos" (other, 14,091): `other`. It holds every language not listed (Tatar, Karaim, Armenian,
    Yiddish, Hebrew and migrants' languages alike); the cube does not let an indigenous remainder
    be told apart from a foreign one.
  Not stated: the cube has no such column, and nationally it is 0 (sources/lt_census.py).
"""
IE = "indoeuropean"
SL = f"{IE}.slavic"

NAMES = {
    "Lietuvių": f"{IE}.baltic.lithuanian",
    "Latvių": f"{IE}.baltic.latvian",
    "Rusų": f"{SL}.east.russian",
    "Baltarusių": f"{SL}.east.belarusian",
    "Ukrainiečių": f"{SL}.east.ukrainian",
    "Lenkų": f"{SL}.west.polish",
    "Vokiečių": f"{IE}.germanic.continental.german",
    "Romų": f"{IE}.indoaryan.romani.romani",
    "Dvi  gimtosios kalbos": "other.two_mother_tongues",
    "Kitos": "other",
}
TOTAL = {"Iš viso pagal gimtąją kalbą"}


def resolve(label):
    if label in TOTAL:
        return None
    if label not in NAMES:
        raise KeyError(f"lt2021: unmapped label {label!r}")
    return NAMES[label]
