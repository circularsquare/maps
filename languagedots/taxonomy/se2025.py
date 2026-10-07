"""Sweden, 2025: language labels written by sources/se_build.py -> node.

Sweden keeps no language statistics, so every label here is the build's own (sources/se.md):
  * "Swedish": everyone the proxies do not place elsewhere.
  * immigrant languages: SCB's population by country of birth (first generation) and by
    parents' country of birth (second generation), each country on its main language
    (France's table, sources/fr_build.py COUNTRY_LANG, with se_build.SE_OVERRIDES), Iraq,
    Syria, Turkey, Iran and Ethiopia split by Parkvall's per-language counts. "Chinese" (born
    in China) is the Sinitic group: the source names a country. "Serbo-Croatian" is the
    Yugoslavia-born, whose successor state SCB does not know.
  * national minority languages from cited estimates: Meänkieli, North, Lule and South Sami,
    Romani, Yiddish. Finnish comes out of the Finland-born and their children.
  * "Other": people whose own or parents' birth country SCB does not name anywhere.
"""
import nl2026

NAMES = dict(nl2026.NAMES)
NAMES.update({
    "Meänkieli": "uralic.meankieli",                     # torn1244
    # Glottolog's three Saami languages of Sweden that still have speakers, siblings of
    # Finland's undivided "Sami" leaf (a child would make that leaf a group, drawn washed out)
    "North Sami": "uralic.saami_north",                  # nort2671
    "Lule Sami": "uralic.saami_lule",                    # lule1254
    "South Sami": "uralic.saami_south",                  # sout2674
    "Romani": "indoeuropean.indoaryan.romani.romani",    # varieties not split (cz.txt's leaf)
    "Yiddish": "indoeuropean.germanic.continental.yiddish",
    # Iraq's Assyrians speak Assyrian (and Chaldean) Neo-Aramaic; Turkey's and Syria's
    # Syriacs, from Tur Abdin, speak Turoyo (turo1239). Parkvall counts both as "Aramaic".
    "Assyrian Neo-Aramaic": "afroasiatic.assyrian",      # assy1241
    "Turoyo": "afroasiatic.turoyo",
    "Iraqi Turkmen": "turkic.iraqi_turkmen",
    "Oromo": "afroasiatic.cushitic.lowland.oromo",
    "Serbo-Croatian": "indoeuropean.slavic.south.serbocroatian",
    "Other": "other",
})
EXTRA_NODES = []


def resolve(label):
    if label[:1].islower():   # a node id, from sources/origin_mix.py
        return label
    if label not in NAMES:
        raise KeyError(f"se2025: unmapped label {label!r}")
    return NAMES[label]
