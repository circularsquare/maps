"""Latvia, Tautas skaitīšana 2011, language mostly spoken at home (mājās pārsvarā lietotā valoda)
-> node. sources/lv_census.py; keyed by CSB's English labels exactly as data/normalized/lv.csv
carries them (PxWeb TSG11-07, lower case in the source).

Seven answers at municipality level. CALLS (sources/lv.md says more):
  latvian, russian, belarusian, ukrainian, polish, lithuanian: one leaf each, the nodes every
    other country uses.
  other (6,922): `other`. It holds every language not listed (Romani, German, Armenian,
    Estonian, Livonian and migrants' languages alike); the table does not let an indigenous
    remainder be told apart from a foreign one.
  Latgalian: NOT A HOME-LANGUAGE ANSWER. Latgalian speakers answered "Latvian" (the law treats
    Latgalian as a variety of Latvian); a separate question asked who uses Latgalian daily
    (TSG11-08, 164,506, 8.8%, a quarter of them Russian at home). Daily use is not the language
    mostly spoken at home, so nothing is moved onto a Latgalian node; sources/lv.md has the
    reasoning and the figures.
  Not stated (193,559, 9.35%): not drawn, in `gap`.
"""
IE = "indoeuropean"
SL = f"{IE}.slavic"

NAMES = {
    "latvian": f"{IE}.baltic.latvian",
    "lithuanian": f"{IE}.baltic.lithuanian",
    "russian": f"{SL}.east.russian",
    "belarusian": f"{SL}.east.belarusian",
    "ukrainian": f"{SL}.east.ukrainian",
    "polish": f"{SL}.west.polish",
    "other": "other",
}
NOT_DRAWN = {"Population", "Not stated (population less those with a home language)"}


def resolve(label):
    if label in NOT_DRAWN:
        return None
    if label not in NAMES:
        raise KeyError(f"lv2011: unmapped label {label!r}")
    return NAMES[label]
