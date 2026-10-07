"""Iraq, World Values Survey waves 4, 6, 7 and Arab Barometer II, III (2004-2018), first / home
language (sources/iq_surveys.py) -> node. The labels are iq_surveys.py's normalised ones.

  "Arabic": Iraqi Arabic, a node of its own beside `arabic` (Mesopotamian Arabic: Glottolog
    meso1252 Gilit and nort3142 North Mesopotamian; no survey tells them apart), following
    Morocco's Darija and Sudan's Sudanese Arabic.
  "Kurdish": the existing `indoeuropean.iranian.kurdish` leaf. Every card had one Kurdish answer,
    so Sorani (Central Kurdish), Badini (Northern Kurdish / Kurmanji) and Feyli (Southern Kurdish)
    are not told apart and none is guessed from the governorate. In Duhok it is Arab Barometer
    VI-3 / VII's ethnic Kurd and Yazidi (sources/iq_surveys.py).
  "Turkmen": Iraqi Turkmen, a new leaf `turkic.iraqi_turkmen`, not `turkic.turkmen`: Glottolog
    files Iraq's Turkmen as the Kirkuk dialect (kirk1242) of South Azerbaijani.
  "Assyrian Neo-Aramaic": the existing `afroasiatic.assyrian` (assy1241). WVS 7's "Assyrian
    Neo-Aramaic" and Arab Barometer III's "Assyrian", one answer each.
  "Shabaki": shab1251, a new leaf. Arab Barometer II, two answers in Nineveh.
  "Other": WVS's bare other: `other`.
"""
NAMES = {
    "Arabic": "afroasiatic.iraqi_arabic",
    "Kurdish": "indoeuropean.iranian.kurdish",
    "Turkmen": "turkic.iraqi_turkmen",
    "Assyrian Neo-Aramaic": "afroasiatic.assyrian",
    "Shabaki": "indoeuropean.iranian.shabaki",
    "Other": "other",
}


def resolve(label):
    if label in NAMES:
        return NAMES[label]
    raise KeyError(f"iq2018: unmapped label {label!r}")
