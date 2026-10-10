"""Iraq, MICS6 2018, language of the household head (HC1B; Kurdish split by HH16, the
respondent's native language; sources/iq_mics6.py) -> node. The labels are iq_mics6.py's normalised ones. Until 2026-10-09 this mapped the pooled
WVS / Arab Barometer answers (sources/iq_surveys.py); those labels are kept so that file still
resolves.

  "Arabic": Iraqi Arabic, a node of its own beside `arabic` (Mesopotamian Arabic: Glottolog
    meso1252 Gilit and nort3142 North Mesopotamian; no source tells them apart), following
    Morocco's Darija and Sudan's Sudanese Arabic.
  "Kurdish (Sorani)": MICS's "Kurdish Surani", Central Kurdish (Glottolog cent1972), a new child
    of `indoeuropean.iranian.kurdish`.
  "Kurdish (Badini)": MICS's "Kurdish Badinani", the Badini (Bahdini) dialect of Northern Kurdish /
    Kurmanji (nort2641), a new child of the same. Other countries' Kurdish stays on the parent,
    since none of their sources names a variety.
  "Turkmen": Iraqi Turkmen, `turkic.iraqi_turkmen`, not `turkic.turkmen`: Glottolog files Iraq's
    Turkmen as the Kirkuk dialect (kirk1242) of South Azerbaijani.
  "Assyrian Neo-Aramaic": MICS's "Asserian", the existing `afroasiatic.assyrian` (assy1241).
  "Other": MICS's "others" (Nineveh 5.2%, plausibly Shabak, but nothing names it): `other`.
  "Kurdish": the parent. MICS's Kurdish-headed households whose respondent named no variety
    (Baghdad, Diyala: plausibly Feyli, Southern Kurdish, which MICS has no code for); also the
    survey pool's one Kurdish answer. "Shabaki" (shab1251): survey pool only (iq_surveys.py).
"""
NAMES = {
    "Arabic": "afroasiatic.iraqi_arabic",
    "Kurdish (Sorani)": "indoeuropean.iranian.kurdish.central",
    "Kurdish (Badini)": "indoeuropean.iranian.kurdish.northern",
    "Turkmen": "turkic.iraqi_turkmen",
    "Assyrian Neo-Aramaic": "afroasiatic.assyrian",
    "Other": "other",
    "Kurdish": "indoeuropean.iranian.kurdish",
    "Shabaki": "indoeuropean.iranian.shabaki",
}


def resolve(label):
    if label in NAMES:
        return NAMES[label]
    raise KeyError(f"iq2018: unmapped label {label!r}")
