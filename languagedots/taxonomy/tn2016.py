"""Tunisia, Arab Barometer II-IV (2011-2016) first language, plus Gabsi 2011's Tunisian Berber
estimate (sources/tn_surveys.py) -> node. Labels are tn_surveys.py's normalised ones.

  "Arabic": Tunisian Arabic (tuni1259), a node of its own beside `arabic`, as Algeria's and
    Morocco's Darija.
  "Berber": Tunisian Berber (tuni1262, Tunisian-Zuwara Berber). The one survey answer
    ("Amazigh", Tunis) and Gabsi's south-eastern speakers: every surviving Tunisian variety
    (Djerba, Chenini, Douiret, Matmata) is in this one Glottolog language, so the named leaf.
  "French", "English", "Italian": as answered, first language (ten respondents of 3,595).
"""
NAMES = {
    "Arabic": "afroasiatic.tunisian_arabic",
    "Berber": "afroasiatic.berber.tunisian_berber",
    "French": "indoeuropean.romance.french",
    "English": "indoeuropean.germanic.english",
    "Italian": "indoeuropean.romance.italian",
}


def resolve(label):
    if label in NAMES:
        return NAMES[label]
    raise KeyError(f"tn2016: unmapped label {label!r}")
