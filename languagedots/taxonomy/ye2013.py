"""Yemen, Arab Barometer III (2013) first language plus Socotra (sources/ye_surveys.py) -> node.

  "Arabic": Yemeni Arabic, sa.txt's node; Sanaani, Ta'izzi-Adeni and Hadrami are not told apart
    (no source places them by count).
  "Soqotri (Socotra)": Soqotri (soqo1240), Socotra governorate's whole population.
"""
NAMES = {
    "Arabic": "afroasiatic.yemeni_arabic",
    "Soqotri (Socotra)": "afroasiatic.soqotri",
}


def resolve(label):
    if label in NAMES:
        return NAMES[label]
    raise KeyError(f"ye2013: unmapped label {label!r}")
