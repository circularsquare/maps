"""Libya: the labels sources/ly_surveys.py writes -> language nodes. Survey answers (WVS 6 and 7,
Arab Barometer III) plus cited estimates for Nafusi, Tuareg and Tebu; every row `modelled`.
Record: sources/ly.md.

- Arabic: every survey answer "Arabic" from Libya is Libyan Arabic (Glottolog liby1240), Egypt's
  `libyan_arabic` node (drawn there for the Awlad Ali of Matrouh).
- Berber: the cards' "Berber; Amazigh; Tamazight" and Arab Barometer's "Amazigh". In Libya's
  north-west that is the Nafusi cluster (Glottolog nafu1238: Jadu, Nalut, Yafran, and Zuwara's
  dialect); a new leaf, since the answer names Libyan Berber, not Algeria's or Morocco's.
  Ghadames (Nalut) and Awjila (Ajdabiya) speakers, a few thousand each, sit inside it unsplit.
- Tamahaq: Tahaggart Tamahaq (Glottolog taha1241, LY among its countries), Algeria's `tamahaq` leaf.
- Tedaga: Tebu, on the `tubu` (Teda-Daza) leaf.
"""
NAMES = {
    "Arabic": "afroasiatic.libyan_arabic",
    "Berber": "afroasiatic.berber.nafusi",
    "Tamahaq": "afroasiatic.berber.tamahaq",
    "Tedaga": "nilosaharan.tubu",
    "English": "indoeuropean.germanic.english",
    "Other": "other",
}

EXTRA_NODES = []


def resolve(label):
    return NAMES.get(label)
