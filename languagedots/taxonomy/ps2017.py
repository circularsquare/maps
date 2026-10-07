"""Palestine, Arab Barometer IV (2016) first language and VII (2021-22) ethnic group
(sources/ps_surveys.py) -> node. Every answer is Arabic (and "Arab"; VII's two "Other" go on the
country's language, as Algeria): Levantine Arabic, sa.txt's node (nort3139; the Palestinian
varieties are South Levantine, sout3123, a dialect of it in Glottolog).
"""
NAMES = {"Arabic": "afroasiatic.levantine_arabic"}


def resolve(label):
    if label in NAMES:
        return NAMES[label]
    raise KeyError(f"ps2017: unmapped label {label!r}")
