"""Burundi: Afrobarometer R5-R6 home-language answers (sources/bi_afro.py, sources/bi.md).
R5's "Kiswahili" and R6's "Swahili" are one answer, merged in bi_afro.py's LABELS.
"""
NAMES = {
    "Kirundi": "nigercongo.bantu.kirundi",
    "Swahili": "nigercongo.bantu.swahili",
    "French": "indoeuropean.romance.french",
}


def resolve(label):
    if label in NAMES:
        return NAMES[label]
    raise KeyError(f"bi2014: unmapped label {label!r}")
