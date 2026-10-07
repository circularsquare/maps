"""Belgium, census 2021: language labels written by sources/be_census.py -> node.

Belgium's census asks no language, so every label is the build's own, not a census category
(sources/be.md says how each is derived): Dutch, French and German for the language areas and
the BRIO surveys, and immigrant languages by country of birth on France's country -> language
table (sources/fr_build.py COUNTRY_LANG), so the labels and nodes are France's.
"""
from fr2023 import NAMES as _FR

NAMES = dict(_FR)
EXTRA_NODES = []


def resolve(label):
    if label[:1].islower():   # a node id, from sources/origin_mix.py
        return label
    if label not in NAMES:
        raise KeyError(f"be2021: unmapped label {label!r}")
    return NAMES[label]
