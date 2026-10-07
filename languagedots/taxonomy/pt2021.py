"""Portugal, 2021: language labels written by sources/pt_censos.py -> node.

Portugal's census asks no language, so every label here is the build's own, not a census
category (sources/pt.md says how each is derived):
  * "Portuguese": Portuguese nationals, plus foreign nationals of Portuguese-speaking countries
    (Brazil, Angola, Mozambique, Sao Tome), plus the share of other foreigners moved by the
    retention step.
  * "Mirandese": Costas's 2020 estimate of regular users, in Miranda do Douro and two Vimioso
    freguesias.
  * immigrant languages: foreign nationals by nationality (Censos 2021), each nationality on its
    country's main language, labels as France's build writes them (fr2023.NAMES, reused so one
    label means one node in both countries). "Chinese" is the Sinitic group, as in France: the
    source names a country, not one of its languages.
  * "Other": the stateless (149 people).
"""
from fr2023 import NAMES as _FR

NAMES = dict(_FR)
NAMES.update({
    "Mirandese": "indoeuropean.romance.mirandese",
    "Other": "other",
})
EXTRA_NODES = []


def resolve(label):
    if label[:1].islower():   # a node id, from sources/origin_mix.py
        return label
    if label not in NAMES:
        raise KeyError(f"pt2021: unmapped label {label!r}")
    return NAMES[label]
