"""Jordan: sources/jo_build.py writes node ids straight into data/normalized/jo.csv, so this
mapping is the identity, checked against the nodes it wrote.

Jordanians: Arab Barometer II-IV first language ("Arabic" -> Levantine Arabic, nort3139; English,
French, German as answered; four "Serbo-Croatian" answers in Jerash and Mafraq, where no Bosnian
or Serbian nationals live, on `other`) and VII's ethnic group (Circassian at Rannut 2009's
home-use share, on `abkhazadyghe.circassian`). Non-Jordanians: 2015 census nationality on its
home language or drawn home mix (sources/gulf_mix.py); Libyans on plain `arabic` (Libya is not
drawn); nationalities under 500 people on `other`.
"""
from pathlib import Path

import pandas as pd

_CSV = Path(__file__).resolve().parent.parent / "data" / "normalized" / "jo.csv"
NAMES = {}
EXTRA_NODES = sorted(pd.read_csv(_CSV, usecols=["source_category"])["source_category"].unique())


def resolve(label):
    if label not in EXTRA_NODES:
        raise KeyError(f"jo2015: {label!r} is not a node sources/jo_build.py wrote")
    return label
