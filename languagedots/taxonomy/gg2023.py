"""Guernsey, 2023 Electronic Census country of birth (sources/gg_census.py, sources/gg.md). No
language question: the Guernsey-born and British-Irish on English, every other birthplace through
sources/origin_mix.py. The source script writes node ids, which pass straight through."""
EXTRA_NODES = []


def resolve(label):
    if label[:1].islower():   # a node id, from sources/origin_mix.py
        return label
    raise KeyError(f"gg2023: unmapped label {label!r}")
