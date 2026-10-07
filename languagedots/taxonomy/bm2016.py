"""Bermuda, 2016 census country of birth (sources/bm_census.py, sources/bm.md). No language
question: the Bermuda-born on English, every birthplace through sources/origin_mix.py (Azores on
Portugal's mix). The source script writes node ids, which pass straight through."""
EXTRA_NODES = []


def resolve(label):
    if label[:1].islower():   # a node id, from sources/origin_mix.py
        return label
    raise KeyError(f"bm2016: unmapped label {label!r}")
