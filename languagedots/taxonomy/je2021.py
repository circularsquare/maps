"""Jersey, 2021 census ethnic and cultural identity (sources/je_census.py, sources/je.md). No
language question: English for the island's own and the British-Irish, every other group through
sources/origin_mix.py. The source script writes node ids, which pass straight through."""
EXTRA_NODES = []


def resolve(label):
    if label[:1].islower():   # a node id, from sources/origin_mix.py
        return label
    raise KeyError(f"je2021: unmapped label {label!r}")
