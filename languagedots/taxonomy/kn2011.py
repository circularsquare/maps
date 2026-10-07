"""St Kitts and Nevis, 2011 census country of birth by island (sources/kn_census.py,
sources/kn.md). No language question: the native-born on the Leeward creole (`antiguan`,
Glottolog anti1245), the foreign-born by birthplace. The source script writes node ids, which
pass straight through."""
EXTRA_NODES = []


def resolve(label):
    if label[:1].islower():   # a node id, from sources/kn_census.py / origin_mix
        return label
    raise KeyError(f"kn2011: unmapped label {label!r}")
