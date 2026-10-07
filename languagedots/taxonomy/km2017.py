"""Comoros: no language question; each island's people on its own Comorian language
(sources/km_pop.py, sources/km.md), Glottolog's split: Ngazidja (ngaz1238), Ndzwani (ndzw1235),
Mwali (mwal1237). They are siblings of the existing `comorian` and `shimaore` leaves rather than
children of `comorian`, because France draws its "Comorian" answer on that node and a named label
must never sit on a group node. French (learned; AGENT_BRIEF §2) is not drawn.
"""
NAMES = {
    "Shingazidja": "nigercongo.bantu.comorian_ngazidja",
    "Shindzuani": "nigercongo.bantu.comorian_ndzwani",
    "Shimwali": "nigercongo.bantu.comorian_mwali",
}


def resolve(label):
    if label in NAMES:
        return NAMES[label]
    raise KeyError(f"km2017: unmapped label {label!r}")
