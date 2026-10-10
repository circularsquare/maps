"""Micronesia (FSM) 2010 census, Table B10A, language mainly spoken at home, persons 3+
(sources/fm_census.py, sources/fm.md).
"""
OC = "austronesian.oceanic"
NAMES = {
    "English": "indoeuropean.germanic.english",
    "Yapese": "austronesian.oceanic.yapese",   # under Oceanic (Glottolog), merged 2026-10-08
    # Ulithian, Woleaian, Satawalese and the other Chuukic languages of Yap's outer islands; the
    # census prints them as one answer, so one leaf (a named label never sits on a group node)
    "Yapese - Outer Island Languages": f"{OC}.yap_outer_islands",
    "Chuukese": f"{OC}.chuukese",          # Mortlockese and the other Chuuk lagoon/outer varieties
    "Pohnpeian": f"{OC}.pohnpeian",
    "Sapwuahfikese": f"{OC}.sapwuahfik",
    "Pingelapese": f"{OC}.pingelapese",
    "Mwoakilese": f"{OC}.mokilese",
    # two Polynesian outlier languages printed as one answer
    "Nukuoroan / Kapingamarangian": f"{OC}.nukuoro_kapingamarangi",
    "Kosraean": f"{OC}.kosraean",
    "Other Pacific Island Languages": "austronesian",   # unnamed remainder (Palauan is not Oceanic)
    "Filipino": "austronesian.philippine.tagalog",
    "Chinese / Taiwanese": "sinotibetan.sinitic",       # unnamed Chinese variety: the group node
    "Japanese": "japonic.japanese",
    "Other Languages": "other",
}


def resolve(label):
    if label in NAMES:
        return NAMES[label]
    raise KeyError(f"fm2010: unmapped label {label!r}")
