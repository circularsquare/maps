"""Netherlands: ProRail's line id is its two end stations' codes, "Asd-Rtd", and it is the line
nl.wikipedia calls "spoorlijn Amsterdam - Rotterdam". The id is the ref, and the name is read
from the codes through RINF's own points (uopid "NL" + code). OSM has two line relations in
the whole country, and Wikidata's route numbers (001-166) are ProRail trajectories that do not
join these ids."""
import re


def nl_id_name(lid, uop):
    """"Asd-Rtd" -> "Amsterdam - Rotterdam": each code is the uopid "NL<code>" of a point."""
    parts = (lid or "").split("-")
    if len(parts) != 2:
        return None
    names = [uop("NL" + p) for p in parts]
    if not all(names):
        return None
    return " - ".join(re.sub(r"\s+(Centraal|CS)$", "", n) for n in names)


COUNTRY = {
    "iso3": "NLD", "wikidata": "Q55", "langs": ["nl", "en"],
    "ref": lambda lid: lid or None, "rule_certain": True,
    "id_name": lambda lid, uop: nl_id_name(lid, uop),
    "im": {"0084_IM": "ProRail"},
}
