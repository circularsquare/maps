"""South Africa's rules for build_model.py (build_model.country_rules lists what it reads).

Register lines are za_register.py's track pieces; OSM's Metrorail and Gautrain route relations
are the services over them (Gautrain's are its lines: it has no register line). za_sources.md
has the reasoning.
"""
import re

# Named trains, not lines:
#   - Shosholoza Meyl's long-distance routes (operator "Shosholoza Meyl"), the Blue Train,
#     Rovos Rail and Premier Classe: one or two trains a week at most, a long-distance train
#     each. Their track counts through the register lines it lies on.
#   - Tourist and excursion trains (service=tourism: Apple Express, Ceres Rail, Hexpas
#     Express), NRZ's Bulawayo - Beitbridge, and the stop-less old routes OSM keeps
#     (Komatipoort - Maputo, Beit Bridge - Pretoria, Musina - Johannesburg, Komatipoort -
#     Pretoria, Kaapmuiden - Barberton): none is scheduled more often than weekly today.
#   - STOPGAP, as Mexico's Línea Z route: OSM routes of services that do not run today at all,
#     wholly over greyed (suspended) register lines. As lines they would be drawn and counted
#     as running beside the greyed track; build_model has no way to grey an OSM line, and as
#     named trains they count towards nothing. Kei Rail (Mthatha - Amabele - East London; no
#     service found since the late 2010s) is here too: its track is no register line.
#     Routes that run in part today stay lines (Durban - Kelso: to Winklespruit; Cape Town -
#     Stellenbosch: to Du Toit; the Red Line via President and the Light Green Line via Crown
#     inside masters that run): build_model builds a master's routes as one line, and the
#     part over greyed track then owns that track as running (za_sources.md "Known faults").
NAMED_OPERATOR = re.compile(r"Shosholoza|Blue Train|Rovos|Premier Classe|National Railways of "
                            r"Zimbabwe|Kei Rail", re.IGNORECASE)
NAMED_NAME = re.compile(
    r"Blue Train|Rovos|Premier Classe|Apple Express|Ceres Rail|Hexpas|Bulawayo|Komatipoort|"
    r"Beit ?Bridge|Musina|Kaapmuiden|"
    # not running (za_sources.md "Not running today")
    r"Daveyton|Springs|Nigel|New Canada.*Germiston|Germiston.*New Canada|"
    # every Vereeniging and Oberholzer service: none runs (trains turn at Lenasia, and the
    # Johannesburg - Lenasia trains have no OSM route of their own)
    r"Vereeniging|Oberholzer|"
    r"Faraday|Westgate|\(via Crown\)|\(via President\)|"
    r"North Coast Line|West Line|Bluff",
    re.IGNORECASE)


# A route_master whose routes are all named trains by the rules above is one too (Metrorail's
# Black, Brown, White and Violet lines, Durban's North Coast, West and Umlazi-Bluff lines: the
# masters' own names, "Metrorail Black Line", say nothing).
SERVICE_IF_ALL_ROUTES_ARE = True


def looks_like_service(tags, name, name_en):
    if set((tags.get("service") or "").split(";")) & {"tourism", "national", "international",
                                                       "long_distance", "night"}:
        return True
    if NAMED_OPERATOR.search(tags.get("operator") or ""):
        return True
    return bool(NAMED_NAME.search(name or ""))
