"""Mexico's rules for build_model.py (build_model.country_rules lists what it reads).

Register lines (mx_register.py) are the named passenger track, metros and light rail
included; OSM route relations are the services over them. mx_sources.md has the reasoning.
"""
import re

# Named trains, not lines:
#   - El Chepe Express (Los Mochis - Creel, three days a week each way, a faster tourist train
#     over part of the line the Chepe Regional runs whole): the Regional is the line's
#     service, the Express a named train over it, its track counting through the register
#     line "Chihuahua al Pacífico".
#   - the tequila excursion trains out of Guadalajara (José Cuervo Express, Tequila Express):
#     Saturdays only, a day-tour package; under Anita's "more often than about once a week"
#     they do not count, and their track (Línea T, Línea I) is no register line.
#   - Amtrak's Sunset Limited and Texas Eagle, which the Geofabrik extract carries for the few
#     km it holds around El Paso (US trains; built with the USA).
NAMED = re.compile(r"Chepe Express|Cuervo Express|Tequila Express|Amtrak", re.IGNORECASE)

# OSM's route for the Tren Interoceánico's Línea Z ("Ferrocarril del Istmo de Tehuantepec",
# network Ferroistmo) is left out. No passenger train has run since 28 Dec 2025 and the
# register line is `suspended` (greyed); the route would otherwise be a running line beside it
# (not the register line's twin: it lists 8 of its 13 stops). Was flagged as a named train
# until SKIP_ROUTES existed (2026-10-04).
SKIP_ROUTES = {16929728}

# Proposed (mx_sources.md, not yet read by build_model): an OSM line matched by name to a
# register line is its twin when it shares the register line's stations, whatever its length.
# OSM's Metrorrey lines 1-3 and Mexico City's Línea 4 list their stops out of order in one
# direction, which gives them a section from one end to the other (Exposición - Talleres
# 18.7 km on Metrorrey Línea 1), so they measure 1.3-2 times their register line and stay
# beside it as "(as operated)" copies with that false section.
TWIN_ON_STATIONS = True


def looks_like_service(tags, name, name_en):
    if set((tags.get("service") or "").split(";")) & {"night", "long_distance", "tourism"}:
        return True
    text = " ".join([name or "", name_en or "", tags.get("network") or "",
                     tags.get("operator") or ""])
    return bool(NAMED.search(text))
