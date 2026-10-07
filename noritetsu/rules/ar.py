"""Argentina's rules for build_model.py (build_model.country_rules lists what it reads).

Register lines (ar_register.py) are the track of the passenger routes, grouped by hand; OSM
route relations are the services over them. ar_sources.md has the reasoning.
"""
import re

# Named trains, not lines (a single long-distance train, or an excursion):
#   - the long-distance trains (OSM service=long_distance): Constitución - Mar del Plata (six
#     days a week), Retiro - Rosario and Retiro - Junín (daily), Once - Bragado (three days a
#     week), Viedma - Bariloche (the Tren Patagónico, weekly in season). One or two trains a
#     day at most, each a single train over a long corridor; their track counts through the
#     register lines it lies on.
#   - excursions (service=tourism): La Trochita, the Tren a las Nubes, the Expreso Río Negro's
#     Perito Moreno outing, the Villa Elisa heritage train; and the Expreso Río Negro
#     (Bariloche - Jacobacci), the Tren Patagónico's own regional working.
# NOT named trains, although OSM tags them tourism: the Tren del Fin del Mundo (Ushuaia,
# several departures a day) and the Tren Ecológico de la Selva (Iguazú, every half hour),
# both scheduled interval services a visitor rides as a line.
LINES_NOT_NAMED = re.compile(r"Fin del Mundo|Ecol[oó]gico de la Selva", re.IGNORECASE)
NAMED = re.compile(r"Expreso R[ií]o Negro|Tren Patag[oó]nico|La Trochita|Tren a las Nubes|"
                   r"Villa Elisa|Ferrocarril (Roca|Mitre|San Mart[ií]n|Sarmiento)\b|"
                   r"larga distancia", re.IGNORECASE)
# A route_master ("Línea larga distancia del FC Mitre") is a named train when all its routes are.
SERVICE_IF_ALL_ROUTES_ARE = True

# Every OSM line whose stations all lie on its register match is its twin and is dropped:
# the register's Buenos Aires lines are the union of OSM's route_masters' routes, and a
# route_master missing one branch would otherwise stay beside it as an "(as operated)" copy.
TWIN_ON_STATIONS = True


def looks_like_service(tags, name, name_en):
    text = " ".join([name or "", name_en or "", tags.get("network") or "",
                     tags.get("operator") or ""])
    if LINES_NOT_NAMED.search(text):
        return False
    if set((tags.get("service") or "").split(";")) & {"night", "long_distance", "tourism"}:
        return True
    return bool(NAMED.search(name or ""))
