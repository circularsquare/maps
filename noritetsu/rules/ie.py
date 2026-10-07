"""Ireland's rules for build_model.py (country_rules): which route=train relations are named
trains rather than lines.

What OSM Ireland has (all 55 route=train relations read 2026-10-03, after the Northern
Ireland clip): Iarnród Éireann's InterCity routes by their ends ("Dublin - Cork", "Dublin -
Galway", "D. Connolly - Sligo", "Dublin - Rosslare", "Dublin - Westport", "Dublin - Waterford",
"Mallow - Tralee", "Limerick - Galway", "Limerick – Ballybrophy", "Ballina - Manulla Jctn"),
the commuter routes (Northern, Western and South Western Commuter, the Cork commuter lines),
the DART's four patterns, and the Enterprise (Dublin - Belfast, Irish Rail;Translink). Every
one is tagged service=long_distance, regional or commuter, so the tag decides nothing.

All of them are lines: the Enterprise runs about hourly (15 a day Monday to Saturday, 8 on
Sundays; en.wikipedia "Belfast–Dublin line"), as gb's rule has it on the other side; the
thinnest InterCity routes (Rosslare, Westport, Ballina, Limerick - Ballybrophy) run several
times every day, which is a line a rider uses, not a single train. The Luas lines are trams
and never named trains.

Named trains, none of which OSM maps today, written by name so they come out right when
mapped: Belmond's Grand Hibernian (a cruise train), the Railway Preservation Society of
Ireland's steam specials, and anything tagged as a night or car train.
"""
import re

NAMED_OPERATORS = ("Belmond", "Railway Preservation Society", "RPSI")
NAMED_SERVICE = {"night", "car", "car_shuttle"}
NAMED_NAME = re.compile(r"\b(?:Grand Hibernian|Steam Train|Sleeper|RPSI)\b", re.IGNORECASE)


def looks_like_service(tags, name, name_en):
    if set((tags.get("service") or "").split(";")) & NAMED_SERVICE:
        return True
    op = " ".join(tags.get(k) or "" for k in ("operator", "network"))
    if any(o in op for o in NAMED_OPERATORS):
        return True
    return bool(NAMED_NAME.search(name or "") or NAMED_NAME.search(name_en or ""))
