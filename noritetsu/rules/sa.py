"""Saudi Arabia's rules for build_model.py (build_model.country_rules lists what it reads).

SAR's three lines are register lines (mideast_register.py). OSM's own route relations for
them are left out: each is the register line again under another name, so it stayed beside it
as a second line over the same track ("SAR Line 1", "القريات - الرياض"), and the Haramain's
lists no stops at all.
"""

SKIP_ROUTES = {
    8273934,     # SAR Line 1: Riyadh -> Dammam (= East Train)
    13880223,    # SAR Line 1: Dammam -> Riyadh (its ways run 706 km: a detour mapped in)
    13880224,    # Al Qurayyat -> Riyadh (= North Train)
    13880225,    # Riyadh -> Al Qurayyat
    7597735,     # Haramain High Speed Line: no stops, both tracks
}
