"""Kenya's rules for build_model.py (build_model.country_rules lists what it reads).

Kenya Railways' lines are register lines (eafrica_register.py). OSM's two SGR route relations
are the register's Mombasa - Nairobi SGR again under another name, so each would stay beside it
as a second line over the same track: left out. Nairobi Commuter Rail's five routes stay OSM
lines over the register's metre-gauge pieces.
"""

SKIP_ROUTES = {
    7190306,     # Mombasa-Nairobi SGR (no stops, both tracks)
    7329392,     # Madaraka Express : Nairobi - Mombasa (= the register's SGR line)
}
