"""Malawi's rules for build_model.py (build_model.country_rules lists what it reads).

CEAR's two passenger lines are register lines (eafrica_register.py). OSM's Limbe - Balaka and
Balaka - Nayuchi routes are the register lines again: left out.
"""

SKIP_ROUTES = {
    14482778, 14482779,     # Limbe - Balaka, Balaka - Limbe
    14482012, 14482013,     # Nayuchi - Balaka, Balaka - Nayuchi
}
