"""Mozambique's rules for build_model.py (build_model.country_rules lists what it reads).

CFM's and CDN's passenger lines are register lines (eafrica_register.py). OSM's train routes
over them are the register lines again under other names, most with no stops: left out. The
Maputo - Manhiça commuter route stays an OSM line over the Limpopo line.
"""

SKIP_ROUTES = {
    5419781,     # Maputo–Chicualacuala (= Linha do Limpopo)
    8468034,     # Maputo–Ressano Garcia (= Linha de Ressano Garcia)
    21440998,    # Maputo / Ressano Garcia (as above)
    9420998,     # Linha Beira - Moatize (= the Machipanda line to Dondo + the Sena line)
    14499856,    # Linha Moatize - Beira (as above)
    14483127,    # Nampula - Cuamba (= the register's)
    14483128,    # Cuamba - Nampula
}
