"""The UAE's rules for build_model.py (build_model.country_rules lists what it reads).

Etihad Rail's passenger route is a register line (mideast_register.py). OSM's route relations
for it are left out: they are the register line again under another name, and list one or
two of its stops (the Fujairah ones only Mohamed Bin Zayed City; the Dubai one carries the
Fujairah route's Arabic name, a slip).
"""

SKIP_ROUTES = {
    21261180,    # Etihad Rail: Fujairah -> Abu Dhabi
    21261181,    # Etihad Rail: Abu Dhabi -> Fujairah
    21493966,    # Etihad Rail: Dubai -> Abu Dhabi
}
