"""Each country's own rules for build_model.py, one file per country: `rules/<cc>.py`, owned by
that country's agent, so a country-specific rule never waits on a change to build_model.py.
build_model.country_rules() loads the file and its docstring lists every name it reads; each
is optional, and a country with no file builds by the defaults. `shared.py` holds patterns
several countries use (`from rules.shared import EU_TRAIN`)."""
