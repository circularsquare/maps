"""pk: not RINF. pk_register.py writes Pakistan's hand-written line list (pk_lines.py) into
rinf.py's input files (data/raw/pk/) and names.json; this file is its settings
(pk_register.country_conf). pk_sources.md has the sources and numbers."""
from pk_register import country_conf

COUNTRY = country_conf()
