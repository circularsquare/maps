"""np: not RINF. asia_register.py writes Nepal Railway Company's line (asia_lines.py) into rinf.py's input
files (data/raw/rinf/np/) and names.json; this file is its settings (asia_register.country_conf).
np_sources.md has the sources and numbers."""
from asia_register import country_conf

COUNTRY = country_conf("np")
