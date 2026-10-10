"""mm: not RINF. asia_register.py writes Myanma Railways' lines (asia_lines.py) into rinf.py's input
files (data/raw/rinf/mm/) and names.json; this file is its settings (asia_register.country_conf).
mm_sources.md has the sources and numbers."""
from asia_register import country_conf

COUNTRY = country_conf("mm")
