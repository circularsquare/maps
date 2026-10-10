"""ph: not RINF. asia_register.py writes PNR's lines (asia_lines.py) into rinf.py's input
files (data/raw/rinf/ph/) and names.json; this file is its settings (asia_register.country_conf).
ph_sources.md has the sources and numbers."""
from asia_register import country_conf

COUNTRY = country_conf("ph")
