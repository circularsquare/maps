"""mn: not RINF. asia_register.py writes UBTZ's lines (asia_lines.py) into rinf.py's input
files (data/raw/rinf/mn/) and names.json; this file is its settings (asia_register.country_conf).
mn_sources.md has the sources and numbers."""
from asia_register import country_conf

COUNTRY = country_conf("mn")
