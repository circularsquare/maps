"""kh: not RINF. asia_register.py writes Royal Railway's lines (asia_lines.py) into rinf.py's input
files (data/raw/rinf/kh/) and names.json; this file is its settings (asia_register.country_conf).
kh_sources.md has the sources and numbers."""
from asia_register import country_conf

COUNTRY = country_conf("kh")
