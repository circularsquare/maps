"""cm: not RINF. wafrica_register.py writes the hand line list (wafrica_lines.py) into rinf.py's
input files (data/raw/rinf/cm/) and names.json; this file is its settings
(wafrica_register.country_conf). cm_sources.md has the sources and numbers."""
from wafrica_register import country_conf

COUNTRY = country_conf("cm")
