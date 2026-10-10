"""sa: not RINF. mideast_register.py writes SAR's line list (mideast_lines.py) into rinf.py's
input files (data/raw/rinf/sa/) and names.json; this file is its settings
(nafrica_register.country_conf). sa_sources.md has the sources and numbers."""
from mideast_register import country_conf

COUNTRY = country_conf("sa")
