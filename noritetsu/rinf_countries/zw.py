"""zw: not RINF. eafrica_register.py writes the line list (eafrica_lines.py) into rinf.py's
input files (data/raw/rinf/zw/) and names.json; this file is its settings
(nafrica_register.country_conf). zw_sources.md has the sources and numbers."""
from eafrica_register import country_conf

COUNTRY = country_conf("zw")
