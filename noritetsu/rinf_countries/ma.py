"""ma: not RINF. nafrica_register.py writes ONCF's line list (nafrica_lines.py) into rinf.py's
input files (data/raw/rinf/ma/) and names.json; this file is its settings
(nafrica_register.country_conf). nafrica_sources.md has the sources and numbers."""
from nafrica_register import country_conf

COUNTRY = country_conf("ma")
