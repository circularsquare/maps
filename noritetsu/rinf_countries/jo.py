"""jo: not RINF. mideast_register.py writes the Hejaz Railway's line (mideast_lines.py) into
rinf.py's input files (data/raw/rinf/jo/) and names.json; this file is its settings
(nafrica_register.country_conf). jo_sources.md has the sources and numbers."""
from mideast_register import country_conf

COUNTRY = country_conf("jo")
