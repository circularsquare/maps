"""bd: not RINF. bd_register.py writes Bangladesh Railway's hand-written line list (bd_lines.py)
into rinf.py's input files (data/raw/rinf/bd/) and names.json through lk_register.py's engine;
this file is its settings. bd_sources.md has the sources and numbers."""
from bd_register import country_conf

COUNTRY = country_conf("bd")
