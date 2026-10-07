"""za: not RINF. za_register.py writes South Africa's hand-written line list into rinf.py's input
files (data/raw/rinf/za/) and names.json; this file is its settings (za_register.country_conf).
za_sources.md has the sources and numbers."""
from za_register import country_conf

COUNTRY = country_conf()
