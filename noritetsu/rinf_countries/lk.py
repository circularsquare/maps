"""lk: not RINF. lk_register.py writes Sri Lanka Railways' lines (en.wikipedia's by-line station
tables, km from Colombo Fort) into rinf.py's input files (data/raw/rinf/lk/) and names.json; this
file is its settings (lk_register.country_conf). lk_sources.md has the sources and numbers."""
from lk_register import country_conf

COUNTRY = country_conf("lk")
