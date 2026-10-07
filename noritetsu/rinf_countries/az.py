"""Azerbaijan: not RINF but the CIS tariff guide's Azerbaijani sheet, which
caucasus_register.py writes into rinf.py's input files (data/raw/rinf/az/) with names.json;
this file is its settings (caucasus_register.country_conf). caucasus_sources.md has the
sources and numbers."""
from caucasus_register import country_conf

COUNTRY = country_conf("az", "AZE")
