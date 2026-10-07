"""Georgia: not RINF but the CIS tariff guide's Georgian sheet, which caucasus_register.py
writes into rinf.py's input files (data/raw/rinf/ge/) with names.json; this file is its
settings (caucasus_register.country_conf). caucasus_sources.md has the sources and numbers.

A line is one tariff section ("57-004"), named by its two ends as OSM names their stations in
Georgian; no line takes a number from OSM. Tariff km are whole kilometres (`tol_abs` 1.5)."""
from caucasus_register import country_conf

COUNTRY = country_conf("ge", "GEO")
