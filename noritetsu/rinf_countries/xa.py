"""Abkhazia (xa, a user-assigned code like Kosovo's xk): not RINF but the Abkhazian part of the
CIS tariff guide's Georgian Railway sheet (sections 57-001 and 57-007), which
caucasus_register.py writes into rinf.py's input files (data/raw/rinf/xa/) with names.json;
this file is its settings (caucasus_register.country_conf). caucasus_sources.md has the
sources and numbers.

A line is one tariff section, its Abkhazian part ("57-001" from the Psou bridge), named by its
two ends as OSM names their stations (in Abkhaz; English from OSM's name:en). The operator is
the Abkhazian Railway (manager code "57A"), whose track it is; the trains are Russian
Railways'. Tariff km are whole kilometres (`tol_abs` 1.5). `iso3` is only read by rinf.py's
--fetch, which this region never runs."""
from caucasus_register import country_conf

COUNTRY = country_conf("xa", "XAB")
