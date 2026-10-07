"""kg: not RINF but the CIS tariff guide (Тарифное руководство № 4), whose sheets
casia_register.py --convert writes into rinf.py's own input files (data/raw/rinf/kg/) plus
names.json. This file is its settings (casia_register.country_conf); casia_sources.md has the
sources and numbers. As Ukraine's (rinf_countries/ua.py): a line is one tariff section, named by
its two ends; tariff km are whole kilometres (tol_abs 1.5); the operator is the administration
whose sheet lists the section."""
from casia_register import country_conf

COUNTRY = country_conf("kg")
