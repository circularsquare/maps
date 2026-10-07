"""Malaysia's rules for build_model.py (build_model.country_rules lists what it reads)."""
import re

# KTM's intercity trains are a brand over the West Coast Line, as Korea's KTX is over its
# lines: OSM maps them as "ETS Gold (Padang Besar - Gemas)", "ETS Gold (KL Sentral - Ipoh)",
# network "KTM ETS". Named trains: their track counts through the register line it lies on.
# KTM Komuter's lines (Seremban Line, Port Klang Line, Butterworth - Padang Besar...) and the
# KLIA Ekspres (non-stop every 15-20 minutes) are what riders use as lines and stay lines.
# KTM's Intercity trains (Ekspres Rakyat Timuran, Shuttle Timur, Shuttle Tebrau) are mapped
# only as route=railway relations, which build_model does not read.
ETS = re.compile(r"^ETS\b")
# The Skypark Link (KL Sentral - Terminal Skypark) has been suspended since 2023-02-15; OSM
# still has its route relations (Eastbound, Westbound), but tags the branch railway=disused, so
# the OSM line came out as KL Sentral - Subang Jaya over the Port Klang branch, drawn as
# running. The register's Skypark branch is greyed (suspended); the two routes are left out
# (was a named-train stopgap until SKIP_ROUTES existed, 2026-10-04).
SKIP_ROUTES = {8391024, 9985660}


def looks_like_service(tags, name, name_en):
    return (tags.get("network") or "").strip() == "KTM ETS" or bool(ETS.match(name or ""))


# OSM writes the station codes into the name: "KA07 Kepong Sentral", "AG7 SP7 KJ13 Masjid
# Jamek", "KJ15 KL Sentral". The name is what follows them.
PLATFORM_SUFFIX = re.compile(r"^(?:[A-Z]{1,3}\d{1,2}[A-Z]?\s+)+")
