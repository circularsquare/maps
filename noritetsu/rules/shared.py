"""Patterns more than one country's rules use."""
import re

# International and long-distance trains mapped one relation per train: EC, EN, NJ, TGV,
# European Sleeper, Eurostar, Lyria. Used as the whole rule in at, be, nl, ch, cz, si, bg, sk,
# and inside se, hr, it, es, de.
# "ICE" followed by a one- or two-digit number is a DB interval line ("ICE 43", "ICE 91"), the
# same route_master built in Germany, where it is a line (DE_LINE); not a single train.
EU_TRAIN = re.compile(r"^(?:Train\s+)?(?:EC|EN|ICE(?!\s?\d{1,2}(?:\.\d)?(?!\d))|NJ|TGV|ES|ECE"
                      r"|INT)(?:[\s\d:]|$)"
                      r"|\bEuro(?:City|Night)\b|\bNightjet\b|\bEuropean Sleeper\b"
                      r"|^(?:Eurostar|Thalys|TGV Lyria|Lyria)\b")

# VR's single trains by number (fi; also in se, where VR's night train runs to Narvik).
FI_TRAIN = re.compile(r"\bPYO\s?\d|^Taajamajuna\s+\d")

# Amtrak's long-distance and once-a-day trains by name (us; also in ca).
US_TRAIN = re.compile(r"\b(?:Auto Train|California Zephyr|Capitol Limited|Cardinal|"
                      r"City of New Orleans|Coast Starlight|Crescent|Empire Builder|Floridian|"
                      r"Lake Shore Limited|Palmetto|Silver Meteor|Silver Star|Southwest Chief|"
                      r"Sunset Limited|Texas Eagle|Adirondack|Maple Leaf|Pennsylvanian|"
                      r"Vermonter|Carolinian|Ethan Allen Express|Heartland Flyer|Pere Marquette|"
                      r"Blue Water|Borealis|Berkshire Flyer|Winter Park Express|VIA Rail)\b")
