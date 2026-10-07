"""Norway: NOT built from RINF. This file exists only so `python rinf.py --fetch no` runs.

Norway is built by no_register.py from Bane NOR's Banenettverk and Entur's timetable
(`--register no_register:data/raw/no`). Bane NOR's RINF has no era:lineId (each section's
nationalLine is labelled "B05-Nordlandsbanen" instead) and its 375 sections, 3,234 km, leave
out Støren's sections, Drammen - Galleberg, Barkåker - Sem, every border section and half of
Ofotbanen (no_sources.md). Building `--register rinf:data/raw/rinf/no` would give lines
grouped by connected piece only, named "first - last": do not.
"""

COUNTRY = {
    "iso3": "NOR", "wikidata": "Q20", "langs": ["nb", "no", "nn", "en"],
}
