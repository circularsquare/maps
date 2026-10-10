"""Israel's rules for build_model.py (build_model.country_rules lists what it reads).

Nothing to set: `il_register.py --clip` drops every OSM route relation from data/proc/il
(Israel Railways' are numbered by its old line scheme, one still to Jerusalem Malha; the
light-rail ones repeat the register's lines, the Malha - HaTurim one labelled "Yellow Line";
Haifa's Metronit is a bus), so the build has register lines only. IR's service patterns are
not lines (il_sources.md). If OSM routes are ever kept, IR's would be listed in SKIP_ROUTES
here rather than read as lines.
"""
