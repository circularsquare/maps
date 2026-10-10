# East and Southern Africa build (ke, et, dj, mz, zm, zw, tz, mg, mw, ug, mu), 2026-10-08: shared-file changes for the managing session

All eleven are built in `dist/data/` (model, tiles, check_model passing; the one flag is
Mauritius' 3.4 km Réduit branch built platform to platform, 0.89). Nothing below has been
applied. Each country's numbers and decisions: `<cc>_sources.md`, "Build (2026-10-08)".

## 1. tools/rebuild.py: REGISTER entries

```diff
     **{cc: f"wafrica_register:data/raw/rinf/{cc}"
        for cc in ("ng", "cm", "ao", "ga", "cg", "sn", "gh", "bf", "cd")},
+    # East and Southern Africa: hand lists through rinf.py (eafrica_register.py, nafrica's
+    # code). After an extract: `eafrica_register.py --clip <cc>`; then `--fill <cc>` for ke,
+    # mg, ug (Mombasa Central, Toamasina, Mukono are not in OSM). Mauritius is all light rail:
+    # no register (None); `--clip mu` still names its two route masters.
+    **{cc: f"eafrica_register:data/raw/rinf/{cc}"
+       for cc in ("ke", "et", "dj", "mz", "zm", "zw", "tz", "mg", "mw", "ug")},
+    "mu": None,
 }
```

## 2. borders.py: two EXTRA points

```diff
     ("eXARUPSOU", 40.008443, 43.393733, ["ru", "xa"]),
+    # Ethiopia - Djibouti (Addis Ababa - Djibouti Railway, a train every second day): where
+    # OSM's track (ways 967769108 / 1197034121) crosses OSM's boundary (way 31304862), 4.0 km
+    # past Dewele (eafrica_lines.BORDERS["XDJET1"]; et_sources.md "Build").
+    ("eXDJET1", 42.642891, 11.090904, ["dj", "et"]),
+    # Tanzania - Zambia at Tunduma - Nakonde (TAZARA's Mukuba Express, weekly): where OSM's
+    # track (way 200454396) crosses OSM's boundary (way 363623831), 0.8 km east of Nakonde
+    # (eafrica_lines.BORDERS["XTZZM1"]; zm_sources.md "Build").
+    ("eXTZZM1", 32.763516, -9.315615, ["tz", "zm"]),
 ]
```

Both are the "e" + uopid convention (as eXDZTN1): each register line ends at the point
`#XDJET1` / `#XTZZM1`, op uopid `XDJET1` / `XTZZM1`. Until they land the point shows under its
bare id ("XDJET1") in et, dj, tz, zm. After landing: rebuild et, dj, tz, zm (`python
tools/rebuild.py et dj tz zm`), so the point is named neutrally and the two sides join.
eafrica_lines' `XMWMZ1` (Nayuchi) is only for the clip: no line ends there and it needs no
EXTRA entry (no train crosses at Nayuchi or Victoria Falls).

## 3. tools/build_regions.py

Run it to put the eleven in `regions.json`. Built neighbours: South Africa shares no running
crossing with any of them (Ressano Garcia - Komatipoort has no through train; Beitbridge and
Botswana have no service), so no neighbour rebuild is needed; et/dj and tz/zm are each other's.

## 4. check_model.py

Edited in place (owned entries): REGISTER ke, et, dj, mz, zm, zw, tz, mg, mw; KNOWN et (Addis
LRT), mu (Metro Express). Uganda has no published length (ug_sources.md).

## Files the agent wrote (for the record)

New: `eafrica_register.py`, `eafrica_lines.py`, `rinf_countries/{ke,et,dj,mz,zm,zw,tz,mg,mw,ug}.py`,
`rules/{ke,et,dj,mz,zm,zw,tz,mg,mw,ug}.py`, `colours/{ke,et,dj,mz,zm,zw,tz,mg,mw,ug}.csv`,
`data/raw/rinf/<cc>/`, `data/proc/<cc>/`, `dist/data/<cc>*` for all eleven. Edited:
`check_model.py`, the eleven `<cc>_sources.md` ("Build (2026-10-08)").

eafrica_register imports nafrica_register (and mideast_register for `--join`, unused so far)
and extends nafrica's LINES / NOT_SERVICE / BORDERS / LANGS / ISO3 in its own process only;
nafrica_register.py is unchanged, but a change to its function signatures (convert, clip,
fill, trace_cmd, fork_cmd, build, split_pieces, country_conf) or to rinf.osm_stations would
break these ten. All extracts are deleted; `data/proc/<cc>/` holds what a rebuild needs. A
re-extract needs `eafrica_register.py --clip <cc>` (and `--fill` for ke, mg, ug) before
build_model.
