# NJ Transit Morristown Line and the Morris & Essex diagram (Anita's items 4 and 5, 2026-10-08)

US agent's working notes. Two separate causes.

## Item 4: Dover -> Hoboken could not be entered (app, every country)

Not the data. On the Morristown Line (u974aaf02cc) the diagram already listed Dover past the
Denville junction (uj486552, on the Montclair-Boonton Line) and Secaucus Junction past the
Northeast Corridor junction, and `pickedRun()` returned a good run (Dover -> Hoboken 64.3 km,
two legs). But `renderLine` then threw `TypeError: Cannot read properties of undefined (reading
'length')` at `run.ways.length`: the two junction branches of `pickedRun` (`PICK.through`,
`PICK.past`) return no `ways` (nor `drawn`), which the "by the other track" choices of 10-07 read.
The panel stayed as it was, with no "Add ride". Every pick through or past a junction end, on
every line in every country, failed the same way. tools/line_100_probe.js did not see it because
it calls pickedRun, not render.

**Landed in dist/index.html** (after the app agent finished; lint OK): both returns now carry
`drawn: null, ways: []`:

```diff
-    return { line, short: p, alt: null, around: false, p, legs };
+    return { line, short: p, drawn: null, alt: null, ways: [], around: false, p, legs };
```
(twice, in pickedRun's `PICK.through` and `PICK.past` branches; a comment above the first.)

Evidence (headless Chrome, clicking the diagram rows, on a private copy of index.html with the
same edit): 22 picks through junction ends on 9 US lines failed before, 22 passed after.
Morristown Line: Dover -> Hoboken, Hoboken -> Dover, Secaucus -> Hoboken, Hoboken -> Secaucus,
Dover -> Secaucus (three legs); Gladstone Branch Summit <-> Gladstone; LIRR Port Washington
Branch Woodside <-> Port Washington; Metro-North New Canaan Branch Stamford <-> New Canaan; SEPTA
Main Line Wayne Junction <-> Lansdale, West Chester Branch Penn Medicine <-> Wawa, Norristown
Branch (5 picks, one through two junctions), Airport Branch Eastwick <-> Airport. Dover ->
Hoboken added as two rides (Montclair-Boonton Dover - junction with cuts, Morristown Line junction
- Hoboken) and credited the Morristown Line 57.81 of 58.47 km (the rest is the 0.66 km stub to
the Northeast Corridor junction). Not re-run against the live index.html: the browser run was
refused by the permission classifier after the edit.

## Item 5: the Morris & Essex Lines' diagram (data: broken OSM relations)

The diagram Anita saw is the OSM line m10441055 "NJ Transit Morris & Essex Lines" (an operating
pattern, not listed). Its two route relations, 1377998 "Morristown Line: New York <=>
Hackettstown" and 1377996 "Gladstone Branch: New York <=> Gladstone", are old both-ways
relations whose ways are in no order: build_model.assemble makes 79 and 231 runs of them,
place_stations reads stops in run order, and each pair consecutive across two runs was traced
over the shortest track: sections Dover - Mount Tabor (past Denville), Mount Arlington -
Denville (past Dover), Denville - Convent Station 13.8 km, East Orange - Hoboken 16.2 km,
Secaucus - Hoboken, New York Penn - Hoboken. 246.8 km for ~140 km of route. The layout cannot
fix a graph like that. 13 US train relations are this broken (4+ runs holding stops): the NJT
Morristown, Gladstone, Main, North Jersey Coast, Montclair-Boonton, Bergen County, Raritan
Valley, Port Jervis; Metro-North Hudson, New Haven, Harlem; LIRR Port Jefferson; SEPTA
Paoli/Thorndale.

**Fix**: a new country hook, `route_runs`, in rules/us.py (written), calling
`us_register.repair_route_runs` (written): for a relation with REPAIR_MIN_RUNS (4) or more runs
holding stops, its own track as a graph; each track node to its nearest stop along the track;
stops joined where their regions touch; the cheapest joins that connect all stops (a spanning
tree); parts with no way between them joined at their nearest loose ends as a gap (build_model
traces those pairs as before); the tree walked as one run from one end of its longest path to
the other, out along each branch and back. On the M&E (stand-alone test): Morristown relation 79
runs -> 3, Gladstone 231 -> 5; sections are exactly the neighbouring stops (Denville - Dover
6.8, Denville - Mount Tabor 0.9, Summit - New Providence 2.7, Chatham - Summit 5.5, ...; Broad
Street - Hoboken and Broad Street - Secaucus across the relations' gaps).

**Needs build_model.py** (managing session): call the hook after assemble in `build()`'s
per-variant loop. A no-op unless a rules file defines `route_runs` (only us does).

```diff
@@ def build(region, log):  (the loop `for rid in rids:` after `def trace(na, nb)`)
         for rid in rids:
             tags, members = routes[rid]
             runs = assemble(members, ways, coords)
+            # A country's repair of a route relation whose ways are in no order (rules/<cc>.py
+            # route_runs; the US's old both-ways NJ Transit relations, 2026-10-08): its runs
+            # rebuilt from its own track and stops, or None to keep assemble's.
+            if runs and rules is not None and hasattr(rules, "route_runs"):
+                runs = rules.route_runs(rid, runs, members, ways, coords, station_nodes,
+                                        stations) or runs
             if not runs:
                 continue
```

and in `country_rules()`'s docstring, after SKIP_ROUTES:

```diff
+      route_runs(rid, runs, members, ways, coords, station_nodes, stations) -> runs or None
+            A route relation's runs rebuilt (assemble's [(node ids, lon/lat)]) where its
+            ways are in no order; None keeps assemble's. Called per variant before
+            travel_dirs and place_stations. Default: none.
```

The trial (below) patched exactly this in at run time (scratch `bm_trial.py`).

## Also: New Providence off the Morristown Line (us_register.NOT_ON)

The Morristown Line register line had New Providence between Chatham and Summit (sections
Chatham - New Providence 3.1, New Providence - Summit 2.3). It is a Gladstone Branch stop; the
Morristown Line passes 560 m off (inside STATION_M), and the broken Gladstone relation lists a
Morristown Line way (334724341) there, enough for ALONG_M. A NOT_ON row (NJT, MORRISTOWN LINE).

## Trial and results

Trial build of us with the hook patched in (scratch bm_trial.py, via tools/slot.py 2, ~10 min),
compared with ab.compare against dist/data/us: 19 lines differ (18 OSM + the Morristown Line),
0 station ids gone or new, no line id gone; foot.json and ways.json differ (the OSM lines'
ways). 23 relations repaired. us_sources.md "Route relations in no order" has the km per line.
Morris & Essex after: 36 sections, all neighbouring stops (Hackettstown ... Denville - Mount
Tabor ... Summit - Short Hills ... Broad Street, Summit - New Providence ... Gladstone, Broad
Street - Hoboken, Broad Street - Secaucus - New York Penn), 148.5 km.
`python -m unittest discover -s tests`: 27 OK.

Not done: the app rows of the rebuilt M&E diagram were not looked at in a browser (the browser
run was refused after the index.html edit); its sections form a plain tree, which layoutPiece
draws as a main lane with branches. No layoutPiece/branchSide change needed.

## To land (managing session)

1. The build_model.py diff above.
2. `python tools/compare_lines.py save us`, `python tools/slot.py 2 -- python tools/rebuild.py
   -j 1 us`, `python tools/compare_lines.py diff us`, `python check_model.py --region us`,
   then `tools/build_regions.py`. Canada needs no rebuild: rules/ca.py has no route_runs and
   ca_register swaps in its own NOT_ON.
3. `tools/line_100_probe.js` on us and ca (I could not run it: browser refused).

