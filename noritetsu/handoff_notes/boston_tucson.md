# Boston's East Subdivision and Tucson's Lordsburg Subdivision (2026-10-09)

Anita's two strip diagrams of 2026-10-09, diagnosed by the US agent. Data fixes are in
us_register.py (landed, us and ca rebuilt). Two shared changes wait on other sessions: a
three-line build_model.py change (managing session) and four hunks of dist/index.html (app
session). Both diffs are below. They were tested on a private copy, `dist/index_us.html`,
deleted afterwards.

## What was wrong

**Boston, East Subdivision.** The diagram opened Newmarket (Fairmount Line), junction "East /
Fairmount / Old Colony", junction "near South Station", South Station, junction "near South
Station", then the Red Line's Downtown Crossing and Broadway, Newmarket on CR-Fairmount and
JFK/UMass on Kingston/Plymouth, and only then Back Bay. Four causes:

1. **Data: the Old Colony's approach filed as East.** NARN files South Station's approach to the
   Old Colony (segments 379801, 380038, and the unnamed 374683 that joined East as the line it
   touches; 1.1 km, coded C, commuter only) under SUBDIV "EAST". So the East Subdivision had a
   1.1 km branch off its Back Bay approach that only Old Colony and Fairmount trains use, and
   the diagram laid it out first.
2. **Data: South Station stood 5 m before NARN's end of track.** That made a 0.005 km section to
   a dead-end junction, "near South Station".
3. **App: the hub fallback.** At that 5 m end, the walk found nothing. `nearRows` then offered
   every section passing within 300 m. `hubStop` (South Station) put them all in `byHub`.
   throughEnds then put them all back, because "rows" was empty ("unless that leaves this
   line's own track to the junction with no way to ride it"). But a 5 m stub is a `lineTails`
   tail: any ride reaching South Station already credits it.
4. **App: `nearRows` ignores kind.** The walk (`departures`) only steps onto track of the same
   kind. `nearRows`, the last resort, took any line, so the Red Line (subway) was offered off a
   commuter railroad.

Also found: **the Old Colony Line was `kind: subway`.** `register_way_lines` re-kinds a register
line from the OSM track it lies on, and from JFK/UMass to Braintree the Red Line runs beside
it. So it matched only subway ways (it owned Red Line track), and the same-kind walk from
East's end could not reach JFK/UMass on it. In NARN countries the same rule also re-kinded NS's
Amtrak Connection (Cleveland) to light rail and CPKC's Canpa (Toronto) to subway. All three
are wrong: every NARN line us_register keeps is a railroad's (PASSNGR R is left out).

**Tucson, Lordsburg Subdivision.** The diagram read Tucson (on Gila Subdivision), Gila /
Lordsburg, end of Lordsburg, "Lordsburg / Lordsburg Subdivision (second track)", Tucson (on the
Sunset Limited), Benson... Tucson itself is right: it is 6.4 km past the subdivision boundary,
on the Gila. The second Tucson was a data fault. Between Vail and Benson, UP's second main
runs on its own alignment, 0.1-1.1 km from the first. NARN codes its first 13.1 km as
passenger (so it was a section of Lordsburg, ending at a junction) and the rest not (the holes
file brought that in as "Lordsburg Subdivision (second track)", folded as a companion, then
pruned whole as leading nowhere). So Lordsburg had a 13.1 km stub whose end lay 48 m from its
own main track. The walk never steps back onto the line itself, so `nearRows` offered the
Sunset Limited's Tucson - Benson section. Benson was left out as the line's own stop, but
Tucson was not.

## Data fixes (us_register.py, rules/us.py, rules/ca.py; landed and rebuilt)

- `SEGMENT_NAME` reads 379801, 380038 and 374683 as "(MBTA) OLD COLONY". East is now South
  Station - Back Bay - ... - Attleboro, plus its 0.2 km to the Franklin junction at Readville.
  South Station is now the Old Colony's own end stop: South Station - JFK/UMass is 3.71 km, its
  traced track sharing South Station's last 0.45 km of throat with East. The Fairmount Line
  still ends at node 495012.
- `TWIN_STUB_KM` (50 m): a dead-end section that short past a stop, at a node no other line
  meets and not by a border, is left out in build(), and the stop ends the line. 20 in the US
  (South Station on both East and the Old Colony, Rockport, Newburyport, Needham Heights,
  Gladstone, Elburn, Seward, South Bend Airport, Downtown Carrollton, ...) and 5 in Canada
  (Waterfront, Saint-Jérôme, LaSalle, Arnaud Junction).
- `fold_second_track_stubs`: a line's stop-less stub from one of its branch points to a dead end
  where its own folded "(second track)" carries on is moved into that companion, so ownership
  gives its ways to the line (as for every second track). Four in the US: Lordsburg 13.1 km,
  Pittsburgh Line 0.3, Gallup 0.15, Cajon 0.12. Canada has no holes-file second tracks.
- `REGISTER_KIND_SURE = True` in rules/us.py and rules/ca.py. This does nothing until
  build_model reads it (next section).

## build_model.py (managing session): keep a NARN line's kind

```diff
@@ def register_way_lines(region, lines, geoms, log):
     rekinded = []
+    # A country whose register knows its lines' kind (rules REGISTER_KIND_SURE: NARN's are all
+    # railroads) keeps it: the track test called the MBTA's Old Colony Line "subway" from the
+    # Red Line running beside it.
+    kind_sure = bool(getattr(country_rules(region), "REGISTER_KIND_SURE", False))
     for l in lines:
         if l.get("src") == "osm":
             continue
@@
-        if (fam in ("rail", "tram") and top != fam
+        if (fam in ("rail", "tram") and top != fam and not kind_sure
                 and by_fam[top] > 0.5 * sum(by_fam.values())):
```

Also add to `country_rules`' docstring:

```
      REGISTER_KIND_SURE = True
            Register lines keep the register's kind: register_way_lines never re-kinds them
            from the OSM track they lie on (NARN: the Old Colony beside the Red Line came out
            "subway"). Default: off.
```

Only us and ca set it. Trial (the change patched in at run time, `scratchpad/bt/bm_kind.py`,
together with the register changes): **0 register lines re-kinded** (was 2 in us, 1 in ca).
Old Colony Line `rail`. NS's Amtrak Connection (Cleveland, 0.44 km) now ships (it had been
dropped as light rail). Otherwise the same lines move as in the data-only build; foot.json and
ways.json differ (the Old Colony stops owning Red Line track). After landing: `python tools/slot.py 2 --
python tools/rebuild.py -j 1 us ca`.

## dist/index.html (app session): two changes in throughEnds / nearRows

Against index.html as of the morning of 2026-10-09. These hunks do not touch anything the app
session has changed since.

```diff
@@ -6977,6 +6977,7 @@ function wayRun(row, atEnd) {
    stop is not offered: the ride is entered on that line from the hub. Further out (the Port
    Washington Branch's end before Woodside) the stub is a real stretch, and stays. */
 const HUB_THROAT_KM = 1.5;
+const HUB_TWIN_KM = 0.1;          // a junction this near its hub is the hub's end of track (throughEnds)
 function hubStop(line, j) {
@@ -7200,8 +7201,14 @@ function throughEnds(line) {
       rows.push({ ...r, j, key: r.legs.map(l => l.line.id).join('>') });
     }
     /* Ridden from the hub instead, unless that leaves this line's own track to the junction
-       with no way to ride it (the Seaford Branch Line's 1.1 km beside the Seaford Single). */
-    if (!rows.length && !offered.size)
+       with no way to ride it (the Seaford Branch Line's 1.1 km beside the Seaford Single). A
+       junction within HUB_TWIN_KM of the hub, on a stub that comes with it (lineTails: any
+       ride reaching the hub credits it), is the hub's own end of track, and nothing is
+       offered past it (Anita, 2026-10-09: South Station's end 5 m on offered the Red Line,
+       Newmarket and JFK/UMass). */
+    const twin = hub && (lineTails(line).get(hub) || []).includes(last[3])
+      && (pathBetween(line, hub, j) || { km: Infinity }).km <= HUB_TWIN_KM;
+    if (!rows.length && !offered.size && !twin)
       for (const r of byHub) rows.push({ ...r, j, key: r.legs.map(l => l.line.id).join('>') });
     const out1 = r => r.legs[0].line.id;
@@ -7216,7 +7223,8 @@ function throughEnds(line) {
 /* THE LAST RESORT AT A JUNCTION END: the build keeps a junction end only where other track
    comes within CONTACT_M (300 m, build_model.prune_dead_track), but the walk follows only
    register lines and meets them within JUNCTION_SNAP_KM. Where it finds nothing, the stops
-   at either end of any line's section passing within NEAR_ROW_KM are offered, ridden on that
+   at either end of any line's section of the same kind (named trains included) passing within
+   NEAR_ROW_KM are offered, ridden on that
    line from the junction: at Lincoln the Hastings Subdivision ends 104 m from the next
@@ -7227,7 +7235,9 @@ function nearRows(line, j, own, missing) {
   const kx = Math.cos(J.y * Math.PI / 180) * 111.32, ky = 110.57;
   const best = new Map();
   for (const { line: M, sec } of sectionsNear(J.x, J.y)) {
-    if (M === line) continue;
+    // The same kind of track only, as the walk (departures): a metro under a railway's
+    // terminus is no way on from it (Anita, 2026-10-09: the Red Line past South Station).
+    if (M === line || kindOf(M) !== kindOf(line)) continue;
     const [a, b, km, gid] = sec;
```

`node tools/lint_map_expressions.js` passes on the patched copy.

### What it changes: the stops past every junction end, before and after

The test is the same page with and without the hunks, on the data as shipped. A headless
survey takes every listed register line with a junction end and records contRows at each
end (every line's geometry is loaded first). It is compared line by line. Run-to-run noise,
from the same page surveyed twice: de 3 lines (round BER), gb 12, others 0.

| | lines changed | rows of another kind gone | hub-twin rows gone | rows that come in |
|---|---|---|---|---|
| us | 12 | 20 | 8 | 4 |
| ca | 6 | 10 | 2 | 0 |
| de | 34 (3 noise) | 22 | 13 | 31 |
| ch | 17 | 21 | 7 | 2 |
| gb | 15 (12 noise) | 6 | (noise) | (noise) |
| jp, ru | 2 (ru) | 4 | 0 | 1 |

- Kind rows gone: subway and light-rail stops off railway lines (the Red Line at South Station;
  LA Metro A and D at Union Station; DART at Dallas and Carrollton; TRAX at Salt Lake;
  Montréal's REM and Orange Line; Toronto's Line 2; Vancouver's Expo Line; Edmonton's Capital
  Line; the NYC Subway's E/F at Sunnyside and J at 121st Street; Edinburgh Trams at Waverley;
  Metrolink at Navigation Road; the Central line at Greenford; Moscow's Line 7). And the
  reverse: S-Bahn and IR stops off Zürich, Bern and Basel tram lines, rail stops off Berlin
  S-Bahn lines.
- "Rows that come in" are stops of the same kind that the other-kind rows had pushed out of
  the list: Victory on the TRE at Dallas, S2's Berg am Laim instead of Tram 19's, Haymarket
  and Brunstane at Waverley.
- Lost: two genuine through runs that only `nearRows` had been offering. AVG's S4 tram-train
  at "Heilbronn Übergang DB/AVG" (de 4950) now offers Nordheim and Leingarten instead.
  Hamburg's S3/S5 at Neugraben comes back as S5's Fischbek. The walk already refused both.
- Ends left with nothing because their only rows were of another kind: de 6, ch 9 (mostly tram
  termini that had offered S-Bahn stations), us 4 (two of them South Station's 5 m stub and the
  mis-kinded Old Colony, both gone with the data fixes), ca 4.

**The 100% probe** (`tools/line_100_probe.js`, us and ca, shipped data). Part 1: us 563 -> 562
pass, ca 93 -> 92. Groups (part 3): 280 -> 276 read 100% when all their lines do; groupsWhole
294 = 294. The two lines are data gaps the kind rule uncovers:

- us `u1638c7c68a` LIRR Main Line (Greenport – Woodside): the end "near 121st Street" offered
  only the J train.
- ca `ue8a442e465` CPKC Westmount Subdivision: the end "near Lucien-L'Allier" offered only the
  Orange Line. OSM's exo1 and exo4 routes both end at Vendôme, and Lucien-L'Allier is in the
  model only as the metro station, so the Westmount's last 3.6 km past Vendôme reach no
  train stop.

### Not proposed: the hub fallback for every stub the hub credits

Dropping the `byHub` fallback wherever the junction's stub is a `lineTails` tail of the hub (no
HUB_TWIN_KM) would be the consistent rule: tails came in on 2026-10-07, after the fallback
was written, and a tail needs no ride past it. But it ends a lot of diagrams at their last
station: **de 496 lines (728 rows), ch 77, us 37, ca 10**. Most are German lines ending at a
"W 46"-style junction 0.2-1 km past a station, the Gießen case Anita asked for on 2026-10-05.
By stub length, de/ch: up to 0.1 km 10 lines, up to 0.2 km 61, up to 0.5 km 301, up to 1 km
185, longer 70. That is Anita's call (app behaviour everywhere), so only the 0.1 km version
is in the diff.

## Rebuild (2026-10-09)

`python tools/slot.py 2 -- python tools/rebuild.py -j 1 us ca`: models **and tiles** for both,
18 min, no failed steps (so the us and ca tiles carry the new build_tiles.py).

- compare_lines: **us 500 -> 500 register lines, 23 differ**. East 56.08 -> 54.54, Old Colony
  16.66 -> 18.23, Lordsburg 506.57 -> 493.45, Lordsburg (second track) new at 13.12 (a hidden
  companion), the 20 station-twinned stubs (-0.03 to -0.05 km each), Gallup / Pittsburgh /
  Cajon -0.12 to -0.30 into their second tracks. The Pittsburgh Line (second track) piece
  changed id (3.27 -> 3.58 km). The MBTA's 0.19 km South Side Subdivision at Greenbush is
  gone. **ca 131 -> 131, 4 differ** (Adirondack, Cascade, Parc, Wacouna, -0.01 to -0.07 km).
- check_model us: chainage median 0.996; Wikipedia table unchanged. Two lines are off by over
  5%. Austin Subdivision (second track) 0.67 was already so. Lordsburg Subdivision (second
  track) 0.63: its `km_official` still counts the 7.75 km of hole track that
  prune_dead_track drops afterwards, and that function does not recompute it. Cosmetic: the
  line owns nothing. check_model ca: median 0.995, none off.
- The 100% probe on the rebuilt data with the current page: us 562 of 565 pass (was 563 of 566;
  the 1 is the South Side stub line), the same 3 failing; e2e 374 (was 373). ca 93 of 97,
  unchanged. Groups 280, groupsWhole 294, both unchanged.
- The strip diagrams with the current page: East reads South Station, Back Bay, Forest Hills,
  Hyde Park, Readville, (Endicott on the Franklin Line past the junction), Route 128 ...
  Attleboro, South Attleboro. Lordsburg reads Tucson (on the Gila Subdivision) once, then
  Benson, Lordsburg, Deming, El Paso.

**Until the build_model change lands**, the Fairmount Line's north end (now mid-section on the
Old Colony, which is still "subway") offers JFK/UMass on the Greenbush train, the wrong way:
`nearRows` has no turn test. With the Old Colony set to rail in the page (a run-time test), it
offers Boston South Station on the Old Colony Line, as it should (plus Back Bay over CSX's
0.6 km East Subdivision connection near Tufts, a register line this did not change). So the
three-line build_model change and a us rebuild should follow soon.
