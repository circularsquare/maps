# Drawing track across borders (noritetsu) — proposal

Written 2026-10-01 for the managing session (maps-ee) and Anita. Everything here was measured
read-only on the project outputs, or built in a scratch copy; no project file was touched.

## As built (2026-10-01, landed; project outputs not yet rebuilt)

Anita's answers to section 7: (1) border track counts in the country it lies in, a ride over
the border credits both; (2) yes to R2; (3) crossings RINF has no point for are added by hand
later, when the country on the other side is built (`borders.EXTRA`, empty now); (4) a border
point shows one neutral name, the same in every country.

Landed in the project, against build_model.py as of the Greece changes (the diff below
applied cleanly with offsets):

- `borders.py` and `border_points.json` (229 RINF border points, every one with two countries;
  83 that only one country files get the other as the nearest Natural Earth country within
  50 km, which names Röszke's point Hungary – Serbia and the Channel Tunnel portal France –
  United Kingdom). `python borders.py --fetch` regenerates it.
- `build_model.py`: `border_tails`, `add_border_sections`, `split_at_borders` (R2) and
  `name_border_points` as in section 5, plus, since the proposal:
  - every border point station, a RINF register's own junction included, is renamed to the
    neutral name ("Belgium – France border": Natural Earth names, alphabetical, en dash) just
    before the outputs are written, so gtfs_served sees the register's names as before;
  - `carry_aliases(..., abroad)`: stations R2 leaves to a neighbour count as live through
    their alias, so an older alias chain through one ends at the neighbour's id instead of
    being dropped (the first rebase lost be's n5695983384 -> Kleinbettingen this way);
  - the ru branch of looks_like_service (from the Russia agent; ru only).

Validated in scratch on be, nl, fr, cz, hu and lu against an unmodified build of the same
inputs (gtfs check live on be, cz and hu): register lines identical in all six; every
border point named the same in every country (92 checked); R2 aliases resolve to the
neighbour's ids. Per country, before -> after:

| | sections to a border point (km) | OSM line km, net | new lines | whole sections cut by R2 / far part dropped | stations no longer shipped, aliased to the neighbour | register km made creditable |
|---|---|---|---|---|---|---|
| be | 25 (1,122) | +412 | 8 | 4 / 4 | 2 (Baisieux -> fr, Kleinbettingen -> lu) | +136.4 |
| nl | 16 (328) | +324 | 1 | 1 / 1 | 1 (Essen -> be) | +32.4 |
| lu | 17 (102) | +32 | 5 | 0 / 0 | 0 | +16.8 |
| cz | 34 (418) | +338 | 5 | 7 / 2 (the rest end in Germany, unbuilt) | 1 (Gmünd NÖ -> at) | +160.9 |
| hu | 15 (84) | +67 | 2 | 4 / 4 | 2 (Loipersbach-Schattendorf, Mogersdorf -> at) | +27.6 |
| fr | 62 (1,127) | +658 | 12 | 3 / 2 | 2 (Erquelinnes -> be, La Plaine -> ch) | +98.4 |

Own track (the app's OWN reading of credits.json) barely moves: be -1, nl +12, cz, hu, fr,
lu within 2 km. Border sections that read as own track: the parts R2 keeps in an unbuilt
neighbour (cz: Schöna, Sebnitz, Schirnding, 2 km each, in Germany; fr: Kehl 0.5 km),
which were own track before too, as whole sections; fr R1/R11 Delle - border (0.4 km) and
TER 04 Menton-Garavan - border (1.0 km), where France's register stub does not reach the
point; nl 9200 Breda - border 32% and 9500 13%, be CFL-70 Athus - border 41%.

The rest of this file is the proposal as written before the answers.

Scratch folder (temporary): `C:\Users\anita\AppData\Local\Temp\claude\c--Users-anita-projects-maps\2b091358-b381-48f3-94de-72ef6687d688\scratchpad\border\`
(called `border\` below).

## 1. What is actually missing

TER K80 Kortrijk - Lille is one OSM route_master (m2998846) and is built in both Belgium and
France under that id. Belgium's extract has the stop at Mouscron but not Tourcoing; France's
has Tourcoing but not Mouscron. So be builds Kortrijk - Mouscron, fr builds Tourcoing - Lille,
and the section Mouscron - Tourcoing exists in neither. The app joins the two pieces into one
line (combineParts) but they share no station, so:

- the selected line has a hole between Mouscron and Tourcoing;
- a ride Kortrijk -> Lille finds **no path** at all (checked: `appsim.py base`), so it counts nothing;
- the register track from the last station to the border can never be completed: Belgium's
  L.75 Mouscron - Mouscron-Frontière (3.0 km) and France's Tourcoing - Bif. Tourcoing-Frontière
  (2.1 km) are credited by no section in their own builds.

**The background track is not the problem.** Geofabrik extracts overlap at a border, so both
be.pmtiles and fr.pmtiles carry the crossing ways (K80: be has 47 of its 113 ways, fr 64,
five at the border in both). The gap is in the lines, the rides and the percentages.

## 2. Size of the problem (all built countries)

Method (`border\measure2.py`, `analyze2.py`): every route relation a built country reads
(11,098), its stops in order, each labelled with the country it lies in (Natural Earth 10m),
every consecutive stop pair in two countries checked against every built lines.json. Lengths
are along the route's own ways over the union of all extracts on disk. Results in
`border\m2\crossings.json` (one row per crossing), `gaps.json`, `whole.json`.

| | crossings with no section | km, station to station | of which lines rather than named trains only | with a RINF border point on the track |
|---|---|---|---|---|
| both sides built | 73 | 3,794 | 49 crossings, 1,505 km | 71 |
| one side built | 95 | 2,473 on the built side | 78 crossings, 1,968 km | 48 |
| neither (ru, by...) | 2 | 91 | | 0 |

km count each stop pair once, but a corridor carrying several named trains with different
stops is counted per pair (Bruxelles-Midi to Paris Nord, to CDG and to Lille Europe all run
over HSL 1). The unique-track measure is the register one below.

Both sides built, by pair: at-ch 5 (54 km), at-cz 3 (161), at-hu 5 (478), at-si 3 (272),
at-sk 1 (6), be-fr 7 (1,147), be-lu 3 (41), be-nl 7 (420), bg-ro 1 (16), ch-fr 15 (508),
cz-pl 5 (115), cz-sk 4 (142), ee-lv 1 (159), fr-lu 3 (54), hr-hu 1 (11), hr-si 2 (41),
hu-ro 1 (13), hu-sk 4 (91), lt-lv 1 (48), lt-pl 1 (18). Examples: K80 Mouscron - Tourcoing
5.1 km; Eurostar/TGV Bruxelles-Midi - Paris Nord 312 km (89 in be, 223 in fr) and - Lille
Europe 107 km; LIMAX Visé - Eijsden 4.0 km; Railjet Wien Hbf - Břeclav 89 km; Ex7 Summerau -
České Budějovice 64 km; Rīga-Valga 159 km; Joniškis - Jelgava 48 km; Léman Express
Chêne-Bourg - Annemasse 3.3 km.

One side built, biggest: fr-uk 5 (Eurostar Paris/Lille/Calais - London and Le Shuttle,
1,438 km on the French side; no RINF point), at-de 4 (121), pl-ua 4 (210), de-nl 9 (153),
cz-de 10 (103), ch-de 9 (66, **no RINF point at any**: Basel Bad Bf, Schaffhausen, the
Basel trams and S-Bahn), ch-it 7 (58), fr-it 5 (46), de-pl 11 (45), be-de 2 (41), fr-mc 2.

**Register track at borders that nothing can credit** (`border\stubs2.py`, project
dist/data): 269 register sections end at a border, 1,074 km. 687 km of that can be credited
by no ride on any line; 126 sections (610 km) have no credit at all. The worst: be L.1
Halle - Esplechin-Frontière, which is all of HSL 1 (74.5 km, 0%); be L.3 Chênée -
Hammerbrücke (35.8 km, 0%); be L.4 HSL 4 (16.2 km, 0%); nl HSL Zuid's last 10.6 km (0%);
fr LGV Nord's last section (98.6 km, 90%). Per country, km nothing can credit: at 49, be 136,
bg 25, ch 19, cz 45, ee 2, fr 96, gr 6, hr 12, hu 41, lt 38, lu 17, lv 33, nl 41, pl 34, pt 13,
ro 25, si 27, sk 29.

**Sections already built whole over a border** (one extract happens to hold both stations):
about 45 stop pairs between two built countries, 265 km (coarse: the 10m outlines put some
border stations on the wrong side, e.g. Chiasso, Narva, Barcs, so the true number is lower).
Some are built by both countries under different station ids (ch-fr, cz-sk, cz-pl, pl-sk, hr-si).

Also measured, and the reason for the approach below: **country outlines are far too coarse
to cut at.** Over 164 RINF border points, Natural Earth 10m and religiondots' shapes put the
border a median 600 m away (p90 1.4 km, max 5.5 km). religiondots' shapes also have no
Luxembourg, Monaco or North Korea.

## 3. The approach: each country builds its own side, to a shared border point

### Considered

(a) **Each country keeps the section from its last station to the border point** (chosen).
(b) A separate border pass after both builds, making the whole station-to-station section
from both extracts: needs both countries built (nothing for de, it, es, uk...), must re-run
after every rebuild of either side because section ids are renumbered each build, needs a new
file type the app loads per country pair, and has to decide which country's percentage a
whole section counts in.
(c) Each country builds the whole section from its own and its neighbour's extract: neither
extract has the far station, so it needs the neighbour's data and the neighbour's station
ids; both copies then exist, overlap, and the app keeps whichever country loaded first, so
per-country totals would depend on load order.
(d) Cut at the country outline: the outlines are 600 m off (above), and the two builds must
cut at exactly the same point or the pieces do not join.

### How (a) works

**The border point table.** ERA RINF types every border point (op-type 90) and gives it one
uopid shared by both countries: Mouscron-Frontière is EU00084 in Belgium's RINF and in France's.
They lie on the track: median 2 m from OSM's rails, 90% within 9 m, over 415 measured
crossings. RINF countries already name their register junction there `"e" + uopid`, and be and
nl both ship `eEU00089` (Meer-Grens / HSL Breda grens) today. `borders.py --fetch` pulls all
229 of them (two SPARQL queries, seconds) into a tracked `border_points.json`, with the
countries whose sections end at each point (144 of 227 have both). A short hand list (`EXTRA`)
can add crossings RINF lacks.

**R1, the tail** (`build_model.border_tails`). For each route variant, at its first and last
station placed in this country: if the route lists stops beyond that end which the extract
does not have at all, follow the route's own track on from the station to the first border
point within 60 m of it, and keep that piece as a section `station -> eEU000xx`. The neighbour
does the same from its side and ends at the same id, so the app's existing join makes one
connected line with nothing new in the app. A line whose only station in this country is
its last before the border (Paris - Amsterdam has only Paris Nord in France) becomes a line
for the first time. No border point on the track: nothing is added, and the build log says
so ("no border point"), which is how every crossing is today.

The tails are kept aside in build() and added after merge_sources (`add_border_sections`), so
the merge's twin and station decisions see exactly what they see today. A line that the merge
folds into a twin hands its tails over.

**R2, sections already built whole** (`split_at_borders`). A section whose two stations lie on
two sides of a border point, one a station of this country's register and the other not, is
cut at the point. The far part is dropped when that country is built (dist/regions.json), so it
is built once, by the country it is in; while the neighbour is unbuilt it stays, so nothing
drawn today disappears. A far station this build stops shipping is aliased in aliases.json to
the neighbour's station of the same name within 500 m (Baisieux: be's bare stop node
n663600155 -> fr87286872), so a saved ride naming it still resolves.

### Against the questions in the brief

- **Crediting and whose percentage.** Rides attach to track: each country's piece is
  credited in its own build against its own register, and register lines already stop at the
  border point, so border track counts in the percentage of the country it is in. A ride
  Kortrijk -> Lille credits L.75's last 3.0 km in Belgium and Ligne de Fives à Mouscron's
  first 2.1 km in France. OSM lines and named trains are not counted in a register country
  today; if they are later (the 2026-10-01 decision), the same split applies, since each
  country's piece is the part on its side.
- **Stable ids.** Line ids do not change (OSM's). R1 changes no existing section or station
  id; it only adds junction stations `eEU000xx` (RINF's own ids, which nobody can board at, so
  no ride names one) and sections. R2 removes a few far stations from a country (4 in the
  prototype), each aliased to the neighbour's id. carry_aliases no longer maps a vanished id
  onto a junction.
- **Countries loaded separately.** Each country's output is self-contained and its border point
  is a station record in its own stations.json, which the app already merges when two countries
  ship the same id. Section ids stay per country (`be:`, `fr:`).
- **Only one side built.** R1 draws this side to the border and makes its register stub
  creditable now (be: Hergenrath-Frontière for the ICE and S41 to Aachen; Visé, Gouvy). The other
  side arrives with the neighbour's first build, with no rebuild of this side for R1.
- **Double drawing.** R1 halves meet at one point and never overlap. Overlap remains only where
  a whole section was already built, which R2 removes once the neighbour is built.

## 4. Prototype: be, nl and fr

Copies of the shared modules and of be/nl/fr's proc data in `border\proto\`; outputs in
`border\base\` (unchanged code, same inputs), `border\r1\` (tails only) and `border\fin\`
(tails and R2, the proposed code). Compared with `compare.py base fin be nl fr` and
`appsim.py`.

**Unchanged:** every register line in all three (name, sections, km, closed): 145, 96 and 278
lines, 0 differ. Aliases: no existing alias changed or removed.

| | be | nl | fr |
|---|---|---|---|
| sections to a border point added | 24 (1,094 km) | 16 (328 km) | 62 (1,127 km) |
| OSM lines with new sections | 17 | 13 | 52 |
| new lines (only stop here is the last before the border) | 8 (TGVs and Eurostars from Bruxelles-Midi) | 1 (RE13 Venlo) | 12 (Eurostar Paris - Amsterdam / Dortmund, Léman Express L4, RE33, R1, R11, RB52, RB53...) |
| whole sections cut by R2 | 4 (IC-19 and TER P81 Froyennes - Baisieux; L-12 and CFL-50 Arlon - Kleinbettingen) | 1 (S32 Roosendaal - Essen) | 3 (S63 Maubeuge - Erquelinnes, RS4 Krimmeri - Kehl, 151 Pougny-Chancy - La Plaine) |
| stations no longer shipped (aliased) | 2 (Baisieux, Kleinbettingen) | 1 (Essen) | 1 (Erquelinnes) |
| new junction stations | 1 | 0 | 28 (France's register has no RINF ids) |
| register km made creditable | +136.4 | +32.4 | +98.4 |

Register track at the borders of these three that nothing can credit: **273 km -> 19.5 km**.
In be all of it is now creditable: L.1 (HSL 1) 0% -> 100%, L.3 0 -> 100, L.4 0 -> 100, L.75
28 -> 100, L.37, L.40, L.42, L.96, L.12 all 0 -> 100.

**The app's join** (`appsim.py`, the same rule as combineParts): line ids shared by be, nl
and fr go from 13 to 19, and those that join into one connected piece from **0 to 16**.

- K80 Kortrijk -> Lille: no path before; now 29.7 km via Mouscron [be], Mouscron-Frontière
  [be], Tourcoing [fr]... Lille Flandres [fr].
- IC-19 Namur -> Lille Flandres: 149.3 km via Froyennes, Blandain-Frontière, Baisieux [fr].
- Eurostar m5189989 Amsterdam -> CDG: 503 km over three countries, via Meer-Grens and
  Esplechin-Frontière.

The 3 still in two pieces (IC-04, IC-26, L-29) are not border crossings: France's extract
carries short Belgian sections inside its buffer under OSM ids (Wervik - Comines, a doubled
Herseaux). Across all countries there are about 30 such sections, 108 km
(`foreign_only.py`); a separate cleanup.

Build time: within noise (be 48 -> 50 s, nl 48 -> 48 s, fr 361 -> 327 s).

## 5. Shared-file changes

1. **New `borders.py`** (`border\proposed\borders.py`): fetch, load, `Index.along` (border points
   on a polyline), `name_for`. **New tracked `border_points.json`** (`border\proposed\border_points.json`,
   229 points, written by `python borders.py --fetch`).
2. **`build_model.py`**: `border\proposed\build_model.diff` (against the current file, which was
   still identical when checked); full file at `border\proposed\build_model.py`. In short:
   - `border_tails()` new; build() calls it per variant and keeps lines with one station when a
     tail exists; tails ride on the line as `_tails`, border-only lines in `build.border_only`;
     a log line counts them and lists route ends near a border that met no point.
   - `add_border_sections()` and `split_at_borders()` new, called in main() right after
     merge_sources.
   - merge_osm_twins hands a dropped twin's `_tails` to the kept line.
   - carry_aliases never carries onto a junction; main() adds the far-station aliases after it.
   - Not wired into the no-register path (no country uses it now).
3. **No change** to extract.py, build_tiles.py, rinf.py, the app, or tools/build_regions.py
   (shared_lines picks the new line ids up by itself). Rebuild tiles after the model as usual:
   ways.json gains way ids outside the extract (7,530 in be, 8 of them inside it), which the
   tiles ignore.
4. **Workflow**: after a country's first build, rebuild its built neighbours too, so R2 hands
   that country its side. Worth one line in HANDOFF's "After any shared-file change".
5. **Optional, France**: fr_register's own border junctions sit a few metres to 1 km from the
   RINF points, so some French stubs reach only 80-99% (Longwy 83%, Thionville - Apach 80%,
   Morteau 94%, Modane 99%) and those lines cannot show finished. Snapping fr_register's
   "Frontière" junctions onto the RINF point (and its `eEU` id) fixes it; Switzerland's
   register likely the same.

Roll-out: land 1-2, rebuild every register country (`compare_lines.py save`, `rebuild.py`,
`compare_lines.py diff`; register lines should come out identical everywhere, as in the
prototype), then `build_regions.py`.

## 6. Risks

- **Only measured on be, nl and fr.** The GTFS hook (cz, hu, and the countries the GTFS agent is
  rolling out) was not exercised: no feed for these three. Rebuild cz and hu first and diff.
- **A border point within 60 m of the wrong track**: parallel lines at a border. 60 m keeps out
  the Kehl tram 110 m from the rail bridge; 13 of 415 measured points lie 60-145 m off their own
  track (Český Těšín, Vrbovce, Görlitz), so those crossings will log "no border point".
- **A tail follows only its own run**: a relation gap right at the border loses it (logged).
- **R2 reads dist/regions.json and the neighbour's stations.json**, so its output depends on what
  else is built: a rebuild order dependency (hence item 4 above).
- **Twin hand-over** can attach a twin's tail to a line that does not run there in the next
  country: in be, the European Sleeper was already merged into the Eurostar (existing behaviour),
  so the Eurostar now also runs Antwerpen - Essen-Grens, one-sided.
- **Small coverage shifts**: one French register section lost 0.8 km of creditable length
  (Lyon-Perrache - Genève, Pougny-Chancy, 27% -> 23%) where R2 cut the line 151 in two.
- **Strip diagram**: border points are `j` rows; a country loaded first names it (be's
  "Mouscron-Frontière", fr's "Frontière FR - BE (Tourcoing - Mouscron)").
- Crossings RINF has no point for stay undrawn: all of ch-de, the Basel and Geneva trams,
  Strasbourg - Kehl tram, the Channel Tunnel, Monaco, every non-EU border (cn-hk, cn-ru, pl-ua
  partly, fi/ee/lv/lt-ru). Each needs a hand-added row.

## 7. Decisions for Anita

1. **Whose percentage does border track count in?** Proposed: the country it lies in (each
   side's register already ends at the border point), with a ride over the border crediting both.
   The alternative is the whole station-to-station section counting in one country (the one
   departed from, or both).
2. **Cut sections already built whole over a border, and give the far part to the neighbour
   once it is built (R2)?** It removes overlaps and makes both registers creditable, at the
   price of a rebuild of the neighbours after each first build and 4 stations moving to the
   neighbour's id (aliased). Without it, those ~45 crossings stay as they are.
3. **Crossings RINF has no point for**: add them by hand (Basel x9, Swiss and Geneva trams,
   Strasbourg - Kehl, Channel Tunnel, later the non-EU borders), or leave them undrawn?
4. **Name shown for a border point**: RINF's name in whichever country loads first, or one
   neutral name such as "Belgium - France border".

## Files

- `border\PROPOSAL.md` (this file)
- `border\proposed\borders.py`, `border\proposed\border_points.json`,
  `border\proposed\build_model.diff`, `border\proposed\build_model.py`
- Measurement: `border\measure2.py`, `border\analyze2.py`, `border\m2\crossings.json`,
  `border\stubs2.py`, `border\stubs2_dist.json`, `border\accuracy.py`, `border\foreign_only.py`
- Prototype: `border\proto\` (scratch copy), `border\base\`, `border\r1\`, `border\fin\`
  (outputs and build logs), `border\compare.py`, `border\appsim.py`, `border\run_builds.py`
