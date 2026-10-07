# noritetsu — spec

A global map of passenger rail lines, on which you record the sections you have ridden, and
which tells you what fraction of each line, operator, region and country you have completed.
Plus a poster generator that draws your own ridden network at every scale at once.

To live at anita.garden/noritetsu. 乗り鉄 is the general Japanese subculture term for the
railfan whose thing is riding the trains, as against 撮り鉄 who photograph them. It names the
audience rather than the activity, and no one service owns the word.

`noritsubushi` (乗りつぶし, riding every line to completion) describes the activity better and
was considered first, but noritsubushi.org (乗りつぶしオンライン) is the established Japanese
service that already does exactly this for Japan, so that name would read as a clone of theirs.

Written 2026-09-29. Later sections reverse earlier ones where they disagree.

## 1. Scope

In:

- Every passenger rail line on earth, drawn, pannable, LOD'd enough to feel smooth on a phone.
- Recording ridden sections, several input routes, each with an optional date.
- Completion percentages by line, operator, region, country.
- A generated poster: global map cropped to what you have ridden, plus city and country insets.
- Line colours taken from the system's own colours (2 train in New York is red).

Out, for now:

- Scheduled trains per day. That is `../traincounts/` (Finland GTFS pilot) and stays there.
  Frequencies come back in phase 5 as a line *detail*, not as the map's subject.
- Accounts, sync, social features. Rides live in the browser and export to a file.
- **Freight-only track, yards, sidings and industrial spurs. Not drawn at all** (revised
  2026-09-29; an earlier draft of this file kept them as faint background). The map is
  passenger rail. See §12a for how that is decided, which is not as simple as a tag.

## 2. Prior art

- **noritsubushi.org** — Japan only. Per-section records between any two stations, riding
  kilometres and completion rate, and a maintained line database that tracks openings, closures
  and relocations. The closest thing to what this should be, and the model for the section unit.
- **viaduct.world** — global ride tracking. Not yet examined; do that before finalising the UI.
- **Tetsulog** — iOS, Japan, 163 operators and 571 lines. Useful as a sanity check on how many
  lines a national inventory should contain.

## 3. The three hard problems

1. **A global line inventory with ordered stations does not exist as a dataset.** OSM route
   relations are the only global source. Coverage is excellent for metros and trams, good for
   national rail in Europe, Japan and Korea, thin for heavy rail in the USA and China.
2. **Drawing the whole world's rail at phone speed.** Roughly 1.3 million km of track.
3. **Attributing a ride to a unit that makes percentages honest** without double counting track
   that several lines share.

## 4. Data model

All derived from OpenStreetMap. Four objects.

### edge

A piece of physical track between two topological junctions. Geometry, `length_m`, `kind`
(rail, light_rail, subway, tram, monorail, funicular, narrow_gauge), `usage`, `service`, gauge,
electrification, maxspeed, operators. Id derived from the OSM way id plus a split index.

This is the **unique-track denominator**. "% of Japan's network" is ridden edge length over
total edge length inside the boundary, counting each edge once however many lines use it.

### station

A served stopping place, one record per real station. Merged from `railway=station|halt|
tram_stop`, `public_transport=stop_position|station` and station areas, with platforms collapsed
onto the parent. Name, `name:en`, position, and the set of lines serving it.

### line

A thing a rider would call a line. From OSM `route=train|subway|light_rail|tram|monorail|
funicular` relations. Carries name, ref, colour, operator, network, kind, an ordered station
list, and the ordered edge list for each inter-station section.

**Deduplicating relations into lines is the crux of the project.** OSM normally has one relation
per direction, plus short-turn and branch variants. Grouping is by `route_master` where one
exists — the mapper's own statement that these variants are one line, and it covers 82% of
Japan's route relations — and by (operator, network, ref, name) for the orphans.

Two findings from building Japan, both of which cost a rebuild to learn:

- **Sections must be cut at the finest granularity available, not per variant.** A
  limited-stop variant lists only its calling points, so its consecutive pairs are 30 km
  apart where the local variant's are 3 km. Unioning both counts the shared track twice, at
  two granularities: the Tokaido Main Line came out at 5,719 km against a published 590. The
  fix is to pool the line's stations over every variant first, then walk each variant's
  **path** past all of them — an express physically passes the local stations, so it yields
  the fine sections too.
- **OSM does not split lines the way operators do.** The Chuo Main Line is officially one
  424.6 km line; in OSM it is two route_masters, split at Shiojiri where JR East hands over
  to JR Central. Neither is wrong. The unit this project counts is OSM's line, and the
  completion percentages are against that.

Validated with `python check_model.py --region jp`, which compares built lengths against
published operating lengths (営業キロ). Every line checked is within 2%.

Where no route relation exists, **synthesize** a line by chaining `railway=rail` ways that share
`name`, `operator` and `usage=main`. Mark these `synthetic: true` and say so in the UI. A
synthetic line has no authored colour and no reliable ref.

### Lines that are not lines

OSM in Japan maps **named trains** as route relations too, with no tag separating them from
lines: のぞみ is an object of the same type as 東海道本線 and runs over it. 144 of Japan's 714
lines are these, and they account for 23,213 km of the 49,539 built — the Shinkansen corridors
counted once as lines and again under every service over them. Without them the total is
**26,326 route-km against a real network of about 27,300**, which is the number to trust.

They are flagged `service: true`, by a crude and region-specific rule (a Japanese line name
ends in 線, a train's does not), and nothing is dropped. Which of them a percentage counts is
a judgement about what the hobby means, and is open:

- **Operating patterns** are a third category the flag does not catch. 京浜東北線 and
  中央線快速 are named with 線 and are lines to any rider, but are not lines in JR's register;
  they run over 東北本線 and 中央本線. noritsubushi.org counts the register.

Settled 2026-10-01 (Anita): everything with scheduled passenger service counts, and named
trains get no percentage of their own (option B). Operating patterns count only for track they
own, which is track no register line covers (one owner per piece of track, §6); an OSM line at
least 40% off the register is listed as a line of its own (metros, trams, private railways
outside the register).

### section

The atomic ridden unit: a line plus a consecutive pair of stations it serves. Has `length_m` and
an ordered edge list. A ride is a contiguous run of sections on one line.

## 5. Rides

```json
{"line": "<line id>", "from": "<station id>", "to": "<station id>", "date": "2026-09-29", "note": ""}
```

`date` and `note` optional, on every input path. A ride names a line and two stations because
that is what the rider did; what it *counts* for is worked out from the track (§6).

**Browser storage and file export first, accounts later, and account-optional at the end**
(decided 2026-09-29). Rides live in localStorage and export to / import from a JSON file. The
file format is the durable thing: accounts have to be able to read exactly this, so it wants a
stable ride id and a timestamp from the start even though nothing uses them yet. Accounts do
not need to be elaborate; what they need is to be optional.

## 6. Percentages, and what a ride credits

**A ride is against track, not against the line label it was entered under** (decided
2026-09-29). You record "I rode the Yamanote from Tokyo to Tabata"; what is counted is the
corridor you sat on, so the stretch north of Tabata — which the register calls the Tohoku Main
Line, not the Yamanote Line — completes part of that too.

**Completion is reported for lines.** Operating patterns and named trains stay as things you
can look up and enter a trip against, because a rider looking up the Yamanote wants the whole
loop, not the 20.6 km the register calls the Yamanote Line. Named trains are not the things
whose percentage is being kept; operating patterns count only for their own track (§4,
settled 2026-10-01).

### How crediting works

**Since 2026-10-01: one owner per piece of track** (Anita: "no double counting, every piece of
track belongs to exactly one line"; for crediting only, rides are still entered on any service
and path). Every piece of track (an OSM way) belongs to exactly one line: a register line where
one lies on it (assigned by geometry), else one OSM line by the lowest ref; named trains own
nothing. Riding a section credits its footprint, the stretches of owner sections its own ways
lie on, found by node ids, not a buffer (`ownership.py`, `foot.json`). A ride credits the track
pair its route uses, a junction crossing credits each side's line, and a through service
credits the railway it runs onto. `ownership_prototype/WRITEUP.md` has the comparison with the
buffer model below, which it replaced.

*Superseded, kept for the record:* section A credited section B over the fraction of B that
lay inside a 45 m buffer around A, stored as a **range** `[start, end]` in `credits.json`. So:

- two lines on the same rails credit each other in full;
- one express section spanning three local ones credits all three;
- **and three local sections together complete the one express section**, which whole-section
  crediting could not express and which is the usual shape of the problem, since register
  lines are almost always coarser than the service you actually rode.

Guarded by `kind`: Tokyo has metro tunnels directly under JR lines and they are not each other
however close they run.

Rail fingerprinting — the exact set of OSM node pairs a section runs over — was tried first and
is not enough on its own. The Yamanote loop runs on its own pair of tracks, so it shares
**0.0 of its 34.5 km** with anything by that test. Geometry is what sees the corridor. Measured:
riding the whole Yamanote credits 12.7 km of the Tokaido Main Line, 15.5 km of the
Keihin-Tohoku, and some part of 46 lines in total.

### The numbers to show

- **per line** — the share of its footprint ridden.
- **per operator or network** and **per country or region** — owned track ridden over owned
  track, each piece once by construction. A named train has no total of its own; its track
  counts through the lines that own it (Anita, 2026-10-01).

## 7. Rendering

- A pmtiles archive with layers `track` (z0–14) and `station` (z8–14), written with the
  LineString extension to `../religiondots/mvt.py`.
- LOD ladder: z0–4 main line only, hard simplified, metro systems collapsed to a point;
  z5–8 main and branch; z9+ everything passenger; z12+ service and yard track.
  Per-zoom Douglas-Peucker at about one pixel.
- **Ridden track is a separate source drawn on top**, built client-side from the ride list plus
  per-line geometry fetched on demand, coloured by the line's own colour. Not feature-state on
  the base tiles: the ridden set is small and wants its own width and colour rules.
- If z0–5 is still too heavy on a phone, pre-render the faint network as raster tiles for those
  zooms and switch to vector above. Decide that by measuring on a phone, not in advance.

## 8. Colours

Ridden lines take `colour` from the route relation where it exists (65% of Japan's lines do).
Otherwise a fixed per-operator palette, otherwise a per-kind default. Authored and stable,
never generated per view.

**The base network has two palettes, one per mode** (decided 2026-09-29):

- **All lines** — full saturation. The network is the subject, so it is drawn properly.
- **Ridden** — much darker and nearly grey. Your own lines go on top in real system colours and
  cannot win against a base that is competing with them.

Not one compromise palette in between, which is too dull to browse and too loud to read your
own lines against. The legend is drawn from whichever palette is live, so it always shows what
is actually on the screen.

## 9. Trip input

Every mode ends in the same confirm step and takes an optional date.

1. **Station to station, routed.** Search an origin, search a destination, get a suggested route,
   confirm. Dijkstra over the section graph with a line-change penalty. Each leg becomes a ride.
2. **Click a station.** Station bubbles are on the map; clicking one lists the lines serving it;
   picking one opens a noritsubushi-style line strip diagram; click from and to.
3. **Search a line by name.** Same strip diagram as mode 2.
4. Others worth having: click successive stations to draw a path by hand; "rode the whole line"
   in one click; mark a whole operator or system at once. A GPX or ticket import is a maybe.

## 10. Poster

Cluster ridden geometry by scale, choose insets automatically — a city inset per metro area with
intracity rides, a country or region inset where several lines are ridden, plus the global map —
and lay them on one sheet so every scale is legible at once. Print output follows `../posters/`
conventions (300 DPI, sRGB embedded).

## 11. Phases

1. **Japan vertical slice.** Extract, model, tiles, map, input modes 2 and 3, percentages by
   line, operator and prefecture. Japan because the OSM rail data is excellent, the hobby is
   Japanese, and `../riders/japanriders/` already has station and operator reference data.
2. Mode 1 router, dates, export and import.
3. Global scale-out. **Unblocked 2026-09-30: 73 GB free, where the earlier postponement was
   written against 23 GB.** The 90 GB figure was always total DOWNLOAD, not peak disk: the
   build holds one region's `.pbf` at a time and the largest single-country extract is about
   5 GB, so the ceiling was never really the issue and is now comfortable either way. The
   pipeline is region-agnostic; see §12c for where each country's line register comes from.
4. Poster generator.
5. Line detail: type, frequency, gauge, electrification. Frequencies lean on `../traincounts/`.

## 12a. Deciding what is passenger track

OpenStreetMap has no passenger flag, and the two available signals each fail on their own.
Measured on Japan (`python inspect_region.py --region jp`):

| track | km on a passenger route relation |
|---|---|
| main line | 86.4% |
| branch | 68.5% |
| industrial | 12.2% |
| yard, siding | 6.9% |

So filtering on route-relation membership alone would delete a seventh of the main network —
freight-shared main lines, and double track where only one direction made it into the
relation. Filtering on `usage` and `service` tags alone would keep every goods yard.

**The rule: keep track whose usage is main or branch, keep anything a passenger route relation
runs over whatever its tags say, drop everything else.** The second half is what keeps a
station's own approach tracks and the sidings a service actually calls at.

Tiles carry `p`, recording which of the two rules let a piece of track in. `p=0` is not drawn
differently — a colour on the map has to be in the legend, and "OSM has no route relation
here" is a fact about the data, not about the railway — but clicking the line says so.

Expect this rule to need revisiting per region: `usage` is well kept in Japan and Europe and
thin elsewhere, and where it is absent everything falls to branch.

## 12. Sources and constraints

- **OpenStreetMap**, ODbL. Attribution required, and a published derived database has to be
  offered under ODbL too. Both fine, but the map needs the attribution line and the download,
  if one is offered, needs the licence stated.
- Extraction is Geofabrik regional `.osm.pbf` read with pyosmium (see §13 for what is left) (`pip install --user osmium`,
  4.3.1 on the system Python 3.9). GDAL's OSM driver is not in this build of fiona, and it loses
  relation member order anyway, which is exactly what the line model needs.
- **Disk: 73 GB free on C: as of 2026-09-30** (it was 23 GB when this file first said disk
  was the binding constraint, and that is no longer true). Work one region at a time anyway,
  deleting each `.pbf` before fetching the next: the transient peak is then one country-level
  extract, about 5 GB at most. Roughly 90 GB of **download** in total for global coverage,
  for a few hundred MB of output — 1.3M km of track at 50 m vertex spacing is about 26M
  vertices, which delta-encodes small. Japan's 2.5 GB extract yields a 12 MB tile archive, so
  the world is plausibly 300-700 MB.

## 12b. Where OpenStreetMap has trains but no line

**Found 2026-09-30 and not yet resolved.** The San'in Main Line is 673 km of JR West main
line. OSM has **no route relation for it at all**, and its track is not named either: 27 ways
carry the name, 7 km in total. What does run over that track is the limited expresses. So the
only thing calling at Matsue is スーパーおき, which is a named train, which completion excludes.

Measured with `python check_model.py --region jp --coverage`:

- **220 stations** are served only by a named train and never by a line
- **17,847 km** of track is reached only by a named train and so counts towards nothing

This is not a bug in the model; it is what the source says. The Tokyo-centric checks in
`check_model.py` all pass because Tokyo is mapped densely. Options, none taken yet:

1. **Count a named train where nothing else covers that track.** Small change, and it makes
   Matsue countable, but under the label "Super Oki", which is not a line anyone completes.
2. **Synthesise a line from track connectivity** where no relation covers it, named by its
   endpoints. This is §4's synthetic line, and it is the honest fix, but an unnamed line
   needs a name from somewhere: Wikidata has the Japanese line register.
3. Leave the gap and say so in the UI, so a rider knows the tool cannot see that line.

## 12c. Getting lines from the register, not only from OpenStreetMap

Researched 2026-09-30, after §12b showed 17,847 km reachable only by a named train.

**OSM is not a line register and never claimed to be.** It maps what someone chose to map. In
Tokyo that is everything; along the San'in coast it is the limited expresses and nothing else.
No amount of work on our side recovers a line that is not in the source.

### What the other tools do

- **noritsubushi.org** maintains its own line database by hand, and keeps it current with
  openings and closures. That is why it can count 営業キロ against the register.
- **viaduct.world** sidesteps lines entirely. It logs journeys from **operator timetables**
  (most of Europe, plus the USA, Canada, Australia, India) against a station database of
  100,000+ stations **linked to Wikidata**, and uses OSM only for the basemap. It answers
  "which trains have I been on", not "how much of this line have I finished".

Neither is a source we can copy. But both point the same way: the line inventory has to come
from somewhere authoritative, per country.

### Japan: 国土数値情報 N02 is exactly it

MLIT's own railway dataset. Downloaded and inspected (`python probe_n02.py --zip
data/raw/N02-24_GML.zip`):

- `N02-24_RailroadSection.geojson`, 14.3 MB, **21,932 track sections**, each carrying
  `N02_003` 路線名 and `N02_004` 運営会社. **553 distinct line names**, 179 operators.
- `N02-24_Station.geojson`, 3.3 MB, **10,235 station records**, with line name, operator,
  station name, and **`N02_005g`, a station group code** — the register's own statement of
  which platforms are one station complex, which is what this project currently guesses at
  with name-plus-distance.
- The whole archive is **12.7 MB**, at a stable URL:
  `https://nlftp.mlit.go.jp/ksj/gml/data/N02/N02-24/N02-24_GML.zip`
- Public Data Licence 1.0, commercial use permitted, attribution required.

山陰線 has 342 sections and 161 stations in it. The line that does not exist in OSM is simply
there. 553 register lines also sits right next to Tetsulog's 571 and our own 594 non-service
OSM lines, which is a good sign that all three are counting roughly the same thing.

### The shape of the fix

**N02 becomes the spine, OSM stays for what N02 has not got.** Neither alone is enough:

| | N02 | OSM |
|---|---|---|
| every line in the country | yes | no |
| official line name and operator | yes | patchy |
| station complex identity | yes, `N02_005g` | guessed |
| line colours | no | 65% of lines |
| operating patterns (京浜東北線) | no, not register lines | yes |
| named trains (のぞみ) | no | yes |
| the rest of the world | no | yes |

So: build lines and stations from N02, keep the OSM relations matched onto them for colour,
and keep operating patterns and named trains as OSM-only objects that credit N02 lines through
the existing spatial mechanism. The credit machinery does not change at all; only where the
lines come from.

### Built 2026-09-30

`python build_model.py --region jp --n02 data/raw/N02-24_GML.zip`, and it is now the default
way to build Japan. `n02.py` does the register half.

- **593 register lines, 28,156 km**, against a real passenger network of about 27,300.
- **9,048 register station groups**; 8,919 of the 11,100 OSM stations matched one by name and
  distance and were folded onto it, so both sources share one station registry.
- 293 OSM lines matched a register line and handed over their colour and English name. They
  are **kept, not dropped**: the OSM object is usually the line as OPERATED where the register
  line is the line as REGISTERED, and the Yamanote runs a 34.5 km loop over a 20.6 km
  register line. Dropping it lost the loop, which is the thing a rider looks up.
- **Matsue now has 山陰線 on it**, which was the whole point.

Two things had to be got right, both found by checking against published lengths:

- **The line's track is a graph, not a chain.** Merging a double-track line end to end gives a
  run that goes out on one track and back on the other, so slicing between two stations
  traverses both: the San'in Line came out at 1,330 km against 674, the Ou Line at four times
  its length. A shortest path over a vertex graph uses one track and ignores the parallel one.
- **Sections come from an absorbing search, not from an ordering.** Ordering stations by
  distance from a terminus cannot work on a loop, because there is no terminus: the Oedo Line
  came out at 178 km against 40.7. A Dijkstra from each station that stops the moment it
  reaches another station gives the inter-station sections directly, whatever shape the line
  is.

What is still off, and is a property of N02 rather than a bug: **the register labels freight
branches with the main line's name**, so JR trunk lines come out 5% to 20% over their
published operating length (山陽線 1.10, 奥羽線 1.10, 東海道線 1.12). Metro lines, where there
are no freight branches, land on 1.01 to 1.03. 山手線 is the worst at 1.44, because N02 gives
the name to more track than the 20.6 km register line.

### Elsewhere

- **Europe**: ERA's Register of Infrastructure (RINF) is a SPARQL knowledge graph of sections
  of line and operational points, per member state. Infrastructure-shaped rather than
  rider-shaped, but authoritative. `bergmannjg/RInfData` on GitHub already queries it.
- **Globally**: Wikidata is the glue rather than the geometry. `?x wdt:P31/wdt:P279* wd:Q728937`
  gives railway lines, with P137 operator, P17 country, and **P402 the OSM relation id**,
  which is what lets an OSM relation be recognised as a particular named line rather than
  guessed at from its tags.
- Expect to do this country by country. There is no global line register.

## 13. Todo

This project's own list. The repo root `todo` is Anita's and is not for this.

**Done** (detail in §4-§12c, and in the comments of the file that does each job): OSM
extraction, tiles with an LOD ladder and a passenger-only filter, the viewer, the line model,
spatial crediting, the register build from N02, and the tracker — record a ride from a strip
diagram or from the map, ridden track drawn in each line's colour, percentages per line,
operator and overall, localStorage with file export and id migration across rebuilds.
Station-to-station routing was considered and dropped (§9).

Added 2026-09-30 in `dist/index.html`: the picked run lit on the diagram and on the map, from
one circled station to the other with the stops between as white rings; undo (Ctrl+Z, one
step per action, in memory only) and editing a ride's date and note; "the other way round"
between two stations where the line has a second way (stored as `around: true`, since
section ids do not survive a rebuild); an operator view with "rode every line"; and tracing a
journey by clicking or searching its stations in order, each leg resolved to one line.
Later the same day: the rides list shows TRIPS (rides saved back to back, each starting where
the last ended, same date, added within 30 minutes), and opening one draws it on the map and
brings it into view without leaving the list. A click on track now searches 10 px around
itself for the nearest line, and below z10, where the tiles carry no way id and clicks
used to do nothing, it resolves the line from the model's station-to-station runs.
Countries load as the map pans over them (`dist/regions.json`, from `tools/build_regions.py`),
merged into one model. The strip diagram draws a line's whole track graph (`lineLayout`):
a main route down lane 0 that prefers stopping at stations over the shortest path, every
other route that leaves and rejoins or dead-ends in a lane beside it with its stations kept
together as one block and a thin line back to where it rejoins, a loop at either end of
the main route walked in one column with only its closing section drawn back (the Oedo
Line reads in the order its trains run), shortcuts (express track beside a stopping route,
duplicate sections, the Oedo's Shinjuku to Shinjuku-nishiguchi past Tochomae) set aside
before layout (a ride over one lights the way beside it), other track
with no station of its own as thin arcs in one shared outer lane, loops with caps, separate
pieces with a gap. 959 of 995 register lines are one straight lane; the JR East Tohoku Line
needs the most, 8.

### Next, roughly in order

**1. Wrong in things that get used daily.**

- [x] **No heavy-rail ride on an OSM line credited any register line.** Fixed 2026-09-30:
      `build_credits` compares `kind_family`, so `train` and `rail` are one. 山陰線 is now
      credited by 17 OSM routes, 山手線 by 27. That at first let every Shinkansen service
      credit the conventional line beside it, about 1,600 km of false credit over 16
      services, so high-speed and conventional sections no longer credit each other
      (`section_highspeed`); 108 km on the Ou Line remains and is real (Tsubasa, Komachi).
      Heavy-rail credit pairs went from 0 to about 26,000 in Japan
- [x] **Clicking the San'in Line on the map selected the Super Oki.** Build half done
      2026-09-30: `build_model.register_way_lines` puts register lines into `ways.json` by
      geometry, nearest line wins, a way's own name tag settles ties, and high-speed track
      only matches high-speed lines. 843 of the Super Oki's 1,061 ways now also offer 山陰線;
      3 of 593 register lines remain unclickable (Rumoi Line, closed 2026; Nagoya guideway
      bus; 1.4 km of Sapporo tram)
- [x] App half, done 2026-09-30: `pickTrack` opens the register line directly when it is the
      only register line among the hits, and lists the other lines on that track under it
- [x] **Line colours on the base map** (asked 2026-09-30, reversing §8's kind-only palette):
      done 2026-09-30, `c` per track feature from `build_tiles.way_colours`, which reads
      build_model's output, so tiles are now built after the model
- [x] **"Tokyo Metro Ginza Line" beside "(as operated)" was noise.** Done 2026-09-30:
      `merge_sources` drops an OSM line within 6% of its register line's length and sharing
      85% of its stations. 148 dropped in Japan; the Yamanote loop and every other line that
      really differs stays. Their ids go to `aliases.json` under `lines`, and the app moves
      saved rides through it

**2. The two features from the original ask that do not exist yet.**

- [ ] Poster generator: the ridden network cropped to itself, with city and country insets
      chosen automatically so every scale is legible at once (§10)
- [ ] Line detail: type, gauge, electrification, frequency. Frequency is what
      `../traincounts/` was for

**3. Tracker gaps.**

- [x] On a phone the open panel covers the whole map, so picking on the map, and tracing a
      journey by clicking, only work through search there. Done 2026-10-04: a bottom sheet
      over half the screen, the map above it (HANDOFF thread 10)
- [ ] Undo lasts only as long as the page. Enough for a slip; not a history
- [ ] Where a line's far end is itself a loop, the diagram walks the loop in one column but
      cannot know which station of it is the terminus: the Chuo Line ends at Shiojiri, and
      the diagram reads Midoriko, Shiojiri, Ono, ... Okaya. The register's own terminus, if
      the build passed it on, would let it start the loop there
- [x] N02 puts Tochomae 152 m from the Oedo loop track, so the build made a Shinjuku to
      Shinjuku-nishiguchi section that does not call there. Fixed 2026-09-30 with the
      station footprints in §4: the Oedo Line is 40.5 km against 40.7
- [x] The JR East Tokaido Line's diagram was tangled from Tokyo to Tsurumi (Omori not joined
      to Kamata, Hazawa between Musashi-Kosugi and Shin-Kawasaki). Fixed 2026-09-30: the
      data half by the §4 footprints, and the app's `looksStraight` guess now runs only on
      lines with `straight_sections`; on this line it had removed the real, straight
      Musashi-Kosugi to Shin-Kawasaki section and sent the main column down the non-stop
      Sotetsu-through track. The line now reads Shinagawa, Oimachi ... Kawasaki, Tsurumi, with
      the Yokosuka route as a block beside it and Hazawa as a spur. Osaka reads Shin-Osaka,
      Osaka, Tsukamoto with Fukushima as the Umekita spur

**4. Data quality, in descending size of error.**

- [x] JR trunk lines measured 5-20% over their published length (山陽線 1.10, 奥羽線 1.10,
      東海道線 1.12, 山手線 1.44), put down to N02 naming freight branches after the main
      line. It was mostly `n02.py` instead: a station claimed only the one track vertex
      nearest its centre, so a multi-track line's other tracks slid past it and gave sections
      that skip stations (田町-大井町 through 品川, 東京-品川 past three). Fixed 2026-09-30: a
      station claims every track within 80 m of its platform, and a section that doubles back
      at a junction (大井町 out towards 北品川 and back down the 大崎 branch to 西大井) is
      dropped where the line's other sections still join its stations; 38 were. The register
      went from 28,156 to 27,122 km and every checked line is now 0.99-1.01, except JR
      Central's 東海道線 at 1.06, which is its 美濃赤坂 branch and the 垂井 bypass, both real
- [x] The register still lists lines that no longer run, and OSM has no rails there, so the
      map drew nothing where the line's panel listed stations: the 日田彦山線 from 添田 to 夜明
      (BRT since 2023), 留萌線 石狩沼田-深川 (closed 2026), the suspended 美祢線, 肥薩線
      八代-吉松 and the 津軽線's far end, and the Nagoya guideway bus. **Kept, greyed as not
      running** (Anita, 2026-09-30): `not_running.py` flags a register section with drawn
      track beside under half of it (`closed` on the line; 45 sections, 154 km in Japan, 18 in
      Switzerland), and the app draws those dashed grey on the map and the strip, clickable,
      and leaves them out of the line's length and every percentage. A ride from before the
      closure can still be entered on them
- [x] 471 sections fell back to a straight line where an OSM route relation has a gap. Fixed
      2026-09-30: the gap is traced along track (`TrackGraph`), the line's own ways first and
      then the network with a detour cap, and cut at any of the line's stations it passes.
      20 remain in Japan, 14 of them named trains, where OSM's track itself does not join
- [x] The app's `looksStraight` (over 2 km, within 1% of the crow-flies distance) flagged
      mostly real track. Since 2026-09-30 it only guesses on a line whose `straight_sections`
      is over 0 (12 lines in Japan, none of them register lines), which is precise enough
- [x] Track that touches no line was drawn and could not be clicked: tourist monorails,
      amusement-park railways, a roller coaster, pedal trolleys, harbour freight lines. Since
      2026-09-30 `build_tiles.drop_islands` leaves out a connected piece of track when no line
      runs over any of it; 635 ways in Japan. An OSM passenger route over it does not save it:
      Disneyland's railway and the harbour freight lines are route=train with no stops, the same
      as the Swiss funiculars the model was missing (now given lines by `funicular_ends`)
- [x] A register line in pieces, or with a dead end another line joins to another dead end of
      it, says which line joins them ("to Niiya by 内子線" on the 予讃線), since the register
      splits lines by legal name and the diagram otherwise read as a line with a hole in it
- [x] **Light theme** (2026-09-30, so Anita can see it before deciding whether to switch to
      it): the Light/Dark button, or `?theme=`. Positron basemap with its labels repainted a
      faint grey and its city dots faded, no country names (in either theme) and paler
      province and country boundaries, darker kind colours, station dots white with a
      black ring. Page colours are CSS variables; `THEMES` in index.html holds the ones CSS
      cannot reach. A line too close to the background gets a thin edge (dark round bright
      lines in light, light round near-black ones in dark; `CASED_*`), drawn narrower over
      it so it weighs the same
- [ ] **Duplicate lines**, done 2026-09-30 as far as they could be found: register matches
      that failed only on operator spelling or station spelling (+17: the Hanzomon,
      Yurakucho and Toei Shinjuku lines), and OSM lines that are each other
      (`merge_osm_twins`, 40 in Japan). Still unmerged by design:
      a through service and the line it runs over when their lengths differ by more than
      6%, as the 98 km Hanzomon-Den-en-toshi-Skytree service does
- [ ] **23 OSM lines have a `display` order that is not a line.** It goes out and back
      (城端線 runs 城端 to 高岡 to 城端; the 阪神なんば線 repeats 9 stations), or ends where it
      started with only a straight-line gap joining the two (西鉄天神大牟田線). The diagram
      now keeps each station's first visit and draws a loop only where a real section closes
      it, but the order should come out right from `build_model`
- [ ] Credit ranges stop a few tens of metres short of a station, where two parallel lines
      part to reach their own platforms: riding the Marunouchi as operated credited 97.3% of
      the register line's Ginza to Kasumigaseki section. The app now closes gaps under 150 m
      (capped at 10% of the section) in `creditSpans`; the build could do it at the source.
      Kept in the app under track ownership (`mergeSpans`)
- [x] The 45 m corridor buffer was a guess (a Marunouchi ride credited 0.4 km of the Oedo
      line). Gone 2026-10-01: one owner per piece of track, crediting by the ways a service
      runs on (§6)
- [ ] A jointly operated line counts as one combined operator ("A · B") rather than towards
      each of them

**5. Names, which are a convenience and not a correctness problem.**

**The native name is the real name.** Someone riding trains in a country reads the names on
the trains, and those are not in English; an English name is a nicety on top, not the thing
the app is for (decided 2026-09-30). `lineName()` already falls back to the native name and
should keep doing so — do not "fix" a Japanese line name into an id or a transliteration.

- [ ] 61% of register lines have no English name (232 of 593 do), and 70% of register
      operators have none (54 of 178 do). Wikidata is the source:
      `?x wdt:P31/wdt:P279* wd:Q728937` with P17 for the country gives English labels to
      match on the native name plus operator (§12c)
- [ ] Only 30% of register lines have a colour, because a colour only arrives when an OSM
      line matched. **Wikidata done 2026-09-30** (`line_colours.py`, applied in
      `build_model.main` after the merge, marked `colour_src: "wikidata"`): Japan 183 -> 338
      of 593 register lines (a dry run; lands on the next jp build), Korea 18 -> 33 of 82.
      Many Wikidata colours are Japanese Wikipedia infobox picks, 54 of Japan's 155 a CSS
      named colour (008000, FF0000), not the operator's. Korea's intercity lines have no line
      colours at all; Wikidata's Korail blue is skipped as generic.
      **Tables done 2026-09-30** (`colours/<region>.csv`, which win over OSM and Wikidata):
      `kr.csv` from the widely circulated Korea network map, `jp.csv` 105 register lines in
      the JR companies' own colours, read from the vector fills of their route-map PDFs
      (JR West, Kyushu, Shikoku, Hokkaido, Central, and JR East's Tokyo area). A line whose
      sections have different official colours and none covers 40% has an empty colour and
      its sections listed in the note (山陰線, 山陽線, 函館線, 東北線...), for a per-section
      pass later. Still open: JR East outside Tokyo (the South Tohoku map
      jreast.co.jp/map/pdf/minamitohoku.pdf returns 403 to scripts), and 八高線's grey,
      which may be that map's neutral default rather than a line colour

**6. Reach.**

- [x] **Switzerland**, 2026-09-30: `schienennetz.py` reads the BAV network register plus
      the national service-point list. 402 register lines, 5,567 km; built length against
      the register's own chainage has a median of 0.996, and five lines checked against
      Wikipedia are within 2%. In the app at `?region=ch`
- [x] Junction-ended sections were judged on ALL drawn track within 40 m, so a yard line
      beside a main line passed on the main line's trains. Now only track assigned to the
      line itself counts (`register_way_lines`); 26 more yard, depot and harbour lines went
      (Limmattal RBL, Lausanne-Triage, Kleinhüningen). "Basel SBB - Basel GB - Basel RB"
      stays, correctly: its kept 1.9 km ends at Basel St. Jakob, the stadium stop
- [ ] Swiss register sections with no OSM track within 700 m (found 2026-09-30 by maps-9f's
      not-running check): Sonceboz-Sombeval - Moutier - Delémont (Tavannes-Sonceboz, 7 of 8
      sections), Ruchfeld - Aesch BL Dorf (Reinach, tram 11 under rebuilding), and single
      sections elsewhere. The register geometry is sound (no jumps, starts at its stations);
      the gap is on OSM's side. Tavannes-Sonceboz not yet explained: Overpass was down and the
      ch extract is deleted; check whether that stretch is railway=construction in OSM
- [ ] Track under construction is not extracted (`railway=construction`), so a line closed
      for rebuilding shows its stops with no track between them: BLT Tram 11 through Reinach,
      closed as of 2026. Correct as data, odd on the map; it returns when OSM does
- [ ] Swiss km-lines are infrastructure axes, not what a rider calls a line: Zurich's trams
      come as 0.1-3 km pieces named after a stop ("Basel, Aeschenplatz"), and there are
      dozens of sub-kilometre connecting curves. They are real track and count, but they
      clutter a per-line list
- [ ] Where a km-line has two routes between neighbouring stops, only the shorter becomes a
      section (the absorbing search). 8 km in Switzerland outside the Gotthard's second bore
- [x] **South Korea**, 2026-09-30: `kr_register.py` takes OSM's named track as the geometry
      register (98% of main-line km carries its legal line's name) and the published station
      lists read by `kr_sources.py` (Korail's 거리표, KRIC 1294) for which stations are on
      which line. 82 register lines, 4,644 km; against the register's own km per section a
      median of 0.997, and every metro and nearly every Korail line within 3% of its
      published length. In the app at `?region=kr`. `HISTORY.md` "Korea" has the detail
- [ ] Korea: a register line whose named track stops short of its first listed station loses
      that section: 호남고속선 오송-공주 (0.76 of its length), 영동선's 영주 and 강릉 ends,
      대구선, 광주선, 수서평택고속선's 지제 end. Trace such a station over the wider network
      to the line's own track
- [ ] Korea: register lines have no English names yet (stations do, from RAFIS and KRIC)
- [x] **Hong Kong and Singapore**, 2026-09-30, on Taiwan's pattern: 13 and 10 register lines,
      293 and 269 km, station lists from MTR's DATA.GOV.HK files and LTA DataMall. Worst
      register lines 0.91 (Peak Tram, measured along its slope) and 1.09 (Light Rail, whose
      one-way street pairs count twice). `hk_sources.md`, `sg_sources.md`
- [ ] Hong Kong: 機場快綫 is listed twice (OSM's and the register's), because Airport has no
      station node and its stop lands on the people mover's Terminal 2 station; the tram and
      Peak Tram OSM relations fail the 6% length test. Light Rail's one-way pairs should be one
      section each way, not two sections
- [x] **Belgium, Austria and the Netherlands**, 2026-09-30, from `rinf.py`, the generic ERA
      RINF reader (§12c "Elsewhere"): 145, 125 and 96 register lines, 3,196, 4,342 and
      2,780 km, median 0.996-1.000 against RINF's own section lengths. Metros, trams and
      private railways RINF lacks stay OSM lines
- [ ] Netherlands: Breda - Rotterdam builds 17% over its published length and nobody has
      worked out why; Dutch line names are ProRail's end points ("Lelystad Opstelterrein
      Aansl. - Zwolle" for the Hanzelijn); Enschede De Eschmarke is cut off from Enschede in OSM
- [ ] Austria: each private line's stub to its ÖBB junction (0.5-3.5 km) is lost, because
      those RINF points have no coordinates
- [x] **France**, 2026-09-30: `fr_register.py`, 278 register lines, 24,227 km, chainage
      median 0.996
- [x] **Czechia, Poland, Hungary, Portugal**, 2026-09-30, from `rinf.py` with a settings file
      each in `rinf_countries/`: 236, 352, 133 and 24 register lines (9,101, 16,171, 6,721 and
      2,138 km). HISTORY.md "Europe from ERA RINF" has the hooks they needed
- [x] **Mainland China**, 2026-09-30: `cn_register.py`, 417 register lines, 121,894 km, from
      OSM's named track and 12306's station list
- [ ] China: named track stopping short of a terminus loses that stretch (宝成线 at 广汉北,
      青藏线 from 湟源, 京广线 from 房山东); 成昆线 and 京港高速线 are each two pieces;
      stations are placed by proximity, so parallel lines can share one wrongly; 瓦日线 and
      唐包线 are freight lines that pass the passenger-station test
- [x] **Completion counts everything with scheduled passenger service** (Anita, 2026-10-01).
      Done the same day (option B): named trains no percentage of their own; OSM lines count
      for their own track, each piece once (§4, §6, HANDOFF)
- [x] **Draw track across borders**: done 2026-10-01 (`borders.py`, `border_tails`,
      `split_at_borders`; HISTORY.md "what maps-ee did")
- [x] RINF countries: where OSM has no passenger route relation, a junction-ended section was
      dropped even when trains run; where OSM still maps a closed line as railway=rail, it
      stayed drawn. Settled by the timetable check (`gtfs_served.py`), live in 16 countries
- [x] **Croatia, Greece, Luxembourg** (RINF) and **Russia** (the tariff guide, one line per
      tariff section), 2026-10-01. Russia's 2022-annexed railways are left out, assigned to no
      country, until a source says which trains run there (Anita); Crimea is built
- [ ] France: the Grande Ceinture carries RER C, T12 and T13 but a register line has one kind,
      so T12/T13 rides did not credit it. Track ownership no longer matches by kind: check
      after the 2026-10-01 rebuild
- [ ] Austria: the Vienna S-Bahn Stammstrecke through Wien Mitte has no railway=rail in the
      extract (probably mapped as construction), and the Mattersburger Bahn stops 416 m short
      of Wiener Neustadt; both leave sections out
- [ ] Singapore: "LRT Sengkang Line" is listed beside the register line (OSM names its stop
      positions "Sengkang - East Loop Anticlockwise"); the KTM shuttle's 1 km in Singapore is
      left out until Malaysia is built
- [ ] Regions beyond Japan, Switzerland and Korea. Per-country registers are in §12c, and
      `HANDOFF.md` has the contract a new region's reader has to meet
- [ ] Everything Japan-specific has to be replaced per region: `check_model.KNOWN` is
      Japanese lines. `REGISTER` has Switzerland and Korea too. The named-train rule has a
      Korean branch (a train brand in the name: KTX, SRT, ITX, 새마을, 무궁화...; 22 flagged).
      In Switzerland every OSM route relation is a service pattern (IC 1, S12), and none is
      flagged a named train
- [ ] Station areas mapped as ways rather than nodes are dropped by the extractor. Fine for
      Japan, likely not elsewhere. `public_transport=stop_area` relations are not read either
      and would be better evidence than name-plus-distance where there is no register

**7. Housekeeping.**

- [x] `build_tiles.py` wrote a `station` layer that nothing read. Dropped 2026-09-30
- [ ] Whether a named train should be searchable at all, or only reachable through the lines
      it runs over

**8. Long term** (not now; Anita, 2026-10-01).

- [ ] Russia: a Yandex Rasp API key (free, needs a Yandex account; per-station queries,
      restrictive terms) would say which stations are passenger stops, instead of relying on
      OSM's train routes. Skipped for now
- [ ] Russia's 2022-annexed railways: add back, with Russia (de facto), once a source says
      which trains run there (`ANNEX_RUNNING` in rinf_countries/ru.py, `EXTRA_AREAS` in
      tools/build_regions.py)
