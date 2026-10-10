# Handoff

The file a managing session or a country agent reads first (rewritten short on 2026-10-02).
`HISTORY.md` has how each country was built, the declined proposals with their numbers, the
traps that cost a rebuild and the session logs. `spec.md` is the design (§13 the todo),
`README.md` how to run things.

## Start here

**72 regions are built** (Vietnam, Argentina, Chile, New Zealand, South Africa, Brazil,
Iran, Morocco, Algeria, Tunisia, Egypt, Belarus, Moldova, the five Central Asian ones and the
three Caucasian ones added overnight 2026-10-03/04; Ireland, Türkiye, Ukraine, the six
Balkan ones, Malaysia, Thailand, Mexico and Indonesia late on 2026-10-03; Denmark, Norway, India and the UK
earlier that day; the USA, Canada, Sweden and Australia 2026-10-02), all in
`dist/regions.json` and `dist/data/`. Register
lines and km as shipped (`src != "osm"`, closed sections included; de, it, es, ru as rebuilt
2026-10-03):

| cc | register from | lines | km | notes in |
|---|---|---|---|---|
| jp | N02 (`n02.py`) | 593 | 27,122 | spec, HISTORY |
| ch | Schienennetz (`schienennetz.py`) | 402 | 5,567 | HISTORY |
| fr | SNCF Réseau RFN (`fr_register.py`) | 278 | 24,227 | `fr_sources.md` |
| kr | OSM named track + Korail/KRIC lists (`kr_register.py`) | 85 | 4,957 | `kr_sources.md` |
| tw | the same recipe (`tw_register.py`) | 36 | 1,792 | `tw_sources.md` |
| cn | OSM named track + 12306 (`cn_register.py`) | 423 | 123,009 | `cn_sources.md` |
| hk, sg | MTR / LTA lists (`hk_register.py`, `sg_register.py`) | 13, 10 | 293, 269 | `hk_`, `sg_sources.md` |
| be, nl, at | ERA RINF (`rinf.py`) | 148, 96, 125 | 3,204, 2,780, 4,654 (holes filled 2026-10-05) | `<cc>_sources.md` |
| cz, pl, hu, pt | RINF | 244, 358, 133, 24 | 9,164, 16,270, 6,736, 2,145 | `<cc>_sources.md` |
| si, sk, bg, ro, fi | RINF | 22, 70, 32, 86, 31 | 1,157, 3,389, 3,617, 9,015, 4,029 | `<cc>_sources.md` |
| lt, lv, ee | RINF | 16, 10, 13 | 1,215, 923, 721 | `baltics_sources.md` |
| hr, gr, lu | RINF | 39, 12, 16 | 2,339, 1,726, 237 | `<cc>_sources.md` |
| ru | the tariff guide, through rinf.py (`ru_register.py`); the 2022-annexed railways included since 2026-10-04 | 928 | 77,913 | `ru_sources.md` |
| de, it, es | RINF (DB InfraGO; RFI + 8 regional managers; Adif, Adif AV, FGC) | 1,102, 310, 171 | 31,437, 16,856, 14,660 | `<cc>_sources.md` |
| us | FRA NARN subdivisions + OSM stations (`us_register.py`) | 448 | 39,807 | `us_sources.md` |
| ca | NARN's Canadian part, through us_register (`ca_register.py`) | 99 | 13,477 | `ca_sources.md` |
| se | RINF, Trafikverket's stråk names (`rinf_countries/se.py`) + a timetable feed | 56 | 9,124 | `se_sources.md` |
| au | Geoscience Australia's rail lines + OSM stations (`au_register.py`) | 124 | 17,070 | `au_sources.md` |
| dk | RINF, ids grouped by Banedanmark's line number (`rinf_countries/dk.py`) + a timetable feed | 46 | 2,408 | `dk_sources.md` |
| no | Bane NOR's Banenettverk + Entur's journey planner (`no_register.py`) | 25 | 3,689 | `no_sources.md` |
| in | Wikidata IR lines + the unofficial IR GTFS's km, traced by rinf.py (`in_register.py`) | 720 | 67,135 | `in_sources.md` |
| gb | OSM named track with repairs (`gb_register.py`); lines in pieces bridged over other lines' track or split (2026-10-04) | 472 | 15,537 | `gb_sources.md` |
| ie | RINF sections grouped into IÉ's lines (`rinf_countries/ie.py`) + NTA's feed | 17 | 1,626 | `ie_sources.md` |
| tr | OSM named track + TCDD's ticket list for stops (`tr_register.py`) | 41 | 8,118 | `tr_sources.md` |
| ua | Russia's tariff guide, UZ sheets, through rinf.py (`ua_register.py`) + a crawled timetable | 318 | 15,734 | `ua_sources.md` |
| rs, ba, me, mk, al, xk | network statements and hand lists through rinf.py (`balkans_register.py`) + feeds | 26, 5, 3, 6, 6, 3 | 2,563, 649, 237, 600, 370, 224 | `balkans_sources.md` |
| my | KTM's and Prasarana's GTFS stop lists laid on OSM track (`my_register.py`) | 17 | 1,960 | `my_sources.md` |
| th | OSM named track + route=railway relations (`th_register.py`) | 16 | 3,919 | `th_sources.md` |
| mx | OSM named passenger track (`mx_register.py`) | 27 | 3,376 | `mx_sources.md` |
| id | id.wikipedia's line tables (KAI km posts) through rinf.py (`id_register.py`) | 39 | 4,274 | `id_sources.md` |
| nz | KiwiRail's line register + km posts through rinf.py (`nz_register.py`) | 15 | 1,455 | `nz_sources.md` |
| ir | OSM route=railway relations + RAI's timetable via iranrail.net (`ir_register.py`; `--construction <pbf>` on a new extract, then `--clip`; recrawl with `--crawl`, `--parse`, `--timetable`) | 26 | 9,414 | `ir_sources.md` |
| kz, uz, kg, tj, tm | the tariff guide's sheets through rinf.py (`casia_register.py`) + a feed crawled from KTZ's ticket site | 74, 37, 3, 5, 10 | 15,016, 3,574, 285, 406, 2,937 | `casia_sources.md` |
| by, md | Russia's tariff guide sheets through rinf.py (`bymd_register.py`; `--cc <cc> --clip` after every extract) + crawled timetables; md includes Transnistria, greyed (2026-10-04) | 75, 15 | 4,931, 1,025 | `by_`, `md_sources.md` |
| xa | Abkhazia (2026-10-04): Book 1's 57-001 through rinf.py (`caucasus_register.py`; `--clip xa` after Georgia's extract) + a hand timetable | 1 | 156 | `caucasus_sources.md` |
| ma, dz, tn, eg | hand line lists from the operators' timetables through rinf.py (`nafrica_register.py`; extract with `--station-areas`, then `--clip`, `--fill`) | 13, 24, 11, 30 | 1,868, 4,850, 1,364, 2,949 | `nafrica_sources.md` |
| ge, am, az | Russia's tariff guide sheets through rinf.py (`caucasus_register.py`; `--clip <cc>` after every extract) + feeds (az's by hand) | 18, 10, 17 | 948, 573, 1,595 | `caucasus_sources.md` |
| ar, cl | the ways of hand-listed OSM passenger routes (`ar_register.py`, `cl_register.py`; `--clip` after every extract) | 23, 7 | 3,218, 767 | `ar_`, `cl_sources.md` |
| br | OSM named track (CPTM) and route relations' ways, passenger track only (`br_register.py`) | 19 | 2,136 | `br_sources.md` |
| za | a hand line list traced by rinf.py (`za_register.py`); PRASA's recovery reports decide what runs | 53 | 4,701 | `za_sources.md` |
| vn | OSM named track, cut as DRVN's line list (`vn_register.py`; extract with `--station-areas`, then `--clip`) | 10 | 2,456 | `vn_sources.md` |

**State.** Every country has `foot.json` (track ownership) and no `credits.json`. Track over
a border is drawn where ERA RINF has a border point (`borders.py`, `border_points.json`). A
national timetable feed decides which register sections trains run over in 21 countries (cz,
hu, pt, pl, be, si, sk, ro, bg, fi, lt, lv, ee, hr, lu, gr, de, se, it, es, dk; off for at and
nl; `gtfs_served.py`, `gtfs_sources.md`); Norway's reader does its own check from Entur. **build_model.py names no country**: each country's rules are in `rules/<cc>.py`
(2026-10-03; `country_rules()`'s docstring lists the hooks). Germany, Italy and Spain were added on 2026-10-02 and
every country rebuilt in one batch the same day (`tools/rebuild.py`, 3 at once: 30 countries
in 23 min; register lines unchanged everywhere; the doubled border ids now one id on both
sides; 305 line ids shared between countries).

### How the work is run

- **One managing session holds the shared files**: `build_model.py`, `ownership.py`,
  `build_tiles.py`, `extract.py`, `rinf.py`, `borders.py`, `line_colours.py`,
  `not_running.py`, `gtfs_served.py`'s hook, `tools/` and `dist/index.html`. It lands every
  shared change and runs `tools/build_regions.py`.
- **Country agents, up to eight at once, own one country each**: `<cc>_register.py` or
  `rinf_countries/<cc>.py`, `<cc>_sources.md`, `colours/<cc>.csv`, `data/raw/<cc>*` (and
  `data/raw/rinf/<cc>`, `data/raw/gtfs/<cc>`), `data/proc/<cc>`, `dist/data/<cc>*`, and their
  own `REGISTER["<cc>"]` in `check_model.py`. For a shared file they send the managing session
  an exact diff. Only one session rebuilds a given country, so outputs are never clobbered.
- An app session, if one runs beside them, owns `dist/index.html` and asks for a new output
  field rather than editing the Python build. Never two sessions in `build_model.py`.
- Builds run freely in noritetsu unless one would take over ~2 hours (Anita). The permission
  classifier blocks subagents' all-country rebuilds and new downloads: the managing session
  runs those, with Anita's say-so. Ask her before big downloads (she paused them for a few
  hours on 2026-10-03); `.osm.pbf` extracts are fetched and extracted by the managing session.
- **Many agents at once** (2026-10-03: ten, Anita "feel free to start lots of agents"): give
  each a written list of the files it owns (a country agent also owns `rules/<cc>.py`; one
  agent may own a shared file for the session), allow no-op-unless-set hooks in rinf.py, and
  have **every build, extract, trial and tiling run go through `python tools/slot.py [n] --
  <cmd>`** (six shared slots = Anita's 6-core cap; threads capped to the slots held) and every
  `ab.py` run set its own `NORITETSU_AB_DIR`. Agents report diffs for files they do not own;
  the managing session lands them and does the batch rebuild and `build_regions.py` at the end.

### Anita's standing decisions

- **Decide local specifics yourself** (Anita, 2026-10-03): "i dont really know about the
  specifics of any rail systems outside the us and japan so i generally wont be able to make
  informed decisions, and you should just trust your gut and not ask me about local stuff like
  this." Later the same day, for the US too: "i guess in general just dont ask me for
  confirmation on factual things. ill let you know if i see something thats wrong." In any
  country, whether a train stops somewhere, whether a service runs, whether something is a
  line or a named train, whether a heritage or seasonal train counts, which line owns a
  stretch, how to cut a city's lines: make the call, write the reasoning in
  `<cc>_sources.md`, and do not put it to her. Ask her only about rules that apply
  everywhere, app behaviour and design, and who a map depicts.

- **Completion counts everything with scheduled passenger service** ("if it is scheduled at
  all, more often than about once a week"). **Named trains (option B)** have no percentage of
  their own and totals leave them out; their track counts through the lines it lies on.
  **Except in the US, Canada and Australia** (Anita, 2026-10-09: "for US and canada and
  australia lets do this change yeah. routes become what operator lists show. though we can
  leave corridors that are well known and/or have lots of different services on them like
  northeast corridor"): there the register is track (FRA subdivisions, Geoscience Australia's
  names), so the lists show the routes riders know, every OSM line of the country, named
  trains included, each with its own percentage and in the totals; a register line stays
  listed only as a corridor or where the routes leave part of it uncovered. Crediting is
  unchanged. App: "Working on the app", routes.
- **One owner per piece of track**: "no double counting, every piece of track belongs to
  exactly one line; for crediting only, trips can still be entered on any service or path"
  (`ownership.py`).
- **Border track** counts in the country it lies in; a ride over the border credits both. A
  section built whole over a border is cut and the far part handed to the neighbour once it is
  built. A crossing RINF has no point for is added to `borders.EXTRA` by hand when the country
  on the other side is built. A border point shows one neutral name ("Belgium – France
  border"), not either country's RINF name.
- **Seasonal lines are drawn as running** (Poland's 96 Muszyna - Leluchów, Croatia's
  Metković - Ploče). **Not running is greyed** and left out of completion (Porvoo is fine
  greyed).
- **Russia**: the tariff section is the line unit. The 2022-annexed railways (Donetsk,
  Luhansk, Melitopol - Kherson) are built with Russia since 2026-10-04 (Anita: "makes sense if
  theres trains run"): `ANNEX_RUNNING = True`, `annex.geojson` in build_regions'
  `EXTRA_AREAS`; running only where daily trains run (poizdato's suburban trains, kept in
  `data/raw/ru/annex_trains.json`: 17 lines / 437 km), the rest greyed (64 / 1,703 km,
  Melitopol - Kherson whole). ru_sources.md "The annexed railways: what runs". Crimea is
  built with Russia. **Abkhazia is its own region, `xa`; Transnistria is Moldova's, greyed**
  (Anita, 2026-10-04).
- **Israel and North Korea** (Anita, 2026-10-08): "jerusalem: if israel administers it we can
  draw it under israel" (the Red Line through East Jerusalem, and the high-speed line's West
  Bank stretch, both under `il`); "north korea: if it runs passenger service to the best of our
  knowledge, we should build it". Tenerife's tram goes into `es`.
- **Croatia's single "B" fast trains stay named trains.** Czechia's JHMD (228/229 out of
  Jindřichův Hradec): unknown whether it runs, leave as is.
- **Where it is going (Anita, 2026-10-02)**: eventually timetables for every country and real
  stop routing, so the map is right close to 100% of the time and says which cases it is not
  sure about. OSM route relations are a fine source for large parts of the world on the way
  there. In the meantime new countries come first ("timetables mostly verify details", the
  same day). Steps: stations from route stop positions (the way a stop node lies on decides
  its line) wherever OSM has them; a feed per country as feeds allow; the per-section
  certainty gtfs_served already computes (served, ambiguous, unknown, border) carried into the
  data so it can be shown.
- **Data**: inspecting data only reads it (no caches or side files beside it). Ask her before
  new downloads or anything else that writes to data beyond an agreed build.

### Open threads, in the order I would take them

**NEXT SESSION, START HERE: Anita's notes from 2026-10-08 (after testing the 10-07 work).**
She liked: one ride filling a line ("works much better now"), same-track crediting, the
operators, the English names. To do, in her words where they matter:
1. DONE (app session, 2026-10-08). **100% must hold for countries and operators too.** "if you
   click everything you should be able to 100% a line ... should also apply to countries.
   100%ing a country should be possible. it should be the union of 100%ing all the individual
   lines. same for an agency." A country's (`c:<cc>`), operator's (`o:<key>`) and
   operator-in-a-country's (`r:<cc>|<key>`) total is now the union of the footprints of its
   listed lines' running sections (`unionTotals`); km ridden is each of those sections' credited
   part carried through the same footprints, a line at 100% counting whole (`unionDone`), so
   both sums add the same terms when every line is done. Used by `stats()`, `buildOps`,
   `renderOp`; `regionTotals` and `ownedDone` are gone, `creditState` now wraps
   `creditFor(rides)`. A loaded country shows its own union; regions.json `km` only before it
   loads (cz differs by 6.5 km with neighbours loaded, from border sections both built).
   `build_regions.owned_totals` does the same union (listed lines per `listed`, across
   countries); country totals fell 0-942 km (ru -942, de -329, pl -140, it -112: track only
   operating patterns or named trains owned). `line_100_probe.js` part 3 (`groups`,
   `groupsWhole`): 22 countries probed in three batches (ch lu be; de at nl dk cz pl fr; jp
   us ca gb ie it es pt kr tw hk sg), every group whose lines all finish reads exactly 100%
   (done === total in 1,829 of 1,829 groups; 1,996 of 1,996 with the lines the picks cannot
   finish ridden whole: jp, ie, pt, kr, tw already 100% on picks alone). The country groups
   held back are by lines the per-line test still fails (stops missing, item 6).
2. DONE. **Operator name in the line view is a link** (`opLinks`, `goOp`): to the Operators
   tab with that operator's row open, drawn and scrolled to, as a click on the row does; an
   operator with nothing ridden (no row there) or in "View all lines" opens `showOp`'s view.
3. DONE. **Default line order** (`lineListOrder`, `lineDone`): longest not at 100% first,
   then the 100% ones longest first; Lines tab, an operator's open row (Operators tab and
   Countries tree) and the operator view. In "View all lines", longest first. The Lines tab's
   reset button reads "unfinished longest first".
4. **NJ Transit Morristown Line, Dover -> Hoboken Terminal can't be entered**: picks work
   Denville - Newark Broad St but not past the junction at either end. Picking through a
   junction end to stops on the far side must work in one ride.
5. **The Morris & Essex Lines' diagram branches wrongly**: stops out of order and not on the
   right branches (her screenshot: Dover / Denville / Convent Station / Mount Arlington on
   side lanes, the Gladstone branch's stops interleaved, Madison / Chatham on a side lane
   beside Summit). Look at the us register/OSM pieces there (Morristown Line, Gladstone
   Branch, Montclair-Boonton) and the strip layout.
6. Still open from 10-07 (below): stops missing from the data (Russian tariff lines, a few
   deleted ids like Myronivka - Bohuslav). Paris Metro 10's loop: FIXED IN TRIAL, WAITING ON
   A SHARED DIFF (fr agent, 2026-10-08). Cause: build_model.place_stations snapped each
   direction's path onto the other arm's stations (150-330 m away, under STOP_SNAP_M 400), so
   both arms were threaded into one chain. `handoff_notes/metro10_loop.md` / `.diff`: snap a
   station whose stop node is on another route's track only where that track runs alongside,
   gated on `SNAP_ALONGSIDE` (rules/fr.py already sets it; other countries build exactly as
   before). Trial: line 10 a real loop (14.73 km, both arms), 22 fr OSM lines change (false
   chords gone from RER A/C/D, J, 13 tram lines; 3 TGV named trains +149 km at Nîmes), no
   register line or station id moves; the diagram draws the loop in headless Chrome. To land:
   apply the diff, rebuild fr, `build_regions.py`. search.json's operator
   text: DONE 2026-10-08 (`build_regions.passenger_ops`, from ops.json and operators.json'
   short names, as the app's opName; 9,863 of 19,754 lines changed: La Junta
   Subdivision "BNSF Railway" -> "Amtrak", Spain's 200 "Adif" -> "Renfe", ICE 91 "DB
   Fernverkehr" -> "DB"); the app's search matches the shown names too.

**Countries not yet built: surveyed 2026-10-08** (six research agents, nothing built or
downloaded beyond small samples in `data/raw/<cc>/survey/`). Each country with service has a
`<cc>_sources.md` with a "Survey (2026-10-08)" section (sources, licence, recipe, size, what
runs, extract name and size); countries with none are one line in `wafrica_survey.md`,
`eafrica_survey.md`, `mideast_survey.md`, `latam_survey.md`, `asia_survey.md`.
- South Asia: pk (priority), bd, lk, np. Hand lists through rinf.py; lk on id's recipe.
- West/Central Africa: ng (priority), gh, sn, bf, cm, ga, cg, cd, ao; one shared reader
  proposed (`wafrica_register.py`, nafrica's pattern). Nigeria needs a gauge preference
  (three parallel railways Ebute Metta - Agbado).
- East/Southern Africa: ke, tz, et+dj, ug, zm, zw, mz, mw, mg, mu; one shared reader proposed
  (`eafrica_register.py`); border points needed at Tunduma/Nakonde and Dewele/Guelile.
  Decided by the managing session: Malawi kept running, Sudan out until a timetable, Bulawayo -
  Harare and Nairobi - Kisumu greyed.
- Middle East: il (MOT GTFS via Mobility Database mdb-2519, my's recipe), sa+ae+qa (one Gulf
  reader, gcc-states extract), iq (ir's recipe), jo (Hejaz excursion, counted). Jerusalem's
  light rail through East Jerusalem is Anita's call (asked 2026-10-08).
- Latin America: pe, bo, co, ec, ve, pa, cr, uy, cu, do, pr (pr is not in us); ar's recipe,
  six small ones could share one reader.
- Rest of Asia: ph, mm, kh, la, mn (UBTZ timetable blocks scripts: Anita's browser), kp
  (Anita's call whether to depict), Canary Islands into es (`canaries_survey.md`).
**Built the same day (Anita: "we can start building")**: 45 new regions, each by a country
agent, each probed with line_100_probe.js (every line, operator and country exact unless noted):
pk (`pk_register.py`); lk, bd (`lk_register.py` is the engine for both); sa ae qa iq jo
(`mideast_register.py`, nafrica's code; qa no register); ng cm ao ga cg sn gh bf cd
(`wafrica_register.py`); ke et dj mz zm zw tz mg mw ug mu (`eafrica_register.py`; mu no
register); cr pa cu pe bo ec uy ve co + do pr with no register (`latam_register.py`, ar's code);
kh la ph mm mn np (`asia_register.py`, lk's engine). Each `<cc>_sources.md` has a "Build
(2026-10-08)" section, each reader's shared diffs are in `handoff_notes/<group>_build.md` and are
landed (rebuild.py REGISTER, `"<cc>": None` = no register, run_country skips --register; new
borders.EXTRA points eXIRPK1, eXDJET1, eXTZZM1, xBotenMohan, xNaushkiSukhbaatar,
xZamynUudErenhot, xEreentsavSolovyevsk, xJaynagarInarwa; cn_register CN_BORDERS for Boten and
Zamyn-Üüd; cn la mn np et dj tz zm rebuilt for them). **Not done**: Russia's side at Naushki
(~6 km under mn's line id) and India's at Jaynagar (~3 km) are in handoff_notes/asia_build.md:
until then mn's Sükhbaatar - border tail is unfinishable (mn 99.3% whole); Iran's side at Taftan
(ML-4 greyed anyway). The Tenerife tram is in es since 2026-10-08 (Anita: fold it into
Spain): `python -m rinf_countries.es --canaries` after every es extract joins the islands'
extract (region `ic`, extract only) into data/proc/es; es_sources.md "The Canary Islands". Doha's two campus trams
count (no "drawn, not counted" mechanism yet; proposal in handoff_notes/mideast_build.md).
**Later the same day, after her answers**: Israel built (`il_register.py`, my's Track, MOT feed
rail subset in data/raw/il/gtfs; East Jerusalem and the A1's West Bank stretch under il as she
ruled); North Korea built (`kp_register.py`, Korea's recipe; running by evidence class A-D in
kp_sources.md; borders xSinuijuDandong, eXKPRUTUMANGANG, xManpoJian; cn CN_BORDERS and
ru_register BORDER for them, cn and ru rebuilt). **Open**: Manp'o - Ji'an does not join (cn's
track ends 74 m from the point, PIECE_BORDER_M is 60); two Pyongyang tram lines named "2"/"3"
(unnamed route_masters). Probe: il exact, kp 99.8% (the Manp'o and Tumangang tails).

**Missing stops (item 6), landed 2026-10-08**: `rinf.py` gives a stop reached only through a
clone link its station record, so `split_pieces` folds the clone in (handoff_notes/
missing_stops.md; ab.py on kz uz kg tj tm by md ge am az xa: only kz moved). `ru_register`
makes a no-letter point a stop where an OSM train stops by name (116 points); pl's Olecko a stop
(lines 39 and 41 back). ru ua kz pl rebuilt. Probe: ru 57 lines / 3,164 km unfinishable -> 14 /
187 km, ua 5 / 33, kz 4 / 32, mostly border tails with the neighbour unloaded and points Book 1
does not place (Kartaly I, Orsk, Rossosh). The deleted ids (Myronivka - Bohuslav, Taraz -
Zhanatas, Nukus - Shymbay...) were checked: no passenger trains, they stay deleted.

**Flavour stats (Anita, 2026-10-09)**: line type and station ridership are being added as
post-build files (`tools/line_types.py` -> `types.json`, `tools/station_riders.py` ->
`riders.json`, app patches in handoff_notes/). **Ridership per line and per segment: later**
("a thing to look into later"); flavour_stats_checklist.md §3-4 has the sources.

Every new extract needs Anita's say-so (asked 2026-10-08).

**2026-10-08, landed by the managing session:** `build_model.py` got two hooks, both no-ops
unless a rules file sets them: `SNAP_ALONGSIDE` (fr only: Métro 10's Auteuil loop and 21 other
false links; handoff_notes/metro10_loop.md) and `route_runs` (us only: 23 OSM relations whose
ways are in no order, the Morris & Essex; handoff_notes/njt_morristown.md). fr and us rebuilt,
compare_lines clean (only the Morristown Line moved, New Providence off it), build_regions run.
`line_100_probe.js` on us+ca: every group exact; 5 us and 5 ca lines unfinishable (stopless
tails, e.g. Bellingham - border 40.8 km). The app's junction-pick crash (no "Add ride" for any
pick through/past a junction end, every country) fixed in `pickedRun`.

**2026-10-07, app session (dist/index.html, dist/poster.js; no build change, nothing rebuilt).**
Landed: the poster as a shareable 2400 px image (`poster.js`: auto insets with a city map for
every city with 15 km+ ridden, starred stats as rings, zoom/pan preview, Download and Copy);
stars on countries/operators/lines/trips (`STARS`, `starredStats()`); rings instead of bars in
"Travel tracker" mode, none in "View all lines" (the buttons renamed); no all-caps headings and
no "Ridden in <country>" headline; Countries tab a tree (country > operators > lines), every
list drag-reorderable (`ORDERS`, one key per list, "sort by distance ridden" per tab); a joint
line counts under each of its operators (`opKeys`); an operator row draws its lines on the map,
a click on open map clears it; lines picked from a list are brought into view; one blue for
every line without a colour (kind colours gone, legend three rows); pale line colours darkened
in the light theme (`forTheme`, `LIGHT_MAX_LUM`, the tiles' TRACK_LINE_COLOUR_LIGHT); a solid
selection halo (the translucent one stacked into spikes at curves); flags from flag-icons
(jsdelivr); English / local names menu with Cyrillic, Greek, Georgian, Armenian transliteration
(`translit(s, cc)`, per language: Ukrainian 2010, Bulgarian, Serbian Latin, Kazakh, ELOT
digraphs; generic line words "Ligne N" / "N号线" / "Пригородный электропоезд" in English via
`englishLine`; the rest of translation_coverage.md's order, build-side, is open); intercity stations (named train or high-speed) from z8 (`station-ic`); "Also on
this track" folds after six; register/ownership notes gone from the line view; OP_EN no longer
lets one line's operator:en rename Network Rail. Crediting (handoff_notes/full_line_not_100.md):
a fully ridden line is exactly 100%, stubs under 1 km / 10% past a stop come with the stop,
"siblings" (same two stops, 80%+ shared track) credit each other on their own lines, and
`SAME_TRACK` pairs NYC 6/<6>, 7/<7>, F/<F>.
Later the same day: every line 100%-able from its own diagram (picks ride the drawn way,
junction ends pickable only where nothing else reaches, "by the other track" choices;
`tools/line_100_probe.js` passes all 12,428 listed lines; handoff_notes/full_line_not_100.md);
a selected line's rings sit on its own track (`linePoint`); English names filled after the
build (`tools/english_names.py` -> `dist/data/<cc>/names_en.json` and `dist/data/names_en.json`,
upload with the data); view-only share links with the rides packed into the URL hash
(`dist/share.js`, `#s=`; a short-link Worker later if wanted); opening a country moves the
camera only when none of it is on screen; poster: Osaka + Kyoto style merged city maps, a
city map needs a trip that began and ended there, inset count 0-25 in the dialog.
Research written up: `operator_branding_sources.md` (colours, logos, short names: no bulk
colour source; Wikidata linking 86%; `data/operators.json` is read by the app if present,
`{key: {short, colour, logo}}`, not yet written), `flavour_stats_checklist.md` and
`translation_coverage.md` (agents, check they landed).
**Round 2, same evening (all 73 countries rebuilt):** junction ride ends removed (Anita: track
past a junction with no station is not passenger rail); `build_model.prune_dead_track` drops
it (-1,321 km, 81 line ids gone: handoff_notes/dead_track_km_2026-10-07.txt); `along.py`
writes `dist/data/<cc>/along.json` (sections within 30 m, same kind, not HSR vs conventional,
not different register lines) and the app credits a section 90% alongside a ride (60% with
both ends stops passed) on its own line: SAME_TRACK hand list deleted; one end-to-end pick
fills 9,226 of 9,313 unbranched lines. Legend removed; "View all lines" shows only Countries
and Trips, countries by length. **Open: stops missing from the data.** Most of the 147 lines
no picks can finish (57 Russian, 3,164 km, e.g. Tommot - Nizhny Bestyakh with 0 stops), and
a few of the 81 deleted ids (ua Myronivka - Bohuslav, kz Taraz - Zhanatas, uz Nukus - Shymbay,
pl 39) are lines whose stations the build never marked as stops: add the stops (tariff guide
/ timetables), then they come back. Anita's calls: Indian Railways one row; Italy stays RFI
until timetables. Upload `along.json` with the data.

**Operators are the passenger companies now** (Anita's call): `tools/operators.py`
(measure / seed / logos / build) writes `dist/data/<cc>/ops.json` (register line -> its
passenger operators by share, from the OSM services over its track), `dist/data/operators.json`
(short name, colour from Wikidata or the logo, logo, group) and `dist/data/logos/` (216 free
PNGs, `credits.json`); `colours/operators.csv` is the hand-editable table (filled cells are
never overwritten). DB, SNCF, RZD, Indian Railways, China Railway, metros by brand (Shanghai
Metro...) are one row each. Left: it 5,848 km still under RFI (no OSM services there).
`search.json`'s operator text names the passenger operators since 2026-10-08. Upload
`ops.json`, `operators.json`, `logos/` with the data. Paris Métro 10's one-way loop at Auteuil is drawn as one sequence in the strip
diagram: the fr build ships it as one chain (2026-10-08 list, item 6: a build_model diff waits
in handoff_notes/metro10_loop.md), not a layoutPiece case. Operators of the same name in two
countries share a row (India's and the UK's Southern Railway): key by country.

0. **Next session, start here (written 2026-10-05 by maps-33, which finished the three
   threads the 10-04/05 managing session handed off).** Nothing from 10-02 onward is
   committed (Anita commits herself). All three are landed and rebuilt; HISTORY.md
   "2026-10-05" has the detail, `handoff_notes/<topic>.md` the working notes.
   - **Opposite-direction track and "abroad"** (`handoff_notes/opposite_directions.md`):
     ownership.py's abroad test needs another country's land, and a way no register line
     owns takes its pair's owner (the same line's other-direction track). Trialled on all 73
     countries, the new-owner rows reviewed (ru, de, pl, at, us, jp, gb: per-direction
     trains and ring trams sharing an owner, named-train track to the line under it; no two
     different lines merged), then every changed country rebuilt model-only. Anita's PATH
     ride on the shipped data: JSQ-33rd 73% -> 95%, 33rd-Hoboken 59% -> 97%. Left: 33rd
     Street's ~3% (each line's own platform road, not a directional pair).
     The au rename seen in the trial was not this change: `us_register.osm_line_name` broke
     a tie between two OSM relations over the same track by set order, which string hashing
     changes per run. Ties now go by name (keeps "Melbourne - Adelaide Rail Corridor").
   - **Channel Tunnel** (`handoff_notes/channel_tunnel.md`): gb and fr rebuilt with tiles.
     New register line "Tunnel sous la Manche" (fr, 23.1 km); HS1 reaches St Pancras
     International (149.2 km); Eurostar Paris/Brussels/Amsterdam and Le Shuttle are named
     trains in both countries and credit HS1 and the tunnel. Left: Amsterdam - London has no
     French piece (no stop in France in OSM); Calais-Fréthun has no Eurostar stop in OSM.
   - **RINF lines in pieces** (`handoff_notes/rinf_pieces.md`): landed in rinf.py. Every
     RINF-read country folds a stop's 0 km link junctions back into the stop
     (`split_pieces`, the build_model hook; the eight wrapper readers pass it on), and
     `fill_holes: True` (set in at, de, cz, pl, be, bg) joins a line's pieces over its own
     OSM relation's track. Two guards added on trial: a fill may lie at most 30% on track
     another RINF line was traced over (`HOLE_OTHER`), and a fill's OSM stop of a register
     stop's name within 1 km is that stop (OSM's second "Wien Hauptbahnhof" node had taken
     30 lines). Lines in pieces: at 47 -> 11 (Tauernbahn whole, 114.5 km, Pusarnitz a stop),
     ru 304 -> 36, ua 62 -> 29, kz 15 -> 5, cz 22 -> 18, de 50 -> 44, by 6 -> 2, uz 4 -> 2;
     1,297 junction ids gone (ru 1,142; junctions only, never a ride's end). de 4721
     Untertürkheim - Nürnberger Str is gone, correctly: it was only credited because the
     timetable check routed S-Bahn trains over it while 4713's Bad Cannstatt - Nürnberger
     Straße was a hole. **Not done**: `bridge_pieces` (written, never trialled; meant for
     pt's Linha do Sul and it's five drop-gaps); Austria's 10 and Germany's 44 left. RINF
     has no section of the Tauernbahn's three holes under any id or country (queried with
     Anita's OK, 2026-10-05), so filling over OSM's track is the right fix.
   - **Lines in pieces, the worklist** (Anita, 2026-10-05: "make a list and go one by one"):
     `handoff_notes/lines_in_pieces.md`, from `tools/pieces_report.py --list` (reads dist
     only; a first guess per line: crossing / near / hole / far). 323 register lines, 259
     OSM lines and 32 named trains were in pieces. Fixed and rebuilt the same day: transit
     sections (`build_model.transit_tail`: Eurostar Amsterdam - London across France),
     fill_holes retrying pairs (Kamptalbahn), out-and-back spurs (`build_model.fold_spurs`:
     58 of 74, mostly gb's bridged detours), and the app drawing a continuation's track from
     the wrong half of its section (Hell Gate -> Penn Station). Then, Anita's call, register
     lines whose pieces lie over 25 km apart split into one line per piece
     (`build_model.split_far_pieces`, 15 countries rebuilt). **She asked to stop there**: the
     rest of the worklist (crossings, near gaps, holes) is parked until she asks again.
   Done 10-04/05 and live (details in the bullets below and HISTORY-worthy): light theme
   default; muting by a second track layer (no tile re-parse); place names under the lines;
   seas' names hidden; Countries tab by km ridden; branch side by angle; phone bottom sheet;
   search ranks stations over route-named lines; border-point countries load with a line;
   China extend_ends fix; outline-sided border cuts (Irun/Hendaia, Varnsdorf); US/CA lines
   in pieces split (+ Penn Station); UK/kr/cn/tr pieces bridged or split (`pieces.py`);
   Russia's annexed railways (running where trains run); Abkhazia `xa`; Transnistria greyed
   in md; junction continuations only where services run, ending at a hub within 1.5 km,
   pruned past nearer stops, one list for map and diagram, track drawn to each; quiet/stub
   junction rows hidden; straight-line gaps out of ride routing. Anita's open asks: none.
   The older start block follows.
   **Next session, first** (written 2026-10-03 by maps-0d; what that day's ten agents did is
   in `HISTORY.md`, "2026-10-03: ten agents at once"). **Nothing from 2026-10-02 or 10-03 is
   committed**: Anita commits herself. In order:
   - **Anita, 2026-10-04, for the next session**:
     - **Branch orientation in the strip diagram: done 2026-10-04** (`branchSide` in
       `layoutPiece`). Her preference was by where trains run (Springfield - New York, not
       Springfield - Boston), failing that by the track's angle. The app has no per-route
       stop lists (lines.json ships sections and one `display`), so the angle stands in: a
       branch off the main route's middle hangs on the side whose heading (to a station ~5 km
       out) is nearer its own, since trains off a branch carry on the way the curve points.
       Springfield now reads Springfield ... Wallingford, State Street, New Haven, New York.
       In jp, de, gb, ch it turns 191 of 407 side blocks upward; samples checked right
       (Gala-Yuzawa, RE7's Kiel, Oban, the WCML's Edinburgh). It is wrong where trains
       reverse at the junction (the Okhotsk at Engaru). If that matters, the build would
       have to ship each OSM route's stop list.
     - **Done 2026-10-04 (US agent + managing session; us_sources.md "Lines in pieces, and
       Penn Station's west end", ca_sources.md "Lines in pieces")**: a register line whose
       shipped sections do not all connect becomes one line per piece
       (`us_register.split_pieces`, called by a build_model hook after
       `drop_unridden_sections` for any register module defining it; JOIN_KM still joins first
       so the holes file can fill gaps). The biggest piece keeps the id, others get
       `piece_id` (line id + lowest NARN segment) and a name from their ends ("Michigan Line
       (Albion – Dearborn)"); aliases.json `pieces` lets the app's migrateRide move a ride
       between two stations of another piece onto it. US 47 lines -> 102, Canada 16 -> 48.
       Penn Station: NARN filed the North River tunnels' New York end under New York
       Terminal; `SEGMENT_NAME` reads it as the NEC, and the NEC, West Subdivision and Empire
       Connection meet at one junction (uj489969). Left open: cutting a line where another ends
       inside its junction-ended section (declined, moves Austin +43 km), 13 US one-stop lines
       with a junction end leading nowhere (listed in us_sources.md). The original note:
       **US subdivisions that are not one piece.** "New York Terminal Subdivision" is two
       disconnected pieces (Spuyten Duyvil - near 34th Street, and near 46th Street - the
       Main Line / West junction): NARN names track by owner's subdivision, and the two
       pieces share the name, so us_register makes one line of them. Split a register line
       whose sections do not connect into one line per piece (us, ca; check others), and
       look at the many US subdivisions with one stop and junctions at both ends. Her
       principle: inputting a trip offers only station to station, so a line's diagram
       should run station to station (the app half landed 2026-10-04: the strip diagram
       continues past each junction end to the stops throughEnds/pastEnds find, in thin
       lanes of the reached line's colour, branching where a junction leads to several
       lines; junctions stay unpickable pass-through dots; the old "→ stop" rows are gone).
       Also: Penn Station's west end goes nowhere: West Subdivision's "New York Terminal /
       West" junction is 314 m from the Empire Connection's own end junction "near 34th
       Street - Hudson Yards", so neither continues; make the two share one junction in
       us_register (or find the missing track).
     - **Muting was laggy; fixed 2026-10-04.** `setMuted` used to swap `track-<cc>`'s
       data-driven `line-color`, and MapLibre 4.7 re-parses every loaded tile of a source on
       any data-driven paint change: the dim landed 120-240 ms after the selection. Now each
       country has a second layer, `track-mute-<cc>` (THEME.muted), and muting swaps the two
       layers' constant `line-opacity` (`TRACK_ALPHA` / 0): 0 tiles re-parsed. The branch
       opacity (0.78 beside 0.9) moved into the colour (`RANK_MIX`) so the opacity can be a
       constant. MapLibre's 300 ms paint fades are off (`applyBasemap`). A single dimming
       layer over the map was rejected: it dims the basemap too, and Anita wants only the
       lines dimmed. **Light is now the default theme** (Anita, same day; a stored choice
       still wins). Still open: a long line's `showLine` costs a 60-100 ms frame of its own.
     - **Export / import** "soonish". Today: Export writes every ride as JSON
       (`exportRides`), Import merges a file by ride id and migrates old ids
       (`importRides`); spec §5 makes that file the durable format. Asked her what is missing.
   - **UK lines in pieces, done 2026-10-04** (gb_sources.md "Lines in pieces"): 47 of 464 were
     in pieces, 29 of them because OSM names shared track once (Birmingham - Peterborough over
     the MML at Leicester). `gb_register.bridge_gaps` joins pieces over the other line's track
     (cheapest passenger path, bounded), listed as `borrowed` in lines.json and padded
     `BORROW_PAD_M` so ownership still gives that track to its own line (keep it between
     ownership's EXACT_TIE_M and TIE_M); the rest split as in the US. Now 0 in pieces.
     Done for kr, cn, tr the same day (pieces.py, shared by all four; <cc>_sources.md "Lines in
     pieces"): kr 5 -> 0 (수인선 over 안산선; 호남고속선, 인천 1호선 over own track; 경강선,
     서해선 split), cn 7 -> 1 (京港高速线 over 昌九城际线; 成昆线, 沈佳, 甬广, 川藏, 龙龙 split;
     青荣城际线 left whole, OSM's track breaks at 烟台南), tr 3 -> 1 (both İstanbul - Ankara lines
     bridged; Mersin - Adana - Gaziantep left whole, track missing from the extract).
     Uncertain: 京港高速线's 45 km through Nanchang may not be the track trains use.
   - **Every agent lean from 2026-10-02 and 10-03 stands** (Anita, 2026-10-03: "i trust your
     leans"); the list is in `HISTORY.md`, "2026-10-03: decisions settled by the agents'
     leans". She asked for the names-only search index (thread 9) and the US station override
     list (thread 8); both started 2026-10-03.
   - **Landed 2026-10-04**: `split_at_borders` cuts a section with neither end a register
     station (an OSM line over a crossing) at its one border point, the end deeper inside
     this country's outline staying (`borders.depth_m`; at least `SIDE_M` 300 m deeper, and
     no further than `OUTLINE_SLACK_M` 1.5 km outside it at one end, inside at the other: the
     outline puts Irun Ficoba 1.1 km inside France, and Ebersbach - Neugersdorf, both ends in
     Germany, dips through Czechia past one point). `xIrunHendaia` is live. The de-borders
     agent's own diff was never saved, so this is a rewrite. Country outlines moved from
     tools/build_regions.py into borders.py (`SHAPES`, `OUTLINE`, `outline()`), shared by
     both. ab.py on all 52 border countries: only de (Seifhennersdorf - Varnsdorf), fr
     (Hendaia - Irun Ficoba) and es (the E2's tail to the point) move; each sided section is
     logged ("border: <line>: <home> (… m in) kept, …"). Next: the UBB at Ahlbeck.
   - **Timetable follow-ups from 2026-10-03**: Italy's high-speed rule (AV Treviglio - Brescia,
     53.8 km, still dropped as "ambiguous"); India's feed in gtfs_served (the in agent's diff in
     in_sources.md; Mumbai locals are not in NTES, so they must come out unknown); gb needs a
     `gb` -> `uk` outline mapping before any UK feed; Ferrotramviaria's official feed
     (Mobility Database mdb-1058); the empty GTT/TFT zips in data/raw/gtfs/it (deleting them
     was blocked for the agent).
   - **US/Canada station placement** (thread 8), done 2026-10-03: no rule reaches Long Island
     City (no OSM route lists it, so it has no stop nodes), so `us_register.NOT_ON` (and
     `ca_register.CA_NOT_ON`) is an explicit override list, "station X is not on line Y" or
     "X is no stop", each row pinned to a point so a moved station fails the build loudly
     (us_sources.md "Station overrides"). LIC and Hunterspoint Avenue are off Amtrak's West
     Subdivision; museum depots, Northstar's six stations, Marceline and Clifton are no stops.
     OSM lines in us/ca now get the stations their relations leave out
     (`extra_route_stops` in rules/us.py and rules/ca.py, read by build_model.build()): 204
     stations on 70 US routes (LIRR Oyster Bay and Babylon, SEPTA Chestnut Hill East, NJT
     Princeton Branch appear; MBTA Newburyport/Rockport 36 -> 85 km). Open: a relation with
     broken runs or two branches in one gets a long section across the gap (MBTA Fall River /
     New Bedford Church Street - Freetown 27 km, Providence/Stoughton Canton Junction -
     Pawtucket 39.7 km; Harlem Line 160 km against ~132): build_model's gap tracing. US
     questions left for Anita in us_sources.md: Lexington NC, Newport News old/new station,
     the state-fair stops (kept).
   - Note: the `<cc>_sources.md` files written before 2026-10-03 speak of "the xx branch of
     build_model.looks_like_service", EU_TRAIN_REGIONS, METRO_DUP_REGIONS and
     PLATFORM_NAME_REGIONS. Those rules are unchanged but now live in `rules/<cc>.py`.
   - **Answered 2026-10-04** (Anita): the annexed Donetsk/Luhansk railways go with Russia where
     trains run (a ru agent is on it); Abkhazia is its own region, `xa` (built: Psou - Sukhum
     - Guma running, Guma - Ochamchira greyed, Psou crossing a shared point `eXARUPSOU`);
     Transnistria is Moldova's, greyed. The original note follows.
     **For Anita (who the map depicts)**: the ua agent found a source for the 2022-annexed
     railways: poizdato.net lists daily Russian-run suburban trains in occupied Donetsk and
     Luhansk (Donetsk - Ilovaisk, Yasynuvata - Yenakiieve, Donetsk - Uspenska, Debaltseve -
     Rodakove; pages in data/raw/ua/poizdato/rozklad-elektrychky/). Her 2026-10-01 rule says
     "until a source says which trains run; then set ANNEX_RUNNING". Asked, not acted on.
     **Abkhazia** (Psou - Sukhum - the Inguri, ~230 tariff km) is clipped out of Georgia and
     built as no region's. Russian FPC trains from Moscow and St Petersburg and Sochi's
     "Dioskuria" electric trains run there daily, so the de facto rule would give it to Russia
     like Crimea (needs Abkhazia in Russia's outline and the Abkhazian part of Book 1's
     sections 57-001 and 57-007). Asked, not acted on. South Ossetia (no trains since 2008)
     is no region's and needs no decision. **Transnistria** (77 tariff km: Bender - Tiraspol
     - Novosavitskaya, Rîbnița - Colbasna) is no region's, Moldova's outline less OSM's
     relation 65335: no passenger train since CFM's Bender trains stopped in January 2025.
     The alternative is greyed as CFM's. Asked, not acted on.
   - **Open from the 2026-10-03 evening round** (each in its `<cc>_sources.md`):
     - extract.py keeps a construction way only under a route: Türkiye loses Adana - Ceyhan
       (48 km) and Karaman - Ulukışla (~110 km), rebuilt track OSM still maps as construction.
       Keeping them needs ab.py on every country.
     - Stale OSM routes: `SKIP_ROUTES` in a country's rules file now leaves out route
       relations by id (Brazil's Teresina routes, 2026-10-03; Malaysia's Skypark Link
       8391024/9985660 and Mexico's Línea Z 16929728 switched over from the named-train
       stopgap 2026-10-04, live at my's and mx's next rebuild). Stale OSM routes on
       no-train lines in mk, xk, al are still built as running OSM lines. South Africa flags
       ten stale Metrorail services the same way (`rules/za.py` `NAMED_NAME`), and proposes
       the general fix in ownership.py: a not-running register section keeps its ways from
       OSM lines (today an OSM route over it makes that track count as running: za's South
       Coast over Winklespruit - Kelso, 33 km). Needs ab.py on every country with greyed lines.
     - gtfs_served needs a scope (judge only sections in a feed's own area) before Indonesia's
       KRL feed or any city-only feed can be used: it closed 1,372 km of KAI's intercity
       network. Malaysia's and Thailand's feeds were tried and set aside (missing lines).
     - Ex-USSR borders (ru_sources.md "Borders with Belarus and Kazakhstan"): Russia now
       ends at all five Belarus points and eight Kazakh ones (`ru_register.BORDER`, `FOREIGN`
       for KTZ's track on Russian soil at Iletsk). Still apart: Кигаш, Озинки, Уютный (Russia's
       pieces drop with no OSM route over them though KTZ's timetable has trains); kz has no
       station at eXKZRU04, 05, 06, 17; Kaliningrad - Lithuania (eEU00252/253: Russia does not
       end there yet; the Moscow - Kaliningrad transit trains cross); Odesa - Izmail via
       Basarabeasca and Kuchurhan - Reni (no border point). About 40 Russian line names carry
       "(эксп.)" from Book 1's export codes (colours/ru.csv is keyed by name: rename both).
     - **Crossings with no OSM route over them** (2026-10-04): Johor - Woodlands, Futian -
       Hong Kong West Kowloon (the XRL; new point `xFutian`) and Pingxiang - Đồng Đăng are now
       one ride each because the far piece ships **under the neighbour's line id** (sg's piece
       is my's West Coast Line id, hk's is cn's 广深港高速线, cn's is vn's): the first register
       ids shipped in two countries; the app joins them by id (`mergeRegion`, per-country
       `parts`). **Hat Yai - Padang Besar done 2026-10-04, app-side**: the through-junction
       walk already steps over a border point onto the neighbour's register line once that
       country is loaded, so selecting a line now loads the countries a border point at its end
       names ("Malaysia – Thailand border", `lineRegionsToLoad`; outlines at 2 km could not
       tell at Padang Besar). The th line lists "Padang Besar, on West Coast Line, 0.5 km past
       the junction". Id-sharing is no longer needed for a new crossing. China - Russia
       (Manzhouli - Zabaikalsk twice a week, Suifenhe - Grodekovo daily) have no border points
       and ru stops at its stations.
     - **cn bug**: `cn_register.extend_ends` queries its k-d tree with the wrong latitude's
       cosine, landing 2-24 km off: 81 of 88 extend_ends sections jump more than 1 km at one
       end (滨洲线 哈尔滨北 - 哈尔滨 11.1 km). Two-line fix; needs a cn trial and rebuild
       (cn_sources.md "Still off").
     - gtfs_served's "weak" rule drops junction-ended sections whose only trains run more
       than 40 km between calls; in Iran it would have dropped Qazvin - Rasht, Yazd - Eqlid and
       Arak - Malayer whole (ir_register works round it with `served_sections`). A per-country
       setting in gtfs_served would be cleaner.
     - Indonesia: Whoosh's OSM relation is not merged as the register line's twin (same four
       stations, 141.1 km).
     - Croatia's single-country RINF border points (EU00221-EU00227) are missing from
       border_points.json: the table fetch probably drops them.
     - Bosnia's feed is the 2025 timetable: refetch when 2026's appears. Ukraine's crawled
       timetable covers about two months: recrawl monthly.
     - Thailand: a few 24-28 km sections may hide halts OSM lacks (th_sources.md).
     - The Chepe (Mexico) has 19 of ~30 stations; ARTF's station layer would fill it (its host
       answers 403 to scripts: a browser download for Anita).
   - **More countries** next: the Canary Islands for es; Vietnam, the Philippines, Argentina,
     Chile, Brazil's commuter networks, New Zealand, Israel, Morocco, Egypt, South Africa,
     Kazakhstan and Uzbekistan, Belarus and Moldova, Georgia and Armenia, Iran, Saudi Arabia,
     with the recipes above (OSM named track where `probe_kr_ways` is high; a national list
     through rinf.py's input format otherwise, as in, ua, id, balkans do).

1. **Germany, Italy and Spain** (built 2026-10-02; `de_`, `it_`, `es_sources.md`). Done
   2026-10-03: the timetable check is live in it and es (closed lines checked one by one, in
   each `<cc>_sources.md`); Italy's "Nodo di ..." lines split; every Swiss-German crossing,
   Konstanz, Selb, Kehl's tram, Tønder and the Öresund bridge share one border id
   (`borders.EXTRA`, `MOVE`, `SAME`); German freight bypasses 1280 and 1750/1751 greyed
   (`FREIGHT` in de.py; the Werntalbahn 5230 does run, RE 55 at weekends); DB InfraGO's data is
   CC BY 4.0. Open:
   - Germany: unchecked stop-to-stop sections with no OSM passenger route (2324 Rath - Opladen,
     2990, 6170 Berlin's ring, 6369, 2400, 2500, 5201, 5922; de_sources.md).
   - Spain's catalogue names and its two operators (Adif, Adif AV) stay as built (Anita: OK).
   - Italy: weak Frecce sections (Bologna San Ruffillo - Bivio Emilia 8.3 km) and Treviglio -
     Brescia need a high-speed rule in gtfs_served.
   - The Canary Islands (Tenerife tram) need Geofabrik's africa/canary-islands extract.
2. **Look at the border crossings in the app** (built, not yet seen on the map): K80 Kortrijk -
   Lille, the Eurostar Amsterdam - Paris, HSL 1. Open: fr's extract still builds
   ~30 short foreign sections in its buffer (IC-04, IC-26, L-29; 108 km in all); Longwy
   (202 000) stays 95% (its last 1.2 km has no trains and no node); ch's Feldkirch - Buchs
   border section is greyed as not running although ÖBB trains run there.
   Done 2026-10-02 (cleanup agent, `fr_sources.md`, schienennetz.py docstring): fr_register's
   dead ends end at the RINF border point under its `eEU` id (`snap_borders`, 30 ends; Basel,
   Portbou, Le Locle, Lyon - Genève now 100%); Swiss ends within 30 m of a point take the
   `eEU` id. **Crossings RINF files twice** (a point per country at one spot: bg EU00208 / ro
   EU00209, cz-pl EU00072/73, at-ch EU00118 / CH15472, ch EU00157 / CH15452, sk EU00162/163)
   are one id: `borders.canon` (within 15 m; eEU first, then lowest), applied to a register's
   output by `build_model.canon_border_ids` and to tracing by `borders.load(canonical_only=True)`;
   gtfs_served counts any table id as a border. Checked with tools/ab.py on at ch cz pl bg ro sk.
3. **After ownership** (cleanup agent, 2026-10-02): Seoul 1/3/4, Athens M3, the Austrian
   Railjets and T12 now credit their register lines; PKM's own 14 km is not in RINF (the OSM
   line owns it). Open:
   - **T13** on the Grande Ceinture: register line 990 000 is rail, T13's track light_rail, so
     990 000 owns 13.5 km no ride can credit. Needs a design choice in ownership.py (let a rail
     register line own light_rail ways where no other register line is there, or split it).
   - **France's missing lines**: done 2026-10-02 with Anita's OK. SNCF's line files lack them;
     its per-track file `data/raw/fr/voies-de-ligne.geojson` (39 MB, ODbL) has them, and
     fr_register reads it when present: +226 310 LGV Interconnexion Est 57.3 km, 262 000 Douai -
     Blanc-Misseron 30.1, 657 000 Lamothe - Arcachon 15.8, 768 300 Pasilly - Aisy 15.1, 958 000
     Bondy - Aulnay (T4) 7.8; LGV Est phase 2 on the finer track shape; 258 000's 0.6 km stub
     now lies on 262 000's rails and drops.
   - **Long connecting curves (Rac)** that TGVs use, ~100 km, count nowhere (excluded by design
     in fr_register); 48 exploited Racs of 3 km or more total 250 km.
   - **Geneva CEVA**: absent from the BAV file (2021-07-06 edition, still the only one).
     Lötschberg base tunnel fixed (330/331, 39.7 km; schienennetz now counts an end's
     neighbours, not segments). be 161A on 161's rails is correct: 1613 is 161's second pair.
   - fr: 894 000 CEVA-Annemasse 85% (a gap mid-line); 457 000 Segré - Nantes-État is a 2.4 km
     tram-kind stub (kept); fr's tram-trains on rail track (519 000, 782 000) now own their
     track. (fr's unnamed "Ouigo" and ru's 001/002 Красная стрела are named trains since
     2026-10-02.)
4. **Timetable follow-ups**: spot-check pl 281 Nakło - Chojnice (may have restarted 2026), ro
   600 and 700. Works closures (pl Opole - Nysa to 24 Oct, Jelenia Góra to 30 Oct, Cieszyn to
   13 Dec) stay greyed until the feed is refetched and pl rebuilt. Greece's feed ends
   2026-12-01: refetch then. pl 295 Węgliniec - border may be a summer works diversion.
5. **Russia**: done 2026-10-03 (ru_sources.md "English names and line colours"): English
   station names from Wikidata by ESR code (P2815) where the label reads as a romanisation of
   the Russian (41% of register stops), line English names from their ends (46% of lines), one
   picked colour per regional railway. Open: the line_colours.py `"*"` row the ru agent
   proposed (ru.csv 845 rows -> 18; ru_sources.md).
6. **Known general issues** (detail in `HISTORY.md`, "second EU round"): `rinf.line_hash`
   gives an unnumbered id the same line id as a public number with the same digits (Romania's
   Blaj - Praid; the fix needs line aliases for saved rides); `n02.walk_order` lists only the
   first piece of a two-piece line in `display` (Bulgaria's 3 and 4); a line with only an infra
   relation is no line and its track is dropped (Bulgaria's Septemvri - Dobrinishte, 125 km,
   Greece's Pelion, Zagreb's funicular).
7. Fixed 2026-10-02, live at the next rebuild: names differing only in a dash ("Charles de
   Gaulle-Étoile" / "Charles de Gaulle — Étoile") are one station (`fold_dashes`; ab.py on 24
   countries: only pl and lu move, one tram stop each, plus fr's 6).
8. **USA: built 2026-10-02** (`us_register.py`, `us_sources.md`; a country agent under
   maps-12): FRA NARN subdivisions, 448 register lines / 39,807 km, 863 lines with OSM's,
   5,722 stations; NARN's own chainage median 0.996, nine Wikipedia figures within 0.96-1.02,
   NEC Washington - Boston path 1.011. Amtrak's long-distance and once-a-day trains are named
   trains (`looks_like_service` us branch; Brightline is a line, hourly); the share of a US
   section ridden is measured on its own length (`ROUTE_SHARE_BY_LENGTH` in rules/us.py: OSM maps an Amtrak
   train both ways over one track). Holes in NARN's passenger coding filled from its other
   segments under OSM passenger routes (`--fetch-holes`, data/raw/us/narn_holes.geojson):
   way under named trains only, owned by nobody, 971 -> 182 km. **Second tracks** (UP's
   paired track, split tracks at Cajon, the New River...) are folded into their line where
   90% lies within 3 km (`companion_of`, us_register `COMPANION_KM`; ownership.py honours a
   declared companion, projecting within `DECLARED_FAR_M`): Anita, 2026-10-02, "both
   directions of a line one track generally, unless they're super far apart". Two stay
   lines (Austin, a 4 km Front Range piece). **Metro stations**: with `METRO_DUP` (rules/us.py)
   same-named metro stations merge within 150 m, not 500, and a metro stop node finds its
   station by name within 200 m: Manhattan's 23rd Streets on 8th, 7th, 6th Avenues and
   Broadway had merged into one. **Known fault, stations by proximity**: a station whose route
   runs beside another line for 300 m goes on that line too (Long Island City on Amtrak's West
   Subdivision gives a section Long Island City - Penn Station no train runs). Three rules
   were trialled and dropped (place_stations' comment: they took real stations off their
   lines, Tucson, Sacramento, Oshawa, Jamaica, Toronto Union, and one still left LIC on). The
   fix that should work: which register line owns the way a route's stop node lies on (stop
   nodes sit on the track the train stops on; 86% of US routes have them), which needs a
   way-to-line match before placement; Canada inherits it. Also: OSM lines keep
   their relation's stop list, which in the US is often partial ("Port Washington Branch (as
   operated)" lists 5 stops); give them the unlisted-station rule the register lines have.
   Open: the 182 km left; DART's Silver Line open?;
   NS Kansas City District (26.5 km, only via OSM's Southwest Chief); line colours (no
   colours/us.csv; OSM's names "Metro-North Harlem Line" do not match NARN's).
   **Canada, Sweden, Australia: built 2026-10-02** by three country agents in parallel (their
   `<cc>_sources.md` have the detail and open questions). Canada reuses us_register through
   `ca_register.adopt()` (it repoints us_register's settings; renaming one breaks Canada
   loudly); 2,373 km NARN still codes passenger but no train runs on (Algoma Central, ex-BC
   Rail, Ontario Northland north of North Bay until the Northlander returns) is dropped.
   Sweden's RINF ids are Trafikverket's stråk numbers, named from its network statement and
   sv.wikipedia; its timetable feed is live (Mälartåg has no OSM routes at all); the Öresund
   bridge stops at Lernacken until Denmark is built. Australia: GA has no ROUTENAME, only
   each state's line `name`, no passenger flag, every track drawn; OSM routes decide
   passenger track; platform-numbered stop names ("Box Hill 3") are read without the platform
   (`PLATFORM_SUFFIX` in rules/au.py); the Ghan has no app line (OSM lists one stop) but its track
   counts. (India, the UK, Norway, Denmark: built 2026-10-03.)
9. **Loading by zoom** (Anita, 2026-10-02: "start zoomed out and see everything, and only load
   detailed data for the places we zoom into"). **First step done 2026-10-03** (app agent):
   line data loads from z8 (`DATA_MIN_ZOOM`), for countries on screen plus 10%; country totals
   come from regions.json `km` (build_regions `owned_totals`, a Python copy of the app's
   ownTrack + regionTotals: change both together); not-running track for every country comes
   from `dist/data/closed.json` (written by build_regions; **upload it when publishing**). A
   track click or a closed-line click on a country not loaded loads it first. Europe z4: 33 MB
   of JSON, 158 MB heap, track at 5.3 s -> nothing, 23 MB, 1.3 s. **Search index done the
   same day**: build_regions writes `dist/data/search.json` (every line and station name; 6.0
   MB raw, 2.0 MB gzipped), loaded on the first keystroke; a hit loads only its own country.
   A world-view search answers in ~250 ms (before: nothing found; Europe z4 2.8-5.9 s, 34 MB).
   **Upload search.json and closed.json when publishing**; without search.json the app falls
   back to loading the countries on screen. Open: operator rows still need their country
   loaded; within a score tie a long line outranks a station ("paddington" lists 13 GWR lines
   before London Paddington). Next: split Russia's data by area (any z8 view in Russia loads 11.7 MB; Moscow's 2° cell
   would be 3.8 MB); a merged overview tileset is not needed (world view 86 small requests,
   0.77 MB, ~1 s). The lint now also checks the whole script parses.
10. **App**: on a phone the panel is now a bottom sheet (2026-10-04): 52% of the screen, the
   map shrunk to the top half above it (`body.sheet`, `layoutSheet`, `map.resize()` keeps the
   centre), "expand"/"shrink" for the whole screen, the legend hidden while it is open. Not
   yet seen on a real phone. A headless `?region=kr` once opened on
   Japan (FOCUS 'jp'), probably a race between `regions.json` and the START check. The rest is
   spec §13, item 3 onward.

## The process for shared changes

- **Trial first.** `python tools/ab.py <cc...>` builds the working tree into a temp folder and
  compares with `dist/data` (lines, km, sections, closed, station ids, whether foot.json and
  ways.json are byte-identical); it writes nothing in the project. Run it with the change in
  the working tree, before rebuilding, only on the countries the change can touch: a change
  gated on one country (`if region == "hr"`, a COUNTRY key nobody else sets) needs no run on
  the others. A generic change that moves another country's output gets scoped or declined;
  record a declined one with its numbers in `HISTORY.md`.
- **Batch** shared changes into one rebuild.
- **After any shared-file change**, the real rebuild: `python tools/compare_lines.py save
  <cc...>`, `python tools/rebuild.py <cc...>` (model and tiles, 3 countries at once, `-j 1` for
  one, `--model-only`; logs in `data/logs/rebuild_<cc>_<step>.txt`), `python
  tools/compare_lines.py diff <cc...>`, then `python tools/build_regions.py`.
- **`tools/build_regions.py` after every rebuild, not only a first build**: since 2026-10-02 it
  also publishes `line_aliases` in regions.json, every country's twin merges (aliases.json
  `lines`) folded across countries, so a line one country merged and another kept apart still
  joins over the border (the Eurostar: fr kept "Eurostar: Paris - Amsterdam" m5189990 apart,
  be and nl merged it into m5189989, so Paris Nord was cut off at the border). It only folds
  into an OSM line, a piece of 5 km or more, with the same name (untrained) or same first
  word and ref, and prints what it folded and refused. The app folds every line id through it
  on load; two lines of one country folded together become one piece. Below z10, a click's
  "Also on this track" now comes from the footprints (foot.json), not station chords, and a
  long-chord line is measured on its drawn geometry.
- **After a country's first build**, rebuild its built neighbours too, so `split_at_borders`
  hands that country its side of each crossing.
- `check_model.py --region <cc>` after any model change: register lines stay near 1.00 of their
  published lengths, OSM-derived objects within 2%. `python -m unittest discover -s tests`.
- Several extracts at once: `OSMIUM_POOL_THREADS=2` each (extract.py defaults to 4).
  `rebuild.py` sets `OMP_NUM_THREADS=2`, so three countries stay within the ~6 cores builds may
  take; ru peaks near 3 GB, cn and fr near 2 GB.

**Running one country**: `python extract.py --region <cc> --pbf <file>`, `python
build_model.py --region <cc> --register <arg>` (every country's arg is in `tools/rebuild.py`'s
`REGISTER`; any other code is RINF, `rinf:data/raw/rinf/<cc>`), then `build_tiles.py`, which
reads line colours from the model, then `check_model.py`. After every extract, clip where the
reader clips: `hk_register.py --clip`, `sg_register.py --clip` (sg's extract takes `--bbox
103.6,1.2,104.05,1.452` from Malaysia's file; hk's comes from openstreetmap.fr),
`cn_register.py --clip`, `ru_register.py --clip`; Russia then `ru_register.py --convert`;
`gb_register.py --clip` and `fr_register.py --clip` (each drops the other's half of the
Channel Tunnel, 2026-10-05).
Added 2026-10-03: `python -m rinf_countries.ie --clip` (the extract holds Northern Ireland,
which is gb's), `my_register.py --clip` (Singapore, Brunei, Thai stubs; extract with
`--station-areas`), `balkans_register.py --clip ba` (the Belgrade - Bar line's 9 km through
Štrpci stays Serbia's), `ua_register.py --esr`, `--disused` (both read the .pbf) and `--clip`
(Crimea and the annexed area out; frontline track OSM retags disused put back), and
`tr_register.py --construction <pbf>` then `--clip`. Ukraine's timetable is crawled
(`ua_register.py --timetable`, ~70 min, monthly; ua_sources.md), not fetched.
Each `<cc>_sources.md` has its commands, `HISTORY.md` the 2026-10-01 list with timings (cn,
ru, fr take 6-11 min). `.osm.pbf` extracts are deleted once extracted and checked;
`data/proc/<region>/` holds all a rebuild short of re-extracting needs (use Geofabrik's dated
`<country>-YYMMDD.osm.pbf` if `-latest` redirect-loops).

## Adding a country

The pipeline is region-agnostic; only the register reader and the check tables are not.

1. **OSM half.** The Geofabrik extract (a new download: Anita's say-so), `extract.py`, then
   `python inspect_region.py --region <cc>` before trusting any of it (are `usage` tags kept,
   how many route relations carry a colour, how much of the network a passenger route covers).
   Delete the `.pbf` afterwards; keep one at a time. `python probe_kr_ways.py --region <cc>`
   gives the share of main-line km whose `name` is its line (Korea 98%): if high, Korea's
   recipe works without a geometry register.
2. **The register** (spec §12c: FRA NARN for North America, ERA RINF for the EU, GTFS via the
   Mobility Database as the broad fallback, Wikidata as the glue). Either a module beside
   `n02.py` with `build(path, log) -> (lines, stations, geoms)`, run as `--register
   <module>:<path>` (`kr_register.py` is the template for named-track countries), or for RINF
   **one file `rinf_countries/<cc>.py`** defining `COUNTRY` (rinf.py's docstring lists the keys
   and hooks, each a no-op unless set; be.py, at.py, nl.py are worked examples), then `python
   rinf.py --fetch <cc>`, an extract and `--register rinf:data/raw/rinf/<cc>`.
3. **Check it.** Published line lengths in `REGISTER` in `check_model.py`, passing. A build
   with no outside number checked against it is not finished. A register with its own
   chainage (`km_official`) checks every line.
4. **Put it on the map**: `tools/build_regions.py` after the first build (managing session),
   then rebuild the built neighbours. The app needs no change; a country missing from
   `regions.json` only loads when someone has rides there. Outlines come from religiondots'
   `country_shapes.geojson`, else Natural Earth 1:10m (Luxembourg).
5. **Colours.** `colours/<cc>.csv` (`line,operator,colour,source,url,note`), applied by
   `line_colours.py` over OSM's and Wikidata's. Official sources first (route-map PDFs: read
   the vector fills with PyMuPDF, not pixels); a widely used map where the operator publishes
   none (Korea); mark our own choices `picked`. `python line_colours.py --fetch <cc>` after
   adding it to `COUNTRY` there gives the Wikidata fill; an operator's corporate colour goes in
   `GENERIC`. So far jp, kr, tw, hk, sg have a CSV and jp, kr, ch a Wikidata fill.
6. **The country's own rules** in `rules/<cc>.py`, which the country agent owns:
   `looks_like_service(tags, name, name_en)` (a single long-distance or international train is a
   named train; an interval product a rider uses as a line stays a line) and the other hooks
   `build_model.country_rules()` lists; shared patterns (EU_TRAIN...) in `rules/shared.py`. Via
   the managing session: `norm_line_name` prefixes; a `rinf.py` hook may be added by the
   country agent if it is a no-op unless its own COUNTRY sets it. After a first build, read
   the `drop_islands` line of the `build_tiles` log (ways left out, by kind): a real line
   missing from the model shows up there as vanished track.

### What `build()` returns

```python
lines = [{"id": str,       # stable across builds; hash the name and operator, never an index
  "src": "n02",            # any non-"osm" value marks a register line, which is what counts
  "service": False,        # True only for a named train rather than a line
  "name": str, "name_en": str, "ref": str, "colour": str,
  "operator": str, "operator_en": str, "network": str,
  "kind": str,             # rail | subway | light_rail | tram | monorail | funicular | narrow_gauge
  "km": float,             # the sum of its sections
  "variants": int, "straight_sections": int,
  "display": [station_id, ...],                   # reading order for the strip diagram
  "sections": [[station_id, station_id, km], ...]}]   # build_model adds a 4th element
stations = {id: {"id", "name", "name_en", "lon", "lat", "lines": set()}}  # + "junction": True
geoms = {line_id: {"<from>|<to>": [[lon, lat], ...]}}                    # one per section
```

Optional on a line: `"highspeed": bool` only if the register knows (it then only matches OSM
ways with the same `highspeed=yes`, keeping the Shinkansen and the conventional line apart), or
per section `"highspeed_sections": {"a|b": bool}`; `"guided": True` for a guideway (never
shares track with trams); `"km_official"` and `"chain": {"a|b": km}`, the register's own
chainage, which `check_model` compares every line against. `"junction": True` marks a section
end that is not a stop: such a section is kept only if OSM passenger routes run over at least
half of it (`drop_unridden_sections`) or, in a GTFS country, trains do; that keeps base tunnels
and drops freight curves. A section between two stops is never questioned. Then
`merge_sources` matches OSM onto the register, `register_way_lines` ties register lines to OSM
ways (and corrects their kind), `ownership.py` writes `foot.json`, and `not_running.py` marks
sections with no drawn track `closed` (its `SOURCES`; RINF countries use `suspended` and the
GTFS check). `HISTORY.md` "The reader contract's notes" has more.

## Working on the app

`dist/index.html` is one file: constants, map setup, data loading, rides, crediting, then the
panel. Things that will bite:

- **Paint expressions must stay plain literal `const`s in SCREAMING_CASE.** `node
  tools/lint_map_expressions.js dist/index.html` evaluates each `addLayer` with those hoisted;
  run it on every edit. It catches an invalid expression that would silently drop a whole
  layer, and a top-level name declared twice (a leftover second `function goRegion` won
  silently and the page reloaded forever). `['*', ['interpolate', ['zoom'], ...], factor]` is
  invalid: `['zoom']` may only be the direct input of a top-level `interpolate` or `step`.
- **Every layer the page adds goes over the whole basemap, its labels included** (Anita,
  2026-10-04: place names under the lines; `carryOver` keeps that on a theme switch); each
  country's track goes in under `sel-halo`, and only the page's own `station-label` is above
  the lines. Seas' and lakes' names are hidden (`baseHide`). The basemap's own near-black
  railways are hidden on load. Station bubbles come from the model, never from the tiles.
- The Countries tab lists the most ridden first, then by km of line (Anita, 2026-10-04).
- **A country's or operator's total is the union of its lines** (`unionTotals` /
  `unionDone`, Anita 2026-10-08): never sum owned track or per-line km for a group, or riding
  every line stops short of 100%. `build_regions.owned_totals` is its Python copy for
  regions.json; change both together. `line_100_probe.js` part 3 checks it.
- **No CSS transitions anywhere**: state changes are instant.
- **A reload opens with nothing selected** (Anita, 2026-10-02): only the home tab survives
  (`bootUi`); a line, station, trip or traced journey from last time muted every other line.
- **Past a junction end** (`pastEnds`, `pickPast`; Anita, 2026-10-02): a register line
  ending at a junction shows "→ <stop>" rows past it, the stops an operating pattern's
  section reaches from one of this line's stops over the very end of its last section (by
  foot.json). A ride to one is two legs: this line to the boarding stop, the pattern on (US:
  LIRR Port Washington Branch, Great Neck → Woodside). Switzerland: 110 of 277 lines ending at
  a junction get one. Cutting the register at such junctions instead was tried in
  us_register and dropped (it moved sections on double track).
- **Through a junction** (Anita, 2026-10-04: lines ending at junctions left "nothing to
  click"): `throughEnds` walks on from a junction end along register lines of the same kind
  (no named trains, no turn sharper than 100°, up to 6 sections / 80 km) to the first stops,
  shown as "→ stop" rows and faint rings beside pastEnds'. A junction usually lies mid-section
  on the line it meets (87% in Germany), so a ride from it carries `cuts` and credits only that
  part of the section (`gid~lo~hi` parts, `spanOf`). A trip through a junction is joined rides,
  one per line, and a junction may end a ride only where the next one carries on.
  `junctionRoute` lets tracing find a route changing only at junctions when no single line
  joins the two stops. **Only ways on trains take** (Anita, 2026-10-04, "a bit aggressive"
  at Hanau): a way on is listed only if a mapped service (OSM route, named trains included)
  runs over both this line's end at the junction and the way's first leg (`wayRun`,
  `servicesAtEnd`, from the footprints); with no service over the end, or where that would
  leave the end with nothing, the whole walk stays. de 283 -> 230 rows, ch 386 -> 323.
  Also (Anita, 2026-10-05): **a junction end within `HUB_THROAT_KM` (1.5 km) of the line's
  last stop offers nothing one line takes from that stop** (3700 Gießen - Fulda now just
  ends at Gießen, 4113 at Hanau Hbf; `hubStop`); **a way on that runs past a nearer listed
  stop goes** (`passesBy`, applied across service and walk rows together in renderLine's
  `contGroups`; the MML's north end offers only Swinton, the services' Doncaster and
  Wakefield Westgate lying beyond it); **a junction the line
  only passes through gets no row** (`.st.quiet`: degree 2 in the piece, not an end, nothing
  past it; the WCML hides 15 of 18), nor **a short dead-end stub's junction** (`.st.stub`:
  degree 1 inside the piece, no track drawn to it; the MML's curve near Cricklewood).
  **One list for diagram and map**: `contRows(line, j)` / `contEnds(line)` are the stops past
  a junction end as shown, used by renderLine's `contGroups` and paintSelection's rings (they
  diverged once, 2026-10-05); paintSelection also draws each continuation's track from the
  junction to its stop as `base` in the reached line's colour (Anita: a stop past a junction
  should not float away from the line). Labels: "junction", "on <line>".
- **Straight-line gaps are left out of ride routing** (`straightGaps`, `lineGraph`; Anita,
  2026-10-05, a ghost ridden line Hoboken - Penn Station): on a line with
  `straight_sections`, a section drawn as one straight segment whose ends the line joins
  otherwise only by over 1.5x its length (NJT Morris & Essex: Hoboken - New York Penn,
  Secaucus - Hoboken). Waits for the line's geometry, then redoes its rides (`GAPS_GEN`).
  The build still makes these sections; 15 of 2 km or more in the US alone. The strip diagram draws those stops as its own rows past each junction
  end (thin lanes in the reached line's colour, one lane per line), so a line with no stops
  of its own still shows two stations to ride between. Coverage of junction ends with a stop past them: ch 123 -> 364 of 419,
  de 676 -> 1,373 of 1,514, us 219 -> 594 of 878. Russia's junction ids often twin a station
  (1,152 of 1,371 junction ends within 300 m of a stop): merge them in the ru build.
- **Track no line runs on** is drawn faint grey (`#303030` dark, `#d2d2ce` light, never
  clickable, a legend row "No passenger line"): `build_tiles.mark_no_line` sets `n = 1` on
  drawn ways missing from ways.json (Anita, 2026-10-04, after Mexico's freight network drew
  like passenger rail).
- **The native name is the real name.** Someone riding trains in a country reads the names on
  the trains, and those are not in English. `lineName()` and `stName()` show an English name
  where one exists and fall back to the native one, and that fallback is correct: never
  substitute an id, a transliteration or a placeholder. 61% of register lines have no English
  name and the app has to read well anyway.
- **Ridden and muted states DARKEN line colours inside the expression**, never lower opacity:
  double and quadruple track are stacked ways whose opacity adds up to nearly full strength.
- **Never change a data-driven paint property on a state change that happens often**: in
  MapLibre 4.7 it re-parses every loaded tile of the source. Muting swaps two layers'
  constant opacities (`track-<cc>` / `track-mute-<cc>`, `applyTrackColour`); the colour
  expressions change only with the theme and Browse/Ridden. A constant opacity must stay
  constant (constant <-> data-driven reloads too).
- **Keep the main thread free** (2026-10-08, Anita: "freezy for the first 3 or so seconds",
  and on selecting a line). Measured with `tools/perf_probe.js` (headless, CPU profile + long
  tasks; numbers in `handoff_notes/app_perf.md`). What it changed, and what not to undo:
  - MapLibre checks every `addSource`/`addLayer`/`setFilter`/`setPaintProperty` by
    serializing the WHOLE style, so with ~120 countries adding them was cubic (8.7 s in one
    task at the world view). Per-country layers go through `addTrackLayer`: only the first
    country is checked, later ones pass `{validate: false}` (the lint now reads
    `addTrackLayer({...})` too, and is their only check). Each is created with the current
    theme/mode/muting values (`trackNow`), noted in `TRACK_SET`, and `applyTrackColour` only
    sets what differs from that note. Countries are added a few per task, nearest the view
    first (`queueRegionTiles` / `drainRegionTiles`, 30 ms budget).
  - A country's arrival only merges its data; the redraws (stations, ridden, selection,
    panel) run once per burst (`dataArrived`, 80 ms), then `warmIndexes` builds the walk
    indexes and OPS in idle time. `OPS` is built when read: call `opsNow()` before reading
    `OPS`/`OP_INDEX`. `along.json` is kept raw and expanded per country on first lookup
    (`ALONG`/`ALONG_INTO` are small objects with `get`/`clear`). `renderHome` calls `stats()`
    only on the tabs that show it (not Trips).
  - `passThrough` and `nearRows` look sections up in a 0.25° grid (`sectionsNear`, built
    per DATA_GEN) instead of scanning every loaded section per junction; `hubStop` rules out
    stops farther than `HUB_THROAT_KM` with one search from the junction. Same rows as
    before on de/gb/ch/us/jp (old and new functions swapped in one page), 10-23x faster.
  - Any new loop over every loaded line or station on a click, a selection or a country's
    arrival should be cached per DATA_GEN or indexed like these.
  - **The selection has a MapLibre worker of its own** (2026-10-08, Anita: a line sometimes
    took ~600 ms to draw, "feels random"). MapLibre 4 parses all tiles and GeoJSON in one
    worker, so the selection's setData queued behind tiles in flight after a pan (0.2-1 s).
    `setWorkerCount(2)` (before `setRTLTextPlugin`, which starts the workers) and
    `selectionWorker` puts `sel`/`selst` alone on the second; everything else stays on the
    first. Patches 4.7 internals (`Dispatcher.getActor`, `GeoJSONSource._updateWorkerData`):
    recheck on a MapLibre upgrade. `opLinks` no longer builds OPS for a listed line.
    `tools/perf_probe.js selects` measures click to drawn; numbers in handoff_notes/app_perf.md.
  - **The dimming lands with the selected line, never before it** (2026-10-09, Anita: "could we
    prioritize redrawing the selected line"). Do not call `setMuted` directly: `paintSelection`
    asks `wantMuted`, and the mute goes on when the `sel` source's new tiles are loaded
    (`selDrawn` / `selLinesDrawn`; cap `MUTE_HOLD_MS`). A clear undims and hides the selection
    layers by constant opacity (`selHidden`) in one frame. Selection data goes through `selSet`
    (one setData in flight). `showLine` paints before it renders the panel. A newly selected
    line goes out alone first only when the rest would wait on a fetch or uncached walks.
    `perf_probe.js frames` shows it frame by frame (app_perf.md).
- **Browse draws a selected line at full colour, ridden or not** (Anita, 2026-10-09): the held
  back colour and the `done` overlay mean "not ridden", which is the tracker's (`drawLine`,
  `heldBack`, `drawStrip` check `MODE === 'ridden'`). **"View all lines" shows no ridden state
  at all** (same day): the ridden overlay goes to opacity 0 there (`riddenShown`, a constant
  swap, called by `setMode` and `setMuted`; its data is kept, so switching back is instant);
  the line view has no "Trips on this line", the operator view no ring or "rode every line",
  a country's operators sort by length, and the strip's closed stops are grey whether reached
  or not. Station bubbles never carried ridden state. The Trips tab stays (Anita, 2026-10-07).
- **Line types** (`types.json` per country, `tools/line_types.py`, line_types.md; read in
  `fillTypes` before LINE_FOLD): the type leads the line view's small line ("Commuter rail ·
  Metro-North · 109.0 km") and a search hit's. The muted "Intercity trains also run here."
  went (Anita, 2026-10-09); `lalso` is still read, nothing shows it. search.json has a `types`
  column (build_regions `write_search`) for countries not loaded.
- **Station ridership** (`riders.json` per country, `riders_sources.json`,
  `tools/station_riders.py`, station_riders.md): one small grey line under a station's name,
  "450,000 passengers / day" (Anita, 2026-10-09: no "about", no year in the text; two
  significant figures, "2.7 million passengers / day" from a million up; "/ weekday" where
  every source counts working days, as NS does; nothing when there is no figure). The source
  and year are the hover (`title`), and the line links to a single source. riders.json is
  fetched on the first station view of its country (`fetchRiders`), never with the lines, and
  the view re-renders once when it lands.
- **A mode switch redraws everything the mode decides at once** (Anita, 2026-10-09: it "doesn't
  correct whats dimmed and whats not until you actually click a line"): `setMode` calls
  `applyTrackColour` (the track's colour expression and edge differ per mode, and only
  muting called it), `riddenShown`, `render`, then `paintSelection`. A colour expression
  changing reloads the tiles: fine on a mode switch, as on a theme switch; never on a selection.
- **Routes in the US, Canada and Australia** (`ROUTE_CCS`, `routesOf`, `listed` vs
  `listedBase`; Anita, 2026-10-09, see the standing decisions). In these countries `listed()`
  is: every OSM line of the country with track (a ROUTE: metro and tram lines as before, and
  operating patterns and named trains, now with percentages), plus a register line that is a
  CORRIDOR or that the routes leave uncovered. `listedBase` is the old rule and still decides
  which register lines own track (`openTrackLines`, `joinedBy`, `endJoins`). All measured on
  footprints, as unionTotals counts track (`pieceFoot`):
  - Covered: the routes' union leaves at most ROUTE_GAP_KM (0.15) or ROUTE_GAP_SHARE (1%) of
    the register line's footprint. What a hidden line leaves (a few hundred metres past a
    terminus, a siding) drops out of the totals, as an operating pattern's register gaps
    always did: us 33.6 km over 86 of 337 hidden lines, ca 10.1 km, au 32.7 km. Everything
    else stays listed (us 124 register lines, ca 30, au 71), so riding every listed line still
    reads 100%.
  - Corridor: 20 km or more, at least 3 routes of at least 2 operators each over a quarter of
    it or more, tourist trains aside (types.json). Many routes of one operator (Amtrak's
    Florida trains, au's duplicated XPT relations) do not make one. As of 2026-10-09: us,
    Northeast Corridor (both pieces), Keystone Corridor, RF&P, Hudson Line (both: Metro-North's,
    and Amtrak's north of Poughkeepsie with Mohawk, Rochester and Buffalo Terminal
    Subdivisions, the Empire Corridor, because the Maple Leaf is VIA's too), New Haven Line,
    Hartford Line, Washington District, Miami and Orlando Subdivisions, San Diego, Orange,
    Ventura, San Bernardino and River Subdivisions (LOSSAN and Metrolink), Seattle and Lakewood
    Subdivisions, East Subdivision (Boston), Chicago, C&M and Joliet Subdivisions; ca, Oakville,
    Kingston, Weston (Malton - Bloor) and St-Hyacinthe Subdivisions; au, Main Suburban, East
    Hills, North Shore Railway, Main Northern (Asquith - Cowan), Main Western (Orange East Fork -
    Dubbo), North Eastern (Springhurst - Broadmeadows) and Melbourne's Pakenham Line.
  - Single: a covered register line exactly one route runs over (at least 0.5 km or 5% of it;
    us 224, ca 73, au 23). A click on its track opens that route, with "Track: Nevada
    Subdivision" under the name (`VIEW.track`); search and the Lines tab still open the
    register line itself.
  - A route's line view says "Track:" with the register lines it runs over in the order it
    reaches them (`routeTracks`), folded; a register line says "Routes on this track:", the
    routes most of whose length lies on it first. Rows in lists note "corridor" or "track";
    track rows come after the routes and corridors in the default order (`lineListOrder`).
  - A route with no operator tag goes under its network (`routeOpKeys`; "CTrail", "SunRail";
    `ROUTE_NETWORK` maps ARR and "VIA Rail"), else "(no operator)" as before.
  - Names (`lineName`, display only): a route there drops "(as operated)" (its register twin is
    the "track" row beside it); other bracketed parts of OSM names there are real ("MBTA Green
    Line (D)") and stay. A ref of one letter and one or two digits goes before a name that
    lacks it (`PUBLIC_REF`): "T1 North Shore & Western Line", "M1 Metro North West &
    Bankstown Line", "L2 Randwick Line". Longer refs are internal codes (Melbourne's "FKN") or
    abbreviations (Metra's "UP-N", Amtrak's "CZ") and are not shown.
  - Totals: us 43,069.8 -> 43,127.1 km, ca 13,970.2 -> 13,983.1, au 17,192.8 -> 17,207.5 (route
    track no register line owned in, slivers out). `build_regions.owned_totals` has the same
    rule (`route_rows`, `ROUTE_CCS` and the constants copied; change both together), and
    matches the app to 0.0001 km for each country loaded alone (2026-10-09).
  - `line_100_probe.js` tests every listed line (routes included) and prints `routes` per
    country; us/ca/au groups exact and 100% once every listed line is ridden (2026-10-09).
- **A new colour goes in both themes** (the two `:root` blocks, and `THEMES` for map paint and
  the strip's SVG). Layers are added with the dark theme's literals so the lint checks them.
- **To see a change**: `python serve.py`, then `node tools/screenshot.js
  http://localhost:8767/ out.png 15000 probe.js` with Chrome headless on its own
  `--user-data-dir` (exact invocation in `tools/screenshot.js`'s header). **Use a private debug
  port** (`CDP_PORT=9341`): 9222 is shared by every session, and a second Chrome on it silently
  lands on someone else's browser. A probe file can drive the map and assert against it, which
  is how every UI change here has been checked; wait on the map's `idle` with a timeout, or a
  probe can hang. Stop only that Chrome.

How it is put together (countries loading as the map pans, per-country sources and layers,
track clicks, the strip diagram's layout): `HISTORY.md`, "The app as of 2026-10-02".

## Data files, what is gitignored, publishing

`dist/data/<region>/` holds `lines.json`, `stations.json`, `foot.json` (track ownership: what
riding each section credits, [owner section, from, to, a, b]), `ways.json` (lines per drawn
way, owner first), `aliases.json` and `geom/<line>.json`; `dist/data/<region>.pmtiles` is the
tiles. All generated and excluded by the repo's `data/` rule, as are `data/raw/`, `data/proc/`
and `data/logs/`. Tracked: source, `dist/index.html`, `dist/regions.json`, `border_points.json`.

**`aliases.json` matters more than it looks.** Station ids move when a build merges OSM onto a
register, and saved trips name stations by id. The file ships the moves and the app migrates
saved rides on load; `carry_aliases` maps every id the previous build shipped (old alias, same
name within 500 m, nearest within 200 m) and logs what had nothing in reach. A change that
moves ids without updating it silently voids people's history. Its `lines` map does the same
for line ids dropped as a register twin.

**Publishing** (to live at anita.garden/noritetsu) must upload foot.json and remove any old
credits.json; every country needs a foot.json before index.html goes up (without one the app
counts that country's OSM lines whole). The .pmtiles need a server that answers HTTP range
requests (why `serve.py`, not `python -m http.server`). **`python tools/deploy.py`** does it
(first run 2026-10-09; README.md "Deploy"): data and regions.json to R2, the page and its two
scripts to the website repo, the page switching between the two by `regions.json` beside it.
A new file the app fetches needs nothing extra if it is under `dist/data/`; anything outside
it has to be added to deploy.py.

## Pointers

- `spec.md`: the design; §12c the registers per country, §13 the todo. `README.md`: how to run.
- `HISTORY.md`: how things got this way, declined proposals, the traps, the session logs.
- `<cc>_sources.md` (the Baltics share `baltics_sources.md`): sources, commands, what is off.
  `multi_sources.md`: sources covering many countries. `gtfs_sources.md`: the timetable feeds,
  and "Rollout (2026-10-01)" with every closed line per country.
- `border_proposal/PROPOSAL.md`: the cross-border design, with an "As built" section.
  `ownership_prototype/WRITEUP.md`: track ownership's prototype and comparison.
- Docstrings: `rinf.py` (COUNTRY keys and hooks), `gtfs_served.py` (the check and its fixes),
  `ownership.py` (who owns a way), `n02.py` and `kr_register.py` (traps that cost a rebuild),
  `tools/rebuild.py`, `tools/ab.py`.
