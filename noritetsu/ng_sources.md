# Nigeria (ng): sources

## Build (2026-10-08)

    python extract.py --region ng --pbf data/raw/nigeria-latest.osm.pbf --station-areas
    python wafrica_register.py --clip ng       # other countries' track, NOT_SERVICE routes,
                                               # track gaps joined, gauge breaks split
    python wafrica_register.py --fill ng       # Iddo; names Ebute Metta Junction
    python wafrica_register.py --convert ng
    python build_model.py --region ng --register wafrica_register:data/raw/rinf/ng
    python build_tiles.py --region ng
    python check_model.py --region ng

Built: **8 register lines, 808 km; 6 running (764 km), 2 greyed (Abuja metro, 44 km)**, plus
LAMATA's Blue Line (11.8 km) as an OSM line: it is mapped railway=subway, which rinf.py's
track graph does not take, and its OSM route (15668733/4) lists its five stations.

| line | km built | published | stops |
|---|---|---|---|
| Lagos – Ibadan | 154.8 | 156.8 | NRC's 9, listed (Agege is OSM's LAMATA-tagged "Agege" at the SGR's station) |
| Abuja – Kaduna | 183.9 | 186.5 | Idu, Kubwa, Jere, Rijana, Kakau, Rigasa, listed: from Idu to Kubwa the trains run on the metro Blue Line's track past its halts. Asham and Katari are missing (no OSM station, no coordinate found) |
| Warri – Itakpe | 310.0 | 326 (with the Warri port end) | the 14 of OSM's NRC route |
| Port Harcourt – Aba | 62.3 | 63 | the two ends only: no list of the intermediate halts found (the press names Elelenwo and Imo River) |
| Iddo – Ijoko | 28.7 | none found | Iddo, Ebute Metta Junction, Yaba, Mushin, Oshodi, Ikeja, Agege, Iju, Agbado |
| Red Line | 24.4 | 27 (LAMATA) | its 8, listed |
| Yellow Line, Blue Line (Abuja) | 26.5, 17.8 | 45 for both | greyed (`suspended`) |

Decisions and repairs:
- **Lagos's three railways.** Ebute Metta – Agbado has NRC's Cape gauge single track and the
  standard gauge double track 90 m apart. OSM ties the two gauges together at 20 nodes, and
  every station snaps to both tracks, so traces jumped between them (Mobolaji Johnson – Agege
  came out as 14 km of Cape gauge). Two fixes in the reader, no shared-file change:
  `--clip` gives each gauge its own copy of a node two gauges share (`split_gauges`), and
  `wafrica_lines.OWN` makes 1067 track the Cape gauge line's own, the "Lagos–Ibadan SGR"
  named track the SGR's and other standard gauge the Red Line's, through rinf.py's
  existing `way_line` hook (pass 2 prefers a line's own track).
- **The Red Line and the SGR** each end up on one of OSM's two standard gauge tracks, so
  they are two lines that do not credit each other. That matches LAMATA building its own
  tracks along the corridor (en.wikipedia: it "shares the right-of-way"); OSM has not drawn
  them apart. If OSM later maps four standard gauge tracks, nothing here needs changing.
  The Red Line's last 2 km into Oyingbo run on the Cape gauge track in OSM (its own 1435
  track there is mapped only in pieces); 0.9 km of it counts as Iddo – Ijoko.
- **Iddo – Ijoko is drawn to Agbado plus 2.6 km**: OSM's Cape gauge track stops there and
  resumes only at Ijoko's sidings 4 km on. Traced over the gap it took the SGR beside it, so
  the line ends at a junction where OSM's track ends (`~Cape gauge track ends (OSM)`).
- **Track gaps**: `--clip` joins two dead ends of one gauge within 60 m that continue each
  other (`join_gaps`): the SGR north of Papalanto (a bridge's end 23 m from the next way)
  and the Cape gauge at Agbado (56 m). Without the first, Papalanto – Abeokuta had no trace.
- **OSM routes dropped** (`NOT_SERVICE`): the Port Harcourt – Kano train, the Kano SGR
  (under construction), Port Harcourt's monorail (never opened), the Abuja metro's routes
  (its lines are greyed register lines), and the three NRC routes the register lines are
  under other names (merge_sources twins by name, and "Itakpe - Warri Rail Line" is not
  "Warri – Itakpe": each was drawn a second time).
- Red Line colour #E30613 and the Abuja lines' yellow and blue: picked to match the line
  names, no official values looked up.

## Survey (2026-10-08)

Research only; nothing built, nothing downloaded beyond small Overpass answers. Nigeria has
three standard gauge intercity lines, two Cape gauge commuter services and two Lagos metro
lines running; the old Cape gauge trunk lines (Lagos – Kano, Port Harcourt – Maiduguri) are
mostly idle.

### What runs (freshest evidence)

| line | gauge | operator | status | evidence |
|---|---|---|---|---|
| **Lagos – Ibadan** (LITS): Mobolaji Johnson station (Ebute Metta) – Agege – Agbado – Kajola – Papalanto – Abeokuta – Olodo – Omi Adio – Obafemi Awolowo station (Moniya) | 1435 | NRC | **running**, 3 trips each way daily (March 2026; Lagos 07:45, 13:40, 16:00; Moniya 08:00, 10:50, 16:30) | nrc.gov.ng (lists LITS, booking nrc.gsds.ng); pmnewsnigeria.com 2026/03/31; gazettengr.com "Easter: NRC to run three trips..." (31 Mar 2026) |
| **Abuja – Kaduna** (AKTS): Idu – Kubwa – Dutse? – Jere – Asham – Katari – Rijana – Kaduna (Rigasa) | 1435 | NRC | **running**: resumed 1 Oct 2025 after the 26 Aug 2025 derailment at Asham; 2 round trips most days, more Fri-Mon from 6 March 2026 | nrc.gov.ng (booking nrc.tps.ng); allafrica.com/stories/202601130534.html; primebusiness.africa "NRC increases Abuja–Kaduna trips"; nairametrics 2025/09/28 |
| **Warri – Itakpe** (WITS): Ujevwu (Warri) – Abraka? – Agbor – Igbanke? – Uromi – Agenebode – Itogbo – Ajaokuta – Itakpe | 1435 | NRC | **running**, six days a week (from May 2025); repeated derailments and pauses (1 Nov 2025 at km 212 near Agbor; one at Abraka reported by gazettengr.com, date unclear), each followed by a resumption; still listed by NRC | nrc.gov.ng (booking nrc-fane.ng); nairametrics 2025/04/28; gazettengr.com (resumption Oct 2025, derailment 2 Nov 2025) |
| **Lagos mass transit, Iddo/Apapa – Ijoko** (Western line) | 1067 | NRC | **running**, daily ("still running Mass Transit Train Service every day of the week", NRC, Feb 2025) | lekkibizchronicle.com/2025/02/01/narrow-guage-active-says-nrc/ |
| **Port Harcourt – Aba** (Eastern line, rebuilt 2024, 63 km) | 1067 | NRC | **running**, daily except Monday; 08:00 from Port Harcourt, 15:00 from Aba; Saturday trips added Aug 2026; short pauses (Sept 2025 breakdown, resumed 9 Sept) | nairametrics 2024/11/29 (handover), 2025/09/05; allafrica.com/stories/202509100275.html; search snippet for Aug 2026 |
| **Lagos Blue Line**, Marina – National Theatre – Iganmu – Alaba – Mile 2 (13 km) | 1435, electric | LAMATA | **running**, 94 trips a day from 15 June 2026; phase 2 Mile 2 – Okokomaiko due 2027 | nairametrics.com 2026/06/09; gazettengr.com "LAMATA increases Blue Line daily trips from 90 to 94" |
| **Lagos Red Line**, Agbado – Iju – Agege – Ikeja – Oshodi – Mushin – Yaba – Oyingbo (27 km) | 1435 | LAMATA | **running**, 9 trips a day from Feb 2025 (opened 15 Oct 2024); new trains due Q3 2026 | thisdaylive.com 2024/10/16; nairametrics.com/tag/lagos-red-line/ |
| Abuja Rail Mass Transit (Yellow: Metro – Airport; Blue: Idu – Gbazango/Kubwa; 45 km, 12 stations) | 1435 | FCTA (CCECC) | **unclear**: relaunched 29 May 2024, free rides to end of 2024, four trips a day; no report of service in 2025-2026 found either way | newtelegraphng.com (250,000 passengers in 100 days), thecable.ng (free rides to end of 2024) |
| Lagos – Kano express, Kano – Kaduna, Kano – Minna, Bauchi – Inkil and the rest of the Cape gauge network | 1067 | NRC | **not running**: repeatedly "to resume" (Lagos – Kano Q1 2024; Kano – Minna "before end of 2025") with no report of it happening | thecable.ng, nairametrics 2025/07/05, dailytrust "NRC moves to revive old narrow gauge networks" |
| Kaduna – Kano SGR, Kano – Maradi, Port Harcourt – Maiduguri rebuild beyond Aba | | | under construction | |

Decisions for a build:
- Register lines, running: Lagos – Ibadan, Abuja – Kaduna, Warri – Itakpe, Iddo – Ijoko,
  Port Harcourt – Aba, Lagos Blue Line, Lagos Red Line.
- **Abuja metro**: built as two register lines but greyed (`suspended`) until a 2025-2026
  report of trains turns up. That is my call on thin evidence; flip it if a report appears.
- Everything else on the Cape gauge network: not built (as Algeria's idle lines). If Anita
  wants the network visible, Lagos – Kano and Port Harcourt – Kano could be greyed lines,
  ~3,500 km, but that is a lot of grey for a country whose real network is ~800 km.

### OSM (Overpass, 2026-10-08, bbox 4.2,2.6,13.9,14.7; it also catches Benin, Niger and Cameroon)

Saved in this agent's scratch only (osm_NG.json); summary:
- **Route relations that matter**: route=train "Express Train : Lagos <-> Ibadan" (13184642,
  NRC), "Abuja-Kaduna Railway" (6441947, NRC), "Itakpe - Warri Rail Line" (9285998, NRC),
  "Port Harcourt - Kano" (8527053, a historic service, not running), "Kaduna – Zarai - Kano
  Railway" (18278186, probably the SGR under construction); route=railway "Lagos–Ibadan SGR"
  (10699301), "Itakpe - Ajaokuta - Warri Standard Gauge" (9110139); light_rail "Yellow Line :
  Abuja - Airport" (8441841/2) and "Blue Line : Gbazango - Idu" (8442723/4), network Abuja Rail
  Mass Transit; subway "Blue Line: Mile 2 ↔ Marina" (15668733/4, LAMATA); monorail "Port
  Harcourt Monorail" (13035324, never opened).
- **Missing**: no route relation for the Lagos Red Line, the Iddo – Ijoko mass transit or Port
  Harcourt – Aba. They need hand lists.
- **Track**: 1,621 `railway=rail` ways in the bbox, only 216 named (13%): "Lagos–Ibadan SGR"
  132, "Eastern Rail Line" 38, "Linking Line" 15, "West Line"/"Western Line"/"West Line
  Branch" 16. Gauge is tagged on 73% (1435: 511, 1067: 460, Benin/Niger 1000: 212).
  So Korea's named-track recipe does not work; the gauge tag helps.
- **Stations**: 211 station/halt objects in the bbox, 186 named.

### Line lists, km, coordinates

No register, no operator line list with km. What exists:
- Published lengths for checks: Lagos – Ibadan 156.8 km (WP "Lagos–Kano Standard Gauge
  Railway": 156), Abuja – Kaduna 186.5 (WP 187), Warri – Itakpe 326 (WP "Warri-Itakpe
  Railway"), Port Harcourt – Aba 63 (nairametrics), Blue Line phase 1 13 km, Red Line 27 km
  (LAMATA; some sources 37 km for the full Agbado – Marina plan), Abuja metro 45 km for both
  lines.
- Stops: the three NRC SGR route relations above carry them; for Iddo – Ijoko, Port Harcourt –
  Aba and the Red Line, every OSM station on the traced section (`osm_stops`), which for the
  Red Line's 8 stations should be checked against LAMATA's list.
- Wikidata: the WDQS endpoint was rate-limited to 1 request a minute during an outage on
  2026-10-08, and both queries failed; Wikidata is glue only here anyway.

### Timetables / GTFS

None for rail. Transitous has no Nigerian feed (api.transitous.org/gtfs/ index, checked).
The Mobility Database catalogue (share.mobilitydata.org/catalogs-csv) answers scripts with a
Cloudflare challenge; Anita can open it in a browser and search "NG" (I expect only Lagos BRT
feeds, if anything). NRC's three booking sites (nrc.gsds.ng, nrc.tps.ng, nrc-fane.ng) hold the
current SGR timetables; not crawled.

### Recipe

**A hand line list traced by rinf.py**, as `za_register.py` and `nafrica_register.py`: each
line its ends and enough stations to pin the path, written into rinf.py's input files and
traced over OSM track; stops from the route relation where there is one, else `osm_stops`.
About 7 register lines (9 with the Abuja metro), **~800 km running** (~850 with the Abuja
metro greyed).

One trap: **Lagos has three parallel railways in one corridor** from Ebute Metta to Agbado:
NRC's Cape gauge (Iddo – Ijoko), the Lagos – Ibadan SGR and the Red Line (standard gauge,
its own tracks). A shortest-path trace will jump between them. The trace needs either a
gauge preference (1067 for Iddo – Ijoko) or za's `own` relation preference (only the SGR has
one); check whether rinf.py can take a gauge filter, and if not, a no-op-unless-set hook is
the fix (the managing session's call).

Extract: `africa/nigeria-latest.osm.pbf`, **676 MB** (Geofabrik, 2026-10-08), the largest in
this survey; extract.py only needs the rail objects, so it can be deleted after.

### A shared reader for the region

Every country in this survey is a handful of hand-listed lines with no register, as North
Africa was. **One reader, `wafrica_register.py` with the lists in `wafrica_lines.py`**, on
`nafrica_register.py`'s pattern (writes rinf.py's input files; `--clip`, `--fill`, `--fork`),
would cover ng, gh, sn, bf, cm, ga, cg, cd, ao: about 30 lines, ~6,000 km, ~3,700 running.
Countries with their own chainage (ga from SETRAG, cg from CFCO's PK, ao from fahrplancenter)
get `chain` checks; the rest `no_chain`, checked by published line lengths in
`check_model.REGISTER`.

### Licences

OSM ODbL. Press and operator figures are facts. Nothing gated.

### Open questions

- Abuja metro: running in 2026 or not? (Press: thecable.ng, punchng.com, dailytrust.com.)
- Iddo – Ijoko after the Red Line opened (Oct 2024): NRC said it still runs daily in Feb 2025;
  nothing later found.
- Warri – Itakpe's and Abuja – Kaduna's intermediate stations: names in OSM's routes; NRC's
  booking sites list them too.
- Kano – Kaduna Cape gauge: NRC keeps announcing it; re-check before a build.
