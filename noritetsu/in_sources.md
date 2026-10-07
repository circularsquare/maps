# India register sources (surveyed 2026-10-03)

How India's register is put together, from what, with which numbers, and what is still off.
`in_register.py`'s docstring says how the reader works; `rinf_countries/in.py` holds the
settings rinf.py reads; `rules/in.py` says which OSM train relations are named trains.

## Commands

    python in_register.py --fetch        # Wikidata (6 SPARQL queries, ~2 min) + the IR GTFS (4 MB)
    python extract.py --region in --pbf data/raw/india-latest.osm.pbf     (1.7 GB, ~10 min)
    python in_register.py --convert      # data/raw/in/{sections,points,names}.json
    python build_model.py --region in --register in_register:data/raw/in
    python build_tiles.py --region in
    python check_model.py --region in
    python in_register.py --report       # the register without OSM: km per zone against IR's
    python in_register.py --line "Kalka" # one line's sections, km and where each km came from

`--register in_register:data/raw/in` runs the conversion and then rinf.build on its output, so
`--convert` beforehand is only needed to look at the files. `tools/rebuild.py` needs
`"in": "in_register:data/raw/in"` in its REGISTER (managing session).

## Sources

| source | what it gives | licence | size | file |
|---|---|---|---|---|
| Wikidata (query.wikidata.org) | line items (566 railway-line items with P17 India), station adjacency qualified by line (P197 + P81: 338 lines, 7,815 stations), station points (P625), IR station codes (P5696, 9,585 codes), division (P137) and its zone, 221 station km on lines (P6710, unused) | CC0 | 9 MB | `data/raw/in/wd_*.json` |
| Neo2308/indianrailways-gtfs (github.com/Neo2308/indianrailways-gtfs, `gtfs/gtfs.zip`) | every NTES train: 10,594 routes, 8,550 stops keyed by IR station code with a point, stop_times with `shape_dist_traveled` = the train's cumulative km at each call (whole km). Feed version 2026-08-30, valid to 2026-09-30, "data generated around 9 Nov 2025" per its README | none stated (scraped from NTES; also listed by the Mobility Database as mdb-2867 and used by Transitous) | 4.2 MB | `data/raw/in/ir_gtfs.zip` |
| en.wikipedia line articles | infobox `tracklength` for 183 of the register's lines (the check table) | CC BY-SA | | `data/raw/in/wp_lengths.json` |
| en.wikipedia "Indian Railways organisational structure" | route km per zone (as of June 2026, from IR's route map): 69,393 km over 18 zones | CC BY-SA | | (numbers in `in_register.report`) |
| en.wikipedia "Indian Railways" | route km 69,181 (IR Year Book 2023-24, 31 March 2024), broad gauge 66,820; broad-gauge route 70,412 km on 31 July 2026 (IR electrification status) | | | |
| OSM API, single relations (about 25 calls, generic User-Agent) | how OSM India names train relations (rules/in.py) and the Mumbai suburban colours | ODbL | | |
| Geofabrik `asia/india-latest.osm.pbf` (india-261002) | track (106,615 ways), stations (24,577 stop records; 7,710 station nodes carry `wikidata`, 9,127 IR codes in `ref`), 402 route relations (255 train), 627 infrastructure relations | ODbL | 1,709 MB | fetched and extracted by the managing session; data/proc/in is all a rebuild needs |

OSM India's track is not named for its line (30% of main and branch rail km carries a name,
`probe_kr_ways.py`), so Korea's named-track recipe does not work; and OSM passenger routes run
over only 44% of main-line km (`inspect_region.py`), so they cannot decide which track is
passenger track either. Hence the timetable decides the register and rinf.py traces it.

Not used: datameet/railways (CC0, 2016: `stations.json` 1.9 MB, `trains.json` 15 MB,
`schedules.json` 82 MB) is the same NTES data nine years older; the GTFS supersedes it.
OSM's 600 India `route=railway` relations (14 with a ref; "Delhi - Kalka Line" carries its stop
nodes) were not needed; they could name lines later.

## Line unit (Anita's decision)

**What is built**: a register line is a Wikidata line item. 338 items carry a station chain;
60 of them are left out (metro, RRTS, monorail and closed items, and chains where fewer than
60% of the stations have an IR code: metros filed as plain railway lines, Kanpur's Orange
Line, Nepal's Janakpur - Jaynagar), leaving 278 IR lines: "Mathura–Vadodara Section" (860
km), "Konkan Railway", "Jolarpettai–Shoranur line", the Sealdah and Howrah suburban branches,
Mumbai's Central / Western / Harbour lines, the hill railways. They hardly overlap (52 station
pairs listed by two lines, each kept on one), so they partition the track the way IR's own
sections do, and they are the units en.wikipedia writes articles about.

They cover only part of the passenger network. The timetable's passenger network (every pair
of consecutive calls of every NTES passenger train, less the express hops local calls already
cover) is 75,261 km; after the chains take theirs, 22,318 km lies on no chain (North Eastern
Railway: 94% of it). So the rest is built from the timetable (`FILL = True`):

- 20 Wikidata line items with no chain, named for their two ends, laid over the timetable's
  network between stations of those names (2,292 km: Jammu - Baramulla 324, Bikaner - Rewari
  379, Barkakana - Son Nagar 312, Jogbani - Katihar, Saharsa - Forbesganj...). A name path is
  refused when the network's own shortest path between the two ends is shorter by over 10%
  (Lucknow - Gorakhpur found 394 km via Sitapur against 274 by the main line) or the item's
  own length (P2043) disagrees (Mainpuri - Etawah).
- 17 chains carried on over the feed from an end no other line touches to the next junction
  (694 km: Kalka - Shimla from Barog to Shimla, Jalandhar - Jammu... see the build log).
- 431 lines junction to junction over the rest (19,333 km), named "<first station> -
  <last station>" from Wikidata's labels or the feed's names ("Gonda Junction - Barabanki
  Junction"); line id `T-<code>-<code>`.

Totals before OSM: 728 lines, 70,257 km (Wikidata chains 48,506, chainless items 2,418,
timetable 19,333), against IR's published 69,393 route km (which includes freight-only
track). Per zone (published km / built km): NR 7,363 / 7,411; NWR 5,706 / 6,331; SR 5,093 /
5,290; WR 6,157 / 4,862 (0.79: metre-gauge lines closed for conversion are in the published
figure and run no trains); ECR 4,238 / 4,822; NFR 4,348 / 4,687; NCR 3,523 / 3,932; CR 4,203 /
3,830; SCoR 3,532 / 3,820; SCR 3,572 / 3,483; SWR 3,692 / 3,457; NER 3,470 / 3,450; WCR
3,060 / 3,348; SER 2,759 / 3,106; ER 2,823 / 2,884; ECoR 2,701 / 2,548; SECR 2,397 / 2,247;
KR 756 / 757. Zones over 1.10 still hold some double counting: feed hops whose junction no
train calls at and that are not cut cleanly (see "What is off").

**The alternatives**:
1. Wikidata chains only (278 lines, 48,500 km, about 70% of the network): the NER, most of
   NCR and SECR count towards nothing; their track would only be ridden through named trains.
2. Timetable junction-to-junction sections everywhere, Wikidata only for names: one rule for
   the whole country, but about 2,000 lines, most of them short pieces between junctions, and
   "Mathura–Vadodara Section" cut into a dozen.
3. What is built: Wikidata where it has a line, the timetable for the rest.

My lean is 3. Question for Anita: are the junction-to-junction "A - B" lines acceptable as
lines (they are IR sections in all but name; IR has no open list of section names), or should
the fill be dropped or merged into longer lines?

## How the km are measured

- A pair of chain neighbours that both have trains calling is the shortest path between them
  over the timetable's network (at most 1.5x the chain's crow-fly + 5 km); calls between them
  that Wikidata lacks are put into the line (222 stations). 5,759 runs measured so.
- A station no train in the feed calls at is not a stop. It is merged away (560), unless it is
  a branch point, an end, or on two lines (kept as a junction point, rinf type 80), or an OSM
  train route stops there (after the extract: Mumbai's suburban halts, which NTES does not
  carry, would otherwise go).
- Runs the feed cannot measure (384 with no feed path, 78 with an unserved end, and every
  run of a line no train in the feed calls at) take the chain's crow-fly x 1.10 and are marked
  "crow" in sections.json: 441 of 8,956 section rows.
- Wikidata's point against the feed's for the same code: median 46 m apart, 95% within 2 km,
  106 over 5 km. Where they are over 3 km apart the one nearer the station's chain
  neighbours wins (34 times the feed's: Khudiram Bose Pusa is 81 km off in Wikidata, Hamrapur
  1,307 km). 66 Wikidata stations whose code the feed does not know take the one feed stop
  within 300 m (renamed codes).
- Feed hops that run over register track: 25 lie on it end to end (3,402 km) and go; 57 leave
  it at a junction no train calls at and are cut there (2,700 km of register track).

## First build (2026-10-03, india-261002 extract)

    python tools/slot.py -- python build_model.py --region in --register in_register:data/raw/in   (~4 min)
    python tools/slot.py -- python build_tiles.py --region in                                      (~2 min)

- Points placed at their OSM station node: 5,204 by its `wikidata` tag, 2,747 by `ref` (the
  IR code), 77 by name within 3 km; 32 more than 1.2 km from any OSM rail track left unplaced
  (wrong coordinates: Reasi), 575 at Wikidata's or the feed's point. rinf.py then found an
  OSM station for 8,083 of 8,579 stops; 469 have none and become junction points (their
  names are in the build log's "no OSM station" line: halts OSM lacks or names otherwise).
- Traced: 8,974 section rows, 83 merged sections left out for a rejected trace, 18 traced end
  to end. 727 rinf lines, 67,509 km traced against 68,545 km of timetable km.
- Built: 873 lines, 720 of them register lines, 67,135 km (Wikidata chains 280 pieces /
  46,339 km, chainless items 19 / 2,383 km, timetable junction-to-junction 421 / 18,413 km),
  74 named trains, 8,935 stations. Junction-ended sections: 6 kept (276 km), 29 dropped
  (374 km). Against the published 69,393 route km: 0.97. Per zone (built / published): NR
  0.98, NWR 1.09, SR 1.01, WR 0.73, ECR 1.05, NFR 1.02, SCoR 1.05, CR 0.86, NCR 1.02, NER 0.97,
  SWR 0.91, SCR 0.90, WCR 1.04, SER 1.07, ER 0.97, ECoR 0.92, SECR 0.88, KR 0.97.
- `check_model`: every line against its timetable km, median 0.996, 139 of 714 off by more
  than 5% (mostly short timetable sections, whole-km rounding); 40 of 44 published lengths
  within 5%. Off: Darjeeling Himalayan 0.78 (Tindharia - Rangtong, the loops, has no path over
  OSM track), Kolkata Circular 0.82, Asansol - Gaya 0.93, Guntakal - Vasco 0.94 (Castle Rock -
  Tinai Ghat on the Braganza ghat: timetable 9 km, track 12.8, rejected).
- Ownership: 822 km of track only OSM's named trains run over (owned by nobody); the biggest
  pieces (near Kishanganj, Haripur, Duraundha, Patchur, Tikri, 18-37 km each) are register
  gaps from rejected traces or stations OSM lacks, worth a look one by one.
- Tiles: 11.5 MB; 943 ways (677 rail) on track no line touches.

## Checks

`check_model.REGISTER["in"]`: 44 lines against en.wikipedia's infobox length (or Wikidata's
P2043), chosen where the article's extent is the chain's. Before OSM, the timetable's km came
to 0.97-1.07 of them (Mathura - Vadodara 860 / 852, Konkan 757 / 756.25, Kalka - Shimla 95.5 /
96.6, Nilgiri 46 / 46, Jammu - Baramulla 324.5 / 324). Every line is also checked against its
own timetable km (`km_official`, rinf.py's chain).

Wikipedia figures not used, and why: Gudur - Chennai (WP 455, an article spanning more),
Vijayawada - Gudur (455, same), Chennai Central - Bengaluru (561 "full route to
Chamarajanagar"), Mumbai - Ahmedabad (493 from Mumbai Central; Wikidata's chain starts at Dahanu Road,
where the Western Line's ends, built 371), Coimbatore - Shoranur (366, the Jolarpettai - Shoranur article's), Samastipur -
Muzaffarpur (53, fine: built 54.5, but the chain had an 81 km coordinate error before the
point check), Varanasi - Sultanpur - Lucknow (infobox reads "2"), Jharsuguda - Vizianagaram
(the article gives two parts, 230 + 266 = 496, built 498), New Bongaigaon - Guwahati (both
routes, 157.8 + 182.7, built 343), Guntakal - Renigunta (article length is another line's),
Sealdah - Namkhana / Bangaon / Ranaghat - Gede (articles count branches), Ludhiana - Jakhal
(Wikidata's chain runs on to Hisar), Muzaffarpur - Sitamarhi (carried on to Raxaul by the
fill).

## What is off, and open

- **Feed coverage**: NTES carries no Mumbai suburban locals (Churchgate 16 calls, all
  long-distance), so Mumbai's lines are measured crow-fly and their halts stay only where OSM
  routes stop. Kolkata's and Chennai's suburban trains are in it (Sealdah 732 calls). Metros
  are not, and stay OSM lines.
- **The feed's km are whole kilometres per train**, so a short section can be a kilometre off;
  `tol_abs` 1.5 in rinf_countries/in.py.
- **Junction-to-junction lines name stations by Wikidata label or the feed's abbreviated name**
  (title-cased: "Bmby Church Gte" where Wikidata has no item for the code). After the extract
  points are renamed to their OSM node's name.
- **Over-counting** where a train runs through a junction no train calls at onto another line:
  the hop is cut at the register point furthest along it, which is right when the two lines
  meet at a Wikidata station and wrong when they meet at a cabin Wikidata does not list. NWR,
  ECR, NCR and SER build 1.11-1.14 of their published km.
- **GTFS check (gtfs_served.py)**: not switched on. It needs `CODE["in"]` and `FEEDS["in"]` in
  gtfs_served.py (a shared file; the diff is in the agent's report), then the zip copied into
  data/raw/gtfs/in/.
- **Freight-only lines** never enter the register: it is built only where trains call. A line
  closed for gauge conversion is absent rather than greyed.
- **The feed is unofficial and dated**: NTES scraped around November 2025, calendars to
  2027-08-30. New lines since (Bairabi - Sairang opened 2025) are in it; anything opened after
  November 2025 is not.

## Disputed territory and borders

Anita's rule elsewhere is de facto: track is drawn as trains actually run.

- **Jammu and Kashmir**: the Jammu - Baramulla line (USBRL, through to Srinagar and
  Baramulla since 2025) and Jammu - Katra are in Indian-administered territory and run Indian
  Railways trains: built with India. Pakistan claims it; nothing to draw on the other side.
- **Arunachal Pradesh** (claimed by China): Naharlagun and the Murkongselek line are built
  with India, as trains run.
- **Pakistan**: Attari - Wagah (Samjhauta Express) and Munabao - Khokhrapar (Thar Express)
  have carried no passenger train since 2019; the register stops where the feed's trains
  stop (Attari, Munabao).
- **Bangladesh**: the international trains (Maitree via Gede - Darsana, Bandhan via Petrapole -
  Benapole, Mitali via Haldibari - Chilahati) have been suspended since mid-2024; the feed's
  domestic trains end on the Indian side. Akhaura - Agartala is built but carries no
  passengers.
- **Nepal**: Jaynagar - Janakpur - Kurtha is Nepal Railways' (left out by NOT_REGISTER); Raxaul
  - Birgunj is freight. Jaynagar and Raxaul are Indian line ends.
- The country outline used by tools/build_regions.py (religiondots' country_shapes, else
  Natural Earth) should be checked for which line it draws in Kashmir before India is put on
  the map, so the Srinagar valley track falls inside India's outline.
