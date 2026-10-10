# Line types

A type per line, in words a rider uses (Anita, 2026-10-09: "ok lets do line type for sure").
Research behind it: `flavour_stats_checklist.md` §1. Written after the build, like the English
names: `tools/line_types.py` reads the shipped files and writes `dist/data/<cc>/types.json`; no
country is rebuilt. The app side is a ready patch in `handoff_notes/line_types_app.md` (not
applied).

    python tools/line_types.py                    # every country: write types.json, print the summary
    python tools/line_types.py jp de --sample 5   # print 5 typed lines per source
    python tools/line_types.py us --explain m10322749   # why one line got its type
    python tools/line_types.py --dry --untyped de # list what is left without a type

Reads `dist/data/<cc>/{lines,foot,stations}.json` and `data/proc/<cc>/rels.pkl` (the route
relations' `service=`, kept by `extract.REL_TAGS`). Runs in a few minutes for all 120
regions, one process. Rerun it after a country is rebuilt (ids can move).

## The types

| key | label in the app | what it is |
|---|---|---|
| high_speed | High-speed rail | Shinkansen, TGV, ICE, AVE, KTX, CR G/D/C trains; their dedicated track |
| intercity | Intercity rail | long-distance and fast trains between cities: IC, EC, Amtrak, VIA, Indian expresses, 特急 |
| night | Night train | sleeper and night trains (OSM lines only; on track they count as intercity) |
| regional | Regional rail | stopping and regional trains: RE/RB, TER, Regionale, JR locals outside the city belts |
| commuter | Commuter rail | S-Bahn, RER, Cercanías, elektrichki, LIRR, Metra, GO, airport trains |
| metro | Metro | `kind` subway |
| light_rail | Light rail | `kind` light_rail (Stadtbahn, US light rail, DLR) |
| tram | Tram | `kind` tram |
| monorail | Monorail | `kind` monorail, or by name |
| people_mover | People mover | Japan's AGT (N02 guideway flag), airport and other automated people movers |
| maglev | Maglev | Linimo, Shanghai, Changsha, Incheon |
| funicular | Funicular | `kind` funicular, or by name (Standseilbahn, ケーブル) |
| tourist | Tourist railway | heritage and excursion lines, mountain rack railways, children's railways |

Thirteen, in the app's sentence case. Shipped: regional 6,778, commuter 4,559, intercity
3,557, tram 2,289, metro 1,011, high-speed 518, light rail 411, tourist 230, funicular 178,
night 65, monorail 63, people mover 50, maglev 7.

## The file

    {"m3313695": "intercity",
     "r424e737107": {"type": "regional", "also": ["intercity", "high_speed", "commuter"]}}

Keyed by the id in that country's `lines.json` (before the app's LINE_FOLD). A plain string is
a line of one kind (every OSM line, every urban register line). An object is a register line,
which is track: `type` is its main kind, `also` the other kinds of train over it (each covering
at least 15% of it), biggest share first, at most three. A line with nothing to go on is left
out. Sizes: 27 bytes (Djibouti) to 100 KB (Russia), 0.5 MB in all.

## How a line gets its type

The order of the rules is in the tool's docstring; in short:

- **OSM lines.** Urban kinds by `kind`. Train routes: `service=` on the route_master, else the
  majority over its routes (81% of train routes carry one), normalised to the list above; then
  a country rule on network, operator, name or train number; then high-speed track (an untagged
  train running 70% of its length on it); then words in the name the same everywhere (S-Bahn,
  RER, Express, TER, Heritage...); last, a named train is intercity and a route regional.
- **Register lines.** The kinds of the OSM services whose footprint (foot.json) lies on the
  line, each as the share of the line's km at least one of them covers (a union, the way
  `tools/operators.py` measures operators for ops.json). High-speed track with high-speed trains
  on it is high-speed. Urban registers (N02, MTR, LTA, the Korean and Taiwanese metros) keep
  their kind. Where no service reaches, the country default below.

## Decisions (made here, not asked)

- **Russia's 1,622 `regional` elektrichki** are prigorodny (suburban) trains. Up to 150 km
  they are commuter rail; longer ones (Velikiye Luki - Pskov, 302 km, a diesel twice a day) are
  regional rail. Same for every post-Soviet country (ru, by, ua, kz, uz, kg, tj, tm, ge, am,
  az, md, xa, mn), where untagged trains are read by the Soviet train number: 1-599
  long-distance, 600-699 local, 700-799 fast day trains, 800-899 fast regional, 6000-7999
  suburban.
- **High-speed track needs high-speed trains** to make a register line high-speed. Britain's
  125 mph main lines (GWML, WCML Trent Valley) are tagged `highspeed=yes` but carry intercity
  trains: they are intercity. Germany's 200 km/h upgraded lines with ICEs on them (1700
  Hannover - Hamm) are high-speed, as the EU counts them. China is the exception: its G and D
  trains are rarely mapped, so there the track alone decides (and 高速线 / 客专 by name).
- **A high-speed train on conventional track is an intercity train there** (KTX over the
  conventional 경부선: intercity, also high-speed). A night train on a register line counts as
  intercity (the Southwest Chief alone on the Gallup Subdivision).
- **Which service is a register line's main type.** On lines under 200 km the operating
  patterns decide once one covers half the line (Mumbai's Western Line stays commuter; one
  sleeper a night does not make a branch line intercity). On longer trunk lines every service
  counts (Taishet - Lena, 731 km, where elektrichki cover half: intercity). Within 5 points of
  the top share the more local kind wins (the Hudson Line, with Metro-North and Amtrak over all
  of it: commuter, also intercity). Where services cover under half the line and the country
  has a default, the default is the main type and the services seen go in `also` (京通线: one
  Beijing S-line over a fifth of it; intercity, also commuter).
- **Amtrak**: Acela is high-speed; every other Amtrak train intercity, corridor and
  long-distance alike (a rider calls both "Amtrak", and both run city to city). Trains tagged
  `night` (the Southwest Chief, the Empire Builder) show as night trains.
- **Airport trains** are commuter rail (Heathrow, KLIA Ekspres, Flytoget, UP Express), but
  only up to 100 km: a long route calling at an airport is not an airport train.
- **Tourist railway** covers heritage lines, excursions and mountain railways (Glacier
  Express, Gornergrat, the Jungfrau railways, PeruRail, India's mountain railways, children's
  railways). Lines that are also everyday transport (Wengernalpbahn) are tourist all the same.
- **Japan**: Shinkansen by the N02 flag; limited expresses (特急, Romancecar, Skyliner) are
  intercity; everything else is commuter where at least half the line lies inside a city belt
  (50 km of Tokyo, 45 km of Osaka, 30 of Nagoya, 25 of Fukuoka and Sapporo, 20 of Sendai and
  Hiroshima), regional outside. Mini-shinkansen trains (Tsubasa, Komachi) are high-speed; the
  Ōu Line they share is regional, also high-speed. Guideway lines (Yurikamome, New Shuttle) are
  people movers; Linimo is a maglev.
- **China**: G, D and C trains high-speed; K, T, Z intercity; 1001-5998 (普快) intercity;
  6001-7598 and 8xxx (普客) regional; S lines (Beijing's 市郊铁路) commuter; 城际 regional.
  Conventional register lines default to intercity (numbered long-distance trains, cn_sources).
- **Closed lines are not typed** (greyed, nothing runs). Neither are junction curves and yard
  links under 5 km running, nor Germany's unmapped register lines (1280, 1750, 5230: freight
  bypasses).

## Country rules (untagged OSM trains)

us: Acela high-speed; Amtrak, Brightline, Alaska Railroad intercity; LIRR, Metro-North, NJ
Transit, SEPTA, MARC, VRE, Metra, NICTD, Metrolink, Caltrain, ACE, SMART, Sounder, Tri-Rail,
SunRail, FrontRunner, Coaster, TRE, Rail Runner, RTD, CTrail, MBTA and the rest commuter;
everything else in the untagged list is an excursion or park railway, tourist. ca: VIA intercity,
Rocky Mountaineer tourist, GO / exo / West Coast Express / UP Express commuter, Ontario
Northland and Keewatin regional. id: Whoosh high-speed, KAI Commuter and KAI Bandara commuter,
KAI named trains intercity and its other routes regional. in: MMTS, suburban and local
commuter; Passenger, MEMU, DEMU regional; named expresses intercity. tr: YHT high-speed,
Ekspres and Mavi Tren intercity, Bölgesel regional, Banliyö / İZBAN / Marmaray commuter,
Turistik tourist. rs: Soko high-speed, IR intercity, Re/REx regional, BG:Voz commuter,
Nostalgija tourist. kr: KTX/SRT high-speed, ITX / Saemaeul / Mugunghwa intercity. tw: 自強,
太魯閣, 普悠瑪, 莒光 intercity; 區間 regional. my: ETS intercity, KTM Komuter and KLIA Ekspres
commuter. th: Airport Rail Link and the Red Lines commuter, rapid and express intercity, local
regional. vn: SE/TN intercity, HP/LP (Hà Nội - Hải Phòng) regional. kp: international trains
intercity, the rest regional. lv: LTG Link intercity, up to 100 km commuter, longer regional.
es: Cercanías C lines, Euskotren and SFM commuter. it: FL lines and Genova's metropolitana
commuter, Avellino - Rocchetta tourist. cz: S lines commuter only in Prague (PID), elsewhere
regional. hu: S/G/Z lines commuter. gb: Merseyrail, Overground, Elizabeth line, Thameslink
commuter; CrossCountry, TransPennine, LNER, Avanti and GWR's London trains over 150 km
intercity; the gb network tag "national" is not read as a service (it means National Rail).
ch: the Jungfrau railways and cogwheel (CC) lines tourist. no: L lines commuter. se: SL's
Roslagsbanan and Saltsjöbanan commuter. au: Transperth etc. commuter. ma: TNR regional, Al
Atlas intercity. dz: SNTF Alger commuter. eg: the LRT commuter. ir: urban and suburban (شهری,
حومه) commuter. lk: Kelani Valley regional. bo: Buscarril regional.

## Register defaults (track no OSM service reaches)

Applied to lines with at least 5 km running, and as the main type where services cover under
half a line. Named lines first:

- **regional**: it, cz, ro, es, si, pl, gb, ua, by, bg, hr, gr, al, ba, me, mk, xk, md, ge, am,
  az, kg, tj, se, lt, lv, ee, lu, ie, no, sk, hu, at, be, ch, nl, ru, fr, rs, ar, cl, au, my,
  vn, gh, mg, mw, mz, np, uy, pe, ph, il, tr, dk, nz, tw: every running line carries the
  stopping service, and the long-distance trains are mapped where they run. Germany is left
  out (its unmapped lines are freight).
- **intercity**: cn (China Railway's numbered long-distance trains), in (NTES express and
  passenger trains on every line), pk (a named express on every running line, pk_sources), fi
  (VR), kr (Korail's Mugunghwa and ITX), ca (VIA and Amtrak on the freight subdivisions), and
  the one-service countries ae, bf, cg, cm, dj, et, ga, kh, sa, cd, ao, ng.
- **by length** (intercity from 250 km, else regional): bd, lk, kp, kz, uz, tm, mm, mn, eg,
  dz, tn, tz, zm, zw, iq, th, ir, id. On a national network with one operator and few trains
  the long lines carry the long-distance trains and the short branches local ones.
- **commuter**: br, cr, pa, sn, ug, ve (city lines). **tourist**: jo (excursions only,
  jo_sources), co, ec. ke and mx regional.
- **Named**: Linha de Cascais, Adelaide's and Brisbane's suburban lines, Limache and Biotren,
  El Insurgente, the KTM Skypark link, PRASA's lines, Lagos's lines, Luanda's and Lobito's
  suburban trains, Kinshasa's and Antananarivo's urban trains, Algiers's airport line, the
  Métro du Sahel, Başkentray / Gaziray, Wellington's Johnsonville Line, Tainan's Shalun line:
  commuter. The Ghan's track, Tren Maya, the Interoceanic, the KTM East Coast line, Vietnam's
  North-South and Lào Cai lines, Kenya's SGR and Kisumu line, Transnet's lines, Tel Aviv -
  Jerusalem, Baghdad - Basra, Kars - Tbilisi: intercity. Puffing Billy, Serra Verde Express,
  Đà Lạt, Alishan, Tren Turístico de la Sabana, Nariz del Diablo: tourist.
- **Overrides** (before services): Mumbai's suburban Western, Central and Harbour lines are
  commuter (OSM's Western Line route stops at Virar, the register line runs to Dahanu); the
  Darjeeling, Nilgiri, Kalka - Shimla and Matheran railways are tourist.
- **sibling**: a US "(second track)" register line takes its subdivision's type (43 lines).

## Coverage, 2026-10-09

"Before" is what the data already said without this work: urban `kind`, tagged `service=` and
register lines with high-speed track (the app showed none of it). "Untyped, running" counts
lines with 5 km or more not closed and still without a type.

| cc | lines | before | after | OSM lines by source | register lines by source | untyped, running |
|---|---|---|---|---|---|---|
| ae | 6 | 3 | 6 | kind 3, name 2 | default 1 | 0 |
| al | 9 | 1 | 3 | default 1, tag 1 | default 1 | 0 |
| am | 18 | 1 | 16 | default 6, rule 1, kind 1 | services 7, default 1 | 0 |
| ao | 7 | 0 | 6 |  | default 6 | 0 |
| ar | 51 | 31 | 47 | tag 19, kind 9 | services 10, default 6, kind 3 | 0 |
| at | 356 | 234 | 338 | tag 166, kind 64, name 2, default 1 | services 89, default 12, hs 4 | 0 |
| au | 268 | 126 | 261 | tag 88, kind 38, rule 1 | services 111, default 23 | 0 |
| az | 29 | 4 | 25 | rule 7, kind 4 | services 8, default 5, sibling 1 | 0 |
| ba | 13 | 7 | 12 | kind 6, default 1, tag 1 | default 3, services 1 | 0 |
| bd | 31 | 1 | 27 | kind 1 | default 26 | 0 |
| be | 291 | 155 | 269 | tag 104, kind 47 | services 109, hs 4, default 4, sibling 1 | 0 |
| bf | 2 | 0 | 1 |  | default 1 | 0 |
| bg | 61 | 22 | 58 | kind 21, default 6, tag 1 | default 28, services 2 | 0 |
| bo | 12 | 7 | 12 | tag 4, kind 3, rule 1 | services 4 | 0 |
| br | 79 | 56 | 79 | kind 47, tag 9, default 2, name 2 | default 10, services 9 | 0 |
| by | 162 | 75 | 157 | tag 55, kind 20, default 10, rule 2, name 1 | services 53, default 15, sibling 1 | 0 |
| ca | 197 | 45 | 196 | kind 36, rule 21, tag 9 | services 118, default 7, sibling 5 | 0 |
| cd | 2 | 0 | 1 |  | default 1 | 0 |
| cg | 1 | 0 | 1 |  | default 1 | 0 |
| ch | 785 | 393 | 748 | tag 225, name 94, kind 65, rule 8, default 1 | services 240, kind 103, default 10, name 2 | 0 |
| cl | 25 | 16 | 25 | tag 9, kind 7, default 1, name 1 | default 5, services 2 | 0 |
| cm | 3 | 0 | 3 |  | default 3 | 0 |
| cn | 869 | 518 | 863 | kind 374, pre 62, name 8, default 1 | default 257, hs 144, services 17 | 0 |
| co | 4 | 3 | 4 | kind 2 | name 1, kind 1 | 0 |
| cr | 3 | 0 | 3 |  | default 3 | 0 |
| cu | 8 | 0 | 8 | default 4 | services 4 | 0 |
| cz | 513 | 264 | 490 | tag 176, kind 88, name 4, rule 3 | services 129, default 90 | 0 |
| de | 2457 | 1349 | 2278 | tag 853, kind 459, name 46, default 4 | services 878, hs 27, kind 10, sibling 1 | 8 |
| dj | 1 | 0 | 1 |  | default 1 | 0 |
| dk | 109 | 56 | 108 | tag 47, kind 9, name 7 | services 44, default 1 | 0 |
| do | 2 | 2 | 2 | kind 2 |  | 0 |
| dz | 42 | 13 | 41 | kind 10, default 5, tag 2, rule 1 | default 15, services 7, hs 1 | 0 |
| ec | 3 | 2 | 3 | kind 2 | default 1 | 0 |
| ee | 40 | 25 | 40 | tag 19, kind 6, default 2 | services 13 | 0 |
| eg | 38 | 4 | 38 | kind 4, rule 2, default 1, name 1 | default 28, services 2 | 0 |
| es | 422 | 249 | 396 | tag 150, kind 80, rule 17, name 6, default 1 | services 92, default 31, hs 19 | 0 |
| et | 3 | 2 | 3 | kind 2 | default 1 | 0 |
| fi | 78 | 47 | 76 | tag 28, kind 19 | services 24, default 5 | 0 |
| fr | 974 | 608 | 967 | tag 452, kind 142, name 87, track 4, default 2 | services 257, hs 10, default 9, kind 4 | 0 |
| ga | 1 | 0 | 1 |  | default 1 | 0 |
| gb | 871 | 342 | 836 | tag 290, kind 51, default 38, rule 15, name 5 | services 406, default 27, sibling 3, hs 1 | 0 |
| ge | 34 | 3 | 30 | rule 12, kind 2, tag 1, name 1 | services 12, default 2 | 0 |
| gh | 3 | 0 | 2 |  | default 2 | 0 |
| gr | 39 | 25 | 35 | tag 18, kind 7, default 1, name 1 | services 6, default 2 | 0 |
| hk | 33 | 30 | 33 | kind 16, name 3 | kind 13, hs 1 | 0 |
| hr | 129 | 90 | 126 | tag 69, kind 21, default 1 | services 32, default 3 | 0 |
| hu | 350 | 188 | 318 | tag 123, kind 65, rule 14, name 13, default 4 | services 91, default 8 | 0 |
| id | 157 | 12 | 157 | rule 108, kind 7, tag 4 | services 25, default 12, hs 1 | 0 |
| ie | 48 | 30 | 48 | tag 28, kind 2, default 1 | services 16, default 1 | 0 |
| il | 22 | 4 | 20 |  | default 16, kind 4 | 0 |
| in | 877 | 75 | 855 | rule 74, kind 59, tag 16, default 4 | default 554, services 138, name 10 | 0 |
| iq | 5 | 0 | 3 |  | default 3 | 0 |
| ir | 57 | 23 | 56 | kind 14, tag 9, default 5, rule 2, name 1 | default 18, services 7 | 0 |
| it | 659 | 333 | 640 | tag 224, kind 103, name 14, default 6, rule 3, track 1 | services 186, default 96, hs 6, sibling 1 | 0 |
| jo | 1 | 0 | 1 |  | default 1 | 0 |
| jp | 1099 | 233 | 1099 | pre 412, kind 71, name 19 | rule 435, kind 154, hs 8 | 0 |
| ke | 14 | 0 | 11 | name 5 | default 3, services 3 | 0 |
| kg | 8 | 0 | 7 | rule 5 | default 1, services 1 | 0 |
| kh | 3 | 0 | 2 |  | default 2 | 0 |
| kp | 80 | 9 | 57 | rule 13, kind 9 | default 27, services 8 | 0 |
| kr | 135 | 75 | 133 | tag 18, kind 17, rule 15 | kind 34, services 26, default 16, hs 6, name 1 | 0 |
| kz | 118 | 26 | 116 | tag 20, rule 14, kind 6 | services 39, default 37 | 0 |
| la | 4 | 0 | 4 | default 2 | services 2 | 0 |
| lk | 13 | 0 | 11 | rule 1 | default 10 | 0 |
| lt | 34 | 16 | 34 | tag 15, rule 2, kind 1 | services 10, default 6 | 0 |
| lu | 50 | 27 | 49 | tag 26, name 7, kind 1 | services 15 | 0 |
| lv | 52 | 23 | 52 | rule 19, kind 13, tag 10 | services 8, default 2 | 0 |
| ma | 30 | 7 | 29 | rule 9, kind 6, name 2 | services 11, hs 1 | 0 |
| md | 18 | 3 | 9 | tag 3 | services 4, default 2 | 0 |
| me | 6 | 2 | 5 | tag 2, name 1 | services 2 | 0 |
| mg | 5 | 0 | 3 |  | default 3 | 0 |
| mk | 8 | 0 | 4 | default 2 | default 2 | 0 |
| mm | 29 | 7 | 13 | kind 7 | default 6 | 0 |
| mn | 8 | 2 | 7 | tag 2, rule 1 | default 2, services 2 | 0 |
| mu | 2 | 2 | 2 | kind 2 |  | 0 |
| mw | 2 | 0 | 2 |  | default 2 | 0 |
| mx | 29 | 20 | 26 | name 2 | kind 20, default 3, services 1 | 0 |
| my | 25 | 9 | 24 | rule 6, kind 1, name 1 | kind 8, services 5, default 3 | 0 |
| mz | 6 | 0 | 6 |  | default 6 | 0 |
| ng | 9 | 3 | 9 | kind 1 | default 6, kind 2 | 0 |
| nl | 283 | 194 | 278 | tag 132, kind 59 | services 81, default 3, hs 2, kind 1 | 0 |
| no | 76 | 48 | 76 | tag 29, kind 19, name 2, rule 1 | services 23, default 2 | 0 |
| np | 1 | 0 | 1 |  | default 1 | 0 |
| nz | 32 | 16 | 32 | tag 13, kind 3, name 1 | services 14, default 1 | 0 |
| pa | 3 | 2 | 3 | kind 2 | default 1 | 0 |
| pe | 9 | 2 | 8 | name 3, kind 2 | default 2, services 1 | 0 |
| ph | 7 | 3 | 5 | kind 3 | default 2 | 0 |
| pk | 36 | 1 | 28 | name 6, default 1, kind 1 | default 13, services 7 | 0 |
| pl | 889 | 512 | 806 | tag 293, kind 218, name 16, default 2 | services 235, default 40, sibling 1, hs 1 | 0 |
| pr | 1 | 1 | 1 | kind 1 |  | 0 |
| pt | 103 | 72 | 103 | tag 46, kind 26, name 6 | services 24, default 1 | 0 |
| qa | 9 | 9 | 9 | kind 9 |  | 0 |
| ro | 202 | 113 | 191 | kind 77, tag 36, name 3 | default 47, services 28 | 0 |
| rs | 91 | 13 | 88 | rule 53, kind 10, tag 2 | services 19, default 3, hs 1 | 0 |
| ru | 3541 | 2607 | 3469 | tag 2120, kind 480, default 6, name 6, rule 4 | services 795, default 46, hs 7, sibling 5 | 0 |
| sa | 10 | 7 | 10 | kind 6, name 1 | default 2, hs 1 | 0 |
| se | 156 | 93 | 155 | tag 63, kind 30, rule 5, default 2 | services 47, default 8 | 0 |
| sg | 13 | 12 | 12 | kind 2 | kind 10 | 0 |
| si | 36 | 14 | 36 | tag 12, kind 2 | default 12, services 10 | 0 |
| sk | 174 | 104 | 154 | tag 81, kind 23 | services 47, default 3 | 0 |
| sn | 1 | 0 | 1 |  | default 1 | 0 |
| th | 33 | 8 | 33 | rule 8, kind 8, name 1 | default 14, services 2 | 0 |
| tj | 6 | 1 | 6 | tag 1 | default 4, services 1 | 0 |
| tm | 12 | 0 | 9 |  | default 9 | 0 |
| tn | 19 | 8 | 19 | kind 7, tag 1 | default 10, services 1 | 0 |
| tr | 145 | 65 | 143 | kind 57, rule 40, tag 4, name 2, default 1 | services 25, default 10, hs 4 | 0 |
| tw | 56 | 28 | 52 | rule 10, tag 6, kind 4 | kind 17, services 10, default 4, hs 1 | 0 |
| tz | 9 | 0 | 7 |  | default 7 | 0 |
| ua | 510 | 168 | 422 | kind 120, tag 48, rule 17, default 8 | default 116, services 113 | 0 |
| ug | 1 | 0 | 1 |  | default 1 | 0 |
| us | 920 | 262 | 911 | kind 220, rule 136, tag 41, name 23 | services 447, sibling 43, kind 1 | 4 |
| uy | 1 | 0 | 1 |  | default 1 | 0 |
| uz | 48 | 14 | 41 | kind 6, tag 5, rule 1 | default 16, services 10, hs 3 | 0 |
| ve | 9 | 8 | 9 | kind 8 | default 1 | 0 |
| vn | 20 | 4 | 19 | rule 7, kind 3 | default 7, services 1, kind 1 | 0 |
| xa | 4 | 3 | 4 | tag 3 | services 1 | 0 |
| xk | 5 | 1 | 4 | tag 1, default 1 | default 1, services 1 | 0 |
| za | 98 | 45 | 78 | tag 44, kind 1 | services 33 | 0 |
| zm | 3 | 0 | 2 |  | default 2 | 0 |
| zw | 3 | 0 | 2 |  | default 2 | 0 |
| all | 20,622 | 10,376 (50%) | 19,716 (95.6%) | | | 12 |

The 906 untyped lines are closed (greyed: za 20, ua 88, kp 23, mm 16, md 9, ...) or under 5
km running, except twelve: Germany's 5230 Waigolshausen - Gemünden, 1153 Lüneburg - Stelle,
6291, 6259, 6274, 2658, 5525, 4040 (freight bypasses and connecting curves with no mapped
service), and four US connectors (Kansas City District, Babylon Connector, Salt Lake
Subdivision, Youngstown Line).

## Spot checks (2026-10-09)

- **jp**: 東海道新幹線 high-speed; 東海道線 (JR East) commuter, also intercity and regional; 東海道線
  (JR Central) regional, also intercity; 山手線 commuter; 奥羽線 regional, also high-speed
  (Tsubasa, Komachi) and intercity; 常磐線快速 commuter; ふじさん, こうのとり intercity; つばめ
  high-speed; ゆりかもめ, ポートライナー, the New Shuttle people mover; Linimo maglev; 叡山ケーブル
  funicular.
- **de**: 6020 Berlin Ring commuter (S-Bahn, `light_rail` in RINF); 2690 Köln - Frankfurt
  high-speed; 1700 Hannover - Hamm high-speed, also regional, intercity, commuter; 4000
  Mannheim - Basel regional, also intercity, high-speed, commuter; 5500 München - Regensburg
  regional, also intercity, commuter; Fichtelbergbahn and Windbergbahn tourist.
- **fr**: LGV Sud-Est and LGV Est high-speed; Paris-Est - Strasbourg regional, also intercity,
  high-speed; RER A commuter; TER lines regional; TGV InOui and Eurostar high-speed; Chemin de
  fer de la Baie de Somme tourist.
- **gb**: WCML and ECML intercity, also regional; GWML intercity, also regional, commuter; HS1
  high-speed; Merseyrail and the Elizabeth line commuter; Settle - Carlisle and the Far North
  Line regional; West Somerset Railway tourist.
- **us**: Acela high-speed; the Northeast Corridor intercity, also high-speed, commuter; Hudson
  Line (Metro-North) commuter, also intercity; Metra BNSF commuter; Southwest Chief and Empire
  Builder night; Empire Service intercity.
- **ru**: Sapsan high-speed; Tver - Khovrino high-speed, also regional, intercity, commuter;
  Lastochka trains intercity; MCD lines and Aeroexpress commuter; Taishet - Irkutsk intercity;
  Moscow-Kurskaya - Kuskovo commuter, also regional and intercity; children's railways tourist.
- **cn**: 京沪高铁 and 京哈高速线 high-speed; 京广线, 京沪线, 青藏线 intercity; 京通线 intercity,
  also commuter; 北京市郊铁路S2线 commuter; 上海磁浮示范运营线 maglev; Shanghai Metro 16 and 18
  metro (not maglev, though the maglev company runs them).
- **in**: Mumbai's Western, Central and Harbour lines commuter; Konkan Railway intercity; Delhi
  - Kalka intercity; Kalka - Shimla, Darjeeling, Nilgiri tourist; Rajdhani trains intercity;
  Chennai Beach - Tambaram commuter.

## Soft spots

- Tagged values are taken as they are. Some taggers call a whole Japanese or Korean line
  `regional`; Japan ignores the tags for that reason, other countries do not.
- `also` lists what OSM maps. Where OSM maps only some services (Korea's Mugunghwa trains, most
  of India's expresses), `also` is short.
- The by-length default is a rule of thumb for 18 countries with almost no mapped services.
- A register line's main type between two kinds of the same share is the more local one, which
  makes 4000 Mannheim - Basel "regional, also intercity" rather than the other way round.
