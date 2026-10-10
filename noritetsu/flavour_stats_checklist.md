# Flavour statistics: what exists, per region

Research of 2026-10-07 for four optional extras on the map: a type label per line, average-day
ridership per station, ridership per line, and ridership per segment (the small grey number
beside each stretch of a line diagram). Every one of the 73 regions has a row in each table.
Nothing here was downloaded beyond small Wikidata query results; licences are as the source
pages state them and were not checked clause by clause.

How the numbers were measured:

- **Rail type**: read from `dist/data/<cc>/lines.json` (closed lines left out) and from the
  OSM route relations kept in `data/proc/<cc>/rels.pkl` (`service=` on the route, or on the
  routes of a route_master).
- **Wikidata**: one pass over the QLever Wikidata mirror (query.wikidata.org was rate-limited
  to one query a minute during an outage and kept answering 429). An item counts as a
  *station* if it has P197 (adjacent station), P81 (connecting line) or a station P31, and as
  a *line* if it has P559 (terminus), P1192 or a line P31. Items with P3872 (patronage) or
  P1373 (daily patronage) that are neither are mostly airports (US: 1,567 airports alone).
- **Effort**: easy = an open file or API, a join by name or code; moderate = scraping PDFs, a
  free account, several city sources, or a rule to write; hard = only press figures or a
  single city; impossible-ish = nothing published.

Sibling projects in `riders/` already hold:

| project | what it has that applies here |
|---|---|
| japanriders | S12 station 乗降客数 (8,532 grouped stations, mostly FY2021), gtfs-gis.jp 輸送密度 per section (CC BY 4.0, FY2018-2024), the 2015 census 駅間通過人員 for the Tokyo, Chukyo and Kinki areas |
| koreariders | Korail station 승하차 by direction (철도통계연보), KRIC per-line and per-train-type tables, the 광역철도 per-station board; per-segment loads *reconstructed* for 22 Korail lines and five city metros (labelled est.) |
| seoulriders | OA-12914 daily boardings per station for all 27 Seoul-area lines, 서울교통공사 혼잡도 (measured riders on board per station, direction and half hour, lines 1-9), one Sunday's OD and modelled segment loads |
| londonriders | NUMBAT 2024 per-line OD by quarter hour (Tube, DLR, Elizabeth, Overground) |
| nycriders | MTA subway OD 2024 (Wednesdays, hourly) |
| tokyoriders | census line-internal OD; undercounts through-running 2-4x, do not reuse for loads |

## 1. Rail type per line

What the build already gives: `kind` types every urban line (subway, light_rail, tram,
monorail, funicular; 3,962 lines), and register lines (`kind: rail`, 8,122) carry
`highspeed_sections` from OSM's `highspeed=yes` (346 register lines touch high-speed track).
What is missing is **commuter vs regional vs intercity**, and that belongs to services, not to
track: a register line is track, and the same track carries S-Bahn and ICE. So the label goes
on the OSM route lines (`kind: train`, 7,766), and a register line can show the mix of
services that run over it.

Measured: **81% of OSM train routes (6,297 of 7,766) already carry `service=`**
(`regional` 3,914, of which 1,622 are Russia's elektrichki; `long_distance`/`national`/
`international`/`night` 1,153; `commuter`/`suburban` 1,000; `high_speed` 92; the rest
`tourism` and odd values). It is not carried into `lines.json` yet: the extract keeps it
(`extract.REL_TAGS`), so adding a `service` field in `build_model.py` is the whole job for most
of Europe, Russia, Australia and South Africa. The gaps are Japan (5%), China (3%),
Indonesia (4%), Serbia (4%), Türkiye (9%), the US (21%), Canada (30%) and India (17%), where
a per-country rule on `network`/operator/name does it (listed in the notes).

Two other sources, not needed for a first pass:
- **RINF line category** (TSI traffic code P1-P6, P1 = new high-speed line) and maximum
  speed are on the same ERA endpoint `rinf.py` already queries, but the cached
  `data/raw/rinf/<cc>/sections.json` does not hold them; one extra query per RINF country
  would cross-check `highspeed=yes`. Moderate.
- **Japan's N02** already split Shinkansen / JR conventional / private and the guideway type
  when the register was built (`guided` marks AGT; Linimo is the one maglev).

Gaps worth knowing: the US has no track tagged high-speed (the Acela's NEC stretches run at
up to 150 mph, OSM leaves them untagged), Switzerland correctly has none, and AGT/people
movers outside Japan are typed `light_rail` or `monorail` (no `guided` flag); maglev exists
only in Japan (Linimo), China (Shanghai, Changsha, Beijing S1) and Korea (Incheon, closed).

| cc | name | lines (register / urban / OSM routes) | HS register lines (km) | OSM routes with `service=` | effort | note |
|---|---|---|---|---|---|---|
| al | Albania | 1 / 0 / 2 | 0 | 1/2 (50%) | easy (by hand) | few lines: 50% of routes tagged, the rest settled by hand in minutes |
| am | Armenia | 8 / 1 / 7 | 0 | 0/7 (0%) | easy (by hand) | few lines: 0% of routes tagged, the rest settled by hand in minutes |
| ar | Argentina | 17 / 12 / 19 | 0 | 19/19 (100%) | easy (by hand) | few lines: 100% of routes tagged, the rest settled by hand in minutes |
| at | Austria | 125 / 65 / 168 | 5 (324) | 167/168 (99%) | easy | 99% tagged; Westbahn HS sections |
| au | Australia | 141 / 38 / 89 | 0 | 88/89 (99%) | easy | 99% of routes tagged |
| az | Azerbaijan | 11 / 4 / 7 | 0 | 0/7 (0%) | easy (by hand) | few lines: 0% of routes tagged, the rest settled by hand in minutes |
| ba | Bosnia and Herz. | 2 / 6 / 2 | 0 | 1/2 (50%) | easy (by hand) | few lines: 50% of routes tagged, the rest settled by hand in minutes |
| be | Belgium | 147 / 47 / 104 | 4 (212) | 104/104 (100%) | easy | all tagged |
| bg | Bulgaria | 30 / 21 / 7 | 0 | 1/7 (14%) | easy (by hand) | few lines: 14% of routes tagged, the rest settled by hand in minutes |
| br | Brazil | 19 / 47 / 13 | 0 | 9/13 (69%) | easy | 69% of routes tagged |
| by | Belarus | 71 / 20 / 68 | 0 | 55/68 (81%) | easy | 81% tagged |
| ca | Canada | 131 / 36 / 30 | 0 | 9/30 (30%) | moderate | most OSM routes untagged; GO/exo = commuter, VIA = intercity by network |
| ch | Switzerland | 291 / 194 / 304 | 0 | 225/304 (74%) | easy | 74% tagged; no track tagged HS (correct: none above 200 km/h); many funiculars and trams |
| cl | Chile | 7 / 8 / 10 | 0 | 9/10 (90%) | easy (by hand) | few lines: 90% of routes tagged, the rest settled by hand in minutes |
| cn | China | 423 / 382 / 63 | 154 (49,425) | 2/63 (3%) | moderate | 154 register lines (49,425 km) carry `highspeed=yes`; OSM train routes have no `service=`; 12306 train-number letters (G/D/C vs K/T/Z) would label services; metro well typed |
| cz | Czechia | 237 / 88 / 183 | 1 (8) | 176/183 (96%) | easy | 96% tagged |
| de | Germany | 1054 / 534 / 872 | 50 (2,879) | 853/872 (98%) | easy | 98% tagged; S-Bahn mostly `commuter`, RE/RB `regional`; 50 register lines with HS sections (2,879 km) |
| dk | Denmark | 38 / 23 / 47 | 2 (79) | 47/47 (100%) | easy | 100% of routes tagged |
| dz | Algeria | 24 / 10 / 8 | 1 (252) | 2/8 (25%) | easy (by hand) | few lines: 25% of routes tagged, the rest settled by hand in minutes |
| ee | Estonia | 13 / 6 / 21 | 0 | 19/21 (90%) | easy (by hand) | few lines: 90% of routes tagged, the rest settled by hand in minutes |
| eg | Egypt | 30 / 5 / 3 | 0 | 0/3 (0%) | easy (by hand) | few lines: 0% of routes tagged, the rest settled by hand in minutes |
| es | Spain | 158 / 84 / 168 | 33 (3,753) | 164/168 (98%) | easy | 98% tagged; Cercanías `commuter`; 33 register lines with HS sections (3,753 km) |
| fi | Finland | 30 / 19 / 28 | 7 (415) | 28/28 (100%) | easy | 100% of routes tagged |
| fr | France | 278 / 155 / 539 | 10 (2,364) | 453/539 (84%) | easy | 84% tagged; TER as `regional`, Transilien/RER partly `commuter`; LGV from RFN category |
| gb | United Kingdom | 472 / 51 / 348 | 8 (826) | 277/348 (80%) | easy | 80% tagged; HS1 and parts of the ECML/WCML tagged HS |
| ge | Georgia | 13 / 3 / 13 | 0 | 1/13 (8%) | easy (by hand) | few lines: 8% of routes tagged, the rest settled by hand in minutes |
| gr | Greece | 6 / 8 / 19 | 0 | 18/19 (95%) | easy (by hand) | few lines: 95% of routes tagged, the rest settled by hand in minutes |
| hk | Hong Kong | 1 / 32 / 0 | 1 (25) | – | easy | all metro-type; East Rail and the XRL distinguishable by name |
| hr | Croatia | 35 / 21 / 70 | 0 | 69/70 (99%) | easy | 99% of routes tagged |
| hu | Hungary | 110 / 66 / 153 | 0 | 122/153 (80%) | easy | 80% tagged |
| id | Indonesia | 38 / 7 / 112 | 1 (141) | 4/112 (4%) | moderate | 108 of 112 OSM routes untagged; KAI Commuter vs KAI by `network` gives commuter vs intercity; Whoosh HS line tagged |
| ie | Ireland | 17 / 2 / 29 | 0 | 26/29 (90%) | easy (by hand) | few lines: 90% of routes tagged, the rest settled by hand in minutes |
| in | India | 724 / 59 / 94 | 0 | 16/94 (17%) | moderate | `service=` only on suburban routes; long-distance trains are named trains; no HS in service |
| ir | Iran | 24 / 14 / 17 | 0 | 9/17 (53%) | easy (by hand) | few lines: 53% of routes tagged, the rest settled by hand in minutes |
| it | Italy | 292 / 105 / 246 | 14 (975) | 232/246 (94%) | easy | 94% tagged; 14 lines with HS sections |
| jp | Japan | 438 / 239 / 416 | 9 (2,949) | 19/416 (5%) | moderate | N02 attributes already split Shinkansen / JR / private and the guideway type (monorail, AGT `guided`, funicular, tram, maglev Linimo); OSM `service=` is almost never set on Japanese routes, so commuter vs limited express needs a rule (e.g. 特急/新幹線 named trains = intercity, everything else local) |
| kg | Kyrgyzstan | 2 / 0 / 5 | 0 | 0/5 (0%) | easy (by hand) | few lines: 0% of routes tagged, the rest settled by hand in minutes |
| kr | South Korea | 50 / 52 / 33 | 8 (1,413) | 18/33 (55%) | moderate | KTX/SRT high-speed tagged; 수도권 전철 vs Korail intercity separable by network/operator |
| kz | Kazakhstan | 79 / 6 / 34 | 0 | 20/34 (59%) | easy | 59% tagged, mostly long-distance |
| lt | Lithuania | 16 / 1 / 17 | 0 | 15/17 (88%) | easy (by hand) | few lines: 88% of routes tagged, the rest settled by hand in minutes |
| lu | Luxembourg | 16 / 2 / 32 | 0 | 26/32 (81%) | easy (by hand) | few lines: 81% of routes tagged, the rest settled by hand in minutes |
| lv | Latvia | 10 / 13 / 29 | 0 | 10/29 (34%) | easy (by hand) | few lines: 34% of routes tagged, the rest settled by hand in minutes |
| ma | Morocco | 12 / 6 / 11 | 1 (193) | 0/11 (0%) | moderate | Al Boraq LGV tagged HS; routes untagged |
| md | Moldova | 4 / 0 / 3 | 0 | 3/3 (100%) | easy (by hand) | few lines: 100% of routes tagged, the rest settled by hand in minutes |
| me | Montenegro | 2 / 0 / 3 | 0 | 2/3 (67%) | easy (by hand) | few lines: 67% of routes tagged, the rest settled by hand in minutes |
| mk | North Macedonia | 1 / 0 / 2 | 0 | 0/2 (0%) | easy (by hand) | few lines: 0% of routes tagged, the rest settled by hand in minutes |
| mx | Mexico | 4 / 21 / 1 | 0 | 0/1 (0%) | easy | mostly metro; the one suburban line by name |
| my | Malaysia | 8 / 9 / 7 | 0 | 0/7 (0%) | moderate | routes untagged but few: KTM Komuter = commuter, ETS = intercity |
| nl | Netherlands | 96 / 60 / 132 | 3 (99) | 132/132 (100%) | easy | all tagged; NS intercity vs sprinter (`long_distance`/`regional`) |
| no | Norway | 25 / 19 / 32 | 0 | 29/32 (91%) | easy | 91% of routes tagged |
| nz | New Zealand | 15 / 4 / 13 | 0 | 13/13 (100%) | easy (by hand) | few lines: 100% of routes tagged, the rest settled by hand in minutes |
| pl | Poland | 336 / 218 / 311 | 3 (352) | 293/311 (94%) | easy | 94% tagged; CMK tagged HS |
| pt | Portugal | 23 / 32 / 46 | 0 | 46/46 (100%) | easy | 100% of routes tagged |
| ro | Romania | 66 / 77 / 39 | 0 | 36/39 (92%) | easy | 92% of routes tagged |
| rs | Serbia | 19 / 10 / 55 | 2 (127) | 2/55 (4%) | moderate | Belgrade - Novi Sad/Subotica tagged HS; 53 of 55 routes untagged |
| ru | Russia | 866 / 480 / 2136 | 9 (637) | 2120/2136 (99%) | easy | `service=` on 99% (1,622 regional = elektrichka, 404 long-distance); Sapsan track tagged HS |
| se | Sweden | 55 / 30 / 70 | 2 (212) | 63/70 (90%) | easy | 90% of routes tagged |
| sg | Singapore | 1 / 12 / 0 | 0 | – | easy | all metro/LRT/monorail, fully typed by `kind` |
| si | Slovenia | 22 / 2 / 12 | 0 | 12/12 (100%) | easy (by hand) | few lines: 100% of routes tagged, the rest settled by hand in minutes |
| sk | Slovakia | 49 / 23 / 81 | 0 | 81/81 (100%) | easy | 100% of routes tagged |
| th | Thailand | 16 / 9 / 8 | 0 | 0/8 (0%) | moderate | routes untagged; SRT Red Line = commuter, SRT intercity by name |
| tj | Tajikistan | 4 / 0 / 1 | 0 | 1/1 (100%) | easy (by hand) | few lines: 100% of routes tagged, the rest settled by hand in minutes |
| tm | Turkmenistan | 9 / 0 / 0 | 0 | – | easy (by hand) | few lines; label by hand |
| tn | Tunisia | 10 / 7 / 1 | 0 | 1/1 (100%) | easy (by hand) | few lines: 100% of routes tagged, the rest settled by hand in minutes |
| tr | Turkey | 40 / 59 / 45 | 6 (1,290) | 4/45 (9%) | moderate | YHT track tagged HS (1,290 km); 41 of 45 TCDD routes untagged |
| tw | Taiwan | 17 / 23 / 16 | 1 (348) | 6/16 (38%) | moderate | THSR tagged high-speed; TRA routes untagged |
| ua | Ukraine | 220 / 120 / 73 | 0 | 48/73 (66%) | easy | 66% tagged |
| us | United States of America | 502 / 244 / 177 | 0 | 38/177 (21%) | moderate | no track tagged high-speed (Acela runs on NEC at up to 150 mph, OSM leaves it untagged); 139 of 177 OSM routes have no `service=`; commuter networks (LIRR, Metra, NJT...) identifiable by `network`/operator |
| uz | Uzbekistan | 28 / 6 / 6 | 10 (575) | 5/6 (83%) | easy | Afrosiyob track tagged HS |
| vn | Vietnam | 8 / 4 / 7 | 0 | 0/7 (0%) | moderate | untagged; all intercity apart from Hanoi/HCMC metro |
| xa | Abkhazia | 0 / 0 / 3 | 0 | 3/3 (100%) | easy (by hand) | few lines: 100% of routes tagged, the rest settled by hand in minutes |
| xk | Kosovo | 2 / 0 / 2 | 0 | 1/2 (50%) | easy (by hand) | few lines: 50% of routes tagged, the rest settled by hand in minutes |
| za | South Africa | 33 / 1 / 44 | 1 (3) | 44/44 (100%) | easy | all tagged; Metrorail commuter, Shosholoza long-distance |

## 2. Ridership per station, average day

Wikidata is not a shortcut outside a few countries: **7,432 station items worldwide carry
P3872/P1373 in the 73 regions, and 5,533 of them are Japanese.** Belgium (558, all 2023),
Taiwan (270), Australia (220, none after 2013), the UK (129, none after 2018), Italy (101,
2019 or older), Switzerland (98), Luxembourg (81, no year), Germany (79) and Türkiye (78) are
the only others above 50. OSM station nodes carry a `wikidata=` tag at 1-81% depending on the
country (jp 81%, be 81%, cz 80%, gb 78%; ru 14%, kz 4%), so a Wikidata join would also need
that coverage. Column "WD" below is station items with patronage (of them, dated 2015 or
later).

| cc | name | status | source (licence, format, years, grain, login) | effort | note |
|---|---|---|---|---|---|
| jp | Japan | yes, all stations | MLIT 国土数値情報 S12 駅別乗降客数 (open, GeoJSON/shp, FY2011-2021 in the copy japanriders holds, newer editions out); WD 5,533 (4,278) | easy | already in `riders/japanriders/data/S12-22`; per operator, group by S12_001g; mind the FY2019 joint-figure convention change |
| kr | South Korea | yes, all | KRIC `railstapassmonList.jsp` (Korail, monthly, by train type), Korail 광역철도 board (commuter lines, per station, 2017-2026), Seoul OA-12914 (27 metro lines, daily), KRIC `citystapassList` (city metros); no login; WD 58 (6) | easy | koreariders and seoulriders already parse these; SR (SRT) stations missing from KRIC |
| tw | Taiwan | yes, all | data.gov.tw: TRA daily entries/exits per station (dataset 8792), Taipei MRT hourly per station since 2015 + OD, Taichung MRT per station (175718), THSR per station (not opened); Open Government Data Licence; WD 270 (254) | easy | Kaohsiung MRT and the Taoyuan airport line not checked |
| gb | United Kingdom | yes, all NR + London | ORR Estimates of station usage, table 1410/1415 (CSV/ODS, Apr-Mar years 1997-2025, entries+exits per station, ~2,580 stations), dataportal.orr.gov.uk/station-usage, no login; TfL station entry/exit counts on crowding.data.tfl.gov.uk; WD 129 (6) | easy | annual totals, divide by 365 for an average day; Glasgow Subway, Tyne and Wear Metro, trams need their operators |
| fr | France | yes, all SNCF + Paris | SNCF "Fréquentation en gares" (data.gouv.fr / ressources.data.sncf.com, ~3,000 stations, annual 2015-2024, SNCF open data licence); IDFM "Validations sur le réseau ferré" (725 RER/metro/Transilien stops, daily, quarterly files); WD 19 (7) | easy | SNCF figures include RATP traffic at shared stations; trams and provincial metros need each network |
| ch | Switzerland | yes, SBB stations | SBB `passagierfrequenz` (data.sbb.ch / opendata.swiss, CSV/JSON, DTV/DWV per station, annual, "open use, cite source", updated Sept 2025); WD 98 (87) | easy | private railways (RhB, BLS, MGB) partly covered; trams and funiculars not |
| nl | Netherlands | yes, all NS | NS in/uitstappers per station, average weekday, 2000-2024 (file on the Zuid-Holland open data portal; also treinreiziger.nl CSVs) | easy | metros and trams (GVB, RET, HTM) not included |
| be | Belgium | yes, all | SNCB October counts, boardings per station (weekday/Sat/Sun), published yearly; WD 558 (555), all 2023 | easy | Wikidata alone covers it; boardings only, double for "users" |
| au | Australia | yes, 3 cities | NSW: Train, Metro and Light Rail Station Entries and Exits (opendata.transport.nsw.gov.au, CC BY, monthly CSV, Opal); Victoria: Annual metropolitan (and regional) train station patronage (data.vic.gov.au, CC BY 4.0, daily averages by day type and time band); Queensland: Translink origin-destination trips 2022 on (data.qld.gov.au, monthly, stop level); WD 220 (0) | easy | Perth and Adelaide not checked; NSW TrainLink and V/Line regional partly covered |
| my | Malaysia | yes, Klang Valley + ETS | data.gov.my `ridership_od_rapidrail_daily` (every LRT/MRT/monorail station pair, daily, parquet), `ridership_od_ets` (hourly), KTM Komuter likely the same family | easy | an OD file gives station totals and segment loads both |
| sg | Singapore | yes, all | LTA DataMall "Passenger Volume by Train Stations" (PV/Train, monthly, tap in/out by hour, weekday/weekend) and PV/ODTrain; free account + API key | moderate | account only, no identity check |
| th | Thailand | yes, Bangkok | Department of Rail Transport (DRT) "ปริมาณผู้โดยสารระบบรถไฟฟ้าขนส่งมวลชน" (datagov.mot.go.th drt2566_02 and later, CSV, per station, open data licence); SRT intercity none | easy | BTS, MRT, ARL, Red Line |
| tr | Türkiye | yes, Istanbul | İBB "Raylı Sistemler İstasyon Bazlı Yolcu ve Yolculuk Sayıları" (data.ibb.gov.tr, monthly per station, İBB open data licence); WD 78 (72) | easy | Istanbul only (incl. Marmaray?); Ankara, İzmir (İZBAN) and TCDD not found |
| mx | Mexico | yes, Mexico City metro | datos.cdmx.gob.mx Metro afluencia diaria por estación (CSV, daily, open); WD 2 | easy | Monterrey, Guadalajara, Tren Suburbano not checked |
| us | United States | patchwork | Amtrak State Fact Sheets (PDF per state, boardings+alightings per station, FY2024/25); MTA subway hourly ridership and OD (data.ny.gov); CTA 'L' station entries daily since 2001 (Chicago portal); BART monthly OD; WMATA ridership portal; NTD is agency-level only; WD 34 (18) | moderate | one parser per agency; Amtrak needs PDF extraction; commuter rail (LIRR, Metro-North, NJT, Metra) not checked |
| ca | Canada | partial | TTC subway ridership by station (PDF, Sept 2023-Aug 2024); STM and GO not found in this pass; VIA none; WD 23 (22) | moderate | |
| es | Spain | partial | Renfe Data "Volumen de viajeros por franja horaria" per Cercanías núcleo (CSV per station and time band; Barcelona Rodalies regularly); Adif station totals only in the press; WD 44 (44) | moderate | long-distance and metro stations not covered |
| it | Italy | partial | Regione Lombardia open data (Trenord saliti/discesi surveys, dati.lombardia.it); WD 101 (12) | moderate | RFI publishes nothing per station |
| se | Sweden | partial | Region Stockholm "Fakta om SL och regionen" (PDF, boardings per station for tunnelbana, pendeltåg, lokalbanor); WD 6 | moderate | national: Trafikverket/Trafikanalys have nothing per station |
| pl | Poland | yes, banded | UTK "wymiana pasażerska" average daily boardings+alightings per station (dane.utk.gov.pl); small stations shown as bands (0-9, 20-49 ...) | moderate | exact figures for big stations only; WD 1 |
| ie | Ireland | yes, one day | NTA Heavy Rail Census (every station and service, one November day, boardings and alightings; report PDFs at nationaltransport.ie, 2014-2021+) | moderate | PDF tables; Luas not included |
| ar | Argentina | partial | Buenos Aires subte turnstile entries per station, 15-minute (data.buenosaires.gob.ar); CNRT statistics reports for the commuter lines (PDF, per line, some per station) | moderate | |
| lu | Luxembourg | WD only | WD 81 (0, no date on the statements); CFL publishes per-line totals only | moderate | the WD figures need a date before use |
| ru | Russia | Moscow only | data.mos.ru metro datasets (entrances/vestibules; per-station flow dataset not confirmed); RZD nothing per station; WD 2 | hard | |
| ua | Ukraine | WD only | WD 15 (11), Kyiv metro | hard | |
| in | India | one city | Bengaluru BMRCL hourly per station (data.opencity.in, from Aug 2025); Indian Railways none (station footfall only through RTI answers); WD 40 (31) | hard | |
| br | Brazil | partial | Metrô São Paulo transparency site (indicator PDFs); CPTM, SuperVia not found; WD 4 | hard | |
| cl | Chile | none open | Metro de Santiago annual results PDF; DTPM per-station data not found; WD 1 | hard | |
| cz | Czechia | partial | PID press releases (train passengers per Prague station); no SŽ or ČD dataset; WD 9 (8) | hard | |
| de | Germany | none open | DB publishes no station footfall (data.deutschebahn.com and OpenStation hold infrastructure only); WD 79 (40) | hard | some Verkehrsverbünde publish counts in reports |
| at | Austria | none open | ÖBB/Wiener Linien publish only examples; WD 8 | hard | |
| hk | Hong Kong | none | MTR publishes system totals and patronage updates only | impossible-ish | |
| cn | China | none | no operator or ministry publishes station counts; a 2017 Shanghai research set (figshare 10.6084/m9.figshare.28844942); WD 1 | impossible-ish | |
| fi, no, dk, hu, pt, ro, gr, id, nz, za | Finland, Norway, Denmark, Hungary, Portugal, Romania, Greece, Indonesia, New Zealand, South Africa | not found | searched: Väylä/Statistics Finland (StatFin has national railway tables, station split unverified), Bane NOR (per-station counts exist in planning PDFs, no file), Danmarks Statistik (national totals), MÁV, Auckland Transport (2013-14 only, via an FYI request), KAI/BPS (operating-area totals); WD 0-6 each | hard | |
| al, am, az, ba, bg, by, dz, ee, eg, ge, hr, ir, kg, kz, lt, lv, ma, md, me, mk, rs, si, sk, tj, tm, tn, uz, vn, xa, xk | the remaining 30 | not found | only the Wikidata count was run (0 or 1 station each) plus general knowledge: none of these railways or statistics offices is known to publish station counts | impossible-ish | |


## 3. Ridership per line

Wikidata has patronage on only **369 line items** across the 73 regions (us 101, de 70,
jp 32, fr 31, cn 15, ca 14, it 14, ar 13, tr 8, in 7, ru 7, kr 6, mx 6; most others 0-5), and
those are mostly metro, S-Bahn and named commuter lines. A line figure can also be built by
summing a line's station counts, but only where a station belongs to one line; at
interchanges it cannot be split.

| cc | name | status | source | effort | note |
|---|---|---|---|---|---|
| jp | Japan | yes, every line and section | gtfs-gis.jp 輸送密度 (CC BY 4.0, FY2018-2024, per line or published section); WD 32 | easy | in japanriders already; 輸送密度 is passengers per km per day, the natural line figure |
| kr | South Korea | yes | KRIC `raillinepassdivList` (Korail per line, 주운행선 rule), `capitallinepassList` (수도권 per line, monthly), `cityorganpassList` (city metros); 철도통계연보 선별 통과인원; WD 6 | easy | koreariders has the fetchers |
| tw | Taiwan | yes | data.gov.tw per-line monthly tables (Taipei MRT, Taichung MRT); TRA per line via station sums | moderate | |
| my | Malaysia | yes | data.gov.my `ridership_headline` (daily per service: each LRT/MRT/monorail line, KTM Komuter, ETS) | easy | |
| th | Thailand | yes | DRT open data, daily per line | easy | |
| tr | Türkiye | Istanbul | İBB "Raylı Sistemler Hat Bazlı Yolculuk Sayıları" (per line); WD 8 | easy | |
| gb | United Kingdom | London only | TfL annual per-line figures and NUMBAT (londonriders); ORR reports by operator, not by line; WD 4 | moderate | National Rail has no per-line ridership |
| fr | France | Paris + some | RATP annual traffic per metro/RER line; SNCF TER by region only; WD 31 | moderate | |
| ch | Switzerland | derivable | ARE section loads (stat 4) give any line's figure; WD 1 | moderate | |
| us | United States | patchwork | Amtrak per route (fact sheets, monthly performance reports); NTD per agency and mode (open, annual); MTA/CTA per line by station sums; WD 101 | moderate | |
| ar | Argentina | yes, commuter | CNRT statistics reports per line (PDF, quarterly/annual); WD 13 | moderate | |
| au | Australia | partly | NSW and Victoria publish patronage by line or by mode in their open data; WD 1 | moderate | not opened |
| de | Germany | WD only | WD 70 (S-Bahn/U-Bahn lines, mixed years) | moderate | DB publishes nothing per line |
| cn | China | per city, some lines | CAMET annual urban rail statistics report (per city; some per-line figures), city transport commission releases; WD 15 | hard | |
| ca, it, in, mx, ru, es, ua, sg, vn, id, fi, hu, br, cl | | WD only or press | WD counts: ca 14, it 14, in 7, ru 7, mx 6, es 5, ua 3, sg 4, id 3, br 5, cl 3, fi 1, hu 1, vn 1 | hard | figures for metros and a few commuter lines only |
| hk | Hong Kong | none | MTR publishes totals by service type, not by line | impossible-ish | |
| be, nl, at, pl, cz, se, dk, no, pt, ie, lu, gr, ro, sk, si, hr, bg, rs, ba, me, mk, al, xk, ee, lv, lt, by, md, kz, uz, kg, tj, tm, ge, am, az, xa, ir, eg, ma, dz, tn, za, nz | the remaining 44 | not found (station sums possible where stat 2 has data: be, nl, pl, ie) | WD 0-1 each; no operator per-line series found | hard (be, nl, pl, ie: moderate by station sums) / impossible-ish elsewhere | |

## 4. Ridership per segment

Published per-segment loads exist in four places. Elsewhere a segment figure has to be
estimated, and how well depends on what the station data says:

- **OD between stations** (Taipei, Klang Valley, Singapore, Queensland, NYC, BART, Seoul's
  one Sunday) gives segment loads by routing every trip, which is what the riders projects do.
- **Boardings by direction** (Korea's 상행/하행) let loads be cumulated along a line: how
  koreariders works.
- **Plain station totals** only fix how many board and alight; a load still needs an
  assumption about where people go (a gravity model with decay fitted to a known mean trip
  length). Reasonable on a single line with no branches, weak on networks; label it estimated.

| cc | name | status | source | effort | note |
|---|---|---|---|---|---|
| jp | Japan | published | gtfs-gis.jp 輸送密度 per section (national, CC BY 4.0); 大都市交通センサス 駅間通過人員 per station pair for the three metro areas (2015; MLIT xlsx) | easy | japanriders has both, stitched per segment; census undercounts long-distance trips |
| ch | Switzerland | published (model) | ARE "Belastung (Personen) des schweizerischen Schienennetzes" (NPVM model; average day and workday, peak hours; opendata.swiss / data.geo.admin.ch, open use, cite source, updated July 2025); trams included | easy | modelled, not counted, but national and per section |
| gb | United Kingdom | London only | TfL NUMBAT link loads by quarter hour (crowding.data.tfl.gov.uk); londonriders has the per-line OD | easy (London) | National Rail: none |
| kr | South Korea | measured for Seoul, reconstructed elsewhere | 서울교통공사 혼잡도 (data.go.kr 15071311, lines 1-8; OA-22197 line 9); koreariders reconstructs Korail and five city metros | moderate | already built; say "est." outside Seoul |
| ie | Ireland | published, one day | NTA Heavy Rail Census reports give loads between stations per service | moderate | PDF |
| tw | Taiwan | estimable | Taipei MRT OD (data.gov.tw), TRA station counts | moderate | |
| my | Malaysia | estimable | Rapid Rail daily OD, ETS OD | moderate | |
| sg | Singapore | estimable | DataMall OD by train station | moderate | free account |
| au | Australia | estimable | Queensland Translink OD (monthly); NSW and Victoria station totals only | moderate (Qld) / hard | |
| us | United States | estimable in NYC and SF | MTA subway OD (nycriders), BART OD; elsewhere station totals only | moderate | |
| be, nl, fr, pl, th, tr, mx, ar | | station totals only | estimates need a gravity model per line | hard | |
| all other regions (55) | | none | no OD, no directional counts, no published loads | impossible-ish | |

## Recommended first wins

1. **Rail type, most of the world in one change**: carry the OSM route's `service=` into
   `lines.json` (the extract already keeps it). 81% of the 7,766 OSM train routes get a label
   at once; urban lines are already typed by `kind`, high-speed track by
   `highspeed_sections`. Then small per-country rules for jp, cn, us, ca, id, in, tr, rs, kr,
   tw, my, th, vn (network or operator or train-name prefix).
2. **Japan, everything**: station counts (S12), line and section figures (輸送密度) and the
   metro census are already downloaded and joined in japanriders; noritetsu's largest region
   by station count could show all four statistics. Wikidata also has 5,533 Japanese
   stations with patronage as a cross-check.
3. **Station counts from six open national files**: ORR (gb), SNCF fréquentation (fr), SBB
   passagierfrequenz (ch), NS (nl), Belgian counts through Wikidata (be), TRA + Taipei MRT
   (tw). Each is one CSV and a name join; together about 12,000 stations.
4. **Korea**: reuse koreariders and seoulriders (station counts, per-line, per-segment
   estimates, Seoul's measured 혼잡도).
5. **Switzerland per segment**: the ARE load layer is per section for the whole network and
   would be the second country (after Japan) with a grey number on every segment.
6. **City OD sets** (Klang Valley, Taipei, Singapore, Brisbane, NYC, BART) for station and
   segment figures on those networks, after the national files.

Coverage headline: rail type is available now or with a short rule for all 73 regions;
station ridership has an open source covering most stations in 14 regions (jp, kr, tw, gb,
fr, ch, nl, be, au, my, sg, th, tr, mx) and partial, banded or single-city data in 14 more; line
ridership is complete in about 6; segment ridership is published for Japan, Switzerland,
London and (one day) Ireland, measured for Seoul, and estimable from OD in about six more
networks.
