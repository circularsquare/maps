# Timetable feeds as the "is this section ridden" check (surveyed 2026-10-01)

The question: for each built and next country, is there an open GTFS (or NeTEx, or a HAFAS
export) of national passenger rail that can tell which register sections passenger trains
run over, so `build_model.drop_unridden_sections` can stop relying on OSM route relations?

Everything below was checked on 2026-10-01 from the session scratchpad (scripts not kept in
the project): the Mobility Database catalogue CSV, the Transitous repository's `feeds/*.json`,
and every feed named here was opened remotely (HTTP range requests on the zip's central
directory, then agency, routes, feed_info, calendar and the head of stops.txt). Nothing used
needed a login. Where the original needs one, an open mirror is named beside it.

## Short answer

Every built country except mainland China has an open feed now, at least through the
Transitous mirror. So does every "next" country except Russia. Czechia and Hungary were
tested against their RINF registers and the method works: in Czechia the feed covers
9,085 of RINF's 9,685 km and keeps 238's approach to Havlíčkův Brod, 292 around Bludov and
201 Nasavrky - Tábor, all of which the OSM rule dropped. It also says 253 Vranovice -
Pohořelice and 345 Nemotice - Koryčany have no trains. In Hungary it finds no trains at all
on lines 27, 37 and 62, or on 150, which the build currently draws.

Feeds that need a key or registration, with an open mirror that works without one:
Hungary (MÁV form), Austria (Mobilitätsverbünde account), Slovenia (NAP account), Sweden
(Trafiklab key), Taiwan (TDX key), New South Wales (Transport for NSW key). Nothing at all:
China, Russia. Partial: South Korea (Korail intercity trains only), Japan (no official
national feed; an all-operator feed of unknown origin plus ODPT for Tokyo).

## Aggregators

- Mobility Database (MobilityData; the successor to OpenMobilityData/TransitFeeds, whose
  archive it absorbed: rows pointing at `openmobilitydata-data.s3...` are old snapshots).
  Catalogue CSV `https://files.mobilitydatabase.org/feeds_v2.csv`, no login; the API needs a
  free account. Useful for finding feeds and their licence pages. It marks which need a key
  (`urls.authentication_type`), e.g. every Swedish Samtrafiken feed.
- Transitous (`github.com/public-transport/transitous`, sources in `feeds/<cc>.json`). Its
  processed copies are all at `https://api.transitous.org/gtfs/<cc>_<name>.gtfs.zip`, open,
  refreshed daily or so, and the server answers HTTP range requests, so a zip's file list and
  small tables can be read without downloading it. It holds keyed sources too (MÁV,
  Mobilitätsverbünde, nap.si, Trafiklab, NSW), and it generates feeds where no operator
  publishes one (`jbb.ghsq.de`: Hellenic Train, Romanian railways from data.gov.ro, LTG Link,
  Japan, Ukraine, Georgia, Armenia). The licence is the source's; Transitous says the files
  are there "for use again". This is the one-stop source for a generic reader.
- gtfs.de: free CC BY 4.0 feeds for Germany from DELFI data, split by mode:
  `https://download.gtfs.de/germany/fv_free/latest.zip` (long distance, 0.4 MB) and
  `.../rv_free/latest.zip` (regional rail incl. non-DB railways, 11 MB); also `nv_free`
  (local) and `full`. A rolling 30-day window, so seasonal lines can be missed.
- transit.land: free API key; its atlas (feed URLs and licences) is on GitHub. Used here
  only to find CP's official URL.
- EU National Access Points (MMTIS), from the NAPCORE monitoring list
  (`eunapmonitoring.napcore.imet.gr/national_access_points.html`). Rail timetables were found
  in: Austria `mobilitydata.gv.at` (points to data.mobilitaetsverbuende.at, account), Belgium
  `transportdata.be`, Czechia `registr.dopravniinfo.cz` (the CIS JŘ files are at
  portal.cisjr.cz), Germany `mobilithek.info` (DELFI), Estonia `peatus.ee`, Finland
  `finap.fi` (rail itself is digitraffic), France `transport.data.gouv.fr`, Italy `cciss.it`
  (Trenitalia and Italo NeTEx), Portugal `nap-portugal.imt-ip.pt` (CP), Slovenia `nap.si`
  (account), Spain `nap.transportes.gob.es` (Renfe also publishes directly), Latvia
  `transportdata.gov.lv`, Greece `data.nap.gov.gr`, Hungary `napportal.kozut.hu`, Romania
  (several data.gov.ro pages), Sweden `trafficdata.se` (Trafiklab), Switzerland
  `opentransportdata.swiss`, Denmark `du.vd.dk`, Bulgaria
  (`mtc.government.bg`), Croatia `promet-info.hr` (HŽPP's own GTFS is simpler). The list
  shows no MMTIS NAP for Poland, Slovakia, Lithuania, Ireland, Norway or Cyprus; these
  countries' feeds come from operators or community converters instead. Only some NAPs
  were opened to confirm rail is in them (Italy, Portugal, Slovenia, Spain, Czechia).

A correction to `multi_sources.md`: Entur's national GTFS needs no key
(`storage.googleapis.com/marduk-production/outbound/gtfs/rb_norway-aggregated-gtfs.zip`, 607
MB, downloaded headers checked).

## Built countries

Station ids: "UIC" means the stop_id contains the 7-digit UIC station code; "RINF code"
means it contains the same number as RINF's `uopid`.

| country | feed | covers | shapes | licence | access | updated | verdict |
|---|---|---|---|---|---|---|---|
| Czechia | Oběhy CZPTT GTFS `motis.obehy.cz/get-feeds/cz-czptt-gtfs.zip`, converted from SŽ's CIS JŘ XML (`portal.cisjr.cz/pub/draha/celostatni/szdc/`) | every carrier on SŽ's network, 41 agencies (ČD, RegioJet, Leo Express, Arriva, GW Train, KŽC, cross-border DB/ÖBB/PKP/ZSSK); not JHMD | no | CC0 (as Transitous lists it) | direct | daily; whole timetable year | works, tested. stop_id `CZ:54943` is RINF `CZ54943`, so 2,664 points match by code |
| Hungary | MÁV `mavcsoport.hu/gtfs/gtfsMavMenetrend.zip` (HTTP basic auth; free credentials from the form at `mavcsoport.hu/gtfs-igenybejelento`); open mirror `api.transitous.org/gtfs/hu_mav.gtfs.zip` | MÁV, GYSEV, Gyermekvasút; replacement buses as route_type 3 | no | CC0 (as Transitous lists it) | registration; mirror direct | daily; June - December 2026 | works, tested. MÁV's own stop ids, so names (97% matched) |
| Poland | Polish-Trains `mkuran.pl/gtfs/polish_trains.zip` (12 regional operators, CC BY 4.0) + PKP Intercity `gtfs.kasznia.net/static/pkp-ic.zip` + narrow gauge `gtfs.kasznia.net/static/narrow-gauge.zip` | all passenger operators between them | PKP IC yes (412 MB uncompressed), regional no | CC BY 4.0 | direct | daily; regional covers about 30 days | works. PLK station numbers (`plk_secondary_id` too), not RINF's PL00010 form, so names |
| Austria | Mobilitätsverbünde "Railway Current Reference Data" (`data.mobilitaetsverbuende.at`, free account; 401 without); mirror `api.transitous.org/gtfs/at_Railway-Current-Reference-Data-2026.gtfs.zip` | ÖBB, WESTbahn, Raaberbahn/GYSEV, Montafonerbahn, cross-border DB, ČD, MÁV, SŽ, SBB, PKP IC | no | Mobilitätsverbünde terms | account; mirror direct | weekly; whole year | works for the RINF network. Zillertalbahn, GKB, Stern & Hafferl, Wiener Lokalbahnen and Salzburger Lokalbahn are in the regional PTA feeds of the same portal |
| Belgium | SNCB `sncb-opendata.hafas.de/gtfs/static/c21ac6758dd25af84cca5b707f3cb3de` (token in the URL) | NMBS/SNCB | no | NMBS open-data terms | direct | weekly; whole year | works. UIC (88xxxxx) stop ids; RINF's are PTCAR letters (BEMGV), so names |
| Netherlands | OVapi `gtfs.ovapi.nl/nl/gtfs-nl.zip` (250 MB, all modes) | NS, NS International, Arriva, Blauwnet, R-net, RRReis, Qbuzz trains | yes | CC0 | direct | daily | works; RINF ids are station codes (NLAc), names needed |
| Portugal | CP `publico.cp.pt/gtfs/gtfs.zip` (via Portugal's NAP) | CP only; Fertagus separately (Transitland f-eyce-fertagus) | no | NAP licence | direct | per timetable; whole year | works, live. stop_id `94_3046` is RINF's `PT03046` (the `CODE` rule) |
| France | SNCF `eu.ftp.opendatasoft.com/sncf/plandata/Export_OpenData_SNCF_GTFS_NewTripId.zip` (also NeTEx) | SNCF Voyageurs: TGV INOUI, OUIGO, Intercités, TER | no | ODbL | direct | daily | works. UIC in stop_id (`OCE87313759`). Add IDFM for Transilien and RER (`eu.ftp.opendatasoft.com/stif/GTFS/IDFM-gtfs.zip`), and Eurostar, Trenitalia France, CFC and Chemins de fer de Provence from transport.data.gouv.fr |
| Switzerland | opentransportdata.swiss `data.opentransportdata.swiss/de/dataset/timetable-2026-gtfs2020/permalink` (289 MB) | everything, 456 agencies incl. funiculars and buses | no | opentransportdata.swiss terms (free, attribution) | direct for static GTFS | weekly; whole year | works. didok/sloid ids (UIC 85xxxxx). Swiss sections end at junctions more than any other register, so this is where shapes are most missed |
| Japan | none official national. `api.transitous.org/gtfs/jp_japan-rail.gtfs.zip` (generated at jbb.ghsq.de, 130 operators: JR, private railways, metros, every train its own route) | most of Japan | no | not stated | direct | daily | usable as a check, but origin unknown. ODPT (`developer.odpt.org`, free key) has JR East and Tokyo operators; `mkuran.pl/gtfs/tokyo/rail.zip` (46 Tokyo-area operators, shapes, MIT); GTFS-JP repository (`gtfs-data.jp`) for small railways. N02 already marks passenger lines, so this matters least here |
| South Korea | KorailGTFS `mkuran.pl/gtfs/korail.zip` | Korail intercity only (KTX, ITX, Mugunghwa, Nuriro); not SRT, commuter lines or metros | yes | CC BY 4.0 | direct | weekly | partial. The national KTDB GTFS needs a request at ktdb.go.kr; a 2025 copy in Transitous (`kr_korea`) labels ferries as rail |
| Taiwan | TDX (`tdx.transportdata.tw`, free key); mirror `github.com/jcanizalez/tdx-gtfs-mirror/releases/download/latest/tw-gtfs.zip` (465 MB, all modes) | TRA, THSR, Alishan, metros | yes | Taiwan open government data licence, attribution | key; mirror direct | daily | works |
| Hong Kong | community `feed.justusewheels.com/hk.gtfs.zip` | MTR, Light Rail, tram as frequency patterns | yes | ODbL | direct | | not needed: every HK register line carries passengers |
| Singapore | LTA-derived feed (Mobility Database mdb-1076) | buses plus 13 MRT/LRT routes, frequency-based | yes | not stated | direct | stale (2024) | not needed, as for Hong Kong |
| China | none | | | | | | nothing. 12306's `train_list.js` still answers (15 MB) but its newest dates are 2022; per-train stop lists exist only behind the ticketing site's JSON. Not tried further |

## Next countries

| country | feed | covers | shapes | licence | access | updated | verdict |
|---|---|---|---|---|---|---|---|
| Slovakia | ŽSR `data.slovensko.sk/download?id=c5a63281-3c44-4dba-82c9-7b7ad603db5d` | ZSSK, RegioJet, Leo Express, TEŽ, LTE | no | not stated on the download | direct | per timetable (version 2026-05-13, valid to December) | works. UIC stop ids (56xxxxx) |
| Romania | `jbb.ghsq.de/gtfs/ro-railway.gtfs.zip` (Transitous), from the operators' XML timetables on data.gov.ro | CFR Călători, Regio Călători, Astra Trans Carpatic, Softrans, Transferoviar, InterRegional, Ferotrafic | yes (generated) | OGL-ROU 1.0 | direct | weekly; whole year | works, live. stop_id `60309` is RINF's `RO60309` (Tecuci; the `CODE` rule) |
| Bulgaria | livetransport.eu `gtfs.livetransport.eu/gtfs/bdz.zip` | BDZ | yes | CC BY 4.0 | direct | daily; 30-day window | works, unofficial; seasonal trains can fall outside the window |
| Finland | Fintraffic `rata.digitraffic.fi/api/v1/trains/gtfs-passenger-stops.zip` (send `Accept-Encoding: gzip`, else 406) | VR and HSL commuter trains, i.e. all passenger rail | yes | CC BY 4.0 | direct | daily; to end of 2027 | works. Station short codes; RINF `FI00094` is probably digitraffic's station number (not tested) |
| Slovenia | NAP `b2b.nap.si/data/b2b.gtfs` (free NAP account; 401 without); mirror `api.transitous.org/gtfs/si_nap.gtfs.zip` | SŽ passenger + buses | yes | CC BY-SA 4.0 | account; mirror direct | daily | works |
| Lithuania | `jbb.ghsq.de/gtfs/lt-ltglink.gtfs.zip` (Transitous, generated from LTG Link) | LTG Link domestic | yes | not stated | direct | daily; 2-month window | works, unofficial. Stop ids are names |
| Latvia | pieturas.lv aggregate (Mobility Database mdb-2337; mirror `api.transitous.org/gtfs/lv_pieturas.gtfs.zip`) | Vivi (all passenger rail), Gulbene - Alūksne narrow gauge, buses | yes | CC0 | direct | daily | works. The NAP (transportdata.gov.lv) was not opened |
| Estonia | Elron `eu-gtfs.remix.com/elron.zip` (national: `eu-gtfs.remix.com/estonia_unified_gtfs.zip`) | Elron, all domestic passenger rail | yes | CC0 | direct | weekly | works |
| Croatia | HŽPP `hzpp.hr/GTFS_files.zip` | HŽ Putnički prijevoz only: no SŽ, MÁV or ŽRS trains | yes | CC0 | direct | daily; whole year | works, live (`NO_INTERNATIONAL`) |
| Luxembourg | national feed on data.public.lu (`data.public.lu/api/1/datasets/horaires-et-arrets-des-transport-publics-gtfs/`; a new file name each week, `fetch` takes the newest) | CFL and every bus; slimmed to 0.2 MB | | CC BY 4.0 | direct (Python's certificate check fails there on this machine; `fetch` falls back to curl) | weekly; 30 Sep - 12 Dec 2026 | live (2026-10-01). Stop ids are 12-digit numbers, names end ", Gare" (a generic word), so names match |
| Greece | `jbb.ghsq.de/gtfs/gr-hellenic-train.gtfs.zip` (Transitous, generated) | Hellenic Train (plus ferries in the same file) | yes | not stated | direct | daily; 2026-07-10 to 2026-12-01 | live (2026-10-01). Stop ids are hex strings, not UIC, and no Greek OSM station has `uic_ref`, so names (Greek folded to Latin; `NAME_ALIAS` for Ska, Paleopharsalos, Aegion, Σέρραι...). Four ferry agencies (route_type 4) and Hellenic Train replacement buses (3) in the same file; trains are 2, 102, 106, 109. Kept whole, not slimmed: the buses are what tell a closed line. Refetch before December: the window ends 1 December |
| Germany | DELFI (official, all modes, 470 MB): `opendata-oepnv.de/ht/de/datensaetze/sharing?...` (the Transitous sharing link resolves without login); easier: gtfs.de `fv_free` + `rv_free` (12 MB together) | all operators incl. non-DB railways | no | CC BY 4.0 | direct | weekly (DELFI); daily, 30 days (gtfs.de) | works. DELFI uses DHIDs; gtfs.de its own numbers, so names |
| Italy | `raw.githubusercontent.com/deryclem/trenitalia-gtfs/refs/heads/main/gtfs-trenitalia.zip`, converted from Trenitalia's NeTEx on the NAP (`cciss.it`) | Trenitalia: high speed, Intercity, regional | yes (from NeTEx ServiceLinks) | CC BY 4.0 | direct | weekly | works for Trenitalia. Trenord (Transitland f-u0n-trenord), Italo (`github.com/deryclem/italo-gtfs`), FNM, EAV, FSE, FdC and FAL each need their own feed |
| Spain | Renfe, direct: `ssl.renfe.com/gtransit/Fichero_AV_LD/google_transit.zip` (AVE, long and medium distance), `ssl.renfe.com/ftransit/Fichero_CER_FOMENTO/fomento_transit.zip` (Cercanías), FEVE via data.renfe.com | Renfe; FGC, Euskotren, FGV, SFM and OUIGO are separate feeds | Cercanías yes, long distance no | CC BY 4.0 | direct (the NAP itself wants an account) | daily | works. Adif 5-digit station codes in stop_id |
| Sweden | Trafiklab GTFS Sverige (`developer.trafiklab.se/register`, free key); mirror `api.transitous.org/gtfs/se_Trafiklab.gtfs.zip` | all operators | no | Trafiklab terms | key; mirror direct | weekly | works |
| Norway | Entur `storage.googleapis.com/marduk-production/outbound/gtfs/rb_norway-aggregated-gtfs.zip` | all: Vy, SJ Norge, Go-Ahead, Flytoget | yes | NLOD 2.0 | direct, no key | daily | works |
| Denmark | Rejseplanen Labs `rejseplanen.info/labs/GTFS.zip` | DSB, S-tog, Lokaltog, the regional railways, metro, light rail | yes | Rejseplanen terms | direct | weekly | works |
| UK | Aubin's GB GTFS `beta.aubin.app/gtfs/great_britain_gtfs.zip` (1 GB, rail and every bus), built from National Rail's timetable; the original needs a free National Rail Open Data / Rail Data Marketplace account | every train operator, Underground, Elizabeth line, trams | no | National Rail open data terms | direct (Aubin); account (NRE) | daily | works; big, but rail is a small slice of it |
| Ireland | TFI `transportforireland.ie/transitData/Data/GTFS_Irish_Rail.zip` | Iarnród Éireann incl. DART and Enterprise | yes | NTA licence (Transitous lists CC BY-SA 4.0) | direct | daily | works. NI Railways is in Translink's own feed (opendatani.gov.uk) |
| USA | Amtrak `content.amtrak.com/content/gtfs/GTFS.zip`; each commuter railroad separately (Metra, LIRR, Metro-North, NJ Transit, SEPTA, MBTA, MARC, VRE, Caltrain, Metrolink, Coaster, Sounder, Tri-Rail, SunRail, Brightline, NICTD...) via the Mobility Database | Amtrak only in the national feed | yes | operator terms | direct for most | weekly | works, assembled from about 25 feeds |
| Canada | VIA Rail `viarail.ca/sites/all/files/gtfs/viarail.zip`; GO Transit, UP Express (`assets.metrolinx.com/...`), exo trains, West Coast Express | | yes | operator terms | direct | | works, assembled |
| Russia | none | | | | | | nothing bulk. Yandex Rasp API (free key at yandex.ru/dev/rasp/raspapi, per-station queries, restrictive terms) is the only machine source |
| India | `github.com/Neo2308/indianrailways-gtfs` (unofficial, 10,594 trains) | all Indian Railways trains | yes | not stated | direct | monthly | usable. IR station codes (ABR), which Wikidata also carries. datameet/railways (CC0) is older |
| Australia | per state: Victoria (`opendata.transport.vic.gov.au`, open), South East Queensland (`gtfsrt.api.translink.com.au/GTFS/SEQ_GTFS.zip`, open), Transperth and Adelaide Metro (open), NSW (Sydney Trains and NSW TrainLink, free key at `opendata.transport.nsw.gov.au`) | each state's metro and regional trains; interstate trains (Ghan, Indian Pacific) in none | yes | CC BY 4.0 | mostly direct; NSW key | | works, assembled |

## Test: Czechia and Hungary against RINF

Method (scratch `gtfs_test.py`, read-only on the project): RINF sections and points from
`data/raw/rinf/<cc>/`; the feed from the Transitous mirror; rail trips only (route_type 2 and
100-117; buses and rail-replacement buses 3/714 left out); every trip collapsed to its
sequence of stations (parent_station), giving the distinct consecutive call pairs. RINF points
matched to feed stations by code (Czechia), then by normalised name within 2 km, then a
passenger-type point to the nearest feed station within 300 m. Two measures: (a) a
station-to-station section is served if some train calls at both ends consecutively; (b) any
section is covered if it lies on the shortest RINF path between the two stations of some
consecutive call pair, with the path no longer than 1.6 times the straight distance plus 4
km. (b) is what handles junction-ended sections and trains that pass stops without calling.

Czechia (feed: 46,207 rail trips, 2,728 stations, whole 2026 timetable year):

- Matching: 2,664 points by code, 14 by name, 4 by distance; 2,667 of the feed's 2,728
  stations matched. The unmatched are abroad (Bad Schandau, Zittau, Kúty...).
- (a) 2,160 station-to-station sections (6,520 km) have both ends matched; trains call at both
  ends consecutively on 2,142 of them (6,478 km).
- (b) 3,528 of 3,905 sections, 9,085 of 9,685 km, lie on some train's path. Of the 1,592
  sections with a non-stop end (2,747 km), 1,279 (2,325 km) are covered.
- The sections `cz_sources.md` lists as dropped for want of an OSM route: 238's approach
  Havlíčkův Brod - Odb Kubešův Mlýn - Břevnice is covered, so is 292 around Bludov and 201
  Nasavrky - Tábor; the curve Havl.Brod Tunel - Kubešův Mlýn (582-01) is not, correctly. 253
  Vranovice - Pohořelice and 345 Nemotice - Koryčany have no trains in the feed, so dropping
  those was right.
- Against the build: of `dist/data/cz` register sections (9,101 km), 8,764 km lie on covered
  RINF paths and 337 km do not. The largest: 228 and 229 (JHMD, 68 km; JHMD is not among the
  feed's 41 agencies, so either it is not running or its timetable is published outside SŽ's
  data, which needs a look), 317 Opava východ - Hlučín 22, 096 Roudnice - Zlonice 18, 256
  Čejč - Uhřice 14, 046 Hněvčeves - Smiřice 12, 013 Bošice - Bečváry 11, 318 Kravaře -
  Chuchelná 10, 137 Chomutov - Vejprty 10, and a few km each on main lines (250, 220), which
  are more likely matching misses or works than closed track.

Hungary (feed: 25,693 rail trips, 1,149 stations, June - December 2026):

- Matching by name and distance only: 1,120 points; 1,114 of the feed's 1,149 stations. The
  unmatched are mostly Szeged tram-train stops and a few spelling differences.
- (a) 955 station-to-station sections (3,643 km) with both ends matched; 948 (3,625 km)
  served consecutively.
- (b) 1,449 of 1,977 sections, 5,039 of 6,592 km. Junction-ended: 139 of 260 (359 of 614 km).
- Lines 27 (Lepsény - Hajmáskér, 27 km), 37 (Somogyszob - Balatonkeresztúr, 50 km) and 62
  (Villány - Középrigóc, 85 km) have no train anywhere on them: the closed lines the build
  draws. Also wholly unserved: 150 (Budapest - Kelebia, 134 of 135 km), 13, 121/2, 84, 151,
  103, 78/2, 106/2, 152 and others.
- Against the build: of 6,721 km of `dist/data/hu` register sections, 5,794 km lie on covered
  paths and 928 km do not; the largest are 150 (155 km), 121 (66), 84 (54), 103 (44), 42 (40),
  78 (40), 152 (38), 37 (34), 153 (31), 98 (30), 151 (30). A few km on main lines (1, 100)
  are matching misses in Budapest. Some of these will be works closures rather than
  abandonment (150 is the Budapest - Belgrade upgrade route), which a feed cannot tell apart
  from closure; see the design note.

Run time: under a minute per country for the whole test.

## Design note: `gtfs_served.py`

What the data supports, in order:

1. Per country, a list of feeds (`GTFS[cc] = [url, ...]`, national plus private and
   cross-border operators), fetched from the original where open and from
   `api.transitous.org/gtfs/` otherwise. Keep rail route_types (2, 100-117; 0, 1, 400-405 and
   12 where the register has metros and trams), drop buses and rail-replacement buses (3,
   200, 700-799 incl. 714) and placeholder agencies (Czechia's "nabídková trasa"). Collapse
   trips to distinct station patterns at parent_station level, keeping per pattern the number
   of trips and of service days in the window.
2. Match register stations to feed stations: a per-country code rule first (Czechia: RINF
   `CZnnnnn` = stop_id `CZ:nnnnn`; France, Belgium, Slovakia, Greece and Switzerland have UIC
   numbers in the stop_id, which join to the register through OSM's `uic_ref` on the matched
   station; Portugal and Romania look like RINF's own numbers), then normalised name within 2
   km, then nearest within 300 m for a stop-type point. Report the unmatched on both sides;
   Hungary matched 97% on names alone.
3. For every consecutive call pair, take the shortest register path between the two matched
   stations, capped at 1.6 times the crow-fly distance plus 4 km, and credit each section on
   it with the pair's trips. This handles junction-ended sections and non-stopping trains
   (3.5% of Czech and 2.8% of Hungarian pairs found no path, nearly all cross-border).
4. Where the feed has shapes (Netherlands, Norway, Denmark, Ireland, Finland, Croatia,
   Latvia, Estonia, Slovenia, Spain's Cercanías, Italy, Taiwan, PKP IC, the generated
   Transitous feeds), use the shape instead of the shortest path: a section is run over if at
   least half its geometry lies within about 50 m of some trip's shape. This is what settles
   the cases the shortest path cannot (below). Note that several of these shapes are made by
   routing over OSM (pfaedle and the like), so they agree with OSM's track, not with OSM's
   route relations, which is the part that was patchy.
5. Output per register section: trips, service days, and the feed date. In `build_model`, a
   section with no trains is marked rather than deleted, whichever kind of end it has, so that
   Hungary's 27/37/62 stop being drawn as running while a works closure (Hungary's 150) can
   be shown as suspended. Keep the OSM route_share rule as the fallback for countries without
   a feed (China, Russia) and for sections the feed cannot reach (unmatched ends).

Where it will not work, or needs care:

- No shapes and two register paths between the same consecutive calls of similar length:
  a high-speed line beside the classic line with no intermediate stop (the shortest path
  always picks one), triangles and chords at junction stations, and four-track sections
  drawn as two register lines. Germany, Switzerland, Austria, France, Belgium, Czechia and
  Hungary have no shapes. A run-time check (scheduled minutes against path km) or the
  register's `highspeed` flag can break some ties; otherwise mark the section ambiguous and
  fall back to OSM.
- Long non-stop runs over junctions (Paris - Lyon, Wien - Salzburg) are fine on length but
  credit every section of whichever path is shortest, including chords no train uses.
- Short windows: gtfs.de, Polish regional, Bulgaria and Lithuania cover 30-60 days, so
  summer-only and weekend tourist lines can be missing. Use whole-year feeds where they
  exist, or keep a union of past runs ("seen in the last 12 months").
- Works closures look like abandonment for as long as the feed's window, as Hungary's 150
  does; that is why step 5 marks rather than drops.
- Feeds that stop at the border lose the pair across it; the neighbouring country's feed,
  or the cross-border trains in it (the Czech feed has DB, ÖBB and PKP trains), close it.
- Operators missing from the national feed: JHMD in Czechia, private railways in Austria's
  rail feed, non-Trenitalia operators in Italy, non-Renfe in Spain, commuter railroads in the
  USA. Each needs its own feed added to `GTFS[cc]`, or the line keeps the OSM rule.

## Built: `gtfs_served.py` (2026-10-01)

Live since the 2026-10-01 rollout (below) for cz, hu, pt, pl, be, si, sk, ro, bg, fi, lt,
lv, ee, hr, lu and gr. Not for at and nl (switched off, see "Rollout"), nor the non-RINF
countries. `build_model.main` calls `gtfs_served.check` just before
`drop_unridden_sections` and `gtfs_served.mark` just after `not_running.mark`. A region with
no zip in `data/raw/gtfs/<cc>/` gets `None` back and builds exactly as before. The module
docstring has the method; in short:

- Feed stations are matched to the build's register section ends: by RINF code where the
  stop_id has it (`CODE`: Czechia, Portugal), otherwise by name within 2 km, otherwise the
  nearest stop within 300 m.
- Every pair of consecutive calls credits the shortest register path between them. Since the
  rollout: paths prefer sections OSM routes run over when two are about as long (`OSM_TIE`);
  register line ends that meet nothing are linked to a node within 1 km (`dangling_links`);
  a section a train calls at both ends of, one after the other, is served whatever the path
  search found; "abroad" means inside another country's outline.
- Each register section then gets one of these states:
  - served
  - weak: junction-ended, no OSM route, and only on long non-stop runs or winding paths;
    left as it is (that is, dropped as before)
  - ambiguous: on a path nearly as short as the one credited; left as it is
  - border: ends at a border point the feed calls nowhere beyond; left as it is (for hr,
    also the track from there back to the last stop the feed knows)
  - unknown: an operator missing from the feed; left as it is
  - not running: marked `closed`
  - osm: a junction-ended section with no train but with OSM routes; left as it is
  - dropped: a junction-ended section with no train and no OSM route; dropped, as it always was

The build log lists every rescued section and every closed section, by line, with lines
beginning `timetable:`.

The Czechia and Hungary figures below are from the first build; "Rollout" has what the later
fixes changed (Czechia 55 km kept instead of 49, Hungary 726 km closed instead of 724).

**Czechia** (Oběhy CZPTT via Transitous, 7.9 MB, 46,205 rail trips, 14 Dec 2025 to 12 Dec
2026, 40 agencies; "nabídková trasa" left out). 2,633 of the 2,745 feed stations are matched,
nearly all by code. Of 9,253 register km before the drop, 8,876 km are served.

- Rescued: 21 junction-ended sections, 49 km on 15 lines.
  - 238 Havlíčkův Brod - Odb Kubešův Mlýn - Břevnice (5.1 km)
  - 292 around Bludov (3.6 km) and its two border sections to Poland at Mikulovice and
    Jindřichov ve Slezsku (6.6 km)
  - 270 Přerov - Vyh Dluhonice (3.3 km)
  - the Česká Třebová approaches (11 km)
  - Brno dolní nádraží (6 km)
  - Turnov, Chodov, Pila - Havlovice, Bohumín-Vrbice to the border (5.3 km), Děčín
- Register km went from 9,101 to 9,151.
- Not running: 74 km on 7 lines.
  - 317 Opava východ - Hlučín (20.2 km; Hlučín is in the feed's stops, but no train calls there)
  - 256 Čejč - Uhřice u Kyjova (15.1)
  - 013 Bošice - Bečváry (10.9)
  - 318 Kravaře ve Slezsku - Chuchelná (9.9)
  - Heřmanův Městec - Prachovice (7.4; trains run Přelouč - Heřmanův Městec and no further)
  - 245 Hrušovany - Hevlín (6.7)
  - 244 Oslavany - Ivančice (3.5)
- Left as they are:
  - unknown, 79 km: JHMD's 228 and 229 (JHMD is not in the feed; caught by the
    infrastructure-manager test, since OSM has no routes there either), the Zubrnice museum
    line, and 130 Novosedlice - Oldřichov
  - ambiguous, 80 km: station throats and border stubs
  - osm, 75 km: 28 junction-ended pieces on running lines, e.g. Praha Masarykovo nádraží -
    Sluncová, Hluboká - Nemanice I, Suchovršice - Trutnov-Poříčí
- check_model is unchanged apart from the longer register.

**Hungary** (MÁV via Transitous, 1.9 MB, 25,693 rail trips, 1 Jun to 13 Dec 2026, MÁV, GYSEV
and Gyermekvasút). 1,174 of the 1,199 feed stations are matched, all by name. Of 6,932
register km before the drop, 5,852 km are served.

- Rescued: 1 section (8 Győr-GYSEV - nyugati elágazás, 0.9 km).
- Not running: 724 km on 23 lines.
  - 150 Budapest - Kelebia (150.2 km; the Budapest - Belgrade works closure)
  - 121 Szeged - Makó - Mezőhegyes (65.9)
  - 84 Kisterenye - Kál-Kápolna (47.4)
  - 42 Pusztaszabolcs - Paks (40.4)
  - 78 Balassagyarmat - Ipolytarnóc (40.3)
  - 152 Kecskemét - Fülöpszállás (38.3)
  - 103 Karcag - Tiszafüred (38.0)
  - 37 Somogyszob - Balatonkeresztúr (35.0)
  - 130 Hódmezővásárhely - Makó (30.8)
  - 153 Kiskőrös - Kalocsa (30.5)
  - 98 Hidasnémeti - Abaújszántó (30.1)
  - 151 Kunszentmiklós - Dunapataj (29.6)
  - 146 (27.3), 114 (25.0), 62 (21.5), 125 (17.0), 88 (16.4), 372 (12.2), 27 (8.5),
    38 (8.2), 24 (5.9), 13 (3.0), 89 (2.4)
- Every one I spot-checked has no train calling there in the feed: Makó, Paks, and Hlučín
  in Czechia.
- Left as they are:
  - border, 69 km: 21 sections to the border. MÁV's feed stops international trains at the
    last Hungarian station, so Hegyeshalom - Nickelsdorf could not be decided and stays.
  - ambiguous, 103 km: Budapest and Szolnok junctions
  - osm, 30 km: Kispest, Józsefváros, Dunakeszi-Főműhely and the like
- Register km went from 6,721 to 6,736, from junction pieces of closed lines kept greyed.

## Adding a country's feed

1. Put the feed in `FEEDS[cc]` in `gtfs_served.py` as `(file name, url, slim)`. Entries for
   pl, be, nl, at, pt, si, sk, ro, bg, fi, lt, lv, ee, lu, hr and gr are there, with the URLs
   from the tables above. A url starting `udata:` is a udata dataset (data.public.lu), and
   `fetch` takes its newest zip.
   - The Transitous mirror is `https://api.transitous.org/gtfs/<cc>_<name>.gtfs.zip`. The
     names are in the index at `https://api.transitous.org/gtfs/`, or in the Transitous
     repository's `feeds/<cc>.json`.
   - Set `slim` to True for a multimodal feed. The download is then cut down to rail routes
     and rail-replacement buses, and the full file deleted. Measured: OVapi Netherlands 238 MB
     to 5.6 MB; PKP Intercity 135 MB to 7.0 MB.
2. Run `python gtfs_served.py --fetch <cc>`. It writes `data/raw/gtfs/<cc>/*.zip`. Every zip
   in that folder is read, and parsed stop patterns are cached beside them in `patterns.pkl`.
   Delete the folder to switch the country back to the OSM rule alone.
3. If the stop_ids carry RINF's own numbers, add a `CODE[cc]` rule. Compare a few stop_ids
   with `uopid` in `data/raw/rinf/<cc>/points.json`. Portugal's `94_3046` is `PT03046`;
   Romania's `60309` is `RO60309`; both rules are in. Slovakia's UIC `5616116` is RINF's
   `SK161166` with a check digit added, so no rule (names match well there). Add any agency
   whose trips are not trains to `SKIP_AGENCY[cc]`, and a feed with no international trains
   at all to `NO_INTERNATIONAL`.
4. Run `python build_model.py --region <cc> --register rinf:data/raw/rinf/<cc>`, then
   build_tiles and check_model. In the build log, read the `timetable:` lines:
   - how many feed stations matched, and the "called at inside the country but matched to
     nothing" list
   - the "stations abroad called at" list. These should be real foreign stations; see the
     Portugal trap below.
   - the "infrastructure managers with no agency of their name" line. A manager listed there
     has its no-train sections left as unknown.
   - every closed line. A whole line closed that is plainly running means a missing
     operator or a matching miss. Search the feed for one of its stations: if nothing calls
     there, the feed really has no train there.

## Rollout (2026-10-01)

Feeds fetched on 2026-10-01 with `python gtfs_served.py --fetch <cc>`; every country rebuilt
with `tools/rebuild.py` and checked with `check_model.py` (all clean). "Kept" is register km
that OSM routes alone would have dropped; "closed" is km now marked not running. Debug
scripts beyond `tools/gtfs/` were scratch and are not kept.

### Changes to the check, and what they did to Czechia and Hungary

Two fixes the Portugal trial asked for, and four more the rollout showed were needed:

1. **Abroad = inside another country's outline.** Before, a station outside this country's
   outline (+300 m) counted as abroad, so 10 Portuguese estuary and coast stations (Alhandra,
   Alverca, Bobadela, Cascais, Figueira da Foz...) did. Countries religiondots has no outline
   for (Luxembourg, Monaco, San Marino, Vatican) come from Natural Earth 1:10m, as in
   `tools/build_regions.py`. Portugal now has one station abroad, Vilar Formoso (on the
   border). Czechia's and Hungary's lists are unchanged (67 and 2).
2. **Links for line ends that meet nothing.** A register line end at a junction that no other
   section touches is joined to the nearest node of another line within 1 km, unless the
   register already joins them within 5 km + 3x the gap. The link is no section, credits
   nothing, and weighs 2x its length + 0.5 km so it never beats real track. Portugal's
   Variante de Alcácer is now the Lisboa - Algarve path, and the old Linha do Sul, Pinheiro -
   Grândola Norte, is no longer credited (it stays dropped).
3. **Prefer what OSM's routes take** when two paths are about as long (sections with no OSM
   route weigh 5% more, and are not credited as a parallel beside one with a route).
   Portugal's freight loops beside the Linha do Norte (Plataforma de Cacia, Bobadela, Ramal
   TER-TIR) were being "rescued" as register lines of their own.
4. **Weak evidence is not a rescue.** A junction-ended section with no OSM route is not kept
   when only long non-stop runs (over 40 km) or winding paths (over 1.3x crow-fly + 1 km)
   cross it, unless a run starts or ends at it. Gliwice - Chałupki non-stop ran over Zabrze
   Makoszowy colliery track; Amsterdam Bijlmer - Centraal went through the Watergraafsmeer
   depot because the Dutch register has no direct Centraal - Muiderpoort section.
5. **Direct evidence wins.** A section whose two ends a train calls at one after the other is
   served, whatever the path search did (Austria: St. Michael ob Bleiburg - Bleiburg came out
   closed after a wrong match one call earlier).
6. **Feeds with no international trains** (`NO_INTERNATIONAL`, Croatia only): the track from
   an unseen border point back to the last stop the feed knows is "border", left as it is.
7. **A closed line closes whole** (added with Greece): an "unknown" stop-to-stop section on a
   line with closed sections, reached from them through stops no train calls at, is closed
   too. Replacement buses do not call at every halt, so Velestino - Volos closed while
   Velestino - Stefanovikeio stayed unknown. A line whose operator is missing from the feed
   has no closed section to grow from. Elsewhere it changed only Romania's 806 (Nazarcea -
   Dorobanțu, 9.2 km, freight).
8. **Name variants** (`NAME_ALIAS`, Greece): another name tried for a feed station.
9. **Seasonal lines** (`PAST`, Poland; pending Anita's approval of the download): see
   "Seasonal lines" below.

Measured on cz and hu (`tools/compare_lines.py` save, rebuild, diff, plus a section-by-section
comparison of old and new states):

- **Czechia**: closed unchanged (74 km, the same 7 lines). Now 26 junction-ended sections
  (55 km) kept that OSM alone drops, against 21 (49 km). Register lines that differ:
  - new lines: Kamenický Šenov - hi Česká Kamenice (4.3 km; regular trains to Benešov nad
    Ploučnicí), Děčín východ - Děčín východ St. 1 (0.8 km, 48 trips), Chomutov prům. kolej -
    Chomutov St. 2 (0.7 km: the register has no other way from Chomutov to the marshalling
    yard on line 140, so trains to Málkov run over this industrial piece and a link; this one
    is wrong but small)
  - longer: 063 +2.8 km (Buda - Odb Zaluci, 4 trips), 120 +1.0 km (Praha-Bubny
    - Masarykovo viadukt, 701 trips), 314 +2.0 km (Otice - Odb Moravice, 18 trips)
  - gone: the unnumbered Třebovice v Čechách - Česká Třebová odjezdová skupina (5.8 km), a
    yard approach no OSM route takes; trains are credited to 270's approach instead
- **Hungary**: register lines identical; closed 724 -> 726 km (103 Karcag-Ipartelep -
  Karcag-Vásártér, 2.4 km, joins the closed 103).

### Per country

| cc | feed (window) | stations matched | kept (OSM alone drops) | closed |
|---|---|---|---|---|
| cz | Oběhy CZPTT (14 Dec 2025 - 12 Dec 2026) | 2,633 of 2,745 | 55 km, 26 sections | 74 km, 7 lines |
| hu | MÁV (1 Jun - 13 Dec 2026) | 1,174 of 1,199 | 1 km | 726 km, 23 lines |
| pt | CP + Fertagus (whole year) | 429 of 468 | 3.5 km | 8.5 km, 2 lines |
| pl | Polish-Trains (19 Sep - 24 Oct 2026), PKP IC (22 Sep - 31 Dec), narrow gauge; past: Polish Trains 13 Jul - 14 Aug 2026 | 3,272 of 3,765 | 110 km | 910 km, 28 lines (96 and 131 seasonal) |
| be | SNCB (whole year) | 560 of 679 | 2.5 km | 9.2 km, 1 line |
| si | nap.si via Transitous (whole year) | 265 of 267 | 0 | 0 |
| sk | ŽSR (whole year) | 688 of 887 | 12.8 km | 398 km, 20 lines (66 km beyond sk.py's SUSPENDED) |
| ro | data.gov.ro via jbb (whole year) | 1,542 of 1,643 | 47 km | 793 km, 20 lines |
| bg | livetransport BDZ (30 Sep - 30 Oct 2026) | 625 of 661 | 19 km | 83 km, 3 lines |
| fi | Fintraffic (24 Sep 2026 - end 2027) | 196 of 198 | 3.5 km | 34 km, 1 line (Porvoo) |
| lt | LTG Link via jbb (1 Oct - 30 Nov 2026) | 115 of 122 | 0 | 0 |
| lv | pieturas.lv via Transitous (whole year) | 103 of 147 | 0 | 0 |
| ee | Elron (31 Aug - 31 Dec 2026) | 238 of 245 | 0 | 0 |
| hr | HŽPP (8 Dec 2025 - 13 Dec 2026) | 458 of 460 | 22.6 km | 139 km, 3 lines (15.5 beyond hr.py's SUSPENDED) |
| at | switched off | | | |
| nl | switched off | | | |
| lu | national feed (30 Sep - 12 Dec 2026) | 69 of 91 | 0 | 0 |
| gr | Hellenic Train via jbb (10 Jul - 1 Dec 2026) | 280 of 395 | 0 | 509 km, 5 lines |

Unmatched feed stations are nearly all abroad. Where many inside the country are unmatched
(Latvia 43: the Skulte, Gulbene and Saldus lines' halts; Slovakia 26: Trenčianske Teplice,
Bardejov line) the register has no node there; those calls are stepped over.

**Portugal.** Kept: Linha do Oeste, Agulha 1 da Amieira - Bifurcação de Lares (2.7 km; 2
trips Louriçal - Bifurcação de Lares), Linha do Sul, Praias do Sado-A - Ramal Sado-Sapec
(0.8 km). Closed: Linha de Leixões, Leça do Balio - Guifões (3.5 km; the Urbanos turn at Leça
do Balio) and Guifões - Leixões (4.4 km, the freight end into the port, kept now as greyed
because it adjoins the closed section, where before it was dropped); Linha de Cintura,
Alcântara-Mar - Alcântara-Terra (0.6 km). The Valença border stub is "border": CP's feed has
no Celta beyond Valença.

**Poland.** Kept: 39 Suwałki - Olecko (42.8 km; PKP IC Szczecin - Suwałki), 506 and 503
(Warszawa Wschodnia Towarowa curves, 6.8 km), 957 Rybnik - Rybnik Towarowy (4 km), 764
Siechnice - Wrocław Brochów (5.9 km), 200 Gliwice - Sośnica (5.1 km), 159, 165, and closed
lines' junction pieces kept greyed (108 to Krościenko 11.4 km, 171 16.9 km, 406 Police -
Trzebież 13.3 km). Closed, by kind:
- No passenger service for years (checked): 12 Pilawa - Łuków (60.7 km; none for 20 years,
  modernisation tendered), 14 Głogów - Żagań (59.8; none since 2000), 34 Małkinia - Ostrołęka
  (53.7; none since 1993, freight diversions in 2026), 144 Tarnowskie Góry - Zawadzkie (34.5),
  204 Braniewo - Bogaczewo (42.0; Braniewo is reached only from Olsztyn), 218 Kwidzyn - Prabuty
  (21.2), 410 Złocieniec - Kalisz Pomorski (40.2; nostalgia specials only).
- Freight lines and bypasses (no train between their ends; trains between the same places go
  another way): 13 Krusze - Pilawa (58.8), 131 Kraski - Karsznice/Babiak (91.3), 171 (29.3),
  179 (21.9), 349 (19.2), 394 (16.0), 273 Drzeńsko - Jerzmanice (10.6, a chord; trains go via
  Rzepin), 275 Żagań - Bieniów (16.2), 57 Sokółka - Kuźnica (16.2, broad gauge; trains use
  line 6 beside it), 208 Lniano - Leosia (20.1).
- Works closures, buses in the feed or in the news: 211 Chojnice - Kościerzyna (62.2;
  replacement buses since 30 June), 287 Opole - Nysa (45.6; buses until 24 October 2026),
  274 Rybnica - Jelenia Góra Zabobrze (9.2; buses 31 August - 30 October 2026), 90 Zebrzydowice
  - Cieszyn (13.1; buses in the feed, trains in summer and hourly from 13 December 2026),
  108 Jedlicze - Tarnowiec (9.2, buses), 107 and 108 in the Bieszczady (Zagórz - Łupków 48.7,
  Zagórz - Ustrzyki Dolne - Krościenko 47.6; line 108 modernisation, no train at Zagórz at all
  in the window), 201 Łąg - Lipowa Tucholska (16.7), 406 Szczecin (34.3; the Szczecin
  Metropolitan Railway, not open yet), 103 Alwernia - Regulice (2.2), 249 Gdańsk Stocznia -
  Stadion Expo (3.3).
- Seasonal: 96 Muszyna - Leluchów (7.0; Koleje Małopolskie's summer weekend trains to Poprad,
  14 June - 29 August 2026); 363 Skwierzyna - Wierzbno (21.0; seasonal trains since 2023, not
  found for 2026). Both read as closed because the regional window is September - October.
- Unsure: 281 Nakło - Chojnice (74.9): local press says a "Krajna" train ran in April 2026 and
  the line is in PLK's planning timetable from 29 June, but nothing runs in the feed.
- Unknown (left drawn): 31 Hajnówka - Siemianówka, 291 Mieroszów, 290 to the border, 364
  Międzyrzecz - Wierzbno, Zagórz - Nowy Zagórz.
- Sources: rynek-kolejowy.pl (Łupków 2026, 108), nakolei.pl, halolukow.pl (line 12),
  eostroleka.pl (34), zawszepomorze.pl and pl.wikipedia (218), polregio.pl (287 ZKA),
  kolejedolnoslaskie.pl (274), kolejemalopolskie.com.pl (Muszyna - Poprad), rynek-kolejowy.pl
  (144, 364), gazetawiecborska.eu (281).

**Belgium.** Kept 2.5 km of junction pieces (L.130/1 at Moustier, L.37/1 Kinkempois, L.66/1
Kortrijk, and L.19 Hamont - border 0.9 km from 2 diverted trains). Closed: L.147 Auvelais -
Fleurus (9.2 km, freight). The Belgian register is in pieces (Lier - Berlaar, Genk - Bokrijk,
Brugge - Heist have no register path), so 731 calls find no path; those sections stay with OSM.

**Slovakia.** Closed agrees with sk.py's SUSPENDED on 17 lines and adds 141 Leopoldov -
Kozárovce (48.8 km; Zlaté Moravce is reached only from Šurany), 133 Sereď - Leopoldov (17.1;
Sereď's trains run Galanta - Trnava) and Hronec - Chvatimech (1.4). Kept: border track at
Kúty, Horné Srnie (Vlárský průsmyk) and Plaveč (12.8 km). Unknown, left drawn: 153 Zvolen -
Šahy (nothing in the feed calls at its stops), a piece of 120 at Žilina, and TEŽ's Tatra lines.

**Romania.** Kept: Giurgiu line pieces at Jilava (10.2 km), Curtici - border (8.4), Săcuieni
(7.4), Golenți (3.3). Closed 793 km on 20 lines (806 includes Nazarcea - Dorobanțu since the
closed-line growth rule): 300 Cluj-Napoca - Oradea (139.6 km; closed
for modernisation since January 2024, trains back in stages from 2027), 600 Făurei - Tecuci
(89.6; Bucharest - Iași trains run via Focșani), 700 Urziceni - Făurei (67.4; Urziceni is a
terminus in the feed), 603, 314, 703, 316, 806 (Constanța - Năvodari - Capul Midia, freight),
205, 107, 119, 701, 510, 214, 218, 105, 207, 500 Dornești - Vicșani (to Ukraine), 508, 104.

**Bulgaria.** Kept: line 8 Trakia - Plovdiv Razpredelitelna (5.5 km, every Plovdiv - Burgas
train: the piece HANDOFF.md hoped the check would rescue), Vidin - Kapitanovtsi - border
(13 km) and Kalotina border (0.8). Closed: 83 Nova Zagora - Radnevo - Galabovo - Simeonovgrad
(61.4), 61 Razmenna - Batanovtsi (17.6), line 1 Sofia - Poduyane marshalling yard (4.4). A
30-day window: nothing seasonal stood out.

**Finland.** The Porvoon rata, kept by Anita's decision (removed from `FREIGHT`), comes back
whole, Kytömaa - Nikkilä - Porvoo (34.3 km), drawn as not running: Fintraffic's feed carries VR
and HSL only, not the museum trains. Olli - Sköldvik (the refinery branch) is cut off track
number 131 and stays out. Kept: Tornio - Haparanda border (3.5 km).

**Croatia.** Kept: M304 Ploče - Metković (22.6 km, 178 trips, the Sarajevo train; OSM has no
stations on it, so it was gone). Closed: M606 Knin - Zadar and L103 Karlovac - Kamanje (both
already greyed by hr.py's SUSPENDED), R104 Dalj - Vukovar-Borovo Naselje (15.5 km; one train
in HŽ Infrastruktura's 2025 statistics). M201 Koprivnica - Novo Drnje - Botovo carries only
international trains, which HŽPP's feed lacks; `NO_INTERNATIONAL` leaves it as "border".
M303 to Slavonski Šamac and L204 Daruvar - Pčelić have trains in the feed and stay running.

**Slovenia, Lithuania, Latvia, Estonia.** No change: nothing kept, nothing closed. Slovenia's
SŽ trains are all route_type 2 (its 56 route_type 3 routes are numbered replacement buses,
"BUS 14006"). Koper's own section is missing from the register, so line 62 is "unknown" and
stays drawn.

**Austria: switched off** (data/raw/gtfs/at/ deleted; the build never used it). The register
graph is in 93 unconnected pieces before the drop (the Unterinntalbahn split at Wörgl, no path
from Wien Meidling to St. Pölten), 5,080 calls found no path, Salzburg Hauptbahnhof did not
match ("Hbf"), and the result closed running track: the Westbahn at Straßwalchen - Steindorf,
the Raaberbahn between Wiener Neustadt and Mattersburg, and the Koralmbahn came out unknown.
Worth another try once Austria's register pieces join up.

**Netherlands: switched off** (data/raw/gtfs/nl/ deleted; the build never used it). The feed
closed nothing and its only rescue was a depot track (Watergraafsmeer HSA), reached because the
register has no direct Amsterdam Centraal - Muiderpoort section. Every Dutch passenger section
already has OSM routes, so the feed adds nothing.

**Luxembourg** (rolled out later on 2026-10-01, after Anita approved the downloads). CFL's
trains are route_type 2 in the national feed (slimmed 16.8 MB to 0.2 MB). No change: nothing
kept, nothing closed; the border stubs (Troisvierges, Kleinbettingen, Bettembourg, Rodange,
Esch) stay as OSM has them.

**Greece** (rolled out later on 2026-10-01). The feed is kept whole (5.8 MB): slimmed, it lost
Hellenic Train's 81 bus routes, and without them the closed Thessaly lines read as "unknown"
(an operator missing from the feed) and stayed drawn. Closed 509 km on 5 lines:
- 25 Thessaloniki - Alexandroupoli from Serres east (278.6 km: Serres - Drama - Xanthi -
  Komotini - Alexandroupoli); Thessaloniki - Serres keeps its trains (route 1634 and others in
  the feed and in the 2026 news). So partial greying, as expected.
- 12 Palaiofarsalos - Kalambaka (79.9 km) and 13 Larissa - Volos (61.0 km): no train in the
  10 July - 1 December timetable, buses in the feed and in the news (Storm Daniel; the
  expected summer 2026 reopening has not happened in the feed).
- 10 Lianokladi - Stylida (19.7 km).
- Thessaloniki - Idomeni (69.8 km; gr.py already greyed 62.4 of it).
Left drawn: Strymonas - Promachonas (13.4 km, "unknown": only the Sofia train, not in the
feed) and two Thessaloniki freight-area pieces. Not matched: the Kiato - Patras line's stations
(the register has no such line) and a few on the Athens lines. The window ends 1 December 2026;
the check ignores today's date, so the snapshot keeps working, but refetch for the next
timetable.

### Seasonal lines (decided and built 2026-10-01)

Anita: seasonal lines are drawn as running. Poland's regional feed covers about five autumn
weeks, so its summer-only lines read as closed. Options measured on the Polish closed sections
(scratch scripts, read-only):

- **A feed covering under N months cannot close a section OSM routes run over:** reopens
  nothing seasonal. 96 Muszyna - Leluchów and 363 Skwierzyna - Wierzbno have no OSM passenger
  route (share 0.00 and 0.02). It would reopen 287 Opole - Nysa (works) and freight lines
  (131, 171, 13, 57, 273) instead: 207 km.
- **...unless corroborated (trains at both ends, or buses):** reopens 96 and 363 but also 13,
  131, 208, 406, 201, 249, 103 and the Bieszczady lines (327 km). In one autumn feed a seasonal
  line and a closed one look the same.
- **A whole-year source:** none found for Polish regional trains. Koleje Małopolskie's own
  GTFS covers 9 September - 24 October. Polregio's would not download (TLS error; not
  retried).
- **A past snapshot (chosen):** Mobility Database keeps every daily download of a feed, and
  the files are public at `files.mobilitydatabase.org/<id>/<id>-YYYYMMDDHHMM/<same>.zip`. The
  listing needs an account, but a date's file can be found by trying the minutes after
  midnight (scratch `mdb_find.py`). The 15 July 2026 Polish Trains snapshot (mdb-3191) has
  regional and PKP IC trains from 13 July to 14 August. It has Koleje Małopolskie's Kraków -
  Muszyna - Plaveč - Poprad weekend trains, and nothing at Leluchów's neighbours Wierzbno,
  Ustrzyki Dolne or Łupków.

Rule (`PAST`, `SEASONAL_MIN_DAYS`): the past snapshot's trains are put through the same paths.
A section the current feed would close (or leave unknown) is "seasonal", drawn as running,
when it has at least 16 trip-days in the snapshot. That is one train each way weekly over an
8-week summer, Anita's "more often than about once a week". Two exceptions: a section with
junctions at both ends, and a section where replacement buses now call at a stop no train
calls at. The latter is a works closure that began after the snapshot (Opole - Nysa, closed 3
August), which stays greyed.

Built on Poland (`--fetch-past pl`, approved by Anita; rebuild diffed against the build
before it: only 96 and 131 changed, from closed to running):
- Reopens 96 Muszyna - Leluchów (7.0 km; 32 trip-days).
- Reopens 131 Kraski - Zduńska Wola Karsznice and Kraski - Babiak (91.3 km; 64 trip-days of
  the summer InterCity coast trains Gdynia - Inowrocław - Zduńska Wola Karsznice - Katowice,
  so 131 is not freight-only after all).
- Keeps 295 Węgliniec - Bielawa Dolna to the German border (13.3 km; 128 trip-days).
- Stays closed: line 12 (one day, under the threshold) and 273 (8 days). Also 363 Wierzbno
  and the Bieszczady lines (no trains in summer 2026 either), and every works closure.
- Closed in Poland: 1,007 -> 910 km.

No other country has a past snapshot: Bulgaria's and Greece's feeds are not in Mobility
Database (Greece's covers July - December anyway), and Lithuania's has no closures.

The snapshot is in `data/raw/gtfs/pl/past/` (7.2 MB slimmed from 28.3). Its own
patterns.pkl cache sits beside it. Poland's row in the table above is now 910 km closed on
28 lines.

### Things to know

- **Inspecting data only reads it** (Anita, 2026-10-01). Inspection scripts set
  `gtfs_served.WRITE_CACHE = False` so no patterns.pkl is written beside a feed, and keep
  their own state in the scratch folder. Only `--fetch` / `--fetch-past` and an agreed build
  write to `data/`.
- A feed is a snapshot. Works closures that end (Opole - Nysa 24 October, Jelenia Góra 30
  October, Cieszyn 13 December) stay greyed until the feed is fetched again and the country
  rebuilt.
- Seasonal trains are missed wherever the window is short (Poland's regional feed, Bulgaria,
  Lithuania). The fix is a past snapshot (`PAST`, "Seasonal lines"), for Poland only so far.
- `fetch` falls back to curl when Python's certificate check fails (data.public.lu on this
  machine).
