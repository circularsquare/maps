# Indonesia register sources (built 2026-10-03)

How Indonesia's register is put together, from what, with which numbers, and what is still off.
`id_register.py`'s docstring says how the reader works; `rinf_countries/id.py` holds the settings
rinf.py reads; `rules/id.py` says which OSM train relations are named trains.

## Commands

    python id_register.py --fetch        # id.wikipedia: 38 line articles + 1,097 station points (~2 min)
    python extract.py --region id --pbf data/raw/indonesia-latest.osm.pbf     (1.74 GB, ~6 min)
    python id_register.py --report       # the register without tracing: km per line against the infobox
    python id_register.py --show "Cirebon–Semarang"   # one article's runs, section by section
    python build_model.py --region id --register id_register:data/raw/id      (~45 s)
    python build_tiles.py --region id                                         (~15 s)
    python check_model.py --region id

`--register id_register:data/raw/id` converts and then runs rinf.build on its output.
`tools/rebuild.py` needs `"id": "id_register:data/raw/id"` in its REGISTER (managing session).

## Sources

| source | what it gives | licence | file |
|---|---|---|---|
| id.wikipedia, "Daftar jalur kereta api aktif di Indonesia" and the 38 line articles it links (+ Makassar–Parepare) | each line's station table ({{DaftarStasiun}} rows in line order): name, KAI code (`singkatan`), status (Beroperasi / Tidak beroperasi / Konstruksi / Reaktivasi), KAI's km posts (`letak=km 219+168`, one per chainage at junctions); the infobox length; branches under their own headings | CC BY-SA 4.0 | `data/raw/id/idwiki_articles.json` |
| id.wikipedia station articles ("Stasiun <nama>", MediaWiki API `prop=coordinates`) | a point for 742 of 1,097 listed stations (688 of the 703 open ones) | CC BY-SA 4.0 | `data/raw/id/idwiki_stations.json` |
| Geofabrik `asia/indonesia-latest.osm.pbf` (managing session) | track (14,520 ways), 8,284 stops, 283 route relations (268 train: 169 KAI, 70 KAI Commuter), 74 route=railway relations | ODbL | data/proc/id is all a rebuild needs |
| en.wikipedia "Jakarta–Bandung high-speed railway" | Whoosh's chainage: Halim 0, Karawang 41.17, Tegalluar 142.80 (and Padalarang 97.22, which is wrong: see below) | CC BY-SA | in `id_register.HAND` |
| DJKA (Ministry of Transport), via detik.com 2025-09-26 and 2024-08-07 | active track: Java 4,921 km (473 stations), Sumatra 1,871 (146), Sulawesi 109 (10); total 6,945 (2024: 6,880) | | numbers here |
| open-krl/pipeline v0.1.0 (github.com/open-krl/pipeline) | a GTFS of KRL Jabodetabek and Merak: 6 routes, 94 stops, 1,139 weekday trips, 2026-09-17 to 12-31 | CC BY 4.0 (data) | `data/raw/id/krl_gtfs_v1.zip` (kept out of data/raw/gtfs/id on purpose, see Timetables) |

Looked at and not used:
- **Wikidata**: 291 railway-line items with P17 Indonesia, 83 with station adjacency, but the
  chains are partial (the Jakarta–Cikampek–Padalarang item has 41 pairs for ~70 stations), no
  P2043 on most, and many items are named trains, not lines. The id.wikipedia tables are the
  same lines with every station and KAI's own km.
- **OSM named track** (`probe_kr_ways.py`): 96.5% of main and branch rail km carries a name, so
  Korea's recipe would work, but the names are KAI's operating segments ("Jalur Kereta Api
  Solo Balapan–Kertosono", "Kutoarjo–Solo Balapan", "Tegal–Semarang–Brumbung"), cut elsewhere
  than any published list and with no km to check against. The wiki tables carry km posts.
- **DJKA's / KAI's own line lists**: no open per-line file found (KAI's GAPEKA is not
  published as data; DJKA publishes island totals only).
- **Transjakarta GTFS** (Mobility Database mdb-1909): bus only (240 routes, all route_type 3).
  No Indonesian feed on Transitous.

## The line unit

A register line is one id.wikipedia line article: the lines the Dutch-era companies built
(Staatsspoorwegen, NIS, SCS, DSM, ZSS), which KAI still measures as one chainage each (its km
posts run Jakarta - Cikampek - Cirebon - Kroya without a break). "Daftar jalur kereta api aktif
di Indonesia" lists them as the active network, and together they partition it. 36 lines are
built (38 articles; Bukit Putus–Indarung is freight-only and Padang Panjang–Sawahlunto has no
timetabled train): 39 line pieces, since Bogor–Padalarang–Kasugihan is two (Cipatat -
Padalarang closed) and lintas Jakarta two. Names are the article's less "Jalur kereta api"
("Cikampek–Cirebon–Kroya", "Lintas Jakarta"). Whoosh is written in by hand as "Kereta Cepat
Jakarta–Bandung" (no wiki station table). The operator is KAI, Whoosh's KCIC.

The KRL and other commuter lines are not register lines: they are OSM route relations
("Lin Bogor", "Commuter Line Prameks", "Lin Srilelawangsa") running over the register lines,
as Seoul's 1호선 runs over 경부선. MRT Jakarta, LRT Jakarta, LRT Jabodebek, LRT Palembang and the
Soekarno-Hatta Kalayang are OSM lines, as metros are everywhere.

## How the km are measured

- A section is two consecutive rows of a table. Its km is the difference of the two stations'
  km posts on the chainage they share: of every pair of posts, the one whose difference best
  fits the crow-fly distance (0.92 x crow - 0.4 to 3 x crow + 1.5; track winds, Batu Ceper -
  Bandara Soekarno-Hatta is 12.3 km for 5.2 crow-fly). 721 sections by posts, 13 crow-fly x 1.15
  (posts missing).
- Rows across a heading or a new table join only if the posts fit and carry on the chainage the
  run came in on, in the same direction (Jakarta's tables end and start at stations that fit by
  chance: Kemayoran, km 4.7 from Ancol, and Tanah Abang, km 0.0).
- A run is trimmed to its first and last Beroperasi station, so a line runs only as far as
  trains call (Cilacap Pelabuhan, Garut - Cikajang, Padang Panjang drop off). Closed halts
  inside stay as junction points or are merged away.
- **Points**: the OSM station or train stop carrying the station's KAI code in `ref` (385
  codes), else OSM's station of exactly that name if within 15 km of the wiki's point, else
  id.wikipedia's point. The wiki's are off by kilometres in places (Kedungbanteng 5.6 km, Sragi
  1.9; Larangan and Suradadi carry one shared wrong point): a point whose posts put it closer
  to its neighbours than the crow flies is dropped (worst first), and the station is found by
  name instead.
- **KAI codes are not unique** across islands and old lists (Bantarkadu and Barru are both BAR,
  Tigaraksa and Tegalluar TGS): a second station under a shared code, by another name and over
  5 km away, gets a point of its own.
- Hand lists (`id_register.py`): `INACTIVE` (Cipatat - Padalarang, closed since the Siliwangi
  turns at Cipatat; Padang Panjang–Sawahlunto, the Mak Itam steam train's occasional runs),
  `FREIGHT` (Bukit Putus–Indarung, Labakkang - Mangilu, Muara Enim - Tanjung Enim Baru,
  Perlanaan - Sei Mangkei - Bandar Tinggi - Kuala Tanjung), `FREIGHT_STATIONS` (Jakarta Gudang,
  Pasoso, Sungai Lagoa, Kalimas, Benteng, Sidotopo, Mesigit, Cigading, Tanjung Enim Baru,
  Indarung, Blokpos Garuntang: not stops, so sections to them are junction-ended and kept only
  where an OSM passenger route runs), `HAND` (Whoosh; the Araskabu - Kualanamu airport branch,
  which no table lists), `NAME_ALIAS` (Tanjung Priok = OSM's Tanjung Priuk).
- Whoosh: en.wikipedia puts Padalarang at km 97.22, but OSM's track gives Karawang - Padalarang
  68.2 km and Padalarang - Tegalluar 31.8 (Padalarang is west of Bandung, Tegalluar east), which
  sum to the published 142.8. Padalarang takes no post and its two sections are crow-fly x 1.15.

## First build (2026-10-03, indonesia-latest extract)

- 691 points, 589 stops; rinf.py found an OSM station for 585 (3 by distance alone: Blimbing
  Pendopo = Belimbing Pendopo, Bandara Adi Soemarmo = Adi Soemarmo, Sumberejo = Sumberrejo);
  Pagar Gunung has none and is a junction point.
- Traced: 39 rinf lines, 4,279 km against 4,292 km of KAI km posts; 1 section rejected
  (Tanjungkarang - Blokpos Garuntang, goods), 2 kept as "length off" (Angke - Kampung Bandan on
  the Anyer line: posts 0.70, track 4.44; Kutablang - Geurugok, Aceh: posts 6.85, track 4.45).
- Built: 158 lines: 39 register pieces (4,274 km), 81 named trains, 38 OSM lines (KRL and the
  other commuter lines, the airport trains, Pangrango, Siliwangi, Batara Kresna, Kedungsepur,
  Whoosh's own relation, MRT, LRTs, Kalayang); 745 stations. Junction-ended sections: kept 4 (15
  km) that passenger routes run over, dropped 3 (6 km). Not running: none (no timetable check).
- By island (register km): Java 3,019.5 (with Whoosh's 141), Sumatra 1,152.6, Sulawesi 101.4.
- Ownership: 12 ways (2.0 km) run over only by named trains; likely register gaps 1.4 km in 2
  places. Tiles: 1.2 MB; 80 ways on track no line touches (47 narrow gauge: sugar mills).

## Checks (check_model.py --region id)

- Against KAI's own km posts (km_official, every line): median 1.000; 2 lines over 5% off:
  Kutablang–Muara Satu 0.86 (29.1 of 33.9: Aceh's new standard-gauge line, the posts there are
  not where OSM's track is), Lintas Surabaya 1.12 (12.8 of 11.4: short pieces).
- `REGISTER["id"]`, 22 lines against the id.wikipedia infobox (or KAI's posts, or DJKA): worst
  0.07. Exact or close: Cikampek–Cirebon–Kroya 292.7 / 293.0, Cirebon–Semarang 223.6 / 225.6,
  Gundih–Surabaya Pasarturi 228.4 / 230, Kertosono–Bangil 217.1 / 215.5, Surabaya–Bangil–Kalisat
  214.4 / 214.4, Kisaran–Rantau Prapat 113.9 / 114, Tegal–Prupuk 38.5 / 38.5, Whoosh 141.1 /
  142.3. Off by extent, as noted: Bogor–Padalarang–Kasugihan 0.96 (Cipatat - Padalarang closed),
  Prabumulih–Panjang 0.96 (passengers end at Tanjungkarang), Cilacap–Yogyakarta 1.04 (YIA and
  Karangtalun branches), Medan–Tebing Tinggi 1.06 (Kualanamu branch), Anyer Kidul–Kampung
  Bandan 1.07 (the infobox's 147 against KAI's posts' 154.5), Makassar–Parepare 0.93 (DJKA's
  109 km for Sulawesi includes the Tonasa goods branch).
- `KNOWN["id"]`, 7 OSM lines: MRT North-South 0.93, LRT Jakarta 0.95, Jabodebek Bekasi 0.92 and
  Cibubur 0.96, LRT Palembang 0.95 (published lengths take in depot tails), Lin Tangerang 1.01,
  Lin Rangkasbitung 1.00.
- **Island totals do not compare with DJKA's figures**: DJKA's "panjang jalur aktif" (Java
  4,921, Sumatra 1,871, Sulawesi 109) looks like track length, double track counted, plus
  freight-only lines: Java's double-tracked Jakarta - Surabaya north and south lines (~1,500 km)
  account for Java's 3,019 against 4,921. Sumatra had 1,348 km operational in 2013 (en.WP "Rail
  transport in Indonesia"), freight lines included; built 1,153 without them. Sulawesi's 109 is
  route km, and 101.4 is built (no Tonasa branch).

## Lines vs named trains (rules/id.py)

OSM maps KAI's trains one relation per train and direction ("Argo Bromo Anggrek: Gambir →
Surabaya Pasarturi"), in route masters, network "KAI". The rule, read through KAI's own two
classes: **KA antarkota** (intercity, long and medium distance: Argo Bromo Anggrek twice a
day each way, Taksaka twice, Gajayana once; Kaligung, Kamandaka, Joglosemarkerto, Sribilah
Utama, Putri Deli, Siantar Ekspres a few times) are named trains: each is one fixed train,
and its track counts through the register lines, which cover every km it runs. **KA lokal,
komuter and bandara** are lines: network KAI Commuter, KAI Bandara, Whoosh, and KAI's local
trains by name (Pangrango, Siliwangi, Batara Kresna, Kedungsepur, Feeder KCJB, BIAS, Sri
Lelawangsa, the West Sumatra and Aceh locals if mapped): several a day, the way a rider
travels those lines. 81 named trains, 38 lines. Relations mistagged route=train for a railway
itself ("Kertosono–Bangil railway") are flagged named trains so they never stand as lines.

## Timetables

No open national KAI feed exists. The open-krl GTFS (KRL Jabodetabek + Merak, weekday only,
CC BY 4.0) was tried through gtfs_served by accident (any zip in data/raw/gtfs/id is read):
it marked 1,372 km of KAI's intercity network "not running", because gtfs_served treats every
register section of the country's main manager with no train as closed. So no
`FEEDS["id"]` is proposed. It would be usable only with a scope hook in gtfs_served (judge only
sections inside the feed's own area, or only lines its stations lie on); the zip is kept in
data/raw/id for that.

## Open

- Whoosh's own OSM relation ("Jakarta–Bandung high speed rail", m16115037) stays an OSM line
  beside the register line though both have the same four stations and 141.1 km; build_model's
  twin merge did not take it ("0 OSM lines matched a register line"), worth a look there.
- No `colours/id.csv`: KAI publishes no line colours; the commuter lines carry OSM's (KAI
  Commuter's official colours).
- Lines with no OSM route relation at all (West Sumatra, Aceh, Lampung's Kuala Stabas) are
  register lines only, so their stops come from the wiki tables alone.
- Kutablang–Muara Satu (Aceh) 0.86 of its posts; Angke - Kampung Bandan on the Anyer line
  traced 4.44 km for posts' 0.70.
- Not built: Garut - Cikajang (reactivation stopped at Garut), Kutoarjo - Purworejo
  ("Reaktivasi", not running), Rangkasbitung - Labuan, the Trans-Sulawesi north of Garongkong
  (construction).
- No land border with rail.
