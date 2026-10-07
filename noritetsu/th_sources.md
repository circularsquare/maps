# Thailand (th): sources, decisions, what is off

Built 2026-10-03 by the th agent. Register reader `th_register.py`, rules `rules/th.py`.

    python extract.py --region th --pbf data/raw/thailand-latest.osm.pbf
    python th_register.py --report          # way-to-line assignment, km per line
    python build_model.py --region th --register th_register:data/raw/th
    python build_tiles.py --region th
    python check_model.py --region th

Build time: about 35 s for the model, 5 s for the tiles.

## Result (2026-10-03)

16 register lines, 3,919 km, 660 register stations; 33 lines in all (17 OSM lines: the BTS,
MRT, monorails, Airport Rail Link, SRT Red Lines, Gold Line, Suvarnabhumi APM, and 5 named
trains), 823 stations. Every register line is within 2% of its published length
(`check_model.py`, worst 1.02):

| line | built km | published | source |
|---|---|---|---|
| สายเหนือ Northern Line | 767.5 | 751.48 | EN; built adds the 2025 Lop Buri bypass and the Krung Thep Aphiwat approach |
| สายสวรรคโลก Sawankhalok | 28.7 | 29.007 | EN |
| สายชุมทางบ้านภาชี–อุบลราชธานี (Ubon) | 486.9 | 485.15 | NE km posts 575.10 − 89.95 |
| สายชุมทางถนนจิระ–หนองคาย (Nong Khai) | 357.4 | 354.82 | NE km posts 621.10 − 266.28; +2.7 over the bridge to the border |
| สายชุมทางแก่งคอย–ชุมทางบัวใหญ่ | 252.4 | 249.887 | EN |
| สายตะวันออก Eastern (Aranyaprathet) | 259.2 | 255 | EN; built Yommarat - Ban Khlong Luk Border |
| สายชุมทางฉะเชิงเทรา–สัตหีบ | 134.3 | 134 | EN (Chuk Samet) |
| สายใต้ Southern | 1,156.3 | 1,144.16 | EN; built adds Bang Sue - Taling Chan |
| สายสุพรรณบุรี | 78.0 | 78.09 | EN |
| สายน้ำตก (Nam Tok, the Burma Railway) | 130.1 | 130.989 | EN |
| สายคีรีรัฐนิคม | 31.0 | 31.25 | EN |
| สายกันตัง | 92.7 | 92.802 | EN |
| สายนครศรีธรรมราช | 34.9 | 35.081 | EN |
| สายชุมทางหาดใหญ่–ปาดังเบซาร์ | 44.4 | 45 | EN (to Malaysia's Padang Besar station) |
| สายแม่กลอง (วงเวียนใหญ่–มหาชัย) | 31.0 | 31.22 | EN |
| สายแม่กลอง (บ้านแหลม–แม่กลอง) | 33.9 | 33.75 | EN |

Against SRT's own network total: 4,044 km open (th.wikipedia "การรถไฟแห่งประเทศไทย": North
781, Northeast 1,094, East 534, South 1,570, Mae Klong 65). Less the freight-only lines left
out (Khlong Sip Kao - Kaeng Khoi 81.4, Map Ta Phut 24.1, Laem Chabang 13.5, Mae Nam 6.6) that
is 3,918 km; built 3,919 (the Lop Buri bypass and the Bangkok approaches in, the Hua
Lamphong trunk counted once).

Sources: EN = en.wikipedia "Rail transport in Thailand" (SRT's list of main and branch
lines with chainage to the metre); NE = en.wikipedia "Northeastern Line (Thailand)" (km posts
Ban Phachi Jn 89.95, Kaeng Khoi Jn 125.10, Thanon Chira Jn 266.28, Bua Yai Jn 345.50, Nong
Khai 621.10, Ubon 575.10); Wikidata P2043 agrees within 1 km where it has a figure (Chiang Mai
Main Line 751.42, Ubon Main Line 575.1). en.wikipedia "Nong Khai railway station" gives
623.58 km from Bangkok, 2.5 km more than the line article; the line article's is used.

## Where the register comes from

No open SRT line register exists (no shapefile, no station-km table on data.go.th that could
be found; the national timetable feed is partial, below). OSM Thailand has what is needed:

- **Named track**: 79% of main-line rail km carries a line name (`probe_kr_ways.py --region
  th`), but coarse ones: "สายตะวันออกเฉียงเหนือ" for both Isan main lines, "ทางรถไฟสายตะวันออก"
  for the Aranyaprathet and Sattahip lines.
- **route=railway relations** (infra.pkl, 50): Northern 1820650, Northeastern 1820651, Lower
  Isan 8425142, Thanon Chira - Khamsavath 17458406, Eastern 1820706 (+ unnamed 9407411),
  Southern 8342730 / 8425164 / 8425165, and one per branch. They cover most unnamed track.

So each way gets a line by its name (`NAME_LINE`), else its relation (`REL_LINE`, the
Northern Line first where the Northern and Northeastern relations both hold the Bangkok - Ban
Phachi trunk), else its neighbours (`propagate`); the coarse names are split by geography
away from the junctions and by track connectivity near them (`split`, `split_components`).
Then kr_register's recipe (as gb_register uses it) makes sections between stations along
each line's own track. Additions for Thailand:

- **Station lists by proximity**: every OSM rail station within 250 m of a line's track is
  on that line (no per-line list exists). Bangkok's ARL, BTS and MRT station and stop
  records are left out first (the ARL runs beside the Eastern Line; its Ramkhamhaeng has no
  SRT halt). The Red Lines' records stay: they stand where the at-grade line's halts are.
- **Junction stations**: a branch whose track ends on another line's takes the station
  nearest that end within 1 km (Kantang 370 m from Thung Song Jn, Suphan Buri 350 m from Nong
  Pladuk Jn, Bua Yai 890 m from Bua Yai Jn, Sattahip 390 m from Chachoengsao Jn), as SRT
  counts branches from the junction station.
- **Gaps**: a line in pieces is joined over other track where two of its stations are within
  12 km and track joins them within 1.5x the crow-fly (3 gaps on the Northern Line, 15 km:
  Ban Phachi Jn - Don Ya Nang, Ban Pa Wai - Lop Buri, Khlong Phutsa - Bang Pa-in).

## Decisions (made here, not put to Anita)

- **Line unit**: SRT's list of main and branch lines, the Northeastern Line as its three
  parts (Ban Phachi - Ubon, Thanon Chira - Nong Khai, Kaeng Khoi - Bua Yai). The Bangkok -
  Ban Phachi trunk is the Northern Line's (SRT's 751 km to Chiang Mai include it), and
  Hua Lamphong - Yommarat too, so the Eastern Line starts at Yommarat. The Mae Klong Railway is
  two lines: its two pieces do not meet (a ferry over the Tha Chin at Maha Chai).
- **Named trains**: every SRT train has a number, and OSM maps SRT trains one relation per
  number (405/406 Sila At - Sawankhalok, 147/148 Udon Thani - Khamsavath, 4302 on the Mae
  Klong, 1123/1124 Thon Buri - Nakhon Pathom, Bangkok Connex 9001-9006 Krung Thep Aphiwat -
  Ayutthaya). Each is a single train: a named train (rules/th.py). That holds even where the
  service is frequent (Mae Klong, ~17 a day; Bangkok Connex), because there the line a rider
  uses is the register line every train runs over; OSM's per-train relations would otherwise
  be several copies of one line (Bangkok Connex was three identical 63.5 km lines).
  SRT's special express, express, rapid, ordinary and commuter classes change nothing: the
  register lines are the lines, the numbered trains are trains.
- **Lines kept as OSM lines**: SRT Dark Red and Light Red (SRTET, every 10-20 min), the
  Airport Rail Link, BTS Sukhumvit, Silom, Gold, MRT Blue, Purple, Yellow, Pink (+ its
  Muang Thong Thani branch), and the Suvarnabhumi airport people mover (airside; kept like
  other countries' airport movers).
- **Freight only, no register line**: Khlong Sip Kao - Kaeng Khoi (freight since 1995, en.
  wikipedia "Khlong Sip Kao Junction railway station"), Map Ta Phut, the Laem Chabang branch
  (its station is in `NOT_STOPS`), Mae Nam (no track carries that name in the extract).
- **Not open, no register line**: Den Chai - Chiang Rai - Chiang Khong (OSM has 170 km of it
  as railway=rail; it shows in build_tiles' "pieces no line touches"), Ban Phai - Nakhon
  Phanom, the Bangkok - Nakhon Ratchasima high-speed line, the three-airports link.
- **Running, kept**: Chuk Samet (train 283 extended Ban Phlu Ta Luang - U-Tapao - Chuk Samet
  in 2023, mgronline / thansettakij); Khiri Rat Nikhom (one local a day, 489/490); Sawankhalok
  (405/406); the Lop Buri bypass with Lopburi 2 (opened 5 Dec 2025 for 14 long-distance
  trains; bangkokbiznews, thairath); the at-grade line Hua Lamphong - Rangsit (the ordinary
  and commuter trains from Hua Lamphong).

## Borders

Track over a border ends at the point where it crosses the national boundary in OSM
(Overpass, admin_level=2 boundary against the rail ways, `data/raw/th/borders_osm.json`):

| id | lon | lat | countries | trains over it |
|---|---|---|---|---|
| xPadangBesar | 100.322477 | 6.665252 | my, th | yes: SRT 45/46 and the Hat Yai shuttles to Padang Besar's joint station on the Malaysian side, KTM's ETS and Komuter from it. OSM node 11494429960, where SRT's way 1419678317 meets KTM's 1237531937 |
| xNongKhaiThanaleng | 102.715092 | 17.880451 | la, th | yes: 133/134 and 147/148 to Khamsavath. Mid-Mekong on the Friendship Bridge (no node) |
| xAranyaprathetPoipet | 102.550138 | 13.661698 | kh, th | no: SRT's trains end at Ban Khlong Luk Border station, 0.4 km short |

th_register offers them as stations itself until they are in `borders.EXTRA`. A border
section is a junction section, kept by build_model only where an OSM route runs over it: Nong
Khai's is (147/148), Padang Besar's (0.46 km) is dropped until build_model reads the line's
`served_sections` (proposed to the managing session), Khlong Luk's is dropped, rightly.

## Timetables

- **OTP's national feed** (Office of Transport and Traffic Policy and Planning, Ministry of
  Transport; Mobility Database mdb-1831, `https://namtang-api.otp.go.th/download/namtang-
  gtfs.zip`, CC BY 4.0, 42 MB, version 20261002, kept as `data/raw/th/namtang-gtfs.zip`):
  buses, BTS, MRT, ARL, Red Lines and 188 SRT route ids. Not usable as the running check:
  SRT trips are truncated (167 Bangkok - Kantang stops at Nakhon Pathom, 133 to Nong Khai at
  Rangsit), regional trains are missing (no 4xx locals outside the Mae Klong and Sala Ya),
  no shape_dist_traveled. Its SRT trips call at 178 stations; on the Sattahip, Nakhon Si
  Thammarat, Khiri Rat Nikhom and Sawankhalok lines at none. With it, gtfs_served would grey
  most of the network. **No `gtfs_served.FEEDS["th"]` entry**; do not put it in
  data/raw/gtfs/th.
- The Mobility Database has nothing else live for Thailand (2018 Chiang Mai bus feeds).
- `https://www.thaitrainguide.com/all-the-lines-thailand/` (an unofficial per-line guide with
  lengths and services) answers 403; not retried.

## Open

- Padang Besar's border section waits for the `served_sections` hook in build_model.
- OSM names named trains with the number stripped by build_model's name tidying: "ขบวน" /
  "Train" for 1123/1124, "/ กรุงเทพอภิวัฒน์ - อยุธยา ..." for Bangkok Connex (reported to the
  managing session; the ref still carries the number).
- No line colours: SRT publishes none for its intercity lines; OSM's colours on the urban
  lines are used.
- Long sections that may hide an unmapped halt: Udon Thani - Na Phu 24.2 km, Na Phu - Na Tha
  24.8, Kapang - Huai Yot 24.5, Huai Yot - Trang 28.5, Khlong Ngae - Hat Yai 24.1.
- The Northern Line's at-grade (Hua Lamphong) and elevated (Krung Thep Aphiwat) tracks
  between Bang Sue and Rangsit are one line; the Red Line's station names stand for the
  at-grade halts.
