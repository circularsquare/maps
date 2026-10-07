# Hong Kong register sources (surveyed 2026-09-30)

What `hk_register.py` reads, where each piece came from, and what is wrong with it. Downloads
are in `data/raw/hk/` (gitignored); this file is the tracked record of them. Nothing here
needed a login, a key or an account.

## Run

```powershell
curl -L -o data/raw/hk-260929.osm.pbf https://download.openstreetmap.fr/extracts/asia/china/hong_kong.osm.pbf
python extract.py --region hk --pbf data/raw/hk-260929.osm.pbf        # 5 s
python hk_register.py --clip                                          # cut the Shenzhen side out
python build_model.py --region hk --register hk_register:data/raw/hk  # 3 s
python build_tiles.py --region hk                                     # 2 s
python check_model.py --region hk
```

`--clip` must run after every extract: it rewrites `data/proc/hk` in place.

## The short answer

- **Geometry: OSM's named track**, as in Korea and Taiwan. 99.7% of rail, 99.9% of subway
  and all light-rail km carry the line's name (`python probe_kr_ways.py --region hk`), written
  bilingually in one tag: "港鐵東鐵綫 MTR East Rail Line". Licence ODbL.
- **MTR lines and Light Rail: MTR's own open data** for which stations, in which order. No
  distances and no positions, so there is no per-section km_official.
- **Trams and the Peak Tram: OSM's own route relations** for their stops.
- **Lengths to check against**: the Highways Department for East Rail, en.wikipedia for the
  rest (below, and `check_model.REGISTER["hk"]`).

## The OSM extract

Geofabrik has no Hong Kong extract (only `asia/china`, 1.5 GB). openstreetmap.fr has one:
https://download.openstreetmap.fr/extracts/asia/china/hong_kong.osm.pbf (42 MB, data to
2026-09-29). It runs a few km past the border and holds Shenzhen Metro lines 1, 2, 4, 7, 8, 9,
10 and 13, the Guangshen railway at 深圳站 and the Shenzhen half of the high-speed line.

`hk_register.py --clip` cuts those out with OSM's own boundary of Hong Kong, relation 913110,
from https://polygons.openstreetmap.fr/get_geojson.py?id=913110&params=0
(`hk_boundary.geojson`). A way stays if at least half its nodes are inside. Every HK-named way
is wholly inside; what goes is 44 unnamed Shenzhen yard ways, 广深线, 深圳地铁 1/2/4/7/8/9/10/13
and the two 广深港高速线 ways north of the border. 羅湖 and 落馬洲 stay; 罗湖, 深圳, 福田口岸 and
深圳湾口岸 go. The two 50 m ways over the Lo Wu bridge (A线, B线) stay, being half inside. (The
religiondots country outline is a land outline and puts half of Victoria Harbour's tunnels
outside Hong Kong, so it is not used.)

## MTR lines and stations (DATA.GOV.HK)

https://opendata.mtr.com.hk/data/mtr_lines_and_stations.csv (`mtr_lines_and_stations.csv`),
listed on DATA.GOV.HK as "MTR Lines and Stations". Terms: DATA.GOV.HK terms of use, free
with attribution.

Line code, direction (UT/DT, plus LMC-UT/LMC-DT for East Rail's Lok Ma Chau branch and
TKS-UT/TKS-DT for LOHAS Park), station code, Chinese and English name, sequence. 10 lines,
98 station rows.

- It writes 茘景 and 茘枝角 with U+8318; OSM has the usual 荔 (U+8354). Folded in `zh_key`.
- 馬場 Racecourse (East Rail, race days only) is not in it, so it is not on the register
  line; OSM's East Rail relations that call there stay as they are.
- A section is kept only between two stations the file has one after the other, which is
  what keeps 落馬洲-羅湖 (two termini of one line whose track meets north of 上水) out.

## Light Rail routes and stops (DATA.GOV.HK)

https://opendata.mtr.com.hk/data/light_rail_routes_and_stops.csv
(`light_rail_routes_and_stops.csv`), "Light Rail Routes and Stops". Same shape, per route and
direction; 12 routes, 68 stops. Light Rail is built as ONE register line (輕鐵), the network MTR
counts as 36.2 km; the routes (505, 507, 610 ...) stay OSM lines over it.

**The build reads 39.6 km against 36.2 (1.09)**, and why: sections run between stops that are
consecutive on some route, so where the two directions use different streets (the one-way
pairs around 石排/新圍/大興, 屯門碼頭's terminal loop, 天水圍) both streets are sections. Measured
on the built geometry, 2.65 km of section runs beside another section away from any stop, so
about 1.3 km is counted twice. The rest is the network's triangles (洪水橋-塘坊村-坑尾村 and
others), each leg of which is a section because some route runs it; checked on a plot of the
built sections over OSM's track, with no detours found. MTR's figure presumably counts each
piece of track once.

## Trams and Peak Tram: OSM route relations

- **Hong Kong Tramways** (香港電車): the 12 route relations' stop members. Each direction's
  stop is its own OSM node with its own name (荷蘭街 eastbound is 堅彌地城海旁 westbound, at one
  kerb), so the two records of a stop on different tracks within 120 m are one station,
  named "A / B" where the names differ. 68 stations. 16.2 km against 15.9 (13.3 main line +
  2.6 Happy Valley loop, en.wikipedia Hong Kong Tramways), 1.02; probably the terminal loops
  (堅尼地城, 上環(西港城), 北角, 筲箕灣), not checked stop by stop.
- **Peak Tram** (山頂纜車): the one route relation; six stops. 1.25 km against 1.364. The
  published figure is along the slope, and the line climbs 369 m (27 to 396 m, en.wikipedia
  Peak Tram), so on the map it is about 1.31 km; OSM's track end to end is 1.26 km and the
  stations sit at its ends.

## Published lengths (`check_model.REGISTER["hk"]`, retrieved 2026-09-30)

- East Rail 46.0: Highways Department, Shatin to Central Link,
  https://www.hyd.gov.hk/en/our_projects/railway_projects/scl/index.html, "approximately
  46km, connecting the Admiralty Station and the Lo Wu Station/Lok Ma Chau Station".
  en.wikipedia gives 紅磡-羅湖 34 and the Lok Ma Chau spur 7.4; the build has the spur at 7.38.
- Every other MTR line: the "Line length" in its en.wikipedia infobox (Tuen Ma 56.193, Kwun
  Tong 17.32, Tsuen Wan 15.59, Island 14.93, South Island 7.4, Tung Chung 31.1, Airport
  Express 35.2, Tseung Kwan O 12.3), not the "Track length" some also give (Island 16.3,
  Tsuen Wan 16.9, Kwun Tong 18.4). Disneyland Resort 3.3 is from the line table in
  en.wikipedia "MTR"; its own page says 3.8 in the infobox and 3.5 in the text.
- **South Island (0.93) and Disneyland Resort (0.93)** are figures to the ends of the track:
  OSM's own track end to end is 7.05 and 3.51 km, against station-to-station 6.91 and 3.08.

## Line colours (`colours/hk.csv`)

MTR's system map, https://www.mtr.com.hk/archive/en/services/routemap.pdf (`mtr_routemap_en.pdf`):
the stroke colours of the 16.3 pt line paths, read with PyMuPDF. The PDF's text is outlined,
so each colour was tied to its line by hue and extent (brown the longest, Tuen Ma; the pink one
the shortest, Disneyland Resort), and every one agrees with OSM's relation colours to within a
shade. Light Rail is the thin gold path. The grey 16.3 pt path (#9C948B) is the high-speed line,
which is not built. The trams and the Peak Tram have no published line colour; theirs are
`picked` from their liveries.

## Over the border (2026-10-04)

**The high-speed line, built as Hong Kong's piece of China's 广深港高速线.** Anita saw no stop
on the Hong Kong side of the line from Shenzhen. Hong Kong's section (香港西九龍 to the border,
26 km) has one station and no OSM route relation, so until now it had no second station to
make a section to, and China's build ended the line at 福田. Now:

- A border point `xFutian` (borders.EXTRA): OSM splits both tracks on its boundary of Hong
  Kong (relation 913110) in the tunnel under the Shenzhen River, at nodes 11 m apart; the
  point is between them. Named "China – Hong Kong border".
- Hong Kong builds **香港西九龍 - China – Hong Kong border, 25.46 km** over the ways OSM names
  廣深港高速鐵路 (the platform roads included, yard roads not). Published: "26 km" for the Hong
  Kong section (en.wikipedia, Hong Kong section of the XRL), which runs to the end of the
  platforms: 0.98. China builds 福田 - border (cn_sources.md).
- Hong Kong's piece **takes China's line id** (`c1d25d1a02a`, cn_register.line_id of
  广深港高速线), name 廣深港高速鐵路, MTR as operator, high-speed. The app joins lines of one
  id from every country into one line, each country's totals counting its own piece, so 福田 ->
  香港西九龍 (or 广州南 -> 香港西九龍) is one ride on one line and credits both. Two lines of
  their own would have met at the border point and left no line calling at both stations: the
  app never joins register lines of different ids, and there is no OSM route to carry the
  ride. Colour: MTR's map draws the high-speed line in grey, #9C948B (`colours/hk.csv`); China
  has no line colours, so the joined line takes it.
- Kept by build_model through the line's `served_sections`. `BORDER_PIECES` in hk_register.py.

**East Rail at 羅湖 and 落馬洲 stays as it is.** No train crosses: passengers walk over the
border to 罗湖 and 福田口岸 on the Shenzhen side, so East Rail ending at both stations is right.
**The Intercity Through Train** (Hung Hom - Guangzhou East, Beijing West, Shanghai) is gone:
suspended in January 2020, never resumed, and its four mainland ports were closed by the State
Council on 2024-07-05 (The Standard, "Time's up for intercity through trains", 2024-08-01;
MTR had already called it the end in 2022, SCMP "End of an era"); China Railway replaced the
Beijing and Shanghai trains with high-speed sleepers to West Kowloon on 2024-06-15. Nothing
to build.

## Not built, and known faults

- **The airport's people mover** (airside), **Ocean Park's 海洋列車** and the
  **Disneyland Railroad** are not register lines. The first two have OSM route relations and
  stay as OSM lines.
- **機場快綫 is listed twice**: the register line and OSM's "機場快綫 Airport Express". The
  Airport station has no station node in OSM, so build_model resolves the relation's stop
  "機場 Airport" onto the nearest station record, the people mover's "二號客運大樓 Terminal 2"
  200 m away; the two then share 4 of 5 stations and are not recognised as one line. A fix in
  build_model (by mode, or named stops only) moved more stops elsewhere for the worse than it
  fixed, so it was left.
- OSM's tram route_master (17.9 km) and Peak Tram relation (2.1 km) also stay beside their
  register lines: they differ from them in length by more than build_model's 6%.
- Station names are the Chinese half of OSM's bilingual names; English names come from MTR's
  files for MTR and Light Rail, else OSM `name:en`.
