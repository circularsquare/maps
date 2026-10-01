# Singapore register sources (surveyed 2026-09-30)

What `sg_register.py` reads, where each piece came from, and what is wrong with it. Downloads
are in `data/raw/sg/` (gitignored); this file is the tracked record of them. Nothing needed a
login, a key or an account.

## Run

```powershell
# Geofabrik ships Singapore inside Malaysia-Singapore-Brunei; the -latest URL redirect-loops,
# so take the dated file from https://download.geofabrik.de/asia/malaysia-singapore-brunei.html
python extract.py --region sg --pbf data/raw/sg/malaysia-singapore-brunei-260929.osm.pbf --bbox 103.6,1.2,104.05,1.452   # 20 s
python sg_register.py --clip          # drops Johor's Pasir Gudang and Tanjung Pelepas track the box keeps
python build_model.py --region sg --register sg_register:data/raw/sg        # 2 s
python build_tiles.py --region sg                                           # 1 s
python check_model.py --region sg
```

The `.pbf` is deleted once extracted; `data/proc/sg/` holds what was pulled out of it.

## The short answer

- **Geometry: OSM's named track**, as in Korea and Taiwan. 99.4% of Singapore's metro track
  in the extract (data to 2026-09-29) carries its line's name (`python probe_kr_ways.py
  --region sg`). The Sengkang and Punggol LRTs are the exception: their track is unnamed, so
  their track is the ways of OSM's own route relations for them. Licence ODbL.
- **Which stations each line has, in order: LTA's own list** (DataMall's station code file),
  brought up to date from LTA's 2026 system map.
- **Where each station is on the track: OSM's route relations' stop nodes.** They list every
  open station of every line and nothing else; the build log checks them against LTA's list
  and found no disagreement in 2026-09.
- **Line lengths: LTA's line pages**, whole km, and English Wikipedia where LTA gives none.
  There is no open per-section chainage, so no `km_official`.

## LTA DataMall: Train Station Codes and Chinese Names

https://datamall.lta.gov.sg/content/dam/datamall/datasets/PublicTransportRelated/Train%20Station%20Codes%20and%20Chinese%20Names.zip
(`lta_station_codes.zip`, unzipped to `Train Station Codes and Chinese Names.xls`), from
DataMall's static datasets. Licence: Singapore Open Data Licence.

203 rows of `stn_code, mrt_station_english, mrt_station_chinese, mrt_line_english,
mrt_line_chinese`. The codes give each line's order (NS1-NS28, EW1-EW33, CG1-CG2, STC and
SE1-SE5 / SW1-SW8, and so on). What is wrong with it:

- **Stale.** It stops at TE22 and has no Hume (DT4, opened 2025-02-28), Punggol Coast (NE18,
  2024-12-10), Teck Lee (PW2, 2024-08-15), TEL Stage 4's TE23-TE29 (2024-06-23) or Circle Line
  Stage 6's CC30-CC32 (2026-07-12). `ADDED` in sg_register.py puts them in from the system map.
- It lists the Changi Airport branch (CG) and the Circle Line Extension (CE) as lines of
  their own. LTA's line pages and system map count them in the East-West and Circle Lines,
  and so does the register. The system map renumbers CE1 Bayfront as CC34 and CE2 Marina Bay
  as CC33 (`RECODE`).
- Some Chinese names carry stray spaces and a trailing 站 (大 士 西 路站); they are cleaned but
  not used, since a station dict has one native name and Singapore's OSM names are English.

## LTA's 2026 system map

https://www.lta.gov.sg/content/dam/ltagov/getting_around/public_transport/rail_network/pdf/SM_EN_(Ver210726)_CCL6.pdf
(`lta_system_map_en_260721.pdf`, map SM-26-01-EN, version dated 21-07-26), linked from
https://www.lta.gov.sg/content/ltagov/en/getting_around/public_transport/rail_network.html.

- Station codes of the stations the DataMall file lacks, and the CC33/CC34 renumbering.
- It shows TEL Stage 5 (TE30 Bedok South, TE31 Sungei Bedok), DT36 Xilin and DT37 Sungei
  Bedok as under construction. OSM agrees: no relation calls there and no track is open. They
  are not built.
- **Line colours** (`colours/sg.csv`) are the map's own vector strokes, read with PyMuPDF
  (`page.get_drawings()`, the 14.7 pt line strokes): NSL E12219, EWL 00953B, NEL 9E26B5,
  CCL FF9F10, DTL 0056B8, TEL 9D5918, and 6F8473 for all three LRTs.
- It draws the Sentosa Express in its "other transport modes" near-black (2D2A26), so that
  colour is not a line colour. The Sentosa Express keeps OSM's route_master colour, DD0B61,
  which nothing official confirms.

## Line lengths (`check_model.REGISTER["sg"]`)

LTA's line pages, https://www.lta.gov.sg/content/ltagov/en/getting_around/public_transport/rail_network/<line>.html
(north_south_line, east_west_line, north_east_line, circle_line, downtown_line,
thomson_east_coast_line, bukit_panjang_lrt, sengkang_punggol_lrt), 2026-09-30:

| Line | LTA | stations | used |
|---|---|---|---|
| North-South | 45 km | 27 | 45.0 |
| East-West | "approximately 57km" | 35 | 57.0 |
| North East | 22 km | 17 | 21.6: 20 km plus the 1.6 km Punggol Coast extension (en.wikipedia) |
| Circle | 39 km, with Stage 6 | 33 | 39.0 |
| Downtown | 42 km | 35 | 42.0 (en.wikipedia 41.9) |
| Thomson-East Coast | 40.6 km | 27 | 40.6 |
| Bukit Panjang LRT | 8 km | (13) | 8.0 |
| Sengkang LRT | none | 14 | 10.7, en.wikipedia |
| Punggol LRT | none | 14, plus Punggol | 10.3, en.wikipedia (Mochidome & Masukawa 2003) |
| Sentosa Express | not LTA's | 4 | 2.1 per direction, en.wikipedia |

Every station count matches what the build makes.

## What is still off, and why

A published length runs to the ends of the running track; the build runs station centre to
station centre. OSM's own running track per direction is in each `REGISTER` note.

- **Downtown Line 0.95**: 1.5 km of its track is the lead from Bukit Panjang to Gali Batu
  depot, beyond the last station. Add it and the build is 41.3 of 42.
- **Bukit Panjang LRT 0.95**: LTA's 8 km still counts the Ten Mile Junction spur, closed
  2019-01-13, and OSM's whole remaining track is 7.7 km a direction.
- **Punggol LRT 0.92**: the build is the track OSM's route relations run over (their ways
  total 9.7 km, the build 9.5). The 10.3 km figure is a 2003 design figure and must count
  track no passenger service uses. My guess is the link to the Sengkang depot, but nothing
  confirms it.
- **Sentosa Express 0.95**: OSM's whole running track is 1.99 km a direction and all of it
  is built; the 2.1 is rounded or counts track beyond the platforms.
- North East 0.95 against 21.6, with 0.7 km of tail track past HarbourFront and Punggol Coast.

## Left out

- **Changi Airport Skytrain**: an airport people mover (Changi Airport Group), not on LTA's
  list. OSM keeps it as an OSM line.
- **Jurong Region Line**: under construction (first stage due 2027). OSM tags its station
  nodes `railway=station`, but it has no open track and no stops in a service relation.
- **RTS Link** (Woodlands North to Bukit Chagar, Johor Bahru): not open (planned end-2026).
- **KTM Shuttle Tebrau** (Woodlands Train Checkpoint to JB Sentral): open, but its Singapore
  part is 1.1 km of track from the checkpoint to the middle of the causeway, with one station.
  OSM has no route relation for it, so a section ending at the border would be dropped as
  unridden anyway. Left out (decided 2026-09-30); it comes back naturally if Malaysia is
  ever built, and the RTS Link replaces it.
- **Sentosa's other transport**: the cable car is `aerialway`, not rail.

## Shared-code change it needed

`build_model.norm_line_name` (landed 2026-09-30): "LRT X Line" reads as "X LRT", a leading
"MRT " is dropped, and hyphen, en dash and space between words are one spelling. Without it
only the Sentosa Express matched its OSM route_master by name, and nine OSM lines stayed
listed beside their register twins. It changed no match in jp, ch, kr or tw.

What is left from OSM after the merge: the Changi Airport Skytrain, and "LRT Sengkang Line"
kept as a duplicate (9.0 km against the register's 10.7). OSM names the Sengkang LRT's stop
positions "Sengkang - East Loop Anticlockwise" and so on, so the OSM-built line never
recognises them as Sengkang and its sections skip the station. The register line is right.

## Station and line English names

Line names are LTA's, as on its system map; OSM's route relations are called "MRT
North-South Line" and "LRT Bukit Panjang Line" and its track "North South Line (NS)".
Station names are OSM's `name`, which in Singapore is English and matches LTA's spelling.
