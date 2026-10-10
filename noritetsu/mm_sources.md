# Myanmar (mm): sources

Built 2026-10-08 (asia agent): see "Build (2026-10-08)" at the end. The survey below was the research for it.

## Survey (2026-10-08)

### What runs

Myanma Railways (MR), metre gauge, 6,207.6 km and 960 stations (en.wikipedia "Myanmar
Railways"). Since the 2021 coup and the civil war a large part of the network has no passenger
train; what is known on 2026-10-08:

| line | status | evidence |
|---|---|---|
| Yangon - Mandalay (620 km) | running: three daily departures each way (MR3/4, 5/6, 9/10, 11/12) | baolau's MR page; ICS Travel news, 2026 |
| Yangon - Mawlamyine (296 km) | running, daytime only after a 2024 mine blast; new DEMU Express 83/84 from 1 July 2026 | ICS Travel news; BNI "Yangon-Mawlamyine train trips temporarily canceled after railroad mine blast"; baolau sells MR81/82, 83/84 |
| Yangon Circular Railway (about 46 km loop; 81 km with the suburban branches) | assumed running (Yangon is under junta control; OSM maps the circular and the Thilawa, Insein - Hlawga, Dagon University, Eastern University, Ywathagyi and Computer University routes) | no 2026 source read; check |
| Mandalay - Myitkyina (361 km), Mandalay - Shwebo, Mandalay - Khin-U | **not running** since 2021: MR staff joined the CDM, and Sagaing and Kachin are contested | DVB, "Over two years with no train service to Upper Burma" |
| Mandalay - Lashio (441 km, the Gokteik viaduct) | unknown; Lashio was held by the MNDAA from Aug 2024 to Apr 2025. Probably at most Mandalay - Pyin Oo Lwin. Check | none read |
| Sittwe Pyidawtha - Yaychanpyin (Rakhine, short line) | BNI reported a resumption; Rakhine is mostly Arakan Army territory now. Check | BNI "Train services to resume this month along Sittwe's Pyidawtha-Yaychanpyin rail route" (date not read) |
| Yangon - Pyay, Thazi - Shwenyaung - (Kalaw, the Inle line), Tanintharyi line (Mawlamyine - Ye - Dawei), Pyinmana - Taungdwingyi, the Bagan lines, Pakokku - Kalay, Hinthada - Pathein, Loikaw, Myingyan... | unknown | none read |

Anita's rules apply: each line gets a call from the build agent, written down; a line with no
evidence of a train since 2021 is built greyed (`suspended`), as Kép - Cái Lân in Vietnam, rather
than left out. A greyed default for "unknown" is the safe side.

### Line list with stations and mileposts

**en.wikipedia "List of railway stations in Myanmar"** (raw wikitext saved:
`data/raw/mm/survey/enwiki_list_of_railway_stations_in_myanmar.wikitext`, 2026-10-08, CC BY-SA):
every MR division's lines, each with its stations in order and **mileposts in miles from
Yangon** (or from the line's own origin: "Myitkyina 722 3/4 miles from Yangon", "Naba
(Junction) 590", "Katha 605"), 923 list rows over 11 divisions and about 45 line headings:
Mandalay - Myitkyina, Naba - Katha, Yangon - Mandalay, Mandalay - Lashio, Myingyan, Tha Ye Ze,
Madaya, Kalaw - Thazi, Loikaw, Kalaw - Lawksawk, Shwenyaung - Taunggyi - Nansan - Mong Nai,
Pyinmana - Taungdwingyi, Nyaunglebin - Madauk, Yangon - Mawlamyine, Yangon - Pyay, Pyay -
Aunglan, Letpadan - Tharawaw, Yangon Circular, Dagon University, Thilawa, Hlehlaw-In,
Tanintharyi line, Hpa-An branch, Kyangin - Hinthada - Pathein, Pakokku - Kalay, Aunglan,
Taungdwingyi - Bagan, Meiktila - Myingyan, Kyaukpadaung - Kyeeni, Taungdwingyi - Magway, Magway
- Kanbya. Its source is the old Ministry of Rail Transportation division maps
(ministryofrailtransportation.com, dead). Mileposts are fractions ("566 1/2"): convert to km.
A line crossing two divisions appears under both, so join the pieces.

Also saved: en.wikipedia "Rail transport in Myanmar" (`enwiki_rail_transport_in_myanmar.wikitext`).

### Other sources

| source | gives | licence | where |
|---|---|---|---|
| OSM via Geofabrik `asia/myanmar-latest.osm.pbf` (271 MB) | 6,970 `railway=rail` ways (5,107 `usage=main`, only 808 named), 786 stations and halts; `route=railway` 7170454 Lashio Line, 13760507 Mandalay-Lashio, 18841554-7 (Shwenyaung - Taunggyi - Pinpat, Shwenyaung - Mongnai, Kalaw - Lawksawk, Kalaw - Thazi); 14 Yangon commuter route relations (`route=light_rail`, MR, coloured: Circular both ways, Thilawa, Insein - Hlawga, Dagon University, Eastern University, Ywathagyi, Computer University) | ODbL | `data/raw/mm/survey/osm_route_relations.json` |
| Mobility Database (2026-10-08) | nothing for MM | | |
| baolau.com MR page | train numbers on the two sold corridors | (reading) | https://www.baolau.com/th/van-tai/myanmar/tau-hoa/mr-myanma-railways/ |
| MR's own timetables | MR posts timetables on Facebook (not read; not a fetchable source) | | |

### Recipe

The Indonesia pattern (`id_register.py`): wiki line tables with km posts through rinf.py.

- Parse the station list: per line heading, stations in order with milepost -> km. Join the
  divisional pieces of a line (Yangon - Mandalay is in four divisions).
- Place stations on OSM's 786 stations by name. OSM's `name` is Burmese; matching the wiki's
  English romanisations ("Hto Pu", "Kya Gyi Kwin") needs `name:en` where present, else a
  romanisation compare plus position along the traced line. This is the main work; expect a
  hand alias table.
- Lines: the wiki headings, MR's own line names ("Yangon - Mandalay line"). Branches that are
  freight or industrial sidings are left out.
- Yangon's commuter routes: the Circular is a register line (it is in the station list as its
  own line); the other OSM `light_rail` relations (Thilawa, Dagon University...) run over
  register lines and stay OSM lines. Note OSM tags them `route=light_rail`: build_model may
  need them read as rail (`rules/mm.py`).
- Running status per line as above; unknown -> greyed.
- `check_model` REGISTER: the wiki's line km (Yangon - Mandalay 620, Mawlamyine 296, Pyay 259,
  Tanintharyi 339, Myitkyina 361, Lashio 441) and the mileposts themselves as `chain`.

Expected: 30-40 register lines, about 5,500-6,000 km, with perhaps 1,000-1,500 km running.
Extract about 2-4 min; model under a minute.

### Open

- Running status of every line outside Yangon - Mandalay, Yangon - Mawlamyine and Upper Burma
  (above). A build agent should spend its searches here (DVB, Irrawaddy, Myanmar Now, BNI,
  Global New Light of Myanmar for junta announcements).
- The wiki list predates the 2010s: newer lines (e.g. to Kyaukphyu? no; the Thanbyuzayat -
  Ye extension, the Pakokku - Kalay pieces) may be missing or partly built.
- Who the map depicts: nothing unusual. Lines through areas held by armed groups are drawn as
  track like any other; "running" means MR trains, whoever holds the ground.

## Build (2026-10-08)

    python tools/slot.py 2 -- python extract.py --region mm --pbf data/raw/myanmar-latest.osm.pbf --station-areas
    python asia_register.py --clip mm
    python asia_register.py --join mm             # 4 track ends a few metres apart
    python asia_register.py --convert mm
    python build_model.py --region mm --register asia_register:data/raw/rinf/mm
    python build_tiles.py --region mm; python check_model.py --region mm

Reader: `asia_register.py` (lk_register.py's engine), list `MM` in `asia_lines.py`: each line
its ends, junctions and a few stations between, every point an OSM station at its own
coordinate (OSM has name:en on 670 of 738 stations, spelled its own way); `osm_stops: "all"`
adds every OSM station on the traced track. Lengths are our traces (`no_chain`); the
en.wikipedia list's mileposts are the outside check. Settings `rinf_countries/mm.py`, rules
`rules/mm.py`, colours `colours/mm.csv` (picked).

**What runs** (the call, 2026-10-08): MR's own list of operating sections, as the state press
reported it in February 2025 (asianews.network, "Naypyidaw-Yangon route to be operational from
February 28"; the article itself refused our fetch, the list is from its search summary): four
express and two mail trains Yangon - Mandalay, two express and two mail Yangon - Pyay, two
mail Pathein - Hinthada - Kyangin, two mail Thazi - Shwenyaung, two express Pyin Oo Lwin -
Gokteik; Yangon - Mawlamyine runs (DEMU express 83/84 from 1 July 2026, ICS Travel; baolau
sells MR81/82); Yangon's circular and suburban lines run. Mandalay - Myitkyina has had no train
since 2021 (DVB; a 100-day repair of Myitkyina - Mohnyin in April 2026, no service yet). Seat61
(last updated January 2024, and no longer kept up) is not used for status. Everything else:
no evidence of a train, greyed.

| line | km | status | check (mileposts, en.WP list) |
|---|---|---|---|
| Yangon–Mandalay | 621.1 | running | 620 (en.WP): 1.00 |
| Yangon–Mawlamyine (from Bago) | 209.5 | running | 218.5: 0.96 |
| Yangon–Pyay | 259.2 | running | 259.1: 1.00 |
| Kyangin–Hinthada–Pathein | 237.2 | running | 236.6: 1.00 |
| Thazi–Shwenyaung | 155.9 | running | 157.7: 0.99 |
| Mandalay–Lashio (Pyin Oo Lwin – Gokteik) | 65.6 | running | 65.2: 1.01 |
| Mandalay–Lashio (Mandalay – Pyin Oo Lwin) | 67.3 | greyed | |
| Mandalay–Lashio (Gokteik – Lashio) | 157.3 | greyed | 157.3: 1.00 |
| Mandalay–Myitkyina (from Sagaing) | 531.4 | greyed | 532.3: 1.00 |
| Naba–Katha | 22.9 | greyed | 24.1: 0.95 |
| Tanintharyi (Mawlamyine – Dawei) | 310.3 | greyed | 307: 1.01 |
| Shwenyaung–Lawksawk | 59.9 | greyed | 60.4: 0.99 |
| Pakokku–Kalay (from Bagan) | 393.4 | greyed | |
| Pyay–Aunglan (to Taungdwingyi) | 166.7 | greyed | |
| Taungdwingyi–Bagan | 165.6 | greyed | |
| Loikaw (from Aungpan) | 164.1 | greyed | |
| Thazi–Myingyan | 112.9 | greyed | |
| Pyinmana–Taungdwingyi | 107.5 | greyed | |
| Sagaing–Monywa | 101.9 | greyed | |
| Monywa–Ye-U | 94.1 | greyed | |
| Madaya | 45.2 | greyed | |
| Shwenyaung–Taunggyi | 35.0 | greyed | |

Register 4,084 km in 22 lines: 1,549 km running (6 lines), 2,535 km greyed (16). OSM lines:
the Yangon Circular (46.0 km, 38 of 38 stations, en.WP 45.9) and six suburban routes (Thilawa,
Computer University, Eastern University, Ywar Thar Gyi, Dagon University, Insein - Hlawga),
running.

Decisions:
- **Pyin Oo Lwin – Gokteik running, Mandalay – Pyin Oo Lwin greyed**: the list names the
  Pyin Oo Lwin - Gokteik section alone; with nothing saying trains start at Mandalay, the
  stretch below is greyed (the safe side the survey set for unknowns).
- **The Yangon–Mawlamyine line starts at Bago**, where it leaves the Mandalay line (the
  list's own heading starts it there); Yangon - Bago belongs to the Mandalay line.
- **The Myitkyina line starts at Sagaing**: OSM has no track over the Irrawaddy between
  Mandalay's side and Sagaing (way 40299319 ends 1.9 km short of way 40300474). It is greyed in
  any case. Lines whose own route is otherwise unclear start at the station where they leave
  their parent (Lawksawk at Shwenyaung, Loikaw at Aungpan, Pakokku - Kalay at Bagan over the
  Pakokku bridge).
- **Shwenyaung – Taunggyi ends at Taunggyi**: OSM's track on to Nansang and Mong Nai is not
  joined to it (65 km apart in the graph), so the rest of that line is not drawn.
- **Left out**: Nyaunglebin - Madauk, Letpadan - Tharrawaw, the Hpa-An branch, Kyaukpadaung -
  Kyeeni, Taungdwingyi - Magway - Kanbya, the Burma Mines Railway (industrial), Yangon's tram.
  Short or unclear branches with no train since well before 2021; the managing session can
  ask for them.
- **Yangon's commuter routes are OSM lines** (route=light_rail in OSM, kind light_rail in the
  model, over MR's track). The Circular matches en.WP exactly, so no register line for it.
