# Tanzania (tz): sources

## Survey (2026-10-08)

Research only, nothing built. By the eafrica survey agent. TAZARA's Zambian half is in
`zm_sources.md`.

### What runs (freshest evidence)

| service | status | evidence |
|---|---|---|
| SGR Dar es Salaam (Magufuli) - Morogoro - Dodoma (Samia), electric | running, several a day (Express 06:00, EMU 08:00, Ordinary 09:30, more added for Parliament sittings; an extra Dodoma EMU from 25 June 2026) | TRC timetable from 3 Jan 2026 (Daily News, dailynews.co.tz "TRC releases new SGR timetable for Dar-Dodoma route"); The Citizen June 2026 |
| SGR Dodoma - Makutupora - Tabora and beyond | **not built**: Makutupora - Tabora under construction; no passenger trains past Dodoma | |
| TRC metre gauge, Central line: Dar - Dodoma - Tabora - Kigoma | running, 2-3 a week each way | TRC long-distance timetable image (trc.co.tz/pages/long-distance-train, uploaded June 2023; copy in `data/raw/tz/survey/`): Kigoma Sun + Thu out, Tue + Sun back; The Citizen: Dar - Kigoma raised to three a week |
| TRC metre gauge: Tabora - Mwanza | running, weekly (Dar - Mwanza Sun out, Tue back) | same timetable |
| TRC metre gauge: Kaliua - Mpanda | running, weekly (Dar - Mpanda Sun out, Tue back) | same |
| TRC metre gauge: Dar - Ruvu - (Link line) - Mruazi - Moshi - Arusha | running, twice a week (Mon + Fri out, Tue + Sat back) | same |
| Tanga - Mruazi (Tanga line's eastern end), Manyoni - Singida, Kilosa - Kidatu | **greyed or left off**: no passenger train in TRC's timetable | |
| Dar es Salaam commuter (TRC "treni ya mjini"): Dar (Stesheni) - Pugu on the Central line; Stesheni - Ubungo Maziwa | running weekdays (fares on trc.co.tz/pages/commuter-city-train; no timetable on the page) | trc.co.tz |
| TAZARA: Dar es Salaam (TAZARA) - Mbeya - Tunduma - Kapiri Mposhi, "Mukuba Express" | running, weekly each way (Fri from Dar, Tue from Kapiri) since 10 Feb 2026; Kilimanjaro ordinary cancelled | TAZARA notice (tazarasite.com "TAZARA Announces Resumption of Cross-Border Services"), Railways Africa, seat61 Zambia page (8 Apr 2026) |
| TAZARA Dar commuter on TAZARA track (Dar TAZARA - Mwakanga, 2 routes, 20.5 km, since 2012) | probably running; not confirmed for 2026 | en.wikipedia TAZARA |

The 2023 timetable is the latest TRC publishes; that the MGR trains still run in 2026 rests on
The Citizen's report of more Kigoma coaches and trips and TRC's 2026 news about MGR coaches. I
take them as running.

### Line list (TRC's own route diagrams give every station in order)

`data/raw/tz/survey/trc_routes.pdf` (trc.co.tz/site/files/trl-routes.pdf, 8 pages) lists every
station on each TRC route in order:

1. **Central line, Dar es Salaam - Kigoma** (Pugu, Mpiji, Soga, Ngeta, Ruvu, Kwala, Msua, Kidugalo, Ngerengere, Kinonko, Mikese, Kingolwira, Morogoro, Masimbu, Mkata, Kimamba, Kilosa, Munisigara, Mzaganza, Kidete, Godegode, Gulwe, Msagali, Igandu, Munase, Kikombo, Humwa, Dodoma, Zuzu, Kigwe, Bahi, Kintinku, Makutopora, Saranda, Manyoni, Aghondi, Itigi, Kitaraka, Kazikazi, Karangasi, Tura, Malongwe, Nyahua, Goweko, Igalula, Itulu, Tabora, Lulangulu, Mabama, Usoke, Urambo, Kaliua, Kombe, Usinge, Nguruka, Malagarasi, Ilunde, Uvinza, Lugufu, Kazuramimba, Kandaga, Kalenge, Luiche, Kigoma). Published 1,254 km. Cut at Ruvu, Tabora and Kaliua so each piece is one line between junctions, or keep it whole as TRC does.
2. **Mwanza route, Tabora - Mwanza** (Kakola, Nzubuka, Ipala, Bukene, Mahene, Gusule, Isaka, Luhumbo, Usule, Shinyanga, Songwe, Seke, Malampaka, Malya, Bukwimba, Mantare, Fela, Mwanza South, Mwanza). Published 378 km.
3. **Mpanda route, Kaliua - Mpanda** (Uyumbu, Nengeme, Lumbe, Usangu, Ugala River, Katumba, Mpanda). Published 210 km.
4. **Link line, Ruvu Junction - Mruazi** (Kwalaza, Usigwa, Kidomole, Wami, Mvave, Mkalamo, Gendagenda, Makinyumbi). About 190 km.
5. **Tanga line, Mruazi - Moshi - Arusha** (Mnyusi, Korogwe, Ngombezi, Maurui, Makuyuni, Mombo, Mazinde, Mkumbara, Mkomazi, Buiko, Hedaru, Makanya, Same, Lembeni, Kisangiro, Kahe, Moshi, Kikuletwa, Usa River, Arusha). About 350 km.
6. **Tanga - Mruazi** (Pongwe, Ngomeni, Muheza, Kihumwi): greyed (no passenger train).
7. **Manyoni - Singida**: greyed or not built (no service).
8. **SGR Dar es Salaam - Dodoma** (Dar es Salaam, Pugu, Soga, Ruvu, Ngerengere, Morogoro, Mkata, Kilosa, Kidete, Igandu, Dodoma). TRC: 444 km. Its own register line with `highspeed` to keep it off the metre-gauge ways (the SGR runs beside the MGR for long stretches).
9. **TAZARA, Dar es Salaam - Tunduma (border)**: 969.6 km (en.wikipedia TAZARA: Dar 0.0, Yombo 3.0, Ifakara 360.0, Mbeya 848.8, Tunduma 969.6). Stops: OSM stations on it; TAZARA's own station list is not online.

Expected: about 9 register lines, ~4,600 km, of which ~4,000 running (MGR ~2,420 incl. the Link and Tanga lines to Arusha, SGR 444, TAZARA ~970) and Tanga - Mruazi / Singida greyed.

### Sources

- Station order: TRC's route PDF (above), TRC's long-distance timetable image; SGR stations from TRC/Daily News.
- km: en.wikipedia "Central Line (Tanzania)", TAZARA article (chainage at a few stations); traced km otherwise (`no_chain`).
- Coordinates: OSM, 200 railway=station/halt, 190 named. Names should be checked against TRC's spellings (Makutupora/Makutopora).
- Timetables/GTFS: none. sgrticket.trc.co.tz and eticketing.trc.co.tz are booking front-ends; not crawled.
- Licence: OSM ODbL; TRC documents are public timetables.

### OSM quality (Overpass, 2026-10-08)

- 4,436 km of track, only 859 named (19%): "Central Line" 343, "Morogoro - Makutopora SGR" 247, "Dar es Salaam - Morogoro SGR" 192, a little TAZARA. Named track does not work; hand list traced by rinf.py.
- Relations: the query timed out twice on the public Overpass servers; read them from the extract.
- 4,436 km looks short of the real network (~2,700 km MGR in TRC's use plus Tanga/Singida, 970 TAZARA, ~700 SGR built): check after extract.py whether parts are tagged `disused`/`abandoned` or unmapped, especially the Mpanda and Tanga lines.
- Geofabrik: `africa/tanzania-latest.osm.pbf`, 673 MB (the biggest in the set).

### Recipe

Hand list traced by rinf.py (za/nafrica), TRC's route diagrams as the list; `listed_only` with
TRC's stations (OSM maps loops where nothing stops); shared eafrica reader. TAZARA a register line
each side of a border point at Tunduma/Nakonde (`borders.EXTRA`). The Mukuba Express a named
train. Dar commuter services as OSM lines if mapped.

### Open questions

- MGR running in 2026 rests on news, not a 2026 timetable. TRC's current long-distance image is from 2023.
- Does any MGR train still run Dar - Dodoma now the SGR does, or do Kigoma/Mwanza trains start from Dodoma? Some reports say the MGR long-distance trains still leave from Dar; assumed so.
- TAZARA Dar commuter: confirm.
- The order is read off TRC's snake diagram (labels alternate above and below the line); check it against OSM's station positions when tracing.

## Build (2026-10-08)

Built with `eafrica_register.py` (commands as ke_sources.md "Build"; no `--fill`). **Result**:
9 register lines, 4,004 km: running 7, 3,825 km (Central Line Dar es Salaam - Kigoma 1,255.1,
TAZARA Dar es Salaam - border 970.3, SGR Dar es Salaam - Dodoma 443.6, Mwanza Line 379.7, Tanga
Line Mruazi - Moshi - Arusha 372.9, Mpanda Line 210.6, Link Line Ruvu - Mruazi 192.7); greyed
2, 179 km (Singida Line 114.4, Tanga - Mruazi 64.9). check_model: SGR 443.6 of 444 (TRC),
Central 1,255.1 of 1,254 (en.WP), Mwanza 379.7 of 378, Mpanda 210.6 of 210, TAZARA 970.3 of
970.3 (chainage to the border).

Decisions:
- The SGR stops at its own stations only (listed_only; each pinned to the SGR station's
  coordinate, as OSM maps an SGR and a metre-gauge station of the same name at most stops);
  "New Central Railway Station" renamed "Dar es Salaam (Magufuli)".
- The metre-gauge Central Line starts at Kamata: OSM's metre-gauge main line ends there, 1 km
  short of the old central station, whose approach is yard track beside the SGR terminus (a
  trace from the old station runs over the SGR).
- Mruazi junction (38.61076, -5.24470, `--fork`): the Arusha trains run Ruvu - Link Line -
  Mruazi - Moshi - Arusha; Tanga - Mruazi is greyed (no passenger train in TRC's timetable).
- MGR lines take every OSM station as a stop. TAZARA's Dar commuter trains run on its first
  20 km, inside the TAZARA line.
- eafrica_register.build keeps the junction-ended sections of greyed lines too (nafrica marks
  running lines' only), or Tanga - Mruazi lost its last 25 km.
- OSM's Mpanda - Tabora route: rules/tz.py SKIP_ROUTES.
- The Tunduma - Nakonde border point XTZZM1: zm_sources.md "Build".
