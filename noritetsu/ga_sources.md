# Gabon (ga): sources

## Build (2026-10-08)

    python extract.py --region ga --pbf data/raw/gabon-latest.osm.pbf --station-areas
    python wafrica_register.py --clip ga       # other countries' track, NOT_SERVICE routes,
                                               # track gaps joined, gauge breaks split
    python wafrica_register.py --fill ga
    python wafrica_register.py --convert ga
    python build_model.py --region ga --register wafrica_register:data/raw/rinf/ga
    python build_tiles.py --region ga
    python check_model.py --region ga

Built: **1 register line, 646 km, running**: Transgabonais Owendo (Owendo Viré) –
Franceville, all 23 of SETRAG's stations as stops (every one is an OSM station on the line).
Check: 646.4 against SETRAG's 670 (0.96; built platform to platform from Owendo Viré). No OSM
route relation for the line exists; COMILOG's (in Congo) is cut by --clip.

## Survey (2026-10-08)

Research only; nothing built. One line runs: the Transgabonais.

### What runs

- **Owendo (Libreville) – Franceville, the Transgabonais** (SETRAG, an Eramet subsidiary,
  standard gauge, ~648-670 km). Four passenger trains a week each way: the "Trans-Ogooué"
  express (Tuesday and Saturday per seat61, updated 2026) and the "L'Équateur" omnibus
  (Thursday and Sunday), overnight. seat61.com/Gabon.htm; setrag.ga and its app have the
  current times. Running in 2026; no suspension found.
- Freight (manganese from Moanda, timber) on the same line. No other passenger railway.

### Line list with km

fahrplancenter.com/Gabon.html (SETRAG timetable valid 1 May 2019, with km): Owendo 0, Ntoum,
Andem, M'bel 85, Oyan, Abanga, **Ndjolé 183**, Alembé, Otoumbi, Bissouma, Ayem, Lopé,
Offoué, **Booué 340**, Ivindo, Mouyabi, Milolé, **Lastourville 485**, Doumé, Lifouta,
Mboungou 595, **Moanda 625**, **Franceville 670**. The page lists km for every station
(the fetch summarised only some); read the full table at build time. 23 stations. Express
trains skip some of them; the omnibus calls at all.

### OSM (Overpass, 2026-10-08, bbox -4.0,8.6,2.4,14.6)

- **No route=train for the Transgabonais.** The only rail route relations are COMILOG's
  (route=railway 7168260 "Ligne de la COMILOG", route=train 8359854 "COMILOG Mbinda - Mont
  Belo", both in Congo, Cape gauge, not running).
- Track: 393 `railway=rail` ways, 127 named; **"Transgabonais" on 87 ways**, gauge tagged on
  nearly all (1435: 353). 48 station objects, all named.
- So stations come from the hand list (SETRAG's table) matched to OSM's 48 stations, and the
  trace runs over the named standard gauge track.

### Timetables / GTFS

None in Transitous. SETRAG's site (www.setrag.ga) and app; seat61. The 2019 table above.

### Recipe

One hand line traced by rinf.py in the shared reader (see `ng_sources.md`): `Owendo –
Franceville`, every station from SETRAG's table with its km (a `chain`, so check_model checks
every section). ~670 km, 23 stations. Express vs omnibus are services, not lines.

Extract: `africa/gabon-latest.osm.pbf`, **24.3 MB**.

### Licences

OSM ODbL; SETRAG timetable figures are facts.

### Open questions

- Owendo to Libreville proper: the passenger station is at Owendo; no city line.
