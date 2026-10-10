# Costa Rica register sources

Built 2026-10-08 (latam agent). The survey below is the research record.

## Build (2026-10-08)

`latam_register.py` (ar_register's code, cl_register's pattern; one module for the eleven Latin
American regions). Not my_register's GTFS-stop pattern after all: OSM's 18 INCOFER routes run
over every stretch but Cartago - Plaza Paraíso, so ar's recipe gives the track, and INCOFER's
feed gives each line's stop list (`lists`) and the station names.

| line | built km | stations | INCOFER shape km | ratio |
|---|---|---|---|---|
| L1 San José - Heredia - Alajuela (Atlántico - Alajuela) | 20.8 | 11 | 20.83 | 1.00 |
| L2 San José - Cartago (Atlántico - Plaza Paraíso) | 24.3 | 10 | 24.62 | 0.99 |
| L3 Curridabat - San José - Pavas - Belén (CFIA - Belén) | 21.9 | 18 | 21.9 | 1.00 |

3 register lines, 67 km (Atlántico - CFIA counted once, as L2's: ownership), 34 stations (all
of the feed's), all running. No OSM lines left: OSM's three route_masters are the twins.

Calls:
- **Stations are named as INCOFER's feed names them.** OSM's records carry older or longer
  names: "Fátima" is the feed's Bulevar Aeropuerto (10 m apart), "Hospital Alajuela" is
  Alajuela, "Flores" is San Joaquín, "Tubo Tico (AyA)" is AyA, "Cuba" Barrio Cuba, and so on
  (`CR_ALIAS`).
- **Cartago - Plaza Paraíso** (10 of L2's weekday trips) is an `extent` over the track OSM
  names "Tren Interurbano | Cartago - Paraíso": OSM's routes stop at Los Ángeles.
- **L3 runs CFIA - Belén**: 2 of its 20 trips start at CFIA, over L2's track (a shared extent,
  as ar's Lobos); ownership gives the track to L2. **L1 is Atlántico - Alajuela**: its 7 trips
  a day on to UCR and U Latina run over the same L2 track, and a shared extent there found no
  station (kr places a listed station only on the line's own ways); left out, and OSM's two
  Heredia - U. Latina routes clipped so OSM's L1 is the register line's twin. The stretch is
  L2's and L3's either way.
- The clip renames OSM's route_masters to INCOFER's names (L1, L2, L3), so build_model finds
  each its register line's twin and drops it.
- Monday to Friday only: scheduled, so all three count.

Commands:

    python extract.py --region cr --pbf data/raw/costa-rica-latest.osm.pbf     # ~5 s
    python latam_register.py --clip cr      # after every extract
    python build_model.py --region cr --register latam_register:data/raw/cr    # ~25 s
    python build_tiles.py --region cr
    python check_model.py --region cr

## Survey (2026-10-08)

### What runs

INCOFER's Tren Interurbano in the Central Valley, Monday to Friday only (the feed's one service
is `entresemana`; no weekend trains). Three lines in INCOFER's own feed (August 2026, valid
1 Jan - 31 Dec 2026):

| line | stations in order | trips a weekday | shape km |
|---|---|---|---|
| L1 San José - Heredia - Alajuela | Estación Atlántico, Calle Blancos, Colima, Santa Rosa, Miraflores, Estación Heredia, San Francisco, San Joaquín, Río Segundo, Bulevar Aeropuerto, Alajuela (11) | 46 | 20.9 |
| L2 San José - Cartago | Estación Atlántico, UCR, U Latina, CFIA, UACA, Tres Ríos, Estación Cartago, Los Ángeles, Oreamuno, Plaza Paraíso (10) | 28 | 24.6 |
| L3 Curridabat - San José - Pavas - Belén | CFIA, U Latina, UCR, Estación Atlántico, La Corte, Plaza Víquez, Estación Pacífico, Barrio Cuba, Contraloría, La Salle, AyA, Jack's, Pavas Centro, Pecosa, Demasa, Metrópoli, (Pedregal, Estación Belén on some trips) | 20 | 21.9 to Metrópoli |

L2 and L3 share CFIA - Estación Atlántico. Distinct route-km about 60-65 (Atlántico - Alajuela
21, Atlántico - Paraíso 25, Atlántico - Belén ~19). 34 stops in the feed, every one with a
coordinate.

Not counted: the Limón-area trains (none running), the Caldera line (freight / not passenger),
"Tren Turístico Arenal" (a 2.5 km narrow-gauge attraction track in OSM, no timetable found),
any heritage excursion.

### Sources

- **INCOFER GTFS**, Mobility Database `mdb-3480`, published by SIMOVI (Universidad de Costa
  Rica) as INCOFER's official feed. Producer URL `https://feeds.simovi.org/incofer/schedule/feed.zip`
  answered 404 on 2026-10-08; the Mobility Database copy
  `https://files.mobilitydatabase.org/mdb-3480/mdb-3480-202608172132/mdb-3480-202608172132.zip`
  (72 KB, feed_version v.20260727) is saved as
  `data/raw/cr/survey/incofer_gtfs_20260817.zip`. Routes, stops, shapes, colours (L1 CE1126,
  L2 002B7F, L3 245C02). Licence: none stated in the feed or the catalogue; treat as
  attribution to INCOFER / SIMOVI. Lost the Mobility Database "seal of reliability" on
  3 Sep 2026 (the producer URL going dead, probably).
- **OSM**: see the OSM section below.
- Geofabrik `central-america/costa-rica-latest.osm.pbf`, 37.3 MB.

### OSM (Overpass, 2026-10-08; `data/raw/cr/survey/osm_summary.txt`)

- **18 PTv2 train routes** (operator Incofer, network "Tren Urbano"), one per direction and
  pattern, no route_masters. Their `ref` is not INCOFER's current line number: OSM ref 1 =
  Pacífico / CFIA - Metrópoli - Belén (the feed's L3), ref 2 = Heredia / Alajuela (L1), ref 4 =
  Cartago (L2). Some names are stale ("Heredia - U. Latina", "Los Ángeles - Estación
  Atlántico"; the feed now runs Cartago trains to Plaza Paraíso).
- **Named track per corridor**: "Tren Interurbano | San José - Heredia" 9.6 km, "| Alajuela -
  Heredia" 11.2, "| San José - Cartago" 20.5, "| Cartago - Paraíso" 6.6, "| San José - San
  Antonio" 14.3, "| San Antonio - Ciruelas" 4.5 (+ 3.7 disused), "| Atlántico - Pacífico" 3.0.
  Plus "Ferrocarril al Atlántico" 109 km in use (Limón freight / part of the Heredia line) and
  "Ramal La Estrella" 52.9 km (banana freight). 232 km of rail usage=main in all; 290 km
  disused (Ferrocarril al Pacífico, the old ramales).
- So kr_register's named-track recipe would also work (the corridor names are the passenger
  lines'), but the GTFS gives INCOFER's actual line split and stops.

### Recipe

GTFS stop lists laid on OSM track, as `my_register.py` does with KTM's and Prasarana's feeds
(the closest pattern; or nafrica's hand list through rinf.py with the GTFS supplying the list).
Three register lines named as INCOFER names them (L1, L2, L3, with the long names), stations
from the feed. L3's Belén tail (Metrópoli - Pedregal - Belén) comes from the trips that run
there. A timetable feed exists, so `gtfs_served` could check sections, though with three
lines all between stops there is little for it to decide.

Expected: 3 register lines, ~62 km, 34 stations.

### Open questions

- The producer URL is dead; whether SIMOVI moved the feed or stopped (the copy is valid to
  31 Dec 2026, so fine for a first build).
- L3's Belén trips: how many a day (the longest trip in the feed stops at Metrópoli).
