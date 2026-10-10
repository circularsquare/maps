# Dominican Republic register sources

Built 2026-10-08 (latam agent). The survey below is the research record.

## Build (2026-10-08)

No register: the two metro lines are OSM lines (`build_model.py --region do`, no
`--register`, as Qatar). `latam_register.py --clip do` takes Haiti out of the shared extract
and drops the Santiago monorail's 40 ways named "(en construcción)" that are tagged
railway=monorail.

| line | built km | stations | published | ratio |
|---|---|---|---|---|
| Línea 1 (OSM) | 13.0 | 16 | 14.5 (en.WP) | 0.89 (tails) |
| Línea 2 with 2C (OSM) | 20.2 | 23 | 21 | 0.96 |

2 lines, 33 km, 39 stations, all running. The sugar-estate railways (Central Romana) are
drawn as track, uncounted. A 0.2 km unnamed monorail crossover is left in the extract.

Commands:

    python extract.py --region do --pbf data/raw/haiti-and-domrep-latest.osm.pbf
    python latam_register.py --clip do
    python build_model.py --region do
    python build_tiles.py --region do
    python check_model.py --region do

## Survey (2026-10-08)

### What runs

**Metro de Santo Domingo** (OPRET), two lines, daily:

| line | termini | stations | km | note |
|---|---|---|---|---|
| Línea 1 | Mamá Tingó - Centro de los Héroes | 16 | ~14.5 | |
| Línea 2 (2A + 2B + 2C) | Pablo Adón Guzmán (Los Alcarrizos) - Concepción Bona | 14 + 4 + 5 = 23 | 21 (en.wikipedia: system 35.5 km as of April 2026) | 2C (Los Alcarrizos extension, 5 stations: Pedro Martínez, Franklin Mieses Burgos, 27 de Febrero, Freddy Gatón Arce, Pablo Adón Guzmán) opened 24 Feb 2026, public from 25 Feb (OPRET; acento.com.do, eldinero.com.do) |

Not rail: the Teleférico de Santo Domingo (cable car). Not open: the Monorriel de Santiago (under
construction), Línea 1's extension. No other passenger trains on the island (Haiti has none;
the Haiti and DR extract is one file).

### Sources

- OPRET (opret.gob.do), es.wikipedia "Metro de Santo Domingo" for station lists and lengths.
- No feed in the Mobility Database for the metro (only Santiago de los Caballeros buses).
- **OSM**: Overpass summary in `data/raw/do/survey/osm_summary.txt` (below).
- Geofabrik `central-america/haiti-and-domrep-latest.osm.pbf`, 84 MB (one extract for both
  countries; `--clip` to the DR outline, nothing on Haiti's side).

### OSM (Overpass, 2026-10-08; `data/raw/do/survey/osm_summary.txt`)

- Both lines have PTv2 routes in both directions (operator OPRET, ref 1 blue / 2 red, no
  route_masters). Línea 2's already run to **Pablo Adón Guzmán**, so 2C is mapped; track
  "Línea 2 del Metro de Santo Domingo (Etapa 2C)" 14.7 km of ways.
- Línea 1's extension "Etapa 1B" is railway=construction (left out by itself).
- **Trap**: the Monorriel de Santiago is partly tagged `railway=monorail` though its names say
  "(en construcción)" / "(2da etapa en construcción)": 12.6 km. No route relation, so no line
  builds, but build_tiles would draw it as track; a `--clip`-style rule should drop ways whose
  name says "en construcción" (cl_register and mx_register already do this for routes).
- The rest is sugar-estate railways (La Romana's Central Romana network, ~360 km of
  rail usage=main/branch, "Ferrocarril La Romana - Batey Romanita", "Ferrocarril Guaymate",
  bateyes): freight, no passengers, not register track.

### Recipe

OSM route relations as lines (US/UK/BR metro rule), no register reader, or a two-line hand list
in a shared small-countries reader with Ecuador, Panama and Puerto Rico if the routes need it.

Expected: 2 lines, 35.5 km, 39 stations.

### Open questions

- None blocking: OSM's routes already include 2C. Drop the Santiago monorail's mistagged ways.
