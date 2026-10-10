# Republic of the Congo (cg): sources

## Build (2026-10-08)

    python extract.py --region cg --pbf data/raw/congo-brazzaville-latest.osm.pbf --station-areas
    python wafrica_register.py --clip cg       # other countries' track, NOT_SERVICE routes,
                                               # track gaps joined, gauge breaks split
    python wafrica_register.py --fill cg
    python wafrica_register.py --convert cg
    python build_model.py --region cg --register wafrica_register:data/raw/rinf/cg
    python build_tiles.py --region cg
    python check_model.py --region cg

Built: **1 register line, 508 km, running**: Pointe-Noire – Brazzaville, 46 stops
(every OSM station on the traced line, the same as OSM's Gazelle route lists). Check: 508.0
against CFCO's PK 512 (0.99). The list pins the trace to the Mayombe realignment (Les Saras –
Mvouti – Les Bandas). Dropped by --clip (`NOT_SERVICE`): the Gazelle route (the register
line under another name), CFCO Bilinga – Dolisie (a local over the old Mayombe line, not
reported since 2023) and COMILOG Mbinda – Mont Belo.

## Survey (2026-10-08)

Research only; nothing built. One line runs: the Congo-Océan (CFCO) main line.

### What runs

- **Pointe-Noire – Brazzaville, "La Gazelle"** (CFCO, Cape gauge 1067 mm, 512 km by CFCO's
  PK, 502-510 km in other sources). Passenger service came back in **May 2023** after seven
  years (africanews.com/2023/05/17/rail-connecting-congos-brazzaville-pointe-noire-resumes-passenger-service);
  travel guides list it as running in 2026, **weekly each way** (Pointe-Noire and Brazzaville
  departures on fixed days; one 2026 guide gives Monday and Friday). 9 intermediate stops
  (en.wikipedia "La Gazelle"). No suspension reported in 2025-2026 by anything found. The
  2017 timetable (fahrplancenter.com/CongoCFCO04.html, valid Dec 2017) also had the "Océan"
  express and mixed trains, which are not reported since.
- **Not running**: the Mont-Belo – Mbinda branch (285 km, ex-COMILOG; no passenger train
  reported), Pointe-Noire suburban trains (nothing found).

Decision: the whole main line is one running register line (a weekly train meets the "about
weekly" rule); the Mbinda branch is not built.

### Line list with km

fr.wikipedia "Chemin de fer Congo-Océan", the line diagram (PK from Pointe-Noire):
Pointe-Noire 0, Tié Tié, Ngondji 18, Hinda 35, Tchitondi 57, Bilala 71, Bilinga 76,
Mfoubou 89, Les Saras 110, Mvouti 127, Les Bandas 148, **Dolisie 167**, Moubotsi 190,
**Mont-Bélo 200** (junction for Mbinda), Loudima 219, **Nkayi 248**, Bodissa 264, **Madingou
278**, Bouansa 304, Loutété 318, Kimbédi 331, Loulombo 346, Kinkembo 361, **Mindouli 384**,
Missafou, Matoumbou 430, Madzia, Kibouende 457, Goma Tsé-Tsé 484, Mfilou 501, **Brazzaville
512**. This is the register's own chainage (`km_official` / `chain`) for a check of every
section. La Gazelle stops at about 9 of them (the bold ones, plus a few); which ones is an
open question (OSM's route relation, if any, or every OSM station on the line).

### OSM (Overpass, 2026-10-08, bbox -5.1,11.1,3.8,18.7; it also catches Kinshasa and Gabon)

- **route=train 8360001 "La Gazelle Pointe Noire - Brazzaville"** (CFCO): its stops are the
  train's stops. Also route=train 8360346 "CFCO Bilinga - Dolisie" (a local train; no
  evidence it runs now) and 8359854 "COMILOG Mbinda - Mont Belo" (not running);
  route=railway 1789163 "Ligne de Chemin de Fer Congo-Océan", 8358716 "Réalignement Chemin
  de Fer Congo-Océan" (the Mayombe realignment), 7168260 "Ligne de la COMILOG".
- Track: **"Congo - Océan" names 164 ways**; 1067 gauge on 304 ways in the bbox. 122 station
  objects, 118 named (with Kinshasa's).
- Good enough for a trace over the named track with the Gazelle's stops.

### Timetables / GTFS

None in Transitous; nothing found on cfco.cg. The 2017 fahrplancenter table has the stops and
km of each train of the time.

### Recipe

One hand line traced by rinf.py in the shared reader (see `ng_sources.md`): `Pointe-Noire –
Brazzaville`, all stations from the PK list (so the chain check works on every section), or
just La Gazelle's stops with the rest as `junction`-free passing points. ~512 km.

Extract: `africa/congo-brazzaville-latest.osm.pbf`, **31.1 MB**. `--clip` matters little:
the line stays inside the country (Brazzaville sits on the DRC border; Kinshasa's lines are
across the river, no rail link).

### Licences

OSM ODbL; Wikipedia CC BY-SA (only the PK numbers, facts).

### Open questions

- La Gazelle's actual stops and days in 2026 (CFCO publishes nothing online; seat61 has no
  Congo page any more).
