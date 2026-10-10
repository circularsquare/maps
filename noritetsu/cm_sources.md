# Cameroon (cm): sources

## Build (2026-10-08)

    python extract.py --region cm --pbf data/raw/cameroon-latest.osm.pbf --station-areas
    python wafrica_register.py --clip cm       # other countries' track, NOT_SERVICE routes,
                                               # track gaps joined, gauge breaks split
    python wafrica_register.py --fill cm
    python wafrica_register.py --convert cm
    python build_model.py --region cm --register wafrica_register:data/raw/rinf/cm
    python build_tiles.py --region cm
    python check_model.py --region cm

Built: **3 register lines, 979 km, all running**: Douala – Yaoundé 259.4 (263, 0.99),
Yaoundé – Ngaoundéré 621.9 (622, 1.00), Douala – Mbanga – Kumba 98.1 (survey ~96). Stops:
every OSM station on the traced line (the omnibus calls at all). Douala – Kumba starts at
Douala Bessengué: OSM has no Bonabéri station across the Wouri, where seat61 says the trains
leave, so the line may run ~5 km further than the trains. Twin records merged
(`merge_twins`): Binguela, Tête d'Éléphant. OSM's two Camrail routes (the register lines
under other names) and an unnamed one-way route are dropped by --clip.

## Survey (2026-10-08)

Research only; nothing built. Camrail (metre gauge, AGL/Bolloré-era concession) runs three
routes.

### What runs (seat61.com/Cameroon.htm, updated 1 Jan 2026)

| route | km | trains |
|---|---|---|
| Douala – Yaoundé (via Edéa, Messondo, Eséka, Makak, Otélé, Ngoumou) | 263 | Express 185/186 Mon-Fri (since 1 July 2021), ordinary 181/182 daily, omnibus 103/104 Mon, Wed, Fri |
| Yaoundé – Belabo – Ngaoundéré (the Transcamerounais) | 622 (seat61 gives 667 by train) | night train 191/192 daily, couchettes |
| Douala (Bonabéri) – Mbanga – Kumba | 66 + ~30 | Douala – Mbanga twice a week; Mbanga – Kumba three local trains a day |

Not running: Mbanga – Nkongsamba and Otélé – Ngoumou – Mbalmayo ("service supprimé" already
in Camrail's 2014 table, fahrplancenter.com/KamerunHoraires04.html; nothing since). Douala
has no suburban train. The Kumba branch lies in the anglophone South-West Region; seat61
lists it as running in 2026, and I found no suspension notice, so it counts.

Decision: four register lines, all running: Douala – Yaoundé, Yaoundé – Ngaoundéré,
Douala – Mbanga, Mbanga – Kumba (split at Mbanga so the twice-weekly and the daily parts are
separate lines; or one line, since both pass "about weekly"; I would keep one line Douala –
Kumba). Mbanga – Nkongsamba and Otélé – Mbalmayo not built.

### Line list

- No km per station published (Camrail's 2014 table has times only; camrail.cm has only the
  MyCamrail booking site). Station names in order are in that 2014 table
  (fahrplancenter.com/KamerunHoraires04.html) and in OSM.
- Published line lengths for checks: Douala – Yaoundé 263 km (seat61, Camrail), Yaoundé –
  Ngaoundéré 622 km (Transcamerounais, en/fr.wikipedia), Douala – Kumba ~ 96 (Mbanga – Kumba
  29 km, fr.wikipedia "Ligne de Douala à Kumba" if it has it).

### OSM (Overpass, 2026-10-08, bbox 2.8,9.2,7.6,14.0; route relations and named track only,
the full query timed out)

- route=train **8503524 "Douala-Yaoundé"** and **8503527 "Yaoundé-Ngaoundéré"** (ref
  191-192), both Camrail; route=railway 6905289 "Mbanga - Kumba mixed passenger and freight
  train"; 11701793 an unnamed route=train; 3517190 "Ancienne Voie Ferrée Douala Yaoundé"
  (the old alignment; the trace must avoid it, check it is tagged abandoned/disused).
- Track is hardly named (9 ways "Yaoundé-Ngaoundéré", single ways "Ligne de l'Ouest :
  Douala-Mbanga", "Pont Ferroviaire du Wouri"). The trace runs on unnamed metre gauge track;
  station counts were not obtained.
- Nothing for Douala – Mbanga as a train route: stops from the 2014 table's names matched to
  OSM stations.

### Timetables / GTFS

None in Transitous. seat61 (above) is the best current timetable; Camrail's own
(www.camrail.cm, MyCamrail) has no public timetable page.

### Recipe

Hand line list traced by rinf.py in the shared reader (see `ng_sources.md`), stops from
OSM stations along the trace (`osm_stops`) or from the 2014 table's station names. ~4 lines,
~985 km. Extract: `africa/cameroon-latest.osm.pbf`, **213 MB**.

### Licences

OSM ODbL; timetable facts.

### Open questions

- Express and omnibus stops: Express calls only at Edéa, Messondo, Eséka, Makak, Ngoumou
  (237online.com). Station list = omnibus's.
- Kumba: running in practice given the anglophone crisis? seat61 says yes.
