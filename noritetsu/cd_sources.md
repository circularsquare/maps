# DR Congo (cd): sources

## Build (2026-10-08)

    python extract.py --region cd --pbf data/raw/congo-democratic-republic-latest.osm.pbf --station-areas
    python wafrica_register.py --clip cd       # other countries' track, NOT_SERVICE routes,
                                               # track gaps joined, gauge breaks split
    python wafrica_register.py --fill cd
    python wafrica_register.py --convert cd
    python build_model.py --region cd --register wafrica_register:data/raw/rinf/cd
    python build_tiles.py --region cd
    python check_model.py --region cd

Built: **2 register lines, 377 km; 1 running (20 km), 1 greyed (356 km)**.
- Train urbain : Gare Centrale – Ndjili, running, 20.3 km: Kinshasa Est (taken as the Gare
  Centrale, Gombe), Ndolo, Limete Amicongo, Masina Mapela, Ndjili Aéroport. The press says
  "nearly 25 km" Tshenke – Gare Centrale; OSM's track ends at Ndjili Aéroport and has no
  Tshenke, so 0.81 against 25 is the press's rounding or track OSM lacks.
- Kinshasa – Matadi, greyed (suspended about April 2026), 356 km to Kenge: OSM's track
  has a ~12 km hole between Kenge and Matadi (two dead ends at 13.627,-5.829 and
  13.515,-5.860), so the last section had no trace and the line stops at Kenge.
Every other DRC route relation is dropped by --clip (`NOT_SERVICE`): the SNCC's, below the
weekly bar, Kisangani – Ubundu and Lubumbashi – Sakania (no service reported), the Kasangulu
urban line and the Ango Ango branch. The SNCC network is not built (survey).

## Survey (2026-10-08)

Research only; nothing built. Three networks; only Kinshasa's has a train at least weekly in
2026, and even that is fragile.

### What runs

| line | operator | gauge | status (freshest evidence) |
|---|---|---|---|
| Kinshasa urban train, Tshenke (Masina) – Gare Centrale (Gombe), "nearly ten stops", ~1 h 20 | ONATRA / SCTP | 1067 | **relaunched 19 Aug 2026** after 15 years; one rotation a day (in 8:05 arr., out 16:30); derailed 3 Sept 2026, running again the same evening (radiookapi.net 2026/08/19 and 2026/09/03; congoquotidien.com 2026/08/19; eco24.cd 2026/07/01 on the Gare Centrale – N'djili airport plan). **Running.** |
| Kinshasa – Matadi (express) | ONATRA / SCTP | 1067 | reopened 5 Sept 2025 after five years (logistafrica.com); cut Nov 2025 by storm damage at Kimwenza – Lemba; relaunched March 2026 with a second weekly round trip from 15 March (Sat Kinshasa 7:30, Sun Matadi 7:30; infos27.cd 2026/03/05, bankable.africa fares); **suspended ~April 2026 for railcar maintenance**, "resumption very soon" (radiookapi.net 2026/06/14, bankable.africa 21 June 2026). No resumption found by 8 Oct 2026. **Greyed (suspended)** unless a resumption turns up. |
| SNCC network (Lubumbashi – Kananga, – Mwene Ditu, – Kindu, – Kalemie, – Dilolo; Kananga – Ilebo) | SNCC | 1067 | named trains (New Express Colombe, Diamant Béton, Kambelembele, Hirondelle) **once or twice a month** per route (snccsa.com "Programme de trains"; acp.cd). Below the weekly bar: **not counted**. |
| CFU Bumba – Isiro (Uele, 600 mm) | CFU | 600 | derelict; no service. |
| Kinshasa metro (MetroKin), Kinshasa – N'djili airport line | - | - | planned / unfunded. |

Decision: register line **Kinshasa Gare Centrale – Tshenke (urban), running**; **Kinshasa –
Matadi, greyed (suspended)**; SNCC lines not built (or, if Anita wants the network visible,
greyed as `suspended`, as Shosholoza Meyl in za). DRC is the weakest "yes" here after
Burkina Faso.

### Line list with km

- Kinshasa – Matadi: 366 km (Matadi – Léopoldville railway; en.wikipedia "Matadi–Kinshasa
  Railway"). Stations in order there and in fr.wikipedia "Chemin de fer Matadi-Kinshasa".
- Urban line: the 2026 service runs east from Gare Centrale (Gombe) to Tshenke (Masina), on
  the branch toward N'djili airport; about 20-25 km by the timings (1 h 20 at 20-25 km/h).
  Stops not published ("nearly ten"); OSM's along the trace.

### OSM (Overpass, 2026-10-08)

Kinshasa and Bas-Congo fell inside Congo-Brazzaville's bbox (-5.1,11.1,3.8,18.7):
- route=train **401053 "Matadi-Kinshasa"** (ONATRA), **1281981 "Ligne urbaine de
  Kasangulu"**, **1281986 "Ligne urbaine de l'Aéroport"** (ONATRA; the airport branch the 2026
  urban train runs on, toward N'djili), route=railway 1281991 "Ligne urbaine de Kintambo",
  1548333 "Chemins de fer Vicinaux du Mayumbe" (closed). So the Kinshasa lines have routes
  with stops already.
- Track around Kinshasa is not named; the trace runs on Cape gauge track ("Pont N'djili" is
  tagged).
Angola's bbox caught the south: route=train 8480451 "Lubumbashi - Ilebo" (SNCC),
route=railway 401455 "Tenke-Dilolo", 452929 "Kamina-Ilebo" (named track "Kamina - Ilebo" on
74 ways). The whole-country query timed out three times; the SNCC network was not probed
further, since nothing there is counted.

### Timetables / GTFS

None (no `cd_` in Transitous).

### Recipe

Hand list traced by rinf.py in the shared reader (see `ng_sources.md`). 2 lines, ~390 km
(~25 running). Extract: `africa/congo-democratic-republic-latest.osm.pbf`, **396 MB** (big
for so little track; a `osmium extract` by bbox of Kinshasa – Matadi would do, if Anita
prefers not to keep the whole file).

### Licences

OSM ODbL; press facts.

### Open questions

- Kinshasa – Matadi after June 2026: resumed? (radiookapi.net, mediacongo.net, actualite.cd).
- The urban train's exact stops and whether it reaches N'djili airport yet.
