# Angola (ao): sources

## Build (2026-10-08)

    python extract.py --region ao --pbf data/raw/angola-latest.osm.pbf --station-areas
    python wafrica_register.py --clip ao       # other countries' track, NOT_SERVICE routes,
                                               # track gaps joined, gauge breaks split
    python wafrica_register.py --fill ao
    python wafrica_register.py --convert ao
    python build_model.py --region ao --register wafrica_register:data/raw/rinf/ao
    python build_tiles.py --region ao
    python check_model.py --region ao

Built: **7 register lines, 2,529 km; 6 running (2,475 km), 1 greyed (54 km)**.

| line | km | check |
|---|---|---|
| Luanda (Bungo) – Baía | 36.0 | CFL 36, 1.00 |
| Baía – Malanje | 386.8 | CFL 386, 1.00 |
| Zenza do Itombe – Dondo (greyed) | 54.4 | CFL 46, 1.18 (fahrplancenter's list has the branch merged into the main line, so the 46 is uncertain) |
| Lobito – Benguela | 31.6 | none |
| Lobito – Huambo – Luau | 1,276.0 | ~1,344, 0.95 (the CFB was rebuilt 2006-2014 on new alignments; no newer figure found) |
| Namibe – Lubango | 245.3 | CFM 246, 1.00 |
| Lubango – Matala – Menongue | 499.3 | CFM 510, 0.98 |

Decisions: **the CFB express is counted to Luau**: the survey's 2026 headline "Expresso do
CFB retoma viagens entre Lobito e Luau" (angop.ao) says it runs there, so Lobito – Luau is one
running line. Lobito – Benguela's first 13.4 km (Lobito – Luongo) are the main line's
track; ownership gives them to Lobito – Benguela. The CFL's Dondo branch is greyed (no
evidence). OSM station names keep CFM's "(Km N)" suffix; twin records merged (Arimba,
Nanguluve, Tchavola, Sacomar). OSM's three main routes (the register lines under other
names) and the Matadi – Kinshasa route (DR Congo's) are dropped by --clip.

## Survey (2026-10-08)

Research only; nothing built. All three state railways carry passengers, Cape gauge
(1067 mm). The best-served country in this survey after Nigeria.

### What runs

| railway | lines | status |
|---|---|---|
| CFL, Caminho de Ferro de Luanda | Luanda (Bungo) – Viana – Baía suburban; Baía – Catete – Zenza do Itombe – N'dalatando – Malanje; Zenza – Dondo branch | suburban trains daily (four a day after COVID, two express suburban trips Bungo – Viana added Dec 2024: verangola.net 122024/42705); Malanje and N'dalatando trains weekly (the long-distance train out on a weekday, back Saturday; ciam.gov.ao/ao/noticia/548). **Running.** Dondo branch: no evidence; greyed. |
| CFB, Caminho de Ferro de Benguela (infrastructure under the Lobito Atlantic Railway concession since 2023; CFB keeps passengers) | Lobito – Benguela suburban; Lobito – Huambo – Kuito – Luena – Luau (1,344 km) | Express weekly: Lobito Monday 07:45 → Luena Tuesday, back Thursday (2026 search results citing angop.ao "CFB retoma circulação no troço Lobito-Luena" and "Expresso do CFB retoma viagens entre Lobito e Luau"; angop.ao refused connections, so the dates are the search snippets'). Lobito – Benguela suburban daily in the 2015 table; no evidence it stopped. **Running.** Whether the express goes past Luena to Luau weekly is open. |
| CFM, Caminho de Ferro de Moçâmedes | Namibe – Lubango – Matala – Menongue (756 km); Namibe – Bibala local (resumed Aug 2023); Lubango suburban; branches (Chiange/Jamba, Cassinga) | 600,000+ passengers in 2025; **Lubango – Menongue express relaunched April 2026, weekly** (out one day, back the next; stops Chicungo, Quipungo, Matala, Cuvango, Cuchi; verangola.net 042026/48362 and 48391). **Running.** Branches: freight/disused, not built. |

### Line lists with km

fahrplancenter.com transcribes each railway's timetable with km at every station:
- **CFL** (AngolaCFL.html, valid 30 July 2019): Luanda Bungo 0, Textang 2, Rotunda 4,
  Muceques 8, Filda 11, Gamek 14, Estalagem 17, Comarca 20, Viana 22, Capalanca 26,
  Entroncamento 30, Baía 36, Catete 64, Barraca 103, Zenza do Itombe 134, Cassoalala 154,
  Km 34 159, N'dalahui 161, Dondo 180 (branch), Luinha 185, Canhoca 208, Queta 222,
  N'dalatando 240, Lucala 288, Quizenga 314, Cambunze 335, Cacuso 349, Matete 368, Zanga 378,
  Lombe 398, Malanje 422. (The fetch merged the Dondo branch into the list; untangle at
  build time: Zenza – Dondo is the branch.)
- **CFB** (AngolaCFB.html, June-July 2015): Lobito – Benguela suburban and Lobito – Huambo
  with times; km to Luau ~1,344.
- **CFM/CFN** (AngolaCFN.html, valid 30 July 2019): Namibe 0, Caraculo 76, Lubango 246,
  Matala 424, Menongue 756, with the stations between.
These are the chainage for `chain` checks, as Algeria's SNTF km were.

### OSM (Overpass, 2026-10-08, bbox -18.1,11.6,-4.3,24.1; it also catches Kinshasa, Katanga and Namibia)

- route=train **5414666 "Train: Luanda - Malanje"**, **8477956 "Lobito - Luau"** (CFB),
  **8477951 "Namibe - Lubango"** (CFM), **12476585 "Serviço Suburbano Lubango"** (CFM).
  route=railway 2148314 "Caminho de Ferro de Luanda", 2796985 "Zenza Do Itombe to Dondo",
  2152051 "Caminho de Ferro de Moçâmedes (CFM)", 1125202 (CFB, unnamed, network CFB),
  7115613 "CFB 2nd Alignment Benguela-Cubal", and old alignments (7115614, 12266756: the CFB
  was rebuilt on new alignments in 2006-2014; the trace must not use the old ones, which
  should be tagged abandoned, but check).
- No route for the Luanda suburban trains, Lobito – Benguela or Lubango – Menongue.
- Track: 1,856 ways in the bbox, 588 named: "Caminho de Ferro de Moçâmedes (CFM)" 285,
  "Caminho(s) de Ferro de Luanda" 67, "Caminhos de Ferro de Benguela" 24, "Ramal da
  Jamba-Chamutete (CFM)" 32. Gauge on 68% (1067: 1,251). 299 stations, 278 named.
- So: a trace over named or Cape gauge track, stops from the four routes plus the
  fahrplancenter lists.

### Timetables / GTFS

None in Transitous. The fahrplancenter tables above (old, but station lists and km do not
change); the railways' own sites: cflep.co.ao (CFL).

### Recipe

Hand line list traced by rinf.py in the shared reader (see `ng_sources.md`), stations and km
from fahrplancenter, like Algeria in `nafrica_register.py`. Lines, roughly:
CFL Bungo – Baía (suburban), Baía – Malanje, Zenza – Dondo (greyed); CFB Lobito – Benguela,
Lobito – Luena, Luena – Luau (running if the express reaches Luau); CFM Namibe – Lubango,
Lubango – Menongue. About 8 lines, ~2,600 km. Extract: `africa/angola-latest.osm.pbf`,
**81 MB**.

The CFB continues into DR Congo (Dilolo); no passenger train crosses. The new Luanda
airport (AIAAN) rail link: under construction, not built.

### Licences

OSM ODbL; timetable facts.

### Open questions

- CFB express beyond Luena to Luau in 2026.
- CFL Dondo branch and CFM's Bibala local: still running?
