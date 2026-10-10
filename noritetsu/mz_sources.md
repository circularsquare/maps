# Mozambique (mz): sources

## Survey (2026-10-08)

Research only, nothing built. By the eafrica survey agent.

### What runs (freshest evidence)

CFM runs three separate systems; the north is run by the concessionaire CDN.

| service | status | evidence |
|---|---|---|
| CFM Sul: Maputo - Ressano Garcia (88 km) | running, daily (07:45 weekdays from Maputo, 12:05 back) | seat61.com/Mozambique.htm (1 Jan 2026); CFM "Transporte de passageiros" page ("daily services" on the Goba, Ressano Garcia and Limpopo lines) |
| CFM Sul: Maputo - Goba (the Goba line, ~74 km to the Eswatini border) | running daily per CFM; how far the daily train goes (Boane or Goba) not stated | cfm.co.mz/transporte-de-passageiros |
| CFM Sul: Maputo - Chicualacuala (Limpopo line, ~530 km) | running, twice a week (one with a Zimbabwe connection) | seat61; CFM |
| CFM Sul: Maputo commuter ("inter-urbano") to Matola-Gare, Boane, Marracuene | running | CFM; seat61 mentions Matola, Machava, Marracuene, Manhiça |
| CFM Centro: Beira - Machipanda (Zimbabwe border), ~317 km | running, twice a week each way since 5 Dec 2023 (Mon + Sat out, Tue + Sun back) | AIM news 23 Nov 2023; CFM lists it in 2026 |
| CFM Centro: Beira - Moatize (Sena line via Dondo, Inhamitanga, Mutarara/Dona Ana) | running, twice a week (Tue 09:00, Sat 18:30 from Beira) | seat61 Jan 2026; CFM lists "Beira to Moatize" |
| CDN: Nampula - Cuamba (Nacala corridor, ~350 km) | running, twice a week (Tue + Sat 05:00) | seat61 Jan 2026 |
| Cuamba - Lichinga, Cuamba - Entre Lagos (Malawi), Nacala - Nampula, Marromeu branch | **not built**: no passenger service found | |
| Maputo - Johannesburg through train | none (passengers change at Ressano Garcia/Komatipoort) | za_sources.md |

### Line list

CFM's line pages (cfm.co.mz/linhas-de-ressano-garcia/, /linha-de-goba/, /linha-do-limpopo/,
/linha-de-machipanda/, /linha-de-sena/, /linha-nacala-cuamba/, /linha-cuamba-lichinga/) give
each line's length and station count but no station list:

1. **Linha de Ressano Garcia**, Maputo - Machava - Matola - Moamba - Ressano Garcia: 88 km, 11 stations and 2 halts (CFM).
2. **Linha de Goba**, Machava (or Matola) - Boane - Goba: ~70 km. Runs to the Eswatini border; no train crosses.
3. **Linha do Limpopo**, Maputo - Marracuene - Manhiça - Xinavane? - Chókwè - Chicualacuala: ~530 km. Cut at Marracuene or Manhiça if the commuter part should read separately (not needed: all of it runs).
4. **Linha de Machipanda**, Beira - Dondo - Nhamatanda - Gondola - Chimoio - Manica - Machipanda: ~317 km (AIM).
5. **Linha de Sena**, Dondo - Inhamitanga - Dona Ana/Mutarara: 357 km to Malawi per CFM, of which the Beira - Moatize trains use Dondo - Dona Ana; **Dona Ana - Moatize**: 254 km (CFM). The Sena line beyond Dona Ana towards Malawi (Vila Nova da Fronteira) and the Inhamitanga - Marromeu branch (88 km): not built/greyed.
6. **Nacala corridor, Nampula - Cuamba**: ~350 km (CDN). Nacala - Nampula greyed or left off.

Expected: 6-7 register lines, ~1,950 km running (88 + 70 + 530 + 317 + ~300 + 254 + 350).

### Sources

- Stops: none published as a list. seat61 gives termini and a few stops; OSM's station objects (128, 115 named) on the traced lines, Egypt's way (`osm_stops`: every named OSM station a line passes is a stop) unless OSM has routes.
- km: CFM's line pages (88, 357, 254, 88 Marromeu); traced km otherwise.
- Wikidata: 28 station items, 13 with coordinates; no line adjacency.
- GTFS: none.
- Licence: OSM ODbL; CFM pages are public.

### OSM quality (Overpass, 2026-10-08)

3,224 km of track, 710 named (22%): "Linha de Sena" 289, "Cuamba-Lichinga" 262, "Inhamitanga -
Marromeu" 82, a little Ressano Garcia. The relation query timed out (public Overpass overloaded
today); read from the extract. Geofabrik `africa/mozambique-latest.osm.pbf`, 243 MB.

### Recipe

Hand list traced by rinf.py in the shared eafrica reader; `osm_stops` for stops (no published
stop lists); CFM's line lengths in `check_model.REGISTER`. No border point (no train crosses).

### Open questions

- How far the daily Goba-line train runs (Boane or Goba).
- The Sena line passenger train: does it run Beira - Moatize through, or Beira - Mutarara with a change? seat61's Jan 2026 page says Beira - Moatize.
- Nampula - Cuamba after Vale/CDN's 2024-25 changes: still twice weekly per seat61.

## Build (2026-10-08)

Built with `eafrica_register.py` (commands as ke_sources.md "Build"; no `--fill`). **Result**:
6 register lines, 1,860 km, all running: Linha de Sena Dondo - Moatize 546.1, Linha do Limpopo
515.4, Linha de Nacala Nampula - Cuamba 340.5, Linha de Machipanda 313.7, Linha de Ressano
Garcia 88.4, Linha de Goba 56.2. check_model: Ressano Garcia 88.4 of 88 (CFM), Machipanda
313.7 of 317 (AIM).

Decisions:
- Every OSM station is a stop (no published stop lists).
- The Goba line counts as running to Goba (CFM: "daily services" on the Goba line; how far the
  daily train goes is not stated). It starts where OSM's track leaves the Ressano Garcia line,
  2.4 km past Machava; the Limpopo line at Infulene; the Sena line at Dondo.
- The Beira - Moatize train is one line from Dondo (Sena line to Dona Ana/Mutarara, then the
  Tete line); Mutarara - Vila Nova da Fronteira (Malawi), the Marromeu branch, Nacala -
  Nampula and Cuamba - Lichinga/Entre Lagos are not built (no passenger train).
- OSM's routes duplicating the register lines: rules/mz.py SKIP_ROUTES; Komatipoort - Maputo,
  Marromeu - Beira, Harare - Beira: NOT_SERVICE.
- Malawi's later clip fix at Nayuchi (BORDERS["XMWMZ1"]) also applies to Mozambique's next
  clip: Mozambique's extract was clipped before it, so ~2 km of track west of the real border
  at Entre Lagos is still drawn (owned by no line).
