# Morocco, Algeria, Tunisia, Egypt: sources and decisions (nafrica agent, 2026-10-03)

One reader for the four, `nafrica_register.py`, with the line lists in `nafrica_lines.py`.
It writes each country's list into rinf.py's input files (balkans_register's recipe), and
rinf.py traces it over OSM track. Settings: `rinf_countries/{ma,dz,tn,eg}.py`; rules:
`rules/ma.py`, `rules/dz.py`, `rules/eg.py`.

## Commands

    python extract.py --region ma --pbf data/raw/morocco-latest.osm.pbf --station-areas
    python nafrica_register.py --clip ma      # what lies in another country's outline, NOT_SERVICE routes,
                                              # and a route_master for routes mapped one per direction
    python nafrica_register.py --fill ma      # timetable stations OSM lacks or leaves unnamed
    python build_model.py --region ma --register nafrica_register:data/raw/rinf/ma
    python build_tiles.py --region ma
    python check_model.py --region ma
    (likewise dz, tn, eg; algeria-, tunisia-, egypt-latest.osm.pbf)

    python nafrica_register.py --trace eg "Tanta@30.99421,30.78133" "Qallin@30.84035,31.05020"
    python nafrica_register.py --fork ma "Bouskoura" "Berrechid" "Aeroport Mohammed V"
    python nafrica_register.py --crawl-eg --list-only      # ENR's train list (see Egypt)

`--station-areas` matters: Rabat Ville, Rabat Agdal, Salé, Fès, Témara are mapped only as areas.
`--fill` runs after every `--clip`.

## Why a hand list

None of the four is in ERA RINF. OSM names almost none of the track (Morocco: only the LGV;
Egypt: the Suez line), and route relations are thin (Morocco 17 train routes, Egypt 4, most
Tunisian and Algerian train routes have no stops). No operator publishes a line register. So
each line is written by hand from the operator's timetable: its stations in order, junctions
where OSM's track shows the branch (`--fork`), km traced over OSM. The km are ours (`no_chain`),
so `check_model.REGISTER` holds the only outside numbers.

Every line is a timetabled line, so a section that ends at a junction or a border point is one
trains run over: `build()` lists it in `served_sections` and build_model keeps it without an
OSM route.

## Morocco (ma)

Source: the unofficial ONCF GTFS built from oncf-voyages.ma (github.com/orhazal/
oncf-gtfs-unofficial, ODbL, version 2026-08-18; data/raw/gtfs/ma/oncf-gtfs.zip; Transitous
lists it). Every station on a list is one ONCF's trains call at, and only those (`listed_only`):
OSM maps stations at loops where nothing stops (Zenata, Sidi Bouknadel, Laassilate, Lbir Jdid,
M'Saada, Oulad Khitib, Arbaoua...). The stations' names are OSM's (French, often with Arabic).

13 lines, 1,868 km: Casablanca – Rabat – Kénitra (with Casa Port's line to Roches Noires),
Kénitra – Fès, Fès – Oujda, Taourirt – Nador (from the junction 6.5 km west of Taourirt, where
Nador's trains run in and back out), Sidi Kacem – Tanger, Sidi Yahya – Mechraa Bel Ksiri (the
Tanger – Kénitra trains' shortcut), the LGV Tanger – Kénitra (its stops only Tanger Ville and
Kénitra; `osm_stops_skip`, a rinf.py hook, keeps OSM's halts beside it off it), Casablanca –
Marrakech, the airport branch, El Jadida, Sidi El Aïdi – Oued Zem, Benguerir – Safi.

Decisions:
- **Al Boraq** runs about 16 a day each way: a line, and its own register line is the LGV.
  OSM's "LGV Al Boraq" routes stay OSM lines over it. Al Atlas and TNR routes likewise lines.
- **Al Bidaoui** (Casa Port – airport, ~17 a day) is OSM's route over casa-ken, casa-mar and
  the airport branch. Casablanca's RER does not run yet; the Kénitra – Marrakech LGV opens 2029.
- **Tanger – Tanger Med**: no train in ONCF's timetable; built and greyed (suspended); OSM's
  route 8530906 dropped.
- **Oujda – Bouarfa**: the "Oriental Désert Express" is a tourist charter: not built as a line
  (route 10038327 dropped); its track stays drawn.
- Msoun: OSM's unnamed station there is named by `--fill`.
- Casablanca trams T1-T4 and Rabat-Salé L1-L2 are OSM lines.
- Western Sahara: no railway; the outline used is religiondots' `ma`, which includes it.

Checks (en./fr.wikipedia): LGV 193.0 vs 186 (1.04, built platform to platform); Casablanca –
Kénitra 132.8 vs 137 (0.97); Sidi Yahya – Mechraa 43.5 vs 45; Tanger – Tanger Med 41.9 vs 45
from 3 km out; Taourirt – Nador 108.2 vs 110.

## Algeria (dz)

Sources: SNTF's timetable tables as fahrplancenter.com transcribes them (2017-2019, with
SNTF's km at every station; data/raw/dz/fahrplancenter/), the opening news of the lines since
(Tissemsilt – M'Sila Dec 2022, Saïda – Frenda Jan 2023, Boughezoul – Laghouat Oct 2023,
Khenchela – Constantine June 2024, Béchar – Tindouf Feb 2026, one train a day), fr.wikipedia's
"Liste des lignes de chemin de fer d'Algérie". Stations are the timetable's where OSM maps them
(`listed_only`: OSM keeps "Gare abandonnée", "Gare désaffectée" on the lines). Timetable stops
OSM lacks (El Eulma, Draa El Mizan, Beni Amrane, Aïn Torki...) are left out: the tables give no
coordinate to place them.

24 lines, 4,850 km. Checks against SNTF's km: Alger – Oran 421.0 vs 419; El Harrach –
Constantine 449.8 vs 454; Touggourt line 416.5 vs 419; Annaba – Tébessa 230.4 vs 231; Oued
Tlelat – Béchar 645.6 vs 649; Tabia – Ghazaouet 185.6 vs 185; Moulay Slissen – Frenda 220.2 vs
221; worst Souk Ahras – border 47.3 vs fr.WP 53 (0.89).

Decisions:
- **Annaba – Tunis**: running again since 15 Sept 2026, three a week each way (SNTF, SNCFT;
  La Presse, El Watan). So Souk Ahras – the Tunisian border is built, to a border point
  `XDZTN1` at 8.35785, 36.40970 (where the line from OSM's last Algerian track meets the
  outline). The train itself has no OSM route; it would be a named train.
- **Ghazaouet**: one daily train in the 2017 table (1153; OSM's route carries that number):
  built as running. Unsure; nothing newer found.
- **Jijel** (Ramdane Djamel – Jijel): "out of service" in the 2018 table, nothing since:
  not built. **Beni Saf**: OSM's route ends at Aïn Témouchent; the line ends there.
- Not built, under construction: Relizane – Tiaret, Tiaret – Tissemsilt, Touggourt – Hassi
  Messaoud, Mécheria – El Bayadh. Freight only: Tébessa – Djebel Onk, Ouenza, Bouarfa side.
- "Batna – Alger" (OSM, 3 stops over 680 km) is a named train (`rules/dz.py`).
- Algiers metro, the trams (Algiers, Oran, Constantine, Sidi Bel Abbès, Sétif, Ouargla,
  Mostaganem) are OSM lines; `--clip` gives master-less one-per-direction routes a route_master.

## Tunisia (tn)

Source: SNCFT's own GTFS, gps.sncft.com.tn/gps/gtfs_ALL.ZIP ("Spring 2021 version D";
data/raw/gtfs/tn/), with SNCFT's distances (shape_dist_traveled), checked against the 2026
summer timetable press release (Tunis – Kalaâ Khasba, 3 a day, Le Kef by one express) and
tunismapper.com's 2026 line list. RFR line D (Tunis – Gobaâ, opened 25 Jan 2025) runs on line
1's track; its stations Le Bardo, Mannouba, Les Orangers, Gobaâ are on line 1. RFR line E
(opened 2023) is its own line. GTFS coordinates can be kilometres off (Sidi Bou Goubrine 6 km):
five points carry OSM's coordinate instead; `--fill` names two unnamed OSM halts and adds five
timetable stops OSM lacks at the timetable's point (Trika, Oued Sarrath, Cheria, Mg 28/29, El
Ayoun).

11 lines, 1,364 km. Checks against SNCFT's own km: Ghardimaou 210.6 vs 211.1, Bizerte 72.1 vs
72.5, Kalaâ Khasba 229.2 vs 229.7, Le Kef 30.8 vs 31.2, Tozeur 235.4 vs 234.1, Nabeul 16.9 vs
17.1, Redeyef 44.0 vs 44.1: all within 1%.

Decisions:
- **The border**: OSM has no track for the last ~3 km from Ghardimaou to the Algerian border
  (both extracts stop: Tunisia's at 8.388 E, Algeria's at 8.355 E). Tunisia's line ends at
  Ghardimaou; only Algeria's side reaches the border point.
- **The Gafsa mining basin**: Metlaoui – Redeyef and Tabeddit – Om El Araies have a daily train
  in SNCFT's feed and on tunismapper in 2026: built as lines. tunismapper also lists Gafsa –
  Cheria (8 a day) and Aguila – M'dhilla, not in the feed: probably workers' trains, not built.
- The Métro du Sahel is SNCFT's: a register line, with Monastir's stub and the link from Sousse's
  goods station (Tunis – Monastir/Mahdia trains). The TGM and the six metro léger lines are OSM
  lines.

## Egypt (eg)

Source: ENR's timetable as egypttrains.com republishes it (unofficial, "data sourced from the
official Egyptian National Railways timetable", updated 18 Aug 2026; ENR's own site answers
only station-pair searches). Its one-page train list (858 trains, 819 active, each with its
first and last station and ENR's km; data/raw/eg/enr_train_list.json, made by
`--crawl-eg --list-only` from data/raw/eg/egypttrains/_trains.html) gives the 212 routes trains
run; the lines are those routes cut at their junction stations. The per-train pages (stop
lists) answered 429 after nine pages and were left: so every named OSM station a line passes
is a stop (`osm_stops`), not a timetable's list. Python's and Windows' curl get 403 there; Git's
curl is let through.

30 lines, 2,949 km. Against ENR's km: Cairo – Alexandria 207.5 vs 208, Cairo – Aswan 877.3 vs
879, and 24 more all within 1-3%; worst Maamoura – Rosetta 51.7 vs 49 (1.05).

Decisions:
- **Sinai**: Qantara East – Bir al-Abd has 4 trains a day (ENR, Youm7/Maspero, April-July 2026):
  built. Ferdan bridge – Qantara East has no passenger train; not on a line.
- **Monorail**: the East Nile line opened 6 May 2026 (full line 27 June): OSM line. The LRT
  (Adly Mansour – 10th of Ramadan / Capital) and Cairo Metro 1-3 are OSM lines.
- **Cairo – Suez**: ENR runs Adly Mansour – Suez (115 km, 4 a day), but OSM's track west of
  Badr has three gaps no trace crosses; the line starts at Badr (87.5 km).
- **Mansoura – Bilqas** (ENR 28 km): OSM's track has no direct link; the trains' track is on
  the Damietta and Kafr el-Sheikh – Shirbin lines, so no separate line.
- Not built: the Kharga / Western Desert line, Qena – Safaga (no trains in the list); the
  high-speed lines (stations mapped, no service yet); Alexandria's Abu Qir line (closed 2024 for
  the metro). Alexandria's Raml tram closed 1 April 2026 for rebuilding; its city trams run but
  OSM's routes have no stops, so they are not built (their track is left out).
- OSM's "Alexandria-Aswan line" (a corridor relation, 13 stops) is a named train (`rules/eg.py`).

## Open

- Egypt's stop lists (the per-train pages) when the site lets a slow crawl through
  (`--crawl-eg`, 4 s apart).
- ONCF's and SNCFT's feeds could drive gtfs_served (a FEEDS entry each).
- Colours: no `colours/<cc>.csv` yet.
- Algeria's missing halts would need coordinates (SNTF publishes none).
