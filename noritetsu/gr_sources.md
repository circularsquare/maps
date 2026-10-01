# Greece register sources (built 2026-10-01)

What the Greek build reads, where each piece came from, and what is still wrong with it.
Greece is built with `rinf.py`; how that reader works is in its docstring, and the per-country
entry is `rinf_countries/gr.py`, whose docstring has the line grouping, the fixes to RINF and
why. Downloads live in `data/raw/rinf/gr/` (gitignored). Nothing needed a login or a key.

## Run

```powershell
python rinf.py --fetch gr                                              # RINF + Wikidata, ~10 s
curl -L -o data/raw/greece-260930.osm.pbf https://download.geofabrik.de/europe/greece-260930.osm.pbf   # 341 MB
$env:OSMIUM_POOL_THREADS=2; python extract.py --region gr --pbf data/raw/greece-260930.osm.pbf --station-areas   # 30 s; delete the .pbf after
python inspect_region.py --region gr
python build_model.py --region gr --register rinf:data/raw/rinf/gr     # 7 s
python build_tiles.py --region gr                                      # 6 s
python check_model.py --region gr
python rinf.py --dry gr          # the reader alone, with its full log
```

Geofabrik's `greece-latest.osm.pbf` was redirect-looping on 2026-10-01 (curl: 50 redirects), as
Switzerland's was; the dated file from https://download.geofabrik.de/europe/greece.html works.
`--station-areas` adds one station (of 15 station areas) and costs nothing.

## Sources

- **ERA RINF**, SPARQL at `https://graph.data.era.europa.eu/repositories/rinf-plus`, fetched
  2026-10-01 (`sections.json`: 193 sections, 25 line ids, 1,830 km once line 04 is read in
  metres; `points.json`: 188 points, every one with a coordinate). Country code GRC; point ids
  are `EL` + five digits. Only OSE is registered (`0073_IM`), standard gauge only, no validity
  periods.
- **OpenStreetMap**, Geofabrik `greece-260930.osm.pbf` (data to 2026-09-30), ODbL: track,
  stations, 47 `route=train` relations (Hellenic Train's Proastiakos A1-A4, Patras Π1/Π2, IC,
  ICE, regional T1-T3 and ΑΠ, RT3 Katakolo - Olympia, Kiato - Aigio), 12 metro (Athens M1-M3,
  Thessaloniki 1 and 2), 4 tram (T6, T7), 54 `route=railway` relations. Station names are Greek;
  only 142 of 320 rail stations carry `name:en`. No Greek OSM station has `uic_ref`.
- **Wikidata** (`wikidata.json`), CC0: only three route numbers (P1671), none of them OSE's
  (Thessaloniki Metro 1 and 2, and "E 851"). Unused.
- **en.wikipedia / el.wikipedia** line articles, retrieved 2026-10-01, CC BY-SA: the figures in
  `check_model.REGISTER["gr"]`. "Piraeus–Platy railway" (456.60 km; Tithorea - Domokos new
  alignment 107.33 km; branches Oinoi - Chalkida 21.69, Lianokladi - Stylida 22.61, Palaiofarsalos
  - Kalambaka 80.44, Larissa - Volos 60.76), "Thessaloniki–Bitola railway" (chainage Platy 36.3,
  Florina 187.5), "Alexandroupoli–Svilengrad railway" (178.5), "Athens Airport–Patras railway"
  (Airport - SKA 30, SKA - Kiato 105, Kiato - Aigio 71), el "Σιδηροδρομική γραμμή Θεσσαλονίκης -
  Αλεξανδρούπολης" (440.8). Mostly uncited.
- **Service status**, used only to decide `suspended` and listed below: en.wikipedia
  "Thessaloniki Regional Railway" and "Thessaloniki–Skopje railway" (no Idomeni service), el
  "Σιδηροδρομική γραμμή Θεσσαλονίκης - Αλεξανδρούπολης" (no trains Serres - Alexandroupoli, buses
  instead), and the Hellenic Train GTFS below.
- **Hellenic Train GTFS**, `https://jbb.ghsq.de/gtfs/gr-hellenic-train.gtfs.zip` (Transitous,
  generated, feed 2026-07-10 to 2026-12-01, attribution "OpenStreetMap contributors"): read once
  by hand to see which station pairs have trains; not wired into the build (another agent owns
  `gtfs_served.py`).

## Lines

RINF's ids are OSE's ("01.00.00"), not public numbers, so lines carry no ref and are named by
their ends, in Greek with an English name (rinf.py reads `id_name`'s (name, name_en)):

| RINF ids | line | built km |
|---|---|---|
| 01, 22 | Πειραιάς – Θεσσαλονίκη / Piraeus – Thessaloniki (with the new Lianokladi - Domokos line added) | 490.2 |
| 25 | Θεσσαλονίκη – Αλεξανδρούπολη | 429.5 |
| 27, 28 | Αλεξανδρούπολη – Ορμένιο | 180.2 |
| 16 | Πλατύ – Φλώρινα | 157.6 |
| 03, 04, 05, 06 | Αεροδρόμιο – Κιάτο / Athens Airport – Kiato | 142.8 |
| 12 | Παλαιοφάρσαλος – Καλαμπάκα | 79.9 |
| 20, 21 | Θεσσαλονίκη – Ειδομένη (greyed, not running) | 62.4 |
| 13 | Λάρισα – Βόλος | 61.0 |
| 09 | Τιθορέα – Λειανοκλάδι (παλαιά γραμμή) (greyed, not running) | 56.1 |
| 10 | Λειανοκλάδι – Στυλίδα | 22.5 |
| 08 | Οινόη – Χαλκίδα | 21.5 |
| 26 | Στρυμόνας – Προμαχώνας | 14.4 |

Not built: 07, 23, 24 (freight, `skip_line`), 11 (the old Lianokladi - Domokos line, skipped:
its middle has no track in OSM and its ends lie on the new line), 17 Amyntaio - Ptolemaida and 18
Mesonisi - Neos Kafkasos (no track in OSM; no passengers for years), 19 Axios - Gefyra link
(unridden, dropped by build_model).

No `colours/gr.csv`: OSE's lines have no colours. The Proastiakos, metro and tram lines carry
OSM's colours.

## Counts (2026-10-01)

- 12 register lines, 1,718 km (14 from the reader, 1,731 km; build_model dropped 4 unridden
  junction sections, 13 km, which emptied 18 and 19). Two lines greyed as not running, 119 km.
- 39 lines in all: those 12, and 27 from OSM: 19 Hellenic Train lines (Proastiakos A1-A4,
  Patras Π1/Π2, ICE, three IC, T1-T3, four ΑΠ regionals, RT3 Katakolo - Olympia, Kiato - Aigio),
  5 metro (Athens M1-M3, Thessaloniki 1 and 2), 2 tram (T6, T7), the Lycabettus funicular. No named trains: Greek IC and ICE run
  as lines (no `gr` branch in `looks_like_service`). 393 stations, 206 on register lines (82 with
  an English name). gr.pmtiles 0.6 MB.

## Check

`python check_model.py --region gr`: against RINF's own section lengths, 12 lines, median 0.998,
none off by more than 5%. Against Wikipedia, 9 lines, 7 within 3%:

- **Αεροδρόμιο – Κιάτο** (1.06): the figure is Airport - SKA - Kiato; the built line also has
  the Kato Acharnai - Zefiri and Kato Acharnai - Miden links (RINF 05's and 03's ends) and line
  06's 2.6 km stub at Ano Liosia, 7.7 km.
- **Πλατύ – Φλώρινα** (1.04): RINF's own is 156.6 against the article's chainage difference of
  151.2, and the build agrees with RINF.
- **Θεσσαλονίκη – Αλεξανδρούπολη** (0.97): the line starts at the TX1 junction, 5.6 km out of
  Thessaloniki over line 01's track.

## Stations

RINF types 172 points as passenger stations or halts. With rinf.py's `GREEK_FOLD` (added for
Greece) 97 match their OSM station by name and 50 by distance (all within 200 m, listed in the
build log); 25 have no OSM station: the closed halts of the old lines 09 and 11 (Arpini, Asopos,
Eleftherochori, Karya, Kallipefki, Styrfaka, Therme, Xynias), line 17's, and halts no train calls
at (Chalki, Melia, Latomio, Melissiatika on Larissa - Volos; Chrysso, Dimitra, Potamos on 25;
Doxaras, Evangelismos, Mezourlo on 01). They merge away as junctions. `osm_stops: True` adds 58
OSM stations an OSM train route stops at: the newer Proastiakos halts (Lefka, Tavros, Pyrgos
Vasilissis, Sfendali), the Lamia halts on Lianokladi - Stylida, the Ormenio line's halts.

## Not running

Two lines are greyed through rinf.py's `suspended`: 09, the old Tithorea - Lianokladi line over
Bralos (bypassed by the new line since 2018; OSM tags it usage=tourism; no train in the timetable)
and 20+21, Thessaloniki - Idomeni (no passenger trains since 2020).

**For Anita to decide**, from the Hellenic Train GTFS (2026-07-10 to 2026-12-01) and the
el.wikipedia line article: these register lines are drawn as running but have no trains in that
timetable.
- Serres - Drama - Xanthi - Komotini - Alexandroupoli, about 290 km of line 25: no passenger train
  for years, buses instead (el.wikipedia). OSM's T3 route still runs to Drama.
- 12 Palaiofarsalos - Kalambaka and 13 Larissa - Volos: closed for flood repairs since Storm
  Daniel (2023), buses; reopening was expected in summer 2026 but the timetable to December has
  none.
- 10 Lianokladi - Stylida: no trains in the timetable.
- 26 Strymonas - Promachonas: only the Sofia - Thessaloniki train, not in Hellenic Train's feed.
`suspended` greys a whole line, so 25's partial closure needs the timetable check
(`gtfs_served.py`), which marks sections no train runs over; it can settle all of these once a
Greek feed is wired.

## Still off, and why

- **Pending shared changes** (with the managing session, 2026-10-01): (1) build_model's
  merge_osm_twins strips ": A ↔ B" from Greek route_master names as if it were a direction, so
  three lines show as "Τρένο IC" / "Train IC" and four as "Τρένο ΑΠ"; a two-way-mark rule keeps
  those names whole. (2) build_stations doubles a station mapped only as two stop positions of
  one name (Ελίκη, Λυκοποριά, Ακράτα, Ξυλόκαστρο, Πλάτανος on Kiato - Aigio, Κατεχάκη on Metro
  M3). Both need a gr rebuild once landed (build_model, build_tiles, check_model).
- **Metro M3 does not credit the Airport line.** M3 trains run on OSE's track from Doukissis
  Plakentias to the Airport (20 km, four stations shared with Proastiakos), but subway and rail
  never credit each other in build_credits, so riding M3 to the airport counts nothing towards
  Αεροδρόμιο – Κιάτο.
- **Narrow gauge.** None is in RINF. Patras suburban (Π1, Π2, metre) and Katakolo - Olympia (RT3)
  are OSM lines. Diakopto - Kalavryta (750 mm rack) is tagged `railway=disused` in OSM, so it is
  not extracted at all; the timetable has no train on it either. The Pelion railway (600 mm,
  "Μουτζούρης", seasonal weekend tourist train) and the Thessaly museum railway have track but no
  route relation, so they are not lines and build_tiles leaves their track out (the general
  "infra relation only" issue in HANDOFF). The rest of the Peloponnese metre-gauge network is
  disused in OSM.
- **Kiato - Aigio** (new standard gauge, 71 km) is not in RINF; it is the OSM line "Τρένο 3xx",
  so it counts as a line but not as register track. Aigio - Rododafni is not open.
- **Thessaloniki suburban** (Thessaloniki - Sindos) has route relations with no stops, so it
  builds no line; its track is line 01's anyway.
- **Border stubs**: Pythio - Turkish border (1.35 km) and Ormenio - Bulgarian border (4.8 km)
  stay because OSM's "Pythio-Edirne-Svilengrad" relation, a historical service, runs over them.
  The Promachonas border stub stays on the Sofia - Thessaloniki relation.

## For the timetable check (gtfs_served.py)

The feed's stop ids are hex strings, not UIC numbers as `gtfs_sources.md` says, and no Greek
OSM station has `uic_ref`, so stations have to join by name and distance. The file also holds
ferries (route_type 4, four agencies) and Hellenic Train's replacement buses (route_type 3);
trains are 2, 102, 106, 109. Names to fold: "Σιδηροδρομικός Κέντρο Αχαρνών (ΣΚΑ)" and "Ska" =
OSM "Σιδηροδρομικό Κέντρο Αχαρνών"; "Σέρραι" = "Σέρρες"; "Πεδινός" = "Πεδινό"; "Πετρίτσιον" =
"Νέο Πετρίτσι"; Latin names for some ("Paleopharsalos", "Rodopolis", "Aegion"); and at least
one wrong name ("Rodopolis - Λιβαδειά", which is Λιβάδια Κερκίνης, not Livadeia on line 01).
