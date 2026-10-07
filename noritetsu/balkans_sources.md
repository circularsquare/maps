# Western Balkans register sources (built 2026-10-03)

Serbia `rs`, Bosnia and Herzegovina `ba`, Montenegro `me`, North Macedonia `mk`, Albania `al`
and Kosovo `xk`. None of the six is in ERA RINF. One reader, `balkans_register.py`, writes each
country's line list into rinf.py's input files (`data/raw/rinf/<cc>/sections.json`,
`points.json`, `names.json`), and rinf.py traces it over OSM track, matches stops, merges
sections and names lines, as for Russia and India. Its docstring says how; this file says
where the lists come from, what was decided and what is still off. Settings rinf.py reads:
`rinf_countries/<cc>.py` (each `balkans_register.country_conf("<cc>")`). Named-train rules:
`rules/rs.py` (`me` uses it), `rules/ba.py` (Croatia's), `rules/mk.py` (`xk` uses it).

## Run

```powershell
# extracts: data/proc/<cc> from Geofabrik serbia, bosnia-herzegovina, montenegro, macedonia,
# albania, kosovo (managing session). Serbia was re-extracted with --station-areas (no change).
python balkans_register.py --clip ba        # after every ba extract: see "Territory"
python build_model.py --region rs --register balkans_register:data/raw/rinf/rs   # ~55 s
python build_tiles.py --region rs
python check_model.py --region rs
python balkans_register.py --convert rs     # the conversion alone, with its log
```

Each of the six builds in under a minute. `--register rinf:data/raw/rinf/<cc>` builds from
the last conversion without converting again.

## Why one reader and not OSM's named track

`probe_kr_ways.py`-style counts on the extracts (2026-10-03): main and branch track carrying a
line number or name in `ref`/`name` is 85% of Serbia's 3,422 km (`ref` 101, 102...), but 2%
of Bosnia's, 2% of North Macedonia's and none of Albania's. OSM's route=railway relations
cover more (Serbia 61, Bosnia 42, North Macedonia 20) but are patchy and partly about
long-closed narrow gauge. So each country has a line list, and OSM gives stations and track.

## Serbia: IŽS's own register

- **Infrastruktura železnice Srbije, Network Statement 2026**,
  https://arhiva.infrazs.rs/IzjavaMreza/2026_NS.pdf (`data/raw/rs_ns_2026.pdf`, 181 pages;
  the copy on infrazs.rs answers 404). Its Appendix 6, "Register of infrastructure data", is
  every service point of every line in order with its chainage and the distance from the
  point before (scanned tables, pages 139-158). I transcribed columns 4 and 5 by hand from page
  images into `data/raw/rs_ns2026_appendix6.txt` (677 rows); the file's header lists the few
  edits. Every section carries IŽS's km, so check_model compares every line with its chainage.
- **Uredba o kategorizaciji železničkih pruga** (Sl. glasnik RS 92/2020, 6/2021), the decree
  numbering the lines (`data/raw/rs_uredba_kategorizacija_2021.htm`): the line list.
- The NS gives the network as 3,357.3 km (1,759.0 main, 1,598.4 other); the 30 lines read here
  are 2,640 km of chainage, the rest yard, connecting, industrial and closed lines.

**Built**: 26 register lines, 2,563 km (1,970 km running, 590 km greyed), 539 stations, 91
lines in all with OSM's (Srbijavoz's routes, BG:Voz, Belgrade's trams), 5 of them named
trains. Every line within 4% of IŽS except three short ones (below); median 1.000 against the
chainage.

**Line names** are IŽS's number and two or three places, in Cyrillic as IŽS and OSM write
them: "102 Београд Центар – Ниш – Прешево", English "Line 102 (Belgrade Centre – Niš –
Preševo)". The IŽS names are long legal strings ("(Beograd Centar) - Resnik - Požega - Vrbnica -
državna granica - (Bijelo Polje)"); the shown name keeps the places a rider knows.

**Stations**: a list name finds its OSM station by one key for Latin and Cyrillic (`lat_key`),
nearest the line's previous placed point within the km plus 3 km (there are two Vitkovac,
two Leskovac). 61 list names have no OSM station and are left out, their km going to the
section across them: halts OSM does not map (Pinosava, Ripanj Kolonija, Stevanac, Letovica),
and stations OSM tags `disused:railway=station` (Bor, Borska Slatina, Kuršumlija, Žitorađa).
OSM stations the list lacks are added where they lie on a traced section (`osm_stops: "all"`):
Zemunsko polje, Trnjane, Kalenić, Brvenik, Kneževac, Osipaonica stajalište, TPS Novi Sad.

**Lines built**: 101-110, 120 (Karađorđev park - Dedinje, BG:Voz's link from Pančevački most to
Rakovica), 121 (Inđija - Golubinci, the Novi Sad - Šid trains' chord), 201, 202, 205, 207, 208,
211, 213, 216, 218, 219, 223, 308, 309, 501 (the Šargan Eight, 760 mm, IŽS's museum line, run
daily in season: counted). **Left out**: 226 Vrbas - Sombor, 306 Rimski Šančevi - Žabalj, 311
Markovac - Resavica and 313 Vršac - Bela Crkva have no passenger trains and OSM tags their
stations disused (226's track also has gaps: the trace ran round by Bogojevo at 2.2x IŽS's
54.4 km); the yard, connecting and industrial lines 111-119, 122-128, 203/204 (abolished),
206, 209/210, 212, 214/215, 217, 220-222 (Kuršumlija's two legs are read into 223), 301-305,
310, 4xx.

**Junctions**: IŽS gives junctions no coordinate. The five other lines branch at
(`SHARED_JUNCTIONS`: Ćuprija, Dedinje, Sajlovo, Vražogrnac 2) are placed on the line they lie
inside at their chainage along the traced track; the rest are left out. 308 starts at Donja
Borina station rather than its junction 0.9 km on (the junction lies between Donja Borina and
an outline border point that has no chainage, so it could not be placed).

**Which sections run**: Srbijavoz's GTFS (vekejsn's merged feed,
https://gitlab.com/api/v4/projects/vekejsn%2Fgtfs-generators/packages/generic/srbijavoz-merged-gtfs/latest/srbijavoz_merged.zip,
the one Transitous uses; `data/raw/gtfs/rs/srbijavoz.zip`, 264 trips, 10 July - 12 December
2026, with MÁV's Szeged trains and ŽPCG's Bar trains), through gtfs_served. 326 of 336 feed
stations matched. Not running (greyed, 590 km):
- 102 Niš - Preševo (148 km): no train south of Niš in the feed, and OSM has no route there
  either.
- 106 Niš - Dimitrovgrad (97 km): no train (works on the line).
- 103 Rakovica - Mala Krsna - Velika Plana (89 km), 223 Doljevac - Kuršumlija - Merdare
  (83 km), 208 Novi Sad - Orlovat (53 km), 202 Kikinda - Banatsko Veliko Selo (11 km).
- 218 Požarevac - Kučevo - Majdanpek (91 km), and also **Mala Krsna - Požarevac (17 km), which
  is wrong**: Srbijavoz runs Smederevo - Mala Krsna - Požarevac (Re 6750-6761, in OSM and in
  the timetable to 12 December 2026) but the feed lacks it, and gtfs_served's "a closed line
  closes whole" rule grows the closure from Požarevac. The report to the managing session has
  a two-line fix (`NO_GROW_OVER_ROUTES`), trialled: it leaves those 17 km drawn and changes
  nothing else.

Border stubs with no train are dropped, as in Croatia: Šid - border 5.6, Preševo - border 8.1,
Subotica - Kelebija 8.0, Dimitrovgrad - border 6.5, Vršac - border 14.1, Rudnica - Kosovo line
1.9, Bogojevo - border 2.7, Kikinda line to Romania, Donja Borina - border, Mokra Gora - border.

**Off against IŽS** (check_model, ratio built / IŽS):
- 120 Karađorđev park - Dedinje 0.59: OSM's Dedinje junction lies 0.6 km nearer than the
  junction placed by chainage on 102.
- 309 Pančevo Varoš - Vojlovica 1.19: OSM's Vojlovica node lies beyond IŽS's axis.
- 501 Šargan Vitasi - Mokra Gora 0.81: the trace short-cuts part of the figure-eight.
- Kept on their own line's track although a piece disagrees with IŽS (rinf's "length off",
  13 pieces): Grejač/Tešica and Tomića Brdo look misplaced in OSM; Beograd Centar - Rakovica
  traces 5.8 for 8.5 (IŽS's chainage goes round by Dedinje and junction G).

## Montenegro: ŽICG

- **ŽICG Network Statement 2017** (Izjava o mreži),
  https://www.zicg.me/AdminCMS/public/pdf/izjavaomrezi/IZJAVA%20O%20MREZI%20%202017.pdf
  (`data/raw/me_zicg_izjava_o_mrezi_2017.pdf`; the newest I found). Annex 4 has every service
  point with chainage. Network 250.51 km of open line. The border on the Bar line is at
  287+438.70, IŽS's own figure for the same point.
- Lines: Bar - Vrbnica (the border; ŽICG's Bijelo Polje - Bar), Podgorica - Nikšić, Podgorica -
  Tuzi - border. Points carry ŽICG's chainage.
- Running: ZPCG's GTFS (Transitous mirror `https://api.transitous.org/gtfs/me_zpcg.gtfs.zip`,
  Mobility Database mdb-2377, ODbL; `data/raw/gtfs/me/zpcg.zip`): Bar - Bijelo Polje, Bar -
  Podgorica, Podgorica - Nikšić (passenger service back since 2012), and the Belgrade trains.
  Podgorica - Tuzi (13.7 km) greyed; Tuzi - Albanian border (11.1, freight only) dropped.
- Check: Bar - Vrbnica 166.9 / 167.4, Nikšić 56.0 / 56.2, Tuzi 13.7 / 13.7.
- Named trains: IR 1130/1131 „Тара“ and „Ловћен“ (rules/rs.py, shared).

## Bosnia and Herzegovina: ŽFBH and ŽRS

- Lines by ŽFBH's and ŽRS's numbers, as OSM's relations carry them: 11 Sarajevo - Čapljina,
  12 Šamac - Doboj - Sarajevo, 13 Novi Grad - Banja Luka - Doboj - Tuzla, 14 Brčko - Banovići,
  15 Živinice - Zvornik, 17 Dobrljin - Novi Grad - Bihać. Freight branches (16 Omarska -
  Tomašica, 24 Podlugovi - Vareš) are not passenger lines. Operator per line by its larger
  part (`Željeznice Federacije BiH`, `Željeznice Republike Srpske`).
- No km list was found (ŽFBH's Izjava o mreži link on zfbh.ba answers 404; ŽRS publishes
  none I could find): km are traced. Outside figures: ŽFBH 608.5 km and ŽRS 418.3 km of
  line (bs/sh.wikipedia, from the companies), Sarajevo 0+000 - Čapljina 170+390 (zfbh.ba
  infrastructure page), Šamac - Sarajevo 242 (Wikidata Q1279793).
- OSM has no station at Šamac, Tuzla or Brčko: those ends are placed at the town's station by
  coordinate and become junctions, so the no-train sections to them are dropped (Šamac -
  Doboj 63 km, Petrovo Novo - Tuzla 30 km). Line 14 has gaps in OSM's track and builds no
  line at all.
- Running: the GTFS Transitous uses for ŽFBH (`https://owncloud.cesnet.cz/index.php/s/yWJhi9wjUIc3IC2/download`,
  Petr Novák's, ŽFBH and ŽRS trains; `data/raw/gtfs/ba/zfbih.zip`). **Its window is 1
  January - 13 December 2025**: last year's timetable. gtfs_served ignores the date, so the
  2025 trains decide. Greyed: 17 Novi Grad - Bihać (66 km), 12 Doboj - Maglaj (23 km), 15
  Živinice - Kalesija (20 km). The seasonal Sarajevo - Ploče train keeps Čapljina - border
  (7.3 km) drawn: the crossing needs a border point (report).
- Check: 11 is 169.9 against ŽFBH's 170.4 + 7.3 (0.96); 12 is 0.71 of Wikidata's 242 with
  Šamac - Doboj dropped.

## North Macedonia: MŽ Infrastruktura

- Lines (named as Wikidata and OSM's relations do; MŽ's own line numbers were not found):
  Tabanovce - Gevgelija, Skopje - Volkovo - Blace (to Kosovo), Gjorče Petrov - Kičevo, Veles -
  Bitola - Kremenica, Veles - Kočani, Kumanovo - Beljakovce. Km traced; checked against
  Wikidata (Q3239944 214.9, Q3239932 31.1, Q3239602 102.6, Q3239995 145.3, Q3239994 85.5): all
  within 4%.
- Running: MŽ Transport's 2025/26 timetable as GTFS (Petr Novák's rehost,
  https://hoermalmeister.github.io/gtfs-rehost/mzi/mzi.zip; `data/raw/gtfs/mk/mzi.zip`):
  Skopje - Kumanovo, - Gevgelija, - Veles - Bitola - Žabeni. Greyed: Veles - Kočani,
  Skopje - Volkovo, Kumanovo - Beljakovce, Tabanovce - Kumanovo, and **Gjorče Petrov - Kičevo
  (`suspended` in the list)**: the timetable has no train, but OSM's 2013 route relation made
  gtfs_served leave it "unknown", i.e. drawn as running.
- Veles - Bitola ends at Žabeni: OSM has no track on to Kremenica and the Greek border
  (EU00190).
- Named train: IC 891/892 (Pristina - Skopje, suspended).

## Albania: HSH

- No timetable feed and no line list: lines written from HSH's network as OSM and
  en.wikipedia ("Rail transport in Albania") describe it, km traced. Checked: Elbasan -
  Pogradec 77.4 / 78 (Wikidata Q31667925), Fier - Ballsh 24.7 / 25 (Q130927911). Wikidata's
  Shkodër - Vorë (103.6) disagrees with the 83 km traced between the two stations and is not
  used.
- Running: only Durrës - Elbasan, Fridays to Sundays (more than weekly: counted), from Durrës
  Plazh since Durrës station's rebuilding (en.wikipedia, citing hekurudha.al, January 2025).
  So the line is "Shkozet - Rrogozhinë - Elbasan", 73.7 km; Shkozet - Durrës Plazh (1.5 km) is
  drawn although trains start at Plazh. Everything else is greyed (`suspended`): Durrës -
  Tiranë (rebuilt as Tirana - Durrës - Rinas; service now planned for the end of 2027,
  citizens.al 2026-05-22, gazetatema.net), Elbasan - Pogradec (closed 2012), Rrogozhinë -
  Fier - Vlorë (Albrail freight), Fier - Ballsh, Vorë - Shkodër - Hani i Hotit (the Shkodër -
  Laç trains ran only in spring 2023).
- OSM gap: Lushnjë - Fier has no path, so that section is missing from the greyed Vlorë line.

## Kosovo: Infrakos and Trainkos

**Kosovo is its own region because Trainkos runs its railway** (Anita's territory rule: de
facto, drawn as trains run). Infrakos manages the infrastructure and Trainkos runs the trains;
Serbia's IŽS still lists the lines in Kosovo in its register (109 beyond the administrative
line, 223 beyond Merdare, 224, 225, 312) and says they are "temporarily under the
supervision of UNMIK". Srbijavoz's trains stop at Rudnica and Merdare, on the Serbian side;
no Serbian train runs in Kosovo, so the north Kosovo track (Jarinje - Leshak - Zvečan -
Mitrovica North) is drawn with Kosovo, greyed, by where it lies.

- Lines (Infrakos 2025 annual report: 333 km of open line; Line 10 Hani i Elezit - Leshak is
  its number): Linja 10 Leshak - Fushë Kosovë - Hani i Elezit, Fushë Kosovë - Pejë, Fushë
  Kosovë - Prishtinë, Klinë - Prizren. Km traced. No per-line published length was found, so
  check_model has no Kosovo rows; the built 224 km plus what is left out (Klinë - Prizren
  35 km, Prishtinë - Podujevë - Merdare, the Drenica freight line) is consistent with 333.
- Running: Trainkos's GTFS (Petr Novák's rehost of trainkos.com's timetable,
  https://hoermalmeister.github.io/gtfs-rehost/trainkos/trainkos.zip; `data/raw/gtfs/xk/trainkos.zip`):
  Prishtinë - Pejë, three pairs a day. Its Prishtinë - Skopje pair runs on no day: suspended
  since 2020 for Line 10's rebuilding (en.wikipedia "Rail transport in Kosovo", mid-2026).
  Line 10 is `suspended` (greyed, 136 km). Running: Fushë Kosovë - Pejë and Fushë Kosovë -
  Prishtinë, 88 km.
- Left out: Klinë - Prizren (no train; OSM has no station at Prizren, so the one section
  ends at a junction and is dropped), Prishtinë - Podujevë (no OSM track), Zvečan -
  Mitrovicë (a gap in OSM's track at the Ibar, so Line 10 is drawn in two pieces).

## Territory and borders

- **Štrpci**: the Belgrade - Bar line's 9 km through Bosnia (Goleš - Štrpci - Rača) are run by
  Srbijavoz and ŽPCG and stay on Serbia's 108 (IŽS lists Štrpci on 108). `clip("ba")` takes
  that track, the 3 station records and 14 Serbian and Montenegrin route relations out of
  Bosnia's extract, so Bosnia does not draw them again.
- Border points: the shared table's where it has one (EU00199 Kelebija, EU00211 Dimitrovgrad,
  EU00248 Vršac, EU00189 Gevgelija, EU00190 Kremenica); Croatia's own RINF points, which the
  table lacks (EU00221 Slavonski Šamac, EU00222 Metković, EU00223 Volinja, EU00225 Drenovci,
  EU00226 Tovarnik, EU00227 Erdut), under their uopid; ours elsewhere (`BORDERS` in
  balkans_register.py). Vrbnica - Bijelo Polje (XMERS1) and Preševo - Tabanovce (XMKRS1) are
  measured along OSM's track from IŽS's chainage; the others where the track crosses
  religiondots' outline.
- **The table's Röszke (EU00200) and Jimbolia (EU00247) points are not at the border**: RINF
  files Röszke's at Röszke station, 7 km inside Hungary, and Jimbolia's 6 km off the Kikinda
  line (4.725 km from Jimbolia on 100A, 45.7522 N 20.7011 E). Serbia's 201 and 202 end at the
  real crossing under those ids; the report proposes `borders.MOVE` for both.

## Still open

- Mala Krsna - Požarevac greyed though trains run (gtfs_served fix in the report).
- Bosnia's feed is the 2025 timetable; refetch when a 2026 one appears.
- Stale OSM route relations on lines with no train are still built as OSM lines: Kičevo
  (mk), Fushë Kosovë - Shkup (mk and xk), Shkodër - Vorë and Shkodra - Tirana (al).
- No colours: no operator in the six publishes line colours.
- No English station names beyond OSM's `name:en`.
