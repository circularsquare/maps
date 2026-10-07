# Georgia (ge), Armenia (am), Azerbaijan (az), Abkhazia (xa): sources, measurements, recipe (built 2026-10-03; xa 2026-10-04)

## Built (2026-10-03)

| cc | register lines | register km | greyed (no train) | OSM lines | stations |
|---|---|---|---|---|---|
| ge | 18 | 948 | 271 km on 5 lines | 16 (8 train lines, 5 named trains, 2 metro, 1 funicular) | 240 |
| am | 10 | 573 | 93 km on 2 lines | 8 (6 train lines, 1 named train, 1 metro) | 102 |
| az | 17 | 1,595 | 702 km on 7 lines | 11 (5 train lines, 2 named trains, 4 metro) | 143 |
| xa | 1 | 156 | 52 km on 1 line (Guma - Ochamchira) | 3 (3 named trains) | 33 |

One register line is one tariff section, named by its two ends as OSM names their stations in
the country's language ("სამტრედია-1 — ხაშური", "Գյումրի — Մասիս", "Hacıqabul — Böyük
Kəsik"), operator the railway (საქართველოს რკინიგზა, Հարավկովկասյան երկաթուղի, Azərbaycan Dəmir
Yolları), one picked colour per railway (colours/<cc>.csv). Points no OSM station names keep
Book 1's Russian name (posts and loops: "Пост 2865 км", "Кармир-Блур").

**Checks.**
- Against the register's own chainage (every line of 2 km or more, `km_official`): median
  0.994 (ge, 17 lines), 0.997 (am, 10), 0.995 (az, 17). Off by more than 5% are lines built
  only in part (Brotseula - Tskaltubo, Rioni - Tkibuli, Ingiri - Jvari: the rest is disused
  track in OSM), Navtlughi - Gardabani 1.12 (OSM's track through Rustavi is not the tariff
  route; a 9 km piece Gachiani - km 23 is rejected), and three short Yerevan junction pieces.
- Against outside figures (`REGISTER` in check_model.py; de.wikipedia's route-diagram
  chainage for "Bahnstrecke Poti–Baku" and "Bahnstrecke Tiflis–Jerewan", Wikidata P2043):
  Samtredia-1 - Khashuri 121.0 of 124.0 (0.98), Khashuri - Navtlughi 125.2 of 125.7 (1.00),
  Senaki - Samtredia-1 27.3 of 28.1 (0.97), Natanebi - Ozurgeti 18.4 of 20.0 (0.92; tariff 18);
  Gyumri - Masis 140.2 of 140.2 (1.00), Ayrum - Gyumri 143.9 of 143.1 (1.01); Hacıqabul -
  Böyük Kəsik 373.7 of 375.1 (1.00), Ələt - Hacıqabul 44.0 of 42.9 (1.03), Yevlax - Balakən
  163.4 of 165.0 (0.99).
- Abkhazia: Ԥсоу — Очамчыра (Psou — Ochamchira) 156.3 against ru.wikipedia's route diagram
  ("Абхазская железная дорога": Psou 1998.0, Ochamchyra 2153.0, so 155.0), 1.01; against Book
  1's chainage 0.995 (157 tariff km from the export code at the bridge).
- OSM metros against en.wikipedia (proposed `KNOWN`, which is shared): Tbilisi Akhmeteli-
  Varketili 19.5 of 19.6, Saburtalo 7.8 of 7.7; Baku Line 1 20.0 of 20.1, Line 3 5.9 of 6.1,
  Line 2B 2.2 of 2.3; Yerevan 12.1 of 13.4 (Wikipedia's figure takes in more than the
  stations; not proposed, and its OSM name "1" is too short a fragment for KNOWN).

## The register: Russia's tariff guide, Book 1

`data/raw/ru/tr4_kniga1_2026-09-30.xls` (the file the ru and ua builds read) has a sheet for
each of the three railways: **Грз** (Georgian Railway, road 57, 25 sections, Abkhazia's and
South Ossetia's included), **Ю-Кав** (South Caucasus Railway, Armenia, road 56, 14 sections)
and **Азерб** (Azerbaijan Railways, road 55, 32 sections, Nakhchivan, Karabakh and the
Horadiz - Meghri - Julfa line through Armenia included). Every station, halt and post in
order with its ESR code and integer km. `caucasus_register.py` reads the three sheets once
and writes each country's pairs (both points inside it) into rinf.py's input files.

Other registers looked at: Wikidata has 49 station items with an ESR code in the three
countries (used for placement), and line lengths only for a few (used as checks). OSM's
`route=railway` relations: 1 in Georgia, 1 in Armenia, 5 in Azerbaijan. No line register
otherwise.

**Placement.** Georgia: 324 ESR codes on OSM nodes, nearly every station. Armenia: 59.
Azerbaijan: 2, so Azerbaijan places by name: Book 1's Russian names against OSM's Azerbaijani
ones (31 of 214 OSM stations carry name:ru) through a consonant skeleton both spellings share
(`skeleton`: Гянджа / Gəncə, Евлах / Yevlax, Гаджигабул / Hacıqabul, Тауз / Tovuz; v and f are
left out because Russian writes Azerbaijani "ov" as a vowel). Of several candidates, the one
whose distances from the placed neighbours best fit the tariff km wins (Пойлы is Poylu, 13 km
from Ağstafa, not Yeni Poylu 2 km away); metro stations never (Tbilisi's metro station
Samgori took the Kakheti line's halt Самгори before). Pinned by hand: Baku's Soviet names
(Нариманов, Кишлы, ОП Кишлы, Беюк-Шор, Сумгаит, Сумгаит-Новый, Г.З. Тагиев-Сортировочная),
Alyat (Baş Ələt, Yeni Ələt), and two junctions by Masis at the OSM junction nodes their km fit
(Пост 2865 км, Разъезд 9 км). An ESR node far off its line is ignored (568205, Аревик on the
Gyumri - Maralik line, is tagged on a halt by Lake Sevan). 225 of 766 points stay unplaced,
nearly all unnamed halts ("Платформа 54 км"); rinf.py traces past them.

Sumgait: Book 1's "Сумгаит" is 11 km from Hacı Zeynalabdin and 7 km from "Сумгаит-Новый", which
fits OSM's Sumqayıtçay (name:ru Сумгаит-Главный) and makes the ring's Sumqayıt station
Сумгаит-Новый. The ring's Bakı - Xırdalan - Sumqayıt route then runs 7.7 km over track no
tariff section has (owned by the OSM line).

## Which trains run (2026)

**Georgia**: Georgian Railway's feed (Transitous, generated at jbb.ghsq.de from GR's ticket
system; ODbL; 1 Oct 2026 - 28 Jan 2027). Tbilisi - Batumi (801-808, three pairs a day), Tbilisi
- Zugdidi, Tbilisi - Poti (once a day each), Tbilisi - Ozurgeti (every other day), Tbilisi -
Gori - Borjomi (two pairs), Kutaisi - Batumi (two pairs), Kutaisi - Zestafoni - Sachkhere,
Tbilisi - Rustavi - Gardabani (two pairs), Zestafoni - Khashuri (two pairs), and the
international Tbilisi - Yerevan and Tbilisi - Baku. The feed has no Kakheti, Akhalkalaki,
Tkibuli or Tskaltubo trains, and the Borjomi - Bakuriani narrow gauge has not run since 2020
(restoration announced for about January 2027; thegeorgianguide.com, en.wikipedia).
**Greyed**: Marabda - Akhalkalaki (127 km), Navtlughi - Gurjaani (94; Kakheti), Khashuri -
Vale beyond Borjomi (43), Brotseula - Tskaltubo (7), Rioni - Tkibuli (1). Of 994 register km
before the drop, 666 served.

**Armenia**: the South Caucasus Railway's feed (Transitous, jbb.ghsq.de; ODbL; 5 Jun 2026 -
13 Jun 2027, so the summer trains are in it). Yerevan - Gyumri (local and express), Yerevan -
Araks, Yerevan - Yeraskh, Yerevan - Hrazdan - Sevan - Shorzha (summer, 51 days in 2026:
seasonal is drawn running), Yerevan - Tbilisi (every other day), Yerevan - Batumi (summer).
**Greyed**: Shorzha - Sotk (75 km), Gyumri - Maralik (19). Hrazdan - Dilijan - Ijevan is in
the register but could not be traced (no OSM stations of its names, no path at Hrazdan) and
has no trains. Of 573 register km, 451 served.

**Azerbaijan**: ADY publishes no feed; ady.az, ticket.ady.az (also its timetable PDFs under
/storage/pages/) and rail.day.az answer 403 to scripts. `--timetable` writes
`data/raw/gtfs/az/az_ady_handlist.gtfs.zip` from `AZ_TRAINS`, built from ADY's announcements
in the press:
- Baku - Gazakh, two pairs a day via Kürdəmir, Ucar, Ləki, Yevlax, Gəncə, Ağstafa (report.az,
  15 Jun 2025; Kürdəmir stop from Feb 2026, modern.az)
- Baku - Balakən night train via Şəki (resumed 3 Jan 2025; times changed 1 Jul 2026, apa.az)
- Baku - Qəbələ (2023; weekend extras summer 2026, apa.az 29 Jun 2026); Gəncə - Qəbələ (from 12
  Jan 2026)
- Baku - Ağdam via Yevlax, Bərdə, Köçərli, Təzəkənd: Saturdays, plus holiday extras (report.az
  30 Aug 2025; 1news.az Feb 2026). Once a week is the edge of "more often than about once a
  week": drawn running, as the only train on Yevlax - Ağdam and a regular weekly one.
- Gəncə - Mingəçevir - Ağstafa, twice a day (from 28 Sep 2025, report.az)
- the Absheron ring (Bakı - Keşlə - Sabunçu - Pirşağı - Sumqayıt - Xırdalan - Biləcəri -
  Dərnəgül - Bakı, 84-100 trips a weekday) and Bakı - Xırdalan (apa.az, Aug 2026)
- Baku - Tbilisi, daily since 26 May 2026 (in Georgian Railway's feed, copied in)

**Greyed**: Osmanlı - Horadiz (and the Nakhchivan end of the same section, Ordubad - Culfa;
195 km), Sumqayıt - Yalama (162; no passenger trains, "Bakı-Yalama" listed by ADY among routes
whose resumption is most requested), Salyan - Astara (136; ADY, Sep 2026: the line cannot carry
passengers until rebuilt, 2027-2028), Culfa - Şərur (108; Nakhchivan: no source says a
passenger train runs; the railway there is being rebuilt from late 2025, and ADY says its
trains "will be able to operate up to Nakhchivan" after the projects), Güzdək - Qaradağ (38,
freight bypass), Bakı - Zirə (35), Alabaşlı - Bayan (29). Of 1,621 register km, 830 served.
Horadiz - Ağbənd (the new line to the Armenian border, due 2026) and Ağdam - Xankəndi are
under construction and not in OSM as track: not built.

The Gabala branch (Ləki - Qəbələ, opened 2023) is not in Book 1; its track is the OSM line's
(Bakı - Qəbələ, 739/740).

## Lines and named trains (`rules/<cc>.py`)

- **Georgia**: lines are the regional trains (6xx, 63xx, 64xx) and Tbilisi - Batumi (801-808,
  three pairs a day, the corridor's interval service); named trains are the once-a-day
  long-distance trains (Zugdidi, Poti, Ozurgeti every other day) and the international ones
  (Yerevan, Baku). `--clip` groups Georgian Railway's numbered trains by their two ends, so the
  three Tbilisi - Batumi route_masters are one line.
- **Armenia**: lines are the suburban trains and Yerevan - Gyumri (local and express); named
  trains Yerevan - Tbilisi and Yerevan - Batumi.
- **Azerbaijan**: lines are the Absheron ring and the 7xx regional day trains (Gazakh, Gabala,
  Aghdam); named trains the Balakən night train (63/64) and Baku - Tbilisi.
- Metros (Tbilisi 2 lines, Baku 4, Yerevan 1) and the Tbilisi funicular are OSM lines.

## Territory (who the map depicts)

Anita's de facto rule, applied as follows (all in `caucasus_register.py`'s docstring):
- **Abkhazia is its own region, `xa`** (Anita, 2026-10-04: "for abkhazia maybe we should make
  it its own country?"). See "Abkhazia" below. Georgia's line stops at Ingiri, as before.
- **South Ossetia** (Gori - Tskhinvali, 57-017): no train since 2008. Its part inside South
  Ossetia is no region's; the Georgian-held first kilometres from Gori are drawn (5 km, the rest
  is disused track in OSM).
- **Karabakh** (Yevlax - Ağdam - Xankəndi) and **Nakhchivan**: Azerbaijan's, as everyone's
  trains there are ADY's (Ağdam weekly; Nakhchivan none).
- Outlines: Georgia as drawn is OSM relation 28699 less 1152720 and 1152717
  (`data/raw/ge/outline.geojson`, for tools/build_regions.py); Abkhazia is OSM relation
  1152720 (`data/raw/xa/outline.geojson`); Armenia and Azerbaijan are religiondots' outlines,
  unchanged. Natural Earth's admin-0 file has no Abkhazia, so its name ("Abkhazia") comes from
  borders.NAME, for tools/build_regions.py and the border point's name.

## Abkhazia (xa)

**The code.** `xa`, user-assigned as Kosovo's `xk` is. Nothing in the pipeline reads a region
code as ISO except the outline and name lookups (borders.outline, tools/build_regions.py), and
both take an explicit entry (borders.OUTLINE, borders.NAME); the app names a region from
regions.json. `iso3` in rinf_countries/xa.py is only read by rinf.py's --fetch.

**The register.** Book 1's Грз sheet: 57-001 "ГАНТИАДИ (ЭКСП.) - СЕНАКИ" starts at the export
code 574704 at km 0, the handover to Russia's 51-032 "ТУАПСЕ-СОРТИРОВОЧНАЯ - ВЕСЕЛОЕ (ЭКСП.)",
which ends 2 km past Veseloe: the Psou bridge. OSM carries 574704 on Psou halt, 0.4 km east of
the bridge. Abkhazia's part runs to Тагилони (km 190) before the Inguri, and 57-007 Ochamchira -
Akarmara (36 km) is the Tkvarcheli branch. One line, the Abkhazian part of 57-001, as in every
tariff-guide country:
- **Built from the Psou bridge to Ochamchira** (157 tariff km, 156.3 km traced), "Ԥсоу —
  Очамчыра" / "Psou — Ochamchira". OSM has no railway beyond Ochamchira nor on the Tkvarcheli
  branch; ru.wikipedia says the track from Achguara to Ingiri was damaged or taken up in the
  1990s and that the branch carries coal only. `TRACK_END` leaves those pairs out (115 tariff
  km), so the line is named by where its track ends rather than by an untraceable Taglan.
- **Names**: OSM's `name`, which in Abkhazia is Abkhaz ("Аҟəа", "Гəдоуҭа", "Афон Ҿыц"); English
  from OSM's name:en where it romanises Book 1's name or OSM's own Russian one (Tsandrypsh is
  Book 1's Gantiadi). A Georgian `name` on a node there gives way to OSM's Russian, else Book
  1's. Ochamchira's ESR node is its yard, which carries the station's names, so it is read.
  Points with no OSM name keep Book 1's Russian ("Пицунда", "Калдахвара", "Багнашени").
- **Operator**: the Abkhazian Railway (Абхазская железная дорога; rows' manager code "57A",
  `IM_OF`), whose track it is; one picked green (colours/xa.csv).
- **Two ESR nodes unplaced** (`UNPLACE`): Бармыш (574136) is 4.4 km of track past Мюссера where
  Book 1 has 7 and ru.wikipedia's diagram 6-8, and Цицквара (574013) 3.2 km from New Athos where
  Book 1 has 7 and the diagram 6. With them the trace rejected two sections and broke the line
  in three; unplaced, the trace steps over them and the line is whole.

**The trains** (poezdato.net's Sukhum station page and train pages, 2026-10-04; the hand feed
`XA_TRAINS`, data/raw/gtfs/xa/xa_handlist.gtfs.zip):
- 304М/304С Moscow (Kazansky) - Sukhum and 479А/480С St Petersburg (Vitebsky) - Sukhum, FPC,
  "по особому графику" (on set dates); in Abkhazia they call at Tsandrypsh (border control),
  Gagra, Gudauta, New Athos, Sukhum.
- 929С/930Ж "Dioskuria" Olympic Park - Sukhum, daily, and 925Э/926Й Olympic Park - Guma (set
  dates); calls Tsandrypsh, Abaata, Gagripsh, Gagra, Bzypta, Gudauta, Psyrtskha, New Athos,
  Sukhum (Guma).
- Nothing runs beyond Guma (the Sukhum - Ochamchira page on transport.marshruty.ru: no train).
  **Greyed**: Guma - Ochamchira, 52 km. Psou - Guma, 104 km, runs.
- **Named trains, all of them** (rules/xa.py): Russia's rules make every one of these OSM
  relations a named train (three digits and a letter), so they are here too and the same
  relation is the same kind of line on both sides. The Dioskuria is a tourist train of a few
  trips a day, not a line a rider uses as one. Their route_masters (m19256388, m17137776,
  m19256669) are the ids Russia's build ships, so the app joins them over the border.
- The New Athos cave railway (OSM route=subway 16248679 and its tourism track) is a ride inside
  the cave, part of the cave tour: left out of the extract.

**The border with Russia.** One crossing, the Psou bridge: point `eXARUPSOU` (40.008443,
43.393733), where way 124797554, the only track over the river, crosses OSM's boundary of
Abkhazia; "Abkhazia – Russia border". Book 1's 574704 is that point (BORDER, BORDER_XY), so
Abkhazia's line starts there with Gyachrypsh 2.2 km on; Psou halt, where no train stops, is not
on the line. Russia's 51-032 needs ru_register.BORDER ("XARUPSOU", Veseloe 532701, its export
code 532608, "at") to end there too, and borders.EXTRA needs the point (proposed in the report
of 2026-10-04). Until then the two builds do not share the id and the line stops at the bridge
on Abkhazia's side.

**Georgia - Abkhazia** (the Inguri): no track, no trains, no point.

## Borders

Crossings with passenger trains, both with no RINF point, so ours (`BORDER` / `BORDER_XY`;
both countries' register lines end at the same id):
- **eXAMGE1, Sadakhlo - Ayrum** (44.898762, 41.211857): the OSM node on the Debed bridge where
  ways 48754465 / 1103672429 cross the boundary. Yerevan - Tbilisi, Yerevan - Batumi.
- **eXAZGE1, Gardabani - Böyük Kəsik** (45.165160, 41.405885): ways 453284165 and 1186547878
  (two tracks 3 m apart) over the boundary. Baku - Tbilisi.

No passenger trains, no point: Georgia - Türkiye (Akhalkalaki - Kartsakhi on the BTK: freight;
the "full launch" of 2 June 2026 was the corridor, and no Tbilisi - Kars passenger timetable
exists; tr_register leaves the BTK out too), Azerbaijan - Russia at Samur/Yalama (no Baku -
Moscow train since 2020; Russia's Derbent - "2454 km" electric train stops at the border on the
Russian side), Azerbaijan - Iran at Astara and Julfa, Armenia - Türkiye (Gyumri - Akhuryan -
Kars, closed), Armenia - Azerbaijan (Ijevan - Gazakh, Yeraskh - Sadarak, Meghri: closed or
not built), Georgia - Abkhazia (the Inguri: no track left; see "Abkhazia").

## How it was built

    # sources (no login; User-Agent "noritetsu-build/1.0 (hobby rail map)")
    #   jbb.ghsq.de/gtfs/ge-georgian-railway.gtfs.zip -> data/raw/ge/ (ODbL)
    #   jbb.ghsq.de/gtfs/am-railway.gtfs.zip          -> data/raw/am/ (ODbL)
    #   polygons.openstreetmap.fr/get_geojson.py?id=<rel>&params=0 for 28699, 364066, 364110,
    #     1152720, 1152717 -> data/raw/<cc>/*_boundary.geojson
    #   Wikidata (MediaWiki API: haswbstatement:P2815 with P17 each country, then
    #     wbgetentities; WDQS was rate-limiting to 1 request a minute) -> data/raw/ge/wd_stations_caucasus.json
    python caucasus_register.py --esr ge data/raw/georgia-latest.osm.pbf     # and am, az: 1-2 s each
    python caucasus_register.py --outline --timetable                        # --timetable: xa's hand feed too
    python caucasus_register.py --clip ge                                    # and am, az: after every extract
    python caucasus_register.py --clip xa                                    # after Georgia's: cut from data/proc/ge/full
    python build_model.py --region xa --register caucasus_register:data/raw/rinf/xa    # 30 s
    python build_model.py --region ge --register caucasus_register:data/raw/rinf/ge    # 30 s each
    python caucasus_register.py --colours ge                                 # then build_model again
    python build_tiles.py --region ge
    python check_model.py --region ge

The .pbf files are no longer needed once `--esr` has run: everything else reads data/proc/<cc>
(and data/proc/<cc>/full for a re-clip). To refresh the timetables, refetch the two jbb.ghsq.de
feeds into data/raw/ge and data/raw/am, edit AZ_TRAINS from ADY's news, and rerun
`--timetable` and the builds.

## Sources and licences

| source | licence | used for |
|---|---|---|
| Tariff Guide No. 4, Books 1-2 (sovetgt.org), the Грз, Ю-Кав, Азерб sheets | official intergovernmental document, its preface calls the data publicly available | the register |
| OSM via Geofabrik (georgia, armenia, azerbaijan, 2026-10-03), boundaries via polygons.openstreetmap.fr | ODbL | track, stations, ESR codes, routes, metros, outlines |
| Georgian Railway and South Caucasus Railway feeds (Transitous, jbb.ghsq.de) | ODbL as Transitous lists them | which sections trains run over |
| poezdato.net (Sukhum station page, train pages 304, 930) | facts only | XA_TRAINS, Abkhazia's hand feed |
| ru.wikipedia "Абхазская железная дорога" | CC BY-SA (figures only) | Abkhazia's outside check, where its track ends |
| ADY's announcements in the Azerbaijani press (report.az, apa.az, modern.az, 1news.az, far.az, trend.az) | facts only | AZ_TRAINS, the hand-made Azerbaijani feed |
| Wikidata | CC0 | station places, line lengths |
| de.wikipedia route diagrams | CC BY-SA (figures only) | outside checks |

Tried and not usable: ady.az, corp.ady.az, ticket.ady.az (and its PDFs), rail.day.az: 403 to
scripts (browser URLs: https://ticket.ady.az/hereket-cedveli). WDQS answered 429 throughout.

## What is still off

- Azerbaijan's feed is a hand list: no times, and intermediate calls only where the press
  names them; a train's path credits the sections between its calls. Recheck AZ_TRAINS when ADY
  adds or drops a route (Baku - Astara is expected back after 2027-2028; Nakhchivan's trains
  after its rebuilding; Ağdam - Xankəndi and Horadiz - Ağbənd on opening).
- Lines on track OSM tags disused (Kakheti east of Gurjaani, Gurjaani - Telavi, Tkibuli,
  Tskaltubo, Gori - Tskhinvali, Kazreti, Akhaltsikhe - Vale) are built only as far as the
  track is live; none has trains. ua_register's `--disused` recipe would draw them greyed.
- Unnamed halts are not stops (no OSM node to place them at): the Yeraskh line's halts and
  Ayrum - Gyumri's "Платформа" stops are stepped over by the timetable check.
- Book 1's Georgian sheet is a Soviet-era list: the Tbilisi - Batumi trains' bypass at Rikoti
  (28 km) and the Rustavi - Gachiani line are owned by the OSM lines, not a tariff section.
