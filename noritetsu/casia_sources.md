# Kazakhstan, Uzbekistan, Kyrgyzstan, Tajikistan, Turkmenistan (kz, uz, kg, tj, tm)

Built 2026-10-03 by the casia agent. One reader, `casia_register.py`, serves all five (its
docstring is how it works); `rinf_countries/<cc>.py` and `rules/<cc>.py` are thin settings.

## Built

| cc | register lines | register km | greyed (no train) | stations | other lines |
|---|---|---|---|---|---|
| kz | 74 | 15,016 | 20 km (Селекционная — Щебзавод) | 1,088 | Almaty metro, Astana LRT, 4 tram lines, 2 suburban OSM lines (Petropavl - Petukhovo, Uzen - Bolashak); 20 named trains |
| uz | 37 | 3,574 | 428 km on 9 lines | 340 | Tashkent metro (4 lines), 2 tram lines; 2 named trains |
| kg | 3 | 285 | 22 km | 46 | Bishkek-1 - Tokmok suburban (OSM) |
| tj | 5 | 406 | 21 km (Kanibadam - Istiklol) | 30 | 1 named train |
| tm | 10 | 2,937 | 646 km on 3 lines | 145 | none |

Register km are what the tariff sections inside each country's OSM boundary trace to, after
build_model dropped junction-ended sections no train and no OSM route runs over (freight
spurs, port lines, the Turkmen lines to Iran and Afghanistan, Balykchy - Kochkor).

**Checks.**
- Against the tariff guide's own km (every line, `km_official`): median 0.995 (kz, 73 lines
  of 2 km or more, 4 off by over 5%), 0.994 (uz, 37, 6 off), 0.975 (kg, 3), 0.992 (tj, 5),
  0.998 (tm, 10).
- Outside figures (`REGISTER`): Ақтоғай — Достық 308.4 of 304 (Wikidata, 1.01); Angren —
  Pop-1 122.4 of 123.1 (Wikidata, 0.99); Türkmenabat — Gazojak 321.2 of 322 (railway.gov.tm's
  timetable km, 1.00); Мары — Serhetabat 293.3 of 316 (0.93: the Saryýazy - Sandykgaçy piece's
  trace is rejected, OSM's track there does not meet the placed points).
- Metros (`KNOWN`): Almaty 13.2 of 13.4 (0.98), Astana LRT 21.1 of 22.4 (0.94, opened 16 May
  2026), Tashkent Uzbekistan line 14.2 of 14.3, Circle line 22.3 of 21.9; Chilonzor 21.6 of 23.7
  and Yunusobod 9.4 of 10.5 have every station and are station to station (published lengths
  take in depot tails).
- Network totals: KTZ's operating length is about 16,000 km; the tariff pairs inside
  Kazakhstan sum to 16,010 km and 15,016 are built. Uzbekistan's sheets inside the country
  4,357 tariff km (UTY: about 4,700), Kyrgyzstan's 393 (KTZh: 424), Turkmenistan's 3,778.
- The timetable: 377 trains (362 from KTZ's site, 15 more by hand), 7,210 calls, 5,794 placed
  (most of the rest are in Russia). Served register km: kz 13,448 of 15,653 traced, uz 2,553
  of 4,017, kg 247 of 391, tj 385 of 588, tm 2,291 of 3,662.

## The short answer

- **The register is the CIS tariff guide, Book 1**, the file Russia's and Ukraine's builds
  already read (`data/raw/ru/tr4_kniga1_2026-09-30.xls`). It has a sheet for each of the five
  administrations: Кзх (road 68, 99 sections), Узбк (73, 59), Кирг (70, 4), Тадж (74, 6), Трк
  (75, 18): every station, halt and post in order with its ESR code and integer tariff km. One
  tariff section is one register line, as in Russia and Ukraine. Book 2 gives each point's
  passenger operations.
- **A sheet is an administration, not a territory.** The Kyrgyz sheet's first section starts
  at Lugovaya in Kazakhstan (Merke and Muňke are Kazakh); the Turkmen sheet's Kerki - Kelif
  line enters Uzbekistan at Talimarjan; the Almaty - Shymkent main line crosses a sliver of
  Kyrgyzstan at Kurkureu-su; Russia's sheets run through Kazakhstan at Petropavl (South Urals),
  Kostanay - Kartaly, the Kulunda line (West Siberian) and Dzhanybek - Saykhin - Shungay on
  Krasny Kut - Verkhny Baskunchak (Privolzhskaya). So every sheet touching the five is read
  (193 sections), and each pair of consecutive points goes to the country both lie in (OSM's
  boundary relations 214665, 196240, 178009, 214626, 223026). The operator shown is the
  administration whose sheet lists the section, the track's country is where it lies.
- **The timetable** is KTZ's ticket site, bilet.railways.kz (robots.txt allows everything). It
  shows every train in the CIS Express system calling at a station on a date, and each train's
  calls: Kazakh trains of every kind (KTZ's suburban 6xxx/7xxx included), Uzbek, Kyrgyz and
  Tajik long-distance trains, and the international ones. Turkmenistan's trains are not in it;
  railway.gov.tm publishes its whole schedule. The rest is written out by hand (below).

## Lines and named trains (`rules/kz.py`, shared by the five)

The project's rule: the line a rider uses is the register line the train runs over. In all
five, passenger service is long-distance and regional trains numbered 1-999 that run once or
twice a day, every other day or weekly (Talgo 1/2 Almaty - Tashkent, the Afrosiyob 7xx,
Moscow - Dushanbe), so the tariff sections are the lines and every numbered OSM train is a
named train. Suburban trains (the CIS 6000-7999 numbers, "пригородный", "қала маңындағы")
are lines: Bishkek-1 - Tokmok 6050/6051, Petropavl - Petukhovo 6975, Uzen - Bolashak
6993/6994. Unnumbered OSM routes here are long-distance trains drawn without a number ("Astana
- Arkalyk", "Қарағанды – Семей"): named trains. Metros, the LRT and trams are lines.

## What runs: sources and decisions

- **KTZ's site.** `--crawl` fetched every station with an Express code in osm.sbin.ru's ESR
  list (1,312) for Wednesday 7 and Saturday 10 October 2026 (odd and even dates, a weekday and
  a weekend day), then each train's route popup (362 trains; 15 popups never answered). Each
  call's name was looked up in the site's station search for its Express code, then in
  osm.sbin.ru's table for the ESR code, kept only where Book 1's name for that code agrees
  (Kazakhstan has renumbered since 2021: sbin's 667909 "Актобе" is Book 1's Кардон). Calls
  are placed by the shortest path through them, with the call times as a speed limit (200
  km/h), so a Russian "Курган" is not a Kyrgyz station.
- **Running days** from the site's "Регулярность": daily, odd/even dates, weekdays, until a
  date, listed dates. A train running fewer than 7 days in the eight weeks 5 October - 29
  November is left out (one: Moscow - Dushanbe 423/424, every other week).
- **Turkmenistan**: railway.gov.tm/en/schedule lists 15 trains with every stop and day
  (Ashgabat - Türkmenabat, - Daşoguz twice, - Serhetabat, - Türkmenbaşy, - Amyderýa,
  Türkmenabat - Gazojak, Balkanabat - Gazojak weekly). Taken whole. No Turkmen train runs to
  Kazakhstan (Bereket - Serhetýaka is greyed), Iran or Afghanistan, or Kerki - Kelif.
- **Tajikistan**: railway.tj's 2026 schedule. Dushanbe - Pakhtaabad 6373/6374 daily and
  Dushanbe - Kulyab 601/602 weekly are not on KTZ's site and are written out by hand;
  Dushanbe - Kanibadam 367/368 (weekly) too, its Tajik part Spitamen - Khujand - Kanibadam
  (it runs in from Uzbekistan at Bekabad, as the old Tashkent - Fergana line ran). Kulyab -
  Volgograd runs Kulyab - Khatlon - Yavan - Dushanbe, not by Hoshadi, so Hoshadi - Khatlon has
  no train and is dropped.
- **Kyrgyzstan**: Bishkek-2 - Kaindy and Bishkek-1 - Tokmok, two daily suburban trains (vb.kg,
  confirmed for 2026 by economist.kg), and the Issyk-Kul train 608/609 Bishkek-2 - Balykchy,
  daily 26 June - 13 September 2026 (economist.kg 2026-05-19, extra runs 3-4 October):
  seasonal, drawn as running (Anita's rule). Bishkek - Moscow and Bishkek - Shu come from KTZ.
  No train in the south (Osh, Jalal-Abad, Tash-Kumyr): those lines are dropped.
- **Uzbekistan**: UTY's own site needs a login for schedules, so its long-distance trains come
  from KTZ's site (the Afrosiyob, Sharq and every regional train are in the Express system).
  Tashkent's suburban electric trains (tashtrans.uz, updated 8 September 2026; daily) by hand,
  directions only: to Khodjikent, to Khavast via Yangiyul - Chinaz - Gulistan, to Angren.
  Greyed (no train): Qarshi - Kitob, Irjarskaya - Jizzax, Termiz - Xotinrabot, Qarshi -
  Nishon, Suvonobod - Qo'qon, Andijon - Savay, Qizilquduq - Muruntau, Sultanabad - Xonobod,
  part of Surxonobod - Kudukli. The Kitob train poezdato.net lists (774/773) was in no
  schedule for the crawled dates.
- **Kazakhstan**: Селекционная — Щебзавод (20 km) greyed. 688 km stay "unknown" (OSM routes but
  no train in the feed: Көкпекті — Карагайлы 247 km, the Ridder line, the Petropavl - Isilkul
  stretch of the South Urals sheet...), drawn as running.

## Borders

`--borders` finds every place a section leaves one of the five (50 crossings,
`data/raw/kz/casia_borders.json`): the boundary crossing of OSM's track nearest to the last
point inside, ids `X<AA><BB><nn>`. Each country's line runs from its last point to that
crossing ("eX..." in the build), so the two sides meet at one id. Crossings with passenger
trains (from the timetable's consecutive calls): Kazakhstan - Russia at Semiglavy Mar/Ozinki,
Iletsk (both lines), Petukhovo, Isilkul, Lokot, the Kulunda line (Karasuk - Irtyshskoye mixed
trains), Kigash (Atyrau - Astrakhan) and four on Krasny Kut - Verkhny Baskunchak (RZD's
Astrakhan trains, by OSM's routes); Kazakhstan - Kyrgyzstan at Merke - Kaindy and twice at
Kurkureu-su; Kazakhstan - Uzbekistan at Saryagash - Keles and Oasis - Karakalpakstan;
Tajikistan - Uzbekistan at Bekabad - Spitamen and Kudukli - Pakhtaabad. **No passenger train
crosses to China** (Altynkol - Khorgos, Dostyk - Alashankou): KTZ's site has none, and OSM's
"13/14 乌鲁木齐<=>Алматы" route is stale. None between Turkmenistan and its neighbours.

## Who the map depicts

No disputed territory carries track here. The Fergana valley's enclaves have no railway; the
Kyrgyz sliver at Kurkureu-su (Almaty - Shymkent main line) is Kyrgyzstan's track by the rule
that border track counts where it lies, though only Kazakh trains run there.

## Sources

| source | where | used for |
|---|---|---|
| Tariff Guide No. 4, Books 1-2 | data/raw/ru (Russia's build fetched them from sovetgt.org) | the register, stops |
| OSM via Geofabrik (kazakhstan, uzbekistan, kyrgyzstan, tajikistan, turkmenistan -latest, 2026-10-03), ODbL | data/proc/<cc> (clipped to the boundary + 0.03°) | track, stations, ESR codes, routes, metros, trams |
| OSM boundary relations | polygons.openstreetmap.fr, data/raw/<cc>/<cc>_boundary.geojson | which country a point is in |
| osm.sbin.ru esr.csv and osm2esr.csv (a 2021 snapshot) | data/raw/kz/sbin_*.csv | ESR to Express codes and country; 2021 OSM nodes for codes OSM no longer tags |
| Wikidata, P2815 items in the five (MediaWiki API; the query service answered 429 "1 req / min") | data/raw/<cc>/wd_stations.json | positions, national-language names |
| bilet.railways.kz station schedules and train routes (crawled 2026-10-03, 1 s apart) | data/raw/kz/ktz/ | the timetable |
| railway.gov.tm/en/schedule | data/raw/tm/railway_gov_tm_schedule.html | Turkmenistan's trains |
| railway.tj/ru/passengers/schedule | data/raw/tj/railway_tj_schedule.html | Tajikistan's domestic trains |
| economist.kg (2026-05-19, 2026-09-29), vb.kg | casia_register.HAND | Kyrgyzstan's suburban and Issyk-Kul trains |
| tashtrans.uz (CC BY-NC-SA; directions only) | data/raw/uz/tashtrans_*.html | Tashkent's suburban electric trains |

Tried and not usable: UTY's ticket site (eticket.railway.uz, now eticket.uzrailpass.uz): the
station list answers, every schedule and route call (`/api/v1/handbook/...`) answers 401
without a login; ticket.railway.kg is a script page with no data in it.

## How it was built

    python casia_register.py --esr kz data/raw/kazakhstan-latest.osm.pbf   # and uz, kg, tj, tm
    python casia_register.py --clip kz                                     # and uz, kg, tj, tm
    python casia_register.py --crawl --names                               # ~110 min, network only
    python casia_register.py --borders --timetable
    python casia_register.py --convert kz --colours kz                     # and the other four
    python tools/rebuild.py kz uz kg tj tm
    python check_model.py --region kz                                      # and the other four

The .pbf files are no longer needed: --esr has written data/raw/<cc>/osm_esr.json and
osm_stations.json, and data/proc/<cc>/full holds the unclipped extract.

## Lines deleted as track to no station (checked 2026-10-08)

`prune_dead_track` deleted kz Тараз — Жаңатас (66.5 km), Ақтаутас — Бугунь (13.2) and uz Nókis —
Shımbay (56.2) on 2026-10-07. None has a passenger train: KTZ's site (crawled for every station
with an Express code) shows none at Жаңатас or Бугунь, and UTY's Nukus trains run to Tashkent
and Mangistau, none to Shımbay. Phosphate and freight branches; left deleted. The lines still
not finishable by picks in kz, uz, tj, tm end at a border point whose next stop is in the
neighbour (Shemonaikha, Shyngyrlau, Xojeli, Takhiatash, Bekobod, 4613 km) or, in tm, at Owadan
Depe, a junction towards the Daşoguz line where no train stops (not in railway.gov.tm's
schedule): not missing stops.

## What is still off

- **672 of 2,348 Book 1 points have no source position** (no OSM ESR node, no name match):
  mostly halts named by line kilometre ("Ост. пункт 1174 км"). Those between two placed points
  are put on the track at their share of the tariff km, the rest are retraced past.
- **Names**: Uzbek and Turkmen stations OSM has show in the national Latin script; the rest
  show Book 1's Russian (Cyrillic) spelling, so a Uzbek line can read "Buxoro 1 — Кашкадарё".
- **Some traces are rejected** where a placed point and OSM's track disagree (Ashgabat's
  northern bypass Бюзмейин - Рзд № 0001 is not in OSM; Saryýazy - Sandykgaçy; Bukhara -
  Miskin's 306 km across the desert, whose points have no position). `cut_at_junctions` keeps
  a rejected piece from taking the rest of its section with it.
- **Timetable gaps**: Uzbekistan's regional suburban trains outside Tashkent (Fergana valley,
  Samarkand) are not in KTZ's system and not written out; their lines are left to OSM's
  routes or greyed. UTY's own timetable would settle them (it needs a login).
- English names: 156 of kz's 1,088 stations, 163 of uz's 340 (OSM name:en and Wikidata, kept
  where they read as a romanisation of Book 1's name).
- Line colours: one picked colour per administration (colours/<cc>.csv, `picked`).
