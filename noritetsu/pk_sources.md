# Pakistan register sources

## Build (2026-10-08)

Built: `pk_register.py` (the reader, nafrica's recipe with id_register's km posts),
`pk_lines.py` (the line list, one comment per line on what runs), `rinf_countries/pk.py`,
`rules/pk.py`, `colours/pk.csv`, `check_model.REGISTER["pk"]` and `KNOWN["pk"]`.

    python extract.py --region pk --pbf data/raw/pakistan-latest.osm.pbf --station-areas
    python pk_register.py --fill        # after EVERY extract (rewrites data/proc/pk/stops.pkl)
    python build_model.py --region pk --register pk_register:data/raw/pk
    python build_tiles.py --region pk
    python check_model.py --region pk
    python pk_register.py --dry         # the conversion alone, with its log
    python pk_register.py --trace "Lahore Junction" "Wagah"
    python pk_register.py --find "narowal|shahinabad"   # OSM stations by any name tag

**Result**: 28 register lines, 6,572 km: 20 running (4,789 km), 8 greyed (1,782 km). Plus 7
OSM named trains (PR's route=train relations) and Lahore's Orange Line (OSM, 25.3 km). 648
stations.

**Checks** (check_model): the 7 lines with PR km posts (ML-1, Lodhran - Khanewal via Multan,
Rohri - Quetta, Quetta - Chaman, Sher Shah - Kot Adu, Khanewal - Wazirabad, Shorkot - Lala
Musa) median 0.997 of their own chainage, none off by more than 5%. Outside figures: ML-1
1,678.5 / 1,682 (WP 1,687 from Kiamari, less 5), Quetta - Chaman 141.9 / 142 (the Chaman
Passenger's), Rohri - Quetta 379.8 / 384 (WP ML-3 526 less 142). Whole train routes over
several lines (`PATH_CHECKS`, build log "PK path check", published distances from the train
articles on en.WP): Jaffar Express 1,628 / 1,632 (via Multan), Khushhal Khan Khattak Express
1,504 / 1,512 (by ML-2), Fareed Express 1,251 / 1,250, Hazara Express 1,576 / 1,594, Kohat
Express 176 / 177, Faiz Ahmed Faiz Passenger 85.4 / 84, Chaman Passenger 141.9 / 142, Thal
Express 571 / 595 (0.96; the one loose figure). Orange Line 25.3 / 27.1 (OSM's routes run
station to station).

### How the reader works

- **Points** are station names (OSM's name, name:en and alt names, with "Railway Station",
  "Junction", "Cantonment"/"Cantt" folded), each taken as the OSM station of that name nearest
  the line's previous point, optionally with "|km", PR's km post from the RDT. One OSM node per
  station across lines (OSM maps many stations twice, as the station and a stop position).
- **Stations OSM lacks** (`--fill`): 69 list stations (Narowal Junction, Shahinabad Junction,
  Bahawalnagar Junction, many ML-1 halts) are added from Wikidata's coordinate (QLever,
  `data/raw/pk/wikidata_stations.json`, 986 station items), put on the nearest OSM track if
  within 1.5 km; 6 unnamed OSM station nodes take their list name. Gujranwala City is left out:
  Wikidata puts it on Gujranwala.
- **km**: a chained line's sections take the difference of their posts; a post that disagrees
  with the track on both sides while the span across it agrees is out of place and is
  interpolated (logged: ML-1's Yousafwala 1,066 -> 1,073.6, ML-3's Mangoli 101 -> 110 and
  Peshi 298 -> 292.3, Shorkot - Lala Musa's Chak Sher Muhammad 285 -> 280.9). Lines with no
  posts are measured on the trace and ship no km_official.
- **Stops**: `osm_stops: "all"`: every OSM rail station on a traced section (OSM has only 9
  PR train routes, so "a route stops there" would leave most halts out).

### Decisions (Anita's standing rule: decide, record here)

- **ML-1 by the chord**: PR's km posts run Lodhran - Khanewal by the chord (Shahidanwala,
  Dunyapur, Jahania; 91 km, the Allama Iqbal Express calls there), so the chord is ML-1;
  Lodhran - Multan - Khanewal (136 km, posts from 0 at Lodhran) is its own line, running (most
  expresses).
- **Kiamari - Karachi City** (ML-1's first 5 km) left out: no passenger train, and neither OSM
  nor Wikidata places Kiamari station.
- **ML-2 cut in four** where service changes: Kotri - Habib Kot running (the Mohenjo Daro
  Express, restored 21 September 2026 after six months off); Habib Kot - Jacobabad is ML-3's
  (it has the posts); Jacobabad - Kashmor - Dera Ghazi Khan **greyed** (only the Khushhal Khan
  Khattak Express ran there, suspended since May 2026 for fuel costs); Dera Ghazi Khan - Kot
  Adu running (the DGK Shuttle, 2025); Kot Adu - Attock City running (Thal Express, Attock and
  Jand Passengers).
- **ML-3 / Quetta**: Rohri - Quetta **running**. The Jaffar Express is scheduled daily and keeps
  coming back after each attack or security suspension (days at a time, e.g. February, March,
  May and August 2026); the Bolan Mail is suspended since May 2026 (fuel). Quetta - Chaman
  **greyed**: the Chaman Passenger, its only train, is one of the eight trains suspended in May
  2026 (Bolan Mail, Khushhal Khan Khattak, Mehran, Chaman Passenger, Marvi, Saman Sarkar,
  Mohenjo Daro, Ravi; propakistani.pk and pakobserver.net, 20-30 May 2026); only the Mohenjo
  Daro Express is reported back (September 2026).
- **ML-4** Spezand - Koh-i-Taftan **greyed**: no passenger train since February 2020 (the
  Taftan Express and the twice-monthly Zahedan Mixed; Wikipedia: "no longer runs"). Built to
  the border point XIRPK1 (OSM's track over OSM's boundary, 550 m west of Taftan station),
  proposed for borders.EXTRA. Iran left Zahedan - Mirjaveh out of its register, so nothing
  joins on the far side.
- **The Qila Sattar Shah bridge**: 30 of its 110 piers fell in the July 2026 floods; the Badar
  and Ghouri Expresses (Lahore - Faisalabad) are suspended, the Mianwali Express goes via Lala
  Musa, and no reopening was found by October 2026. OSM has no track over it. Shahdara Bagh -
  Sangla Hill **greyed**, and Shorkot - Sheikhupura **greyed** (the Ravi Express is suspended
  since May 2026, and the Waris Shah Passenger ran to Lahore over that bridge).
- **Running branches**: Khanewal - Wazirabad (Millat, Hazara; the Wazirabad Passenger,
  Faisalabad - Wazirabad, for Sangla Hill - Wazirabad: the least certain call, kept running);
  Shorkot - Lala Musa (Sandal Express via Jhang, back August 2026; Hazara, Millat, the Sargodha
  and Pind Dadan Khan Shuttles); Chak Jhumra - Chiniot - Shahinabad (Millat); Sargodha -
  Kundian and Daud Khel - Mari Indus (Mianwali Express, Attock Passenger); Lodhran - Raiwind
  (Fareed Express; Bulleh Shah Passenger Lahore - Pakpattan since 17 July 2026); Shahdara Bagh
  - Narowal and Wazirabad - Sialkot - Narowal (Allama Iqbal Express, Faiz Ahmed Faiz and
  Narowal Passengers, Shaheen Passenger, Lasani and Sialkot Expresses); Golra Sharif - Basal
  and Jand - Kohat (Kohat Express, Thal Express); Taxila - Havelian (Hazara Express, Rawalpindi
  Passenger); Malakwal - Pind Dadan Khan (Pind Dadan Khan Shuttle); Sher Shah - Kot Adu (Thal
  Express, DGK Shuttle); Hyderabad - Mirpur Khas (the Shah Latif Express; the Mehran and Saman
  Sarkar suspended in May 2026).
- **Greyed branches**: Mirpur Khas - Khokhrapar - Zero Point (the Marvi Express only, suspended
  May 2026; the Thar Express to India ended 2019); Lahore - Wagah (the Samjhauta Express ended
  August 2019); Samasata - Bahawalnagar - Minchinabad (no passenger train since the 2000s; OSM's
  track ends short of Amruka).
- **Left out** (no OSM track, or none worth greying): Hyderabad - Badin (Badin Express ended
  2020; no track in OSM), Bahawalnagar - Fort Abbas (no track), Narowal - Chak Amru (no train
  found, OSM's track has gaps), Kohat - Thal, Nowshera - Dargai, Mardan - Charsadda, the Khyber
  Pass line, Bostan - Zhob, Sibi - Harnai (the Harnai Passenger ran 2023-24), Daud Khel -
  Lakki Marwat, Larkana - Jacobabad, the Karachi Circular Railway (not running).
- **Named trains**: all nine PR route=train relations in OSM (rules/pk.py). The suspended
  ones' track (Zahedan Mixed over ML-4, Khushhal Khan Khattak over Jacobabad - DG Khan,
  Chaman Mixed) is greyed register track, owned by nobody for completion.
- **No running crossing**: Wagah - Attari and Khokhrapar - Munabao have carried nobody since
  2019; no border points proposed for them.
- **RDT corrections**: Serai Alamgir's "1,365" is a typo (left out); Shorkot - Lala Musa's
  "325" at Lala Musa is Khanewal - Wazirabad's end figure (OSM: 9 km from Akhtar Karnana at
  303, the line 313 km), so that last section is measured on the track.
- **Colours**: en.WP's list of lines gives ML-1 to ML-4 a colour box each (#4444CC, #CC44CC,
  #44CC44, #CC4444); the branches are our own (`picked`).

### Known gaps

- Station names are OSM's (Urdu `name`, with `name:en`); the 69 Wikidata-filled stations have
  an English name only.
- `osm_stops: "all"` also takes OSM halts that may be closed (OSM keeps some on ML-1).
- RABTA's train-itinerary endpoints (below) would give PR's own stop lists; not used.

## Survey (2026-10-08)

Research only: nothing built, no existing file touched. Samples in `data/raw/pk/survey/`
(about 0.4 MB). Every source below was reached without a login, a key or an account.

### The short answer

- **No open line register with geometry exists.** Pakistan Railways (PR) publishes route
  totals only (Year Book). OSM names 36% of main-line track km for its line, so Korea's
  named-track recipe does not work.
- **en.wikipedia's route diagram templates carry PR km posts for the main lines**: ML-1
  (Kiamari 0 ... Peshawar), ML-2 (Kotri - Attock), ML-3 (Rohri - Chaman), ML-4
  (Quetta/Spezand - Taftan), Khanewal - Wazirabad, Shorkot - Lala Musa, Sher Shah - Kot Addu.
  The rest of the branches have stations in order but no km.
- **Recommended recipe: a hand line list traced by rinf.py**, the za / nafrica pattern, with
  Indonesia's twist (id_register: Wikipedia km posts as `chain` / `km_official` where the
  RDT has them). A new `pk_register.py` with `LINES`, each line its stations in order from
  the RDT or the line article, traced over OSM track preferring the ways of the OSM
  `route=railway` relation that names the line (`own`, as za does). Which branches are
  running comes from PR's train list (below).
- **Expected size**: about 25-35 register lines; PR's network is 7,791 route km (BG 7,479,
  MG 312). Running passenger track is likely 5,500-6,500 km; much of the rest is closed
  branches (Zhob Valley, Bannu - Tank, Kohat - Thal, Nowshera - Dargai, Mardan - Charsadda,
  Khanpur - Chachran, Larkana - Jacobabad...) to be greyed or left out.
- **Extract**: Geofabrik `asia/pakistan-latest.osm.pbf`, 149 MB (2026-10-08).

### Sources

| source | what it gives | licence | sample |
|---|---|---|---|
| en.wikipedia route diagram templates (category "Templates for railway lines of Pakistan", 49 pages; fetch via `api.php?action=query&prop=revisions`) | Stations in order with PR km on 7 lines: ML-1 RDT 227 rows / 83 with km (from Kiamari; halts often blank), Kotri - Attock RDT 116 / 101, Rohri - Chaman RDT 51 / 43, Khanewal - Wazirabad RDT 46 / 37, Shorkot - Lala Musa RDT 64 / 41, Quetta - Taftan RDT 43 / 23, Sher Shah - Kot Addu RDT 17 / 9. Station lists without km for the others (Lodhran - Raiwind, Shorkot - Sheikhupura, Wazirabad - Narowal, Shahdara Bagh - Chak Amru / Sangla Hill, Hyderabad - Khokhrapar / Badin, Lahore - Wagah, Orange Line...) | CC BY-SA | `wp_rdt_templates.json`, `rdt_ml1.wikitext` |
| en.wikipedia "List of railway lines in Pakistan" and line articles | the line inventory: 4 main lines (ML-1 1,687 km, ML-2 1,246, ML-3 526, ML-4 632) and about 15 branch lines; "Stations" sections in order (ML-1's marks abandoned halts "(Abandoned)") | CC BY-SA | `wp_list_lines.wikitext`, `wp_ml1.wikitext` |
| en.wikipedia "List of named passenger trains of Pakistan" | every named train with its ends and an "Operated" column ("1940 - Present", "Suspended", "2011-Suspended, 2026-Present"); each train article lists its stops. This decides which branches carry passengers | CC BY-SA | `wp_trains_list.json` |
| Wikidata (via QLever, qlever.dev/api/wikidata; WDQS was rate-limited to 1 query/min on 2026-10-08) | 45 railway-line items with P17 Pakistan (17 with P2043 length, 3 with P402 OSM relation); 1,226 station items, 947 with coordinates; station adjacency (P197 + P81) on 11 lines, 571 station-line pairs: Karachi - Peshawar 239, Kotri - Attock 138, Rohri - Chaman 67, Hyderabad - Khokhrapar 33, Quetta - Taftan 29, Orange Line 26, Karachi Circular 15, Khyber Pass 13, Lahore - Wagah 9 | CC0 | `wikidata_counts_pk_bd_lk_np.json` |
| PR Year Book 2024-25 (`pakrail.gov.pk/images/yearbook/yearbook2024-25.pdf`, 40 MB; editions back to 2015-16 on `pakrail.gov.pk/YearBook.aspx`) | route km 7,791 (BG 7,479, MG 312), track km 11,881, 460 stations excluding halts (30 June 2024/2025). No per-line or per-section table | none stated (government publication) | `yearbook2024-25.txt` (text only; the PDF was deleted, too big) |
| RABTA, PR's ticketing app (`pakrailways.gov.pk/train`, API `isapi.pakrailways.gov.pk/v1/`) | `GET /ticket/getStations` is open (no token): 342 stations with PR code, city code and English name, no coordinates. The JS bundle also names `/ticket/trainInfo/trainInfoList` (POST), `/ticket/trainInfo/stopTimeTable/<id>` (GET), `/ticket/getTrainItinerary` and `/ticket/searchAvailableTrains` (POST, `boardStationCode`, `arrivalStationCode`, `travelDate`), all marked `isToken:false`, but the right request bodies were not worked out (E0119 / E0001 errors; `stopTimeTable/1` returns an empty list). `/ticket/trainTimes` needs a login | terms not found | `rabta_getStations.json` |
| OSM, Overpass (overpass.kumi.systems; the main instance was overloaded) | measured below | ODbL | `osm_track_survey.json` |
| pakrail.gov.pk fare tables (`/images/fareRatesTable/Passenger_Fare_20_04_2025.pdf`) | fare slabs; not opened | | |

Not found: any GTFS (Mobility Database has no Pakistan feed; Transitous none), an official
station-km table, a working timetable book online. trainstracking.com claims 117 active PR
trains "from the official timetable" but is a commercial aggregator with no stated licence;
not used. The Pakistan Bureau of Statistics yearbook (pbs.gov.pk) and opendata.com.pk have
route-km time series only.

### OSM (measured 2026-10-08, Overpass, non-service track by `usage`)

| track | km | named | on a `route=train` relation |
|---|---|---|---|
| rail, usage=main | 5,728 | 2,061 (36%) | 4,136 (72%) |
| rail, usage=branch | 1,572 | 482 (31%) | 19 |
| rail, no usage | 534 | 126 | 13 |
| narrow_gauge | 80 | 7 | 0 |
| subway (Lahore Orange Line) | 51 | 51 | |

(The km come from overpass.kumi.systems, whose relation queries were patchy that day: it
returned no `route=railway` relation at all, against the 33 the main instance listed earlier,
so re-measure relation coverage with `inspect_region.py` on the extract.)

Total non-service rail about 7,830 km against PR's 7,791 route km, so the track is all
there. Named track mostly carries generic names ("Karachi-Lahore-Peshawar main railway line"
1,494 km in Urdu, "Pakistan Railway" 403 km), so it does not separate lines.

Relations (main Overpass instance, earlier the same day): 9 `route=train` (Rehman Baba
Express Peshawar - Karachi, Khushhal Khan Khattak Express via ML-2, Bolan Mail, Sukkur
Express, Subak Kharam, Chaman Mixed both ways, Zahedan Mixed; colour #7FB17F on most), 29
`route=railway` + 4 `route=tracks` infrastructure relations naming lines (ML-3 with
ref, ML-4, Lahore - Wagah, Hyderabad - Badin, Lodhran - Raiwind, Lodhran - Khanewal,
Shorkot - Sheikhupura, Wazirabad - Narowal, Shahdara Bagh - Chak Amru, Bahawalnagar - Fort
Abbas, Sangla Hill - Kundian, Nowshera - Dargai, Mardan - Charsadda, Kohat - Thal, Mirpur
Khas - Nawabshah, Daud Khel - Bannu, Lakki Marwat - Tank, Zhob Valley, Khyber Pass...), and
2 `route=subway` for the Orange Line (Punjab Masstransit Authority, #F7943A). 665 station /
halt nodes in Pakistan's outline.

### Urban rail

- **Lahore Orange Line** (27 km, 26 stations; Wikidata has its chain): OSM line, as other
  countries' metros. Its route relations are complete with colour.
- **Karachi Circular Railway**: not running (partial revival 2020-21 lapsed); greyed or left
  out. Karachi's Green/Red lines are BRT, not rail.
- Rawalpindi - Islamabad and Peshawar circular railways are plans only.

### Recommended recipe

1. Extract with `--station-areas` (as nafrica), then `python inspect_region.py --region pk`.
2. `pk_register.py`: `LINES` written from the RDTs (stations in order + PR km → `chain`,
   `km_official`) for ML-1, ML-2, ML-3, ML-4, Khanewal - Wazirabad, Shorkot - Lala Musa,
   Sher Shah - Kot Addu; from the line articles' station lists (km traced, `no_chain`) for
   the other running branches. Trace with rinf.py, `own` = the OSM `route=railway`
   relation of the same name.
3. Which lines run: the trains list's "Present" trains and their articles' stops. Running
   branches look like Hyderabad - Mirpur Khas - Khokhrapar (Mehran, Shah Latif, Marvi),
   Lodhran - Pakpattan - Raiwind (Bulleh Shah Passenger, revived 2026), Wazirabad - Narowal
   and Shahdara Bagh - Chak Amru (Narowal and Faiz Ahmed Faiz Passengers), Wazirabad -
   Sialkot (Shaheen Passenger), Multan - Kot Addu - D.G. Khan (DGK Shuttle 2025), Mari
   Indus - Kundian - Sargodha - Lahore (Mianwali Express), Rawalpindi - Havelian
   (Rawalpindi Passenger), Mari Indus - Attock (Attock Passenger); closed branches greyed
   (`suspended`) where OSM still has track, else left out.
4. Stations: OSM station nodes; Wikidata items (947 with coordinates) fill gaps, RABTA's
   342 codes as the station-code key. `listed_only` as nafrica (OSM keeps many abandoned
   halts on ML-1).
5. Check: `check_model.REGISTER["pk"]` from the RDT km (ML-1 Karachi City - Peshawar Cantt,
   ML-2 Kotri - Attock City 1,246 or Wikidata 1,519, ML-3 523/526, ML-4) and the Year Book
   total.

### Open questions (for the country agent to decide)

- **Security suspensions on ML-3 / Quetta**: the Jaffar Express (Quetta - Peshawar) was
  hijacked in March 2025 and Quetta services were suspended on and off afterwards. Check
  what runs Sibi - Quetta - Chaman and Quetta - Taftan - Zahedan in late 2026 before
  calling ML-3 / ML-4 running.
- **Cross-border**: Attari - Wagah (Samjhauta) and Munabao - Khokhrapar (Thar Express) have
  carried nobody since 2019 (India's build stops at Attari and Munabao); Marvi Express runs
  Mirpur Khas - Khokhrapar on the Pakistani side. Quetta - Zahedan mixed train: Iran's
  register already includes the Zahedan side (`ir_sources.md`); a `borders.EXTRA` point at
  Koh-i-Taftan / Mirjaveh would be needed.
- **ML-1 km**: the RDT starts at Kiamari (0); Karachi City is km 5. Keep PR's chainage or
  re-zero at Karachi City? Wikipedia's "1,687 km" is Karachi - Peshawar; Wikidata's 1,872 is
  the CPEC upgrade scope.
- **RABTA**: the train-itinerary endpoints are open in principle; working out their request
  bodies (Anita's browser network tab on `pakrailways.gov.pk/train` would show one call)
  would give PR's official stop lists and maybe distances. Not needed for a first build.
- Metre-gauge 312 km in the Year Book: mostly closed lines (Mirpur Khas - Nawabshah, Zhob
  Valley's narrow gauge is separate); no MG passenger service found.

### Downloads Anita must do by hand

None. The extract (149 MB) is the managing session's usual step.
