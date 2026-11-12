# City-pair trip figures for capping the demand model (pull2 research, 2026-11-11)

All figures converted to **both directions, a day = annual / 365** unless marked. "Line" means a count on a
railway line or route (includes intermediate stations and through passengers), not the city pair.
"FR24" = the project's own `maps/flights/routes_fr24.json` (two days in May 2026, flights x seats x modelled
load factor, annualised x182): airport-pair passengers **including connecting passengers**, so on hub feeder
routes (FCO-NAP, FRA-DUS, WAW-KRK, CDG-x, MUC-x, IST-ESB) it overstates city-pair trips. Script:
`fr24pairs.py` in this folder.

## Table

| Pair | Rail, both ways a day | Air, both ways a day | Coach / car / all-mode | Cap I'd use (all modes, a day) | Confidence |
|---|---|---|---|---|---|
| Berlin - Hamburg | **Line**: about 20,000 long-distance passengers a day on the Hamburg - Berlin connection (DB board member Peterson, DB press release 11 Jun 2026, https://www.deutschebahn.com/de/presse/pressestart_zentrales_uebersicht/Bahn-sagt-Danke-fuer-die-Geduld-100-000-Extra-Schnaeppchentickets-zur-Wiedereroeffnung-der-Strecke-Hamburg-Berlin-13900604); DB's own earlier material and ZDF said 30,000 a day before the 2025-26 closure (https://www.zdfheute.de/wirtschaft/bahnstrecke-berlin-hamburg-freigegeben-100.html). City-pair share not published | 0 (no flights; FR24 none) | none found (car and FlixBus are large, unpublished) | **20,000** (the lower DB line figure; it over-counts rail's city pair and leaves out car and coach, so it is a stand-in, not a bound either way) | low |
| Rome - Naples | Trenitalia AV only, Naples - Rome: 3.0M in 2018 = **8,200** (FS press note 3 Jun 2019, https://fsitaliane.it/content/dam/fsitaliane/Documents/old-media-ed-eventi/comunicati-stampa-e-news/anno-2019/giugno/2019_06_03_NS_10_anni_AV.pdf). Italo not included | negligible (Cascetta: "air demand is negligible on such OD pairs"); FR24 FCO-NAP 1,047 = connecting passengers | **All modes, average day, 2010**: car 6,719 + HSR/Eurostar 7,185 + Intercity 2,287 = **16,192** (Cascetta & Coppola, survey-corrected OD, Table 7 "average daily city-to-city flow estimates", https://art.torvergata.it/retrieve/e291c0d4-c49f-cddb-e053-3a05fe0aa144/Cascetta-Coppola%20ETC2011_HSR-Italy.pdf; same table in disP 49(2) 2013). Direction not stated; both ways inferred (one-way would make 2010 HSR 5.2M/yr, above FS's 3.0M for 2018) | **20,000** (2010's 16.2k all-mode plus Italo since 2012 and Trenitalia's growth, minus car) | medium |
| Moscow - St Petersburg | Jan-Sep 2024: 8.3M "between the two capitals by train" (all trains; Sapsan about 4M) = **30,300** a day over those 9 months (RZD via Delovoy Peterburg, 18 Oct 2024, https://www.dp.ru/a/2024/10/18/sapsani-za-leto-perevezli); full year about 11M | 2025: 2.868M on Moscow - St Petersburg (Aeroflot citing Rosaviatsia, https://samokatus.ru/travel-telegram-feed/aeroflot_pr/1250) = **7,860**; FR24 9,180 | none found (M-11 car, coach unpublished) | **38,000** (rail + air; lower bound since car and coach are out) | medium-high for rail+air |
| Istanbul - Ankara | **Line**: YHT Ankara - Istanbul line 16,723 a day (Transport Minister Uraloglu, Mar 2026, via Milliyet https://www.milliyet.com.tr/haber/7554439/); includes Eskisehir, Bilecik, Izmit etc. | FR24 ESB-IST + ESB-SAW 3.89M = **10,670** (includes transfers at IST). No DHMI/SHGM route table found | none found (intercity coach is big; Obilet says Istanbul - Ankara is the most-booked bus route, no count) | **27,000** (YHT line + air pair; both over-count their city share, coach and car left out) | low |
| Madrid - Barcelona | 2024: **8,944,328** AVE/AVLO/Ouigo/Iryo on the Madrid - Barcelona relation = **24,500** (CNMC INF/DTSP/269/25, Table 1, https://www.cnmc.es/sites/default/files/6606989.pdf). The corridor as a whole carried 14.6M (CNMC annual rail report 2024, includes Zaragoza, Lleida, Tarragona etc.) | Derived from CNMC's real shares (rail 79.9%, air 18.2%, bus 2.0%, Table 5): about 2.04M = **5,580**; FR24 4,079 | CNMC modelled all modes 13,961,154 = **38,250** (car 20% is modelled, "cannot be checked for lack of real data"). Observed collective modes (rail+air+bus) 11.19M = **30,650** | **30,650** (observed rail+air+bus; take the modelled 38k only if car must be in) | high |
| Madrid - Valencia | 2024: **5,298,112** = **14,520** (CNMC same table) | CNMC real share 6.6% of collective: about 0.38M = **1,040**; FR24 609 | CNMC modelled all modes 11,843,254 = **32,450** (car 51%, modelled). Observed collective 5.74M = **15,730** | **15,730** observed; 32k if modelled car is accepted. Rule says lower: 15.7k | high (observed), medium (all-mode) |
| Warsaw - Krakow | PKP IC H1 2019: 1.3M = **7,180** (avg day over H1) (IC via T-3240; H1 2018 1.2M, https://www.intercity.pl/pl/site/o-nas/dzial-prasowy/aktualnosci/pkp-intercity-sprawdzilo,-dokad-najchetniej-jezdza-polacy.html). IC total 48M (2019) to 89.2M (2025): scaled about 13,400 (estimate, not published) | FR24 WAW-KRK 1,227 (mostly LOT connections) | none found | **12,000** (rail, scaled; lower bound) | medium for rail 2019, low for scaling |
| Warsaw - Tricity | PKP IC H1 2019 about 0.95M = **5,250**; H1 2018 Gdansk - Warsaw nearly 0.9M (same IC release). Busiest IC route in 2020 summer (30% above Krakow) and Q2 2024 | FR24 WAW-GDN 869 (mostly connections) | none found | **10,000** (rail scaled with IC's growth; lower bound) | low-medium |
| Berlin - Munich | DB: 4.4M "Reisende zwischen Berlin und München" in the first year of the new line (Dec 2017 - Dec 2018) = **12,050**, likely includes some intermediate travel (https://www.l-iz.de/wirtschaft/mobilitaet/2018/12/Die-Bahn-hat-ihre-Fahrgastzahlen-zwischen-Muenchen-und-Berlin-mehr-als-verdoppelt-247632) | Munich Airport 2025: Berlin 959,477 = **2,630** (both ways, incl. transfers; https://www.munich-airport.de/_b/0000000000000031515196bb67ed2d3f/annual-traffic-report-2024.pdf p.10); FR24 2,472 | DB + Telefonica mobile-data split one year after opening: rail 46%, air 30%, car/bus 24% (same l-iz article). 4.4M / 0.46 = 9.6M = 26,200 | **18,000** (T-1508's rail+air, kept; air has halved since 2018 while rail grew) | medium |
| Hamburg - Munich | none found | Munich Airport 2025: Hamburg 1,306,482 = **3,580** (incl. transfers); FR24 4,000 | none found | **7,000** (air x2 for rail, a guess) | low |
| Paris - Lyon | 2008: Paris - Lyon/St-Etienne train group 8,869,242 both ways = **24,300** (Setec for RFF, LGV POCL traffic study, June 2011, p.4, https://cpdp.debatpublic.fr/cpdp-lgv-pocl/DOCS/DOCUMENTS_DEBAT/ETUDES_MO/ETUDES_DE_TRAFIC/PERSPECTIVES_DE_TRAFIC_ET_DE_CI.PDF). SNCF: Paris - Lyon traffic about 20% above 2019 in 2024 (voyages-d-affaires.com) | FR24 CDG/ORY-LYS 1,615 (mostly connections) | none found | **28,000** (rail 2008 plus some growth; lower bound, car left out) | medium |
| Milan - Rome | Trenitalia AV Rome - Milan 3.6M in 2018 = **9,860** (FS note above) | ENAC 2019: Linate-Fiumicino not among routes over 620k one way (Linate shut Jul-Oct 2019); FR24 2026 2,758 | FS 2018 modal split: train 69%, air 19.5%, car 9.4% (same FS note). 3.6M/0.69 = 5.2M = 14,300; FR24 air / 0.195 = 14,100 | **14,000** | medium |
| Beijing - Shanghai | **Line**: Jinghu HSR company's own-line trains 52.0M in 2024 = 142,500 (all stations; 36Kr citing the company report, https://www.36kr.com/p/3333263540464134). City pair not found | 2024: 8.613M on the Beijing - Shanghai air route = **23,600** (Hangban Guanjia data via 36Kr); 2023 6.9M; FR24 24,760 | none | **50,000** (air 23.6k plus a rail city share I estimated at 25-30k; floor 23.6k) | low |
| Paris - London | Eurostar London - Paris about 8.3M in 2024 = **22,700** (T-809, sources there) | FR24 5,351 (2026) | T-809: about 30k a day all modes | **28,000** (rail + FR24 air) | medium-high |
| Paris - Brussels | Thalys about 3.5-4M + Ouigo 1M = **12-13k** (T-809, half of Thalys's network flow; from TourMaG and Thalys) | FR24 CDG-BRU 775 (connections) | none | **12,000** | medium |
| Vienna - Budapest | not found (ÖBB/MAV give no route figure) | FR24 556 (connections) | none | none to cap with | - |
| Singapore - Kuala Lumpur | none | about 4M a year = **11,000** (Malaysiakini, pre-2020); OAG 2024 5.4M seats; FR24 KUL/SZB-SIN 4.41M = **12,070** (includes hub connections) | coach: no count found | **11,000** (air alone; coach and car left out) | medium for air |
| Zurich - Milan | all EuroCity Switzerland - Italy 2.4M in 2023 (all routes and ODs) = 6,600 (treinreiziger.nl citing SBB/Trenitalia); Zurich - Milan share unknown | FR24 MXP-ZRH 1,246 (connections) | none | **4,000** (guess: a third of the EC flow plus air) | low |
| Madrid - Lisbon | no through train | FR24 MAD-LIS 1.85M = **5,060** | coach exists, no count | **5,000** | medium |
| Paris - Amsterdam | not found (Eurostar gives only growth: Paris - NL +140k in 2024) | FR24 CDG-AMS 3,486 (AF-KLM hubs, many connections) | none | none firm; air alone 3.5k | low |
| Copenhagen - Stockholm | not found | FR24 CPH-ARN 3,915 (SAS hub, many connections) | none | **4,000** (air) | low |
| Frankfurt - Cologne | not found (about 120 ICE a day on the line, DB) | FR24 FRA-CGN 74 (feeder) | none | none | - |
| Munich - Stuttgart | not searched beyond above | none | none | none | - |
| Beijing - Tianjin | **Line**: Beijing - Tianjin intercity about 73,000 a day (2023, 15-year report; Beijing Daily https://news.bjd.com.cn/2023/08/08/10523576.shtml), plus Jinghu trains | none | none | **70,000** (line figure as a lower bound for the pair) | low-medium |
| Mumbai - Pune | not found | none | not found | none | - |

## Notes per pair

- **Berlin - Hamburg.** DB gives two numbers for the same thing: 30,000 long-distance passengers a day (pre-closure
  material, repeated by ZDF and Nordkurier) and 20,000 (board member Peterson, June 2026). Touristik-aktuell calls
  the 30,000 "local and long-distance". Both are line counts: they include Spandau, Wittenberge, Ludwigslust, Büchen
  and trains running on to Dresden, Prague or Vienna. No city-pair, car or coach figure found; a
  Verkehrsverflechtungsprognose OD matrix exists (BVWP) but is not published as a pair number. The model's 115-128k
  is 4-6x anything published.
- **Rome - Naples.** The only all-mode figure is Cascetta's (2010, 16.2k a day, survey-corrected; air negligible;
  coach not in the table). FS's 2018 Trenitalia-only 3.0M (8.2k a day) is close to Cascetta's 2010 HSR 7.2k, so
  Trenitalia grew little; Italo (since 2012) is the main addition, no route figure found. The earlier 60-80k guess
  has no support; the model's 249k is about 12x.
- **Moscow - St Petersburg.** RZD's 8.3M is Jan-Sep 2024 (summer heavy); a full year is about 11M (my
  extrapolation). 2019 full year not found (H1 2019 Sapsan Moscow - SPb 2.48M). Air: Aeroflot's 2.868M (2025,
  "citing Rosaviatsia", may be Aeroflot group only) against FR24's 3.35M (all airlines, includes connections);
  air fell 18% in H1 2026 (Kommersant). Lower used.
- **Istanbul - Ankara.** The 16,723 is a line average and includes Eskisehir/Izmit/Bilecik trips; TCDD gives no
  city-pair split. A 2023 holiday peak was 17,777 a day on the same line. No air route figure from DHMI/SHGM (the
  SHGM 2023 statistics PDF refused the connection); FR24's 10.7k includes Istanbul Airport transfers. Coach and car,
  likely the biggest modes, unpublished.
- **Madrid - Barcelona / Valencia.** CNMC's totals are rail real figures divided by modelled rail shares: rail, air
  and bus are observed (Table 5's real shares match the model to a point), car is not. Valencia's car share (51%)
  is large and unverified, so the observed 15.7k is the safe cap and 32k the generous one. The CNMC "corridor"
  figures (Madrid-Barcelona 14.6M, Levante 5.6M) are line-like and include intermediate stations; Table 1's
  "relación" figures are the pair. These are strong primary figures; the model's 66k and 88k are 2x and 3-5x.
- **Warsaw - Krakow / Tricity.** Only IC half-year figures (2018, 2019) are published; PKP IC names these as its top
  routes in 2024 but gives no counts. Scaling by IC's total growth (1.86x 2019-2025) is my estimate. Air is LOT
  feeder traffic; car and coach unpublished. T-3240's 25k targets are not supported by anything published; rail
  alone is about 12-13k.
- **Berlin - Munich.** The 4.4M and the 46/30/24 split come from the same DB release but do not fit together: air
  at 30% of 9.6M would be 2.9M, more than flights carried (Munich Airport's 2025 Berlin figure is 0.96M including
  transfers; the 2017-18 air figure was "almost two million" per wissenschaft.de, undated). So the 4.4M probably
  includes trips to and from Leipzig, Erfurt, Nuremberg etc. The 18k (rail + air, T-1508) stays the cap.
- **Hamburg - Munich.** Only air found. Rail (ICE about 6 h) is real but uncounted.
- **Paris - Lyon.** Setec's 8.87M (2008) is the Paris - Lyon/St-Etienne train group, both ways, reconstructed from
  load factors; a few of those riders go to Le Creusot, Macon or Saint-Exupery, and some Paris - Lyon riders use
  PACA trains stopping at Part-Dieu, roughly cancelling. SNCF says the Paris - Lyon axis was about 20% above 2019 in
  2024. No car figure; 45k all-mode is plausible but unsourced.
- **Milan - Rome.** FS's 3.6M (2018) is Trenitalia only; FS's 69/19.5/9.4 split covers all passengers. Two routes
  to an all-mode total agree: 3.6M/0.69 = 14.3k a day, and FR24 air / 0.195 = 14.1k. If the 69% includes Italo
  and 3.6M does not, the true total is higher (perhaps 20k); the rule says take the lower.
- **Beijing - Shanghai.** No city-pair rail figure found in Chinese sources (the 52M is the line company's own
  trains, every station). Air 8.6M (2024) is solid. My 25-30k rail estimate: about 40 through train pairs a day x
  about 1,100 seats x 0.9 load x an end-to-end share near 0.4. Rail is "several times air on the line" (36Kr),
  which is about the line, not the pair.
- **Cross-border.** London - Paris and Paris - Brussels are from T-809 (sourced there). Singapore - Kuala Lumpur:
  air sources run 2M ("6,000 a day from Changi", one way) to 4M; 4M fits OAG's seats and FR24. No coach count;
  T-3240's "buses about as many again as air" is unsourced. Zurich - Milan's 2.4M is every EuroCity between
  Switzerland and Italy (Geneva, Basel, Lugano, Como included), from one Dutch site. Vienna - Budapest, Paris -
  Amsterdam, Copenhagen - Stockholm: no rail figure found after one search each; FR24 air there is mostly hub
  connections, so it is not a floor for the pair either.
- **Frankfurt - Cologne, Munich - Stuttgart, Mumbai - Pune:** nothing found (one search each). Beijing - Tianjin's
  73k is the intercity line (Wuqing, Binhai stops included), 2023.

## What I tried and did not find

DB route statistics (not published per pair), Destatis air by route (not reached), SHGM 2023 route PDF (connection
reset), ENAC 2019 route table (Linate-Fiumicino not in the text list), CNMC 2019 OD, Renfe Statista series (2017:
just over 4M Madrid - Barcelona, Renfe only), Rosaviatsia route table, UTK pair data, SBB/OBB/SJ route figures,
Malaysian/Singapore coach counts, Mumbai - Pune corridor studies.
</content>
</invoke>
