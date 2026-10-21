# Running costs: what real metros spend per car-km, and a rule for the game

Research note, 2026-10-21 (T-806). The game charges US$3 per 20 m car-km (docs/track-and-service.md
6.4), before the 100x economy. All figures below are real figures, before that scaling.
**[SRC]** = read in the source. **[DERIVED]** = my arithmetic on sourced numbers (inputs stated).
**[EST]** = my estimate. Currency: **market exchange rate** of that year unless it says PPP, since
construction is priced at market-rate Korea/Spain costs. Inflation to 2026 is not applied (add
roughly 20-30% to the 2017-2018 figures).

## Summary

| Metro | Year | Per car-km | What it includes | Tag |
|---|---|---|---|---|
| Barcelona (TMB, FMB) | 2017 | €3.09 = **US$3.49** | operation, staff, energy, outsourced upkeep of track and stations; no depreciation | SRC |
| Barcelona, all-in | 2017 | €3.63 = US$4.10 | plus depreciation and financial costs | SRC |
| Seoul Metro lines 1-8 | 2025 | ~₩8,500-9,500 = **US$6.0-6.7** | total cost incl. depreciation, interest | DERIVED |
| Tokyo Metro, O&M only | FY2018 | ~¥650 = **US$5.9** | direct work and staff; no admin, depreciation, tax | DERIVED |
| Tokyo Metro, all railway costs | FY2018 | ~¥1,000 = US$9.1 | everything | DERIVED |
| China, national average | 2022 | ¥23.49 CNY = **US$3.49** | "operating cost" (CAMET); scope not stated | SRC |
| China, later year | ~2024 | ¥35.28 CNY = US$4.9 | same, 35 cities | SRC |
| London Underground | 2024 | ~US$6.2 **PPP** | CoMET-style opex (operations, maintenance, admin) | SRC (imputed by source) |
| Paris, London, Berlin | 2013 | ~US$6 **PPP** | CoMET opex | SRC |
| New York (NYCT subway) | 2023 | **US$10.93** | NTD opex; no depreciation | DERIVED from $17.59/vehicle-mile |
| **Game today** | | **US$3.00** | all running costs | |

Cheap, efficient operators sit at about **US$3.5-6 per car-km** before depreciation; New York is
about twice that. The game's US$3 is just under the cheapest real figure, which matches the intent
in 6.4 (fares can cover it). Cars are 18-22 m in all these systems, close to the game's 20 m.

**How the cost splits** (Tokyo Metro and Osaka yardstick, Barcelona, Beijing; details below):

| Part | What drives it | Share of opex | Typical cheap-metro size |
|---|---|---|---|
| Car maintenance | cars, car-km | 10-15% | US$0.6-1 per car-km [EST from Tokyo ¥8M/car/yr] |
| Traction energy | car-km | 8-11% | ~2 kWh/car-km, US$0.25-0.55 per car-km |
| Train crew and control | train-hours, not length | 10% (20% of the yardstick items) | US$2-7 per train-km (driver alone ~US$60/train-hour) |
| Track, power, signals upkeep | route-km | 10% | US$0.5-1.3M per route-km a year |
| Stations (staff, cleaning, escalators, ticketing) | stations | 20-25% (43% of the yardstick items) | US$1-3M per station a year |
| Admin, overheads, tax | everything | 15-25% | |

So only about a quarter of a real metro's opex grows with car-km. The rest is per train or fixed.
The best natural experiment: London's DLR added a third car to its trains and ran **46% more car-km
for 5% more cost** (2010/11-2013/14, TfL). Real costs per car-km fall steeply as trains get longer.

## Recommendation

**Option B: US$9 per train-km plus US$2 per car-km.** A 9-car train costs US$27/train-km, as now;
8 cars US$25, 10 cars US$29. A 4-car train costs US$17 (US$4.25 per car-km), a 12-car one US$33
(US$2.75). One rule, a base plus a rate, like the fare curve. Keep upkeep (option C) for later, or
skip it; it is the honest fix for empty-country lines but adds a cost that runs with no trains.

| Option | Rule | 4-car train | 9-car train | 12-car train | Notes |
|---|---|---|---|---|---|
| A | US$3 per car-km (today) | 12 | 27 | 36 | cost per car flat; short trains cost the same per seat as long |
| **B** | **US$9/train-km + US$2/car-km** | **17** | **27** | **33** | fixed costs folded into the per-train part |
| C | US$6/train-km + US$1.50/car-km, plus upkeep US$0.6M per route-km and US$1M per station a year | 12 | 19.5 | 24 | upkeep: about US$1,640 per route-km and US$2,740 per station a real day |

(US$ per train-km. Upkeep in C would be shown per real day like the others, then scaled 100x:
US$164k per route-km and US$274k per station per game day.)

Why B's train part is bigger than the real crew cost: station staff and track upkeep do not grow
with train length, so if they are not charged separately they belong with frequency, not cars.
US$9 per train-km at ~33 km/h is about US$300 a train-hour, against ~US$60 for a Madrid driver
alone; the rest stands for the fixed costs.

What each option does to play:

- **Short trains, high frequency.** A 4-car train every 2.5 min against an 8-car train every 5 min
  (same car-km): A charges both the same; B charges the short trains 36% more; C 33% more on the
  moving part. That is the real trade (DLR above; also why metros run long trains). Driverless
  lines (Paris 14, Copenhagen, Shenzhen's new lines) escape most of it; if the game ever gets an
  automated-line flag, halving the train part would model that.
- **Long lines through empty country.** A 40 km line, 6 stations, 4 trains an hour each way for
  18 hours, 4-car trains: A US$69k a day, B US$98k, C US$151k (half of it upkeep). Under A and B a
  thin line costs little when it runs few short trains; only C makes the track itself cost money.
- **A typical city line** (20 km, 18 stations, 12 an hour each way for 18 h, 9 cars): A and B
  US$233k a day, C US$250k. C's rates were chosen to keep this about the same.
- **Close station spacing.** Only C charges for it (per station), which is real: stations are the
  biggest single fixed cost in Japan's yardstick data.

## Details

**Barcelona (TMB metro, FMB), 2017.** 92.0M car-km. Operating cost per car-km €3.087: staff 1.885
(61%), outside services 0.819 (27%), electricity 0.270 (9%), supplies 0.105 (3%). Total with
depreciation (0.324) and financial costs (0.223) €3.63. Excludes train leasing and L9/L10 concession
fees. FMB maintains its own track and stations, so upkeep is in. €1 = US$1.13 (2017 average).
TMB Informe anual 2017, pp. 37-38: https://ddd.uab.cat/pub/infanu/47225/iaTMBa2017icat1.pdf

**Seoul Metro (lines 1-8).** Total cost 2025 ₩3.1996 trillion (revenue ₩2.3728 trillion, loss
₩826.8bn; read in search excerpts, not the full pages):
https://www.sidae.com/article/2026061215283820934 and https://www.dt.co.kr/article/12067238 . Cost per passenger ₩1,817, fare ₩1,036, cost recovery 57%:
https://www.hankyung.com/article/202606121978Y . Trains ran 42.04M km in 2024:
https://news.tf.co.kr/read/life/2180176.htm . [DERIVED] At 8-9 cars a train (lines 1-4 run 10
cars, 5-7 eight, 8 six) that is 340-380M car-km, so ₩8,400-9,500 per car-km, US$6.0-6.7 at
~₩1,420. Includes depreciation and interest, and the cost of free rides for over-65s. Electricity
₩274.3bn (2025), ~₩760 = US$0.54 per car-km including stations:
https://www.heraldk.com/article/2026022218205323250 . Korail's metropolitan lines: not found.

**Tokyo Metro, FY2018 (MLIT data in JICA's O&M handbook, p. 61).** Railway operating expenditure
¥301.3bn; O&M expenditure (direct work and staff, excluding admin, depreciation, taxes, welfare)
¥193.9bn; 9,666 staff; 195.1 km. Toei (Tokyo's city-run subway) in the same table: ¥126.0bn /
¥79.8bn. https://www.jica.go.jp/activities/issues/transport/handbook/v4lphn0000004n35-att/handbook_eng_02.pdf
Car-km is not in it; [DERIVED] Tokyo Metro ran 33.69M train-km in FY2011
(https://www1.mlit.go.jp:8088/common/001023424.pdf ), about 300M car-km at ~9 cars, giving ¥1,000
and ¥650 per car-km. ¥1 = US$0.0091 (2018). Osaka Metro: no per-car-km figure found.

**Japan yardstick, FY2005 (MLIT).** The regulator sets fare benchmarks from five cost items, each
priced per its own driver: track upkeep per track-km, power and signals per km of wire, car
maintenance per car, train operation (crew and control, no energy) per route-km, station work per
station. They are 52% of the ten subway operators' railway opex. Tokyo Metro: stations ¥55.3bn
(43%), train operation ¥25.9bn (20%), cars ¥20.0bn (16%), track ¥14.5bn (11%), power and signals
¥12.2bn (10%). Unit rates: station ¥329M a year (Osaka ¥249M, Fukuoka ¥114M); car ¥8.0M a year;
track and wires together ~¥145M per route-km (Osaka ~¥117M).
https://www.mlit.go.jp/kisha/kisha06/08/080731/03.pdf (¥110/US$).

**China.** National average 2022 ¥23.49 per car-km, revenue ¥11.24 (CAMET 2022 statistics, via The
Paper): https://m.thepaper.cn/newsDetail_forward_23247455 . ¥35.28 for 35 cities in a later year
(Railway Journal; year not stated, page blocked):
https://www.railjournal.com/passenger/metros/chinas-metros-reach-record-length/ . Shanghai and
Shenzhen do not publish per-car-km figures; Shanghai's group accounts are not public. Beijing 2022
(Beijing Subway Operation Co.): staff nearly half of cost, repairs ~¥2.0bn, electricity ¥1.3bn,
security screening ¥1.3bn (11%): https://m.jiemian.com/article/10405234.html

**New York, London, Paris.** MTA 2023 benchmarking: NYCT subway US$17.59 per vehicle-mile,
maintenance US$9.92 of it, of which facilities (track, signals, stations) US$6.87; 23% above the
CoMET average per vehicle-mile, two-person crews. https://www.mta.info/document/182116 (pp. 8-10).
London ~US$6.2 PPP (2024), New York US$11.58 (2024), London, Paris and Berlin ~US$6 PPP (2013 CoMET),
Japan ~US$5: https://pedestrianobservations.com/2026/05/22/quick-note-on-subway-operating-costs/ .
LU 21% above the CoMET average, DLR 2% below; DLR 3-car trains, +46% car-km for +5% cost:
https://content.tfl.gov.uk/rup-20150212-part-1-item08-international-benchmarking.pdf (pp. 4-7).

**Crew and energy.** Madrid metro driver base pay €37,713 a year
(https://theobjective.com/economia/2024-05-27/conductor-metro-madrid-sueldo-maquinista/ ), 1,635
hours a year under the agreement
(https://www.vozpopuli.com/espana/madrid/comunidad-madrid-renueva-nuevo-convenio-colectivo-metro-mejora-servicio-fomenta-conciliacion-laboral-sg.html ).
[EST] With extras and ~30% employer charges, ~€58k, ~1,000 hours at the controls: ~US$60 a
train-hour, ~US$2 per train-km. Japan's train-operation item (conductors, dispatch, supervision
included) is ~¥770 = US$7 per train-km [DERIVED]. Madrid traction energy 1.99 kWh per car-km (2021):
https://www.eldebate.com/espana/madrid/20220606/metro-madrid-entre-tres-mas-eficientes-mundo.html
Strong returns to density and scale in train-service costs (24 metros):
https://www.worldtransitresearch.info/research/5831

## Uncertain

- Seoul and Tokyo per-car-km figures rest on my train-length assumptions (8-9 cars) and, for Tokyo,
  a 2011 train-km figure against 2018 costs. Treat them as ±15%.
- China's "operating cost" scope (depreciation in or out) is not stated in the secondary sources.
- Madrid, Shanghai, Shenzhen, Osaka Metro and Korail: no published per-car-km cost found.
- The station and per-route-km upkeep sizes come from Japan (2005), where stations are staffed;
  Spanish and Korean stations are cheaper. Option C's US$1M a station is set near Fukuoka's.
