# National long-trip rates against the model (research for pull2, 2026-11-11)

Model units: one-way trips a resident a weekday, all modes, domestic, straight-line km between
place centres, bands 100-300 and 300+. Conversions used throughout:

- round trips x 2 = one-way trips; annual / 365 = average day.
- route km to straight km: divide by 1.25 (so 300 straight = 375 route; 100 straight = 125 route).
- average day to weekday: x 0.89. This is Japan's own ratio (MLIT 2010: the weekday-only annual
  estimate 1,449M against the all-days estimate 1,624M), borrowed for France, Germany, the US ATS
  and the per-mode countries because long trips lean to weekends and holidays everywhere.
- Where a published band straddles a model boundary (e.g. 200-499 km), the split is an assumption,
  stated in the row. Arithmetic: `scripts/convert.py` in this folder.

## Table

| Country | Published figure (what it counts, year, source) | Converted, one-way a resident a day, 100-300 / 300+ straight | Conservative ceiling (weekday) | Model now / planned 300+ | Confidence |
|---|---|---|---|---|---|
| US | (a) NHTS 2017, project file `research/intercity/nhts_long.json`: weekday, linked trips, route mi x 1.609 / 1.25. (b) ATS 1995 Profile: 1,001M person round trips to US destinations of 100+ route miles one way; household trips by round-trip miles: <300 29.6%, 300-499 26.6%, 500-999 21.4%, 1,000-1,999 11.6%, 2,000+ 10.9% (person trips <300: 30.6%). https://rosap.ntl.bts.gov/view/dot/4664 (Travel profile, Table 1) | (a) weekday: 100-300 0.0237 non-commute, 0.0268 with commutes; 300+ 0.0098, 0.0103 with commutes. (b) average day: 129-300 0.0108 (ATS starts at 161 route km = 129 straight); 300+ 0.0098 (466+ round-trip miles; the 466-499 slice taken pro rata); weekday-scaled 0.0096 / 0.0088 | 100-300: 0.0268 (NHTS all purposes; 0.0237 non-commute). 300+: 0.0098 | 0.0303 / 0.0098 | High for 300+ (two surveys 22 years apart agree to 1%). Medium for 100-300 (see disagreements) |
| Japan | MLIT 6th Inter-Regional Travel Survey (Zenkoku kansen ryokyaku jun-ryudo chosa), FY2015: 1,793M one-way trips a year that cross a prefecture line, all modes; by distance band (route): <100 20%, 100-200 45%, 200-300 12%, 300-500 12%, 500-700 5%, 700-1,000 3%, 1,000+ 4%. https://www.mlit.go.jp/report/press/sogo23_hh_000108.html (reference material p.3-4: https://www.mlit.go.jp/common/001297857.pdf) | Average day: all 0.0386; 300+ route 0.0092 (24%). Straight: 100-300 0.0188 avg / 0.0168 weekday (assumes 68% of the 100-200 band is over 125 km and 55% of 300-500 is under 375 km); 300+ 0.0067 avg / 0.0059 weekday. T-376's weekday figure (4.02M a weekday) gives the same 300+ within 5% | 100-300: no ceiling (MLIT leaves out trips inside a prefecture, so 0.0168 is a floor). 300+: 0.006 | 0.0199 / 0.0046 | Medium-high. Published bands; only the 300-500 split is assumed |
| Germany | (a) INVERMO 2002/03 (Zumkeller et al. 2005), via the KIT dissertation, Table 2: 7.5 long trips (100+ route km one way, round trips) a person a year, 84% domestic, plus 1.3 long-commute trips; domestic trips by 100 km route class by purpose (Abb. 20, read by eye). https://publikationen.bibliothek.kit.edu/1000176300/155563125 (pp. 31, 40). (b) MOP and KONTIV 2002: 1.3-1.47% of all trips are over 100 route km, about 8.5-9 long trips a person a year (same source p. 32-33). (c) ifmo/INFRAS 2014: MiD 2008 and Swiss MZ give 0.05-0.06 one-way trips over 100 km a person a day, all destinations. https://www.infras.ch/media/filer_public/7c/de/7cdef9bf-aaf7-4b43-823a-39b1f681ef56/7248a_ifmo_schlussbericht_140228.pdf (pp. 43, 88). (d) MiD 2017 Ergebnisbericht: only "a good 1%" of car trips are over 100 km; no person-trip split published | INVERMO, domestic: all 80+ straight 0.0414 avg day. 100-300 0.0268 avg / 0.0239 weekday; 300+ 0.0093 avg / 0.0083 weekday (assumes 30% of the 100-199 route class is under 125 km and 75% of 300-399 is under 375 km) | 100-300: 0.024. 300+: 0.008 | 0.0265 / 0.0043, planned ~0.008 | Medium-low. 20+ years old, curve read by eye, but four surveys agree on the 100 km+ total (8.5-9 a year, 0.05-0.06 a day) |
| France | EMP 2019 (SDES/Insee): 374.1M "voyages" (round trips from home) to more than 80 km straight from home, residents 6+, 6.3 a person a year; destination distance from home (straight): <200 49.9%, 200-499 29.5%, 500-899 12.7%, 900+ 7.9%; 85.8% to metropolitan France. https://www.statistiques.developpement-durable.gouv.fr/comment-les-francais-voyagent-ils-en-2019-resultats-de-lenquete-mobilite-des-personnes ; table file https://www.statistiques.developpement-durable.gouv.fr/media/5045/download?inline (sheet 1) | Domestic only (trips abroad assumed 10/25/30/35% across the four bands); per whole resident population 64.8M. All domestic 80+: 0.0271 avg. 100-300: 0.0147 avg / 0.0131 weekday (2/3 of 80-200 over 100 km; 55% of 200-499 under 300). 300+: 0.0073 avg / 0.0065 weekday; 0.0069-0.0077 for a 60/40 to 50/50 split; hard bounds 0.0036 (500+ only) to 0.0118 (all of 200-499) | 100-300: 0.013. 300+: 0.0065 | 0.0228 / 0.0045, planned ~0.011 | Medium. Published straight-line bands; the 200-499 split and the abroad mix are assumed |
| Spain | Ministry of Development big-data pilot (Orange phones, July, August and October 2017): "more than 2 million interprovincial trips of more than 50 km a day", peaks over 4M in summer; mean 221 km on a working day; residents and non-residents. https://www.esmartcity.es/2019/12/17/fomento-publica-resultados-estudio-movilidad-interprovincial-viajeros-aplicando-tecnologia-big-data | No band split published. A d^-1.5 distance curve on 50-1,000 km reproduces the 221 km mean; it puts 24% over 300 km and 38% at 100-300. If the published km are straight: 300+ 0.0102, 100-300 0.0165. If they are route km (/1.25): 300+ 0.0078, 100-300 0.0148. Cross-check by modes: domestic air ~30M trips over 300 km, AVE/long-distance rail ~15M, coach ~15M, so public modes ~60M; at a 40% public share, 300+ is ~0.0087 | 100-300: no ceiling (interprovincial only; 0.015 is a floor). 300+: 0.008 | 0.0163 / 0.0053, planned ~0.015 | Low. One total and one mean; the shape is assumed; non-residents included |
| Poland | No national travel survey with long trips found. GUS tourism survey: 57.7M domestic overnight trips by residents 15+ in 2018 (https://stat.gov.pl/obszary-tematyczne/kultura-turystyka-sport/turystyka/ ; figure via a Jagiellonian University repository summary). PKP Intercity 78.5M passengers in 2024 (https://railwaypro.com/wp/?p=126858) | Overnight trips alone: 0.0083 one-way a day, distance unknown; at 25-40% over 300 km straight (Warsaw - Gdańsk 285, Warsaw - Kraków 253, Warsaw - Wrocław 300 straight sit just under the line) 300+ is 0.0021-0.0033 from these alone (day trips and business trips not counted) | 100-300: none. 300+: 0.003 | 0.0221 / 0.0024 | Low |
| China | NBS 2024 statistical communiqué, Table 6: rail 4.31bn trips, 1,579.9bn pkm (mean 366 km); road (commercial) 11.78bn, 511.7bn pkm (mean 43 km); air 0.73bn (CAAC: domestic 0.665bn incl. 0.0096bn Hong Kong, Macau, Taiwan). https://www.stats.gov.cn/sj/zxfb/202502/t20250228_1958817.html | 300+ straight (375+ route): rail 25-35% of trips (a skewed distribution with a 366 km mean has most trips below it; lognormal sigma 0.9-1.0 gives ~30%), air 93-97%, coach 0.1-0.3bn, private car 0.3-0.8bn (no source). Total 2.1-3.3bn a year, mid 2.6bn: 0.0041-0.0063 avg day, mid 0.0051 avg / 0.0045 weekday. Rail + air alone 0.0033-0.0042. 100-300: not estimable (road statistics have no distance split and leave out private cars) | 300+: 0.005 | 0.0154 / 0.0065, planned ~0.0084 | Low (car volume and rail distance distribution assumed). Counts are boardings: a two-ticket rail transfer or a connecting flight counts twice, so person trips are a little lower still |
| Russia | Russian Railways long-distance (dalneye sledovaniye) passengers: 126.7M in 2025, about 127M in 2024 (https://finance.mail.ru/article/rzhd-v-2025-godu-perevezli-bolee-13-mlrd-passazhirov-69194615/); Rosaviatsia: Russian airlines carried 84.7M on domestic routes in 2024 (https://www.vedomosti.ru/analytics/krupnyy_plan/articles/2025/01/19/1087102-rossiiskie-aviakompanii-uvelichili-passazhiropotok-vtoroi-god-podryad). Mean long-distance rail trip about 800 km (derived from Rosstat's ~100bn pkm, not checked this session) | 300+ straight: rail 65-80% of long-distance trips, air 85-92% (transfers via Moscow count twice), intercity bus 15-45M, car 50-150M (no source). 220-375M a year: 0.0041-0.0070 avg day, mid 0.0056 avg / 0.0050 weekday. Rail + air alone 0.0029-0.0034 | 300+: 0.005 | not computed | Low |
| Turkey | DHMI: domestic airport passengers 91.06M in 2023 (counted at both ends, so ~45.5M trips; 2024 ~47M); Ministry (UETDS): 82.2M passengers on ticketed intercity buses and 60.4M on tour buses in 2024 (https://www.milliyet.com.tr/ekonomi/bakan-uraloglu-2024-yili-karayolu-tasima-verilerini-acikladi-7282025); TCDD: 12.46M high-speed (YHT) passengers in 2024 (https://www.ekonomigazetesi.com/sektor-haberleri/demiryolu/2024te-demiryollarinda-279-4-milyon-yolcu-tasindi-48024) | 300+ straight: air ~95%, intercity bus 30-50%, YHT 30-50% (Ankara - Istanbul is 350 km straight; Ankara - Konya and Eskişehir under 300), mainline ~1.5M, tour buses 4-10M, car 25-70M (no source). 103-175M a year: 0.0033-0.0056 avg day, mid 0.0044 avg / 0.0039 weekday. Scheduled public modes alone 0.0024-0.0030 | 300+: 0.004 | not computed | Low |
| Italy (cheap pass) | Isfort Audimob: trips over 50 km were 2.6% of all weekday trips in 2019 (2.3-2.9% in 2015-2017; https://www.assoporti.it/media/3735/15-rapporto-sulla-mobilit%C3%A0-degli-italiani_sintesi.pdf). No 100 or 300 km split found. Per-mode, from memory and not checked: domestic air ~64M airport passengers (~32M trips), Frecce + Italo ~75M, Intercity ~13M | 300+: air ~30M, high-speed rail 30-40% of ~75M, Intercity ~4M, coach ~8M, car 40-100M (no source): 0.0048-0.0079 avg day, mid 0.0063 avg / 0.0056 weekday | 300+: 0.0055 | 0.0239 / 0.0034 | Low |

## Disagreements and notes

1. **US 100-300.** NHTS 2017 (one-day diary) gives 0.0211 non-commute trips a weekday at 129-300 km
   straight; ATS 1995 (four quarterly recall interviews) gives 0.0108 an average day for the same band,
   about half. At 300+ the two agree (0.0098 both). Recall surveys lose short and routine long trips more
   than memorable far ones, and US long-distance travel grew 1995-2017, so NHTS is more likely right;
   but by the project rule (unclear, take the lower) 100-300 could be argued down to about 0.018
   (ATS plus NHTS's own 100-129 km slice). Either way the model's 0.0303 is over: 1.13x NHTS with
   commutes, 1.28x without.

2. **France 100-300 is far below the model.** EMP gives about 0.013 a weekday against the model's
   0.0228 (1.7x). EMP's long-distance module is a recall of voyages; day trips of 80-150 km are the
   likeliest to be forgotten, so 0.013 is probably low, but no other French source reaches 0.023.
   Germany, at the same model level (0.0265), is supported by INVERMO (0.024-0.027), so the gap
   looks like France's survey or France's real behaviour, not the model's general level.

3. **The planned 300+ raises overshoot for France, Spain and China.** Planned values against the
   ceilings: Germany 0.008 against 0.008 (at the ceiling, fine); France 0.011 against 0.0065
   (1.7x; inside the hard bound of 0.0118 only if every 200-499 km trip were over 300); Spain 0.015
   against 0.008-0.010 (1.5-1.9x); China 0.0084 against 0.0045-0.0051 (1.7-1.9x; the model's current
   0.0065 is already 1.3x over). Japan (0.0046 against 0.006) and Italy (0.0034 against ~0.0055) are
   under; the US is at its figure.

4. **China.** T-3240's "300 km+ about 3.9bn a year" seems to treat most rail trips as long. A 366 km
   mean on a skewed distribution (the Pearl River Delta, Yangtze Delta and Chengdu - Chongqing intercity
   flows are all under 300 km) puts roughly 30% of rail trips over 375 route km, ~1.3bn, not ~3bn.
   Plus ~0.63bn by air and an unknown, probably under 1bn, by coach and car: 2.1-3.3bn.

5. **Japan.** MLIT's 2015 total is 0.0386 trips an average day crossing a prefecture line, 24% of them
   over 300 route km. Trips within one prefecture (all of Hokkaido, Niigata, Nagano) are not in it, so
   the 100-300 band is a floor; the 300+ band misses only intra-Hokkaido trips like Sapporo - Kushiro
   (small). The weekday rate is 0.89x the average day.

6. **Germany.** The best person-trip distance split is INVERMO 2002/03; MiD 2017 publishes no split of
   person trips over 100 km (its microdata needs an application, already in the data register). The
   100 km+ level is well pinned (MOP, KONTIV, MiD 2008, INVERMO, DB's own survey: 8.5-9 long trips a
   person a year, 0.05-0.06 one-way trips a day including trips abroad); the 300 km split rests on
   a graph read by eye and could be +-25%.

7. **Spain.** The Ministry's 2017 pilot gives only a daily total and a mean; whether its km are route or
   straight is not stated, which moves 300+ between 0.0078 and 0.0102. Non-residents (tourists with
   Spanish SIMs or roaming) are in the total, so the per-resident figure is if anything lower. The
   ministry's 2022+ open data has distance bands only up to "more than 50 km". The EsadeEcPol report
   (Jan 2025) on medium and long trips refused automated download (403); the PDF is
   https://www.esade.edu/ecpol/wp-content/uploads/2025/01/TransporteInterurbano_v9.pdf if a person
   wants to check its bands.

8. **Per-mode countries (China, Russia, Turkey, Italy).** The published floor is rail + air (+ scheduled
   coach in Turkey); the car volume over 300 km is a judgement in every case. The ceilings above assume
   cars carry 40-60% of 300 km+ trips, about France's and Germany's survey shares; a lower car share
   would lower them.

9. **Weekday against average day.** Only NHTS and MLIT's weekday table are weekday figures. EMP,
   INVERMO and ATS are annual averages, scaled by Japan's 0.89; if Europe's weekday dip is deeper,
   the weekday ceilings drop a little further.

## Sources not used, or tried and failed

- UBA Texte 51/2022 (Langstreckenmobilität, MiD 2017 and INVERMO chapters): the PDF came back as an
  HTML page; WebFetch 404. MiD 2023 short report: no long-trip split.
- BTS NTS Table 1-33 (ATS 1995 by distance): 403; the same numbers came from the ATS travel profile PDF.
- Eurostat tour_dem tables: domestic trips a head only, no distance (T-3240 5.1 already used them).
- Verkehrsverflechtungsprognose 2030 (Germany OD by distance class): not found in reach.
- MITMA Movilia 2006/07 long-distance tables: not found online.
