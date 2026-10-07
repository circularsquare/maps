# Greece (gr): the record

Drawn 2026-10-05 by session edd42a8c-gr. 10,477,450 people (2021 census, less 5,013 of unknown
citizenship), 52 regional units (NUTS 3) plus Mount Athos, 113 languages, 10,449 dots. Every row
is `derived`.

Build: `python sources/gr_build.py` (writes `data/normalized/gr.csv` and the placement weights
`data/geo/gr/gr_weights.csv`). Mapping `taxonomy/gr2021.py` (France's labels plus Greece's), nodes
`taxonomy/tree.d/gr.txt`, entry `countries/gr.py`. Read-only from religiondots: the Eurostat
census file `data/raw/gr/cens_21ctz_r3_el.json` and the LAU layer `data/geo/gr/gr_lau.gpkg`.

## 1. The rule

No Greek census has asked language since 1951. Anita's 2026-10-05 ruling for rich countries with
no language question (AGENT_BRIEF §2): Greek, plus (a) minority and regional languages, (b)
immigrant languages by citizenship; proxy rows derived. Brief §4.4: the 1951 mother-tongue table
by nomos is a placement guide.

## 2. Sources

- **2021 census, residents by citizenship, NUTS 3** (ELSTAT via Eurostat `cens_21ctz_r3`, the file
  religiondots fetched). TOTAL 10,482,482; NAT 9,716,914; FOR 758,597 (200 named citizenships plus
  continental `_OTH` remainders); STLS 1,990; UNK 5,013. Cells are not exactly additive (worst unit
  off by 9 and 33 people); Greek absorbs it.
- **1951 census, vol. II (ESYE 1958), table 7b**, population by religion, mother tongue and sex, by
  nomos: `dlib.statistics.gr/Book/GRESYE_02_0101_00029.pdf` (599 scanned pages; table 7b is PDF
  pp. 217-251, table 7a's national row p. 189). Transcribed by hand into
  `data/raw/gr/census1951_table7b_mothertongue.csv` (52 nomoi, 12 columns). The scan PDF (131 MB)
  is in the session scratchpad, not the repo.
- **Muslim minority of Thrace**: 120,000 (Greek government figure, as in ELIAMEP 2006, "A case
  study on Muslims in Western Thrace", p. 3, citing Alexandris); composition 50% Turkish origin,
  35% Pomak, 15% Roma (1991 census as reported, Wikipedia "Minorities in Greece"; ELIAMEP gives
  45/36/18). Pomaks by regional unit: 23,000 Xanthi, 11,000 Rodopi, 2,000 Evros (estimate based on
  the 2001 census, same page).
- **Turks of Rhodes and Kos**: 2,500 and 2,000 (their association's president, Wikipedia "Turks of
  the Dodecanese"); under 5,000 in all.
- **Roma**: 117,495 (General Secretariat for Social Solidarity, 2021 mapping, National Roma
  Strategy 2021-2030); by region from the 2017 mapping, Operational Action Plan 2017-2021 table 3
  (104,210: Attica 30,363, East Macedonia-Thrace 16,435, West Greece 15,898, Central Macedonia
  15,374, Thessaly 14,534, Central Greece 4,958, Peloponnese 1,661, Epirus 1,500, Crete 1,211,
  North Aegean 755, Ionian 639, South Aegean 554, West Macedonia 328). Its "CONTINENT" row is a
  machine translation of Ipeiros (Epirus).
- **Aromanian**: 50,000 native speakers in Greece (2018 estimate, Wikipedia "Aromanian language").
- **Arvanitika**: 50,000 (Sasse 1991, as cited in Wikipedia "Arvanites"; Ethnologue 2000 says
  150,000, MRG 1997 200,000 Arvanites).
- **Slavic speakers of Greek Macedonia**: 50,000, the low end of the usual 50,000-250,000
  (Wikipedia "Slavic speakers of Greek Macedonia"). Van Boeschoten found 64% speakers in 43
  villages of the Florina area.

## 3. Method (per regional unit)

1. **Immigrant languages**: each named citizenship on its main language (`fr_build.COUNTRY_LANG`,
   overrides in `gr_build.GREECE_OVERRIDES`: Cyprus Greek, India Punjabi, Switzerland German,
   Belgium Dutch, Canada English), times **61.5% retention** (Italy's ISTAT 2024 tav. 11, share of
   non-Italian mother tongues speaking that language at home). No Greek source gives a share:
   Gogonas (2009, 2015) documents strong shift to Greek among second-generation Albanians but gives
   no first-generation home-language rate. `_OTH` remainders and stateless people on `other`.
   Mount Athos's foreign monks at 100%. 467,928 of 760,587 foreign or stateless residents drawn on
   their language. **Naturalised citizens are not visible**: many Albanians have become Greek
   citizens (most ethnic Greeks from southern Albania among them), and they are drawn as Greek.
2. **Minorities**, out of Greek citizens (NAT):
   - Turkish (Thrace) 60,000, split over Evros, Xanthi, Rodopi by the 1951 Muslim Turkish speakers
     as a share of each nomos, times 2021 Greek citizens: Rodopi 36%, Xanthi 17%, Evros 3%.
   - Pomak 36,000 by the 2001 unit estimates: Xanthi 21%, Rodopi 11%, Evros 1.5%.
   - Turkish (Dodecanese) 4,500.
   - Romani 117,495: 2017 regional shares scaled to the 2021 total, then by Greek citizens across a
     region's units. All Roma drawn as Romani speakers (Muslim Roma in Thrace partly speak Turkish).
   - Aromanian 50,000 and Slavic 50,000 by `by_1951()`: the 1951 rate (speakers / nomos
     population) times the unit's 2021 Greek citizens, so emptied units (Florina, Evrytania) are not
     drawn at 1951 size. A count-based spread put Macedonian at 52% of Florina; the rate gives 40%.
     Slavic restricted to Macedonia (EL52, EL53, Drama, Kavala), Athos and the Cyclades' 1951
     answers left out. **Drama, Kavala and Serres drawn Bulgarian, the rest Macedonian** (Trudgill
     2000 classes the eastern dialects with Bulgarian); a place-dependent label split by place.
   - **Arvanitika is not spread by 1951.** The 1951 "Albanian" answers run against every modern
     account (Attica 0.09%, Argolida 2.4%, Evros 2.9%, Thesprotia's Orthodox Chams 11%): Arvanites
     of Old Greece mostly declared Greek. Drawn instead on Euromosaic's areas (East and West Attica,
     Piraeus and islands, Boeotia, Euboea, Argolida-Arkadia, Corinthia), by Greek citizens: 2.7-2.8%
     of each. Epirus's Cham Albanian (5,426 + 1,588 in 1951, Thesprotia and Preveza) has no modern
     figure and is not drawn; Andros (in the Cyclades unit) and Evros's Arvanites are left out.
3. **Greek**: everyone else.

**The 1951 transcription checks to the national row** (table 7a, p. 184 of the print): total
7,632,801, Greek 7,297,878, Turkish 179,895, Aromanian 39,855, Albanian 22,736, Romani 7,429 all
exact; Pomak 18,670 against 18,671. **Florina's printed row has 4,303 under "Russian"**: Slavic
across nomoi sums to 36,715 against the national 41,017, and Russian to 8,086 against 3,815; both
reconcile (41,018 and 3,783) only with Florina's 4,303 moved to Slavic, which the build does. (The
Greek Wikipedia's 7,297,827 Greek speakers is a typo for 7,297,878.) Most 1951 "Turkish" outside
Thrace and the Dodecanese is Christian refugee Turkophones (86,838 Orthodox nationally); only the
Muslim column is used.

## 4. Geography and placement

Religiondots' 6,137 GISCO LAU 2021 polygons with their NUTS 3 (`place_unit` = `nuts3`); LAU
population as weight. Pomak on the Pomak villages (LAU codes: Myki municipality 0603 in Xanthi,
Kechros 010203 and Organi 010204 in Rodopi, Mikro Derio 03050206 in Evros, whose LAU holds Mega
Derio) up to 85% of their people, the rest over the unit: Xanthi 13,209 of 23,000 in the villages,
Rodopi 2,894 of 11,000, Evros 1,618 of 2,000. Thracian Turkish on every LAU but those villages;
Dodecanese Turkish on Rhodes town and Kos town. Everything else by LAU population.

## 5. Checks and result

Units sum to the census total less UNK within 1 person; minorities at most 51% of a unit's Greek
citizens (Rodopi). check_country ok; scatter 10,449 dots on 2,533 polygons, 81 rings (all
derived), 28,450 people (0.27%) under one dot per language.

National: Greek 92.1%, Albanian 230,630 (2.2%), Romani 117,495, Turkish 67,946, Aromanian 50,000,
Arvanitika 50,000, Macedonian 45,117, Pomak 36,000, Punjabi 29,336, Bulgarian 27,400 (Slavic of
Drama, Kavala, Serres plus Bulgarian citizens), Romanian, Arabic, Georgian. Greek share lowest:
Rodopi 49%, Florina 56%, Xanthi 57%, Pella 83%.

Beside religiondots' Greece: its Muslim minority split gives Anatoliki Makedonia-Thraki 22%
Muslim; here Turkish + Pomak are 17% of that region, plus Muslim Roma within Romani.

## 6. Colours

New: Pomak hand-set teal-green (0.64 0.12 165), Arvanitika light orange (0.76 0.13 50).
**Turkish (ae/cz, 0.70 0.16 330) and Romani (cz, 0.70 0.12 330) share a hue and lightness and
both sit in Rodopi and Xanthi**; not changed here, since both are defined by other countries.
Worth a look by whoever owns the colours.

## 7. Room for improvement

- The three Slavic, Aromanian and Arvanitika totals are speaker estimates of unclear date and
  definition, not home language; they likely overstate first-language use (Florina Macedonian 40%).
- Within Thrace, Turkish is spread on all non-Pomak LAUs by population, so Komotini and Xanthi
  towns get Turkish at the unit rate; a village list of Muslim communities would place it better.
- Pomaks increasingly speak Turkish at home; no figure, so all are drawn Pomak.
- Retention borrowed from Italy; a Greek survey of immigrant home language would replace it.
- Not drawn: Pontic Greek and Tsakonian (Glottolog languages; estimates are of Pontic descent, not
  speech), Cham Albanian in Epirus, Armenian of Greek citizens (8,990 in 1951), Ladino.
