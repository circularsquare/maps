# Azerbaijan (`az`): 2019 census mother tongue, placed by rayon through a nationality model

Drawn 2026-10-04 by session `d9e44929-az` at national grain. Ask 006 ruled 2026-10-05 (Anita:
"okay yes"): the nationality model is now drawn. Re-scattered 2026-10-05 by session
`d9e44929-rulings`.

Files: `sources/az_census.py` (normaliser), `sources/az_model.py` (the model, drawn),
`taxonomy/az2019.py`, `taxonomy/tree.d/az.txt`, `countries/az.py`. Outputs
`data/normalized/az.csv`, `data/normalized/az_model.csv`, `data/processed/dots_az.geojson` (9,934
dots), `rings_az.geojson` (0).

## 0. What is drawn now (ask 006, 2026-10-05)

`MODEL = True` in `countries/az.py`. **66 rayons and cities, 9,943,958 people (the 2019 existing
population), 21 nodes, every row `modelled`.** The 2019 national mother-tongue split of each
nationality (Table 30) is placed by that nationality's 2009 census count by rayon; Azerbaijanis
are each unit's remainder. Section 4's second bullet describes the model in full.

- `az_model.py` re-run 2026-10-05: 74 units joined one-to-one, every 2009 column sums to the
  national row, the model sums to 9,943,958. `check_country` ok. Scatter: 9,934 dots, 0 rings
  (modelled rows do not ring, so the four rings of the national build are gone); 9,958 people
  (0.10%) under one dot per language nationally; the 4 layer units with no population draw
  nothing.
- National shares are unchanged to the tenth of a point (Azerbaijani 9,549,520, 96.0%). The model
  totals 7,451 fewer people than section 3, because it uses the existing (de facto) population by
  rayon where Table 30 counts the permanent population; the difference falls on Azerbaijani.
- Rewritten: `how` ("model: the 2019 census's native language split by nationality, printed for
  the whole country, placed by each nationality's 2009 census count by rayon"), `grain` (66 rayons
  and cities, 151,000 people on average, languages within each modelled), `note_public` (says the
  places are modelled, names the flat split as the weakness, and gives Gusar, Zagatala, Balakan,
  Astara and Lerik). `gap` unchanged.
- The national-grain wording: `how` "census, 2019, native language"; `grain` "the whole country,
  9.95 million people; the census prints mother tongue for no smaller area"; the note said every
  language was spread over the country and the dots showed how many, not where.

## 1. The table

*Population Census in the Republic of Azerbaijan 2019*, Volume B (State Statistical Committee,
2022), **Table 30**, "National (ethnic) composition and mother tongue of population", printed
pp.415-429. The question is "ana dili", native language, one answer. Rows: the country and 21
nationalities, each split into men, women, urban, rural. Columns: population, then "language of the
nationality that it belongs", Azerbaijani, Turkish, Russian, Talish, Lezgi, Tat, Kurd, Georgian,
Avar, Sakhur, Udin, other languages.

The PDF is read in place from `religiondots/data/raw/az/` (read-only; religiondots fetched the
volume zips from stat.gov.az for its own Azerbaijan). `--fetch` downloads into
`languagedots/data/raw/az/` only if neither copy exists.

**Grain: the whole country.** Searched for anything finer:
- Volume A (470 pp.) has no language table at all (no page mentions "ana dili" or mother tongue).
- Volume B's language tables are 30 (mother tongue) and 31 (languages spoken fluently), both
  national. Its rayon tables are housing and population.
- stat.gov.az's demography page has table 1.12 (own-language share and fluent languages by
  nationality, national).
- The 2009 census's *XIX cild* (nationality, age, mother tongue, other languages) is not online;
  Bespyatov's pop-stat transcription has only its nationality-by-rayon table.

The population is the permanent (de jure) 9,951,409. The existing (de facto) population is 9,943,958;
the 7,451 difference is national only, so it does not matter at this grain.

## 2. Checks (all asserted in `az_census.py`)

1. Every one of the 198 rows sums across its 13 columns to its population; men + women and urban +
   rural equal the total in every cell.
2. The 21 nationality rows sum to the national row in every column, for total, urban and rural.
3. Each nationality's population equals table 1.11 (`001_11-12en.xls`, 2019, thousands) to 0.1
   thousand.
4. **Second table of the same census:** table 1.12's share of each nationality "who consider the
   language of their nationality native" equals Table 30's own column over the population, to the
   0.1 point printed, for all 21.

## 3. Result

9,951,409 people, 21 nodes. Azerbaijani 9,556,968 (96.0%), Lezgian 126,464 (1.3%), Russian 77,190
(0.8%), Avar 45,790, Talysh 44,342, Turkish 27,460, Tatar 15,750, Tsakhur 12,953, Tat 9,546,
Ukrainian 8,314, Georgian 8,275, Jewish 4,966, other 4,407, Udi 3,535, Haput 1,694, Kurdish 1,256,
Khinalug 1,186, Ingilo 842, Kryts 320, Budukh 99, Armenian 52. Rings: the four under one dot.

How many of each nationality named Azerbaijani: Talysh 50.5%, Lezgins 24.6%, Tats 68.3%, Kurds
69.9%, Khinalugs 65.8%, Grysz 84.6%, Budukhs 90.5%, Ingiloys 53.7%. Ukrainians mostly named Russian
(5,477 of 13,947).

## 4. Calls someone might reverse

- **National grain, one unit** (the 2026-10-04 build, replaced by the model on 2026-10-05). The 66
  populated rayons' hexes were one unit `AZ`; the 8 units the census found empty keep their ids and
  draw nothing (religiondots' Karabakh mask is already in the layer). Placement inside is Kontur
  population (9.82M against the census's 9.94M; Baku and Sumgayit under, Nakhchivan city over).
- **The nationality model is drawn** (ask 006, ruled yes 2026-10-05): a proxy. `sources/az_model.py`: each
  nationality's 2019 count on its 2009 rayon distribution, Azerbaijanis as the remainder, each
  nationality's national mother-tongue split applied flat. 66 units, 9,943,958 people, all
  `modelled`. Under 90% Azerbaijani: Gusar (56.5% Lezgian), Zagatala (18.6% Avar, 9.3% Tsakhur),
  Balakan (23.8% Avar), Gakh (12.0% Georgian), Astara (14.1% Talysh), Gabala (11.0% Lezgian, 3.4%
  Udi), Khachmaz (10.1% Lezgian), Saatly (10.9% Turkish). Lerik is almost all Azerbaijani because
  the 2009 census counted 2,281 Talysh in 74,522. 2019 nationalities with no 2009 column: Ingiloys
  on the Georgians, Grysz and Haputs on the Kryts, Budukhs on "other".
- **"Language of the Jews" on `other.jewish`**, a named leaf: the census names no language and
  Juhuri, Hebrew and Yiddish are all possible. Ukraine's "єврейську" went to Yiddish because the
  Soviet term means Yiddish; Azerbaijan's Jews are mostly Mountain Jews, so that reading does not
  carry over, and Juhuri would be a guess.
- **Ingilo a sibling of Georgian** (Glottolog: a dialect), and **Haput a sibling of Kryts**
  (Glottolog: a dialect): the census counts both as nationalities with their own language.
- **"Tat" is Muslim Tat** (musl1236). Only 2 Jews are in the Tat column (and 62 in Russian), so
  the Mountain Jews' Juhuri was coded as the Jews' own language, which is why `other.jewish` is
  probably mostly Juhuri; still not guessed into it.
- **Nakh-Daghestanian is a new root** (colour 225, blue-cyan), Lezgic and Avar-Andic under it,
  Khinalug a branch of its own. A Russia or Georgia agent will hang Chechen, Ingush and the rest here.

## 5. Not done, and noticed

- Turkish and Tatar get the same generated colour (`#ea6878`), the sibling bug ask 002 flagged.
  They mix at national grain anyway; not recoloured, since both are other countries' nodes.
- The urban/rural split (both in Table 30 and in the 2009 transcription) is unused. It could
  sharpen the model (a nationality's urban split for Baku and the towns, its rural split for the
  rest); not done in the 2026-10-05 switch, which drew the model as built.
- Talysh and Lezgin counts are often said to be low in Azerbaijan's censuses. No source for that was
  opened here, so the note says nothing about it.

## Karabakh (added 2026-10-06, session `5d7dac7e-cau`)

Anita, 2026-10-06: Karabakh was hatched; draw the current situation, which is essentially the
Azerbaijani resettlement, and do not draw the pre-2023 population as current. Files:
`sources/az_karabakh.py`, `taxonomy/az2026_karabakh.py`, `countries/az.py` (counts, `parts`,
`drawn_named`, place, note). Outputs `data/normalized/az_karabakh.csv`,
`data/geo/az/az_plus_hexes.gpkg` (religiondots' AZ hexes plus 15 settlement discs; `countries/az.py`
reads it). Units `KB-*`, inside `az` as religiondots keeps the area.

- **No census, register or official table by settlement exists.** Three official statements:
  more than 40,000 former IDPs returned (Hikmet Hajiyev, Assistant to the President, Trend,
  11 September 2026: the total drawn); more than 23,000 "residents" in the Khankendi, Aghdara and
  Khojaly districts (Sabuhi Gahramanov, deputy Special Representative, APA, 24 November 2025: that
  zone's share; may include state employees); and Researching Internal Displacement's settlement
  table compiled from press reports to January 2025 (Fuzuli city 3,132, Lachin 2,090, Shusha 1,386,
  Jabrayil 1,346, Aghali 871, Zabukh 823, Sus 215: the weights for the other 17,000; Zangilan city,
  settled 2025, at 300).
- **Assumed, nothing published**: inside the Khankendi-Aghdara-Khojaly zone, Khankendi half, the
  six resettled places named in convoy reports (Khojaly, Ballija, Aghdara, Talish, Hasanriz,
  Sugovushan) a twelfth each. Shukurbeyli is folded into Jabrayil (its disc touched the hexes
  2019's census populated).
- A district split attributed to Caspian News (23 Feb 2026; Lachin 2,176, Fuzuli 3,136, Zangilan
  1,837, Aghdara 2,996, Shusha 1,396, Khojaly 3,954, Jabrayil 3,646) was seen only as a search
  excerpt: the page is behind a Cloudflare wall and not in the Wayback Machine. It contradicts the
  January 2025 settlement table (Lachin district 3,128 then), so it was not used.
- All 40,000 drawn as Azerbaijani, `modelled`. **Double count**: they are also in the 2019 census
  where they lived then; 0.4% of the country, said in `note_public`. Workers and students living
  there without resettling (Hajiyev: about 100,000 "working, living and studying" in all) are not
  drawn; said in `gap` and the note.
- The Armenians: more than 100,000 left in September 2023 (UNHCR); the few dozen who stayed in
  Khankendi are under one dot and not drawn. Said in `note_public`.
- **Placement**: a disc around each OSM place point (Nominatim, 2026-10-06), 3 km for Khankendi,
  2 km for Fuzuli, Lachin and Shusha, 1.2 km otherwise; asserted to touch none of the 30,446 hexes of
  religiondots' layer. Kontur's 2023 grid shows the pre-2023 population there and is not used.
- Room for improvement: a published count by settlement (the State Committee announces each
  convoy, so a running tally could be built from APA/Trend reports, about 200 of them).
