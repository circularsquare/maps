# Philippines: 2020 census ethnicity, drawn as languages, checked against the census's home-language table

Built 2026-10-05, session `edd42a8c-ph`. Scripts: `sources/ph_census.py` (table, joins, retention
shift), `taxonomy/ph2020.py` (crosswalk, its docstring has every mapping call),
`taxonomy/tree.d/ph.txt` (270 lines, generated from ph2020.py), `countries/ph.py`. Raw files in
`data/raw/ph/`, outputs `data/normalized/ph.csv` and `ph_retention.csv` (all gitignored).

**This is a proxy**, built under AGENT_BRIEF §2's ethnicity rule (Anita, 2026-10-05). Rows read
straight from the ethnicity table are `derived`; the people moved by the retention check are
`modelled`.

## 1. The search for a language table

The supervisor asked for a language table at province or finer to be preferred over 2020
ethnicity, even if older. What exists:

| source | question | grain published | verdict |
|---|---|---|---|
| **2020 CPH, "Language/Dialect Generally Spoken at Home"** (PSA press release 2023-42, 7 March 2023) | "What is the language/dialect generally spoken at home by members of this household?" One answer per HOUSEHOLD | **national only**: 257 categories, 26,388,654 households | used as the retention check (§4) |
| 2020 CPH regional special releases | same | top five per province, only where a regional office wrote one: Cordillera (CAR-SSR-2024-40), Oriental Mindoro (2024-SR-078). A Wayback CDX sweep of all 17 `rsso*.psa.gov.ph` sites for language/dialect/spoken found no others | used as a check (§5) |
| 2020 CPH municipal special releases (searched again 2026-10-06, session 5d7dac7e-edge) | same | one municipality at a time, where a provincial office wrote one: Wayback holds only Oriental Mindoro's province release and Victoria (Oriental Mindoro). The MIMAROPA regional release (`rssomimaropa.psa.gov.ph/content/languages-dialects-spoken-home-mimaropa-region-...`) is province level and not archived; the live rsso sites return Cloudflare 403. A Wayback CDX sweep of psa.gov.ph and every subdomain for language/dialect/ethnic URLs found nothing more; an FOI request titled "Language/Dialect Generally Spoken at Home" exists on foi.gov.ph (403 to scripts, not read) | no municipal table exists for the country; placement inside provinces is estimated instead (§7) |
| PSA OpenSTAT (PxWeb API, open) | | 2020 CPH tables there: population, households, housing; no language, no ethnicity | household counts used (§4) |
| **2000 CPH, Report No. 2 Vol. I, Table 27** "Language or Dialect Generally Spoken in the Households" | same household question, 10% sample | **by province**, one PDF per province/HUC (~100) | **not reachable**: psa.gov.ph returns Cloudflare 403 to scripts (as religiondots found); the Wayback Machine holds only six (Abra, Basilan, Catanduanes, Cebu, Maguindanao, Marinduque). Abra's Table 27 shows the table exists and also its quality: 3,190 Ibanag households (7.8%) in Abra, a Cagayan language, beside 5,845 "Tinggian" and 1,080 "Itneg" |
| 2010 CPH | ethnicity; language "yet to be published" per NSCB in 2013 | | none found |
| 1990 CPH (IPUMS MTONGPH, LANGPH) | mother tongue, household language | microdata | IPUMS account blocked (memory note) |

So the only real language question at province grain is 2000's household table, behind a wall,
on 2000 boundaries (since then Zamboanga Sibugay, Dinagat, Davao Occidental, Davao de Oro's
rename, the BARMM interim province, several new HUCs), counting households from a 10% sample.
I built from 2020 ethnicity, which is full count, per person, 290 categories, on religiondots'
exact 117 units, and used 2020's own national language table to correct it. **Call someone might
reverse**: if Anita fetches the ~100 Report No. 2 PDFs by browser, a 2000 household-language
build is possible (Table 27 is text, parseable); it would need a 2000-to-2020 unit crosswalk and
the households-to-people conversion this build already does.

## 2. The table

`ph2020_ethnicity.xlsx` = PSA `Ethnicity_Statistical Table.xlsx` (sheet `Table`), from the press
release "Ethnicity in the Philippines" (2023-77, 3 July 2023), fetched from the Wayback Machine
(`id_` snapshot 20230812114311; psa.gov.ph is Cloudflare-walled). Question: "What is ____'s
ethnicity by descent/blood relation/consanguinity?", asked of a household respondent about every
member; categories from the National Commission on Indigenous Peoples and the National
Commission on Muslim Filipinos. Household population 108,667,043. Licence: PSA attribution (see
../religiondots/sources/ph.md §8).

Checks (all pass, `sources/ph_census.py`):
1. 290 columns; every row sums to its household population exactly. PSA prints `Buhid Mangyan`
   twice (18,636 and 169 people); the second is written `Buhid Mangyan [2]` and merged.
2. The 135 rows join religiondots' `ph.csv` row for row on label AND household population, to
   the person. The 117 fine units (provinces less their HUCs, 33 HUCs, Isabela City, Pateros,
   the BARMM Interim Province) sum to the national row in every column.
3. The press release's Table 1 (top ten and Not Reported) matches the national row.
4. OpenSTAT `0011A6DPHH0` (households by province/HUC) joins all 135 rows on household
   population (unique) with names agreeing; it lists HUCs in its own order.
5. The language table's 257 categories sum to 26,388,654 households.

## 3. Mapping calls (taxonomy/ph2020.py has all of them)

- Bisaya/Binisaya (15.5M, the second answer) -> **Cebuano**. It is what Cebuano speakers call
  their language in Mindanao, Leyte and Negros Oriental (Glottolog files Binisaya under Cebuano),
  and the language table agrees in total: Bisaya + Cebuano + Boholano households are 0.97 of what
  the three ethnicities imply. Boholano keeps a leaf beside Cebuano (the language table prints it).
- Caviteño and Batangan -> Tagalog (Tagalogs of Cavite and Batangas; "Caviteño" is 1,199
  households in the language table against 498,271 people). Bago -> Ilocano (631 households
  speak "Bago").
- **Place-split labels**, each with its evidence: `Other Local Ethnicity` is 88% of Aklan and 23%
  of Zambales; the list has no Aklanon or Sambal, and the language table's 133,121 households of
  the garbled code "Bukidnon/Binukid-Akeanon/Aklanon" are Aklan's household count. So it is
  Aklanon in Aklan, Sambal in Zambales, the unnamed Philippine remainder elsewhere. `Dumagat` is
  43% in Northern Mindanao, where the word is the Lumad name for lowland Visayan settlers: in
  Mindanao it is Cebuano. `Bukidnon` in the Visayas is the Panay/Negros highlanders, not Binukid.
  The Zambales call is the weakest (nothing in the language table names Sambal); reversing it
  puts 148,345 people on the unnamed remainder.
- Cluster generics (Kalinga, Itneg/Tinguian, Ifugao, Manobo, Mangyan, Agta, Dumagat, Panay
  Bukidnon) sit on their group node, drawn as "not named": the census offered the subgroups.
  Generic Aeta/Ayta, Ata/Negrito, Bagobo, Cotabateño and Other Local Ethnicity sit on
  `austronesian.philippine`.
- "Agta and Dumagat" is a readability group, not a Glottolog unit (Northern Luzon Agta, Bikol
  Agta and Manide are different branches). Sama-Bajaw is its own branch under `austronesian`.
  Chavacano varieties under `creole.spanish_based`. Eskaya -> `other.eskayan` (Glottolog:
  artificial). Indian -> Indo-Aryan, not named; Swiss, Singaporean, Afghan, Other Foreign -> other.
- Colours: Tagalog, Cebuano and Ilocano keep us.txt's (all cyan-teal; Tagalog/Ilocano in Central
  Luzon are close, flagged for Anita's tuning). Hiligaynon, Bikol, Waray, Kapampangan,
  Pangasinan, Kinaray-a and Kankanaey had no colour anywhere (Canada's generated ones) and get
  hand-picked hues here, as do the other large regional languages and the new groups. Canada's
  tiny Philippine-language rows change colour with them.

## 4. Retention: the national home-language table

For each ethnic group with a counterpart in the language table (77 sets; one table splits what
the other lumps, e.g. 40 Kalinga subgroups vs one "Kalinga"), households expected = sum over
units of persons x (households / household population) in that unit (OpenSTAT); retention =
language-table households / expected, capped at 1. The deficit is removed first from units where
the group is not the largest ethnicity (the diaspora), then from the rest in proportion, and drawn
on the unit's largest lingua franca other than the group's own (Tagalog, Cebuano, Hiligaynon,
Ilocano, Bikol, Waray, Tausug, Maranao, Maguindanao, Chavacano). 28 small ethnicities (461,152
people) have no counterpart and are kept whole.

Selected ratios (all in `ph_retention.csv`): Tagalog 1.44 (the receiver); Cebuano set 0.97;
Hiligaynon 0.92; Tausug 0.91; Maguindanao 0.89; Ilocano 0.85; Kapampangan 0.83; Maranao 0.82;
Waray 0.71; Pangasinan 0.67; Bikol 0.62; Kankanaey set 0.59; Ibanag 0.55; Ibaloi 0.46; Manobo set
0.40; Mansakan set 0.38; Subanen 0.31; Bontok set 0.28; Bukidnon (Binukid) 0.24; Capiznon 0.18;
Agta set 0.15; Davawenyo 0.10; Higaonon 0.10; Panay Bukidnon 0.06; Ati 0.04.

Moved: 14,323,818 people (13.2%): to Tagalog 6.67M, Cebuano 4.26M, Hiligaynon 1.47M, Ilocano
1.45M, the rest under 140k each.

**Drawn vs the language table** (households x 4.12 national persons per household): Tagalog
35.5M vs 43.4M (0.82), Cebuano 27.8M vs 24.4M (1.14), Hiligaynon 1.18, Ilocano 1.17, Bikol 1.07,
Waray 1.02, Kapampangan 1.01, Pangasinan 0.98; Maguindanao, Maranao, Tausug 1.26-1.37 (BARMM's
households are larger than 4.12, so these are inflated by the conversion, not by the map). The
remaining gap is Visayans and Ilocanos in and around Manila who speak Tagalog at home: their own
groups retain well nationally, so the diaspora-first rule moves only part of them. Known
direction, not corrected.

**Caveats.** A household is counted once under one language, so mixed-descent households blur
both sides. Retention is national per group; a group's shift really differs by place. Moved
people go to the unit's largest lingua franca, which in a mixed province (Cotabato: Hiligaynon,
Cebuano, Maguindanao) picks one.

## 5. Check against the regional releases (Cordillera, 2020, top five per province, households)

| province | drawn (people) | language table (households) |
|---|---|---|
| Abra | Ilocano 77.1, Itneg-Inlaud 5.5, Maeng 3.9, Adasen 2.6, Masadiit 2.5 | Ilocano 75.5, Maeng 5.2, Masadiit 3.6, Adasen 3.1, Inlaud 2.3 |
| Apayao | Ilocano 61.6, Isnag 23.4, Malaueg 3.5 | Ilocano 61.2, Isnag 27.5, Malaueg 3.7 |
| Benguet | Kankanaey 46.5, Ilocano 27.0, Ibaloi 13.1, Tagalog 4.1 | Kankanaey 36.3, Ilocano 33.9, Ibaloi 16.5, Tagalog 7.3 |

Abra and Apayao agree within a few points. Benguet overdraws Kankanaey by ten points: Kankanaey is
Benguet's largest ethnicity, so the diaspora-first rule protects it there and takes the deficit
elsewhere; in fact Benguet's Kankanaey shift to Ilocano too.

## 6. Room for improvement

- A municipal language or ethnicity table for any province would replace §7's estimate there;
  the regional offices hold them (Victoria, Oriental Mindoro, was published).

- The 2000 Table 27 PDFs (one per province, psa.gov.ph `main-publication/<PROVINCE>.pdf`) by
  browser: a real household-language table by province.
- A 2020 province-by-language table: PSA could release it (the regional offices clearly hold it);
  watch `psa.gov.ph/statistics/population-and-housing` and OpenSTAT.
- English as a home language: 37,164 households answered "American/English", far more than
  the ~25,000 people of anglophone nationality, so English is under-drawn here.

## 7. Placement

Religiondots' `ph_barangays.gpkg` (42,042 barangays, 2020 barangay population as `pop`, unit =
10-digit PSGC; ../religiondots/sources/ph_geo.md), joined 117 = 117 both ways.

**Until 2026-10-06** every language was spread over each unit's barangays by population. Anita
(2026-10-06) named the result as one of the map's two most visible false edges: Negros
Occidental drawn all Hiligaynon beside Negros Oriental all Cebuano, and the same at every
province line. No municipal language table exists (§1), so placement inside a unit is now an
estimate, under AGENT_BRIEF §4.4 (counts stay the unit's; session 5d7dac7e-edge).

**The rule as first built** (weakened the same day, §7a; `countries/ph.py`, `_PhWeighter`): per unit, a barangay x language table is raked
(IPF, 300 rounds) to the barangays' population, scaled to the unit's drawn total, and the unit's
count of each language. Seeds:
- a language with 40% or more of the unit: even (it fills what the others leave);
- a smaller language with a **homeland** (units where it holds 40% or more) whose nearest
  barangay is within 25 km of the unit (adjacent, or across a narrow strait):
  `0.002 + 0.998 x exp(-(d - dmin) / 12 km)`, d each barangay's distance to the nearest
  homeland barangay;
- a smaller language with no homeland unit at all: the same kernel from its Glottolog points
  (`POINTS` in ph.py, 45 nodes, codes looked up by name in `data/raw/glottolog/languages.csv`;
  a child node with no entry takes its nearest listed ancestor's; Glottolog's only Iranun point
  is in Sabah and is not used), when within 150 km;
- everything else even. Migrant languages (Cebuano in Manila, Tagalog in Davao) have no homeland
  within 25 km and stay on population.
Asserted: every unit's raked column sums equal its counts. scatter: 97 (unit, language) pairs
pulled to a neighbouring homeland, 2,073 to Glottolog points, 11,903 even; 108,529 dots, 0.11%
of people below one dot per language, 63 rings.

**What it does**, by municipality (Cebuano share of drawn people): Negros Occidental's 312,494
Cebuano now sit along the Negros Oriental side: the towns under the mountain spine (Moises
Padilla 31%, Isabela and La Castellana 27%, Kabankalan 26%), San Carlos 27%, Hinoba-an 23%,
against 1-4% on the north coast (Silay, Victorias, Cadiz, Manapla); PSGC municipality codes
checked against the barangay layer's positions. In Cotabato, Cebuano runs from 3% in the west
(by Maguindanao) to 50% in the east (by Davao), Hiligaynon being the majority and even. The rule cannot know a minority's town when
it does not border its homeland: Sipalay, which is largely Cebuano, gets 5%.

Constants (`HOME`, `MAJORITY`, `NEAR_KM`, `KERNEL_KM`, `LAMBDA`, `FAR_KM`) were set by looking
at Negros and Cotabato, nothing more. A kernel of 25 km (tried first) left Negros Occidental's
Cebuano nearly flat, 5-20% everywhere.

### 7a. Weakened, 2026-10-06 (session 5d7dac7e-php)

Anita (2026-10-06): "the softening is kinda sus tbh. i think we can do it but maybe slightly
weaken it. in the philippines im not sure if its realistic. ... especially for small languages
in super mountainous areas we dont wanna dilute much." The rule above is replaced by:

- **Only big regional languages are pulled towards a neighbouring homeland**: the twelve drawn
  by 1M+ people nationally (`PULL_MIN`): Tagalog, Cebuano, Boholano, Hiligaynon, Ilocano, Bikol,
  Waray, Kapampangan, Pangasinan, Maguindanao, Maranao, Tausug. The next language down is
  Kinaray-a at 0.60M, so the line falls in a clear gap. Every smaller language uses only its
  Glottolog points (the point pull is unchanged: 12 km kernel, λ 0.998, 150 km), so small
  languages stay gathered where Glottolog puts them. The eleven smaller languages that hold 40%+
  of some unit and were neighbour-pulled before now have points in `POINTS` (Kankanaey,
  Kinaray-a, Aklanon, Mandaya, Masbatenyo, Romblomanon, Surigaonon, Sama, Yakan, Chavacano, each
  looked up by name in Glottolog's languages.csv); Ivatan has no language-level point on Batanes
  and stays even.
- **Shorter reach**: the homeland must be within 15 km of the unit (was 25 km, which also reached
  across straits such as Cebu-Negros Occidental).
- **Weaker strength**: seed `0.2 + 0.8 x exp(-(d - dmin)/12 km)` (floor was 0.002), so a barangay
  far from the line keeps at least a fifth of the seed of one on it.
- **A cap**: of a pulled language's speakers in a unit, the share in barangays within 15 km of
  its homeland is at most twice the share of the unit's people who live there (`BAND_CAP`); λ is
  halved until that holds (13 of 53 pulled pairs were held back by it).
- **Not into the hills**: the pull fades with barangay elevation, full below 300 m and nothing
  above 700 m (GEBCO 2026 15" grid at each barangay's representative point,
  `sources/ph_elev.py` -> `data/geo/ph/ph_barangay_elev.csv`; Baguio's barangays come out at a
  median 1,446 m, Manila's 7 m). Maranao is exempt from the fade because its own homeland, the
  Lanao plateau, is at about 700 m.
- **No pull at all in units led by a smaller language** (12 units: Benguet, Mountain Province,
  Ifugao, Batanes, Masbate, Aklan, Antique, Romblon, Surigao del Norte, Zamboanga City, Basilan,
  Tawi-Tawi): there the big languages follow population. Kalinga and Apayao are led by Ilocano in
  this build, so there the elevation fade is what keeps Ilocano out of the hills.

Counts are untouched: the rake still meets every unit's per-language count (asserted).
scatter: 53 (unit, language) pairs pulled towards a neighbouring homeland (was 95), 2,129 to
Glottolog points (was 2,073), 11,891 even; 108,529 dots, 63 rings.

**Before and after.** Per unit, each big language that is a minority there (1-40%): its share
of drawn people in barangays within 15 km of the homeland unit(s) next door ("band", distance
between barangay representative points, so it runs a few km wide of the true line), in the rest
of the unit, and in barangays at 700 m or more. "-" means no homeland within 15 km, so the
language was never neighbour-pulled there.

| unit | language (unit share) | homeland next door | band holds | band: before -> after | rest: before -> after | 700 m+: before -> after |
|---|---|---|---|---|---|---|
| Negros Occidental | Cebuano (11.9%) | Negros Oriental, Cebu | 17.5% of people | 31.3 -> 18.8 | 7.8 -> 10.5 | 16.9 -> 7.8 |
| Negros Oriental | Hiligaynon (1.5%) | Negros Occidental | 24.6% | 4.3 -> 2.4 | 0.6 -> 1.2 | 3.2 -> 0.9 |
| Cotabato | Cebuano (22.9%) | Bukidnon, Davao del Sur, Davao City | 21.6% | 48.2 -> 30.5 | 16.0 -> 20.8 | 41.8 -> 17.3 |
| Cotabato | Maguindanao (11.8%) | Maguindanao, BARMM interim province | 60.9% | 17.1 -> 15.1 | 3.6 -> 6.6 | 4.8 -> 5.3 |
| Cotabato | Ilocano (4.0%), Boholano (2.0%) | - | | | 4.0, 2.0 (no change) | |
| Benguet | Ilocano (27.0%) | La Union, Ilocos Sur, Nueva Vizcaya, Baguio | 82.1% | 30.1 -> 27.0 | 12.9 -> 27.4 | 26.4 -> 26.9 |
| Mountain Province | Ilocano (29.8%) | Ilocos Sur, Abra, Kalinga, Isabela | 66.2% | 34.8 -> 30.4 | 20.0 -> 28.5 | 24.8 -> 25.5 |
| Bukidnon | Hiligaynon (5.2%) | Cotabato | 8.3% | 26.5 -> 11.6 | 3.2 -> 4.6 | 3.2 -> 3.8 |
| Bukidnon | Boholano (5.5%), Ilocano (1.1%) | - | | | no change | |
| Occidental Mindoro | Cebuano (18.7%), Ilocano (7.2%) | - | | | no change | |
| Oriental Mindoro | Cebuano (6.8%), Ilocano (1.3%) | - | | | no change | |

Majority languages are even and only take what the minorities leave: Negros Occidental's
Hiligaynon goes from 82.3% to 91.4% in its upland barangays; Cotabato's Hiligaynon from 44.1% to
64.1% in its. Mindoro's big languages were never neighbour-pulled (Tagalog is the majority on
both sides; Cebuano and Ilocano have no homeland near), and the Mangyan languages keep their
point pull, so Mindoro is unchanged. In Benguet and Mountain Province Ilocano is now spread by
population (the small residual in Mountain Province is the Bontok and Kankanaey point pulls
taking the barangays near their points). Negros Occidental's Cebuano by municipality now runs
8-21% (was 1-31%); the north coast is no longer near zero.

Comparison script: kept in the session scratchpad only (it re-imports the pre-weakening
`countries/ph.py`); the numbers above are its output.

## Moved from countries/ph.py text (2026-10-06 sweep)

From `note_public`: "Higaonon, Bagobo and Davawenyo households number a tenth to a fifth of
their people." "Smaller languages are drawn nearest to where Glottolog places them (the Manobo,
Subanen and Blaan languages towards their home areas)."
