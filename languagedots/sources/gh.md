# Ghana: 2021 census ethnicity, read as home language through Afrobarometer

Built 2026-10-05 (session edd42a8c-gh). 30,484,536 people (all Ghanaians), 272 units, 57
answers, every row `modelled`. Tier D (ethnicity only), built under AGENT_BRIEF §2's
2026-10-05 ruling with the retention check.

| | |
|---|---|
| census | GSS 2021 PHC, StatsBank `Population/ethnic_table.px`: 9 major groups + Others, Ghanaians only, on 261 districts + 17 sub-metros |
| survey | Afrobarometer R4-R9 (2008-2022), Ghana: 13,169 respondents, ethnic group (Q79/Q84/Q87/Q81/Q84A) and home language (Q3/Q2/Q2B/Q2); religiondots' merged .sav files, read-only |
| split | GSS StatsBank `Education and Literacy/GHLang_table.px`: literate population by Ghanaian language of literacy, same 272 units |
| geography | religiondots' `gh_districts.gpkg` (GSS 2021 polygons, Lake Volta cut out), read-only; Kontur 400 m hexes keyed to it (`sources/gh_geo.py` -> `data/geo/gh/gh_hexes.gpkg`) |
| scripts | `sources/gh_afro.py --fetch`, `sources/gh_census.py [--fetch]`, `sources/gh_geo.py`; `taxonomy/gh2021.py`, `taxonomy/tree.d/gh.txt`, `countries/gh.py` |

## 1. What was searched for a real language table

- **2021 PHC.** No language question. Its language items are literacy (Vol 3D; StatsBank
  `GHLang_table`, `Lang_of_Lit_table`, `Lang_of_Lit_table_2`: which languages a literate
  person reads and writes, and how many). Literacy is not first language: 5.1M are literate
  in Asante Twi against 0.39M in Dagbani. Used only to split two pairs (§4).
- **Vol 3C** (downloaded this time, 81 pp): Table 5.5 is the nine groups by region and sex,
  nothing finer. StatsBank's district cube is strictly better (same groups, 272 units); its
  national figures equal Table 5.5 exactly (30,484,536; Akan 13,925,576).
- **Finer ethnic sub-groups** (the ~90 of 2010): only in IPUMS microdata (account blocked)
  and 2010 report tables; not used. The district cube plus the survey does that job.
- **DHS / GLSS**: microdata gated (DHS registration is off; GLSS on the GSS microdata
  catalogue needs an application). **CLEAR Global** Ghana is itself Afrobarometer-based at
  region level, so the raw rounds are better.

## 2. Retention check (AB, pooled, weighted, share speaking a language of their own group)

Akan 97% (n 6,749); Mole-Dagbani 90% (2,059); Ga-Dangme 89% (1,099; 8% Akan); Ewe 85%
(1,850; 10% Akan); Others 85% (Hausa 73%); Gurma 79% (487); Guan 76% (274; national mix
dominated by Gonja); Grusi 69% (255; 15% Akan); Mande 61% (112; 23% Akan). The rest speak
Akan, English or Hausa at home, and are drawn so: the model moves them onto what they said,
not onto one national language (Ghana's shift language is Akan, not English).

## 3. The model (`sources/gh_census.py`)

Each unit's count of group G is shared over P(answer | G) from the survey, estimated at four
nested levels, each shrunk towards the one above with a prior of K = 30 respondents:
national -> 2018 region (all rounds) -> 2021 region (R8, R9, and R4/R6/R7 respondents whose
district name places them; 80% of respondents) -> district (8,048 of 8,271 respondents who
name a district matched by name, prefix or close match within their 2018 region; a parent
district spreads a respondent over its 2021 children).

The prior inside each level is not flat: each language with a home area (Glottolog's point,
glottocode in `HOME`, plus a hand-set radius 12-100 km, floor 0.02) is weighted towards its
home, normalised so the national total is the survey's. Without it, region-level shares
smeared Kusaal 13% into Bolgatanga, gave Builsa North 41% Buli and Effutu 17% Gonja; with it
Bawku is 62% Kusaal, Builsa North 79% Buli, Bolgatanga 60% Farefare, Effutu 3% Gonja. The same
kernel places dots inside a district (`countries/gh.py`, a placement weight only).

Checks: groups sum to each unit's total; units sum to 30,484,536 = Vol 3C; the drawn rows sum
back to it; 272 units match religiondots' polygons both ways; Kontur join r = 0.689 against
0.172 shuffled (7 metro sub-units outside x3, Kontur's urban spread).

## 4. Calls

- **Ga and Dangme**: AB codes one "Ga/Dangbe" answer; split per unit by the literacy cube's Ga :
  Dangme (region's ratio under 200 literate). Ada East 77% Dangme, Accra sub-metros Ga.
- **Nzema**: AB rarely met it (16 respondents); Akan and Nzema answers are pooled and split by
  Nzema : (Asante + Akuapem Twi + Fante) literacy. Jomoro 65% Nzema; 261k nationally.
- **English, Hausa and Akan at R7's mother tongue** (§4a). English was kept as answered (555k)
  before 2026-10-05.
- Akan one node (AB has no Twi/Fante split). Dagaare on bf.txt's `dagara`. "Dagaare/Waale"
  (one round's combined code) split by the respondent's ethnicity.
- Census group of a sub-group, where arguable, read off the district cube: Chokosi in Akan
  (Chereponi 66% Akan), Bisa and Banda in Mande, Builsa and Kusasi in Mole-Dagbani, Bimoba in
  Gurma, Kotokoli in Grusi.
- Unidentifiable "Other" verbatims go on the remainder of the respondent's group (`guan`,
  `gurunsi`, `gur`, `kwa`, `africa_other`). Non-Ghanaians (~350k) not drawn (`gap`).
- New groups `kwa.guan` (Guan) and `kwa.gtm` (Ghana-Togo Mountain, which the census files
  under Guan). Hand colours: Dagbani, Mampruli, Farefare, Konkomba, Gonja (tree.d/gh.txt says
  why).

## 4a. Lingua francas at R7's mother tongue (2026-10-05, ask 018)

Anita's ruling: lingua francas are drawn at Afrobarometer R7's separate **mother tongue**
question (Q2A). `calibrate` in `sources/gh_census.py`: for English, Hausa and Akan, each census
group's P(answer | group) is set to R7's Q2A share among that group's R7 respondents (2,367
matched by RESPNO), shrunk by 50 respondents (`wafr_afro.K_SHRINK`) towards the pooled share x
the national Q2A/pooled ratio. The factor is applied to every unit's modelled share, the
group's other answers scaled to fill; the four-level geography is unchanged.

| | before | after | R7 Q2A, national |
|---|---|---|---|
| English | 555k, 1.82% | 273k, 0.89% | 0.87% |
| Hausa | 615k, 2.02% | 371k, 1.22% | 0.94% |
| Akan | 15.0M, 49.2% | 14.2M, 46.7% | 50.7% (all groups) |

Akan among non-Akan groups falls most: Ewe 10.3 -> 3.6%, Ga-Dangme 8.5 -> 4.4%, Grusi 15.1 ->
6.5%, Mole-Dagbani 5.4 -> 3.0%. Hausa among "Others" 73 -> 49% (27 R7 respondents). English is
highest in Accra's districts (2-3%, was 4-7%).

## 5. Room for improvement

- **Bimoba (58k) and the southern Guan languages (Efutu 4k, Larteh 2k) are under drawn**: the
  survey met few of them, and the home-area prior cannot create speakers the survey never
  counted. Bunkpurugu comes out 45% Konkomba, 9% Bimoba; it is mostly Bimoba.
- A real language or detailed-ethnicity table (2010 detailed groups by district, or 2021
  microdata) would replace the survey's split of each cluster and its retention shares.
- AB's district sample is thin (median one Mole-Dagbani respondent per district), so the
  district level mostly follows the home-area prior.

## Cross-border groups (2026-10-06, 5d7dac7e-xb)

Ewe is in a Gbe group with Togo's and Benin's Gbe languages. The Akan group now also holds Côte d'Ivoire's Abron (Bono), which this census already counts as Akan. Groups only (taxonomy/regroup.txt): no label's node or count changed, each keeps its leaf and colour; dots rewritten in place. The full table of cases is in followups.md (2026-10-06, languages that stop at a border).
