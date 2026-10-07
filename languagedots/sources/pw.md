# Palau: the record

Drawn 2026-10-05 (session edd42a8c-pw). 17,351 people in 16 states, 10 nodes, 15 dots and 7
rings at 1:1000. Scripts: `sources/pw_census.py` (transcribed table and checks),
`sources/pw_geo.py` (placement layer), `taxonomy/pw2015.py`, `taxonomy/tree.d/pw.txt`,
`countries/pw.py`. No asks.

## 1. Source, vintage, question

2015 Census of Population, Housing and Agriculture, Office of Planning and Statistics (Ministry
of Finance), report "2015 Census of Population, Housing and Agriculture Tables", Table 16
"Language Spoken and Language Used Most by Usual Residence, Palau: 2015", report page 25 (PDF
index 24). `data/raw/pw/2015-Census-of-Population-Housing-Agriculture.pdf` (31.8 MB), from
https://www.palaugov.pw/wp-content/uploads/2017/02/2015-Census-of-Population-Housing-Agriculture-.pdf
(listed on the ministry's census page). The tables are images; the figures were transcribed
by hand from a 300 dpi render into `pw_census.py`.

Questions (questionnaire, PDF pages 232-233), asked of everyone, all ages:
C22 does the person speak Palauan at home (Palauan only / Palauan and another / another),
C23 the other language spoken at home (one write-in), C24 whether it is spoken more often than
Palauan, equally, less often, or Palauan not spoken. So each person is either "Palauan only"
or under one named other language: Palauan only 8,657 + other languages 9,004 = 17,661.

**Why 2015, not 2020.** The 2020 report (`2020-Census-of-Population-and-Housing.pdf`, Table
16, p.25, same layout) tabulates the other language only for people who speak Palauan at home
(12,399 rows, 12,297 of them English, Philippine languages 16); its 5,215 "No, another
language", 4,509 of them of Asian ethnicity (Table 88), get no language at all. Every 2020
language table (16, 33, 48, 62, 75, 88, 100, 109, 122, 130, 140) has that shape. 2015 names the
language for everyone. 2005 (`2005-Census-of-Population-Housing.pdf`, Table 16, text layer) is
also complete and splits Asian languages finer (Filipino, Japanese, Korean, Chinese/Taiwanese),
but is ten years older; it is used below only as a check. Palau's SPC microdata for 2020
(pacificdata.org, spc_plw_2020_phc_v01_m) was not tried: the 2015 table is enough at this size.

## 2. Checks (`pw_census.py`)

All pass on the transcription, which is what vouches for it:
1. In all 18 rows, the Total column equals the sum of the 16 states, Outside of Palau and Unknown.
2. In all 19 columns: C22's three answers add to the total; Palauan-and-another + another =
   "Other language spoken"; the nine languages add to it; C24's four answers add to it;
   Palauan only + the nine languages = the total.
3. Reported, not fixed: C22 "No, another language" 4,785 against C24 "does not speak Palauan"
   4,447 (338 apart). The census's own inconsistency.

The 16 state ids and names match geoBoundaries' both ways (asserted in `pw_geo.py`).

## 3. The C24 call (the one most worth reversing)

The brief wants home language / main language. C23 alone would file a Palauan-and-English home
under English (5,640 English, 32%, mostly ethnic Palauans). C24 is the census's "language used
most" item, so `countries/pw.py` applies it per state: the 1,768 (1,757 in the states) who use their other language
**less often than Palauan** are drawn as Palauan. The table does not cross C24 with the
language, so they are taken off **English**. Evidence: the 2005 census does cross them (Table
16, 2005): of 431 people using another language less often than Palauan, 408 (95%) spoke
English. In every state English holds more than the moved count (asserted). Moved rows (English
and Palauan, all 16 states) are `tier="derived"`. People who use both "equally" (1,422) stay on
their other language, the census's C23 answer. Result: Palauan 10,379 (59.8%), English 3,683
(21.2%), Philippine 2,034 (11.7%).

Without the move, Palauan would be 8,622 and English 5,440 in the 16 states (1,757 moved). Reversing it is a
one-line change in `_counts()`.

## 4. Mapping (`taxonomy/pw2015.py`)

| label | node |
|---|---|
| Yes, Palauan only (+ C24 move) | austronesian.palauan |
| English | indoeuropean.germanic.english |
| Carolinian | austronesian.oceanic.sonsorol_tobi (new, "Sonsorolese and Tobian") |
| Other micronesian | pacific_other |
| Philippine languages | austronesian.philippine (group, draws as "language not named") |
| Japanese | japonic.japanese |
| Korean | koreanic.korean |
| Chinese languages | sinotibetan.sinitic (group) |
| Taiwanese | sinotibetan.sinitic.min_nan |
| Other language | other |

- "Carolinian" in Palau's census is the southwest islanders' language (the ethnic category of
  the same name; Sonsorol and Hatohobei states), not Saipan's Carolinian (gu.txt's node).
  Glottolog: Sonsorol sons1242 and Tobian tobi1238, family Sonsorol-Tobi sons1245, Chuukic, in
  Oceanic. One leaf for both: the 12 speakers live in Airai 5, Koror 3, Sonsorol 4, so a split
  by state would not separate them. No glottocode on the node.
- "Other micronesian" on `pacific_other` because Palau's "Micronesian" is the region and can
  include Chamorro, which is not Oceanic.
- "Taiwanese" is printed as its own row beside "Chinese languages", so it is read as the
  language the word names, Taiwanese Hokkien.

Colours: nothing hand-picked. Palauan and Philippine and the Oceanic node are all in
Austronesian's part of the wheel; Palauan (gu.txt's node) is the only big language that borders
another on the ground, and the second is English, far apart.

## 5. Geography (`pw_geo.py`)

religiondots draws Palau as one unit (`../religiondots/countries/pw.py`), so its 175 Kontur hexes
(`../religiondots/data/geo/pw/pw_hexes.gpkg`, read only) are re-keyed to the 16 states of
geoBoundaries gbOpen PLW ADM1 (OSM, ODbL, commit 9469f09, `data/raw/pw/`). 40 hexes (4,496
Kontur people) have centroids outside every OSM state outline and go to the nearest state, all
within 0.4 km except Ngarchelong (2.7 km) and Sonsorol (its three islands, Sonsorol, Pulo Anna
and Merir, sit outside its own outline; nearest is still Sonsorol). Helen Reef falls inside
Hatohobei. Every state holds populated hexes.

Kontur against the census per state is poor (normalised ratios 0.14 Ngardmau to 8.67 Angaur;
Koror 0.61), but the counts are the census's; Kontur only places people inside a state.

## 6. Gap and grain

Not drawn: "Outside of Palau" (284, people counted whose usual residence is abroad) and
"Unknown" (26), 1.8%. Grain: 16 states, mean 1,084, Koror 11,444, Hatohobei 25. At one dot per
thousand most states draw nothing; their people show only in the rings and the counts.
