# Malaysia: 2020 census ethnic group, read as language, 160 districts

Drawn 2026-10-05 (session edd42a8c-my). `sources/my_census.py`, `taxonomy/my2020.py`,
`taxonomy/tree.d/my.txt`, `countries/my.py`. Build tail left to the supervisor.

**REWORKED 2026-10-19 (session eca0185b-my), see §9, which supersedes the numbers and the
Chinese, Indian and non-citizen calls below. Now: 32,445,200 people (29,754,600 citizens and
2,690,600 non-citizens), 160 districts, 119 nodes, 32,387 dots at 1:1000; 19.0M rows `derived`,
13.4M `modelled`. 3.2% on a group node or `other` (was 27.3%).**

First build (2026-10-05, session edd42a8c-my): 29,754,600 citizens, 160 administrative districts
(203,000 people on average), 48 nodes, 29,733 dots at 1:1000. 25.2M rows `derived` (district
counts read as language), 4.55M (15.3%) `modelled` (sub-ethnic groups counted by state and
spread over districts). 2,691,070 non-citizens not drawn.

## 1. The tables

Malaysia's census asks ethnic group (kumpulan etnik), never language: tier D, built under
AGENT_BRIEF §2's ethnicity rule.

| table | grain | what |
|---|---|---|
| OpenDOSM `population_district.parquet`, 2020 rows | 160 districts | Malay, Other Bumiputera, Chinese, Indians, Other citizens, Non-citizens; thousands to one decimal. CC BY 4.0, no key |
| State volumes *JADUAL 1 HINGGA 16*, Table 5 | 16 states | sub-ethnic groups: Orang Asli (Negrito, Senoi, Proto-Malay), 15 Sabah groups, 4 (peninsular states) or 27 (Sarawak) Sarawak groups, Chinese, Indians, Others |
| Census 2010 Table 11.1 (Sabah), 12.1 (Sarawak), *Total population by ethnic group, Local Authority area and state* | 2010 districts | Other Bumiputera split into Kadazan Dusun, Bajau, Murut (Sabah); Iban, Bidayuh, Melanau (Sarawak) |
| *JADUAL BANCI MALAYSIA 2020 MUKIM BANDAR PEKAN* | mukim | broad groups 2010 and 2020; the check on the others |
| National volume Table 4 (T) | state | broad groups (check) |

The 2020 workbooks are religiondots' downloads (eStatistik, free login), read-only. The 2010
PDFs were at `statistics.gov.my/portal/download_Population/files/population/04Jadual_PBT_negeri/`
(dead), fetched from the Wayback Machine (2012-02-27), `data/raw/my/`. The other twelve states'
PBT PDFs exist there too but carry only the broad groups.

**Why OpenDOSM and not the mukim workbook for districts:** the workbook's mukim, bandar and pekan
rows do not add to the published district totals everywhere (Kota Bharu +103,861 with Tanah Merah
-103,198; Sepang and Ulu Langat swap 138,000; Jelebu and Port Dickson short), and some
continuation sheets carry the wrong district heading. OpenDOSM's 2020 rows match the religion
table's district totals within 50 everywhere.

## 2. Nothing finer, and what was searched

- **Sub-ethnic groups are state-only in 2020.** The district and mukim tables carry the broad
  groups; Sabah's and Sarawak's state volumes have no district sub-ethnic table (Sarawak's
  Table 5 by district is broad, and its value cells are empty, the same decoy pattern as its
  religion Table 7). Wikipedia's Sabah district pages cite 2020 district sub-ethnic figures from a
  DOSM OwnCloud share (`cloud.stats.gov.my/index.php/s/BG11nZfaBh09RaX`); the share is dead (404,
  503 on /download), and is probably the per-district Sabah volumes on eStatistik. **That is
  the one improvement worth a browser session**: Sabah's 27 and Sarawak's 40 per-district volumes
  (eStatistik, *Penemuan Utama ... <district>*), if their Table 5 is by sub-ethnic group, would
  replace the 2010 pattern with 2020 counts.
- **Chinese dialect groups** (first build's search; §9 found the 2000 table in Voon 2024): the
  1970-2000 censuses coded Hokkien, Cantonese, Hakka, Teochew,
  Hainanese, Kwongsai, Hokchiu, Henghua, Hokchia (the 1970 master-file codebook, SEAFERT,
  `seafert.csde.washington.edu/n/m/malay70.pdf`), but no open table by state or district was
  found; the microdata (SEAFERT, IPUMS) need the statistics office's or IPUMS's permission
  (IPUMS account blocked, gated). Wikipedia's *Malaysian Chinese* has no dialect table. 2010 and
  2020 publish none.
- **Indian sub-groups** (first build; §9 has a share table): the same 1970 coding (Indian Tamil, Telugu, Malayali, Punjabi, Ceylon
  Tamil...), same access; nothing published since.
- **Non-citizens by citizenship**: no 2020 census table by state was found; UN DESA's migrant
  stock (Indonesia 1.24M, Nepal 0.59M, Bangladesh 0.42M, Myanmar 0.35M, Philippines 0.12M) is
  national and not a census count.

## 3. Checks (all in `my_census.py`)

1. OpenDOSM: 160 districts join by name one to one to religiondots' units; every district's
   total within 50 of religiondots' religion table (a different table of the same census).
2. Districts summed per state against Table 4 within 2,000 for every group; Table 4 files some
   Bumiputera between Malay and Other Bumiputera differently from Table 5 (Johor 4,841), and
   OpenDOSM follows Table 5, so only Bumiputera as a whole is compared there.
3. The mukim workbook agrees with OpenDOSM within 100 on Bumiputera, Chinese, Indians and
   non-citizens in 143 of 160 districts.
4. Table 5: the leaves add to the state's citizens and its Bumiputera leaves to Table 4's
   Bumiputera, exactly, in 15 states; Labuan's are 15 short.
5. 2010 PDFs: every row adds up (total = citizens + non-citizens, citizens = Bumiputera +
   Chinese + Indians + Others, Bumiputera = its five columns). Each 2010 district heading equals
   the mukim workbook's 2010 column summed over the 2020 districts that borrow it: exactly for
   all 25 Sabah headings (3,117,405), and in Sarawak except three boundary moves (Siburan,
   32,299, Kuching to Serian; 1,728 Kapit to Belaga; 5,237 Miri to Marudi).

## 4. How the counts are made (`countries/my.py`)

- **Others**: the district's own count, `derived`. (Chinese, Indians and non-citizens: §9;
  first build had Chinese on `sinitic`, Indians 80% Tamil and 20% `other`, non-citizens undrawn.)
- **Bumiputera**: for each state, Table 5's groups (columns) are spread over the state's
  districts so that each district's groups add to its 2020 Bumiputera (Malay + Other
  Bumiputera) and each group's districts add to its state count (iterative proportional
  fitting; Table 5 rescaled by at most OpenDOSM's rounding). The starting pattern:
  - peninsular states, KL, Labuan, Putrajaya: Malay follows the district's 2020 Malay (and comes
    out as it, `derived`); every other group follows the district's 2020 Other Bumiputera
    (`modelled`).
  - Sabah and Sarawak: each group follows its 2010 column's share of the 2010 heading's
    Bumiputera (Malay; Kadazan Dusun; Bajau; Murut, also for Lundayeh; Iban; Bidayuh; Melanau;
    Other Bumiputera for the rest), times the district's 2020 Bumiputera. New 2020 districts
    borrow their parent heading's mix (`SEED_PARENT` in my_census.py: Kalabakan from Tawau,
    Telupid from Beluran, Tebedu from Serian, Pusa from Betong, Kabong from Saratok, Tanjung
    Manis from Daro, Sebauh from Bintulu, Bukit Mabong from Kapit, Subis from Miri, Beluru and
    Telang Usan from Marudi). All `modelled`.
  - **Home districts** (`HOME` in my2020.py) for the Sabah and Sarawak groups 2010 does not name:
    without them Rungus, Suluk, Orang Sungai, Kayan, Kenyah and the rest would follow 2010's
    "Other Bumiputera" everywhere in the state (Rungus over Tawau). Each is restricted to the
    districts its language is spoken in. These lists are mine, from where the language and
    people articles place them; the state counts are unchanged. Kudat comes out Rungus first
    (31,700), Kinabatangan Orang Sungai first, Belaga Kayan, Penan and Kenyah, Lawas Lundayeh
    second to Malay.
- In 2010 the Rungus were inside "Other Bumiputera", not Kadazan Dusun (Kudat: Kadazan Dusun
  3,674, Other Bumiputera 45,420), so Rungus follows Other Bumiputera inside its home districts.

## 5. Retention (§2: check retention first)

**No measured share was found for any Malaysian group.** Searched: Kadazandusun language-shift
studies (UNIMAS case study, ir.unimas.my/id/eprint/7485: first and second generations speak it,
third and fourth mostly Sabah Malay or English; no population share), the "560,000 speakers"
figure (it is the 2010 ethnic count restated), UNESCO's 300,000 (2005; no denominator year
given, not used). Nothing for Iban, Bidayuh, Melanau (known to be shifting to Sarawak Malay),
Murut, Bajau, Orang Asli, Chinese (some English-dominant homes) or Indians. So every group is
drawn whole on its language, and `note_public` says the smaller languages are likely drawn too
large.

## 6. Tree and calls

New nodes in `tree.d/my.txt` (its header carries the Glottolog checks): group North Borneo with
36 leaves (Kadazan-Dusun, Rungus, Bisaya, Orang Sungai, Murut, Tagal, Ida'an, Lundayeh (Lun
Bawang), Bulongan, Melanau, Kajang, Kayan, Kenyah, Kelabit, Penan, Berawan and Sarawak's small
groups down to Narom, 17 people); group Aslian; leaves Brunei Malay, Kedayan, Bidayuh, Bajau
(under ph.txt's Sama-Bajaw), Tausug and Iranun and Molbog (repeating ph.txt's). Calls:

- (Superseded by §9: Chinese on `sinitic`, unnamed; Indians 80/20.) Others on `other`.
- Senoi and Negrito on `aslian`, unnamed; Proto-Malay on `malayic`, unnamed (Semelai, Aslian,
  is inside it and cannot be split out).
- "Other Sabah / Sarawak Bumiputera" (505,000 drawn) on `austronesian`, unnamed.
- Lun Bawang/Murut (Sarawak) and Lundayuh/Lundayeh (Sabah) on one node: one people, one language.
- (Superseded by §9: non-citizens, 2,691,070, 8.3%, 23.7% of Sabah, were not drawn.)
- Hand-picked colours: Iban (generated, it sat beside Malay and Malayic), Brunei Malay, Kedayan
  and Orang Sungai (generated, both light greens beside Malay), Melanau (violet, against Malay
  and Iban on the Mukah coast), Kadazan-Dusun and Murut (yellow and amber in the interior),
  Bajau (blue, on the coast), Bidayuh (rose, beside Iban and Malay around Kuching), Aslian.

## 7. Geography

religiondots' `my_hexes.gpkg` (144,439 Kontur hexes, `unit` = MYS_ss_dd, 160 districts),
read-only, placed on population. Not new Kontur, so no cap check. Scatter: 942 rows on
population, none on equal shares; 21,600 people under one dot per language nationally drew none.

## 8. Room for improvement

- 2020 sub-ethnic groups by district for Sabah and Sarawak (the per-district eStatistik volumes,
  or the dead OwnCloud share): would replace the 2010 pattern and the home-district lists.
- Chinese dialect groups by DISTRICT, or from 2010 or 2020 (the 2010 microdata clearly has them:
  Penang Institute quotes Penang's 2010 split, but it is unpublished); a home-language measure
  for Chinese by dialect group, and a national one (§9: one small qualitative study).
- Non-citizens by citizenship and state (the census has the answer; no published table found),
  which would replace §9's nationality shares from work passes.
- A retention measure for any other group, and the Indian sub-groups by district.

## 9. Rework, 2026-10-19 (session eca0185b-my): Chinese dialects, Indian sub-groups, non-citizens

Anita, 2026-10-19: "chinese dialects in malaysia", and Malaysia was the third-greyest country
(27.3%: Chinese 6.89M on `sinitic`, Indians' 20%, Others). Files: `sources/my_extra.py` (the three
tables, writes `data/normalized/my_chinese_shares.csv`, `my_indian_mix.csv`,
`my_noncitizen_mix.csv`; run `python sources/my_extra.py`, no network), `countries/my.py`
(`_chinese`, `_noncitizens`), `taxonomy/my2020.py` (RETAIN, DIALECT_PLACE, SABAH_HOME),
`taxonomy/tree.d/my.txt` (Kwongsai, and a borrowed-nodes block from
`python sources/origin_mix.py --fragment my`: 45 nodes). Result: 3.2% on a group node or `other`
(Austronesian unnamed 505k, `other` 270k, Chinese unnamed 167k, Malayic unnamed 95k).

### 9.1 The Chinese table

**Found:** Voon Phin-Keong, "The Hakkas of Malaysia to 1970: Population, Livelihood, and Culture",
Malaysian Journal of Chinese Studies 13(1), 2024, 37-59 (CC BY 4.0,
ejournal.newera.edu.my/mjcs/article/view/23), Table 2, "Population and Percentage Share of Seven
Major Dialect Groups by State, 2000". It is the Department of Statistics' unpublished 2000
tabulation by dialect group (dialect identity self-reported by the household; Voon's note 2: the
tabulation is 330,000 or 5.7% short of the published Chinese total). 14 state rows (no Labuan or
Putrajaya), Hakka, Hokkien, Cantonese, Teochew, Hainanese, Foochow, Kwongsai, as counts in
10,000s and as % of the state's Chinese. Nationally (the same census, as quoted in Wikipedia,
"Malaysian Chinese", from Ember, Ember and Skoggard 2004 p. 698; Table 2's printed totals agree):
Hokkien 2,020,914, Hakka 1,092,835, Cantonese 1,068,008, Teochew 497,280, Hokchew (Foochow)
251,553, Hainanese 141,045, Kwongsai 51,674, Henghua 24,654, Hockchia 14,935, other Chinese
202,977, total 5,365,875. Not found: 2010 (Penang Institute holds it and quotes Penang's split,
Hokkien 63.9%, Teochew 17.8%, Cantonese 8.3%, Hakka 5.2%, Hainanese 1.5%, but says the language
data are not published) and 2020; no district grain from any year. Checks (my_extra.py): shares
add to 90.5-97.5% in every state (the rest is Henghua, Hockchia and other Chinese); the counts
over the shares give one state total per row within 12% (KL's Hakka count 10.9 is the odd one, a
typo for about 10.1; shares are used, not counts); the printed column totals match the rows
within rounding (Hakka 110.3 against 109.3 is the KL cell).

**Calls:**
1. **State shares from 2000 applied to 2020's Chinese per district.** The groups' shares by state
   cannot have moved much in 20 years; a mismatch of up to a few points is likely, and Mandarin's
   rise (below) is the bigger error. Labuan takes Sabah's shares, Putrajaya Selangor's.
2. **Dialect group is an ancestry label, not a home language, so only 53% are drawn on it.**
   Carstens (Portland State University), study of Malaysian Chinese language practice presented at
   UKM 27 May 2014 (ukm.my/news/?p=4668, read through the Wayback Machine because the live page
   was down on 2026-10-19): "currently 53% of the respondents speak Chinese dialects with their
   parents compared with 42% in 1970"; 93% of families use Mandarin, with dialects; none speak
   Malay at home; 25% use English with their families. Working people in towns, qualitative, so
   this is the best figure found, not a measure. The 47% who do not speak a dialect with their
   parents are drawn as `mandarin` (3.24M, the largest Chinese node). One rate for every
   group: nothing separates them, though Cantonese in KL and Ipoh, Hokkien in Penang and Foochow
   in Sibu are known to hold up better than Hakka (Liao 2017 on Sandakan Hakka; UNIMAS survey of
   300 in Kuching: Mandarin used more than Foochow or Hokkien, Hokkien more than Foochow). English
   homes are not drawn (the 25% is not exclusive with Mandarin or dialects). The retention rate
   is `RETAIN` in my2020.py: one number to change. Drawing the dialects whole (RETAIN = 1) would
   be the other defensible call; it gives the Hokkien 2.7M, Hakka 1.4M and Cantonese 1.4M of the
   ancestry labels.
3. **Henghua, Hockchia and other Chinese (4.5% nationally; 5.5% in Sarawak) stay on `sinitic`,
   the unnamed Chinese group**, at the same 53% (the rest Mandarin). The census names Henghua
   (24,654) and Hockchia (14,935) nationally, but no source gives them by state (Tien 1983 has
   Henghua 10,642 in Sarawak's 1970 census, 4,806 in Kuching, no other state), so placing them
   would be invented.
4. **Foochow is `min_dong` (Eastern Min; Fuzhou is its prestige variety)** whose legend label is
   "Min Dong", shared with cn, bn and ca. Relabelling to "Min Dong (Foochow)" would take one
   line in each of those fragments; not done here. Kwongsai (Guangxi Chinese, Yue varieties) has
   a leaf of its own, `sinitic.kwongsai`, colour gold (`0.76 0.13 85`) so it does not sit on
   Cantonese's pink in Bentong and Raub. Hokkien is `min_nan` ("Min Nan (Taiwanese, Hokkien)").
5. **Placement inside a state** (`DIALECT_PLACE`): the state's group totals are the census's, and
   each group is spread over the state's districts by iterative proportional fitting against
   the district Chinese counts, on soft weights (home districts 1, the rest 0.05 to 0.5). Only
   `modelled`. Each line's evidence:
   - Sarawak Foochow, the Rejang basin (Sibu, Sarikei, Maradong, Dalat, Mukah, Kanowit, Selangau,
     Kapit, Song, Julau, Bintulu, Tatau, Daro) at 1 and the rest at 0.1: Voon, "the Foochows were
     clustered in the Sibu area" (1960, p. 40); Tien Ju-K'ang 1983 (Southeast Asian Studies
     21(3), the Foochow settlement of Sibu); Wikipedia on Sibu as "mini-Fuzhou". Result: Sibu
     41,700 Foochow of 114,000 Chinese after retention (79,000 before), Kuching 20,900.
   - Sarawak Hakka, flat, which falls where the Foochow are not: Voon calls them "rather widely
     dispersed". Check: 61% of Sarawak's Hakka come out in the First Division (Kuching, Bau,
     Lundu, Samarahan, Serian, Simunjan, Asajaya, Tebedu) against 76% in the 1970 census (Tien
     1983: 91,610 Hakka, 69,471 in the First Division).
   - Perak Foochow in Manjung (Sitiawan), 1 against 0.05: Voon, p. 46, the Methodist mission
     brought Foochow settlers to Sitiawan. Result: 10,000 of Perak's 15,000 retained Foochow.
   - Pahang Kwongsai in Bentong and Raub, 1 against 0.1: Wikipedia, "Malaysian Chinese",
     Kwongsai "have a substantial speaking population in Bentong and Raub".
   - Hokkien in Klang, Batu Pahat, Muar and Timur Laut (George Town); Cantonese in KL, Kinta
     (Ipoh), Seremban and Kuantan; Hakka in Cameron Highlands and Kulai; Teochew in Perak's
     west coast (Manjung, Hilir Perak, Larut Dan Matang, Kerian) and Johor Bahru: the same
     Wikipedia passage (notes 159-166, Gabaccia and Hoerder 2011 and others), each at 1 against
     0.5, so a mild lean only. A secondary source; reverse them by emptying `DIALECT_PLACE`.
   - Sabah and every other state: flat. A forum and Liao 2017 say Sandakan is Cantonese-speaking
     or at least Cantonese-led in business; Liao also finds "little evidence" that Cantonese
     speakers are the majority, so Sandakan is not boosted.
6. **Result (national, people drawn):** Mandarin 3,239,000; Hokkien 1,425,000; Cantonese 730,000;
   Hakka 719,000; Teochew 334,000; Chinese unnamed 167,000; Foochow 150,000; Hainanese 96,000;
   Kwongsai 33,000. The 6.89M Chinese citizens were 6.89M on `sinitic`.

### 9.2 Indians

The first build's 80% Tamil came from the Tamil share of the migration to Malaya (Rajoo 1985).
Replaced by a sub-group table: P. Bala, "Malaysian Indians", *Worldmark Encyclopedia of Cultures
and Daily Life* (encyclopedia.com), whose table of Indian sub-groups has no printed source or
year beside it (the article's population line says 2004). It is the census's coding (Indian
Tamil, Sri Lankan Tamil, Malayali, Telugu, Sikh, Punjabi, other Indian, Pakistani, Bangladeshi,
Singhalese) and its total, 1,593,777, is 4% under the 2000 census's Indians, the same kind of
gap as Voon's for the Chinese, so it is taken as the 2000 tabulation. **Used as shares, not
counts:** Tamil 88.2% (Indian Tamil 87.6 and Sri Lankan Tamil 0.5, with the Pakistani and the
Singhalese rounded in), Punjabi 3.6 (Punjabi 1.5 and Sikh 2.1), Telugu 2.4, Malayali 2.3, other
Indian 2.6 on `other`, Pakistani 0.7 on Pakistan's mix (Punjabi 37, Pashto 18, Sindhi 15...),
Bangladeshi 0.2 Bengali, Singhalese 0.1 Sinhala. The shares are flat over districts: Punjabi
(Sikh) are known to be Perak, Selangor and Penang, but no source gives the places. The older
figure is higher on Tamil (1970, 80%); the Indian High Commission and the 2018 minorities
directory say "about 80%", so 88% may be a little high. Rows `modelled`. Not drawn: English as a
home language of Malaysian Indians (large in towns, no figure).

### 9.3 Non-citizens (2,690,600 in OpenDOSM's 2020 district rows; 810,500 in Sabah, 23.7%)

The census counts them by district and says nothing public of their nationality. Searched: DOSM
Key Findings 2020, the migration statistics release (2025: national, by district for 2025), the
UN migration forum slides (national), the 2010 volumes: no nationality table by state or district.
Not an ask (§7: no emails to offices).

- **Nationality, outside Sabah:** temporary-pass (PLKS) foreign workers to June 2018, Ministry of
  Human Resources (mohr.gov.my, BPA Report, page 27): Indonesia 39%, Nepal 22%, Bangladesh 17%,
  India 6%, Myanmar 6%; the other 10% (many small countries) left out and the five scaled up:
  Indonesia 43.3, Nepal 24.4, Bangladesh 18.9, India 6.7, Myanmar 6.7. Workers are not all
  non-citizens (dependants, students, professionals, undocumented), but nothing else gives a mix.
- **Nationality, Sabah:** Sabah's quota allows only Indonesian and Filipino workers. 50%
  Indonesian, 45% Filipino, 5% others (the default mix without Indonesia). Evidence bracketing it:
  plantation workers 90% Indonesian (Bernama 2016 via MPOB); registered workers 2019-20 about 30%
  Filipino (Pappas, Sabah recruiters); Filipino refugees with IMM13 permits 136,055 (ESSCom, Dec
  2020), a majority of IMM13 holders; prison inmates in 2015, 78% of foreigners Filipino;
  Wikipedia's undated estimate 1,000,000 Filipino and 500,000 Indonesian (not census counts). A
  judgement; moving it is `SABAH_NAT` in my_extra.py.
- **Languages, by nationality** (origin_mix.py, the home mix, AGENT_BRIEF §2 diaspora rule):
  Indonesia, Nepal, Bangladesh, Myanmar as their drawn maps. **India: Tamil Nadu's 2011 mix
  (uncited):** Indian workers in Malaysia are taken as largely Tamil, not India's Hindi-first
  mix. **Sabah's Indonesians:** Wan Shawaluddin Wan Hassan, Marja Azlima Omar and Ramli Dollah,
  "The Illegal Immigrants in Sabah: Why Do They Come Back?", Borneo Research Journal 4 (2010),
  115-128: "the Indonesian immigrants consist of the Bugis, Torajas and Flores from Sulawesi and
  Flores islands or the Muruts from the Kalimantan-Sabah borders", the Bugis the estate
  supervisors and main workforce. Bugis 50%, Toraja 10, Lamaholot 5, Manggarai 5 (the Flores
  groups), the other 30% at Indonesia's own mix (Javanese, Indonesian, Sundanese...): the shares
  are my judgement, the names are the paper's. **Sabah's Filipinos:** "Most of the immigrants
  from the Philippines are Muslims and they are either of the Bajaus or Sulu'", the Bajau
  "bigger in number" than the Suluk (same paper): Sama 50%, Tausug 40, the Philippines' own mix 10
  (Christian Filipinos from the north; Pappas's 30% of registered workers are mostly these).
  Sama is ph.txt's `sama_bajaw.sama` ("Sama"), not Malaysia's own Sabah `Bajau` node.
  Myanmar is its home mix, though many in Malaysia are Rohingya or Chin refugees (UNHCR: Myanmar
  86% of registered refugees, Dec 2022).
- **Where in Sabah:** the same paper: Indonesians "in Tawau, Kunak, Lahad Datu, Sandakan and also
  in Kota Kinabalu"; Filipinos "in the eastern part of Sabah especially Sandakan, Lahad Datu,
  Kudat" (it also names Semporna and Tawau). Soft weights (`SABAH_HOME`, home 1, the rest 0.3),
  fitted to each district's census count and the state's nationality totals. Indonesians also
  lean to the plantation districts (Kinabatangan, Beluran, Tongod, Telupid, Kalabakan). Result:
  the largest non-citizen districts are Sandakan 141,000, Tawau 109,000, Kinabatangan 100,000;
  nationally Bugis 220,000,
  Sama 182,000, Tausug 219,000 (Sabah's Tausug plus the Philippines' own), Nepali 235,000,
  Bengali 332,000.
- Rows `derived` (Anita). Placed on population inside the district. The non-citizens' census
  counts are the only measured part; the districts' foreign-worker-heavy places (Kinabatangan,
  Kunak, Kalabakan, Sepang) are right because the census counts them there.

### 9.4 What else changed

`countries/my.py` ENTRY: `how`, `source`, `parts` (Chinese 6.89M, Indians 2.00M, non-citizens
2.69M, the rest), `note_public`; `gap` removed (non-citizens are drawn). The check that the
counts add up to the districts' citizens plus non-citizens (32,445,200) replaces the citizens-only
check. 119 nodes (was 48): the rest are the nationality languages. `scatter.py --country my`:
32,387 dots, 58,200 people (0.18%) under one dot per language nationally draw none.

### 9.5 Room for improvement (added)

A Chinese table by district or from 2010 or 2020; a national home-language survey of Chinese
Malaysians by dialect group (RETAIN is one rate for all); the census's own nationality counts for
non-citizens by state or district; Sabah's and Sarawak's per-district 2020 sub-ethnic tables
(§2).
