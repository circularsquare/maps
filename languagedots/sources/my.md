# Malaysia: 2020 census ethnic group, read as language, 160 districts

Drawn 2026-10-05 (session edd42a8c-my). `sources/my_census.py`, `taxonomy/my2020.py`,
`taxonomy/tree.d/my.txt`, `countries/my.py`. Build tail left to the supervisor.

**29,754,600 citizens, 160 administrative districts (203,000 people on average), 48 nodes,
29,733 dots at 1:1000. 25.2M rows `derived` (district counts read as language), 4.55M (15.3%)
`modelled` (sub-ethnic groups counted by state and spread over districts). 2,691,070
non-citizens not drawn.**

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
- **Chinese dialect groups**: the 1970-2000 censuses coded Hokkien, Cantonese, Hakka, Teochew,
  Hainanese, Kwongsai, Hokchiu, Henghua, Hokchia (the 1970 master-file codebook, SEAFERT,
  `seafert.csde.washington.edu/n/m/malay70.pdf`), but no open table by state or district was
  found; the microdata (SEAFERT, IPUMS) need the statistics office's or IPUMS's permission
  (IPUMS account blocked, gated). Wikipedia's *Malaysian Chinese* has no dialect table. 2010 and
  2020 publish none.
- **Indian sub-groups**: the same 1970 coding (Indian Tamil, Telugu, Malayali, Punjabi, Ceylon
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

- **Chinese, Others**: the district's own count, `derived`. **Indians**: 80% Tamil, 20% `other`,
  `derived`.
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

- Chinese on `sinitic`, unnamed. Indians 80/20 (my2020.py docstring has the source). Others on
  `other`.
- Senoi and Negrito on `aslian`, unnamed; Proto-Malay on `malayic`, unnamed (Semelai, Aslian,
  is inside it and cannot be split out).
- "Other Sabah / Sarawak Bumiputera" (505,000 drawn) on `austronesian`, unnamed.
- Lun Bawang/Murut (Sarawak) and Lundayuh/Lundayeh (Sabah) on one node: one people, one language.
- Non-citizens (2,691,070, 8.3%; 23.7% of Sabah) not drawn, in `gap`: the census gives them
  no ethnic group, and a nationality proxy would need a census table by state that was not found.
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
- A Chinese dialect table at any grain (1970-2000 census reports in a library, or microdata
  access): would split `sinitic` into Hokkien, Hakka, Cantonese, Teochew, Foochow, Hainanese.
- An Indian sub-group table, and a retention measure for any group.
- Non-citizens by citizenship and state, to draw them by nationality.
