# Indonesia: 2010 census, language used daily at home, 33 provinces

Drawn 2026-10-04 (session d9e44929-id). `sources/id_sp2010.py`, `taxonomy/id2010.py`,
`taxonomy/tree.d/id.txt`, `countries/id.py`. Follow-up 2026-10-05 (session d9e44929-id2): ask
009 ruled yes, and the 42.6M "other regional languages" remainder is now shared out by the same
book's province x ethnic group table (§5), every such row `modelled`. Widened 2026-10-05
(session edd42a8c-id2, Anita: "more regional languages of Indonesia"): the share-out is raked to
BPS's national language table, split into ~100 languages with Ananta et al.'s ethnic counts, and
placed inside provinces by Glottolog points (§5, §6; `sources/id_shareout.py`). Papua,
2026-10-06 (session 5d7dac7e-pap, Anita: Indonesian New Guinea did not read as diverse): the
Papuan-language speakers of Papua and Papua Barat are split among 234 Glottolog languages by
regency (§10; `sources/id_papua.py`).

**213,495,218 people aged five and over, 333 nodes, 33 provinces (6.5 million people on average),
213,349 dots and 114 rings; 42,560,185 (19.9%) modelled, of whom 1,745,279 (0.8%) are still
unnamed.**

## 1. The table

*Kewarganegaraan, Suku Bangsa, Agama, dan Bahasa Sehari-hari Penduduk Indonesia: Hasil Sensus
Penduduk 2010* (BPS, 2011; ISBN 978-979-064-417-5, catalogue 2102032, 64 PDF pages). Publication
page `https://www.bps.go.id/en/publication/2012/05/23/55eca38b7fe0830834605b35/...` (Cloudflare
403 to a script; read through the Wayback Machine, snapshot 20250505150932). The page links a
`https://web-api.bps.go.id/download.php?f=<token>` URL, and **web-api.bps.go.id serves the PDF
with no key or session** (the full token is in `id_sp2010.py`). Saved as
`data/raw/id/sp2010_kewarganegaraan_suku_bangsa_agama_bahasa.pdf`. Encrypted with an owner
password only; PyMuPDF reads its text layer, one cell per line.

The question (Catatan Teknis, PDF p.14): *bahasa sehari-hari*, the language usually used with the
other household members at home, persons aged 5+, one answer, recorded as one of three boxes
(Indonesian / regional, named / foreign, named) and coded to BPS's 1,211-language list. The
three-box form puts Indonesian first, so a household that talks Indonesian at home is Indonesian.

Tables used, all "only from document SP2010-C1", 214,056,929 persons:

| table | PDF p. | what |
|---|---|---|
| L4.1 | 56 | national, 35 language groups, ranked |
| L4.2 | 57 | 33 provinces x Indonesian / regional / foreign / not answered / total |
| L4.3 | 58 | L4.2 as row percentages (used only as a check) |
| L4.5 | 60-61 | 33 provinces x Jawa, Indonesia, Sunda, Melayu, Madura, Minangkabau, Banjar, Bugis, not answered, Lainnya |
| L2.6 | 45-50 | 33 provinces x 31 ethnic groups (citizens, every age, forms C1 and C2-Apartemen) |
| P1.2 | 37 | national count per ethnic group (check on L2.6) |

What is written per province: the eight named groups, not answered, and L4.5's *Lainnya* split
by L4.2's boxes: **"Bahasa daerah lainnya" = L4.2 regional box minus L4.5's seven regional
languages**, and **"Bahasa asing" = L4.2 foreign box**. That is arithmetic on two tables of the
same census, not a model; check 4 asserts that the two parts are non-negative and sum to Lainnya
in every province.

## 2. Nothing finer, and what was searched

- `sensus.bps.go.id` (the open SP2010 microsite religiondots uses to kecamatan) has, for
  language, only "Kemampuan Berbahasa Indonesia" (ability to speak Indonesian), datasets
  `/topik/dataset/sp2010/1`-`9` read through on 2026-10-04.
- The BPS province and regency SP2010 publications seen (e.g. Klaten's *Penduduk menurut wilayah
  dan karakteristik*) carry no language table.
- The 2020 long form asked only the three-way split (coverage sweep, USCB workbook).
- IPUMS 2010: the sweep reports language coded Indonesian / regional / foreign; gated anyway.

So province is the finest grain, and only eight languages are named at it.

Searched again 2026-10-05 (session edd42a8c-id2) for anything finer on ethnicity or language:
- sensus.bps.go.id: its topic page lists no ethnicity dataset (citizenship, religion, age, urban,
  document type only), and its 316 SP2010 publications (`/publikasi/index/sp2010`, all 32 pages
  read) include no ethnicity or language title: early aggregate counts per regency, youth,
  elderly, housing. The national book says (§1.3) that province and regency analyses "will be
  presented in a separate publication"; none was found.
- BPS Bali, *Peta Sebaran Penduduk menurut Suku Bangsa Provinsi Bali, SP2000 dan 2010*: Bali's
  top ten groups by regency and kecamatan, as maps. Not fetched: Bali is 97% Balinese in the
  remainder, so it would move almost nothing. The one provincial ethnic-map publication seen.
- Ananta et al., *Demography of Indonesia's Ethnicity* (ISEAS 2015): province tables of the
  largest groups exist (Appendix 1-2) but the book is paywalled (Cambridge Core); its national
  145-group table (pp.119-122) is reproduced in English Wikipedia and is used (§5). Their ISEAS
  working paper 2014/1 is open but gives the classification only, no counts. Their Papua paper
  (Asia & the Pacific Policy Studies 3(3), 2016, open access) has regency figures for Papua.
  Wiley returns 403, but the Wayback Machine holds the full text; used since 2026-10-06 (§10).
- Ananta et al., IUSSP 2013 poster paper (open): Table 4, own-language retention for the 15
  largest groups, used (§5).
- Batak sub-groups: only Katadata's (databoks, 2021) North Sumatra percentages, attributed to
  SP2010 (Tapanuli/Toba 25.62, Mandailing 11.27, Karo 5.09, Simalungun 2.04, Pakpak 0.73). They
  match figures usually quoted for the 2000 census, so the vintage is uncertain; used as shares.
- No source splits Dayak, so it stays one leaf (placed by every Dayak point in Glottolog).
- SUSENAS and the 2020 long form name no regional languages (coverage sweep).

## 3. Checks (all asserted in `id_sp2010.py`)

1. L4.2: every province's four columns sum to its Total; the 33 provinces sum to the printed
   INDONESIA row, every column (214,056,929).
2. L4.5: every province's ten columns sum to L4.2's Total for that province; the 33 sum to the
   printed Total Nasional row, every column.
3. L4.2 and L4.5 agree on Indonesian and not answered in every province.
4. Lainnya = (regional box - seven named) + foreign box, both parts >= 0, every province.
5. L4.5's national totals equal L4.1's rows (Jawa 68,044,660; Indonesia 42,682,566; Sunda
   32,412,752; Melayu 7,901,386; Madura 7,743,533; Minangkabau 4,232,226; Banjar 3,651,626; Bugis
   3,510,249; not answered 561,711).
   **L4.1's 35 printed rows sum to 213,744,867, 312,062 short of its own Grand Total.** The gap is
   exactly L4.2's foreign box (756,035) minus L4.1's foreign row (443,973), and L4.1's regional
   rows plus its sign-language row (40,373) are exactly L4.2's regional box (170,056,617). So L4.1
   leaves one foreign row unprinted (plausibly the Chinese varieties, which BPS codes 1161-1175
   under "Tionghoa"; not provable), and **BPS filed sign language in the regional-language box**.
   Both are asserted.
6. L4.3's percentages recomputed from L4.2 to 0.01 point, every province.
7. L2.6: each of the 31 groups' 33 provinces sums to its P1.2 national figure, in printed order
   (which also proves the six pages' columns were read in order), and all sum to P1.2's Total,
   236,728,379 citizens.
8. The share-out: every province's modelled rows sum exactly to its measured remainder
   (largest-remainder rounding), and the national total is 42,560,185, unchanged. Since
   2026-10-05 also: L4.1's 24 groups plus sign language sum to the remainder exactly, and the
   rake meets every group's national total (to 0.004 people before rounding).

Not available: a second table per province to cross-check L4.5's eight languages against; the
book has none. Ethnicity (L2.6) is related but is not the same question.

## 4. Mapping calls (`taxonomy/id2010.py`)

- The eight on their own leaves: Javanese, Sundanese, Madurese, Buginese (Malayo-Polynesian,
  directly under `austronesian` as other fragments already had them); Indonesian, Malay,
  Minangkabau and Banjar under `malayic`. **Banjar is new** (Glottolog banj1239, Malayic).
- **"Melayu" is BPS's Melayu group only.** L4.1 prints Melayu Perdagangan (trade Malays such as
  Ambonese and Manado Malay, 2.24M), Melayu Tengah (1.44M), Musi/Palembang/Sekayu (2.18M) and
  Betawi (2.24M) as separate groups, and none of them is in L4.5's Melayu column. They are in the
  remainder.
- **What the share-out (§5) cannot name goes on a new areal root, `indonesia_other`** ("Other
  regional languages of Indonesia"), the region's equivalent of `americas_other`. It crosses Austronesian
  and the Papuan families (Timor-Alor-Pantar in NTT and Maluku, North Halmahera, Papua), so no
  family node contains it (the signers are on `signlanguage` since 2026-10-05). A split by place (`austronesian` in the western
  provinces) was considered and not done: migrants carry Papuan languages west, signers are
  everywhere, and both nodes are drawn washed out anyway.
- The foreign box goes on `other`: mostly Chinese by its geography (Kalimantan Barat 260,156,
  Sumatera Utara 227,061, Kepulauan Riau 72,649, Riau, Jakarta, Bangka Belitung), but BPS never
  names it by province. Not `sinitic`: Arabic, English, Dutch, Hindi and others are in it too.
- Not answered (561,711) is not drawn; it is in `gap`.

## 5. The remainder, shared out (ask 009; widened 2026-10-05)

The measured remainder, "Bahasa daerah lainnya", is 42,560,185 people, 19.9% of those drawn, and
the majority in North Sulawesi (98%), Maluku (94%), North Maluku (93%), NTB (93%), Bali (84%),
Aceh (79%), NTT (75%), West Sulawesi (69%), South Sumatra (62%) and Papua (59%).

**Anita's ruling, 2026-10-05: "okay this multiplication seems fine"**, and later the same day
"nice would be getting more regional languages of Indonesia". The first share-out (session
d9e44929-id2) divided each province's remainder among L2.6's ethnic groups by their numbers and
left 18.9M on `indonesia_other`. It now runs in `sources/id_shareout.py` (its docstring is the
method), in three stages at province grain, every row `modelled`:

0. **Fine groups per province.** Each L2.6 group is split into the groups of Ananta et al.'s
   "new classification" of the same census (*Demography of Indonesia's Ethnicity*, 2015,
   pp.119-122: national counts of the 145 largest, read from English Wikipedia's copy of the
   table), by a rake to L2.6's province counts and NC's national counts, seeded by nearness of
   each province's people to the group's Glottolog point(s) (50 km kernel, 95%). "Suku asal NTT"
   becomes Atoni, Manggarai, Sumba, Lamaholot, Ngada, Rote, Alor, Lio, Savu ...; "Sulawesi
   lainnya" becomes Toraja, Mandar, Buton, Kaili, Tolaki, Sangir ...; "Papua" becomes Dani, Mee,
   Biak, Yali, Asmat ... NC members under 20,000 and groups NC does not split stay as the group's
   residual. Crosswalk calls are commented in `GROUPS`:
   - Jambi: NC moved Jambi Malay into Malay, and Jambi's remainder is 33,000 against 1.34M ethnic
     Jambi there: their language was coded Melayu (L4.5). Only Kerinci takes a share. The same for
     Bangka, Belitung and Akit (Bangka Belitung's remainder is 24,000).
   - South Sumatra's residual is NC's Melayu Lahat and Semendo, which Glottolog files as South
     Barisan (Central) Malay: on BPS's "Melayu Tengah".
   - "Timor Leste origin" (269,368) is read as Tetun, on tl.txt's Tetun Terik (the Tetun of Belu).
     "Flores" (an island) and Makian (two languages of two families) stay unnamed.
   - Ambon, Saparua, Haruku and Banda people go to the trade Malay column only.
1. **Province x L4.1 group.** L4.1 counts the remainder nationally in 24 groups (Bali 3.37M,
   Batak 3.32M, Cirebon-Indramayu 3.09M, NTT 3.00M, Sasak, Aceh, Betawi, Melayu Perdagangan 2.24M,
   Musi/Palembang 2.18M, Sulselbar 1.95M, ...) plus sign language; they sum to the remainder
   exactly. A rake to the provinces' remainders (rows) and those national totals (columns), seeded
   by stage 0 times the share of the group that used its own language at home in 2010 (IUSSP 2013
   Table 4: Batak 43%, Betawi 25%, Balinese 93%, Acehnese 84%, Dayak 62%, Sasak 94%, all others
   32%), each group seeding the group(s) its language is filed in. Eastern groups also seed
   "Melayu Perdagangan" (Minahasa and Papua and Maluku groups half, Sulawesi Utara groups a fifth,
   NTT 15%), which no ethnic group stands for. "Sulawesi Utara" and "lain asal Sulawesi" are raked
   as one constraint because nothing says which of the two holds Gorontalo. Sign language is
   seeded by population.
2. **Languages.** Within a province x group, by stage 0 x retention. Batak by Katadata's North
   Sumatra shares in every province (§2). Melayu Perdagangan is named by place (AGENT_BRIEF §3, a
   label whose meaning depends on place): Manado Malay in North and Central Sulawesi and
   Gorontalo, Ambonese Malay in Maluku, North Moluccan Malay in North Maluku, Papuan Malay in both
   Papuas, Kupang Malay in NTT; elsewhere (122,549) it stays on the group node, BPS's own label.

What this changes against the first share-out: the L4.1 groups are now met exactly, so Batak
(was 1.26x L4.1), Betawi (1.30x) and Cirebonese (0.50x) are right nationally, and Manado Malay
(1.31M) now takes most of North Sulawesi, where Minahasan had been overstated. Unnamed falls
from 18.9M to 2.18M. Largest new languages: Musi/Palembang 2.13M, Toba Batak 1.90M, Central
Malay 1.31M, Manado Malay 1.31M, Mandailing-Angkola 0.84M, Bima 0.70M, Uab Meto 0.65M, Toraja
0.62M, Manggarai 0.52M, Mandar 0.49M, Sumba 0.47M, Rejang 0.46M, Komering 0.44M, Buton 0.44M,
Dani 0.38M. 121 nodes in all.

Calls someone might reverse:
- Reading Katadata's North Sumatra Batak shares (perhaps 2000's) into every province.
- Retention from IUSSP's national table applied in every province; "Others" (32%) for every
  group outside the fifteen largest.
- The trade Malay seed shares (half, a fifth, 15%) are guesses; only the national total is
  measured.
- Group leaves for ethnic clusters that speak several languages (Sumba, Buton, Kaili, Seram,
  Tanimbar, Aru, Babar, Rote, Alor-Pantar), as Dayak already was. (Yapen and Arfak were too,
  until §10 split Papua's clusters into languages.)

## 6. Geography

religiondots' `data/geo/id/id_hexes.gpkg` (Kontur 400 m hexes keyed to 5,122 kecamatan, 89
regencies and one residual), read-only. `countries/id.py` keys each hex to its province by the
first two digits of its BPS code, which every unit id carries; the one 2-digit unit, `65`, is
religiondots' Kalimantan Utara residual, and Kalimantan Utara was carved out of Kalimantan Timur
in 2012, so in 2010 it is `64`. Asserted: every unit id is a 2-, 4- or 7-digit code, and the
re-keyed set is exactly the 33 provinces of id.csv. Provinces use the 2010 codes (Papua Barat 91,
Papua 94), which are religiondots' too.

**Placement inside provinces (2026-10-05; AGENT_BRIEF §4.4, placement only).** Per province a
kecamatan x language table is raked to the kecamatan's Kontur population and the province's
counts, seeded for each regional language by exp(-distance to its nearest Glottolog point / 30
km) (99.8% of the seed; the rest even). `sources/id_units.py` writes each unit's
population-weighted centre to `data/geo/id/id_units.csv`. Even seeds for Indonesian, Malay,
Banjar, Javanese, Sundanese, `other`, sign language, the unnamed remainder, any language whose
points are over 150 km from all of a province's people, and any language with 40% or more of its
province (Acehnese in Aceh, Balinese in Bali, Sasak in NTB). Minangkabau and Buginese use their
points where they are minorities; Madurese is seeded on Madura's four regencies (its point,
used as a kernel, filled Surabaya at 59% before Sumenep; since 2026-10-19 Java's big
languages are seeded by homeland priors instead, §12). Dayak uses every non-Malay Austronesian
language Glottolog places in Indonesian Borneo. Checked by regency (kernel 30 km): Madura 89%
Madurese, Toba Samosir 81% Toba, Tana Toraja 41% Toraja, Majene 59% Mandar, Jayawijaya 52% Dani,
Minahasa 53% Manado Malay. **Known weakness:** languages with even seeds take their provincial
share everywhere, so a homeland draws its language under the truth (Samosir 48% Toba, 31%
Indonesian; Aceh Tengah 20% Gayo, because Gayo's single point sits east of Takengon). At kernel
40 km and 90% Tana Toraja drew 27% Toraja.

Scatter: two Papua highland blocks hit Kontur's cap and were lowered by religiondots' registered
entries (regencies 9431 and 9432). 213,436 dots, no rings.

## 7. Colours (`taxonomy/tree.d/id.txt`)

The eight measured languages as before: Javanese a mid blue (0.66 0.14 248), Sundanese a light
lime, Madurese a mid teal-green, Indonesian a light cyan, Malay sg.txt's light green, Minangkabau
an olive, Banjar a light lilac, Buginese a light leaf green; `indonesia_other` a muted teal. The
first share-out's twelve sit violet to rose (Acehnese deep violet, Batak pink, Balinese magenta,
Sasak light salmon, Dayak deep rose, Makassarese magenta, Minahasan violet, Gorontalo light pink).

The 2026-10-05 languages are hand-picked where they meet, the rest generated (the fragment's
comment lists the pairs): Toba pink, Mandailing coral, Karo pale lilac, Simalungun deep rose,
Gayo amber; Musi deep teal, Central Malay khaki, Komering light pink, Rejang coral; Bima
blue-violet, Sumbawa sky; Manggarai magenta, Ngada pale yellow, Lio blue-violet, Lamaholot coral,
Sumba orchid, Uab Meto tl.txt's Baikenu blue (one language across the Oecusse border), Rote amber,
Hawu coral; Toraja orange, Mandar blue-violet, Tae' pale gold, Buton coral, Muna orchid, Tolaki
blue-violet, Kaili red, Pamona orchid, Sangir light coral, Mongondow deep red; the trade Malays
mint to light blue; North Halmahera a yellow group, Bird's Head a green one, Dani green, Ekari
pale yellow-green. Checked on dot renders of Sulawesi, Nusa Tenggara and North Sumatra
(`review_id_sulawesi.png`). Tetun's teal sits near Indonesian's cyan on Timor; left.

**Diaspora mixes.** `sources/origin_mix.py` takes Indonesia's languages over 1% into other
countries' immigrant mixes. Musi/Palembang (1.02%) is the only new node over 1%, and it is defined
in this fragment; countries whose borrowed-node blocks list Indonesia's nodes may want
`python sources/origin_mix.py --fragment <cc>` rerun. Batak and the unnamed remainder drop
under 1%.

## 8. Wording

- `how`: "census, 2010, language used daily at home; the 20% who used a regional language other
  than the eight largest estimated by language from the census's ethnic groups and its national
  language table".
- `grain`: the 33 provinces, and that inside a province each language is placed by Glottolog.
- `source`: L4.1, L4.2, L4.5, L2.6; Ananta et al. 2015; Glottolog.
- `gap`: 22,678,702 children under five (sensus.bps.go.id, SP2010 *Penduduk Menurut Kelompok Umur
  dan Jenis Kelamin*, 0-4) were not asked; 237,641,326 - 22,678,702 - 214,056,929 = 905,695
  people were counted on the shorter forms (C2, L2), which do not carry the question (Catatan
  Teknis 2.3); and 561,711 did not answer.
- `note_public`: the eight-languages limit; how the other fifth is estimated (national groups,
  ethnic groups, retention, Ananta's counts, Batak shares); the trade Malays named by province;
  about 2 million unnamed; placement by Glottolog with the big languages filling the rest;
  Madurese filling Madura and the rest spread across East Java; the Indonesian-first coding.

## 9. Room for improvement

A real province or regency table of languages would replace all of §5. Short of that: the open
Papua paper's regency figures (§2); Ananta's province tables (paywalled) in place of stage 0's
nearness seeds; a Batak split with a known vintage; regency homelands for the even-seeded big
languages, which would let minority homelands fill to their true share.

## 10. Indonesian New Guinea by regency (2026-10-06, session 5d7dac7e-pap)

Anita: the Papua provinces did not read as diverse beside PNG's 779 languages. **Before:** Papua
(94, 2.47M aged 5+) drew Indonesian 37.0%, one Dani cluster 15.5%, Ekari 7.4%, Papuan Malay 7.3%,
"other regional languages" 5.7% and 18 more clusters; Papua Barat (91, 0.66M) Indonesian 70.0%,
unnamed 8.4%, Javanese 6.2% and eight clusters (Arfak, Maybrat, Biak, Baham, Yapen, Moi ...).
§5 split the Papua column among Ananta 2015's 22 national clusters plus a residual, each placed
by one Glottolog point.

**After:** the same column (§5 stage 1, unchanged: the people who used a Papuan language at home,
1,251,393 in Papua and 115,123 in Papua Barat) is shared among 234 languages, 208 drawn in Papua
and 68 in Papua Barat. Unnamed in the two provinces falls from 196,703 to 35,067 (what is left is
other groups' residuals). Indonesian, Javanese, Bugis and the rest of the measured eight, and
Papuan Malay, are untouched. Papua now: Indonesian 37.0%, Western Dani 9.5%, Ekari 7.5%, Papuan
Malay 7.3%, Biak 3.5%, Mid Grand Valley Dani 3.0%, Javanese 2.6%, Nduga 2.3%, Angguruk Yali 1.6%,
Ninia Yali 1.2%, and 87 Papuan languages over 1,000 speakers. Papua Barat stays 70% Indonesian
(measured), its Papuan 17% now over 68 languages (Biak, Maybrat, Sougb, Meyah, Wandamen, Baham,
Moi, Tehit ...).

### Sources

- **Ananta, Utami and Handayani, "Statistics on Ethnic Diversity in the Land of Papua,
  Indonesia"**, Asia & the Pacific Policy Studies 3(3), 2016, pp.458-474, doi 10.1002/app5.143,
  CC BY-NC-ND, from the 2010 census's raw ethnicity file. Wiley's PDF and full text answer 403
  to a script (OpenAlex and Semantic Scholar point only there; Europe PMC has nothing; CORE
  rate-limited; cyberleninka refused the connection). The full text is in the Wayback Machine,
  `https://web.archive.org/web/20230204023049id_/https://onlinelibrary.wiley.com/doi/full/10.1002/app5.143`,
  saved as `data/raw/id/ananta2016_papua_app5.143_wayback20230204.html`. Tables 1-2: each
  province's 25 largest groups (citizens, all ages: Papua Barat 753,399, Papua 2,780,144); Table
  3: each of the 40 regencies' Papuan share, and its largest group with its share; Table 4: each
  regency's Javanese share. Typed into `id_papua.py` (REGENCY, PROVINCE). One misprint: Table 4
  gives Biak Numfor's Javanese share as 69.89, which is Table 3's Biak share there; read as
  missing.
- Glottolog 5 (points, classification, endangerment); Joshua Project's Indonesia rows summed by
  ISO code (the speaker estimates, as `pg.md`); religiondots' 2010 census regency totals
  (`religiondots/data/normalized/id.csv`, "Total" rows, read-only; 3,593,803 for the 40);
  religiondots' regency polygons.
- Not found: a BPS regency table of ethnicity or language for Papua beyond what Ananta prints;
  SIL's Indonesian survey reports are per language, not per regency, with no counts to sum.

### The model (`sources/id_papua.py`; its docstring is the method; every row `modelled`)

1. Languages: every Glottolog language whose point is in one of the 40 regencies of 2010, or
   within 40 km of one if Glottolog lists Indonesia (16 snapped: Pyu, Sougb, Meyah, Amanab,
   Ngalum, Yonggom ...). Left out: 9 pidgin/unclassifiable, Papuan Malay (the share-out's own
   row), 46 extinct, nearly extinct or moribund (Inanwatan among them: Glottolog, after de Vries
   2006, says only people over 50 speak it). 234 drawn; 226 have a Joshua Project figure (2.18M
   in all), 8 take their regency's median.
2. Indigenous people per regency: census population x Table 3's Papuan share.
3. A regency x language table: each language's estimate spread over regencies by the Kontur
   people near its point (exp(-d/25 km)). Table 3's largest Papuan group is fixed in its regency
   (31 of 40) and shared among its languages by that spread; the rest is raked to the regencies'
   remaining indigenous people and to Tables 1-2's province counts of 27 named groups, with no
   other language or group allowed above the fixed largest one. Kota Jayapura and Kota Sorong
   take half their seed from the whole province's mix (PNG's Port Moresby takes the national
   mix).
4. A province's Papuan-language speakers are shared by this table's province totals; outside
   the two provinces (Papuan migrants elsewhere) by both together.

Ethnic label -> languages (`GROUP_LANGS`): Dani = Western Dani, Mid/Upper/Lower Grand Valley
Dani, Walak, Nggem; Ngalik = the three Yali; Dauwa = Nduga (98% of Nduga regency); Auwye/Mee =
Ekari; Mimika = Kamoro; Ayfat = Maybrat; Karon = Abun and Karon Dori; Arfak = Hatam, Meyah, Sougb,
Moskona; Asmat = Central, Casuarina Coast, Yaosakor and North Asmat with Citak and Tamnim Citak;
Yapen = every language on Kepulauan Yapen but Biak; Wandamen and Wamesa (Table 1 prints both) =
Wandamen; Mandobo = the three Mandobo. Not identified, so not fixed: "Aikwakai" (Teluk Bintuni's
largest, 20%) and "Biga" (Sarmi's largest, 15%; Glottolog's Biga is a Raja Ampat language, so
probably a different BPS code).

### Checks (printed by `python sources/id_papua.py`)

- The rake meets every regency's indigenous people (to 0.4 people) and every named group's
  province count exactly; every fixed group is the largest in its regency (Pegunungan Bintang:
  Ketengban had come out at 49% against Ngalum's 43% before the cap was added).
- In the six regencies whose largest group is Javanese (Sorong, Kota Sorong, Merauke, Nabire,
  Keerom, Kota Jayapura), no Papuan language exceeds the Javanese share (asserted).
- Against Joshua Project: the census groups are bigger than JP in the highlands (Western Dani
  407k vs 277k, Ekari 321k vs 177k, Nduga 99k vs 18k) and smaller for Sentani (29k vs 128k) and
  Maybrat (46k vs 99k). The census count wins wherever it exists.

### Tree and colours (`taxonomy/tree.d/id.txt`, the block between the GENERATED markers)

Generated by `id_papua.py`, as pg.txt is by `pg_build.py`. A language PNG also draws keeps
**pg's node**, so it is one colour across the border (14: Ngalum, Ninggerum, Yonggom, Mandobo
Atas, Suganga, Amanab, Manem, Waris, Sowanda, Wutung, Yei, Ngkontar Ngkolmpu, Pyu, Dera). The
rest go under pg's Trans-New Guinea group where one contains them (Asmat-Awyu-Ok), else under
Glottolog's group just below Nuclear TNG (new: Dani, Paniai Lakes, Mek); the other Papuan
families each under `papuan` (29: Lakes Plain, Greater Kwerba, Tor-Orya, Geelvink Bay, West,
East and South Bird's Head, Maybratic ...; pg's Border, Sko, Anim, Yam, Senagi, Pauwasi reused);
isolates on `isolate`; Austronesian under a new South Halmahera-West New Guinea group (Biak,
the Yapen and Raja Ampat languages, Wandamen, Waropen) or Oceanic's new Sarmi-Jayapura Bay
group; the Central Malayo-Polynesian languages of the Bomberai coast (Irarutu, Onin, Sekar,
Kowiai ...) flat under `austronesian` like the rest of Indonesia's. The old `papuan.birds_head`
areal group and the 22 cluster leaves are gone (no other fragment used them);
`papuan.tng.dani` is now the Dani group; `isolate.damal`, `papuan.anim.marind` and `yaqay` keep
their ids. `taxonomy/id2010.py` maps the "Papua: <name> [<glottocode>]" labels from the CSV the
script writes.

Colours: hand-picked for the highland neighbours (`HAND_COLOUR`): Ekari pale yellow-green, Moni
rust, Damal light amber, Western Dani green, Nggem light green, Walak olive, Mid Grand Valley
Dani deep teal, Upper light aqua, Lower pale yellow, Nduga coral red, Hupla blue-violet, the
Yali amber and golds, Ketengban orange, Kamoro violet; Biak magenta, Irarutu violet. Warm hues
are used (PNG's Torricelli and Sepik already are); Indonesian's cyan is avoided, since it is in
every highland town. Checked on a dot render across the border (135.5-142 E). The rest are
generated around their group.

### Placement (`countries/id.py`, `_papua_seed`)

Inside the two provinces every language is placed by regency: a Papuan language by its share
of each regency's people in the table above, and inside the regency 75% towards its point (30 km
kernel), 25% even; Javanese by Table 4's Javanese share (Biak Numfor: its non-Papuan share x the
other regencies' Javanese-to-non-Papuan ratio); every other migrant language (Bugis,
Makassarese, Butonese, Torajan ...) by the regency's non-Papuan share; Indonesian, Papuan Malay,
sign language and the unnamed remainder even. The kecamatan rake then meets Kontur's people and
the province counts as before. Scatter: 213,349 dots, 114 rings (small Papuan languages under
one dot, each ringed once at its largest concentration).

### Calls someone might reverse

- One retention for every Papuan language: the province's speakers are shared by indigenous
  people, so coastal and town groups (Biak, Sentani, Yapen), who have shifted to Malay more, are
  probably drawn too large against the highlands. No per-language or per-regency retention is
  published.
- Joshua Project (Ethnologue) estimates for the ~200 languages no census group names.
- Half of each city's Papuans from the province mix.
- Moribund languages dropped (as pg); Inanwatan's people go to their neighbours' languages.
- Aikwakai and Sarmi's "Biga" not fixed.

### Room for improvement

A BPS table of ethnicity by regency with more than the largest group (Ananta worked from the raw
2010 file); any measure of home-language retention by group or regency; language polygons
instead of one Glottolog point each.

## 11. Indonesian placed by regency, 2020 long form (2026-10-09, session 32a047f0)

Viewers saw a sharp Betawi / Indonesian edge at the DKI Jakarta border, and earlier "too
Sundanese" in places (followups.md). Cause: Indonesian had an even seed, so West Java's 19% and
Banten's 39% (2010, measured) were spread over every kecamatan, and Bekasi, Depok and Tangerang
drew their province's rural mix beside Jakarta's 89%; Betawi, seeded to its Glottolog point,
crowded the same fringe (Kota Bekasi 10.8%).

**Source.** Sensus Penduduk 2020 Long Form (fieldwork 2022), table 201: persons 5+ by regency,
"uses a regional language to talk daily in the family", Ya / Tidak (Tidak = Indonesian or foreign).
All 514 regencies, open JSON at `https://sensus.bps.go.id/topik/tabular/sp2022/201/<area>/3`.
`sources/id_lf2020.py` -> `data/normalized/id_lf2020_regency.csv` (raw in `data/raw/id/lf2020/`);
its docstring has the checks (regencies sum to provinces within 3, national 253,679,348). Tidak
nationally 25.2%. It names no regional language below the nation, and a family using both answers
Ya, so it is not comparable to 2010's Indonesian-first boxes; used for placement only.

**Seed** (`countries/id.py`, `_lf2020_tidak`, every province but Papua and Papua Barat): t = the
regency's Tidak share, clipped to 0.005-0.995; Indonesian and foreign (`other`) seeds x t, every
regional language's seed (Glottolog-pointed or even) x (1 - t), sign language unchanged. The rake
then shifts each province's log-odds to meet its 2010 counts. The layer's regencies are 2010's:
the 17 created since are folded into their parents (`LF2020_PARENT`), Kalimantan Utara's five
into the residual unit 65.

**Effect** (drawn share of Kontur population, before -> after): Kota Bekasi Indonesian 13 -> 63,
Sundanese 48 -> 8, Betawi 10.8 -> 3.0; Depok 15 -> 58; Tangerang Selatan 26 -> 63; Kota Tangerang
26 -> 60; Kab. Bekasi 14 -> 40; Garut and Tasikmalaya 15 -> 0, Sundanese 52 -> 66; Serang 28 -> 5;
Medan 57 -> 72; Makassar 25 -> 61; Surabaya 3 -> 13; Jakarta's five cities unchanged (74-78).
Across 452 regencies outside Papua, drawn Indonesian against 2020 Tidak: Pearson 0.72 -> 0.93.

Calls someone might reverse:
- Betawi moves with every regional language out of the cities, so West Java's 1.42M modelled
  Betawi now sit more in Kab. Bogor, Kab. Bekasi and Karawang (7.9%, was 4.5); Karawang is
  Sundanese, so some of that is likely too much. BPS's *Profil Suku* (2024, Fig. 3.19-3.20) says
  98.69% of ethnic Betawi use Indonesian or a foreign language in the family, so 2010's 2.24M
  Betawi speakers (L4.1) may themselves be the shaky part.
- 2020 shares placing 2010 counts: twelve years of urban growth are assumed not to move the
  pattern, only its level.
- Papua and Papua Barat left on their own regency model (Indonesian even there).

## 12. Java's big languages placed by homeland (2026-10-19, session eca0185b-id)

Anita, 2026-10-08 and 2026-10-19: "too Sundanese in places ... the totals are reasonable but
its probably just like allocation / shifting", and the Madurese of the Tapal Kuda missing.
Javanese, Sundanese and Madurese had even seeds inside their province (Madurese: Madura's four
regencies at 1, the rest of East Java at 0.02), Cirebonese and Betawi one Glottolog point each,
so Sundanese filled Indramayu (72%) and Serang (83%), and Jember, Bondowoso and Situbondo drew
East Java's off-Madura average (14% Madurese).

**Counts or placement?** Placement, for Madurese: L4.5 has 7,380,244 Madurese speakers in East
Java (21.5% of the province), about what Madura (3.6M people) and the Tapal Kuda can hold. No
count is changed. One count looks wrong and is left (see the last call below): Banten's Javanese.

**Source search** (on top of §2): sensus.bps.go.id's 2020 long form has, besides table 201,
tables 197-198 "Bahasa Pertama yang Dikuasai" (first language) and 202-204 (regional language
with neighbours) by regency, all in four boxes only (Indonesian / regional / foreign / sign), no
language named; tables 190-214 were probed. Web searches for a regency suku table of East Java
(2010 or 2000) found only national and province figures. Ananta's province appendix stays
paywalled and would not give regencies anyway.

**So the priors are homeland knowledge, recorded as such**, in `sources/id_homeland.py` (its
comments name each kecamatan): for each language a rough share of a regency's or kecamatan's
regional-language speakers, used as its seed in the kecamatan rake (times 1 - the 2020 Tidak
share, as every regional language). The rake still meets the 2010 province counts and Kontur's
people, so a prior says only where. Main lines: Sundanese 0.95 in Priangan and in Pandeglang,
Lebak and southern Serang; Cirebonese 0.9 in Cirebon and Indramayu, with Sundanese in south
Indramayu (Haurgeulis, Gantar, Kroya) and south-west Cirebon (Pasaleman, Sedong, Beber) and a
Cirebonese/Sundanese mix on Subang's coast; Betawi in Kota Bekasi, Depok, north-west Kab. Bekasi
and north-east Kab. Tangerang, eastern Kab. Bekasi Sundanese; Banten Javanese in northern Serang,
Kota Serang (Kasemen), Cilegon and north-west Tangerang; Sundanese in Brebes (Salem,
Bantarkawung, Banjarharjo ...) and Cilacap (Dayeuhluhur, Wanareja, Majenang ...), Javanese the
rest of Central Java (Suyanto, NUSA 2018); Madurese 0.97 on Madura, 0.92 Situbondo and
Bondowoso, 0.85 Probolinggo east of Wonomerto, Jember's north and east 0.85 and its south-west
0.12, north-east Lumajang 0.6, eastern Pasuruan 0.65, northern Banyuwangi (Wongsorejo 0.9), the
Osing kecamatan and Tengger (Tosari, Senduro, Sukapura) 0.02, Surabaya 0.2, Javanese the rest of
East Java. `python sources/id_homeland.py` asserts every code is a unit of the hex layer.

**Effect** (drawn share of Kontur population, before -> after):

| regency | before | after |
|---|---|---|
| Kota Bekasi | Indonesian 84, Sundanese 11, Betawi 4 | Indonesian 86, Betawi 6, Sundanese 5, Javanese 2.5 |
| Depok | Indonesian 77, Sundanese 18, Betawi 3 | Indonesian 77, Betawi 10, Sundanese 9.5, Javanese 3 |
| Kab. Bekasi | Indonesian 54, Sundanese 34, Betawi 10 | Indonesian 56, Sundanese 20, Betawi 19, Javanese 5 |
| Cirebon | Cirebonese 49, Sundanese 46 | Cirebonese 85, Sundanese 7 |
| Indramayu | Sundanese 72, Cirebonese 23 | Cirebonese 82, Sundanese 13 |
| Kuningan | Sundanese 56, Cirebonese 40 | Sundanese 91, Cirebonese 5 |
| Garut | Sundanese 88, Cirebonese 7 | Sundanese 97.5 |
| Serang | Sundanese 83, Javanese 4 | Sundanese 76, Javanese 12 |
| Cilegon | Sundanese 53, Indonesian 42 | Indonesian 58, Sundanese 28, Javanese 10 |
| Brebes | Javanese 97, Sundanese 2 | Javanese 78, Sundanese 21 |
| Sumenep | Madurese 90 | Madurese 94 |
| Bondowoso, Situbondo | Javanese 81, Madurese 14 | Madurese 82, Javanese 11.5 |
| Probolinggo | Javanese 81, Madurese 14 | Madurese 57, Javanese 37.5 |
| Jember | Javanese 82, Madurese 14 | Javanese 55, Madurese 42 |
| Banyuwangi | Javanese 83, Madurese 14 | Javanese 80, Madurese 17 |
| Malang | Javanese 83, Madurese 14.5 | Javanese 90, Madurese 8 |

Scatter: 213,349 dots, 114 rings, unchanged in number.

Calls someone might reverse:
- Priors from knowledge at 5-10 point resolution, no count below the province behind them.
  Kab. Bogor and Karawang are at regency grain (Betawi in northern Bogor and Javanese on
  Karawang's coast are not split out); Surabaya's Madurese north is not split either.
- Osing (Banyuwangi) stays Javanese, as BPS counts it; its kecamatan are kept for Javanese.
- **Banten's Javanese count, left alone at this stage; split on 2026-10-21, §15.** L4.5 has 259,981 Javanese in Banten against a Banten
  Javanese homeland (north Serang, Kota Serang, Cilegon, north-west Tangerang) of perhaps 1.5M
  aged 5+, and 4.97M Sundanese against a Sundanese homeland of about 4.2M. BPS's L4.1 has no
  Banten language (§1), so Banten Javanese was most likely coded Sunda. Moving ~1M from
  Sundanese to Javanese in Banten would be a count change on knowledge alone, which this
  session's brief did not allow; so Serang stays 76% Sundanese and the Javanese it does have sit
  in the north. This is the one thing left in Anita's 2026-10-08 complaint; it needs a ruling
  (split Banten's "Sunda" by place, as India's Pahari was) or a BPS table naming the code.

Outside Java: §13.

## 13. Malay, Banjar, Javanese and Bugis outside Java (2026-10-19, same session, stage 2)

The same method (`sources/id_homeland.py`, the `PRIORS.update` block), regency grain only, for
the languages that were still even-seeded in Sumatra and Kalimantan: Malay (BPS's Melayu) in
North Sumatra, Riau, Jambi, the Riau Islands, West and East Kalimantan; Banjar in Riau, Jambi
and the three other Kalimantan provinces; Javanese transmigrants and plantation workers in
North Sumatra, Riau, Jambi, South Sumatra, Lampung, the Riau Islands and Kalimantan; Sundanese
in Lampung; Madurese in West Kalimantan; Bugis in Jambi, South and East Kalimantan (its point in
Bone is over 150 km from them, so it was even). Sulawesi, Nusa Tenggara, Bengkulu, Bangka
Belitung and Aceh are left as they were. 37 (province, language) pairs now have homeland seeds,
11 of them Java's (§12, unchanged by this stage). Counts unchanged.

**Effect** (drawn %, before -> after):

| regency | before | after |
|---|---|---|
| Indragiri Hilir (Riau) | Malay 39, Javanese 23, Banjar 5 | Banjar 29, Malay 23, Javanese 18 |
| Siak (Riau) | Malay 23, Javanese 14 | Malay 19, Javanese 23 |
| Kuantan Singingi (Riau) | Malay 25, Javanese 14 | Malay 37, Javanese 6 |
| Tanjung Jabung Timur (Jambi) | Malay 58, Javanese 21, Banjar 2, Bugis 2 | Malay 32, Javanese 27, Bugis 13, Banjar 12 |
| Kota Jambi | Malay 59, Javanese 21 | Malay 64, Javanese 19 |
| OKU Timur (S. Sumatra) | Javanese 33 | Javanese 56 |
| Batu Bara (N. Sumatra) | Malay 6, Javanese 15 | Malay 23, Javanese 20 |
| Sambas (W. Kalimantan) | Malay 36, Dayak 47 | Malay 56, Dayak 34 |
| Landak (W. Kalimantan) | Malay 37, Dayak 41 | Malay 8, Dayak 64 |
| Kubu Raya (W. Kalimantan) | Madurese 7 | Madurese 17 |
| Kotawaringin Timur (C. Kal.) | Banjar 32 | Banjar 41 |
| Gunung Mas (C. Kal.) | Banjar 34, Dayak 11 | Banjar 10, Dayak 17, unnamed 38 |
| Banjarmasin | Banjar 81, Javanese 9 | Banjar 90, Javanese 2 |
| Hulu Sungai Selatan | Banjar 84, Javanese 10 | Banjar 93, Javanese 2 |
| Tanah Bumbu (S. Kal.) | Banjar 77, Javanese 9, Bugis 2 | Banjar 52, Javanese 25, Bugis 9 |
| Berau (E. Kal.) | Malay 7 | Malay 14 |

Calls someone might reverse:
- Kerinci kept Malay's default prior at this stage (a low one filled the valley with Javanese,
  28%); superseded by §14, which moves the valley's Malay onto Kerinci.
- Gunung Mas and Murung Raya now draw the unnamed remainder (38%, 23%) more than before: Central
  Kalimantan's 387,000 unnamed are most likely Dayak languages nobody names, and they fill what
  Banjar and Javanese leave upriver. Left even.
- North Sumatra's Javanese stays small beside Indonesian (Deli Serdang 12%) because the 2020
  Tidak share puts Indonesian there; the 2010 count is 885,000 Javanese speakers.
- Regency grain only; the priors are judgement, as in §12.

## 14. Kerinci split from Jambi's Malay (2026-10-21, session eca0185b-id)

Anita, 2026-10-21: "sure we can draw kerinci as own". Jambi's whole "other regional languages"
remainder in 2010 is 33,000 people, of whom the share-out gave Kerinci 26,655, so most of the
Kerinci valley's speakers were coded Melayu in L4.5; §13's map drew Kerinci regency 68% Malay
and 9% Kerinci.

**The count.** No census or survey counts Kerinci speakers by regency. Joshua Project's
Indonesian figure (312,000, after Ethnologue; `data/raw/pg/joshuaproject_pgic.csv`) is all ages
and about the two units' whole 2010 population (229,495 + 82,293 = 311,788), so it is not used
as the count, only as a ceiling. The count is from knowledge, recorded as such: Kerinci speakers
are about 85% of Kerinci regency's people and 80% of Sungai Penuh's (the rest are Minangkabau
traders, Javanese transmigrants and Jambi Malays). Applied to the people aged 5+ the map draws
there (164,772 and 60,241: Kontur scaled to Jambi's 2010 count, so the moved people fit the
valley's rows; the census's own 5+ figure would be about 279,000 and would spill Kerinci out of
it) gives 188,249, less the 20,273 Kerinci already drawn there: **168,000 people moved from
Jambi's measured Malay (1,670,695 -> 1,502,695) to Kerinci (26,655 -> 194,655)**, the moved row
`modelled`. In `countries/id.py` (`KERINCI_FROM_MALAY`, applied in `_counts()`, asserted against
Jambi's one measured Malay row); `parts` adjusted by the same amount.

**Placement.** `sources/id_homeland.py`: Kerinci's prior 0.95 in Kerinci regency and 0.9 in
Sungai Penuh, 0.002 elsewhere in Jambi (it had followed its Glottolog point, keri1250, confirmed
in `data/raw/glottolog/languages.csv`); Malay's there 0.03 and 0.08.

| unit | before | after |
|---|---|---|
| Kerinci regency | Malay 68, Minangkabau 14, Kerinci 9, Javanese 5 | Kerinci 83, Minangkabau 10, Javanese 3, Malay 1 |
| Sungai Penuh | Malay 68, Kerinci 10, Minangkabau 10, Javanese 5 | Kerinci 82, Minangkabau 7, Malay 4, Javanese 3 |
| Merangin, Bungo, Kota Jambi | Malay 64-68 | Malay unchanged; Bungo's Minangkabau 5 -> 6.5 |

Minangkabau in the valley falls a little as Kerinci fills it, and the rake puts it in Bungo
and Merangin instead. **Tree:** `austronesian.malayic.kerinci` already existed (the share-out's Kerinci).
It now has a hand-picked colour, 0.62 0.14 300, a mid purple, apart from Malay's light green,
Minangkabau's olive and Rejang's coral across the Bengkulu border. No regroup line: Kerinci is a
language of its own under Malayic, not a dialect of Minangkabau, though Glottolog puts both in
one Kerinci-Minangkabau group. Scatter: 213,349 dots, 114 rings.

Call someone might reverse: the 85% and 80% shares, which are judgement.

## 15. Banten Javanese split from Banten's Sundanese (2026-10-21, session eca0185b-id)

Anita, 2026-10-21: yes to splitting Banten's census Sundanese by place (§12's open call). L4.5
has 4,972,928 Sundanese and 259,981 Javanese in Banten; BPS's L4.1 has no Banten language, and
the Sundanese count is about 0.8M more than Banten's Sundanese homeland holds, so the Banten
Javanese of the north were mostly coded Sunda.

**Source search.** No speaker estimate for Banten Javanese (Jawa Serang) and no BPS suku table by
regency was found (a web search on 2026-10-21; Ethnologue and Joshua Project do not separate it
from Javanese; the papers found describe its area, northern Kab. Serang, Kota Serang and
Cilegon, without counts). So the count is from knowledge, recorded as such.

**The count.** In the 39 kecamatan of the Banten Javanese homeland as `sources/id_homeland.py`
draws it (north Serang's twelve at a Javanese prior of 0.85, Anyar, Mancak and Waringinkurung
0.45, all of Kota Serang 0.55 with Kasemen 0.9, Cilegon 0.8, north-west Tangerang's seven 0.6,
Pasarkemis, Sindang Jaya and Balaraja 0.3), each prior times the people the map drew there as
regional-language speakers (1,418,408 of 2,155,867; the rest Indonesian) gives 864,462 Javanese,
less the 235,879 already drawn there: **629,000 people moved from Banten's measured Sundanese
(4,972,928 -> 4,343,928) to Javanese (259,981 -> 888,981)**, the moved row `modelled`. The
Sundanese left (4.34M) is near the 4.2M §12 estimated Banten's Sundanese homeland holds.
In `countries/id.py` (`SPLITS`, with Kerinci; a check that Banten's measured Sundanese row is
L4.5's 4,972,928 before anything moves); `parts` has a row for the two splits (797,000).

**Placement:** the §12 priors, unchanged.

| regency | before | after |
|---|---|---|
| Serang | Sundanese 76, Javanese 12, Indonesian 8 | Sundanese 60, Javanese 32, Indonesian 6 |
| Kota Serang | Sundanese 55, Indonesian 35, Javanese 7 | Sundanese 38, Javanese 30, Indonesian 30 |
| Cilegon | Indonesian 58, Sundanese 28, Javanese 10 | Indonesian 44, Javanese 37, Sundanese 16 |
| Kab. Tangerang | Sundanese 53, Indonesian 33, Betawi 11, Javanese 2 | Sundanese 45, Indonesian 35, Betawi 11, Javanese 7 |
| Pandeglang | Sundanese 98 | Sundanese 96 |
| Lebak | Sundanese 98 | Sundanese 97 |

Of the 889,000 Javanese, 764,000 now sit in the homeland kecamatan; the rest are spread over
the Tangerang cities and the south at the low default prior, as migrants. Serang regency stays
Sundanese-majority overall because its southern half is Sundanese; its north is Javanese.
Indonesian falls in Cilegon (58 -> 44) because the rake had been filling Cilegon with Indonesian
for want of Javanese to put there. Cilegon's 2020 Tidak share (57%) is the check on that, and it
is now under it.

Calls someone might reverse: the homeland priors and shares; computing the move from the drawn
table, so it depends on where Indonesian was drawn at the time (recomputing after the move would
give about 200,000 more, because Indonesian moves out of the homeland; not iterated).

## Moved from countries/id.py text (2026-10-06 sweep)

From `note_public`: "In Papua and Papua Barat (the 2010 provinces, six since 2022), ..."

## Cross-border groups (2026-10-06, 5d7dac7e-xb)

Uab Meto is a group holding Timor-Leste's Baikenu (Oecusse). Groups only (taxonomy/regroup.txt): no label's node or count changed, each keeps its leaf and colour; dots rewritten in place. The full table of cases is in followups.md (2026-10-06, languages that stop at a border).
