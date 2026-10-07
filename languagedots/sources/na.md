# Namibia: the 2011 census public sample, main household language, at 107 constituencies split town and countryside

**Drawn 2026-10-04** (d9e44929-na). 2,064,890 people drawn of 2,115,377 in the weighted sample
(census count 2,113,077), 11 nodes, 2,060 dots.

```
python sources/na_pums.py --fetch   # PUMS + DDI + six regional profiles -> data/normalized/na.csv, all checks
python sources/na_geo.py            # religiondots' hexes re-keyed to constituencies, cut town/countryside
python taxonomy/build.py
python tools/check_country.py na
python scatter.py --country na
```

Files: `sources/na_pums.py`, `sources/na_geo.py`, `taxonomy/na2011.py`, `taxonomy/tree.d/na.txt`,
`countries/na.py`; raw files in `data/raw/na/` (the PUMS text is 47 MB).

## 1. Why 2011, and why the sample

- **2023 census: no language table.** The *2023 Population and Housing Census Main Report* (NSA,
  28 October 2024, `nsa.org.na/wp-content/uploads/2024/10/2023-Population-and-Housing-Census-Main-Report-28-Oct-2024.pdf`,
  122 pp.) has no language table at all; it has ethnicity (Table 3.6, top 20 groups, national),
  which is not a language question. The 2023 PUMS (NADA catalog 17) has no language variable
  (coverage sweep, 2026-10-03). `census.nsa.org.na`'s constituency pages carry population
  topics only (checked: Walvis Bay Urban). The figures some web pages give as "2023" household
  language shares (Oshiwambo 49.7%, Lozi 4.9%...) could not be traced to any NSA release.
- **2011 published tables** give language only by region and urban/rural: each regional
  profile's Table 6.7 (households) and its annexure tables (households and population by main
  language, region / urban / rural; numbered 6.12-6.14 or 6.16-6.18 depending on the profile).
  Constituency tables in the profiles do not include language.
- **2011 PUMS**: NSA's National Data Archive, `microdata.nsanamibia.com/index.php/catalog/9`,
  `NAM_NSA_PHC_2011_V01_PUMS` v1.0. Access policy "Public use" (statistical and research purposes,
  no re-identification, no redistribution of the dataset itself); the data and documentation are
  listed openly under "related materials" (download 172 the data, 171 the documentation PDF; no
  login). This map publishes aggregates only. 20% of households, stratified by constituency x
  urban/rural, weight 5; the seven strata under 250 households and every household of 50+ people
  taken whole at weight 1. It has constituency (107) and urban/rural. Chosen: constituency is 8x
  finer than region, and the sample reproduces the full-count regional tables (check 6).

The question (H13, housing record): *"What is the MAIN language spoken in this household?"*, one
answer per household; every member is drawn at it. Not asked in institutions and special
populations (HH_TYPE 2xx/3xx: hostels, barracks, prisons, hospitals, hotels, travellers): their
H13 is blank, 49,952 people weighted. "Don't know": 535.

## 2. Reading the file

One fixed-width text file, three record types (1 person, 92 chars; 2 death, 22; 3 housing, 54).
Household key `line[1:16]`: region 2, constituency 2, "0", urban/rural 1 (1 urban, 2 rural),
household type 3, serial 6. Weight = the last character of every record. The DDI's start positions
are per original file and do not hold for this combined text (they are off by 8 at H13): H13 is
`line[42:44]` of the housing record, proven by all 15 of its codes' counts equalling the DDI's
`catStat` exactly. Region codes 1-13 are 2011's 13 regions (Caprivi, Erongo, ..., Otjozondjupa).

## 3. Constituency codes -> COD-AB pcodes

The file's constituency codes (1-12 within region) carry no labels ("consult the code book",
which is not online). They are the NSA codes COD-AB Namibia's admin-2 pcodes carry, `NA<rr><cc>`
(COD-AB's admin 2 is the 2008 delimitation's 107 constituencies, filed under the 14 regions of
2013), except that Kavango's 01 Kahenge, 02 Kapako and 04 Mpungu are NA1401/02/04 in Kavango West.
Witnesses neither key decides:

- **the seven whole strata.** The documentation names them (p.6): Arandis rural, Rehoboth Urban
  East rural, Walvis Bay Rural rural, Mpungu urban, Etayi urban, Kalahari urban, Ondobe urban.
  Through the mapping, the strata whose every household weighs 1 are exactly those seven.
- **Urban-named constituencies are urban**: Katima Mulilo Urban, Walvis Bay Urban, Rehoboth Urban
  West, Keetmanshoop Urban, Rundu Urban 100%; Rehoboth Urban East 98%; Mariental Urban 81%.
  (Rural-named, not asserted: Rundu Rural West 75% urban and Rundu Rural East 63%, because Rundu
  town spreads over all three Rundu constituencies; Walvis Bay Rural 98%, the doc lists its rural
  part as a tiny stratum.)
- 107 codes, the same count per region as COD-AB (Kavango 9 = East 6 + West 3).

## 4. Checks (sources/na_pums.py, all pass)

1. records 94,774 housing / 441,929 person / 4,401 death, the DDI's case counts
2. H13 counts per code = DDI catStat, all 15
3. H13 blank exactly for the non-conventional households; every person joins a household
4. weighted people 2,115,377 against the census's 2,113,077 (+0.11%)
5. the code witnesses (section 3)
6. **against the full count.** Six regional profiles are on the web (mirrored at `cms.my.na`;
   Erongo, Karas, Kavango, Ohangwena, Omaheke, Otjozondjupa; the other seven were not found).
   Their annexure tables of population by language x urban/rural, 80 cells of 100+ households:
   largest |z| 2.6 against the sampling error of a 20% household sample (rel. SE ~ 2/sqrt(h)),
   7 cells past 2 where ~4 are expected by chance; urban and rural totals within 3.0% (Omaheke
   urban +3.0%, the largest). Erongo's profile prints "Erongo languages" where every other
   profile has "Caprivi languages" (same row position, all three tables); read as Caprivi.

Sampling: a constituency's small languages are estimates from about a fifth of its households
(a language held by 10 sampled households is 50 households +-30%). Every row is drawn as
measured; the note says it is the sample.

## 5. Mapping (taxonomy/na2011.py, tree.d/na.txt)

| census answer | node | people |
|---|---|---|
| Oshiwambo languages | `nigercongo.bantu.oshiwambo` (leaf) | 1,019,047 |
| Nama/Damara languages | `khoisan.khoe.khoekhoe` | 233,594 |
| Kavango languages | `nigercongo.bantu` (washed) | 214,261 |
| Herero languages | `nigercongo.bantu.otjiherero` (leaf) | 185,239 |
| Afrikaans | `...continental.afrikaans` | 181,566 |
| Caprivi languages | `nigercongo.bantu` (washed) | 92,158 |
| English | `indoeuropean.germanic.english` | 51,407 |
| Other African languages | `africa_other` | 35,860 |
| San languages | `khoisan` (washed) | 18,784 |
| Other European languages | `other` | 14,883 |
| German | `...continental.german` | 11,069 |
| Setswana | `...sotho_tswana.setswana` | 5,437 |
| Asian languages | `other` | 1,585 |
| Don't know / not asked | not drawn (gap) | 535 / 49,952 |

**The calls.**
- **Oshiwambo and Otjiherero drawn as languages**, though the census label is plural. Oshiwambo's
  varieties (Kwanyama, Ndonga, Kwambi, Ngandjera...) are mutually intelligible and Namibia names
  and teaches Oshiwambo as one language; Herero's varieties (Mbanderu, Himba) likewise. Washed
  out as groups they would make half the country read as a remainder, the reverse of the
  "Chinese" case Anita ruled on (build.py `UNWASHED`). Zemba (Glottolog Herero R.30, a few
  thousand in Kunene) goes in with Otjiherero; the census does not separate it.
- **Kavango and Caprivi languages on Bantu, washed.** They are groups of different, not mutually
  intelligible languages (Rukwangali/Rumanyo are Glottolog's Kwangali-Diriku in Njila,
  Thimbukushu Greater Luyana; Silozi Sotho-Tswana, Subiya/Fwe/Totela Botatwe, Yeyi), so the
  narrowest node holding each is Bantu (spec §3.2). No source splits them by constituency;
  an areal "Kavango languages" node was considered and not made (it would duplicate zm.txt's
  Mbukushu and is not a classification).
- **Khoisan as one new root** (families as readers know them; Glottolog's Khoe-Kwadi khoe1240,
  Kx'a kxaa1236, Tuu tuuu1241). Khoekhoegowab under `khoisan.khoe`; the San languages span all
  three, so they sit on the root.
- Other European and Asian languages on `other` (they cross families); Other African on
  `africa_other` (Anita's indigenous-remainder rule). Kavango's profile puts 12.6% of its
  households in Other African; no table says which languages.

Not done, possible later: splitting Kavango/Caprivi languages with a survey's regional mix
(Afrobarometer asks home language finely) would be a proxy and is Anita's to allow; not asked.

## 6. Geography (sources/na_geo.py)

religiondots' `data/geo/na/na_hexes.gpkg` (Kontur 2023-11, 69,799 populated hexes keyed to 14
regions, border hexes already dropped or snapped there) re-keyed to COD-AB admin 2 by centroid;
366 hexes (19,932 people) with a centroid in no constituency go to the nearest constituency of
their own region (farthest 3.5 km). Every hex's constituency is in religiondots' region for it.
Kontur 2023 against the 2011 counts per constituency: log correlation within regions 0.95, 1,000
within-region shuffles at most 0.44; ratio over the national 1.24: p10 0.83, median 0.94, p90
1.15. Windhoek's Katutura constituencies move most (Katutura Central 0.45, Katutura East 1.60,
Samora Machel 1.53, Moses Garoeb 1.47: growth and informal settlement since 2011; only placement
inside each constituency uses Kontur). Town/countryside: the densest hexes of each constituency
labelled town up to the sample's urban share (zm_geo.py's method); 43 constituencies split, 48 all
rural, 16 all urban; largest gap between the town share of Kontur people and the census's urban
share 0.066 (Mpungu). 150 units. No Kontur block at the cap (religiondots `kontur_cap.py na`).

## 7. Colours

Oshiwambo orange #f97c3d, Otjiherero magenta #cd5cac, Khoekhoegowab light yellow #e8d34f,
Khoisan root green (San washed #9dc2ad), Afrikaans blue (us.txt's). **Known clash, not fixed:**
Bantu's washed colour (#c3a48c, Kavango and Caprivi languages) is nearly `africa_other`
(#c4a582), and in Kavango the two sit together (79% and 13% of households). Both nodes belong to
other fragments and build.py; left for the colour pass.

## Cross-border groups (2026-10-06, 5d7dac7e-xb)

Oshiwambo is now a group; the census's 'Oshiwambo' draws on its own leaf 'Oshiwambo (variety not given)', beside Angola's Kwanyama and Ndonga. Groups only (taxonomy/regroup.txt): no label's node or count changed, each keeps its leaf and colour; dots rewritten in place. The full table of cases is in followups.md (2026-10-06, languages that stop at a border).
