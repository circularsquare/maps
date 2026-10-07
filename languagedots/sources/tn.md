# Tunisia (tn): record

Drawn 2026-10-05 (session edd42a8c-arab). Three Arab Barometer rounds with a first-language
question, pooled per governorate and applied to the RGPH 2024 governorate populations, plus
Tunisian Berber from Gabsi (2011): 11,972,169 people, 24 governorates, 5 nodes, every row
`modelled`. Placed on religiondots' Kontur 400 m hexes. 11,969 dots at 1:1,000.

Files: `sources/tn_surveys.py`, `sources/ab_firstlang.py` (shared Arab Barometer reader for
tn/jo/lb/ps/ye), `taxonomy/tn2016.py`, `taxonomy/tree.d/tn.txt`, `countries/tn.py`,
`data/normalized/tn.csv`.

```
python sources/tn_surveys.py
python taxonomy/build.py
python tools/check_country.py tn
python scatter.py --country tn
```

## 1. What exists

| source | language item | grain | used |
|---|---|---|---|
| RGPH 2024, 2014 | none (2014: literacy languages only) | - | population only |
| **Arab Barometer II 2011** (1,196) | `q10191` first language | 24 governorates | yes |
| **Arab Barometer III 2013** (1,199) | `q1019_1` first language | 24 | yes |
| **Arab Barometer IV 2016** (1,200) | `q1019a` first language | 24 | yes |
| Arab Barometer VII 2021-22 (2,400) | `Q1012B` ethnicity | 24 | no: 829 "Other", 778 "Don't know" |
| WVS 6 (2013), 7 (2019) | language at home | regions | not fetched (room for improvement) |
| Gabsi 2011, IJSL 211 | 45,000-50,000 Berber speakers, villages named | villages | **yes**, 47,500 |

Population: religiondots' `tn_lookup.csv` (RGPH 2024, 11,972,169).

## 2. The pool

3,595 answers: Arabic 3,580, French 6, English 3, Italian 1, Amazigh 1 (Tunis, 2013). Kept as
answered, one respondent one vote, per governorate, the Iraq rule. The single foreign answers
become 2,800-10,000 people each in their governorate (English in Jendouba, Kasserine, Medenine;
Italian in Beja): noise of a small sample, said in `note_public`.

## 3. Tunisian Berber

The surveys interviewed in Arabic and found one Berber speaker in 3,595. Estimates run from
10,000 (UNESCO-style "severely endangered" counts) to 60,000 (Belgacem's thesis, via L. Souag's
blog) and "1%" claims; Gabsi 2011 is the peer-reviewed one: 45,000-50,000, today in Guellala,
Sedouikech and Ouirsighen (Djerba, Medenine), Chenini and Douiret (Tataouine), with Matmata's
villages (Gabes) in older descriptions. Drawn 47,500 over Gabes, Medenine and Tataouine **by
population** (17,600 / 23,000 / 7,000), since nothing splits it; that is placement within the
south-east, not a count per governorate. Taken out of the governorates' surveyed Arabic. Plus
the Tunis survey answer, scaled (3,055): national Berber 50,555 (0.42%).

## 4. Mapping and tree

| answer | node | people |
|---|---|---:|
| Arabic | `afroasiatic.tunisian_arabic` (new, tuni1259) | 11,888,511 |
| Berber | `afroasiatic.berber.tunisian_berber` (new, tuni1262) | 50,555 |
| French | `indoeuropean.romance.french` | 20,739 |
| English | `indoeuropean.germanic.english` | 9,583 |
| Italian | `indoeuropean.romance.italian` | 2,781 |

Tunisian Arabic hand-set 0.84 0.08 158, between Algerian Arabic and the Levantine mints.

## 5. Calls someone might reverse

- Gabsi's figure shared by population over three governorates (it spreads Berber dots over
  Gabes city and Zarzis rather than the villages).
- Foreign first-language answers kept as answered, not folded into Arabic.
- Foreigners (Libyans, Algerians, sub-Saharan migrants, ~60,000) not modelled: the survey shares
  apply to the whole population.

## 6. Room for improvement

- A delegation-level placement for the Berber villages (Djerba's Guellala and Sedouikech
  delegations, Tataouine Sud, Matmata) would put the dots where the speakers are.
- WVS 6 and 7 Tunisia samples (online tool, `sources/ir_wvs.py`) would add ~2,400 answers.

## Terms

Arab Barometer: free download, citation requested. RGPH via religiondots (INS). Glottolog CC BY.
Kontur CC BY 4.0.
