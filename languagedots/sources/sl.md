# Sierra Leone: 2015 PHC, main language, by district

Drawn 2026-10-05 (session edd42a8c-sl). 6,954,701 household members with a stated main language,
14 districts (the 2015 set), 19 nodes, every row `derived`. 6,946 dots at 1:1000, 2 rings.

```
python sources/sl_place.py            # religiondots' hexes + each hex's 2017 district
python sources/sl_census.py --fetch   # the analytical report (digest pinned) + CLEAR's CSVs
python taxonomy/build.py
python tools/check_country.py sl
python scatter.py --country sl
```

`sl_place.py` runs first: `sl_census.py` reads its layer for the 2015/2017 district mix.

## What Statistics Sierra Leone publishes on language

The 2015 census asked every household member's main language (P10) and a secondary language
(P11). The **National Analytical Report** (584 pp, `2015_census_national_analytical_report.pdf`,
live on statistics.sl, Wayback 2020-11-14) prints:

- **Table 3.22** (PDF p134): main language **nationally, in persons**: 15 local languages,
  "Foreign language" 11,231 and "Other" 5,499, total 6,954,702. That is the household population
  (7,076,119, Table 4.1) less 121,417 with no answer: Table 3.23's national Mende 29.2% of
  7,076,119 is Table 3.22's 2,065,349.
- **Table 3.23** (p135): for the nation, 4 regions and **14 districts**, the three most common
  main languages with their % of the household population, and "Others". 42 district cells.
- Table 3.24 (p136): number of languages spoken by district. Not used.

Searched and not there: none of the fifteen 2015 thematic reports in the Wayback CDX
(`statistics.sl/.../Census/2015/sl_2015_phc_thematic_report_on_*`) is about language; the Census
Atlas (190 pp) maps ethnicity by chiefdom but not language; the analytical report has no
language annex. No full district x language table is published. The microdata is IPUMS only
(account blocked).

## The construction

**CLEAR Global** (HDX `sierra-leone-languages`, CC BY-SA 4.0): "main language spoken"
proportions for the 16 districts of 2017 (COD-AB pcodes), from the IPUMS 10% sample of this
census (extract ipumsi_00291). 18 named languages (the 15 local ones plus English, French and
Standard Arabic) and `Unknown`.

1. **Each 2015 district's starting shares** are its 2017 pieces' CLEAR shares weighted by Kontur
   population: Bombali = Bombali 0.68 + Karene 0.32, Port Loko = Port Loko 0.85 + Karene 0.15,
   Koinadugu = Koinadugu 0.52 + Falaba 0.48. The other eleven are one piece each.
2. **Table 3.23's 42 printed district cells are fixed** (share x district household population).
3. **The free cells are fitted** (iterative proportional fitting) to what is left of each
   district's household population (religiondots' `sl_lookup.csv`: Table 2.2 x 7,076,119 /
   7,092,113), of each Table 3.22 national total, and of each of the 8 Table 3.23 region cells
   that still has a free district under it (Krio in Southern pins Moyamba's Krio, for one).
4. "Foreign language" is split English / French / Arabic by CLEAR's national ratio; "Other" and
   the 121,417 with no answer share CLEAR's `Unknown` in their national ratio, and the no-answer
   column is dropped.

So every national total is the census's, every district total is the census's, and the three
largest languages of every district are the census's to the printed decimal. Only the rest of
each district (5-26% of it; Kono and Bombali are the most mixed) comes from the sample. Every row
is `derived` because the district split of all but the printed cells is fitted.

CLEAR also has a row SL05XXX, "northwestern: level 2 unknown" (94.5% Temne), with no population;
it is left out. It is probably Port Loko's: CLEAR alone gives Port Loko 74.7% Temne against the
printed 81.7. Fixing the printed cells corrects that.

## Checks (`sources/sl_census.py`, `sources/sl_place.py`, all pass)

| check | result |
|---|---|
| file | 584 pages, 12,614,169 bytes, sha256 pinned, %%EOF |
| Table 3.22 parse | 17 rows off p134 = the transcription; sum 6,954,702 = its printed total |
| Table 3.23 parse | 19 rows off p135 = the transcription; each district's three + Others = 100 within 0.15 |
| districts | household population sums to 7,076,119; = religiondots' sl_hexes units both ways |
| CLEAR | 16 pcodes = COD-AB sle_admin2 both ways; shares sum to 1 |
| CLEAR alone vs Table 3.22 | worst Temne 25.79 vs 26.16% |
| CLEAR alone vs Table 3.23's 42 district cells | mean 0.59 points; worst Temne in Port Loko 74.7 vs 81.7, Temne in Bombali 45.9 vs 42.8, Susu in Port Loko 6.0 vs 3.7, Limba in Koinadugu 12.5 vs 14.2; the other ten districts within 1.7 |
| fit | district totals and Table 3.22 totals met within 1 person |
| rank | the printed three stay each district's top three (closest: Pujehun's Fula 0.1 points under its third, Temne 0.6) |
| region cells | all 12 within 0.06 points of Table 3.23 (8 were fitted, 4 are pure checks) |
| placement | 43,526 hexes, each 2015 district holds exactly its 2017 districts; 497 offshore or line centroids put on the nearest allowed one; population = religiondots' |

The CLEAR-alone misses in Bombali and Port Loko come from step 1: Karene's shares are used for
both its halves, but its Port Loko chiefdoms (Buya Romende, Dibia, Sanda Magbolontor) are likely
more Temne and its Bombali ones (Sanda Loko, Sella Limba, Tambakha..., Loko and Limba by their
names) more Limba and Loko; the fit's direction agrees. The fixed
cells take most of it out, but the smaller languages in those two districts carry some of it:
Susu in Bombali comes out 5.8% (Karene's Susu, which probably sits on the Port Loko side).

## Placement inside districts

Within a 2015 district, a language's dots go to each hex by Kontur population times CLEAR's share
of that language in the hex's 2017 district (AGENT_BRIEF §4.4; it moves people only inside the
unit counted). This matters only in Bombali/Karene and Koinadugu/Falaba: Falaba's Kuranko,
Yalunka and Fula go north-east, Koinadugu's Limba south-west. "Other" goes on population.

## Mapping calls (`taxonomy/sl2015.py`, `taxonomy/tree.d/sl.txt`)

- **Reused:** Temne and Krio (defined bare by au/pl/uk; coloured here), Fula, Susu, Kissi,
  Kuranko (gn), Yalunka, Mandinka (sn), English, French, Arabic.
- **New:** Mende, Loko, Kono (Sierra Leone, kono1268, a Vai-Kono language; the plain `kono` node,
  Guinea's unrelated Kono being `kono_guinea`), Vai (Mande); Limba, Sherbro, Krim (Atlantic, flat
  as Temne is). Limba is a two-language family in Glottolog directly under Atlantic-Congo; one
  leaf, conventionally Atlantic. Krim is a dialect of Bom-Kim in Glottolog; named, so a node.
- **Madingo on `mandinka`**, following CLEAR's coding (mand1436). Sierra Leone's Mandingo are
  partly of Guinean Maninka origin; the census says only "Madingo".
- **Foreign language split** into English (10.1k), French and Arabic (under 1k each) by the same
  sample. The census form coded them apart; the report prints one row.
- **"Other" on `other`.**

**Colours.** Mende blue, Temne orange, Krio pale mint, Limba purple, Kono dark teal, Loko pale
pink, Sherbro green. Neighbours that matter: Mende/Sherbro/Temne in Moyamba and Bonthe,
Mende/Kissi/Kono in the east, Temne/Limba/Loko/Susu in Bombali, Kuranko/Limba/Fula/Yalunka in
Koinadugu. Kuranko (pale lime) and Krio (pale mint) are both light and about 60 degrees apart;
they share Koinadugu and Kono at 5-8% each. Not tuned further.

## Not escalated

Language here follows ethnicity closely. Counts are Statistics Sierra Leone's own published tables at district level; the rest comes from
CLEAR's public humanitarian dataset. Nothing finer than a district is drawn.

## Terms

The analytical report is a public download with no licence text. CLEAR Global's dataset is CC
BY-SA 4.0 (attribution: CLEAR Global, from IPUMS International and Statistics Sierra Leone);
used to split districts and place dots, its file is not republished. COD-AB Sierra Leone CC
BY-IGO; Kontur CC BY 4.0; Glottolog CC BY.

## Would change it

- A full district (or chiefdom) language table from Statistics Sierra Leone, or IPUMS access:
  either replaces the fitted cells, and IPUMS's chiefdom geography would fix Karene's split.
- The 2021 mid-term census, if it asked language and publishes it.
