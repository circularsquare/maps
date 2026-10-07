# Finland: population register, language (mother tongue as registered), 31 Dec 2025

Built 2026-10-04 (session d9e44929-fi). Rebuild:

```
python sources/fi_register.py [--fetch]   -> data/normalized/fi.csv (also fetches the 2026 boundaries)
python taxonomy/build.py
python sources/fi_geo.py [--fetch]        -> data/geo/fi/fi_grid1km.gpkg
python tools/check_country.py fi
python scatter.py --country fi
```

Drawn: 5,650,672 people on 308 municipalities (Aland's 16 included), 166 register codes of which
162 hold anyone, on 161 nodes; 5,610 dots, 50 rings. Not drawn: 2,209 people whose language the
register records as unknown (0.04%, the gap).

## 1. The table

Statistics Finland, StatFin database `vaerak` (population structure), PxWeb API
`https://pxdata.stat.fi/PxWeb/api/v1/en/StatFin/vaerak/`, CC BY 4.0, no login, fetched as
json-stat2 with a browser User-Agent:

- **11rm** "Language according to sex by municipality, 1990-2025": 308 municipalities of the
  1 January 2026 division x 169 language codes, sex = total, year = 2025 (population on 31 Dec).
  The drawn table.
- **11rl** "Language according to age and sex by region, 1990-2025": the 19 maakunnat (regions),
  the same 169 codes, age and sex total. The second table.

**The question.** There is none: Finland has no census questionnaire. The Population Information
System holds one language per person (aidinkieli, mother tongue), given when a birth is registered
(by the parents) or when an immigrant is first registered, and seldom changed later. One code
only, so a bilingual Finnish-Swedish child is registered under one of the two. The queue's
coverage note says the same.

**The codes** are ISO 639-1 plus Statistics Finland's own: `98` other language, `X` unknown and
the subtotals `SSS` (total), `01` (national languages: Finnish, Swedish, Sami) and `02` (foreign
languages). The subtotals are checked and dropped. fi.csv labels each row `<code> <English text>`
("sv Swedish") and the mapping keys on the code.

**Suppression.** 11rm marks every cell under 10 people `...` (its own note: "for reasons of
privacy protection"), zero included: 46,961 of 52,221 cells. 11rl is not suppressed at all, and
neither is any municipality's total or national-language subtotal.

**Vintage.** 2025 is the latest year in both tables. The coverage sweep's "2024" was the latest
when it ran.

## 2. Suppressed cells (`fill_suppressed()`)

Within each maakunta, and separately for the national-language block (fi, sv, se) and the
foreign block (everything else, `98` and `X` included):

- per language, the suppressed cells must add up to 11rl's region figure less the municipalities'
  printed cells;
- per municipality, to its printed block subtotal less its printed cells (the foreign subtotal
  `02` is derived as total less `01` where it is itself suppressed);
- each cell is 0 to 9.

Iterative proportional fitting from an even seed meets both margins, clipping at 9 after each
pass. Every margin was checked to lie inside 0..9 x its number of suppressed cells before fitting
(it would stop otherwise; none failed, which is itself a check that the two tables and the
maakunta join agree). Worst margin error after fitting 0.0013 people. 122 cells end at the cap.

The estimates hold **21,720 people, 0.38% of Finland**, tier `derived`, in 28,603 non-zero cells:
the median municipality has 1.1% of its people estimated, the largest share is Kumlinge (Aland,
8.9%). By language the largest estimated totals are German 668, Tagalog 599, English 596, Thai 554
and other 543: small foreign communities spread thin over small municipalities. The national
languages are almost all printed; every Sami figure in the eight municipalities with the most
Sami speakers is printed.

## 3. Checks (asserted in `fi_register.py` and `fi_geo.py`; all pass)

| check | result |
|---|---|
| municipalities in 11rm | 308, all `KUnnn` |
| fully printed subtotals equal their codes summed | 11rm 20, 11rl 66 (19 maakunnat, 2 halves, the country, x 3), no difference |
| whole-country row of 11rm against 11rl | all 169 codes 11rm prints there agree |
| 2026 municipality layer (geo.stat.fi `kunta1000k_2026`) against 11rm | the same 308 codes, both ways |
| municipality -> maakunta (point inside each polygon on `maakunta1000k_2026`) | 19 maakunnat, each with municipalities |
| maakunta x code, 11rl against its municipalities | 86 cells with nothing suppressed match exactly; in the other 3,125 the printed cells stay under the region's figure and the remainder fits 0..9 per suppressed cell |
| filled cells meet both margins | worst error 0.0013 people |
| drawn = total less unknown | 5,650,672 = 5,652,881 - 2,209 |
| grid municipality codes against the table | 308 = 308, both ways |
| grid residents per municipality against the table | national 0.986; p10 0.970, median 0.987, p90 0.995; extremes Kauniainen 0.749, Jokioinen 1.094 |

The maakunta check is the strong one: 11rl is a separate table at another level, and the fill
could only be built because every one of 19 x 166 region remainders and 308 x 2 municipality
remainders fell in the range suppression allows. A wrong municipality-to-region join would have
broken it.

National: Finnish 4,719,802 (83.5%), Swedish 284,611 (5.0%), Russian 102,618, Estonian 48,495,
Ukrainian 46,696, Arabic 44,956, English 39,100, Somali 28,167, Persian 24,047, Chinese 20,320,
Albanian 19,600, Kurdish 18,460, Vietnamese 17,682, Bengali 14,013, Turkish 13,790, Tagalog 13,380,
other 13,258, Thai 12,712, Nepali 12,204, Spanish 12,088. Sami 2,076, mostly Inari 513, Utsjoki
458, Rovaniemi 221, Enontekio 181, Oulu 159, Sodankyla 126, Helsinki 75. 31 municipalities have a
Swedish majority.

## 4. Labels and nodes (`taxonomy/fi2025.py`, `tree.d/fi.txt`)

Every code with anyone has its own node; all but three existed.

- **Sami** (`se`, 2,076): `uralic.saami`, the tree's single Sami leaf (pl.txt). One register code
  covers North, Inari and Skolt Sami, though ISO's `se` is North Sami alone; nothing finer is
  published. Registered Sami very likely understates Sami speakers (a family registers one
  language, and Finnish is the usual second one); nothing here measures by how much.
- **Chinese** (`zh`, 20,320): `sinotibetan.sinitic`, drawn unwashed as for the US and Canada.
- **Twi** (`tw`, 942) and **Akan** (`ak`, 705): the register prints both. Glottolog has Twi as a
  dialect of Akan (twii1234 under akan1250). New leaf `nigercongo.kwa.twi` as a sibling of us.txt's
  Akan, not a child, so Akan does not become a group. Its legend then reads "Akan (Twi)" beside
  "Twi"; us.txt's label is not mine to change.
- **Ewe** (`ee`, 170): new leaf `nigercongo.kwa.ewe` beside us.txt's "Gbe (Ewe, Fon)" (Glottolog
  ewee1241, in Gbe), same reasoning.
- **Ido** (`io`, 25): new leaf `other.ido` beside `other.esperanto` (Glottolog: artificial).
- **Moldavian** (`mo`, 365): cz.txt's Moldovan, apart from Romanian (7,310) as printed.
  **Serbo-Croatian** (`sh`, 1,145) apart from Bosnian, Croatian and Serbian, as printed.
- **Chichewa** (`ny`): pl.txt's Chewa. **Guarani** (`gn`, 2): Paraguayan Guarani, because ar.txt's
  `guarani` is a group and a named label may not sit on one. **Romansh** (`rm`, 2): ch.txt's
  Rhaeto-Romance, which is Switzerland's Romansh.
- **Bihari languages** (`bh`, 2) and **Cree** (`cr`, 5) name groups and sit on the group nodes.
- **Other language** (`98`, 13,258): `other`. The list is ISO 639-1's, so a language with no
  two-letter code (Karelian has none) is here or under a listed language; not split anywhere.
- **Unknown** (`X`): not drawn; the gap.

fi.txt repeats, with unchanged labels, every node fi2025 uses that tree.txt lacks, so the
fragment stands alone when the build tail reads only drawn countries' fragments.

## 5. Placement (`fi_geo.py`)

Not Kontur. Statistics Finland's own 1 km population grid, `vaestoruutu:vaki2025_1km` on
geo.stat.fi (CC BY 4.0): 96,904 inhabited squares, each with its resident count (small squares
included; only their age and sex are blanked) and the municipality the office files it under,
all 308 present. It is the same register as the language table, so a square's weight is where
registered residents live; Kontur models people from buildings, and Finland's half a million
summer cottages are the failure religiondots met in Iceland (playbooks/geography.md). The office
gives each square's municipality, so there is no centroid join. Squares are not clipped to
municipal lines (a border square's dots can fall up to 1 km over the line); scatter.py clipped the
sea from 4,358 squares.

The grid holds 5,573,552 residents against the table's 5,652,881: it counts only residents the
register can place on a coordinate, and its layer name does not say whether it is the end of 2024
or of 2025. It is a weight inside each municipality only. Kauniainen (0.749 of its table total)
is an enclave in Espoo whose border squares are filed under Espoo, so its dots sit on its inner
squares; harmless at this scale.

No Kontur layer, so no cap blocks and nothing in `kontur_cap.csv`.

## 6. Colour (hand-picked in `tree.d/fi.txt`)

Generated, Finnish came out light green beside Russian's green and Sami's sea-green, and Estonian
sky blue beside Swedish's light cyan. Hand-picked, all within the Uralic and North Germanic parts of
the wheel: Finnish 0.80 0.10 190 (light teal, #67d2cc), Swedish 0.76 0.13 265 (periwinkle, #88afff;
us.txt, bo.txt and pr.txt use it bare, where it sits beside English's near-white and Norwegian's
dark teal), Sami 0.62 0.12 165 (dark sea-green, #249c74), Estonian 0.66 0.11 220 (steel blue,
#29a1c1). Arabic (#96d5b2, mint) is the nearest big language to Finnish left as generated.

## 7. Not done

- 1990-2024 are in the same tables and not read.
- `11rz` (marital status by age, language and sex by area) and `11s2` (Finnish and Swedish speakers
  by area) could be a third check on the national languages; not read.
- Postal-code (Paavo) data was not looked at; the register table is already at municipality.
