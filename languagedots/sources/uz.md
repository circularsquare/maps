# Uzbekistan (`uz`): 2026 census native language, 14 regions

Drawn 2026-10-05 by session `d9e44929-uz`. No asks filed.

Files: `sources/uz_census.py` (normaliser), `taxonomy/uz2026.py`, `taxonomy/tree.d/uz.txt`,
`countries/uz.py`. Outputs `data/raw/uz/uz_census2026_results_uz.pdf` (the Uzbek edition),
`data/normalized/uz.csv` (112 rows), `data/processed/dots_uz.geojson` (39,044 dots), no rings.
The English edition and the placement layer are religiondots', read in place.

## 1. The table

*Preliminary Results of the Population and Agriculture Census of the Republic of Uzbekistan,
2026*, National Statistics Committee, Tashkent 2026, 94 pp. Census moment 15 January 2026,
enumeration 15 January to 28 February, the first census since 1989. Table "Distribution of the
population by main language of communication (mother tongue), by region":

| edition | URL | printed page | PDF page |
|---|---|---|---|
| English | `https://stat.uz/img/news/english_natija_merged-2_p42445.pdf` (cached by religiondots as `data/raw/uz/uz_census2026_results_en.pdf`) | 71 | 73 |
| Uzbek | `https://aholi.stat.uz/images/kitob-uzb_p83506.pdf` (linked from aholi.stat.uz, *Nashrlar*), 4,708,380 bytes, `%%EOF` present | 51 | 72 |

Both certificates fail verification; the fetch skips it, as religiondots' does.

**The question.** Item 13 of the individual form, "Ваш родной язык?", one answer from a list,
with "другой" and free text. The enumerators' instruction (religiondots
`data/raw/uz/uz_census2026_instruction_ru.txt`, line 1374) defines it as the language learned in
childhood and mainly used in daily communication, "the first language learned in childhood";
for children under 6 the parents' or family's language; for deaf people the language they were
taught in. So it is closer to a first-language question than the Soviet "rodnoy yazyk", and the
table's own title calls it the main language of communication. It is still the ex-USSR question
in name, and `note_public` says it leans towards identity, as the brief asks.

**Columns.** Population, then Uzbek, Karakalpak, Kazakh, Tajik, Kyrgyz, Russian, Turkmen, other.
The English edition heads them "Uzbeks", "Karakalpaks" and so on, copied from the ethnicity
table; the Uzbek edition prints language names (o'zbek, qoraqalpoq, qozoq, tojik, qirg'iz, rus,
turkman, boshqa). No "not stated" column: every row sums to its population.

**Rows.** The nation and the 14 regions: Karakalpakstan, the 12 viloyats, Tashkent city. Nothing
finer in this volume. The office says detailed final results will follow; a district table, if
it comes, would be a straight upgrade (the normaliser and mapping carry over).

**Population.** De jure, 39,047,321, including 2,090,953 counted at their usual residence while
temporarily away (65% of them men; printed p.8). The table covers everyone, so
`gap` is left out.

**Search.** aholi.stat.uz's results page also offers an xlsx (`zhadvallar-sajt-uchun-uzb_p85033.xlsx`);
it holds only the agriculture tables 1.1 to 1.11, as religiondots found for the three earlier
xlsx. siat.stat.uz has no language indicator (religiondots `sources/uz.md` §2).

## 2. Checks (all asserted in `uz_census.py`)

1. Every row's eight columns sum to its population; the 14 regions sum to the national row in
   every column; the nation is 39,047,321.
2. The English and Uzbek editions print the same 135 numbers. A reprint, so this catches a
   misread, not an office error.
3. **A second table of the same census**: each region's population equals the ethnic composition
   table's (English PDF p.64), and that table's groups sum per region.

Language against nationality per region (printed by the script, not asserted), thousands:

| | Uzbek | Karakalpak | Kazakh | Tajik | Kyrgyz | Russian | Turkmen | other |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| nation | +753 | +69 | -146 | -426 | -140 | +181 | -81 | -212 |
| Karakalpakstan | +18 | +72 | -69 | 0 | 0 | +4 | -20 | -3 |
| Namangan | +158 | +1 | 0 | -151 | -5 | +4 | 0 | -5 |
| Fergana | +130 | +1 | 0 | -89 | -42 | +7 | 0 | -8 |
| Surkhandarya | +100 | +1 | 0 | -80 | 0 | +2 | -18 | -4 |
| Samarkand | -40 | +1 | -1 | +46 | 0 | +10 | 0 | -16 |
| Tashkent region | +110 | 0 | -59 | -37 | -4 | +39 | -1 | -49 |
| Tashkent city | +11 | +1 | -12 | -8 | -1 | +94 | -1 | -84 |

It reads the way the two questions should differ: minorities naming Uzbek (Fergana valley Tajiks
and Kyrgyz, Karakalpakstan's Kazakhs naming Karakalpak), the "other" nationalities (Tatars,
Koreans, Ukrainians) naming Russian in Tashkent. Samarkand is the one region where more people
named Tajik than were counted Tajik by nationality.

## 3. Mapping and nodes

All eight columns land on existing nodes (`taxonomy/uz2026.py`): `turkic.uzbek`,
`turkic.karakalpak`, `turkic.kazakh`, `indoeuropean.iranian.tajik`, `turkic.kyrgyz`,
`indoeuropean.slavic.east.russian`, `turkic.turkmen`, and "other" (boshqa, 88,244, 0.23%) on the
root `other`: it holds every language without a column, Korean, Ukrainian and Armenian as well as
Tatar and Uyghur, so no narrower node contains it. No new nodes; `tree.d/uz.txt` repeats them so
the fragment stands alone.

**Colour.** Turkmen was generated `#f58de0`, a light pink almost on Uzbek's `#ec81c0`; they meet
in Khorezm, Karakalpakstan and the south. Hand-picked `0.58 0.18 5` (`#cb3e6c`, crimson) in the
fragment, checked on a dot-cloud swatch against Uzbek, Karakalpak (`#c5521d`), Kazakh, Kyrgyz,
Tajik and Russian. Turkmen was bare in every other fragment (au, cz, fi, kg, pl: migrants), so
nothing else depended on the old colour. Kazakh's pale rose next to Uzbek's pink (Tashkent
region) is kg's hand pick and distinguishable; left.

## 4. Geography

religiondots' Uzbekistan layer, read-only: `data/geo/uz/uz_hexes.gpkg` (103,753 Kontur hexes over
14 units, with `pop`), keyed on the COD-AB p-codes UZ35, UZ03 ... UZ26 that `uz_lookup.csv`
carries and `uz_census.py` writes directly. The census rows run in the office's SOATO order,
which is the order religiondots' lookup and its own parse of the same volume follow; the region
names were matched by hand once (`REGIONS` in the script) and `countries/uz.py` asserts all 14
join. `pop_weight`, not religiondots' `_uz_place_weight`: both read the same `pop` column.

The scatter capped two Kontur blocks already registered (Tashkent city's southern districts,
58% of the city's Kontur population lowered to its ring median, and the block beside it in
Tashkent region). Kontur holds 1.8 million in Tashkent city against the census's 3.2 million;
it is a weight within the unit only, so that does not move any count.

## 5. Result

39,047,321 people, 8 nodes, 14 units. Uzbek 35,654,647 (91.3%), Karakalpak 910,462 (2.3%),
Tajik 851,326 (2.2%), Russian 787,493 (2.0%), Kazakh 561,316 (1.4%), Turkmen 114,274 (0.3%),
other 88,244 (0.2%), Kyrgyz 79,559 (0.2%). 39,044 dots at 1:1,000; every row `measured`.

## 6. Calls

- **Region grain.** The preliminary volume prints nothing finer, and the brief sets no floor. A
  language is spread over its whole region by population, so Tashkent region's 277,000 Kazakh
  speakers are not pulled to the Kazakh border districts, nor Surkhandarya's Tajik speakers to
  the north of the region. A nationality-by-district crosswalk would place them better, but no
  district table of the 2026 census is out and any other base is a proxy, Anita's to allow; not
  built.
- **The Tajik figure is drawn as printed.** It is the census's own count and the census is the
  source. The figure is disputed (outside estimates of Tajik speakers run far higher, above all
  for the cities of Samarkand and Bukhara), and `note_public` says so with the census's numbers
  (Samarkand region 3.6%, Bukhara region 0.7%) and the nationality comparison. At region grain
  nothing is drawn finely enough to single anyone out, so this was not raised as an ask; it is the
  call most worth a second look.
- **Preliminary results.** Final results may revise counts and add a finer table; `how` says
  "census, 2026", the source line says preliminary.
- **Not used:** the 1989 Soviet census (oblast native language). Thirty-seven years old, before
  the emigration of most Russians, Germans and Jews, and the 2026 table covers the same grain.
