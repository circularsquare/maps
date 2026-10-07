# Latvia: Tautas skaitīšana 2011, language mostly spoken at home

Built 2026-10-05 (session d9e44929-lv). Rebuild:

```
python sources/lv_census.py [--fetch]   -> data/normalized/lv.csv
python sources/lv_geo.py                -> data/geo/lv/lv_hexes.gpkg
python taxonomy/build.py
python tools/check_country.py lv
python scatter.py --country lv
```

Drawn: 1,876,812 people on 119 municipalities, 7 labels on 7 nodes, 1,872 dots, 1 ring. Not
drawn: 193,559 people (9.35% of 2,070,371) whose home language the census did not record.

## 1. Which source

The coverage sweep's lead (tier B, question `none`) was right about 2021: Latvia's 2021 census
was compiled from registers and collected no language. Its pointer, CSB's press release 21052
(23.10.2023), is the **Adult Education Survey 2022**, ages 18-69, national with Latgale picked
out for Latgalian only: mother tongue Latvian 64.3%, Russian 37.7% (several allowed), Ukrainian
1.7%, Latgalian 1.3% (8.4% in Latgale); language used at home, one answer: Latvian 62.0%, Russian
34.6%, Latgalian 1.2% (8.8% in Latgale). National survey shares, so not drawable.

**Drawn: the 2011 census**, the last count of everyone's home language with geography. PxWeb on
data.stat.gov.lv, folder `OSP_OD/tautassk/taut/tsk2011`, open, no key; the API
(`/api/v1/en/OSP_OD/tautassk/taut/tsk2011/<table>`) takes a JSON POST. Tables used:

| table | content | use |
|---|---|---|
| `TSG11-07` | territory x sex x home language x age | drawn (sex and age totals) |
| `TSG11-060` | territory x ethnicity | the population per unit (its total) |
| `TSG11-071` | territory x ethnicity x home language | check |
| `TSG11-08` | territory x age x daily Latgalian use x home language | check, and the Latgalian figures |

Territory in all four: Latvia, 6 statistical regions, 119 municipalities (9 republican cities, 110
novadi; the map of 2009-2021). Nothing finer exists for language; the 2000 census (mother tongue)
is older and was not used.

**The question**: `mājās pārsvarā lietotā valoda`, the language mostly spoken at home, one
answer, seven printed: Latvian, Russian, Belarusian, Ukrainian, Polish, Lithuanian, other.

## 2. Checks (sources/lv_census.py prints them)

- TSG11-07's total is exactly the sum of its seven answers in every unit, 1,876,812 nationally,
  which is **not** the population: TSG11-060's total is 2,070,371, the published census figure.
  The difference, 193,559, is written per unit as `Not stated (population less those with a home
  language)`. CSB's release gives shares "of the population" computed on the 1,876,812 (Latvian
  62.1%). Ethnicity, by contrast, is known for all but 8,198 (register-held), so the missing home
  language is presumably people the census took from registers without an answer; CSB's pages
  checked do not say. Not stated is 3.6% (Aknīste) to 23.7% (Cibla); by region Riga 11.2%,
  Latgale 9.6%, Kurzeme 9.4%, Pierīga 8.7%, Zemgale 7.7%, Vidzeme 6.4%. It is not filled.
- Unit counts 1 / 6 / 119; regions and municipalities each sum to the country in every column.
- TSG11-071 (summed over 8 ethnicities, and its own total) equals TSG11-07 in all 1,008
  (unit, language) cells exactly.
- TSG11-08 (summed over Latgalian yes/no) equals it everywhere except 5 people: Riga city 2 (one
  Latvian, one Russian at home) and one Latvian-at-home person in each of the Pierīga, Vidzeme and
  Zemgale region rows, whose municipalities match. Held to 2 per cell, 5 nationally.

National, of those with a home language: Latvian 62.07%, Russian 37.23%, other 0.37%,
Lithuanian 0.12%, Polish 0.09%, Ukrainian 0.09%, Belarusian 0.03%.

## 3. Calls

- **Latgalian is not drawn as its own language.** The home-language question did not offer it;
  by law it is a variety of Latvian, so its speakers answered Latvian. TSG11-08's separate
  question, "do you use Latgalian on a daily basis", got 164,506 yes (8.8% of those answering;
  Latgale 97,590, 35.5%; Rēzekne city 40.5%; Riga city 29,390, 5.0%), of whom 123,052 mostly
  speak Latvian at home, 40,553 Russian and 901 something else. Moving the Latvian-at-home daily
  users onto a Latgalian node would draw 6.6% as Latgalian-at-home, where the 2022 survey's
  one-answer home-language question gives 1.2% (ages 18-69): daily use is several times main home
  use, so the split would overstate Latgalian about fivefold. note_public gives both figures. If
  Anita would rather see Latgale's Latgalian on the map, the Latvian-at-home x daily-use cell is
  per municipality in data/raw/lv/TSG11-08.json; it would be a `derived` split, labelled as daily
  use.
- `other` (6,922) on `other`: the table does not separate Livonian or Romani from migrants'
  languages, so no indigenous remainder can be told apart.
- No new nodes, so no tree fragment. Latvian (#ead981, pale yellow) and Russian (#54b85b, green)
  are far apart; Lithuanian (#cf8e52) borders Latvian in the south and is distinct too.
- Vintage 2011 is stated in `how` and note_public (population since down about a tenth; the
  post-2022 Ukrainian arrivals are absent).

## 4. Geography

`sources/lv_geo.py`: religiondots' `data/geo/lv/lv_lau.gpkg` (GISCO LAU 2021, the same 119
municipalities; read only) as units, joined on the ATVK code (census `LV` + seven digits = GISCO
LAU_ID), asserted both ways and by name (119 of 119 agree). Kontur 2023 hexes (downloaded into
languagedots/data/geo/kontur/, 57,786 hexes) keyed by centroid: 497 hexes, 7,854 people, outside
every unit (coast and border); Kontur/census nationally 0.882, per unit p10 0.92, median 1.06,
p90 1.31; log r = 0.983 against a shuffled best of 0.361. One unit outside a factor of 3: Kocēni
(0960200, 3.62), which rings Valmiera city (0.84); Kontur's Valmiera edge falls in Kocēni. It
moves only where Kocēni's census dots sit, not how many.

**Found on the way, in religiondots (not touched):** GISCO's LAU 2021 workbook swaps the
populations of Jelgava city (0090000, given 21,629; census 2011 59,511) and Jēkabpils city
(0110000, given 50,248; census 2011 24,635); both are in Zemgale (LV009), so only the placement
inside that region is affected. religiondots' lv_lau.gpkg `pop` carries the swap
and its `_lv_place_weight` uses it, so in religiondots Zemgale's dots under-weight Jelgava and
over-weight Jēkabpils. lv_geo.py here asserts the swap and does not use that column.
