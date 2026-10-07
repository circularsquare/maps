# Poland: NSP 2021, language used at home

Built 2026-10-04 (session d9e44929-pl). Rebuild:

```
python sources/pl_nsp.py [--fetch]     -> data/normalized/pl.csv
python taxonomy/build.py
python sources/pl_geo.py               -> data/geo/pl/pl_hexes.gpkg
python tools/check_country.py pl
python scatter.py --country pl
```

Drawn: 38,003,737 people on 2,477 gminas, 344 nodes, 37,975 dots, 31 languages reaching a dot.

## 1. The table

Statistics Poland (GUS), Narodowy Spis Powszechny 2021, final results, page "Tablice z
ostatecznymi danymi w zakresie przynależności narodowo-etnicznej, języka używanego w domu oraz
przynależności do wyznania religijnego":

```
https://stat.gov.pl/download/gfx/portalinformacyjny/pl/defaultaktualnosci/6536/10/1/1/
  jezyk_uzywany_w_domu_-_dane_nsp_2021_dla_kraju_i_jednostek_podzialu_terytorialnego.xlsx
  wyniki_ostateczne_nsp2021_narodowsc_jezyk_wyznanie_2023_11_29.xlsx      (summary annex)
```

Saved in `data/raw/pl/`. stat.gov.pl omits its intermediate certificate, so `--fetch` turns
verification off for that host and checks the files are xlsx (religiondots/sources/pl.md §1 found
the same).

The main workbook: TABL.1/2 national (alphabetical / by size, 348 labels plus Polish, "other than
Polish" and not established), TABL.3 voivodeship, TABL.4 powiat, TABL.5 gmina (2,477, seven-digit
TERYT). Census reference date 31 March 2021.

**The question.** Language usually used at home: Polish and/or up to two other languages. So a
person names one, two or three languages. Per gmina the table gives T (persons), P (persons naming
Polish), O (persons naming at least one other language), N (not established), then each other
language as a count of mentions. The annex's Tab3_JDom is the national cross-table by number of
languages: 36,379,882 named one, 1,623,855 more than one (1,611,784 of them including Polish),
32,381 not established.

**Suppression.** A cell under 10 is not printed at gmina level, under 3 at powiat and voivodeship.
O itself is "<10" in two gminas (Regnów, Młynarze). 85,019 of the 1,903,511 non-Polish mentions
(4.5%) are in cells the gmina table does not print; 246 of the 348 labels appear in no gmina row.

## 2. How a gmina's people are drawn (spec §3.6)

**Step 1, unprinted cells.** Per language, what a level lacks against its parent goes to the
parent's units that print nothing for that language, by population, capped at the largest hidden
value (2, 2, 9), national -> voivodeship -> powiat -> gmina, then rounded to whole people by
carrying the fractions along TERYT code order. Every allocation fits its caps (asserted). Before
this, the annex's voivodeship table (Tab4_JDom_woj, 33 languages) prints 6 cells of 1-2 people
that TABL.3 hides; those are used as printed.

**Step 2, persons.** The table gives more than the mention counts the spec's plain scaling uses:
P and O are person counts, so

- B = P + O - (T - N) is the number who named Polish and another language (nationally 1,611,784,
  exactly Tab3's figure);
- of those, B3 named Polish and two others; only the national share is known: 144,537 / 1,611,784
  = 0.0897 (Tab3: 156,608 people named two non-Polish languages, 12,071 of them without Polish);
- Polish = P - B/2 - B3/6, the non-Polish total = O - B/2 + B3/6, and each other language gets
  the non-Polish total in proportion to its mentions in the gmina.

So Polish and the non-Polish total per gmina are the census's own combinations (exact but for B3
being the national share), and only the split among non-Polish languages within a gmina is the
mention scaling. Nationally Polish comes to 37,038,636.5, which is Tab3's exact figure
(36,256,834 Polish only + 1,467,247 / 2 + 144,537 / 3). Plain scaling of mentions to the
population would have given Polish about 36.2M, under-drawing it by 0.8M. Every row is
`derived`; "not established" (32,381) is the gap.

The two "<10" O cells together hold 9 people (the national O less the printed gminas); each gets
its floor (B >= 0) and the rest by its mentions.

## 3. Checks (all in `pl_nsp.py`, all pass)

| check | result |
|---|---|
| T, P, O, N summed at each of four levels equal the national | exact (O at gmina 9 short: the two "<10") |
| annex Tab3: one + several + not established = T; B from P,O,T,N = Tab3's "with Polish" | 1,611,784 both |
| annex Tab4 (voivodeship x 33 languages) against TABL.2 and TABL.3 | 538 of 538 printed cells agree; 6 cells only the annex prints |
| after step 1, every level sums to TABL.2 per language | exact, 1,903,511 mentions |
| per gmina 0 <= B <= min(P, O); persons sum to T - N | holds; 38,003,737 nationally |

Placement (`pl_geo.py`): the 2,477 gminas of the table and of religiondots' boundaries match both
ways on six-digit TERYT. Kontur hexes by centroid: 1,175 hexes (68,513 people) fall outside every
gmina (border and coast); every gmina has a populated hex; Kontur / census 1.078 nationally, per
gmina p10 0.96, median 1.03, p90 1.13; 30 of 2,477 outside a factor of 3; log correlation r =
0.967 against a best of 0.065 over 500 shuffles.

**The 30 outliers are Kontur's, not the join's.** They are pairs: a town and the rural gmina
round it (Elbląg city 0.13 of its census, its rural ring 16x; likewise Włocławek, Suwałki,
Grudziądz, Skierniewice, Ciechanów, Słupsk, Przemyśl, Augustów). The polygons do not overlap
(tested: each cut by smaller ones it overlaps, none did), the city polygons have the right areas
and contain the city centres, and Kontur puts only 17,000 people within 3 km of Elbląg's centre.
Kontur's 2023 model has moved these towns' people into the countryside round them. Each gmina's
dot count comes from the census, so the effect is only where inside a town or its ring the dots
fall. Wiśniowa (8.6x) and Miedzichowo (5.3x) are Kontur high in a village gmina, not looked into;
the scatter raised no Kontur cap block.

## 4. Calls

- **Every printed label is drawn as its own node** (spec §3.1), including the list-picking tail:
  Abkhaz 446, Acholi 576, Afar 541, Adyghe 836, Arapaho 52, Aleut 38 read like the top of an online
  form's drop-down. In practice the scatter draws none of them: under §3.6 every Polish row is
  `derived`, and `scatter.py` gives a ring only to a language with `measured` rows, so the 313
  nodes under 1,000 shared people draw nothing (28,737 people, 0.08%). That also drops the small
  real ones: the Polish gwary (Goral 369 mentions, Cieszyn 136, Kurpie 111), Wymysorys (10),
  Upper and Lower Sorbian, Karaim, Tatar. Dropping the drop-down labels outright would be a
  one-line change per label in `pl2021.py`; not done, because nothing in the data says which
  answers were mis-clicks.
- **Merges**: gwara śląska into Silesian (same speech, "dialect" against "language" the only
  difference); azerski + azerbejdżański (Azerbaijani); pilipino + filipiński (Filipino); lushai
  into Mizo; bini into Edo; twi and fanti into Akan; ewe and fon onto the tree's Gbe node.
- **Group labels on groups**, as us2024/uk2021/ca2021 put unspecified Chinese: chiński (1,167
  drawn, washed out as "Chinese, language not named"), dolnoniemiecki (Low German), malgaski,
  slavey, guarani, bihari, irański, hmong-mien.
- **ruski** (2,297 mentions, mostly Warsaw, Wrocław, Kraków, Białystok, Gdańsk) is its own node,
  "Ruski (Ruthenian, or colloquial Russian)": in Polish it is both the old word for Ruthenian and
  slang for Russian, and GUS prints Russian, Lemko and Belarusian apart.
- **Lemko** is its own East Slavic leaf beside Rusyn, not under it (Glottolog files it as a Rusyn
  dialect; a child would wash Rusyn out as a group in Canada).
- **"różne inne gwary regionalne i lokalne"** (446) sits on Slavic: GUS's named gwary are Polish
  and Podlachian (East Slavic). **"inne - niesklasyfikowane"** (180) on `other`.
- **tonga** taken as Tongan (ISO 639-2 `ton`), **cziczewa** as Chewa (zm's node).
- **Placement on Kontur hexes rather than religiondots' gmina polygons**, so Warsaw's dots follow
  its built-up area instead of spreading over its forests (religiondots/sources/pl_geo.md §4).
- Colours: Silesian a light yellow-green, Kashubian a pale cyan, Lemko a dark olive, Podlachian a
  mid blue (generated it was a near-twin of Belarusian's green, and they share Hajnówka county).

## 5. Second source

Not required for a home-language census. The national-ethnic question of the same census (annex
Tab1_Etno) agrees in shape: 596,224 declared a Silesian identity (first or second), 179,685
Kashubian, 144,177 German, against 467,145, 89,198 and 216,342 naming the language at home.
