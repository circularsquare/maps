# Moldova: BNS, Recensământul Populaţiei şi al Locuinţelor 2024, mother tongue

Drawn 2026-10-05 (session edd42a8c-md). Rebuild:

```
python sources/md_census.py --fetch     # data/raw/md/ -> data/normalized/md.csv
python sources/md_geo.py                # data/geo/md/md_hexes.gpkg (Kontur MD downloaded once)
python tools/check_country.py md
python scatter.py --country md
```

Mother tongue (`limba maternă`) for **901 UATs** (towns, communes, and Chişinău's five
sectors), 2,409,207 people with usual residence, about 2,700 a unit. Eight answers, every one
measured, nothing suppressed or rounded.

## 1. The table

`https://statistica.gov.md/files/files/ComPresa/Recensamant/2024/Ro/Anexa_Caracteristici_Etnoculturale_RPL2024.xlsx`,
the ethnocultural annexe to the final 2024 results (1,310,944 bytes, open, no login). The same
file religiondots reads for religion (its sheet 5.31); this reads:

| sheet | what | used as |
|---|---|---|
| `5.15` | mother tongue by town/commune (UAT) | the drawn level |
| `5.13` | mother tongue by raion/municipality | cross-check |
| `5.10` | mother tongue, national, 2024 beside 2014 | cross-check |
| `5.37` | mother tongue × language usually spoken, ages 3+, national | the record only (§4) |

Sheets `5.22` (language usually spoken, ages 3+, by UAT) and `5.38` (language competences) exist
too. Mother tongue was chosen because it covers everyone, not ages 3+, and it is the question the
rest of the map draws (§4 compares the two).

Answers: Moldovenească, Română, Ucraineană, Rusă, Găgăuză, Bulgară, Romani (Ţigănească), Altă
limbă, and Nu au declarat limba maternă. The `Moldovenească sau Română` column is the sheet's own
sum of the first two (footnote 3, "for information/analysis") and is not drawn.

**Not covered** (footnote 1): the left bank of the Nistru, Bender (with Proteagailovca), Chiţcani
commune (with Mereneşti and Zahorna), Cremenciug and Gîsca (Căuşeni), Corjova with Mahala and
Roghi (Dubăsari). That is the `gap`.

## 2. Checks (all asserted by `sources/md_census.py`)

- Rows are interleaved: 35 raion rows (codes ending `00000`), the `or. Chişinău` city row
  (`0101000`, the sum of the five sectors under it), and 901 leaves. Raions and the city row are
  dropped by code, as in religiondots. Codes lose their leading zero in some cells; zero-filled.
- Every leaf's nine columns sum to its total; Moldovenească + Română equals the sheet's own sum
  column in all 901.
- The 901 leaves sum to the sheet's `Total` row in every column, and to 2,409,207.
- All 35 raion rows equal the sum of their own leaves, column by column; sheet 5.13 carries the
  same raion figures as 5.15.
- The five Chişinău sectors sum to 567,038, the dropped city row.
- Sheet 5.10's national figures agree with 5.15 for every named answer. 5.10 names nothing
  inside `Altă limbă` either.

National: Moldovan 1,159,857 (48.1%), Romanian 765,838 (31.8%), Russian 280,050, Gagauz 87,407,
Ukrainian 71,878, Bulgarian 28,839, Romani 7,640, other 6,116, not declared 1,582.

## 3. Geography

Units are religiondots' `data/geo/md/md_uat.gpkg` (read only): BNS's own commune layer on
`gis.statistica.md`, keyed by the CUATM code, with the Chişinău sectors cut from OSM and clipped
to the city (religiondots `sources/md_geo.py` and `md.md` §4 have the checks, including BNS's
layer population matching the census for all 896 non-sector units). The census table's code is
the polygon key, so there is **no name join**; `sources/md_geo.py` asserts the two code sets are
identical (901 each).

Placement: Kontur 2023 r8 hexes by centroid (`sources/_grid.py`), downloaded into
`data/geo/kontur/`. Religiondots draws Moldova on the bare UAT polygons; hexes keep dots off the
fields. Results:

- 20,935 hexes, 3.46M Kontur people; 2,513 hexes (422,119 people) outside every unit, which is
  Transnistria and Bender, as expected.
- Kontur / census nationally 1.262 (Kontur's population predates the 2024 count's emigration
  losses). Per unit, normalised: p10 0.65, median 1.05, p90 1.71; 6 of 901 outside a factor of 3
  (lowest 3401000 0.25, highest 6223000 3.20). Log r 0.896 against 0.123 for the best of 500
  shuffles. Since only the placement inside a unit is borrowed, these do not move any count.
- Every UAT has at least one populated hex (the fallback that gives an empty UAT its own polygon
  was not needed).
- Scatter: 2,403 dots at 1:1000; 4,625 people (0.19%) are under one dot per language and draw
  none. No Kontur cap block was hit.

## 4. Calls

- **Moldovan and Romanian are two nodes** (`romance.moldovan`, `romance.romanian`), as in
  Ukraine, Czechia, Finland, Poland and Russia on this map. Glottolog has one language
  (Moldavian, mold1248, a dialect of Romanian), and BNS prints a sum column, but the census asked
  and printed the answers apart and the split is an identity statement people made. The split
  is geographic: in Chişinău city Romanian leads (260,989 to 149,169), while the countryside and
  the north lean Moldovan. The colours (amber and pink, from ua.txt and cz.txt) are unchanged.
  Folding them into one node would be a one-line change in `taxonomy/md2024.py`.
- **`Altă limbă` on `other`.** Not named by UAT. Sheet 5.37 (ages 3+, national) breaks the
  mother-tongue row down: Turkish 659, Arabic 605, English 507, Belarusian 420, Armenian 401,
  Azerbaijani 374, Italian 255, German 172, Polish 145, Ivrit 126, French 119, Georgian 93,
  Tatar 86, Spanish 75, Czech 71, Greek 35, Yiddish 31, Portuguese 19, other 1,795. A mix of
  minority, Soviet-era and migrant languages; no narrower node holds it, and none is drawn on
  its own.
- **Mother tongue, not usual language.** The ex-USSR "native language" caveat applies, and
  `note_public` says so: 280,050 gave Russian as mother tongue, while 370,599 aged 3+ usually
  speak it (5.37). The usual-language table (5.22) is by UAT too and could be drawn instead; the
  mother-tongue one covers everyone and matches the rest of the map's question.
- **Colours** checked for the south (Gagauz #d863a0 beside Moldovan #ffb86a and Bulgarian
  #00978a) and the north (Ukrainian #bfd869, Russian #54b85b, Romani #a95c90). Romanian
  #e2a0c5, Gagauz and Romani are all in the pink range but differ clearly in lightness, and
  Gagauz and Romanian rarely share a commune (Găgăuzia: 165 Romanian against 80,026 Gagauz). No
  fragment `tree.d/md.txt` was needed: every node existed.

Places that show the table at work: UTA Găgăuzia is 77% Gagauz (80,026 of 103,668); Taraclia raion
is 61% Bulgarian (16,230 of 26,435); Bălţi is 37% Russian.

## Transnistria (added 2026-10-06, session `5d7dac7e-cau`)

Anita, 2026-10-06: draw the hatched left bank. Files: `sources/md_pmr.py`, `taxonomy/md2015_pmr.py`,
`taxonomy/tree.d/md.txt` (new: Moldova had no fragment and borrowed every node; it now repeats
them bare), `countries/md.py` (counts, `parts`, `drawn_named`, note). Outputs
`data/normalized/md_pmr.csv`, `data/geo/md/md_pmr_units.gpkg`, `md_plus_hexes.gpkg` (Moldova's
layer plus Transnistria's hexes; `countries/md.py` reads it, so re-run `md_pmr.py` after
`md_geo.py`). Geography follows religiondots: inside `md`, units `PMR-*`.

- **Table.** The Transnistrian authorities' 2015 census, nationality by city and raion (Bender,
  Tiraspol, Dnestrovsk, Grigoriopol, Dubossary, Kamenka, Rybnitsa, Slobodzeya), through
  `pop-stat.mashke.org/pmr-ethnic2015.htm`: 475,007. Checks: every row's nationalities sum to its
  total; the 8 units sum to the total row in 11 of 12 columns. Not "refused": 4,603 over the units
  against 1,974 in the total row. "Refused" is a subset of "undeclared" (rows sum without it) and is
  not drawn either way.
- **No native-language table** by raion was found (the census asked it: searched ru.wikipedia's
  census and population articles, search engines for the statistics service's results). Read as
  nationality, every row `derived`.
- **Retention**: Moldova's own 2024 census, nationality by mother tongue (BNS table 5.33, the
  annexe `md_census.py` reads), national shares: Moldovans 92.9% Moldovan/Romanian, 6.5% Russian;
  Ukrainians 50.2% Ukrainian, 44.1% Russian; Russians 95.2% Russian; Gagauz 87.8% Gagauz; Bulgarians
  71.7% Bulgarian; Belarusians 14.8%, Germans 13.0%, Poles 8.6% their own. **These are right-bank
  shares; the left bank is more Russian-speaking** (89% of pupils taught in Russian, 2024/25,
  ru.wikipedia "Naselenie PMR"), so Russian is probably understated. No left-bank source says by how
  much; said in `note_public`.
- **Calls.** Moldovan and Romanian answers are summed and drawn as Moldovan (the Transnistrian
  official name; the right bank's split is not carried over). "Transnistrians" (1,013): Russian.
  "Other" (4,209): `other`. Undeclared (66,432, 14.0%): not drawn, in `gap`.
- Drawn: 408,575. Russian 195,685, Moldovan 137,366, Ukrainian 56,415, Bulgarian 8,368, Gagauz
  5,539, other 4,397, then Belarusian, German, Romani, Polish.
- **Units**: OSM relations via polygons.openstreetmap.fr (Tiraspol city council 1702219 less
  Dnestrovsk 8290840, Bender 944727, raions 1702214-1702218). No overlaps.
- **Placement**: Kontur MD hexes not already in `md_hexes.gpkg`, by centroid; 91 more (2,219
  people) within 1 km of a unit snapped to it. 16 hexes of Moldova's own layer (2,850 Kontur people,
  8 UATs: the left-bank communes Chişinău administers) sit inside the OSM raions and stay Moldova's,
  since BNS counted them. Kontur/census normalised 0.72-1.64, except Dnestrovsk 0.10 (OSM's relation
  is the town core).

Room for improvement: the 2015 census's native-language tables by raion, if they were ever
published, would replace the right-bank shares.
