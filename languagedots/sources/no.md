# Norway (no): record

Drawn 2026-10-05 (session edd42a8c-mono4). 5,488,984 people (SSB, 1 January 2023), 356
kommuner, 533 nodes (nearly all from origin mixes), every row `derived`. 5,381 dots, 438 rings.

```
python sources/no_ssb.py --fetch      # SSB 09817, 07459, KLASS 91->552 into data/raw/no/
python sources/no_ssb.py              # data/normalized/no.csv; data/geo/no/no_hexes.gpkg if missing
python sources/origin_mix.py --fragment no
python taxonomy/build.py
python tools/check_country.py no
python scatter.py --country no
```

Files: `sources/no_ssb.py`, `sources/no_svalbard.py` (§5), `taxonomy/no2023.py` (identity), `taxonomy/tree.d/no.txt`,
`countries/no.py`, `data/normalized/no.csv`, `data/raw/no/`, `data/geo/no/no_hexes.gpkg`.
Kontur NO downloaded into `data/geo/kontur/` (religiondots had none).

## 1. The rule and the sources

No census or register asks language. Anita's 2026-10-05 rule for rich countries: the national
language, regional languages from cited estimates, immigrant languages by origin; all `derived`.
Sweden's build (`sources/se.md`) is the model, done leaner.

- **SSB 07459** population per kommune and **09817** immigrants (B) and Norwegian-born to two
  immigrant parents (C) by country background per kommune, 1 January 2023: the last year on the
  2020-2023 kommune codes, which religiondots' GISCO LAU 2021 polygons carry (2024 renumbered
  Viken's kommuner). Immigrants 877,227; Norwegian-born to immigrant parents 213,812; 247
  country backgrounds; countries sum to SSB's "all" within 2 per kommune.
- SSB 3-digit country codes -> ISO alpha-3 by KLASS correspondence 91->552, -> alpha-2 by
  queue.csv; Kosovo and the UK hand-coded (KLASS gives no single match); 36 people (Gibraltar,
  Channel Islands, Falklands, French Guiana, unknown) on `other`.

## 2. How the counts are made

- Each country background through `origin_mix.mix(iso, "no")` (home mix if drawn, else main
  language). Immigrants keep it whole; Norwegian-born keep it at **78%**, the rest Norwegian:
  Parkvall 2009's figure for Sweden's second generation with two foreign-born parents, borrowed
  (no Norwegian figure found). Everyone else Norwegian.
- **North Sami 10,000**: Samisk språkundersøkelse 2012 (Solstad, ed.): "nærmere 20.000"
  North Sami speakers in Norway, Sweden and Finland, "om lag halvparten i Norge". Half in
  Kautokeino and Karasjok (93% of each; the report: nearly everyone there speaks it), half over
  Tana, Nesseby, Porsanger, Kåfjord, Lavangen, Tjeldsund by population (34% of each). Carved out
  of Norwegian.
- Drawn: Norwegian 80.8%, Polish 2.2%, Russian 0.8%, Arabic 0.8%, Lithuanian 0.8%, Swedish 0.7%.

**Placement**: Kontur hexes keyed to the kommuner by centroid (`_grid.hex_layer`): log
correlation with SSB's kommune populations 0.978 (shuffled best 0.174); 3 of 356 kommuner
outside a factor of 3 (Kontur's cabins and holiday areas, Valdres/Hallingdal codes 34xx/30xx);
310,413 Kontur people in hexes outside every LAU polygon (coastline generalisation) are dropped,
which only affects placement weights. Counts are SSB's.

## 3. Calls someone might reverse

- North Sami placement: half on Kautokeino and Karasjok is a reading of the report's "nearly
  everyone", not a count; the rest are spread at one rate. Sami in Tromsø, Alta and Oslo (many
  speakers) are not drawn there.
- Lule and South Sami (a few hundred each), Kven, Romani and Norwegian Romani not drawn.
- 78% second-generation retention borrowed from Sweden.
- Vintage 2023, for the polygon codes.
- Svalbard: see §5 (drawn since 2026-10-06).

## 5. Svalbard (added 2026-10-06, session 5d7dac7e-sj)

Anita: Svalbard was hatched as not drawn; draw it. religiondots draws it inside `no` as one
unit `NO-21` (its sources/no_geo.py, ruling 2026-10-04), so it goes inside `no` here too, as three
units, without touching a mainland row. `python sources/no_svalbard.py --fetch` (raw files in
`data/raw/no/`) writes `data/normalized/no_svalbard.csv` and appends 272 Svalbard hexes to
`data/geo/no/no_hexes.gpkg` (mainland's 170,240 hexes kept as they were; re-run it if no_ssb.py
ever rebuilds the hex layer). Not a named breakaway area, so no `drawn_named`: the not-drawn test
counts every placement polygon of a counted unit as drawn ground.

Counts, 1 January 2026, all `derived`:
- **NO-21-LYR Longyearbyen and Ny-Ålesund, 2,512** (SSB 07430: 1,648 resident on the mainland
  + 864 from abroad). Citizenship from the Flourish chart (figur 2) of SSB's article of 3 March
  2026, "Høy befolkningsutskiftning og økende mangfold på Svalbard ved inngangen av 2026": Norway
  1,598, Philippines 136, Thailand 115, Germany 78, Sweden 59, Russia 57, UK 45, France 43,
  Denmark 34, Poland 33, USA 25, Ukraine 22, Finland 18, rest 249. **Check**: sums to 2,512, its
  Norway equals table 12622's, and 12622's six groups less the chart's countries give exactly the
  chart's rest (Nordic 3, old EU/EFTA 140, other Europe 15, Africa and Asia 32, Americas and
  Oceania 59).
- **NO-21-BAR Barentsburg and Pyramiden, 392** (07430). No table; the article's all-Svalbard
  largest citizenships (Russia 305, Philippines 136, Thailand 115, Tajikistan 83, Ukraine 78,
  Germany 78, Sweden 59, of 2,904) less the chart: Russia 248, Ukraine 56, Tajikistan 83 (all 83
  put here: the chart has no Tajik row and Longyearbyen's whole Africa-and-Asia rest is 32), 5
  unassigned on `other`. Philippines, Thailand, Germany and Sweden match the chart exactly, so
  none of them is in Barentsburg.
- **NO-21-HOR Hornsund, 10** (07430): the Polish Academy of Sciences' polar station, drawn as
  Polish (no citizenship figure).

Languages: Norwegian citizens Norwegian; each citizenship through `origin_mix.mix(iso, "no")`,
as the mainland. **Barentsburg's Ukrainians** take Donetsk oblast's 2001 native-language split
(this map's ua.csv: 75.7% Russian, 24.3% Ukrainian), because Arktikugol's Ukrainian miners mostly
came from the Donbas (Wikipedia, "Arktikugol" and "Barentsburg"; not a measured figure for
Svalbard). The chart's rest (249) is on `other`: some 50 citizenships, languages unknown.
Drawn: Norwegian 1,598, Russian 336, other 254, German 78, Tajik 73, English 66, Swedish 60.

**Double count**: the 1,648 Longyearbyen residents registered in a mainland kommune are in no.csv
too (0.03% of Norway). religiondots takes them out of the mainland pro rata; here the brief was
to leave mainland rows alone, so they are counted twice. Vintage 2026 against the mainland's 2023.

Placement: religiondots' Svalbard hexes (Kontur SJ calibrated to 07430, read-only) re-keyed.
Ny-Ålesund gets 35 of LYR and Pyramiden 10 of BAR, on Kontur's own proportions there (both
uncited placement figures: Kings Bay's usual winter count, a caretaker crew). Every other
Svalbard hex (Sveagruva, closed 2017, where Kontur puts 1,900) is in LYR at weight 0.

Tool fixes made on the way (2026-10-06): `origin_mix.mix()` now returns regrouped (drawn) ids,
so no_svalbard.py maps them back to written ids; `origin_mix.py --fragment` read regrouped
counts and wrote drawn ids (bantu.zone_a, kwa.gbe...) that build.py rejects, now fixed to read
the written counts and take written labels from the fragments.

## 4. Room for improvement

A Sami speaker count by kommune (the Folkeregister has recorded Sami language since 2021 but
publishes nothing yet); Kven estimates for Nord-Troms and Finnmark; 2024 codes once LAU 2024
polygons are available.
