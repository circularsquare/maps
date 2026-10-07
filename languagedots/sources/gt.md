# Guatemala: the record

Drawn 2026-10-04 (agent d9e44929-gt). 13,552,901 people aged 4 and over of the 14,901,286 the
2018 census counted, 340 municipios, 28 language nodes. 13,540 dots at 1:1000, 1 ring.

## Source

INE, XII Censo Nacional de Población y VII de Vivienda 2018, through INE's own open REDATAM
webserver over the full person file (base `CPVGT2018`,
`https://redatam2018.ine.gob.gt/bingtm/`). `ine.gob.gt` itself is behind a Radware bot wall;
the REDATAM host is not. Its Frequency web form answers HTTP 500, but the program route
(`RpWebStats.exe/CmdSet`, `ITEM=PROGRED`) runs, and so does the dictionary page.
`python sources/gt_censo.py --fetch` runs seven programs (saved beside their output in
`data/raw/gt/*.program.txt`) and writes `data/normalized/gt.csv` and `gt_units.csv`.

The question: `PCP15` "Idioma en el que aprendió a hablar", the language the person learned to
speak in: a mother-tongue question with one answer each, asked of everyone aged 4 and over
(13,566,897; the 1,334,389 under 4 are "No Aplica"). 29 categories: 22 Mayan languages, Xinka,
Garífuna, Español, Inglés, Señas, Otro idioma, No habla. No "not stated" category exists. The
census also asks `PCP12` pueblo, `PCP13` comunidad lingüística and `PCP25` up to three other
languages spoken; none of those is drawn.

## Checks (numbers from the run)

1. 340 municipios in the AREALIST (municipio x language) and in both AREABREAK runs, with the
   same codes; every AREALIST row sums to its printed total.
2. The AREALIST equals `FREQUENCY OF PCP15 AREABREAK MUPIO` (a second engine path, which also
   prints each code's name) in all 340 x 29 cells.
3. Summed over municipios, every language equals the national frequency; asked total
   13,566,897; all ages (from `PCP6 AREABREAK MUPIO`) 14,901,286, the census count.
4. In every municipio, asked + No Aplica = all ages. Under-4 share runs 5.8% (Estanzuela) to
   13.0% (San Miguel Acatán).
5. An age split (`PCP7 >= 4`) by PCP15: nobody under 4 answered, everyone 4+ did.
6. Geography, below.

## Geography

OCHA COD-AB Guatemala admin2 (valid_on 2019-02-07), read from religiondots' download
(`religiondots/data/raw/gt/shp/`, read-only). Religiondots draws Guatemala by department, so
its hex layer is not reused; `sources/gt_geo.py` builds a municipio layer with `_grid.hex_layer`.

- 342 features: the 340 municipios plus two lakes (Amatitlán `GT0100`, Atitlán `GT0700`),
  dropped as units. 63 Kontur hexes centred in the lakes (32,587 Kontur people) go to the
  nearest municipio, at most 597 m away.
- **COD swaps two pcodes.** COD's `GT0206` is named Sanarate and `GT0207` Sansare; INE's
  census has 0206 Sansare and 0207 Sanarate. Codes join 340 for 340 and every total passes
  either way; the name witness caught it. The polygons follow their names: COD's "Sanarate"
  is 274 km2 holding 40,553 Kontur people, "Sansare" 144 km2 holding 13,574, and the census
  counts 39,444 in Sanarate and 13,154 in Sansare. Each census code takes the polygon of the
  same name (`POLYGON_FOR`); both then sit at 0.85 of the national Kontur/census ratio
  (asserted), where the code join would have put Sansare at 2.54 and Sanarate at 0.28.
- The other 22 name differences are the census's full name against COD's short one ("San Juan
  Comalapa" / "Comalapa", "Playa Grande Ixcán" / "Ixcán") or a spelling ("San Raymundo" /
  "San Raimundo"); each pinned in `NAME_PINNED`. Department prefix agrees for all 340.
- Kontur over the census per municipio: national ratio 1.213, normalised p10 0.65, median
  0.91, p90 1.37; 3 of 340 outside a factor of 3 (GT1706, GT0416, GT0808, all high); log r =
  0.933 against a best of 0.167 over 500 shuffles. 692 hexes (61,766 Kontur people) outside
  every unit, across the borders and at sea, dropped. Kontur only places dots inside a
  municipio; the counts are the census's. No Kontur cap block stopped the scatter.

## Calls

- **Every INE label is a node.** New: Achi, Tz'utujil, Sakapulteko, Sipakapense, Uspanteko,
  Poqomam, Poqomchi' under `mayan.kichean`; Chalchiteko under `mayan.mamean`; Ch'orti' under
  `mayan.cholan_tzeltalan`; Itza' and Mopan under `mayan.yucatecan`; `isolate.xinka`;
  `arawakan.garifuna`. Reused from mx.txt: K'iche', Q'eqchi', Kaqchikel (INE spells it
  "Kaqchiquel"), Mam, Ixil, Awakateko, Q'anjob'al, Chuj, Akateko, Jakalteko (INE
  "Jakalteko/Popti'"); INE's "Tektiteko" is Mexico's "Teko", one node.
- **Chalchiteko is its own node** though Glottolog has no entry for it (it files the speech of
  eastern Aguacatán under Awakateko): INE prints it as a language and 21,550 people answered
  it. Of those, 5,620 also identified as Ladino rather than Maya (the pueblo crosstab), a
  quarter, which no other Mayan language comes near; drawn as answered.
- **Xinka under `isolate`.** Glottolog has Xincan as a family of four varieties, all but
  extinct; INE prints one "Xinka" and readers know it as one language, the reasoning mx.txt
  used for Huave and P'urhépecha. 2,755 people.
- **Señas** (sign language, unnamed) on the `signlanguage` root, as au2021's "nfd". Almost all
  will be LENSEGUA, but the census does not say.
- **Otro idioma** on `other`: foreign languages and indigenous languages of other countries
  share one line, so it cannot go on `americas_other`.
- **Under 4 and "No habla" are not drawn** (`gap`), as Mexico's, the UK's and the US's
  children who were not asked.
- **Colours.** Hand-picked in `tree.d/gt.txt` for the big languages and their neighbours:
  K'iche' mid blue, Q'eqchi' light cyan, Mam violet, Kaqchikel pale sky, Tz'utujil and Achi
  dark teals, Poqomchi' indigo, Ixil pale lilac, Q'anjob'al light teal, Chuj lavender,
  Awakateko dark blue, Chalchiteko teal-green, Mopan blue. These nodes were bare in mx.txt, so
  Mexico's few speakers of them recolour too; Mam (305) was kept away from Tsotsil (272) for
  Chiapas. Build.py's generator gave Chalchiteko and Mopan exactly the Mayan root's colour
  (a symmetric pair of steps cancelling: group = root + step, member = group - step); hand
  picks avoid it here, but other countries' generated colours can hit the same thing.

## For note_public: pueblo against mother tongue (`gt_pueblo_x_pcp15`)

Aged 4+: 5,584,885 Maya by pueblo (41.2%) against 4,021,870 with a Mayan mother tongue
(29.6%); 1,580,789 Maya (28.3%) learned Spanish first. Xinka by pueblo 238,542 (264,167 all
ages), Xinka mother tongue 2,755. Garífuna by pueblo 18,290, of whom 12,091 learned Spanish
first; 2,856 people have Garífuna as mother tongue.

## Not done

- No second source for the remainder was needed (the census asked everyone 4+).
- The census's `PCP25` (other languages spoken, up to three) could give a speakers layer;
  not built.

## Immigrant languages (2026-10-05, session edd42a8c-latn): nothing to add

Guatemala was listed with the indigenous-only countries, but PCP15 asks everyone aged 4+ their
first language and prints Inglés and "Otro idioma", both already drawn; Garífuna is drawn from
the same question. Splitting "Otro idioma" (on `other`) by country of birth was not done.
