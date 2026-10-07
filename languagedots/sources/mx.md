# Mexico: the record

Drawn 2026-10-04 (agent d9e44929-mx). 119,862,013 people aged 3+ in 2,469 municipios: 7,364,645
indigenous-language speakers, 112,497,368 drawn as Spanish.

## Source

INEGI, Censo de Población y Vivienda 2020, cuestionario básico (the full count).

- **The language table: INEGI's "Consulta interactiva de datos" OLAP cube "Población de 3 años y
  más"** (`inegi.org.mx/sistemas/Olap/Proyectos/bd/censos/cpv2020/P3Mas.asp`, database
  `PV2020_AMD_Poblacion`, footnote "Cuestionario Básico"), dimension "Habla indígena y lengua
  (INALI)" crossed with "Entidad y municipio". Exported as CSV through the cube's own export
  endpoint (`/sistemas/olap/exporta/exporta.aspx`) with an MDX query (`_MDX` in
  `sources/mx_censo.py`). No login or key. The endpoint returns an HTML error page unless
  `Lc_encabeza`, `Cant_Col`, `Cant_Fil` and `completo` are posted; their values do not limit
  the export.
- **The coverage sweep's lead was wrong in the useful direction.** It said municipio x language
  existed only in the ampliado sample microdata. The per-entity tabulados
  (`cpv2020_b_<ent>_05_etnicidad.xlsx`, `chh` checked) do carry only the yes/no per municipio,
  but the cube has the full-count cross. The same cube also has a Religión dimension by
  municipio, which religiondots' `sources/mx.md` §8 lists as not existing (its sweep looked at
  the PX-Web tabulados, not the OLAP cubes); worth telling that project.
- `cpv2020_b_eum_05_etnicidad.xlsx`, sheet 04 (entidad x language): the check.
- ITER 2020 (`iter_00_cpv2020_csv.zip`, religiondots' `data/raw/mx/`, read in place): POBTOT,
  P_3YMAS and P3YM_HLI per municipio and per locality, with coordinates.
- Boundaries: INEGI Marco Geoestadístico 2020 (`00mun`, `00l`), religiondots' `data/geo/mx/mg2020/`,
  read in place. Kontur MX 2023-11-01, downloaded to `data/geo/kontur/`.

The question, asked of everyone aged 3+: does (NAME) speak an indigenous language or dialect;
which one (one answer, coded to INALI's catalogue at the agrupación level: 68 groupings plus three
"insuficientemente especificado", "Otras lenguas indígenas de América" and "No especificado");
does (NAME) also speak Spanish. So it is a "speaks" question, not mother tongue, and indigenous
only (spec §3.5).

`python sources/mx_censo.py --fetch` writes `data/raw/mx/` (2 files, 1.4 MB),
`data/normalized/mx.csv` and `mx_status.csv`; `python sources/mx_geo.py` writes
`data/geo/mx/mx_hexes.gpkg`.

## Checks (numbers from the run)

1. 2,469 municipios. In every one, the 72 language columns sum to "Habla lengua indígena", and
   speakers + non-speakers + not specified = aged 3+.
2. Per municipio, the cube's aged 3+ and speakers equal ITER's P_3YMAS and P3YM_HLI exactly, all
   2,469 (ITER is a separate INEGI product, so this is a real second table).
3. Per entidad and language, the municipios' sum equals the national tabulado's sheet 04 exactly:
   1,628 entidad x language cells and 72 national ones.
4. 7,364,645 speakers of 119,976,584 aged 3+ (6.14%), INEGI's published figures; population
   126,014,024.
5. Units: `00mun` and ITER join both ways, 2,469 = 2,469.
6. Kontur against census population per municipio: national ratio 1.018, normalised p10 0.87,
   median 1.02, p90 1.25, 8 of 2,469 outside a factor of 3 (all tiny municipios: Santa María
   Yalina, Oaxaca, 250 people, sits in one Kontur hex holding 3); log r 0.991 against 0.079 for
   the best of 500 shuffles. 2,137 hexes (283,223 Kontur people) fall outside every municipio.
7. `spk` (speakers per hex) sums to every municipio's P3YM_HLI.

## Calls

- **Remainder (spec §3.5)**: Spanish = the municipio's people aged 3+ who said they do not speak
  an indigenous language, `derived`. **Children under 3 (6,037,440, 4.8%) are not drawn**, as the
  US draws no under-5s: they were not asked, and drawing them as Spanish would put 5-7% Spanish
  dots into municipios where nearly everyone speaks an indigenous language. Brazil drew its
  under-2s as Portuguese; this is the other choice, made on purpose. The 114,571 aged 3+ who did
  not say whether they speak one are not drawn either; both are in `gap`.
- **Second source for the remainder**: none checked. No Mexican census or survey asks everyone
  their language; the census asks only indigenous-language speakers whether they also speak
  Spanish (87% do). Low German in the Mennonite colonies (Chihuahua, Campeche, Durango) and
  immigrant languages are inside "Spanish"; note_public says so.
- **Placement**: Kontur hexes per municipio, and inside a municipio the indigenous-language dots
  go where ITER counts speakers, locality by locality (sources/mx_geo.py's docstring has the
  rules: polygon localities spread by Kontur, point localities on the nearest hex of their
  municipio, 81,257 masked localities sharing the 1,390 speakers otherwise unaccounted for, the
  one- and two-dwelling roll-ups, 43,760 speakers, municipio-wide). Spanish dots go on Kontur
  scaled to the census's aged 3+, less those speakers. All languages of one municipio share the
  speakers' weight, since locality x language is not published; note_public says that two
  languages in one municipio mix. Spot check, dot centroids: Tarahumara -107.1, 27.3 (Sierra
  Tarahumara), Maya -89.0, 20.5, Huasteco -98.9, 22.1, Huave -95.2, 16.4 (the lagoons of
  Tehuantepec), Yaqui -110.3, 27.8, Tsotsil -92.9, 17.0.
- **Not on religiondots' AGEB layer**: `mx_place.gpkg` has no population, so religiondots places
  Mexico in equal shares per AGEB; rural AGEBs average 93 km². Kontur plus ITER's localities puts
  the indigenous villages and the mestizo towns of one municipio apart, which is most of what a
  language map of Mexico shows.
- **Tree** (taxonomy/mx2020.py's docstring and tree.d/mx.txt): one node per INALI agrupación,
  including the clusters (Zapoteco, Mixteco, Chinanteco, Mixe, Náhuatl, Otomí), which the census
  does not split. New roots: Mayan, Oto-Manguean, Mixe-Zoque, Totonac-Tepehua, Cochimí-Yuman,
  Tequistlatecan (Chontal de Oaxaca; a small family in Glottolog, so not an isolate). Huave
  (Glottolog: Huavean), Tarasco (Tarascan) and Seri under `isolate`, as each is one language to
  most readers. Kickapoo under us.txt's Algic. Middle levels: Kaufman's Mayan subgroups (Huastecan is Huasteco alone, so left out); the
  low-level Oto-Manguean groups (Oto-Pamean, Popolocan, Zapotecan, Mixtecan) without the eastern/
  western split; Corachol, Taracahitan, Tepiman (with a Tepehuan group) for Uto-Aztecan; Mixean
  and Zoquean.
- **Remainders**: "Popoluca insuficientemente especificado" (8,427) on Mixe-Zoque, since the
  Veracruz Popolucas are both Mixean and Zoquean; "Tepehuano insuficientemente especificado"
  (317) on the Tepehuan group; "Chontal insuficientemente especificado" (1,704) on
  `americas_other`, because the two Chontals share no family and these speakers are mostly in
  Quintana Roo, Chiapas, México and Campeche, not in Tabasco or Oaxaca, so a split by place has
  nothing to go on. "Otras lenguas indígenas de América" (2,453) and "No especificado" (22,777,
  speaks one, not given) on `americas_other`, as Colombia drew its "indígena sin información".
  Brazil left its unnamed speakers out instead; drawing them keeps every speaker in the count.
- **Colours**: every big language hand-picked in tree.d/mx.txt. Pairs that share municipios were
  checked in OKLab; moved after the first build: Otomí (too close to Zapoteco), Zapoteco darker,
  Mazateco to olive (it sat 0.055 from Zapoteco, 20,878 people shared), Chatino to pale teal
  (beside Mixteco on the Oaxaca coast), Chontal de Tabasco darker (beside Tseltal). Left: the
  small Guatemalan Mayan languages of the Chiapas border (Q'anjob'al, Akateko, Q'eqchi', Chuj)
  sit 0.04-0.05 apart, and Mixteco is close to Huichol and Mayo, which meet it only among migrant
  farm workers in Sinaloa and Baja California.
- **Glottocodes**: none given.

## Files

`sources/mx_censo.py`, `sources/mx_geo.py`, `taxonomy/mx2020.py`, `taxonomy/tree.d/mx.txt`,
`countries/mx.py`, `data/raw/mx/` (2 files), `data/normalized/mx.csv`, `mx_status.csv`,
`data/geo/mx/mx_hexes.gpkg`, `data/geo/kontur/kontur_population_MX_20231101.gpkg` (+ .gz),
`dots_mx.geojson`, `rings_mx.geojson`.

## Immigrant and settler languages (2026-10-05, session edd42a8c-latn)

Anita's priority: the non-indigenous minority languages of the Latin American maps that drew
everyone but indigenous speakers as Spanish. Mexico now draws three more groups, all `derived`,
all out of the old Spanish remainder (people aged 3+ who said they speak no indigenous language).
Spanish 112,497,368 -> 112,195,697; English 188,882, Plautdietsch 54,198, Portuguese 7,049,
French 6,398, German 5,553, Mandarin 4,677, Haitian Creole 4,082, Japanese 3,798, Korean 3,715,
Italian 3,327, Venetian 2,585. 119,808 dots, 360 rings.

Files: `sources/mx_origin.py` (fetch, checks), `sources/latam_immig.py` (the shared rule, below),
`data/raw/mx/cpv2020_olap_p3mas_municipio_{nacimiento,eeuu_edad}_nohli.csv` (+ `.mdx.txt`),
`data/normalized/mx_origin.csv`, `mx_settlers.csv`; `countries/mx.py` changed.

### 1. Country of birth (`sources/mx_origin.py`)

The same INEGI cube as the languages ("Población de 3 años y más") has "Lugar de nacimiento":
entidad, or country under its continent (INEGI's own three-digit codes, mapped to ISO in
`INEGI_ISO`). One MDX export gives municipio x country **sliced to "No habla lengua indígena"**,
so foreign-born who speak an indigenous language (Guatemalans speaking Mam) stay on that
language and nobody is counted twice. A second gives the US-born by age. Checks (asserted): 2,469
municipios; the six continent members sum to "En otro país"; US-born by age sum to the US
column; municipios sum to the query's national row. Aged 3+, no indigenous language: born in
the US 732,142 (464,325 aged 3-17), elsewhere 394,688 (Venezuela 52,452, Guatemala 44,101,
Colombia 35,871, Honduras 34,080, Cuba 25,753, Spain 20,518 ... Canada 12,030, China 10,441,
France 8,939, Brazil 8,499, Haiti 5,277); country not stated 3,419 (drawn as Spanish).

### 2. The shared rule (`sources/latam_immig.py`, used by every country in this pass)

- Spanish-speaking origins (Spain, Hispanic America, Equatorial Guinea) stay on Spanish.
- Everyone else takes `origin_mix.mix(iso, dest)`, less any indigenous-American node (in Mexico
  those speakers answered the census's own question; Belize's Q'eqchi' and Mopan drop out).
- Retention: France's TeO2 share speaking only the host language with their children
  (`fr_build.TEO_FRENCH`) moves onto Spanish; the Americas take "Americas, Oceania" (77.4% keep),
  Europe and Asia their TeO2 regions. TeO2 measures settled families in France; no Latin
  American survey of immigrant home language was found.
- **US-born aged 3-17 are Spanish** (no cited share found). They are 63% of the US-born and
  overwhelmingly children of returning Mexican families (Mexico's US-born population doubled
  2000-2010 with return migration). US-born adults take the US mix (English 85 / Spanish 15)
  with TeO2 retention: 65.9% English. That adult group mixes retirees (San Miguel, Chapala,
  Baja California) with Mexican-American returnees, so English may be high there; a cited
  split would replace it.

### 3. Mennonite colonies: Plautdietsch 54,198 aged 3+

- The census does not find them: its religion question counts 6,109 "Anabautista/Menonita" aged
  3+ nationally (Cuauhtémoc 420, Nuevo Ideal 2,413, Hopelchén 1,314). Die Mennonitische Post's
  colony census (Oct 2022, "Mexico colony census brings surprises", anabaptistworld.org) counts
  74,122 Colony Mennonites, Manitoba Colony 17,212, Swift Current 3,480.
- The colony villages are ITER 2020 localities named "Campo ..." or "...Menonit..." in sixteen
  colony municipios (`COLONY_MUN`: Chihuahua's Cuauhtémoc, Riva Palacio, Cusihuiriachi,
  Namiquipa, Ascensión, Janos, Ahumada, Nuevo Casas Grandes; Durango's Nuevo Ideal; Zacatecas's
  Miguel Auza and Sombrerete; Campeche's Hopelchén, Hecelchakán, Tenabo, Candelaria; Tamaulipas's
  Casas), dropping any locality with 10%+ indigenous-language speakers. Witness: Cuauhtémoc's
  matched villages hold 16,978 people against the Post's Manitoba Colony 17,212.
- Counts are the villages' census population aged 3+ (Chihuahua 30,126 in 278 villages,
  Zacatecas 4,243, Tamaulipas 538), except where the villages' names miss most of the colony:
  Campeche, scaled to 15,000 (Reuters 2022, via Wikipedia) -> 13,307 aged 3+; Durango, scaled to
  6,500 (El Siglo de Durango 2012, via Wikipedia) -> 5,985. All on Plautdietsch (ca.txt's node),
  as Old Colony Mennonites. Total about 57,000 all ages against the Post's 74,122: smaller
  colonies (Quintana Roo, San Luis Potosí, more in Chihuahua) are missed.
- In the colony municipios the Canada-, Belize-, Bolivia- and Paraguay-born (1,704) are taken
  to be colony Mennonites already inside the count, not added again as English, Kriol or
  Spanish.
- Placement: each village's people on the hex nearest its ITER point within its municipio.
  Median Plautdietsch dot -106.70, 28.47 (Cuauhtémoc).

### 4. Chipilo: Venetian 2,416 (+169 from Italy-born)

Ethnologue 18th ed. (2015), Venetian (Mexico): 2,500 speakers (2011), placed at Chipilo de
Francisco Javier Mina, San Gregorio Atzompa (21125; ITER 4,059 people), scaled to aged 3+.

### 5. Not done

- Haitians in Mexico are those of the 2020 census (5,277, Tijuana and the south); the larger
  arrivals after 2021 are not in it.
- No cited US-born adult split; no Latin American retention survey; the smaller Mennonite
  colonies.
