# Brazil: the record

Drawn 2026-10-04 (agent d9e44929-br). 203,080,756 people, 5,570 municípios; 474,856
indigenous-language speakers in 1,990 of them, the rest drawn as Portuguese.

## Source

IBGE, Censo Demográfico 2022, "Etnias e Línguas Indígenas: principais características
sociodemográficas, Resultados do Universo" (ftp.ibge.gov.br/Censos/Censo_Demografico_2022/).

- **Tabela complementar 26**: indigenous people aged 2+ by indigenous language spoken or used
  at home, per município. 8,061 rows, 310 labels. The coverage sweep had only found state level
  (SIDRA 10423) and Terra Indígena level (10402); this table is municipal and was the reason
  not to build Terra Indígena geography.
- **SIDRA 10392 at N6**: the same people per município by how many languages they named (one,
  two, three, não-determinada, mal definida, não sabe, does not speak one).
- **Tabela complementar 21**: national figure per language, for the label check.
- **Agregados por setores, pessoas indígenas**, V01690: indigenous people per census setor,
  used only to place dots.

The question: asked only of indigenous people aged 2+ (indigenous = declared by cor ou raça,
or, inside indigenous localities, by "se considera indígena"). "Fala ou utiliza língua
indígena no domicílio?", then up to three languages, unordered. So it is a home-language
question, several allowed. Portuguese at home was asked too and is not used: an indigenous-
language speaker is drawn on the indigenous language.

`python sources/br_censo.py --fetch` fetches all four into `data/raw/br/` and writes
`data/normalized/br.csv`, `br_status.csv` and `data/processed/br_setor_indigenous.csv`.

## Checks (numbers from the run)

1. Per município, Tabela 26's rows sum to one + 2 x two + 3 x three + não-determinada + mal
   definida + não sabe from SIDRA 10392, exactly, in all 5,570; its three non-language rows
   equal SIDRA's three status counts exactly. 485,051 mentions.
2. Per label, Tabela 26 summed over municípios equals Tabela 21's national figure for 301 of
   306 labels. Five are 1-3 people short in Tabela 26: Ka'apor 3,816 / 3,817, Tiriyó 2,043 /
   2,044, Wapixana 5,191 / 5,192, Xavante 21,772 / 21,773, Yanomami 19,174 / 19,177. Tolerated;
   Tabela 26 agrees with SIDRA per município exactly.
3. 474,856 speakers aged 2+, IBGE's published figure. 463,163 named one language, 8,201 two,
   997 three; 2,495 speak one the census could not name.
4. Setor V01690 totals 1,570,841 indigenous people (all ages) against SIDRA's 1,626,851 aged
   2+: IBGE suppresses ("X") 80,369 setores' counts. Shortfall 88,884 people, handled in
   placement (below). Not enforced.
5. Drawn total 203,078,261 + 2,495 unnamed = 203,080,756, the 2022 population.

## Calls

- **Município, not Terra Indígena.** Tabela 26 gives every language per município, which
  covers people inside and outside TIs alike. TI-level language tables (SIDRA 10402,
  Tabela complementar 4) exist and IBGE publishes its TI-by-UF polygons
  (Indigenas_Primeiros_resultados_do_universo/Arquivos_geoespaciais_vetoriais...), but they
  cover 78% of speakers and would need a second unit system. Placement inside the município
  does most of the same work.
- **Placement**: religiondots' 2022 setor layer (`br_setores_2022.gpkg`, read-only).
  Indigenous-language dots are weighted by each setor's indigenous people, V01690, plus the
  município's suppressed shortfall spread by setor population. Portuguese dots by setor
  population less its expected speakers (indigenous estimate x the município's speaker share).
  Spot check: Tikuna's dots centre on -69.5, -4.0 (Alto Solimões), Kaiowá on -55.0, -22.9
  (southern Mato Grosso do Sul), Kaingang on -52.6, -27.4, Yanomami on -63.7, 2.3.
- **Several languages a person** (spec §3.6): within each município, count = mentions x
  people / mentions. 491 municípios are scaled at all (factor median 0.95, min 0.33). A row is
  `derived` only where the factor is under 0.95 (60,119 speakers); otherwise `measured`
  (412,242). Marking every scaled município derived would have put 83% of speakers there over
  a handful of bilingual people, and derived rows may not ring, which hid 79 small languages.
  With the 0.95 cut, 8 languages still draw nothing (all their rows derived and under one
  dot): Akuriyó, Faruk'woto, Katxuyana, Puinave, Barasána, Mirititapuia, Tanimuka, Kaimbé.
- **Remainder** (spec §3.5): Portuguese = município population minus all its speakers aged 2+,
  `derived`. It includes indigenous children under 2 and indigenous people who speak no
  indigenous language. Second source for the remainder: none checked. The 2010 census
  also asked only indigenous people; older censuses and national surveys were not checked,
  and IPOL's language inventories (Pomeranian, Hunsrik, Talian) are local studies. Not
  corroborated.
- **Not drawn**: "Não determinada" 337, "Mal definida" 1,176, "Não sabe" 982: speakers whose
  language could not be named. Left out of Portuguese as well, stated in `gap`.
- **Tree**: IBGE's classification (SIDRA classification 2105 = the publication's Apêndice 2,
  Rodrigues' troncos). Where Glottolog differs, the census wins (Guató, Kariri, Yatê/Fulniô are
  isolates in Glottolog; Jabutí sits in Nuclear-Macro-Je; Ticuna is Ticuna-Yuri; Aweti is a
  sister of Tupi-Guarani). Mapping decisions are in `taxonomy/br2022.py`'s docstring:
  "não especificado" labels on their family; "Tupi-Guarani (**)" (IBGE: a family declared as
  a language) on Tupi-Guarani; "Tupi-Mondé" on Mondé; "Guarani" on a Guarani group over
  Kaiowá, Mbya, Nhandeva, Avá Guarani; Timbira, Kanela and Kawahiva groups added for their
  unnamed labels; Kayapó, Nambikwára, Yanomami and Wari' stay leaves beside their dialects.
  Eñepá (Panare) moved from "other languages of the Americas" to Cariban. Ypy follows SIDRA
  (unclassified), although its code sits among Tupi-Guarani's. 24 new roots: ask/002-br.md.
- **Colours**: every node in `tree.d/br.txt` coloured explicitly (build.py's generator gives
  siblings one colour; ask/002-br.md). Close pairs among languages sharing a município were
  checked in OKLab; the remaining near pairs (Baniwa/Kuripako 0.083, Marubo/Matsés 0.066,
  Karajá/Tapirapé 0.063) were left.
- **Migrant languages**: Warao (4,329, Venezuelan refugees counted as indigenous), Aymara,
  Quechua, Quíchua, Mapuche, Wayuu are in the table because their speakers declared themselves
  indigenous; drawn as measured.

## Files

`sources/br_censo.py`, `taxonomy/br2022.py`, `taxonomy/tree.d/br.txt`, `countries/br.py`,
`data/raw/br/` (4 files, 24 MB), `data/normalized/br.csv`, `br_status.csv`,
`data/processed/br_setor_indigenous.csv`, `dots_br.geojson`, `rings_br.geojson`.

## Immigrant languages (2026-10-06, session 5d7dac7e-br)

`python sources/br_immig.py` -> `data/normalized/br_immig.csv`; `countries/br.py` `_immigrants`
turns it into languages with `sources/latam_immig.py` (origin_mix home mixes, France's TeO2
retention, out of the `derived` Portuguese remainder), as ar cl co ve.

- **Count**: Censo 2022, SIDRA 10157 at N6, naturalizados (216,334) + estrangeiros (792,996)
  per município = 1,009,330. The municípios sum 7 and 4 people under the N1 figures (IBGE's
  small-cell perturbation); tolerated at 0.01%. The census has not published country of birth
  or nationality per município.
- **Mix**: SISMIGRA-ATIVOS (Polícia Federal register via OBMigra/UnB; Anita downloaded it
  2026-10-06; copied unmodified to `data/raw/br/sismigra_ativos.zip`). One row per registration:
  UF, nationality, legal basis, class, sex, status. **No município column**, so the mix is per
  UF, applied to each of its municípios. Only `situacao == "Ativo"` (2,080,214 of 3,879,289;
  "Prazo vencido" 1.23M, "Cancelado" 500k are mostly people who left, naturalised or died);
  2,064,609 with a UF and a nationality give the shares. 195 nationalities, all mapped to ISO
  (`ISO` dict; CONGO = Brazzaville since the DRC is listed apart; SÉRVIA E MONTENEGRO -> RS,
  IUGOSLÁVIA -> YU, UNIÃO SOVIÉTICA -> SU).
- **Call**: naturalised Brazilians take the same UF mix as active foreign registrations. They
  are older and more Portuguese, Japanese, Italian and Lebanese than that mix, so those
  languages are somewhat under-drawn and Venezuelan Spanish over-drawn. No source splits them.
- **Result**: estimate 692,162; 1,910 already counted as indigenous speakers (Quechua, Aymara
  among Bolivians and Peruvians who declared indigenous; `DEDUPE`); 690,252 added: Spanish
  471,957, Haitian Creole 46,593, English 18,289, Japanese 17,730, Mandarin 14,411, Paraguayan
  Guarani 13,215. Venezuelans are Spanish (uncited override, origin_mix.md §2b).
- Placement: immigrant dots go on the município's non-indigenous weight, like Portuguese.

## Settled communities (2026-10-06, session 5d7dac7e-br)

`python sources/br_settlers.py` -> `data/normalized/br_settlers.csv`; rows `modelled` (ask 019
estimate route), taken out of the Portuguese remainder. The docstring has every source and
rule. Drawn: Hunsrik 1,173,256, Talian 585,919, Pomerano 120,000.

| language | figure | region | source |
|---|---:|---|---|
| Hunsrik | 784,000 | Rio Grande do Sul | Altenhofen, Morello et al. 2018, *Hunsrückisch: inventário de uma língua do Brasil*, p.120: 80% of RS's ~980,000 German speakers (BIRS conscript survey 1988-90 x 1991 population) |
| Hunsrik | 294,000 | Santa Catarina | same, "an index near 30%" of RS |
| Hunsrik | 98,000 | Sudoeste Paranaense (IBGE meso 4107) | same, 10% |
| Talian | 587,411 | Rio Grande do Sul | same book, Table 6 (p.117): Italian speakers in RS, BIRS 1990, 6.43% |
| Pomerano | 120,000 | Espírito Santo | IPOL 2014 ("cerca de 120 mil dos estimados 300 mil descendentes"), the figure pt.wikipedia's Pomeranos gives as speakers |

Not drawn, no usable figure: the inventory's 49,000 Hunsrik "in other regions" (mixes Brazil,
Argentina, Paraguay); the other 20% of RS German speakers (Pomerano around Pelotas and
São Lourenço do Sul, Westphalian, standard German); Pomerano in RS, RO (Espigão d'Oeste: 15,000
descendants, a local newspaper via IPOL), MG, SC (only descendant counts); Talian outside RS
(IPHAN's INDL page: "Não existe estimativa"; the media's 500,000 in 133 towns is uncited);
Japanese among Brazilian-born Nikkei (only generation-retention studies of single colonies);
Plautdietsch (Witmarsum PR, Colônia Nova RS: a few thousand, no figure found). Old censuses
were checked as placement sources: 1940 and 1950 asked everyone's home language but tabulate
it only by state (1940 RS Tomo 1 Table 12, 1950 SC Table 14); no municipal table.

Placement inside each region (`place()`), the region's total fixed:
- Hunsrik and Pomerano by 2010 Lutherans per município (SIDRA 2094, from religiondots'
  normalized br.csv, read-only; 999,480 nationally; RS 449,768) as a share of 2010 population x
  2022 population. Hunsrik co-official towns under 30,000 people take their whole population
  as weight (the Catholic Hunsrik towns: São João do Oeste, Santa Maria do Herval, Harmonia,
  Barão, Antônio Carlos, Ipumirim, Itá, Horizontina); Ijuí (83,000) keeps its Lutheran weight,
  since a city's co-official law says less about its share (uncapped, it drew 70% Hunsrik).
  The Pelotas microrregião (Canguçu, São Lourenço do Sul, Pelotas, Arroio do Padre...) is
  kept off Hunsrik: its Lutherans are Pomeranians. Pomerano in ES takes Lutherans only (no
  co-official boost: Pomeranians there are Lutheran; the boost had put 61% in four towns).
- Hunsrik in southwestern Paraná by population (migrants from RS's Catholic colonies).
- Talian by population over the IBGE microrregiões holding an RS município with Talian
  co-official (96 municípios, 1.6M people: 36% each, Caxias do Sul 168,508, Passo Fundo 74,971).
- Cap 70% per language per município (IPOL: 70% of Santa Maria de Jetibá is of Pomeranian
  origin, the highest share any source gives), overflow re-spread; all settler languages
  together held to 90% (Agudo, Barão, Brochier, Harmonia, Linha Nova).
- Results: Hunsrik tops Porto Alegre 41,338 (3%), Novo Hamburgo 31,617 (14%), Santa Cruz do
  Sul 28,956 (22%), Teutônia 62%; SC Blumenau 29,298 (8%), Pomerode 43%; Pomerano Santa Maria
  de Jetibá 29,145 (70%), Domingos Martins 42%.

Calls someone might reverse: the Lutheran proxy (misses Catholic German areas without a law,
and Blumenau/Pomerode's Lutherans speak other varieties, per the inventory's footnote 83);
Talian's flat 36% across the Serra and Passo Fundo; using ~1990 BIRS-based figures for 2022;
Pomerano descendants read as speakers. Room for improvement: the IPOL Pomerano inventory
(ILP, fieldwork 2019) and Santa Maria de Jetibá's 2009/2012 linguistic census, if published;
BIRS per-município tables (Koch 1996), not online.

Scatter 2026-10-06: 202,953 dots, 783 rings; 125,261 people under one dot per language.
New files: `sources/br_immig.py`, `sources/br_settlers.py`, `data/raw/br/sismigra_ativos.zip`,
`sidra_10157_n6.json`, `sidra_10157_n1.json`, `ibge_municipios.json`,
`hunsrik_inventario_2018.pdf`, `data/normalized/br_immig.csv`, `br_settlers.csv`.
