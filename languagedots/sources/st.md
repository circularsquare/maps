# São Tomé and Príncipe: RGPH 2012, languages spoken, district

**Drawn** 2026-10-05 (agent edd42a8c-st). 173,015 people (aged 1+) on 7 districts, 6 nodes, 170
dots at 1:1000, 0 rings. Every row `derived` (a multi-answer question, spec §3.6).

## Source

- INE São Tomé e Príncipe, IV Recenseamento Geral da População e da Habitação 2012, Resultados
  Distritais (2013): **Quadro 10, "População residente segundo sexo e grupos etários por língua
  falada"**, in each of the seven district reports and the national one. For each of eight
  languages it prints Total / Sim / Não by age and sex (and urban/rural in some reports):
  Português, Fôrro, Angolar, Lunguié, Cabo verdiano, Francês, Inglês, `Outra(s) língua(s)
  (inclusive sinais)`. Universe: residents aged 1 and over (173,015 of 178,739).
- The PDFs are the same eight religiondots fetched for religion (Quadro 8). Route:
  `www.ine.st/phocadownload/userupload/Documentos/Recenseamentos/2012/Dados Distritais e Nacional
  Recenseamento 2012/`, an open LiteSpeed autoindex (`../religiondots/sources/st.py` describes
  it). `sources/st_rgph.py` reads `data/raw/st/` if `--fetch` has been run, otherwise
  religiondots' copies read-only. Nothing was downloaded for this build.
- The coverage sweep had "national (district split unclear)" and Kabuverdianu 7.9%; the district
  reports settle the grain, and Kabuverdianu is 14,654 of 173,015 = 8.5% (the 7.9% presumably
  used all ages or another denominator).
- **2024 has no language table.** The V RGPH 2024 folder (`.../Recenseamentos/2024/`, listed
  2026-10-05) holds the results report, the leaflet, an older copy of the report and two videos;
  neither PDF mentions Forro, Angolar or any language question. 2012 is the latest.
- No finer table: the 2016 locality publication (`localidades2012.pdf` in religiondots' raw
  folder) has no table titled by language (its text searched for "língua"/"idioma" in table and
  quadro titles; none). Its hits on "Angolar" are place names.

## Checks (`sources/st_rgph.py`, asserted, all pass)

1. Every language's Total is the same figure in each report, and Sim + Não = Total for every
   language, pinning the parse to the total row's cells.
2. The seven districts sum to the national Quadro 10 on all eight Sims and the 1+ total (173,015).
3. Each district's 1+ population is 2.9-3.6% below its all-ages Quadro 8 figure (religiondots'
   parse of the same reports): the under-ones.
4. National shares reproduce the secondary figures the coverage row cited: Portuguese 98.4%,
   Forro 36.2%, Angolar 6.6%, Lung'ie 1.0%.

National mentions: Português 170,223, Fôrro 62,707, Cabo verdiano 14,654, Francês 11,697,
Angolar 11,377, Inglês 8,556, Outras 4,184, Lunguié 1,753.

## Drawing

Spec §3.6: within each district every label's mentions are scaled by population / sum(mentions)
(k = 0.57-0.63), so each person is shared across what they named. Drawn (people):

| district | Portuguese | Forro | Kabuverdianu | Angolar | Lung'ie | other |
|---|---|---|---|---|---|---|
| Água-Grande | 47,283 | 14,727 | 2,165 | 1,444 | 466 | 1,211 |
| Mé-Zóchi | 29,387 | 10,943 | 960 | 1,151 | 134 | 765 |
| Lobata | 11,723 | 4,524 | 1,979 | 249 | 52 | 142 |
| Cantagalo | 10,018 | 4,136 | 1,331 | 977 | 41 | 120 |
| Lembá | 9,090 | 2,712 | 912 | 1,316 | 37 | 81 |
| Caué | 3,443 | 459 | 156 | 1,609 | 10 | 179 |
| Príncipe | 4,666 | 552 | 1,386 | 111 | 331 | 37 |

**Folded into Portuguese** (AGENT_BRIEF §2, learned second languages): French, 11,697 mentions,
7,058 people's worth after scaling; English, 8,556 mentions, 5,157 after scaling. Both are school
languages here. Their age profile says so: in Água-Grande, French is 20 of 8,304 children aged 1-4 and 929 of 7,004 aged
15-19. Neither is a first language for any sizeable group in the country, so no flag.

Placement: religiondots' Kontur 400 m hexes for the same seven COD-AB districts
(`RD_GEO/st/st_hexes.gpkg`, unit = pcode ST11..ST26, the same ids `sources/st_rgph.py` writes),
plain population weight. Nothing published places Angolar or Lung'ie speakers inside a district,
so Angolar in Caué is spread over the whole district rather than the São João dos Angolares coast.

## Calls

- **Kabuverdianu shared in, not folded.** It is a first language of the Cape Verdean community
  brought to the roças as contract labour; 31% of Príncipe named it and its age profile is flat
  (no school pattern).
- **"Outra(s) língua(s) (inclusive sinais)" on `other`, not folded.** It mixes sign languages,
  immigrant languages and possibly learned ones, with no split. Folding it like French was the
  alternative (El Salvador folded its `Otro`), but in Caué it is 308 people, 5.3%, including 44 of
  744 children aged 1-4, which no school language explains; it behaves like a home language there
  (plausibly the African languages of the plantation workers' descendants, unverified). `other` is
  the narrowest node holding a sign language and an unnamed spoken one (spec §3.2). Cost of
  reversing: 2,535 people, about 2 dots, move to Portuguese.
- **Under-ones not drawn** (5,724, 3.2%, in `gap`). The table excludes them; drawing them as
  Portuguese (as El Salvador drew its under-3s as Spanish) would be a count proxy.
- **Lung'ie is drawn mostly on São Tomé**, as the census has it: 1,228 of 1,753 people who named
  it live on São Tomé island, 783 in Água-Grande; only 525 on Príncipe.
- Node names: `Forro (Sãotomense)`, `Lung'ie (Principense)`, `Angolar`; three new leaves under
  `creole.portuguese_based` beside `kabuverdianu` (us.txt). Glottolog saot1239, prin1242, ango1258
  (filed there under Indo-European, as all lexifier creoles).
- Colours (tree.d/st.txt): Forro 0.56 0.11 190 (darker blue-green, apart from Portuguese's blue
  and Kabuverdianu's light green), Angolar 0.74 0.13 80 (ochre), Lung'ie 0.66 0.16 35
  (orange-red). Not checked on screen.

## Files

`sources/st_rgph.py`, `data/normalized/st.csv`, `taxonomy/st2012.py`, `taxonomy/tree.d/st.txt`,
`countries/st.py`, `dots_st.geojson` / `rings_st.geojson` (scatter output), this file.

## Moved from countries/st.py text (2026-10-06 sweep)

Cut from the reader-facing text and not recorded above; verbatim from the old `note_public`.

- Kabuverdianu came with Cape Verdean plantation workers and is spoken most of all on Príncipe, where 31% named it, more than named Lung'ie (7.4%); most of the 1,753 people who named Lung'ie live on São Tomé, 783 of them in the capital district.
