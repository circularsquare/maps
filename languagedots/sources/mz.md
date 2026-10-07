# Mozambique: IV RGPH 2017, mother tongue, province by urban and rural

**Drawn** 2026-10-05 (`d9e44929-mz`). 21,566,224 people aged 5 and over on 11 provinces, each
split into towns and countryside (21 units; Maputo Cidade is all urban), 25 nodes, 21,555 dots
at 1:1000, 0 rings. Files: `sources/mz_rgph.py`, `sources/mz_geo.py`, `taxonomy/mz2017.py`,
`taxonomy/tree.d/mz.txt`, `countries/mz.py`.

## Source

- INE Moçambique, IV Recenseamento Geral da População e Habitação 2017, definitive results
  (2019): **Quadro 22**, *População de 5 anos e mais por idade, segundo área de residência, sexo e
  língua materna*, one xlsx per province and one national. Quadro 23 (*língua que fala com mais
  frequência em casa*, the language spoken most at home) is read as a witness only.
- The question is mother tongue, one answer per person, asked of everyone aged 5 and over
  (22,243,373 of 26,899,105).
- INE's old Plone site (`www.ine.gov.mz/iv-rgph-2017/<province>/`) is gone; the files are read
  from the Wayback Machine's 2019-11-14 captures with the `id_` flag, as religiondots'
  `sources/mz.py` reads Quadro 11 (its `sources/mz.md` §1 has the history of the live site:
  timeouts, Liferay APIs answering 403). The CDX query that lists them:
  `http://web.archive.org/cdx/search/cdx?url=ine.gov.mz/iv-rgph-2017/*&output=txt&limit=5000&fl=original,statuscode,mimetype,timestamp,length&collapse=urlkey&filter=original:.*lingua.*`
  All eleven provinces' Quadro 22 are captured; Quadro 23 for all but Zambézia.
- **Fetching:** on 2026-10-05 the Wayback Machine answered Python `requests` with 429 for about an
  hour while `curl` and `urllib` from the same machine got 200. `fetch()` uses urllib.

## Nothing finer than province exists

- 2017: INE's list of tables has only Quadros 3, 6, 7, 8 by district (religiondots `sources/mz.md`
  §1). The microdata on mozdata.ine.gov.mz is licensed (gated, not tried); IPUMS is not available
  to us.
- 2007: the *Indicadores Sócio-Demográficos Distritais* volumes religiondots used for religion
  have no language table (searched Nampula's, 31 pages: "língua" appears only in literacy).
- 1997: INE's 2001 site had `censo2/<nn>/brochura/<nn>linguas.htm`, province pages (Wayback CDX);
  not read, and province grain anyway.
- So the grain is the province, and the urban/rural split is the one finer thing the tables give.

## What the tables print

Each province names its own five to seven languages and puts the rest under `Outras línguas
moçambicanas`; the national table names nine. Labels per province (canonical spelling):

| province | named |
|---|---|
| Niassa | Português, Emakhuwa, Ciyao, Cinyanja (printed CINHANJA), Elomwe, Xichangana |
| Cabo Delgado | Português, Emakhuwa, Shimakonde, Kimwani, Ciyao, Kiswahili (printed KISWALHILI) |
| Nampula | Português, Emakhuwa, Coti, Elomwe, Shimakonde, Echuwabo |
| Zambézia | Português, Elomwe, Echuwabo, Cisena, Lolo/Malolo, Emakhuwa |
| Tete | Português, Cinyanja, Cinyungwe, Cisena, Cishona |
| Manica | Português, Cindau, Chitewe, Cimanika, Cisena, Cinyungwe, Chibalke |
| Sofala | Português, Cisena, Cindau, Echuwabo, Xitswa, Emakhuwa |
| Inhambane, Gaza, Maputo Província, Maputo Cidade | Português, Xichangana, Xirhonga, Cicopi/Chichopi, Xitshwa, Bitonga |

Plus `Outras línguas estrangeiras`, `Mudo(s)` and `Desconhecida` everywhere. Maputo Cidade's
table has no residence split (the census counts the city as urban).

## Two mislabelled captures, and which file is read

Inhambane and Maputo Cidade each have two captures of Quadro 22, the plain name and a `-1`
re-upload. **The plain files print the right numbers against labels shifted one row**: in
Inhambane's, `PORTUGUÊS` 26,566, `BITONGA` 703,856, `CHICHOPI` 758; in the `-1` file the same
numbers sit as Outras línguas moçambicanas 26,566, Xitshwa 703,856, Xirhonga 758, Português
157,894, Cicopi 181,071, Bitonga 151,871. Maputo Cidade's plain file has Portuguese at 41,473; the
`-1` file 602,939 (62%). The `-1` files are read. Evidence: with them, the eleven provinces'
Portuguese sums to the national table's 3,686,890 to the person, which the plain files miss by
hundreds of thousands; and the `-1` Inhambane (Xitswa 56%, Cicopi 14%, Gitonga 12%) is the
province's known pattern, where the plain one made Gitonga, a language of Inhambane town and
Maxixe, 56%.

## Checks (all in `sources/mz_rgph.py`, all pass)

1. Every block's categories sum to its printed universe; Homens + Mulheres equals the block's
   all-sex row, category by category; Urbana + Rural equals Total, category by category (every
   Quadro 22).
2. The provinces sum to the national table: 22,243,373 aged 5+ to the person; Português and Mudo
   to the person.
3. **The national table is a different edit in three cells, pinned (`NATIONAL_DIFF`)**, national
   minus provinces: foreign +26,261, which is exactly Cabo Delgado's Kiswahili (the national
   table counts Swahili as foreign); unknown −265,049; Mozambican languages +238,788. So the
   national table gave a language to about 265,000 people the provincial tables leave unknown,
   most of them presumably among Cabo Delgado's 290,465 unknowns. Nobody publishes that edit by
   province; the provinces are drawn and a re-issued file fails the pin.
4. For each language the national table names, the provinces that name it sum to no more than
   the national figure (the rest is in other provinces' remainders). Largest remainder:
   Cinyanja, 598,358 of 1,790,831 not named by any province, most likely Zambézia's (Milange,
   Morrumbala on the Malawi border) and Manica's `Outras` (425,341 and 173,370).
5. Quadro 23 (home language) has the same province totals as Quadro 22 in all ten provinces
   captured. Sofala's Quadro 23 urban block has a men + women slip, so Quadro 23's sex check is
   off; it is a witness only.

Mother tongue against home language, printed by the script: within a point or two everywhere
except the south's towns, where Portuguese gains at home (Maputo Província 49.3% to 54.2%,
Maputo Cidade 62.3% to 68.3%, Sofala 21.7% to 24.9%) and Xitswa and the unnamed remainder lose.

## Mapping calls (`taxonomy/mz2017.py` has the full reasoning)

- Every printed label is its own node. Spelling variants merged in `SPELLINGS`: CINHANJA and
  CINYANJA; ELOMWE and ELOMWUE; XITSHWA and XITSWA; CHICHOPI and CICOPI/CHICHOPI; MUDO and
  MUDOS; OUTRAS, OURAS and Outas.
- Xichangana on za.txt's Xitsonga node (the same language, Mozambique's name for it).
- Cinyanja on zm.txt's Nyanja, though in Tete it is the speech Malawi calls Chewa.
- `Outras línguas moçambicanas` (947,455) on `nigercongo.bantu`, washed out as "Bantu, language
  not named": every indigenous language of Mozambique is Bantu, so that is the narrowest node.
  Angola put its `Outras línguas` on `africa_other`, but sign language was inside Angola's.
- Foreign (86,124) on `other`. Not drawn: Desconhecida (672,976, 3.0%) and Mudo (4,173).

New nodes (`taxonomy/tree.d/mz.txt`, families checked against Glottolog `values.csv`): a Makhuwa
group (P.30: Emakhuwa, Elomwe, Echuwabo, Lolo, Koti; Glottolog's maku1247); Sena, Nyungwe and
Barwe under zm.txt's Nyanja-Sena (Glottolog puts Barwe in Senaic); Ndau, Tewe, Manyika from
Bantu beside the Shona leaf (bw.txt's Kalanga precedent); Makonde from Bantu; Tswa and Ronga
under za.txt's Tswa-Ronga; a Chopi group (S.60: Cicopi, Gitonga). Yao (pl.txt) and Mwani
(ca.txt, label "Mwani" kept after the coordinator flagged a label clash) were defined bare and
are coloured here.

Colours were hand-picked for neighbours on the ground (the fragment's header lists them); not
checked on a screenshot.

## Placement (`sources/mz_geo.py`)

religiondots' `data/geo/mz/mz_hexes.gpkg` (Kontur 400 m hexes on 2007 districts and two whole
provinces, read-only). The first four characters of each unit are the province code. Within a
province the hexes are ranked by Kontur density and the densest are urban until they hold the
census's urban share of aged 5+ (Russia's method, `sources/ru_geo.py`). Every province lands within
0.0003 of its census share. The urban parts are small and dense: 120 to 714 hexes per province,
the sparsest urban hex between 538/km² (Sofala) and 3,650/km² (Cabo Delgado). INE's urban line is
administrative (cities and vilas, whose limits take in farmland), so this is a stand-in for it.

Why the split is worth it: Portuguese is 38.3% of urban Mozambicans aged 5+ and 5.1% of rural
ones; without it Portuguese would be spread evenly over every village.

## Calls someone might reverse

- The `-1` captures for Inhambane and Maputo Cidade (evidence above; strong).
- Drawing the provincial tables, not the national edit, which leaves 265,000 more people unknown.
- The other-Mozambican remainder on Bantu rather than `africa_other`.
- The density-ranked urban/rural cut.

## What would reopen it

- District tables: INE's custom tabulation offer or the licensed microdata (both gated).
- A Cabo Delgado volume or report explaining its 15.6% unknown mother tongue.

## Cross-border groups (2026-10-06, 5d7dac7e-xb)

Cindau, Cimanika and Chitewe now sit in a Shona group with Zimbabwe's Shona (Glottolog's Core Shona); Elomwe in a Lomwe group with Malawi's Lomwe; Cisena in a Sena group with Malawi's Sena. Cinyanja stays on the Nyanja leaf, already in the Nyanja group beside Malawi's Chewa (one language; Mozambique prints the other name). Swati has no label here, so Eswatini's Swati stops at the border: its speakers are inside 'Outras línguas moçambicanas' and are not split out. Groups only (taxonomy/regroup.txt): no label's node or count changed, each keeps its leaf and colour; dots rewritten in place. The full table of cases is in followups.md (2026-10-06, languages that stop at a border).
