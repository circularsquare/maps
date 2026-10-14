# France (fr): the record

Drawn 2026-10-05 by session edd42a8c-fr. 68,350,798 people, 101 départements (96 metropolitan,
Corsica included, and the five overseas), 135 languages, 68,292 dots. Every row is `derived`.

Build: `python sources/fr_insee.py --fetch` (INSEE tables into `data/raw/fr/`; the 73 MB
population zip resets urllib's connection, so if it fails fetch it with
`curl -C - --retry 5` to the same name), `python sources/fr_geo.py` (placement layer),
`python sources/fr_insee.py` (runs `sources/fr_build.py`, writes `data/normalized/fr.csv`).
Survey figures and their citations: `sources/fr_regional.py`. Mapping `taxonomy/fr2023.py`,
nodes `taxonomy/tree.d/fr.txt`, entry `countries/fr.py`.

## 1. The rule

France's census asks no language and never has. Anita's 2026-10-05 ruling for rich countries
with no language question (AGENT_BRIEF §2): French, plus (a) regional languages from regional
surveys, (b) immigrant languages by country of birth or nationality, all proxy rows derived.

## 2. Immigrants: INSEE RP 2023 by country of birth

INSEE's open Melodi API (no key; catalogue `https://api.insee.fr/melodi/catalog/all`):
- `DS_RP_TD_IMMI_AGESEX_PAYSNAISS_D_PRINC` (ex IMG1B, detailed): 41 named countries plus six
  remainders (other EU, other Europe, other Africa, other Asia, other Americas, Oceania), only
  for areas of 500,000+: 52 départements, 13 régions, France.
- `DS_RP_TD_IMMI_AGESEX_PAYSNAISS_R_PRINC` (grouped): 7 countries (Algeria, Morocco, Tunisia,
  Portugal, Italy, Spain, Turkey) and four groups, all 100 départements but Mayotte.
- `DS_RP_TD_POPULATION_AGESEX_PRINC`: population by single year of age, département and
  commune. RP 2023 = the 2021-2025 annual surveys.

INSEE's *immigré* is born a foreigner abroad, so French citizens born in Algeria (pieds-noirs)
are not in it. **Trap:** four country codes (24 Angola, 56 Belgium, 76 Brazil, 332 Haiti) are
also département codes, and the metadata file labels them with the département's name. Read
`AREA_COUNTRY` as ISO numeric, not through the label file.

**Remainders** (a group in a small département, "other Africa" in a large one; 1,553,340
people) are split into countries by Eurostat's 2021 census `cens_21ctz_r3` (foreign citizens by
NUTS 3, religiondots' raw copy, read-only): the département's citizens of that group's countries
give the shares, falling back to France's where the département has none.

Checks (printed, the first stops the build): the D table regrouped into the R groups matches
nationally to the person except other-EU / other-Europe, 13 apart (a country INSEE files
differently from Eurostat's continent order); 7,257,155 immigrants either way; every
département's countries sum to its total; RP 2023 départements sum to France (68,094,280).

**Country -> language** (`COUNTRY_LANG` in fr_build.py): the country's main first language.
Splits and calls:
- **Algeria**: 70% Arabic, 30% Kabyle. Salem Chaker (Inalco/CRB, "La langue berbère en
  France", 1997, and the Rhône-Alpes note) estimates 30-40% of the Algerian-origin population
  Berber-speaking, mostly Kabyle; 30% is his lower bound. A secondary citation puts INED's 1992
  MGIS survey at 28% (Journal of Amazigh Studies 1; 403, not opened). All Algerian Berber goes
  on Kabyle (Chaouia has no node; Chaker says the majority is Kabyle).
- **Morocco**: Morocco's own 2024 census mix of languages used, normalised (Darija 78.2%,
  Tachelhit 12.1%, Tamazight 6.3%, Tarifit 2.7%, Hassaniya 0.7%). Chaker puts Moroccan
  emigration at 40-50% Berber (the Souss and the Rif emigrated first); the census mix is the
  conservative choice.
- **French** for Belgium, Switzerland, Canada, Monaco (emigrants to France overwhelmingly
  francophone) and for Cameroon, Côte d'Ivoire and Gabon (no majority first language; French
  is the main home language of their cities).
- Mali -> Bambara, Mauritania -> Hassaniya, though emigration from both to France is largely
  Soninke (Kayes, the Senegal river). No figure to split by.
- Sri Lanka -> Tamil (the refugee migration). India -> Hindi. Pakistan -> Punjabi.
  Afghanistan -> Dari. China -> Chinese (the Sinitic group: the source names a country, and
  much of France's Chinese migration is Wenzhou Wu). Taiwan -> Mandarin.
- Congo and DR Congo -> Lingala. Dominica and Saint Lucia -> Antillean Creole (their Kwéyòl).
  Suriname -> Ndyuka (nearly all Suriname-born in France live in Guyane, mostly Maroons).
- Turkey -> Turkish; Kurds are not split out (no figure).

Their French-born children are drawn as French (said in note_public; TeO2 IMMFRA23-F19: 68% of
descendants with two immigrant parents were spoken to in both French and the parents' language).

## 2b. Retention: the share who speak only French at home (session edd42a8c-fr2)

Anita, 2026-10-05: many immigrant groups mostly speak French, so move that share onto French.
Table: TeO2 (Ined-Insee 2019-20), INSEE "Immigrés et descendants d'immigrés" édition 2023,
fiche 6793258, `IMMFRA23-F18.xlsx` figure 3 (copy in `data/raw/fr/`), metropolitan immigrants
18-59. Two columns: share whose family reference language is foreign (`ref`), and, among
parents with it, share who use it with their children (`kids`). French share = 1 - ref x kids,
i.e. speaking only French with the children. Applied to every language row of the origin
(Algeria's Arabic and Kabyle alike, Morocco's five languages alike, so the splits keep their
proportions); countries already on French are untouched. **TeO2 publishes regions, not
countries**: every country takes its region's share (`TEO2`, `_REGION_OF` in fr_build.py).

| TeO2 origin | ref | kids | French | countries |
|---|---|---|---|---|
| Spain, Italy | 99 | 63 | 37.6% | ES IT |
| Portugal | 99 | 65 | 35.7% | PT |
| other EU27 | 89 | 75 | 33.2% | rest of the EU |
| Europe (all) | 95 | 71 | 32.6% | non-EU Europe (TeO2 names no subgroup) |
| Algeria | 98 | 63 | 38.3% | DZ |
| Morocco, Tunisia | 98 | 59 | 42.2% | MA TN |
| Sahel | 97 | 54 | 47.6% | SN MR GM GW GN ML BF NE TD |
| Guinean or Central Africa | 84 | 33 | 72.3% | CI GH TG BJ NG CM CF GA CG CD GQ |
| Africa (all) | 95 | 55 | 47.8% | rest of Africa (Madagascar, Comoros, Mauritius, Horn, Egypt...) |
| Southeast Asia | 93 | 61 | 43.3% | KH LA VN TH MM MY SG ID PH BN TL |
| China | 99 | 67 | 33.7% | CN, TW |
| Turkey, Middle East | 98 | 76 | 25.5% | TR, Israel, Iran, the Arab Middle East |
| Asia (all) | 96 | 73 | 29.9% | South, East and Central Asia otherwise |
| Americas, Oceania | 91 | 85 | 22.6% | AME, OCE |

Moved onto French: **2,544,580** of 7,109,318 metropolitan immigrants (of whom some were already
French: Belgium, Switzerland...). Largest: Arabic 468,643, Moroccan Arabic 286,056, Portuguese
235,312, Lingala 131,311, Spanish 121,407, Italian 105,703, Kabyle 105,549. French 86.0% -> 89.7%.

Calls: children, not partner (TeO2 also prints partner: 64% overall), because "with the
children" is the household's home language and matches TeO 2008's own French-only column.
Mixed use (French and the language) stays on the language. The five overseas départements are
not adjusted (TeO is metropolitan; Guyane's Ndyuka check needs the Suriname-born whole).
The rule assumes immigrants without children behave like those with them.

Cross-check, TeO 2008 (Condon & Régnard, ch. 4 table 4 in Beauchemin, Hamel, Simon eds.,
*Trajectoires et origines*, Ined 2016, books.openedition.org/ined/816; image only), "français
seulement" with resident children: Algeria 38, Morocco-Tunisia 30, sub-Saharan Africa 55,
Southeast Asia 26, Turkey 10, Portugal 43, Spain-Italy 53, other EU27 35, all 36. TeO2 agrees
overall (40 vs 36) and for Algeria; it is higher for Morocco-Tunisia, Southeast Asia and Turkey
and lower for southern Europe. TeO2 is used: newer, and splits Sahel from Central Africa.

## 3. Regional and overseas languages (`sources/fr_regional.py`)

The measure: first or childhood language where the survey has it, a "both" answer counting
half; where it publishes only ability, speakers x its own share who learned it from their
parents. Each share is applied to the département's population at the survey's ages (RP
2023); children below that age get the survey's youngest band as a ratio where published,
otherwise French. All are subtracted from French; immigrants are not touched.

| language | survey | measure | drawn |
|---|---|---|---|
| Basque | EEP, VII Inkesta Soziolinguistikoa 2021, 16+ | first language by zone: BAB 6.75%, Lapurdi 17.15%, Behe Nafarroa-Zuberoa 43.65% | 52,234 |
| Catalan | Generalitat, EULCN 2015, 15+ | llengua inicial 9.2% + 3.5%/2 | 46,065 |
| Corsican | Collectivité de Corse 2021, 18+ | home language to age 6: 15.6% + 40.2%/2 + 0.9%/3; under-18s 2% (2012, 18-24) | 106,207 |
| Breton | Région Bretagne / TMO 2024, 15+ | speakers by département x 44% learned mainly with parents | 52,585 |
| Gallo | same | speakers x 51%, Côtes-d'Armor and Ille-et-Vilaine only | 44,846 |
| Alsatian | OLCA / EDinstitut 2012, 18+; CeA 2022 age gradient | Bas-Rhin 46%, Haut-Rhin 38% x 89% from parents | 619,412 |
| Lorraine Franconian | DRAC Grand Est / TMO 2024, 18+ | 190,000 x 73% in Moselle x 49% from parents | 67,963 |
| Occitan | OPLO 2020; Midi-Pyrénées 2010 | ~600,000 speakers x 71% learned in the family | 426,000 |

Numbers worth knowing:
- Basque: RP 2023 gives the Pays Basque 279,783 people 16+ against the survey's 255,940 (an
  older base). The zones are the Communauté d'agglomération du Pays Basque's 158 communes
  (INSEE EPCI file 2023), Labourd's 41 by name (fr.wikipedia list), BAB = Bayonne, Anglet,
  Biarritz. Basque dots go on the zones weighted by their share.
- Breton: Loire-Atlantique is printed "<1%", taken as 0.5%. The survey's total (107,000
  speakers) and its pays figures (Morlaix 11%, Trégor 10%) are in the report; placement inside
  a département is by population, so western Côtes-d'Armor and Morbihan are not favoured.
- Occitan: the survey prints 7% and "nearly 600,000"; the 25 départements' RP 2023 15+ is
  10.38M, so 7% would be 727,000. The count anchors it. Printed: Haute-Garonne, Gironde, Hérault
  2%, Lozère 22%. **Assumed** at 2%: Charente-Maritime, Deux-Sèvres, Vienne (langue d'oïl) and
  Pyrénées-Orientales (Catalan); the other 17 share the rest evenly at 9.65%. In
  Pyrénées-Atlantiques Occitan is placed outside the Pays Basque (Béarn). Provence, Auvergne,
  the Drôme and Ardèche: no survey, French.
- Alsatian: 619,412 is high beside EHF 1999's 548,000 adult speakers, but it is the formula on
  the survey's numbers: the 2012 figures are 18+ ability, and 89% of speakers learned it from
  parents. Children get 9/46 of the adult share (CeA 2022, 18-24).
- Not drawn: West Flemish (no survey; uncited estimates of 20,000), Franco-Provençal, Picard,
  Norman and the other langues d'oïl. The 1999 EHF (INED, Population & Sociétés 376) is the
  only source on them and gives no regional table.
- Sources the research found walled (403): the 2018 Brittany report, OPLO 2020's full results,
  the CeA 2022 Alsace report (connection refused). The last two may have département figures.

**Overseas.**
- Guadeloupe, Martinique, Réunion: INED/INSEE Migrations, Famille et Vieillissement 2009-10,
  native-born parents 18-79 (erudit.org summary): childhood language Creole only / both,
  Guadeloupe 34.2/47.7, Martinique 23.5/57.2, Réunion 79.7/17.6; under-15s at what parents
  passed on (Antilles 9/47, Réunion 51.2/34.3). Applied to the non-immigrant population (RP
  2023 minus INSEE's immigrants, under-15s from its Y_LT15 band). Antillean Creole 50.6% of
  Guadeloupe, 47.6% of Martinique; Réunion Creole 81.8%.
- Guyane: INSEE Pratiques culturelles 2019-20, languages used daily (coastal Guyane): Guianese
  Creole 20%, Maroon languages 8%, applied to the whole population. The Suriname-born already
  drawn as Ndyuka (28,062) exceed 8% (23,520), so nothing is added. Amerindian languages and
  Hmong have no figure and are French.
- Mayotte: no RP 2023 table. Census 2017 (INSEE 3713016): 256,518; 36% born abroad (95%
  Comorian, 4% Malagasy by nationality), 6% born in France; the rest split 82:33 Shimaore to
  Kibushi by INSEE 2019's ability shares among natives (ability only; no first-language source
  found; a circulating "2006 mother tongue" split had no traceable source). Population vintage
  2017, against 2023 elsewhere.

## 4. Geography

`data/geo/fr/fr_place.gpkg` (`sources/fr_geo.py`): religiondots' `fr_lau.gpkg` read-only (GISCO
LAU 2021 communes with INSEE RP 2021 TD_NAT1 French and foreign nationals; Kontur hexes for the
DOM) plus Corsica's 360 communes, which religiondots dropped, from the same GISCO shapefile and
TD_NAT1 (all 360 found). 101 units, unit <-> NUTS 3 asserted one to one. Placement (AGENT_BRIEF
§4.4): French on French nationals, immigrant languages on foreign nationals, regional languages
on population, Basque on its zones; overseas hexes carry no nationality, so population. The
counting unit is the département (680,000 people on average); the communes only place.

## 5. Checks and result

After the retention step (2b): check_country ok; scatter 68,292 dots on 20,105 polygons, 33
rings (all derived rows), 58,798 people (0.09%) under one dot per language. French is 89.7%
nationally (86.0% before retention).

## 6. Colours

New nodes take generated colours except Lorraine Franconian (hand-set light blue against
Alsatian's dark blue; they meet in Moselle and Alsace bossue) and Comorian (pushed red against
Shimaore's ochre in Mayotte). Occitan (teal) sits near Corsican and Gallo, which never meet it.

## 7. Room for improvement

- Département figures for Occitan and Alsatian 2022 (the walled reports).
- A Berber share for Algerian and Moroccan immigrants from a survey that asked (MGIS 1992,
  TeO); Kurdish among Turkish-born; Soninke among Malian- and Mauritanian-born.
- Retention by country rather than TeO2 region (TeO2 microdata at Quetelet-Progedo would give
  it for the large origins).

## 8. Minority-language audit (2026-10-13)

No counts changed. Concept checked: language spoken first or mainly at home.

Yardstick found: INSEE/INED Etude de l'histoire familiale 1999 (EHF), adults, "language your
parents usually spoke to you at about age 5" (Heran, Population & Societes 376; figures as
quoted in Bachoc's summary, files.eke.eus/pdf/topaketen_aktak/insee/
enquete_transmission_familiale_des_langues.pdf): usual / occasional / re-transmitted to own
children. Alsatian 660k / 240k / 410k; langue d'oc 610k / 1.06M / 250k; langues d'oil 570k /
850k / 300k; Breton 280k / 400k / 100k; hexagonal Creole 110k / 120k / 90k; Catalan 80k / 100k /
50k; Basque 63k / 13k / 34.5k. This is a 1999 stock of mostly older adults, so today's drawn
figures should sit below it; they do, except Alsatian (below).

Verdicts:
- Alsatian 619k: roughly right but the top of the plausible range. 1999's usual-childhood-language
  stock was 660k; the drawn figure rests on a 2012 ability survey x 89% from parents, with no
  further decay to 2026. Kept; the 2022 CeA report (walled) would settle it.
- Occitan 426k: right level (EHF 610k usual in 1999, 250k re-transmitted). Kept.
- Breton 52.6k, Basque 52.2k, Catalan 46.1k, Corsican 106k, Lorraine Franconian 68k, Gallo 45k:
  consistent with EHF and with the newer surveys already used. Kept.
- Creoles in the DOM: from INED/INSEE MFV 2009-10 childhood language, a direct first-language
  measure. Kept.
- Not drawn, no survey: West Flemish (Nord, uncited ~20k), Franco-Provencal (Savoie, Lyon area),
  Yenish/Romani/Sinte, Bressan, Walloon in the Ardennes. Arbereshe, Piedmont Occitan and Alsatian
  left as they are (Anita's call, via the coordinator, 2026-10-13).

### 8a. Added: langues d'oil other than Gallo (modelled; code in `fr_build.extras()`, constants in
`fr_regional.py`)

EHF 1999 (see above) has 570k adults whose parents usually spoke a langue d'oil to them at about
five, Gallo included. Our estimate, not a survey figure:
1. Scale by Breton's decline between EHF 1999 and today's drawn Breton: 52,584 / 280,000 = 0.188,
   giving 107,046 for all langues d'oil. Why Breton: the nearest case with both a 1999 and a
   recent figure, a rural language that lost transmission in the same cohorts. Alsatian (0.94) or
   Occitan (0.70) would give five times more; Breton is the low end and so the cautious choice.
2. Take out Gallo's drawn count (44,846), leaving 62,200.
3. Share that out among the seven traditional areas by rural population (RP 2021 communes under
   2,000 inhabitants in the area's departements; "our call", flagged in note_public). Areas under
   5,000 stay French.
4. Within an area, by each departement's rural population; inside a departement, on population.

| area | departements | drawn |
|---|---|---|
| Picard | 59, 62, 80, 02, 60 | 16,429 |
| Norman | 14, 27, 50, 61, 76 | 12,493 |
| Poitevin-Saintongeais | 85, 79, 86, 16, 17 | 8,738 |
| Burgundian (Morvandiau) | 21, 58, 71, 89 | 7,791 |
| Franc-Comtois | 25, 39, 70, 90 | 5,969 |
| Champenois | 08, 10, 51, 52 | 5,868 |
| Lorrain Romance | 54, 55, 88 | 4,912, below the 5,000 cut, not drawn |

57,288 people in all, taken from French in the same departement. Angevin, Mayennais, Berrichon
and the other oil varieties of the Loire valley have no area here and stay French. New tree nodes
(tree.d/fr.txt, taxonomy/fr2023.py): picard, norman, poitevin_saintongeais, champenois, burgundian,
franc_comtois, all under indoeuropean.romance beside Gallo. Gallo is regrouped under French in
taxonomy/regroup.txt; these six are not (shared file, not touched): the orchestrator may want
them there too. Part names in fr.csv: "langues d'oil, modelled". Tier stays `derived`, as every
row of this country.

### 8b. Added: Creole of people born overseas and living in the mainland

Not INSEE immigrants, so they were drawn as French. INSEE Premiere 1389 (census 2008,
insee.fr/fr/statistiques/1281122): born in Martinique 117,000, Guadeloupe 115,400, Reunion
108,000, Guyane 24,400, living in the mainland. Two thirds of the Antilleans live in
Ile-de-France, 28% of the Reunionese, about half of the Guyanese. Each group x the DOM's own
childhood-language share from MFV 2009-10 (sections 3, Creole alone plus half of both:
Guadeloupe 58.1%, Martinique 52.1%, Reunion 88.5%; Guyane 20%): Antillean Creole 66,990 +
60,957, Reunion Creole 95,580, Guianese Creole 4,880; 228,407 in all. Split between Ile-de-France
and the rest by the groups' printed shares, then by departement population.
Calls: first generation only (no source gives the second generation's Creole; their French-born
children are drawn as French, as for immigrants). No discount for language loss after moving
(no source; the DOM itself uses the same childhood-language share). 2008 is the newest
birthplace count found, so this is a floor; INSEE's RP 2023 tables found in Melodi carry no
place-of-birth by departement. EHF 1999 counted 110k adults with a Creole parent language in the
hexagon, so 228k for a larger 2023 group is of the right order. Placement is by population, not
by where the Antilleans live inside Ile-de-France.
