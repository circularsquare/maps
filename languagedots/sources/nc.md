# New Caledonia: 2019 census, Kanak languages spoken (15+), 33 communes

Drawn 2026-10-05 (session d9e44929-nc). `sources/nc_rp2019.py`, `taxonomy/nc2019.py`,
`taxonomy/tree.d/nc.txt`, `countries/nc.py`. No asks filed. Build tail left to the supervisor.

**210,981 people aged 15 and over, 31 nodes, 33 communes (6,400 on average), 197 dots and 15
single dots of their own weight.** French (derived) 64.0%, Drehu 7.5%, Kanak language not named
5.1%, Nengone 4.4%, Paicî 3.1%, Xârâcùù 2.7%, Ajië 2.1%, Iaai 1.8%, Cèmuhî 1.1%, Fagauvea 1.0%.
`check_country nc` ok; 13,981 people (6.6%) in languages under one dot nationally.

## 1. The tables

All from ISEE's page "Organisation coutumière kanak" (https://www.isee.nc/organisation-coutumiere-kanak)
and its census page (https://www.isee.nc/population/recensement), Recensement de la population
2019, INSEE-ISEE. Fetched by `sources/nc_rp2019.py --fetch` into `data/raw/nc/`:

- `langues-vernaculaires-locuteurs.xls` (updated 13/10/2021):
  - **'par commune-langue'**, "Nombre de locuteurs d'une langue kanak par commune de résidence",
    29 languages x 33 communes. Footnotes: "un locuteur peut parler une ou plusieurs langues";
    "certaines personnes ont déclaré parler une langue kanak sans préciser laquelle. Celles-ci ne
    sont donc pas réparties ici"; "le nombre de locuteurs distincts est de 75853". **The table
    drawn.**
  - 'par commune': distinct speakers aged 15+ per commune, 1996-2019.
  - 'langue vernaculaire': national mentions per language, 1996-2019 (title says 14+, the
    commune sheets 15+; the 2019 figures are identical, check 2).
- `rp2019-pop-logement-menages-communes.xls`: **P21** (15+: parle / comprend / aucune
  connaissance, by commune) and **P01** (population, all ages). P09 (community) used for the note.
- `langues-vernaculaires-connaissance.xls`: 'commune_2019' (15+: parle ou comprend / aucune).

The coverage sweep's lead (languages_spoken, commune, 28-29 languages) was right; the per-commune
language table was not in its notes. I did not read the 2019 questionnaire itself; the question's
shape is from the tables' headings (speaks / only understands / neither, then which).

ISEE also publishes the census as open microdata (`RP2019NC_OD_ind_*.xlsx`), but it has no
language variable and no commune, so it cannot give the combinations (checked, file deleted).

## 2. Checks (all in the normaliser, all pass)

1. 33 communes in every table, each name to one INSEE code (98801-98833; Kouaoua is 98833).
2. Every language's communes sum to its TOTAL, and the TOTALs equal the national 2019 column:
   65,185 mentions.
3. P21 "Parle" = 'par commune' 2019 in all 33 communes (two tables of the same census); they
   sum to 75,853, the footnote's distinct speakers.
4. 'connaissance' "Parle ou comprend" = P21 Parle + Comprend, and the 15+ totals agree, in all 33.
5. The table's 'Total Locuteurs' row is the **sum of mentions**, not persons (65,185 = the 29
   languages' total, in every commune). No commune has more mentions than speakers.
6. P01's population equals religiondots' `nc_units.gpkg` `pop` for every INSEE code: the
   name-to-code join checked against an independent copy of the census count. 271,407 people.

## 3. How it is drawn (countries/nc.py)

Per commune, within the 15+ population:

- **each language: its mentions, `measured`.** A mention is a census count of people who speak
  that language in the commune.
- **Parle minus all mentions on `austronesian.oceanic.kanak` ("Kanak languages, language not
  named"), `derived`.** 10,668 nationally. It is the speakers who named no language less any who
  named two (the overlap is not published), so it is a floor on the unnamed and the named
  languages are drawn at their full counts. The overlap looks small: in the rural communes
  where nearly everyone speaks one language the remainder is only 1-5% (Lifou 4.6%, Maré 2.5%,
  Bélep 1.8%, Hienghène 0.9%), and it cannot be negative. It is large in the towns (Païta 28%,
  Nouméa 22%, Dumbéa 20%), where people away from their home area more often did not say which.
  Spec 3.6's scaling would only bite where mentions exceed speakers, which happens nowhere.
- **Everyone else aged 15+ on French, `derived` (spec 3.5).** 135,128: 16,941 who only
  understand a Kanak language and 118,187 with no knowledge of one. The census asks about no
  other language.
- Children under 15 (60,426, 22.3%) are not asked and not drawn (`gap`), as for every other
  age-limited census here.

Speakers are NOT shared with French, though nearly all speak it: the question is "do you speak a
Kanak language", Mexico's shape (spec 3.5, indigenous languages drawn as measured), not El
Salvador's "besides Spanish".

**Second source on the remainder (spec 3.5, wanted).** None asks home language. The census's own
community table (P09) says who the French remainder hides: 22,520 people of Wallisian and
Futunian community (8.3%), 21,255 of them in Greater Nouméa, plus Tahitian, Indonesian,
Vietnamese and Ni-Vanuatu communities, and 30,758 of several communities. So French is
overcounted in Greater Nouméa by Wallisian and Futunian speakers above all; note_public says so.
It does not corroborate French as such; nothing found does.

## 4. Mapping and tree (taxonomy/nc2019.py, tree.d/nc.txt)

Every one of the 29 labels is a leaf (spec 3.1). Groups follow Glottolog (checked in
`data/raw/glottolog`): Loyalty Islands (loya1239: Drehu, Nengone, Iaai), Northern New Caledonian
(nort3325), Southern New Caledonian (sout3313), under an areal node **"Kanak languages"**
(`austronesian.oceanic.kanak`) that also holds Fagauvea (West Uvean, Polynesian, west2516). The
areal node exists to hold the unnamed remainder, as `austronesian.oceanic.vanuatu` does for
Vanuatu. Glottolog's "Mainland" level is skipped.

Label calls:
- "Paicï" (ISEE's typing) -> Paicî. "Yalâyu" -> "Yalâyu (Nyelâyu)" (Glottolog nyal1254 "Belep").
  "Fwa Kumak" -> "Fwa Kumak (Nêlêmwa-Nixumwak)" (kuma1276). "Faga uvea" -> "Fagauvea (West
  Uvean)".
- "Dialectes de Voh-Koné" -> one leaf "Voh-Koné dialects" (Glottolog's Voh-Kone group under
  Mid-Northern); "Dialectes de l'extrême sud" -> "Far South dialects (Numèè)" (nume1242, Extreme
  Southern; 931 of its 1,618 on the Île des Pins, i.e. largely Kwényï). Neither is split, since
  the census does not.
- **Tayo** is a French-based creole (Glottolog tayo1238, Macro-French) and sits under
  `creole.french_based`, though ISEE counts it as Kanak. So the unnamed remainder's node leaves
  out one language ISEE files there: Tayo is 1.6% of named mentions, 86% in Mont-Dore (908 of
  1,052), so at most a few dozen unnamed people are on a node that does not contain their
  language.

Colours: group hues in Oceanic's green-to-blue band (Kanak 175, Loyalty 245, Northern 145,
Southern 210), big languages hand-picked so ground neighbours differ: Iaai violet against
Fagauvea yellow-green on Ouvéa; Cèmuhî yellow, Paicî green, Ajië light blue down the east coast;
Xârâcùù teal against Xârâgùrè pale green at Canala and Thio; all away from French's pink. Small
languages generated. Looked at the hex values only, not a render.

## 5. Placement

religiondots' NC layer (`RD_GEO/nc/nc_hexes.gpkg`, Kontur 400 m hexes keyed to the 33 INSEE
codes), `pop_weight`. Every language in a commune shares the commune's population weight, so in
Nouméa the Kanak-language dots are spread over the southern, mostly European quartiers as much as
the northern ones. religiondots' Loyalty Islands correction to Kontur is not needed here: it
moves weight between communes only, and placement is within a commune.

**Follow-up, not done (placement proxy, no ask needed):** the quartier workbook
`https://www.isee.nc/sites/default/files/2025-10/rp2019-quartiers-communes-plus-de-10000-habitants.xls`
has **P09GN, Kanak-language knowledge (15+) per quartier of Greater Nouméa** (Nouméa, Dumbéa,
Mont-Dore, Païta: 33,505 speakers, 44% of all). Weighting Kanak-language hexes by each quartier's
speakers would put Nouméa's Kanak dots in the right half of the city. Needs quartier polygons:
data.gouv.nc lists `quartiers-noumea` (Licence Ouverte 2.0, the official quartiers, 36 at the 1982 deliberation) but its API
returns no records (probably a file attachment); nothing found for the other three communes.

## 6. Numbers worth keeping

- 75,853 speakers of 15+ (36.0%); 16,941 understand only; 118,187 neither. 65,185 mentions.
- Most-mentioned: Drehu 15,875 (Lifou 5,745, Nouméa 5,272, Dumbéa 2,493), Nengone 9,356, Paicî
  6,530, Xârâcùù 5,645, Ajië 4,449, Iaai 3,714. Smallest: Zîchë 10, Arhâ 19, Pwâpwâ 27.
- Nouméa holds 18,528 speakers, 24% of all.
